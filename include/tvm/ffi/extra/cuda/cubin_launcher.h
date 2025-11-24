/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
/*!
 * \file tvm/ffi/extra/cuda/cubin_launcher.h
 * \brief CUDA CUBIN launcher utility for loading and executing CUDA kernels.
 *
 * This header provides a lightweight C++ wrapper around CUDA Driver API
 * for loading CUBIN modules and launching kernels. It supports:
 * - Loading CUBIN from memory (embedded data)
 * - Multi-GPU execution using CUDA primary contexts
 * - Kernel parameter management and launch configuration
 */
#ifndef TVM_FFI_EXTRA_CUBIN_LAUNCHER_H_
#define TVM_FFI_EXTRA_CUBIN_LAUNCHER_H_

#include <cuda.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/string.h>

#include <cstdint>
#include <cstring>

namespace tvm {
namespace ffi {

/*!
 * \brief Macro for checking CUDA driver API errors.
 *
 * This macro checks the return value of CUDA driver API calls and throws
 * a RuntimeError with detailed error information if the call fails.
 *
 * \param stmt The CUDA driver API call to check.
 */
#define TVM_FFI_CHECK_CUDA_DRIVER_ERROR(stmt)                                       \
  do {                                                                              \
    CUresult __err = (stmt);                                                        \
    if (__err != CUDA_SUCCESS) {                                                    \
      const char* __err_name = nullptr;                                             \
      const char* __err_str = nullptr;                                              \
      cuGetErrorName(__err, &__err_name);                                           \
      cuGetErrorString(__err, &__err_str);                                          \
      __err_name = __err_name ? __err_name : "UNKNOWN";                             \
      __err_str = __err_str ? __err_str : "No description";                         \
      TVM_FFI_THROW(RuntimeError) << "CUDA Driver Error: " << __err_name << " ("    \
                                  << static_cast<int>(__err) << "): " << __err_str; \
    }                                                                               \
  } while (0)

/*!
 * \brief A simple 3D dimension type for CUDA kernel launch configuration.
 *
 * This struct mimics the behavior of dim3 from CUDA Runtime API, but works
 * with the CUDA Driver API. It can be constructed from 1, 2, or 3 dimensions.
 */
struct dim3 {
  /*! \brief X dimension (number of blocks in x-direction or threads in x-direction) */
  unsigned int x;
  /*! \brief Y dimension (number of blocks in y-direction or threads in y-direction) */
  unsigned int y;
  /*! \brief Z dimension (number of blocks in z-direction or threads in z-direction) */
  unsigned int z;

  /*! \brief Default constructor initializes to (1, 1, 1) */
  dim3() : x(1), y(1), z(1) {}

  /*! \brief Construct with x dimension, y and z default to 1 */
  explicit dim3(unsigned int x_) : x(x_), y(1), z(1) {}

  /*! \brief Construct with x and y dimensions, z defaults to 1 */
  dim3(unsigned int x_, unsigned int y_) : x(x_), y(y_), z(1) {}

  /*! \brief Construct with all three dimensions */
  dim3(unsigned int x_, unsigned int y_, unsigned int z_) : x(x_), y(y_), z(z_) {}
};

/*!
 * \brief Macro to embed a CUBIN module with static initialization.
 *
 * This macro declares external symbols for embedded CUBIN data and creates
 * a singleton struct to manage the CubinModule instance. The CUBIN data
 * symbols should be named `__tvm_ffi__cubin_<name>` and `__tvm_ffi__cubin_<name>_end`,
 * typically created using objcopy and ld.
 *
 * \par Creating Embedded CUBIN Symbols
 * To embed a CUBIN file into your binary, follow these steps:
 *
 * \par Step 1: Compile CUDA kernel to CUBIN
 * \code{.bash}
 * nvcc --cubin -arch=sm_75 kernel.cu -o kernel.cubin
 * \endcode
 *
 * \par Step 2: Convert CUBIN to object file with ld
 * \code{.bash}
 * ld -r -b binary -o kernel_data.o kernel.cubin
 * \endcode
 * This creates an object file with symbols: `_binary_kernel_cubin_start`,
 * `_binary_kernel_cubin_end`, and `_binary_kernel_cubin_size`.
 *
 * \par Step 3: Rename symbols with objcopy
 * \code{.bash}
 * objcopy --rename-section .data=.rodata,alloc,load,readonly,data,contents \
 *         --redefine-sym _binary_kernel_cubin_start=__tvm_ffi__cubin_<name> \
 *         --redefine-sym _binary_kernel_cubin_end=__tvm_ffi__cubin_<name>_end \
 *         kernel_data.o
 * \endcode
 * Replace `<name>` with your chosen identifier (e.g., "env", "my_kernels").
 *
 * \par Step 4: Link the object file with your library/executable
 * \code{.bash}
 * g++ -o mylib.so -shared mycode.cc kernel_data.o -Wl,-z,noexecstack -lcuda
 * \endcode
 * \note The `-z,noexecstack` flag marks the stack as non-executable, which is
 * required for security as the embedded object file lacks a .note.GNU-stack section.
 *
 * \par CMake Example
 * \code{.cmake}
 * add_custom_command(OUTPUT kernel_data.o
 *   COMMAND ${CMAKE_LINKER} -r -b binary -o kernel_data.o kernel.cubin
 *   COMMAND ${CMAKE_OBJCOPY}
 *     --rename-section .data=.rodata,alloc,load,readonly,data,contents
 *     --redefine-sym _binary_kernel_cubin_start=__tvm_ffi__cubin_env
 *     --redefine-sym _binary_kernel_cubin_end=__tvm_ffi__cubin_env_end
 *     kernel_data.o
 *   DEPENDS kernel.cubin)
 *
 * add_library(mylib SHARED mycode.cc kernel_data.o)
 * target_link_libraries(mylib PRIVATE cuda)
 * target_link_options(mylib PRIVATE "LINKER:-z,noexecstack")
 * \endcode
 *
 * \par Usage in C++ Code
 * \code{.cpp}
 * // Declare the embedded CUBIN module (use the same name as in objcopy)
 * TVM_FFI_EMBED_CUBIN(env);
 *
 * void MyFunction() {
 *   // Get kernel from embedded CUBIN (cached in static variable)
 *   static auto kernel = TVM_FFI_EMBED_CUBIN_GET_KERNEL(env, "my_kernel");
 *   // Use kernel...
 * }
 * \endcode
 *
 * \par Python Integration
 * When using `tvm_ffi.cpp.load_inline()` with the `embed_cubin` parameter,
 * the CUBIN data is automatically embedded and the symbols are created for you.
 * \code{.py}
 * mod = cpp.load_inline(
 *     "my_module",
 *     cuda_sources=cpp_code,
 *     embed_cubin={"env": cubin_bytes}
 * )
 * \endcode
 *
 * \param name The identifier for this embedded CUBIN module (must match the
 *             symbol names created with objcopy or the key in embed_cubin dict).
 *
 * \see TVM_FFI_EMBED_CUBIN_GET_KERNEL
 * \see CubinModule
 * \see CubinKernel
 */
#define TVM_FFI_EMBED_CUBIN(name)                                                       \
  extern "C" const char __tvm_ffi__cubin_##name[];                                      \
  extern "C" const char __tvm_ffi__cubin_##name##_end[];                                \
  namespace {                                                                           \
  struct EmbedCubinModule_##name {                                                      \
    tvm::ffi::CubinModule mod{tvm::ffi::Bytes(                                          \
        __tvm_ffi__cubin_##name,                                                        \
        static_cast<size_t>(__tvm_ffi__cubin_##name##_end - __tvm_ffi__cubin_##name))}; \
    static EmbedCubinModule_##name* Global() {                                          \
      static EmbedCubinModule_##name inst;                                              \
      return &inst;                                                                     \
    }                                                                                   \
  };                                                                                    \
  } /* anonymous namespace */

/*!
 * \brief Macro to get a kernel from an embedded CUBIN module.
 *
 * This macro retrieves a kernel by name from a previously declared embedded
 * CUBIN module (using TVM_FFI_EMBED_CUBIN). The result is a CubinKernel object
 * that can be used to launch the kernel with specified parameters.
 *
 * \par Performance Tip
 * It's recommended to store the result in a static variable to avoid repeated
 * kernel lookups, which improves performance:
 * \code{.cpp}
 * static auto kernel = TVM_FFI_EMBED_CUBIN_GET_KERNEL(my_kernels, "kernel_name");
 * \endcode
 *
 * \par Complete Example
 * \code{.cpp}
 * // Declare embedded CUBIN module
 * TVM_FFI_EMBED_CUBIN(my_kernels);
 *
 * void LaunchKernel(tvm::ffi::TensorView input, tvm::ffi::TensorView output) {
 *   // Get kernel (cached in static variable for efficiency)
 *   static auto kernel = TVM_FFI_EMBED_CUBIN_GET_KERNEL(my_kernels, "add_one");
 *
 *   // Prepare kernel arguments
 *   void* in_ptr = input.data_ptr();
 *   void* out_ptr = output.data_ptr();
 *   int64_t n = input.size(0);
 *   void* args[] = {&in_ptr, &out_ptr, &n};
 *
 *   // Configure launch
 *   tvm::ffi::dim3 grid((n + 255) / 256);
 *   tvm::ffi::dim3 block(256);
 *
 *   // Get stream and launch
 *   DLDevice device = input.device();
 *   CUstream stream = static_cast<CUstream>(
 *       TVMFFIEnvGetStream(device.device_type, device.device_id));
 *
 *   CUresult result = kernel.Launch(args, grid, block, stream);
 *   TVM_FFI_CHECK_CUDA_DRIVER_ERROR(result);
 * }
 * \endcode
 *
 * \param name The identifier of the embedded CUBIN module (must match the name
 *             used in TVM_FFI_EMBED_CUBIN).
 * \param kernel_name The name of the kernel function as it appears in the CUBIN
 *                    (typically the function name for `extern "C"` kernels).
 * \return A CubinKernel object for the specified kernel.
 *
 * \see TVM_FFI_EMBED_CUBIN
 * \see CubinKernel::Launch
 */
#define TVM_FFI_EMBED_CUBIN_GET_KERNEL(name, kernel_name) \
  (EmbedCubinModule_##name::Global()->mod[kernel_name])

// Forward declaration
class CubinKernel;

/*!
 * \brief CUDA CUBIN module loader and manager.
 *
 * This class provides a RAII wrapper around CUDA Driver API's library management.
 * It loads a CUBIN module from memory and manages the library handle automatically.
 * The library is unloaded when the CubinModule object is destroyed.
 *
 * \par Features
 * - Load CUBIN from memory (embedded data or runtime-generated)
 * - Automatic resource management (RAII pattern)
 * - Multi-GPU execution using CUDA primary contexts
 * - Retrieve multiple kernels from the same module
 *
 * \par Example Usage
 * \code{.cpp}
 * // Load CUBIN from memory
 * tvm::ffi::Bytes cubin_data = ...;
 * tvm::ffi::CubinModule module(cubin_data);
 *
 * // Get kernels by name
 * tvm::ffi::CubinKernel kernel1 = module["add_one"];
 * tvm::ffi::CubinKernel kernel2 = module.GetKernel("mul_two");
 *
 * // Launch kernels
 * void* args[] = {...};
 * tvm::ffi::dim3 grid(32), block(256);
 * CUstream stream = ...;
 * kernel1.Launch(args, grid, block, stream);
 * \endcode
 *
 * \note This class is movable but not copyable.
 * \see TVM_FFI_EMBED_CUBIN for embedding CUBIN at compile time
 * \see CubinKernel for kernel launching
 */
class CubinModule {
 public:
  /*!
   * \brief Load CUBIN module from memory.
   *
   * \param bytes CUBIN binary data as a Bytes object.
   * \note Calls cuInit(0) to ensure CUDA is initialized.
   */
  explicit CubinModule(const Bytes& bytes) {
    TVM_FFI_CHECK_CUDA_DRIVER_ERROR(cuInit(0));
    TVM_FFI_CHECK_CUDA_DRIVER_ERROR(
        cuLibraryLoadData(&library_, bytes.data(), nullptr, nullptr, 0, nullptr, nullptr, 0));
  }

  /*! \brief Destructor unloads the library */
  ~CubinModule() {
    if (library_ != nullptr) {
      cuLibraryUnload(library_);
    }
  }

  /*!
   * \brief Get a kernel function from the module by name.
   *
   * \param name Name of the kernel function.
   * \return CubinKernel object representing the loaded kernel.
   */
  CubinKernel GetKernel(const char* name);

  /*!
   * \brief Operator[] for convenient kernel access.
   *
   * \param name Name of the kernel function.
   * \return CubinKernel object representing the loaded kernel.
   */
  CubinKernel operator[](const char* name);

  /*! \brief Get the underlying CUlibrary handle */
  CUlibrary GetHandle() const { return library_; }

  // Non-copyable
  CubinModule(const CubinModule&) = delete;
  CubinModule& operator=(const CubinModule&) = delete;

  /*!
   * \brief Move constructor for CubinModule.
   *
   * Transfers ownership of the CUDA library handle from another CubinModule instance.
   *
   * \param other The source CubinModule to move from (will be left in an empty state).
   */
  CubinModule(CubinModule&& other) noexcept : library_(other.library_) { other.library_ = nullptr; }

  /*!
   * \brief Move assignment operator for CubinModule.
   *
   * Transfers ownership of the CUDA library handle from another CubinModule instance.
   * Cleans up any existing library handle in this instance before taking ownership.
   *
   * \param other The source CubinModule to move from (will be left in an empty state).
   * \return Reference to this CubinModule.
   */
  CubinModule& operator=(CubinModule&& other) noexcept {
    if (this != &other) {
      if (library_ != nullptr) {
        cuLibraryUnload(library_);
      }
      library_ = other.library_;
      other.library_ = nullptr;
    }
    return *this;
  }

 private:
  CUlibrary library_ = nullptr;
};

/*!
 * \brief CUDA kernel handle for launching kernels.
 *
 * This class represents a loaded CUDA kernel function and provides
 * methods to launch it with specified grid/block dimensions, arguments,
 * and stream configuration. Obtained from CubinModule by kernel name.
 *
 * \par Usage Pattern
 * \code{.cpp}
 * // Get kernel from module
 * tvm::ffi::CubinKernel kernel = module["kernel_name"];
 *
 * // Prepare arguments (must be pointers to actual values)
 * void* data_ptr = tensor.data_ptr();
 * int64_t size = tensor.size(0);
 * void* args[] = {&data_ptr, &size};
 *
 * // Configure launch dimensions
 * tvm::ffi::dim3 grid(32);    // 32 blocks
 * tvm::ffi::dim3 block(256);  // 256 threads per block
 *
 * // Launch on stream
 * CUstream stream = ...;
 * CUresult result = kernel.Launch(args, grid, block, stream);
 * TVM_FFI_CHECK_CUDA_DRIVER_ERROR(result);
 * \endcode
 *
 * \note This class is movable but not copyable.
 * \see CubinModule for loading CUBIN and getting kernels
 * \see dim3 for grid/block dimension specification
 */
class CubinKernel {
 public:
  /*!
   * \brief Construct a CubinKernel from a library and kernel name.
   *
   * \param library The CUlibrary handle.
   * \param name Name of the kernel function.
   */
  CubinKernel(CUlibrary library, const char* name) {
    TVM_FFI_CHECK_CUDA_DRIVER_ERROR(cuLibraryGetKernel(&kernel_, library, name));
  }

  /*! \brief Destructor (kernel handle doesn't need explicit cleanup) */
  ~CubinKernel() = default;

  /*!
   * \brief Launch the kernel with specified parameters.
   *
   * This function launches the kernel on the current CUDA context/device using
   * the CUDA Driver API. The kernel executes asynchronously on the specified stream.
   *
   * \par Argument Preparation
   * The `args` array must contain pointers to the actual argument values, not the
   * values themselves. For example:
   * \code{.cpp}
   * void* data_ptr = tensor.data_ptr();
   * int64_t size = 100;
   * void* args[] = {&data_ptr, &size};  // Note: addresses of the variables
   * \endcode
   *
   * \par Launch Configuration
   * Grid and block dimensions determine the kernel's parallelism:
   * - Grid: Number of thread blocks (can be 1D, 2D, or 3D)
   * - Block: Number of threads per block (can be 1D, 2D, or 3D)
   * - Total threads = grid.x * grid.y * grid.z * block.x * block.y * block.z
   *
   * \par Error Checking
   * Always check the returned CUresult:
   * \code{.cpp}
   * CUresult result = kernel.Launch(args, grid, block, stream);
   * TVM_FFI_CHECK_CUDA_DRIVER_ERROR(result);
   * \endcode
   *
   * \param args Array of pointers to kernel arguments (must point to actual values).
   * \param grid Grid dimensions (number of blocks in x, y, z).
   * \param block Block dimensions (threads per block in x, y, z).
   * \param stream CUDA stream to launch the kernel on (use 0 for default stream).
   * \param dyn_smem_bytes Dynamic shared memory size in bytes (default: 0).
   * \return CUresult error code from cuLaunchKernel (CUDA_SUCCESS on success).
   *
   * \note The kernel executes asynchronously. Use cudaStreamSynchronize() or
   *       cudaDeviceSynchronize() to wait for completion if needed.
   */
  CUresult Launch(void** args, dim3 grid, dim3 block, CUstream stream,
                  uint32_t dyn_smem_bytes = 0) {
    CUfunction function;
    // Get function handle for the current context from the kernel
    CUresult result = cuKernelGetFunction(&function, kernel_);
    if (result != CUDA_SUCCESS) {
      return result;
    }

    return cuLaunchKernel(function, grid.x, grid.y, grid.z, block.x, block.y, block.z,
                          dyn_smem_bytes, stream, args, nullptr);
  }

  /*! \brief Get the underlying CUkernel handle */
  CUkernel GetHandle() const { return kernel_; }

  // Non-copyable
  CubinKernel(const CubinKernel&) = delete;
  CubinKernel& operator=(const CubinKernel&) = delete;

  /*!
   * \brief Move constructor for CubinKernel.
   *
   * Transfers ownership of the CUDA kernel handle from another CubinKernel instance.
   *
   * \param other The source CubinKernel to move from (will be left in an empty state).
   */
  CubinKernel(CubinKernel&& other) noexcept : kernel_(other.kernel_) { other.kernel_ = nullptr; }

  /*!
   * \brief Move assignment operator for CubinKernel.
   *
   * Transfers ownership of the CUDA kernel handle from another CubinKernel instance.
   *
   * \param other The source CubinKernel to move from (will be left in an empty state).
   * \return Reference to this CubinKernel.
   */
  CubinKernel& operator=(CubinKernel&& other) noexcept {
    if (this != &other) {
      kernel_ = other.kernel_;
      other.kernel_ = nullptr;
    }
    return *this;
  }

 private:
  CUkernel kernel_ = nullptr;
};

// Implementation of CubinModule methods that return CubinKernel
inline CubinKernel CubinModule::GetKernel(const char* name) { return CubinKernel(library_, name); }

inline CubinKernel CubinModule::operator[](const char* name) { return GetKernel(name); }

}  // namespace ffi
}  // namespace tvm

#endif  // TVM_FFI_EXTRA_CUBIN_LAUNCHER_H_
