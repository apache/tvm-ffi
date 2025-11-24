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
 * \file tvm/ffi/extra/cubin_launcher.h
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
 * symbols should be named __tvm_ffi__cubin_<name> and __tvm_ffi__cubin_<name>_end,
 * typically created using objcopy and ld.
 *
 * ## Creating Embedded CUBIN Symbols with objcopy
 *
 * To embed a CUBIN file into your binary, follow these steps:
 *
 * ### Step 1: Compile CUDA kernel to CUBIN
 * \code{.bash}
 * nvcc --cubin -arch=sm_75 kernel.cu -o kernel.cubin
 * \endcode
 *
 * ### Step 2: Convert CUBIN to object file with ld
 * \code{.bash}
 * ld -r -b binary -o kernel_data.o kernel.cubin
 * \endcode
 *
 * This creates an object file with symbols based on the input filename:
 * - _binary_kernel_cubin_start
 * - _binary_kernel_cubin_end
 * - _binary_kernel_cubin_size
 *
 * ### Step 3: Rename symbols with objcopy
 * \code{.bash}
 * objcopy --rename-section .data=.rodata,alloc,load,readonly,data,contents \
 *         --redefine-sym _binary_kernel_cubin_start=__tvm_ffi__cubin_<name> \
 *         --redefine-sym _binary_kernel_cubin_end=__tvm_ffi__cubin_<name>_end \
 *         kernel_data.o
 * \endcode
 *
 * Replace `<name>` with your chosen identifier (e.g., "env", "my_kernels").
 *
 * ### Step 4: Link the object file with your library/executable
 * \code{.bash}
 * g++ -o mylib.so -shared mycode.cc kernel_data.o -Wl,-z,noexecstack
 * \endcode
 *
 * Note: The `-z,noexecstack` flag marks the stack as non-executable, which is
 * required for security as the embedded object file lacks a .note.GNU-stack section.
 *
 * ## CMake Example
 *
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
 * target_link_options(mylib PRIVATE "LINKER:-z,noexecstack")
 * \endcode
 *
 * ## Usage in C++ Code
 *
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
 * \param name The identifier for this embedded CUBIN module (must match the
 *             symbol names created with objcopy).
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
 * CUBIN module (using TVM_FFI_EMBED_CUBIN). The result is a CubinKernel object.
 * It's recommended to store the result in a static variable to avoid repeated
 * kernel lookups.
 *
 * Example usage:
 * \code
 * static auto kernel = TVM_FFI_EMBED_CUBIN_GET_KERNEL(my_kernels, "kernel_name");
 * \endcode
 *
 * \param name The identifier of the embedded CUBIN module.
 * \param kernel_name The name of the kernel function (as a string literal).
 * \return A CubinKernel object for the specified kernel.
 */
#define TVM_FFI_EMBED_CUBIN_GET_KERNEL(name, kernel_name) \
  (EmbedCubinModule_##name::Global()->mod[kernel_name])

// Forward declaration
class CubinKernel;

/*!
 * \brief CUDA CUBIN module loader and manager.
 *
 * This class provides a RAII wrapper around CUDA driver API's library management.
 * It loads a CUBIN module from memory and manages the library handle.
 * Supports multi-GPU execution using CUDA primary contexts.
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
 * methods to launch it with specified parameters and configuration.
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
   * This function launches the kernel on the current CUDA context/device.
   *
   * \param args Array of pointers to kernel arguments.
   * \param grid Grid dimensions (number of blocks).
   * \param block Block dimensions (threads per block).
   * \param stream CUDA stream to launch the kernel on.
   * \param dyn_smem_bytes Dynamic shared memory size in bytes (default: 0).
   * \return CUresult error code from cuLaunchKernel.
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
