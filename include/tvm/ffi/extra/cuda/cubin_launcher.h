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
 * This header provides a lightweight C++ wrapper around CUDA Runtime API
 * for loading CUBIN modules and launching kernels. It supports:
 * - Loading CUBIN from memory (embedded data)
 * - Multi-GPU execution using CUDA primary contexts
 * - Kernel parameter management and launch configuration
 */
#ifndef TVM_FFI_EXTRA_CUDA_CUBIN_LAUNCHER_H_
#define TVM_FFI_EXTRA_CUDA_CUBIN_LAUNCHER_H_

#include <cuda_runtime.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/cuda/base.h>
#include <tvm/ffi/string.h>

#include <cstdint>
#include <cstring>

namespace tvm {
namespace ffi {

/*!
 * \brief A simple 3D dimension type for CUDA kernel launch configuration.
 *
 * This struct mimics the behavior of dim3 from CUDA Runtime API and provides
 * a compatible interface for kernel launch configuration. It can be constructed
 * from 1, 2, or 3 dimensions.
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
 * \par Creating Embedded CUBIN with TVM-FFI Utilities
 * TVM-FFI provides utilities to simplify CUBIN embedding. You have two options:
 *
 * \par Option 1: CMake Utility (Recommended)
 * Use the `tvm_ffi_embed_cubin` CMake function:
 * \code{.cmake}
 * # Find tvm_ffi package (provides tvm_ffi_embed_cubin utility)
 * find_package(tvm_ffi CONFIG REQUIRED)
 * find_package(CUDAToolkit REQUIRED)
 *
 * # Compile CUDA kernel to CUBIN
 * tvm_ffi_generate_cubin(
 *   OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/kernel.cubin
 *   SOURCE src/kernel.cu
 *   ARCH native  # or sm_75, sm_80, etc.
 * )
 *
 * # Embed CUBIN into C++ object file
 * tvm_ffi_embed_cubin(
 *   OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/mycode_with_cubin.o
 *   SOURCE src/mycode.cc
 *   CUBIN ${CMAKE_CURRENT_BINARY_DIR}/kernel.cubin
 *   NAME my_kernels  # Must match TVM_FFI_EMBED_CUBIN(my_kernels) in code
 * )
 *
 * # Link into shared library
 * add_library(mylib SHARED ${CMAKE_CURRENT_BINARY_DIR}/mycode_with_cubin.o)
 * target_link_libraries(mylib PRIVATE tvm_ffi_header CUDA::cudart)
 * \endcode
 *
 * \par Option 2: Python Utility
 * Use the `tvm_ffi.utils.embed_cubin` command-line tool:
 * \code{.bash}
 * # Step 1: Compile CUDA kernel to CUBIN
 * nvcc --cubin -arch=sm_75 kernel.cu -o kernel.cubin
 *
 * # Step 2: Compile C++ source to object file
 * g++ -c -fPIC -std=c++17 -I/path/to/tvm-ffi/include mycode.cc -o mycode.o
 *
 * # Step 3: Embed CUBIN using Python utility
 * python -m tvm_ffi.utils.embed_cubin \
 *     --output-obj mycode_with_cubin.o \
 *     --input-obj mycode.o \
 *     --cubin kernel.cubin \
 *     --name my_kernels
 *
 * # Step 4: Link into shared library
 * g++ -o mylib.so -shared mycode_with_cubin.o -lcudart
 * \endcode
 *
 * The utilities automatically handle:
 * - Symbol renaming to __tvm_ffi__cubin_<name> format
 * - Adding .note.GNU-stack section for security
 * - Symbol localization to prevent conflicts
 *
 * \par Usage in C++ Code
 * In your C++ source file, use the embedded CUBIN:
 * \code{.cpp}
 * #include <tvm/ffi/extra/cuda/cubin_launcher.h>
 *
 * // Declare the embedded CUBIN module (name must match CMake NAME parameter)
 * TVM_FFI_EMBED_CUBIN(my_kernels);
 *
 * void MyFunction() {
 *   // Get kernel from embedded CUBIN (cached in static variable for efficiency)
 *   static auto kernel = TVM_FFI_EMBED_CUBIN_GET_KERNEL(my_kernels, "my_kernel");
 *   // Use kernel...
 * }
 * \endcode
 *
 * \note CMake Setup: To use the utilities, add to your CMakeLists.txt:
 * \code{.cmake}
 * find_package(tvm_ffi CONFIG REQUIRED)  # Provides tvm_ffi_embed_cubin utility
 * \endcode
 *
 * \par Option 3: Python Integration with load_inline
 * When using `tvm_ffi.cpp.load_inline()` with the `embed_cubin` parameter,
 * the CUBIN data is automatically embedded using the Python utility internally:
 * \code{.py}
 * from tvm_ffi import cpp
 * from tvm_ffi.cpp import nvrtc
 *
 * # Compile CUDA source to CUBIN
 * cubin_bytes = nvrtc.nvrtc_compile(cuda_source)
 *
 * # Load with embedded CUBIN - automatically handles embedding
 * mod = cpp.load_inline(
 *     "my_module",
 *     cuda_sources=cpp_code,
 *     embed_cubin={"my_kernels": cubin_bytes},
 *     extra_ldflags=["-lcudart"]
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
#define TVM_FFI_EMBED_CUBIN(name)                        \
  extern "C" const char __tvm_ffi__cubin_##name[];       \
  extern "C" const char __tvm_ffi__cubin_##name##_end[]; \
  namespace {                                            \
  struct EmbedCubinModule_##name {                       \
    tvm::ffi::CubinModule mod{__tvm_ffi__cubin_##name};  \
    static EmbedCubinModule_##name* Global() {           \
      static EmbedCubinModule_##name inst;               \
      return &inst;                                      \
    }                                                    \
  };                                                     \
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
 *   cudaStream_t stream = static_cast<cudaStream_t>(
 *       TVMFFIEnvGetStream(device.device_type, device.device_id));
 *
 *   cudaError_t result = kernel.Launch(args, grid, block, stream);
 *   TVM_FFI_CHECK_CUDA_ERROR(result);
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
 * This class provides a RAII wrapper around CUDA Runtime API's library management.
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
 * cudaStream_t stream = ...;
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
   */
  explicit CubinModule(const Bytes& bytes) {
    TVM_FFI_CHECK_CUDA_ERROR(
        cudaLibraryLoadData(&library_, bytes.data(), nullptr, nullptr, 0, nullptr, nullptr, 0));
  }

  /*!
   * \brief Load CUBIN module from raw memory buffer.
   *
   * \param code Pointer to CUBIN binary data.
   * \note The `code` buffer points to an ELF image.
   */
  explicit CubinModule(const char* code) {
    TVM_FFI_CHECK_CUDA_ERROR(
        cudaLibraryLoadData(&library_, code, nullptr, nullptr, 0, nullptr, nullptr, 0));
  }

  /*! \brief Destructor unloads the library */
  ~CubinModule() {
    if (library_ != nullptr) {
      cudaLibraryUnload(library_);
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
   * \brief Get a kernel function from the module by name with maximum dynamic shared memory.
   *
   * \param name Name of the kernel function.
   * \param dynamic_smem_max Maximum dynamic shared memory in bytes to set for this kernel.
   *                         -1 (default) means maximum available dynamic shared memory
   *                         (device max - static shared memory used by kernel).
   * \return CubinKernel object representing the loaded kernel.
   */
  CubinKernel GetKernelWithMaxDynamicSharedMemory(const char* name, int64_t dynamic_smem_max);

  /*!
   * \brief Operator[] for convenient kernel access.
   *
   * It's equivalent to calling GetKernel(name, -1).
   *
   * \param name Name of the kernel function.
   * \return CubinKernel object representing the loaded kernel.
   */
  CubinKernel operator[](const char* name);

  /*! \brief Get the underlying cudaLibrary_t handle */
  cudaLibrary_t GetHandle() const { return library_; }

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
        cudaLibraryUnload(library_);
      }
      library_ = other.library_;
      other.library_ = nullptr;
    }
    return *this;
  }

 private:
  cudaLibrary_t library_ = nullptr;
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
 * cudaStream_t stream = ...;
 * cudaError_t result = kernel.Launch(args, grid, block, stream);
 * TVM_FFI_CHECK_CUDA_ERROR(result);
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
   * \param library The cudaLibrary_t handle.
   * \param name Name of the kernel function.
   */
  CubinKernel(cudaLibrary_t library, const char* name) {
    TVM_FFI_CHECK_CUDA_ERROR(cudaLibraryGetKernel(&kernel_, library, name));
  }

  /*! \brief Destructor (kernel handle doesn't need explicit cleanup) */
  ~CubinKernel() = default;

  /*!
   * \brief Launch the kernel with specified parameters.
   *
   * This function launches the kernel on the current CUDA context/device using
   * the CUDA Runtime API. The kernel executes asynchronously on the specified stream.
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
   * Always check the returned cudaError_t:
   * \code{.cpp}
   * cudaError_t result = kernel.Launch(args, grid, block, stream);
   * TVM_FFI_CHECK_CUDA_ERROR(result);
   * \endcode
   *
   * \param args Array of pointers to kernel arguments (must point to actual values).
   * \param grid Grid dimensions (number of blocks in x, y, z).
   * \param block Block dimensions (threads per block in x, y, z).
   * \param stream CUDA stream to launch the kernel on (use 0 for default stream).
   * \param dyn_smem_bytes Dynamic shared memory size in bytes (default: 0).
   * \return cudaError_t error code from cudaLaunchKernel (cudaSuccess on success).
   *
   * \note The kernel executes asynchronously. Use cudaStreamSynchronize() or
   *       cudaDeviceSynchronize() to wait for completion if needed.
   */
  cudaError_t Launch(void** args, dim3 grid, dim3 block, cudaStream_t stream,
                     uint32_t dyn_smem_bytes = 0) {
    // Cast cudaKernel_t to const void* for use with cudaLaunchKernel
    // The Runtime API accepts cudaKernel_t directly as a function pointer
    auto kernel = reinterpret_cast<const void*>(kernel_);
    return cudaLaunchKernel(kernel, {grid.x, grid.y, grid.z}, {block.x, block.y, block.z}, args,
                            dyn_smem_bytes, stream);
  }

  /*! \brief Get the underlying cudaKernel_t handle */
  cudaKernel_t GetHandle() const { return kernel_; }

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
  /*!
   * \brief Set maximum dynamic shared memory for this kernel across all devices.
   *
   * This method configures the maximum dynamic shared memory that can be allocated
   * when launching this kernel. It must be called after the kernel is loaded.
   *
   * \param dynamic_smem_max Maximum dynamic shared memory in bytes to set.
   *                         -1 (default) means maximum available dynamic shared memory,
   *                         which is computed as (device max shared memory - static shared memory).
   *                         For -1, the method queries the kernel's static shared memory usage
   *                         and sets the attribute to the remaining available shared memory.
   *
   * \note This sets the maximum cap but doesn't force allocation. The actual dynamic
   *       shared memory used is controlled by the dyn_smem_bytes parameter in Launch().
   * \note This method attempts to set the attribute for all available devices and will
   *       only throw an error if it fails for ALL devices.
   */
  void SetMaxDynamicSharedMemory(int64_t dynamic_smem_max = -1) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
      return;  // No devices available, nothing to configure
    }

    bool any_success = false;
    for (int device_id = 0; device_id < device_count; ++device_id) {
      // Query device's maximum shared memory per block
      int max_shared_mem = 0;
      err = cudaDeviceGetAttribute(&max_shared_mem, cudaDevAttrMaxSharedMemoryPerBlock, device_id);
      if (err != cudaSuccess) {
        continue;  // Skip this device if we can't get its attribute
      }

      int shared_mem_to_set;
      if (dynamic_smem_max == -1) {
        // Query the kernel's static shared memory usage
        cudaFuncAttributes func_attr;

        // According to the documentation, we can use cudaFuncGetAttributes to get the attributes of
        // cudaKernel_t returned by cudaLibraryGetKernel, just cast the kernel_ to const void*
        err = cudaFuncGetAttributes(&func_attr, reinterpret_cast<const void*>(kernel_));
        if (err != cudaSuccess) {
          continue;  // Skip this device if we can't get kernel attributes
        }

        // Calculate available dynamic shared memory:
        // device max shared memory - static shared memory used by kernel
        int64_t static_shared = static_cast<int64_t>(func_attr.sharedSizeBytes);
        int64_t max_shared = static_cast<int64_t>(max_shared_mem);
        int64_t available = max_shared - static_shared;
        shared_mem_to_set = (available > 0) ? static_cast<int>(available) : 0;
      } else {
        shared_mem_to_set = static_cast<int>(dynamic_smem_max);
      }

      // Set the maximum dynamic shared memory size for this device
      err = cudaKernelSetAttributeForDevice(kernel_, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            shared_mem_to_set, device_id);
      if (err == cudaSuccess) {
        any_success = true;
      }
      // Don't error out for individual device failures - user may only use some GPUs
    }

    // Only error out if setting failed for ALL devices
    if (!any_success && device_count > 0) {
      TVM_FFI_THROW(RuntimeError) << "Failed to set dynamic shared memory attribute for any device";
    }
  }

  cudaKernel_t kernel_ = nullptr;

  friend class CubinModule;
};

// Implementation of CubinModule methods that return CubinKernel
inline CubinKernel CubinModule::GetKernelWithMaxDynamicSharedMemory(const char* name,
                                                                    int64_t dynamic_smem_max = -1) {
  auto kernel = CubinKernel(library_, name);
  kernel.SetMaxDynamicSharedMemory(dynamic_smem_max);
  return kernel;
}

inline CubinKernel CubinModule::GetKernel(const char* name) {
  auto kernel = CubinKernel(library_, name);
  return kernel;
}

inline CubinKernel CubinModule::operator[](const char* name) { return GetKernel(name); }

}  // namespace ffi
}  // namespace tvm

#endif  // TVM_FFI_EXTRA_CUDA_CUBIN_LAUNCHER_H_
