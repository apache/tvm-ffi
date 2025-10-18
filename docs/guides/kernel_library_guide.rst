.. Licensed to the Apache Software Foundation (ASF) under one
.. or more contributor license agreements.  See the NOTICE file
.. distributed with this work for additional information
.. regarding copyright ownership.  The ASF licenses this file
.. to you under the Apache License, Version 2.0 (the
.. "License"); you may not use this file except in compliance
.. with the License.  You may obtain a copy of the License at
..
..   http://www.apache.org/licenses/LICENSE-2.0
..
.. Unless required by applicable law or agreed to in writing,
.. software distributed under the License is distributed on an
.. "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
.. KIND, either express or implied.  See the License for the
.. specific language governing permissions and limitations
.. under the License.

====================
Kernel Library Guide
====================

This guide serves as a quick start for composing a kernel from scratch, or migrating a kernel from externel frameworks. It covers the core concepts in TVM FFI, such as tensor, stream.

Tensor
======

Tensor is the most important input for a kernel libaray. In PyTorch C++ extensions, kernel library usually takes ``at::Tensor`` as tensor input. In TVM FFI, we introduce two types of tensor, ``ffi::Tensor`` and ``ffi::TensorView``.

Tensor and TensorView
---------------------

Both ``ffi::Tensor`` and ``ffi::TensorView`` are designed to represent tensors in TVM FFI eco-system. The main difference is whether it is an owning tensor pointer.

:ffi::Tensor:
 ``ffi::Tensor`` is a completely onwing tensor pointer, pointing to TVM FFI tensor object. TVM FFI handles the lifetime of ``ffi::Tensor`` by retaining a strong reference.

:ffi::TensorView:
 ``ffi::TensorView`` is a light weight non-owning tnesor pointer, pointeing to a TVM FFI tensor or external tensor object. TVM FFI does not retain its reference. So users are responsible for ensuring the lifetime of tensor object to which the ``ffi::TensorView`` points.

TVM FFI can automatically convert the input tensor at Python side, e.g. ``torch.Tensor``, to both ``ffi::Tensor`` or ``ffi::TensorView`` at C++ side, depends on the C++ function arguments. However, for more flexibility and better compatibility, we **recommand** to use ``ffi::TensorView`` in practice. Since some frameworks, like JAX, cannot provide strong referenced tensor, as ``ffi::Tensor`` expected.

Tensor as Argument
------------------

Typically, we expect that all tensors are pre-allocated at Python side and passed in via TVM FFI, including the output tensor. And TVM FFI will convert them into ``ffi::TensorView`` at runtime. For the optional arguments, ``ffi::Optional`` is the best practice. Here is an example of a kernel definition at C++ side and calling at Python side.

.. code-block:: c++

 // Kernel definition
 void func(ffi::TensorView input, ffi::Optional<ffi::Tensor> optional_input, ffi::TensorView output, ffi::TensorView workspace);

.. code-block:: python

 # Kernel calling
 input = torch.tensor(...)
 output = torch.empty(...)
 workspace = torch.empty(...)
 func(input, None, output, workspace)

Ideally, we expect the kernel function to have ``void`` as return type. However, if it is necessary to return the ``ffi::Tensor`` anyway, please pay attention to convert the output ``ffi::Tensor`` to original tensor type at Python side, like ``torch.from_dlpack``.

Tensor Attributes
-----------------

For the sake of convenience, ``ffi::TensorView`` and ``ffi::Tensor`` align the following attributes retrieval mehtods to ``at::Tensor`` interface:

``dim``, ``sizes``, ``size``, ``strides``, ``stride``, ``numel``, ``data_ptr``, ``device``, ``is_contiguous``

:DLDataType:
 In TVM FFI, tensor data types are stored as ``DLDataType`` which is defined by DLPack protocol.

:DLDevice:
 In TVM FFI, tensor device information are stored as ``DLDevice`` which is defined by DLPack protocol.

:ShapeView:
 In TVM FFI, tensor shapes and strides attributes retrieval are returned as ``ShapeView``. It is an iterate-able data structure storing the shapes or strides data as ``int64_t`` array.

Tensor Allocation
-----------------

Sometimes we have to allocate the tensor within the kernel. TVM FFI provides several methods to allocate tensors.

:FromNDAlloc:
 ``FromNDAlloc`` is the most basic tensor allocator. Besides of the basic attributes like shape, data type and device, it requires a custom allocator struct to handle the allocation and free. The allocator must consist of ``void AllocData(DLTensor*)`` and ``void FreeData(DLTensor*)`` methods. Here are the examples of CPU, CUDA and NVSHMEM allocation:

 .. code-block:: c++

  // CPU Allocator
  struct CPUNDAlloc {
    void AllocData(DLTensor* tensor) { tensor->data = malloc(ffi::GetDataSize(*tensor)); }
    void FreeData(DLTensor* tensor) { free(tensor->data); }
  };

  // CUDA Allocator
  struct CUDANDAlloc {
    void AllocData(DLTensor* tensor) {
      size_t data_size = ffi::GetDataSize(*tensor);
      void* ptr = nullptr;
      cudaError_t err = cudaMalloc(&ptr, data_size);
      TVM_FFI_ICHECK_EQ(err, cudaSuccess) << "cudaMalloc failed: " << cudaGetErrorString(err);
      tensor->data = ptr;
    }
    void FreeData(DLTensor* tensor) {
      if (tensor->data != nullptr) {
        cudaError_t err = cudaFree(tensor->data);
        TVM_FFI_ICHECK_EQ(err, cudaSuccess) << "cudaFree failed: " << cudaGetErrorString(err);
        tensor->data = nullptr;
      }
    }
  };

  // NVSHMEM Allocator
  struct NVSHMEMNDAlloc {
    void AllocData(DLTensor* tensor) {
      size_t size = tvm::ffi::GetDataSize(*tensor);
      tensor->data = nvshmem_malloc(size);
      TVM_FFI_ICHECK_NE(tensor->data, nullptr) << "nvshmem_malloc failed. size: " << size;
    }
    void FreeData(DLTensor* tensor) { nvshmem_free(tensor->data); }
  };

  // Allocator usage
  ffi::Tensor cpu_tensor = ffi::Tensor::FromNDAlloc(CPUNDAlloc(), ...);
  ffi::Tensor cuda_tensor = ffi::Tensor::FromNDAlloc(CUDANDAlloc(), ...);
  ffi::Tensor nvshmem_tensor = ffi::Tensor::FromNDAlloc(NVSHMEMNDAlloc(), ...);

:FromEnvAlloc:
 For the case of using external tensor allocator, like``at::empty`` in PyTorch C++ extensions, ``FromEnvAlloc`` is the better choice. Besides of the basic attributes like shape, data type and device, it requires a thread-local environmental allocator ``TVMFFIEnvTensorAlloc``. ``TVMFFIEnvTensorAlloc`` gets the global tensor allocator in the current context. The context can be switched based on the arguments of the kernel.

:FromDLPack:
 ``FromDLPack`` enables creating ``ffi::Tensor`` from ``DLManagedTensor*``.

:FromDLPackVersioned:
 ``FromDLPackVersioned`` enables creating ``ffi::Tensor`` from ``DLManagedTensorVersioned*``.

Stream
======

TVM FFI maintains the stream context per device type and index. Use ``TVMFFIEnvGetStream`` to get the current stream on device:

.. code-block:: c++

 ffi::DLDevice device = input.device();
 cudaStream_t stream = reinterpret_cast<cudaStream_t>(TVMFFIEnvGetStream(device.device_type, device.device_id));

Similar to ``TVMFFIEnvTensorAlloc``, TVM FFI updates the context stream based on the arguments of the kernel, by calling ``TVMFFIEnvSetStream``.
