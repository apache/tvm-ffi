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

This guide serves as a quick start for shipping python version and framework agnostic kernel libraries with TVM FFI.

Tensor
======

TVM FFI provide minimal set of data structures to represent tensors from frameworks and allows us to build kernels for frameworks. In TVM FFI, we support two types of tensor constructs, ``ffi::Tensor`` and ``ffi::TensorView`` that can be used to represent a tensor from machine learning frameworks, such as PyTorch, XLA, JAX, and so on.

Tensor and TensorView
---------------------

Both ``ffi::Tensor`` and ``ffi::TensorView`` are designed to represent tensors from ML frameworks that interact with the TVM FFI ABI. The main difference is whether it is an owning tensor structure.

ffi::Tensor
 ``ffi::Tensor`` is a completely onwing tensor pointer, pointing to a TVM FFI tensor object. TVM FFI handles the lifetime of ``ffi::Tensor`` by retaining a strong reference.

ffi::TensorView
 ``ffi::TensorView`` is non-owning view of an existing tensor. It is backed by ``DLTensor`` structure in DLPack. Since it is a non-owning view, it is the user's responsibility to ensure the lifetime of underlying tensor data and attributes of the viewed tensor object.

We **recommend** to use ``ffi::TensorView`` when possible, that helps us to support more cases, including cases where only view but not strong reference are passed, like XLA buffer. It is also more lightweight.

Tensor Attributes
-----------------

For the sake of convenience, ``ffi::TensorView`` and ``ffi::Tensor`` align the following attributes retrieval mehtods to ``at::Tensor`` interface:

``dim``, ``sizes``, ``size``, ``strides``, ``stride``, ``numel``, ``data_ptr``, ``device``, ``is_contiguous``

DLDataType
 In TVM FFI, tensor data types are stored as ``DLDataType`` which is defined by DLPack protocol.

DLDevice
 In TVM FFI, tensor device information are stored as ``DLDevice`` which is defined by DLPack protocol.

ShapeView
 In TVM FFI, tensor shapes and strides attributes retrieval are returned as ``ShapeView``. It is an iterate-able data structure storing the shapes or strides data as ``int64_t`` array.

Tensor Allocation
-----------------

TVM FFI provides several methods to allocate tensors, when dynamic tensor allocation is necessary.

FromEnvAlloc
 Usually TVM FFI works together with a ML framework with its own tensor allocator. ``FromEnvAlloc`` is tailor-made for this case, so that it is possible to use framework tensor allocator when allocating ``ffi::Tensor``. And TVM FFI automatically sets the framework tensor allocator when the corresponding framework tensor exists in FFI arguments. For example, when calling TVM FFI packed kernels, if there are any input arguments of type ``torch.Tensor`` at Python side, TVM FFI will bind the ``at::Empty`` as the global framework tensor allocator - ``TVMFFIEnvTensorAlloc``. Here is an example:

 .. code-block:: c++

  void func(ffi::TensorView arg0, ffi::TensorView arg1, ...) {
   ffi::Tensor tensor0 = ffi::Tensor::FromEnvAlloc(TVMFFIEnvTensorAlloc, ...);
   ffi::Tensor tensor1 = ffi::Tensor::FromDLPackVersioned(at::toDLPackImpl<DLManagedTensorVersioned>(at::empty(...)))
   // tensor0 and tensor1 are equivalent once arg{i} at Python side has type of torch.Tensor.
  }

 We **recommend** to use ``FromEnvAlloc`` when possible, since the framework tensor allocator has adavantages:

 * Benefit from the framework's native caching allocator or related mechanism.
 * Help framework tracking memory usage and planning globally.

FromNDAlloc
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

FromDLPack
 ``FromDLPack`` enables creating ``ffi::Tensor`` from ``DLManagedTensor*``, working with ``ToDLPack`` for DLPack C Tensor Object ``DLTensor`` exchange protocol. Both are used for DLPack pre V1.0 API.

FromDLPackVersioned
 ``FromDLPackVersioned`` enables creating ``ffi::Tensor`` from ``DLManagedTensorVersioned*``, working with ``ToDLPackVersioned`` for DLPack C Tensor Object ``DLTensor`` exchange protocol. Both are used for DLPack post V1.0 API.

Tensor Passing FFI
------------------

TVM FFI does two conversions when calling the compiled kernels to pass the tensor across FFI. It first converts the framework tensor at Python side to ``ffi::Tensor`` or ``ffi::TensorView``. And then it converts the output ``ffi::Tensor`` back to the framework tensor at Python side. When converting back, TVM FFI will convert to the same framework as arguments. If there are no framework tensors provided in the arguments, TVM FFI will output tensors with the type of ``tvm_ffi.core.Tensor`` still.

Actually, in practie, we **recommend** that all input and output tensors are pre-allocated at Python side by framework alreadly. As for the optional arguments, use ``ffi::Optional`` as wrapper. So, for the kernel function, it returns nothing with a ``void`` return type. Here is a paradigm of TVM FFI interact with Pytorch:

.. code-block:: c++

 // Kernel definition
 void func(ffi::TensorView input, ffi::Optional<ffi::Tensor> optional_input, ffi::TensorView output, ffi::TensorView workspace);

.. code-block:: python

 # Kernel calling
 input: torch.Tensor = ...
 output: torch.Tensor = ...
 workspace: torch.Tensor = ...
 func(input, None, output, workspace)

Stream
======

TVM FFI maintains the stream context per device type and index. And TVM FFI automatically updates the context stream when handling the arguments. For example, if there is an argument of ``torch.Tensor(device="cuda:3")``, TVM FFI will set the current stream of cuda device 3 from torch current context stream. Then at C++ side, use ``TVMFFIEnvGetStream`` to get the current stream on the specific device. Here is an example:

.. code-block:: c++

 void func(ffi::TensorView arg0, ...) {
  ffi::DLDevice device = arg0.device();
  cudaStream_t stream0 = reinterpret_cast<cudaStream_t>(TVMFFIEnvGetStream(device.device_type, device.device_id));
  cudaStream_t stream1 = reinterpret_cast<cudaStream_t>(at::cuda::getCurrentCUDAStream(device.device_id).stream());
  // stream0 and stream1 are the same cuda stream handle once arg0 is of type torch.Tensor at Python side, or any other torch.Tensor arguments at PYthon side are on the same device as arg0.
 }
