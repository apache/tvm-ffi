#=
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
=#

"""
    TVMFFI

Julia bindings for TVM FFI (Foreign Function Interface).

This package provides a Julia interface to TVM's C API, enabling
machine learning model compilation and execution from Julia.

# Core Types
- `DLDevice`: Device abstraction (CPU, CUDA, etc.)
- `DLDataType`: Data type descriptor
- `TVMError`: Error handling
- `TVMString`: ABI-stable string type
- `TVMFunction`: Function objects
- `TVMTensor`: N-dimensional array type

# Design Philosophy
- Direct C API calls via ccall (no intermediate layer)
- Julia's GC manages object lifetimes via finalizers
- Simple, clear abstractions without over-engineering
"""
module TVMFFI

# Export core types
export DLDevice, DLDataType, DLDeviceType, DLDataTypeCode
export DLTensor, DLTensorHolder
export TVMError, TVMErrorKind
export TVMString, TVMBytes
export TVMFunction
export TVMTensor

# Export error kinds
export ValueError, TypeError, RuntimeError, AttributeError, KeyError, IndexError

# Export utility functions
export check_call, shape, dtype, device
export get_global_func
export to_julia_array, copy_to_julia, from_julia_array
export cpu, cuda, opencl, vulkan, metal, rocm

# Export GPU support functions and types
export GPUDLTensorHolder
export from_gpu_array
export supports_gpu_backend, list_available_gpu_backends
export print_gpu_info, gpu_array_info

# Include submodules in dependency order
include("LibTVMFFI.jl")
include("error.jl")
include("dtype.jl")
include("device.jl")
include("string.jl")
include("object.jl")
include("tensor.jl")              # Defines DLTensor and basic conversions
include("gpuarrays_support.jl")   # GPU abstraction (uses DLTensor, extends from_julia_array)
include("function.jl")            # Uses DLTensor

end # module

