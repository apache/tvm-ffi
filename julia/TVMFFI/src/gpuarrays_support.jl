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
GPUArrays.jl Integration for Hardware-Agnostic GPU Support

GPUArrays.jl provides a unified interface for GPU programming that works across:
- CUDA.jl (NVIDIA)
- AMDGPU.jl (AMD ROCm)
- Metal.jl (Apple)
- oneAPI.jl (Intel)

This module provides integration between TVMFFI and GPUArrays, enabling
hardware-agnostic GPU tensor operations.

# Architecture

```
                    TVMFFI.jl
                        ↓
                  GPUArrays.jl (abstract interface)
                   ↙    ↓    ↘
            CUDA.jl  AMDGPU.jl  Metal.jl
               ↓        ↓         ↓
            NVIDIA    AMD       Apple
```

# Design Philosophy (Linus-style)

**Simple**: One interface, multiple backends
**Direct**: Minimal abstraction, just type mapping
**Practical**: Solve real problem (multi-vendor GPU support)

# Usage

```julia
using TVMFFI
using CUDA  # or AMDGPU, or Metal, or oneAPI

# Works with any GPU backend!
x_gpu = CUDA.CuArray(Float32[1, 2, 3])  # NVIDIA
# x_gpu = AMDGPU.ROCArray(Float32[1, 2, 3])  # AMD
# x_gpu = Metal.MtlArray(Float32[1, 2, 3])  # Apple

# Generic conversion (detects backend automatically)
x_dl = from_gpu_array(x_gpu)

# Call TVM function (backend-agnostic)
tvm_func(x_dl, y_dl)
```
"""

# Check if GPUArrays is available
const HAS_GPUARRAYS = Ref(false)

function __init_gpuarrays__()
    try
        # Try to load GPUArrays
        if isdefined(Main, :GPUArrays)
            HAS_GPUARRAYS[] = true
            @info "GPUArrays.jl integration enabled"
            return true
        end
    catch e
        @debug "GPUArrays.jl not available" exception=e
    end
    return false
end

"""
    gpu_backend_to_dldevice(backend::Symbol) -> DLDeviceType

Map GPU backend name to DLDeviceType.

# Arguments
- `backend::Symbol`: GPU backend (:CUDA, :ROCm, :Metal, :oneAPI)

# Returns
- `DLDeviceType`: Corresponding DLPack device type
"""
function gpu_backend_to_dldevice(backend::Symbol)
    backend_map = Dict(
        :CUDA => LibTVMFFI.kDLCUDA,
        :ROCm => LibTVMFFI.kDLROCM,
        :AMDGPU => LibTVMFFI.kDLROCM,  # AMDGPU.jl uses ROCm
        :Metal => LibTVMFFI.kDLMetal,
        :oneAPI => LibTVMFFI.kDLExtDev,  # Intel oneAPI
        :Vulkan => LibTVMFFI.kDLVulkan,
        :OpenCL => LibTVMFFI.kDLOpenCL,
    )
    
    if haskey(backend_map, backend)
        return backend_map[backend]
    else
        error("Unsupported GPU backend: $backend. Supported: $(keys(backend_map))")
    end
end

"""
    detect_gpu_backend(arr) -> (Symbol, Int)

Detect GPU backend and device ID from a GPU array.

Uses type-based detection - no runtime introspection of Main module.

# Arguments
- `arr`: GPU array (CuArray, ROCArray, MtlArray, etc.)

# Returns
- `(backend::Symbol, device_id::Int)`: Backend name and device ID

# Design Philosophy (Linus-style)
Old code: isdefined(Main, :CUDA) - hacky runtime introspection
New code: Type name pattern matching - simple and direct
No special cases for device ID - just use 0 as default

# Examples
```julia
using CUDA
x = CUDA.CuArray([1, 2, 3])
backend, dev_id = detect_gpu_backend(x)  # (:CUDA, 0)
```
"""
function detect_gpu_backend(arr)
    arr_type = typeof(arr)
    type_name = string(arr_type.name.name)
    
    # Detect backend from array type name
    # Simple pattern matching - no runtime introspection needed
    if occursin("Cu", type_name) || occursin("CUDA", type_name)
        return (:CUDA, 0)  # Default to device 0
        
    elseif occursin("ROC", type_name) || occursin("AMD", type_name)
        return (:ROCm, 0)  # Default to device 0
        
    elseif occursin("Mtl", type_name) || occursin("Metal", type_name)
        return (:Metal, 0)
        
    elseif occursin("oneAPI", type_name)
        return (:oneAPI, 0)
        
    else
        error("Cannot detect GPU backend from array type: $arr_type. " *
              "If you're using a custom GPU array type, specify backend explicitly.")
    end
end

"""
    GPUDLTensorHolder{T}

GPU version of DLTensorHolder - self-contained holder for GPU arrays.
"""
mutable struct GPUDLTensorHolder{T}
    tensor::DLTensor
    shape::Vector{Int64}
    strides::Vector{Int64}
    source::T  # GPU array (CuArray, ROCArray, etc.)
    
    function GPUDLTensorHolder(arr::T; backend::Symbol=:auto, device_id::Int=-1) where T
        # Auto-detect backend if not specified
        if backend == :auto || device_id == -1
            detected_backend, detected_device_id = detect_gpu_backend(arr)
            if backend == :auto
                backend = detected_backend
            end
            if device_id == -1
                device_id = detected_device_id
            end
        end
        
        # Get DLDeviceType
        dl_device_type = gpu_backend_to_dldevice(backend)
        
        # Create device
        device = DLDevice(Int32(dl_device_type), Int32(device_id))
        
        # Get element type
        elem_type = eltype(arr)
        dt = DLDataType(elem_type)
        
        # Get shape
        shape_tuple = size(arr)
        ndim = length(shape_tuple)
        shape_vec = collect(Int64, shape_tuple)
        
        # Calculate strides (Julia is column-major)
        strides_vec = ones(Int64, ndim)
        for i in 2:ndim
            strides_vec[i] = strides_vec[i-1] * shape_vec[i-1]
        end
        
        # Create DLTensor
        tensor = DLTensor(
            pointer(arr),           # GPU memory pointer
            device,                 # Device context
            Int32(ndim),           # Number of dimensions
            dt,                    # Data type
            pointer(shape_vec),    # Shape pointer
            pointer(strides_vec),  # Strides pointer
            UInt64(0)              # Byte offset
        )
        
        return new{T}(tensor, shape_vec, strides_vec, arr)
    end
end

# Allow Ref() on GPU holder too
Base.Ref(holder::GPUDLTensorHolder) = Ref(holder.tensor)

# Define unsafe_convert for GPU holder
function Base.unsafe_convert(::Type{Ptr{DLTensor}}, holder::GPUDLTensorHolder)
    # Get pointer to the tensor field within the holder
    return Ptr{DLTensor}(pointer_from_objref(holder))
end

"""
    from_gpu_array(arr; backend::Symbol=:auto, device_id::Int=-1) -> GPUDLTensorHolder

Create a self-contained DLTensor holder from a GPU array (hardware-agnostic).

This function works with any GPUArrays.jl-compatible backend:
- CUDA.jl → CuArray
- AMDGPU.jl → ROCArray
- Metal.jl → MtlArray
- oneAPI.jl → oneArray

# Arguments
- `arr`: GPU array
- `backend`: GPU backend (auto-detected if :auto)
- `device_id`: Device ID (auto-detected if -1)

# Returns
- `GPUDLTensorHolder`: Self-contained holder (keep this alive!)

# Examples
```julia
using CUDA
x = CUDA.CuArray(Float32[1, 2, 3, 4, 5])
holder = from_gpu_array(x)
# Automatically detects CUDA backend
func(holder)  # Pass holder directly

using AMDGPU
x = AMDGPU.ROCArray(Float32[1, 2, 3, 4, 5])
holder = from_gpu_array(x)
# Automatically detects ROCm backend
```

# Design
New safe API - single holder object instead of three-tuple.
"""
function from_gpu_array(arr; backend::Symbol=:auto, device_id::Int=-1)
    return GPUDLTensorHolder(arr; backend=backend, device_id=device_id)
end

"""
    supports_gpu_backend(backend::Symbol) -> Bool

Check if a specific GPU backend is available.

# Design Philosophy (Linus-style)
Old code: Try to introspect Main module and call .functional()
New code: Just check if the package exists in the current environment
Simpler, no weird Main.CUDA hacks

# Arguments
- `backend::Symbol`: Backend to check (:CUDA, :ROCm, :Metal, :oneAPI)

# Returns
- `Bool`: Whether the backend package is loaded

# Examples
```julia
if supports_gpu_backend(:CUDA)
    println("CUDA is available!")
end
```
"""
function supports_gpu_backend(backend::Symbol)
    # Check if the module is in the namespace by trying to get its const
    # This is cleaner than isdefined(Main, :...)
    try
        if backend == :CUDA
            return isdefined(@__MODULE__, :CUDA) || Base.get_extension(@__MODULE__, :CUDAExt) !== nothing
        elseif backend == :ROCm || backend == :AMDGPU
            return isdefined(@__MODULE__, :AMDGPU) || Base.get_extension(@__MODULE__, :AMDGPUExt) !== nothing
        elseif backend == :Metal
            return isdefined(@__MODULE__, :Metal) || Base.get_extension(@__MODULE__, :MetalExt) !== nothing
        elseif backend == :oneAPI
            return isdefined(@__MODULE__, :oneAPI) || Base.get_extension(@__MODULE__, :oneAPIExt) !== nothing
        end
    catch
        # Fallback: just return false if extension system not available (Julia < 1.9)
        return false
    end
    return false
end

"""
    list_available_gpu_backends() -> Vector{Symbol}

List all available and functional GPU backends.

# Returns
- `Vector{Symbol}`: List of available backends

# Examples
```julia
backends = list_available_gpu_backends()
println("Available GPU backends: ", backends)
# Output: [:CUDA, :ROCm]  (depending on system)
```
"""
function list_available_gpu_backends()
    all_backends = [:CUDA, :ROCm, :Metal, :oneAPI]
    available = Symbol[]
    
    for backend in all_backends
        if supports_gpu_backend(backend)
            push!(available, backend)
        end
    end
    
    return available
end

"""
    print_gpu_info()

Print information about available GPU backends and devices.

# Examples
```julia
print_gpu_info()
# Output:
# Available GPU Backends:
#   ✓ CUDA (NVIDIA)
#     • Device 0: NVIDIA GeForce RTX 3090
#     • Device 1: NVIDIA GeForce RTX 3080
#   ✓ ROCm (AMD)
#     • Device 0: AMD Radeon RX 6900 XT
```
"""
function print_gpu_info()
    println("Available GPU Backends:")
    
    backends = list_available_gpu_backends()
    
    if isempty(backends)
        println("  ❌ No GPU backends available")
        println("\nTo enable GPU support, install one of:")
        println("  • CUDA.jl:   using Pkg; Pkg.add(\"CUDA\")")
        println("  • AMDGPU.jl: using Pkg; Pkg.add(\"AMDGPU\")")
        println("  • Metal.jl:  using Pkg; Pkg.add(\"Metal\")")
        println("  • oneAPI.jl: using Pkg; Pkg.add(\"oneAPI\")")
        return
    end
    
    for backend in backends
        vendor = if backend == :CUDA
            "NVIDIA"
        elseif backend == :ROCm || backend == :AMDGPU
            "AMD"
        elseif backend == :Metal
            "Apple"
        elseif backend == :oneAPI
            "Intel"
        else
            "Unknown"
        end
        
        println("  ✓ $backend ($vendor)")
        
        # Simplified: just report backend is available
        # Device enumeration requires calling into the backend packages
        # which we've intentionally avoided for simplicity
        println("    • Backend available (device enumeration requires importing package)")
    end
end

# Extend from_julia_array to handle GPU arrays via type dispatch
# Note: We're in the TVMFFI module, so we can just add a new method
"""
    from_julia_array(arr::AbstractArray)

Extended method that handles both CPU and GPU arrays automatically.

Type dispatch:
- Array{T} → DLTensorHolder{T} (CPU)
- CuArray/ROCArray/etc. → GPUDLTensorHolder (GPU)

This is cleaner than runtime type checking.
"""
function from_julia_array(arr::AbstractArray)
    # For non-Array types (GPU arrays), use GPU holder
    # This assumes any non-Array is a GPU array
    # More specific methods can be added for specific GPU types if needed
    return GPUDLTensorHolder(arr)
end

# Extend to_tvm_any for GPU holder
# Note: We're in the TVMFFI module, so we can just add a new method
"""
    to_tvm_any(holder::GPUDLTensorHolder)

Convert GPU DLTensor holder to TVMFFIAny.
"""
function to_tvm_any(holder::GPUDLTensorHolder)
    # Convert holder to DLTensor pointer
    # Holder keeps GPU array alive, we just borrow the reference
    # Use unsafe_convert which we defined for GPUDLTensorHolder
    tensor_ptr = Base.unsafe_convert(Ptr{DLTensor}, holder)
    LibTVMFFI.TVMFFIAny(
        Int32(LibTVMFFI.kTVMFFIDLTensorPtr),
        0,
        reinterpret(UInt64, tensor_ptr)
    )
end

"""
    gpu_array_info(arr)

Print diagnostic information about a GPU array.

# Example
```julia
using CUDA
x = CUDA.CuArray(Float32[1, 2, 3])
gpu_array_info(x)
# Output:
#   Backend: CUDA
#   Device: 0
#   Type: Float32
#   Shape: (3,)
#   Pointer: 0x...
```
"""
function gpu_array_info(arr)
    println("GPU Array Information:")
    
    try
        backend, dev_id = detect_gpu_backend(arr)
        println("  Backend: $backend")
        println("  Device ID: $dev_id")
        println("  Element Type: ", eltype(arr))
        println("  Shape: ", size(arr))
        println("  Size: ", length(arr), " elements")
        println("  Memory Pointer: ", repr(UInt(pointer(arr))))
        
        # Map to DLDevice
        dl_type = gpu_backend_to_dldevice(backend)
        dl_dev = DLDevice(Int32(dl_type), Int32(dev_id))
        println("  DLDevice: ", dl_dev)
        
    catch e
        println("  Error getting info: ", e)
    end
end

# Initialize
__init_gpuarrays__()
