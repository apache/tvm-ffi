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

# Arguments
- `arr`: GPU array (CuArray, ROCArray, MtlArray, etc.)

# Returns
- `(backend::Symbol, device_id::Int)`: Backend name and device ID

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
    
    # Detect backend from array type
    if occursin("Cu", type_name) || occursin("CUDA", type_name)
        # CUDA.jl
        backend = :CUDA
        device_id = try
            # Try to get device from CUDA
            if isdefined(Main, :CUDA)
                Main.CUDA.device().handle
            else
                0
            end
        catch
            0
        end
        return (backend, device_id)
        
    elseif occursin("ROC", type_name) || occursin("AMD", type_name)
        # AMDGPU.jl (ROCm)
        backend = :ROCm
        device_id = try
            if isdefined(Main, :AMDGPU)
                Main.AMDGPU.device_id(Main.AMDGPU.device())
            else
                0
            end
        catch
            0
        end
        return (backend, device_id)
        
    elseif occursin("Mtl", type_name) || occursin("Metal", type_name)
        # Metal.jl
        backend = :Metal
        device_id = 0  # Metal typically has one default device
        return (backend, device_id)
        
    elseif occursin("oneAPI", type_name)
        # oneAPI.jl
        backend = :oneAPI
        device_id = 0
        return (backend, device_id)
        
    else
        error("Cannot detect GPU backend from array type: $arr_type")
    end
end

"""
    from_gpu_array(arr::T, backend::Symbol=:auto, device_id::Int=-1) where T

Create a DLTensor from a GPU array (hardware-agnostic).

This function works with any GPUArrays.jl-compatible backend:
- CUDA.jl → CuArray
- AMDGPU.jl → ROCArray
- Metal.jl → MtlArray
- oneAPI.jl → oneArray

# Arguments
- `arr`: GPU array
- `backend`: GPU backend (auto-detected if not specified)
- `device_id`: Device ID (auto-detected if -1)

# Returns
- `Ref{DLTensor}`: DLTensor reference
- `Vector{Int64}`: Shape vector (keep alive!)
- `Vector{Int64}`: Strides vector (keep alive!)

# Examples
```julia
using CUDA
x = CUDA.CuArray(Float32[1, 2, 3, 4, 5])
x_dl, shape, strides = from_gpu_array(x)
# Automatically detects CUDA backend

using AMDGPU
x = AMDGPU.ROCArray(Float32[1, 2, 3, 4, 5])
x_dl, shape, strides = from_gpu_array(x)
# Automatically detects ROCm backend
```
"""
function from_gpu_array(arr::T; backend::Symbol=:auto, device_id::Int=-1) where T
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
    dltensor = DLTensor(
        pointer(arr),           # GPU memory pointer
        device,                 # Device context
        Int32(ndim),           # Number of dimensions
        dt,                    # Data type
        pointer(shape_vec),    # Shape pointer
        pointer(strides_vec),  # Strides pointer
        UInt64(0)              # Byte offset
    )
    
    return Ref(dltensor), shape_vec, strides_vec
end

"""
    supports_gpu_backend(backend::Symbol) -> Bool

Check if a specific GPU backend is available.

# Arguments
- `backend::Symbol`: Backend to check (:CUDA, :ROCm, :Metal, :oneAPI)

# Returns
- `Bool`: Whether the backend is available and functional

# Examples
```julia
if supports_gpu_backend(:CUDA)
    println("CUDA is available!")
end
```
"""
function supports_gpu_backend(backend::Symbol)
    if backend == :CUDA
        try
            return isdefined(Main, :CUDA) && Main.CUDA.functional()
        catch
            return false
        end
    elseif backend == :ROCm || backend == :AMDGPU
        try
            return isdefined(Main, :AMDGPU) && Main.AMDGPU.functional()
        catch
            return false
        end
    elseif backend == :Metal
        try
            return isdefined(Main, :Metal) && Main.Metal.functional()
        catch
            return false
        end
    elseif backend == :oneAPI
        try
            return isdefined(Main, :oneAPI) && Main.oneAPI.functional()
        catch
            return false
        end
    else
        return false
    end
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
        
        # Try to get device info
        try
            if backend == :CUDA && isdefined(Main, :CUDA)
                for (idx, dev) in enumerate(Main.CUDA.devices())
                    println("    • Device $(idx-1): ", Main.CUDA.name(dev))
                end
            elseif (backend == :ROCm || backend == :AMDGPU) && isdefined(Main, :AMDGPU)
                # AMDGPU device info
                println("    • Device 0: ROCm GPU")
            elseif backend == :Metal && isdefined(Main, :Metal)
                println("    • Device 0: Metal GPU")
            end
        catch e
            println("    • Device info unavailable")
        end
    end
end

"""
    from_abstract_gpu_array(arr::AbstractGPUArray{T}) where T

Create DLTensor from any GPUArrays.jl-compatible array.

This is the **hardware-agnostic** interface that works with:
- CUDA.jl → CuArray
- AMDGPU.jl → ROCArray  
- Metal.jl → MtlArray
- oneAPI.jl → oneArray

# Design Philosophy

**Linus-style simplicity**:
1. Detect backend automatically (no manual specification)
2. Use existing pointer/device methods (no reinvention)
3. One function, all backends (eliminate special cases)

# Example
```julia
using CUDA  # or AMDGPU, Metal, oneAPI
using TVMFFI

# Create GPU array (any backend)
x_gpu = CUDA.CuArray(Float32[1, 2, 3, 4, 5])

# Generic conversion (detects CUDA automatically)
x_dl, shape, strides = from_abstract_gpu_array(x_gpu)

# Call TVM function (backend-agnostic)
tvm_func(x_dl, y_dl)
```
"""
function from_abstract_gpu_array(arr; backend::Symbol=:auto, device_id::Int=-1)
    # Auto-detect if needed
    if backend == :auto || device_id == -1
        detected_backend, detected_device_id = detect_gpu_backend(arr)
        backend = (backend == :auto) ? detected_backend : backend
        device_id = (device_id == -1) ? detected_device_id : device_id
    end
    
    # Map to DLDevice
    dl_device_type = gpu_backend_to_dldevice(backend)
    device = DLDevice(Int32(dl_device_type), Int32(device_id))
    
    # Get element type and create dtype
    elem_type = eltype(arr)
    dt = DLDataType(elem_type)
    
    # Get shape
    shape_tuple = size(arr)
    ndim = length(shape_tuple)
    shape_vec = collect(Int64, shape_tuple)
    
    # Calculate strides (column-major)
    strides_vec = ones(Int64, ndim)
    for i in 2:ndim
        strides_vec[i] = strides_vec[i-1] * shape_vec[i-1]
    end
    
    # Create DLTensor with GPU pointer
    dltensor = DLTensor(
        pointer(arr),           # GPU device pointer
        device,                 # Device (CUDA/ROCm/Metal/etc.)
        Int32(ndim),
        dt,
        pointer(shape_vec),
        pointer(strides_vec),
        UInt64(0)
    )
    
    return Ref(dltensor), shape_vec, strides_vec
end

# Convenience alias
const from_gpu_array = from_abstract_gpu_array

# Extend from_julia_array to handle GPU arrays
# This overrides the fallback defined in tensor.jl
"""
    from_julia_array(arr::AbstractArray) -> (Ref{DLTensor}, Vector, Vector)

Extended method that handles both CPU and GPU arrays automatically.

This method is extended by gpuarrays_support.jl to handle GPU arrays.
"""
function TVMFFI.from_julia_array(arr::AbstractArray)
    # Runtime check: is this a CPU array?
    if arr isa Array
        # CPU array - delegate to tensor.jl implementation
        return TVMFFI.from_julia_array(arr, cpu())
    else
        # GPU array - use GPU-specific conversion
        return from_abstract_gpu_array(arr)
    end
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
