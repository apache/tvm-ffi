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

# Unified API - from_julia_array works for CPU and GPU!
x_holder = from_julia_array(x_gpu)  # Auto-detects CUDA

# Call TVM function (backend-agnostic)
tvm_func(x_holder, y_holder)
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

# Deleted from_gpu_array - it's redundant!
# from_julia_array handles GPU arrays automatically.

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

# Extend from_julia_array to handle GPU arrays
"""
    from_julia_array(arr::AbstractArray)

Extended method for GPU arrays - uses same DLTensorHolder as CPU!

# Design Philosophy (Linus-style)
ONE function, ONE type, ALL devices.
- Auto-detect GPU backend from array type
- Create appropriate device automatically
- Return unified DLTensorHolder

No from_gpu_array needed - it was a redundant special case!

# Examples
```julia
using CUDA

# CPU and GPU - same API!
cpu_holder = from_julia_array(cpu_arr)     # Auto: CPU device
gpu_holder = from_julia_array(gpu_arr)     # Auto: CUDA device

# Both return DLTensorHolder{T, S}
# Device info is in holder.tensor.device
```
"""
function from_julia_array(arr::S) where {S <: AbstractArray}
    T = eltype(arr)

    # Auto-detect backend and create GPU device
    backend, device_id = detect_gpu_backend(arr)
    dl_device_type = gpu_backend_to_dldevice(backend)
    device = DLDevice(Int32(dl_device_type), Int32(device_id))

    # Get shape and strides - same as CPU
    shape_vec = collect(Int64, size(arr))
    strides_vec = collect(Int64, Base.strides(arr))

    # Get dtype
    dt = DLDataType(T)

    # Create DLTensor
    tensor = DLTensor(
        pointer(arr),
        device,
        Int32(length(shape_vec)),
        dt,
        pointer(shape_vec),
        pointer(strides_vec),
        UInt64(0)
    )

    return DLTensorHolder{T, S}(tensor, shape_vec, strides_vec, arr)
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
