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

using .LibTVMFFI

"""
    DLDevice

Re-export from LibTVMFFI for convenience.
Represents a device (CPU, GPU, etc.) and device ID.
"""
const DLDevice = LibTVMFFI.DLDevice
const DLDeviceType = LibTVMFFI.DLDeviceType

"""
    DLDevice(device_type::DLDeviceType, device_id::Integer=0)

Create a device context.

# Examples
```julia
cpu = DLDevice(kDLCPU, 0)
cuda0 = DLDevice(kDLCUDA, 0)
cuda1 = DLDevice(kDLCUDA, 1)
```
"""
function DLDevice(device_type::DLDeviceType, device_id::Integer=0)
    DLDevice(Int32(device_type), Int32(device_id))
end

"""
    cpu(id::Integer=0) -> DLDevice

Create a CPU device context.
"""
cpu(id::Integer=0) = DLDevice(LibTVMFFI.kDLCPU, id)

"""
    cuda(id::Integer=0) -> DLDevice

Create a CUDA device context.
"""
cuda(id::Integer=0) = DLDevice(LibTVMFFI.kDLCUDA, id)

"""
    opencl(id::Integer=0) -> DLDevice

Create an OpenCL device context.
"""
opencl(id::Integer=0) = DLDevice(LibTVMFFI.kDLOpenCL, id)

"""
    vulkan(id::Integer=0) -> DLDevice

Create a Vulkan device context.
"""
vulkan(id::Integer=0) = DLDevice(LibTVMFFI.kDLVulkan, id)

"""
    metal(id::Integer=0) -> DLDevice

Create a Metal device context.
"""
metal(id::Integer=0) = DLDevice(LibTVMFFI.kDLMetal, id)

"""
    rocm(id::Integer=0) -> DLDevice

Create a ROCm device context.
"""
rocm(id::Integer=0) = DLDevice(LibTVMFFI.kDLROCM, id)

# Pretty printing
function Base.show(io::IO, device::DLDevice)
    type_names = Dict(
        Int32(LibTVMFFI.kDLCPU) => "CPU",
        Int32(LibTVMFFI.kDLCUDA) => "CUDA",
        Int32(LibTVMFFI.kDLCUDAHost) => "CUDAHost",
        Int32(LibTVMFFI.kDLOpenCL) => "OpenCL",
        Int32(LibTVMFFI.kDLVulkan) => "Vulkan",
        Int32(LibTVMFFI.kDLMetal) => "Metal",
        Int32(LibTVMFFI.kDLVPI) => "VPI",
        Int32(LibTVMFFI.kDLROCM) => "ROCm",
        Int32(LibTVMFFI.kDLExtDev) => "ExtDev",
    )
    
    type_name = get(type_names, device.device_type, "Unknown")
    print(io, "DLDevice(", type_name, ":", device.device_id, ")")
end

