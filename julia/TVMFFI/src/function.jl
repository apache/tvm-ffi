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
    TVMFunction

Wrapper for TVM function objects with automatic argument conversion.
"""
mutable struct TVMFunction
    handle::LibTVMFFI.TVMFFIObjectHandle

    """
        TVMFunction(handle; own=true)

    Create a TVMFunction from a raw handle.

    # Arguments
    - `handle`: The raw function handle
    - `own`: If true, increment refcount (default). If false, take ownership without IncRef.
    """
    function TVMFunction(handle::LibTVMFFI.TVMFFIObjectHandle; own::Bool=true)
        if handle == C_NULL
            error("Cannot create TVMFunction from NULL handle")
        end

        # Optionally increase reference count
        if own
            LibTVMFFI.TVMFFIObjectIncRef(handle)
        end

        func = new(handle)

        # Finalizer
        finalizer(func) do f
            if f.handle != C_NULL
                LibTVMFFI.TVMFFIObjectDecRef(f.handle)
            end
        end

        return func
    end
end

"""
    get_global_func(name::AbstractString) -> Union{TVMFunction, Nothing}

Get a global function by name.

# Arguments
- `name::AbstractString`: The name of the global function

# Returns
- `TVMFunction` if the function exists
- `nothing` if the function does not exist

# Examples
```julia
func = get_global_func("my_custom_function")
if func !== nothing
    result = func(arg1, arg2)
end
```
"""
function get_global_func(name::AbstractString)
    name_str = String(name)
    byte_array = LibTVMFFI.TVMFFIByteArray(pointer(name_str), sizeof(name_str))

    ret, handle = LibTVMFFI.TVMFFIFunctionGetGlobal(byte_array)

    if ret != 0
        # Check if error is "function not found" or something else
        error_handle = LibTVMFFI.TVMFFIErrorMoveFromRaised()
        if error_handle != C_NULL
            throw(TVMError(error_handle; own=false))  # C API transferred ownership
        else
            # No error but non-zero return
            error("Failed to get global function '$name' with code $ret")
        end
    end

    if handle == C_NULL
        return nothing
    end

    # C API returns a new reference, so we take ownership without IncRef
    return TVMFunction(handle; own=false)
end

"""
    to_tvm_any(value) -> LibTVMFFI.TVMFFIAny

Convert Julia value to TVMFFIAny for function arguments.
"""
function to_tvm_any(value::Int64)
    # POD type - no refcounting
    LibTVMFFI.TVMFFIAny(Int32(LibTVMFFI.kTVMFFIInt), 0, reinterpret(UInt64, value))
end

function to_tvm_any(value::Float64)
    # POD type - no refcounting
    LibTVMFFI.TVMFFIAny(Int32(LibTVMFFI.kTVMFFIFloat), 0, reinterpret(UInt64, value))
end

function to_tvm_any(value::Bool)
    # POD type - no refcounting
    LibTVMFFI.TVMFFIAny(Int32(LibTVMFFI.kTVMFFIBool), 0, UInt64(value))
end

function to_tvm_any(value::DLDevice)
    # POD type - no refcounting
    packed = UInt64(value.device_type) | (UInt64(value.device_id) << 32)
    LibTVMFFI.TVMFFIAny(Int32(LibTVMFFI.kTVMFFIDevice), 0, packed)
end

function to_tvm_any(value::DLDataType)
    # POD type - no refcounting
    packed = UInt64(value.code) | (UInt64(value.bits) << 8) | (UInt64(value.lanes) << 16)
    LibTVMFFI.TVMFFIAny(Int32(LibTVMFFI.kTVMFFIDataType), 0, packed)
end

function to_tvm_any(value::TVMString)
    # Object type - create new reference
    if value.data.type_index >= Int32(LibTVMFFI.kTVMFFIStaticObjectBegin)
        obj_ptr = reinterpret(LibTVMFFI.TVMFFIObjectHandle, value.data.data)
        if obj_ptr != C_NULL
            LibTVMFFI.TVMFFIObjectIncRef(obj_ptr)
        end
    end
    return value.data
end

function to_tvm_any(value::AbstractString)
    # Convert to TVMString then to Any
    to_tvm_any(TVMString(value))
end

function to_tvm_any(value::TVMFunction)
    # Object type - create new reference
    if value.handle != C_NULL
        LibTVMFFI.TVMFFIObjectIncRef(value.handle)
    end
    LibTVMFFI.TVMFFIAny(
        Int32(LibTVMFFI.kTVMFFIFunction),
        0,
        reinterpret(UInt64, value.handle)
    )
end

function to_tvm_any(value::TVMObject)
    # Generic object - create new reference
    if value.handle != C_NULL
        LibTVMFFI.TVMFFIObjectIncRef(value.handle)
    end
    LibTVMFFI.TVMFFIAny(
        type_index(value),
        0,
        reinterpret(UInt64, value.handle)
    )
end

function to_tvm_any(::Nothing)
    # Special value - no refcounting
    LibTVMFFI.TVMFFIAny(Int32(LibTVMFFI.kTVMFFINone), 0, 0)
end

function to_tvm_any(value::Base.RefValue{DLTensor})
    # Pointer type - no refcounting (borrowed reference)
    LibTVMFFI.TVMFFIAny(
        Int32(LibTVMFFI.kTVMFFIDLTensorPtr),
        0,
        reinterpret(UInt64, Base.unsafe_convert(Ptr{DLTensor}, value))
    )
end

function to_tvm_any(holder::DLTensorHolder)
    # Convert holder to DLTensor pointer
    # Holder keeps data alive, we just borrow the reference
    # Use unsafe_convert which we defined for DLTensorHolder
    tensor_ptr = Base.unsafe_convert(Ptr{DLTensor}, holder)
    LibTVMFFI.TVMFFIAny(
        Int32(LibTVMFFI.kTVMFFIDLTensorPtr),
        0,
        reinterpret(UInt64, tensor_ptr)
    )
end

# Note: to_tvm_any for GPUDLTensorHolder is defined in gpuarrays_support.jl

"""
    from_tvm_any(any::LibTVMFFI.TVMFFIAny) -> Any

Convert TVMFFIAny back to Julia value.
"""
function from_tvm_any(any::LibTVMFFI.TVMFFIAny)
    type_idx = any.type_index

    if type_idx == Int32(LibTVMFFI.kTVMFFINone)
        return nothing
    elseif type_idx == Int32(LibTVMFFI.kTVMFFIInt)
        return reinterpret(Int64, any.data)
    elseif type_idx == Int32(LibTVMFFI.kTVMFFIBool)
        return any.data != 0
    elseif type_idx == Int32(LibTVMFFI.kTVMFFIFloat)
        return reinterpret(Float64, any.data)
    elseif type_idx == Int32(LibTVMFFI.kTVMFFIDevice)
        device_type = Int32(any.data & 0xFFFFFFFF)
        device_id = Int32((any.data >> 32) & 0xFFFFFFFF)
        return DLDevice(device_type, device_id)
    elseif type_idx == Int32(LibTVMFFI.kTVMFFIDataType)
        code = UInt8(any.data & 0xFF)
        bits = UInt8((any.data >> 8) & 0xFF)
        lanes = UInt16((any.data >> 16) & 0xFFFF)
        return DLDataType(code, bits, lanes)
    elseif type_idx == Int32(LibTVMFFI.kTVMFFISmallStr) || type_idx == Int32(LibTVMFFI.kTVMFFIStr)
        # Take ownership without extra IncRef
        return String(TVMString(any; own=false))
    elseif type_idx == Int32(LibTVMFFI.kTVMFFISmallBytes) || type_idx == Int32(LibTVMFFI.kTVMFFIBytes)
        # Take ownership without extra IncRef
        return Vector{UInt8}(TVMBytes(any; own=false))
    elseif type_idx == Int32(LibTVMFFI.kTVMFFIFunction)
        handle = reinterpret(LibTVMFFI.TVMFFIObjectHandle, any.data)
        # Take ownership without extra IncRef
        return TVMFunction(handle; own=false)
    elseif type_idx == Int32(LibTVMFFI.kTVMFFIError)
        handle = reinterpret(LibTVMFFI.TVMFFIObjectHandle, any.data)
        # Take ownership without extra IncRef
        return TVMError(handle; own=false)
    elseif type_idx >= Int32(LibTVMFFI.kTVMFFIStaticObjectBegin)
        # Generic object - take ownership without extra IncRef
        handle = reinterpret(LibTVMFFI.TVMFFIObjectHandle, any.data)
        return TVMObject(handle; own=false)
    else
        error("Unsupported type index for conversion: $type_idx")
    end
end

"""
    (func::TVMFunction)(args...) -> Any

Call a TVM function with arguments.

Arrays are automatically converted to DLTensorHolder for convenience.
Pre-created holders can be passed for performance optimization.

# Examples
```julia
# Simple - auto-conversion
func(x, y)

# Optimized - reuse holders
holder = from_julia_array(x)
for i in 1:1000000
    func(holder)
end
```
"""
function (func::TVMFunction)(args...)
    # Auto-convert AbstractArrays to DLTensorHolder
    # This allows users to pass arrays directly for convenience,
    # or pass pre-made holders for performance optimization
    processed_args = Any[
        arg isa AbstractArray && !(arg isa DLTensorHolder) ?
            from_julia_array(arg) : arg
        for arg in args
    ]

    # Convert arguments to TVMFFIAny array
    num_args = length(processed_args)
    args_array = Vector{LibTVMFFI.TVMFFIAny}(undef, num_args)

    for (i, arg) in enumerate(processed_args)
        args_array[i] = to_tvm_any(arg)
    end

    # Prepare result
    result = Ref{LibTVMFFI.TVMFFIAny}(
        LibTVMFFI.TVMFFIAny(Int32(LibTVMFFI.kTVMFFINone), 0, 0)
    )

    # CRITICAL GC Safety:
    # We must preserve BOTH the original args AND processed_args
    # - args: contains original arrays (data source)
    # - processed_args: contains holders (pointers extracted from these)
    # The C call uses pointers from holders, which reference data in arrays
    local ret
    GC.@preserve args processed_args begin
        # Call function
        ret = if num_args > 0
            LibTVMFFI.TVMFFIFunctionCall(
                func.handle,
                pointer(args_array),
                Int32(num_args),
                Base.unsafe_convert(Ptr{LibTVMFFI.TVMFFIAny}, result)
            )
        else
            LibTVMFFI.TVMFFIFunctionCall(
                func.handle,
                C_NULL,
                Int32(0),
                Base.unsafe_convert(Ptr{LibTVMFFI.TVMFFIAny}, result)
            )
        end
    end

    check_call(ret)

    # Convert result back to Julia type
    julia_result = from_tvm_any(result[])

    # Cleanup: decrease ref counts for object arguments we created
    for arg_any in args_array
        if arg_any.type_index >= Int32(LibTVMFFI.kTVMFFIStaticObjectBegin)
            obj_ptr = reinterpret(LibTVMFFI.TVMFFIObjectHandle, arg_any.data)
            if obj_ptr != C_NULL
                LibTVMFFI.TVMFFIObjectDecRef(obj_ptr)
            end
        end
    end

    return julia_result
end

Base.show(io::IO, func::TVMFunction) = print(io, "TVMFunction(@", repr(UInt(func.handle)), ")")
