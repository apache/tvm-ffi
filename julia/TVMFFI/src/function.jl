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

Wrapper for TVM function objects.

Allows calling TVM functions from Julia with automatic argument conversion.

# Design Philosophy
The core is simple: wrap a handle, provide call interface.
Julia's multiple dispatch handles type conversions naturally.
"""
mutable struct TVMFunction
    handle::LibTVMFFI.TVMFFIObjectHandle
    
    function TVMFunction(handle::LibTVMFFI.TVMFFIObjectHandle)
        if handle == C_NULL
            error("Cannot create TVMFunction from NULL handle")
        end
        
        # Increase reference count
        LibTVMFFI.TVMFFIObjectIncRef(handle)
        
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
            throw(TVMError(error_handle))
        else
            # No error but non-zero return
            error("Failed to get global function '$name' with code $ret")
        end
    end
    
    if handle == C_NULL
        return nothing
    end
    
    return TVMFunction(handle)
end

"""
    to_tvm_any(value) -> LibTVMFFI.TVMFFIAny

Convert Julia value to TVMFFIAny for function call arguments.
"""
function to_tvm_any(value::Int64)
    LibTVMFFI.TVMFFIAny(Int32(LibTVMFFI.kTVMFFIInt), 0, reinterpret(UInt64, value))
end

function to_tvm_any(value::Float64)
    LibTVMFFI.TVMFFIAny(Int32(LibTVMFFI.kTVMFFIFloat), 0, reinterpret(UInt64, value))
end

function to_tvm_any(value::Bool)
    LibTVMFFI.TVMFFIAny(Int32(LibTVMFFI.kTVMFFIBool), 0, UInt64(value))
end

function to_tvm_any(value::DLDevice)
    # Pack device into UInt64
    packed = UInt64(value.device_type) | (UInt64(value.device_id) << 32)
    LibTVMFFI.TVMFFIAny(Int32(LibTVMFFI.kTVMFFIDevice), 0, packed)
end

function to_tvm_any(value::DLDataType)
    # Pack dtype into UInt64
    packed = UInt64(value.code) | (UInt64(value.bits) << 8) | (UInt64(value.lanes) << 16)
    LibTVMFFI.TVMFFIAny(Int32(LibTVMFFI.kTVMFFIDataType), 0, packed)
end

function to_tvm_any(value::TVMString)
    # Return the internal TVMFFIAny
    # Need to increment ref count if it's a heap-allocated object
    if value.data.type_index >= Int32(LibTVMFFI.kTVMFFIStaticObjectBegin)
        obj_ptr = reinterpret(LibTVMFFI.TVMFFIObjectHandle, value.data.data)
        if obj_ptr != C_NULL
            LibTVMFFI.TVMFFIObjectIncRef(obj_ptr)
        end
    end
    return value.data
end

function to_tvm_any(value::AbstractString)
    to_tvm_any(TVMString(value))
end

function to_tvm_any(value::TVMFunction)
    # Increase ref count since we're creating a new reference
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
    # Generic object - use its handle directly
    # Increase ref count since we're creating a new reference
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
    LibTVMFFI.TVMFFIAny(Int32(LibTVMFFI.kTVMFFINone), 0, 0)
end

function to_tvm_any(value::Base.RefValue{DLTensor})
    # DLTensor pointer (for passing tensors to functions)
    LibTVMFFI.TVMFFIAny(
        Int32(LibTVMFFI.kTVMFFIDLTensorPtr),
        0,
        reinterpret(UInt64, Base.unsafe_convert(Ptr{DLTensor}, value))
    )
end

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
        return String(TVMString(any))
    elseif type_idx == Int32(LibTVMFFI.kTVMFFISmallBytes) || type_idx == Int32(LibTVMFFI.kTVMFFIBytes)
        return Vector{UInt8}(TVMBytes(any))
    elseif type_idx == Int32(LibTVMFFI.kTVMFFIFunction)
        handle = reinterpret(LibTVMFFI.TVMFFIObjectHandle, any.data)
        return TVMFunction(handle)
    elseif type_idx == Int32(LibTVMFFI.kTVMFFIError)
        handle = reinterpret(LibTVMFFI.TVMFFIObjectHandle, any.data)
        return TVMError(handle)
    elseif type_idx >= Int32(LibTVMFFI.kTVMFFIStaticObjectBegin)
        # Generic object
        handle = reinterpret(LibTVMFFI.TVMFFIObjectHandle, any.data)
        return TVMObject(handle)
    else
        error("Unsupported type index for conversion: $type_idx")
    end
end

"""
    (func::TVMFunction)(args...) -> Any

Call a TVM function with arguments.

# Examples
```julia
func = get_global_func("my_function")
result = func(1, 2.0, "hello")
```
"""
function (func::TVMFunction)(args...)
    # Convert arguments to TVMFFIAny array
    num_args = length(args)
    args_array = Vector{LibTVMFFI.TVMFFIAny}(undef, num_args)
    
    for (i, arg) in enumerate(args)
        args_array[i] = to_tvm_any(arg)
    end
    
    # Prepare result
    result = Ref{LibTVMFFI.TVMFFIAny}(
        LibTVMFFI.TVMFFIAny(Int32(LibTVMFFI.kTVMFFINone), 0, 0)
    )
    
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
