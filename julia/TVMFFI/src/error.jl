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
    TVMErrorKind

Error kind wrapper. Simply wraps a string for clarity.
"""
struct TVMErrorKind
    name::String
end

Base.string(k::TVMErrorKind) = k.name

# Define standard error kinds
const ValueError = TVMErrorKind("ValueError")
const TypeError = TVMErrorKind("TypeError")
const RuntimeError = TVMErrorKind("RuntimeError")
const AttributeError = TVMErrorKind("AttributeError")
const KeyError = TVMErrorKind("KeyError")
const IndexError = TVMErrorKind("IndexError")

"""
    TVMError

TVM FFI error type. Wraps a TVM error object handle.

# Fields
- `handle::TVMFFIObjectHandle`: The underlying error object
- `kind::String`: Error kind (ValueError, TypeError, etc.)
- `message::String`: Error message
- `backtrace::String`: Stack backtrace

# Design Notes
Julia's exception system handles error propagation, so we don't need
complex reference counting here. The finalizer handles cleanup.
"""
mutable struct TVMError <: Exception
    handle::LibTVMFFI.TVMFFIObjectHandle
    kind::String
    message::String
    backtrace::String
    
    function TVMError(kind::TVMErrorKind, message::AbstractString, backtrace::AbstractString="")
        # Create byte arrays for C API
        kind_bytes = LibTVMFFI.TVMFFIByteArray(pointer(kind.name), sizeof(kind.name))
        msg_bytes = LibTVMFFI.TVMFFIByteArray(pointer(message), sizeof(message))
        bt_bytes = LibTVMFFI.TVMFFIByteArray(pointer(backtrace), sizeof(backtrace))
        
        ret, handle = LibTVMFFI.TVMFFIErrorCreate(kind_bytes, msg_bytes, bt_bytes)
        if ret != 0
            # Error creating error object - this is bad, but we can't recurse
            error("Failed to create TVM error object (out of memory?)")
        end
        
        err = new(handle, kind.name, string(message), string(backtrace))
        
        # Register finalizer to decrease reference count
        finalizer(err) do e
            if e.handle != C_NULL
                LibTVMFFI.TVMFFIObjectDecRef(e.handle)
            end
        end
        
        return err
    end
    
    # Constructor from existing handle (e.g., from TLS)
    function TVMError(handle::LibTVMFFI.TVMFFIObjectHandle)
        if handle == C_NULL
            error("Cannot create TVMError from NULL handle")
        end
        
        # Get the error cell pointer
        cell_ptr = LibTVMFFI.TVMFFIErrorGetCellPtr(handle)
        cell = unsafe_load(cell_ptr)
        
        # Extract kind, message, and backtrace
        kind_str = unsafe_string(cell.kind.data, cell.kind.size)
        msg_str = unsafe_string(cell.message.data, cell.message.size)
        bt_str = unsafe_string(cell.backtrace.data, cell.backtrace.size)
        
        err = new(handle, kind_str, msg_str, bt_str)
        
        # Register finalizer
        finalizer(err) do e
            if e.handle != C_NULL
                LibTVMFFI.TVMFFIObjectDecRef(e.handle)
            end
        end
        
        return err
    end
end

"""
    check_call(ret::Integer)

Check the return code from a TVM FFI C API call.
If non-zero, retrieve and throw the error from TLS.

# Arguments
- `ret::Integer`: Return code from C API (0 = success, non-zero = error)

# Example
```julia
ret = LibTVMFFI.TVMFFISomeFunction(...)
check_call(ret)
```
"""
function check_call(ret::Integer)
    if ret != 0
        # Retrieve error from thread-local storage
        error_handle = LibTVMFFI.TVMFFIErrorMoveFromRaised()
        if error_handle != C_NULL
            throw(TVMError(error_handle))
        else
            # No error in TLS, but ret was non-zero
            error("TVM FFI call failed with code $ret but no error in TLS")
        end
    end
    nothing
end

# Pretty printing for TVMError
function Base.show(io::IO, e::TVMError)
    println(io, "TVMError: $(e.kind)")
    println(io, "  ", e.message)
    if !isempty(e.backtrace)
        println(io, "Backtrace (most recent call last):")
        # Reverse backtrace lines to match Python convention
        bt_lines = split(e.backtrace, '\n')
        for line in reverse(bt_lines)
            if !isempty(strip(line))
                println(io, "  ", line)
            end
        end
    end
end

function Base.showerror(io::IO, e::TVMError)
    show(io, e)
end

