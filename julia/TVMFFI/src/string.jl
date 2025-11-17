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
    TVMString

TVM ABI-stable string type.

This wraps the TVM FFI string representation which can be either
a small string (≤7 bytes, stored inline) or a heap-allocated object.

# Design Notes
Unlike Rust, Julia doesn't need manual memory management here.
We use a simpler approach: store the TVMFFIAny directly and use
finalizers for cleanup.
"""
mutable struct TVMString
    data::LibTVMFFI.TVMFFIAny
    
    function TVMString(s::AbstractString)
        str = String(s)  # Convert to Julia String
        byte_array = LibTVMFFI.TVMFFIByteArray(pointer(str), sizeof(str))
        
        ret, any_result = LibTVMFFI.TVMFFIStringFromByteArray(byte_array)
        check_call(ret)
        
        tvmstr = new(any_result)
        
        # Only need finalizer for heap-allocated strings
        if any_result.type_index >= Int32(LibTVMFFI.kTVMFFIStaticObjectBegin)
            finalizer(tvmstr) do s
                # Decrease reference count
                obj_ptr = reinterpret(LibTVMFFI.TVMFFIObjectHandle, s.data.data)
                if obj_ptr != C_NULL
                    LibTVMFFI.TVMFFIObjectDecRef(obj_ptr)
                end
            end
        end
        
        return tvmstr
    end
    
    # Constructor from TVMFFIAny (internal use)
    # Note: Takes ownership without IncRef by default (C API returns new reference)
    function TVMString(any::LibTVMFFI.TVMFFIAny; own::Bool=false)
        tvmstr = new(any)
        
        # Add finalizer for heap objects
        if any.type_index >= Int32(LibTVMFFI.kTVMFFIStaticObjectBegin)
            obj_ptr = reinterpret(LibTVMFFI.TVMFFIObjectHandle, any.data)
            if obj_ptr != C_NULL
                # Optionally increase ref count
                if own
                    LibTVMFFI.TVMFFIObjectIncRef(obj_ptr)
                end
            end
            
            finalizer(tvmstr) do s
                obj_ptr = reinterpret(LibTVMFFI.TVMFFIObjectHandle, s.data.data)
                if obj_ptr != C_NULL
                    LibTVMFFI.TVMFFIObjectDecRef(obj_ptr)
                end
            end
        end
        
        return tvmstr
    end
end

"""
    Base.String(s::TVMString) -> String

Convert TVMString to Julia String.
"""
function Base.String(s::TVMString)
    if s.data.type_index == Int32(LibTVMFFI.kTVMFFISmallStr)
        # Small string optimization - data stored inline in the union
        str_len = Int(s.data.small_str_len)
        if str_len == 0
            return ""
        end
        # The bytes are stored in the UInt64 data field
        # Extract them by reinterpreting the UInt64 as bytes
        bytes_vec = collect(reinterpret(UInt8, [s.data.data]))
        # Take only the first str_len bytes and convert to String
        return String(copy(bytes_vec[1:str_len]))
    elseif s.data.type_index == Int32(LibTVMFFI.kTVMFFIStr)
        # Heap-allocated string
        obj_handle = reinterpret(LibTVMFFI.TVMFFIObjectHandle, s.data.data)
        byte_array_ptr = LibTVMFFI.TVMFFIBytesGetByteArrayPtr(obj_handle)
        byte_array = unsafe_load(byte_array_ptr)
        return unsafe_string(byte_array.data, byte_array.size)
    else
        error("Invalid TVMString type index: $(s.data.type_index)")
    end
end

Base.length(s::TVMString) = length(String(s))
Base.sizeof(s::TVMString) = sizeof(String(s))
Base.iterate(s::TVMString) = iterate(String(s))
Base.iterate(s::TVMString, state) = iterate(String(s), state)
Base.getindex(s::TVMString, i) = getindex(String(s), i)

Base.show(io::IO, s::TVMString) = print(io, "TVMString(\"", String(s), "\")")
Base.print(io::IO, s::TVMString) = print(io, String(s))

"""
    TVMBytes

TVM ABI-stable bytes type (for binary data).

Similar to TVMString but for arbitrary binary data.
"""
mutable struct TVMBytes
    data::LibTVMFFI.TVMFFIAny
    
    function TVMBytes(bytes::Vector{UInt8})
        byte_array = LibTVMFFI.TVMFFIByteArray(pointer(bytes), length(bytes))
        
        ret, any_result = LibTVMFFI.TVMFFIBytesFromByteArray(byte_array)
        check_call(ret)
        
        tvmbytes = new(any_result)
        
        # Only need finalizer for heap-allocated bytes
        if any_result.type_index >= Int32(LibTVMFFI.kTVMFFIStaticObjectBegin)
            finalizer(tvmbytes) do b
                obj_ptr = reinterpret(LibTVMFFI.TVMFFIObjectHandle, b.data.data)
                if obj_ptr != C_NULL
                    LibTVMFFI.TVMFFIObjectDecRef(obj_ptr)
                end
            end
        end
        
        return tvmbytes
    end
    
    # Constructor from TVMFFIAny
    # Note: Takes ownership without IncRef by default (C API returns new reference)
    function TVMBytes(any::LibTVMFFI.TVMFFIAny; own::Bool=false)
        tvmbytes = new(any)
        
        if any.type_index >= Int32(LibTVMFFI.kTVMFFIStaticObjectBegin)
            obj_ptr = reinterpret(LibTVMFFI.TVMFFIObjectHandle, any.data)
            if obj_ptr != C_NULL
                # Optionally increase ref count
                if own
                    LibTVMFFI.TVMFFIObjectIncRef(obj_ptr)
                end
            end
            
            finalizer(tvmbytes) do b
                obj_ptr = reinterpret(LibTVMFFI.TVMFFIObjectHandle, b.data.data)
                if obj_ptr != C_NULL
                    LibTVMFFI.TVMFFIObjectDecRef(obj_ptr)
                end
            end
        end
        
        return tvmbytes
    end
end

"""
    Vector{UInt8}(b::TVMBytes) -> Vector{UInt8}

Convert TVMBytes to Julia byte vector.
"""
function Base.Vector{UInt8}(b::TVMBytes)
    if b.data.type_index == Int32(LibTVMFFI.kTVMFFISmallBytes)
        # Small bytes - stored inline
        byte_len = Int(b.data.small_str_len)
        if byte_len == 0
            return UInt8[]
        end
        # Extract bytes by reinterpreting the UInt64 as bytes
        bytes_vec = collect(reinterpret(UInt8, [b.data.data]))
        return copy(bytes_vec[1:byte_len])
    elseif b.data.type_index == Int32(LibTVMFFI.kTVMFFIBytes)
        # Heap-allocated bytes
        obj_handle = reinterpret(LibTVMFFI.TVMFFIObjectHandle, b.data.data)
        byte_array_ptr = LibTVMFFI.TVMFFIBytesGetByteArrayPtr(obj_handle)
        byte_array = unsafe_load(byte_array_ptr)
        # Copy to Julia array
        return unsafe_wrap(Array, byte_array.data, byte_array.size) |> copy
    else
        error("Invalid TVMBytes type index: $(b.data.type_index)")
    end
end

Base.length(b::TVMBytes) = length(Vector{UInt8}(b))
Base.sizeof(b::TVMBytes) = sizeof(Vector{UInt8}(b))

Base.show(io::IO, b::TVMBytes) = print(io, "TVMBytes(", Vector{UInt8}(b), ")")
