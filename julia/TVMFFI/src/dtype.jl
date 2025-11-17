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
    DLDataType

Re-export from LibTVMFFI for convenience.
Represents a data type in DLPack format.
"""
const DLDataType = LibTVMFFI.DLDataType
const DLDataTypeCode = LibTVMFFI.DLDataTypeCode

"""
    Base.string(dtype::DLDataType) -> String

Convert DLDataType to string representation (e.g., "int32", "float64").
"""
function Base.string(dtype::DLDataType)
    ret, any_result = LibTVMFFI.TVMFFIDataTypeToString(dtype)
    check_call(ret)

    # Extract string from TVMFFIAny
    # C API returns a new reference, so we take ownership
    if any_result.type_index == Int32(LibTVMFFI.kTVMFFISmallStr)
        # Small string optimization - no refcounting needed
        str_len = Int(any_result.small_str_len)
        if str_len == 0
            return ""
        end
        # Extract bytes by reinterpreting the UInt64
        bytes_vec = collect(reinterpret(UInt8, [any_result.data]))
        return String(copy(bytes_vec[1:str_len]))
    elseif any_result.type_index == Int32(LibTVMFFI.kTVMFFIStr)
        # Heap-allocated string object
        # Create TVMString with own=false to take ownership
        tvmstr = TVMString(any_result; own=false)
        return String(tvmstr)
    else
        error("Unexpected type index from TVMFFIDataTypeToString: $(any_result.type_index)")
    end
end

"""
    DLDataType(dtype_str::AbstractString) -> DLDataType

Parse a string to create a DLDataType.

# Examples
```julia
dt = DLDataType("int32")
dt = DLDataType("float64")
dt = DLDataType("bool")
```
"""
function DLDataType(dtype_str::AbstractString)
    byte_array = LibTVMFFI.TVMFFIByteArray(pointer(dtype_str), sizeof(dtype_str))
    ret, dtype = LibTVMFFI.TVMFFIDataTypeFromString(byte_array)
    check_call(ret)
    return dtype
end

"""
    DLDataType(::Type{T}) where T -> DLDataType

Get the DLDataType corresponding to a Julia type.
"""
function DLDataType(::Type{Int8})
    DLDataType(UInt8(LibTVMFFI.kDLInt), UInt8(8), UInt16(1))
end

function DLDataType(::Type{Int16})
    DLDataType(UInt8(LibTVMFFI.kDLInt), UInt8(16), UInt16(1))
end

function DLDataType(::Type{Int32})
    DLDataType(UInt8(LibTVMFFI.kDLInt), UInt8(32), UInt16(1))
end

function DLDataType(::Type{Int64})
    DLDataType(UInt8(LibTVMFFI.kDLInt), UInt8(64), UInt16(1))
end

function DLDataType(::Type{UInt8})
    DLDataType(UInt8(LibTVMFFI.kDLUInt), UInt8(8), UInt16(1))
end

function DLDataType(::Type{UInt16})
    DLDataType(UInt8(LibTVMFFI.kDLUInt), UInt8(16), UInt16(1))
end

function DLDataType(::Type{UInt32})
    DLDataType(UInt8(LibTVMFFI.kDLUInt), UInt8(32), UInt16(1))
end

function DLDataType(::Type{UInt64})
    DLDataType(UInt8(LibTVMFFI.kDLUInt), UInt8(64), UInt16(1))
end

function DLDataType(::Type{Float32})
    DLDataType(UInt8(LibTVMFFI.kDLFloat), UInt8(32), UInt16(1))
end

function DLDataType(::Type{Float64})
    DLDataType(UInt8(LibTVMFFI.kDLFloat), UInt8(64), UInt16(1))
end

function DLDataType(::Type{Bool})
    DLDataType(UInt8(LibTVMFFI.kDLBool), UInt8(8), UInt16(1))
end

# Pretty printing
function Base.show(io::IO, dtype::DLDataType)
    print(io, "DLDataType(", string(dtype), ")")
end
