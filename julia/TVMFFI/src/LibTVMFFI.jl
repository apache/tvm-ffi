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
    LibTVMFFI

Low-level C API bindings for TVM FFI.

This module provides direct Julia bindings to the TVM FFI C API.
All functions follow the naming convention from c_api.h.

# Design Notes
- Uses ccall for direct C function invocation
- Matches C struct layouts exactly with Julia struct definitions
- No intermediate abstractions - keep it simple and direct
"""
module LibTVMFFI

# Find the TVM FFI shared library
# This assumes libtvm_ffi.so is in a standard location or LD_LIBRARY_PATH
function _find_libtvm_ffi()
    candidates = [
        "libtvm_ffi.so",  # Linux
        "libtvm_ffi.dylib",  # macOS
        "libtvm_ffi.dll",  # Windows
        # Also check build directory relative to this file
        joinpath(@__DIR__, "../../../build/lib/libtvm_ffi.so"),
        joinpath(@__DIR__, "../../../build/lib/libtvm_ffi.dylib"),
        joinpath(@__DIR__, "../../../build/lib/libtvm_ffi.dll"),
    ]
    
    for lib in candidates
        try
            # Try to dlopen to verify it exists
            if Sys.iswindows()
                # On Windows, just check if file exists
                if isfile(lib)
                    return lib
                end
            else
                # On Unix, try dlopen
                handle = Libc.Libdl.dlopen(lib, Libc.Libdl.RTLD_LAZY)
                if handle != C_NULL
                    Libc.Libdl.dlclose(handle)
                    return lib
                end
            end
        catch
            continue
        end
    end
    
    # Default fallback - let the linker find it
    return "libtvm_ffi"
end

# Module-level constant for the library path
const libtvm_ffi = _find_libtvm_ffi()

# Section: Type definitions matching C API structures

# DLPack types (from dlpack.h, used by TVM)
@enum DLDeviceType::Int32 begin
    kDLCPU = 1
    kDLCUDA = 2
    kDLCUDAHost = 3
    kDLOpenCL = 4
    kDLVulkan = 7
    kDLMetal = 8
    kDLVPI = 9
    kDLROCM = 10
    kDLExtDev = 12
end

"""
    DLDevice

Device context for array execution.

# Fields
- `device_type::Int32`: Device type (CPU, CUDA, etc.)
- `device_id::Int32`: Device ID (e.g., GPU 0, 1, 2...)
"""
struct DLDevice
    device_type::Int32
    device_id::Int32
end

@enum DLDataTypeCode::UInt8 begin
    kDLInt = 0
    kDLUInt = 1
    kDLFloat = 2
    kDLBfloat = 4
    kDLComplex = 5
    kDLBool = 6
    kDLOpaqueHandle = 3
end

"""
    DLDataType

Data type descriptor.

# Fields
- `code::UInt8`: Type code (int, uint, float, etc.)
- `bits::UInt8`: Number of bits
- `lanes::UInt16`: Number of lanes for vector types (1 for scalar)
"""
struct DLDataType
    code::UInt8
    bits::UInt8
    lanes::UInt16
end

# TVM FFI Type Index
# Note: In C, kTVMFFIStaticObjectBegin and kTVMFFIObject both have value 64.
# Julia @enum requires unique values, so we define boundary markers as constants.
@enum TVMFFITypeIndex::Int32 begin
    kTVMFFIAny = -1
    kTVMFFINone = 0
    kTVMFFIInt = 1
    kTVMFFIBool = 2
    kTVMFFIFloat = 3
    kTVMFFIOpaquePtr = 4
    kTVMFFIDataType = 5
    kTVMFFIDevice = 6
    kTVMFFIDLTensorPtr = 7
    kTVMFFIRawStr = 8
    kTVMFFIByteArrayPtr = 9
    kTVMFFIObjectRValueRef = 10
    kTVMFFISmallStr = 11
    kTVMFFISmallBytes = 12
    # Static object types start at 64
    kTVMFFIObject = 64
    kTVMFFIStr = 65
    kTVMFFIBytes = 66
    kTVMFFIError = 67
    kTVMFFIFunction = 68
    kTVMFFIShape = 69
    kTVMFFITensor = 70
    kTVMFFIArray = 71
    kTVMFFIMap = 72
    kTVMFFIModule = 73
    kTVMFFIOpaquePyObject = 74
end

# Boundary markers (not enum members due to value conflicts)
const kTVMFFIStaticObjectBegin = Int32(64)   # Same as kTVMFFIObject
const kTVMFFIStaticObjectEnd = Int32(75)     # After last static type
const kTVMFFIDynObjectBegin = Int32(128)     # Dynamic types start here

# Object handle - opaque pointer
const TVMFFIObjectHandle = Ptr{Cvoid}

"""
    TVMFFIObject

Object header for all heap-allocated TVM FFI objects.
Contains reference counting and type information.
"""
struct TVMFFIObject
    combined_ref_count::UInt64  # Strong ref in lower 32 bits, weak ref in upper 32 bits
    type_index::Int32
    __padding::UInt32
    deleter::Ptr{Cvoid}  # Function pointer or alignment padding
end

"""
    TVMFFIAny

Type-erased value container. Can hold POD types or object references.
"""
struct TVMFFIAny
    type_index::Int32
    small_str_len::UInt32  # Only used for small strings/bytes
    # Union of 8 bytes - Julia doesn't have unions, so we use a single UInt64
    # and provide accessors
    data::UInt64
end

"""
    TVMFFIByteArray

Byte array descriptor used by String and Bytes objects.
"""
struct TVMFFIByteArray
    data::Ptr{UInt8}
    size::UInt
end

"""
    TVMFFIErrorCell

Error cell structure following the object header in error objects.
"""
struct TVMFFIErrorCell
    kind::TVMFFIByteArray
    message::TVMFFIByteArray
    backtrace::TVMFFIByteArray
    update_backtrace::Ptr{Cvoid}  # Function pointer
end

"""
    TVMFFIFunctionCell

Function cell structure following the object header in function objects.
"""
struct TVMFFIFunctionCell
    safe_call::Ptr{Cvoid}  # TVMFFISafeCallType
    cpp_call::Ptr{Cvoid}   # Nullable C++ call pointer
end

# Section: Core C API functions

"""
    TVMFFIObjectIncRef(obj::TVMFFIObjectHandle)

Increase the strong reference count of an object.
"""
function TVMFFIObjectIncRef(obj::TVMFFIObjectHandle)
    ccall((:TVMFFIObjectIncRef, libtvm_ffi), Cint, (TVMFFIObjectHandle,), obj)
end

"""
    TVMFFIObjectDecRef(obj::TVMFFIObjectHandle)

Decrease the strong reference count of an object.
"""
function TVMFFIObjectDecRef(obj::TVMFFIObjectHandle)
    ccall((:TVMFFIObjectDecRef, libtvm_ffi), Cint, (TVMFFIObjectHandle,), obj)
end

"""
    TVMFFIErrorCreate(kind, message, backtrace) -> (Int32, TVMFFIObjectHandle)

Create an error object.
Returns (return_code, error_handle).
"""
function TVMFFIErrorCreate(kind::TVMFFIByteArray, message::TVMFFIByteArray, 
                          backtrace::TVMFFIByteArray)
    out_handle = Ref{TVMFFIObjectHandle}(C_NULL)
    ret = ccall((:TVMFFIErrorCreate, libtvm_ffi), Cint,
                (Ref{TVMFFIByteArray}, Ref{TVMFFIByteArray}, Ref{TVMFFIByteArray}, 
                 Ptr{TVMFFIObjectHandle}),
                Ref(kind), Ref(message), Ref(backtrace), out_handle)
    return ret, out_handle[]
end

"""
    TVMFFIErrorMoveFromRaised() -> TVMFFIObjectHandle

Move the last error from TLS (Thread Local Storage).
"""
function TVMFFIErrorMoveFromRaised()
    out_handle = Ref{TVMFFIObjectHandle}(C_NULL)
    ccall((:TVMFFIErrorMoveFromRaised, libtvm_ffi), Cvoid,
          (Ptr{TVMFFIObjectHandle},), out_handle)
    return out_handle[]
end

"""
    TVMFFIErrorSetRaised(error::TVMFFIObjectHandle)

Set a raised error in TLS.
"""
function TVMFFIErrorSetRaised(error::TVMFFIObjectHandle)
    ccall((:TVMFFIErrorSetRaised, libtvm_ffi), Cvoid,
          (TVMFFIObjectHandle,), error)
end

"""
    TVMFFIFunctionCall(func, args, num_args, result)

Call a TVM function with arguments.
"""
function TVMFFIFunctionCall(func::TVMFFIObjectHandle, args::Ptr{TVMFFIAny}, 
                           num_args::Int32, result::Ptr{TVMFFIAny})
    ccall((:TVMFFIFunctionCall, libtvm_ffi), Cint,
          (TVMFFIObjectHandle, Ptr{TVMFFIAny}, Int32, Ptr{TVMFFIAny}),
          func, args, num_args, result)
end

"""
    TVMFFIFunctionGetGlobal(name) -> (Int32, TVMFFIObjectHandle)

Get a global function by name.
"""
function TVMFFIFunctionGetGlobal(name::TVMFFIByteArray)
    out_handle = Ref{TVMFFIObjectHandle}(C_NULL)
    ret = ccall((:TVMFFIFunctionGetGlobal, libtvm_ffi), Cint,
                (Ref{TVMFFIByteArray}, Ptr{TVMFFIObjectHandle}),
                Ref(name), out_handle)
    return ret, out_handle[]
end

"""
    TVMFFIDataTypeFromString(str) -> (Int32, DLDataType)

Parse a string to DLDataType.
"""
function TVMFFIDataTypeFromString(str::TVMFFIByteArray)
    out_dtype = Ref{DLDataType}()
    ret = ccall((:TVMFFIDataTypeFromString, libtvm_ffi), Cint,
                (Ref{TVMFFIByteArray}, Ptr{DLDataType}),
                Ref(str), out_dtype)
    return ret, out_dtype[]
end

"""
    TVMFFIDataTypeToString(dtype) -> (Int32, TVMFFIAny)

Convert DLDataType to string.
"""
function TVMFFIDataTypeToString(dtype::DLDataType)
    out_any = Ref{TVMFFIAny}(TVMFFIAny(Int32(kTVMFFINone), 0, 0))
    ret = ccall((:TVMFFIDataTypeToString, libtvm_ffi), Cint,
                (Ref{DLDataType}, Ptr{TVMFFIAny}),
                Ref(dtype), out_any)
    return ret, out_any[]
end

"""
    TVMFFIStringFromByteArray(input) -> (Int32, TVMFFIAny)

Create a String from TVMFFIByteArray.
"""
function TVMFFIStringFromByteArray(input::TVMFFIByteArray)
    out_any = Ref{TVMFFIAny}(TVMFFIAny(Int32(kTVMFFINone), 0, 0))
    ret = ccall((:TVMFFIStringFromByteArray, libtvm_ffi), Cint,
                (Ref{TVMFFIByteArray}, Ptr{TVMFFIAny}),
                Ref(input), out_any)
    return ret, out_any[]
end

"""
    TVMFFIBytesFromByteArray(input) -> (Int32, TVMFFIAny)

Create a Bytes from TVMFFIByteArray.
"""
function TVMFFIBytesFromByteArray(input::TVMFFIByteArray)
    out_any = Ref{TVMFFIAny}(TVMFFIAny(Int32(kTVMFFINone), 0, 0))
    ret = ccall((:TVMFFIBytesFromByteArray, libtvm_ffi), Cint,
                (Ref{TVMFFIByteArray}, Ptr{TVMFFIAny}),
                Ref(input), out_any)
    return ret, out_any[]
end

# Section: Inline accessor functions (from C++ inline functions)

"""
    TVMFFIObjectGetTypeIndex(obj::TVMFFIObjectHandle) -> Int32

Get the type index of an object.
"""
function TVMFFIObjectGetTypeIndex(obj::TVMFFIObjectHandle)
    unsafe_load(Ptr{TVMFFIObject}(obj)).type_index
end

"""
    TVMFFIErrorGetCellPtr(obj::TVMFFIObjectHandle) -> Ptr{TVMFFIErrorCell}

Get pointer to the error cell from an error object.
"""
function TVMFFIErrorGetCellPtr(obj::TVMFFIObjectHandle)
    Ptr{TVMFFIErrorCell}(obj + sizeof(TVMFFIObject))
end

"""
    TVMFFIBytesGetByteArrayPtr(obj::TVMFFIObjectHandle) -> Ptr{TVMFFIByteArray}

Get pointer to the byte array from a string or bytes object.
"""
function TVMFFIBytesGetByteArrayPtr(obj::TVMFFIObjectHandle)
    Ptr{TVMFFIByteArray}(obj + sizeof(TVMFFIObject))
end

"""
    TVMFFIFunctionGetCellPtr(obj::TVMFFIObjectHandle) -> Ptr{TVMFFIFunctionCell}

Get pointer to the function cell from a function object.
"""
function TVMFFIFunctionGetCellPtr(obj::TVMFFIObjectHandle)
    Ptr{TVMFFIFunctionCell}(obj + sizeof(TVMFFIObject))
end

end # module LibTVMFFI

