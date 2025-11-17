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
    TVMObject

Base wrapper for TVM FFI object handles with automatic memory management.
"""
mutable struct TVMObject
    handle::LibTVMFFI.TVMFFIObjectHandle

    """
        TVMObject(handle; own=true)

    Create a TVMObject from a raw handle.

    # Arguments
    - `handle`: The raw object handle
    - `own`: If true, increment refcount (default). If false, take ownership without IncRef.
    """
    function TVMObject(handle::LibTVMFFI.TVMFFIObjectHandle; own::Bool=true)
        if handle == C_NULL
            error("Cannot create TVMObject from NULL handle")
        end

        # Optionally increase reference count
        if own
            LibTVMFFI.TVMFFIObjectIncRef(handle)
        end

        obj = new(handle)

        # Finalizer to decrease ref count when GC collects this
        finalizer(obj) do o
            if o.handle != C_NULL
                LibTVMFFI.TVMFFIObjectDecRef(o.handle)
            end
        end

        return obj
    end
end

"""
    type_index(obj::TVMObject) -> Int32

Get the type index of an object.
"""
function type_index(obj::TVMObject)
    LibTVMFFI.TVMFFIObjectGetTypeIndex(obj.handle)
end

"""
    is_type(obj::TVMObject, idx::LibTVMFFI.TVMFFITypeIndex) -> Bool

Check if an object has a specific type index.
"""
function is_type(obj::TVMObject, idx::LibTVMFFI.TVMFFITypeIndex)
    type_index(obj) == Int32(idx)
end

Base.show(io::IO, obj::TVMObject) = print(io, "TVMObject(type_index=", type_index(obj), ")")
