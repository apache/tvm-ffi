# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# Extern declarations from tvm_ffi_type_converter.h
cdef extern from "tvm_ffi_type_converter.h":
    cdef enum TVMFFIPyTypeConverterKind:
        kTCKAny = 0
        kTCKNone = 1
        kTCKInt = 2
        kTCKBool = 3
        kTCKFloat = 4
        kTCKDataType = 5
        kTCKDevice = 6
        kTCKOpaquePtr = 7
        kTCKString = 8
        kTCKBytes = 9
        kTCKTensor = 10
        kTCKFunction = 11
        kTCKObject = 12
        kTCKOpaquePyObject = 13
        kTCKOptional = 14
        kTCKUnion = 15
        kTCKArray = 16
        kTCKList = 17
        kTCKMap = 18
        kTCKTuple = 19

    ctypedef struct TVMFFIPyTypeConverterNode:
        pass

    ctypedef struct TVMFFIPyTypeConverter:
        pass

    TVMFFIPyTypeConverterNode* TVMFFIPyTypeConverterNodeCreate(
        int kind, int32_t type_index, const char* type_repr, int type_repr_len,
        TVMFFIPyTypeConverterNode** children, int num_children
    )
    void TVMFFIPyTypeConverterNodeDelete(TVMFFIPyTypeConverterNode* node)
    TVMFFIPyTypeConverter* TVMFFIPyTypeConverterCreate(
        TVMFFIPyTypeConverterNode* root,
        TVMFFIObjectHandle array_ctor,
        TVMFFIObjectHandle list_ctor,
        TVMFFIObjectHandle map_ctor,
        TVMFFIObjectHandle map_iter_func,
        TVMFFIObjectHandle map_size_func,
    )
    int TVMFFIPyTypeConverterSafeCall(
        void* self, const TVMFFIAny* c_args, int32_t num_args, TVMFFIAny* rv
    ) noexcept
    void TVMFFIPyTypeConverterDeleter(void* self) noexcept


# Origin string to converter kind mapping
_ORIGIN_TO_TC_KIND = {
    "Any": kTCKAny,
    "None": kTCKNone,
    "int": kTCKInt,
    "bool": kTCKBool,
    "float": kTCKFloat,
    "dtype": kTCKDataType,
    "Device": kTCKDevice,
    "ctypes.c_void_p": kTCKOpaquePtr,
    "str": kTCKString,
    "bytes": kTCKBytes,
    "Tensor": kTCKTensor,
    "Callable": kTCKFunction,
    "Object": kTCKObject,
    "Optional": kTCKOptional,
    "Union": kTCKUnion,
    "list": kTCKArray,
    "dict": kTCKMap,
    "tuple": kTCKTuple,
}


cdef TVMFFIPyTypeConverterNode* _build_converter_node(object schema) except NULL:
    """Build a converter node tree from a TypeSchema."""
    cdef str origin = schema.origin
    cdef tuple args = schema.args
    cdef int kind
    cdef int32_t type_index = 0
    cdef int32_t tindex
    cdef int num_children
    cdef vector[TVMFFIPyTypeConverterNode*] children_vec
    cdef int i
    cdef str type_repr_str
    cdef bytes type_repr_bytes
    cdef const char* type_repr_ptr
    cdef int type_repr_len
    cdef TVMFFIPyTypeConverterNode* node

    # Look up the kind
    kind_val = _ORIGIN_TO_TC_KIND.get(origin)
    if kind_val is not None:
        kind = <int>kind_val
        if kind == kTCKObject and origin == "Object":
            # Generic Object: use kTVMFFIObject as the type_index
            type_index = kTVMFFIObject
    else:
        # Unknown origin: try to resolve as a type key
        type_key_arg = ByteArrayArg(c_str(origin))
        ret = TVMFFITypeKeyToIndex((<ByteArrayArg>type_key_arg).cptr(), &tindex)
        if ret == 0:
            kind = kTCKObject
            type_index = tindex
        else:
            raise ValueError(f"Unknown type origin in TypeSchema: {origin}")

    # Build children recursively
    num_children = len(args)
    children_vec.resize(num_children)
    for i in range(num_children):
        children_vec[i] = _build_converter_node(args[i])
        if children_vec[i] == NULL:
            # Cleanup previously allocated children
            for j in range(i):
                TVMFFIPyTypeConverterNodeDelete(children_vec[j])
            return NULL

    # Get type_repr for error messages
    type_repr_str = repr(schema)
    type_repr_bytes = type_repr_str.encode("utf-8")
    type_repr_ptr = type_repr_bytes
    type_repr_len = len(type_repr_bytes)

    node = TVMFFIPyTypeConverterNodeCreate(
        kind, type_index, type_repr_ptr, type_repr_len,
        children_vec.data() if num_children > 0 else NULL,
        num_children
    )
    return node


# Cached global function handles for container constructors
cdef Function _CONSTRUCTOR_LIST = _get_global_func("ffi.List", True)
cdef Function _MAP_FORWARD_ITER_FUNC = _get_global_func("ffi.MapForwardIterFunctor", True)
cdef Function _MAP_SIZE_FUNC = _get_global_func("ffi.MapSize", True)


def create_type_converter(schema) -> Function:
    """Create a type converter function from a TypeSchema.

    Parameters
    ----------
    schema : TypeSchema
        The type schema describing the target type.

    Returns
    -------
    Function
        A function ``(Any) -> Any`` that validates/converts input values
        to match the target type.
    """
    cdef TVMFFIPyTypeConverterNode* root = _build_converter_node(schema)

    # Get constructor handles and incref for converter ownership
    cdef TVMFFIObjectHandle array_ctor = (<Object>_CONSTRUCTOR_ARRAY).chandle
    cdef TVMFFIObjectHandle list_ctor = (<Object>_CONSTRUCTOR_LIST).chandle
    cdef TVMFFIObjectHandle map_ctor = (<Object>_CONSTRUCTOR_MAP).chandle
    cdef TVMFFIObjectHandle map_iter_func = (<Object>_MAP_FORWARD_ITER_FUNC).chandle
    cdef TVMFFIObjectHandle map_size_func = (<Object>_MAP_SIZE_FUNC).chandle

    TVMFFIObjectIncRef(array_ctor)
    TVMFFIObjectIncRef(list_ctor)
    TVMFFIObjectIncRef(map_ctor)
    TVMFFIObjectIncRef(map_iter_func)
    TVMFFIObjectIncRef(map_size_func)

    cdef TVMFFIPyTypeConverter* converter = TVMFFIPyTypeConverterCreate(
        root, array_ctor, list_ctor, map_ctor, map_iter_func, map_size_func
    )

    # Create the Function wrapping the converter
    cdef TVMFFIObjectHandle func_handle
    cdef int ret_code = TVMFFIFunctionCreate(
        <void*>converter,
        TVMFFIPyTypeConverterSafeCall,
        TVMFFIPyTypeConverterDeleter,
        &func_handle
    )
    if ret_code != 0:
        TVMFFIPyTypeConverterDeleter(<void*>converter)
        CHECK_CALL(ret_code)

    func = Function.__new__(Function)
    (<Object>func).chandle = func_handle
    return func
