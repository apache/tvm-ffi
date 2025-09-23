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
import warnings
from typing import Any


_CLASS_OBJECT = None


def _set_class_object(cls):
    global _CLASS_OBJECT
    _CLASS_OBJECT = cls


def __object_repr__(obj):
    """Object repr function that can be overridden by assigning to it"""
    return type(obj).__name__ + "(" + str(obj.__ctypes_handle__().value) + ")"


def _new_object(cls):
    """Helper function for pickle"""
    return cls.__new__(cls)


class ObjectConvertible:
    """Base class for all classes that can be converted to object."""

    def asobject(self):
        """Convert value to object"""
        raise NotImplementedError()


class ObjectRValueRef:
    """Represent an RValue ref to an object that can be moved.

    Parameters
    ----------
    obj : tvm.runtime.Object
        The object that this value refers to
    """

    __slots__ = ["obj"]

    def __init__(self, obj):
        self.obj = obj


cdef class Object:
    """Base class of all TVM FFI objects.
    """
    cdef void* chandle

    def __cinit__(self):
        # initialize chandle to NULL to avoid leak in
        # case of error before chandle is set
        self.chandle = NULL

    def __dealloc__(self):
        if self.chandle != NULL:
            CHECK_CALL(TVMFFIObjectDecRef(self.chandle))
            self.chandle = NULL

    def __ctypes_handle__(self):
        return ctypes_handle(self.chandle)

    def __chandle__(self):
        cdef uint64_t chandle = <uint64_t>self.chandle
        return chandle

    def __reduce__(self):
        cls = type(self)
        return (_new_object, (cls,), self.__getstate__())

    def __getstate__(self):
        if _OBJECT_TO_JSON_GRAPH_STR is None:
            raise RuntimeError("ffi.ToJSONGraphString is not registered, make sure build project with extra API")
        if not self.__chandle__() == 0:
            # need to explicit convert to str in case String
            # returned and triggered another infinite recursion in get state
            return {"handle": str(_OBJECT_TO_JSON_GRAPH_STR(self, None))}
        return {"handle": None}

    def __setstate__(self, state):
        # pylint: disable=assigning-non-slot, assignment-from-no-return
        if _OBJECT_FROM_JSON_GRAPH_STR is None:
            raise RuntimeError("ffi.FromJSONGraphString is not registered, make sure build project with extra API")
        handle = state["handle"]
        if handle is not None:
            self.__init_handle_by_constructor__(_OBJECT_FROM_JSON_GRAPH_STR, handle)
        else:
            self.chandle = NULL

    def __repr__(self):
        # exception safety handling for chandle=None
        if self.chandle == NULL:
            return type(self).__name__ + "(chandle=None)"
        return str(__object_repr__(self))

    def __eq__(self, other):
        return self.same_as(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __init_handle_by_constructor__(self, fconstructor, *args):
        """Initialize the handle by calling constructor function.

        Parameters
        ----------
        fconstructor : Function
            Constructor function.

        args: list of objects
            The arguments to the constructor

        Note
        ----
        We have a special calling convention to call constructor functions.
        So the return handle is directly set into the Node object
        instead of creating a new Node.
        """
        # avoid error raised during construction.
        self.chandle = NULL
        cdef void* chandle
        ConstructorCall(
            (<Object>fconstructor).chandle, <PyObject*>args, &chandle, NULL)
        self.chandle = chandle

    def __ffi_init__(self, *args) -> None:
        """Initialize the instance using the ` __init__` method registered on C++ side.

        Parameters
        ----------
        args: list of objects
            The arguments to the constructor
        """
        self.__init_handle_by_constructor__(type(self).__c_ffi_init__, *args)

    def same_as(self, other):
        """Check object identity.

        Parameters
        ----------
        other : object
            The other object to compare against.

        Returns
        -------
        result : bool
             The comparison result.
        """
        if not isinstance(other, Object):
            return False
        return self.chandle == (<Object>other).chandle

    def __hash__(self):
        cdef uint64_t hash_value = <uint64_t>self.chandle
        return hash_value

    def _move(self):
        """Create an RValue reference to the object and mark the object as moved.

        This is a advanced developer API that can be useful when passing an
        unique reference to an Object that you no longer needed to a function.

        A unique reference can trigger copy on write optimization that avoids
        copy when we transform an object.

        Note
        ----
        All the reference of the object becomes invalid after it is moved.
        Be very careful when using this feature.

        Returns
        -------
        rvalue : The rvalue reference.
        """
        return ObjectRValueRef(self)

    def __move_handle_from__(self, other):
        """Move the handle from other to self"""
        self.chandle = (<Object>other).chandle
        (<Object>other).chandle = NULL


cdef class OpaquePyObject(Object):
    """Opaque PyObject container

    This is a helper class to store opaque python objects
    that will be passed to the ffi functions.

    Users do not need to directly create this class.
    """
    def pyobject(self):
        """Get the underlying python object"""
        cdef object obj
        cdef PyObject* py_handle
        py_handle = <PyObject*>(TVMFFIOpaqueObjectGetCellPtr(self.chandle).handle)
        obj = <object>py_handle
        return obj


class PyNativeObject:
    """Base class of all TVM objects that also subclass python's builtin types."""
    __slots__ = []

    def __init_tvm_ffi_object_by_constructor__(self, fconstructor, *args):
        """Initialize the internal tvm_ffi_object by calling constructor function.

        Parameters
        ----------
        fconstructor : Function
            Constructor function.

        args: list of objects
            The arguments to the constructor

        Note
        ----
        We have a special calling convention to call constructor functions.
        So the return object is directly set into the object
        """
        obj = _CLASS_OBJECT.__new__(_CLASS_OBJECT)
        obj.__init_handle_by_constructor__(fconstructor, *args)
        self.__tvm_ffi_object__ = obj


def _object_type_key_to_index(str type_key):
    """get the type index of object class"""
    cdef int32_t tidx
    type_key_arg = ByteArrayArg(c_str(type_key))
    if TVMFFITypeKeyToIndex(type_key_arg.cptr(), &tidx) == 0:
        return tidx
    return None


cdef inline str _type_index_to_key(int32_t tindex):
    """get the type key of object class"""
    cdef const TVMFFITypeInfo* info = TVMFFIGetTypeInfo(tindex)
    cdef const TVMFFIByteArray* type_key
    if info == NULL:
        return "<unknown>"
    type_key = &(info.type_key)
    return bytearray_to_str(type_key)


cdef inline object make_ret_opaque_object(TVMFFIAny result):
    obj = OpaquePyObject.__new__(OpaquePyObject)
    (<Object>obj).chandle = result.v_obj
    return obj.pyobject()

cdef inline object make_fallback_cls_for_type_index(int32_t type_index):
    cdef str type_key = _type_index_to_key(type_index)
    cdef object type_info = _lookup_or_register_type_info_from_type_key(type_key)
    cdef object parent_type_info = type_info.parent_type_info
    assert type_info.type_cls is None

    # Ensure parent classes are created first
    assert parent_type_info is not None
    if parent_type_info.type_cls is None:   # recursively create parent class first
        make_fallback_cls_for_type_index(parent_type_info.type_index)
    assert parent_type_info.type_cls is not None

    # Create `type_info.type_cls` now
    class cls(parent_type_info.type_cls):
        pass
    attrs = dict(cls.__dict__)
    attrs.pop("__dict__", None)
    attrs.pop("__weakref__", None)
    attrs.update({
        "__slots__": (),
        "__tvm_ffi_type_info__": type_info,
        "__name__": type_key.split(".")[-1],
        "__qualname__": type_key,
        "__module__": ".".join(type_key.split(".")[:-1]),
        "__doc__": f"Auto-generated fallback class for {type_key}.\n"
                   "This class is generated because the class is not registered.\n"
                   "Please do not use this class directly, instead register the class\n"
                   "using `register_object` decorator.",
    })
    for field in type_info.fields:
        attrs[field.name] = field.as_property(cls)
    for method in type_info.methods:
        name = method.name
        if name == "__ffi_init__":
            name = "__c_ffi_init__"
        attrs[name] = method.as_callable(cls)
    for name, val in attrs.items():
        setattr(cls, name, val)
    # Update the registry
    type_info.type_cls = cls
    _update_registry(type_index, type_key, type_info, cls)
    return cls


cdef inline object make_ret_object(TVMFFIAny result):
    cdef int32_t type_index
    cdef object cls, obj
    type_index = result.type_index

    if type_index < len(TYPE_INDEX_TO_CLS) and (cls := TYPE_INDEX_TO_CLS[type_index]) is not None:
        if issubclass(cls, PyNativeObject):
            obj = Object.__new__(Object)
            (<Object>obj).chandle = result.v_obj
            return cls.__from_tvm_ffi_object__(cls, obj)
    else:
        # Slow path: object is not found in registered entry
        # In this case create a dummy stub class for future usage.
        # For every unregistered class, this slow path will be triggered only once.
        cls = make_fallback_cls_for_type_index(type_index)
    obj = cls.__new__(cls)
    (<Object>obj).chandle = result.v_obj
    return obj


cdef _get_method_from_method_info(const TVMFFIMethodInfo* method):
    cdef TVMFFIAny result
    CHECK_CALL(TVMFFIAnyViewToOwnedAny(&(method.method), &result))
    return make_ret(result)


cdef _type_info_create_from_type_key(object type_cls, str type_key):
    cdef const TVMFFIFieldInfo* field
    cdef const TVMFFIMethodInfo* method
    cdef const TVMFFITypeInfo* info
    cdef int32_t type_index
    cdef list ancestors = []
    cdef int ancestor
    cdef object fields = []
    cdef object methods = []
    cdef FieldGetter getter
    cdef FieldSetter setter
    cdef ByteArrayArg type_key_arg = ByteArrayArg(c_str(type_key))

    # NOTE: `type_key_arg` must be kept alive until after the call to `TVMFFITypeKeyToIndex`,
    # because Cython doesn't defer the destruction of `type_key_arg` until after the call.
    if TVMFFITypeKeyToIndex(type_key_arg.cptr(), &type_index) != 0:
        raise ValueError(f"Cannot find type key: {type_key}")
    info = TVMFFIGetTypeInfo(type_index)
    for i in range(info.num_fields):
        field = &(info.fields[i])
        getter = FieldGetter.__new__(FieldGetter)
        (<FieldGetter>getter).getter = field.getter
        (<FieldGetter>getter).offset = field.offset
        setter = FieldSetter.__new__(FieldSetter)
        (<FieldSetter>setter).setter = field.setter
        (<FieldSetter>setter).offset = field.offset
        fields.append(
            TypeField(
                name=bytearray_to_str(&field.name),
                doc=bytearray_to_str(&field.doc) if field.doc.size != 0 else None,
                size=field.size,
                offset=field.offset,
                frozen=(field.flags & kTVMFFIFieldFlagBitMaskWritable) == 0,
                getter=getter,
                setter=setter,
            )
        )

    for i in range(info.num_methods):
        method = &(info.methods[i])
        methods.append(
            TypeMethod(
                name=bytearray_to_str(&method.name),
                doc=bytearray_to_str(&method.doc) if method.doc.size != 0 else None,
                func=_get_method_from_method_info(method),
                is_static=(method.flags & kTVMFFIFieldFlagBitMaskIsStaticMethod) != 0,
            )
        )

    for i in range(info.type_depth):
        ancestor = info.type_ancestors[i].type_index
        ancestors.append(ancestor)

    return TypeInfo(
        type_cls=type_cls,
        type_index=type_index,
        type_key=bytearray_to_str(&info.type_key),
        type_ancestors=ancestors,
        fields=fields,
        methods=methods,
        parent_type_info=None,
    )


cdef _update_registry(int type_index, object type_key, object type_info, object type_cls):
    cdef int extra = type_index + 1 - len(TYPE_INDEX_TO_INFO)
    assert len(TYPE_INDEX_TO_INFO) == len(TYPE_INDEX_TO_CLS)
    if extra > 0:
        TYPE_INDEX_TO_INFO.extend([None] * extra)
        TYPE_INDEX_TO_CLS.extend([None] * extra)
    TYPE_INDEX_TO_CLS[type_index] = type_cls
    TYPE_INDEX_TO_INFO[type_index] = type_info
    TYPE_KEY_TO_INFO[type_key] = type_info


def _register_object_by_index(int type_index, object type_cls):
    global TYPE_INDEX_TO_INFO, TYPE_KEY_TO_INFO, TYPE_INDEX_TO_CLS
    cdef str type_key = _type_index_to_key(type_index)
    cdef object info = _type_info_create_from_type_key(type_cls, type_key)
    _update_registry(type_index, type_key, info, type_cls)
    return info


def _set_type_cls(object type_info, object type_cls):
    global TYPE_INDEX_TO_INFO, TYPE_INDEX_TO_CLS
    assert type_info.type_cls is None, f"Type already registered for {type_info.type_key}"
    assert TYPE_INDEX_TO_INFO[type_info.type_index] is type_info
    assert TYPE_KEY_TO_INFO[type_info.type_key] is type_info
    type_info.type_cls = type_cls
    TYPE_INDEX_TO_CLS[type_info.type_index] = type_cls


def _lookup_or_register_type_info_from_type_key(type_key: str) -> TypeInfo:
    if info := TYPE_KEY_TO_INFO.get(type_key, None):
        return info
    info = _type_info_create_from_type_key(None, type_key)
    _update_registry(info.type_index, type_key, info, None)
    return info


cdef list TYPE_INDEX_TO_CLS = []
cdef list TYPE_INDEX_TO_INFO = []
cdef dict TYPE_KEY_TO_INFO = {}

_set_class_object(Object)
