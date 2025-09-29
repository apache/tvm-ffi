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
import dataclasses
import json
from typing import Optional, Any
from io import StringIO


cdef class FieldGetter:
    cdef dict __dict__
    cdef TVMFFIFieldGetter getter
    cdef int64_t offset

    def __call__(self, Object obj):
        cdef TVMFFIAny result
        cdef int c_api_ret_code
        cdef void* field_ptr = (<char*>(<Object>obj).chandle) + self.offset
        result.type_index = kTVMFFINone
        result.v_int64 = 0
        c_api_ret_code = self.getter(field_ptr, &result)
        CHECK_CALL(c_api_ret_code)
        return make_ret(result)


cdef class FieldSetter:
    cdef dict __dict__
    cdef TVMFFIFieldSetter setter
    cdef int64_t offset

    def __call__(self, Object obj, value):
        cdef int c_api_ret_code
        cdef void* field_ptr = (<char*>(<Object>obj).chandle) + self.offset
        TVMFFIPyCallFieldSetter(
            TVMFFIPyArgSetterFactory_,
            self.setter,
            field_ptr,
            <PyObject*>value,
            &c_api_ret_code
        )
        # NOTE: logic is same as check_call
        # directly inline here to simplify backtrace
        if c_api_ret_code == 0:
            return
        elif c_api_ret_code == -2:
            raise_existing_error()
        raise move_from_last_error().py_error()

_TYPE_SCHEMA_ORIGIN_CONVERTER = {
    # A few Python-native types
    "Variant": "Union",
    "Optional": "Optional",
    "Tuple": "tuple",
    "ffi.Function": "Callable",
    "ffi.Array": "list",
    "ffi.Map": "dict",
    "ffi.OpaquePyObject": "Any",
    "ffi.Object": "Object",
    "ffi.Tensor": "Tensor",
    "DLTensor*": "Tensor",
    # ctype types
    "void*": "ctypes.c_void_p",
    # bytes
    "TVMFFIByteArray*": "bytes",
    "ffi.SmallBytes": "bytes",
    "ffi.Bytes": "bytes",
    # strings
    "std::string": "str",
    "const char*": "str",
    "ffi.SmallStr": "str",
    "ffi.String": "str",
}


@dataclasses.dataclass(repr=False, frozen=True)
class TypeSchema:
    """Type schema for a TVM FFI type."""
    origin: str
    args: tuple[TypeSchema, ...] = ()

    def __post_init__(self):
        origin = self.origin
        args = self.args
        if origin == "Union":
            assert len(args) >= 2, "Union must have at least two arguments"
        elif origin == "Optional":
            assert len(args) == 1, "Optional must have exactly one argument"
        elif origin == "list":
            assert len(args) == 1, "list must have exactly one argument"
        elif origin == "dict":
            assert len(args) == 2, "dict must have exactly two arguments"
        elif origin == "tuple":
            pass  # tuple can have arbitrary number of arguments

    def __repr__(self) -> str:
        if self.origin == "Union":
            return " | ".join(repr(a) for a in self.args)
        elif self.origin == "Optional":
            return repr(self.args[0]) + " | None"
        elif self.origin == "Callable":
            if not self.args:
                return "Callable[..., Any]"
            else:
                arg_ret = self.args[0]
                arg_args = self.args[1:]
                return f"Callable[[{', '.join(repr(a) for a in arg_args)}], {repr(arg_ret)}]"
        elif not self.args:
            return self.origin
        else:
            return f"{self.origin}[{', '.join(repr(a) for a in self.args)}]"

    @staticmethod
    def from_json_obj(obj: dict[str, Any]) -> "TypeSchema":
        assert isinstance(obj, dict) and "type" in obj, obj
        origin = obj["type"]
        origin = _TYPE_SCHEMA_ORIGIN_CONVERTER.get(origin, origin)
        args = obj.get("args", ())
        args = tuple(TypeSchema.from_json_obj(a) for a in args)
        return TypeSchema(origin, args)

    @staticmethod
    def from_json_str(s) -> "TypeSchema":
        return TypeSchema.from_json_obj(json.loads(s))


@dataclasses.dataclass(eq=False)
class TypeField:
    """Description of a single reflected field on an FFI-backed type."""

    name: str
    doc: Optional[str]
    size: int
    offset: int
    frozen: bool
    metadata: dict[str, Any]
    getter: FieldGetter
    setter: FieldSetter
    dataclass_field: Any = None

    def __post_init__(self):
        assert self.setter is not None
        assert self.getter is not None

    def as_property(self, object cls):
        """Create a Python ``property`` object for this field on ``cls``."""
        cdef str name = self.name
        cdef FieldGetter fget = self.getter
        cdef FieldSetter fset = self.setter
        cdef object ret
        fget.__name__ = fset.__name__ = name
        fget.__module__ = fset.__module__ = cls.__module__
        fget.__qualname__ = fset.__qualname__ = f"{cls.__qualname__}.{name}"
        ret = property(
            fget=fget,
            fset=fset if (not self.frozen) else None,
        )
        if self.doc:
            ret.__doc__ = self.doc
            fget.__doc__ = self.doc
            fset.__doc__ = self.doc
        return ret


@dataclasses.dataclass(eq=False)
class TypeMethod:
    """Description of a single reflected method on an FFI-backed type."""

    name: str
    doc: Optional[str]
    func: object
    metadata: dict[str, Any]
    is_static: bool

    def __post_init__(self):
        assert callable(self.func)

    def as_callable(self, object cls):
        """Create a Python method attribute for this method on ``cls``."""
        cdef str name = self.name
        cdef object func = self.func
        if not self.is_static:
            func = _member_method_wrapper(func)
        func.__module__ = cls.__module__
        func.__name__ = name
        func.__qualname__ = f"{cls.__qualname__}.{name}"
        if self.doc:
            func.__doc__ = self.doc
        if self.is_static:
            func = staticmethod(func)
        return func


@dataclasses.dataclass(eq=False)
class TypeInfo:
    """Aggregated type information required to build a proxy class."""

    type_cls: Optional[type]
    type_index: int
    type_key: str
    type_ancestors: list[int]
    fields: list[TypeField]
    methods: list[TypeMethod]
    parent_type_info: Optional[TypeInfo]

    def __post_init__(self):
        cdef int parent_type_index
        cdef str parent_type_key
        if not self.type_ancestors:
            return
        parent_type_index = self.type_ancestors[-1]
        parent_type_key = _type_index_to_key(parent_type_index)
        # ensure parent is registered
        self.parent_type_info = _lookup_or_register_type_info_from_type_key(parent_type_key)


def _member_method_wrapper(method_func: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(self: Any, *args: Any) -> Any:
        return method_func(self, *args)

    return wrapper
