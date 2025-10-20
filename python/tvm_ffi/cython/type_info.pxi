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
    "DataType": "dtype",
}


@dataclasses.dataclass(repr=False)
class TypeSchema:
    """Type schema that describes a TVM FFI type.

    The schema is expressed using a compact JSON-compatible structure
    and can be rendered as a Python typing string with
    :py:meth:`repr`.
    """
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
            assert len(args) in (0, 1), "list must have 0 or 1 argument"
            if args == ():
                self.args = (TypeSchema("Any"),)
        elif origin == "dict":
            assert len(args) in (0, 2), "dict must have 0 or 2 arguments"
            if args == ():
                self.args = (TypeSchema("Any"), TypeSchema("Any"))
        elif origin == "tuple":
            pass  # tuple can have arbitrary number of arguments

    def __repr__(self) -> str:
        return self.repr(ty_map=None)

    @staticmethod
    def from_json_obj(obj: dict[str, Any]) -> "TypeSchema":
        """Construct a :class:`TypeSchema` from a parsed JSON object."""
        assert isinstance(obj, dict) and "type" in obj, obj
        origin = obj["type"]
        origin = _TYPE_SCHEMA_ORIGIN_CONVERTER.get(origin, origin)
        args = obj.get("args", ())
        args = tuple(TypeSchema.from_json_obj(a) for a in args)
        return TypeSchema(origin, args)

    @staticmethod
    def from_json_str(s: str) -> "TypeSchema":
        """Construct a :class:`TypeSchema` from a JSON string."""
        return TypeSchema.from_json_obj(json.loads(s))

    def repr(self, ty_map: "Optional[Callable[[str], str]]" = None) -> str:
        """Render a human-readable representation of this schema.

        Parameters
        ----------
        ty_map : Callable[[str], str], optional
            A mapping function applied to the schema origin name before
            rendering (e.g. map ``"list" -> "Sequence"`` and
            ``"dict" -> "Mapping"``). If ``None``, the raw origin is used.

        Returns
        -------
        str
            A readable string using Python typing syntax. Formats include:
            - Unions as ``"T1 | T2"``
            - Optional as ``"T | None"``
            - Callables as ``"Callable[[arg1, ...], ret]"``
            - Containers as ``"origin[arg1, ...]"``

        Examples
        --------
        .. code-block:: python

            # From JSON emitted by the runtime
            s = TypeSchema.from_json_str('{"type":"Optional","args":[{"type":"int"}]}')
            assert s.repr() == "int | None"

            # Callable where the first arg is return type, remaining are parameters
            s = TypeSchema("Callable", (TypeSchema("int"), TypeSchema("str")))
            assert s.repr() == "Callable[[str], int]"

            # Custom mapping to stdlib typing collections
            def _map(t: str) -> str:
                return {"list": "Sequence", "dict": "Mapping"}.get(t, t)

            s = TypeSchema.from_json_str('{"type":"dict","args":[{"type":"str"},{"type":"int"}]}')
            assert s.repr(_map) == "Mapping[str, int]"

        """
        if ty_map is None:
            origin = self.origin
        else:
            origin = ty_map(self.origin)
        args = [i.repr(ty_map) for i in self.args]
        if origin == "Union":
            return " | ".join(args)
        elif origin == "Optional":
            return args[0] + " | None"
        elif origin == "Callable":
            if not args:
                return "Callable[..., Any]"
            else:
                ret = args[0]
                args = ", ".join(args[1:])
                return f"Callable[[{args}], {ret}]"
        elif not args:
            return origin
        else:
            args = ", ".join(args)
            return f"{origin}[{args}]"


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
