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
from functools import cached_property
from typing import Optional, Any
from io import StringIO


cdef class FieldGetter:
    cdef dict __dict__
    cdef TVMFFIFieldGetter getter
    cdef int64_t offset

    def __call__(self, CObject obj):
        cdef TVMFFIAny result
        cdef int c_api_ret_code
        cdef void* field_ptr = (<char*>(<CObject>obj).chandle) + self.offset
        result.type_index = kTVMFFINone
        result.v_int64 = 0
        c_api_ret_code = self.getter(field_ptr, &result)
        CHECK_CALL(c_api_ret_code)
        return make_ret(result)


cdef class FieldSetter:
    cdef dict __dict__
    cdef TVMFFIFieldSetter setter
    cdef int64_t offset

    def __call__(self, CObject obj, value):
        cdef int c_api_ret_code
        cdef void* field_ptr = (<char*>(<CObject>obj).chandle) + self.offset
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
        # backward compact with error already set case
        # TODO(tqchen): remove after we move beyond a few versions.
        if c_api_ret_code == -2:
            raise raise_existing_error()
        # epecial handle env error already set
        error = move_from_last_error()
        if error.kind == "EnvErrorAlreadySet":
            raise raise_existing_error()
        raise error.py_error()


_TYPE_SCHEMA_ORIGIN_CONVERTER = {
    # A few Python-native types
    "Variant": "Union",
    "Optional": "Optional",
    "Tuple": "tuple",
    "ffi.Function": "Callable",
    "ffi.Array": "Array",
    "ffi.List": "List",
    "ffi.Map": "Map",
    "ffi.Dict": "Dict",
    # OpaquePyObject accepts any Python value at the FFI boundary (the C++
    # side wraps it opaquely), so mapping to "Any" is semantically correct.
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
    # C++ STL types (emitted by TypeTraits in include/tvm/ffi/extra/stl.h)
    "std::vector": "Array",
    "std::optional": "Optional",
    "std::variant": "Union",
    "std::tuple": "tuple",
    "std::map": "Map",
    "std::unordered_map": "Map",
    "std::function": "Callable",
    # Rvalue reference (C++ move semantics).  Python has no move semantics,
    # so the checker treats it as a plain Object reference.
    "ObjectRValueRef": "Object",
}

# Sentinel for structural types (Optional, Union) that have no single type_index
_ORIGIN_TYPE_INDEX_STRUCTURAL = -2
# Sentinel for unknown/unresolved origins
_ORIGIN_TYPE_INDEX_UNKNOWN = -3

# Map origin string -> type_index for known types
_ORIGIN_TO_TYPE_INDEX = {
    "None": kTVMFFINone,
    "int": kTVMFFIInt,
    "bool": kTVMFFIBool,
    "float": kTVMFFIFloat,
    "str": kTVMFFIStr,
    "bytes": kTVMFFIBytes,
    "Device": kTVMFFIDevice,
    "dtype": kTVMFFIDataType,
    "ctypes.c_void_p": kTVMFFIOpaquePtr,
    "Tensor": kTVMFFITensor,
    "Object": kTVMFFIObject,
    "Callable": kTVMFFIFunction,
    "Array": kTVMFFIArray,
    "List": kTVMFFIList,
    "Map": kTVMFFIMap,
    "Dict": kTVMFFIDict,
    "Any": kTVMFFIAny,
}

# Reverse map: type_index -> origin string
_TYPE_INDEX_TO_ORIGIN = {v: k for k, v in _ORIGIN_TO_TYPE_INDEX.items()}
# Low-level type indices that alias canonical origins
_TYPE_INDEX_TO_ORIGIN[kTVMFFIDLTensorPtr] = "Tensor"
_TYPE_INDEX_TO_ORIGIN[kTVMFFIRawStr] = "str"
_TYPE_INDEX_TO_ORIGIN[kTVMFFIByteArrayPtr] = "bytes"
_TYPE_INDEX_TO_ORIGIN[kTVMFFISmallStr] = "str"
_TYPE_INDEX_TO_ORIGIN[kTVMFFISmallBytes] = "bytes"
_TYPE_INDEX_TO_ORIGIN[kTVMFFIObjectRValueRef] = "Object"


@dataclasses.dataclass(repr=False)
class TypeSchema:
    """Type schema that describes a TVM FFI type.

    The schema is expressed using a compact JSON-compatible structure
    and can be rendered as a Python typing string with
    :py:meth:`repr`.
    """
    origin: str
    args: tuple[TypeSchema, ...] = ()
    origin_type_index: int = dataclasses.field(default=_ORIGIN_TYPE_INDEX_UNKNOWN, repr=False)

    def __post_init__(self):
        origin = self.origin
        args = self.args
        if origin == "Union":
            if len(args) < 2:
                raise ValueError("Union must have at least two arguments")
        elif origin == "Optional":
            if len(args) != 1:
                raise ValueError("Optional must have exactly one argument")
        elif origin in ("list", "Array", "List"):
            if len(args) not in (0, 1):
                raise ValueError(f"{origin} must have 0 or 1 argument")
            if args == ():
                self.args = (TypeSchema("Any"),)
        elif origin in ("dict", "Map", "Dict"):
            if len(args) not in (0, 2):
                raise ValueError(f"{origin} must have 0 or 2 arguments")
            if args == ():
                self.args = (TypeSchema("Any"), TypeSchema("Any"))
        elif origin == "tuple":
            pass  # tuple can have arbitrary number of arguments
        # Compute origin_type_index if not already set
        if self.origin_type_index == _ORIGIN_TYPE_INDEX_UNKNOWN:
            if origin in ("Optional", "Union"):
                self.origin_type_index = _ORIGIN_TYPE_INDEX_STRUCTURAL
            elif origin in _ORIGIN_TO_TYPE_INDEX:
                self.origin_type_index = _ORIGIN_TO_TYPE_INDEX[origin]
            else:
                # Try to resolve as a registered object type key
                tindex = _object_type_key_to_index(origin)
                if tindex is not None:
                    self.origin_type_index = tindex

    @cached_property
    def _converter(self):
        """Lazily build the type converter on first use.

        Deferred construction ensures all object types are registered
        by the time the converter is built. Raises TypeError for
        unresolvable origins.
        """
        return _build_converter(self)

    def __repr__(self) -> str:
        return self.repr(ty_map=None)

    @staticmethod
    def from_json_obj(obj: dict[str, Any]) -> "TypeSchema":
        """Construct a :class:`TypeSchema` from a parsed JSON object.

        Non-dict elements in the ``"args"`` list (e.g., numeric lengths
        emitted by ``std::array`` TypeTraits) are silently skipped.
        """
        if not isinstance(obj, dict) or "type" not in obj:
            raise TypeError(
                f"expected schema dict with 'type' key, got {type(obj).__name__}"
            )
        origin = obj["type"]
        origin = _TYPE_SCHEMA_ORIGIN_CONVERTER.get(origin, origin)
        raw_args = obj.get("args", ())
        if not isinstance(raw_args, (list, tuple)):
            raw_args = ()
        args = tuple(
            TypeSchema.from_json_obj(a) for a in raw_args
            if isinstance(a, dict)
        )
        return TypeSchema(origin, args)

    @staticmethod
    def from_json_str(s: str) -> "TypeSchema":
        """Construct a :class:`TypeSchema` from a JSON string."""
        return TypeSchema.from_json_obj(json.loads(s))

    @staticmethod
    def from_type_index(type_index: int, args: "tuple[TypeSchema, ...]" = ()) -> "TypeSchema":
        """Construct a :class:`TypeSchema` from a type_index and optional args.

        Parameters
        ----------
        type_index : int
            A valid TVM FFI type index (e.g., ``kTVMFFIInt``, ``kTVMFFIArray``,
            or an object type index from ``_object_type_key_to_index``).
            Passing an unregistered index triggers a fatal C++ assertion;
            callers must ensure the index was obtained from the type registry.
        args : tuple[TypeSchema, ...], optional
            Type arguments for parameterized types (e.g., element type for Array).

        Returns
        -------
        TypeSchema
            A new schema with the origin resolved from the type index.
        """
        origin = _TYPE_INDEX_TO_ORIGIN.get(type_index, None)
        if origin is None:
            origin = _type_index_to_key(type_index)
        return TypeSchema(origin, args, origin_type_index=type_index)

    def check_value(self, value: object) -> None:
        """Validate that *value* is compatible with this type schema.

        Parameters
        ----------
        value : object
            The Python value to check.

        Raises
        ------
        TypeError
            If the value is not compatible with the schema, with a
            human-readable error message describing the mismatch.
        """
        # _type_schema_check_value is defined in type_check.pxi
        _type_schema_check_value(self, value)

    def try_check_value(self, value: object) -> "Optional[str]":
        """Check if *value* is compatible with this type schema.

        Returns
        -------
        str or None
            ``None`` if the value is compatible, or an error message
            string describing the mismatch.
        """
        # _type_schema_try_check_value is defined in type_check.pxi
        return _type_schema_try_check_value(self, value)

    def convert(self, value: object) -> object:
        """Convert *value* according to this type schema.

        Applies the same implicit conversions as the C++ FFI
        ``TypeTraits<T>::TryCastFromAnyView`` rules.  For example,
        ``TypeSchema("float").convert(42)`` returns ``42.0``.

        Parameters
        ----------
        value : object
            The Python value to convert.

        Returns
        -------
        object
            The converted value.

        Raises
        ------
        TypeError
            If the value cannot be converted to this schema's type.
        """
        # _type_schema_convert is defined in type_check.pxi
        return _type_schema_convert(self, value)

    def try_convert(self, value: object) -> "tuple[bool, object]":
        """Try to convert *value* according to this type schema.

        Returns
        -------
        tuple[bool, object]
            A ``(success, result)`` pair.  On success, ``result`` is the
            converted value (which may be ``None`` for e.g. ``Optional``).
            On failure, ``result`` is an error message string.
        """
        # _type_schema_try_convert is defined in type_check.pxi
        return _type_schema_try_convert(self, value)

    def repr(self, ty_map: "Optional[Callable[[str], str]]" = None) -> str:
        """Render a human-readable representation of this schema.

        Parameters
        ----------
        ty_map : Callable[[str], str], optional
            A mapping function applied to the schema origin name before
            rendering (e.g. map ``"Array" -> "Array"`` and
            ``"Map" -> "Map"``). If ``None``, the raw origin is used.

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

            # Container types from C++ FFI schemas
            s = TypeSchema.from_json_str('{"type":"ffi.Map","args":[{"type":"str"},{"type":"int"}]}')
            assert s.repr() == "Map[str, int]"

            s = TypeSchema.from_json_str('{"type":"ffi.Array","args":[{"type":"int"}]}')
            assert s.repr() == "Array[int]"

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
    c_init: bool = True
    c_kw_only: bool = False
    c_has_default: bool = False
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
