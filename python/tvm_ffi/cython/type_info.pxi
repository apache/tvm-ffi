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
import typing
import collections.abc
from functools import cached_property
from typing import Optional, Any
from io import StringIO

try:
    from types import UnionType as _UnionType
except ImportError:
    _UnionType = None


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
    cdef void* setter
    cdef int64_t offset
    cdef int64_t flags

    def __call__(self, CObject obj, value):
        cdef int c_api_ret_code
        cdef void* field_ptr = (<char*>(<CObject>obj).chandle) + self.offset
        TVMFFIPyCallFieldSetter(
            TVMFFIPyArgSetterFactory_,
            self.setter,
            self.flags,
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
    args: tuple["TypeSchema", ...] | None = None
    origin_type_index: int = dataclasses.field(default=_ORIGIN_TYPE_INDEX_UNKNOWN, repr=False)

    def __post_init__(self):
        origin = self.origin
        args = self.args
        if args is not None and not isinstance(args, tuple):
            args = tuple(args)
            self.args = args
        if origin != "tuple" and args is None:
            args = ()
            self.args = args
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
        if "args" not in obj:
            return TypeSchema(origin)
        raw_args = obj["args"]
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

    @staticmethod
    def from_annotation(annotation: object) -> "TypeSchema":
        """Construct a :class:`TypeSchema` from a Python type annotation.

        Parameters
        ----------
        annotation : object
            A Python type annotation such as ``int``, ``list[int]``,
            ``Optional[str]``, ``Union[int, str]``, ``Callable[[int], str]``,
            or a registered :class:`CObject` subclass.

        Returns
        -------
        TypeSchema
            The corresponding schema.

        Raises
        ------
        TypeError
            If the annotation cannot be mapped to a TypeSchema.

        Examples
        --------
        >>> TypeSchema.from_annotation(int)
        int
        >>> TypeSchema.from_annotation(list[int])
        List[int]
        >>> TypeSchema.from_annotation(tuple[int, ...])
        Array[int]
        """
        # --- Singletons ---
        if annotation is type(None) or annotation is None:
            return TypeSchema("None")
        if annotation is typing.Any:
            return TypeSchema("Any")

        # --- Bare builtin scalar types ---
        if annotation is bool:
            return TypeSchema("bool")
        if annotation is int:
            return TypeSchema("int")
        if annotation is float:
            return TypeSchema("float")
        if annotation is str:
            return TypeSchema("str")
        if annotation is bytes:
            return TypeSchema("bytes")

        # --- Bare container types (unparameterised) ---
        if annotation is list:
            return TypeSchema("List")
        if annotation is dict:
            return TypeSchema("Dict")
        if annotation is tuple:
            return TypeSchema("tuple")
        if annotation is collections.abc.Callable:
            return TypeSchema("Callable")

        # --- Python 3.10+ union syntax  (X | Y) ---
        if _UnionType is not None and isinstance(annotation, _UnionType):
            return _annotation_union(typing.get_args(annotation))

        # --- Generic aliases (list[int], Optional[T], etc.) ---
        origin = typing.get_origin(annotation)
        targs = typing.get_args(annotation)

        if origin is typing.Union:
            return _annotation_union(targs)

        if origin is list:
            if len(targs) > 1:
                raise TypeError(
                    f"list takes at most 1 type argument, got {len(targs)}"
                )
            if targs:
                return TypeSchema("List", (TypeSchema.from_annotation(targs[0]),))
            return TypeSchema("List")

        if origin is dict:
            if len(targs) == 1 or len(targs) > 2:
                raise TypeError(
                    f"dict requires 0 or 2 type arguments, got {len(targs)}"
                )
            if len(targs) == 2:
                return TypeSchema("Dict", (
                    TypeSchema.from_annotation(targs[0]),
                    TypeSchema.from_annotation(targs[1]),
                ))
            return TypeSchema("Dict")

        if origin is tuple:
            if len(targs) == 2 and targs[1] is Ellipsis:
                # tuple[T, ...] → homogeneous variable-length → Array
                return TypeSchema("Array", (TypeSchema.from_annotation(targs[0]),))
            if targs:
                return TypeSchema(
                    "tuple",
                    tuple(TypeSchema.from_annotation(a) for a in targs),
                )
            if annotation is not tuple:
                return TypeSchema("tuple", ())
            return TypeSchema("tuple")

        if origin is collections.abc.Callable:
            if len(targs) == 2:
                params, ret = targs
                ret_schema = TypeSchema.from_annotation(ret)
                if isinstance(params, list):
                    # Callable[[P1, P2], R] → (R, P1, P2)
                    param_schemas = tuple(
                        TypeSchema.from_annotation(p) for p in params
                    )
                    return TypeSchema("Callable", (ret_schema,) + param_schemas)
                # Callable[..., R]
                return TypeSchema("Callable", (ret_schema,))
            return TypeSchema("Callable")

        # --- Parameterised CObject subclasses (Array[int], Dict[str, V], …) ---
        if isinstance(origin, type) and issubclass(origin, CObject):
            return _annotation_cobject(origin, targs)

        # --- Bare (unparameterised) CObject subclasses ---
        if isinstance(annotation, type) and issubclass(annotation, CObject):
            return _annotation_cobject(annotation, ())

        # --- PyNativeObject subclasses (String, Bytes) ---
        if isinstance(annotation, type) and issubclass(annotation, PyNativeObject):
            if issubclass(annotation, str):
                return TypeSchema("str")
            if issubclass(annotation, bytes):
                return TypeSchema("bytes")

        # --- Non-CObject cdef classes with known origins ---
        if annotation is DataType:
            return TypeSchema("dtype")
        if annotation is Device:
            return TypeSchema("Device")

        # --- ctypes.c_void_p ---
        import ctypes as _ctypes
        if annotation is _ctypes.c_void_p:
            return TypeSchema("ctypes.c_void_p")

        # --- Types with __dlpack__ protocol (e.g. torch.Tensor) → Tensor ---
        if isinstance(annotation, type) and hasattr(annotation, "__dlpack__"):
            return TypeSchema("Tensor")

        raise TypeError(
            f"Cannot convert {annotation!r} to TypeSchema"
        )

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
        try:
            _type_convert_impl(self._converter, value)
        except RecursionError:
            raise TypeError(
                f"type check failed for {self!r}: "
                f"infinite __tvm_ffi_value__ cycle detected"
            ) from None
        except _ConvertError as err:
            raise TypeError(f"type check failed for {self!r}: {err.message}") from None

    def convert(self, value: object) -> "CAny":
        """Convert *value* according to this type schema, returning a :class:`CAny`.

        Applies the same implicit conversions as the C++ FFI
        ``TypeTraits<T>::TryCastFromAnyView`` rules.  The result is
        always a :class:`CAny` instance that owns the converted value.
        Use ``_to_py_class_value(result)`` to recover the Python object.

        Parameters
        ----------
        value : object
            The Python value to convert.

        Returns
        -------
        CAny
            The converted value wrapped in a CAny.

        Raises
        ------
        TypeError
            If the value cannot be converted to this schema's type.
        """
        try:
            return _type_convert_impl(self._converter, value)
        except RecursionError:
            raise TypeError(
                f"type conversion failed for {self!r}: "
                f"infinite __tvm_ffi_value__ cycle detected"
            ) from None
        except _ConvertError as err:
            raise TypeError(f"type conversion failed for {self!r}: {err.message}") from None

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
        schema_args = self.args
        args = [i.repr(ty_map) for i in (() if schema_args is None else schema_args)]
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
        elif origin == "tuple" and schema_args == ():
            return "tuple[()]"
        elif not args:
            return origin
        else:
            args = ", ".join(args)
            return f"{origin}[{args}]"

    def to_json(self) -> dict[str, Any]:
        """Convert a TypeSchema to a JSON-compatible dict."""
        if self.args is not None and (self.args or self.origin == "tuple"):
            return {
                "type": self.origin,
                "args": [a.to_json() for a in self.args],
            }
        return {"type": self.origin}


def _annotation_union(args):
    """Convert Union type args to a TypeSchema (Optional or Union)."""
    non_none = tuple(a for a in args if a is not type(None))
    has_none = len(non_none) < len(args)
    converted = tuple(TypeSchema.from_annotation(a) for a in non_none)
    if has_none:
        if len(non_none) == 1:
            return TypeSchema("Optional", converted)
        return TypeSchema("Optional", (TypeSchema("Union", converted),))
    return TypeSchema("Union", converted)


def _annotation_cobject(cls, targs):
    """Handle a CObject subclass (bare or parameterised) in from_annotation."""
    info = TYPE_CLS_TO_INFO.get(cls)
    if info is None:
        raise TypeError(
            f"CObject subclass {cls!r} is not registered "
            f"in TYPE_CLS_TO_INFO; use @register_object to register it"
        )
    # Prefer canonical short origin from _TYPE_INDEX_TO_ORIGIN (e.g. "Array")
    # over the registered type_key (e.g. "ffi.Array") when available.
    origin = _TYPE_INDEX_TO_ORIGIN.get(info.type_index, info.type_key)
    n = len(targs)
    if n > 0:
        if origin in ("Array", "List"):
            if n != 1:
                raise TypeError(
                    f"{origin} requires 1 type argument, got {n}"
                )
        elif origin in ("Map", "Dict"):
            if n != 2:
                raise TypeError(
                    f"{origin} requires 2 type arguments, got {n}"
                )
        arg_schemas = tuple(TypeSchema.from_annotation(a) for a in targs)
        return TypeSchema(origin, arg_schemas, origin_type_index=info.type_index)
    return TypeSchema(origin, origin_type_index=info.type_index)


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
    ty: Optional[TypeSchema] = None
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
    fields: Optional[list[TypeField]]
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

    @cached_property
    def total_size(self) -> int:
        """Total object size in bytes (header + all fields).

        For native C++ types with metadata, returns metadata.total_size.
        For Python-defined types, computes from field layout.
        """
        cdef const TVMFFITypeInfo* c_info = TVMFFIGetTypeInfo(self.type_index)
        if c_info != NULL and c_info.metadata != NULL:
            return c_info.metadata.total_size
        cdef int64_t end = sizeof(TVMFFIObject)
        if self.fields:
            for f in self.fields:
                f_end = f.offset + f.size
                if f_end > end:
                    end = f_end
        return (end + 7) & ~7  # align to 8 bytes


def _member_method_wrapper(method_func: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(self: Any, *args: Any) -> Any:
        return method_func(self, *args)

    return wrapper
