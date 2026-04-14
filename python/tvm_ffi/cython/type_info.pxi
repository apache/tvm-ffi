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
        if annotation is DataType or (_CLASS_DTYPE is not None and annotation is _CLASS_DTYPE):
            return TypeSchema("dtype")
        if annotation is Device or (_CLASS_DEVICE is not None and annotation is _CLASS_DEVICE):
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


class FFIProperty(property):
    """Property descriptor for FFI-backed fields.

    When *frozen* is True the public setter (``fset``) is suppressed so
    that normal attribute assignment raises ``AttributeError``.  The
    real setter is stashed in :attr:`_fset` and exposed via the
    :meth:`set` escape-hatch.
    """

    def __init__(self, fget, fset, frozen, fdel=None, doc=None):
        super().__init__(fget, None if frozen else fset, fdel, doc)
        self._fset = fset

    def set(self, obj, value):
        """Force-set the field value, bypassing the frozen guard."""
        self._fset(obj, value)


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
        """Create an :class:`FFIProperty` descriptor for this field on ``cls``."""
        cdef str name = self.name
        cdef FieldGetter fget = self.getter
        cdef FieldSetter fset = self.setter
        cdef object ret
        fget.__name__ = fset.__name__ = name
        fget.__module__ = fset.__module__ = cls.__module__
        fget.__qualname__ = fset.__qualname__ = f"{cls.__qualname__}.{name}"
        ret = FFIProperty(
            fget=fget,
            fset=fset,
            frozen=self.frozen,
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
        # Assert no duplicate field names within this type's own fields.
        if self.fields is not None:
            seen = set()
            for f in self.fields:
                assert f.name not in seen, (
                    f"duplicate field name {f.name!r} in TypeInfo for {self.type_key!r}; "
                    f"TypeInfo.fields must only contain the type's own fields"
                )
                seen.add(f.name)
        if not self.type_ancestors:
            return
        parent_type_index = self.type_ancestors[-1]
        parent_type_key = _type_index_to_key(parent_type_index)
        # ensure parent is registered
        self.parent_type_info = _lookup_or_register_type_info_from_type_key(parent_type_key)
        # Warn if own fields shadow any ancestor field.
        if self.fields and self.parent_type_info is not None:
            parent_names = set()
            ti = self.parent_type_info
            while ti is not None:
                if ti.fields:
                    for f in ti.fields:
                        parent_names.add(f.name)
                ti = ti.parent_type_info
            for f in self.fields:
                if f.name in parent_names:
                    import warnings
                    warnings.warn(
                        f"Field {f.name!r} in {self.type_key!r} duplicates "
                        f"an ancestor field. Child types should not "
                        f"re-register inherited fields.",
                        stacklevel=2,
                    )

    @cached_property
    def total_size(self) -> int:
        """Total object size in bytes (header + all fields).

        For native C++ types with metadata, returns metadata.total_size.
        For Python-defined types, computes from field layout.
        """
        cdef const TVMFFITypeInfo* c_info = TVMFFIGetTypeInfo(self.type_index)
        if c_info != NULL and c_info.metadata != NULL:
            return c_info.metadata.total_size
        if self.parent_type_info is None:
            raise ValueError(f"Cannot find parent type of {type(self)}")
        cdef int64_t end = self.parent_type_info.total_size
        assert end >= sizeof(TVMFFIObject)
        for f in self.fields:
            end = max(end, f.offset + f.size)
        return (end + 7) & ~7  # align to 8 bytes

    def _register_fields(self, fields, structure_kind=None):
        """Register Field descriptors and set up __ffi_new__/__ffi_init__.

        Delegates to the module-level _register_fields function,
        stores the resulting list[TypeField] on self.fields,
        then reads back methods registered by C++ via _read_back_methods.

        Can only be called once (fields must be None beforehand).

        Parameters
        ----------
        fields : list[Field]
            The Field descriptors to register.
        structure_kind : int | None
            The structural equality/hashing kind (``TVMFFISEqHashKind`` integer).
            ``None`` or ``0`` means unsupported (no metadata registered).
        """
        assert self.fields is None, (
            f"_register_fields already called for {self.type_key!r}"
        )
        self.fields = _register_fields(self, fields, structure_kind)
        self._read_back_methods()

    def _register_py_methods(self, py_methods=None, type_attr_names=frozenset()):
        """Register user-defined dunder hooks and re-read the method table.

        Each entry whose name is in *type_attr_names* is registered as a
        TypeAttrColumn entry (for C++ dispatch); the value need not be
        callable (e.g. ``__ffi_ir_traits__``).  All other entries are
        registered as TypeMethod (for reflection introspection).

        Regardless, the full method list is always re-read from the C
        type table so that system-generated methods (``__ffi_init__``,
        ``__ffi_shallow_copy__``) are picked up.

        Parameters
        ----------
        py_methods : list[tuple[str, Any, bool]] | None
            Each entry is ``(name, value, is_static)``.
        type_attr_names : frozenset[str]
            Names to register as TypeAttrColumn instead of TypeMethod.
        """
        if py_methods:
            _register_py_methods(self.type_index, py_methods, type_attr_names)
        self._read_back_methods()

    def _read_back_methods(self):
        """Read methods from the C type table into self.methods.

        Called after C++ registers __ffi_init__, __ffi_shallow_copy__, etc.
        """
        cdef const TVMFFITypeInfo* c_info = TVMFFIGetTypeInfo(self.type_index)
        cdef const TVMFFIMethodInfo* mi
        self.methods = []
        for i in range(c_info.num_methods):
            mi = &(c_info.methods[i])
            self.methods.append(TypeMethod(
                name=bytearray_to_str(&mi.name),
                doc=bytearray_to_str(&mi.doc) if mi.doc.size != 0 else None,
                func=_get_method_from_method_info(mi),
                is_static=(mi.flags & kTVMFFIFieldFlagBitMaskIsStaticMethod) != 0,
                metadata=json.loads(bytearray_to_str(&mi.metadata)) if mi.metadata.size != 0 else {},
            ))


# ---------------------------------------------------------------------------
# Python-defined type field registration helpers
# ---------------------------------------------------------------------------

# Native layout for each TypeSchema origin: (size, alignment, field_static_type_index)
_ORIGIN_NATIVE_LAYOUT = {
    "int": (8, 8, kTVMFFIInt),
    "float": (8, 8, kTVMFFIFloat),
    "bool": (1, 1, kTVMFFIBool),
    "ctypes.c_void_p": (8, 8, kTVMFFIOpaquePtr),
    "dtype": (4, 2, kTVMFFIDataType),
    "Device": (8, 4, kTVMFFIDevice),
    "Any": (16, 8, -1),  # kTVMFFIAny = -1
    # str/bytes can be SmallStr/SmallBytes (inline, not ObjectRef),
    # so store as Any (16 bytes) to handle both inline and heap variants.
    "str": (16, 8, -1),
    "bytes": (16, 8, -1),
    # Optional/Union can hold any type including inline scalars
    "Optional": (16, 8, -1),
    "Union": (16, 8, -1),
}

cdef _register_one_field(
    int32_t type_index,
    object py_field,
    int64_t offset,
    int64_t size,
    int64_t alignment,
    int32_t field_type_index,
    TVMFFIFieldGetter getter,
    CObject setter_fn,
):
    """Build a TVMFFIFieldInfo and register it for the given type."""
    cdef TVMFFIFieldInfo info
    cdef int c_api_ret_code

    # --- name ---
    name_bytes = c_str(py_field.name)
    cdef ByteArrayArg name_arg = ByteArrayArg(name_bytes)
    info.name = name_arg.cdata

    # --- doc ---
    cdef ByteArrayArg doc_arg
    if py_field.doc is not None:
        doc_bytes = c_str(py_field.doc)
        doc_arg = ByteArrayArg(doc_bytes)
        info.doc = doc_arg.cdata
    else:
        info.doc.data = NULL
        info.doc.size = 0

    # --- metadata (JSON with type_schema) ---
    metadata_str = json.dumps({"type_schema": py_field.ty.to_json()})
    metadata_bytes = c_str(metadata_str)
    cdef ByteArrayArg metadata_arg = ByteArrayArg(metadata_bytes)
    info.metadata = metadata_arg.cdata

    # --- flags ---
    cdef int64_t flags = kTVMFFIFieldFlagBitMaskWritable | kTVMFFIFieldFlagBitSetterIsFunctionObj
    if py_field.default is not MISSING or py_field.default_factory is not MISSING:
        flags |= kTVMFFIFieldFlagBitMaskHasDefault
    if py_field.default_factory is not MISSING:
        flags |= kTVMFFIFieldFlagBitMaskDefaultFromFactory
    if not py_field.init:
        flags |= kTVMFFIFieldFlagBitMaskInitOff
    if not py_field.repr:
        flags |= kTVMFFIFieldFlagBitMaskReprOff
    if not py_field.hash:
        flags |= kTVMFFIFieldFlagBitMaskHashOff
    if not py_field.compare:
        flags |= kTVMFFIFieldFlagBitMaskCompareOff
    if py_field.kw_only:
        flags |= kTVMFFIFieldFlagBitMaskKwOnly
    # Structural equality/hashing field annotations
    cdef object field_structure = getattr(py_field, "structural_eq", None)
    if field_structure == "ignore":
        flags |= kTVMFFIFieldFlagBitMaskSEqHashIgnore
    elif field_structure == "def":
        flags |= kTVMFFIFieldFlagBitMaskSEqHashDef
    info.flags = flags

    # --- native layout ---
    info.size = size
    info.alignment = alignment
    info.offset = offset

    # --- getter / setter ---
    info.getter = getter
    info.setter = <void*>setter_fn.chandle

    # --- default value ---
    cdef TVMFFIAny default_any
    default_any.type_index = kTVMFFINone
    default_any.v_int64 = 0
    # Determine which Python object (if any) to store as the default.
    # No memory leak: TVMFFIAny is a POD struct; TVMFFITypeRegisterField
    # copies the bytes into the type table, which owns the reference.
    cdef object default_obj = MISSING
    if py_field.default is not MISSING:
        default_obj = py_field.default
    elif py_field.default_factory is not MISSING:
        default_obj = py_field.default_factory
    if default_obj is not MISSING:
        TVMFFIPyPyObjectToFFIAny(
            TVMFFIPyArgSetterFactory_,
            <PyObject*>default_obj,
            &default_any,
            &c_api_ret_code
        )
        CHECK_CALL(c_api_ret_code)
    info.default_value_or_factory = default_any

    # --- field_static_type_index ---
    info.field_static_type_index = field_type_index

    CHECK_CALL(TVMFFITypeRegisterField(type_index, &info))


cdef int _f_type_convert(void* type_converter, const TVMFFIAny* value, TVMFFIAny* result) noexcept with gil:
    """C callback for type conversion, called from C++ MakeFieldSetter.

    Parameters
    ----------
    type_converter : void*
        A PyObject* pointing to a _TypeConverter instance (borrowed reference).
    value : const TVMFFIAny*
        The packed value to convert (borrowed from the caller).
    result : TVMFFIAny*
        Output: the converted value (caller takes ownership).

    Returns 0 on success, -1 on error (error stored in TLS via set_last_ffi_error).
    """
    cdef TVMFFIAny temp
    cdef _TypeConverter conv
    cdef CAny cany
    try:
        # Unpack the packed AnyView to a Python object.
        # We must IncRef if it's an object, because make_ret takes ownership.
        temp = value[0]
        if temp.type_index >= kTVMFFIStaticObjectBegin:
            if temp.v_obj != NULL:
                TVMFFIObjectIncRef(<TVMFFIObjectHandle>temp.v_obj)
        py_value = make_ret(temp)
        # Dispatch directly through the C-level converter
        conv = <_TypeConverter>type_converter
        cany = _type_convert_impl(conv, py_value)
        # Transfer ownership from CAny to result (zero cany to prevent double-free)
        result[0] = cany.cdata
        cany.cdata.type_index = kTVMFFINone
        cany.cdata.v_int64 = 0
        return 0
    except Exception as err:
        set_last_ffi_error(err)
        return -1


def _register_fields(type_info, fields, structure_kind=None):
    """Register Field descriptors for a Python-defined type and set up __ffi_new__/__ffi_init__.

    For each Field:
    1. Computes native layout (size, alignment, offset)
    2. Obtains a C getter function pointer
    3. Creates a FunctionObj setter with type conversion
    4. Registers via TVMFFITypeRegisterField

    After all fields, registers __ffi_new__ (object allocator),
    __ffi_init__ (auto-generated constructor), and optionally
    type metadata (structural_eq_hash_kind).

    Parameters
    ----------
    type_info : TypeInfo
        The TypeInfo of the type being defined.
    fields : list[Field]
        The Field descriptors to register.
    structure_kind : int | None
        The structural equality/hashing kind (``TVMFFISEqHashKind`` integer).
        ``None`` or ``0`` means unsupported (no metadata registered).

    Returns
    -------
    list[TypeField]
        The registered field descriptors.
    """
    cdef int32_t type_index = type_info.type_index
    # Start field offsets AFTER all parent fields (not at fixed offset 24).
    # This is critical for inheritance: child fields must not overlap parent memory.
    cdef int64_t current_offset = type_info.parent_type_info.total_size
    cdef int64_t size, alignment
    cdef int32_t field_type_index
    cdef TVMFFIFieldGetter getter
    cdef FieldGetter fgetter
    cdef FieldSetter fsetter
    cdef list type_fields = []
    for py_field in fields:
        # 1. Get layout
        layout = _ORIGIN_NATIVE_LAYOUT.get(py_field.ty.origin, (8, 8, kTVMFFIObject))
        size = layout[0]
        alignment = layout[1]
        field_type_index = layout[2]

        # 2. Compute offset (align up)
        current_offset = (current_offset + alignment - 1) & ~(alignment - 1)
        field_offset = current_offset
        current_offset += size

        # 3. Get getter (C function pointer) and setter (FunctionObj).
        # Pointers are transported as int64_t through the FFI boundary.
        getter = <TVMFFIFieldGetter><int64_t>_MAKE_FILED_GETTER(field_type_index)
        setter_fn = <CObject>_MAKE_FIELD_SETTER(
            field_type_index,
            <int64_t><void*>py_field.ty._converter,
            <int64_t>&_f_type_convert,
        )

        # 4. Register field in the C type table
        _register_one_field(
            type_index, py_field, field_offset, size, alignment,
            field_type_index, getter, setter_fn,
        )

        # 5. Build the Python-side TypeField descriptor
        fgetter = FieldGetter.__new__(FieldGetter)
        fgetter.getter = getter
        fgetter.offset = field_offset
        fsetter = FieldSetter.__new__(FieldSetter)
        fsetter.setter = <void*>setter_fn.chandle
        fsetter.offset = field_offset
        fsetter.flags = <int64_t>(kTVMFFIFieldFlagBitMaskWritable | kTVMFFIFieldFlagBitSetterIsFunctionObj)
        type_fields.append(
            TypeField(
                name=py_field.name,
                doc=py_field.doc,
                size=size,
                offset=field_offset,
                frozen=py_field.frozen,
                metadata={"type_schema": py_field.ty.to_json()},
                getter=fgetter,
                setter=fsetter,
                ty=py_field.ty,
                c_init=py_field.init,
                c_kw_only=py_field.kw_only,
                c_has_default=(py_field.default is not MISSING or py_field.default_factory is not MISSING),
            )
        )

    # Align total size to 8 bytes
    cdef int64_t total_size = (current_offset + 7) & ~7
    if total_size < sizeof(TVMFFIObject):
        total_size = sizeof(TVMFFIObject)

    # 7. Register __ffi_new__, __ffi_shallow_copy__, __ffi_init__ TypeAttrColumns
    _PYCLS_REGISTER(type_index, total_size)

    # 8. Register type metadata (structural_eq_hash_kind) if specified.
    if structure_kind is not None and structure_kind != 0:
        _register_type_metadata(type_index, total_size, structure_kind)

    return type_fields


cdef _register_type_metadata(int32_t type_index, int32_t total_size, int structure_kind):
    """Register TVMFFITypeMetadata for the given type with structural eq/hash kind."""
    cdef TVMFFITypeMetadata metadata
    metadata.doc.data = NULL
    metadata.doc.size = 0
    metadata.creator = NULL
    metadata.total_size = total_size
    metadata.structural_eq_hash_kind = <TVMFFISEqHashKind>structure_kind
    CHECK_CALL(TVMFFITypeRegisterMetadata(type_index, &metadata))


cdef _register_py_methods(int32_t type_index, list py_methods, frozenset type_attr_names):
    """Register user-defined methods and type attrs as TypeAttrColumn or TypeMethod.

    For each entry in *py_methods*:
    1. Convert the Python object to a ``TVMFFIAny``.
    2. If the name is in *type_attr_names*, register as TypeAttrColumn
       (for C++ dispatch via ``TypeAttrColumn``).  The value need not be
       callable (e.g. ``__ffi_ir_traits__`` is an Object instance).
    3. Otherwise, register as TypeMethod (for reflection introspection
       via ``TypeInfo.methods``).

    Parameters
    ----------
    type_index : int
        The runtime type index of the type.
    py_methods : list[tuple[str, Any, bool]]
        Each entry is ``(name, value, is_static)``.
    type_attr_names : frozenset[str]
        Names to register as TypeAttrColumn instead of TypeMethod.
    """
    cdef TVMFFIMethodInfo method_info
    cdef TVMFFIAny func_any
    cdef TVMFFIAny sentinel_any
    cdef int c_api_ret_code
    cdef ByteArrayArg name_arg

    sentinel_any.type_index = kTVMFFINone
    sentinel_any.v_int64 = 0

    for name, func, is_static in py_methods:
        func_any.type_index = kTVMFFINone
        func_any.v_int64 = 0
        try:
            name_bytes = c_str(name)
            name_arg = ByteArrayArg(name_bytes)

            # Convert Python object -> TVMFFIAny
            TVMFFIPyPyObjectToFFIAny(
                TVMFFIPyArgSetterFactory_,
                <PyObject*>func,
                &func_any,
                &c_api_ret_code,
            )
            CHECK_CALL(c_api_ret_code)

            if name in type_attr_names:
                # Register as TypeAttrColumn (for C++ dispatch)
                CHECK_CALL(TVMFFITypeRegisterAttr(kTVMFFINone, &name_arg.cdata, &sentinel_any))
                CHECK_CALL(TVMFFITypeRegisterAttr(type_index, &name_arg.cdata, &func_any))
            else:
                # Register as TypeMethod (for reflection introspection)
                method_info.name = name_arg.cdata
                method_info.doc.data = NULL
                method_info.doc.size = 0
                method_info.flags = kTVMFFIFieldFlagBitMaskIsStaticMethod if is_static else 0
                method_info.method = func_any
                method_info.metadata.data = NULL
                method_info.metadata.size = 0
                CHECK_CALL(TVMFFITypeRegisterMethod(type_index, &method_info))
        finally:
            if func_any.type_index >= kTVMFFIStaticObjectBegin and func_any.v_obj != NULL:
                TVMFFIObjectDecRef(<TVMFFIObjectHandle>func_any.v_obj)


def _member_method_wrapper(method_func: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(self: Any, *args: Any) -> Any:
        return method_func(self, *args)

    return wrapper
