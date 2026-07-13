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
"""Cross-language enum singletons."""

from __future__ import annotations

import sys
import typing
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any, ClassVar, overload

from typing_extensions import Self

from .. import core
from ..container import Dict, List
from ..core import Object
from .c_class import c_class
from .py_class import py_class

if TYPE_CHECKING:
    from enum import EnumMeta as _EnumMetaBase
else:
    _EnumMetaBase = type(Object)

if TYPE_CHECKING:

    def _enum_c_class(type_key: str, **kwargs: Any) -> Callable[[type], type]:
        return lambda cls: cls

else:
    _enum_c_class = c_class

__all__ = [
    "ENUM_STATE_ATTR",
    "Enum",
    "EnumAttrMap",
    "EnumState",
    "IntEnum",
    "StrEnum",
    "auto",
    "entry",
]

ENUM_STATE_ATTR = "__ffi_enum__"
_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1


@c_class("ffi.EnumState")
class EnumState(Object):
    """Registry state shared by enum definition and lookup."""

    entries: List[Object]
    indexes: Dict[Any, Object]
    attrs: Dict[str, Dict[Object, Any]]

    if TYPE_CHECKING:

        def __init__(self) -> None: ...

        def __ffi_init__(self) -> None: ...  # ty: ignore[invalid-method-override]


class _EnumEntry:
    __slots__ = ("args", "kwargs")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs

    def __repr__(self) -> str:
        values = [repr(arg) for arg in self.args]
        values.extend(f"{key}={value!r}" for key, value in self.kwargs.items())
        return f"entry({', '.join(values)})"


def entry(*args: Any, **kwargs: Any) -> Any:
    """Declare an enum variant, optionally with declared-field values."""
    return _EnumEntry(*args, **kwargs)


def auto() -> Any:
    """Declare a variant whose indices are inferred from its class-body alias."""
    return _EnumEntry()


class _ClassProperty:
    __slots__ = ("fget",)

    def __init__(self, fget: Callable[[type], Any]) -> None:
        self.fget = fget

    def __get__(self, instance: Any, owner: type | None = None) -> Any:
        return self.fget(owner if owner is not None else type(instance))


def _normalize_index(index: Any) -> int | str:
    if isinstance(index, bool):
        return int(index)
    if isinstance(index, int):
        if not _INT64_MIN <= index <= _INT64_MAX:
            raise OverflowError(f"enum integer index {index} is outside int64")
        return index
    if isinstance(index, str):
        return index
    raise TypeError(f"enum index must be int or str, got {type(index).__name__}")


def _state(cls: type, *, create: bool = False) -> EnumState | None:
    type_info = getattr(cls, "__tvm_ffi_type_info__", None)
    if type_info is None:
        return None
    state = core._lookup_type_attr(type_info.type_index, ENUM_STATE_ATTR)
    if state is None and create:
        state = EnumState()
        core._register_type_attr(type_info.type_index, ENUM_STATE_ATTR, state)
        state = core._lookup_type_attr(type_info.type_index, ENUM_STATE_ATTR)
    return state


def _ordered_entries(cls: type) -> list[Any]:
    state = _state(cls)
    return [] if state is None else list(state.entries)


class _EnumMeta(_EnumMetaBase):
    def __iter__(cls) -> Iterator[Any]:
        return iter(_ordered_entries(cls))

    def __len__(cls) -> int:
        return len(_ordered_entries(cls))

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if not kwargs and len(args) == 1:
            value = args[0]
            if isinstance(value, cls):
                return value
            if isinstance(value, (int, str)):
                try:
                    return cls.get(value)  # ty: ignore[unresolved-attribute]
                except KeyError:
                    raise ValueError(f"{value!r} is not a valid {cls.__name__}") from None
        return super().__call__(*args, **kwargs)


@_enum_c_class("ffi.Enum", init=False)
class Enum(Object, metaclass=_EnumMeta):
    """Base class for FFI-registered enum singletons."""

    if not TYPE_CHECKING:
        __slots__ = ()
        _int_index: int
        _str_index: str

    if TYPE_CHECKING:

        @property
        def _int_index(self) -> int: ...

        @property
        def _str_index(self) -> str: ...

        @property
        def name(self) -> str:
            """Return the canonical string index."""
            ...

        @property
        def value(self) -> Any:
            """Return the payload index exposed by a payload enum."""
            ...

        @overload
        def __new__(cls: type[Self], index: Self, /) -> Self: ...

        @overload
        def __new__(cls: type[Self], index: int | str, /) -> Self: ...

        @overload
        def __new__(cls: type[Self], *args: Any, **kwargs: Any) -> Self: ...

        def __new__(cls: type[Self], *args: Any, **kwargs: Any) -> Self:
            """Normalize a canonical index to its singleton."""
            ...

    def __init_subclass__(
        cls,
        *,
        type_key: str | None = None,
        frozen: bool = True,
        init: bool = True,
        repr: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init_subclass__(**kwargs)
        if type_key is None:
            return
        payload_type = (
            int if issubclass(cls, IntEnum) else str if issubclass(cls, StrEnum) else None
        )
        user_repr = "__repr__" in cls.__dict__
        binders, declarations = _collect_declarations(cls, payload_type)
        cxx_backed = core._object_type_key_to_index(type_key) is not None
        if cxx_backed:
            c_class(type_key, init=init, repr=repr)(cls)
        else:
            py_class(type_key, frozen=frozen, structural_eq="singleton")(cls)
        if payload_type is not None:
            _install_payload_behavior(cls, payload_type, user_repr)
        _resolve(cls, binders, declarations, type_key, cxx_backed, payload_type)

    @classmethod
    def get(cls, index: int | str) -> Enum:
        """Return the singleton at a canonical integer or string index."""
        index = _normalize_index(index)
        state = _state(cls)
        if state is not None and index in state.indexes:
            return typing.cast(Enum, state.indexes[index])
        raise KeyError(index)

    @classmethod
    def all_entries(cls) -> Iterator[Enum]:
        """Iterate over variants in registration order."""
        return iter(_ordered_entries(cls))

    @_ClassProperty
    def attr_dict(cls: type) -> Any:
        """Return the live extensible-attribute dictionary."""
        state = _state(cls)
        return Dict({}) if state is None else state.attrs

    @classmethod
    def def_attr(cls, name: str, *, default: Any = core.MISSING) -> EnumAttrMap:
        """Return a per-variant extensible-attribute view."""
        return EnumAttrMap(cls, name, default=default)


class EnumAttrMap:
    """Mutable singleton-keyed view of one extensible enum attribute."""

    __slots__ = ("_default", "_enum_cls", "_name")

    def __init__(self, enum_cls: type, name: str, *, default: Any = core.MISSING) -> None:
        self._enum_cls = enum_cls
        self._name = name
        self._default = default

    def _check(self, variant: object) -> None:
        if not isinstance(variant, self._enum_cls):
            raise TypeError(f"expected {self._enum_cls.__name__}, got {type(variant).__name__}")

    def _column(self, create: bool) -> Any:
        state = _state(self._enum_cls, create=create)
        if state is None:
            return None
        attrs = state.attrs
        column = attrs.get(self._name)
        if column is None and create:
            column = Dict({})
            attrs[self._name] = column
        return column

    def __setitem__(self, variant: object, value: Any) -> None:
        self._check(variant)
        self._column(True)[variant] = value

    def __getitem__(self, variant: object) -> Any:
        self._check(variant)
        column = self._column(False)
        if column is not None and variant in column:
            return column[variant]
        if self._default is not core.MISSING:
            return self._default
        raise KeyError(f"{variant!r} has no extensible attribute {self._name!r}")

    def __contains__(self, variant: object) -> bool:
        if not isinstance(variant, self._enum_cls):
            return False
        column = self._column(False)
        return column is not None and variant in column

    def get(self, variant: object, default: Any = None) -> Any:
        """Return a stored value or *default*."""
        if variant not in self:
            return default
        return self[variant]

    @property
    def name(self) -> str:
        """Return the extensible-attribute name."""
        return self._name


def _install_payload_behavior(cls: type, payload_type: type, user_repr: bool) -> None:
    def eq(self: Enum, other: object) -> Any:
        if isinstance(other, type(self)):
            return self.value == other.value
        if isinstance(other, payload_type):
            return self.value == other
        return NotImplemented

    def ne(self: Enum, other: object) -> Any:
        result = eq(self, other)
        return NotImplemented if result is NotImplemented else not result

    def name(self: Enum) -> str:
        return str(self._str_index)

    defaults = {
        "__eq__": eq,
        "__ne__": ne,
        "__str__": lambda self: str(self.value),
        "__hash__": lambda self: hash(self.value),
    }
    for key, value in defaults.items():
        if key not in cls.__dict__:
            setattr(cls, key, value)
    if "name" not in cls.__dict__:
        cls.name = property(name)  # ty: ignore[unresolved-attribute]
    if not user_repr:
        cls.__repr__ = lambda self: f"{type(self).__name__}.{self.name}"  # type: ignore[attr-defined]


@_enum_c_class("ffi.IntEnum", init=False)
class IntEnum(Enum):
    """Enum whose ``value`` aliases its signed 64-bit integer index."""

    if not TYPE_CHECKING:
        __slots__ = ()
        value: int

    if TYPE_CHECKING:

        @property
        def value(self) -> int:
            """Return the canonical integer index."""
            ...

    def __init_subclass__(cls, **kwargs: Any) -> None:
        _reject_value_field(cls, "IntEnum")
        super().__init_subclass__(**kwargs)


@_enum_c_class("ffi.StrEnum", init=False)
class StrEnum(Enum):
    """Enum whose ``name`` and ``value`` alias its string index."""

    if not TYPE_CHECKING:
        __slots__ = ()
        value: str

    if TYPE_CHECKING:

        @property
        def value(self) -> str:
            """Return the canonical string index."""
            ...

    def __init_subclass__(cls, **kwargs: Any) -> None:
        _reject_value_field(cls, "StrEnum")
        super().__init_subclass__(**kwargs)


def _collect_declarations(
    cls: type, payload_type: type | None
) -> tuple[list[str], dict[str, _EnumEntry]]:
    annotations = _own_annotations(cls)
    binders = [
        name
        for name, annotation in annotations.items()
        if not name.startswith("_") and _is_class_var(annotation) and name not in cls.__dict__
    ]
    fields = {name for name, annotation in annotations.items() if not _is_class_var(annotation)}
    declarations: dict[str, _EnumEntry] = {}
    for name, value in list(cls.__dict__.items()):
        if name.startswith("_") or name in fields:
            continue
        if isinstance(value, _EnumEntry):
            declarations[name] = value
        elif payload_type is not None and not (
            isinstance(value, (staticmethod, classmethod, property)) or callable(value)
        ):
            declarations[name] = _EnumEntry(value=value)
        else:
            continue
        delattr(cls, name)
    return binders, declarations


def _resolve(
    cls: type,
    binders: list[str],
    declarations: dict[str, _EnumEntry],
    type_key: str,
    cxx_backed: bool,
    payload_type: type | None,
) -> None:
    state = _state(cls, create=True)
    assert state is not None
    entries, indexes = state.entries, state.indexes
    for alias in binders:
        instance = indexes.get(alias)
        if instance is None:
            if cxx_backed:
                raise RuntimeError(f"{type_key!r} has no enum at string index {alias!r}")
            if payload_type is int:
                raise TypeError(f"{cls.__name__}.{alias} requires an integer value")
            instance = _create(cls, len(entries), alias, _EnumEntry(), cxx_backed)
            _register(entries, indexes, instance)
        setattr(cls, alias, instance)

    for alias, declaration in declarations.items():
        int_index, str_index, fields = _indices(cls, alias, len(entries), declaration, payload_type)
        lookup = int_index if payload_type is int else str_index
        instance = indexes.get(lookup)
        if instance is None:
            if int_index in indexes or str_index in indexes:
                raise ValueError(f"duplicate enum index in {cls.__name__}.{alias}")
            instance = _create(cls, int_index, str_index, fields, cxx_backed)
            _register(entries, indexes, instance)
        setattr(cls, alias, instance)


def _indices(
    cls: type,
    alias: str,
    ordinal: int,
    declaration: _EnumEntry,
    payload_type: type | None,
) -> tuple[int, str, _EnumEntry]:
    kwargs = dict(declaration.kwargs)
    if "_int_index" in kwargs or "_str_index" in kwargs:
        raise TypeError(f"{cls.__name__}.{alias}: enum indices are inferred")
    if payload_type is None:
        return ordinal, alias, declaration
    if "value" not in kwargs:
        raise TypeError(f"{cls.__name__}.{alias} requires a {payload_type.__name__} value")
    value = kwargs.pop("value")
    if type(value) is not payload_type:
        raise TypeError(f"{cls.__name__}.{alias} value must be {payload_type.__name__}")
    if payload_type is int:
        _normalize_index(value)
        indices = value, alias
    else:
        indices = ordinal, value
    return indices[0], indices[1], _EnumEntry(*declaration.args, **kwargs)


def _create(
    cls: type, int_index: int, str_index: str, declaration: _EnumEntry, cxx_backed: bool
) -> Any:
    try:
        if not cxx_backed:
            return cls(
                *declaration.args,
                _int_index=int_index,
                _str_index=str_index,
                **declaration.kwargs,
            )
        if declaration.args:
            raise TypeError("C++-backed enum entries require keyword fields")
        type_info = cls.__tvm_ffi_type_info__  # ty: ignore[unresolved-attribute]
        ffi_new = core._lookup_type_attr(type_info.type_index, "__ffi_new__")
        if ffi_new is None:
            raise TypeError("enum type has no allocator")
        instance = ffi_new()
        values = {"_int_index": int_index, "_str_index": str_index, **declaration.kwargs}
        for name, value in values.items():
            descriptor = getattr(cls, name, None)
            if descriptor is None or not hasattr(descriptor, "set"):
                raise TypeError(f"field {name!r} has no reflected setter")
            descriptor.set(instance, value)
        return instance
    except Exception as err:
        raise TypeError(f"{cls.__name__}.{str_index}: invalid enum entry: {err}") from None


def _register(entries: Any, indexes: Any, instance: Any) -> None:
    entries.append(instance)
    indexes[instance._int_index] = instance
    indexes[instance._str_index] = instance


def _own_annotations(cls: type) -> dict[str, Any]:
    if sys.version_info >= (3, 14):
        return dict(getattr(cls, "__annotations__", {}) or {})
    return dict(cls.__dict__.get("__annotations__", {}))


def _is_class_var(annotation: Any) -> bool:
    if annotation is ClassVar or typing.get_origin(annotation) is ClassVar:
        return True
    if isinstance(annotation, str):
        return annotation.replace(" ", "").startswith(("ClassVar", "typing.ClassVar"))
    return False


def _reject_value_field(cls: type, base_name: str) -> None:
    if "value" in cls.__dict__ or "value" in _own_annotations(cls):
        raise TypeError(f"{base_name} reserves `value` as its canonical index alias")
