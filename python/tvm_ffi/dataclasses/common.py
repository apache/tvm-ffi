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
"""``dataclasses``-style helpers unified over stdlib, ``@c_class``, and ``@py_class``."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

from ..container import Array, Dict, List, Map
from .field import Field

__all__ = ["asdict", "astuple", "fields", "is_dataclass", "replace"]

# Exact-type fast path for atomic (immutable, non-recursive) values.  Mirrors
# :data:`dataclasses._ATOMIC_TYPES` from the standard library.
_ATOMIC_TYPES: frozenset[type] = frozenset(
    {
        bool,
        bytes,
        complex,
        float,
        int,
        str,
        type(None),
        type(Ellipsis),
        type(NotImplemented),
    }
)


def is_dataclass(obj: Any) -> bool:
    """Return True if ``obj`` is a ``@c_class`` / ``@py_class`` type or instance."""
    cls = obj if isinstance(obj, type) else type(obj)
    return getattr(cls, "__tvm_ffi_type_info__", None) is not None


def fields(obj_or_cls: Any) -> tuple[Field, ...]:
    """Return the :class:`~tvm_ffi.dataclasses.Field` descriptors for a type.

    Accepts a ``@c_class`` / ``@py_class`` type or instance and walks the
    parent chain so inherited fields appear parent-first, matching the
    order of the auto-generated ``__init__``.

    Raises
    ------
    TypeError
        If ``obj_or_cls`` is not a ``@c_class`` / ``@py_class`` type or instance.

    """
    cls = obj_or_cls if isinstance(obj_or_cls, type) else type(obj_or_cls)
    ti = getattr(cls, "__tvm_ffi_type_info__", None)
    if ti is None:
        raise TypeError(
            f"fields() argument must be a c_class or py_class type or instance, "
            f"got {type(obj_or_cls).__name__}"
        )
    chain = []
    while ti is not None:
        chain.append(ti)
        ti = ti.parent_type_info
    out: list[Field] = []
    for ti in reversed(chain):
        for tf in ti.fields or ():
            if tf.dataclass_field is not None:
                out.append(tf.dataclass_field)
    return tuple(out)


def replace(obj: Any, /, **changes: Any) -> Any:
    """Return a copy of ``obj`` with selected fields replaced.

    Drop-in for :func:`dataclasses.replace` for FFI-backed instances: the
    call is forwarded to ``obj.__replace__`` (installed by the decorator),
    which uses the ``FFIProperty.set()`` escape hatch so frozen fields are
    still replaceable.
    """
    return obj.__replace__(**changes)


def _is_ffi_dataclass_instance(obj: Any) -> bool:
    """Return True when *obj* is a ``@c_class`` / ``@py_class`` **instance** (not a type)."""
    if isinstance(obj, type):
        return False
    return getattr(type(obj), "__tvm_ffi_type_info__", None) is not None


def _asdict_inner(  # noqa: PLR0911, PLR0912
    obj: Any, dict_factory: Callable[..., Any]
) -> Any:
    obj_type = type(obj)
    if obj_type in _ATOMIC_TYPES:
        return obj
    # FFI containers are treated as their stdlib analogues so the result is
    # plain Python data — handy for JSON serialisation, the main use case.
    if isinstance(obj, (Array, List)):
        return [_asdict_inner(v, dict_factory) for v in obj]
    if isinstance(obj, (Map, Dict)):
        return dict_factory(
            [
                (_asdict_inner(k, dict_factory), _asdict_inner(v, dict_factory))
                for k, v in obj.items()
            ]
        )
    if _is_ffi_dataclass_instance(obj):
        fs = fields(obj)
        if dict_factory is dict:
            return {f.name: _asdict_inner(getattr(obj, f.name), dict) for f in fs}  # ty: ignore[invalid-argument-type]
        return dict_factory(
            [(f.name, _asdict_inner(getattr(obj, f.name), dict_factory)) for f in fs]  # ty: ignore[invalid-argument-type]
        )
    if obj_type is list:
        return [_asdict_inner(v, dict_factory) for v in obj]
    if obj_type is dict:
        return {
            _asdict_inner(k, dict_factory): _asdict_inner(v, dict_factory) for k, v in obj.items()
        }
    if obj_type is tuple:
        return tuple(_asdict_inner(v, dict_factory) for v in obj)
    if issubclass(obj_type, tuple):
        if hasattr(obj, "_fields"):  # namedtuple
            return obj_type(*[_asdict_inner(v, dict_factory) for v in obj])
        return obj_type(_asdict_inner(v, dict_factory) for v in obj)
    if issubclass(obj_type, dict):
        if hasattr(obj_type, "default_factory"):
            result = obj_type(obj.default_factory)
            for k, v in obj.items():
                result[_asdict_inner(k, dict_factory)] = _asdict_inner(v, dict_factory)
            return result
        return obj_type(
            (_asdict_inner(k, dict_factory), _asdict_inner(v, dict_factory)) for k, v in obj.items()
        )
    if issubclass(obj_type, list):
        return obj_type(_asdict_inner(v, dict_factory) for v in obj)
    return copy.deepcopy(obj)


def _astuple_inner(obj: Any, tuple_factory: Callable[..., Any]) -> Any:  # noqa: PLR0911
    obj_type = type(obj)
    if obj_type in _ATOMIC_TYPES:
        return obj
    if isinstance(obj, (Array, List)):
        return [_astuple_inner(v, tuple_factory) for v in obj]
    if isinstance(obj, (Map, Dict)):
        return {
            _astuple_inner(k, tuple_factory): _astuple_inner(v, tuple_factory)
            for k, v in obj.items()
        }
    if _is_ffi_dataclass_instance(obj):
        return tuple_factory(
            [_astuple_inner(getattr(obj, f.name), tuple_factory) for f in fields(obj)]  # ty: ignore[invalid-argument-type]
        )
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):  # namedtuple
        return obj_type(*[_astuple_inner(v, tuple_factory) for v in obj])
    if isinstance(obj, (list, tuple)):
        return obj_type(_astuple_inner(v, tuple_factory) for v in obj)
    if isinstance(obj, dict):
        if hasattr(obj_type, "default_factory"):
            result = obj_type(obj.default_factory)
            for k, v in obj.items():
                result[_astuple_inner(k, tuple_factory)] = _astuple_inner(v, tuple_factory)
            return result
        return obj_type(
            (_astuple_inner(k, tuple_factory), _astuple_inner(v, tuple_factory))
            for k, v in obj.items()
        )
    return copy.deepcopy(obj)


def asdict(obj: Any, *, dict_factory: Callable[..., Any] = dict) -> Any:
    r"""Return the fields of a ``@c_class`` / ``@py_class`` instance as a new dict.

    Mirrors :func:`dataclasses.asdict` from the standard library.  The
    function recurses into nested FFI dataclass instances, FFI containers
    (:class:`~tvm_ffi.Array`, :class:`~tvm_ffi.List`,
    :class:`~tvm_ffi.Map`, :class:`~tvm_ffi.Dict`), and the built-in
    ``list`` / ``tuple`` / ``dict``.  FFI sequence containers are
    converted to Python ``list``\ s and FFI mapping containers to
    Python ``dict``\ s so the result is plain Python data, e.g. for
    JSON serialisation.  Any other value is copied with
    :func:`copy.deepcopy`.

    Parameters
    ----------
    obj
        A ``@c_class`` / ``@py_class`` instance.  Passing a type raises
        :class:`TypeError`.
    dict_factory
        Callable used to construct the outer mapping and any nested
        mapping recursed from an FFI dataclass.  Defaults to :class:`dict`.

    Raises
    ------
    TypeError
        If ``obj`` is not a ``@c_class`` / ``@py_class`` instance.

    """
    if not _is_ffi_dataclass_instance(obj):
        raise TypeError("asdict() should be called on c_class / py_class instances")
    return _asdict_inner(obj, dict_factory)


def astuple(obj: Any, *, tuple_factory: Callable[..., Any] = tuple) -> Any:
    """Return the fields of a ``@c_class`` / ``@py_class`` instance as a new tuple.

    Mirrors :func:`dataclasses.astuple` from the standard library.  The
    function recurses into nested FFI dataclass instances, FFI containers
    (:class:`~tvm_ffi.Array`, :class:`~tvm_ffi.List`,
    :class:`~tvm_ffi.Map`, :class:`~tvm_ffi.Dict`), and the built-in
    ``list`` / ``tuple`` / ``dict``.  Any other value is copied with
    :func:`copy.deepcopy`.

    Parameters
    ----------
    obj
        A ``@c_class`` / ``@py_class`` instance.  Passing a type raises
        :class:`TypeError`.
    tuple_factory
        Callable used to construct the outer tuple and any nested tuple
        recursed from an FFI dataclass.  Defaults to :class:`tuple`.

    Raises
    ------
    TypeError
        If ``obj`` is not a ``@c_class`` / ``@py_class`` instance.

    """
    if not _is_ffi_dataclass_instance(obj):
        raise TypeError("astuple() should be called on c_class / py_class instances")
    return _astuple_inner(obj, tuple_factory)
