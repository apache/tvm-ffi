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
"""Public helpers for describing dataclass-style defaults on FFI proxies."""

from __future__ import annotations

from dataclasses import _MISSING_TYPE, MISSING
from typing import Any, Callable, TypeVar, cast

_FieldValue = TypeVar("_FieldValue")


class Field:
    """(Experimental) Descriptor placeholder returned by :func:`tvm_ffi.dataclasses.field`.

    A ``Field`` mirrors the object returned by :func:`dataclasses.field`, but it
    is understood by :func:`tvm_ffi.dataclasses.c_class`.  The decorator inspects
    the ``Field`` instances, records the ``default_factory`` and later replaces
    the field with a property that forwards to the underlying C++ attribute.

    Users should not instantiate ``Field`` directly—use :func:`field` instead,
    which guarantees that ``name`` and ``default_factory`` are populated in a
    way the decorator understands.
    """

    __slots__ = ("default_factory", "name")

    def __init__(
        self,
        *,
        name: str | None = None,
        default_factory: Callable[[], _FieldValue] | _MISSING_TYPE = MISSING,
    ) -> None:
        """Do not call directly; use :func:`field` instead."""
        self.name = name
        self.default_factory = default_factory


def field(
    *,
    default: _FieldValue | _MISSING_TYPE = MISSING,  # type: ignore[assignment]
    default_factory: Callable[[], _FieldValue] | _MISSING_TYPE = MISSING,  # type: ignore[assignment]
) -> _FieldValue:
    """(Experimental) Declare a dataclass-style field on a :func:`c_class` proxy.

    Use this helper exactly like :func:`dataclasses.field` when defining the
    Python side of a C++ class.  When :func:`c_class` processes the class body it
    replaces the placeholder with a property and arranges for ``default`` or
    ``default_factory`` to be respected by the synthesized ``__init__``.

    Parameters
    ----------
    default : Any, optional
        A literal default value that should populate the field when no argument
        is given.  The value is copied into a closure because TVM FFI does not
        mutate the Python placeholder instance.
    default_factory : Callable[[], Any], optional
        A zero-argument callable that produces the default.  This matches the
        semantics of :func:`dataclasses.field` and is useful for mutable
        defaults such as ``list`` or ``dict``.

    Returns
    -------
    Field
        A placeholder object that :func:`c_class` will consume during class
        registration.

    Examples
    --------
    ``field`` integrates with :func:`c_class` to express defaults the same way a
    Python ``dataclass`` would::

        @c_class("testing.TestCxxClassBase")
        class PyBase:
            v_i64: int
            v_i32: int = field(default=16)

        obj = PyBase(v_i64=4)
        obj.v_i32  # -> 16

    """
    if default is not MISSING and default_factory is not MISSING:
        raise ValueError("Cannot specify both `default` and `default_factory`")
    if default is not MISSING:
        default_factory = _make_default_factory(default)
    ret = Field(default_factory=default_factory)
    return cast(_FieldValue, ret)


def _make_default_factory(value: Any) -> Callable[[], Any]:
    """Make a default factory that returns the given value."""

    def factory() -> Any:
        return value

    return factory
