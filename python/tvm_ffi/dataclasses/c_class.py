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
"""The ``c_class`` decorator: register_object + structural dunders."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from typing_extensions import dataclass_transform

_T = TypeVar("_T", bound=type)


@dataclass_transform(eq_default=False, order_default=False)
def c_class(
    type_key: str,
    *,
    init: bool = True,
    repr: bool = True,
    eq: bool = False,
    order: bool = False,
    unsafe_hash: bool = False,
    slots: bool = True,
) -> Callable[[_T], _T]:
    """Register a C++ FFI class and install structural dunder methods.

    Combines :func:`~tvm_ffi.register_object` with structural comparison,
    hashing, and ordering derived from the C++ reflection metadata.

    Parameters
    ----------
    type_key
        The reflection key that identifies the C++ type in the FFI registry.
    init
        If True, install ``__init__`` from C++ reflection metadata.
    repr
        If True, install ``__repr__`` from C++ reflection metadata.
    eq
        If True, install ``__eq__`` and ``__ne__``.
    order
        If True, install ``__lt__``, ``__le__``, ``__gt__``, ``__ge__``.
    unsafe_hash
        If True, install ``__hash__``.
    slots
        Declares whether the class should use ``__slots__``.  The actual
        slotting is controlled by the metaclass keyword ``slots`` in the
        class header (e.g. ``class Foo(Object, slots=False):``).  This
        parameter validates that the class header matches the declared
        intent; a mismatch raises ``TypeError``.

    Returns
    -------
    Callable[[type], type]
        A class decorator.

    """
    from ..registry import _install_dataclass_dunders, register_object  # noqa: PLC0415

    def decorator(cls: _T) -> _T:
        from ..core import Object as _Object  # noqa: PLC0415

        if issubclass(cls, _Object):
            has_slots = "__slots__" in cls.__dict__
            if slots and not has_slots:
                raise TypeError(
                    f"`@c_class({type_key!r}, slots=True)` expects `__slots__`, "
                    f"but `{cls.__name__}` was defined with `slots=False` in the "
                    f"class header. Either remove `slots=False` from "
                    f"`class {cls.__name__}(Object, slots=False)` "
                    f"or pass `slots=False` to @c_class."
                )
            if not slots and has_slots:
                raise TypeError(
                    f"`@c_class({type_key!r}, slots=False)` expects no `__slots__`, "
                    f"but `{cls.__name__}` has `__slots__` set. "
                    f"Add `slots=False` to the class header: "
                    f"`class {cls.__name__}(Object, slots=False)`."
                )
        cls = register_object(type_key)(cls)
        _install_dataclass_dunders(
            cls, init=init, repr=repr, eq=eq, order=order, unsafe_hash=unsafe_hash
        )
        return cls

    return decorator
