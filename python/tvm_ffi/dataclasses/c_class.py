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
) -> Callable[[_T], _T]:
    """Register a C++ FFI class and install structural dunder methods.

    Combines :func:`~tvm_ffi.register_object` with structural comparison,
    hashing, and ordering derived from the C++ reflection metadata.
    User-defined dunders in the class body are never overwritten.

    Parameters
    ----------
    type_key
        The reflection key that identifies the C++ type in the FFI registry.
        Must match a key already registered on the C++ side via
        ``TVM_FFI_DECLARE_OBJECT_INFO``.
    init
        If True (default), install ``__init__`` from C++ reflection metadata.
        The generated ``__init__`` respects ``Init()``, ``KwOnly()``, and
        ``Default()`` traits declared on each C++ field.  If the class body
        already defines ``__init__``, it is kept.
    repr
        If True (default), install ``__repr__`` using
        :func:`~tvm_ffi.core.object_repr`, which formats the object via
        the C++ ``ReprPrint`` visitor.  Skipped if the class body already
        defines ``__repr__``.
    eq
        If True, install ``__eq__`` and ``__ne__`` using the C++ recursive
        structural comparison (``RecursiveEq``).  Returns ``NotImplemented``
        for unrelated types.  Defaults to False.
    order
        If True, install ``__lt__``, ``__le__``, ``__gt__``, ``__ge__``
        using the C++ recursive comparators.  Returns ``NotImplemented``
        for unrelated types.  Defaults to False.
    unsafe_hash
        If True, install ``__hash__`` using ``RecursiveHash``.  Called
        *unsafe* because mutable fields contribute to the hash, so mutating
        an object while it is in a set or dict key will break invariants.
        Defaults to False.

    Returns
    -------
    Callable[[type], type]
        A class decorator.

    Examples
    --------
    Basic usage with default settings (``init`` and ``repr`` enabled):

    .. code-block:: python

        @c_class("my.Point")
        class Point(Object):
            x: float
            y: float

    Enable structural equality, hashing, and ordering:

    .. code-block:: python

        @c_class("my.Point", eq=True, unsafe_hash=True, order=True)
        class Point(Object):
            x: float
            y: float

    See Also
    --------
    :func:`tvm_ffi.register_object`
        Lower-level decorator that only registers the type without
        installing structural dunders.

    """
    from ..registry import _install_dataclass_dunders, register_object  # noqa: PLC0415

    def decorator(cls: _T) -> _T:
        cls = register_object(type_key)(cls)
        _install_dataclass_dunders(
            cls, init=init, repr=repr, eq=eq, order=order, unsafe_hash=unsafe_hash, pre_alloc=True
        )
        return cls

    return decorator
