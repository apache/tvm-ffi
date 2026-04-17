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

from typing import Any

from .field import Field

__all__ = ["fields", "is_dataclass", "replace"]


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
