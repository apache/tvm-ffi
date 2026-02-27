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
"""The ``c_class`` decorator: pass-through to ``register_object``."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

_T = TypeVar("_T", bound=type)


def c_class(type_key: str, **kwargs: Any) -> Callable[[_T], _T]:
    """Register a C++ FFI class by type key.

    This is a thin wrapper around :func:`~tvm_ffi.register_object` that
    accepts (and currently ignores) additional keyword arguments for
    forward compatibility.

    Parameters
    ----------
    type_key
        The reflection key that identifies the C++ type in the FFI registry.
    kwargs
        Reserved for future use.

    Returns
    -------
    Callable[[type], type]
        A class decorator.

    """
    from ..registry import register_object  # noqa: PLC0415

    return register_object(type_key)
