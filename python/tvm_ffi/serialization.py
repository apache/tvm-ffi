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
"""Utilities for serializing and deserializing FFI object graphs.

These helpers produce a stable JSON graph representation that preserves
object identity and references. It is useful for debugging and for
lightweight persistence when pickling is not available.
"""

from __future__ import annotations

from typing import Any

from . import _ffi_api


def to_json_graph_str(obj: Any, metadata: dict[str, Any] | None = None) -> str:
    """Dump an object to a JSON graph string.

    The JSON graph is a textual representation of the object graph that
    preserves shared references. It can be used for debugging or simple
    persistence.

    Parameters
    ----------
    obj
        The object to save.

    metadata
        Extra metadata to save into the json graph string.

    Returns
    -------
    json_str
        The JSON graph string.

    Examples
    --------
    .. code-block:: python

        import tvm_ffi

        a = tvm_ffi.convert([1, 2, 3])
        s = tvm_ffi.serialization.to_json_graph_str(a)
        b = tvm_ffi.serialization.from_json_graph_str(s)
        assert list(b) == [1, 2, 3]

    """
    return _ffi_api.ToJSONGraphString(obj, metadata)


def from_json_graph_str(json_str: str) -> Any:
    """Load an object from a JSON graph string.

    The JSON graph string is produced by :py:func:`to_json_graph_str` and
    can be converted back into the corresponding FFI-backed objects.

    Parameters
    ----------
    json_str
        The JSON graph string to load.

    Returns
    -------
    obj
        The loaded object.

    """
    return _ffi_api.FromJSONGraphString(json_str)


__all__ = ["from_json_graph_str", "to_json_graph_str"]
