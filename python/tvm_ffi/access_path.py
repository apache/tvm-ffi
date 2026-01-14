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
# pylint: disable=invalid-name
"""Access path classes."""

# tvm-ffi-stubgen(begin): import-section
# fmt: off
# isort: off
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from collections.abc import Sequence
    from tvm_ffi import Object
    from typing import Any
# isort: on
# fmt: on
# tvm-ffi-stubgen(end)
from enum import IntEnum

from .core import Object
from .registry import register_object


class AccessKind(IntEnum):
    """Kinds of access steps in an access path."""

    ATTR = 0
    ARRAY_ITEM = 1
    MAP_ITEM = 2
    ATTR_MISSING = 3
    ARRAY_ITEM_MISSING = 4
    MAP_ITEM_MISSING = 5


@register_object("ffi.reflection.AccessStep")
class AccessStep(Object):
    """Access step container."""

    # tvm-ffi-stubgen(ty-map): ffi.reflection.AccessStep -> ffi.access_path.AccessStep
    # tvm-ffi-stubgen(begin): object/ffi.reflection.AccessStep
    # fmt: off
    kind: int
    key: Any
    # fmt: on
    # tvm-ffi-stubgen(end)


@register_object("ffi.reflection.AccessPath")
class AccessPath(Object):
    """Access path container.

    It describes how to reach a nested attribute or item
    inside a complex FFI object by recording a sequence of steps
    (attribute, array index, or map key). It is primarily used by
    diagnostics to pinpoint structural mismatches.

    Examples
    --------
    .. code-block:: python

        from tvm_ffi.access_path import AccessPath

        root = AccessPath.root()
        # Build a path equivalent to obj.layer.weight[2]
        p = root.attr("layer").attr("weight").array_item(2)
        assert isinstance(p, AccessPath)

    """

    # tvm-ffi-stubgen(ty-map): ffi.reflection.AccessPath -> ffi.access_path.AccessPath
    # tvm-ffi-stubgen(begin): object/ffi.reflection.AccessPath
    # fmt: off
    parent: Object | None
    step: AccessStep | None
    depth: int
    if TYPE_CHECKING:
        @staticmethod
        def _root() -> AccessPath: ...
        def _extend(self, _1: AccessStep, /) -> AccessPath: ...
        def _attr(self, _1: str, /) -> AccessPath: ...
        def _array_item(self, _1: int, /) -> AccessPath: ...
        def _map_item(self, _1: Any, /) -> AccessPath: ...
        def _attr_missing(self, _1: str, /) -> AccessPath: ...
        def _array_item_missing(self, _1: int, /) -> AccessPath: ...
        def _map_item_missing(self, _1: Any, /) -> AccessPath: ...
        def _is_prefix_of(self, _1: AccessPath, /) -> bool: ...
        def _to_steps(self, /) -> Sequence[AccessStep]: ...
        def _path_equal(self, _1: AccessPath, /) -> bool: ...
    # fmt: on
    # tvm-ffi-stubgen(end)

    def __init__(self) -> None:
        """Disallow direct construction; use `AccessPath.root()` instead."""
        super().__init__()
        raise ValueError(
            "AccessPath can't be initialized directly. "
            "Use AccessPath.root() to create a path to the root object"
        )

    @staticmethod
    def root() -> AccessPath:
        """Create a root access path.

        Returns
        -------
        AccessPath
            A path representing the root of an object graph.

        """
        return AccessPath._root()

    def __eq__(self, other: Any) -> bool:
        """Return whether two access paths are equal."""
        if not isinstance(other, AccessPath):
            return False
        return self._path_equal(other)

    def __ne__(self, other: Any) -> bool:
        """Return whether two access paths are not equal."""
        if not isinstance(other, AccessPath):
            return True
        return not self._path_equal(other)

    def is_prefix_of(self, other: AccessPath) -> bool:
        """Check if this access path is a prefix of another access path.

        Parameters
        ----------
        other
            The access path to check if it is a prefix of this access path

        Returns
        -------
        bool
            True if this access path is a prefix of the other access path, False otherwise

        """
        return self._is_prefix_of(other)

    def attr(self, attr_key: str) -> AccessPath:
        """Create an access path to the attribute of the current object.

        Parameters
        ----------
        attr_key
            The key of the attribute to access

        Returns
        -------
        AccessPath
            The extended access path

        """
        return self._attr(attr_key)

    def attr_missing(self, attr_key: str) -> AccessPath:
        """Create an access path that indicate an attribute is missing.

        Parameters
        ----------
        attr_key
            The key of the attribute to access

        Returns
        -------
        AccessPath
            The extended access path

        """
        return self._attr_missing(attr_key)

    def array_item(self, index: int) -> AccessPath:
        """Create an access path to the item of the current array.

        Parameters
        ----------
        index
            The index of the item to access

        Returns
        -------
        AccessPath
            The extended access path

        """
        return self._array_item(index)

    def array_item_missing(self, index: int) -> AccessPath:
        """Create an access path that indicate an array item is missing.

        Parameters
        ----------
        index
            The index of the item to access

        Returns
        -------
        AccessPath
            The extended access path

        """
        return self._array_item_missing(index)

    def map_item(self, key: Any) -> AccessPath:
        """Create an access path to the item of the current map.

        Parameters
        ----------
        key
            The key of the item to access

        Returns
        -------
        AccessPath
            The extended access path

        """
        return self._map_item(key)

    def map_item_missing(self, key: Any) -> AccessPath:
        """Create an access path that indicate a map item is missing.

        Parameters
        ----------
        key
            The key of the item to access

        Returns
        -------
        AccessPath
            The extended access path

        """
        return self._map_item_missing(key)

    def to_steps(self) -> Sequence[AccessStep]:
        """Convert the access path to a list of access steps.

        Returns
        -------
        access_steps
            The list of access steps

        """
        return self._to_steps()

    __hash__ = Object.__hash__
