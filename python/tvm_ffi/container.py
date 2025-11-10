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
"""Container classes."""

from __future__ import annotations

import itertools
import operator
import sys
from typing import (
    Any,
    Callable,
    SupportsIndex,
    TypeVar,
    cast,
    overload,
)

from . import _ffi_api, core
from .registry import register_object

if sys.version_info >= (3, 9):
    # PEP 585 generics
    from collections.abc import (
        ItemsView as ItemsViewBase,
    )
    from collections.abc import (
        Iterable,
        Iterator,
        Mapping,
        Sequence,
    )
    from collections.abc import (
        KeysView as KeysViewBase,
    )
    from collections.abc import (
        ValuesView as ValuesViewBase,
    )
else:  # Python 3.8
    # workarounds for python 3.8
    # typing-module generics (subscriptable on 3.8)
    from typing import (
        ItemsView as ItemsViewBase,
    )
    from typing import (
        Iterable,
        Iterator,
        Mapping,
        Sequence,
    )
    from typing import (
        KeysView as KeysViewBase,
    )
    from typing import (
        ValuesView as ValuesViewBase,
    )

__all__ = ["Array", "Map"]


T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")
_DefaultT = TypeVar("_DefaultT")


def getitem_helper(
    obj: Any,
    elem_getter: Callable[[Any, int], T],
    length: int,
    idx: SupportsIndex | slice,
) -> T | list[T]:
    """Implement a pythonic __getitem__ helper.

    Parameters
    ----------
    obj
        The original object

    elem_getter
        A simple function that takes index and return a single element.

    length
        The size of the array

    idx
        The argument passed to getitem

    Returns
    -------
    result
        The element for integer indices or a :class:`list` for slices.

    """
    if isinstance(idx, slice):
        start, stop, step = idx.indices(length)
        return [elem_getter(obj, i) for i in range(start, stop, step)]

    try:
        index = operator.index(idx)
    except TypeError as exc:  # pragma: no cover - defensive, matches list behaviour
        raise TypeError(f"indices must be integers or slices, not {type(idx).__name__}") from exc

    if index < -length or index >= length:
        raise IndexError(f"Index out of range. size: {length}, got index {index}")
    if index < 0:
        index += length
    return elem_getter(obj, index)


@register_object("ffi.Array")
class Array(core.Object, Sequence[T]):
    """Array container that represents a sequence of values in the FFI.

    :py:func:`tvm_ffi.convert` will map python list/tuple to this class.

    Parameters
    ----------
    input_list
        The list of values to be stored in the array.

    Examples
    --------
    .. code-block:: python

        import tvm_ffi

        a = tvm_ffi.Array([1, 2, 3])
        assert tuple(a) == (1, 2, 3)

    See Also
    --------
    :py:func:`tvm_ffi.convert`

    """

    def __init__(self, input_list: Iterable[T]) -> None:
        """Construct an Array from a Python sequence."""
        self.__init_handle_by_constructor__(_ffi_api.Array, *input_list)

    @overload
    def __getitem__(self, idx: SupportsIndex, /) -> T: ...

    @overload
    def __getitem__(self, idx: slice, /) -> list[T]: ...

    def __getitem__(self, idx: SupportsIndex | slice, /) -> T | list[T]:
        """Return one element or a list for a slice."""
        length = len(self)
        result = getitem_helper(self, _ffi_api.ArrayGetItem, length, idx)
        return result

    def __len__(self) -> int:
        """Return the number of elements in the array."""
        return _ffi_api.ArraySize(self)

    def __iter__(self) -> Iterator[T]:
        """Iterate over the elements in the array."""
        length = len(self)
        for i in range(length):
            yield self[i]

    def __repr__(self) -> str:
        """Return a string representation of the array."""
        # exception safety handling for chandle=None
        if self.__chandle__() == 0:
            return type(self).__name__ + "(chandle=None)"
        return "[" + ", ".join([x.__repr__() for x in self]) + "]"

    def __add__(self, other: Iterable[T]) -> Array[T]:
        """Concatenate two arrays."""
        return type(self)(itertools.chain(self, other))

    def __radd__(self, other: Iterable[T]) -> Array[T]:
        """Concatenate two arrays."""
        return type(self)(itertools.chain(other, self))


class KeysView(KeysViewBase[K]):
    """Helper class to return keys view."""

    def __init__(self, backend_map: Map[K, V]) -> None:
        self._backend_map = backend_map

    def __len__(self) -> int:
        return len(self._backend_map)

    def __iter__(self) -> Iterator[K]:
        size = len(self._backend_map)
        functor: Callable[[int], Any] = _ffi_api.MapForwardIterFunctor(self._backend_map)
        for _ in range(size):
            key = cast(K, functor(0))
            yield key
            if not functor(2):
                break

    def __contains__(self, k: object) -> bool:
        return k in self._backend_map


class ValuesView(ValuesViewBase[V]):
    """Helper class to return values view."""

    def __init__(self, backend_map: Map[K, V]) -> None:
        self._backend_map = backend_map

    def __len__(self) -> int:
        return len(self._backend_map)

    def __iter__(self) -> Iterator[V]:
        size = len(self._backend_map)
        functor: Callable[[int], Any] = _ffi_api.MapForwardIterFunctor(self._backend_map)
        for _ in range(size):
            value = cast(V, functor(1))
            yield value
            if not functor(2):
                break


class ItemsView(ItemsViewBase[K, V]):
    """Helper class to return items view."""

    def __init__(self, backend_map: Map[K, V]) -> None:
        self._backend_map = backend_map

    def __len__(self) -> int:
        return len(self._backend_map)

    def __iter__(self) -> Iterator[tuple[K, V]]:
        size = len(self._backend_map)
        functor: Callable[[int], Any] = _ffi_api.MapForwardIterFunctor(self._backend_map)
        for _ in range(size):
            key = cast(K, functor(0))
            value = cast(V, functor(1))
            yield (key, value)
            if not functor(2):
                break

    def __contains__(self, item: object) -> bool:
        if not isinstance(item, tuple) or len(item) != 2:
            return False
        key, value = item
        try:
            existing_value = self._backend_map[key]
        except KeyError:
            return False
        else:
            return existing_value == value


@register_object("ffi.Map")
class Map(core.Object, Mapping[K, V]):
    """Map container.

    :py:func:`tvm_ffi.convert` will map python dict to this class.

    Parameters
    ----------
    input_dict
        The dictionary of values to be stored in the map.

    Examples
    --------
    .. code-block:: python

        import tvm_ffi

        amap = tvm_ffi.Map({"a": 1, "b": 2})
        assert len(amap) == 2
        assert amap["a"] == 1
        assert amap["b"] == 2

    See Also
    --------
    :py:func:`tvm_ffi.convert`

    """

    def __init__(self, input_dict: Mapping[K, V]) -> None:
        """Construct a Map from a Python mapping."""
        list_kvs: list[Any] = []
        for k, v in input_dict.items():
            list_kvs.append(k)
            list_kvs.append(v)
        self.__init_handle_by_constructor__(_ffi_api.Map, *list_kvs)

    def __getitem__(self, k: K) -> V:
        """Return the value for key `k` or raise KeyError."""
        return cast(V, _ffi_api.MapGetItem(self, k))

    def __contains__(self, k: object) -> bool:
        """Return True if the map contains key `k`."""
        return _ffi_api.MapCount(self, k) != 0

    def keys(self) -> KeysView[K]:
        """Return a dynamic view of the map's keys."""
        return KeysView(self)

    def values(self) -> ValuesView[V]:
        """Return a dynamic view of the map's values."""
        return ValuesView(self)

    def items(self) -> ItemsView[K, V]:
        """Get the items from the map."""
        return ItemsView(self)

    def __len__(self) -> int:
        """Return the number of items in the map."""
        return _ffi_api.MapSize(self)

    def __iter__(self) -> Iterator[K]:
        """Iterate over the map's keys."""
        return iter(self.keys())

    @overload
    def get(self, key: K) -> V | None: ...

    @overload
    def get(self, key: K, default: V | _DefaultT) -> V | _DefaultT: ...

    def get(self, key: K, default: V | _DefaultT | None = None) -> V | _DefaultT | None:
        """Get an element with a default value.

        Parameters
        ----------
        key
            The attribute key.

        default
            The default object.

        Returns
        -------
        value
            The result value.

        """
        try:
            return self[key]
        except KeyError:
            return default

    def __repr__(self) -> str:
        """Return a string representation of the map."""
        # exception safety handling for chandle=None
        if self.__chandle__() == 0:
            return type(self).__name__ + "(chandle=None)"
        return "{" + ", ".join([f"{k.__repr__()}: {v.__repr__()}" for k, v in self.items()]) + "}"
