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
        MutableSequence,
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
        MutableSequence,
        Sequence,
    )
    from typing import (
        KeysView as KeysViewBase,
    )
    from typing import (
        ValuesView as ValuesViewBase,
    )

__all__ = ["Array", "List", "Map"]


T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")
_DefaultT = TypeVar("_DefaultT")

MISSING = _ffi_api.MapGetMissingObject()


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

    index = normalize_index(length, idx)
    return elem_getter(obj, index)


def normalize_index(length: int, idx: SupportsIndex) -> int:
    """Normalize and bounds-check a Python index."""
    try:
        index = operator.index(idx)
    except TypeError as exc:  # pragma: no cover - defensive, matches list behaviour
        raise TypeError(f"indices must be integers or slices, not {type(idx).__name__}") from exc
    if index < -length or index >= length:
        raise IndexError(f"Index out of range. size: {length}, got index {index}")
    if index < 0:
        index += length
    return index


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

    Notes
    -----
    For structural equality and hashing, use ``structural_equal`` and ``structural_hash`` APIs.

    See Also
    --------
    :py:func:`tvm_ffi.convert`

    """

    # tvm-ffi-stubgen(begin): object/ffi.Array
    # fmt: off
    # fmt: on
    # tvm-ffi-stubgen(end)

    def __init__(self, input_list: Iterable[T]) -> None:
        """Construct an Array from a Python sequence."""
        self.__init_handle_by_constructor__(_ffi_api.Array, *input_list)

    @overload
    def __getitem__(self, idx: SupportsIndex, /) -> T: ...

    @overload
    def __getitem__(self, idx: slice, /) -> list[T]: ...

    def __getitem__(self, idx: SupportsIndex | slice, /) -> T | list[T]:  # ty: ignore[invalid-method-override]
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

    def __contains__(self, value: object) -> bool:
        """Check if the array contains a value."""
        return _ffi_api.ArrayContains(self, value)

    def __bool__(self) -> bool:
        """Return True if the array is non-empty."""
        return len(self) > 0

    def __add__(self, other: Iterable[T]) -> Array[T]:
        """Concatenate two arrays."""
        return type(self)(itertools.chain(self, other))

    def __radd__(self, other: Iterable[T]) -> Array[T]:
        """Concatenate two arrays."""
        return type(self)(itertools.chain(other, self))


@register_object("ffi.List")
class List(core.Object, MutableSequence[T]):
    """Mutable list container that represents a mutable sequence in the FFI."""

    # tvm-ffi-stubgen(begin): object/ffi.List
    # fmt: off
    # fmt: on
    # tvm-ffi-stubgen(end)

    def __init__(self, input_list: Iterable[T] = ()) -> None:
        """Construct a List from a Python sequence."""
        self.__init_handle_by_constructor__(_ffi_api.List, *input_list)

    @overload
    def __getitem__(self, idx: SupportsIndex, /) -> T: ...

    @overload
    def __getitem__(self, idx: slice, /) -> list[T]: ...

    def __getitem__(self, idx: SupportsIndex | slice, /) -> T | list[T]:  # ty: ignore[invalid-method-override]
        """Return one element or a list for a slice."""
        length = len(self)
        return getitem_helper(self, _ffi_api.ListGetItem, length, idx)

    @overload
    def __setitem__(self, index: SupportsIndex, value: T) -> None: ...

    @overload
    def __setitem__(self, index: slice[int | None], value: Iterable[T]) -> None: ...

    def __setitem__(self, index: SupportsIndex | slice[int | None], value: T | Iterable[T]) -> None:
        """Set one element or assign a slice."""
        if isinstance(index, slice):
            replacement = list(cast(Iterable[T], value))
            length = len(self)
            start, stop, step = index.indices(length)
            if step != 1:
                target_indices = list(range(start, stop, step))
                if len(replacement) != len(target_indices):
                    raise ValueError(
                        "attempt to assign sequence of size "
                        f"{len(replacement)} to extended slice of size {len(target_indices)}"
                    )
                for i, item in zip(target_indices, replacement):
                    _ffi_api.ListSetItem(self, i, item)
                return
            stop = max(stop, start)
            _ffi_api.ListReplaceSlice(self, start, stop, type(self)(replacement))
            return

        normalized_index = normalize_index(len(self), index)
        _ffi_api.ListSetItem(self, normalized_index, cast(T, value))

    @overload
    def __delitem__(self, index: SupportsIndex) -> None: ...

    @overload
    def __delitem__(self, index: slice[int | None]) -> None: ...

    def __delitem__(self, index: SupportsIndex | slice[int | None]) -> None:
        """Delete one element or a slice."""
        if isinstance(index, slice):
            length = len(self)
            start, stop, step = index.indices(length)
            if step == 1:
                stop = max(stop, start)
                _ffi_api.ListEraseRange(self, start, stop)
            else:
                # Delete indices from high to low so that earlier deletions
                # do not shift the positions of later ones.
                indices = (
                    reversed(range(start, stop, step)) if step > 0 else range(start, stop, step)
                )
                for i in indices:
                    _ffi_api.ListErase(self, i)
            return
        normalized_index = normalize_index(len(self), index)
        _ffi_api.ListErase(self, normalized_index)

    def insert(self, index: int, value: T) -> None:
        """Insert value before index."""
        length = len(self)
        if index < 0:
            index = max(0, index + length)
        else:
            index = min(index, length)
        _ffi_api.ListInsert(self, index, value)

    def append(self, value: T) -> None:
        """Append one value to the tail."""
        _ffi_api.ListAppend(self, value)

    def clear(self) -> None:
        """Remove all elements from the list."""
        _ffi_api.ListClear(self)

    def reverse(self) -> None:
        """Reverse the list in-place."""
        _ffi_api.ListReverse(self)

    def pop(self, index: int = -1) -> T:
        """Remove and return item at index (default last)."""
        length = len(self)
        if length == 0:
            raise IndexError("pop from empty list")
        normalized_index = normalize_index(length, index)
        return cast(T, _ffi_api.ListPop(self, normalized_index))

    def extend(self, values: Iterable[T]) -> None:
        """Append elements from an iterable."""
        end = len(self)
        self[end:end] = values

    def __len__(self) -> int:
        """Return the number of elements in the list."""
        return _ffi_api.ListSize(self)

    def __iter__(self) -> Iterator[T]:
        """Iterate over the elements in the list."""
        length = len(self)
        for i in range(length):
            yield cast(T, _ffi_api.ListGetItem(self, i))

    def __repr__(self) -> str:
        """Return a string representation of the list."""
        if self.__chandle__() == 0:
            return type(self).__name__ + "(chandle=None)"
        return "[" + ", ".join([x.__repr__() for x in self]) + "]"

    def __contains__(self, value: object) -> bool:
        """Check if the list contains a value."""
        return _ffi_api.ListContains(self, value)

    def __bool__(self) -> bool:
        """Return True if the list is non-empty."""
        return len(self) > 0

    def __add__(self, other: Iterable[T]) -> List[T]:
        """Concatenate two lists."""
        return type(self)(itertools.chain(self, other))

    def __radd__(self, other: Iterable[T]) -> List[T]:
        """Concatenate two lists."""
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

    def __contains__(self, k: object) -> bool:  # ty: ignore[invalid-method-override]
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
        actual_value = self._backend_map.get(key, MISSING)  # ty: ignore[invalid-argument-type]
        if actual_value is MISSING:
            return False
        # TODO(@junrus): Is `__eq__` the right method to use here?
        return actual_value == value


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

    Notes
    -----
    For structural equality and hashing, use ``structural_equal`` and ``structural_hash`` APIs.

    See Also
    --------
    :py:func:`tvm_ffi.convert`

    """

    # tvm-ffi-stubgen(begin): object/ffi.Map
    # fmt: off
    # fmt: on
    # tvm-ffi-stubgen(end)

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

    def __bool__(self) -> bool:
        """Return True if the map is non-empty."""
        return len(self) > 0

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
        ret = _ffi_api.MapGetItemOrMissing(self, key)
        if MISSING.same_as(ret):
            return default
        return ret

    def __repr__(self) -> str:
        """Return a string representation of the map."""
        # exception safety handling for chandle=None
        if self.__chandle__() == 0:
            return type(self).__name__ + "(chandle=None)"
        return "{" + ", ".join([f"{k.__repr__()}: {v.__repr__()}" for k, v in self.items()]) + "}"
