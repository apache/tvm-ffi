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
"""Structural helper objects and functions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import IntEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .access_path import AccessPath

from . import _ffi_api, core
from .core import Object
from .registry import register_object

__all__ = [
    "DefRegionKind",
    "StructuralKey",
    "StructuralVisitor",
    "VisitInterrupt",
    "WalkOrder",
    "WalkResult",
    "get_first_structural_mismatch",
    "structural_equal",
    "structural_hash",
    "structural_walk",
]


class WalkOrder(IntEnum):
    """Callback placement before or after visiting children for structural walks.

    See Also
    --------
    :py:func:`tvm_ffi.structural_walk`
        Walk an object graph, invoke matching callbacks.

    """

    PREORDER = 0
    POSTORDER = 1


class WalkResult(IntEnum):
    """Control signal for structural walks.

    Advance continues visiting a node's children; skip continues traversal but
    skips the current node's children.

    Use :class:`VisitInterrupt` when traversal should stop entirely.

    See Also
    --------
    :py:func:`tvm_ffi.structural_walk`
        Walk an object graph, invoke matching callbacks.

    """

    ADVANCE = 0
    SKIP = 1


class DefRegionKind(IntEnum):
    """Def-region state active during structural visiting.

    The values mirror ``TVMFFIDefRegionKind`` in the C ABI.

    See Also
    --------
    :py:class:`tvm_ffi.StructuralVisitor`
        Structural traversal visitor that carries object dispatch and def-region
        state across recursive visits.

    """

    NONE = 0
    DEF_RECURSIVE = 1
    DEF_NON_RECURSIVE = 2


def structural_equal(
    lhs: Any, rhs: Any, map_free_vars: bool = False, skip_tensor_content: bool = False
) -> bool:
    """Check structural equality between two values.

    Structural equality compares the *shape/content structure* of two values
    instead of Python object identity. For container-like values, this means
    recursive comparison of elements/fields. For object types that provide
    structural equal hooks, those hooks are used.

    Parameters
    ----------
    lhs
        Left-hand side value.
    rhs
        Right-hand side value.

    map_free_vars
        Whether free variables (variables without a definition site) can be
        mapped to each other during comparison.

    skip_tensor_content
        Whether to skip tensor data content when comparing tensors.

    Returns
    -------
    result
        Whether the two values are structurally equal.

    Examples
    --------
    .. code-block:: python

        import tvm_ffi

        assert tvm_ffi.structural_equal([1, 2, 3], [1, 2, 3])
        assert not tvm_ffi.structural_equal([1, 2, 3], [1, 2, 4])

    See Also
    --------
    :py:func:`tvm_ffi.structural_hash`
        Hash function compatible with structural equality.
    :py:func:`tvm_ffi.get_first_structural_mismatch`
        Mismatch diagnostics with access paths.

    """
    return _ffi_api.StructuralEqual(lhs, rhs, map_free_vars, skip_tensor_content)


def structural_hash(
    value: Any, map_free_vars: bool = False, skip_tensor_content: bool = False
) -> int:
    """Compute structural hash of a value.

    This hash is designed to be consistent with :py:func:`structural_equal`
    under the same options. If two values are structurally equal, they should
    have the same structural hash.

    Parameters
    ----------
    value
        Input value to hash.

    map_free_vars
        Whether free variables mapped to each other during hashing.

    skip_tensor_content
        Whether tensor data content is ignored when hashing tensors.

    Returns
    -------
    hash_value
        Structural hash value.

    Examples
    --------
    .. code-block:: python

        import tvm_ffi

        h0 = tvm_ffi.structural_hash([1, 2, 3])
        h1 = tvm_ffi.structural_hash([1, 2, 3])
        assert h0 == h1

    Notes
    -----
    Structural hash is intended for hash-table bucketing and fast pre-checks.
    Always use structural equality to confirm semantic equivalence.

    """
    # need to mask the result so negative values are converted to u64
    # this is because hash value were stored as int64_t in the C++ code
    return _ffi_api.StructuralHash(value, map_free_vars, skip_tensor_content) & 0xFFFFFFFFFFFFFFFF


def get_first_structural_mismatch(
    lhs: Any, rhs: Any, map_free_vars: bool = False, skip_tensor_content: bool = False
) -> tuple[AccessPath, AccessPath] | None:
    """Like structural_equal(), but returns the AccessPath pair of the first detected mismatch.

    Parameters
    ----------
    lhs
        The left operand.

    rhs
        The right operand.

    map_free_vars
        Whether free variables (i.e. variables without a definition site) should be mapped
        as equal to each other.

    skip_tensor_content
        Whether to skip the data content of tensor.

    Returns
    -------
    mismatch: tuple[AccessPath, AccessPath] | None
        `None` if `lhs` and `rhs` are structurally equal.
        Otherwise, a tuple of two AccessPath objects that point to the first detected mismatch.

    """
    return _ffi_api.GetFirstStructuralMismatch(lhs, rhs, map_free_vars, skip_tensor_content)


@register_object("ffi.StructuralKey")
class StructuralKey(Object):
    """Hash-cached structural key wrapper.

    This wrapper can be used to hint that a dict uses structural equality and hash for the key.

    Examples
    --------
    Use ``StructuralKey`` with Python dictionaries when you want key lookup by
    structural semantics:

    .. code-block:: python

        import tvm_ffi

        k0 = tvm_ffi.StructuralKey([1, 2, 3])
        k1 = tvm_ffi.StructuralKey([1, 2, 3])
        k2 = tvm_ffi.StructuralKey([1, 2, 4])

        d = {k0: "value-a", k2: "value-b"}
        assert d[k1] == "value-a"  # k1 matches k0 structurally
        assert d[k2] == "value-b"

    It can also be used directly with :py:class:`tvm_ffi.Map`:

    .. code-block:: python

        m = tvm_ffi.Map({k0: 1, k1: 2})
        assert len(m) == 1
        assert m[k0] == 2

    See Also
    --------
    :py:func:`tvm_ffi.structural_equal`
        Structural equality comparison.
    :py:func:`tvm_ffi.structural_hash`
        Structural hash computation.

    """

    # tvm-ffi-stubgen(begin): object/ffi.StructuralKey
    # fmt: off
    key: Any
    hash_i64: int
    if TYPE_CHECKING:
        def __init__(self, key: Any, hash_i64: int) -> None: ...
        def __ffi_init__(self, _0: Any, /) -> None: ...  # ty: ignore[invalid-method-override]
    # fmt: on
    # tvm-ffi-stubgen(end)

    def __init__(self, key: Any) -> None:
        """Create a structural key from ``key``.

        Parameters
        ----------
        key
            The underlying value used for structural hash/equality.

        """
        self.__init_handle_by_constructor__(_ffi_api.StructuralKey, key)

    def __hash__(self) -> int:
        """Return cached structural hash."""
        # need to mask the result so negative values are converted to u64
        # this is because hash value were stored as int64_t in the C++ code
        return self.hash_i64 & 0xFFFFFFFFFFFFFFFF

    def __eq__(self, other: Any) -> bool:
        """Compare by structural equality."""
        return isinstance(other, StructuralKey) and _ffi_api.StructuralKeyEqual(self, other)


@register_object("ffi.VisitInterrupt")
class VisitInterrupt(Object):
    """Payload-carrying signal that stops a structural visit.

    This object can be returned from structural walk callbacks and structural
    visit hooks to halt traversal early. The optional payload is preserved in
    :py:attr:`value` and returned to the caller.

    Examples
    --------
    Use ``VisitInterrupt`` to stop a structural walk when a target node is found:

    .. code-block:: python

        import tvm_ffi


        def on_node(node):
            if is_target(node):
                return tvm_ffi.VisitInterrupt(node)
            return None


        result = tvm_ffi.structural_walk(root, (object, on_node))
        if result is not None:
            found = result.value

    See Also
    --------
    :py:func:`tvm_ffi.structural_walk`
        Structural walk API whose callbacks may return ``VisitInterrupt``.

    """

    # tvm-ffi-stubgen(begin): object/ffi.VisitInterrupt
    # fmt: off
    value: Any
    if TYPE_CHECKING:
        def __init__(self, value: Any) -> None: ...
        def __ffi_init__(self, _0: Any, /) -> None: ...  # ty: ignore[invalid-method-override]
    # fmt: on
    # tvm-ffi-stubgen(end)

    def __init__(self, value: Any = None) -> None:
        """Create an interrupt with an optional payload.

        Parameters
        ----------
        value
            Payload returned to the caller when traversal stops.

        """
        self.__init_handle_by_constructor__(_ffi_api.VisitInterrupt, value)


@register_object("ffi.StructuralVisitor")
class StructuralVisitor(Object):
    """Low-level structural traversal visitor.

    This class exposes the low-level visitor object used by structural
    traversal hooks. Instances are supplied by an active traversal and cannot
    be constructed directly.
    """

    def visit(self, value: Any) -> VisitInterrupt | None:
        """Low-level API to visit ``value`` using this visitor's dispatch behavior.

        Parameters
        ----------
        value
            Value to visit.

        Returns
        -------
        result
            ``None`` if traversal should continue, otherwise a
            :class:`VisitInterrupt` carrying the early-exit payload.

        """
        return _ffi_api.StructuralVisitorVisit(self, value)

    def def_region_kind(self) -> DefRegionKind:
        """Low-level API to return the currently active structural def-region kind.

        Returns
        -------
        kind
            The active :class:`DefRegionKind`.

        """
        return DefRegionKind(_ffi_api.StructuralVisitorDefRegionKind(self))

    def with_def_region_kind(
        self,
        kind: int,
        callback: Callable[[], Any],
    ) -> Any:
        """Low-level API to run ``callback`` with a temporarily active def-region kind.

        Parameters
        ----------
        kind
            Def region kind to use while running ``callback``.

        callback
            Nullary callable to execute inside the scoped region.

        Returns
        -------
        result
            The value returned by ``callback``.

        """
        return _ffi_api.StructuralVisitorWithDefRegionKind(self, kind, callback)


def structural_walk(
    root: Any,
    callbacks: tuple | Sequence | Callable = (),
    with_def_region_kind: tuple | Sequence | Callable = (),
    order: str | WalkOrder = "pre",
) -> VisitInterrupt | None:
    """Walk a value structurally and invoke the first matching typed callback.

    Parameters
    ----------
    root
        Root value to traverse.

    callbacks
        Normal callbacks. These callbacks receive one argument: ``value``.
        Callback entries are tried in order.

        May be one of:

        - A single callback, used as a ``typing.Any`` catch-all.
        - A ``(type, callback)`` entry.
        - A grouped ``((type1, type2, ...), callback)`` entry.
        - A sequence of entries.

        Types may be builtins, registered FFI object classes, or
        ``typing.Any``/``object`` as a catch-all.

    with_def_region_kind
        Def-region-aware callbacks. These callbacks receive two arguments:
        ``(value, def_region_kind)``. They accept the same callback entry forms
        as ``callbacks``.

    order
        ``"pre"``/``WalkOrder.PREORDER`` to invoke callbacks before children, or
        ``"post"``/``WalkOrder.POSTORDER`` to invoke callbacks after children.

    Returns
    -------
    result
        ``None`` if traversal completed, otherwise a :class:`VisitInterrupt`
        returned by a callback.

    Examples
    --------
    .. code-block:: python

        visited = []


        uses = []
        result = tvm_ffi.structural_walk(
            node,
            ((int, float), lambda value: visited.append(("leaf", value))),
            with_def_region_kind=(
                Var,
                lambda var, kind: (
                    uses.append(var) if kind == tvm_ffi.DefRegionKind.NONE else None
                ),
            ),
        )

    """
    if isinstance(order, WalkOrder):
        order_int = int(order)
    elif order in ("pre", "post"):
        order_int = int(WalkOrder.PREORDER if order == "pre" else WalkOrder.POSTORDER)
    else:
        raise ValueError(f"Unknown structural walk order: {order!r}")

    def normalize_callbacks(
        callbacks: tuple | Sequence | Callable,
    ) -> list[tuple[object, Callable]]:
        callback_entries = []

        def add_callback_entry(callback_entry: tuple) -> None:
            callback_type, fn = callback_entry
            callback_types = callback_type if isinstance(callback_type, tuple) else (callback_type,)
            callback_entries.extend((t, fn) for t in callback_types)

        if callable(callbacks):
            callback_entries.append((Any, callbacks))
        elif isinstance(callbacks, tuple) and len(callbacks) == 2 and callable(callbacks[1]):
            add_callback_entry(callbacks)
        elif isinstance(callbacks, Sequence) and not isinstance(callbacks, (str, bytes)):
            for callback in callbacks:
                if (
                    not isinstance(callback, tuple)
                    or len(callback) != 2
                    or not callable(callback[1])
                ):
                    raise TypeError(
                        "structural_walk callbacks within a sequence must be "
                        "(type, callback) tuples"
                    )
                add_callback_entry(callback)
        else:
            raise TypeError(
                "structural_walk callbacks must be callbacks, (type, callback) entries, "
                "((type1, type2, ...), callback) entries, or sequences of tuple entries"
            )
        return callback_entries

    def wrap_callback_with_def_region_kind(fn: Callable[..., Any]) -> Callable[[Any, int], Any]:
        return lambda value, kind: fn(value, DefRegionKind(kind))

    callback_entries = normalize_callbacks(callbacks)
    callback_entries_with_def_region_kind = normalize_callbacks(with_def_region_kind)

    entries: list[tuple[int, Callable[[Any], Any]]] = [
        (_callback_type_to_type_index(t), fn) for t, fn in callback_entries
    ]
    entries_with_def_region_kind: list[tuple[int, Callable[[Any, int], Any]]] = [
        (_callback_type_to_type_index(t), wrap_callback_with_def_region_kind(fn))
        for t, fn in callback_entries_with_def_region_kind
    ]
    return _ffi_api.StructuralWalk(root, entries, entries_with_def_region_kind, order_int)


def _callback_type_to_type_index(callback_type: type[Any] | Any) -> int:
    """Convert a callback arg type to a type index."""
    annotation = Any if callback_type is object else callback_type
    try:
        type_index = core.TypeSchema.from_annotation(annotation).origin_type_index
    except TypeError as err:
        raise TypeError(
            "structural_walk callback type must be a supported builtin, "
            "typing.Any/object, or an FFI-registered object class"
        ) from err
    if type_index < 0 and annotation is not Any:
        raise TypeError(
            "structural_walk callback type_index is negative, the only"
            "acceptable negative type_index is -1 for Any"
        )
    return type_index
