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

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .access_path import AccessPath

from . import _ffi_api
from .core import Object
from .registry import register_object

__all__ = [
    "StructuralKey",
    "get_first_structural_mismatch",
    "structural_equal",
    "structural_hash",
]


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
