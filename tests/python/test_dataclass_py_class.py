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
"""Tests for TypeInfo.total_size with multi-level inheritance."""

from __future__ import annotations

from typing import Any

from tvm_ffi.core import TypeInfo
from tvm_ffi.testing import (
    _TestCxxClassBase,
    _TestCxxClassDerived,
    _TestCxxClassDerivedDerived,
)


def _get_type_info(cls: Any) -> TypeInfo:
    """Retrieve the TypeInfo for a registered class."""
    return cls.__tvm_ffi_type_info__


class TestBug1FieldOffsetOverlap:
    """Regression test for Bug #1: field offsets must not overlap in multi-level
    inheritance when an intermediate class has no own fields.
    """

    def test_three_level_total_size_monotonic(self) -> None:
        """Grandparent < Parent < Child total_size must be non-decreasing.

        Uses the existing C++ class hierarchy:
        TestCxxClassBase -> TestCxxClassDerived -> TestCxxClassDerivedDerived
        """
        base_info = _get_type_info(_TestCxxClassBase)
        derived_info = _get_type_info(_TestCxxClassDerived)
        dd_info = _get_type_info(_TestCxxClassDerivedDerived)

        base_size = base_info.total_size
        derived_size = derived_info.total_size
        dd_size = dd_info.total_size

        # Each level adds fields so total_size must grow
        assert base_size > 0, f"Base total_size should be positive, got {base_size}"
        assert derived_size >= base_size, (
            f"Derived total_size ({derived_size}) < Base total_size ({base_size})"
        )
        assert dd_size >= derived_size, (
            f"DerivedDerived total_size ({dd_size}) < Derived total_size ({derived_size})"
        )

    def test_total_size_accounts_for_parent_fields(self) -> None:
        """total_size of a child must be at least as large as parent total_size,
        ensuring child field offsets don't overlap with parent fields.
        """
        derived_info = _get_type_info(_TestCxxClassDerived)

        # Parent link should be set
        assert derived_info.parent_type_info is not None
        parent_size = derived_info.parent_type_info.total_size
        child_size = derived_info.total_size

        assert child_size >= parent_size, (
            f"Child total_size ({child_size}) must be >= parent total_size ({parent_size})"
        )

    def test_child_field_offsets_dont_overlap_parent(self) -> None:
        """Child fields must not overlap with any parent field ranges.

        Verifies that the offset of each child-only field lies outside
        every parent field's [offset, offset+size) range.
        """
        derived_info = _get_type_info(_TestCxxClassDerived)
        base_info = _get_type_info(_TestCxxClassBase)

        # Build parent field ranges
        parent_ranges = {f.name: (f.offset, f.offset + f.size) for f in base_info.fields}
        # Child-only fields (fields not in base)
        base_field_names = {f.name for f in base_info.fields}
        child_only_fields = [f for f in derived_info.fields if f.name not in base_field_names]

        for child_field in child_only_fields:
            for parent_name, (p_start, p_end) in parent_ranges.items():
                assert not (p_start <= child_field.offset < p_end), (
                    f"Child field '{child_field.name}' at offset {child_field.offset} "
                    f"overlaps parent field '{parent_name}' at [{p_start}, {p_end})"
                )

    def test_four_level_total_size_chain(self) -> None:
        """Verify total_size through the parent_type_info chain.

        Walk up from DerivedDerived to Base checking each link.
        """
        dd_info = _get_type_info(_TestCxxClassDerivedDerived)

        sizes = []
        current = dd_info
        while current is not None:
            sizes.append(current.total_size)
            current = current.parent_type_info

        # Should have at least 3 levels (DD -> Derived -> Base)
        assert len(sizes) >= 3, f"Expected >= 3 levels, got {len(sizes)}"
        # Sizes should be non-increasing as we go up (child >= parent)
        for i in range(len(sizes) - 1):
            assert sizes[i] >= sizes[i + 1], (
                f"Level {i} total_size ({sizes[i]}) < level {i + 1} ({sizes[i + 1]})"
            )
