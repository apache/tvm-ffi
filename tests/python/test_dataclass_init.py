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
"""Comprehensive tests for reflection-driven auto-generated ``__ffi_init__``.

This file exercises:
1. metadata emitted by C++ for auto-init traits
2. Python ``__init__`` signature generation
3. constructor behavior across positional/kw-only/default/init=False combinations
4. low-level KWARGS protocol via ``Object.__ffi_init__``
5. inheritance behavior for auto-generated init
6. copy/deepcopy/replace interplay with auto-init objects
7. re-initialization, isinstance checks, and instance isolation
"""

# ruff: noqa: D102
from __future__ import annotations

import copy
import inspect
import sys
from typing import Any

import pytest
from tvm_ffi import core
from tvm_ffi.testing import (
    _TestCxxAutoInit,
    _TestCxxAutoInitAllInitOff,
    _TestCxxAutoInitChild,
    _TestCxxAutoInitKwOnlyDefaults,
    _TestCxxAutoInitParent,
    _TestCxxAutoInitSimple,
    _TestCxxNoAutoInit,
)


def _field_map(type_cls: type) -> dict[str, Any]:
    return {field.name: field for field in getattr(type_cls, "__tvm_ffi_type_info__").fields}


def _method_metadata(type_cls: type, method_name: str) -> dict[str, Any]:
    type_info = getattr(type_cls, "__tvm_ffi_type_info__")
    for method in type_info.methods:
        if method.name == method_name:
            return method.metadata
    raise AssertionError(f"Cannot find method metadata: {type_cls.__name__}.{method_name}")


class TestAutoInitMetadata:
    def test_auto_init_method_marked(self) -> None:
        metadata = _method_metadata(_TestCxxAutoInit, "__ffi_init__")
        assert metadata.get("auto_init") is True

    def test_field_bitmask_init_kw_only_and_defaults(self) -> None:
        fields = _field_map(_TestCxxAutoInit)
        assert fields["a"].c_init is True
        assert fields["b"].c_init is False
        assert fields["b"].c_has_default is True
        assert fields["c"].c_kw_only is True
        assert fields["c"].c_init is True
        assert fields["d"].c_has_default is True

    def test_all_init_off_field_bitmask(self) -> None:
        fields = _field_map(_TestCxxAutoInitAllInitOff)
        assert fields["x"].c_init is False
        assert fields["x"].c_has_default is True
        assert fields["y"].c_init is False
        assert fields["y"].c_has_default is True
        assert fields["z"].c_init is False
        assert fields["z"].c_has_default is False

    def test_kw_only_defaults_field_bitmask(self) -> None:
        fields = _field_map(_TestCxxAutoInitKwOnlyDefaults)
        assert fields["p_default"].c_has_default is True
        assert fields["k_required"].c_kw_only is True
        assert fields["k_default"].c_kw_only is True
        assert fields["k_default"].c_has_default is True
        assert fields["hidden"].c_init is False
        assert fields["hidden"].c_has_default is True


class TestAutoInitSignature:
    def test_auto_init_signature_layout(self) -> None:
        sig = inspect.signature(_TestCxxAutoInit.__init__)
        assert tuple(sig.parameters) == ("self", "a", "d", "c")
        assert sig.parameters["a"].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
        assert sig.parameters["d"].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
        assert sig.parameters["c"].kind == inspect.Parameter.KEYWORD_ONLY

    def test_auto_init_signature_required_vs_default(self) -> None:
        sig = inspect.signature(_TestCxxAutoInit.__init__)
        assert sig.parameters["a"].default is inspect.Parameter.empty
        assert sig.parameters["c"].default is inspect.Parameter.empty
        assert sig.parameters["d"].default is not inspect.Parameter.empty

    def test_simple_signature(self) -> None:
        sig = inspect.signature(_TestCxxAutoInitSimple.__init__)
        assert tuple(sig.parameters) == ("self", "x", "y")
        assert all(
            p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
            for p in (sig.parameters["x"], sig.parameters["y"])
        )

    def test_all_init_off_signature_is_no_arg(self) -> None:
        sig = inspect.signature(_TestCxxAutoInitAllInitOff.__init__)
        assert tuple(sig.parameters) == ("self",)

    def test_kw_only_defaults_signature_layout(self) -> None:
        sig = inspect.signature(_TestCxxAutoInitKwOnlyDefaults.__init__)
        assert tuple(sig.parameters) == (
            "self",
            "p_required",
            "p_default",
            "k_required",
            "k_default",
        )
        assert sig.parameters["p_required"].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
        assert sig.parameters["p_default"].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
        assert sig.parameters["k_required"].kind == inspect.Parameter.KEYWORD_ONLY
        assert sig.parameters["k_default"].kind == inspect.Parameter.KEYWORD_ONLY
        assert sig.parameters["p_required"].default is inspect.Parameter.empty
        assert sig.parameters["k_required"].default is inspect.Parameter.empty
        assert sig.parameters["p_default"].default is not inspect.Parameter.empty
        assert sig.parameters["k_default"].default is not inspect.Parameter.empty

    def test_inheritance_signature_includes_parent_fields(self) -> None:
        sig = inspect.signature(_TestCxxAutoInitChild.__init__)
        # Required positional params must precede default ones in Python,
        # so child_required comes before parent_default.
        assert tuple(sig.parameters) == (
            "self",
            "parent_required",
            "child_required",
            "parent_default",
            "child_kw_only",
        )
        assert sig.parameters["child_kw_only"].kind == inspect.Parameter.KEYWORD_ONLY
        assert sig.parameters["parent_default"].default is not inspect.Parameter.empty

    def test_init_false_field_not_in_signature(self) -> None:
        """B is init=False and should not appear in signature."""
        sig = inspect.signature(_TestCxxAutoInit.__init__)
        assert "b" not in sig.parameters

    def test_hidden_field_not_in_signature(self) -> None:
        """Hidden is init=False and should not appear in signature."""
        sig = inspect.signature(_TestCxxAutoInitKwOnlyDefaults.__init__)
        assert "hidden" not in sig.parameters

    def test_kw_only_params_after_positional(self) -> None:
        """Keyword-only params should come after positional in the signature."""
        sig = inspect.signature(_TestCxxAutoInit.__init__)
        params = list(sig.parameters.values())
        saw_kw_only = False
        for p in params[1:]:  # skip self
            if p.kind == inspect.Parameter.KEYWORD_ONLY:
                saw_kw_only = True
            elif saw_kw_only:
                assert p.kind == inspect.Parameter.KEYWORD_ONLY, (
                    f"Parameter {p.name} is {p.kind} after a KEYWORD_ONLY param"
                )

    def test_child_signature_required_before_optional(self) -> None:
        """The child signature should have all required positional before optional."""
        sig = inspect.signature(_TestCxxAutoInitChild.__init__)
        params = list(sig.parameters.values())[1:]  # skip self
        positional = [p for p in params if p.kind != inspect.Parameter.KEYWORD_ONLY]
        saw_default = False
        for p in positional:
            if p.default is not inspect.Parameter.empty:
                saw_default = True
            elif saw_default:
                pytest.fail(f"Required param '{p.name}' appears after a default param")


class TestAutoInitConstruction:
    def test_auto_init_minimal_required(self) -> None:
        obj = _TestCxxAutoInit(1, c=3)
        assert obj.a == 1
        assert obj.b == 42
        assert obj.c == 3
        assert obj.d == 99

    def test_auto_init_all_keyword_arguments(self) -> None:
        obj = _TestCxxAutoInit(a=10, c=30, d=20)
        assert obj.a == 10
        assert obj.b == 42
        assert obj.c == 30
        assert obj.d == 20

    def test_auto_init_keyword_order_irrelevant(self) -> None:
        obj = _TestCxxAutoInit(c=7, d=8, a=9)
        assert obj.a == 9
        assert obj.c == 7
        assert obj.d == 8
        assert obj.b == 42

    def test_auto_init_second_positional_maps_to_d(self) -> None:
        obj = _TestCxxAutoInit(1, 2, c=3)
        assert obj.a == 1
        assert obj.d == 2
        assert obj.c == 3
        assert obj.b == 42

    def test_mutate_fields_after_construction(self) -> None:
        obj = _TestCxxAutoInit(1, c=2)
        obj.b = 100
        obj.c = 999
        assert obj.b == 100
        assert obj.c == 999


class TestAutoInitErrors:
    def test_missing_required_positional(self) -> None:
        with pytest.raises(TypeError):
            _TestCxxAutoInit(c=3)  # ty: ignore[missing-argument]

    def test_missing_required_kw_only(self) -> None:
        with pytest.raises(TypeError):
            _TestCxxAutoInit(1)  # ty: ignore[missing-argument]

    def test_kw_only_rejects_positional(self) -> None:
        with pytest.raises(TypeError):
            _TestCxxAutoInit(1, 2, 3)  # ty: ignore[missing-argument, too-many-positional-arguments]

    def test_duplicate_argument_detection(self) -> None:
        with pytest.raises(TypeError):
            _TestCxxAutoInit(1, 2, c=3, d=4)  # ty: ignore[parameter-already-assigned]

    def test_unexpected_keyword(self) -> None:
        with pytest.raises(TypeError):
            _TestCxxAutoInit(1, c=2, nope=3)  # ty: ignore[unknown-argument]

    def test_init_false_field_rejected_in_python_init(self) -> None:
        with pytest.raises(TypeError):
            _TestCxxAutoInit(1, b=2, c=3)  # ty: ignore[unknown-argument]

    def test_type_mismatch_for_required_positional(self) -> None:
        with pytest.raises(TypeError):
            _TestCxxAutoInit("x", c=2)  # ty: ignore[invalid-argument-type]

    def test_type_mismatch_for_kw_only(self) -> None:
        with pytest.raises(TypeError):
            _TestCxxAutoInit(1, c="x")  # ty: ignore[invalid-argument-type]

    def test_type_mismatch_for_defaultable_positional(self) -> None:
        with pytest.raises(TypeError):
            _TestCxxAutoInit(1, d="x", c=3)  # ty: ignore[invalid-argument-type]

    def test_init_false_field_rejected_via_dict_unpacking(self) -> None:
        """B is init=False, rejected even when passed via **dict."""
        with pytest.raises(TypeError):
            _TestCxxAutoInit(**{"a": 1, "c": 2, "b": 3})

    def test_positional_and_keyword_same_field(self) -> None:
        """Providing a field both positionally and as keyword should error."""
        with pytest.raises(TypeError):
            _TestCxxAutoInit(1, a=2, c=3)  # ty: ignore[parameter-already-assigned]

    def test_none_for_required_integer_field(self) -> None:
        """Passing None where an int64_t is expected should raise TypeError."""
        with pytest.raises(TypeError):
            _TestCxxAutoInitSimple(None, 1)  # ty: ignore[invalid-argument-type]

    def test_none_for_keyword_integer_field(self) -> None:
        """Passing None as keyword where int64_t is expected."""
        with pytest.raises(TypeError):
            _TestCxxAutoInitSimple(x=None, y=1)  # ty: ignore[invalid-argument-type]


class TestAutoInitLowLevelFfiInit:
    def test_low_level_kwargs_protocol(self) -> None:
        obj = _TestCxxAutoInit.__new__(_TestCxxAutoInit)
        obj.__ffi_init__(core.KWARGS, "a", 1, "c", 3)
        assert obj.a == 1
        assert obj.b == 42
        assert obj.c == 3
        assert obj.d == 99

    def test_low_level_kwargs_even_pairs_required(self) -> None:
        obj = _TestCxxAutoInit.__new__(_TestCxxAutoInit)
        with pytest.raises(TypeError):
            obj.__ffi_init__(core.KWARGS, "a", 1, "c")

    def test_low_level_kwargs_duplicate_name(self) -> None:
        obj = _TestCxxAutoInit.__new__(_TestCxxAutoInit)
        with pytest.raises(TypeError):
            obj.__ffi_init__(core.KWARGS, "a", 1, "a", 2, "c", 3)

    def test_low_level_kwargs_unknown_name(self) -> None:
        obj = _TestCxxAutoInit.__new__(_TestCxxAutoInit)
        with pytest.raises(TypeError):
            obj.__ffi_init__(core.KWARGS, "a", 1, "unknown", 2, "c", 3)

    def test_low_level_kwargs_key_must_be_string(self) -> None:
        obj = _TestCxxAutoInit.__new__(_TestCxxAutoInit)
        with pytest.raises(TypeError):
            obj.__ffi_init__(core.KWARGS, 1, 2, "a", 3, "c", 4)

    def test_low_level_positional_too_many(self) -> None:
        obj = _TestCxxAutoInit.__new__(_TestCxxAutoInit)
        with pytest.raises(TypeError):
            obj.__ffi_init__(1, 2, 3, 4)

    def test_low_level_missing_required(self) -> None:
        obj = _TestCxxAutoInit.__new__(_TestCxxAutoInit)
        with pytest.raises(TypeError):
            obj.__ffi_init__(1)

    def test_low_level_kw_only_field_rejected_positionally(self) -> None:
        obj = _TestCxxAutoInit.__new__(_TestCxxAutoInit)
        with pytest.raises(TypeError):
            obj.__ffi_init__(1, 2)

    def test_low_level_init_false_field_rejected(self) -> None:
        obj = _TestCxxAutoInit.__new__(_TestCxxAutoInit)
        with pytest.raises(TypeError):
            obj.__ffi_init__(core.KWARGS, "a", 1, "b", 2, "c", 3)

    def test_low_level_kwargs_all_init_fields_explicit(self) -> None:
        """Providing all init=True fields via KWARGS."""
        obj = _TestCxxAutoInit.__new__(_TestCxxAutoInit)
        obj.__ffi_init__(core.KWARGS, "a", 10, "c", 30, "d", 40)
        assert obj.a == 10
        assert obj.b == 42  # init=False, default
        assert obj.c == 30
        assert obj.d == 40

    def test_low_level_kwargs_empty_string_key(self) -> None:
        """Empty string as keyword name should be rejected as unknown."""
        obj = _TestCxxAutoInit.__new__(_TestCxxAutoInit)
        with pytest.raises(TypeError, match="unexpected keyword"):
            obj.__ffi_init__(core.KWARGS, "", 1, "a", 2, "c", 3)

    def test_low_level_kwargs_odd_kv_count(self) -> None:
        """Odd number of key-value args after KWARGS sentinel."""
        obj = _TestCxxAutoInitSimple.__new__(_TestCxxAutoInitSimple)
        with pytest.raises(TypeError):
            obj.__ffi_init__(core.KWARGS, "x")

    def test_low_level_positional_only_simple(self) -> None:
        """Positional-only mode (no KWARGS sentinel)."""
        obj = _TestCxxAutoInitSimple.__new__(_TestCxxAutoInitSimple)
        obj.__ffi_init__(10, 20)
        assert obj.x == 10
        assert obj.y == 20

    def test_low_level_zero_args_missing_required(self) -> None:
        """Zero args for a type that requires them."""
        obj = _TestCxxAutoInitSimple.__new__(_TestCxxAutoInitSimple)
        with pytest.raises(TypeError, match="missing required"):
            obj.__ffi_init__()

    def test_low_level_kwargs_sentinel_only_no_kv_pairs(self) -> None:
        """KWARGS sentinel with zero key-value pairs; required fields still missing."""
        obj = _TestCxxAutoInit.__new__(_TestCxxAutoInit)
        with pytest.raises(TypeError, match="missing required"):
            obj.__ffi_init__(core.KWARGS)

    def test_low_level_kwargs_positional_then_sentinel_no_kv(self) -> None:
        """Positional args followed by KWARGS sentinel but no key-value pairs."""
        obj = _TestCxxAutoInit.__new__(_TestCxxAutoInit)
        # a=1 positionally, sentinel, no KV pairs → c is still missing
        with pytest.raises(TypeError, match="missing required"):
            obj.__ffi_init__(1, core.KWARGS)

    def test_low_level_child_positional_routing(self) -> None:
        """Verify the inheritance positional fix at the raw __ffi_init__ level.

        After stable_partition, pos_indices should be:
        [parent_required, child_required, parent_default]
        So 2 positional args map to parent_required=1, child_required=2.
        """
        obj = _TestCxxAutoInitChild.__new__(_TestCxxAutoInitChild)
        obj.__ffi_init__(1, 2, core.KWARGS, "child_kw_only", 3)
        assert obj.parent_required == 1
        assert obj.child_required == 2
        assert obj.parent_default == 5  # default
        assert obj.child_kw_only == 3

    def test_low_level_child_all_three_positional(self) -> None:
        """Three positional args for child at the raw protocol level."""
        obj = _TestCxxAutoInitChild.__new__(_TestCxxAutoInitChild)
        obj.__ffi_init__(1, 2, 3, core.KWARGS, "child_kw_only", 4)
        assert obj.parent_required == 1
        assert obj.child_required == 2
        assert obj.parent_default == 3
        assert obj.child_kw_only == 4

    def test_low_level_child_too_many_positional(self) -> None:
        """Four positional args exceed the 3 positional slots for child."""
        obj = _TestCxxAutoInitChild.__new__(_TestCxxAutoInitChild)
        with pytest.raises(TypeError, match="positional"):
            obj.__ffi_init__(1, 2, 3, 4, core.KWARGS, "child_kw_only", 5)


class TestAutoInitSimple:
    def test_simple_positional(self) -> None:
        obj = _TestCxxAutoInitSimple(10, 20)
        assert obj.x == 10
        assert obj.y == 20

    def test_simple_keyword(self) -> None:
        obj = _TestCxxAutoInitSimple(x=10, y=20)
        assert obj.x == 10
        assert obj.y == 20

    def test_simple_mixed(self) -> None:
        obj = _TestCxxAutoInitSimple(10, y=20)
        assert obj.x == 10
        assert obj.y == 20

    def test_simple_missing_required(self) -> None:
        with pytest.raises(TypeError):
            _TestCxxAutoInitSimple(x=10)  # ty: ignore[missing-argument]

    def test_simple_too_many_positional(self) -> None:
        with pytest.raises(TypeError):
            _TestCxxAutoInitSimple(1, 2, 3)  # ty: ignore[too-many-positional-arguments]

    def test_simple_unexpected_keyword(self) -> None:
        with pytest.raises(TypeError):
            _TestCxxAutoInitSimple(x=1, y=2, z=3)  # ty: ignore[unknown-argument]

    def test_simple_low_level_kwargs(self) -> None:
        obj = _TestCxxAutoInitSimple.__new__(_TestCxxAutoInitSimple)
        obj.__ffi_init__(core.KWARGS, "x", 10, "y", 20)
        assert obj.x == 10
        assert obj.y == 20

    def test_zero_values(self) -> None:
        obj = _TestCxxAutoInitSimple(0, 0)
        assert obj.x == 0
        assert obj.y == 0

    def test_negative_values(self) -> None:
        obj = _TestCxxAutoInitSimple(-1, -9999999)
        assert obj.x == -1
        assert obj.y == -9999999

    def test_large_values(self) -> None:
        large = 2**62
        obj = _TestCxxAutoInitSimple(large, -large)
        assert obj.x == large
        assert obj.y == -large

    def test_float_for_integer_field(self) -> None:
        """Passing float where int64_t is expected - should truncate or error."""
        try:
            obj = _TestCxxAutoInitSimple(1.5, 2)  # ty: ignore[invalid-argument-type]
            assert isinstance(obj.x, int)
        except TypeError:
            pass  # Also acceptable

    def test_bool_for_integer_field(self) -> None:
        """Booleans are valid ints in Python but should work correctly."""
        obj = _TestCxxAutoInitSimple(True, False)
        assert obj.x == 1
        assert obj.y == 0


class TestAutoInitAllInitOff:
    def test_no_arg_constructor(self) -> None:
        obj = _TestCxxAutoInitAllInitOff()
        assert obj.x == 7
        assert obj.y == 9
        assert obj.z == 1234

    def test_rejects_positional_args(self) -> None:
        with pytest.raises(TypeError):
            _TestCxxAutoInitAllInitOff(1)  # ty: ignore[too-many-positional-arguments]

    def test_rejects_keyword_args(self) -> None:
        with pytest.raises(TypeError):
            _TestCxxAutoInitAllInitOff(x=1)  # ty: ignore[unknown-argument]

    def test_low_level_empty_init(self) -> None:
        obj = _TestCxxAutoInitAllInitOff.__new__(_TestCxxAutoInitAllInitOff)
        obj.__ffi_init__()
        assert obj.x == 7
        assert obj.y == 9
        assert obj.z == 1234

    def test_mutate_fields(self) -> None:
        obj = _TestCxxAutoInitAllInitOff()
        obj.x = 101
        obj.y = 202
        obj.z = 303
        assert (obj.x, obj.y, obj.z) == (101, 202, 303)

    def test_empty_kwargs_dict_star(self) -> None:
        """Passing **{} should be fine (empty kwargs)."""
        obj = _TestCxxAutoInitAllInitOff(**{})
        assert obj.x == 7

    def test_low_level_rejects_positional(self) -> None:
        obj = _TestCxxAutoInitAllInitOff.__new__(_TestCxxAutoInitAllInitOff)
        with pytest.raises(TypeError):
            obj.__ffi_init__(1)


class TestAutoInitKwOnlyDefaults:
    def test_minimal_required(self) -> None:
        obj = _TestCxxAutoInitKwOnlyDefaults(1, k_required=2)
        assert obj.p_required == 1
        assert obj.p_default == 11
        assert obj.k_required == 2
        assert obj.k_default == 22
        assert obj.hidden == 33

    def test_override_defaults(self) -> None:
        obj = _TestCxxAutoInitKwOnlyDefaults(p_required=1, p_default=4, k_required=5, k_default=6)
        assert obj.p_required == 1
        assert obj.p_default == 4
        assert obj.k_required == 5
        assert obj.k_default == 6
        assert obj.hidden == 33

    def test_missing_required_positional(self) -> None:
        with pytest.raises(TypeError):
            _TestCxxAutoInitKwOnlyDefaults(k_required=2)  # ty: ignore[missing-argument]

    def test_missing_required_kw_only(self) -> None:
        with pytest.raises(TypeError):
            _TestCxxAutoInitKwOnlyDefaults(1)  # ty: ignore[missing-argument]

    def test_kw_only_rejects_positional(self) -> None:
        with pytest.raises(TypeError):
            _TestCxxAutoInitKwOnlyDefaults(1, 2, 3)  # ty: ignore[missing-argument, too-many-positional-arguments]

    def test_hidden_init_false_field_not_accepted(self) -> None:
        with pytest.raises(TypeError):
            _TestCxxAutoInitKwOnlyDefaults(1, k_required=2, hidden=4)  # ty: ignore[unknown-argument]

    def test_type_mismatch(self) -> None:
        with pytest.raises(TypeError):
            _TestCxxAutoInitKwOnlyDefaults("x", k_required=2)  # ty: ignore[invalid-argument-type]

    def test_low_level_kwargs_call(self) -> None:
        obj = _TestCxxAutoInitKwOnlyDefaults.__new__(_TestCxxAutoInitKwOnlyDefaults)
        obj.__ffi_init__(core.KWARGS, "p_required", 1, "k_required", 2)
        assert obj.p_required == 1
        assert obj.p_default == 11
        assert obj.k_required == 2
        assert obj.k_default == 22
        assert obj.hidden == 33

    def test_kw_only_via_dict_unpacking(self) -> None:
        """Verify kw_only fields work via **dict."""
        kwargs = {"k_required": 100, "k_default": 200}
        obj = _TestCxxAutoInitKwOnlyDefaults(1, **kwargs)
        assert obj.p_required == 1
        assert obj.p_default == 11  # default
        assert obj.k_required == 100
        assert obj.k_default == 200
        assert obj.hidden == 33  # init=False default


class TestAutoInitInheritance:
    def test_parent_constructor(self) -> None:
        obj = _TestCxxAutoInitParent(10)
        assert obj.parent_required == 10
        assert obj.parent_default == 5

    def test_parent_all_keyword(self) -> None:
        obj = _TestCxxAutoInitParent(parent_required=10, parent_default=20)
        assert obj.parent_required == 10
        assert obj.parent_default == 20

    def test_parent_positional_then_keyword(self) -> None:
        obj = _TestCxxAutoInitParent(10, parent_default=20)
        assert obj.parent_required == 10
        assert obj.parent_default == 20

    def test_child_constructor_uses_parent_and_child_fields(self) -> None:
        obj = _TestCxxAutoInitChild(parent_required=1, child_required=2, child_kw_only=3)
        assert obj.parent_required == 1
        assert obj.parent_default == 5
        assert obj.child_required == 2
        assert obj.child_kw_only == 3

    def test_child_all_keyword_with_parent_default_override(self) -> None:
        # fmt: off
        obj = _TestCxxAutoInitChild(parent_required=1, child_required=2, parent_default=99, child_kw_only=3)
        # fmt: on
        assert obj.parent_required == 1
        assert obj.child_required == 2
        assert obj.parent_default == 99
        assert obj.child_kw_only == 3

    def test_child_missing_parent_required(self) -> None:
        with pytest.raises(TypeError):
            _TestCxxAutoInitChild(child_required=2, child_kw_only=3)  # ty: ignore[missing-argument]

    def test_child_two_positional_args_routes_correctly(self) -> None:
        """Calling Child(1, 2, child_kw_only=3) should set parent_required=1, child_required=2.

        The Python signature is:
          (self, parent_required, child_required, parent_default=..., *, child_kw_only)
        So positional arg 0 = parent_required, positional arg 1 = child_required.
        """
        obj = _TestCxxAutoInitChild(1, 2, child_kw_only=3)
        assert obj.parent_required == 1
        assert obj.child_required == 2
        assert obj.parent_default == 5  # should use the default
        assert obj.child_kw_only == 3

    def test_child_three_positional_args_no_silent_swap(self) -> None:
        """Calling Child(1, 2, 3, child_kw_only=4) should map correctly.

        Python signature order: parent_required=1, child_required=2, parent_default=3
        """
        obj = _TestCxxAutoInitChild(1, 2, 3, child_kw_only=4)
        assert obj.parent_required == 1
        assert obj.child_required == 2
        assert obj.parent_default == 3
        assert obj.child_kw_only == 4

    def test_child_one_positional_rest_keyword(self) -> None:
        """Mix of positional and keyword to verify correct mapping."""
        obj = _TestCxxAutoInitChild(10, child_required=20, parent_default=30, child_kw_only=40)
        assert obj.parent_required == 10
        assert obj.child_required == 20
        assert obj.parent_default == 30
        assert obj.child_kw_only == 40


class TestAutoInitCopyBehavior:
    """Test copy/deepcopy/replace interplay with auto-init objects."""

    def test_shallow_copy(self) -> None:
        obj = _TestCxxAutoInitSimple(10, 20)
        obj_copy = copy.copy(obj)
        assert obj_copy.x == 10
        assert obj_copy.y == 20
        assert not obj.same_as(obj_copy)

    def test_deepcopy(self) -> None:
        obj = _TestCxxAutoInitSimple(10, 20)
        obj_copy = copy.deepcopy(obj)
        assert obj_copy.x == 10
        assert obj_copy.y == 20
        assert not obj.same_as(obj_copy)

    @pytest.mark.skipif(sys.version_info < (3, 13), reason="copy.replace requires Python 3.13+")
    def test_replace(self) -> None:
        obj = _TestCxxAutoInit(1, c=3)
        replaced = copy.replace(obj, a=100, c=300)  # type: ignore[attr-defined]
        assert replaced.a == 100
        assert replaced.b == 42
        assert replaced.c == 300
        assert replaced.d == 99

    def test_copy_preserves_init_false_field(self) -> None:
        """After construction, mutating the init=False field and copying."""
        obj = _TestCxxAutoInit(1, c=3)
        assert obj.b == 42
        obj.b = 999
        assert obj.b == 999
        obj_copy = copy.copy(obj)
        assert obj_copy.b == 999

    def test_copy_preserves_default_override(self) -> None:
        """Override a default field, then copy should preserve the override."""
        obj = _TestCxxAutoInit(1, c=3, d=55)
        obj_copy = copy.copy(obj)
        assert obj_copy.d == 55

    def test_deepcopy_all_init_off(self) -> None:
        """Deepcopy of an object with all fields init=False."""
        obj = _TestCxxAutoInitAllInitOff()
        obj.x = 111
        obj.y = 222
        obj.z = 333
        obj_copy = copy.deepcopy(obj)
        assert obj_copy.x == 111
        assert obj_copy.y == 222
        assert obj_copy.z == 333
        assert not obj.same_as(obj_copy)

    @pytest.mark.skipif(sys.version_info < (3, 13), reason="copy.replace requires Python 3.13+")
    def test_replace_kw_only_defaults(self) -> None:
        obj = _TestCxxAutoInitKwOnlyDefaults(1, k_required=2)
        replaced = copy.replace(obj, k_required=99, p_default=88)  # type: ignore[attr-defined]
        assert replaced.p_required == 1
        assert replaced.p_default == 88
        assert replaced.k_required == 99
        assert replaced.k_default == 22
        assert replaced.hidden == 33


class TestAutoInitReinitialization:
    """Test what happens when __ffi_init__ is called multiple times."""

    def test_reinit_changes_handle(self) -> None:
        """Calling __ffi_init__ again should create a new underlying object."""
        obj = _TestCxxAutoInit(1, c=3)
        original_handle = obj.__chandle__()
        assert obj.a == 1

        obj.__ffi_init__(core.KWARGS, "a", 100, "c", 300)
        assert obj.a == 100
        assert obj.c == 300
        assert obj.__chandle__() != original_handle

    def test_reinit_resets_init_false_field(self) -> None:
        """Re-initialization should reset init=False fields to defaults."""
        obj = _TestCxxAutoInit(1, c=3)
        obj.b = 999
        assert obj.b == 999

        obj.__ffi_init__(core.KWARGS, "a", 2, "c", 4)
        assert obj.b == 42  # reset to default


class TestAutoInitTypeChecks:
    """Verify isinstance relationships for auto-init objects."""

    def test_parent_isinstance(self) -> None:
        obj = _TestCxxAutoInitParent(1)
        assert isinstance(obj, _TestCxxAutoInitParent)
        assert isinstance(obj, core.Object)

    def test_child_isinstance_parent(self) -> None:
        obj = _TestCxxAutoInitChild(parent_required=1, child_required=2, child_kw_only=3)
        assert isinstance(obj, _TestCxxAutoInitChild)
        assert isinstance(obj, _TestCxxAutoInitParent)
        assert isinstance(obj, core.Object)

    def test_parent_isinstance_child_due_to_metaclass(self) -> None:
        """Due to _ObjectSlotsMeta, any CObject passes isinstance for any FFI class.

        This is a pre-existing design choice in the TVM FFI type system, not a bug
        introduced by the auto-init feature.
        """
        obj = _TestCxxAutoInitParent(1)
        # _ObjectSlotsMeta.__instancecheck__ returns True for any CObject
        assert isinstance(obj, _TestCxxAutoInitChild)


class TestAutoInitInstanceIsolation:
    """Verify that multiple instances don't share mutable state."""

    def test_separate_instances_are_independent(self) -> None:
        a = _TestCxxAutoInitSimple(1, 2)
        b = _TestCxxAutoInitSimple(3, 4)
        assert a.x == 1
        assert b.x == 3
        assert not a.same_as(b)

    def test_mutating_one_instance_doesnt_affect_another(self) -> None:
        a = _TestCxxAutoInit(1, c=3)
        b = _TestCxxAutoInit(1, c=3)
        assert a.b == 42
        assert b.b == 42
        a.b = 999
        assert a.b == 999
        assert b.b == 42

    def test_all_init_off_instances_are_independent(self) -> None:
        a = _TestCxxAutoInitAllInitOff()
        b = _TestCxxAutoInitAllInitOff()
        a.x = 100
        assert b.x == 7


class TestAutoInitReinitInitOffNoDefault:
    """Test reinit behavior for init=False fields with and without reflection defaults."""

    def test_reinit_init_false_with_default_resets(self) -> None:
        """Fields b (init=False, default=42) should reset to default on reinit."""
        obj = _TestCxxAutoInit(1, c=3)
        obj.b = 999
        obj.__ffi_init__(core.KWARGS, "a", 2, "c", 4)
        assert obj.b == 42

    def test_reinit_init_false_without_reflection_default(self) -> None:
        """Field z has init=False AND no reflection default (c_has_default=False).

        On reinit, z should get whatever the C++ creator sets (1234).
        """
        obj = _TestCxxAutoInitAllInitOff()
        assert obj.z == 1234
        obj.z = 9999
        assert obj.z == 9999
        # Reinit via low-level call
        obj.__ffi_init__()
        # z has no reflection default, so creator's C++ default (1234) is used
        assert obj.z == 1234


class TestAutoInitErrorMessages:
    """Verify that error messages name the correct field after pos_indices reordering."""

    def test_missing_child_required_names_correct_field(self) -> None:
        """When child_required is missing, error should mention 'child_required'."""
        with pytest.raises(TypeError, match="child_required"):
            _TestCxxAutoInitChild(parent_required=1, child_kw_only=3)  # ty: ignore[missing-argument]

    def test_missing_parent_required_names_correct_field(self) -> None:
        """When parent_required is missing, error should mention 'parent_required'."""
        with pytest.raises(TypeError, match="parent_required"):
            _TestCxxAutoInitChild(child_required=2, child_kw_only=3)  # ty: ignore[missing-argument]

    def test_missing_kw_only_names_correct_field(self) -> None:
        """When child_kw_only is missing, error should mention 'child_kw_only'."""
        with pytest.raises(TypeError, match="child_kw_only"):
            _TestCxxAutoInitChild(parent_required=1, child_required=2)  # ty: ignore[missing-argument]

    def test_too_many_positional_error_message(self) -> None:
        """Error for too many positional args should mention the correct count."""
        with pytest.raises(TypeError, match="3 positional"):
            _TestCxxAutoInitChild(1, 2, 3, 4, child_kw_only=5)  # ty: ignore[too-many-positional-arguments]

    def test_unexpected_keyword_error_message(self) -> None:
        """Error for unknown keyword should mention the keyword name."""
        with pytest.raises(TypeError, match="bogus"):
            _TestCxxAutoInit(1, c=2, bogus=3)  # ty: ignore[unknown-argument]

    def test_duplicate_arg_error_message(self) -> None:
        """Error for duplicate argument should mention the field name."""
        with pytest.raises(TypeError, match=r"multiple values.*a"):
            _TestCxxAutoInit(1, c=2, a=3)  # ty: ignore[parameter-already-assigned]


class TestAutoInitLowLevelKwOnlyDefaults:
    """Low-level KWARGS protocol tests for the KwOnlyDefaults type."""

    def test_low_level_positional_plus_kwargs_kw_only(self) -> None:
        """Positional arg for p_required, then KWARGS for kw_only fields."""
        obj = _TestCxxAutoInitKwOnlyDefaults.__new__(_TestCxxAutoInitKwOnlyDefaults)
        obj.__ffi_init__(1, core.KWARGS, "k_required", 2)
        assert obj.p_required == 1
        assert obj.p_default == 11
        assert obj.k_required == 2
        assert obj.k_default == 22
        assert obj.hidden == 33

    def test_low_level_two_positional_plus_kwargs(self) -> None:
        """Two positional args (p_required, p_default) then KWARGS for kw_only."""
        obj = _TestCxxAutoInitKwOnlyDefaults.__new__(_TestCxxAutoInitKwOnlyDefaults)
        obj.__ffi_init__(1, 2, core.KWARGS, "k_required", 3, "k_default", 4)
        assert obj.p_required == 1
        assert obj.p_default == 2
        assert obj.k_required == 3
        assert obj.k_default == 4
        assert obj.hidden == 33

    def test_low_level_all_via_kwargs(self) -> None:
        """All init=True fields via KWARGS, no positional."""
        obj = _TestCxxAutoInitKwOnlyDefaults.__new__(_TestCxxAutoInitKwOnlyDefaults)
        obj.__ffi_init__(
            core.KWARGS, "p_required", 10, "p_default", 20, "k_required", 30, "k_default", 40
        )
        assert obj.p_required == 10
        assert obj.p_default == 20
        assert obj.k_required == 30
        assert obj.k_default == 40
        assert obj.hidden == 33


class TestClassLevelInitFalse:
    """init(false) passed to ObjectDef constructor suppresses __ffi_init__."""

    def test_no_ffi_init_method(self) -> None:
        type_info = getattr(_TestCxxNoAutoInit, "__tvm_ffi_type_info__")
        method_names = [m.name for m in type_info.methods]
        assert "__ffi_init__" not in method_names

    def test_has_fields(self) -> None:
        type_info = getattr(_TestCxxNoAutoInit, "__tvm_ffi_type_info__")
        field_names = [f.name for f in type_info.fields]
        assert field_names == ["x", "y"]

    def test_direct_construction_raises(self) -> None:
        with pytest.raises(TypeError):
            _TestCxxNoAutoInit(1, 2)  # ty: ignore[too-many-positional-arguments]

    def test_has_shallow_copy(self) -> None:
        type_info = getattr(_TestCxxNoAutoInit, "__tvm_ffi_type_info__")
        method_names = [m.name for m in type_info.methods]
        assert "__ffi_shallow_copy__" in method_names
