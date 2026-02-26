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
"""Tests for the c_class decorator (register_object + structural dunders)."""

from __future__ import annotations

import inspect

import pytest
from tvm_ffi.core import Object
from tvm_ffi.dataclasses import c_class
from tvm_ffi.testing import (
    _TestCxxClassBase,
    _TestCxxClassDerived,
    _TestCxxClassDerivedDerived,
    _TestCxxInitSubset,
    _TestCxxKwOnly,
)

# ---------------------------------------------------------------------------
# 1. Custom __init__ preservation
# ---------------------------------------------------------------------------


def test_c_class_custom_init() -> None:
    """c_class preserves user-defined __init__."""
    obj = _TestCxxClassBase(v_i64=10, v_i32=20)
    assert obj.v_i64 == 11  # +1 from custom __init__
    assert obj.v_i32 == 22  # +2 from custom __init__


# ---------------------------------------------------------------------------
# 2. Auto-generated __init__ with defaults
# ---------------------------------------------------------------------------


def test_c_class_auto_init_defaults() -> None:
    """Derived classes use auto-generated __init__ with C++ defaults."""
    obj = _TestCxxClassDerived(v_i64=1, v_i32=2, v_f64=3.0)
    assert obj.v_i64 == 1
    assert obj.v_i32 == 2
    assert obj.v_f64 == 3.0
    assert obj.v_f32 == 8.0  # default from C++


def test_c_class_auto_init_all_explicit() -> None:
    """Auto-generated __init__ accepts all fields explicitly."""
    obj = _TestCxxClassDerived(v_i64=123, v_i32=456, v_f64=4.0, v_f32=9.0)
    assert obj.v_i64 == 123
    assert obj.v_i32 == 456
    assert obj.v_f64 == 4.0
    assert obj.v_f32 == 9.0


# ---------------------------------------------------------------------------
# 3. Structural equality (__eq__)
# ---------------------------------------------------------------------------


def test_c_class_eq() -> None:
    """c_class installs __eq__ using RecursiveEq."""
    a = _TestCxxClassDerived(1, 2, 3.0, 4.0)
    b = _TestCxxClassDerived(1, 2, 3.0, 4.0)
    assert a == b
    assert a is not b  # different objects
    c = _TestCxxClassDerived(1, 2, 3.0, 5.0)
    assert a != c


def test_c_class_eq_reflexive() -> None:
    """Equality is reflexive: an object equals itself."""
    a = _TestCxxClassDerived(1, 2, 3.0, 4.0)
    b = a  # alias, same object
    assert a == b


def test_c_class_eq_symmetric() -> None:
    """Equality is symmetric: a == b implies b == a."""
    a = _TestCxxClassDerived(1, 2, 3.0, 4.0)
    b = _TestCxxClassDerived(1, 2, 3.0, 4.0)
    assert a == b
    assert b == a


# ---------------------------------------------------------------------------
# 4. Structural hash (__hash__)
# ---------------------------------------------------------------------------


def test_c_class_hash() -> None:
    """c_class installs __hash__ using RecursiveHash."""
    a = _TestCxxClassDerived(1, 2, 3.0, 4.0)
    b = _TestCxxClassDerived(1, 2, 3.0, 4.0)
    assert hash(a) == hash(b)


def test_c_class_hash_as_dict_key() -> None:
    """Equal objects can be used interchangeably as dict keys."""
    a = _TestCxxClassDerived(1, 2, 3.0, 4.0)
    b = _TestCxxClassDerived(1, 2, 3.0, 4.0)
    d = {a: "value"}
    assert d[b] == "value"


# ---------------------------------------------------------------------------
# 5. Ordering (__lt__, __le__, __gt__, __ge__)
# ---------------------------------------------------------------------------


def test_c_class_ordering() -> None:
    """c_class installs ordering operators."""
    small = _TestCxxClassDerived(0, 0, 0.0, 0.0)
    big = _TestCxxClassDerived(100, 100, 100.0, 100.0)
    assert small < big  # ty: ignore[unsupported-operator]
    assert small <= big  # ty: ignore[unsupported-operator]
    assert big > small  # ty: ignore[unsupported-operator]
    assert big >= small  # ty: ignore[unsupported-operator]
    assert not (big < small)  # ty: ignore[unsupported-operator]
    assert not (small > big)  # ty: ignore[unsupported-operator]


def test_c_class_ordering_reflexive() -> None:
    """<= and >= are reflexive."""
    a = _TestCxxClassDerived(1, 2, 3.0, 4.0)
    b = a  # alias, same object
    assert a <= b  # ty: ignore[unsupported-operator]
    assert a >= b  # ty: ignore[unsupported-operator]


def test_c_class_ordering_antisymmetric() -> None:
    """If a < b then not b < a."""
    a = _TestCxxClassDerived(0, 0, 0.0, 0.0)
    b = _TestCxxClassDerived(100, 100, 100.0, 100.0)
    if a < b:  # ty: ignore[unsupported-operator]
        assert not (b < a)  # ty: ignore[unsupported-operator]
    else:
        assert not (a < b)  # ty: ignore[unsupported-operator]


# ---------------------------------------------------------------------------
# 6. Equality with different types returns NotImplemented
# ---------------------------------------------------------------------------


def test_c_class_eq_different_type() -> None:
    """__eq__ returns NotImplemented for unrelated types."""
    a = _TestCxxClassDerived(1, 2, 3.0, 4.0)
    assert a != "hello"
    assert a != 42
    assert a != 3.14
    assert a != None


def test_c_class_ordering_different_type() -> None:
    """Ordering against unrelated types raises TypeError."""
    a = _TestCxxClassDerived(1, 2, 3.0, 4.0)
    with pytest.raises(TypeError):
        a < "hello"  # ty: ignore[unsupported-operator]
    with pytest.raises(TypeError):
        a <= 42  # ty: ignore[unsupported-operator]
    with pytest.raises(TypeError):
        a > 3.14  # ty: ignore[unsupported-operator]
    with pytest.raises(TypeError):
        a >= None  # ty: ignore[unsupported-operator]


# ---------------------------------------------------------------------------
# 7. Subclass equality
# ---------------------------------------------------------------------------


def test_c_class_subclass_eq() -> None:
    """Subclass instances can be compared to parent instances without crashing."""
    derived = _TestCxxClassDerived(1, 2, 3.0, 4.0)
    derived_derived = _TestCxxClassDerivedDerived(
        v_i64=1, v_i32=2, v_f64=3.0, v_f32=4.0, v_str="hello", v_bool=True
    )
    # These are different types in the same hierarchy; comparison should
    # return a bool (the result depends on C++ behavior).
    result = derived == derived_derived
    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# 8. KwOnly from C++ reflection
# ---------------------------------------------------------------------------


def test_c_class_kw_only_signature() -> None:
    """kw_only trait comes from C++ reflection, not Python decorator."""
    sig = inspect.signature(_TestCxxKwOnly.__init__)
    params = sig.parameters
    for name in ("x", "y", "z", "w"):
        assert params[name].kind == inspect.Parameter.KEYWORD_ONLY, (
            f"Expected {name} to be KEYWORD_ONLY"
        )


def test_c_class_kw_only_call() -> None:
    """KwOnly fields can be supplied as keyword arguments."""
    obj = _TestCxxKwOnly(x=1, y=2, z=3, w=4)
    assert obj.x == 1
    assert obj.y == 2
    assert obj.z == 3
    assert obj.w == 4


def test_c_class_kw_only_default() -> None:
    """KwOnly field with a C++ default can be omitted."""
    obj = _TestCxxKwOnly(x=1, y=2, z=3)
    assert obj.w == 100


def test_c_class_kw_only_rejects_positional() -> None:
    """KwOnly fields reject positional arguments."""
    with pytest.raises(TypeError, match="positional"):
        _TestCxxKwOnly(1, 2, 3, 4)  # ty: ignore[missing-argument, too-many-positional-arguments]


# ---------------------------------------------------------------------------
# 9. Init subset from C++ reflection
# ---------------------------------------------------------------------------


def test_c_class_init_subset_signature() -> None:
    """init=False fields from C++ reflection are excluded from __init__."""
    sig = inspect.signature(_TestCxxInitSubset.__init__)
    params = tuple(sig.parameters)
    assert "required_field" in params
    assert "optional_field" not in params
    assert "note" not in params


def test_c_class_init_subset_defaults() -> None:
    """init=False fields get their default values from C++."""
    obj = _TestCxxInitSubset(required_field=42)
    assert obj.required_field == 42
    assert obj.optional_field == -1  # C++ default
    assert obj.note == "default"  # C++ default


def test_c_class_init_subset_positional() -> None:
    """Init-subset fields can be passed positionally."""
    obj = _TestCxxInitSubset(7)
    assert obj.required_field == 7
    assert obj.optional_field == -1


def test_c_class_init_subset_field_writable() -> None:
    """Fields excluded from __init__ can still be assigned after construction."""
    obj = _TestCxxInitSubset(required_field=0)
    obj.optional_field = 11
    assert obj.optional_field == 11


# ---------------------------------------------------------------------------
# 10. DerivedDerived with defaults
# ---------------------------------------------------------------------------


def test_c_class_derived_derived_defaults() -> None:
    """DerivedDerived uses positional args; C++ defaults fill in omitted fields."""
    obj = _TestCxxClassDerivedDerived(1, 2, 3.0, True)
    assert obj.v_i64 == 1
    assert obj.v_i32 == 2
    assert obj.v_f64 == 3.0
    assert obj.v_f32 == 8.0  # C++ default
    assert obj.v_str == "default"  # C++ default
    assert obj.v_bool is True


def test_c_class_derived_derived_all_explicit() -> None:
    """DerivedDerived with all fields explicitly provided."""
    obj = _TestCxxClassDerivedDerived(
        v_i64=123,
        v_i32=456,
        v_f64=4.0,
        v_f32=9.0,
        v_str="hello",
        v_bool=True,
    )
    assert obj.v_i64 == 123
    assert obj.v_i32 == 456
    assert obj.v_f64 == 4.0
    assert obj.v_f32 == 9.0
    assert obj.v_str == "hello"
    assert obj.v_bool is True


# ---------------------------------------------------------------------------
# 11. Hash / set usage
# ---------------------------------------------------------------------------


def test_c_class_usable_in_set() -> None:
    """Equal objects deduplicate in a set."""
    a = _TestCxxClassDerived(1, 2, 3.0, 4.0)
    b = _TestCxxClassDerived(1, 2, 3.0, 4.0)
    c = _TestCxxClassDerived(5, 6, 7.0, 8.0)
    s = {a, b, c}
    assert len(s) == 2  # a and b are equal


def test_c_class_unequal_objects_in_set() -> None:
    """Distinct objects are separate entries in a set."""
    objs = {_TestCxxClassDerived(i, i, float(i), float(i)) for i in range(5)}
    assert len(objs) == 5


# ---------------------------------------------------------------------------
# 12. Slots validation
# ---------------------------------------------------------------------------


def test_c_class_slots_default_ok() -> None:
    """Default slots=True passes validation for Object subclass with __slots__."""
    # Slots validation passes; the ValueError comes from register_object
    # because the type key doesn't exist in C++.
    with pytest.raises(ValueError, match="Cannot find object type index"):

        @c_class("test.SlotsDefaultOk", init=False)
        class _SlotsDefaultOk(Object):
            pass


def test_c_class_slots_false_ok() -> None:
    """slots=False passes validation for Object subclass with slots=False in header."""
    # Slots validation passes; the ValueError comes from register_object.
    with pytest.raises(ValueError, match="Cannot find object type index"):

        @c_class("test.SlotsFalseOk", init=False, slots=False)
        class _SlotsFalseOk(Object, slots=False):
            pass


def test_c_class_slots_true_mismatch() -> None:
    """slots=True but class header has slots=False raises TypeError."""
    with pytest.raises(TypeError, match=r"slots=True.*expects.*__slots__"):

        @c_class("test.SlotsTrueMismatch", init=False, slots=True)
        class _SlotsTrueMismatch(Object, slots=False):
            pass


def test_c_class_slots_false_mismatch() -> None:
    """slots=False but class header has default slots=True raises TypeError."""
    with pytest.raises(TypeError, match=r"slots=False.*expects no.*__slots__"):

        @c_class("test.SlotsFalseMismatch", init=False, slots=False)
        class _SlotsFalseMismatch(Object):
            pass
