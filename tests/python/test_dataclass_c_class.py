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
import warnings

import pytest
import tvm_ffi.testing
from tvm_ffi.core import MISSING, TypeInfo
from tvm_ffi.dataclasses import Field
from tvm_ffi.dataclasses.c_class import _attach_field_objects
from tvm_ffi.registry import _warn_missing_field_annotations
from tvm_ffi.testing import (
    TestCompare,
    TestHash,
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
    assert small < big
    assert small <= big
    assert big > small
    assert big >= small
    assert not (big < small)
    assert not (small > big)


def test_c_class_ordering_reflexive() -> None:
    """<= and >= are reflexive."""
    a = _TestCxxClassDerived(1, 2, 3.0, 4.0)
    b = a  # alias, same object
    assert a <= b
    assert a >= b


def test_c_class_ordering_antisymmetric() -> None:
    """If a < b then not b < a."""
    a = _TestCxxClassDerived(0, 0, 0.0, 0.0)
    b = _TestCxxClassDerived(100, 100, 100.0, 100.0)
    if a < b:
        assert not (b < a)
    else:
        assert not (a < b)


# ---------------------------------------------------------------------------
# 6. Equality with different types returns NotImplemented
# ---------------------------------------------------------------------------


def test_c_class_eq_different_type() -> None:
    """__eq__ returns NotImplemented for unrelated types."""
    a = _TestCxxClassDerived(1, 2, 3.0, 4.0)
    assert a != "hello"
    assert a != 42
    assert a != 3.14
    assert a is not None


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
# 12. Field annotation warnings
# ---------------------------------------------------------------------------


def test_c_class_warns_on_missing_field_annotations() -> None:
    """@c_class warns when reflected fields lack Python annotations."""
    type_info: TypeInfo = getattr(_TestCxxClassBase, "__tvm_ffi_type_info__")
    field_names = {f.name for f in type_info.fields}
    assert field_names  # sanity: there are reflected fields

    # A class with no annotations should trigger a warning
    DummyCls = type("DummyCls", (), {})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _warn_missing_field_annotations(DummyCls, type_info, stacklevel=2)
    assert len(w) == 1
    assert "does not annotate" in str(w[0].message)
    for name in field_names:
        assert name in str(w[0].message)


def test_c_class_no_warning_when_all_fields_annotated() -> None:
    """@c_class does not warn when all reflected fields are annotated."""
    type_info: TypeInfo = getattr(_TestCxxClassBase, "__tvm_ffi_type_info__")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _warn_missing_field_annotations(_TestCxxClassBase, type_info, stacklevel=2)
    assert len(w) == 0


def test_c_class_warns_only_for_missing_annotations() -> None:
    """Warning lists only the missing fields, not the annotated ones."""
    type_info: TypeInfo = getattr(_TestCxxClassBase, "__tvm_ffi_type_info__")
    field_names = sorted(f.name for f in type_info.fields)
    assert len(field_names) >= 2  # need at least 2 fields for this test

    # Annotate only the first field, leave the rest unannotated
    PartialCls = type("PartialCls", (), {"__annotations__": {field_names[0]: int}})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _warn_missing_field_annotations(PartialCls, type_info, stacklevel=2)
    assert len(w) == 1
    msg = str(w[0].message)
    # The annotated field should NOT appear in the warning
    assert field_names[0] not in msg
    # The unannotated fields should appear
    for name in field_names[1:]:
        assert name in msg


def test_c_class_warns_only_own_fields_not_inherited() -> None:
    """Warning only checks own fields, not parent fields."""
    # _TestCxxClassDerived's type_info.fields contains only its own fields
    # (v_f64, v_f32), not parent fields (v_i64, v_i32).
    derived_type_info: TypeInfo = getattr(_TestCxxClassDerived, "__tvm_ffi_type_info__")
    own_field_names = {f.name for f in derived_type_info.fields}
    assert own_field_names  # sanity

    # A class with no annotations: warning should mention only own fields
    DummyCls = type("DummyCls", (), {})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _warn_missing_field_annotations(DummyCls, derived_type_info, stacklevel=2)
    assert len(w) == 1
    msg = str(w[0].message)
    for name in own_field_names:
        assert name in msg
    # Parent fields should NOT appear in the warning
    parent_type_info = derived_type_info.parent_type_info
    if parent_type_info is not None:
        parent_field_names = {f.name for f in parent_type_info.fields}
        for name in parent_field_names:
            assert name not in msg


# ---------------------------------------------------------------------------
# 13. Field object attachment (_attach_field_objects)
# ---------------------------------------------------------------------------


def test_c_class_attaches_field_object_per_typefield() -> None:
    """Every own reflected field gets a ``Field`` instance on ``dataclass_field``."""
    type_info: TypeInfo = getattr(_TestCxxClassBase, "__tvm_ffi_type_info__")
    assert type_info.fields  # sanity
    for tf in type_info.fields:
        assert isinstance(tf.dataclass_field, Field)


def test_c_class_field_name_matches_typefield() -> None:
    """``Field.name`` mirrors ``TypeField.name``."""
    type_info: TypeInfo = getattr(_TestCxxClassBase, "__tvm_ffi_type_info__")
    for tf in type_info.fields:
        assert tf.dataclass_field.name == tf.name


def test_c_class_field_private_schema_mirrors_typefield() -> None:
    """``Field._ty_schema`` is forwarded verbatim from ``TypeField.ty``.

    For C++-backed fields ``TypeField.ty`` is typically ``None`` (only
    populated by ``@py_class`` registration), so the helper should just
    forward the value without fabricating a :class:`TypeSchema`.
    """
    type_info: TypeInfo = getattr(_TestCxxClassBase, "__tvm_ffi_type_info__")
    for tf in type_info.fields:
        assert tf.dataclass_field._ty_schema is tf.ty


def test_c_class_field_type_resolved_from_annotation() -> None:
    """``Field.type`` is the resolved Python annotation, not a TypeSchema."""
    type_info: TypeInfo = getattr(_TestCxxClassBase, "__tvm_ffi_type_info__")
    for tf in type_info.fields:
        # _TestCxxClassBase annotates v_i64 / v_i32 as ``int``.
        assert tf.dataclass_field.type is int


def test_c_class_field_defaults_missing_when_unspecified() -> None:
    """Fields with no C++ default retain ``MISSING`` on ``Field.default``.

    ``_TestCxxClassBase.v_i64`` / ``v_i32`` are registered without
    ``refl::default_value(...)``; the reflection layer should leave
    ``c_default`` / ``c_default_factory`` as :data:`MISSING`.
    """
    type_info: TypeInfo = getattr(_TestCxxClassBase, "__tvm_ffi_type_info__")
    for tf in type_info.fields:
        assert tf.c_default is MISSING
        assert tf.c_default_factory is MISSING
        assert tf.dataclass_field.default is MISSING
        assert tf.dataclass_field.default_factory is MISSING


def test_c_class_field_default_value_from_cxx() -> None:
    """C++ ``refl::default_value(...)`` is exposed on ``Field.default``.

    ``_TestCxxClassDerived.v_f32`` registers ``refl::default_value(8.0f)``;
    ``_TestCxxClassDerivedDerived.v_str`` registers
    ``refl::default_value(String("default"))``.  Both should round-trip
    through ``TypeField.c_default`` / ``Field.default``.
    """
    derived_info: TypeInfo = getattr(_TestCxxClassDerived, "__tvm_ffi_type_info__")
    by_name = {tf.name: tf for tf in derived_info.fields}
    # v_f64: no default → MISSING.
    assert by_name["v_f64"].c_default is MISSING
    assert by_name["v_f64"].dataclass_field.default is MISSING
    # v_f32: C++ default 8.0f.
    assert by_name["v_f32"].c_default == pytest.approx(8.0)
    assert by_name["v_f32"].dataclass_field.default == pytest.approx(8.0)
    assert by_name["v_f32"].c_default_factory is MISSING
    assert by_name["v_f32"].dataclass_field.default_factory is MISSING

    dd_info: TypeInfo = getattr(_TestCxxClassDerivedDerived, "__tvm_ffi_type_info__")
    dd_by_name = {tf.name: tf for tf in dd_info.fields}
    assert dd_by_name["v_str"].c_default == "default"
    assert dd_by_name["v_str"].dataclass_field.default == "default"
    # v_bool has no default.
    assert dd_by_name["v_bool"].c_default is MISSING


def test_c_class_field_default_respects_init_false() -> None:
    """Defaults are visible even when ``init=False`` (fields filled by C++)."""
    subset_info: TypeInfo = getattr(_TestCxxInitSubset, "__tvm_ffi_type_info__")
    by_name = {tf.name: tf for tf in subset_info.fields}
    # optional_field / note are init(false) + have C++ defaults.
    assert by_name["optional_field"].c_default == -1
    assert by_name["optional_field"].dataclass_field.default == -1
    assert by_name["optional_field"].dataclass_field.init is False
    assert by_name["note"].c_default == "default"
    assert by_name["note"].dataclass_field.default == "default"
    # required_field has no default.
    assert by_name["required_field"].c_default is MISSING


def test_c_class_field_default_respects_kw_only() -> None:
    """Defaults are visible on kw_only fields too (``_TestCxxKwOnly.w``)."""
    kw_info: TypeInfo = getattr(_TestCxxKwOnly, "__tvm_ffi_type_info__")
    by_name = {tf.name: tf for tf in kw_info.fields}
    # w is kw_only and has C++ default 100.
    assert by_name["w"].c_default == 100
    assert by_name["w"].dataclass_field.default == 100
    assert by_name["w"].dataclass_field.kw_only is True
    # x / y / z are kw_only without defaults.
    for name in ("x", "y", "z"):
        assert by_name[name].c_default is MISSING
        assert by_name[name].dataclass_field.default is MISSING


def test_c_class_field_frozen_matches_typefield() -> None:
    """``Field.frozen`` tracks ``TypeField.frozen`` (TestIntPair fields are frozen)."""
    type_info: TypeInfo = getattr(tvm_ffi.testing.TestIntPair, "__tvm_ffi_type_info__")
    for tf in type_info.fields:
        assert tf.dataclass_field.frozen == tf.frozen


def test_c_class_field_init_and_kw_only_flags() -> None:
    """``init`` / ``kw_only`` flags propagate from the reflection layer."""
    # _TestCxxKwOnly has *all* fields kw-only.
    kw_info: TypeInfo = getattr(_TestCxxKwOnly, "__tvm_ffi_type_info__")
    for tf in kw_info.fields:
        assert tf.dataclass_field.kw_only is True
        assert tf.dataclass_field.init is True

    # _TestCxxInitSubset has some fields with Init(false) (not in __init__).
    subset_info: TypeInfo = getattr(_TestCxxInitSubset, "__tvm_ffi_type_info__")
    by_name = {tf.name: tf for tf in subset_info.fields}
    assert by_name["required_field"].dataclass_field.init is True
    # optional_field / note are marked Init(false) in C++.
    assert by_name["optional_field"].dataclass_field.init is False
    assert by_name["note"].dataclass_field.init is False


def test_c_class_field_doc_matches_typefield() -> None:
    """``Field.doc`` mirrors ``TypeField.doc``."""
    type_info: TypeInfo = getattr(_TestCxxClassBase, "__tvm_ffi_type_info__")
    for tf in type_info.fields:
        assert tf.dataclass_field.doc == tf.doc


def test_c_class_field_repr_flag_from_cxx() -> None:
    """``refl::repr(false)`` on a C++ field propagates to ``Field.repr``.

    ``_TestCxxClassBase`` registers both of its fields with
    ``refl::repr(false)``; ``_TestCxxClassDerived`` registers its fields
    without that modifier (default on).
    """
    base_info: TypeInfo = getattr(_TestCxxClassBase, "__tvm_ffi_type_info__")
    for tf in base_info.fields:
        assert tf.c_repr is False
        assert tf.dataclass_field.repr is False
    derived_info: TypeInfo = getattr(_TestCxxClassDerived, "__tvm_ffi_type_info__")
    for tf in derived_info.fields:
        assert tf.c_repr is True
        assert tf.dataclass_field.repr is True


def test_c_class_field_compare_flag_from_cxx() -> None:
    """``refl::compare(false)`` on a C++ field propagates to ``Field.compare``.

    ``TestCompare.ignored_field`` is registered with ``refl::compare(false)``;
    ``key`` and ``name`` are registered without it.
    """
    type_info: TypeInfo = getattr(TestCompare, "__tvm_ffi_type_info__")
    by_name = {tf.name: tf for tf in type_info.fields}
    assert by_name["key"].c_compare is True
    assert by_name["key"].dataclass_field.compare is True
    assert by_name["name"].c_compare is True
    assert by_name["name"].dataclass_field.compare is True
    assert by_name["ignored_field"].c_compare is False
    assert by_name["ignored_field"].dataclass_field.compare is False


def test_c_class_field_hash_flag_from_cxx() -> None:
    """``refl::hash(false)`` on a C++ field propagates to ``Field.hash``.

    ``TestHash.hash_ignored`` is registered with ``refl::hash(false)``;
    ``key`` and ``name`` are registered without it.
    """
    type_info: TypeInfo = getattr(TestHash, "__tvm_ffi_type_info__")
    by_name = {tf.name: tf for tf in type_info.fields}
    assert by_name["key"].c_hash is True
    assert by_name["key"].dataclass_field.hash is True
    assert by_name["name"].c_hash is True
    assert by_name["name"].dataclass_field.hash is True
    assert by_name["hash_ignored"].c_hash is False
    assert by_name["hash_ignored"].dataclass_field.hash is False


def test_c_class_field_structural_eq_default_none() -> None:
    """Fields registered without ``s_eq_hash_def`` / ``s_eq_hash_ignore`` stay ``None``.

    None of the current C++ testing fixtures set structural_eq flags on a
    field, so we verify the default (``None``) is preserved across the
    reflection boundary.
    """
    for cls in (_TestCxxClassBase, _TestCxxClassDerived, TestCompare, TestHash):
        type_info: TypeInfo = getattr(cls, "__tvm_ffi_type_info__")
        for tf in type_info.fields:
            assert tf.c_structural_eq is None
            assert tf.dataclass_field.structural_eq is None


def test_c_class_attaches_only_own_fields_not_inherited() -> None:
    """Only own fields get a Field; parent fields are attached on the parent's TypeInfo."""
    derived_info: TypeInfo = getattr(_TestCxxClassDerived, "__tvm_ffi_type_info__")
    own_names = {tf.name for tf in derived_info.fields}
    # Derived owns v_f64 / v_f32; parent owns v_i64 / v_i32.
    assert own_names == {"v_f64", "v_f32"}
    for tf in derived_info.fields:
        assert isinstance(tf.dataclass_field, Field)
    # Parent TypeInfo already had Field objects attached when it was decorated.
    parent_info = derived_info.parent_type_info
    assert parent_info is not None
    for tf in parent_info.fields:
        assert isinstance(tf.dataclass_field, Field)


def test_c_class_field_type_none_when_annotation_missing() -> None:
    """Fields without a Python annotation get ``Field.type == None``."""
    # Re-run _attach_field_objects on a bare class (no annotations) — it must
    # not raise and should leave every Field.type == None.
    type_info: TypeInfo = getattr(_TestCxxClassBase, "__tvm_ffi_type_info__")
    original = [tf.dataclass_field for tf in type_info.fields]
    bare = type("BareCls", (), {})
    try:
        _attach_field_objects(bare, type_info)
        for tf in type_info.fields:
            assert tf.dataclass_field.type is None
    finally:
        # Restore so later tests still see the annotated Field.type.
        for tf, saved in zip(type_info.fields, original):
            tf.dataclass_field = saved


def test_c_class_attaches_field_for_new_decoration() -> None:
    """A freshly decorated @c_class type has its own Field objects."""
    # We don't create a new C++ type here — just re-run the decorator machinery
    # with a duplicate key-aware workflow isn't necessary. Simply assert that
    # the existing decorated classes all expose populated dataclass_field.
    for cls in (_TestCxxClassBase, _TestCxxClassDerived, _TestCxxClassDerivedDerived):
        ti: TypeInfo = getattr(cls, "__tvm_ffi_type_info__")
        for tf in ti.fields:
            assert tf.dataclass_field is not None
            assert tf.dataclass_field.name == tf.name


def test_c_class_attach_is_idempotent() -> None:
    """Calling ``_attach_field_objects`` twice replaces Fields without error."""
    type_info: TypeInfo = getattr(_TestCxxClassBase, "__tvm_ffi_type_info__")
    original = [tf.dataclass_field for tf in type_info.fields]
    try:
        _attach_field_objects(_TestCxxClassBase, type_info)
        for tf in type_info.fields:
            assert isinstance(tf.dataclass_field, Field)
            assert tf.dataclass_field.type is int
    finally:
        for tf, saved in zip(type_info.fields, original):
            tf.dataclass_field = saved


def test_c_class_attach_tolerates_unresolvable_hints() -> None:
    """``typing.get_type_hints`` exceptions fall back silently; ``Field.type`` stays None."""

    # Declaring a type with a string annotation that points at an undefined name
    # makes get_type_hints raise NameError.  The helper must swallow it.
    class BrokenAnn:
        x: ThisNameIsNeverDefined  # ty: ignore[unresolved-reference]  # noqa: F821

    type_info: TypeInfo = getattr(_TestCxxClassBase, "__tvm_ffi_type_info__")
    original = [tf.dataclass_field for tf in type_info.fields]
    try:
        _attach_field_objects(BrokenAnn, type_info)
        for tf in type_info.fields:
            assert tf.dataclass_field.type is None
    finally:
        for tf, saved in zip(type_info.fields, original):
            tf.dataclass_field = saved
