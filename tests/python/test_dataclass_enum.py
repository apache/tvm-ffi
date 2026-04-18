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
"""Tests for ``Enum`` subclasses with ``type_key``-parameterized inheritance."""

from __future__ import annotations

import itertools
from typing import ClassVar

import pytest
import tvm_ffi
from tvm_ffi import core
from tvm_ffi.core import Object
from tvm_ffi.dataclasses import Enum, EnumAttrMap, auto, entry
from tvm_ffi.dataclasses.enum import (
    ENUM_ATTRS_ATTR,
    ENUM_ENTRIES_ATTR,
    _EnumEntry,
)

# ---------------------------------------------------------------------------
# Unique type-key generator — ensures no collisions across tests.
# ---------------------------------------------------------------------------
_counter = itertools.count()


def _unique_key(base: str) -> str:
    return f"testing.py_enum_sub.{base}_{next(_counter)}"


# ---------------------------------------------------------------------------
# Attribute-carrying form — entry(...) with extra field kwargs
# ---------------------------------------------------------------------------


def test_attribute_carrying_basic() -> None:
    class Activation(Enum, type_key=_unique_key("Activation")):
        output_zero: bool
        is_monotonic: bool

        relu: ClassVar[Activation] = entry(output_zero=True, is_monotonic=True)
        gelu: ClassVar[Activation] = entry(output_zero=False, is_monotonic=False)
        silu: ClassVar[Activation] = entry(output_zero=False, is_monotonic=True)

    assert isinstance(Activation.relu, Activation)
    assert isinstance(Activation.gelu, Activation)
    assert isinstance(Activation.silu, Activation)

    assert Activation.relu.output_zero is True  # ty: ignore[unresolved-attribute]
    assert Activation.relu.is_monotonic is True  # ty: ignore[unresolved-attribute]
    assert Activation.gelu.output_zero is False  # ty: ignore[unresolved-attribute]
    assert Activation.silu.is_monotonic is True  # ty: ignore[unresolved-attribute]

    # Ordinals auto-assigned in declaration order.
    assert Activation.relu.value == 0
    assert Activation.gelu.value == 1
    assert Activation.silu.value == 2
    assert Activation.relu.name == "relu"
    assert Activation.gelu.name == "gelu"

    assert Activation.get("relu").same_as(Activation.relu)
    assert Activation.get("gelu").same_as(Activation.gelu)
    assert Activation.get("silu").same_as(Activation.silu)


def test_entry_rejects_value_kwarg() -> None:
    """``entry(value=...)`` conflicts with the auto-assigned ordinal."""
    with pytest.raises(TypeError):

        class _Bad(Enum, type_key=_unique_key("BadValue")):
            flag: bool
            a: ClassVar[_Bad] = entry(flag=True, value=5)


def test_entry_rejects_name_kwarg() -> None:
    """``entry(name=...)`` conflicts with the auto-assigned declaration key."""
    with pytest.raises(TypeError):

        class _Bad(Enum, type_key=_unique_key("BadName")):
            flag: bool
            a: ClassVar[_Bad] = entry(flag=True, name="other")


def test_get_missing_raises() -> None:
    class Missing(Enum, type_key=_unique_key("Missing")):
        flag: bool
        yes: ClassVar[Missing] = entry(flag=True)

    with pytest.raises(KeyError):
        Missing.get("no-such-entry")


def test_entries_iteration_order() -> None:
    class Ordered(Enum, type_key=_unique_key("Ordered")):
        tag: str
        a: ClassVar[Ordered] = entry(tag="first")
        b: ClassVar[Ordered] = entry(tag="second")
        c: ClassVar[Ordered] = entry(tag="third")

    values = list(Ordered.entries())
    assert len(values) == 3
    assert values[0].same_as(Ordered.a)
    assert values[1].same_as(Ordered.b)
    assert values[2].same_as(Ordered.c)


def test_frozen_variants() -> None:
    class Frozen(Enum, type_key=_unique_key("Frozen")):
        flag: bool
        yes: ClassVar[Frozen] = entry(flag=True)

    with pytest.raises(AttributeError):
        Frozen.yes.flag = False  # ty: ignore[invalid-assignment]


# ---------------------------------------------------------------------------
# Bare ClassVar[Cls] (no assignment) — Python-side blank entries
# ---------------------------------------------------------------------------


def test_bare_classvar_without_cxx_entries() -> None:
    """``ClassVar[Cls]`` with no value registers a new blank Python entry
    when the type key has no C++ backing.
    """

    class Status(Enum, type_key=_unique_key("Status")):
        ok: ClassVar[Status]
        err: ClassVar[Status]
        retry: ClassVar[Status]

    assert isinstance(Status.ok, Status)
    assert Status.ok.value == 0
    assert Status.err.value == 1
    assert Status.retry.value == 2
    assert Status.ok.name == "ok"
    assert Status.err.name == "err"
    assert list(Status.entries()) == [Status.ok, Status.err, Status.retry]
    assert Status.get("ok").same_as(Status.ok)


def test_bare_classvar_mixed_with_entry() -> None:
    """Bare ``ClassVar`` and ``ClassVar = entry(...)`` may mix in one class."""

    class Kind(Enum, type_key=_unique_key("Kind")):
        blank: ClassVar[Kind]
        tag: str = ""  # ordinary field; extra fields follow
        named: ClassVar[Kind] = entry(tag="hi")

    # ``ClassVar`` binders are processed before ``entry(...)`` assignments.
    assert Kind.blank.value == 0
    assert Kind.named.value == 1
    assert Kind.blank.name == "blank"
    assert Kind.named.name == "named"
    assert Kind.named.tag == "hi"  # ty: ignore[unresolved-attribute]


# ---------------------------------------------------------------------------
# Bare ``name = entry(...)`` sugar (no ClassVar annotation)
# ---------------------------------------------------------------------------


def test_bare_entry_sugar_form() -> None:
    """``name = entry(...)`` without a ``ClassVar`` annotation is picked up."""

    class Activation(Enum, type_key=_unique_key("ActivationBare")):
        output_zero: bool
        is_monotonic: bool

        relu = entry(output_zero=True, is_monotonic=True)
        gelu = entry(output_zero=False, is_monotonic=False)

    assert isinstance(Activation.relu, Activation)
    assert Activation.relu.output_zero is True  # ty: ignore[unresolved-attribute]
    assert Activation.gelu.output_zero is False  # ty: ignore[unresolved-attribute]
    assert list(Activation.entries()) == [Activation.relu, Activation.gelu]


# ---------------------------------------------------------------------------
# ``auto()`` — simple Python-side entries without ClassVar annotation
# ---------------------------------------------------------------------------


def test_auto_basic_no_annotation() -> None:
    """``name = auto()`` registers a py-side entry with dense auto-ordinals."""

    class Priority(Enum, type_key=_unique_key("Priority")):
        low = auto()
        medium = auto()
        high = auto()

    assert isinstance(Priority.low, Priority)
    assert Priority.low.value == 0
    assert Priority.medium.value == 1
    assert Priority.high.value == 2
    assert Priority.low.name == "low"
    assert Priority.high.name == "high"
    assert list(Priority.entries()) == [Priority.low, Priority.medium, Priority.high]


def test_auto_with_classvar_annotation() -> None:
    """``name: ClassVar[Cls] = auto()`` is equivalent to the annotation-less form."""

    class Stage(Enum, type_key=_unique_key("Stage")):
        init: ClassVar[Stage] = auto()
        run: ClassVar[Stage] = auto()
        done: ClassVar[Stage] = auto()

    assert Stage.init.value == 0
    assert Stage.run.value == 1
    assert Stage.done.value == 2


def test_auto_mixed_with_bare_classvar() -> None:
    """``auto()`` may coexist with bare ``ClassVar`` binders in one class.

    Bare ``ClassVar`` binders are processed first (in annotation order),
    then ``auto()`` / ``entry(...)`` sentinels in class-body order — so
    ordinals reflect that deterministic two-phase order.
    """

    class Mixed(Enum, type_key=_unique_key("Mixed")):
        alpha: ClassVar[Mixed]
        beta = auto()
        gamma: ClassVar[Mixed]

    # Binders (alpha, gamma) come first in annotation order, then sentinels.
    assert Mixed.alpha.value == 0
    assert Mixed.gamma.value == 1
    assert Mixed.beta.value == 2
    assert {v.name for v in Mixed.entries()} == {"alpha", "beta", "gamma"}


def test_auto_mixed_with_entry() -> None:
    """``auto()`` and ``entry(...)`` compose on an attribute-carrying enum."""

    class Op(Enum, type_key=_unique_key("OpMixedAuto")):
        arity: int = 0
        noop = auto()
        add = entry(arity=2)
        neg = entry(arity=1)

    assert Op.noop.value == 0
    assert Op.add.value == 1
    assert Op.neg.value == 2
    assert Op.noop.arity == 0  # ty: ignore[unresolved-attribute]
    assert Op.add.arity == 2  # ty: ignore[unresolved-attribute]


def test_auto_rejects_already_registered_name() -> None:
    """``auto()`` on a name already in the entries dict is rejected.

    ``testing.TestEnumVariant`` pre-registers ``Alpha`` / ``Beta`` from C++,
    so attempting to *register* (rather than bind) ``Alpha`` via ``auto()``
    must fail — bare ``ClassVar[Cls]`` is the way to bind to an existing
    entry.
    """
    with pytest.raises(RuntimeError):

        class _Shadow(Enum, type_key="testing.TestEnumVariant"):
            Alpha = auto()


def test_auto_returns_fresh_sentinels() -> None:
    """Each ``auto()`` call returns a distinct sentinel instance."""
    a, b = auto(), auto()
    assert isinstance(a, _EnumEntry)
    assert isinstance(b, _EnumEntry)
    assert a is not b
    assert a.args == ()
    assert a.kwargs == {}


# ---------------------------------------------------------------------------
# by_name / by_value / attr_dict
# ---------------------------------------------------------------------------


def test_by_name_is_live_dict() -> None:
    class K(Enum, type_key=_unique_key("ByName")):
        a: ClassVar[K]
        b: ClassVar[K]

    assert set(K.by_name.keys()) == {"a", "b"}
    assert K.by_name["a"].same_as(K.a)


def test_by_value_indexed_by_ordinal() -> None:
    class K(Enum, type_key=_unique_key("ByValue")):
        a: ClassVar[K]
        b: ClassVar[K]
        c: ClassVar[K]

    by_val = K.by_value
    assert len(by_val) == 3
    assert by_val[0].same_as(K.a)
    assert by_val[1].same_as(K.b)
    assert by_val[2].same_as(K.c)


def test_attr_dict_direct_access() -> None:
    """The ``attr_dict`` class property returns the live per-variant attrs map."""

    class Op(Enum, type_key=_unique_key("OpDirect")):
        arity: int
        add: ClassVar[Op] = entry(arity=2)
        neg: ClassVar[Op] = entry(arity=1)

    has_side_effects = Op.def_attr("has_side_effects", default=False)
    has_side_effects[Op.add] = False
    has_side_effects[Op.neg] = True

    # Direct read via class-level property.
    column = Op.attr_dict["has_side_effects"]
    assert column[Op.add.value] is False
    assert column[Op.neg.value] is True


# ---------------------------------------------------------------------------
# EnumAttrMap / def_attr
# ---------------------------------------------------------------------------


def test_def_attr_basic_get_set() -> None:
    class Op(Enum, type_key=_unique_key("Op")):
        arity: int
        add: ClassVar[Op] = entry(arity=2)
        neg: ClassVar[Op] = entry(arity=1)

    cost = Op.def_attr("cost", default=0)
    assert isinstance(cost, EnumAttrMap)

    assert cost[Op.add] == 0
    assert cost[Op.neg] == 0

    cost[Op.add] = 5
    cost[Op.neg] = 2
    assert cost[Op.add] == 5
    assert cost[Op.neg] == 2


def test_def_attr_missing_raises_without_default() -> None:
    class OpStrict(Enum, type_key=_unique_key("OpStrict")):
        arity: int
        add: ClassVar[OpStrict] = entry(arity=2)

    attr = OpStrict.def_attr("strict_attr")
    with pytest.raises(KeyError):
        _ = attr[OpStrict.add]


def test_def_attr_get_method_default() -> None:
    class Op(Enum, type_key=_unique_key("OpGetDefault")):
        arity: int
        add: ClassVar[Op] = entry(arity=2)

    attr = Op.def_attr("cost")
    assert attr.get(Op.add, -1) == -1
    attr[Op.add] = 9
    assert attr.get(Op.add, -1) == 9


def test_def_attr_rejects_foreign_variant() -> None:
    class Left(Enum, type_key=_unique_key("LeftEnum")):
        v: int
        one: ClassVar[Left] = entry(v=1)

    class Right(Enum, type_key=_unique_key("RightEnum")):
        v: int
        one: ClassVar[Right] = entry(v=1)

    attr = Left.def_attr("x", default=0)
    with pytest.raises(TypeError):
        attr[Right.one] = 1


def test_def_attr_contains() -> None:
    class Op(Enum, type_key=_unique_key("OpC")):
        arity: int
        add: ClassVar[Op] = entry(arity=2)
        neg: ClassVar[Op] = entry(arity=1)

    cost = Op.def_attr("cost", default=0)
    assert Op.add not in cost
    cost[Op.add] = 5
    assert Op.add in cost
    assert Op.neg not in cost


def test_def_attr_rejects_none_write() -> None:
    """``None`` is reserved as the column's "unset" sentinel."""

    class Op(Enum, type_key=_unique_key("OpNone")):
        arity: int
        add: ClassVar[Op] = entry(arity=2)

    cost = Op.def_attr("cost", default=0)
    with pytest.raises(TypeError, match="reserved as the 'unset' sentinel"):
        cost[Op.add] = None
    assert Op.add not in cost


def test_def_attr_accepts_fresh_wrapper_from_get() -> None:
    """Variants returned by ``Cls.get(...)`` may be fresh Python wrappers
    whose ``id`` differs from the cached class attribute.  Ordinal-indexed
    lookup must still resolve correctly.
    """

    class Op(Enum, type_key=_unique_key("OpFresh")):
        arity: int
        add: ClassVar[Op] = entry(arity=2)

    cost = Op.def_attr("cost", default=0)
    cost[Op.add] = 7

    fresh = Op.get("add")
    assert fresh.same_as(Op.add)
    assert cost[fresh] == 7
    assert fresh in cost


# ---------------------------------------------------------------------------
# `entry` sentinel sanity checks
# ---------------------------------------------------------------------------


def test_entry_sentinel_reprs() -> None:
    e = entry(1, 2, name_key="x")
    assert isinstance(e, _EnumEntry)
    assert e.args == (1, 2)
    assert e.kwargs == {"name_key": "x"}
    assert "entry(" in repr(e)


def test_entry_attribute_access_outside_class_body() -> None:
    """A naked ``entry()`` call returns the sentinel — never a real instance."""
    e = entry(output_zero=True)
    assert not isinstance(e, Object)


# ---------------------------------------------------------------------------
# TypeAttr-level verification
# ---------------------------------------------------------------------------


def test_enum_entries_typeattr_is_mapping() -> None:
    class WithAttr(Enum, type_key=_unique_key("WithAttr")):
        v: int
        one: ClassVar[WithAttr] = entry(v=1)

    tinfo = WithAttr.__tvm_ffi_type_info__  # ty: ignore[unresolved-attribute]
    stored = core._lookup_type_attr(tinfo.type_index, ENUM_ENTRIES_ATTR)
    assert stored is not None
    assert "one" in stored


def test_enum_attrs_typeattr_stored_under_unified_column() -> None:
    """``def_attr`` writes into the ``__ffi_enum_attrs__`` Dict column."""

    class WithAttr(Enum, type_key=_unique_key("UnifiedAttrs")):
        v: int
        one: ClassVar[WithAttr] = entry(v=1)

    attr = WithAttr.def_attr("color", default="?")
    attr[WithAttr.one] = "red"

    tinfo = WithAttr.__tvm_ffi_type_info__  # ty: ignore[unresolved-attribute]
    stored = core._lookup_type_attr(tinfo.type_index, ENUM_ATTRS_ATTR)
    assert stored is not None
    assert "color" in stored
    assert stored["color"][WithAttr.one.value] == "red"


# ---------------------------------------------------------------------------
# C++-backed enum — auto-detected when type_key is already registered
# ---------------------------------------------------------------------------


def test_cxx_enum_obj_get_returns_singleton() -> None:
    """``EnumObj::Get<TestEnumVariantObj>`` (wired as ``testing.enum_variant_get``)
    returns the same singleton as ``Enum.get`` and the Python-side binder.
    """
    cxx_get = tvm_ffi.get_global_func("testing.enum_variant_get")

    class Variant(Enum, type_key="testing.TestEnumVariant"):
        Alpha: ClassVar[Variant]
        Beta: ClassVar[Variant]

    assert cxx_get("Alpha").same_as(Variant.Alpha)
    assert cxx_get("Beta").same_as(Variant.Beta)

    with pytest.raises(RuntimeError, match="no instance named"):
        cxx_get("Nope")


def test_cxx_backed_classvar_binds_to_existing_entries() -> None:
    """``ClassVar[Cls]`` (no assignment) binds to entries registered on the
    C++ side via ``refl::EnumDef``.

    ``testing.TestEnumVariant`` is registered in C++ with two entries,
    ``Alpha`` and ``Beta``, each with a ``code`` attr attached via
    ``set_attr``.  The Python subclass picks these up without declaring any
    new entries of its own.
    """

    class Variant(Enum, type_key="testing.TestEnumVariant"):
        Alpha: ClassVar[Variant]
        Beta: ClassVar[Variant]

    assert isinstance(Variant.Alpha, Variant)
    assert Variant.Alpha.name == "Alpha"
    assert Variant.Beta.name == "Beta"
    # Ordinals come from C++ (registered in Alpha, Beta declaration order).
    assert Variant.Alpha.value == 0
    assert Variant.Beta.value == 1
    assert Variant.get("Alpha").same_as(Variant.Alpha)

    # C++-stored `code` attr is visible through attr_dict.
    code_col = Variant.attr_dict["code"]
    assert code_col[Variant.Alpha.value] == 10
    assert code_col[Variant.Beta.value] == 20


def test_cxx_backed_reads_entries_typeattr() -> None:
    class Variant2(Enum, type_key="testing.TestEnumVariant"):
        Alpha: ClassVar[Variant2]

    tinfo = Variant2.__tvm_ffi_type_info__  # ty: ignore[unresolved-attribute]
    stored = core._lookup_type_attr(tinfo.type_index, ENUM_ENTRIES_ATTR)
    assert stored is not None
    assert "Alpha" in stored


def test_cxx_backed_binder_typo_raises_descriptive_error() -> None:
    """A ``ClassVar[Cls]`` binder naming an entry that isn't in the C++
    registry raises a ``RuntimeError`` that names both the typo and the
    known C++-registered entries, rather than falling through to the
    ``Enum``-base ``init=False`` guard.
    """
    with pytest.raises(RuntimeError) as excinfo:

        class _Typo(Enum, type_key="testing.TestEnumVariant"):
            Alpha: ClassVar[_Typo]
            Neta: ClassVar[_Typo]  # intended to be "Beta"

    msg = str(excinfo.value)
    assert "'Neta'" in msg
    assert "'testing.TestEnumVariant'" in msg
    assert "'Alpha'" in msg
    assert "'Beta'" in msg
    assert "C++" in msg
    assert "ClassVar" in msg


def test_cxx_backed_mixed_entries_via_auto() -> None:
    """A Python subclass of a cxx-backed enum may add new Python-side entries
    via ``auto()`` alongside bare ``ClassVar`` binders for existing C++ entries.

    Ordinals for the new Python entries continue from the count of existing
    entries, preserving the dense-ordinal invariant across the mixed set.
    The ``__ffi_enum_entries__`` dict lives at the type-index level and is
    shared with every other Python subclass of the same ``type_key`` — so
    we pick unique variant names (``Mixed*``) to avoid collisions with
    other tests that also bind to ``testing.TestEnumVariant``.
    """

    class Mixed(Enum, type_key="testing.TestEnumVariant"):
        Alpha: ClassVar[Mixed]
        Beta: ClassVar[Mixed]
        MixedOne = auto()
        MixedTwo = auto()

    # Alpha/Beta come from C++ with ordinals 0 and 1.
    assert Mixed.Alpha.value == 0
    assert Mixed.Beta.value == 1
    assert Mixed.Alpha.name == "Alpha"
    assert Mixed.Beta.name == "Beta"

    # Python-side entries extend the dense ordinal sequence from the C++ count.
    assert Mixed.MixedOne.name == "MixedOne"
    assert Mixed.MixedTwo.name == "MixedTwo"
    assert Mixed.MixedOne.value == Mixed.Beta.value + 1
    assert Mixed.MixedTwo.value == Mixed.Beta.value + 2

    # Round-trip through ``get`` / ``by_name`` / ``entries``.
    assert Mixed.get("MixedOne").same_as(Mixed.MixedOne)
    assert Mixed.get("MixedTwo").same_as(Mixed.MixedTwo)
    assert {"Alpha", "Beta", "MixedOne", "MixedTwo"}.issubset(Mixed.by_name.keys())

    # Python-side variants are real subclass instances of the cxx-backed type.
    assert isinstance(Mixed.MixedOne, Mixed)
    assert isinstance(Mixed.MixedTwo, Mixed)

    # Existing C++ attrs remain unaffected; new Python variants have no attrs yet.
    code = Mixed.attr_dict["code"]
    assert code[Mixed.Alpha.value] == 10
    assert code[Mixed.Beta.value] == 20


def test_cxx_backed_python_entry_accepts_def_attr() -> None:
    """``def_attr`` writes still work for Python-side variants on a cxx-backed enum."""

    class WithPy(Enum, type_key="testing.TestEnumVariant"):
        Alpha: ClassVar[WithPy]
        AttrOne = auto()

    tag = WithPy.def_attr("tag", default=None)
    tag[WithPy.AttrOne] = "py-side"
    assert tag[WithPy.AttrOne] == "py-side"
    # Column was widened to the new ordinal; C++-registered entries retain default.
    assert tag.get(WithPy.Alpha) is None


# ---------------------------------------------------------------------------
# Default ReprPrint for EnumObj subclasses + MISSING/KWARGS sentinels
# ---------------------------------------------------------------------------


def test_default_repr_python_backed() -> None:
    """Python-only enum subclasses format each variant as ``<type_key>.<name>``."""
    key = _unique_key("ReprPy")

    class Priority(Enum, type_key=key):
        low = auto()
        medium = auto()
        high = auto()

    assert repr(Priority.low) == f"{key}.low"
    assert repr(Priority.medium) == f"{key}.medium"
    assert repr(Priority.high) == f"{key}.high"


def test_default_repr_cxx_backed() -> None:
    """C++-registered enum subclasses format with the C++ type_key."""

    class Variant(Enum, type_key="testing.TestEnumVariant"):
        Alpha: ClassVar[Variant]
        Beta: ClassVar[Variant]

    assert repr(Variant.Alpha) == "testing.TestEnumVariant.Alpha"
    assert repr(Variant.Beta) == "testing.TestEnumVariant.Beta"


def test_default_repr_in_nested_container() -> None:
    """Enum repr applies recursively when a variant is nested inside a Dict/List."""
    key = _unique_key("ReprNested")

    class Color(Enum, type_key=key):
        red = auto()
        green = auto()

    by_name_repr = repr(Color.by_name)
    assert f"{key}.red" in by_name_repr
    assert f"{key}.green" in by_name_repr

    by_value_entries = [repr(v) for v in Color.by_value]
    assert by_value_entries == [f"{key}.red", f"{key}.green"]


def test_default_repr_with_attribute_carrying_variant() -> None:
    """Attribute-carrying entries still render with the ``<type_key>.<name>`` form."""
    key = _unique_key("ReprWithAttrs")

    class Op(Enum, type_key=key):
        arity: int
        add: ClassVar[Op] = entry(arity=2)
        neg: ClassVar[Op] = entry(arity=1)

    assert repr(Op.add) == f"{key}.add"
    assert repr(Op.neg) == f"{key}.neg"


def test_missing_and_kwargs_sentinel_repr() -> None:
    """The built-in MISSING and KWARGS singletons render with angle-bracket tags."""
    assert repr(core.MISSING) == "<MISSING>"
    assert repr(core.KWARGS) == "<KWARGS>"
