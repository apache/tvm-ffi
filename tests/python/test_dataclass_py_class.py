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
"""Tests for Python-defined TVM-FFI types: ``@py_class`` decorator and low-level Field API."""

# ruff: noqa: D102, PLR0124, PLW1641, UP006, UP045
from __future__ import annotations

import copy
import gc
import inspect
import itertools
import math
from typing import Any, ClassVar, Dict, List, Optional

import pytest
import tvm_ffi
from tvm_ffi import core
from tvm_ffi._ffi_api import DeepCopy, RecursiveEq, RecursiveHash, ReprPrint
from tvm_ffi.core import MISSING, Object, TypeInfo, TypeSchema, _to_py_class_value
from tvm_ffi.dataclasses import KW_ONLY, Field, field, py_class
from tvm_ffi.registry import _add_class_attrs, _install_dataclass_dunders
from tvm_ffi.testing import TestObjectBase as _TestObjectBase
from tvm_ffi.testing.testing import requires_py310 as _needs_310

# ---------------------------------------------------------------------------
# Unique type key generator (avoids collisions across tests)
# ---------------------------------------------------------------------------
_counter = itertools.count()


def _unique_key(base: str) -> str:
    return f"testing.py_class_dec.{base}_{next(_counter)}"


def _get_type_info(cls: type) -> TypeInfo:
    ret = cls.__tvm_ffi_type_info__  # ty: ignore[unresolved-attribute]
    assert isinstance(ret, TypeInfo), f"Expected TypeInfo, got {type(ret)}"
    return ret


# ---------------------------------------------------------------------------
# Low-level helpers for _make_type-based tests
# ---------------------------------------------------------------------------
_counter_ff = itertools.count()


def _unique_key_ff(base: str) -> str:
    """Return a globally unique type key for low-level field tests."""
    return f"testing.py_class.{base}_{next(_counter_ff)}"


def _make_type(
    name: str,
    fields: List[Field],
    *,
    parent: type = core.Object,
    eq: bool = False,
    unsafe_hash: bool = False,
    repr: bool = True,
) -> type:
    """Create, register, and fully set up a Python-defined TVM-FFI type.

    Returns the ready-to-use Python class.
    """
    type_key = _unique_key_ff(name)
    parent_info = core._type_cls_to_type_info(parent)
    assert parent_info is not None
    cls = type(name, (parent,), {"__slots__": ()})
    info = core._register_py_class(parent_info, type_key, cls)
    info._register_fields(fields)
    setattr(cls, "__tvm_ffi_type_info__", info)
    _add_class_attrs(cls, info)
    _install_dataclass_dunders(
        cls,
        init=True,
        repr=repr,
        eq=eq,
        order=False,
        unsafe_hash=unsafe_hash,
    )
    return cls


# ###########################################################################
#  1. Basic registration
# ###########################################################################
class TestBasicRegistration:
    """@py_class decorator with different calling conventions."""

    def test_bare_decorator(self) -> None:
        @py_class(_unique_key("Bare"))
        class Bare(Object):
            x: int

        info = _get_type_info(Bare)
        assert info is not None
        assert len(info.fields) == 1
        assert info.fields[0].name == "x"

    def test_decorator_with_options(self) -> None:
        @py_class(_unique_key("Opts"), eq=True)
        class Opts(Object):
            x: int

        assert hasattr(Opts, "__eq__")
        assert Opts(x=1) == Opts(x=1)

    def test_auto_type_key(self) -> None:
        @py_class(_unique_key("AutoKey"))
        class AutoKey(Object):
            x: int

        info = _get_type_info(AutoKey)
        assert info.type_key.startswith("testing.")

    def test_explicit_type_key(self) -> None:
        key = _unique_key("ExplicitKey")

        @py_class(key)
        class ExplicitKey(Object):
            x: int

        assert _get_type_info(ExplicitKey).type_key == key

    def test_empty_class(self) -> None:
        @py_class(_unique_key("Empty"))
        class Empty(Object):
            pass

        obj = Empty()
        assert obj is not None

    def test_isinstance_check(self) -> None:
        @py_class(_unique_key("InstCheck"))
        class InstCheck(Object):
            x: int

        obj = InstCheck(x=42)
        assert isinstance(obj, InstCheck)
        assert isinstance(obj, Object)


# ###########################################################################
#  2. Field parsing
# ###########################################################################
class TestFieldParsing:
    """Annotation-to-Field conversion."""

    def test_int_field(self) -> None:
        @py_class(_unique_key("IntFld"))
        class IntFld(Object):
            x: int

        obj = IntFld(x=42)
        assert obj.x == 42

    def test_float_field(self) -> None:
        @py_class(_unique_key("FltFld"))
        class FltFld(Object):
            x: float

        obj = FltFld(x=3.14)
        assert abs(obj.x - 3.14) < 1e-10

    def test_str_field(self) -> None:
        @py_class(_unique_key("StrFld"))
        class StrFld(Object):
            x: str

        obj = StrFld(x="hello")
        assert obj.x == "hello"

    def test_bool_field(self) -> None:
        @py_class(_unique_key("BoolFld"))
        class BoolFld(Object):
            x: bool

        obj = BoolFld(x=True)
        assert obj.x is True

    @_needs_310
    def test_optional_field(self) -> None:
        @py_class(_unique_key("OptFld"))
        class OptFld(Object):
            x: Optional[int]

        obj = OptFld(x=42)
        assert obj.x == 42
        obj2 = OptFld(x=None)
        assert obj2.x is None

    def test_multiple_fields(self) -> None:
        @py_class(_unique_key("Multi"))
        class Multi(Object):
            a: int
            b: float
            c: str

        obj = Multi(a=1, b=2.0, c="three")
        assert obj.a == 1
        assert obj.b == 2.0
        assert obj.c == "three"


# ###########################################################################
#  3. Defaults
# ###########################################################################
class TestDefaults:
    """Default values and default_factory."""

    def test_bare_default(self) -> None:
        @py_class(_unique_key("BareDef"))
        class BareDef(Object):
            x: int
            y: int = 10

        obj = BareDef(x=1)
        assert obj.y == 10

    def test_field_default(self) -> None:
        @py_class(_unique_key("FldDef"))
        class FldDef(Object):
            x: int = field(default=42)

        obj = FldDef()
        assert obj.x == 42

    def test_field_default_factory(self) -> None:
        call_count = 0

        def make_default() -> int:
            nonlocal call_count
            call_count += 1
            return 99

        @py_class(_unique_key("FldFact"))
        class FldFact(Object):
            x: int = field(default_factory=make_default)

        obj1 = FldFact()
        assert obj1.x == 99
        obj2 = FldFact()
        assert obj2.x == 99
        assert call_count == 2

    def test_default_and_factory_mutually_exclusive(self) -> None:
        with pytest.raises(ValueError, match="cannot specify both"):
            field(default=1, default_factory=int)

    def test_non_callable_factory_rejected(self) -> None:
        with pytest.raises(TypeError, match="default_factory must be a callable"):
            field(default_factory=42)  # ty: ignore[invalid-argument-type]

    def test_required_before_optional(self) -> None:
        @py_class(_unique_key("ReqOpt"))
        class ReqOpt(Object):
            a: int
            b: int = 10

        obj = ReqOpt(1)
        assert obj.a == 1
        assert obj.b == 10


# ###########################################################################
#  4. KW_ONLY
# ###########################################################################
class TestKwOnly:
    """Keyword-only field support."""

    def test_kw_only_sentinel(self) -> None:
        @py_class(_unique_key("KWSent"))
        class KWSent(Object):
            a: int
            _: KW_ONLY
            b: int = 10

        obj = KWSent(1, b=20)  # ty: ignore[missing-argument]
        assert obj.a == 1
        assert obj.b == 20
        with pytest.raises(TypeError):
            KWSent(1, 2)  # ty: ignore[invalid-argument-type]

    def test_decorator_level_kw_only(self) -> None:
        @py_class(_unique_key("DecKW"), kw_only=True)
        class DecKW(Object):
            a: int
            b: int = 10

        obj = DecKW(a=1)
        assert obj.a == 1
        assert obj.b == 10
        with pytest.raises(TypeError):
            DecKW(1)  # ty: ignore[missing-argument,too-many-positional-arguments]

    def test_field_level_kw_only_override(self) -> None:
        @py_class(_unique_key("FldKW"))
        class FldKW(Object):
            a: int
            b: int = field(default=10, kw_only=True)

        obj = FldKW(1)
        assert obj.a == 1
        assert obj.b == 10
        with pytest.raises(TypeError):
            FldKW(1, 2)  # b is keyword-only


# ###########################################################################
#  5. ClassVar
# ###########################################################################
class TestClassVar:
    """ClassVar annotations are skipped."""

    def test_classvar_skipped(self) -> None:
        @py_class(_unique_key("CV"))
        class CV(Object):
            x: int
            count: ClassVar[int] = 0

        info = _get_type_info(CV)
        field_names = [f.name for f in info.fields]
        assert "x" in field_names
        assert "count" not in field_names

    def test_classvar_preserved_on_class(self) -> None:
        @py_class(_unique_key("CVPres"))
        class CVPres(Object):
            x: int
            tag: ClassVar[str] = "hello"

        assert CVPres.tag == "hello"


# ###########################################################################
#  6. Init generation
# ###########################################################################
class TestInit:
    """Auto-generated __init__."""

    def test_positional_args(self) -> None:
        @py_class(_unique_key("Pos"))
        class Pos(Object):
            a: int
            b: str

        obj = Pos(1, "hello")
        assert obj.a == 1
        assert obj.b == "hello"

    def test_keyword_args(self) -> None:
        @py_class(_unique_key("Kw"))
        class Kw(Object):
            a: int
            b: str

        obj = Kw(a=1, b="hello")
        assert obj.a == 1
        assert obj.b == "hello"

    def test_init_false_field(self) -> None:
        @py_class(_unique_key("NoInit"))
        class NoInit(Object):
            a: int
            b: int = field(default=99, init=False)

        obj = NoInit(a=1)
        assert obj.a == 1
        assert obj.b == 99

    def test_user_defined_init_preserved(self) -> None:
        @py_class(_unique_key("UserInit"), init=False)
        class UserInit(Object):
            a: int

            def __init__(self, val: int) -> None:
                self.__ffi_init__(val)

        obj = UserInit(42)
        assert obj.a == 42

    def test_required_after_optional_reordered(self) -> None:
        """Required positional fields are reordered before optional ones in __init__."""

        @py_class(_unique_key("ReorderOwn"))
        class ReorderOwn(Object):
            x: int = 0
            y: int  # ty: ignore[dataclass-field-order]

        sig = inspect.signature(ReorderOwn.__init__)
        param_names = [n for n in sig.parameters if n != "self"]
        assert param_names[0] == "y"  # required comes first
        assert param_names[1] == "x"  # optional comes second

        obj = ReorderOwn(y=1)  # ty: ignore[missing-argument]
        assert obj.x == 0
        assert obj.y == 1

    def test_required_after_optional_in_parent(self) -> None:
        """Child required fields are reordered before parent optional fields."""

        @py_class(_unique_key("OptParent"))
        class OptParent(Object):
            x: int
            y: int = 0

        @py_class(_unique_key("ReqChild"))
        class ReqChild(OptParent):
            z: int

        sig = inspect.signature(ReqChild.__init__)
        param_names = [n for n in sig.parameters if n != "self"]
        # required (x, z) before optional (y)
        assert param_names == ["x", "z", "y"]

        obj = ReqChild(x=1, z=3)
        assert obj.x == 1
        assert obj.y == 0
        assert obj.z == 3

    def test_kw_only_exempt_from_reorder(self) -> None:
        """kw_only fields are not reordered with positional fields."""

        @py_class(_unique_key("KwReorder"))
        class KwReorder(Object):
            x: int = 0
            _: KW_ONLY  # ty: ignore[dataclass-field-order]
            y: int  # ty: ignore[dataclass-field-order]

        sig = inspect.signature(KwReorder.__init__)
        params = sig.parameters
        assert params["x"].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
        assert params["y"].kind == inspect.Parameter.KEYWORD_ONLY

        obj = KwReorder(y=1)  # ty: ignore[missing-argument]
        assert obj.x == 0
        assert obj.y == 1

    def test_mixed_positional_and_kw_only_with_defaults(self) -> None:
        """Mixed positional/kw_only fields with defaults produce correct signature."""

        @py_class(_unique_key("MixedSig"))
        class MixedSig(Object):
            a: int = 0
            b: int  # ty: ignore[dataclass-field-order]
            _: KW_ONLY  # ty: ignore[dataclass-field-order]
            c: int = 10
            d: int  # ty: ignore[dataclass-field-order]

        sig = inspect.signature(MixedSig.__init__)
        param_names = [n for n in sig.parameters if n != "self"]
        # positional: b (required) before a (optional); kw_only: d (required) before c (optional)
        assert param_names == ["b", "a", "d", "c"]

        obj = MixedSig(b=2, d=4)  # ty: ignore[missing-argument]
        assert obj.a == 0
        assert obj.b == 2
        assert obj.c == 10
        assert obj.d == 4

    def test_init_false_excluded_from_signature(self) -> None:
        """init=False fields do not appear in __init__ signature."""

        @py_class(_unique_key("InitFalseSig"))
        class InitFalseSig(Object):
            a: int
            b: int = field(default=99, init=False)
            c: str

        sig = inspect.signature(InitFalseSig.__init__)
        param_names = [n for n in sig.parameters if n != "self"]
        assert "b" not in param_names
        assert "a" in param_names
        assert "c" in param_names


# ###########################################################################
#  7. __post_init__
# ###########################################################################
class TestPostInit:
    """__post_init__ support."""

    def test_post_init_called(self) -> None:
        post_init_called = False

        @py_class(_unique_key("PostInit"))
        class PostInit(Object):
            x: int

            def __post_init__(self) -> None:
                nonlocal post_init_called
                post_init_called = True

        PostInit(x=1)
        assert post_init_called

    def test_post_init_sees_field_values(self) -> None:
        @py_class(_unique_key("PostInitVal"))
        class PostInitVal(Object):
            x: int
            y: int = 10

            def __post_init__(self) -> None:
                # Fields should be set before __post_init__ is called
                assert self.x is not None
                assert self.y == 10

        PostInitVal(x=5)


# ###########################################################################
#  8. Repr
# ###########################################################################
class TestRepr:
    """__repr__ generation."""

    def test_repr_generated(self) -> None:
        @py_class(_unique_key("Repr"))
        class Repr(Object):
            x: int
            y: str

        obj = Repr(x=1, y="hello")
        r = repr(obj)
        assert "1" in r
        assert "hello" in r

    def test_repr_disabled(self) -> None:
        @py_class(_unique_key("NoRepr"), repr=False)
        class NoRepr(Object):
            x: int

        obj = NoRepr(x=1)
        # Should use default object repr
        r = repr(obj)
        assert "NoRepr" in r or "object at" in r


# ###########################################################################
#  9. Equality
# ###########################################################################
class TestEquality:
    """__eq__ and __ne__ generation."""

    def test_eq_enabled(self) -> None:
        @py_class(_unique_key("Eq"), eq=True)
        class Eq(Object):
            x: int
            y: str

        assert Eq(x=1, y="a") == Eq(x=1, y="a")
        assert Eq(x=1, y="a") != Eq(x=2, y="a")

    def test_eq_disabled_by_default(self) -> None:
        @py_class(_unique_key("NoEq"))
        class NoEq(Object):
            x: int

        a = NoEq(x=1)
        b = NoEq(x=1)
        # Without eq, identity comparison
        assert a != b
        assert a == a


# ###########################################################################
# 10. Order
# ###########################################################################
class TestOrder:
    """Comparison methods."""

    def test_order_enabled(self) -> None:
        @py_class(_unique_key("Ord"), eq=True, order=True)
        class Ord(Object):
            x: int

        assert Ord(x=1) < Ord(x=2)
        assert Ord(x=2) > Ord(x=1)
        assert Ord(x=1) <= Ord(x=1)
        assert Ord(x=1) >= Ord(x=1)


# ###########################################################################
# 11. Hash
# ###########################################################################
class TestHash:
    """__hash__ generation."""

    def test_unsafe_hash(self) -> None:
        @py_class(_unique_key("Hash"), eq=True, unsafe_hash=True)
        class Hash(Object):
            x: int

        a = Hash(x=1)
        b = Hash(x=1)
        assert hash(a) == hash(b)
        # Can be used in sets
        s = {a, b}
        assert len(s) == 1


# ###########################################################################
# 12. Copy
# ###########################################################################
class TestCopy:
    """__copy__, __deepcopy__, __replace__."""

    def test_shallow_copy(self) -> None:
        @py_class(_unique_key("SCopy"))
        class SCopy(Object):
            x: int

        obj = SCopy(x=42)
        obj2 = copy.copy(obj)
        assert obj2.x == 42

    def test_deep_copy(self) -> None:
        @py_class(_unique_key("DCopy"))
        class DCopy(Object):
            x: int

        obj = DCopy(x=42)
        obj2 = copy.deepcopy(obj)
        assert obj2.x == 42

    def test_replace(self) -> None:
        @py_class(_unique_key("Repl"))
        class Repl(Object):
            x: int
            y: str

        obj = Repl(x=1, y="a")
        obj2 = obj.__replace__(x=2)  # ty: ignore[unresolved-attribute]
        assert obj2.x == 2
        assert obj2.y == "a"


# ###########################################################################
# 13. Inheritance
# ###########################################################################
class TestInheritance:
    """Inheritance between py_class types."""

    def test_child_adds_fields(self) -> None:
        @py_class(_unique_key("Parent"))
        class Parent(Object):
            x: int

        @py_class(_unique_key("Child"))
        class Child(Parent):
            y: str

        obj = Child(x=1, y="hello")
        assert obj.x == 1
        assert obj.y == "hello"

    def test_child_isinstance(self) -> None:
        @py_class(_unique_key("P2"))
        class P2(Object):
            x: int

        @py_class(_unique_key("C2"))
        class C2(P2):
            y: str

        obj = C2(x=1, y="hello")
        assert isinstance(obj, C2)
        assert isinstance(obj, P2)
        assert isinstance(obj, Object)

    def test_three_level_inheritance(self) -> None:
        @py_class(_unique_key("L1"))
        class L1(Object):
            a: int

        @py_class(_unique_key("L2"))
        class L2(L1):
            b: int

        @py_class(_unique_key("L3"))
        class L3(L2):
            c: int

        obj = L3(a=1, b=2, c=3)
        assert obj.a == 1
        assert obj.b == 2
        assert obj.c == 3


# ###########################################################################
# 14. Forward references / deferred resolution
# ###########################################################################
class TestForwardReferences:
    """Deferred annotation resolution for mutual and self-references."""

    @_needs_310
    def test_self_reference(self) -> None:
        @py_class(_unique_key("SelfRef"))
        class SelfRef(Object):
            value: int
            next_node: SelfRef | None

        leaf = SelfRef(value=2, next_node=None)
        head = SelfRef(value=1, next_node=leaf)
        assert head.next_node is not None
        assert head.next_node.value == 2

    @_needs_310
    def test_mutual_reference(self) -> None:
        """Two classes that reference each other."""

        @py_class(_unique_key("Foo"))
        class Foo(Object):
            value: int
            bar: Bar | None

        @py_class(_unique_key("Bar"))
        class Bar(Object):
            value: int
            foo: Foo | None

        bar = Bar(value=2, foo=None)
        foo = Foo(value=1, bar=bar)
        assert foo.bar is not None
        assert foo.bar.value == 2

    @_needs_310
    def test_deferred_resolution_on_instantiation(self) -> None:
        """Forward ref resolved on first instantiation."""

        @py_class(_unique_key("Early"))
        class Early(Object):
            value: int
            ref: Late | None

        # At this point, Early's fields are deferred because Late doesn't exist

        @py_class(_unique_key("Late"))
        class Late(Object):
            value: int

        # Now Early should resolve (either via flush or on instantiation)
        obj = Early(value=1, ref=Late(value=2))
        assert obj.ref is not None
        assert obj.ref.value == 2


# ###########################################################################
# 15. User-defined dunder preservation
# ###########################################################################
class TestDunderPreservation:
    """User-defined dunders are not overwritten."""

    def test_user_repr_preserved(self) -> None:
        @py_class(_unique_key("UserRepr"))
        class UserRepr(Object):
            x: int

            def __repr__(self) -> str:
                return f"Custom({self.x})"

        obj = UserRepr(x=42)
        assert repr(obj) == "Custom(42)"

    def test_user_eq_preserved(self) -> None:
        @py_class(_unique_key("UserEq"), eq=True)
        class UserEq(Object):
            x: int

            def __eq__(self, other: object) -> bool:
                return False

        assert not (UserEq(x=1) == UserEq(x=1))


# ###########################################################################
# 16. field() API
# ###########################################################################
class TestFieldAPI:
    """field() function returns a Field."""

    def test_field_returns_field(self) -> None:
        f = field(default=42)
        assert isinstance(f, Field)
        assert f.default == 42

    def test_field_defaults(self) -> None:
        f = field()
        assert f.init is True
        assert f.repr is True
        assert f.hash is None  # None = follow compare
        assert f.compare is True

    def test_field_kw_only_missing_by_default(self) -> None:
        f = field()
        assert f.kw_only is None

    def test_field_repr_false(self) -> None:
        @py_class(_unique_key("FldRepr"))
        class FldRepr(Object):
            x: int
            y: int = field(default=0, repr=False)

        obj = FldRepr(x=1)
        r = repr(obj)
        assert "1" in r
        # y with repr=False should not appear in repr
        # (depends on C++ ReprPrint implementation respecting the flag)


# ###########################################################################
# 17. Edge cases
# ###########################################################################
class TestEdgeCases:
    """Edge cases and error conditions."""

    def test_no_ffi_parent_raises(self) -> None:
        with pytest.raises(TypeError, match="must inherit from"):

            @py_class(_unique_key("NoPar"))
            class NoPar:  # no Object parent!
                x: int

    def test_only_classvar(self) -> None:
        @py_class(_unique_key("OnlyCV"))
        class OnlyCV(Object):
            count: ClassVar[int] = 0

        obj = OnlyCV()
        assert obj is not None

    def test_mutation_after_creation(self) -> None:
        @py_class(_unique_key("Mut"))
        class Mut(Object):
            x: int

        obj = Mut(x=1)
        obj.x = 42
        assert obj.x == 42


# ###########################################################################
# 18. hash=None tri-state
# ###########################################################################
class TestHashTriState:
    """field(hash=None) means 'follow compare' (native dataclass semantics)."""

    def test_hash_none_follows_compare_true(self) -> None:
        """hash=None + compare=True → field participates in hash."""

        @py_class(_unique_key("HNT"), eq=True, unsafe_hash=True)
        class HNT(Object):
            x: int  # default: compare=True, hash=None → hash=True

        a = HNT(x=1)
        b = HNT(x=1)
        assert hash(a) == hash(b)

    def test_hash_none_follows_compare_false(self) -> None:
        """hash=None + compare=False → field excluded from hash."""

        @py_class(_unique_key("HNF"), eq=True, unsafe_hash=True)
        class HNF(Object):
            x: int
            y: int = field(compare=False)  # hash=None → follows compare=False

        # y doesn't participate in hash, so different y values → same hash
        a = HNF(x=1, y=10)
        b = HNF(x=1, y=20)
        assert hash(a) == hash(b)

    def test_hash_explicit_true_with_compare_true(self) -> None:
        """hash=True + compare=True → field participates in hash."""

        @py_class(_unique_key("HET"), eq=True, unsafe_hash=True)
        class HET(Object):
            x: int = field(hash=True)  # compare=True (default)

        a = HET(x=1)
        b = HET(x=2)
        assert hash(a) != hash(b)

    def test_hash_explicit_false(self) -> None:
        """hash=False excludes field from hashing even with compare=True."""

        @py_class(_unique_key("HEF"), eq=True, unsafe_hash=True)
        class HEF(Object):
            x: int
            y: int = field(hash=False)  # compare=True but hash=False

        a = HEF(x=1, y=10)
        b = HEF(x=1, y=20)
        assert hash(a) == hash(b)


# ###########################################################################
# 19. Deferred resolution + user __init__ / init=False
# ###########################################################################
class TestDeferredInitPreservation:
    """Deferred resolution preserves user-defined __init__ and init=False."""

    @_needs_310
    def test_deferred_with_user_init(self) -> None:
        """User-defined __init__ is preserved after deferred resolution."""

        @py_class(_unique_key("DefUI"))
        class DefUI(Object):
            value: int
            ref: DefUILate | None

            def __init__(self, value: int) -> None:
                self.__ffi_init__(value, None)

        @py_class(_unique_key("DefUILate"))
        class DefUILate(Object):
            x: int

        # DefUI should use the user-defined __init__ (one positional arg)
        obj = DefUI(42)
        assert obj.value == 42
        assert obj.ref is None

    @_needs_310
    def test_deferred_with_init_false(self) -> None:
        """init=False is respected after deferred resolution."""

        @py_class(_unique_key("DefNoInit"), init=False)
        class DefNoInit(Object):
            value: int
            ref: DefNoInitLate | None

            def __init__(self, v: int) -> None:
                self.__ffi_init__(v, None)

        @py_class(_unique_key("DefNoInitLate"))
        class DefNoInitLate(Object):
            x: int

        obj = DefNoInit(10)
        assert obj.value == 10


# ###########################################################################
# 21. order=True requires eq=True
# ###########################################################################
class TestOrderEqValidation:
    """order=True without eq=True is rejected."""

    def test_order_without_eq_raises(self) -> None:
        with pytest.raises(ValueError, match="order=True requires eq=True"):

            @py_class(_unique_key("OrdNoEq"), order=True)
            class OrdNoEq(Object):
                x: int


# ###########################################################################
# 23. Registration rollback on failure
# ###########################################################################
class TestRegistrationRollback:
    """Failed decorations don't permanently poison the type registry."""

    def test_failed_decoration_allows_retry(self) -> None:
        key = _unique_key("Rollback")

        with pytest.raises(Exception):

            @py_class(key)
            class Bad(Object):
                x: object  # unsupported annotation type

        # The type key should be available for reuse
        @py_class(key)
        class Good(Object):
            x: int
            y: int = 0

        assert Good(x=1).y == 0


# ###########################################################################
# 24. User-defined __replace__ preserved
# ###########################################################################
class TestUserReplace:
    """User-defined __replace__ is not overwritten by py_class."""

    def test_user_replace_preserved(self) -> None:
        @py_class(_unique_key("UserRepl"))
        class UserRepl(Object):
            x: int

            def __replace__(self, **changes: object) -> str:
                return "custom"

        obj = UserRepl(x=1)
        assert obj.__replace__(x=2) == "custom"


# ###########################################################################
# 25. default_factory=None raises
# ###########################################################################
class TestDefaultFactoryNone:
    """Explicit default_factory=None matches stdlib semantics (raises)."""

    def test_explicit_none_raises(self) -> None:
        with pytest.raises(TypeError, match="default_factory must be a callable"):
            field(default_factory=None)


# ###########################################################################
# 26. Adversarial edge cases for init reordering
# ###########################################################################
class TestInitReorderingAdversarial:
    """Tricky scenarios that catch bugs in naive init-signature generation."""

    def test_positional_call_maps_to_required_not_declared_order(self) -> None:
        """Positional arg 1 maps to the first *required* field, not the first declared."""

        @py_class(_unique_key("PosMap"))
        class PosMap(Object):
            x: int = 0  # optional, declared first
            y: int  # ty: ignore[dataclass-field-order]  # required, declared second

        # Positional call: first arg is y (required), not x (optional)
        obj = PosMap(42)  # ty: ignore[missing-argument]
        assert obj.y == 42
        assert obj.x == 0

    def test_relative_order_preserved_within_groups(self) -> None:
        """Within required and optional groups, declaration order is preserved."""

        @py_class(_unique_key("RelOrder"))
        class RelOrder(Object):
            a: int = 0
            b: int  # ty: ignore[dataclass-field-order]
            c: int = 1
            d: int  # ty: ignore[dataclass-field-order]

        sig = inspect.signature(RelOrder.__init__)
        param_names = [n for n in sig.parameters if n != "self"]
        # required: b, d (declaration order); optional: a, c (declaration order)
        assert param_names == ["b", "d", "a", "c"]

        obj = RelOrder(10, 20)  # ty: ignore[missing-argument]
        assert obj.b == 10
        assert obj.d == 20
        assert obj.a == 0
        assert obj.c == 1

    def test_default_factory_counts_as_optional(self) -> None:
        """default_factory makes a field optional for reordering purposes."""

        @py_class(_unique_key("DFReorder"))
        class DFReorder(Object):
            items: str = field(default_factory=lambda: "hello")
            count: int  # ty: ignore[dataclass-field-order]

        sig = inspect.signature(DFReorder.__init__)
        param_names = [n for n in sig.parameters if n != "self"]
        assert param_names[0] == "count"  # required first
        assert param_names[1] == "items"  # optional (factory) second

        obj = DFReorder(count=5)
        assert obj.count == 5
        assert obj.items == "hello"

    def test_three_level_hierarchy_reorder(self) -> None:
        """Required fields from all levels come before optional fields from all levels."""

        @py_class(_unique_key("G1"))
        class G1(Object):
            a: int  # required

        @py_class(_unique_key("P1"))
        class P1(G1):
            b: int = 0  # optional

        @py_class(_unique_key("C1"))
        class C1(P1):
            c: int  # required

        sig = inspect.signature(C1.__init__)
        param_names = [n for n in sig.parameters if n != "self"]
        # required (a, c) before optional (b)
        assert param_names == ["a", "c", "b"]

        obj = C1(a=1, c=3)
        assert obj.a == 1
        assert obj.b == 0
        assert obj.c == 3

    def test_kw_only_false_overrides_sentinel(self) -> None:
        """kw_only=False on a field after KW_ONLY sentinel makes it positional."""

        @py_class(_unique_key("KwOverride"))
        class KwOverride(Object):
            _: KW_ONLY
            a: int  # kw_only (inherits sentinel)
            b: int = field(kw_only=False)  # positional (explicit override)

        sig = inspect.signature(KwOverride.__init__)
        assert sig.parameters["a"].kind == inspect.Parameter.KEYWORD_ONLY
        assert sig.parameters["b"].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD

        obj = KwOverride(42, a=1)  # ty: ignore[missing-argument,invalid-argument-type]
        assert obj.b == 42
        assert obj.a == 1

    def test_init_false_field_gets_default(self) -> None:
        """init=False field with default is set to default, not left uninitialized."""

        @py_class(_unique_key("InitFalseDef"))
        class InitFalseDef(Object):
            visible: int
            hidden: str = field(default="secret", init=False)

        obj = InitFalseDef(visible=1)
        assert obj.hidden == "secret"

    def test_post_init_sees_reordered_fields(self) -> None:
        """__post_init__ sees correct values even when __init__ reorders fields."""
        seen: Dict[str, int] = {}

        @py_class(_unique_key("PostReorder"))
        class PostReorder(Object):
            x: int = 0
            y: int  # ty: ignore[dataclass-field-order]

            def __post_init__(self) -> None:
                seen["x"] = self.x
                seen["y"] = self.y

        PostReorder(y=10, x=20)
        assert seen == {"x": 20, "y": 10}

    @_needs_310
    def test_deferred_forward_ref_with_reordering(self) -> None:
        """Deferred forward-reference resolution still produces correct reordering."""

        @py_class(_unique_key("DeferReorder"))
        class DeferReorder(Object):
            opt: DeferLate | None = None
            req: int  # ty: ignore[dataclass-field-order]

        @py_class(_unique_key("DeferLate"))
        class DeferLate(Object):
            x: int

        sig = inspect.signature(DeferReorder.__init__)
        param_names = [n for n in sig.parameters if n != "self"]
        assert param_names[0] == "req"
        assert param_names[1] == "opt"

        obj = DeferReorder(req=1)
        assert obj.req == 1
        assert obj.opt is None

    def test_all_optional_preserves_declaration_order(self) -> None:
        """When all fields are optional, declaration order is preserved."""

        @py_class(_unique_key("AllOpt"))
        class AllOpt(Object):
            c: int = 3
            a: int = 1
            b: int = 2

        sig = inspect.signature(AllOpt.__init__)
        param_names = [n for n in sig.parameters if n != "self"]
        assert param_names == ["c", "a", "b"]

        obj = AllOpt()
        assert obj.c == 3
        assert obj.a == 1
        assert obj.b == 2

    def test_all_required_preserves_declaration_order(self) -> None:
        """When all fields are required, declaration order is preserved."""

        @py_class(_unique_key("AllReq"))
        class AllReq(Object):
            c: int
            a: int
            b: int

        sig = inspect.signature(AllReq.__init__)
        param_names = [n for n in sig.parameters if n != "self"]
        assert param_names == ["c", "a", "b"]

        obj = AllReq(10, 20, 30)
        assert obj.c == 10
        assert obj.a == 20
        assert obj.b == 30


# ###########################################################################
#  1. Registration
# ###########################################################################
class TestRegisterPyClass:
    """Low-level _register_py_class: type allocation, ancestors, field lifecycle."""

    def test_basic_registration(self) -> None:
        type_key = _unique_key_ff("RegBasic")
        parent_info = core._type_cls_to_type_info(core.Object)
        assert parent_info is not None
        cls = type("RegBasic", (core.Object,), {"__slots__": ()})
        info = core._register_py_class(parent_info, type_key, cls)
        assert info is not None
        assert info.type_key == type_key

    def test_type_index_allocated(self) -> None:
        type_key = _unique_key_ff("RegIndex")
        parent_info = core._type_cls_to_type_info(core.Object)
        assert parent_info is not None
        cls = type("RegIndex", (core.Object,), {"__slots__": ()})
        info = core._register_py_class(parent_info, type_key, cls)
        assert isinstance(info.type_index, int)
        assert info.type_index > 0

    def test_ancestors_include_parent(self) -> None:
        parent_info = core._type_cls_to_type_info(core.Object)
        assert parent_info is not None
        type_key = _unique_key_ff("RegAncestors")
        cls = type("RegAncestors", (core.Object,), {"__slots__": ()})
        info = core._register_py_class(parent_info, type_key, cls)
        assert parent_info.type_index in info.type_ancestors

    def test_parent_type_info_set(self) -> None:
        parent_info = core._type_cls_to_type_info(core.Object)
        assert parent_info is not None
        type_key = _unique_key_ff("RegParent")
        cls = type("RegParent", (core.Object,), {"__slots__": ()})
        info = core._register_py_class(parent_info, type_key, cls)
        assert info.parent_type_info is parent_info

    def test_initial_fields_none_and_methods_empty(self) -> None:
        parent_info = core._type_cls_to_type_info(core.Object)
        assert parent_info is not None
        type_key = _unique_key_ff("RegEmpty")
        cls = type("RegEmpty", (core.Object,), {"__slots__": ()})
        info = core._register_py_class(parent_info, type_key, cls)
        assert info.fields is None
        assert len(info.methods) == 0

    def test_two_registrations_different_indices(self) -> None:
        parent_info = core._type_cls_to_type_info(core.Object)
        assert parent_info is not None
        cls1 = type("RegDiff1", (core.Object,), {"__slots__": ()})
        cls2 = type("RegDiff2", (core.Object,), {"__slots__": ()})
        info1 = core._register_py_class(parent_info, _unique_key_ff("RegDiff1"), cls1)
        info2 = core._register_py_class(parent_info, _unique_key_ff("RegDiff2"), cls2)
        assert info1.type_index != info2.type_index

    def test_fields_none_before_registration(self) -> None:
        parent_info = core._type_cls_to_type_info(core.Object)
        assert parent_info is not None
        cls = type("Pending", (core.Object,), {"__slots__": ()})
        info = core._register_py_class(parent_info, _unique_key_ff("Pending"), cls)
        assert info.fields is None

    def test_register_fields_is_instance_method(self) -> None:
        parent_info = core._type_cls_to_type_info(core.Object)
        assert parent_info is not None
        cls = type("PendingM", (core.Object,), {"__slots__": ()})
        info = core._register_py_class(parent_info, _unique_key_ff("PendingM"), cls)
        assert hasattr(info, "_register_fields")

    def test_duplicate_type_key_raises(self) -> None:
        parent_info = core._type_cls_to_type_info(core.Object)
        assert parent_info is not None
        type_key = _unique_key_ff("Dup")
        cls1 = type("Dup1", (core.Object,), {"__slots__": ()})
        core._register_py_class(parent_info, type_key, cls1)
        cls2 = type("Dup2", (core.Object,), {"__slots__": ()})
        with pytest.raises((RuntimeError, ValueError)):
            core._register_py_class(parent_info, type_key, cls2)

    def test_duplicate_type_key_preserves_original(self) -> None:
        """After rejected duplicate, original entry is intact."""
        parent_info = core._type_cls_to_type_info(core.Object)
        assert parent_info is not None
        type_key = _unique_key_ff("DupPreserve")
        cls1 = type("DupPreserve1", (core.Object,), {"__slots__": ()})
        info1 = core._register_py_class(parent_info, type_key, cls1)
        info1._register_fields([Field(name="x", ty=TypeSchema("int"))])
        setattr(cls1, "__tvm_ffi_type_info__", info1)
        _add_class_attrs(cls1, info1)

        cls2 = type("DupPreserve2", (core.Object,), {"__slots__": ()})
        with pytest.raises((RuntimeError, ValueError)):
            core._register_py_class(parent_info, type_key, cls2)

        reloaded = core._lookup_or_register_type_info_from_type_key(type_key)
        assert reloaded.type_cls is cls1
        assert [f.name for f in reloaded.fields] == ["x"]


# ###########################################################################
#  2. Field Registration
# ###########################################################################
class TestFieldRegistration:
    """Low-level _register_fields: field types, metadata, offsets."""

    def test_int_field_registered(self) -> None:
        cls = _make_type(
            "FldInt",
            [Field(name="x", ty=TypeSchema("int"), default=MISSING)],
        )
        info = getattr(cls, "__tvm_ffi_type_info__")
        assert len(info.fields) == 1
        assert info.fields[0].name == "x"

    def test_float_field_registered(self) -> None:
        cls = _make_type(
            "FldFloat",
            [Field(name="val", ty=TypeSchema("float"), default=0.0)],
        )
        info = getattr(cls, "__tvm_ffi_type_info__")
        assert info.fields[0].name == "val"

    def test_str_field_registered(self) -> None:
        cls = _make_type(
            "FldStr",
            [Field(name="s", ty=TypeSchema("str"), default="hello")],
        )
        info = getattr(cls, "__tvm_ffi_type_info__")
        assert info.fields[0].name == "s"

    def test_bool_field_registered(self) -> None:
        cls = _make_type(
            "FldBool",
            [Field(name="flag", ty=TypeSchema("bool"), default=False)],
        )
        info = getattr(cls, "__tvm_ffi_type_info__")
        assert info.fields[0].name == "flag"

    def test_multiple_fields_count(self) -> None:
        cls = _make_type(
            "FldMulti",
            [
                Field(name="a", ty=TypeSchema("int"), default=MISSING),
                Field(name="b", ty=TypeSchema("float"), default=0.0),
                Field(name="c", ty=TypeSchema("str"), default="x"),
            ],
        )
        info = getattr(cls, "__tvm_ffi_type_info__")
        assert len(info.fields) == 3
        assert [f.name for f in info.fields] == ["a", "b", "c"]

    def test_field_offsets_increasing(self) -> None:
        cls = _make_type(
            "FldOff",
            [
                Field(name="a", ty=TypeSchema("int"), default=MISSING),
                Field(name="b", ty=TypeSchema("float"), default=MISSING),
                Field(name="c", ty=TypeSchema("str"), default=MISSING),
            ],
        )
        info = getattr(cls, "__tvm_ffi_type_info__")
        offsets = [f.offset for f in info.fields]
        for i in range(1, len(offsets)):
            assert offsets[i] > offsets[i - 1], f"Field offsets not increasing: {offsets}"

    def test_ffi_init_method_registered(self) -> None:
        cls = _make_type(
            "FldInit",
            [Field(name="x", ty=TypeSchema("int"), default=MISSING)],
        )
        info = getattr(cls, "__tvm_ffi_type_info__")
        assert "__ffi_init__" in [m.name for m in info.methods]

    def test_field_metadata_repr_flag(self) -> None:
        cls = _make_type(
            "FldReprMeta",
            [
                Field(
                    name="visible",
                    ty=TypeSchema("int"),
                    default=MISSING,
                    repr=True,
                ),
                Field(
                    name="hidden",
                    ty=TypeSchema("int"),
                    default=0,
                    repr=False,
                ),
            ],
        )
        info = getattr(cls, "__tvm_ffi_type_info__")
        assert len(info.fields) == 2


# ###########################################################################
#  3. Field Descriptor
# ###########################################################################
class TestFieldDescriptor:
    """Field class: validation, defaults, default_factory checks."""

    def test_compare_default_is_false(self) -> None:
        f = Field(name="x", ty=TypeSchema("int"))
        assert f.compare is False

    def test_default_and_factory_mutually_exclusive(self) -> None:
        with pytest.raises(ValueError):
            Field(name="x", ty=TypeSchema("int"), default=0, default_factory=lambda: 0)

    def test_factory_must_be_callable(self) -> None:
        with pytest.raises(TypeError, match="callable"):
            Field(name="x", ty=TypeSchema("int"), default_factory=0)  # ty: ignore[invalid-argument-type]

    def test_non_callable_factory_rejected(self) -> None:
        with pytest.raises(TypeError, match="callable"):
            Field(name="x", ty=TypeSchema("int"), default_factory="not_callable")  # ty: ignore[invalid-argument-type]


# ###########################################################################
#  4. Construction
# ###########################################################################
class TestConstruction:
    """Low-level __init__ via _make_type: positional/keyword args, defaults, errors."""

    def test_keyword_args(self) -> None:
        Cls = _make_type(
            "ConKw",
            [
                Field(name="x", ty=TypeSchema("int"), default=MISSING),
                Field(name="y", ty=TypeSchema("float"), default=MISSING),
            ],
        )
        obj = Cls(x=42, y=3.14)
        assert obj.x == 42
        assert obj.y == pytest.approx(3.14)

    def test_positional_args(self) -> None:
        Cls = _make_type(
            "ConPos",
            [
                Field(name="x", ty=TypeSchema("int"), default=MISSING),
                Field(name="y", ty=TypeSchema("float"), default=MISSING),
            ],
        )
        obj = Cls(10, 2.5)
        assert obj.x == 10
        assert obj.y == pytest.approx(2.5)

    def test_mixed_positional_and_keyword(self) -> None:
        Cls = _make_type(
            "ConMixed",
            [
                Field(name="x", ty=TypeSchema("int"), default=MISSING),
                Field(name="y", ty=TypeSchema("float"), default=MISSING),
            ],
        )
        obj = Cls(7, y=1.5)
        assert obj.x == 7
        assert obj.y == pytest.approx(1.5)

    def test_default_value_int(self) -> None:
        Cls = _make_type(
            "ConDefInt",
            [Field(name="x", ty=TypeSchema("int"), default=99)],
        )
        assert Cls().x == 99

    def test_default_value_float(self) -> None:
        Cls = _make_type(
            "ConDefFloat",
            [Field(name="x", ty=TypeSchema("float"), default=1.5)],
        )
        assert Cls().x == pytest.approx(1.5)

    def test_default_value_str(self) -> None:
        Cls = _make_type(
            "ConDefStr",
            [Field(name="s", ty=TypeSchema("str"), default="hello")],
        )
        assert Cls().s == "hello"

    def test_override_default(self) -> None:
        Cls = _make_type(
            "ConOverride",
            [Field(name="x", ty=TypeSchema("int"), default=0)],
        )
        assert Cls(x=42).x == 42

    def test_required_and_optional_together(self) -> None:
        Cls = _make_type(
            "ConReqOpt",
            [
                Field(name="required", ty=TypeSchema("int"), default=MISSING),
                Field(name="optional", ty=TypeSchema("float"), default=0.0),
            ],
        )
        obj = Cls(required=5)
        assert obj.required == 5
        assert obj.optional == pytest.approx(0.0)

    def test_missing_required_raises(self) -> None:
        Cls = _make_type(
            "ConMissing",
            [Field(name="x", ty=TypeSchema("int"), default=MISSING)],
        )
        with pytest.raises(TypeError):
            Cls()

    def test_extra_kwarg_raises(self) -> None:
        Cls = _make_type(
            "ConExtra",
            [Field(name="x", ty=TypeSchema("int"), default=MISSING)],
        )
        with pytest.raises(TypeError):
            Cls(x=1, bogus=2)

    def test_str_field_construction(self) -> None:
        Cls = _make_type(
            "ConStr",
            [Field(name="name", ty=TypeSchema("str"), default=MISSING)],
        )
        assert Cls(name="world").name == "world"

    def test_bool_field_construction(self) -> None:
        Cls = _make_type(
            "ConBool",
            [Field(name="flag", ty=TypeSchema("bool"), default=MISSING)],
        )
        assert Cls(flag=True).flag is True
        assert Cls(flag=False).flag is False

    def test_kw_only_field(self) -> None:
        Cls = _make_type(
            "ConKwOnly",
            [
                Field(name="x", ty=TypeSchema("int"), default=MISSING),
                Field(
                    name="y",
                    ty=TypeSchema("int"),
                    default=MISSING,
                    kw_only=True,
                ),
            ],
        )
        obj = Cls(1, y=2)
        assert obj.x == 1
        assert obj.y == 2

    def test_kw_only_rejects_positional(self) -> None:
        Cls = _make_type(
            "ConKwOnlyReject",
            [
                Field(name="x", ty=TypeSchema("int"), default=MISSING),
                Field(
                    name="y",
                    ty=TypeSchema("int"),
                    default=MISSING,
                    kw_only=True,
                ),
            ],
        )
        with pytest.raises(TypeError):
            Cls(1, 2)

    def test_isinstance_check(self) -> None:
        Cls = _make_type(
            "ConIsInstance",
            [Field(name="x", ty=TypeSchema("int"), default=MISSING)],
        )
        obj = Cls(x=1)
        assert isinstance(obj, Cls)
        assert isinstance(obj, core.Object)


# ###########################################################################
#  5. Getter / Setter
# ###########################################################################
class TestGetterSetter:
    """Field access: get/set POD, str, bool, mutation isolation."""

    def test_get_int(self) -> None:
        Cls = _make_type(
            "GSInt",
            [Field(name="x", ty=TypeSchema("int"), default=MISSING)],
        )
        assert Cls(x=42).x == 42

    def test_set_int(self) -> None:
        Cls = _make_type(
            "GSSetInt",
            [Field(name="x", ty=TypeSchema("int"), default=MISSING)],
        )
        obj = Cls(x=1)
        obj.x = 100
        assert obj.x == 100

    def test_get_float(self) -> None:
        Cls = _make_type(
            "GSFloat",
            [Field(name="val", ty=TypeSchema("float"), default=MISSING)],
        )
        assert Cls(val=3.14).val == pytest.approx(3.14)

    def test_set_float(self) -> None:
        Cls = _make_type(
            "GSSetFloat",
            [Field(name="val", ty=TypeSchema("float"), default=MISSING)],
        )
        obj = Cls(val=1.0)
        obj.val = 2.718
        assert obj.val == pytest.approx(2.718)

    def test_get_str(self) -> None:
        Cls = _make_type(
            "GSStr",
            [Field(name="s", ty=TypeSchema("str"), default=MISSING)],
        )
        assert Cls(s="hello").s == "hello"

    def test_set_str(self) -> None:
        Cls = _make_type(
            "GSSetStr",
            [Field(name="s", ty=TypeSchema("str"), default=MISSING)],
        )
        obj = Cls(s="hello")
        obj.s = "world"
        assert obj.s == "world"

    def test_get_bool(self) -> None:
        Cls = _make_type(
            "GSBool",
            [Field(name="flag", ty=TypeSchema("bool"), default=MISSING)],
        )
        assert Cls(flag=True).flag is True

    def test_set_bool(self) -> None:
        Cls = _make_type(
            "GSSetBool",
            [Field(name="flag", ty=TypeSchema("bool"), default=MISSING)],
        )
        obj = Cls(flag=True)
        obj.flag = False
        assert obj.flag is False

    def test_mutation_isolated(self) -> None:
        Cls = _make_type(
            "GSIsolate",
            [Field(name="x", ty=TypeSchema("int"), default=MISSING)],
        )
        a = Cls(x=1)
        b = Cls(x=1)
        a.x = 99
        assert a.x == 99
        assert b.x == 1

    def test_multiple_fields_mutation(self) -> None:
        Cls = _make_type(
            "GSMultiMut",
            [
                Field(name="a", ty=TypeSchema("int"), default=MISSING),
                Field(name="b", ty=TypeSchema("float"), default=MISSING),
                Field(name="c", ty=TypeSchema("str"), default=MISSING),
            ],
        )
        obj = Cls(a=1, b=2.0, c="x")
        obj.a = 10
        obj.b = 20.0
        obj.c = "y"
        assert obj.a == 10
        assert obj.b == pytest.approx(20.0)
        assert obj.c == "y"

    def test_set_array_field(self) -> None:
        Cls = _make_type(
            "GSSetArr",
            [
                Field(
                    name="arr",
                    ty=TypeSchema("Array", (TypeSchema("int"),)),
                    default=MISSING,
                ),
            ],
        )
        obj = Cls(arr=[1])
        obj.arr = tvm_ffi.Array([4, 5, 6])
        assert len(obj.arr) == 3
        assert obj.arr[0] == 4


# ###########################################################################
#  6. ObjectRef Fields
# ###########################################################################
class TestObjectRefFields:
    """Fields holding ObjectRef types: Array, custom objects."""

    def test_array_field(self) -> None:
        Cls = _make_type(
            "ObjArr",
            [
                Field(
                    name="arr",
                    ty=TypeSchema("Array", (TypeSchema("int"),)),
                    default=MISSING,
                ),
            ],
        )
        obj = Cls(arr=tvm_ffi.Array([1, 2, 3]))
        assert len(obj.arr) == 3
        assert obj.arr[0] == 1
        assert obj.arr[2] == 3

    def test_array_field_from_list(self) -> None:
        Cls = _make_type(
            "ObjArrList",
            [
                Field(
                    name="arr",
                    ty=TypeSchema("Array", (TypeSchema("int"),)),
                    default=MISSING,
                ),
            ],
        )
        obj = Cls(arr=[10, 20, 30])
        assert len(obj.arr) == 3
        assert obj.arr[1] == 20

    def test_nested_object_field(self) -> None:
        Inner = _make_type(
            "ObjInner",
            [Field(name="val", ty=TypeSchema("int"), default=MISSING)],
        )
        inner_info = getattr(Inner, "__tvm_ffi_type_info__")
        inner_schema = TypeSchema(inner_info.type_key, origin_type_index=inner_info.type_index)
        Outer = _make_type(
            "ObjOuter",
            [Field(name="child", ty=inner_schema, default=MISSING)],
        )
        assert Outer(child=Inner(val=42)).child.val == 42


# ###########################################################################
#  7. Optional Fields
# ###########################################################################
class TestOptionalFields:
    """Optional/Union fields: None and non-None values."""

    def test_optional_int_with_value(self) -> None:
        Cls = _make_type(
            "OptIntV",
            [
                Field(
                    name="x",
                    ty=TypeSchema("Optional", (TypeSchema("int"),)),
                    default=MISSING,
                ),
            ],
        )
        assert Cls(x=42).x == 42

    def test_optional_int_with_none(self) -> None:
        Cls = _make_type(
            "OptIntN",
            [
                Field(
                    name="x",
                    ty=TypeSchema("Optional", (TypeSchema("int"),)),
                    default=None,
                ),
            ],
        )
        assert Cls().x is None

    def test_optional_str_with_value(self) -> None:
        Cls = _make_type(
            "OptStrV",
            [
                Field(
                    name="s",
                    ty=TypeSchema("Optional", (TypeSchema("str"),)),
                    default=MISSING,
                ),
            ],
        )
        assert Cls(s="hello").s == "hello"

    def test_optional_str_with_none(self) -> None:
        Cls = _make_type(
            "OptStrN",
            [
                Field(
                    name="s",
                    ty=TypeSchema("Optional", (TypeSchema("str"),)),
                    default=None,
                ),
            ],
        )
        assert Cls().s is None

    def test_optional_set_to_none(self) -> None:
        Cls = _make_type(
            "OptSet",
            [
                Field(
                    name="x",
                    ty=TypeSchema("Optional", (TypeSchema("int"),)),
                    default=MISSING,
                ),
            ],
        )
        obj = Cls(x=42)
        obj.x = None
        assert obj.x is None

    def test_optional_set_back_to_value(self) -> None:
        Cls = _make_type(
            "OptBack",
            [
                Field(
                    name="x",
                    ty=TypeSchema("Optional", (TypeSchema("int"),)),
                    default=None,
                ),
            ],
        )
        obj = Cls()
        obj.x = 99
        assert obj.x == 99

    def test_all_optional_fields_default_none(self) -> None:
        Cls = _make_type(
            "AllOpt",
            [
                Field(
                    name="a",
                    ty=TypeSchema("Optional", (TypeSchema("int"),)),
                    default=None,
                ),
                Field(
                    name="b",
                    ty=TypeSchema("Optional", (TypeSchema("str"),)),
                    default=None,
                ),
                Field(
                    name="c",
                    ty=TypeSchema("Optional", (TypeSchema("float"),)),
                    default=None,
                ),
            ],
        )
        obj = Cls()
        assert obj.a is None
        assert obj.b is None
        assert obj.c is None
        obj.a = 42
        assert obj.a == 42
        obj.b = "hello"
        assert obj.b == "hello"

    def test_optional_object_none_and_back(self) -> None:
        Cls = _make_type(
            "OptObjRound",
            [
                Field(
                    name="ref",
                    ty=TypeSchema("Optional", (TypeSchema("Object"),)),
                    default=None,
                ),
            ],
        )
        obj = Cls()
        assert obj.ref is None
        obj.ref = tvm_ffi.Array([1])
        assert len(obj.ref) == 1
        obj.ref = None
        assert obj.ref is None

    def test_union_int_str(self) -> None:
        """Union[int, str] field should accept both types."""
        Cls = _make_type(
            "UnionIntStr",
            [
                Field(
                    name="val",
                    ty=TypeSchema("Union", (TypeSchema("int"), TypeSchema("str"))),
                    default=MISSING,
                ),
            ],
        )
        obj = Cls(val=42)
        assert obj.val == 42
        obj.val = "hello"
        assert obj.val == "hello"

    def test_union_int_str_rejects_float(self) -> None:
        """Union[int, str] should reject float (not in union)."""
        Cls = _make_type(
            "UnionReject",
            [
                Field(
                    name="val",
                    ty=TypeSchema("Union", (TypeSchema("int"), TypeSchema("str"))),
                    default=MISSING,
                ),
            ],
        )
        obj = Cls(val=1)
        with pytest.raises((TypeError, RuntimeError)):
            obj.val = 3.14

    def test_optional_union(self) -> None:
        """Optional[Union[int, str]] should accept None, int, and str."""
        Cls = _make_type(
            "OptUnion",
            [
                Field(
                    name="val",
                    ty=TypeSchema(
                        "Optional",
                        (TypeSchema("Union", (TypeSchema("int"), TypeSchema("str"))),),
                    ),
                    default=None,
                ),
            ],
        )
        obj = Cls()
        assert obj.val is None
        obj.val = 42
        assert obj.val == 42
        obj.val = "hi"
        assert obj.val == "hi"
        obj.val = None
        assert obj.val is None


# ###########################################################################
#  8. Any Fields
# ###########################################################################
class TestAnyField:
    """Fields with TypeSchema('Any'): hold any value type."""

    def test_any_holds_int(self) -> None:
        Cls = _make_type(
            "AnyI",
            [Field(name="val", ty=TypeSchema("Any"), default=None)],
        )
        assert Cls(val=42).val == 42

    def test_any_holds_str(self) -> None:
        Cls = _make_type(
            "AnyS",
            [Field(name="val", ty=TypeSchema("Any"), default=None)],
        )
        assert Cls(val="hello").val == "hello"

    def test_any_holds_none(self) -> None:
        Cls = _make_type(
            "AnyN",
            [Field(name="val", ty=TypeSchema("Any"), default=None)],
        )
        assert Cls().val is None

    def test_any_holds_object(self) -> None:
        Cls = _make_type(
            "AnyObj",
            [Field(name="val", ty=TypeSchema("Any"), default=None)],
        )
        arr = tvm_ffi.Array([1, 2])
        assert len(Cls(val=arr).val) == 2

    def test_any_type_change(self) -> None:
        Cls = _make_type(
            "AnyChg",
            [Field(name="val", ty=TypeSchema("Any"), default=None)],
        )
        obj = Cls()
        obj.val = 42
        assert obj.val == 42
        obj.val = "hello"
        assert obj.val == "hello"
        obj.val = None
        assert obj.val is None
        obj.val = tvm_ffi.Array([1])
        assert len(obj.val) == 1


# ###########################################################################
#  9. Default Factory
# ###########################################################################
class TestDefaultFactory:
    """default_factory support: fresh instances, override, various types."""

    def test_factory_produces_fresh_instances(self) -> None:
        Cls = _make_type(
            "DFFresh",
            [
                Field(
                    name="data",
                    ty=TypeSchema("Array", (TypeSchema("int"),)),
                    default_factory=lambda: tvm_ffi.Array([]),
                ),
            ],
        )
        a = Cls()
        b = Cls()
        assert not a.data.same_as(b.data)

    def test_factory_with_content(self) -> None:
        Cls = _make_type(
            "DFContent",
            [
                Field(
                    name="items",
                    ty=TypeSchema("Array", (TypeSchema("int"),)),
                    default_factory=lambda: tvm_ffi.Array([1, 2, 3]),
                ),
            ],
        )
        obj = Cls()
        assert len(obj.items) == 3
        assert obj.items[0] == 1

    def test_factory_override(self) -> None:
        Cls = _make_type(
            "DFOverride",
            [Field(name="x", ty=TypeSchema("int"), default_factory=lambda: 42)],
        )
        assert Cls(x=99).x == 99

    def test_factory_str(self) -> None:
        Cls = _make_type(
            "DFStr",
            [Field(name="s", ty=TypeSchema("str"), default_factory=lambda: "generated")],
        )
        assert Cls().s == "generated"


# ###########################################################################
#  10. Repr
# ###########################################################################
class TestFieldRepr:
    """Low-level repr via _make_type: field values, repr=False exclusion."""

    def test_repr_includes_fields(self) -> None:
        Cls = _make_type(
            "ReprBasic",
            [
                Field(name="x", ty=TypeSchema("int"), default=MISSING),
                Field(name="y", ty=TypeSchema("float"), default=0.0),
            ],
        )
        r = ReprPrint(Cls(x=42, y=3.14))
        assert "x=42" in r
        assert "y=3.14" in r

    def test_repr_str_field(self) -> None:
        Cls = _make_type(
            "ReprStr",
            [Field(name="name", ty=TypeSchema("str"), default=MISSING)],
        )
        assert '"hello"' in ReprPrint(Cls(name="hello"))

    def test_repr_bool_field(self) -> None:
        Cls = _make_type(
            "ReprBool",
            [Field(name="flag", ty=TypeSchema("bool"), default=MISSING)],
        )
        assert "flag=True" in ReprPrint(Cls(flag=True))

    def test_repr_false_excluded(self) -> None:
        Cls = _make_type(
            "ReprExcl",
            [
                Field(name="visible", ty=TypeSchema("int"), default=MISSING),
                Field(
                    name="hidden",
                    ty=TypeSchema("int"),
                    default=0,
                    repr=False,
                ),
            ],
        )
        r = ReprPrint(Cls(visible=42))
        assert "visible=42" in r
        assert "hidden" not in r

    def test_python_repr_delegates(self) -> None:
        Cls = _make_type(
            "ReprDeleg",
            [Field(name="x", ty=TypeSchema("int"), default=MISSING)],
        )
        assert "x=7" in repr(Cls(x=7))

    def test_repr_contains_type_key(self) -> None:
        Cls = _make_type(
            "ReprKey",
            [Field(name="x", ty=TypeSchema("int"), default=MISSING)],
        )
        info = getattr(Cls, "__tvm_ffi_type_info__")
        assert info.type_key in ReprPrint(Cls(x=1))

    def test_repr_optional_none(self) -> None:
        Cls = _make_type(
            "ReprOptNone",
            [
                Field(
                    name="x",
                    ty=TypeSchema("Optional", (TypeSchema("int"),)),
                    default=None,
                ),
            ],
        )
        r = ReprPrint(Cls())
        assert isinstance(r, str)

    def test_repr_array_field(self) -> None:
        Cls = _make_type(
            "ReprArr",
            [
                Field(
                    name="items",
                    ty=TypeSchema("Array", (TypeSchema("int"),)),
                    default=MISSING,
                ),
            ],
        )
        r = ReprPrint(Cls(items=[1, 2, 3]))
        assert isinstance(r, str)


# ###########################################################################
#  11. Hash
# ###########################################################################
class TestFieldHash:
    """Low-level hash via _make_type: equal objects same hash, hash=False exclusion."""

    def test_equal_objects_same_hash(self) -> None:
        Cls = _make_type(
            "HashEq",
            [
                Field(
                    name="x",
                    ty=TypeSchema("int"),
                    default=MISSING,
                    compare=True,
                ),
                Field(
                    name="y",
                    ty=TypeSchema("float"),
                    default=MISSING,
                    compare=True,
                ),
            ],
            eq=True,
            unsafe_hash=True,
        )
        assert RecursiveHash(Cls(x=1, y=2.0)) == RecursiveHash(Cls(x=1, y=2.0))

    def test_different_objects_different_hash(self) -> None:
        Cls = _make_type(
            "HashDiff",
            [
                Field(
                    name="x",
                    ty=TypeSchema("int"),
                    default=MISSING,
                    compare=True,
                ),
            ],
            eq=True,
            unsafe_hash=True,
        )
        assert RecursiveHash(Cls(x=1)) != RecursiveHash(Cls(x=2))

    def test_hash_false_field_ignored(self) -> None:
        Cls = _make_type(
            "HashIgn",
            [
                Field(
                    name="key",
                    ty=TypeSchema("int"),
                    default=MISSING,
                    compare=True,
                ),
                Field(
                    name="ignored",
                    ty=TypeSchema("int"),
                    default=0,
                    hash=False,
                ),
            ],
            eq=True,
            unsafe_hash=True,
        )
        assert RecursiveHash(Cls(key=42, ignored=100)) == RecursiveHash(Cls(key=42, ignored=999))

    def test_hash_dunder_installed(self) -> None:
        Cls = _make_type(
            "HashDunder",
            [Field(name="x", ty=TypeSchema("int"), default=MISSING)],
            eq=True,
            unsafe_hash=True,
        )
        assert isinstance(hash(Cls(x=42)), int)

    def test_usable_as_dict_key(self) -> None:
        Cls = _make_type(
            "HashDict",
            [
                Field(
                    name="x",
                    ty=TypeSchema("int"),
                    default=MISSING,
                    compare=True,
                ),
            ],
            eq=True,
            unsafe_hash=True,
        )
        assert {Cls(x=1): "value"}[Cls(x=1)] == "value"

    def test_usable_in_set(self) -> None:
        Cls = _make_type(
            "HashSet",
            [
                Field(
                    name="x",
                    ty=TypeSchema("int"),
                    default=MISSING,
                    compare=True,
                ),
            ],
            eq=True,
            unsafe_hash=True,
        )
        assert len({Cls(x=1), Cls(x=1), Cls(x=2)}) == 2


# ###########################################################################
#  12. Equality
# ###########################################################################
class TestFieldEquality:
    """Low-level equality via _make_type: structural compare, compare=False exclusion."""

    def test_equal_objects(self) -> None:
        Cls = _make_type(
            "EqEqual",
            [
                Field(
                    name="x",
                    ty=TypeSchema("int"),
                    default=MISSING,
                    compare=True,
                ),
                Field(
                    name="y",
                    ty=TypeSchema("float"),
                    default=MISSING,
                    compare=True,
                ),
            ],
            eq=True,
        )
        assert Cls(x=1, y=2.0) == Cls(x=1, y=2.0)

    def test_different_objects(self) -> None:
        Cls = _make_type(
            "EqDiff",
            [
                Field(
                    name="x",
                    ty=TypeSchema("int"),
                    default=MISSING,
                    compare=True,
                ),
            ],
            eq=True,
        )
        assert Cls(x=1) != Cls(x=2)

    def test_compare_false_field_ignored(self) -> None:
        Cls = _make_type(
            "EqIgn",
            [
                Field(
                    name="key",
                    ty=TypeSchema("int"),
                    default=MISSING,
                    compare=True,
                ),
                Field(
                    name="ignored",
                    ty=TypeSchema("int"),
                    default=0,
                    compare=False,
                ),
            ],
            eq=True,
        )
        assert RecursiveEq(Cls(key=42, ignored=100), Cls(key=42, ignored=999))

    def test_compare_off_excludes_from_eq(self) -> None:
        """Fields with compare=False (default) are ignored by RecursiveEq."""
        Cls = _make_type(
            "CmpOff",
            [Field(name="x", ty=TypeSchema("int"), default=MISSING)],
            eq=True,
        )
        assert RecursiveEq(Cls(x=1), Cls(x=2))

    def test_compare_true_includes_in_eq(self) -> None:
        Cls = _make_type(
            "CmpOn",
            [
                Field(
                    name="x",
                    ty=TypeSchema("int"),
                    default=MISSING,
                    compare=True,
                ),
            ],
            eq=True,
        )
        assert not RecursiveEq(Cls(x=1), Cls(x=2))

    def test_eq_reflexive(self) -> None:
        Cls = _make_type(
            "EqRefl",
            [
                Field(
                    name="x",
                    ty=TypeSchema("int"),
                    default=MISSING,
                    compare=True,
                ),
            ],
            eq=True,
        )
        a = Cls(x=42)
        assert a == a

    def test_eq_symmetric(self) -> None:
        Cls = _make_type(
            "EqSym",
            [
                Field(
                    name="x",
                    ty=TypeSchema("int"),
                    default=MISSING,
                    compare=True,
                ),
            ],
            eq=True,
        )
        a, b = Cls(x=1), Cls(x=1)
        assert a == b
        assert b == a

    def test_eq_with_str_field(self) -> None:
        Cls = _make_type(
            "EqStr",
            [
                Field(
                    name="s",
                    ty=TypeSchema("str"),
                    default=MISSING,
                    compare=True,
                ),
            ],
            eq=True,
        )
        assert RecursiveEq(Cls(s="hello"), Cls(s="hello"))
        assert not RecursiveEq(Cls(s="hello"), Cls(s="world"))

    def test_eq_hash_consistency(self) -> None:
        Cls = _make_type(
            "EqHashCon",
            [
                Field(
                    name="x",
                    ty=TypeSchema("int"),
                    default=MISSING,
                    compare=True,
                ),
                Field(
                    name="y",
                    ty=TypeSchema("float"),
                    default=MISSING,
                    compare=True,
                ),
            ],
            eq=True,
            unsafe_hash=True,
        )
        a, b = Cls(x=1, y=2.0), Cls(x=1, y=2.0)
        assert RecursiveEq(a, b)
        assert RecursiveHash(a) == RecursiveHash(b)


# ###########################################################################
#  13. Edge Cases
# ###########################################################################
class TestFieldEdgeCases:
    """Low-level edge cases via _make_type: empty class, extreme values, init=False."""

    def test_empty_class_no_fields(self) -> None:
        Cls = _make_type("EdgeEmpty", [])
        obj = Cls()
        assert isinstance(obj, core.Object)
        assert isinstance(obj, Cls)

    def test_empty_class_repr(self) -> None:
        Cls = _make_type("EdgeEmptyRepr", [])
        info = getattr(Cls, "__tvm_ffi_type_info__")
        assert info.type_key in ReprPrint(Cls())

    def test_bool_true_and_false(self) -> None:
        Cls = _make_type(
            "EdgeBool",
            [Field(name="flag", ty=TypeSchema("bool"), default=MISSING)],
        )
        assert Cls(flag=True).flag is True
        assert Cls(flag=False).flag is False

    def test_bool_default_false(self) -> None:
        Cls = _make_type(
            "EdgeBoolDef",
            [Field(name="flag", ty=TypeSchema("bool"), default=False)],
        )
        assert Cls().flag is False

    def test_multiple_types_together(self) -> None:
        Cls = _make_type(
            "EdgeMulti",
            [
                Field(
                    name="i",
                    ty=TypeSchema("int"),
                    default=MISSING,
                    compare=True,
                ),
                Field(name="f", ty=TypeSchema("float"), default=MISSING),
                Field(name="s", ty=TypeSchema("str"), default=MISSING),
                Field(name="b", ty=TypeSchema("bool"), default=MISSING),
            ],
        )
        obj = Cls(i=42, f=3.14, s="test", b=True)
        assert obj.i == 42
        assert obj.f == pytest.approx(3.14)
        assert obj.s == "test"
        assert obj.b is True

    def test_pod_and_objectref_mixed(self) -> None:
        Cls = _make_type(
            "EdgeMixed",
            [
                Field(name="count", ty=TypeSchema("int"), default=MISSING),
                Field(
                    name="items",
                    ty=TypeSchema("Array", (TypeSchema("int"),)),
                    default=MISSING,
                ),
                Field(name="label", ty=TypeSchema("str"), default=""),
            ],
        )
        obj = Cls(count=3, items=[1, 2, 3])
        assert obj.count == 3
        assert len(obj.items) == 3
        assert obj.label == ""

    def test_multiple_types_with_defaults(self) -> None:
        Cls = _make_type(
            "EdgeMultiDef",
            [
                Field(name="i", ty=TypeSchema("int"), default=0),
                Field(name="f", ty=TypeSchema("float"), default=1.0),
                Field(name="s", ty=TypeSchema("str"), default="default"),
                Field(name="b", ty=TypeSchema("bool"), default=True),
            ],
        )
        obj = Cls()
        assert obj.i == 0
        assert obj.f == pytest.approx(1.0)
        assert obj.s == "default"
        assert obj.b is True

    def test_zero_values(self) -> None:
        Cls = _make_type(
            "EdgeZero",
            [
                Field(name="i", ty=TypeSchema("int"), default=MISSING),
                Field(name="f", ty=TypeSchema("float"), default=MISSING),
            ],
        )
        obj = Cls(i=0, f=0.0)
        assert obj.i == 0
        assert obj.f == 0.0

    def test_negative_values(self) -> None:
        Cls = _make_type(
            "EdgeNeg",
            [
                Field(name="i", ty=TypeSchema("int"), default=MISSING),
                Field(name="f", ty=TypeSchema("float"), default=MISSING),
            ],
        )
        obj = Cls(i=-42, f=-3.14)
        assert obj.i == -42
        assert obj.f == pytest.approx(-3.14)

    def test_large_int(self) -> None:
        Cls = _make_type(
            "EdgeLargeInt",
            [Field(name="x", ty=TypeSchema("int"), default=MISSING)],
        )
        large = 2**62
        assert Cls(x=large).x == large

    def test_empty_string_field(self) -> None:
        Cls = _make_type(
            "EdgeEmptyStr",
            [Field(name="s", ty=TypeSchema("str"), default=MISSING)],
        )
        assert Cls(s="").s == ""

    def test_long_string_field(self) -> None:
        Cls = _make_type(
            "EdgeLongStr",
            [Field(name="s", ty=TypeSchema("str"), default=MISSING)],
        )
        long_str = "a" * 1000
        assert Cls(s=long_str).s == long_str

    def test_equality_empty_class(self) -> None:
        Cls = _make_type("EdgeEmptyEq", [], eq=True, unsafe_hash=True)
        assert RecursiveEq(Cls(), Cls())
        assert RecursiveHash(Cls()) == RecursiveHash(Cls())

    def test_init_false_field_excluded_from_init(self) -> None:
        Cls = _make_type(
            "EdgeInitFalse",
            [
                Field(name="visible", ty=TypeSchema("int"), default=MISSING),
                Field(
                    name="internal",
                    ty=TypeSchema("int"),
                    default=0,
                    init=False,
                ),
            ],
        )
        obj = Cls(visible=42)
        assert obj.visible == 42
        assert obj.internal == 0

    def test_init_false_field_rejected_as_kwarg(self) -> None:
        Cls = _make_type(
            "EdgeInitFalseReject",
            [
                Field(name="visible", ty=TypeSchema("int"), default=MISSING),
                Field(
                    name="internal",
                    ty=TypeSchema("int"),
                    default=0,
                    init=False,
                ),
            ],
        )
        with pytest.raises(TypeError):
            Cls(visible=1, internal=2)

    def test_init_false_field_writable(self) -> None:
        Cls = _make_type(
            "EdgeInitFalseWrite",
            [
                Field(name="visible", ty=TypeSchema("int"), default=MISSING),
                Field(
                    name="internal",
                    ty=TypeSchema("int"),
                    default=0,
                    init=False,
                ),
            ],
        )
        obj = Cls(visible=1)
        obj.internal = 99
        assert obj.internal == 99


# ###########################################################################
#  14. Inheritance (Python-defined parent)
# ###########################################################################
class TestFieldInheritance:
    """Low-level inheritance via _make_type: field offsets, parent-child layout."""

    def test_child_fields_after_parent(self) -> None:
        Parent = _make_type(
            "InhParent",
            [Field(name="x", ty=TypeSchema("int"), default=MISSING)],
        )
        Child = _make_type(
            "InhChild",
            [Field(name="y", ty=TypeSchema("int"), default=MISSING)],
            parent=Parent,
        )
        obj = Child(1, 2)
        assert obj.x == 1
        assert obj.y == 2

    def test_child_field_offsets_non_overlapping(self) -> None:
        Parent = _make_type(
            "InhParentOff",
            [Field(name="x", ty=TypeSchema("int"), default=MISSING)],
        )
        Child = _make_type(
            "InhChildOff",
            [Field(name="y", ty=TypeSchema("int"), default=MISSING)],
            parent=Parent,
        )
        p_info = getattr(Parent, "__tvm_ffi_type_info__")
        c_info = getattr(Child, "__tvm_ffi_type_info__")
        parent_end = p_info.fields[0].offset + p_info.fields[0].size
        assert c_info.fields[0].offset >= parent_end

    def test_mutation_no_aliasing(self) -> None:
        Parent = _make_type(
            "InhParentAlias",
            [Field(name="x", ty=TypeSchema("int"), default=MISSING)],
        )
        Child = _make_type(
            "InhChildAlias",
            [Field(name="y", ty=TypeSchema("int"), default=MISSING)],
            parent=Parent,
        )
        obj = Child(1, 2)
        obj.y = 9
        assert obj.x == 1
        assert obj.y == 9

    def test_three_level_inheritance(self) -> None:
        """Object → A → B → C: all fields accessible and non-overlapping."""
        A = _make_type(
            "InhA",
            [Field(name="a", ty=TypeSchema("int"), default=MISSING)],
        )
        B = _make_type(
            "InhB",
            [Field(name="b", ty=TypeSchema("str"), default=MISSING)],
            parent=A,
        )
        C = _make_type(
            "InhC",
            [Field(name="c", ty=TypeSchema("float"), default=MISSING)],
            parent=B,
        )
        obj = C(a=1, b="two", c=3.0)
        assert obj.a == 1
        assert obj.b == "two"
        assert obj.c == pytest.approx(3.0)

    def test_three_level_offsets_non_overlapping(self) -> None:
        A = _make_type(
            "InhAOff",
            [Field(name="a", ty=TypeSchema("int"), default=MISSING)],
        )
        B = _make_type(
            "InhBOff",
            [Field(name="b", ty=TypeSchema("int"), default=MISSING)],
            parent=A,
        )
        C = _make_type(
            "InhCOff",
            [Field(name="c", ty=TypeSchema("int"), default=MISSING)],
            parent=B,
        )
        a_info = getattr(A, "__tvm_ffi_type_info__")
        b_info = getattr(B, "__tvm_ffi_type_info__")
        c_info = getattr(C, "__tvm_ffi_type_info__")
        a_end = a_info.fields[0].offset + a_info.fields[0].size
        b_end = b_info.fields[0].offset + b_info.fields[0].size
        assert b_info.fields[0].offset >= a_end
        assert c_info.fields[0].offset >= b_end

    def test_three_level_mutation_no_aliasing(self) -> None:
        A = _make_type(
            "InhAMut",
            [Field(name="a", ty=TypeSchema("int"), default=MISSING)],
        )
        B = _make_type(
            "InhBMut",
            [Field(name="b", ty=TypeSchema("int"), default=MISSING)],
            parent=A,
        )
        C = _make_type(
            "InhCMut",
            [Field(name="c", ty=TypeSchema("int"), default=MISSING)],
            parent=B,
        )
        obj = C(a=1, b=2, c=3)
        obj.c = 99
        assert obj.a == 1
        assert obj.b == 2
        assert obj.c == 99
        obj.a = 77
        assert obj.a == 77
        assert obj.b == 2
        assert obj.c == 99

    def test_three_level_isinstance(self) -> None:
        A = _make_type(
            "InhAIs",
            [Field(name="a", ty=TypeSchema("int"), default=MISSING)],
        )
        B = _make_type(
            "InhBIs",
            [Field(name="b", ty=TypeSchema("int"), default=MISSING)],
            parent=A,
        )
        C = _make_type(
            "InhCIs",
            [Field(name="c", ty=TypeSchema("int"), default=MISSING)],
            parent=B,
        )
        obj = C(a=1, b=2, c=3)
        assert isinstance(obj, C)
        assert isinstance(obj, B)
        assert isinstance(obj, A)
        assert isinstance(obj, core.Object)

    def test_three_level_deep_copy(self) -> None:
        A = _make_type(
            "InhACopy",
            [Field(name="a", ty=TypeSchema("int"), default=MISSING)],
        )
        B = _make_type(
            "InhBCopy",
            [Field(name="b", ty=TypeSchema("int"), default=MISSING)],
            parent=A,
        )
        C = _make_type(
            "InhCCopy",
            [Field(name="c", ty=TypeSchema("int"), default=MISSING)],
            parent=B,
        )
        obj = C(a=1, b=2, c=3)
        obj_copy = DeepCopy(obj)
        assert not obj.same_as(obj_copy)
        assert obj_copy.a == 1
        assert obj_copy.b == 2
        assert obj_copy.c == 3
        obj_copy.c = 99
        assert obj.c == 3


# ###########################################################################
#  15. Mutual / Self References
# ###########################################################################
class TestMutualReferences:
    """Low-level mutual and self-referential type fields via two-phase registration."""

    def _register_bare(self, name: str) -> tuple[type, core.TypeInfo]:
        """Register a type with no fields (phase 1 of two-phase)."""
        parent_info = core._type_cls_to_type_info(core.Object)
        assert parent_info is not None
        cls = type(name, (core.Object,), {"__slots__": ()})
        info = core._register_py_class(parent_info, _unique_key_ff(name), cls)
        return cls, info

    def _finalize(self, cls: type, info: core.TypeInfo, fields: List[Field]) -> None:
        """Register fields and install class attrs (phase 2 of two-phase)."""
        info._register_fields(fields)
        setattr(cls, "__tvm_ffi_type_info__", info)
        _add_class_attrs(cls, info)
        _install_dataclass_dunders(
            cls,
            init=True,
            repr=True,
            eq=False,
            order=False,
            unsafe_hash=False,
        )

    def test_mutual_references(self) -> None:
        """Foo has Optional[Bar], Bar has Optional[Foo]."""
        Foo, foo_info = self._register_bare("MutFoo")
        Bar, bar_info = self._register_bare("MutBar")
        foo_schema = TypeSchema(foo_info.type_key, origin_type_index=foo_info.type_index)
        bar_schema = TypeSchema(bar_info.type_key, origin_type_index=bar_info.type_index)
        self._finalize(
            Foo,
            foo_info,
            [
                Field(name="a", ty=TypeSchema("str"), default=MISSING),
                Field(
                    name="bar",
                    ty=TypeSchema("Optional", (bar_schema,)),
                    default=None,
                ),
            ],
        )
        self._finalize(
            Bar,
            bar_info,
            [
                Field(
                    name="foo",
                    ty=TypeSchema("Optional", (foo_schema,)),
                    default=None,
                ),
            ],
        )
        foo = Foo(a="hello")
        bar = Bar()
        bar.foo = foo
        foo.bar = bar
        assert foo.bar.foo.a == "hello"

    def test_self_referential_field(self) -> None:
        """Bar has Optional[Bar] (self-reference)."""
        Bar, bar_info = self._register_bare("SelfRef")
        bar_schema = TypeSchema(bar_info.type_key, origin_type_index=bar_info.type_index)
        self._finalize(
            Bar,
            bar_info,
            [
                Field(name="val", ty=TypeSchema("int"), default=MISSING),
                Field(
                    name="next",
                    ty=TypeSchema("Optional", (bar_schema,)),
                    default=None,
                ),
            ],
        )
        a = Bar(val=1)
        b = Bar(val=2, next=a)
        assert b.next.val == 1
        assert a.next is None
        # Circular: a → b → a
        a.next = b
        assert a.next.next.val == 1

    def test_typed_mutual_ref_rejects_wrong_type(self) -> None:
        """Optional[Foo] field should reject Bar objects."""
        Foo, foo_info = self._register_bare("TypedFoo")
        Bar, bar_info = self._register_bare("TypedBar")
        foo_schema = TypeSchema(foo_info.type_key, origin_type_index=foo_info.type_index)
        self._finalize(
            Foo,
            foo_info,
            [
                Field(name="x", ty=TypeSchema("int"), default=MISSING),
            ],
        )
        self._finalize(
            Bar,
            bar_info,
            [
                Field(
                    name="foo",
                    ty=TypeSchema("Optional", (foo_schema,)),
                    default=None,
                ),
            ],
        )
        bar = Bar()
        bar.foo = Foo(x=1)  # OK
        assert bar.foo.x == 1
        with pytest.raises((TypeError, RuntimeError)):
            bar.foo = bar  # Bar is not Foo


# ###########################################################################
#  16. Inheritance (native C++ parent)
# ###########################################################################
class TestNativeParentInheritance:
    """Low-level Python child of C++ TestObjectBase: offsets, fields, methods, copy."""

    def test_non_overlapping_offsets(self) -> None:
        parent_info = core._type_cls_to_type_info(_TestObjectBase)
        assert parent_info is not None
        Child = _make_type(
            "InhNativeChild",
            [Field(name="extra", ty=TypeSchema("int"), default=MISSING)],
            parent=_TestObjectBase,
        )
        child_info = getattr(Child, "__tvm_ffi_type_info__")
        parent_end = max(f.offset + f.size for f in parent_info.fields)
        assert child_info.fields[0].offset >= parent_end

    def test_preserves_parent_fields(self) -> None:
        Child = _make_type(
            "InhNativePreserve",
            [Field(name="extra", ty=TypeSchema("int"), default=MISSING)],
            parent=_TestObjectBase,
        )
        obj = Child(extra=7, v_i64=1, v_f64=2.0, v_str="x")
        assert obj.extra == 7
        assert obj.v_i64 == 1
        assert obj.v_f64 == 2.0
        assert obj.v_str == "x"

    def test_mutation_no_aliasing(self) -> None:
        Child = _make_type(
            "InhNativeMut",
            [Field(name="extra", ty=TypeSchema("int"), default=MISSING)],
            parent=_TestObjectBase,
        )
        obj = Child(extra=7, v_i64=1, v_f64=2.0, v_str="x")
        obj.extra = 33
        assert obj.extra == 33
        assert obj.v_i64 == 1
        assert obj.v_f64 == 2.0
        assert obj.v_str == "x"

    def test_parent_method_uses_parent_state(self) -> None:
        Child = _make_type(
            "InhNativeMethod",
            [Field(name="extra", ty=TypeSchema("int"), default=MISSING)],
            parent=_TestObjectBase,
        )
        obj = Child(extra=7, v_i64=1, v_f64=2.0, v_str="x")
        assert obj.add_i64(5) == 6

    def test_copy_preserves_all_fields(self) -> None:
        Child = _make_type(
            "InhNativeCopy",
            [Field(name="extra", ty=TypeSchema("int"), default=MISSING)],
            parent=_TestObjectBase,
        )
        obj = Child(extra=7, v_i64=1, v_f64=2.0, v_str="x")
        obj_copy = copy.copy(obj)
        assert obj_copy.extra == 7
        assert obj_copy.v_i64 == 1
        assert obj_copy.v_f64 == 2.0
        assert obj_copy.v_str == "x"

    def test_deepcopy_preserves_all_fields(self) -> None:
        Child = _make_type(
            "InhNativeDeepCopy",
            [Field(name="extra", ty=TypeSchema("int"), default=MISSING)],
            parent=_TestObjectBase,
        )
        obj = Child(extra=7, v_i64=1, v_f64=2.0, v_str="x")
        obj_copy = copy.deepcopy(obj)
        assert obj_copy.extra == 7
        assert obj_copy.v_i64 == 1
        assert obj_copy.v_f64 == 2.0
        assert obj_copy.v_str == "x"


# ###########################################################################
#  16. Deep Copy
# ###########################################################################
class TestDeepCopy:
    """Low-level DeepCopy via _make_type: nested ObjectRef, mutation independence."""

    def test_deep_copy_basic(self) -> None:
        Cls = _make_type(
            "DCBasic",
            [
                Field(
                    name="x",
                    ty=TypeSchema("int"),
                    default=MISSING,
                    compare=True,
                ),
                Field(
                    name="s",
                    ty=TypeSchema("str"),
                    default=MISSING,
                    compare=True,
                ),
            ],
            eq=True,
        )
        obj = Cls(x=42, s="hello")
        obj_copy = DeepCopy(obj)
        assert not obj.same_as(obj_copy)
        assert RecursiveEq(obj, obj_copy)

    def test_deep_copy_nested_objectref(self) -> None:
        Cls = _make_type(
            "DCNested",
            [
                Field(
                    name="items",
                    ty=TypeSchema("Array", (TypeSchema("int"),)),
                    default=MISSING,
                ),
            ],
        )
        obj = Cls(items=tvm_ffi.Array([1, 2, 3]))
        obj_copy = DeepCopy(obj)
        assert not obj.items.same_as(obj_copy.items)
        assert len(obj_copy.items) == 3

    def test_deep_copy_mutate_independent(self) -> None:
        Cls = _make_type(
            "DCMut",
            [Field(name="x", ty=TypeSchema("int"), default=MISSING)],
        )
        obj = Cls(x=1)
        obj_copy = DeepCopy(obj)
        obj_copy.x = 99
        assert obj.x == 1

    def test_python_deepcopy_dunder(self) -> None:
        Cls = _make_type(
            "DCPython",
            [Field(name="x", ty=TypeSchema("int"), default=MISSING)],
        )
        obj = Cls(x=42)
        obj_copy = copy.deepcopy(obj)
        assert obj_copy.x == 42
        assert not obj.same_as(obj_copy)


# ###########################################################################
#  17. Memory / Lifetime
# ###########################################################################
class TestMemoryLifetime:
    """Low-level ref-counting: ObjectRef/Any fields are properly ref-counted."""

    def test_objectref_field_kept_alive(self) -> None:
        Cls = _make_type(
            "MemAlive",
            [
                Field(
                    name="arr",
                    ty=TypeSchema("Array", (TypeSchema("int"),)),
                    default=MISSING,
                ),
            ],
        )
        arr = tvm_ffi.Array([1, 2, 3])
        obj = Cls(arr=arr)
        del arr
        gc.collect()
        assert len(obj.arr) == 3

    def test_multiple_objects_independent_lifetime(self) -> None:
        Cls = _make_type(
            "MemIndep",
            [
                Field(
                    name="arr",
                    ty=TypeSchema("Array", (TypeSchema("int"),)),
                    default=MISSING,
                ),
            ],
        )
        shared = tvm_ffi.Array([10, 20])
        a = Cls(arr=shared)
        b = Cls(arr=shared)
        del a
        gc.collect()
        assert len(b.arr) == 2
        assert b.arr[0] == 10

    def test_str_field_any_storage(self) -> None:
        Cls = _make_type(
            "MemStr",
            [Field(name="s", ty=TypeSchema("str"), default=MISSING)],
        )
        assert Cls(s="hi").s == "hi"
        long_str = "a" * 500
        assert Cls(s=long_str).s == long_str


# ###########################################################################
#  18. Bool Alignment
# ###########################################################################
class TestBoolAlignment:
    """Low-level bool field layout: 1-byte packing, padding, alternating layouts."""

    def test_bool_then_int_alignment(self) -> None:
        Cls = _make_type(
            "BoolAlign",
            [
                Field(name="flag", ty=TypeSchema("bool"), default=MISSING),
                Field(name="val", ty=TypeSchema("int"), default=MISSING),
            ],
        )
        info = getattr(Cls, "__tvm_ffi_type_info__")
        assert info.fields[0].offset == 24
        assert info.fields[1].offset % 8 == 0
        assert info.fields[1].offset >= info.fields[0].offset + 1

    def test_bool_then_int_values(self) -> None:
        Cls = _make_type(
            "BoolAlignVal",
            [
                Field(name="flag", ty=TypeSchema("bool"), default=MISSING),
                Field(name="val", ty=TypeSchema("int"), default=MISSING),
            ],
        )
        obj = Cls(flag=True, val=42)
        assert obj.flag is True
        assert obj.val == 42
        obj.flag = False
        assert obj.flag is False
        assert obj.val == 42

    def test_multiple_bools_packed(self) -> None:
        Cls = _make_type(
            "MultiBool",
            [
                Field(name="a", ty=TypeSchema("bool"), default=MISSING),
                Field(name="b", ty=TypeSchema("bool"), default=MISSING),
                Field(name="c", ty=TypeSchema("bool"), default=MISSING),
            ],
        )
        info = getattr(Cls, "__tvm_ffi_type_info__")
        assert [f.offset for f in info.fields] == [24, 25, 26]
        obj = Cls(a=True, b=False, c=True)
        assert obj.a is True
        assert obj.b is False
        assert obj.c is True

    def test_bool_int_bool_int_alternating(self) -> None:
        Cls = _make_type(
            "BoolIntBoolInt",
            [
                Field(name="b1", ty=TypeSchema("bool"), default=MISSING),
                Field(name="i1", ty=TypeSchema("int"), default=MISSING),
                Field(name="b2", ty=TypeSchema("bool"), default=MISSING),
                Field(name="i2", ty=TypeSchema("int"), default=MISSING),
            ],
        )
        obj = Cls(b1=True, i1=100, b2=False, i2=200)
        assert obj.b1 is True
        assert obj.i1 == 100
        assert obj.b2 is False
        assert obj.i2 == 200


# ###########################################################################
#  19. Type Conversion Errors
# ###########################################################################
class TestTypeConversionErrors:
    """Low-level type conversion: wrong-type setter/construction raises."""

    def test_set_int_field_to_str_raises(self) -> None:
        Cls = _make_type(
            "ErrIntStr",
            [Field(name="x", ty=TypeSchema("int"), default=MISSING)],
        )
        obj = Cls(x=1)
        with pytest.raises((TypeError, RuntimeError)):
            obj.x = "not_an_int"

    def test_set_str_field_to_int_raises(self) -> None:
        Cls = _make_type(
            "ErrStrInt",
            [Field(name="s", ty=TypeSchema("str"), default=MISSING)],
        )
        obj = Cls(s="hello")
        with pytest.raises((TypeError, RuntimeError)):
            obj.s = 42

    def test_construct_with_wrong_type_raises(self) -> None:
        Cls = _make_type(
            "ErrInit",
            [Field(name="x", ty=TypeSchema("int"), default=MISSING)],
        )
        with pytest.raises((TypeError, RuntimeError)):
            Cls(x="bad")

    def test_set_wrong_type_preserves_old_value(self) -> None:
        """Failed type-checked mutation preserves old value."""
        Cls = _make_type(
            "ErrPreserve",
            [Field(name="x", ty=TypeSchema("int"), default=MISSING)],
        )
        obj = Cls(x=42)
        with pytest.raises((TypeError, RuntimeError)):
            obj.x = "bad_value"
        assert obj.x == 42

    def test_type_schema_convert_raises_directly(self) -> None:
        """TypeSchema.convert raises TypeError for incompatible values."""
        ts = TypeSchema("int")
        assert _to_py_class_value(ts.convert(42)) == 42
        with pytest.raises(TypeError):
            ts.convert("not_an_int")

    def test_set_non_optional_object_field_to_none_raises(self) -> None:
        """A non-Optional Object field must reject None."""
        Cls = _make_type(
            "ErrObjNone",
            [
                Field(
                    name="child",
                    ty=TypeSchema("Object"),
                    default=MISSING,
                ),
            ],
        )
        obj = Cls(child=tvm_ffi.Array([1]))
        with pytest.raises((TypeError, RuntimeError)):
            obj.child = None

    def test_construct_non_optional_object_field_with_none_raises(self) -> None:
        """Constructing with None for a non-Optional Object field must fail."""
        Cls = _make_type(
            "ErrObjNoneInit",
            [
                Field(
                    name="child",
                    ty=TypeSchema("Object"),
                    default=MISSING,
                ),
            ],
        )
        with pytest.raises((TypeError, RuntimeError)):
            Cls(child=None)

    def test_optional_object_field_accepts_none(self) -> None:
        """An Optional[Object] field should accept None."""
        Cls = _make_type(
            "OptObjNone",
            [
                Field(
                    name="child",
                    ty=TypeSchema("Optional", (TypeSchema("Object"),)),
                    default=None,
                ),
            ],
        )
        obj = Cls()
        assert obj.child is None
        obj.child = tvm_ffi.Array([1, 2])
        assert len(obj.child) == 2
        obj.child = None
        assert obj.child is None

    def test_set_bool_field_to_str_raises(self) -> None:
        Cls = _make_type(
            "ErrBoolStr",
            [Field(name="b", ty=TypeSchema("bool"), default=MISSING)],
        )
        obj = Cls(b=True)
        with pytest.raises((TypeError, RuntimeError)):
            obj.b = "not_a_bool"

    def test_set_array_field_to_int_raises(self) -> None:
        Cls = _make_type(
            "ErrArrInt",
            [
                Field(
                    name="arr",
                    ty=TypeSchema("Array", (TypeSchema("int"),)),
                    default=MISSING,
                ),
            ],
        )
        obj = Cls(arr=[1])
        with pytest.raises((TypeError, RuntimeError)):
            obj.arr = 42

    def test_construct_multiple_wrong_types_first_caught(self) -> None:
        """When the first field has a wrong type, the error is caught."""
        Cls = _make_type(
            "ErrMulti",
            [
                Field(name="x", ty=TypeSchema("int"), default=MISSING),
                Field(name="y", ty=TypeSchema("str"), default=MISSING),
            ],
        )
        with pytest.raises((TypeError, RuntimeError)):
            Cls(x="bad", y="ok")

    def test_set_optional_to_wrong_inner_type_raises(self) -> None:
        Cls = _make_type(
            "ErrOptWrong",
            [
                Field(
                    name="x",
                    ty=TypeSchema("Optional", (TypeSchema("int"),)),
                    default=None,
                ),
            ],
        )
        obj = Cls()
        with pytest.raises((TypeError, RuntimeError)):
            obj.x = "not_an_int"


# ###########################################################################
#  20. Setter / Getter Corner Cases
# ###########################################################################
class TestSetterGetterCornerCases:
    """Low-level setter/getter corner cases: conversions, nesting, edge values."""

    # --- Bool / int coercion ---

    def test_bool_field_accepts_true_false(self) -> None:
        Cls = _make_type(
            "SGBool",
            [Field(name="b", ty=TypeSchema("bool"), default=MISSING)],
        )
        obj = Cls(b=True)
        assert obj.b is True
        obj.b = False
        assert obj.b is False

    def test_int_field_accepts_bool(self) -> None:
        """Python bool is a subclass of int — FFI should accept it."""
        Cls = _make_type(
            "SGIntBool",
            [Field(name="x", ty=TypeSchema("int"), default=MISSING)],
        )
        obj = Cls(x=True)
        assert obj.x == 1
        obj.x = False
        assert obj.x == 0

    # --- Float edge values ---

    def test_float_field_inf_nan(self) -> None:
        Cls = _make_type(
            "SGFloatEdge",
            [Field(name="f", ty=TypeSchema("float"), default=MISSING)],
        )
        obj = Cls(f=float("inf"))
        assert math.isinf(obj.f)
        obj.f = float("-inf")
        assert math.isinf(obj.f) and obj.f < 0
        obj.f = float("nan")
        assert math.isnan(obj.f)

    def test_float_field_accepts_int(self) -> None:
        Cls = _make_type(
            "SGFloatInt",
            [Field(name="f", ty=TypeSchema("float"), default=MISSING)],
        )
        obj = Cls(f=42)
        assert obj.f == pytest.approx(42.0)

    # --- String edge values ---

    def test_str_field_unicode(self) -> None:
        Cls = _make_type(
            "SGStrUni",
            [Field(name="s", ty=TypeSchema("str"), default=MISSING)],
        )
        obj = Cls(s="日本語テスト 🎉")
        assert obj.s == "日本語テスト 🎉"

    def test_str_field_null_bytes(self) -> None:
        Cls = _make_type(
            "SGStrNull",
            [Field(name="s", ty=TypeSchema("str"), default=MISSING)],
        )
        s = "hello\x00world"
        obj = Cls(s=s)
        assert obj.s == s

    # --- Multiple mutations ---

    def test_repeated_mutation_same_field(self) -> None:
        Cls = _make_type(
            "SGRepeat",
            [Field(name="x", ty=TypeSchema("int"), default=MISSING)],
        )
        obj = Cls(x=0)
        for i in range(100):
            obj.x = i
        assert obj.x == 99

    def test_repeated_str_mutation(self) -> None:
        """Stress: repeated str assignment should not leak."""
        Cls = _make_type(
            "SGRepeatStr",
            [Field(name="s", ty=TypeSchema("str"), default=MISSING)],
        )
        obj = Cls(s="init")
        for i in range(100):
            obj.s = f"value_{i}"
        assert obj.s == "value_99"

    def test_repeated_objectref_mutation(self) -> None:
        """Stress: repeated ObjectRef assignment should properly DecRef old values."""
        Cls = _make_type(
            "SGRepeatArr",
            [
                Field(
                    name="arr",
                    ty=TypeSchema("Array", (TypeSchema("int"),)),
                    default=MISSING,
                ),
            ],
        )
        obj = Cls(arr=[0])
        for i in range(50):
            obj.arr = tvm_ffi.Array([i])
        assert obj.arr[0] == 49

    # --- Nested object fields ---

    def test_nested_two_levels(self) -> None:
        Inner = _make_type(
            "SGInner",
            [Field(name="val", ty=TypeSchema("int"), default=MISSING)],
        )
        inner_info = getattr(Inner, "__tvm_ffi_type_info__")
        inner_schema = TypeSchema(inner_info.type_key, origin_type_index=inner_info.type_index)
        Outer = _make_type(
            "SGOuter",
            [Field(name="child", ty=inner_schema, default=MISSING)],
        )
        obj = Outer(child=Inner(val=42))
        assert obj.child.val == 42
        # Mutate inner through outer
        new_inner = Inner(val=99)
        obj.child = new_inner
        assert obj.child.val == 99

    def test_self_referential_optional_field(self) -> None:
        """A type with an Optional[Self] field (stored as Any)."""
        Cls = _make_type(
            "SGSelfRef",
            [
                Field(name="val", ty=TypeSchema("int"), default=MISSING),
                Field(
                    name="next",
                    ty=TypeSchema("Optional", (TypeSchema("Object"),)),
                    default=None,
                ),
            ],
        )
        a = Cls(val=1)
        b = Cls(val=2, next=a)
        assert b.val == 2
        assert b.next.val == 1
        assert a.next is None

    # --- Default factory edge cases ---

    def test_default_factory_called_each_time(self) -> None:
        call_count = [0]

        def factory() -> int:
            call_count[0] += 1
            return call_count[0]

        Cls = _make_type(
            "SGFactoryCount",
            [Field(name="x", ty=TypeSchema("int"), default_factory=factory)],
        )
        a = Cls()
        b = Cls()
        c = Cls()
        assert a.x == 1
        assert b.x == 2
        assert c.x == 3

    # --- Mixed types in one class ---

    def test_all_pod_plus_objectref_plus_optional(self) -> None:
        Cls = _make_type(
            "SGKitchenSink",
            [
                Field(name="i", ty=TypeSchema("int"), default=MISSING),
                Field(name="f", ty=TypeSchema("float"), default=MISSING),
                Field(name="b", ty=TypeSchema("bool"), default=MISSING),
                Field(name="s", ty=TypeSchema("str"), default=MISSING),
                Field(
                    name="arr",
                    ty=TypeSchema("Array", (TypeSchema("int"),)),
                    default=MISSING,
                ),
                Field(
                    name="opt",
                    ty=TypeSchema("Optional", (TypeSchema("int"),)),
                    default=None,
                ),
            ],
        )
        obj = Cls(i=1, f=2.0, b=True, s="hi", arr=[10, 20])
        assert obj.i == 1
        assert obj.f == pytest.approx(2.0)
        assert obj.b is True
        assert obj.s == "hi"
        assert len(obj.arr) == 2
        assert obj.opt is None
        # Mutate all fields
        obj.i = -1
        obj.f = -2.0
        obj.b = False
        obj.s = "bye"
        obj.arr = tvm_ffi.Array([30])
        obj.opt = 42
        assert obj.i == -1
        assert obj.f == pytest.approx(-2.0)
        assert obj.b is False
        assert obj.s == "bye"
        assert len(obj.arr) == 1
        assert obj.opt == 42


# ###########################################################################
#  21. FFI Global Function Existence
# ###########################################################################
class TestFFIGlobalFunctions:
    """Low-level FFI global function registration checks."""

    def test_make_ffi_new_exists(self) -> None:
        assert tvm_ffi.get_global_func("ffi.MakeFFINew", allow_missing=True) is not None

    def test_register_auto_init_exists(self) -> None:
        assert tvm_ffi.get_global_func("ffi.RegisterAutoInit", allow_missing=True) is not None

    def test_get_field_getter_exists(self) -> None:
        assert tvm_ffi.get_global_func("ffi.GetFieldGetter", allow_missing=True) is not None

    def test_make_field_setter_exists(self) -> None:
        assert tvm_ffi.get_global_func("ffi.MakeFieldSetter", allow_missing=True) is not None

    def test_make_new_removed(self) -> None:
        assert tvm_ffi.get_global_func("ffi.MakeNew", allow_missing=True) is None


# ###########################################################################
#  22. Container field annotations
# ###########################################################################
class TestContainerFieldAnnotations:
    """Container field annotations: List[T], Dict[K,V], nested."""

    def test_list_int_field(self) -> None:
        @py_class(_unique_key("ListInt"))
        class ListInt(Object):
            items: List[int]

        obj = ListInt(items=[1, 2, 3])
        assert len(obj.items) == 3
        assert obj.items[0] == 1
        assert obj.items[2] == 3

    def test_list_int_from_tuple(self) -> None:
        @py_class(_unique_key("ListIntTup"))
        class ListIntTup(Object):
            items: List[int]

        obj = ListIntTup(items=(10, 20, 30))  # ty:ignore[invalid-argument-type]
        assert len(obj.items) == 3
        assert obj.items[1] == 20

    def test_dict_str_int_field(self) -> None:
        @py_class(_unique_key("DictStrInt"))
        class DictStrInt(Object):
            mapping: Dict[str, int]

        obj = DictStrInt(mapping={"a": 1, "b": 2})
        assert len(obj.mapping) == 2
        assert obj.mapping["a"] == 1
        assert obj.mapping["b"] == 2

    def test_list_list_int_field(self) -> None:
        @py_class(_unique_key("ListListInt"))
        class ListListInt(Object):
            matrix: List[List[int]]

        obj = ListListInt(matrix=[[1, 2, 3], [4, 5, 6]])
        assert len(obj.matrix) == 2
        assert len(obj.matrix[0]) == 3
        assert obj.matrix[0][0] == 1
        assert obj.matrix[1][2] == 6

    def test_dict_str_list_int_field(self) -> None:
        @py_class(_unique_key("DictStrListInt"))
        class DictStrListInt(Object):
            data: Dict[str, List[int]]

        obj = DictStrListInt(data={"x": [1, 2, 3], "y": [4, 5, 6]})
        assert len(obj.data) == 2
        assert tuple(obj.data["x"]) == (1, 2, 3)
        assert tuple(obj.data["y"]) == (4, 5, 6)

    def test_container_field_set(self) -> None:
        @py_class(_unique_key("ContSet"))
        class ContSet(Object):
            items: List[int]

        obj = ContSet(items=[1, 2])
        assert tuple(obj.items) == (1, 2)
        obj.items = [3, 4, 5]
        assert len(obj.items) == 3
        assert obj.items[0] == 3

    def test_dict_field_set(self) -> None:
        @py_class(_unique_key("DictSet"))
        class DictSet(Object):
            mapping: Dict[str, int]

        obj = DictSet(mapping={"a": 1})
        obj.mapping = {"b": 2, "c": 3}
        assert len(obj.mapping) == 2
        assert obj.mapping["b"] == 2

    def test_container_shared_reference(self) -> None:
        @py_class(_unique_key("ContShare"))
        class ContShare(Object):
            a: List[int]
            b: List[int]

        obj = ContShare(a=[1, 2], b=[1, 2])
        assert tuple(obj.a) == tuple(obj.b)

    def test_untyped_list_field(self) -> None:
        @py_class(_unique_key("UList"))
        class UList(Object):
            items: list

        obj = UList(items=[1, "two", 3.0])
        assert len(obj.items) == 3
        assert obj.items[0] == 1
        assert obj.items[1] == "two"

    def test_untyped_dict_field(self) -> None:
        @py_class(_unique_key("UDict"))
        class UDict(Object):
            data: dict

        obj = UDict(data={"a": 1, "b": "two"})
        assert len(obj.data) == 2


# ###########################################################################
#  23. Optional container fields
# ###########################################################################
class TestOptionalContainerFields:
    """Optional[List[T]], Optional[Dict[K,V]] via @py_class."""

    @_needs_310
    def test_optional_list_int(self) -> None:
        @py_class(_unique_key("OptListInt"))
        class OptListInt(Object):
            items: Optional[List[int]]

        obj = OptListInt(items=[1, 2, 3])
        assert len(obj.items) == 3  # ty:ignore[invalid-argument-type]
        obj.items = None
        assert obj.items is None
        obj.items = [4, 5]
        assert len(obj.items) == 2

    @_needs_310
    def test_optional_dict_str_int(self) -> None:
        @py_class(_unique_key("OptDictStrInt"))
        class OptDictStrInt(Object):
            data: Optional[Dict[str, int]]

        obj = OptDictStrInt(data={"a": 1})
        assert obj.data["a"] == 1  # ty:ignore[not-subscriptable]
        obj.data = None
        assert obj.data is None
        obj.data = {"b": 2}
        assert obj.data["b"] == 2

    @_needs_310
    def test_optional_list_list_int(self) -> None:
        @py_class(_unique_key("OptLLI"))
        class OptLLI(Object):
            matrix: Optional[List[List[int]]]

        obj = OptLLI(matrix=[[1, 2], [3, 4]])
        assert obj.matrix[0][0] == 1  # ty:ignore[not-subscriptable]
        obj.matrix = None
        assert obj.matrix is None

    @_needs_310
    def test_optional_dict_str_list_int(self) -> None:
        @py_class(_unique_key("OptDSLI"))
        class OptDSLI(Object):
            data: Optional[Dict[str, List[int]]]

        obj = OptDSLI(data={"x": [1, 2, 3]})
        assert tuple(obj.data["x"]) == (1, 2, 3)  # ty:ignore[not-subscriptable]
        obj.data = None
        assert obj.data is None

    def test_optional_list_with_typing_optional(self) -> None:
        @py_class(_unique_key("OptListTyping"))
        class OptListTyping(Object):
            items: Optional[List[int]]

        obj = OptListTyping(items=[1, 2, 3])
        assert len(obj.items) == 3  # ty:ignore[invalid-argument-type]
        obj.items = None
        assert obj.items is None

    def test_optional_dict_with_typing_optional(self) -> None:
        @py_class(_unique_key("OptDictTyping"))
        class OptDictTyping(Object):
            data: Optional[Dict[str, int]]

        obj = OptDictTyping(data={"a": 1})
        assert obj.data["a"] == 1  # ty:ignore[not-subscriptable]
        obj.data = None
        assert obj.data is None


# ###########################################################################
#  24. Callable / Function fields
# ###########################################################################
class TestFunctionField:
    """Function/Callable field via @py_class decorator."""

    def test_function_field(self) -> None:
        @py_class(_unique_key("FuncFld"))
        class FuncFld(Object):
            func: tvm_ffi.Function

        fn = tvm_ffi.convert(lambda x: x + 1)
        obj = FuncFld(func=fn)
        assert obj.func(1) == 2

    def test_function_field_set(self) -> None:
        @py_class(_unique_key("FuncSet"))
        class FuncSet(Object):
            func: tvm_ffi.Function

        fn1 = tvm_ffi.convert(lambda x: x + 1)
        fn2 = tvm_ffi.convert(lambda x: x + 2)
        obj = FuncSet(func=fn1)
        assert obj.func(1) == 2
        obj.func = fn2
        assert obj.func(1) == 3

    @_needs_310
    def test_optional_function_field(self) -> None:
        @py_class(_unique_key("OptFunc"))
        class OptFunc(Object):
            func: Optional[tvm_ffi.Function]

        obj = OptFunc(func=None)
        assert obj.func is None
        obj.func = tvm_ffi.convert(lambda x: x * 2)
        assert obj.func(3) == 6
        obj.func = None
        assert obj.func is None


# ###########################################################################
#  25. Any-typed fields (decorator level)
# ###########################################################################
class TestAnyFieldDecorator:
    """Any-typed field via @py_class decorator."""

    def test_any_holds_int(self) -> None:
        @py_class(_unique_key("AnyI"))
        class AnyI(Object):
            val: Any

        assert AnyI(val=42).val == 42

    def test_any_holds_str(self) -> None:
        @py_class(_unique_key("AnyS"))
        class AnyS(Object):
            val: Any

        assert AnyS(val="hello").val == "hello"

    def test_any_holds_none(self) -> None:
        @py_class(_unique_key("AnyN"))
        class AnyN(Object):
            val: Any = None

        assert AnyN().val is None

    def test_any_holds_list(self) -> None:
        @py_class(_unique_key("AnyL"))
        class AnyL(Object):
            val: Any

        assert len(AnyL(val=[1, 2, 3]).val) == 3

    def test_any_type_change(self) -> None:
        @py_class(_unique_key("AnyChg"))
        class AnyChg(Object):
            val: Any = None

        obj = AnyChg()
        assert obj.val is None
        obj.val = 42
        assert obj.val == 42
        obj.val = "hello"
        assert obj.val == "hello"
        obj.val = tvm_ffi.Array([1, 2])
        assert len(obj.val) == 2
        obj.val = None
        assert obj.val is None


# ###########################################################################
#  26. Post-init field mutation
# ###########################################################################
class TestPostInitMutation:
    """__post_init__ that mutates field values."""

    def test_post_init_mutates_str(self) -> None:
        @py_class(_unique_key("PostMut"))
        class PostMut(Object):
            a: int
            b: str

            def __post_init__(self) -> None:
                self.b = self.b.upper()

        obj = PostMut(a=1, b="hello")
        assert obj.a == 1
        assert obj.b == "HELLO"

    def test_post_init_computes_derived(self) -> None:
        @py_class(_unique_key("PostDeriv"))
        class PostDeriv(Object):
            x: int
            doubled: int = 0

            def __post_init__(self) -> None:
                self.doubled = self.x * 2

        assert PostDeriv(x=5).doubled == 10


# ###########################################################################
#  27. Custom __init__ with init=False
# ###########################################################################
class TestCustomInitFalse:
    """Custom __init__ with init=False and reordered parameters."""

    def test_custom_init_reordered_params(self) -> None:
        @py_class(_unique_key("CustomOrd"), init=False)
        class CustomOrd(Object):
            a: int
            b: float
            c: str
            d: bool

            def __init__(self, b: float, c: str, a: int, d: bool) -> None:
                self.__ffi_init__(a, b, c, d)

        obj = CustomOrd(b=2.0, c="3", a=1, d=True)
        assert obj.a == 1
        assert obj.b == 2.0
        assert obj.c == "3"
        assert obj.d is True

    def test_custom_init_keyword_only(self) -> None:
        @py_class(_unique_key("CustomKW"), init=False)
        class CustomKW(Object):
            a: int
            b: str

            def __init__(self, *, b: str, a: int) -> None:
                self.__ffi_init__(a, b)

        obj = CustomKW(a=1, b="hello")
        assert obj.a == 1
        assert obj.b == "hello"


# ###########################################################################
#  28. Inheritance with defaults and containers
# ###########################################################################
class TestInheritanceWithDefaults:
    """Inheritance with default values and container fields."""

    def test_base_with_default_factory(self) -> None:
        @py_class(_unique_key("BaseDef"))
        class BaseDef(Object):
            a: int
            b: List[int] = field(default_factory=list)

        obj = BaseDef(a=42)
        assert obj.a == 42
        assert len(obj.b) == 0

    def test_derived_adds_optional_fields(self) -> None:
        @py_class(_unique_key("BaseD"))
        class BaseD(Object):
            a: int
            b: List[int] = field(default_factory=list)

        @py_class(_unique_key("DerivedD"))
        class DerivedD(BaseD):
            c: Optional[int] = None
            d: Optional[str] = "default"

        obj = DerivedD(a=12)
        assert obj.a == 12
        assert len(obj.b) == 0
        assert obj.c is None
        assert obj.d == "default"

    def test_derived_interleaved_required_optional(self) -> None:
        @py_class(_unique_key("BaseIL"))
        class BaseIL(Object):
            a: int
            b: List[int] = field(default_factory=list)

        @py_class(_unique_key("DerivedIL"))
        class DerivedIL(BaseIL):
            c: int
            d: Optional[str] = "default"

        obj = DerivedIL(a=1, c=2)
        assert obj.a == 1
        assert len(obj.b) == 0
        assert obj.c == 2
        assert obj.d == "default"

    def test_three_level_with_defaults(self) -> None:
        @py_class(_unique_key("L1D"))
        class L1D(Object):
            a: int

        @py_class(_unique_key("L2D"))
        class L2D(L1D):
            b: Optional[int] = None
            c: Optional[str] = "hello"

        @py_class(_unique_key("L3D"))
        class L3D(L2D):
            d: str

        obj = L3D(a=1, d="world")
        assert obj.a == 1
        assert obj.b is None
        assert obj.c == "hello"
        assert obj.d == "world"

    def test_derived_with_container_init(self) -> None:
        @py_class(_unique_key("BaseC"))
        class BaseC(Object):
            items: List[int]

        @py_class(_unique_key("DerivedC"))
        class DerivedC(BaseC):
            name: str

        obj = DerivedC(items=[1, 2, 3], name="test")
        assert len(obj.items) == 3
        assert obj.name == "test"


# ###########################################################################
#  29. Decorator-level type validation
# ###########################################################################
class TestFieldTypeValidation:
    """Type validation on set for @py_class fields."""

    def test_set_int_to_str_raises(self) -> None:
        @py_class(_unique_key("ValInt"))
        class ValInt(Object):
            x: int

        obj = ValInt(x=1)
        with pytest.raises((TypeError, RuntimeError)):
            obj.x = "not_an_int"  # ty:ignore[invalid-assignment]

    def test_set_str_to_int_raises(self) -> None:
        @py_class(_unique_key("ValStr"))
        class ValStr(Object):
            x: str

        obj = ValStr(x="hello")
        with pytest.raises((TypeError, RuntimeError)):
            obj.x = 42  # ty:ignore[invalid-assignment]

    def test_set_bool_to_str_raises(self) -> None:
        @py_class(_unique_key("ValBool"))
        class ValBool(Object):
            x: bool

        obj = ValBool(x=True)
        with pytest.raises((TypeError, RuntimeError)):
            obj.x = "not_a_bool"  # ty:ignore[invalid-assignment]

    def test_set_list_to_wrong_type_raises(self) -> None:
        @py_class(_unique_key("ValList"))
        class ValList(Object):
            items: List[int]

        obj = ValList(items=[1, 2])
        with pytest.raises((TypeError, RuntimeError)):
            obj.items = "not_a_list"  # ty:ignore[invalid-assignment]

    def test_set_dict_to_wrong_type_raises(self) -> None:
        @py_class(_unique_key("ValDict"))
        class ValDict(Object):
            data: Dict[str, int]

        obj = ValDict(data={"a": 1})
        with pytest.raises((TypeError, RuntimeError)):
            obj.data = "not_a_dict"  # ty:ignore[invalid-assignment]


# ###########################################################################
#  30. Three-level inheritance with containers
# ###########################################################################
class TestDerivedDerivedContainers:
    """Three-level inheritance with container fields and init reordering."""

    def test_three_level_with_container_and_defaults(self) -> None:
        @py_class(_unique_key("DD_L1"))
        class L1(Object):
            a: int
            b: List[int] = field(default_factory=list)

        @py_class(_unique_key("DD_L2"))
        class L2(L1):
            c: Optional[int] = None
            d: Optional[str] = "hello"

        @py_class(_unique_key("DD_L3"))
        class L3(L2):
            e: str

        obj = L3(a=1, e="world", b=[1, 2])
        assert obj.a == 1
        assert tuple(obj.b) == (1, 2)
        assert obj.c is None
        assert obj.d == "hello"
        assert obj.e == "world"

    def test_three_level_positional_call(self) -> None:
        @py_class(_unique_key("DD2_L1"))
        class L1(Object):
            a: int

        @py_class(_unique_key("DD2_L2"))
        class L2(L1):
            b: List[int] = field(default_factory=list)

        @py_class(_unique_key("DD2_L3"))
        class L3(L2):
            c: str

        obj = L3(a=1, c="x")
        assert obj.a == 1
        assert len(obj.b) == 0
        assert obj.c == "x"


# ###########################################################################
#  31. Container field mutation and type rejection
# ###########################################################################
class TestContainerFieldMutation:
    """Container field set, mutation, and type rejection."""

    def test_untyped_list_mutation(self) -> None:
        obj = _make_multi_type_obj()
        assert len(obj.list_any) == 3
        assert obj.list_any[0] == 1
        obj.list_any = [4, 3.0, "two"]
        assert len(obj.list_any) == 3
        assert obj.list_any[0] == 4
        assert obj.list_any[2] == "two"

    def test_untyped_dict_mutation(self) -> None:
        obj = _make_multi_type_obj()
        assert len(obj.dict_any) == 2
        obj.dict_any = {"4": 4, "3": "two", "2": 3.0}
        assert len(obj.dict_any) == 3
        assert obj.dict_any["4"] == 4
        assert obj.dict_any["3"] == "two"

    def test_list_any_type_rejection(self) -> None:
        @py_class(_unique_key("LAReject"))
        class LAReject(Object):
            items: List[Any]

        obj = LAReject(items=[1, 2])
        with pytest.raises((TypeError, RuntimeError)):
            obj.items = "wrong"  # ty:ignore[invalid-assignment]

    def test_list_list_int_type_rejection(self) -> None:
        @py_class(_unique_key("LLReject"))
        class LLReject(Object):
            matrix: List[List[int]]

        obj = LLReject(matrix=[[1, 2]])
        with pytest.raises((TypeError, RuntimeError)):
            obj.matrix = [4, 3, 2, 1]  # ty:ignore[invalid-assignment]
        with pytest.raises((TypeError, RuntimeError)):
            obj.matrix = None  # ty:ignore[invalid-assignment]
        assert len(obj.matrix) == 1

    def test_dict_any_any_type_rejection(self) -> None:
        @py_class(_unique_key("DAAReject"))
        class DAAReject(Object):
            data: Dict[Any, Any]

        obj = DAAReject(data={1: 2})
        with pytest.raises((TypeError, RuntimeError)):
            obj.data = "wrong"  # ty:ignore[invalid-assignment]
        with pytest.raises((TypeError, RuntimeError)):
            obj.data = 42  # ty:ignore[invalid-assignment]
        with pytest.raises((TypeError, RuntimeError)):
            obj.data = None  # ty:ignore[invalid-assignment]

    def test_dict_str_any_type_rejection(self) -> None:
        @py_class(_unique_key("DSAReject"))
        class DSAReject(Object):
            data: Dict[str, Any]

        obj = DSAReject(data={"a": 1})
        with pytest.raises((TypeError, RuntimeError)):
            obj.data = "wrong"  # ty:ignore[invalid-assignment]
        with pytest.raises((TypeError, RuntimeError)):
            obj.data = 42  # ty:ignore[invalid-assignment]
        with pytest.raises((TypeError, RuntimeError)):
            obj.data = None  # ty:ignore[invalid-assignment]
        with pytest.raises((TypeError, RuntimeError)):
            obj.data = {4: 4, 3.0: 3}  # ty:ignore[invalid-assignment]

    def test_dict_str_list_int_type_rejection(self) -> None:
        @py_class(_unique_key("DSLReject"))
        class DSLReject(Object):
            data: Dict[str, List[int]]

        obj = DSLReject(data={"a": [1, 2]})
        with pytest.raises((TypeError, RuntimeError)):
            obj.data = "wrong"  # ty:ignore[invalid-assignment]
        with pytest.raises((TypeError, RuntimeError)):
            obj.data = 42  # ty:ignore[invalid-assignment]
        with pytest.raises((TypeError, RuntimeError)):
            obj.data = None  # ty:ignore[invalid-assignment]
        with pytest.raises((TypeError, RuntimeError)):
            obj.data = {"a": 1, "b": [2]}  # ty:ignore[invalid-assignment]


# ###########################################################################
#  32. Optional field set/unset cycles with type rejection
# ###########################################################################
class TestOptionalFieldCycles:
    """Optional field set → None → set-back cycles with type rejection."""

    def test_opt_func_type_rejection(self) -> None:
        @py_class(_unique_key("OptFuncR"))
        class OptFuncR(Object):
            func: Optional[tvm_ffi.Function]

        obj = OptFuncR(func=None)
        assert obj.func is None
        obj.func = tvm_ffi.convert(lambda x: x + 2)
        assert obj.func(1) == 3
        obj.func = None
        assert obj.func is None
        with pytest.raises((TypeError, RuntimeError)):
            obj.func = "wrong"  # ty:ignore[invalid-assignment]
        with pytest.raises((TypeError, RuntimeError)):
            obj.func = 42  # ty:ignore[invalid-assignment]

    def test_opt_ulist_cycle(self) -> None:
        @py_class(_unique_key("OptUListC"))
        class OptUListC(Object):
            items: Optional[list]

        obj = OptUListC(items=None)
        assert obj.items is None
        obj.items = [4, 3.0, "two"]
        assert len(obj.items) == 3
        assert obj.items[0] == 4
        with pytest.raises((TypeError, RuntimeError)):
            obj.items = "wrong"  # ty:ignore[invalid-assignment]
        with pytest.raises((TypeError, RuntimeError)):
            obj.items = 42  # ty:ignore[invalid-assignment]
        obj.items = None
        assert obj.items is None

    def test_opt_udict_cycle(self) -> None:
        @py_class(_unique_key("OptUDictC"))
        class OptUDictC(Object):
            data: Optional[dict]

        obj = OptUDictC(data=None)
        assert obj.data is None
        obj.data = {"4": 4, "3": "two"}
        assert len(obj.data) == 2
        with pytest.raises((TypeError, RuntimeError)):
            obj.data = "wrong"  # ty:ignore[invalid-assignment]
        with pytest.raises((TypeError, RuntimeError)):
            obj.data = 42  # ty:ignore[invalid-assignment]
        obj.data = None
        assert obj.data is None

    def test_opt_list_any_cycle(self) -> None:
        @py_class(_unique_key("OptLAC"))
        class OptLAC(Object):
            items: Optional[List[Any]]

        obj = OptLAC(items=[1, 2.0, "three"])
        assert len(obj.items) == 3  # ty:ignore[invalid-argument-type]
        obj.items = [4, 3.0, "two"]
        assert obj.items[0] == 4
        with pytest.raises((TypeError, RuntimeError)):
            obj.items = "wrong"  # ty:ignore[invalid-assignment]
        with pytest.raises((TypeError, RuntimeError)):
            obj.items = 42  # ty:ignore[invalid-assignment]
        obj.items = None
        assert obj.items is None

    def test_opt_list_list_int_cycle(self) -> None:
        @py_class(_unique_key("OptLLIC"))
        class OptLLIC(Object):
            matrix: Optional[List[List[int]]]

        obj = OptLLIC(matrix=[[1, 2, 3], [4, 5, 6]])
        assert tuple(obj.matrix[0]) == (1, 2, 3)  # ty:ignore[not-subscriptable]
        obj.matrix = [[4, 3, 2]]
        assert len(obj.matrix) == 1
        with pytest.raises((TypeError, RuntimeError)):
            obj.matrix = "wrong"  # ty:ignore[invalid-assignment]
        with pytest.raises((TypeError, RuntimeError)):
            obj.matrix = [1, 2, 3]  # ty:ignore[invalid-assignment]
        obj.matrix = None
        assert obj.matrix is None

    def test_opt_dict_any_any_cycle(self) -> None:
        @py_class(_unique_key("OptDAAC"))
        class OptDAAC(Object):
            data: Optional[Dict[Any, Any]]

        obj = OptDAAC(data=None)
        assert obj.data is None
        obj.data = {4: 4, "three": "two"}
        assert len(obj.data) == 2
        with pytest.raises((TypeError, RuntimeError)):
            obj.data = "wrong"  # ty:ignore[invalid-assignment]
        with pytest.raises((TypeError, RuntimeError)):
            obj.data = 42  # ty:ignore[invalid-assignment]
        obj.data = None
        assert obj.data is None

    def test_opt_dict_str_any_cycle(self) -> None:
        @py_class(_unique_key("OptDSAC"))
        class OptDSAC(Object):
            data: Optional[Dict[str, Any]]

        obj = OptDSAC(data={"a": 1})
        assert obj.data["a"] == 1  # ty:ignore[not-subscriptable]
        obj.data = {}
        assert len(obj.data) == 0
        with pytest.raises((TypeError, RuntimeError)):
            obj.data = "wrong"  # ty:ignore[invalid-assignment]
        with pytest.raises((TypeError, RuntimeError)):
            obj.data = 42  # ty:ignore[invalid-assignment]
        obj.data = None
        assert obj.data is None

    def test_opt_dict_any_str_cycle(self) -> None:
        @py_class(_unique_key("OptDASC"))
        class OptDASC(Object):
            data: Optional[Dict[Any, str]]

        obj = OptDASC(data={1: "a", "two": "b"})
        assert obj.data[1] == "a"  # ty:ignore[not-subscriptable]
        obj.data = {4: "4", "three": "two"}
        assert len(obj.data) == 2
        with pytest.raises((TypeError, RuntimeError)):
            obj.data = "wrong"  # ty:ignore[invalid-assignment]
        with pytest.raises((TypeError, RuntimeError)):
            obj.data = 42  # ty:ignore[invalid-assignment]
        obj.data = None
        assert obj.data is None

    def test_opt_dict_str_list_int_cycle(self) -> None:
        @py_class(_unique_key("OptDSLIC"))
        class OptDSLIC(Object):
            data: Optional[Dict[str, List[int]]]

        obj = OptDSLIC(data={"1": [1, 2, 3], "2": [4, 5, 6]})
        assert tuple(obj.data["1"]) == (1, 2, 3)  # ty:ignore[not-subscriptable]
        obj.data = {"a": [7, 8]}
        assert tuple(obj.data["a"]) == (7, 8)
        with pytest.raises((TypeError, RuntimeError)):
            obj.data = "wrong"  # ty:ignore[invalid-assignment]
        with pytest.raises((TypeError, RuntimeError)):
            obj.data = 42  # ty:ignore[invalid-assignment]
        obj.data = None
        assert obj.data is None


# ###########################################################################
#  33. Multi-type field class
# ###########################################################################
@py_class(_unique_key("MultiType"))
class _PyClassMultiType(Object):
    """@py_class with many field types for cross-cutting field tests."""

    bool_: bool
    i64: int
    f64: float
    str_: str
    any_val: Any
    list_int: List[int]
    list_any: list
    dict_str_int: Dict[str, int]
    dict_any: dict
    list_list_int: List[List[int]]
    dict_str_list_int: Dict[str, List[int]]
    opt_bool: Optional[bool]
    opt_int: Optional[int]
    opt_float: Optional[float]
    opt_str: Optional[str]
    opt_list_int: Optional[List[int]]
    opt_dict_str_int: Optional[Dict[str, int]]


def _make_multi_type_obj() -> _PyClassMultiType:
    return _PyClassMultiType(
        bool_=False,
        i64=64,
        f64=2.5,
        str_="world",
        any_val="hello",
        list_int=[1, 2, 3],
        list_any=[1, "two", 3.0],
        dict_str_int={"a": 1, "b": 2},
        dict_any={"x": 1, "y": "two"},
        list_list_int=[[1, 2, 3], [4, 5, 6]],
        dict_str_list_int={"p": [1, 2], "q": [3, 4]},
        opt_bool=True,
        opt_int=-64,
        opt_float=None,
        opt_str=None,
        opt_list_int=[10, 20],
        opt_dict_str_int=None,
    )


class TestMultiTypeFieldOps:
    """Per-field get/set/validation on a many-field @py_class type."""

    def test_bool_field(self) -> None:
        obj = _make_multi_type_obj()
        assert obj.bool_ is False
        obj.bool_ = True
        assert obj.bool_ is True
        with pytest.raises((TypeError, RuntimeError)):
            obj.bool_ = "not_a_bool"  # ty:ignore[invalid-assignment]

    def test_int_field(self) -> None:
        obj = _make_multi_type_obj()
        assert obj.i64 == 64
        obj.i64 = -128
        assert obj.i64 == -128
        with pytest.raises((TypeError, RuntimeError)):
            obj.i64 = "wrong"  # ty:ignore[invalid-assignment]

    def test_float_field(self) -> None:
        obj = _make_multi_type_obj()
        assert abs(obj.f64 - 2.5) < 1e-10
        obj.f64 = 5.0
        assert abs(obj.f64 - 5.0) < 1e-10
        with pytest.raises((TypeError, RuntimeError)):
            obj.f64 = "wrong"  # ty:ignore[invalid-assignment]

    def test_str_field(self) -> None:
        obj = _make_multi_type_obj()
        assert obj.str_ == "world"
        obj.str_ = "hello"
        assert obj.str_ == "hello"

    def test_any_field(self) -> None:
        obj = _make_multi_type_obj()
        assert obj.any_val == "hello"
        obj.any_val = 42
        assert obj.any_val == 42
        obj.any_val = [1, 2]
        assert len(obj.any_val) == 2

    def test_list_int_field(self) -> None:
        obj = _make_multi_type_obj()
        assert tuple(obj.list_int) == (1, 2, 3)
        obj.list_int = [4, 5]
        assert len(obj.list_int) == 2
        with pytest.raises((TypeError, RuntimeError)):
            obj.list_int = "wrong"  # ty:ignore[invalid-assignment]

    def test_untyped_list_field(self) -> None:
        obj = _make_multi_type_obj()
        assert len(obj.list_any) == 3
        assert obj.list_any[0] == 1
        assert obj.list_any[1] == "two"
        obj.list_any = [4, 3.0, "new"]
        assert len(obj.list_any) == 3

    def test_dict_str_int_field(self) -> None:
        obj = _make_multi_type_obj()
        assert obj.dict_str_int["a"] == 1
        obj.dict_str_int = {"c": 3}
        assert obj.dict_str_int["c"] == 3
        with pytest.raises((TypeError, RuntimeError)):
            obj.dict_str_int = "wrong"  # ty:ignore[invalid-assignment]

    def test_untyped_dict_field(self) -> None:
        obj = _make_multi_type_obj()
        assert len(obj.dict_any) == 2
        obj.dict_any = {"new": 42}
        assert obj.dict_any["new"] == 42

    def test_nested_list_field(self) -> None:
        obj = _make_multi_type_obj()
        assert tuple(obj.list_list_int[0]) == (1, 2, 3)
        assert tuple(obj.list_list_int[1]) == (4, 5, 6)

    def test_nested_dict_field(self) -> None:
        obj = _make_multi_type_obj()
        assert tuple(obj.dict_str_list_int["p"]) == (1, 2)
        assert tuple(obj.dict_str_list_int["q"]) == (3, 4)

    def test_optional_bool(self) -> None:
        obj = _make_multi_type_obj()
        assert obj.opt_bool is True
        obj.opt_bool = False
        assert obj.opt_bool is False
        obj.opt_bool = None
        assert obj.opt_bool is None

    def test_optional_int(self) -> None:
        obj = _make_multi_type_obj()
        assert obj.opt_int == -64
        obj.opt_int = None
        assert obj.opt_int is None
        obj.opt_int = 128
        assert obj.opt_int == 128

    def test_optional_float(self) -> None:
        obj = _make_multi_type_obj()
        assert obj.opt_float is None
        obj.opt_float = 1.5
        assert abs(obj.opt_float - 1.5) < 1e-10
        obj.opt_float = None
        assert obj.opt_float is None

    def test_optional_str(self) -> None:
        obj = _make_multi_type_obj()
        assert obj.opt_str is None
        obj.opt_str = "hello"
        assert obj.opt_str == "hello"
        obj.opt_str = None
        assert obj.opt_str is None

    def test_optional_list_int(self) -> None:
        obj = _make_multi_type_obj()
        assert tuple(obj.opt_list_int) == (10, 20)  # ty:ignore[invalid-argument-type]
        obj.opt_list_int = None
        assert obj.opt_list_int is None
        obj.opt_list_int = [30]
        assert len(obj.opt_list_int) == 1

    def test_optional_dict_str_int(self) -> None:
        obj = _make_multi_type_obj()
        assert obj.opt_dict_str_int is None
        obj.opt_dict_str_int = {"z": 99}
        assert obj.opt_dict_str_int["z"] == 99
        obj.opt_dict_str_int = None
        assert obj.opt_dict_str_int is None


class TestMultiTypeCopy:
    """Copy with the comprehensive multi-type class."""

    def test_shallow_copy_comprehensive(self) -> None:
        obj = _make_multi_type_obj()
        obj2 = copy.copy(obj)
        assert obj2.bool_ == obj.bool_
        assert obj2.i64 == obj.i64
        assert obj2.f64 == obj.f64
        assert obj2.str_ == obj.str_
        assert obj2.any_val == obj.any_val
        assert obj.list_int.same_as(obj2.list_int)  # ty:ignore[unresolved-attribute]
        assert obj.dict_str_int.same_as(obj2.dict_str_int)  # ty:ignore[unresolved-attribute]

    def test_deep_copy_comprehensive(self) -> None:
        obj = _make_multi_type_obj()
        obj2 = copy.deepcopy(obj)
        assert obj2.bool_ == obj.bool_
        assert obj2.i64 == obj.i64
        assert not obj.list_int.same_as(obj2.list_int)  # ty:ignore[unresolved-attribute]
        assert not obj.dict_str_int.same_as(obj2.dict_str_int)  # ty:ignore[unresolved-attribute]
        assert tuple(obj2.list_int) == (1, 2, 3)
        assert obj2.dict_str_int["a"] == 1


# ---------------------------------------------------------------------------
# _collect_py_methods allowlist and method introspection
# ---------------------------------------------------------------------------


class TestPyMethodAllowlist:
    """Only names in ``_FFI_RECOGNIZED_METHODS`` are collected by ``_collect_py_methods``."""

    def test_system_methods_not_in_allowlist(self) -> None:
        from tvm_ffi.dataclasses.py_class import _collect_py_methods  # noqa: PLC0415

        @py_class(_unique_key("Allow"))
        class Allow(core.Object):
            x: int

            def __ffi_init__(self, x: int) -> None:  # ty: ignore[invalid-method-override]
                pass

            def __ffi_shallow_copy__(self) -> None:
                pass

            def __ffi_repr__(self, fn_repr: Any) -> str:
                return "repr"

        collected = _collect_py_methods(Allow)
        assert collected is not None
        names = {name for name, _, _ in collected}
        assert "__ffi_repr__" in names
        assert "__ffi_init__" not in names
        assert "__ffi_shallow_copy__" not in names

    def test_arbitrary_ffi_dunder_not_collected(self) -> None:
        from tvm_ffi.dataclasses.py_class import _collect_py_methods  # noqa: PLC0415

        @py_class(_unique_key("Arb"))
        class Arb(core.Object):
            x: int

            def __ffi_custom_op__(self, y: int) -> int:
                return self.x + y

        collected = _collect_py_methods(Arb)
        assert collected is None


class TestPyMethodIntrospection:
    """Registered __ffi_* methods appear in ``TypeInfo.methods``."""

    def test_ffi_repr_in_methods(self) -> None:
        @py_class(_unique_key("IntrRepr"))
        class IntrRepr(core.Object):
            x: int

            def __ffi_repr__(self, fn_repr: Any) -> str:
                return "repr"

        info = getattr(IntrRepr, "__tvm_ffi_type_info__")
        names = {m.name for m in info.methods}
        assert "__ffi_repr__" in names
        # system methods still present
        assert "__ffi_init__" in names
        assert "__ffi_shallow_copy__" in names
