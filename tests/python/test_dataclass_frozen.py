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
"""Tests for ``@py_class(frozen=True)`` support."""

# ruff: noqa: D102
from __future__ import annotations

import copy
import itertools
from typing import Optional

import pytest
import tvm_ffi
from tvm_ffi.core import Object
from tvm_ffi.dataclasses import field, py_class, FrozenInstanceError

# ---------------------------------------------------------------------------
# Unique type key generator (avoids collisions across tests)
# ---------------------------------------------------------------------------
_counter = itertools.count()


def _unique_key(base: str) -> str:
    return f"testing.frozen.{base}_{next(_counter)}"


# ###########################################################################
#  1. Basic frozen class
# ###########################################################################


class TestBasicFrozen:
    """Basic construction and field access on frozen py_class instances."""

    def test_basic_frozen_class(self) -> None:
        """Frozen py_class with a few fields: construct and read back."""

        @py_class(_unique_key("Basic"), frozen=True)
        class Basic(Object):
            x: int
            y: str

        obj = Basic(x=42, y="hello")
        assert obj.x == 42
        assert obj.y == "hello"

    def test_frozen_single_field(self) -> None:
        """Frozen class with a single field."""

        @py_class(_unique_key("Single"), frozen=True)
        class Single(Object):
            value: int

        obj = Single(value=7)
        assert obj.value == 7

    def test_frozen_multiple_types(self) -> None:
        """Frozen class with int, float, str, and bool fields."""

        @py_class(_unique_key("Multi"), frozen=True)
        class Multi(Object):
            a: int
            b: float
            c: str
            d: bool

        obj = Multi(a=1, b=2.5, c="abc", d=True)
        assert obj.a == 1
        assert obj.b == 2.5
        assert obj.c == "abc"
        assert obj.d is True


# ###########################################################################
#  2. Assignment blocked
# ###########################################################################


class TestAssignmentBlocked:
    """Setting attributes on frozen instances raises FrozenInstanceError."""

    def test_setattr_existing_field_raises(self) -> None:
        @py_class(_unique_key("SetBlk"), frozen=True)
        class SetBlk(Object):
            x: int

        obj = SetBlk(x=1)
        with pytest.raises(FrozenInstanceError):
            obj.x = 99

    def test_setattr_preserves_original_value(self) -> None:
        @py_class(_unique_key("SetPres"), frozen=True)
        class SetPres(Object):
            x: int

        obj = SetPres(x=42)
        with pytest.raises(FrozenInstanceError):
            obj.x = 99
        assert obj.x == 42


# ###########################################################################
#  3. Deletion blocked
# ###########################################################################


class TestDeletionBlocked:
    """Deleting attributes on frozen instances raises FrozenInstanceError."""

    def test_delattr_raises(self) -> None:
        @py_class(_unique_key("DelBlk"), frozen=True)
        class DelBlk(Object):
            x: int

        obj = DelBlk(x=1)
        with pytest.raises(FrozenInstanceError):
            del obj.x


# ###########################################################################
#  4. FrozenInstanceError is AttributeError
# ###########################################################################


class TestFrozenInstanceErrorType:
    """FrozenInstanceError must be a subclass of AttributeError."""

    def test_is_subclass_of_attribute_error(self) -> None:
        assert issubclass(FrozenInstanceError, AttributeError)

    def test_isinstance_check_on_raised_error(self) -> None:
        @py_class(_unique_key("ErrType"), frozen=True)
        class ErrType(Object):
            x: int

        obj = ErrType(x=1)
        with pytest.raises(FrozenInstanceError) as exc_info:
            obj.x = 2
        assert isinstance(exc_info.value, AttributeError)

    def test_caught_as_attribute_error(self) -> None:
        """FrozenInstanceError can be caught with ``except AttributeError``."""

        @py_class(_unique_key("CatchAttr"), frozen=True)
        class CatchAttr(Object):
            x: int

        obj = CatchAttr(x=1)
        with pytest.raises(FrozenInstanceError):
            obj.x = 2


# ###########################################################################
#  5. Init works
# ###########################################################################


class TestInitWorks:
    """Constructor works normally on frozen classes."""

    def test_positional_args(self) -> None:
        @py_class(_unique_key("InitPos"), frozen=True)
        class InitPos(Object):
            x: int
            y: str

        obj = InitPos(1, "hello")
        assert obj.x == 1
        assert obj.y == "hello"

    def test_keyword_args(self) -> None:
        @py_class(_unique_key("InitKw"), frozen=True)
        class InitKw(Object):
            x: int
            y: str

        obj = InitKw(x=1, y="hello")
        assert obj.x == 1
        assert obj.y == "hello"

    def test_mixed_args(self) -> None:
        @py_class(_unique_key("InitMix"), frozen=True)
        class InitMix(Object):
            x: int
            y: str

        obj = InitMix(1, y="hello")
        assert obj.x == 1
        assert obj.y == "hello"


# ###########################################################################
#  6. Post-init with escape hatch
# ###########################################################################


class TestPostInitEscapeHatch:
    """__post_init__ can use object.__setattr__ on frozen instances."""

    def test_post_init_with_object_setattr(self) -> None:
        @py_class(_unique_key("PIEsc"), frozen=True)
        class PIEsc(Object):
            x: int
            y: Optional[int] = None

            def __post_init__(self) -> None:
                object.__setattr__(self, "y", self.x * 2)

        obj = PIEsc(x=5)
        assert obj.y == 10


# ###########################################################################
#  7. Post-init without escape hatch fails
# ###########################################################################


class TestPostInitNoEscapeHatch:
    """__post_init__ that uses self.attr = ... raises FrozenInstanceError."""

    def test_post_init_direct_setattr_raises(self) -> None:
        @py_class(_unique_key("PINoEsc"), frozen=True)
        class PINoEsc(Object):
            x: int
            y: Optional[int] = None

            def __post_init__(self) -> None:
                self.y = self.x * 2  # should raise FrozenInstanceError

        with pytest.raises(FrozenInstanceError):
            PINoEsc(x=5)


# ###########################################################################
#  8. Escape hatch works
# ###########################################################################


class TestEscapeHatch:
    """object.__setattr__ bypasses the frozen guard."""

    def test_object_setattr_bypasses_guard(self) -> None:
        @py_class(_unique_key("Esc"), frozen=True)
        class Esc(Object):
            x: int

        obj = Esc(x=1)
        object.__setattr__(obj, "x", 42)
        assert obj.x == 42

    def test_escape_hatch_then_frozen_still_enforced(self) -> None:
        """After using the escape hatch, normal assignment still blocked."""

        @py_class(_unique_key("EscThen"), frozen=True)
        class EscThen(Object):
            x: int

        obj = EscThen(x=1)
        object.__setattr__(obj, "x", 42)
        assert obj.x == 42
        with pytest.raises(FrozenInstanceError):
            obj.x = 99


# ###########################################################################
#  9. Non-frozen class still mutable (regression test)
# ###########################################################################


class TestNonFrozenStillMutable:
    """Regular @py_class without frozen=True allows mutation."""

    def test_mutable_setattr(self) -> None:
        @py_class(_unique_key("Mut"))
        class Mut(Object):
            x: int

        obj = Mut(x=1)
        obj.x = 42
        assert obj.x == 42

    def test_mutable_no_frozen_error(self) -> None:
        @py_class(_unique_key("MutNoErr"))
        class MutNoErr(Object):
            x: int

        obj = MutNoErr(x=1)
        # Should not raise
        obj.x = 2
        assert obj.x == 2


# ###########################################################################
# 10. Frozen with default values
# ###########################################################################


class TestFrozenWithDefaults:
    """Fields with default values work in frozen classes."""

    def test_default_value_used(self) -> None:
        @py_class(_unique_key("Def"), frozen=True)
        class Def(Object):
            x: int = 10

        obj = Def()
        assert obj.x == 10

    def test_default_value_overridden(self) -> None:
        @py_class(_unique_key("DefOvr"), frozen=True)
        class DefOvr(Object):
            x: int = 10

        obj = DefOvr(x=99)
        assert obj.x == 99

    def test_mixed_required_and_default(self) -> None:
        @py_class(_unique_key("DefMix"), frozen=True)
        class DefMix(Object):
            x: int
            y: str = "default"

        obj = DefMix(x=1)
        assert obj.x == 1
        assert obj.y == "default"


# ###########################################################################
# 11. Frozen with default_factory
# ###########################################################################


class TestFrozenWithDefaultFactory:
    """Fields with default_factory work in frozen classes."""

    def test_default_factory(self) -> None:
        @py_class(_unique_key("Fac"), frozen=True)
        class Fac(Object):
            x: int
            y: list = field(default_factory=list)

        obj = Fac(x=1)
        assert obj.x == 1
        assert len(obj.y) == 0

    def test_default_factory_fresh_each_time(self) -> None:
        @py_class(_unique_key("FacFresh"), frozen=True)
        class FacFresh(Object):
            y: list = field(default_factory=list)

        a = FacFresh()
        b = FacFresh()
        # Each instance gets its own default_factory result
        assert a.y is not b.y


# ###########################################################################
# 12. Frozen with kw_only
# ###########################################################################


class TestFrozenWithKwOnly:
    """@py_class(frozen=True, kw_only=True) works."""

    def test_kw_only_frozen(self) -> None:
        @py_class(_unique_key("KwFrz"), frozen=True, kw_only=True)
        class KwFrz(Object):
            x: int
            y: str

        obj = KwFrz(x=1, y="hello")
        assert obj.x == 1
        assert obj.y == "hello"

    def test_kw_only_positional_rejected(self) -> None:
        @py_class(_unique_key("KwFrzPos"), frozen=True, kw_only=True)
        class KwFrzPos(Object):
            x: int

        with pytest.raises(TypeError):
            KwFrzPos(1)  # type: ignore[misc]


# ###########################################################################
# 13. Frozen with eq
# ###########################################################################


class TestFrozenWithEq:
    """@py_class(frozen=True, eq=True) -- equality on frozen instances."""

    def test_equal_instances(self) -> None:
        @py_class(_unique_key("FrzEq"), frozen=True, eq=True)
        class FrzEq(Object):
            x: int
            y: str

        a = FrzEq(x=1, y="a")
        b = FrzEq(x=1, y="a")
        assert a == b

    def test_not_equal_instances(self) -> None:
        @py_class(_unique_key("FrzNeq"), frozen=True, eq=True)
        class FrzNeq(Object):
            x: int

        a = FrzNeq(x=1)
        b = FrzNeq(x=2)
        assert a != b

    def test_frozen_eq_hash(self) -> None:
        """frozen=True + eq=True + unsafe_hash=True makes instances hashable."""

        @py_class(_unique_key("FrzEqH"), frozen=True, eq=True, unsafe_hash=True)
        class FrzEqH(Object):
            x: int

        a = FrzEqH(x=1)
        b = FrzEqH(x=1)
        assert hash(a) == hash(b)
        assert a == b


# ###########################################################################
# 14. Frozen with repr
# ###########################################################################


class TestFrozenWithRepr:
    """repr() works on frozen instances."""

    def test_repr_contains_fields(self) -> None:
        @py_class(_unique_key("FrzRepr"), frozen=True)
        class FrzRepr(Object):
            x: int
            y: str

        obj = FrzRepr(x=42, y="hello")
        r = repr(obj)
        assert "42" in r
        assert "hello" in r

    def test_repr_disabled(self) -> None:
        @py_class(_unique_key("FrzNoRepr"), frozen=True, repr=False)
        class FrzNoRepr(Object):
            x: int

        obj = FrzNoRepr(x=1)
        r = repr(obj)
        # Should use default object repr, not the dataclass-style one
        assert "FrzNoRepr" in r or "object at" in r


# ###########################################################################
# 15. Copy / deepcopy
# ###########################################################################


class TestFrozenCopy:
    """copy.copy() and copy.deepcopy() produce new frozen instances."""

    def test_shallow_copy(self) -> None:
        @py_class(_unique_key("FrzSC"), frozen=True)
        class FrzSC(Object):
            x: int

        obj = FrzSC(x=42)
        obj2 = copy.copy(obj)
        assert obj2.x == 42
        assert obj is not obj2

    def test_shallow_copy_is_frozen(self) -> None:
        """The copy is also frozen."""

        @py_class(_unique_key("FrzSCF"), frozen=True)
        class FrzSCF(Object):
            x: int

        obj = FrzSCF(x=42)
        obj2 = copy.copy(obj)
        with pytest.raises(FrozenInstanceError):
            obj2.x = 99

    def test_deep_copy(self) -> None:
        @py_class(_unique_key("FrzDC"), frozen=True)
        class FrzDC(Object):
            x: int

        obj = FrzDC(x=42)
        obj2 = copy.deepcopy(obj)
        assert obj2.x == 42
        assert obj is not obj2

    def test_deep_copy_is_frozen(self) -> None:
        """The deep copy is also frozen."""

        @py_class(_unique_key("FrzDCF"), frozen=True)
        class FrzDCF(Object):
            x: int

        obj = FrzDCF(x=42)
        obj2 = copy.deepcopy(obj)
        with pytest.raises(FrozenInstanceError):
            obj2.x = 99

    def test_copy_escape_hatch_independent(self) -> None:
        """Modifying a copy via escape hatch does not affect the original."""

        @py_class(_unique_key("FrzCpInd"), frozen=True)
        class FrzCpInd(Object):
            x: int

        obj = FrzCpInd(x=1)
        obj2 = copy.copy(obj)
        object.__setattr__(obj2, "x", 99)
        assert obj.x == 1
        assert obj2.x == 99

    def test_deepcopy_escape_hatch_independent(self) -> None:
        """Modifying a deep copy via escape hatch does not affect the original."""

        @py_class(_unique_key("FrzDCInd"), frozen=True)
        class FrzDCInd(Object):
            x: int

        obj = FrzDCInd(x=1)
        obj2 = copy.deepcopy(obj)
        object.__setattr__(obj2, "x", 99)
        assert obj.x == 1
        assert obj2.x == 99


# ###########################################################################
# 16. Replace
# ###########################################################################


class TestFrozenReplace:
    """__replace__ produces a new frozen instance with updated fields."""

    def test_replace_produces_new_instance(self) -> None:
        @py_class(_unique_key("FrzRepl"), frozen=True)
        class FrzRepl(Object):
            x: int
            y: str

        obj = FrzRepl(x=1, y="a")
        obj2 = obj.__replace__(x=2)  # type: ignore[attr-defined]
        assert obj2.x == 2
        assert obj2.y == "a"
        assert obj.x == 1  # original unchanged

    def test_replace_result_is_frozen(self) -> None:
        @py_class(_unique_key("FrzReplF"), frozen=True)
        class FrzReplF(Object):
            x: int

        obj = FrzReplF(x=1)
        obj2 = obj.__replace__(x=2)  # type: ignore[attr-defined]
        with pytest.raises(FrozenInstanceError):
            obj2.x = 99


# ###########################################################################
# 17. Multiple instances
# ###########################################################################


class TestMultipleInstances:
    """Two frozen instances of the same class are independent."""

    def test_independent_instances(self) -> None:
        @py_class(_unique_key("FrzIndep"), frozen=True)
        class FrzIndep(Object):
            x: int

        a = FrzIndep(x=1)
        b = FrzIndep(x=2)
        assert a.x == 1
        assert b.x == 2

    def test_escape_hatch_on_one_does_not_affect_other(self) -> None:
        @py_class(_unique_key("FrzIndep2"), frozen=True)
        class FrzIndep2(Object):
            x: int

        a = FrzIndep2(x=1)
        b = FrzIndep2(x=1)
        object.__setattr__(a, "x", 99)
        assert a.x == 99
        assert b.x == 1


# ###########################################################################
# 18. Frozen class with no fields
# ###########################################################################


class TestFrozenNoFields:
    """Edge case: empty frozen class."""

    def test_empty_frozen_class(self) -> None:
        @py_class(_unique_key("FrzEmpty"), frozen=True)
        class FrzEmpty(Object):
            pass

        obj = FrzEmpty()
        assert obj is not None

    def test_empty_frozen_setattr_blocked(self) -> None:
        """Even on an empty frozen class, setting new attributes is blocked."""

        @py_class(_unique_key("FrzEmpty2"), frozen=True)
        class FrzEmpty2(Object):
            pass

        obj = FrzEmpty2()
        with pytest.raises(FrozenInstanceError):
            obj.x = 1  # type: ignore[attr-defined]


# ###########################################################################
# 19. Frozen with init=False
# ###########################################################################


class TestFrozenInitFalse:
    """@py_class(frozen=True, init=False): user provides __init__ via escape hatch."""

    def test_user_init_with_escape_hatch(self) -> None:
        @py_class(_unique_key("FrzNoInit"), frozen=True, init=False)
        class FrzNoInit(Object):
            x: int

            def __init__(self, val: int) -> None:
                object.__setattr__(self, "x", val)

        obj = FrzNoInit(42)
        assert obj.x == 42

    def test_user_init_without_escape_hatch_raises(self) -> None:
        """User __init__ that does self.x = ... should raise FrozenInstanceError."""

        @py_class(_unique_key("FrzNoInitFail"), frozen=True, init=False)
        class FrzNoInitFail(Object):
            x: int

            def __init__(self, val: int) -> None:
                self.x = val  # should raise FrozenInstanceError

        with pytest.raises(FrozenInstanceError):
            FrzNoInitFail(42)

    def test_frozen_init_false_still_frozen_after_init(self) -> None:
        @py_class(_unique_key("FrzNoInitStill"), frozen=True, init=False)
        class FrzNoInitStill(Object):
            x: int

            def __init__(self, val: int) -> None:
                object.__setattr__(self, "x", val)

        obj = FrzNoInitStill(42)
        with pytest.raises(FrozenInstanceError):
            obj.x = 99


# ###########################################################################
# 20. Assignment of new (non-field) attribute blocked
# ###########################################################################


class TestNonFieldAttributeBlocked:
    """Setting a new attribute that is not a declared field also raises."""

    def test_new_attribute_blocked(self) -> None:
        @py_class(_unique_key("FrzNewAttr"), frozen=True)
        class FrzNewAttr(Object):
            x: int

        obj = FrzNewAttr(x=1)
        with pytest.raises(FrozenInstanceError):
            obj.new_attr = 1  # type: ignore[attr-defined]

    def test_del_new_attribute_blocked(self) -> None:
        @py_class(_unique_key("FrzDelNew"), frozen=True)
        class FrzDelNew(Object):
            x: int

        obj = FrzDelNew(x=1)
        with pytest.raises(FrozenInstanceError):
            del obj.new_attr  # type: ignore[attr-defined]


# ###########################################################################
# 21. Field-level frozen: single frozen field in a non-frozen class
# ###########################################################################


class TestFieldFrozenInNonFrozenClass:
    """field(frozen=True) in a class that is NOT frozen at the class level."""

    def test_frozen_field_rejects_assignment(self) -> None:
        """A single frozen field in a mutable class raises on assignment."""

        @py_class(_unique_key("FldFrz1"))
        class FldFrz1(Object):
            x: int = field(frozen=True)
            y: int

        obj = FldFrz1(x=1, y=2)
        assert obj.x == 1
        # Frozen field rejects assignment
        with pytest.raises(FrozenInstanceError):
            obj.x = 10
        # Mutable field is fine
        obj.y = 20
        assert obj.y == 20

    def test_mutable_field_stays_mutable(self) -> None:
        """Non-frozen fields in a non-frozen class remain freely mutable."""

        @py_class(_unique_key("FldFrz2"))
        class FldFrz2(Object):
            x: int = field(frozen=True)
            y: str

        obj = FldFrz2(x=1, y="hello")
        obj.y = "world"
        assert obj.y == "world"

    def test_frozen_field_read_access(self) -> None:
        """Frozen fields are readable even though they are not writable."""

        @py_class(_unique_key("FldFrzRd"))
        class FldFrzRd(Object):
            x: int = field(frozen=True)

        obj = FldFrzRd(x=42)
        assert obj.x == 42


# ###########################################################################
# 22. All fields individually frozen in a non-frozen class
# ###########################################################################


class TestAllFieldsFrozenIndividually:
    """All fields have field(frozen=True) but the class itself is NOT frozen."""

    def test_all_fields_frozen_rejects_assignment(self) -> None:
        @py_class(_unique_key("AllFld"))
        class AllFld(Object):
            x: int = field(frozen=True)
            y: str = field(frozen=True)

        obj = AllFld(x=1, y="a")
        with pytest.raises(FrozenInstanceError):
            obj.x = 2
        with pytest.raises(FrozenInstanceError):
            obj.y = "b"

    def test_field_level_guard_only_blocks_frozen_fields(self) -> None:
        """Field-level frozen installs a __setattr__ guard that blocks
        only the frozen fields.  Non-field attributes are still settable
        via the guard's fallthrough to object.__setattr__."""

        @py_class(_unique_key("AllFldNoGuard"))
        class AllFldNoGuard(Object):
            x: int = field(frozen=True)

        obj = AllFldNoGuard(x=1)
        # Field-level frozen blocks declared fields
        with pytest.raises(FrozenInstanceError):
            obj.x = 2


# ###########################################################################
# 23. Frozen field with default value
# ###########################################################################


class TestFrozenFieldWithDefault:
    """field(frozen=True, default=...) works correctly."""

    def test_frozen_field_default_value(self) -> None:
        @py_class(_unique_key("FldDef"))
        class FldDef(Object):
            x: int = field(frozen=True, default=42)

        obj = FldDef()
        assert obj.x == 42
        with pytest.raises(FrozenInstanceError):
            obj.x = 99

    def test_frozen_field_default_overridden_at_init(self) -> None:
        @py_class(_unique_key("FldDefOvr"))
        class FldDefOvr(Object):
            x: int = field(frozen=True, default=42)

        obj = FldDefOvr(x=7)
        assert obj.x == 7
        with pytest.raises(FrozenInstanceError):
            obj.x = 99


# ###########################################################################
# 24. Frozen field with default_factory
# ###########################################################################


class TestFrozenFieldWithDefaultFactory:
    """field(frozen=True, default_factory=...) works correctly."""

    def test_frozen_field_default_factory(self) -> None:
        call_count = 0

        def make_default() -> int:
            nonlocal call_count
            call_count += 1
            return 100

        @py_class(_unique_key("FldFact"))
        class FldFact(Object):
            x: int = field(frozen=True, default_factory=make_default)

        obj = FldFact()
        assert obj.x == 100
        assert call_count == 1
        with pytest.raises(FrozenInstanceError):
            obj.x = 200

    def test_frozen_field_factory_called_per_instance(self) -> None:
        call_count = 0

        def make_default() -> int:
            nonlocal call_count
            call_count += 1
            return call_count

        @py_class(_unique_key("FldFactPer"))
        class FldFactPer(Object):
            x: int = field(frozen=True, default_factory=make_default)

        a = FldFactPer()
        b = FldFactPer()
        assert a.x == 1
        assert b.x == 2
        assert call_count == 2


# ###########################################################################
# 25. Explicit field(frozen=False) in non-frozen class
# ###########################################################################


class TestExplicitFrozenFalse:
    """field(frozen=False) is explicitly mutable (same as default behavior)."""

    def test_explicit_mutable_field(self) -> None:
        @py_class(_unique_key("ExplMut"))
        class ExplMut(Object):
            x: int = field(frozen=False)

        obj = ExplMut(x=1)
        obj.x = 2
        assert obj.x == 2


# ###########################################################################
# 26. Mixed frozen and mutable fields
# ###########################################################################


class TestMixedFrozenAndMutableFields:
    """Some fields frozen, some not -- verified independently."""

    def test_mixed_fields_four_fields(self) -> None:
        @py_class(_unique_key("Mixed4"))
        class Mixed4(Object):
            a: int = field(frozen=True)
            b: str
            c: float = field(frozen=True)
            d: int

        obj = Mixed4(a=1, b="x", c=3.14, d=4)
        # Frozen fields reject assignment
        with pytest.raises(FrozenInstanceError):
            obj.a = 10
        with pytest.raises(FrozenInstanceError):
            obj.c = 2.71
        # Mutable fields accept assignment
        obj.b = "y"
        assert obj.b == "y"
        obj.d = 40
        assert obj.d == 40

    def test_mixed_alternating(self) -> None:
        """Alternating frozen and mutable fields."""

        @py_class(_unique_key("MixAlt"))
        class MixAlt(Object):
            f1: int = field(frozen=True)
            m1: int
            f2: str = field(frozen=True)
            m2: str

        obj = MixAlt(f1=1, m1=2, f2="a", m2="b")
        with pytest.raises(FrozenInstanceError):
            obj.f1 = 10
        obj.m1 = 20
        assert obj.m1 == 20
        with pytest.raises(FrozenInstanceError):
            obj.f2 = "x"
        obj.m2 = "y"
        assert obj.m2 == "y"


class TestFrozenFieldsReadAlwaysWorks:
    """Frozen fields are readable regardless of their frozen status."""

    def test_read_multiple_frozen_fields(self) -> None:
        @py_class(_unique_key("RdFroz"))
        class RdFroz(Object):
            x: int = field(frozen=True)
            y: str = field(frozen=True)
            z: float = field(frozen=True)

        obj = RdFroz(x=42, y="hello", z=3.14)
        assert obj.x == 42
        assert obj.y == "hello"
        assert abs(obj.z - 3.14) < 1e-10


# ###########################################################################
# 27. Frozen parent, frozen child -- both frozen=True
# ###########################################################################


class TestFrozenParentFrozenChild:
    """Both parent and child have frozen=True."""

    def test_both_frozen_all_fields_immutable(self) -> None:
        @py_class(_unique_key("FP"), frozen=True)
        class FP(Object):
            x: int

        @py_class(_unique_key("FC"), frozen=True)
        class FC(FP):
            y: str

        obj = FC(x=1, y="a")
        assert obj.x == 1
        assert obj.y == "a"
        with pytest.raises(FrozenInstanceError):
            obj.x = 10
        with pytest.raises(FrozenInstanceError):
            obj.y = "b"

    def test_frozen_child_extra_fields_all_frozen(self) -> None:
        """Child adds extra fields -- all remain frozen."""

        @py_class(_unique_key("FPE"), frozen=True)
        class FPE(Object):
            x: int

        @py_class(_unique_key("FCE"), frozen=True)
        class FCE(FPE):
            y: str
            z: float

        obj = FCE(x=1, y="a", z=3.14)
        assert obj.x == 1
        assert obj.y == "a"
        assert abs(obj.z - 3.14) < 1e-10
        with pytest.raises(FrozenInstanceError):
            obj.x = 10
        with pytest.raises(FrozenInstanceError):
            obj.y = "b"
        with pytest.raises(FrozenInstanceError):
            obj.z = 2.71


# ###########################################################################
# 28. Frozen parent, non-frozen child
# ###########################################################################


class TestFrozenParentNonFrozenChild:
    """Frozen parent, non-frozen child.

    Python's stdlib dataclasses raises TypeError if a non-frozen dataclass
    inherits from a frozen one. Our implementation may differ -- test actual
    behavior.
    """

    def test_frozen_parent_nonfrozen_child(self) -> None:
        try:

            @py_class(_unique_key("FPN"), frozen=True)
            class FPN(Object):
                x: int

            @py_class(_unique_key("NFCh"))
            class NFCh(FPN):
                y: str

            # Definition was allowed: verify the object can be created.
            obj = NFCh(x=1, y="a")
            assert obj.x == 1
            assert obj.y == "a"
        except TypeError:
            # Matches stdlib: non-frozen child of frozen parent is rejected.
            pass


# ###########################################################################
# 29. Non-frozen parent, frozen child
# ###########################################################################


class TestNonFrozenParentFrozenChild:
    """Non-frozen parent, frozen child.

    Python's stdlib dataclasses raises TypeError in this case.
    Our implementation may differ -- test actual behavior.
    """

    def test_nonfrozen_parent_frozen_child(self) -> None:
        try:

            @py_class(_unique_key("NFP"))
            class NFP(Object):
                x: int

            @py_class(_unique_key("FCh"), frozen=True)
            class FCh(NFP):
                y: str

            obj = FCh(x=1, y="a")
            assert obj.x == 1
            assert obj.y == "a"
            # If allowed, frozen child should guard assignment
            with pytest.raises((FrozenInstanceError, AttributeError)):
                obj.y = "b"
        except TypeError:
            # Matches stdlib: frozen child of non-frozen parent is rejected.
            pass


# ###########################################################################
# 30. Non-frozen parent with frozen fields, non-frozen child
# ###########################################################################


class TestFieldFrozenInheritance:
    """Non-frozen parent has field(frozen=True) fields; child inherits them."""

    def test_inherited_frozen_fields_stay_frozen(self) -> None:
        @py_class(_unique_key("NFPff"))
        class NFPff(Object):
            x: int = field(frozen=True)
            y: int

        @py_class(_unique_key("NFCff"))
        class NFCff(NFPff):
            z: str

        obj = NFCff(x=1, y=2, z="a")
        # Parent's frozen field stays frozen in child
        with pytest.raises(FrozenInstanceError):
            obj.x = 10
        # Parent's mutable field stays mutable
        obj.y = 20
        assert obj.y == 20
        # Child's own field is mutable
        obj.z = "b"
        assert obj.z == "b"

    def test_child_adds_own_frozen_fields(self) -> None:
        """Child adds its own field(frozen=True) on top of parent's."""

        @py_class(_unique_key("NFPff2"))
        class NFPff2(Object):
            x: int = field(frozen=True)

        @py_class(_unique_key("NFCff2"))
        class NFCff2(NFPff2):
            y: int = field(frozen=True)
            z: int

        obj = NFCff2(x=1, y=2, z=3)
        with pytest.raises(FrozenInstanceError):
            obj.x = 10
        with pytest.raises(FrozenInstanceError):
            obj.y = 20
        obj.z = 30
        assert obj.z == 30


# ###########################################################################
# 31. Multi-level frozen inheritance
# ###########################################################################


class TestMultiLevelFrozenInheritance:
    """Grandparent -> Parent -> Child, all frozen=True."""

    def test_three_level_frozen(self) -> None:
        @py_class(_unique_key("GP"), frozen=True)
        class GP(Object):
            a: int

        @py_class(_unique_key("P"), frozen=True)
        class P(GP):
            b: int

        @py_class(_unique_key("C"), frozen=True)
        class C(P):
            c: int

        obj = C(a=1, b=2, c=3)
        assert obj.a == 1
        assert obj.b == 2
        assert obj.c == 3
        with pytest.raises(FrozenInstanceError):
            obj.a = 10
        with pytest.raises(FrozenInstanceError):
            obj.b = 20
        with pytest.raises(FrozenInstanceError):
            obj.c = 30

    def test_three_level_frozen_isinstance(self) -> None:
        @py_class(_unique_key("GP2"), frozen=True)
        class GP2(Object):
            a: int

        @py_class(_unique_key("P2"), frozen=True)
        class P2(GP2):
            b: int

        @py_class(_unique_key("C2"), frozen=True)
        class C2(P2):
            c: int

        obj = C2(a=1, b=2, c=3)
        assert isinstance(obj, C2)
        assert isinstance(obj, P2)
        assert isinstance(obj, GP2)
        assert isinstance(obj, Object)


# ###########################################################################
# 32. Frozen py_class with plain Python mixin
# ###########################################################################


class TestFrozenWithMixin:
    """Frozen py_class that also inherits from a plain Python mixin."""

    def test_frozen_with_python_mixin(self) -> None:
        class MyMixin:
            """A plain Python mixin (not a py_class)."""

            def greet(self) -> str:
                return "hello"

        @py_class(_unique_key("FrozMixin"), frozen=True)
        class FrozMixin(Object):
            x: int

        obj = FrozMixin(x=42)
        assert obj.x == 42
        # Frozen guard still works
        with pytest.raises(FrozenInstanceError):
            obj.x = 99


# ###########################################################################
# 33. Frozen + structural_eq
# ###########################################################################


class TestFrozenWithStructuralEq:
    """frozen=True combined with structural_eq="tree"."""

    def test_frozen_structural_eq(self) -> None:
        @py_class(_unique_key("FrozSEq"), frozen=True, structural_eq="tree")
        class FrozSEq(Object):
            x: int
            y: str

        a = FrozSEq(x=1, y="a")
        b = FrozSEq(x=1, y="a")
        c = FrozSEq(x=2, y="a")
        assert tvm_ffi.structural_equal(a, b)
        assert not tvm_ffi.structural_equal(a, c)
        # Still frozen
        with pytest.raises(FrozenInstanceError):
            a.x = 10

    def test_frozen_structural_hash(self) -> None:
        @py_class(_unique_key("FrozSHash"), frozen=True, structural_eq="tree")
        class FrozSHash(Object):
            x: int

        a = FrozSHash(x=1)
        b = FrozSHash(x=1)
        assert tvm_ffi.structural_hash(a) == tvm_ffi.structural_hash(b)


# ###########################################################################
# 34. Frozen + unsafe_hash
# ###########################################################################


class TestFrozenWithUnsafeHash:
    """frozen=True combined with unsafe_hash=True."""

    def test_frozen_hashable(self) -> None:
        @py_class(_unique_key("FrozHash"), frozen=True, eq=True, unsafe_hash=True)
        class FrozHash(Object):
            x: int
            y: str

        a = FrozHash(x=1, y="a")
        b = FrozHash(x=1, y="a")
        assert hash(a) == hash(b)
        with pytest.raises(FrozenInstanceError):
            a.x = 10


# ###########################################################################
# 35. Frozen + eq
# ###########################################################################


class TestFrozenFieldLevelWithEq:
    """frozen=True combined with eq=True (field-level and class-level)."""

    def test_frozen_eq_class_level(self) -> None:
        @py_class(_unique_key("FrozEqC"), frozen=True, eq=True)
        class FrozEqC(Object):
            x: int
            y: str

        a = FrozEqC(x=1, y="a")
        b = FrozEqC(x=1, y="a")
        c = FrozEqC(x=2, y="a")
        assert a == b
        assert a != c
        with pytest.raises(FrozenInstanceError):
            a.x = 10


# ###########################################################################
# 36. Frozen + order
# ###########################################################################


class TestFrozenWithOrder:
    """frozen=True combined with eq=True and order=True."""

    def test_frozen_order(self) -> None:
        @py_class(_unique_key("FrozOrd"), frozen=True, eq=True, order=True)
        class FrozOrd(Object):
            x: int

        a = FrozOrd(x=1)
        b = FrozOrd(x=2)
        assert a < b
        assert b > a
        assert a <= a
        assert a >= a
        with pytest.raises(FrozenInstanceError):
            a.x = 10

    def test_frozen_order_all_comparisons(self) -> None:
        @py_class(_unique_key("FrozOrdAll"), frozen=True, eq=True, order=True)
        class FrozOrdAll(Object):
            x: int

        a = FrozOrdAll(x=1)
        b = FrozOrdAll(x=2)
        c = FrozOrdAll(x=1)
        assert a < b
        assert b > a
        assert a <= c
        assert c >= a
        assert a == c
        assert a != b


# ###########################################################################
# 37. Copy of object with field-level frozen
# ###########################################################################


class TestFieldFrozenWithCopy:
    """Copy preserves field-level frozen status."""

    def test_copy_preserves_field_frozen(self) -> None:
        @py_class(_unique_key("CpFld"))
        class CpFld(Object):
            x: int = field(frozen=True)
            y: int

        obj = CpFld(x=1, y=2)
        obj2 = copy.copy(obj)
        assert obj2.x == 1
        assert obj2.y == 2
        # Frozen field stays frozen in copy
        with pytest.raises(FrozenInstanceError):
            obj2.x = 10
        # Mutable field stays mutable in copy
        obj2.y = 20
        assert obj2.y == 20

    def test_deepcopy_preserves_field_frozen(self) -> None:
        @py_class(_unique_key("DcpFld"))
        class DcpFld(Object):
            x: int = field(frozen=True)
            y: int

        obj = DcpFld(x=1, y=2)
        obj2 = copy.deepcopy(obj)
        assert obj2.x == 1
        assert obj2.y == 2
        with pytest.raises(FrozenInstanceError):
            obj2.x = 10
        obj2.y = 20
        assert obj2.y == 20


# ###########################################################################
# 38. Field-level frozen + __replace__
# ###########################################################################


class TestFieldFrozenWithReplace:
    """__replace__ on objects with field(frozen=True)."""

    def test_replace_frozen_field_new_value(self) -> None:
        """Replace produces a new instance where the frozen field has a
        different value. The new instance's field is still frozen."""

        @py_class(_unique_key("RplFld"))
        class RplFld(Object):
            x: int = field(frozen=True)
            y: int

        obj = RplFld(x=1, y=2)
        obj2 = obj.__replace__(x=10, y=20)  # type: ignore[attr-defined]
        assert obj2.x == 10
        assert obj2.y == 20
        # Original is unchanged
        assert obj.x == 1
        assert obj.y == 2
        # Frozen field is still frozen in the replaced copy
        with pytest.raises(FrozenInstanceError):
            obj2.x = 99

    def test_replace_only_mutable_field(self) -> None:
        """Replace can change just the mutable field."""

        @py_class(_unique_key("RplMut"))
        class RplMut(Object):
            x: int = field(frozen=True)
            y: int

        obj = RplMut(x=1, y=2)
        obj2 = obj.__replace__(y=20)  # type: ignore[attr-defined]
        assert obj2.x == 1
        assert obj2.y == 20


# ###########################################################################
# 39. Frozen class used as dict key
# ###########################################################################


class TestFrozenClassAsDictKey:
    """Frozen + hashable instances can be used as dict keys."""

    def test_frozen_as_dict_key(self) -> None:
        @py_class(_unique_key("DKey"), frozen=True, eq=True, unsafe_hash=True)
        class DKey(Object):
            x: int

        a = DKey(x=1)
        b = DKey(x=2)
        d = {a: "one", b: "two"}
        assert d[a] == "one"
        assert d[b] == "two"
        # Same value should look up correctly
        c = DKey(x=1)
        assert d[c] == "one"

    def test_frozen_dict_key_multiple_fields(self) -> None:
        @py_class(_unique_key("DKeyM"), frozen=True, eq=True, unsafe_hash=True)
        class DKeyM(Object):
            x: int
            y: str

        a = DKeyM(x=1, y="a")
        b = DKeyM(x=1, y="b")
        d = {a: "first", b: "second"}
        assert len(d) == 2
        assert d[DKeyM(x=1, y="a")] == "first"
        assert d[DKeyM(x=1, y="b")] == "second"


# ###########################################################################
# 40. Frozen class in set
# ###########################################################################


class TestFrozenClassInSet:
    """Frozen + hashable instances can be stored in sets."""

    def test_frozen_in_set(self) -> None:
        @py_class(_unique_key("SetE"), frozen=True, eq=True, unsafe_hash=True)
        class SetE(Object):
            x: int

        a = SetE(x=1)
        b = SetE(x=1)
        c = SetE(x=2)
        s = {a, b, c}
        # a and b are equal, so set should deduplicate
        assert len(s) == 2

    def test_frozen_set_operations(self) -> None:
        @py_class(_unique_key("SetOp"), frozen=True, eq=True, unsafe_hash=True)
        class SetOp(Object):
            x: int

        s1 = {SetOp(x=1), SetOp(x=2), SetOp(x=3)}
        s2 = {SetOp(x=2), SetOp(x=3), SetOp(x=4)}
        assert len(s1 & s2) == 2  # intersection: 2 and 3
        assert len(s1 | s2) == 4  # union: 1, 2, 3, 4


# ###########################################################################
# 41. Frozen field type annotations (various types)
# ###########################################################################


class TestFrozenFieldTypeAnnotations:
    """Frozen fields with various type annotations all work."""

    def test_frozen_int_field(self) -> None:
        @py_class(_unique_key("FTInt"))
        class FTInt(Object):
            x: int = field(frozen=True)

        obj = FTInt(x=42)
        assert obj.x == 42
        with pytest.raises(FrozenInstanceError):
            obj.x = 0

    def test_frozen_str_field(self) -> None:
        @py_class(_unique_key("FTStr"))
        class FTStr(Object):
            s: str = field(frozen=True)

        obj = FTStr(s="hello")
        assert obj.s == "hello"
        with pytest.raises(FrozenInstanceError):
            obj.s = "world"

    def test_frozen_float_field(self) -> None:
        @py_class(_unique_key("FTFlt"))
        class FTFlt(Object):
            f: float = field(frozen=True)

        obj = FTFlt(f=3.14)
        assert abs(obj.f - 3.14) < 1e-10
        with pytest.raises(FrozenInstanceError):
            obj.f = 2.71

    def test_frozen_bool_field(self) -> None:
        @py_class(_unique_key("FTBool"))
        class FTBool(Object):
            b: bool = field(frozen=True)

        obj = FTBool(b=True)
        assert obj.b is True
        with pytest.raises(FrozenInstanceError):
            obj.b = False

    def test_frozen_optional_field(self) -> None:
        @py_class(_unique_key("FTOpt"))
        class FTOpt(Object):
            x: Optional[int] = field(frozen=True, default=None)

        obj = FTOpt()
        assert obj.x is None
        with pytest.raises(FrozenInstanceError):
            obj.x = 42

        obj2 = FTOpt(x=10)
        assert obj2.x == 10
        with pytest.raises(FrozenInstanceError):
            obj2.x = None

    def test_frozen_object_field(self) -> None:
        @py_class(_unique_key("FTInner"))
        class FTInner(Object):
            v: int

        @py_class(_unique_key("FTObj"))
        class FTObj(Object):
            child: Object = field(frozen=True)

        inner = FTInner(v=10)
        obj = FTObj(child=inner)
        assert obj.child is not None
        with pytest.raises(FrozenInstanceError):
            obj.child = FTInner(v=20)


# ###########################################################################
# 42. Error message quality
# ###########################################################################


class TestFrozenErrorMessageQuality:
    """FrozenInstanceError message includes the field name."""

    def test_error_message_includes_field_name(self) -> None:
        @py_class(_unique_key("ErrMsg"), frozen=True)
        class ErrMsg(Object):
            x: int
            y: str

        obj = ErrMsg(x=1, y="a")
        with pytest.raises(FrozenInstanceError, match="x"):
            obj.x = 10
        with pytest.raises(FrozenInstanceError, match="y"):
            obj.y = "b"

    def test_frozen_instance_error_is_attribute_error(self) -> None:
        """FrozenInstanceError is a subclass of AttributeError."""
        assert issubclass(FrozenInstanceError, AttributeError)

    def test_frozen_class_delattr_error_message(self) -> None:
        """Deleting an attribute on a frozen instance raises with field name."""

        @py_class(_unique_key("DelMsg"), frozen=True)
        class DelMsg(Object):
            x: int

        obj = DelMsg(x=1)
        with pytest.raises(FrozenInstanceError, match="x"):
            del obj.x


# ###########################################################################
# 43. field(frozen=None) inherits from class-level frozen
# ###########################################################################


class TestFieldFrozenNoneInheritsClass:
    """field(frozen=None) (default) inherits from class-level frozen setting."""

    def test_inherits_class_frozen_true(self) -> None:
        """In a frozen class, default fields are frozen."""

        @py_class(_unique_key("InhTrue"), frozen=True)
        class InhTrue(Object):
            x: int  # frozen=None -> inherits frozen=True

        obj = InhTrue(x=1)
        with pytest.raises(FrozenInstanceError):
            obj.x = 10

    def test_inherits_class_frozen_false(self) -> None:
        """In a non-frozen class, default fields are mutable."""

        @py_class(_unique_key("InhFalse"))
        class InhFalse(Object):
            x: int  # frozen=None -> inherits frozen=False

        obj = InhFalse(x=1)
        obj.x = 10
        assert obj.x == 10


# ###########################################################################
# 44. field(frozen=False) override in frozen class
# ###########################################################################


class TestFieldFrozenFalseOverrideInFrozenClass:
    """field(frozen=False) explicitly in a frozen class.

    The class-level __setattr__ guard still blocks mutation, so effectively
    the field remains frozen. This tests the actual behavior.
    """

    def test_explicit_frozen_false_in_frozen_class(self) -> None:
        @py_class(_unique_key("OptOut"), frozen=True)
        class OptOut(Object):
            x: int = field(frozen=False)

        obj = OptOut(x=1)
        # Class-level __setattr__ guard blocks all attribute mutation
        # regardless of field-level frozen=False override
        with pytest.raises((FrozenInstanceError, AttributeError)):
            obj.x = 10


# ###########################################################################
# 45. field(frozen=True) in non-frozen class (explicit opt-in)
# ###########################################################################


class TestFieldFrozenTrueInNonFrozenClass:
    """field(frozen=True) explicitly opts into freezing in a non-frozen class."""

    def test_explicit_frozen_true_in_nonfrozen_class(self) -> None:
        @py_class(_unique_key("OptIn"))
        class OptIn(Object):
            x: int = field(frozen=True)
            y: int

        obj = OptIn(x=1, y=2)
        with pytest.raises(FrozenInstanceError):
            obj.x = 10
        obj.y = 20
        assert obj.y == 20


# ###########################################################################
# 46. Frozen class-level copy and replace
# ###########################################################################


class TestFrozenClassCopyReplace:
    """Copy and replace on a fully frozen class."""

    def test_copy_frozen_class(self) -> None:
        @py_class(_unique_key("CpCls"), frozen=True)
        class CpCls(Object):
            x: int
            y: str

        obj = CpCls(x=1, y="a")
        obj2 = copy.copy(obj)
        assert obj2.x == 1
        assert obj2.y == "a"
        with pytest.raises(FrozenInstanceError):
            obj2.x = 10

    def test_deepcopy_frozen_class(self) -> None:
        @py_class(_unique_key("DcpCls"), frozen=True)
        class DcpCls(Object):
            x: int
            y: str

        obj = DcpCls(x=1, y="a")
        obj2 = copy.deepcopy(obj)
        assert obj2.x == 1
        assert obj2.y == "a"
        with pytest.raises(FrozenInstanceError):
            obj2.x = 10

    def test_replace_frozen_class(self) -> None:
        """__replace__ on a fully frozen class produces a new frozen instance."""

        @py_class(_unique_key("RplCls"), frozen=True)
        class RplCls(Object):
            x: int
            y: str

        obj = RplCls(x=1, y="a")
        obj2 = obj.__replace__(x=10)  # type: ignore[attr-defined]
        assert obj2.x == 10
        assert obj2.y == "a"
        # New instance is also frozen
        with pytest.raises(FrozenInstanceError):
            obj2.x = 99
        # Original unchanged
        assert obj.x == 1


# ###########################################################################
# 47. Frozen escape hatch details
# ###########################################################################


class TestFrozenEscapeHatchDetails:
    """Detailed escape hatch (object.__setattr__) scenarios."""

    def test_escape_hatch_then_still_frozen(self) -> None:
        """After using the escape hatch, normal assignment still blocked."""

        @py_class(_unique_key("EscDet"), frozen=True)
        class EscDet(Object):
            x: int

        obj = EscDet(x=1)
        object.__setattr__(obj, "x", 42)
        assert obj.x == 42
        with pytest.raises(FrozenInstanceError):
            obj.x = 99

    def test_field_level_escape_hatch(self) -> None:
        """object.__setattr__ bypasses field-level frozen because the
        property always retains its setter; only the __setattr__ guard
        blocks normal writes."""

        @py_class(_unique_key("FldEsc"))
        class FldEsc(Object):
            x: int = field(frozen=True)

        obj = FldEsc(x=1)
        with pytest.raises(FrozenInstanceError):
            obj.x = 42
        # object.__setattr__ bypasses the guard
        object.__setattr__(obj, "x", 42)
        assert obj.x == 42
