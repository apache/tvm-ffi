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
"""Tests for frozen support in ``@py_class``."""

# ruff: noqa: D102
from __future__ import annotations

import copy
import itertools
from typing import Dict, List, Optional

import pytest
from tvm_ffi.core import FFIProperty, Object  # ty: ignore[unresolved-import]
from tvm_ffi.dataclasses import field, py_class

_counter = itertools.count()


def _unique_key(base: str) -> str:
    return f"testing.frozen.{base}_{next(_counter)}"


# ---------------------------------------------------------------------------
#  Basic frozen class
# ---------------------------------------------------------------------------
class TestFrozenClassBasic:
    def test_init_works_normally(self) -> None:
        @py_class(_unique_key("basic_init"), frozen=True)
        class Pt(Object):
            x: int
            y: int

        p = Pt(1, 2)
        assert p.x == 1
        assert p.y == 2

    def test_assignment_blocked_after_init(self) -> None:
        @py_class(_unique_key("basic_blocked"), frozen=True)
        class Pt(Object):
            x: int
            y: int

        p = Pt(1, 2)
        with pytest.raises(AttributeError):
            p.x = 10  # ty: ignore[invalid-assignment]

    def test_all_fields_blocked(self) -> None:
        @py_class(_unique_key("all_blocked"), frozen=True)
        class Rec(Object):
            a: int
            b: str
            c: float

        r = Rec(1, "hi", 3.0)
        for attr in ("a", "b", "c"):
            with pytest.raises(AttributeError):
                setattr(r, attr, None)

    def test_reading_fields_works(self) -> None:
        @py_class(_unique_key("read_ok"), frozen=True)
        class Pt(Object):
            x: int

        p = Pt(42)
        assert p.x == 42

    def test_del_blocked(self) -> None:
        @py_class(_unique_key("del_blocked"), frozen=True)
        class Pt(Object):
            x: int

        p = Pt(1)
        with pytest.raises(AttributeError):
            del p.x


# ---------------------------------------------------------------------------
#  Frozen with various field types
# ---------------------------------------------------------------------------
class TestFrozenFieldTypes:
    def test_frozen_optional_field(self) -> None:
        @py_class(_unique_key("opt_field"), frozen=True)
        class Opt(Object):
            v: Optional[int]  # noqa: UP045

        o1 = Opt(None)
        assert o1.v is None
        o2 = Opt(42)
        assert o2.v == 42
        with pytest.raises(AttributeError):
            o1.v = 5  # ty: ignore[invalid-assignment]

    def test_frozen_object_field(self) -> None:
        @py_class(_unique_key("obj_inner"), frozen=True)
        class Inner(Object):
            val: int

        @py_class(_unique_key("obj_outer"), frozen=True)
        class Outer(Object):
            child: Inner

        inner = Inner(10)
        outer = Outer(inner)
        assert outer.child.val == 10
        with pytest.raises(AttributeError):
            outer.child = Inner(20)  # ty: ignore[invalid-assignment]

    def test_frozen_list_field(self) -> None:
        @py_class(_unique_key("list_field"), frozen=True)
        class HasList(Object):
            items: List[int]  # noqa: UP006

        h = HasList([1, 2, 3])
        assert len(h.items) == 3
        with pytest.raises(AttributeError):
            h.items = [4, 5]  # ty: ignore[invalid-assignment]

    def test_frozen_dict_field(self) -> None:
        @py_class(_unique_key("dict_field"), frozen=True)
        class HasDict(Object):
            mapping: Dict[str, int]  # noqa: UP006

        h = HasDict({"a": 1})
        assert h.mapping["a"] == 1
        with pytest.raises(AttributeError):
            h.mapping = {"b": 2}  # ty: ignore[invalid-assignment]


# ---------------------------------------------------------------------------
#  Frozen with defaults
# ---------------------------------------------------------------------------
class TestFrozenDefaults:
    def test_frozen_field_with_default(self) -> None:
        @py_class(_unique_key("def_val"), frozen=True)
        class Cfg(Object):
            x: int = 10

        c = Cfg()
        assert c.x == 10
        with pytest.raises(AttributeError):
            c.x = 20  # ty: ignore[invalid-assignment]

    def test_frozen_field_with_default_factory(self) -> None:
        @py_class(_unique_key("def_factory"), frozen=True)
        class Cfg(Object):
            items: List[int] = field(default_factory=list)  # noqa: UP006

        c = Cfg()
        assert len(c.items) == 0
        with pytest.raises(AttributeError):
            c.items = [1]  # ty: ignore[invalid-assignment]

    def test_frozen_init_false_with_default(self) -> None:
        @py_class(_unique_key("init_false_def"), frozen=True)
        class Cfg(Object):
            x: int
            tag: str = field(default="default", init=False)

        c = Cfg(5)
        assert c.tag == "default"
        with pytest.raises(AttributeError):
            c.tag = "other"  # ty: ignore[invalid-assignment]


# ---------------------------------------------------------------------------
#  Per-field frozen on a non-frozen class
# ---------------------------------------------------------------------------
class TestPerFieldFrozen:
    def test_field_frozen_true_on_mutable_class(self) -> None:
        @py_class(_unique_key("per_field"))
        class Rec(Object):
            mutable: int
            immutable: int = field(frozen=True)

        r = Rec(1, 2)
        r.mutable = 10  # OK
        assert r.mutable == 10
        with pytest.raises(AttributeError):
            r.immutable = 20

    def test_field_frozen_true_init_works(self) -> None:
        @py_class(_unique_key("per_field_init"))
        class Rec(Object):
            x: int = field(frozen=True)

        r = Rec(42)
        assert r.x == 42

    def test_multiple_mixed_fields(self) -> None:
        @py_class(_unique_key("mixed_fields"))
        class Rec(Object):
            a: int = field(frozen=True)
            b: int
            c: int = field(frozen=True)

        r = Rec(1, 2, 3)
        r.b = 20  # OK
        with pytest.raises(AttributeError):
            r.a = 10
        with pytest.raises(AttributeError):
            r.c = 30


# ---------------------------------------------------------------------------
#  Escape hatch: type(obj).field.set(obj, val)
# ---------------------------------------------------------------------------
class TestEscapeHatch:
    def test_escape_hatch_sets_frozen_field(self) -> None:
        @py_class(_unique_key("esc_basic"), frozen=True)
        class Pt(Object):
            x: int

        p = Pt(1)
        type(p).x.set(p, 99)  # ty: ignore[unresolved-attribute]
        assert p.x == 99

    def test_escape_hatch_on_field_level_frozen(self) -> None:
        @py_class(_unique_key("esc_field"))
        class Rec(Object):
            val: int = field(frozen=True)

        r = Rec(5)
        type(r).val.set(r, 50)  # ty: ignore[unresolved-attribute]
        assert r.val == 50

    def test_escape_hatch_multiple_fields(self) -> None:
        @py_class(_unique_key("esc_multi"), frozen=True)
        class Pt(Object):
            x: int
            y: int

        p = Pt(1, 2)
        type(p).x.set(p, 10)  # ty: ignore[unresolved-attribute]
        type(p).y.set(p, 20)  # ty: ignore[unresolved-attribute]
        assert p.x == 10
        assert p.y == 20

    def test_escape_hatch_preserves_type_coercion(self) -> None:
        @py_class(_unique_key("esc_coerce"), frozen=True)
        class HasList(Object):
            items: List[int]  # noqa: UP006

        h = HasList([1])
        type(h).items.set(h, [10, 20])  # ty: ignore[unresolved-attribute]
        assert len(h.items) == 2

    def test_regular_setattr_still_blocked_after_escape_hatch(self) -> None:
        @py_class(_unique_key("esc_still_frozen"), frozen=True)
        class Pt(Object):
            x: int

        p = Pt(1)
        type(p).x.set(p, 99)  # ty: ignore[unresolved-attribute]
        with pytest.raises(AttributeError):
            p.x = 100  # ty: ignore[invalid-assignment]

    def test_escape_hatch_on_mutable_field_also_works(self) -> None:
        @py_class(_unique_key("esc_mutable"))
        class Pt(Object):
            x: int

        p = Pt(1)
        type(p).x.set(p, 99)  # ty: ignore[unresolved-attribute]
        assert p.x == 99


# ---------------------------------------------------------------------------
#  copy / deepcopy / __replace__
# ---------------------------------------------------------------------------
class TestFrozenCopy:
    def test_copy_copy_frozen_class(self) -> None:
        @py_class(_unique_key("copy_basic"), frozen=True)
        class Pt(Object):
            x: int
            y: int

        p = Pt(1, 2)
        p2 = copy.copy(p)
        assert p2.x == 1 and p2.y == 2
        assert not p.same_as(p2)

    def test_copy_copy_result_is_also_frozen(self) -> None:
        @py_class(_unique_key("copy_frozen"), frozen=True)
        class Pt(Object):
            x: int

        p2 = copy.copy(Pt(1))
        with pytest.raises(AttributeError):
            p2.x = 10  # ty: ignore[invalid-assignment]

    def test_deepcopy_frozen_class(self) -> None:
        @py_class(_unique_key("deepcopy"), frozen=True)
        class Pt(Object):
            x: int

        p = Pt(42)
        p2 = copy.deepcopy(p)
        assert p2.x == 42
        assert not p.same_as(p2)

    def test_deepcopy_result_is_also_frozen(self) -> None:
        @py_class(_unique_key("deepcopy_frozen"), frozen=True)
        class Pt(Object):
            x: int

        p2 = copy.deepcopy(Pt(1))
        with pytest.raises(AttributeError):
            p2.x = 10  # ty: ignore[invalid-assignment]

    def test_replace_on_frozen_class(self) -> None:
        @py_class(_unique_key("replace"), frozen=True)
        class Pt(Object):
            x: int
            y: int

        p = Pt(1, 2)
        p2 = p.__replace__(x=10)  # ty: ignore[unresolved-attribute]
        assert p2.x == 10 and p2.y == 2
        assert p.x == 1  # original unchanged

    def test_replace_multiple_fields(self) -> None:
        @py_class(_unique_key("replace_multi"), frozen=True)
        class Pt(Object):
            x: int
            y: int

        p = Pt(1, 2)
        p2 = p.__replace__(x=10, y=20)  # ty: ignore[unresolved-attribute]
        assert p2.x == 10 and p2.y == 20

    def test_replace_result_is_frozen(self) -> None:
        @py_class(_unique_key("replace_frozen"), frozen=True)
        class Pt(Object):
            x: int

        p2 = Pt(1).__replace__(x=10)  # ty: ignore[unresolved-attribute]
        with pytest.raises(AttributeError):
            p2.x = 99


# ---------------------------------------------------------------------------
#  Inheritance
# ---------------------------------------------------------------------------
class TestFrozenInheritance:
    def test_frozen_parent_mutable_child(self) -> None:
        @py_class(_unique_key("inh_parent_frozen"), frozen=True)
        class Parent(Object):
            a: int

        @py_class(_unique_key("inh_child_mutable"))
        class Child(Parent):  # ty: ignore[invalid-frozen-dataclass-subclass]
            b: int

        c = Child(1, 2)
        assert c.a == 1 and c.b == 2
        # Parent field stays frozen (property installed by Parent class)
        with pytest.raises(AttributeError):
            c.a = 10  # ty: ignore[invalid-assignment]
        # Child's own field is mutable
        c.b = 20  # ty: ignore[invalid-assignment]
        assert c.b == 20

    def test_frozen_parent_frozen_child(self) -> None:
        @py_class(_unique_key("inh_both_frozen_p"), frozen=True)
        class Parent(Object):
            a: int

        @py_class(_unique_key("inh_both_frozen_c"), frozen=True)
        class Child(Parent):
            b: int

        c = Child(1, 2)
        with pytest.raises(AttributeError):
            c.a = 10  # ty: ignore[invalid-assignment]
        with pytest.raises(AttributeError):
            c.b = 20  # ty: ignore[invalid-assignment]

    def test_mutable_parent_frozen_child(self) -> None:
        @py_class(_unique_key("inh_parent_mutable"))
        class Parent(Object):
            a: int

        @py_class(_unique_key("inh_child_frozen"), frozen=True)
        class Child(Parent):  # ty: ignore[invalid-frozen-dataclass-subclass]
            b: int

        c = Child(1, 2)
        # Parent field is mutable (property installed by Parent class)
        c.a = 10  # ty: ignore[invalid-assignment]
        assert c.a == 10
        # Child's own field is frozen
        with pytest.raises(AttributeError):
            c.b = 20  # ty: ignore[invalid-assignment]

    def test_three_level_frozen_inheritance(self) -> None:
        @py_class(_unique_key("inh_l1"), frozen=True)
        class L1(Object):
            a: int

        @py_class(_unique_key("inh_l2"), frozen=True)
        class L2(L1):
            b: int

        @py_class(_unique_key("inh_l3"), frozen=True)
        class L3(L2):
            c: int

        obj = L3(1, 2, 3)
        assert obj.a == 1 and obj.b == 2 and obj.c == 3
        for attr in ("a", "b", "c"):
            with pytest.raises(AttributeError):
                setattr(obj, attr, 99)

    def test_escape_hatch_on_inherited_frozen_field(self) -> None:
        @py_class(_unique_key("inh_esc_p"), frozen=True)
        class Parent(Object):
            a: int

        @py_class(_unique_key("inh_esc_c"), frozen=True)
        class Child(Parent):
            b: int

        c = Child(1, 2)
        # Escape hatch for parent field must go through Parent class descriptor
        Parent.a.set(c, 10)  # ty: ignore[unresolved-attribute]
        assert c.a == 10

    def test_replace_on_inherited_frozen(self) -> None:
        @py_class(_unique_key("inh_replace_p"), frozen=True)
        class Parent(Object):
            a: int

        @py_class(_unique_key("inh_replace_c"), frozen=True)
        class Child(Parent):
            b: int

        c = Child(1, 2)
        c2 = c.__replace__(a=10, b=20)  # ty: ignore[unresolved-attribute]
        assert c2.a == 10 and c2.b == 20


# ---------------------------------------------------------------------------
#  kw_only + frozen
# ---------------------------------------------------------------------------
class TestFrozenKwOnly:
    def test_frozen_with_kw_only(self) -> None:
        @py_class(_unique_key("kw_frozen"), frozen=True, kw_only=True)
        class Cfg(Object):
            x: int
            y: int

        c = Cfg(x=1, y=2)
        assert c.x == 1 and c.y == 2
        with pytest.raises(AttributeError):
            c.x = 10  # ty: ignore[invalid-assignment]


# ---------------------------------------------------------------------------
#  __post_init__ interaction
# ---------------------------------------------------------------------------
class TestFrozenPostInit:
    def test_post_init_called(self) -> None:
        log: list[bool] = []

        @py_class(_unique_key("post_init_called"), frozen=True)
        class Pt(Object):
            x: int

            def __post_init__(self) -> None:
                log.append(True)

        Pt(1)
        assert log == [True]

    def test_post_init_can_read_fields(self) -> None:
        captured: list[int] = []

        @py_class(_unique_key("post_init_read"), frozen=True)
        class Pt(Object):
            x: int

            def __post_init__(self) -> None:
                captured.append(self.x)

        Pt(42)
        assert captured == [42]

    def test_post_init_cannot_set_frozen_fields(self) -> None:
        @py_class(_unique_key("post_init_set"), frozen=True)
        class Pt(Object):
            x: int

            def __post_init__(self) -> None:
                self.x = 999  # should fail  # ty: ignore[invalid-assignment]

        with pytest.raises(AttributeError):
            Pt(1)

    def test_post_init_can_use_escape_hatch(self) -> None:
        @py_class(_unique_key("post_init_esc"), frozen=True)
        class Pt(Object):
            x: int

            def __post_init__(self) -> None:
                type(self).x.set(self, self.x * 2)  # ty: ignore[unresolved-attribute]

        p = Pt(5)
        assert p.x == 10


# ---------------------------------------------------------------------------
#  eq / hash + frozen
# ---------------------------------------------------------------------------
class TestFrozenEqHash:
    def test_frozen_with_eq(self) -> None:
        @py_class(_unique_key("eq"), frozen=True, eq=True)
        class Pt(Object):
            x: int

        assert Pt(1) == Pt(1)
        assert Pt(1) != Pt(2)

    def test_frozen_with_hash(self) -> None:
        @py_class(_unique_key("hash"), frozen=True, eq=True, unsafe_hash=True)
        class Pt(Object):
            x: int

        assert hash(Pt(1)) == hash(Pt(1))
        s = {Pt(1), Pt(2)}
        assert len(s) == 2


# ---------------------------------------------------------------------------
#  FFIProperty descriptor checks
# ---------------------------------------------------------------------------
class TestFFIPropertyDescriptor:
    def test_frozen_field_is_ffi_property(self) -> None:
        @py_class(_unique_key("desc_type"), frozen=True)
        class Pt(Object):
            x: int

        assert isinstance(Pt.__dict__["x"], FFIProperty)

    def test_frozen_field_fset_is_none(self) -> None:
        @py_class(_unique_key("desc_fset"), frozen=True)
        class Pt(Object):
            x: int

        assert Pt.__dict__["x"].fset is None

    def test_mutable_field_is_ffi_property(self) -> None:
        @py_class(_unique_key("desc_mutable"))
        class Pt(Object):
            x: int

        assert isinstance(Pt.__dict__["x"], FFIProperty)

    def test_mutable_field_fset_is_not_none(self) -> None:
        @py_class(_unique_key("desc_mutable_fset"))
        class Pt(Object):
            x: int

        assert Pt.__dict__["x"].fset is not None

    def test_mutable_field_set_method_also_works(self) -> None:
        @py_class(_unique_key("desc_mutable_set"))
        class Pt(Object):
            x: int

        p = Pt(1)
        type(p).x.set(p, 99)  # ty: ignore[unresolved-attribute]
        assert p.x == 99


# ---------------------------------------------------------------------------
#  Edge cases
# ---------------------------------------------------------------------------
class TestFrozenEdgeCases:
    def test_frozen_class_no_own_fields(self) -> None:
        @py_class(_unique_key("no_fields"), frozen=True)
        class Empty(Object):
            pass

        Empty()  # should not raise

    def test_frozen_single_field(self) -> None:
        @py_class(_unique_key("single"), frozen=True)
        class Single(Object):
            x: int

        s = Single(1)
        assert s.x == 1
        with pytest.raises(AttributeError):
            s.x = 2  # ty: ignore[invalid-assignment]

    def test_frozen_field_with_none_default(self) -> None:
        @py_class(_unique_key("none_def"), frozen=True)
        class Opt(Object):
            v: Optional[int] = None  # noqa: UP045

        o = Opt()
        assert o.v is None
        with pytest.raises(AttributeError):
            o.v = 1  # ty: ignore[invalid-assignment]

    def test_multiple_instances_independent(self) -> None:
        @py_class(_unique_key("multi_inst"), frozen=True)
        class Pt(Object):
            x: int

        a = Pt(1)
        b = Pt(2)
        assert a.x == 1
        assert b.x == 2
        with pytest.raises(AttributeError):
            a.x = 99  # ty: ignore[invalid-assignment]
        with pytest.raises(AttributeError):
            b.x = 99  # ty: ignore[invalid-assignment]

    def test_frozen_instance_as_field_value(self) -> None:
        @py_class(_unique_key("inner_frozen"), frozen=True)
        class Inner(Object):
            val: int

        @py_class(_unique_key("outer_mut"))
        class Outer(Object):
            child: Inner

        inner = Inner(10)
        outer = Outer(inner)
        # Inner is still frozen even when held in a mutable outer
        with pytest.raises(AttributeError):
            outer.child.val = 99  # ty: ignore[invalid-assignment]
        # But the outer field itself is mutable
        outer.child = Inner(20)
        assert outer.child.val == 20
