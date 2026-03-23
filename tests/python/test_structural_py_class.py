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
"""Tests for structural equality/hashing on py_class-defined types.

Mirrors the C++ tests in tests/cpp/extra/test_structural_equal_hash.cc,
porting the object-level tests (FreeVar, FuncDefAndIgnoreField, etc.)
to Python using ``@py_class(structural_eq=...)`` and ``field(structural_eq=...)``.
"""

from __future__ import annotations

import pytest
import tvm_ffi as ffi
from tvm_ffi import get_first_structural_mismatch, structural_equal, structural_hash
from tvm_ffi.dataclasses import field, py_class

# ---------------------------------------------------------------------------
# Type definitions (mirror testing_object.h)
# ---------------------------------------------------------------------------


@py_class("testing.py.Var", structural_eq="var")
class TVar(ffi.Object):
    """Variable node — compared by binding position, not by name.

    Mirrors C++ TVarObj with:
      _type_s_eq_hash_kind = kTVMFFISEqHashKindFreeVar
      name field has SEqHashIgnore
    """

    name: str = field(structural_eq="ignore")


@py_class("testing.py.Int", structural_eq="tree")
class TInt(ffi.Object):
    """Simple integer literal node."""

    value: int


@py_class("testing.py.Func", structural_eq="tree")
class TFunc(ffi.Object):
    """Function node with definition region and ignored comment.

    Mirrors C++ TFuncObj with:
      params has SEqHashDef
      comment has SEqHashIgnore
    """

    params: list = field(structural_eq="def")
    body: list
    comment: str = field(structural_eq="ignore", default="")


@py_class("testing.py.Expr", structural_eq="tree")
class TExpr(ffi.Object):
    """A simple expression node for tree-comparison tests."""

    value: int


@py_class("testing.py.Metadata", structural_eq="const-tree")
class TMetadata(ffi.Object):
    """Immutable metadata node — pointer shortcut is safe (no var children)."""

    tag: str
    version: int


@py_class("testing.py.Binding", structural_eq="dag")
class TBinding(ffi.Object):
    """Binding node — sharing structure is semantically meaningful."""

    name: str
    value: int


# ---------------------------------------------------------------------------
# Tests: FreeVar (mirrors C++ FreeVar test)
# ---------------------------------------------------------------------------


class TestFreeVar:
    """Test structural_eq="var" kind (C++ kTVMFFISEqHashKindFreeVar)."""

    def test_free_var_equal_with_mapping(self) -> None:
        """Two different vars are equal when map_free_vars=True."""
        a = TVar("a")
        b = TVar("b")
        assert structural_equal(a, b, map_free_vars=True)

    def test_free_var_not_equal_without_mapping(self) -> None:
        """Two different vars are NOT equal by default (no mapping)."""
        a = TVar("a")
        b = TVar("b")
        assert not structural_equal(a, b)

    def test_free_var_hash_differs_without_mapping(self) -> None:
        """Without mapping, different vars produce different hashes."""
        a = TVar("a")
        b = TVar("b")
        assert structural_hash(a) != structural_hash(b)

    def test_free_var_hash_equal_with_mapping(self) -> None:
        """With map_free_vars, positional hashing makes them equal."""
        a = TVar("a")
        b = TVar("b")
        assert structural_hash(a, map_free_vars=True) == structural_hash(b, map_free_vars=True)

    def test_free_var_same_pointer(self) -> None:
        """Same variable is always equal to itself."""
        x = TVar("x")
        assert structural_equal(x, x)

    def test_free_var_name_ignored(self) -> None:
        """The name field is structural_eq="ignore", so it doesn't affect comparison."""
        a = TVar("different_name_a")
        b = TVar("different_name_b")
        # Names differ, but with mapping they are still equal
        assert structural_equal(a, b, map_free_vars=True)


# ---------------------------------------------------------------------------
# Tests: FuncDefAndIgnoreField (mirrors C++ FuncDefAndIgnoreField test)
# ---------------------------------------------------------------------------


class TestFuncDefAndIgnore:
    """Test structural_eq="def" and structural_eq="ignore" on fields."""

    def test_alpha_equivalent_functions(self) -> None:
        """fun(x){1, x} with comment_a == fun(y){1, y} with comment_b."""
        x = TVar("x")
        y = TVar("y")
        fa = TFunc(params=[x], body=[TInt(1), x], comment="comment a")
        fb = TFunc(params=[y], body=[TInt(1), y], comment="comment b")
        assert structural_equal(fa, fb)
        assert structural_hash(fa) == structural_hash(fb)

    def test_different_body(self) -> None:
        """fun(x){1, x} != fun(x){1, 2} — body differs at index 1."""
        x = TVar("x")
        fa = TFunc(params=[x], body=[TInt(1), x], comment="comment a")
        fc = TFunc(params=[x], body=[TInt(1), TInt(2)], comment="comment c")
        assert not structural_equal(fa, fc)

    def test_mismatch_path(self) -> None:
        """GetFirstMismatch reports the correct access path."""
        x = TVar("x")
        fa = TFunc(params=[x], body=[TInt(1), x])
        fc = TFunc(params=[x], body=[TInt(1), TInt(2)])
        mismatch = get_first_structural_mismatch(fa, fc)
        assert mismatch is not None

    def test_comment_ignored(self) -> None:
        """Identical structure with different comments are equal."""
        x = TVar("x")
        f1 = TFunc(params=[x], body=[TInt(1)], comment="first")
        f2 = TFunc(params=[x], body=[TInt(1)], comment="second")
        assert structural_equal(f1, f2)
        assert structural_hash(f1) == structural_hash(f2)

    def test_inconsistent_var_usage(self) -> None:
        """fun(x,y){x+x} != fun(a,b){a+b} — inconsistent variable mapping."""
        x, y = TVar("x"), TVar("y")
        a, b = TVar("a"), TVar("b")
        f1 = TFunc(params=[x, y], body=[x, x])
        f2 = TFunc(params=[a, b], body=[a, b])
        assert not structural_equal(f1, f2)

    def test_multi_param_alpha_equiv(self) -> None:
        """fun(x,y){x, y} == fun(a,b){a, b} — consistent variable renaming."""
        x, y = TVar("x"), TVar("y")
        a, b = TVar("a"), TVar("b")
        f1 = TFunc(params=[x, y], body=[x, y])
        f2 = TFunc(params=[a, b], body=[a, b])
        assert structural_equal(f1, f2)
        assert structural_hash(f1) == structural_hash(f2)

    def test_swapped_params_not_equal(self) -> None:
        """fun(x,y){x, y} != fun(a,b){b, a} — reversed usage."""
        x, y = TVar("x"), TVar("y")
        a, b = TVar("a"), TVar("b")
        f1 = TFunc(params=[x, y], body=[x, y])
        f2 = TFunc(params=[a, b], body=[b, a])
        assert not structural_equal(f1, f2)

    def test_nested_functions(self) -> None:
        """Nested function with inner binding — alpha-equivalence is scoped."""
        x = TVar("x")
        y = TVar("y")
        inner_x = TFunc(params=[x], body=[x])
        inner_y = TFunc(params=[y], body=[y])
        outer_a = TFunc(params=[], body=[inner_x])
        outer_b = TFunc(params=[], body=[inner_y])
        assert structural_equal(outer_a, outer_b)
        assert structural_hash(outer_a) == structural_hash(outer_b)


# ---------------------------------------------------------------------------
# Tests: tree kind basics
# ---------------------------------------------------------------------------


class TestTreeNode:
    """Test structural_eq="tree" kind."""

    def test_equal_content(self) -> None:
        """Two tree nodes with identical content are structurally equal."""
        a = TExpr(value=42)
        b = TExpr(value=42)
        assert structural_equal(a, b)
        assert structural_hash(a) == structural_hash(b)

    def test_different_content(self) -> None:
        """Two tree nodes with different content are not equal."""
        a = TExpr(value=1)
        b = TExpr(value=2)
        assert not structural_equal(a, b)
        assert structural_hash(a) != structural_hash(b)

    def test_sharing_invisible(self) -> None:
        """Under "tree", sharing doesn't affect equality."""
        s = TExpr(value=10)
        # Two arrays referencing the same object vs two copies
        shared = ffi.Array([s, s])
        copies = ffi.Array([TExpr(value=10), TExpr(value=10)])
        assert structural_equal(shared, copies)
        assert structural_hash(shared) == structural_hash(copies)


# ---------------------------------------------------------------------------
# Tests: const-tree kind
# ---------------------------------------------------------------------------


class TestConstTreeNode:
    """Test structural_eq="const-tree" kind."""

    def test_equal_content(self) -> None:
        """Two const-tree nodes with identical content are structurally equal."""
        a = TMetadata(tag="v1", version=1)
        b = TMetadata(tag="v1", version=1)
        assert structural_equal(a, b)
        assert structural_hash(a) == structural_hash(b)

    def test_different_content(self) -> None:
        """Two const-tree nodes with different content are not equal."""
        a = TMetadata(tag="v1", version=1)
        b = TMetadata(tag="v1", version=2)
        assert not structural_equal(a, b)

    def test_same_pointer_shortcircuits(self) -> None:
        """Same pointer should be equal (the const-tree optimization)."""
        a = TMetadata(tag="test", version=1)
        assert structural_equal(a, a)


# ---------------------------------------------------------------------------
# Tests: dag kind
# ---------------------------------------------------------------------------


class TestDAGNode:
    """Test structural_eq="dag" kind."""

    def test_same_dag_shape(self) -> None:
        """Two DAGs with the same sharing shape are equal."""
        s1 = TBinding(name="s", value=1)
        s2 = TBinding(name="s", value=1)
        dag1 = ffi.Array([s1, s1])  # shared
        dag2 = ffi.Array([s2, s2])  # shared (same shape)
        assert structural_equal(dag1, dag2)

    def test_dag_vs_tree_not_equal(self) -> None:
        """A DAG (shared) vs tree (independent copies) are NOT equal."""
        shared = TBinding(name="s", value=1)
        copy1 = TBinding(name="s", value=1)
        copy2 = TBinding(name="s", value=1)
        dag = ffi.Array([shared, shared])
        tree = ffi.Array([copy1, copy2])
        assert not structural_equal(dag, tree)

    def test_dag_vs_tree_hash_differs(self) -> None:
        """DAG and tree with same content should hash differently."""
        shared = TBinding(name="s", value=1)
        copy1 = TBinding(name="s", value=1)
        copy2 = TBinding(name="s", value=1)
        dag = ffi.Array([shared, shared])
        tree = ffi.Array([copy1, copy2])
        assert structural_hash(dag) != structural_hash(tree)

    def test_reverse_bijection(self) -> None:
        """(a, b) vs (a, a) where a ≅ b — reverse map detects inconsistency."""
        a = TBinding(name="a", value=1)
        b = TBinding(name="b", value=1)  # same content as a
        lhs = ffi.Array([a, b])
        rhs = ffi.Array([a, a])  # note: same object twice
        assert not structural_equal(lhs, rhs)


# ---------------------------------------------------------------------------
# Tests: unsupported kind (default)
# ---------------------------------------------------------------------------


class TestUnsupported:
    """Test that types without structural_eq= raise on structural comparison."""

    def test_unsupported_raises_on_hash(self) -> None:
        """Structural hashing raises TypeError for types without structural_eq=."""

        @py_class("testing.py.Plain")
        class Plain(ffi.Object):
            x: int

        with pytest.raises(TypeError):
            structural_hash(Plain(x=1))

    def test_unsupported_raises_on_equal(self) -> None:
        """Structural equality raises TypeError for types without structural_eq=."""

        @py_class("testing.py.Plain2")
        class Plain2(ffi.Object):
            x: int

        with pytest.raises(TypeError):
            structural_equal(Plain2(x=1), Plain2(x=1))


# ---------------------------------------------------------------------------
# Tests: validation
# ---------------------------------------------------------------------------


class TestValidation:
    """Test that invalid structural_eq= values are rejected."""

    def test_invalid_type_structure(self) -> None:
        """Invalid type-level structural_eq= value raises ValueError."""
        with pytest.raises(ValueError, match="structural_eq"):
            py_class(structural_eq="invalid")

    def test_invalid_field_structure(self) -> None:
        """Invalid field-level structural_eq= value raises ValueError."""
        with pytest.raises(ValueError, match="structural_eq"):
            field(structural_eq="bad_value")
