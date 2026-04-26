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
"""Tests for ``tvm_ffi.dialect_autogen.finalize_module``.

These tests exercise *what name to register* — not *what to register*.
Every handler installed by ``finalize_module`` is a placeholder that
raises :class:`NotImplementedError`; the assertions check that the
correct attribute appears on the dialect module under the correct name
and sub-key.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

import pytest

from tvm_ffi import Object, finalize_module, parse_hook
from tvm_ffi import ir_traits as tr
from tvm_ffi.dataclasses import py_class
from tvm_ffi.dialect_autogen import (
    DEFAULT_DTYPE_NAMES,
    _DtypeHandle,
    _classify,
    _tier,
    registered_names,
)
from tvm_ffi.parse_hook import (
    RESERVED_DICT_SLOTS,
    RESERVED_FN_SLOTS,
    ParseHookSpec,
    get_hook_spec,
    get_slot_field,
    parse_slot,
)
from tvm_ffi.pyast import OperationKind
from tvm_ffi.stub.dialect_stub import generate_dialect_stub


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dialect() -> ModuleType:
    """Import the fixture module once and finalize it.

    The IR class type-keys are baked into the C-level registry on first
    import; there's no clean reload story. Tests rely on
    ``finalize_module`` being idempotent — verified by
    :func:`test_finalize_module_idempotent` — and inspect the module
    attributes read-only.
    """
    import tests.python.lang_module_fixtures as fixtures  # noqa: WPS433

    finalize_module(fixtures.__name__, auto_stub=False)
    return fixtures


# ---------------------------------------------------------------------------
# Decorator-only sanity checks
# ---------------------------------------------------------------------------


class TestParseHookDecorator:
    """``@parse_hook`` records the right :class:`ParseHookSpec` shape."""

    def test_named_callee_lands_in_named_callees(self) -> None:
        @parse_hook("foo")
        def fn() -> None: ...

        spec = get_hook_spec(fn)
        assert spec is not None
        assert spec.named_callees == ("foo",)
        assert spec.fn_slots == ()
        assert spec.op_kinds == ()

    def test_iterable_named_callees_flatten(self) -> None:
        @parse_hook(["serial", "parallel"], "unroll")
        def fn() -> None: ...

        spec = get_hook_spec(fn)
        assert spec is not None
        assert spec.named_callees == ("serial", "parallel", "unroll")

    def test_reserved_fn_slot_routed(self) -> None:
        @parse_hook("__ffi_parse_func__")
        def fn() -> None: ...

        spec = get_hook_spec(fn)
        assert spec is not None
        assert spec.fn_slots == ("__ffi_parse_func__",)
        assert spec.named_callees == ()

    def test_op_kind_kwarg_routes(self) -> None:
        @parse_hook(callee="Add", op_kind=OperationKind.Add)
        def fn() -> None: ...

        spec = get_hook_spec(fn)
        assert spec is not None
        assert spec.named_callees == ("Add",)
        assert spec.op_kinds == (OperationKind.Add,)

    def test_make_const_format_kwarg_routes(self) -> None:
        @parse_hook(make_const_format="int")
        def fn() -> None: ...

        spec = get_hook_spec(fn)
        assert spec is not None
        assert spec.make_const_formats == ("int",)

    def test_dict_slot_as_positional_raises(self) -> None:
        with pytest.raises(ValueError, match=r"dict-typed reserved slot"):

            @parse_hook("__ffi_parse_op__")
            def _fn() -> None: ...

    def test_no_targets_raises(self) -> None:
        with pytest.raises(ValueError, match=r"at least one target"):

            @parse_hook()
            def _fn() -> None: ...

    def test_non_callable_target_raises(self) -> None:
        with pytest.raises(TypeError, match=r"expected a callable"):
            parse_hook("foo")(42)  # type: ignore[arg-type]


class TestParseSlotDecorator:
    """``@parse_slot`` attaches the trait-field name to the wrapped fn."""

    def test_attaches_slot(self) -> None:
        @parse_slot("extent")
        def fn() -> None: ...

        assert get_slot_field(fn) == "extent"

    def test_empty_field_raises(self) -> None:
        with pytest.raises(TypeError, match=r"non-empty str"):
            parse_slot("")

    def test_non_str_field_raises(self) -> None:
        with pytest.raises(TypeError, match=r"non-empty str"):
            parse_slot(42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Classifier — pure-function unit tests
# ---------------------------------------------------------------------------


class TestTierDetection:
    """``_tier`` returns 1/2/3 from the right TypeAttrColumn presence."""

    def test_tier_3_default(self) -> None:
        @py_class("testing.lang_test.NoTraits_t3")
        class _N(Object):
            x: int

        assert _tier(_N) == 3

    def test_tier_2_when_traits_set(self) -> None:
        @py_class("testing.lang_test.WithTraits_t2")
        class _N(Object):
            __ffi_ir_traits__ = tr.ValueTraits("$field:name", None, None)
            name: str

        assert _tier(_N) == 2

    def test_tier_1_when_text_print_set(self) -> None:
        @py_class("testing.lang_test.WithPrint_t1")
        class _N(Object):
            __ffi_text_print__ = (lambda *_a, **_k: None)  # noqa: PLW0108
            x: int

        assert _tier(_N) == 1


class TestPerTraitClassification:
    """The classifier emits the right :class:`RegEntry` shape per trait.

    These tests use freshly-decorated classes (rather than the shared
    fixture module) so each trait is exercised in isolation.
    """

    def test_binop_sugared_op_lands_in_op_dict(self) -> None:
        @py_class("testing.lang_test.BinAdd_cls")
        class _Add(Object):
            __ffi_ir_traits__ = tr.BinOpTraits("$field:l", "$field:r", "+", None, None)
            l: Any
            r: Any

        entries = _classify(_Add)
        assert len(entries) == 1
        e = entries[0]
        assert e.target == "__ffi_parse_op__"
        assert e.sub_key == OperationKind.Add

    def test_binop_with_func_name_emits_two_entries(self) -> None:
        @py_class("testing.lang_test.BinFD_cls")
        class _FD(Object):
            __ffi_ir_traits__ = tr.BinOpTraits("$field:l", "$field:r", "//", None, "FloorDiv")
            l: Any
            r: Any

        entries = _classify(_FD)
        targets = {(e.target, e.sub_key) for e in entries}
        assert ("__ffi_parse_op__", OperationKind.FloorDiv) in targets
        assert ("FloorDiv", None) in targets

    def test_value_lands_in_make_var(self) -> None:
        @py_class("testing.lang_test.Var_cls")
        class _V(Object):
            __ffi_ir_traits__ = tr.ValueTraits("$field:name", None, None)
            name: str

        entries = _classify(_V)
        assert [e.target for e in entries] == ["__ffi_parse_make_var__"]

    def test_literal_int_format(self) -> None:
        @py_class("testing.lang_test.IntImm_cls")
        class _I(Object):
            __ffi_ir_traits__ = tr.LiteralTraits("$field:value", "int")
            value: int

        entries = _classify(_I)
        assert len(entries) == 1
        e = entries[0]
        assert e.target == "__ffi_parse_make_const__"
        assert e.sub_key == "int"

    def test_literal_no_format_uses_default_subkey(self) -> None:
        @py_class("testing.lang_test.AnyLit_cls")
        class _L(Object):
            __ffi_ir_traits__ = tr.LiteralTraits("$field:value", None)
            value: Any

        entries = _classify(_L)
        assert entries[0].sub_key == "default"

    def test_call_literal_callee_uses_callee_name(self) -> None:
        @py_class("testing.lang_test.Call_cls")
        class _C(Object):
            __ffi_ir_traits__ = tr.CallTraits("foo", "$field:args", None, None, None, None)
            args: Any

        entries = _classify(_C)
        assert [e.target for e in entries] == ["foo"]

    def test_call_opaque_callee_emits_nothing(self) -> None:
        """``$method:`` ops can't be classified — registry waits for an override."""

        @py_class("testing.lang_test.OpaqueCall_cls")
        class _C(Object):
            __ffi_ir_traits__ = tr.CallTraits("$method:_resolve", "$field:args", None, None, None, None)
            args: Any

        assert _classify(_C) == []

    def test_func_with_kind(self) -> None:
        @py_class("testing.lang_test.PrimFunc_cls")
        class _F(Object):
            __ffi_ir_traits__ = tr.FuncTraits(
                "$field:n",
                tr.RegionTraits("$field:body", None, None, None),
                None,
                "prim_func",
                None,
            )
            n: str
            body: list

        entries = _classify(_F)
        assert [e.target for e in entries] == ["prim_func"]

    def test_func_no_kind_falls_back_to_reserved(self) -> None:
        @py_class("testing.lang_test.AnonFunc_cls")
        class _F(Object):
            __ffi_ir_traits__ = tr.FuncTraits(
                "$field:n",
                tr.RegionTraits("$field:body", None, None, None),
                None,
                None,
                None,
            )
            n: str
            body: list

        entries = _classify(_F)
        assert [e.target for e in entries] == ["__ffi_parse_func__"]

    def test_assign_with_kind_uses_kind(self) -> None:
        @py_class("testing.lang_test.Bind_cls")
        class _A(Object):
            __ffi_ir_traits__ = tr.AssignTraits(
                "$field:lhs",
                "$field:rhs",
                None,
                None,
                "bind",
                None,
            )
            lhs: Any
            rhs: Any

        assert _classify(_A)[0].target == "bind"

    def test_assign_no_kind_falls_back_to_reserved(self) -> None:
        @py_class("testing.lang_test.Eval_cls")
        class _A(Object):
            __ffi_ir_traits__ = tr.AssignTraits(None, "$field:rhs", None, None, None, None)
            rhs: Any

        assert _classify(_A)[0].target == "__ffi_parse_assign__"

    def test_for_falls_back_to_reserved_slot(self) -> None:
        @py_class("testing.lang_test.For_cls")
        class _F(Object):
            __ffi_ir_traits__ = tr.ForTraits(
                tr.RegionTraits("$field:body", None, None, None),
                None,
                None,
                None,
                None,
                None,
                None,
                "$method:_kind",
            )
            body: list

        assert _classify(_F)[0].target == "__ffi_parse_for__"

    @pytest.mark.parametrize(
        "trait, target",
        [
            (tr.AssertTraits("$field:c", None), "__ffi_parse_assert__"),
            (tr.ReturnTraits("$field:v"), "__ffi_parse_return__"),
            (tr.WhileTraits(
                "$field:c",
                tr.RegionTraits("$field:body", None, None, None),
            ), "__ffi_parse_while__"),
            (tr.IfTraits(
                "$field:c",
                tr.RegionTraits("$field:t", None, None, None),
                None,
            ), "__ffi_parse_if__"),
            (tr.LoadTraits("$field:s", None, None), "__ffi_parse_load__"),
            (tr.StoreTraits("$field:t", "$field:v", None, None), "__ffi_parse_store__"),
            (tr.WithTraits(
                tr.RegionTraits("$field:body", None, None, None),
                None, None, None, None, None, None,
            ), "__ffi_parse_with__"),
        ],
    )
    def test_simple_traits_target_their_reserved_slot(
        self,
        trait: Any,
        target: str,
    ) -> None:
        """Traits whose only registration target is a fixed reserved slot."""
        # Defensive: build a minimal class around each trait and assert
        # its single classifier output.
        @py_class(f"testing.lang_test.SimpleTrait_{target}")
        class _N(Object):
            __ffi_ir_traits__ = trait
            value: Any

        entries = _classify(_N)
        assert [e.target for e in entries] == [target]

    @pytest.mark.parametrize(
        "trait, name",
        [
            (tr.TensorTyTraits(None, None, None), "Tensor"),
            (tr.BufferTyTraits("$field:s", "$field:d", None, None, None), "Buffer"),
            (tr.FuncTyTraits(None, None), "FuncType"),
            (tr.TupleTyTraits("$field:f"), "Tuple"),
            (tr.ShapeTyTraits(None, None), "Shape"),
        ],
    )
    def test_type_traits_register_fixed_name(self, trait: Any, name: str) -> None:
        @py_class(f"testing.lang_test.TyTrait_{name}")
        class _N(Object):
            __ffi_ir_traits__ = trait

        entries = _classify(_N)
        assert [e.target for e in entries] == [name]

    def test_tier_3_class_registers_class_name(self) -> None:
        @py_class("testing.lang_test.PlainBare_cls")
        class _N(Object):
            x: int

        entries = _classify(_N)
        assert [e.target for e in entries] == ["_N"]


# ---------------------------------------------------------------------------
# End-to-end finalize_module — the dialect surface
# ---------------------------------------------------------------------------


class TestFinalizeModuleSurface:
    """After ``finalize_module``, the dialect carries the right names."""

    def test_reserved_fn_slots_present(self, dialect: ModuleType) -> None:
        """Every reserved C0 fn-slot that any tier-2 class claimed lands."""
        for slot in (
            "__ffi_parse_make_var__",
            "__ffi_parse_assign__",
            "__ffi_parse_assert__",
            "__ffi_parse_return__",
            "__ffi_parse_func__",
            "__ffi_parse_for__",
            "__ffi_parse_with__",
            "__ffi_parse_while__",
            "__ffi_parse_if__",
            "__ffi_parse_load__",
            "__ffi_parse_store__",
        ):
            assert hasattr(dialect, slot), (
                f"missing reserved slot {slot!r} on dialect"
            )

    def test_reserved_dict_slots_present(self, dialect: ModuleType) -> None:
        op_dict = getattr(dialect, "__ffi_parse_op__", None)
        const_dict = getattr(dialect, "__ffi_parse_make_const__", None)
        assert isinstance(op_dict, dict)
        assert isinstance(const_dict, dict)
        # BinAdd contributes Add; UnaryNeg contributes USub; the
        # @parse_hook override contributes Add via "MyOp" too — first
        # wins via override.
        assert OperationKind.Add in op_dict
        assert OperationKind.USub in op_dict
        # IntImm → 'int', FloatImm → 'float', StringLit → 'default'.
        assert {"int", "float", "default"}.issubset(set(const_dict))

    def test_named_callees_for_literal_kinds(self, dialect: ModuleType) -> None:
        """Tier-2 classes with literal text_printer_kind register that name."""
        assert callable(getattr(dialect, "prim_func", None))
        assert callable(getattr(dialect, "bind", None))
        # CallTraits with literal "my_op" callee.
        assert callable(getattr(dialect, "my_op", None))

    def test_type_trait_names(self, dialect: ModuleType) -> None:
        for name in ("Tensor", "Buffer", "FuncType", "Tuple", "Shape"):
            assert callable(getattr(dialect, name, None)), (
                f"missing type-trait registration {name!r}"
            )

    def test_dtype_handles_installed(self, dialect: ModuleType) -> None:
        for name in DEFAULT_DTYPE_NAMES:
            handle = getattr(dialect, name, None)
            assert isinstance(handle, _DtypeHandle), (
                f"dtype handle missing or wrong type for {name!r}"
            )
            assert handle.name == name

    def test_dtype_handle_call_routes_to_make_var(self, dialect: ModuleType) -> None:
        """``T.int32(var_name="x")`` invokes the make_var placeholder."""
        with pytest.raises(NotImplementedError, match=r"Parser dispatch"):
            dialect.int32(var_name="x")

    def test_dtype_handle_call_routes_to_make_const(self, dialect: ModuleType) -> None:
        """``T.int32(value=42)`` invokes the make_const[int] placeholder."""
        with pytest.raises(NotImplementedError, match=r"Parser dispatch"):
            dialect.int32(value=42)

    def test_tier_3_class_name_registered(self, dialect: ModuleType) -> None:
        # Tier 3 fixture is named "PlainNode" in the dialect.
        assert callable(getattr(dialect, "PlainNode", None))

    def test_module_scope_parse_hook_registers(self, dialect: ModuleType) -> None:
        """``@parse_hook("serial", ...)`` at module scope binds those names."""
        for name in ("serial", "parallel", "unroll", "vectorized"):
            assert callable(getattr(dialect, name, None)), (
                f"@parse_hook override missing on {name!r}"
            )

    def test_parse_slot_attaches_to_class(self, dialect: ModuleType) -> None:
        """``@parse_slot('extent')`` lands on the class' ``__ffi_parse_slots__``."""
        for_cls = dialect.For
        slots = getattr(for_cls, "__ffi_parse_slots__", None)
        assert isinstance(slots, dict)
        assert "extent" in slots
        assert callable(slots["extent"])

    def test_default_type_canonicalized(self, dialect: ModuleType) -> None:
        canon = getattr(dialect, "__ffi_default_type__", None)
        assert isinstance(canon, dict)
        assert canon["int"] == "int32"
        assert canon["float"] == "float32"
        assert canon["bool"] == "bool"


# ---------------------------------------------------------------------------
# Conflict resolution
# ---------------------------------------------------------------------------


class TestConflicts:
    """Two tier-2 classes can't claim the same reserved slot without an override."""

    def test_duplicate_make_var_raises(self) -> None:
        """Two ValueTraits classes → both want ``__ffi_parse_make_var__``."""
        mod = ModuleType("testing.lang_conflicts.dup_make_var")
        mod.__file__ = ""  # disable .pyi writeback

        @py_class("testing.lang_conflicts.Var1")
        class _V1(Object):
            __ffi_ir_traits__ = tr.ValueTraits("$field:name", None, None)
            name: str

        @py_class("testing.lang_conflicts.Var2")
        class _V2(Object):
            __ffi_ir_traits__ = tr.ValueTraits("$field:name", None, None)
            name: str

        # Both classes are owned by ``mod``.
        _V1.__module__ = mod.__name__
        _V2.__module__ = mod.__name__
        mod.V1 = _V1
        mod.V2 = _V2

        import sys

        sys.modules[mod.__name__] = mod
        try:
            with pytest.raises(RuntimeError, match=r"duplicate registration"):
                finalize_module(mod.__name__, auto_stub=False)
        finally:
            sys.modules.pop(mod.__name__, None)

    def test_user_override_breaks_tie(self) -> None:
        """A module-scope ``@parse_hook`` claims the slot before the classifier."""
        mod = ModuleType("testing.lang_conflicts.override_dup")
        mod.__file__ = ""

        @py_class("testing.lang_conflicts.OVar1")
        class _V1(Object):
            __ffi_ir_traits__ = tr.ValueTraits("$field:name", None, None)
            name: str

        @py_class("testing.lang_conflicts.OVar2")
        class _V2(Object):
            __ffi_ir_traits__ = tr.ValueTraits("$field:name", None, None)
            name: str

        @parse_hook("__ffi_parse_make_var__")
        def _user_override(*_a: Any, **_k: Any) -> Any:
            raise NotImplementedError

        _V1.__module__ = mod.__name__
        _V2.__module__ = mod.__name__
        mod.V1 = _V1
        mod.V2 = _V2
        mod._user_override = _user_override

        import sys

        sys.modules[mod.__name__] = mod
        try:
            finalize_module(mod.__name__, auto_stub=False)
            # The user override wins — module attr is exactly that fn.
            assert mod.__ffi_parse_make_var__ is _user_override
        finally:
            sys.modules.pop(mod.__name__, None)


# ---------------------------------------------------------------------------
# Stub generation
# ---------------------------------------------------------------------------


class TestStubGeneration:
    """The generated ``.pyi`` reflects every registered name."""

    def test_stub_contains_dtype_handles(self, dialect: ModuleType) -> None:
        text = generate_dialect_stub(dialect)
        for name in ("int32", "float32", "bool"):
            assert f"\n{name}: Any" in text

    def test_stub_contains_reserved_slots(self, dialect: ModuleType) -> None:
        text = generate_dialect_stub(dialect)
        assert "__ffi_parse_make_var__: Callable[..., Any]" in text
        assert "__ffi_parse_op__: dict[int, Callable[..., Any]]" in text
        assert "__ffi_parse_make_const__: dict[str, Callable[..., Any]]" in text

    def test_stub_contains_named_callees(self, dialect: ModuleType) -> None:
        text = generate_dialect_stub(dialect)
        for name in ("prim_func", "bind", "Tensor", "Buffer", "serial", "parallel"):
            assert f"def {name}(" in text, f"missing {name} in stub:\n{text}"

    def test_stub_contains_class_reexports(self, dialect: ModuleType) -> None:
        text = generate_dialect_stub(dialect)
        # PlainNode is the tier-3 fixture; its class re-export must appear.
        assert "class PlainNode: ..." in text


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_finalize_module_idempotent(dialect: ModuleType) -> None:
    """Running ``finalize_module`` twice doesn't corrupt the dialect."""
    before = set(registered_names(dialect))
    finalize_module(dialect.__name__, auto_stub=False)
    after = set(registered_names(dialect))
    assert before == after


# ---------------------------------------------------------------------------
# Sanity: every placeholder really raises
# ---------------------------------------------------------------------------


def test_every_placeholder_raises(dialect: ModuleType) -> None:
    """Walk the module surface and confirm every callable placeholder
    raises ``NotImplementedError`` when called."""
    for name in registered_names(dialect):
        value = getattr(dialect, name, None)
        # Skip dict-typed slots and dtype handles (handles aren't
        # placeholders — they route to placeholder handlers).
        if isinstance(value, (dict, _DtypeHandle)):
            continue
        if not callable(value):
            continue
        with pytest.raises(NotImplementedError):
            value()


# ---------------------------------------------------------------------------
# Re-export surface
# ---------------------------------------------------------------------------


def test_top_level_exports() -> None:
    import tvm_ffi

    assert tvm_ffi.finalize_module is finalize_module
    assert tvm_ffi.parse_hook is parse_hook
    # parse_slot is exported via tvm_ffi.parse_slot
    from tvm_ffi import parse_slot as ps

    assert ps is parse_slot
    # ParseHookSpec is internal but importable from the parse_hook module
    assert ParseHookSpec is not None
    assert RESERVED_FN_SLOTS  # non-empty
    assert RESERVED_DICT_SLOTS  # non-empty
