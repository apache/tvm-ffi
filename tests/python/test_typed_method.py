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
"""Tests for ``@tvm_ffi.method`` — opt-in TypeMethod registration on
``@py_class``-decorated classes.
"""

from __future__ import annotations

import itertools
from typing import Any, ClassVar

import pytest
from tvm_ffi import Object, method
from tvm_ffi.core import TypeInfo
from tvm_ffi.dataclasses import py_class

_counter = itertools.count()


def _unique_key(base: str) -> str:
    """Return a globally unique type key so tests can re-register freely."""
    return f"testing.method_dec.{base}_{next(_counter)}"


def _find_method(info: TypeInfo, name: str) -> Any:
    """Return the ``TypeMethod`` entry for ``name`` or :data:`None`."""
    return next((m for m in info.methods if m.name == name), None)


def _toy_method_resolve(obj: Any, ref: str, *args: Any, **kwargs: Any) -> Any:
    """Stand-in for the C++ trait printer's ``$method:<name>`` resolution.

    Parses the ``$method:NAME`` ref, looks up ``NAME`` in the instance's
    ``TypeInfo.methods`` table, invokes the resolved ``Function`` on
    ``obj`` (plus any extra args). A successful return value means the
    whole ``@method`` → register → FFI callable chain is intact.
    """
    prefix = "$method:"
    if not ref.startswith(prefix):
        raise ValueError(f"Not a $method: ref: {ref!r}")
    name = ref[len(prefix) :]
    info = type(obj).__tvm_ffi_type_info__  # ty: ignore[unresolved-attribute]
    m = _find_method(info, name)
    if m is None:
        raise LookupError(
            f"{type(obj).__name__}.{name}: not in TypeInfo.methods — "
            "was the method decorated with ``@tvm_ffi.method``?",
        )
    return m.func(obj, *args, **kwargs)


# ---------------------------------------------------------------------------
# Registration — ``@method``-marked methods land in TypeInfo.methods
# ---------------------------------------------------------------------------


class TestMethodRegistration:
    """``@method`` drops the function's signature into
    ``TVMFFITypeRegisterMethod``; the name is resolvable from any FFI
    consumer.
    """

    def test_instance_method_registered_and_ffi_callable(self) -> None:
        """A plain instance-style method registers with ``is_static=False``
        and the returned FFI Function accepts the instance as arg 0.
        """

        @py_class(_unique_key("Node"))
        class Node(Object):
            x: int

            @method
            def label(self) -> str:
                return f"N({self.x})"

        m = _find_method(Node.__tvm_ffi_type_info__, "label")  # ty: ignore[unresolved-attribute]
        assert m is not None
        assert m.is_static is False
        # FFI call routes through the C method table — proves the
        # registration landed on the C side, not just the Python attr.
        assert m.func(Node(x=7)) == "N(7)"

    def test_staticmethod_registered_with_is_static_true(self) -> None:
        """``@method`` on top of ``@staticmethod`` marks the underlying
        function; the unwrap happens inside ``_collect_py_methods``.
        """

        @py_class(_unique_key("Nstat"))
        class Nstat(Object):
            x: int

            @method
            @staticmethod
            def constant() -> int:
                return 42

        m = _find_method(Nstat.__tvm_ffi_type_info__, "constant")  # ty: ignore[unresolved-attribute]
        assert m is not None
        assert m.is_static is True
        assert m.func() == 42

    def test_multiple_methods_all_registered(self) -> None:
        """Every ``@method``-marked callable appears in ``info.methods``."""

        @py_class(_unique_key("NodeMulti"))
        class NodeMulti(Object):
            x: int

            @method
            def kind(self) -> str:
                return "multi"

            @method
            def double(self) -> int:
                return self.x * 2

            @method
            def prefixed(self, p: str) -> str:
                return f"{p}-{self.x}"

        names = {m.name for m in NodeMulti.__tvm_ffi_type_info__.methods}  # ty: ignore[unresolved-attribute]
        assert {"kind", "double", "prefixed"}.issubset(names)

    def test_no_decorator_no_registration(self) -> None:
        """Without ``@method``, a class-body function is a plain Python
        attribute — nothing reaches ``TypeInfo.methods``. Protects the
        opt-in contract: users aren't surprised by accidental FFI
        registration of helper methods.
        """

        @py_class(_unique_key("NodeBare"))
        class NodeBare(Object):
            x: int

            def helper(self) -> int:  # no @method
                return self.x

        assert _find_method(NodeBare.__tvm_ffi_type_info__, "helper") is None  # ty: ignore[unresolved-attribute]

    def test_python_attribute_still_callable(self) -> None:
        """Registration doesn't shadow the Python attribute — callers
        can still invoke the method normally as ``instance.name(...)``.
        """

        @py_class(_unique_key("NodeKeep"))
        class NodeKeep(Object):
            x: int

            @method
            def doubled(self) -> int:
                return self.x * 2

        assert NodeKeep(x=5).doubled() == 10


# ---------------------------------------------------------------------------
# End-to-end: ``$method:<name>`` resolves via the toy printer
# ---------------------------------------------------------------------------


class TestDollarMethodResolution:
    """``$method:NAME`` refs reach ``@method``-decorated methods
    through the ``TypeInfo.methods`` table — the fix this PR ships.
    """

    def test_dollar_method_ref_invokes_decorated_method(self) -> None:
        """Trait stores ``$method:label``; the toy printer resolves and
        calls it. Mirrors what a real trait-driven C++ printer does.
        """

        @py_class(_unique_key("Op"))
        class Op(Object):
            kind: str
            __ffi_ir_traits__: ClassVar = {"print_label": "$method:label"}

            @method
            def label(self) -> str:
                return f"op:{self.kind}"

        result = _toy_method_resolve(Op(kind="add"), "$method:label")
        assert result == "op:add"

    def test_dollar_method_ref_threads_extra_args(self) -> None:
        """The toy resolver passes ``*args`` / ``**kwargs`` through to
        the FFI Function — covers prologue-style ``$method:NAME(printer, frame)``
        shapes used by real trait printers.
        """

        @py_class(_unique_key("PrologueOp"))
        class PrologueOp(Object):
            kind: str

            @method
            def print_prologue(self, printer: Any, frame: Any) -> str:
                # Use the extra args so a missing pass-through would show up.
                return f"{printer}-{self.kind}-{frame}"

        op = PrologueOp(kind="add")
        assert _toy_method_resolve(op, "$method:print_prologue", "PR", "FR") == "PR-add-FR"

    def test_dollar_method_missing_surfaces_clear_error(self) -> None:
        """A ``$method:`` ref targeting an undecorated method raises
        at resolution time — the failure mode a user hits when they
        forget the ``@method`` decorator.
        """

        @py_class(_unique_key("OpMiss"))
        class OpMiss(Object):
            kind: str

            def unmarked(self) -> str:  # no @method — not registered
                return self.kind

        with pytest.raises(LookupError, match=r"not in TypeInfo\.methods"):
            _toy_method_resolve(OpMiss(kind="x"), "$method:unmarked")


# ---------------------------------------------------------------------------
# Validation — reserved names / wrong wrappers rejected at decoration
# ---------------------------------------------------------------------------


class TestMethodValidation:
    """``@method`` + the registration path both raise with clear,
    class-scoped messages when a name or wrapper is reserved.
    """

    def test_rejects_classmethod(self) -> None:
        """``@classmethod``'s first-arg is the class, not the instance —
        breaks the packed-call convention. Rejected at decoration time
        (before py_class even sees the method).
        """
        with pytest.raises(TypeError, match=r"@classmethod is not supported"):

            class _Bad:
                @method
                @classmethod
                def maker(cls) -> int:
                    return 0

    def test_rejects_classmethod_method_decorator_order_swap(self) -> None:
        """The decorator catches ``@method @classmethod`` but a user can
        bypass it by writing ``@classmethod @method`` — @method runs
        first on the bare function, then classmethod wraps the marked
        function. The collector must surface this with a clear error;
        without the guard, the entry would silently fail to register
        (Python 3.11+) or register as a malformed instance method.
        """
        with pytest.raises(TypeError, match=r"wrapped by @classmethod"):

            @py_class(_unique_key("CMOrderBad"))
            class _CMOrderBad(Object):
                x: int

                @classmethod
                @method
                def maker(cls) -> int:
                    return 0

    def test_rejects_manually_marked_classmethod(self) -> None:
        """The decorator can also be bypassed by marking a function
        directly (``fn.__ffi_method__ = True``) and then wrapping it
        in ``classmethod``. The collector's classmethod check fires
        on the descriptor regardless of how the marker got there.
        """

        def _maker(cls: type) -> int:
            return 0

        _maker.__ffi_method__ = True  # ty: ignore[unresolved-attribute]
        cm = classmethod(_maker)

        with pytest.raises(TypeError, match=r"wrapped by @classmethod"):

            @py_class(_unique_key("CMManualBad"))
            class _CMManualBad(Object):
                x: int
                maker = cm

    def test_rejects_non_callable(self) -> None:
        """``@method`` applied to a bare value (not a callable) raises."""
        with pytest.raises(TypeError, match=r"expected a callable"):
            method(42)

    def test_rejects_reserved_ffi_prefix(self) -> None:
        """``__ffi_*`` names are routed through TypeAttrColumn — using
        ``@method`` on them is surely a user error (would silently
        double-register), so ``_collect_py_methods`` raises.
        """
        with pytest.raises(NameError, match=r"reserved ``__ffi_`` prefix"):

            @py_class(_unique_key("RFfiPfx"))
            class _RFfiPfx(Object):
                x: int

                @method
                def __ffi_custom__(self) -> int:
                    return 0

    def test_rejects_typeattrcolumn_name(self) -> None:
        """Decorating a TypeAttrColumn dunder with ``@method`` is
        rejected — those are routed to ``TVMFFITypeRegisterAttr``
        already, never to TypeMethod.
        """
        with pytest.raises(NameError, match=r"TypeAttrColumn"):

            @py_class(_unique_key("RAttr"))
            class _RAttr(Object):
                x: int

                @method
                def __ffi_repr__(self, fn_repr: Any) -> str:
                    return "r"

    def test_rejects_python_protocol_dunder(self) -> None:
        """``__len__`` / ``__iter__`` / etc. are reserved for Python
        semantics — cannot be FFI TypeMethods.
        """
        with pytest.raises(NameError, match=r"Python protocol dunder"):

            @py_class(_unique_key("RDun"))
            class _RDun(Object):
                x: int

                @method
                def __len__(self) -> int:
                    return 0
