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
"""Fixture dialect for ``finalize_module`` tests.

This module exists solely to exercise the registration walker. Each IR
class targets one trait kind across one tier so the test suite can
introspect what lands on the dialect surface.

Layout
------
* ``Tier1*``: classes with a manual ``__ffi_text_print__``; rely on a
  module-scope ``@parse_hook`` for explicit registration.
* ``Tier2*``: trait-driven classes; the classifier picks the slot.
* ``Tier3*``: plain ``@py_class`` (no print, no traits); register their
  class name as the default callee.

The test that imports this module is responsible for invoking
``finalize_module(__name__)`` (with ``auto_stub=False``) — no
side-effect at import time.
"""

from __future__ import annotations

import itertools
from typing import Any

from tvm_ffi import Object
from tvm_ffi import ir_traits as tr
from tvm_ffi import parse_hook, parse_slot
from tvm_ffi.dataclasses import py_class
from tvm_ffi.pyast import OperationKind

# Globally-unique type keys so the test file can be re-imported (e.g.
# under pytest watch) without colliding with prior registrations.
_counter = itertools.count()


def _key(base: str) -> str:
    return f"testing.lang_module.{base}_{next(_counter)}"


# ---------------------------------------------------------------------------
# Tier 2 — trait-driven (one fixture per trait kind)
# ---------------------------------------------------------------------------


@py_class(_key("BinAdd"))
class BinAdd(Object):
    """``BinOpTraits`` with sugared ``+`` op → ``__ffi_parse_op__[Add]``."""

    __ffi_ir_traits__ = tr.BinOpTraits("$field:lhs", "$field:rhs", "+", None, None)
    lhs: Any
    rhs: Any


@py_class(_key("BinFloorDiv"))
class BinFloorDiv(Object):
    """BinOp with both an op-kind and a literal func-name → both slots fire."""

    __ffi_ir_traits__ = tr.BinOpTraits(
        "$field:lhs",
        "$field:rhs",
        "//",
        None,
        "FloorDiv",
    )
    lhs: Any
    rhs: Any


@py_class(_key("UnaryNeg"))
class UnaryNeg(Object):
    """``UnaryOpTraits`` → ``__ffi_parse_op__[USub]``."""

    __ffi_ir_traits__ = tr.UnaryOpTraits("$field:operand", "-")
    operand: Any


@py_class(_key("Var"))
class Var(Object):
    """``ValueTraits`` → ``__ffi_parse_make_var__``."""

    __ffi_ir_traits__ = tr.ValueTraits("$field:name", None, None)
    name: str


@py_class(_key("IntImm"))
class IntImm(Object):
    """``LiteralTraits`` with format='int' → ``__ffi_parse_make_const__['int']``."""

    __ffi_ir_traits__ = tr.LiteralTraits("$field:value", "int")
    value: int


@py_class(_key("FloatImm"))
class FloatImm(Object):
    """``LiteralTraits`` with format='float' → ``__ffi_parse_make_const__['float']``."""

    __ffi_ir_traits__ = tr.LiteralTraits("$field:value", "float")
    value: float


@py_class(_key("StringLit"))
class StringLit(Object):
    """``LiteralTraits`` with format=None → ``__ffi_parse_make_const__['default']``."""

    __ffi_ir_traits__ = tr.LiteralTraits("$field:value", None)
    value: str


@py_class(_key("Call"))
class Call(Object):
    """``CallTraits`` with literal callee → ``T.my_op``."""

    __ffi_ir_traits__ = tr.CallTraits("my_op", "$field:args", None, None, None, None)
    args: Any


@py_class(_key("OpaqueCall"))
class OpaqueCall(Object):
    """``CallTraits`` with $method op → no auto-registration (waiting for
    a user override)."""

    __ffi_ir_traits__ = tr.CallTraits("$method:_resolve_callee", "$field:args", None, None, None, None)
    args: Any


@py_class(_key("BufferLoad"))
class BufferLoad(Object):
    """``LoadTraits`` → ``__ffi_parse_load__``."""

    __ffi_ir_traits__ = tr.LoadTraits("$field:buf", "$field:idx", None)
    buf: Any
    idx: Any


@py_class(_key("BufferStore"))
class BufferStore(Object):
    """``StoreTraits`` → ``__ffi_parse_store__``."""

    __ffi_ir_traits__ = tr.StoreTraits("$field:buf", "$field:val", "$field:idx", None)
    buf: Any
    val: Any
    idx: Any


@py_class(_key("Bind"))
class Bind(Object):
    """``AssignTraits`` with literal kind → ``T.bind``."""

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


@py_class(_key("Evaluate"))
class Evaluate(Object):
    """``AssignTraits`` with no kind → ``__ffi_parse_assign__``."""

    __ffi_ir_traits__ = tr.AssignTraits(None, "$field:rhs", None, None, None, None)
    rhs: Any


@py_class(_key("AssertStmt"))
class AssertStmt(Object):
    """``AssertTraits`` → ``__ffi_parse_assert__``."""

    __ffi_ir_traits__ = tr.AssertTraits("$field:cond", "$field:msg")
    cond: Any
    msg: Any


@py_class(_key("ReturnStmt"))
class ReturnStmt(Object):
    """``ReturnTraits`` → ``__ffi_parse_return__``."""

    __ffi_ir_traits__ = tr.ReturnTraits("$field:value")
    value: Any


@py_class(_key("PrimFunc"))
class PrimFunc(Object):
    """``FuncTraits`` with literal kind → ``T.prim_func``."""

    __ffi_ir_traits__ = tr.FuncTraits(
        "$field:name",
        tr.RegionTraits("$field:body", "$field:params", None, None),
        None,
        "prim_func",
        None,
    )
    name: str
    params: list
    body: list


@py_class(_key("AnonFunc"))
class AnonFunc(Object):
    """``FuncTraits`` with no kind → ``__ffi_parse_func__``."""

    __ffi_ir_traits__ = tr.FuncTraits(
        "$field:name",
        tr.RegionTraits("$field:body", "$field:params", None, None),
        None,
        None,
        None,
    )
    name: str
    params: list
    body: list


@py_class(_key("For"))
class For(Object):
    """``ForTraits`` with $method kind → ``__ffi_parse_for__`` (kinds via override)."""

    __ffi_ir_traits__ = tr.ForTraits(
        tr.RegionTraits("$field:body", None, None, None),
        "$field:start",
        "$field:end",
        "$field:step",
        None,
        None,
        None,
        "$method:_kind_prefix",
    )
    start: Any
    end: Any
    extent: Any
    step: Any
    body: list

    @parse_slot("extent")
    def _construct_extent(start: Any, end_value: Any) -> Any:  # noqa: N805
        return end_value - start


@py_class(_key("Block"))
class Block(Object):
    """``WithTraits`` → ``__ffi_parse_with__``."""

    __ffi_ir_traits__ = tr.WithTraits(
        tr.RegionTraits("$field:body", None, None, None),
        None,
        None,
        None,
        None,
        None,
        None,
    )
    body: list


@py_class(_key("WhileLoop"))
class WhileLoop(Object):
    """``WhileTraits`` → ``__ffi_parse_while__``."""

    __ffi_ir_traits__ = tr.WhileTraits(
        "$field:cond",
        tr.RegionTraits("$field:body", None, None, None),
    )
    cond: Any
    body: list


@py_class(_key("IfThenElse"))
class IfThenElse(Object):
    """``IfTraits`` → ``__ffi_parse_if__``."""

    __ffi_ir_traits__ = tr.IfTraits(
        "$field:cond",
        tr.RegionTraits("$field:then_body", None, None, None),
        tr.RegionTraits("$field:else_body", None, None, None),
    )
    cond: Any
    then_body: list
    else_body: list


@py_class(_key("PrimType"))
class PrimType(Object):
    """``PrimTyTraits`` — registration goes through dtype-handles, not
    a direct trait entry."""

    __ffi_ir_traits__ = tr.PrimTyTraits("$field:dtype")
    dtype: Any


@py_class(_key("TensorType"))
class TensorType(Object):
    """``TensorTyTraits`` → ``T.Tensor``."""

    __ffi_ir_traits__ = tr.TensorTyTraits("$field:shape", "$field:dtype", None)
    shape: Any
    dtype: Any


@py_class(_key("BufferType"))
class BufferType(Object):
    """``BufferTyTraits`` → ``T.Buffer``."""

    __ffi_ir_traits__ = tr.BufferTyTraits("$field:shape", "$field:dtype", None, None, None)
    shape: Any
    dtype: Any


@py_class(_key("FuncType"))
class FuncTypeNode(Object):
    """``FuncTyTraits`` → ``T.FuncType``."""

    __ffi_ir_traits__ = tr.FuncTyTraits("$field:params", "$field:ret")
    params: Any
    ret: Any


@py_class(_key("TupleType"))
class TupleType(Object):
    """``TupleTyTraits`` → ``T.Tuple``."""

    __ffi_ir_traits__ = tr.TupleTyTraits("$field:fields")
    fields: Any


@py_class(_key("ShapeType"))
class ShapeType(Object):
    """``ShapeTyTraits`` → ``T.Shape``."""

    __ffi_ir_traits__ = tr.ShapeTyTraits("$field:dims", "$field:ndim")
    dims: Any
    ndim: Any


# ---------------------------------------------------------------------------
# Tier 3 — no traits, no manual print
# ---------------------------------------------------------------------------


@py_class(_key("PlainNode"))
class PlainNode(Object):
    """No traits, no manual print → register class name as default callee."""

    payload: Any


# ---------------------------------------------------------------------------
# Module-scope @parse_hook overrides
# ---------------------------------------------------------------------------


@parse_hook("serial", "parallel", "unroll", "vectorized")
def _for_kinds(*_args: Any, **_kw: Any) -> Any:
    """Tier-2 ForTraits override that supplies the loop-kind names."""
    raise NotImplementedError


@parse_hook(callee=["MyOp"], op_kind=OperationKind.Add)
def _custom_op(*_args: Any, **_kw: Any) -> Any:
    """Tier-1 BinOp override; registers both ``T.MyOp`` and the op-kind dict."""
    raise NotImplementedError
