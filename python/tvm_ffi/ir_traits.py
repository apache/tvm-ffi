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
"""Semantic trait types for IR nodes.

Traits describe WHAT each IR node IS (a loop, a value, a function)
and are consumed by the text printer, optimizers, and other tools.
"""

from __future__ import annotations

from tvm_ffi import Object
from tvm_ffi.core import _lookup_type_attr
from tvm_ffi.dataclasses import c_class

# ---- Region ----


# ---- Base trait ----


@c_class("ffi.ir_traits.IRTraits", init=False)
class IRTraits(Object):
    """Base class for all semantic traits."""


Trait = IRTraits  # backward compatibility alias


@c_class("ffi.ir_traits.ExprTraits", init=False)
class ExprTraits(IRTraits):
    """Base class for expression-level traits (produce a value)."""


@c_class("ffi.ir_traits.StmtTraits", init=False)
class StmtTraits(IRTraits):
    """Base class for statement-level traits (side effects, control flow, scope)."""


# ---- Region ----


@c_class("ffi.ir_traits.RegionTraits")
class RegionTraits(IRTraits):
    """A region describes a scoped body with optional defs and return."""

    body: str
    def_values: str | None
    def_expr: str | None
    ret: str | None


# ---- Expression traits ----


@c_class("ffi.ir_traits.BinOpTraits")
class BinOpTraits(ExprTraits):
    """Binary operation trait (e.g. add, sub, mul)."""

    lhs: str
    rhs: str
    op: str
    text_printer_sugar_check: str | None
    text_printer_func_name: str | None


@c_class("ffi.ir_traits.UnaryOpTraits")
class UnaryOpTraits(ExprTraits):
    """Unary operation trait (e.g. neg, not)."""

    operand: str
    op: str


@c_class("ffi.ir_traits.ValueTraits")
class ValueTraits(ExprTraits):
    """Named value trait with optional type annotation."""

    name: str
    ty: str | None
    text_printer_type: str | None


@c_class("ffi.ir_traits.LiteralTraits")
class LiteralTraits(ExprTraits):
    """Literal / leaf value trait."""

    value: str
    format: str | None


@c_class("ffi.ir_traits.CallTraits")
class CallTraits(ExprTraits):
    """Call expression trait (function call with op and args)."""

    op: str
    args: str
    attrs: str | None
    kwargs: str | None
    text_printer_callee: str | None


@c_class("ffi.ir_traits.LoadTraits")
class LoadTraits(ExprTraits):
    """Load trait for reading from a source with optional indices."""

    source: str
    indices: str | None
    predicate: str | None


# ---- Statement traits ----


@c_class("ffi.ir_traits.AssignTraits")
class AssignTraits(StmtTraits):
    """Assignment / expression-statement trait.

    When def_values is set: renders as ``def_values = rhs`` (assignment).
    When def_values is absent: renders ``rhs`` as a statement (expression-statement).
    """

    def_values: str | None
    rhs: str
    text_printer_pre: str | None
    text_printer_post: str | None
    text_printer_kind: str | None
    text_printer_return_check: str | None


@c_class("ffi.ir_traits.StoreTraits")
class StoreTraits(StmtTraits):
    """Store trait for writing a value to a target with optional indices."""

    target: str
    value: str
    indices: str | None
    predicate: str | None


@c_class("ffi.ir_traits.AssertTraits")
class AssertTraits(StmtTraits):
    """Assertion trait with a condition and optional message."""

    cond: str
    message: str | None


@c_class("ffi.ir_traits.ReturnTraits")
class ReturnTraits(StmtTraits):
    """Return trait carrying a value."""

    value: str


@c_class("ffi.ir_traits.FuncTraits")
class FuncTraits(StmtTraits):
    """Function definition trait with symbol name and body region."""

    symbol: str
    region: RegionTraits
    attrs: str | None
    text_printer_kind: str | None
    text_printer_pre: str | None


@c_class("ffi.ir_traits.ForTraits")
class ForTraits(StmtTraits):
    """For-loop trait with iteration range and body region."""

    region: RegionTraits
    start: str | None
    end: str | None
    step: str | None
    def_carry: str | None
    carry_init: str | None
    attrs: str | None
    text_printer_kind: str | None


@c_class("ffi.ir_traits.WithTraits")
class WithTraits(StmtTraits):
    """Context-manager / sequence trait.

    When text_printer_kind is set: renders as ``with kind(...) as var:`` block.
    When text_printer_kind is absent and region has no defs: renders as inline sequence.
    """

    region: RegionTraits
    def_carry: str | None
    carry_init: str | None
    text_printer_kind: str | None
    text_printer_pre: str | None
    text_printer_post: str | None
    text_printer_no_frame: bool | None


@c_class("ffi.ir_traits.WhileTraits")
class WhileTraits(StmtTraits):
    """While-loop trait with condition and body region."""

    cond: str
    region: RegionTraits


@c_class("ffi.ir_traits.IfTraits")
class IfTraits(StmtTraits):
    """If-else conditional trait with then/else regions."""

    cond: str
    then_region: RegionTraits
    else_region: RegionTraits | None


# ---- Type traits ----


@c_class("ffi.ir_traits.TyTraits", init=False)
class TyTraits(IRTraits):
    """Base class for type-level traits."""


@c_class("ffi.ir_traits.TensorTyTraits")
class TensorTyTraits(TyTraits):
    """Tensor type trait with shape, dtype, and device."""

    shape: str | None
    dtype: str | None
    device: str | None


@c_class("ffi.ir_traits.BufferTyTraits")
class BufferTyTraits(TyTraits):
    """Buffer type trait with shape, dtype, strides, offset, and scope."""

    shape: str
    dtype: str
    strides: str | None
    offset: str | None
    scope: str | None


@c_class("ffi.ir_traits.PrimTyTraits")
class PrimTyTraits(TyTraits):
    """Primitive scalar type trait with dtype."""

    dtype: str


@c_class("ffi.ir_traits.FuncTyTraits")
class FuncTyTraits(TyTraits):
    """Function type trait with parameter and return types."""

    params: str | None
    ret: str | None


@c_class("ffi.ir_traits.TupleTyTraits")
class TupleTyTraits(TyTraits):
    """Tuple type trait with field types."""

    fields: str


@c_class("ffi.ir_traits.ShapeTyTraits")
class ShapeTyTraits(TyTraits):
    """Shape type trait with dimension info."""

    dims: str | None
    ndim: str | None


# ---- Query helpers ----


def get_trait(obj: Object) -> IRTraits | None:
    """Retrieve the semantic trait for an IR object, or None."""
    info = type(obj).__tvm_ffi_type_info__  # type: ignore[attr-defined]
    return _lookup_type_attr(info.type_index, "__ffi_ir_traits__")


def has_trait(obj: Object, trait_cls: type[IRTraits] | None = None) -> bool:
    """Check if obj has a trait (optionally of a specific type)."""
    t = get_trait(obj)
    if t is None:
        return False
    return trait_cls is None or isinstance(t, trait_cls)


def get_type_trait(obj: Object) -> TyTraits | None:
    """Return the Ty-family trait, or None."""
    t = get_trait(obj)
    return t if isinstance(t, TyTraits) else None
