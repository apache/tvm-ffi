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
"""Internal: convert Python standard-library ``ast`` nodes to TVM-FFI text AST nodes.

This module is not part of the public API. Use :func:`tvm_ffi.pyast.from_py`
as the public entry point.
"""

from __future__ import annotations

import ast
import math
import textwrap
from typing import Callable

from tvm_ffi import pyast as tast

# ---------------------------------------------------------------------------
# Operator mapping tables
# ---------------------------------------------------------------------------

_UNARY_OP: dict[type, int] = {
    ast.USub: tast.OperationKind.USub,
    ast.UAdd: tast.OperationKind.UAdd,
    ast.Invert: tast.OperationKind.Invert,
    ast.Not: tast.OperationKind.Not,
}

_BINOP: dict[type, int] = {
    ast.Add: tast.OperationKind.Add,
    ast.Sub: tast.OperationKind.Sub,
    ast.Mult: tast.OperationKind.Mult,
    ast.Div: tast.OperationKind.Div,
    ast.FloorDiv: tast.OperationKind.FloorDiv,
    ast.Mod: tast.OperationKind.Mod,
    ast.Pow: tast.OperationKind.Pow,
    ast.LShift: tast.OperationKind.LShift,
    ast.RShift: tast.OperationKind.RShift,
    ast.BitAnd: tast.OperationKind.BitAnd,
    ast.BitOr: tast.OperationKind.BitOr,
    ast.BitXor: tast.OperationKind.BitXor,
    ast.MatMult: tast.OperationKind.MatMult,
}

_CMPOP: dict[type, int] = {
    ast.Lt: tast.OperationKind.Lt,
    ast.LtE: tast.OperationKind.LtE,
    ast.Gt: tast.OperationKind.Gt,
    ast.GtE: tast.OperationKind.GtE,
    ast.Eq: tast.OperationKind.Eq,
    ast.NotEq: tast.OperationKind.NotEq,
    ast.Is: tast.OperationKind.Is,
    ast.IsNot: tast.OperationKind.IsNot,
    ast.In: tast.OperationKind.In,
    ast.NotIn: tast.OperationKind.NotIn,
}

_BOOLOP: dict[type, int] = {
    ast.And: tast.OperationKind.And,
    ast.Or: tast.OperationKind.Or,
}


# ---------------------------------------------------------------------------
# _Converter — stateless recursive converter
# ---------------------------------------------------------------------------


class _Converter:
    """Converts Python ``ast`` nodes to TVM-FFI text AST nodes."""

    def __init__(self) -> None:
        self._expr_dispatch: dict[type, Callable[..., tast.Expr]] = {
            ast.Constant: self._convert_constant,
            ast.Name: self._convert_name,
            ast.Attribute: self._convert_attribute,
            ast.Subscript: self._convert_subscript,
            ast.Call: self._convert_call,
            ast.UnaryOp: self._convert_unaryop,
            ast.BinOp: self._convert_binop,
            ast.BoolOp: self._convert_boolop,
            ast.Compare: self._convert_compare,
            ast.IfExp: self._convert_ifexp,
            ast.Lambda: self._convert_lambda,
            ast.Tuple: self._convert_tuple,
            ast.List: self._convert_list,
            ast.Dict: self._convert_dict,
            ast.Slice: self._convert_slice,
            ast.Starred: self._convert_starred,
            ast.Set: self._convert_set,
            ast.ListComp: self._convert_listcomp,
            ast.SetComp: self._convert_setcomp,
            ast.DictComp: self._convert_dictcomp,
            ast.GeneratorExp: self._convert_generatorexp,
            ast.Yield: self._convert_yield,
            ast.YieldFrom: self._convert_yieldfrom,
            ast.JoinedStr: self._convert_joinedstr,
            ast.FormattedValue: self._convert_formattedvalue,
            ast.NamedExpr: self._convert_namedexpr,
            ast.Await: self._convert_await,
        }
        self._stmt_dispatch: dict[type, Callable[..., tast.Stmt | list[tast.Stmt]]] = {
            ast.Module: self._convert_module,
            ast.Assign: self._convert_assign,
            ast.AnnAssign: self._convert_annassign,
            ast.AugAssign: self._convert_augassign,
            ast.Expr: self._convert_expr_stmt,
            ast.If: self._convert_if,
            ast.While: self._convert_while,
            ast.For: self._convert_for,
            ast.AsyncFor: self._convert_for,
            ast.With: self._convert_with,
            ast.AsyncWith: self._convert_with,
            ast.FunctionDef: self._convert_functiondef,
            ast.AsyncFunctionDef: self._convert_functiondef,
            ast.ClassDef: self._convert_classdef,
            ast.Return: self._convert_return,
            ast.Assert: self._convert_assert,
            ast.Pass: self._convert_pass,
            ast.Break: self._convert_break,
            ast.Continue: self._convert_continue,
            ast.Import: self._convert_import,
            ast.ImportFrom: self._convert_importfrom,
            ast.Delete: self._convert_delete,
            ast.Raise: self._convert_raise,
            ast.Global: self._convert_global,
            ast.Nonlocal: self._convert_nonlocal,
            ast.Try: self._convert_try,
        }
        if hasattr(ast, "TryStar"):
            self._stmt_dispatch[ast.TryStar] = self._convert_try
        if hasattr(ast, "Match"):
            self._stmt_dispatch[ast.Match] = self._convert_match
        if hasattr(ast, "TypeAlias"):
            self._stmt_dispatch[ast.TypeAlias] = self._convert_typealias  # ty: ignore[unresolved-attribute]

    # -- span helper --------------------------------------------------------

    @staticmethod
    def _set_span(result: tast.Node, source: ast.AST) -> None:
        """Copy source location from a Python AST node to a TVM-FFI AST node."""
        result.lineno = getattr(source, "lineno", -1) or -1
        result.col_offset = getattr(source, "col_offset", -1)
        if result.col_offset is None:
            result.col_offset = -1
        result.end_lineno = getattr(source, "end_lineno", -1) or -1
        result.end_col_offset = getattr(source, "end_col_offset", -1)
        if result.end_col_offset is None:
            result.end_col_offset = -1

    # -- public entry points ------------------------------------------------

    def convert(self, node: ast.AST) -> tast.Node:
        """Convert a Python AST node to a TVM-FFI AST node."""
        if isinstance(node, ast.expr):
            return self.convert_expr(node)
        if isinstance(node, ast.stmt):
            result = self._dispatch_stmt(node)
            if isinstance(result, list):
                return tast.StmtBlock(result)  # ty: ignore[invalid-argument-type]
            return result
        if isinstance(node, ast.Module):
            return self._convert_module(node)
        if isinstance(node, ast.Expression):
            return self.convert_expr(node.body)
        if isinstance(node, ast.Interactive):
            stmts = self._convert_body(node.body)
            return tast.StmtBlock(stmts)
        raise NotImplementedError(f"Unsupported top-level AST node type: {type(node).__name__}")

    def convert_expr(self, node: ast.expr) -> tast.Expr:
        """Convert a Python expression AST node to a TVM-FFI Expr."""
        handler = self._expr_dispatch.get(type(node))
        if handler is None:
            raise NotImplementedError(
                f"Unsupported expression AST node type: {type(node).__name__}"
            )
        result = handler(node)
        self._set_span(result, node)
        return result

    # -- expression handlers ------------------------------------------------

    def _convert_constant(self, node: ast.Constant) -> tast.Expr:  # noqa: PLR0911
        value = node.value
        if isinstance(value, float) and math.isinf(value):
            return tast.Id("-1e999" if value < 0 else "1e999")
        if isinstance(value, float) and math.isnan(value):
            return tast.Id('float("nan")')
        if value is None or isinstance(value, (bool, float, str)):
            return tast.Literal(value)
        if isinstance(value, int):
            # int64 range check: FFI stores integers as int64_t
            if -(2**63) <= value < 2**63:
                return tast.Literal(value)
            return tast.Id(repr(value))
        if value is ...:
            return tast.Id("...")
        # For unsupported literal types (bytes, complex, tuple, frozenset),
        # fall back to rendering via repr
        return tast.Id(repr(value))

    def _convert_name(self, node: ast.Name) -> tast.Expr:
        return tast.Id(node.id)

    def _convert_attribute(self, node: ast.Attribute) -> tast.Expr:
        return tast.Attr(self.convert_expr(node.value), node.attr)

    def _convert_subscript(self, node: ast.Subscript) -> tast.Expr:
        obj = self.convert_expr(node.value)
        slc = node.slice
        has_starred = isinstance(slc, ast.Tuple) and any(
            isinstance(e, ast.Starred) for e in slc.elts
        )
        if isinstance(slc, ast.Tuple) and len(slc.elts) != 1 and not has_starred:
            # Multi-element tuple without Starred: x[a, b] → Index(obj, [a, b])
            indices = [self.convert_expr(e) for e in slc.elts]
        else:
            # Single value, single-element tuple, or tuple with Starred: keep as one index.
            # For Starred in subscript, preserving the Tuple wrapper renders as
            # x[(*a, b)] which is valid on Python 3.9+, unlike x[*a, b] (3.11+).
            # For x[1,] (single-element tuple), this becomes Index(obj, [Tuple([1])])
            # which renders as x[(1,)] — semantically equivalent, roundtrips correctly.
            indices = [self.convert_expr(slc)]
        return tast.Index(obj, indices)

    def _convert_call(self, node: ast.Call) -> tast.Expr:
        callee = self.convert_expr(node.func)
        args = [self.convert_expr(a) for a in node.args]
        kwargs_keys: list[str] = []
        kwargs_values: list[tast.Expr] = []
        for kw in node.keywords:
            kwargs_keys.append(kw.arg if kw.arg is not None else "")
            kwargs_values.append(self.convert_expr(kw.value))
        return tast.Call(callee, args, kwargs_keys, kwargs_values)

    def _convert_unaryop(self, node: ast.UnaryOp) -> tast.Expr:
        operand = self.convert_expr(node.operand)
        kind = _UNARY_OP.get(type(node.op))
        if kind is None:
            raise NotImplementedError(f"Unsupported unary operator: {type(node.op).__name__}")
        return tast.Operation(kind, [operand])

    def _convert_binop(self, node: ast.BinOp) -> tast.Expr:
        kind = _BINOP.get(type(node.op))
        if kind is None:
            raise NotImplementedError(f"Unsupported binary operator: {type(node.op).__name__}")
        left = self.convert_expr(node.left)
        right = self.convert_expr(node.right)
        return tast.Operation(kind, [left, right])

    def _convert_boolop(self, node: ast.BoolOp) -> tast.Expr:
        kind = _BOOLOP[type(node.op)]
        values: list[tast.Expr] = []
        for v in node.values:
            converted = self.convert_expr(v)
            # Wrap nested same-operator BoolOps in Parens to preserve grouping.
            # Without parens, "a and b and c" re-parses as flat And([a,b,c]),
            # losing the nested And([And([a,b]), c]) structure.
            if isinstance(v, ast.BoolOp) and type(v.op) is type(node.op):
                converted = tast.Operation(tast.OperationKind.Parens, [converted])
            values.append(converted)
        return tast.Operation(kind, values)

    def _convert_compare(self, node: ast.Compare) -> tast.Expr:
        left = self.convert_expr(node.left)
        # Wrap Compare left in parens if it's itself a Compare, to prevent
        # (a == b) == c from becoming the chained a == b == c.
        if isinstance(node.left, ast.Compare):
            left = tast.Operation(tast.OperationKind.Parens, [left])
        if len(node.ops) == 1:
            right = self.convert_expr(node.comparators[0])
            kind = _CMPOP.get(type(node.ops[0]))
            if kind is None:
                raise NotImplementedError(
                    f"Unsupported comparison operator: {type(node.ops[0]).__name__}"
                )
            return tast.Operation(kind, [left, right])
        # Chained comparison: a < b < c  ->  Operation(ChainedCompare, [a, Literal(Lt), b, ...])
        operands: list[tast.Expr] = [left]
        for op, comparator in zip(node.ops, node.comparators):
            kind = _CMPOP.get(type(op))
            if kind is None:
                raise NotImplementedError(f"Unsupported comparison operator: {type(op).__name__}")
            operands.append(tast.Literal(kind))
            operands.append(self.convert_expr(comparator))
        return tast.Operation(tast.OperationKind.ChainedCompare, operands)

    def _convert_ifexp(self, node: ast.IfExp) -> tast.Expr:
        test = self.convert_expr(node.test)
        body = self.convert_expr(node.body)
        orelse = self.convert_expr(node.orelse)
        # Wrap body in Parens if it is itself a ternary, because without parens
        # "B if A else C if X else Z" parses as "B if A else (C if X else Z)"
        # which is a different structure from "(B if A else C) if X else Z".
        if isinstance(node.body, ast.IfExp):
            body = tast.Operation(tast.OperationKind.Parens, [body])
        return tast.Operation(tast.OperationKind.IfThenElse, [test, body, orelse])

    def _convert_lambda(self, node: ast.Lambda) -> tast.Expr:
        converted_args = self._convert_arguments(node.args)
        args: list[tast.Expr] = []
        for a in converted_args:
            if a.rhs is not None or a.annotation is not None:
                # Render "name: ann = default" or "name=default" as a text Id
                # since Lambda args are List<ExprAST> (can't hold Assign stmts).
                parts = a.lhs.to_python()
                if a.annotation is not None:
                    parts += ": " + a.annotation.to_python()
                if a.rhs is not None:
                    parts += "=" + a.rhs.to_python()
                args.append(tast.Id(parts))
            else:
                args.append(a.lhs)
        body = self.convert_expr(node.body)
        return tast.Lambda(args, body)

    def _convert_tuple(self, node: ast.Tuple) -> tast.Expr:
        return tast.Tuple([self.convert_expr(e) for e in node.elts])

    def _convert_list(self, node: ast.List) -> tast.Expr:
        return tast.List([self.convert_expr(e) for e in node.elts])

    def _convert_dict(self, node: ast.Dict) -> tast.Expr:
        keys: list[tast.Expr] = []
        values: list[tast.Expr] = []
        for k, v in zip(node.keys, node.values):
            if k is None:
                # Dictionary unpacking: {**d} → use StarredExpr as a sentinel key
                keys.append(tast.StarredExpr(tast.StarredExpr(self.convert_expr(v))))
            else:
                keys.append(self.convert_expr(k))
            values.append(self.convert_expr(v))
        return tast.Dict(keys, values)

    def _convert_slice(self, node: ast.Slice) -> tast.Expr:
        start = self.convert_expr(node.lower) if node.lower else None
        stop = self.convert_expr(node.upper) if node.upper else None
        step = self.convert_expr(node.step) if node.step else None
        return tast.Slice(start, stop, step)

    def _convert_starred(self, node: ast.Starred) -> tast.Expr:
        value = self.convert_expr(node.value)
        # Wrap ternary values in parens: *([x] if c else []) must keep parens
        if isinstance(node.value, ast.IfExp):
            value = tast.Operation(tast.OperationKind.Parens, [value])
        return tast.StarredExpr(value)

    def _convert_set(self, node: ast.Set) -> tast.Expr:
        return tast.Set([self.convert_expr(e) for e in node.elts])

    def _convert_comprehension_iters(
        self, generators: list[ast.comprehension]
    ) -> list[tast.ComprehensionIter]:
        result: list[tast.ComprehensionIter] = []
        for gen in generators:
            target = self.convert_expr(gen.target)
            iter_expr = self.convert_expr(gen.iter)
            # Wrap ternary iters in parens to avoid ambiguity with filter `if`
            if isinstance(gen.iter, ast.IfExp):
                iter_expr = tast.Operation(tast.OperationKind.Parens, [iter_expr])
            ifs = [self.convert_expr(c) for c in gen.ifs]
            result.append(tast.ComprehensionIter(target, iter_expr, ifs))
        return result

    def _convert_listcomp(self, node: ast.ListComp) -> tast.Expr:
        elt = self.convert_expr(node.elt)
        iters = self._convert_comprehension_iters(node.generators)
        return tast.Comprehension(tast.ComprehensionKind.List, elt, None, iters)

    def _convert_setcomp(self, node: ast.SetComp) -> tast.Expr:
        elt = self.convert_expr(node.elt)
        iters = self._convert_comprehension_iters(node.generators)
        return tast.Comprehension(tast.ComprehensionKind.Set, elt, None, iters)

    def _convert_dictcomp(self, node: ast.DictComp) -> tast.Expr:
        key = self.convert_expr(node.key)
        value = self.convert_expr(node.value)
        iters = self._convert_comprehension_iters(node.generators)
        return tast.Comprehension(tast.ComprehensionKind.Dict, key, value, iters)

    def _convert_generatorexp(self, node: ast.GeneratorExp) -> tast.Expr:
        elt = self.convert_expr(node.elt)
        iters = self._convert_comprehension_iters(node.generators)
        return tast.Comprehension(tast.ComprehensionKind.Generator, elt, None, iters)

    def _convert_yield(self, node: ast.Yield) -> tast.Expr:
        value = self.convert_expr(node.value) if node.value else None
        return tast.Yield(value)

    def _convert_yieldfrom(self, node: ast.YieldFrom) -> tast.Expr:
        return tast.YieldFrom(self.convert_expr(node.value))

    def _convert_joinedstr(self, node: ast.JoinedStr) -> tast.Expr:
        values: list[tast.Expr] = []
        for v in node.values:
            values.append(self.convert_expr(v))
        return tast.FStr(values)

    def _convert_formattedvalue(self, node: ast.FormattedValue) -> tast.Expr:
        value = self.convert_expr(node.value)
        conversion = node.conversion
        format_spec = self.convert_expr(node.format_spec) if node.format_spec else None
        return tast.FStrValue(value, conversion, format_spec)

    def _convert_namedexpr(self, node: ast.NamedExpr) -> tast.Expr:
        return tast.WalrusExpr(self.convert_expr(node.target), self.convert_expr(node.value))

    def _convert_await(self, node: ast.Await) -> tast.Expr:
        return tast.AwaitExpr(self.convert_expr(node.value))

    # -- statement helpers --------------------------------------------------

    def _dispatch_stmt(self, node: ast.stmt) -> tast.Stmt | list[tast.Stmt]:
        handler = self._stmt_dispatch.get(type(node))
        if handler is None:
            raise NotImplementedError(f"Unsupported statement AST node type: {type(node).__name__}")
        result = handler(node)
        if isinstance(result, list):
            for r in result:
                self._set_span(r, node)  # ty: ignore[invalid-argument-type]
        else:
            self._set_span(result, node)
        return result

    def _convert_body(
        self, stmts: list[ast.stmt], *, detect_docstring: bool = False
    ) -> list[tast.Stmt]:
        result: list[tast.Stmt] = []
        for i, stmt in enumerate(stmts):
            if (
                detect_docstring
                and i == 0
                and isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Constant)
                and isinstance(stmt.value.value, str)
            ):
                result.append(tast.DocString(stmt.value.value))
                continue
            converted = self._dispatch_stmt(stmt)
            if isinstance(converted, list):
                result.extend(converted)  # ty: ignore[invalid-argument-type]
            else:
                result.append(converted)
        return result

    # -- statement handlers -------------------------------------------------

    def _convert_module(self, node: ast.Module) -> tast.Stmt:
        stmts = self._convert_body(node.body, detect_docstring=True)
        return tast.StmtBlock(stmts)

    def _convert_assign(self, node: ast.Assign) -> tast.Stmt:
        rhs = self.convert_expr(node.value)
        if len(node.targets) == 1:
            return tast.Assign(lhs=self._convert_target(node.targets[0]), rhs=rhs)
        # Multi-target: a = b = c → Assign(lhs=Parens(Tuple([a, b])), rhs=c)
        targets = [self._convert_target(t) for t in node.targets]
        lhs = tast.Operation(tast.OperationKind.Parens, [tast.Tuple(targets)])
        return tast.Assign(lhs=lhs, rhs=rhs)

    def _convert_target(self, node: ast.expr) -> tast.Expr:
        """Convert an assignment target, preserving List vs Tuple distinction."""
        if isinstance(node, ast.List):
            return tast.List([self._convert_target(e) for e in node.elts])
        if isinstance(node, ast.Tuple):
            return tast.Tuple([self._convert_target(e) for e in node.elts])
        if isinstance(node, ast.Starred):
            return tast.StarredExpr(self._convert_target(node.value))
        return self.convert_expr(node)

    def _convert_annassign(self, node: ast.AnnAssign) -> tast.Stmt:
        lhs = self.convert_expr(node.target)
        annotation = self.convert_expr(node.annotation)
        rhs = self.convert_expr(node.value) if node.value else None
        return tast.Assign(lhs=lhs, rhs=rhs, annotation=annotation)

    def _convert_augassign(self, node: ast.AugAssign) -> tast.Stmt:
        kind = _BINOP.get(type(node.op))
        if kind is None:
            raise NotImplementedError(
                f"Unsupported augmented assignment operator: {type(node.op).__name__}"
            )
        lhs = self.convert_expr(node.target)
        rhs = self.convert_expr(node.value)
        return tast.Assign(lhs=lhs, rhs=rhs, aug_op=kind)

    def _convert_expr_stmt(self, node: ast.Expr) -> tast.Stmt:
        return tast.ExprStmt(self.convert_expr(node.value))

    def _convert_if(self, node: ast.If) -> tast.Stmt:
        cond = self.convert_expr(node.test)
        then_branch = self._convert_body(node.body)
        else_branch = self._convert_body(node.orelse)
        return tast.If(cond, then_branch, else_branch)

    def _convert_while(self, node: ast.While) -> tast.Stmt:
        cond = self.convert_expr(node.test)
        body = self._convert_body(node.body)
        orelse = self._convert_body(node.orelse) if node.orelse else []
        return tast.While(cond, body, orelse)

    def _convert_for(self, node: ast.For | ast.AsyncFor) -> tast.Stmt:
        lhs = self.convert_expr(node.target)
        rhs = self.convert_expr(node.iter)
        body = self._convert_body(node.body)
        is_async = isinstance(node, ast.AsyncFor)
        orelse = self._convert_body(node.orelse) if node.orelse else []
        return tast.For(lhs, rhs, body, is_async, orelse)

    def _convert_with(self, node: ast.With | ast.AsyncWith) -> tast.Stmt:
        is_async = isinstance(node, ast.AsyncWith)
        body_stmts = self._convert_body(node.body)
        if len(node.items) == 1:
            item = node.items[0]
            rhs = self.convert_expr(item.context_expr)
            lhs = self.convert_expr(item.optional_vars) if item.optional_vars else None
            return tast.With(lhs, rhs, body_stmts, is_async)
        # Multi-item with: encode as Tuple of context exprs / targets
        rhs_elts = [self.convert_expr(item.context_expr) for item in node.items]
        lhs_elts = [
            self.convert_expr(item.optional_vars) if item.optional_vars else tast.Id("")
            for item in node.items
        ]
        return tast.With(tast.Tuple(lhs_elts), tast.Tuple(rhs_elts), body_stmts, is_async)

    @staticmethod
    def _make_name_with_type_params(name: str, type_params: list) -> tast.Id:
        """Build an Id with type params appended, e.g. 'Foo[T, U: int]'."""
        if not type_params:
            return tast.Id(name)
        parts: list[str] = []
        for tp in type_params:
            if isinstance(tp, ast.TypeVar):  # ty: ignore[unresolved-attribute]
                s = tp.name
                if tp.bound:
                    s += ": " + ast.unparse(tp.bound)
                parts.append(s)
            elif isinstance(tp, ast.TypeVarTuple):  # ty: ignore[unresolved-attribute]
                parts.append("*" + tp.name)
            elif isinstance(tp, ast.ParamSpec):  # ty: ignore[unresolved-attribute]
                parts.append("**" + tp.name)
            else:
                parts.append(str(tp.name))  # pragma: no cover
        return tast.Id(name + "[" + ", ".join(parts) + "]")

    def _convert_functiondef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> tast.Stmt:
        name = self._make_name_with_type_params(node.name, getattr(node, "type_params", []))
        args = self._convert_arguments(node.args)
        decorators = [self.convert_expr(d) for d in node.decorator_list]
        return_type = self.convert_expr(node.returns) if node.returns else None
        body = self._convert_body(node.body, detect_docstring=True)
        is_async = isinstance(node, ast.AsyncFunctionDef)
        return tast.Function(name, args, decorators, return_type, body, is_async)

    def _convert_arguments(self, args: ast.arguments) -> list[tast.Assign]:
        result: list[tast.Assign] = []
        # Positional-only args first, then a / separator
        all_args = list(args.posonlyargs) + list(args.args)
        posonly_count = len(args.posonlyargs)
        # Defaults are right-aligned to args
        num_defaults = len(args.defaults)
        num_no_default = len(all_args) - num_defaults
        for i, arg in enumerate(all_args):
            lhs = tast.Id(arg.arg)
            annotation = self.convert_expr(arg.annotation) if arg.annotation else None
            default_idx = i - num_no_default
            rhs = self.convert_expr(args.defaults[default_idx]) if default_idx >= 0 else None
            result.append(tast.Assign(lhs=lhs, rhs=rhs, annotation=annotation))
            # Insert / separator after the last positional-only arg
            if posonly_count > 0 and i == posonly_count - 1:
                result.append(tast.Assign(lhs=tast.Id("/")))
        # *args or bare * separator
        if args.vararg or args.kwonlyargs:
            if args.vararg:
                vararg_name = tast.StarredExpr(tast.Id(args.vararg.arg))
                ann = self.convert_expr(args.vararg.annotation) if args.vararg.annotation else None
                result.append(tast.Assign(lhs=vararg_name, annotation=ann))
            else:
                # bare * separator for keyword-only args
                result.append(tast.Assign(lhs=tast.StarredExpr(tast.Id(""))))
        # Keyword-only args
        for i, arg in enumerate(args.kwonlyargs):
            lhs = tast.Id(arg.arg)
            annotation = self.convert_expr(arg.annotation) if arg.annotation else None
            default = args.kw_defaults[i]
            rhs = self.convert_expr(default) if default else None
            result.append(tast.Assign(lhs=lhs, rhs=rhs, annotation=annotation))
        # **kwargs
        if args.kwarg:
            kwarg_name = tast.StarredExpr(tast.StarredExpr(tast.Id(args.kwarg.arg)))
            ann = self.convert_expr(args.kwarg.annotation) if args.kwarg.annotation else None
            result.append(tast.Assign(lhs=kwarg_name, annotation=ann))
        return result

    def _convert_classdef(self, node: ast.ClassDef) -> tast.Stmt:
        name = self._make_name_with_type_params(node.name, getattr(node, "type_params", []))
        bases = [self.convert_expr(b) for b in node.bases]
        decorators = [self.convert_expr(d) for d in node.decorator_list]
        body = self._convert_body(node.body, detect_docstring=True)
        kwargs_keys = [kw.arg for kw in node.keywords if kw.arg is not None]
        kwargs_values = [self.convert_expr(kw.value) for kw in node.keywords if kw.arg is not None]
        return tast.Class(name, bases, decorators, body, kwargs_keys, kwargs_values)

    def _convert_return(self, node: ast.Return) -> tast.Stmt:
        value = self.convert_expr(node.value) if node.value else None
        return tast.Return(value)

    def _convert_assert(self, node: ast.Assert) -> tast.Stmt:
        cond = self.convert_expr(node.test)
        msg = self.convert_expr(node.msg) if node.msg else None
        return tast.Assert(cond, msg)

    def _convert_pass(self, node: ast.Pass) -> tast.Stmt:
        return tast.ExprStmt(tast.Id("pass"))

    def _convert_break(self, node: ast.Break) -> tast.Stmt:
        return tast.ExprStmt(tast.Id("break"))

    def _convert_continue(self, node: ast.Continue) -> tast.Stmt:
        return tast.ExprStmt(tast.Id("continue"))

    def _convert_import(self, node: ast.Import) -> tast.Stmt:
        parts: list[str] = []
        for alias in node.names:
            if alias.asname:
                parts.append(f"{alias.name} as {alias.asname}")
            else:
                parts.append(alias.name)
        return tast.ExprStmt(tast.Id(f"import {', '.join(parts)}"))

    def _convert_importfrom(self, node: ast.ImportFrom) -> tast.Stmt:
        module = node.module or ""
        prefix = "." * (node.level or 0)
        names: list[str] = []
        for alias in node.names:
            if alias.asname:
                names.append(f"{alias.name} as {alias.asname}")
            else:
                names.append(alias.name)
        return tast.ExprStmt(tast.Id(f"from {prefix}{module} import {', '.join(names)}"))

    def _convert_delete(self, node: ast.Delete) -> tast.Stmt:
        targets_str = ", ".join(ast.unparse(t) for t in node.targets)
        return tast.ExprStmt(tast.Id(f"del {targets_str}"))

    def _convert_raise(self, node: ast.Raise) -> tast.Stmt:
        if node.exc is None:
            return tast.ExprStmt(tast.Id("raise"))
        if node.cause:
            return tast.ExprStmt(
                tast.Id(f"raise {ast.unparse(node.exc)} from {ast.unparse(node.cause)}")
            )
        return tast.ExprStmt(tast.Id(f"raise {ast.unparse(node.exc)}"))

    def _convert_global(self, node: ast.Global) -> tast.Stmt:
        return tast.ExprStmt(tast.Id(f"global {', '.join(node.names)}"))

    def _convert_nonlocal(self, node: ast.Nonlocal) -> tast.Stmt:
        return tast.ExprStmt(tast.Id(f"nonlocal {', '.join(node.names)}"))

    def _convert_try(self, node: ast.Try) -> tast.Stmt:
        body = self._convert_body(node.body)
        handlers: list[tast.ExceptHandler] = []
        for h in node.handlers:
            h_type = self.convert_expr(h.type) if h.type else None
            h_body = self._convert_body(h.body)
            handlers.append(tast.ExceptHandler(h_type, h.name, h_body))
        orelse = self._convert_body(node.orelse) if node.orelse else []
        finalbody = self._convert_body(node.finalbody) if node.finalbody else []
        return tast.Try(body, handlers, orelse, finalbody)

    def _convert_match(self, node: ast.Match) -> tast.Stmt:  # ty: ignore[unresolved-attribute]
        subject = self.convert_expr(node.subject)
        cases: list[tast.MatchCase] = []
        for case in node.cases:
            pattern = self._convert_match_pattern(case.pattern)
            guard = self.convert_expr(case.guard) if case.guard else None
            case_body = self._convert_body(case.body)
            cases.append(tast.MatchCase(pattern, guard, case_body))
        return tast.Match(subject, cases)

    def _convert_match_pattern(self, node: ast.pattern) -> tast.Expr:  # ty: ignore[unresolved-attribute]
        """Convert match patterns to expression AST using best-effort mapping."""
        # Use ast.unparse for a faithful text representation
        return tast.Id(ast.unparse(node))

    def _convert_typealias(self, node: ast.TypeAlias) -> tast.Stmt:  # ty: ignore[unresolved-attribute]
        """Convert ``type X = ...`` (PEP 695) to an Assign with ``type`` prefix."""
        type_params = getattr(node, "type_params", [])
        name_str = node.name.id  # ty: ignore[unresolved-attribute]
        if type_params:
            name_id = self._make_name_with_type_params(name_str, type_params)
        else:
            name_id = tast.Id(name_str)
        # Render as "type X = value" by prefixing "type " in the lhs name
        lhs = tast.Id("type " + name_id.name)
        rhs = self.convert_expr(node.value)
        return tast.Assign(lhs=lhs, rhs=rhs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_converter = _Converter()


def ast_translate(source: str | ast.AST) -> tast.Node:
    """Convert a Python source string or ``ast.AST`` node to a TVM-FFI AST node.

    Parameters
    ----------
    source
        Either a Python source-code string (parsed with ``ast.parse``) or an
        already-parsed ``ast.AST`` node.

    Returns
    -------
    node
        The corresponding TVM-FFI text AST node.

    Raises
    ------
    NotImplementedError
        If the source contains Python constructs that have no TVM-FFI AST
        equivalent (e.g. ``try``/``except``, f-strings, comprehensions).

    Examples
    --------
    .. code-block:: python

        from tvm_ffi.pyast.ast import from_py

        # From source string
        node = from_py("x + 1")
        print(node.to_python())  # x + 1

        # From ast node
        import ast

        tree = ast.parse("y = 42")
        node = from_py(tree)
        node.print_python()  # y = 42

    Note
    ----
    This function is internal. The public entry point is
    :func:`tvm_ffi.pyast.from_py`.

    """
    if isinstance(source, str):
        source = textwrap.dedent(source)
        tree = ast.parse(source)
        return _converter.convert(tree)
    return _converter.convert(source)
