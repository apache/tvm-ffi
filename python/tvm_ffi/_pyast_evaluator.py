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
"""Runtime expression evaluator for ``tvm_ffi.pyast`` expression trees.

Given a :class:`tvm_ffi.pyast.Expr` tree and a variable scope, the evaluator
produces a Python value by walking the tree. It supports user-defined
operator overrides via an :class:`OperatorDispatch` registry.

The evaluator is a pure tree-walking interpreter; it never calls
``compile``, ``eval``, or ``exec``, never synthesizes source text, and never
round-trips through the stdlib :mod:`ast` module.
"""

from __future__ import annotations

import builtins
import dataclasses as dc
import operator
from collections import ChainMap
from collections.abc import Iterable, Iterator, Mapping
from functools import cached_property
from typing import Any, Callable

from tvm_ffi import pyast

__all__ = [
    "DEFAULT_DISPATCH",
    "BindValue",
    "EvaluationError",
    "ExprEvaluator",
    "OperatorDispatch",
    "UndefinedNameError",
    "eval_assign",
    "eval_expr",
]


class EvaluationError(Exception):
    """Base exception raised for any expression evaluation failure.

    Attributes
    ----------
    node
        The :class:`tvm_ffi.pyast.Node` that triggered the failure, when one
        is available. Callers can inspect this field (for example to
        highlight source spans via ``node.source_paths`` / ``node.lineno``)
        without parsing the exception message.

    """

    node: pyast.Node | None

    def __init__(self, message: str, *, node: pyast.Node | None = None) -> None:
        super().__init__(message)
        self.node = node


class UndefinedNameError(EvaluationError, NameError):
    """Raised when an :class:`~tvm_ffi.pyast.Id` is not resolvable."""


@dc.dataclass
class OperatorDispatch:
    """Registry mapping ``(op, operand_type, operand_index)`` to a handler.

    The key has three axes:

    - **``op``** — a :class:`~tvm_ffi.pyast.OperationKind` enum value (e.g.
      ``Add``, ``USub``, ``Lt``, ``And``, ``IfThenElse``).
    - **``operand_type``** — a Python :class:`type` matched against an
      operand's runtime type via MRO walk. Register on a base class to
      have every subclass inherit the handler.
    - **``operand_index``** — the 0-based position of the operand to match
      on. Register at ``operand_index=1`` (or at both ``0`` **and** ``1``)
      when the "interesting" operand may appear on the right-hand side,
      e.g. ``1 + sym`` or ``True and sym``.

    :meth:`lookup` scans operands left-to-right; at each position it walks
    ``type(operand).__mro__`` and returns the first handler registered at
    ``(op, cls, i)``. If nothing matches, :meth:`invoke` falls back to the
    native Python operator via :data:`_NATIVE_HANDLERS` so unregistered
    types keep working (``int + int``, ``list + list``, ``str == str``…).

    Handler signature
    -----------------
    Handlers receive all operand values positionally, plus a required
    keyword-only ``node`` argument carrying the originating
    :class:`~tvm_ffi.pyast.Node` so they can anchor error messages at the
    correct source span. Declare it explicitly (``def h(a, b, *, node=None):``)
    or absorb it with ``**kwargs`` (``def h(a, b, **_): ...``). Handlers
    that don't accept ``node`` will raise :exc:`TypeError` at invoke time.

    How the evaluator uses this registry
    ------------------------------------
    Different op kinds drive :meth:`lookup` / :meth:`invoke` differently;
    handlers should be registered with the corresponding calling
    convention in mind:

    - **Standard binary / unary / comparison ops** (``Add``, ``Sub``,
      ``USub``, ``Not``, ``Lt``, …): all operands are evaluated first,
      then a single ``invoke`` call runs the handler with the full tuple.
      First hit across MRO x operand position wins.

    - **``IfThenElse``**: only the condition is evaluated first. The
      evaluator probes ``lookup(IfThenElse, (cond,))`` — operand 0 only —
      to decide between eager-pick and eager-both. On a match, **both**
      branches evaluate and the handler is called with
      ``(cond, then, else)``. Otherwise Python's eager-pick semantics
      apply (the unused branch is never evaluated). Register on the
      condition's type (e.g. ``SymBool``) to lift to a symbolic
      ``if_then_else``.

    - **``and`` / ``or``**: pairwise left-fold with **two-sided probing**.
      For each fold step the evaluator first probes at operand index 0
      with the running result; if no match and the result does not
      short-circuit, it evaluates the next operand and probes again on
      the pair ``(result, next)``. This lets ``sym and x``,
      ``True and sym``, and ``sym and sym`` all dispatch, while fully
      native expressions like ``f(0) and f(1)`` still short-circuit.

      **Caveat** — ``False and sym`` short-circuits without firing
      dispatch: the right operand is never evaluated, matching Python
      semantics. If a caller needs ``False`` to lift to a symbolic
      bool as well, they must ensure the LHS is already symbolic, or
      convert statically before evaluation. This is a deliberate
      tradeoff against TVMScript's always-eager fold.

    - **Chained compare** (``a < b < c``): the evaluator unfolds the
      chain pairwise and calls ``invoke`` on each pair; registration
      semantics are identical to standalone compare ops. The outer
      ``ChainedCompare`` node is forwarded as ``node`` so handlers anchor
      at the full expression rather than a synthetic sub-range.

    Examples
    --------
    Symbolic LHS only (the common case for custom arithmetic types)::

        def _add(a, b, *, node=None):
            return PrimExpr.add(a, b)


        dispatch.register(OperationKind.Add, PrimExpr, _add)

    Both sides — fires whether the symbolic operand is left or right.
    Follows TVMScript's ``for i in [0, 1]`` pattern::

        for i in (0, 1):
            dispatch.register(OperationKind.Eq, PrimExpr, _eq, operand_index=i)

    Error reporting with source spans::

        def _div(a, b, *, node=None):
            if _is_zero(b):
                raise MyError("division by zero", node=node)
            return PrimExpr.div(a, b)


        dispatch.register(OperationKind.Div, PrimExpr, _div)

    """

    _handlers: dict[
        tuple[int, type, int],
        Callable[..., Any],
    ] = dc.field(default_factory=dict)

    def register(
        self,
        op: int,
        operand_type: type,
        handler: Callable[..., Any],
        *,
        operand_index: int = 0,
    ) -> None:
        """Register *handler* for ``(op, operand_type, operand_index)``.

        *handler* must accept a keyword-only ``node`` argument (explicitly
        or via ``**kwargs``).
        """
        self._handlers[(op, operand_type, operand_index)] = handler

    def lookup(self, op: int, operand_values: tuple[Any, ...]) -> Callable[..., Any] | None:
        """Find the first handler matching any operand's MRO, left-to-right."""
        for i, v in enumerate(operand_values):
            for cls in type(v).__mro__:
                handler = self._handlers.get((op, cls, i))
                if handler is not None:
                    return handler
        return None

    def invoke(
        self,
        op: int,
        operand_values: tuple[Any, ...],
        *,
        node: pyast.Node | None = None,
    ) -> Any:
        """Run the matching handler, or fall back to the native Python op."""
        if handler := self.lookup(op, operand_values):
            return handler(*operand_values, node=node)
        if native := _NATIVE_HANDLERS.get(op):
            return native(*operand_values)
        raise EvaluationError(f"no native handler for operation kind {op}", node=node)


@dc.dataclass
class ExprEvaluator:
    """Stateful recursive interpreter for :class:`tvm_ffi.pyast` expressions."""

    scope: Mapping[str, Any]
    dispatch: OperatorDispatch
    wrap_errors: bool = True

    @cached_property
    def _vtable(self) -> dict[type, Callable[[ExprEvaluator, Any], Any]]:
        return {
            pyast.Literal: ExprEvaluator._eval_literal,
            pyast.Id: ExprEvaluator._eval_id,
            pyast.Attr: ExprEvaluator._eval_attr,
            pyast.Index: ExprEvaluator._eval_index,
            pyast.Call: ExprEvaluator._eval_call,
            pyast.Operation: ExprEvaluator._eval_operation,
            pyast.Tuple: ExprEvaluator._eval_tuple,
            pyast.List: ExprEvaluator._eval_list,
            pyast.Dict: ExprEvaluator._eval_dict,
            pyast.Set: ExprEvaluator._eval_set,
            pyast.Slice: ExprEvaluator._eval_slice,
            pyast.Lambda: ExprEvaluator._eval_lambda,
            pyast.Comprehension: ExprEvaluator._eval_comprehension,
            pyast.FStr: ExprEvaluator._eval_fstr,
            pyast.FStrValue: ExprEvaluator._eval_fstr_value,
            pyast.StarredExpr: ExprEvaluator._eval_starred,
            pyast.WalrusExpr: ExprEvaluator._eval_walrus,
            pyast.Yield: ExprEvaluator._eval_yield,
            pyast.YieldFrom: ExprEvaluator._eval_yield_from,
            pyast.AwaitExpr: ExprEvaluator._eval_await,
        }

    def eval(self, node: pyast.Node) -> Any:
        handler = self._vtable.get(type(node))
        if handler is None:
            raise EvaluationError(f"unsupported node kind: {type(node).__name__}", node=node)
        try:
            return handler(self, node)
        except EvaluationError as e:
            if e.node is None:
                e.node = node
            raise
        except Exception as e:
            if not self.wrap_errors:
                raise
            raise EvaluationError(f"{type(e).__name__}: {e}", node=node) from e

    def _eval_literal(self, node: pyast.Literal) -> Any:
        return node.value

    def _eval_id(self, node: pyast.Id) -> Any:
        name = node.name
        scope = self.scope
        if name in scope:
            return scope[name]
        if hasattr(builtins, name):
            return getattr(builtins, name)
        raise UndefinedNameError(f"name {name!r} is not defined", node=node)

    def _eval_attr(self, node: pyast.Attr) -> Any:
        obj = self.eval(node.obj)
        try:
            return getattr(obj, node.name)
        except AttributeError as e:
            raise EvaluationError(
                f"attribute lookup failed: {type(obj).__name__}.{node.name}",
                node=node,
            ) from e

    def _eval_index(self, node: pyast.Index) -> Any:
        obj = self.eval(node.obj)
        indices = [self.eval(i) for i in node.idx]
        key = indices[0] if len(indices) == 1 else tuple(indices)
        return obj[key]

    def _eval_call(self, node: pyast.Call) -> Any:
        callee = self.eval(node.callee)
        positional: list[Any] = []
        for arg in node.args:
            if isinstance(arg, pyast.StarredExpr):
                positional.extend(self.eval(arg.value))
            else:
                positional.append(self.eval(arg))
        kwargs: dict[str, Any] = {}
        for key, val_node in zip(node.kwargs_keys, node.kwargs_values):
            if key == "":
                kwargs.update(self.eval(val_node))
            else:
                kwargs[key] = self.eval(val_node)
        return callee(*positional, **kwargs)

    def _eval_operation(self, node: pyast.Operation) -> Any:  # noqa: PLR0911
        op = node.op
        operands = node.operands
        K = pyast.OperationKind
        if op == K.And:
            return self._eval_bool_op_pairwise(node, op, operands, is_short_circuit=lambda v: not v)
        if op == K.Or:
            return self._eval_bool_op_pairwise(node, op, operands, is_short_circuit=bool)
        if op == K.IfThenElse:
            cond = self.eval(operands[0])
            if self.dispatch.lookup(op, (cond,)) is not None:
                then_v = self.eval(operands[1])
                else_v = self.eval(operands[2])
                return self.dispatch.invoke(op, (cond, then_v, else_v), node=node)
            return self.eval(operands[1] if cond else operands[2])
        if op == K.Parens:
            return self.eval(operands[0])
        if op == K.ChainedCompare:
            return self._eval_chained_compare(node)
        if K._UnaryStart < op < K._UnaryEnd:
            value = self.eval(operands[0])
            return self.dispatch.invoke(op, (value,), node=node)
        values = tuple(self.eval(o) for o in operands)
        return self.dispatch.invoke(op, values, node=node)

    def _eval_chained_compare(self, node: pyast.Operation) -> Any:
        operands = list(node.operands)
        prev = self.eval(operands[0])
        i = 1
        while i < len(operands):
            op_literal = operands[i]
            right = self.eval(operands[i + 1])
            if not isinstance(op_literal, pyast.Literal):
                raise EvaluationError(
                    "ChainedCompare: expected Literal operator marker, "
                    f"got {type(op_literal).__name__}",
                    node=op_literal,
                )
            if not self.dispatch.invoke(op_literal.value, (prev, right), node=node):
                return False
            prev = right
            i += 2
        return True

    def _eval_bool_op_pairwise(
        self,
        node: pyast.Operation,
        op: int,
        operands: Any,
        *,
        is_short_circuit: Callable[[Any], bool],
    ) -> Any:
        # Pairwise left-fold with two-sided dispatch probing. Preserves
        # Python short-circuit for native operands; fires dispatch when
        # either operand of a pair has a registered handler.
        result = self.eval(operands[0])
        for child in operands[1:]:
            # Left-side probe: symbolic LHS (e.g. sym and x).
            if self.dispatch.lookup(op, (result,)) is not None:
                val = self.eval(child)
                result = self.dispatch.invoke(op, (result, val), node=node)
                continue
            if is_short_circuit(result):
                return result
            val = self.eval(child)
            # Right-side probe: symbolic RHS (e.g. True and sym).
            if self.dispatch.lookup(op, (result, val)) is not None:
                result = self.dispatch.invoke(op, (result, val), node=node)
            else:
                result = val
        return result

    def _eval_tuple(self, node: pyast.Tuple) -> tuple:
        return tuple(self._expand_sequence(node.values))

    def _eval_list(self, node: pyast.List) -> list:
        return list(self._expand_sequence(node.values))

    def _eval_set(self, node: pyast.Set) -> set:
        return set(self._expand_sequence(node.values))

    def _expand_sequence(self, exprs: Iterable[pyast.Expr]) -> Iterator[Any]:
        for e in exprs:
            if isinstance(e, pyast.StarredExpr):
                yield from self.eval(e.value)
            else:
                yield self.eval(e)

    def _eval_dict(self, node: pyast.Dict) -> dict:
        out: dict = {}
        for k_node, v_node in zip(node.keys, node.values):
            if isinstance(k_node, pyast.StarredExpr) and isinstance(
                k_node.value, pyast.StarredExpr
            ):
                out.update(self.eval(k_node.value.value))
            else:
                out[self.eval(k_node)] = self.eval(v_node)
        return out

    def _eval_slice(self, node: pyast.Slice) -> slice:
        start = self.eval(node.start) if node.start is not None else None
        stop = self.eval(node.stop) if node.stop is not None else None
        step = self.eval(node.step) if node.step is not None else None
        return slice(start, stop, step)

    def _eval_lambda(self, node: pyast.Lambda) -> Callable[..., Any]:
        def _param_name(arg: pyast.Expr) -> str:
            if not isinstance(arg, pyast.Id):
                raise EvaluationError(
                    f"unsupported lambda parameter: {type(arg).__name__}", node=arg
                )
            name = arg.name
            for sep in (":", "="):
                cut = name.find(sep)
                if cut >= 0:
                    name = name[:cut]
            return name.strip()

        def _closure(*args: Any) -> Any:
            if len(args) != len(param_names):
                raise EvaluationError(
                    f"lambda expected {len(param_names)} argument(s), got {len(args)}",
                    node=lambda_node,
                )
            child = ChainMap(dict(zip(param_names, args)), parent_scope)  # ty: ignore[invalid-argument-type]
            return ExprEvaluator(child, dispatch).eval(body)

        param_names = [_param_name(a) for a in node.args]
        body = node.body
        parent_scope = self.scope
        dispatch = self.dispatch
        lambda_node = node
        return _closure

    def _eval_comprehension(self, node: pyast.Comprehension) -> Any:
        def _build_generator() -> Iterator[Any]:
            child_scope: dict[str, Any] = {}
            chain = ChainMap(child_scope, parent_scope)  # ty: ignore[invalid-argument-type]
            child_eval = ExprEvaluator(chain, dispatch)
            first = node.iters[0]
            for item in outer_iter:
                _bind_target(first.target, item, child_scope)
                if all(child_eval.eval(c) for c in first.ifs):
                    yield from _walk_comprehension(child_eval, node, child_scope, 1)

        K = pyast.ComprehensionKind
        kind = node.kind
        if kind == K.Generator:
            if not list(node.iters):
                raise EvaluationError("comprehension requires a 'for' clause", node=node)
            outer_iter = self.eval(node.iters[0].iter)
            parent_scope = self.scope
            dispatch = self.dispatch

            return _build_generator()
        child_scope: dict[str, Any] = {}
        chain = ChainMap(child_scope, self.scope)  # ty: ignore[invalid-argument-type]
        child_eval = ExprEvaluator(chain, self.dispatch)
        if kind == K.List:
            return list(_walk_comprehension(child_eval, node, child_scope, 0))
        if kind == K.Set:
            return set(_walk_comprehension(child_eval, node, child_scope, 0))
        if kind == K.Dict:
            return dict(_walk_comprehension(child_eval, node, child_scope, 0))
        raise EvaluationError(f"unsupported comprehension kind: {kind}", node=node)

    def _eval_fstr(self, node: pyast.FStr) -> str:
        parts: list[str] = []
        for v in node.values:
            if isinstance(v, pyast.Literal) and isinstance(v.value, str):
                parts.append(v.value)
            elif isinstance(v, pyast.FStrValue):
                parts.append(self._eval_fstr_value(v))
            else:
                parts.append(format(self.eval(v), ""))
        return "".join(parts)

    def _eval_fstr_value(self, node: pyast.FStrValue) -> str:
        value = self.eval(node.value)
        conv = node.conversion
        if conv == -1:
            pass
        elif conv == ord("s"):
            value = str(value)
        elif conv == ord("r"):
            value = repr(value)
        elif conv == ord("a"):
            value = ascii(value)
        else:
            raise EvaluationError(f"unknown f-string conversion code: {conv}", node=node)
        if node.format_spec is None:
            spec = ""
        elif isinstance(node.format_spec, pyast.FStr):
            spec = self._eval_fstr(node.format_spec)
        else:
            spec = str(self.eval(node.format_spec))
        return format(value, spec)

    def _eval_starred(self, node: pyast.StarredExpr) -> Any:
        raise EvaluationError(
            "bare starred expression cannot be evaluated outside a "
            "sequence, call, or unpacking context",
            node=node,
        )

    def _eval_walrus(self, node: pyast.WalrusExpr) -> Any:
        value = self.eval(node.value)
        target = node.target
        if not isinstance(target, pyast.Id):
            raise EvaluationError(
                f"walrus target must be an Id, got {type(target).__name__}",
                node=node,
            )
        if isinstance(self.scope, ChainMap):
            self.scope.maps[0][target.name] = value  # ty: ignore[invalid-assignment]
        else:
            try:
                self.scope[target.name] = value  # type: ignore[index]
            except TypeError as e:
                raise EvaluationError("walrus requires a mutable scope", node=node) from e
        return value

    def _eval_yield(self, node: pyast.Yield) -> Any:
        raise EvaluationError("yield expressions are not evaluable", node=node)

    def _eval_yield_from(self, node: pyast.YieldFrom) -> Any:
        raise EvaluationError("yield-from expressions are not evaluable", node=node)

    def _eval_await(self, node: pyast.AwaitExpr) -> Any:
        raise EvaluationError("await expressions are not evaluable", node=node)


def _walk_comprehension(
    evaluator: ExprEvaluator,
    node: pyast.Comprehension,
    scope: dict[str, Any],
    idx: int,
) -> Iterator[Any]:
    iters = list(node.iters)
    if idx >= len(iters):
        if node.kind == pyast.ComprehensionKind.Dict:
            # Dict comprehension guarantees node.value is present.
            assert node.value is not None
            yield (evaluator.eval(node.elt), evaluator.eval(node.value))
        else:
            yield evaluator.eval(node.elt)
        return
    it = iters[idx]
    iterable = evaluator.eval(it.iter)
    for item in iterable:
        _bind_target(it.target, item, scope)
        if all(evaluator.eval(c) for c in it.ifs):
            yield from _walk_comprehension(evaluator, node, scope, idx + 1)


def _bind_target(
    target: pyast.Expr,
    value: Any,
    scope: dict[str, Any],
    *,
    bind_value: BindValue | None = None,
) -> None:
    """Bind *value* against *target* into *scope* (a plain dict)."""
    if isinstance(target, pyast.Id):
        if bind_value is not None:
            value = bind_value(target, target.name, value)
        scope[target.name] = value
        return
    if isinstance(target, (pyast.Tuple, pyast.List)):
        _bind_sequence(
            list(target.values),
            value,
            scope,
            container=target,
            bind_value=bind_value,
        )
        return
    if isinstance(target, pyast.StarredExpr):
        raise EvaluationError("bare starred target outside a sequence unpacking", node=target)
    if isinstance(target, pyast.Attr):
        if bind_value is not None:
            bind_value(target, target.name, value)
            return
        raise EvaluationError(
            f"cannot bind into {type(target).__name__}; "
            "eval_assign only produces new local bindings "
            "(pass bind_value to handle non-Id targets)",
            node=target,
        )
    if isinstance(target, pyast.Index):
        if bind_value is not None:
            bind_value(target, "", value)
            return
        raise EvaluationError(
            f"cannot bind into {type(target).__name__}; "
            "eval_assign only produces new local bindings "
            "(pass bind_value to handle non-Id targets)",
            node=target,
        )
    raise EvaluationError(f"unsupported assignment target: {type(target).__name__}", node=target)


def _bind_sequence(
    targets: list[pyast.Expr],
    value: Any,
    scope: dict[str, Any],
    *,
    container: pyast.Expr,
    bind_value: BindValue | None = None,
) -> None:
    starred_positions = [i for i, t in enumerate(targets) if isinstance(t, pyast.StarredExpr)]
    if len(starred_positions) > 1:
        # Point at the second starred target — it's the offending one.
        raise EvaluationError(
            "at most one starred target permitted in unpacking",
            node=targets[starred_positions[1]],
        )
    items = list(value)
    if not starred_positions:
        if len(items) != len(targets):
            raise EvaluationError(
                f"cannot unpack {len(items)} values into {len(targets)} targets",
                node=container,
            )
        for t, v in zip(targets, items):
            _bind_target(t, v, scope, bind_value=bind_value)
        return
    star_idx = starred_positions[0]
    before = targets[:star_idx]
    after = targets[star_idx + 1 :]
    min_len = len(before) + len(after)
    if len(items) < min_len:
        raise EvaluationError(
            f"cannot unpack {len(items)} values; need at least {min_len}",
            node=container,
        )
    for t, v in zip(before, items[: len(before)]):
        _bind_target(t, v, scope, bind_value=bind_value)
    captured = items[len(before) : len(items) - len(after)]
    starred_target = targets[star_idx]
    inner = starred_target.value  # type: ignore[union-attr]
    if isinstance(inner, (pyast.Id, pyast.Tuple, pyast.List)):
        _bind_target(inner, list(captured), scope, bind_value=bind_value)
    else:
        raise EvaluationError(
            f"starred target must be Id/Tuple/List, got {type(inner).__name__}",
            node=starred_target,
        )
    for t, v in zip(after, items[len(items) - len(after) :]):
        _bind_target(t, v, scope, bind_value=bind_value)


def eval_expr(
    node: pyast.Node,
    scope: Mapping[str, Any] | None = None,
    *,
    dispatch: OperatorDispatch | None = None,
    wrap_errors: bool = True,
) -> Any:
    """Evaluate a :mod:`tvm_ffi.pyast` expression tree to a Python value.

    Parameters
    ----------
    node
        Any :class:`~tvm_ffi.pyast.Expr` (or other supported node). Statement
        nodes raise :class:`EvaluationError`.
    scope
        Read-only mapping used to resolve :class:`~tvm_ffi.pyast.Id` nodes.
        Defaults to an empty scope. Python builtins are consulted as a
        fallback.
    dispatch
        Optional per-call :class:`OperatorDispatch`. Defaults to
        :data:`DEFAULT_DISPATCH`.
    wrap_errors
        When ``True`` (default), raw Python exceptions raised while walking
        the tree are re-raised as :class:`EvaluationError` with the offending
        node attached and the original exception preserved as ``__cause__``.
        Set to ``False`` to let underlying exceptions bubble up unchanged.

    Returns
    -------
    value
        The evaluated Python value.

    """
    if scope is None:
        scope = {}
    if dispatch is None:
        dispatch = DEFAULT_DISPATCH
    # Wrap in a ChainMap so walrus and other internal bindings do not
    # leak into the caller's mapping (unless the caller passes one).
    if not isinstance(scope, ChainMap):
        scope = ChainMap({}, scope)  # ty: ignore[invalid-argument-type]
    return ExprEvaluator(scope, dispatch, wrap_errors=wrap_errors).eval(node)


def eval_assign(
    target: pyast.Expr,
    value: Any,
    *,
    bind_value: BindValue | None = None,
) -> dict[str, Any]:
    """Bind ``value`` against the shape of ``target`` and return the bindings.

    Handles :class:`~tvm_ffi.pyast.Id`, :class:`~tvm_ffi.pyast.Tuple`, and
    :class:`~tvm_ffi.pyast.List` targets (with at most one
    :class:`~tvm_ffi.pyast.StarredExpr` element per sequence).

    Parameters
    ----------
    target
        The assignment target AST node.
    value
        The right-hand side value to bind against *target*.
    bind_value
        Optional callback invoked for each leaf target. For :class:`~tvm_ffi.pyast.Id`
        targets, the callback's return value replaces the bound value in the
        output dict. When provided, :class:`~tvm_ffi.pyast.Attr` and
        :class:`~tvm_ffi.pyast.Index` targets are permitted: the callback is
        invoked for them but they contribute no entry to the returned dict.

    """
    out: dict[str, Any] = {}
    _bind_target(target, value, out, bind_value=bind_value)
    return out


_NATIVE_HANDLERS: dict[int, Callable[..., Any]] = {
    pyast.OperationKind.USub: operator.neg,
    pyast.OperationKind.UAdd: operator.pos,
    pyast.OperationKind.Invert: operator.invert,
    pyast.OperationKind.Not: operator.not_,
    pyast.OperationKind.Add: operator.add,
    pyast.OperationKind.Sub: operator.sub,
    pyast.OperationKind.Mult: operator.mul,
    pyast.OperationKind.Div: operator.truediv,
    pyast.OperationKind.FloorDiv: operator.floordiv,
    pyast.OperationKind.Mod: operator.mod,
    pyast.OperationKind.Pow: operator.pow,
    pyast.OperationKind.LShift: operator.lshift,
    pyast.OperationKind.RShift: operator.rshift,
    pyast.OperationKind.BitAnd: operator.and_,
    pyast.OperationKind.BitOr: operator.or_,
    pyast.OperationKind.BitXor: operator.xor,
    pyast.OperationKind.MatMult: operator.matmul,
    pyast.OperationKind.Lt: operator.lt,
    pyast.OperationKind.LtE: operator.le,
    pyast.OperationKind.Gt: operator.gt,
    pyast.OperationKind.GtE: operator.ge,
    pyast.OperationKind.Eq: operator.eq,
    pyast.OperationKind.NotEq: operator.ne,
    pyast.OperationKind.Is: operator.is_,
    pyast.OperationKind.IsNot: operator.is_not,
    pyast.OperationKind.In: lambda a, b: a in b,
    pyast.OperationKind.NotIn: lambda a, b: a not in b,
    pyast.OperationKind.And: lambda a, b: a and b,
    pyast.OperationKind.Or: lambda a, b: a or b,
}

DEFAULT_DISPATCH = OperatorDispatch()

BindValue = Callable[[pyast.Expr, str, Any], Any]
"""Callback invoked by :func:`eval_assign` on each leaf target.

Signature: ``(target_node, name, value) -> value``. For :class:`~tvm_ffi.pyast.Id`
targets *name* is the identifier; for :class:`~tvm_ffi.pyast.Attr` targets it
is the attribute name; for :class:`~tvm_ffi.pyast.Index` targets it is the
empty string. The return value is what lands in the resulting dict (for
``Id`` targets only — ``Attr``/``Index`` targets do not produce local
bindings).
"""
