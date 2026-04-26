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
"""Decorators for opt-in parser-dispatch registration.

Two decorators are exported:

* :func:`parse_hook` — marks a function as the parser entry-point for one
  or more registry slots on a language module. Used by tier-1 manual
  parsers and as the explicit override mechanism when trait-driven
  classification would conflict.
* :func:`parse_slot` — marks a method on a tier-2 class as the inverse
  for a single trait field that cannot be auto-inverted (i.e. a field
  whose value comes from a ``$method:`` or ``$global:`` ref).

Both decorators only attach metadata to the wrapped callable; the
actual registration walk happens later in :mod:`tvm_ffi.dialect_autogen`
when ``finalize_module`` runs.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

# ---------------------------------------------------------------------------
# Reserved C0-keyword targets for ``parse_hook``
# ---------------------------------------------------------------------------
#
# These are the parser-protocol slots reserved on every dialect module for
# Python builtin syntax that does not carry a unique name (return, if,
# while, var-def, etc.). A tier-1 user supplies a manual parser by
# decorating it with ``@parse_hook("__ffi_parse_*__")``; the registry
# walker honours that override in place of the auto-classification.

#: Function-typed reserved C0 slots (one parser handler per slot).
RESERVED_FN_SLOTS: frozenset[str] = frozenset(
    {
        "__ffi_parse_make_var__",
        "__ffi_parse_return__",
        "__ffi_parse_if__",
        "__ffi_parse_while__",
        "__ffi_parse_assert__",
        "__ffi_parse_func__",
        "__ffi_parse_for__",
        "__ffi_parse_with__",
        "__ffi_parse_call__",
        "__ffi_parse_assign__",
        "__ffi_parse_load__",
        "__ffi_parse_store__",
    },
)

#: Dict-typed reserved C0 slots — registered on the module as a ``dict``;
#: each handler binding lands at a sub-key (op-kind enum value or format
#: string).
RESERVED_DICT_SLOTS: frozenset[str] = frozenset(
    {
        "__ffi_parse_op__",  # keyed by ``OperationKind`` int
        "__ffi_parse_make_const__",  # keyed by literal-format str
    },
)

#: All reserved C0 keywords; ``parse_hook`` accepts any of these as a
#: positional target (``@parse_hook("__ffi_parse_func__")``).
RESERVED_C0_KEYWORDS: frozenset[str] = RESERVED_FN_SLOTS | RESERVED_DICT_SLOTS


# ---------------------------------------------------------------------------
# Hook metadata attached to decorated functions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParseHookSpec:
    """All registration targets a single ``@parse_hook`` declares.

    Attached to the decorated function as ``__ffi_parse_hook__``; consumed
    by the registry walker. A single function may target multiple names
    (e.g. ``@parse_hook(callee=["serial", "parallel"])``) and/or land in
    multiple dict slots simultaneously.

    Attributes
    ----------
    named_callees
        Names to register on the lang module as direct attributes
        (``T.<name>``). Comes from positional args and the ``callee``
        kwarg.
    fn_slots
        Reserved single-handler slots claimed by this hook
        (``__ffi_parse_func__``, ``__ffi_parse_return__``, …).
    op_kinds
        Sub-keys for ``__ffi_parse_op__``. Each value is an
        ``OperationKind`` int.
    make_const_formats
        Sub-keys for ``__ffi_parse_make_const__``. Each value is the
        literal-format string (``"int"``, ``"float"``, ``"default"``).
    """

    named_callees: tuple[str, ...] = ()
    fn_slots: tuple[str, ...] = ()
    op_kinds: tuple[int, ...] = ()
    make_const_formats: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# parse_hook
# ---------------------------------------------------------------------------


_F = TypeVar("_F", bound=Callable[..., Any])


def _flatten_targets(targets: tuple[Any, ...]) -> list[str]:
    """Accept positional args as either ``str`` or ``Iterable[str]``."""
    out: list[str] = []
    for t in targets:
        if isinstance(t, str):
            out.append(t)
        elif isinstance(t, Iterable):
            for item in t:
                if not isinstance(item, str):
                    raise TypeError(
                        f"@parse_hook: positional target must be str or "
                        f"iterable of str, got element of type "
                        f"{type(item).__name__}",
                    )
                out.append(item)
        else:
            raise TypeError(
                f"@parse_hook: positional target must be str or iterable "
                f"of str, got {type(t).__name__}",
            )
    return out


def parse_hook(  # noqa: PLR0913
    *targets: Any,
    callee: str | Iterable[str] | None = None,
    op_kind: int | Iterable[int] | None = None,
    make_const_format: str | Iterable[str] | None = None,
) -> Callable[[_F], _F]:
    """Mark a function as the parser handler for one or more registry slots.

    Tier-1 manual parsers and explicit conflict-overrides funnel through
    this decorator. The wrapped function is unchanged at runtime; the
    registry walker reads :class:`ParseHookSpec` from
    ``fn.__ffi_parse_hook__`` when ``finalize_module`` runs.

    Parameters
    ----------
    *targets
        Names to register. Each positional arg may be a single ``str``
        (e.g. ``"prim_func"``) or an iterable of strings
        (``["serial", "parallel", "unroll"]``). Reserved C0 keywords
        (``"__ffi_parse_*__"``) are auto-routed to the matching slot.
    callee
        Equivalent to a positional target; kept as a kwarg for the
        readable ``@parse_hook(callee="Add", op_kind=OperationKind.Add)``
        BinOp/Call shape.
    op_kind
        ``OperationKind`` int (or iterable of them) — registers the hook
        under ``__ffi_parse_op__[op_kind]``. Used by tier-1
        ``BinOpTraits`` / ``UnaryOpTraits`` overrides.
    make_const_format
        Literal-format string (``"int"``, ``"float"``, ``"default"``) or
        iterable of them — registers the hook under
        ``__ffi_parse_make_const__[format]``. Used by tier-1
        ``LiteralTraits`` overrides.

    Returns
    -------
    Callable
        The decorator that returns ``fn`` unchanged with metadata
        attached.

    Raises
    ------
    TypeError
        On invalid argument types.
    ValueError
        When no targets are supplied (``@parse_hook()`` with everything
        defaulted is a no-op and almost certainly a user error).

    Examples
    --------
    .. code-block:: python

        # Register on the lang module under multiple names.
        @parse_hook("serial", "parallel")
        def _parse_for_loop(...): ...

        # Tier-1 BinOp override.
        @parse_hook(callee="Add", op_kind=OperationKind.Add)
        def _parse_add(...): ...

        # Override a reserved C0 slot.
        @parse_hook("__ffi_parse_make_var__")
        def _parse_var(...): ...
    """
    # Collect named callees from positional + ``callee`` kwarg.
    named: list[str] = _flatten_targets(targets)
    if callee is not None:
        if isinstance(callee, str):
            named.append(callee)
        else:
            named.extend(_flatten_targets((callee,)))

    # Split named into reserved-slot vs free-name targets.
    fn_slots: list[str] = []
    named_free: list[str] = []
    for n in named:
        if n in RESERVED_FN_SLOTS:
            fn_slots.append(n)
        elif n in RESERVED_DICT_SLOTS:
            raise ValueError(
                f"@parse_hook: {n!r} is a dict-typed reserved slot — "
                "supply ``op_kind=`` or ``make_const_format=`` instead "
                "of using it as a positional target.",
            )
        else:
            named_free.append(n)

    # Normalize op_kind / make_const_format to tuples of ints/strs.
    op_kinds: list[int]
    if op_kind is None:
        op_kinds = []
    elif isinstance(op_kind, int):
        op_kinds = [op_kind]
    else:
        op_kinds = []
        for k in op_kind:
            if not isinstance(k, int):
                raise TypeError(
                    f"@parse_hook: op_kind values must be int (got "
                    f"{type(k).__name__})",
                )
            op_kinds.append(k)

    formats: list[str]
    if make_const_format is None:
        formats = []
    elif isinstance(make_const_format, str):
        formats = [make_const_format]
    else:
        formats = []
        for s in make_const_format:
            if not isinstance(s, str):
                raise TypeError(
                    f"@parse_hook: make_const_format values must be str "
                    f"(got {type(s).__name__})",
                )
            formats.append(s)

    if not (named_free or fn_slots or op_kinds or formats):
        raise ValueError(
            "@parse_hook: at least one target is required (named, "
            "reserved slot, op_kind, or make_const_format).",
        )

    spec = ParseHookSpec(
        named_callees=tuple(named_free),
        fn_slots=tuple(fn_slots),
        op_kinds=tuple(op_kinds),
        make_const_formats=tuple(formats),
    )

    def decorator(fn: _F) -> _F:
        if not callable(fn):
            raise TypeError(
                f"@parse_hook: expected a callable, got {type(fn).__name__}",
            )
        fn.__ffi_parse_hook__ = spec  # type: ignore[attr-defined]
        return fn

    return decorator


# ---------------------------------------------------------------------------
# parse_slot
# ---------------------------------------------------------------------------


def parse_slot(field_name: str) -> Callable[[_F], _F]:
    """Mark a method as the inverse for a single tier-2 trait field.

    Tier-2 trait fields whose values come from a ``$method:`` or
    ``$global:`` ref are not auto-invertible — the registry needs an
    explicit reconstruction routine to materialize the field from the
    parsed AST. ``@parse_slot("field_name")`` declares one such routine
    on the IR class.

    Parameters
    ----------
    field_name
        The trait field this method reconstructs (e.g. ``"extent"``,
        ``"end"``).

    Returns
    -------
    Callable
        Decorator that returns the method unchanged with the slot name
        attached as ``__ffi_parse_slot__``.

    Examples
    --------
    .. code-block:: python

        @py_class("mini.tir.For", structural_eq="tree")
        class For(Object):
            start: Any
            extent: Any

            @parse_slot("extent")
            def _construct_extent(start, end_value):
                return end_value - start
    """
    if not isinstance(field_name, str) or not field_name:
        raise TypeError(
            "@parse_slot: field_name must be a non-empty str, got "
            f"{type(field_name).__name__}",
        )

    def decorator(fn: _F) -> _F:
        if not callable(fn):
            raise TypeError(
                f"@parse_slot: expected a callable, got {type(fn).__name__}",
            )
        fn.__ffi_parse_slot__ = field_name  # type: ignore[attr-defined]
        return fn

    return decorator


# ---------------------------------------------------------------------------
# Internal helpers (consumed by the registry walker)
# ---------------------------------------------------------------------------


def get_hook_spec(fn: Any) -> ParseHookSpec | None:
    """Return the :class:`ParseHookSpec` attached to ``fn`` or ``None``."""
    spec = getattr(fn, "__ffi_parse_hook__", None)
    return spec if isinstance(spec, ParseHookSpec) else None


def get_slot_field(fn: Any) -> str | None:
    """Return the trait-field name for a ``@parse_slot`` method or ``None``."""
    name = getattr(fn, "__ffi_parse_slot__", None)
    return name if isinstance(name, str) else None
