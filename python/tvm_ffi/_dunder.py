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
"""Dunder method installation for ``@c_class`` and ``@py_class``."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Callable

from . import core
from .core import TypeInfo, object_repr

if TYPE_CHECKING:
    from .core import Function


def _make_init(
    type_cls: type,
    type_info: TypeInfo,
    ffi_init: Function,
    py_class_mode: bool = False,
) -> Callable[..., None]:
    """Build ``__init__`` that delegates to ``__ffi_init__``.

    Both ``@c_class`` and ``@py_class`` use the same constructor-call path
    (``self.__init_handle_by_constructor__(ffi_init, *args)``).  The only
    difference is how ``super().__init__()`` from a subclass is handled:

    * **c_class** — raises ``TypeError`` (subclass must define its own init).
    * **py_class** — silently skips when the C++ handle is already set, so
      ``super().__init__()`` is a harmless no-op.

    Parameters
    ----------
    type_cls
        The class to build an __init__ for.
    type_info
        The TypeInfo for *type_cls*.
    ffi_init
        The C++ ``__ffi_init__`` resolved at install time.
    py_class_mode
        If True, use a ``chandle`` guard instead of a ``TypeError`` guard.

    """
    sig = _make_init_signature(type_info)
    kwargs_obj = core.KWARGS
    missing = core.MISSING
    has_post_init = hasattr(type_cls, "__post_init__")

    if py_class_mode:

        def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
            if self.__chandle__() != 0:
                return
            ffi_args: list[Any] = list(args)
            if kwargs:
                ffi_args.append(kwargs_obj)
                for key, val in kwargs.items():
                    if val is not missing:
                        ffi_args.append(key)
                        ffi_args.append(val)
            self.__init_handle_by_constructor__(ffi_init, *ffi_args)
            if has_post_init:
                self.__post_init__()

    else:
        type_name = type_cls.__name__

        def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
            if type_info is not type(self).__tvm_ffi_type_info__:
                raise TypeError(
                    f"Calling `super().__init__()` is not supported for @c_class "
                    f"`{type_name}`. Use @py_class for inheritance with manual "
                    f"__init__, or define `{type(self).__name__}` with init=True."
                )
            ffi_args: list[Any] = list(args)
            if kwargs:
                ffi_args.append(kwargs_obj)
                for key, val in kwargs.items():
                    if val is not missing:
                        ffi_args.append(key)
                        ffi_args.append(val)
            self.__init_handle_by_constructor__(ffi_init, *ffi_args)
            if has_post_init:
                self.__post_init__()

    __init__.__signature__ = sig  # ty: ignore[invalid-assignment]
    __init__.__qualname__ = f"{type_cls.__qualname__}.__init__"
    __init__.__module__ = type_cls.__module__
    return __init__


def _make_init_signature(type_info: TypeInfo) -> inspect.Signature:
    """Build an ``inspect.Signature`` from reflection field metadata.

    Walks the parent chain (parent-first) to collect all ``init=True`` fields,
    reorders required-before-optional within each group, and returns a
    Signature for introspection.
    """
    positional: list[tuple[str, bool]] = []  # (name, has_default)
    kw_only: list[tuple[str, bool]] = []  # (name, has_default)

    # Walk the parent chain to collect all fields (parent-first order).
    all_fields: list[Any] = []
    ti: TypeInfo | None = type_info
    chain: list[TypeInfo] = []
    while ti is not None:
        chain.append(ti)
        ti = ti.parent_type_info
    for ancestor_info in reversed(chain):
        all_fields.extend(ancestor_info.fields)

    for field in all_fields:
        if not field.c_init:
            continue
        if field.c_kw_only:
            kw_only.append((field.name, field.c_has_default))
        else:
            positional.append((field.name, field.c_has_default))

    # Required params must come before optional ones within each group.
    pos_required = [(n, d) for n, d in positional if not d]
    pos_default = [(n, d) for n, d in positional if d]
    kw_required = [(n, d) for n, d in kw_only if not d]
    kw_default = [(n, d) for n, d in kw_only if d]

    params: list[inspect.Parameter] = []
    params.append(inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD))

    for name, _has_default in pos_required:
        params.append(inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD))

    for name, _has_default in pos_default:
        params.append(
            inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=core.MISSING)
        )

    for name, _has_default in kw_required:
        params.append(inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY))

    for name, _has_default in kw_default:
        params.append(inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=core.MISSING))

    return inspect.Signature(params)


# ---------------------------------------------------------------------------
# Copy / deepcopy / replace factories
# ---------------------------------------------------------------------------


def _make_copy(type_info: TypeInfo, shallow_copy_fn: Function | None) -> Callable[..., Any]:
    """Build ``__copy__`` with ``__ffi_shallow_copy__`` resolved at install time."""
    if shallow_copy_fn is not None:

        def __copy__(self: Any) -> Any:
            return shallow_copy_fn(self)

    else:

        def __copy__(self: Any) -> Any:
            raise TypeError(
                f"Type `{type(self).__name__}` does not support copy. "
                f"The underlying C++ type is not copy-constructible."
            )

    return __copy__


def _make_deepcopy(_type_info: TypeInfo) -> Callable[..., Any]:
    """Build ``__deepcopy__`` with ``DeepCopy`` resolved at install time."""
    from . import _ffi_api  # noqa: PLC0415

    deep_copy = _ffi_api.DeepCopy

    def __deepcopy__(self: Any, memo: Any = None) -> Any:
        return deep_copy(self)

    return __deepcopy__


def _make_replace(_type_info: TypeInfo) -> Callable[..., Any]:
    """Build ``__replace__`` with ``copy.copy`` resolved at install time."""
    import copy  # noqa: PLC0415

    copy_copy = copy.copy

    def __replace__(self: Any, **kwargs: Any) -> Any:
        obj = copy_copy(self)
        force_set = type(obj)._force_set_field
        for key, value in kwargs.items():
            force_set(obj, key, value)
        return obj

    return __replace__


# ---------------------------------------------------------------------------
# __init__ installation
# ---------------------------------------------------------------------------


def _set_match_args(cls: type, type_info: TypeInfo) -> None:
    """Set ``cls.__match_args__`` from reflected fields.

    Mirrors stdlib :func:`dataclasses.dataclass` semantics: the tuple
    contains the names of positional ``__init__`` fields (``init=True``
    and ``kw_only=False``), walking the parent chain in parent-first
    order.  If ``cls`` already defines ``__match_args__`` in its own
    ``__dict__``, it is left untouched.
    """
    if "__match_args__" in cls.__dict__:
        return
    chain: list[TypeInfo] = []
    ti: TypeInfo | None = type_info
    while ti is not None:
        chain.append(ti)
        ti = ti.parent_type_info
    names: list[str] = []
    for ancestor in reversed(chain):
        for tf in ancestor.fields or ():
            df = tf.dataclass_field
            if df is None:
                continue
            if df.init and not df.kw_only:
                names.append(tf.name)
    setattr(cls, "__match_args__", tuple(names))


def _install_dataclass_dunders(  # noqa: PLR0912, PLR0915
    cls: type,
    *,
    init: bool,
    repr: bool,
    eq: bool,
    order: bool,
    unsafe_hash: bool,
    match_args: bool = True,
    py_class_mode: bool = False,
) -> None:
    """Install structural dunder methods on *cls*.

    Each dunder delegates to the corresponding C++ recursive structural
    operation (``RecursiveEq``, ``RecursiveHash``, ``RecursiveLt``, etc.).
    If the user already defined a dunder in the class body
    (i.e. it exists in ``cls.__dict__``), it is left untouched.

    Parameters
    ----------
    cls
        The class to install dunders on.  Must have been processed by
        :func:`register_object` first (so ``__tvm_ffi_type_info__`` exists).
    init
        If True, install ``__init__`` from C++ reflection metadata.
    repr
        If True, install :func:`~tvm_ffi.core.object_repr` as ``__repr__``.
    eq
        If True, install ``__eq__`` and ``__ne__`` using ``RecursiveEq``.
        Returns ``NotImplemented`` for unrelated types so Python can
        fall back to identity comparison.
    order
        If True, install ``__lt__``, ``__le__``, ``__gt__``, ``__ge__``
        using ``RecursiveLt``/``Le``/``Gt``/``Ge``.  Returns
        ``NotImplemented`` for unrelated types.
    unsafe_hash
        If True, install ``__hash__`` using ``RecursiveHash``.
    match_args
        If True (default), set ``cls.__match_args__`` to the tuple of
        positional ``__init__`` field names for use with ``match``
        statements.  Skipped when the class already defines
        ``__match_args__`` in its body.
    py_class_mode
        If True, use a ``chandle`` guard for ``__init__`` so that
        ``super().__init__()`` is a no-op, and wrap user-defined
        ``__init__`` to allocate via ``__ffi_init__`` before user code.

    """
    from . import _ffi_api  # noqa: PLC0415

    type_info: TypeInfo = cls.__tvm_ffi_type_info__  # type: ignore[assignment]
    type_index: int = type_info.type_index
    # Look up __ffi_init__ from TypeMethod (preferred) or TypeAttrColumn (fallback).
    ffi_init: Function | None = None
    for method in type_info.methods:
        if method.name == "__ffi_init__":
            ffi_init = method.func
            break
    if ffi_init is None:
        ffi_init = core._lookup_type_attr(type_index, "__ffi_init__")
    ffi_new: Function | None = core._lookup_type_attr(type_index, "__ffi_new__")
    ffi_shallow_copy: Function | None = core._lookup_type_attr(type_index, "__ffi_shallow_copy__")

    # __init__
    # ┌─────────────────────┬──────────────────────┬──────────────────────┐
    # │                     │ user __init__        │ no user __init__     │
    # ├─────────────────────┼──────────────────────┼──────────────────────┤
    # │ c_class, init=True  │ keep as-is           │ _make_init           │
    # │ c_class, init=False │ keep as-is           │ TypeError guard      │
    # │ py_class, init=True │ wrap chandle guard   │ _make_init(py_class) │
    # │ py_class, init=False│ wrap chandle guard   │ TypeError guard      │
    # └─────────────────────┴──────────────────────┴──────────────────────┘
    if "__init__" not in cls.__dict__:
        if init and ffi_init is not None:
            cls.__init__ = _make_init(  # type: ignore[attr-defined]
                cls,
                type_info,
                ffi_init=ffi_init,
                py_class_mode=py_class_mode,
            )
        elif not init:
            # init=False, no user __init__: TypeError guard
            msg = (
                f"`{cls.__name__}` cannot be constructed directly. "
                f"Define a custom __init__ or use a factory method."
            )

            def __init___(self: Any, *args: Any, **kwargs: Any) -> None:
                raise TypeError(msg)

            __init___.__qualname__ = f"{cls.__qualname__}.__init__"
            __init___.__module__ = cls.__module__
            cls.__init__ = __init___  # type: ignore[attr-defined]
    elif py_class_mode and ffi_new is not None:
        # User-defined __init__: wrap with chandle guard so the C++ object
        # is allocated (via __ffi_new__) before the user's code runs.
        # We use __ffi_new__ (zero-arg allocator) rather than __ffi_init__
        # because the user's __init__ signature may not match the field
        # layout.  super().__init__() from within the user's code becomes
        # a no-op because chandle is already set.
        import functools  # noqa: PLC0415

        user_init = cls.__dict__["__init__"]

        @functools.wraps(user_init)
        def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
            if self.__chandle__() == 0:
                self.__init_handle_by_constructor__(ffi_new)
            user_init(self, *args, **kwargs)

        cls.__init__ = __init__  # type: ignore[attr-defined]

    # __repr__
    if repr and "__repr__" not in cls.__dict__:
        cls.__repr__ = object_repr  # type: ignore[attr-defined]

    def _is_comparable(self: Any, other: Any) -> bool:
        """Return True if *self* and *other* share a type hierarchy."""
        return isinstance(other, type(self)) or isinstance(self, type(other))

    # Semantic families (__eq__/__ne__, __lt__/__le__/__gt__/__ge__) are
    # treated as a unit: if the user defines any member, the whole family
    # is skipped so generated and user-defined methods don't disagree.
    if eq and not ({"__eq__", "__ne__"} & set(cls.__dict__)):
        recursive_eq = _ffi_api.RecursiveEq

        def __eq__(self: Any, other: Any) -> bool:
            if not _is_comparable(self, other):
                return NotImplemented
            return recursive_eq(self, other)

        def __ne__(self: Any, other: Any) -> bool:
            if not _is_comparable(self, other):
                return NotImplemented
            return not recursive_eq(self, other)

        cls.__eq__ = __eq__  # type: ignore[attr-defined]
        cls.__ne__ = __ne__  # type: ignore[attr-defined]

    if order and not ({"__lt__", "__le__", "__gt__", "__ge__"} & set(cls.__dict__)):
        recursive_lt = _ffi_api.RecursiveLt
        recursive_le = _ffi_api.RecursiveLe
        recursive_gt = _ffi_api.RecursiveGt
        recursive_ge = _ffi_api.RecursiveGe

        def __lt__(self: Any, other: Any) -> bool:
            if not _is_comparable(self, other):
                return NotImplemented
            return recursive_lt(self, other)

        def __le__(self: Any, other: Any) -> bool:
            if not _is_comparable(self, other):
                return NotImplemented
            return recursive_le(self, other)

        def __gt__(self: Any, other: Any) -> bool:
            if not _is_comparable(self, other):
                return NotImplemented
            return recursive_gt(self, other)

        def __ge__(self: Any, other: Any) -> bool:
            if not _is_comparable(self, other):
                return NotImplemented
            return recursive_ge(self, other)

        cls.__lt__ = __lt__  # type: ignore[attr-defined]
        cls.__le__ = __le__  # type: ignore[attr-defined]
        cls.__gt__ = __gt__  # type: ignore[attr-defined]
        cls.__ge__ = __ge__  # type: ignore[attr-defined]

    if unsafe_hash and "__hash__" not in cls.__dict__:
        recursive_hash = _ffi_api.RecursiveHash

        def __hash__(self: Any) -> int:
            return recursive_hash(self)

        cls.__hash__ = __hash__  # type: ignore[attr-defined]

    # __copy__ / __deepcopy__ / __replace__
    if "__copy__" not in cls.__dict__:
        cls.__copy__ = _make_copy(type_info, ffi_shallow_copy)  # type: ignore[attr-defined]
    if "__deepcopy__" not in cls.__dict__:
        cls.__deepcopy__ = _make_deepcopy(type_info)  # type: ignore[attr-defined]
    if "__replace__" not in cls.__dict__:
        cls.__replace__ = _make_replace(type_info)  # type: ignore[attr-defined]

    if match_args:
        _set_match_args(cls, type_info)
