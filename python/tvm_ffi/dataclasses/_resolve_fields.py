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
"""Forward-annotation resolution for ``@py_class`` registration.

``@py_class`` registers a Python type before it resolves field annotations, so
self-referential and mutually recursive annotations can refer to classes that
are already present in the FFI type registry.  This module owns that early
registration step, the per-module local namespace used by
``typing.get_type_hints``, and the queue of classes whose annotations must be
retried after later definitions become available.

When a class is ready to finalize, this module lazily imports
``py_class.on_fields_resolved``.  Keeping the finalizer import local preserves
the boundary between annotation resolution and field registration while avoiding
a top-level import cycle.
"""

from __future__ import annotations

import sys
import typing
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .. import core
from .field import KW_ONLY

if TYPE_CHECKING:
    from ..core import TypeInfo
else:
    TypeInfo = Any

ResolvedFields = tuple[list[type], dict[type, dict[str, Any]]]


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
#
# ``@py_class`` registration happens in two phases:
#
#   Phase 1 (register_type_without_fields)
#       Allocates a C-level type index and inserts the class into the
#       global type registry.  This must happen early so that self-
#       referential and mutually-referential annotations can resolve
#       the class via ``TypeSchema.from_annotation()``.  Phase 1 always
#       succeeds or raises immediately for invalid parents.  Decorator
#       options are stored by py_class.py on ``TypeInfo._decorator_args``.
#
#   Phase 2 (py_class.on_fields_resolved)
#       Resolves string annotations via ``typing.get_type_hints``,
#       returns per-owner resolved hints, and asks py_class.py to
#       materialize ``Field`` objects, register fields/methods/type
#       attributes with the Cython layer, and install dataclass dunders.
#
#       If ``get_type_hints`` raises ``NameError`` (forward reference
#       not yet defined), the class is added to ``_PENDING_CLASSES``
#       and retried after each successful phase-2.  If phase-2 fails
#       for any other reason, ``rollback_registration`` undoes phase-1
#       so the type key can be reused.
# ---------------------------------------------------------------------------


@dataclass
class _PendingClass:
    """Resolution state for a class waiting on unresolved annotations.

    Only data needed to retry annotation resolution is stored here.  Decorator
    options live on ``type_info._decorator_args`` so deferred and immediate
    finalization use the same source of truth.
    """

    cls: type
    type_info: TypeInfo
    globalns: dict[str, Any]


#: Classes whose phase-2 (field registration) was deferred because
#: ``typing.get_type_hints`` raised ``NameError`` on an unresolved
#: forward reference.  Retried after each successful phase-2 via
#: :func:`flush_pending`.
_PENDING_CLASSES: list[_PendingClass] = []

#: Per-module mapping of ``class.__name__ → class`` for every
#: ``@py_class``-decorated type.  Used as *localns* when resolving
#: annotations so that mutual references between classes in the same
#: module work even before the second class is assigned to the module
#: variable by Python.
_PY_CLASS_BY_MODULE: dict[str, dict[str, type]] = {}


class _KWOnlyAnnotation:
    """Type-valued stand-in accepted by ``typing.get_type_hints``."""


def _is_kw_only_annotation(annotation: Any) -> bool:
    if annotation is KW_ONLY or annotation is _KWOnlyAnnotation:
        return True
    if isinstance(annotation, str):
        return annotation == "KW_ONLY" or annotation.endswith(".KW_ONLY")
    forward_arg = getattr(annotation, "__forward_arg__", None)
    if isinstance(forward_arg, str):
        return forward_arg == "KW_ONLY" or forward_arg.endswith(".KW_ONLY")
    return False


# ---------------------------------------------------------------------------
# Phase 1: type registration
# ---------------------------------------------------------------------------


def _registered_type_info(cls: type) -> TypeInfo | None:
    """Return the TypeInfo registered directly for *cls*, not inherited metadata."""
    info = core._type_cls_to_type_info(cls)
    if info is not None:
        return info
    return cls.__dict__.get("__tvm_ffi_type_info__", None)


def register_type_without_fields(cls: type, type_key: str | None) -> TypeInfo:
    """Register ``cls`` in the FFI type registry before resolving fields.

    The returned :class:`~tvm_ffi.core.TypeInfo` has a type index, type key, and
    Python class binding, but no field metadata yet.  This early registration is
    what lets ``TypeSchema.from_annotation`` resolve references to ``cls`` while
    the decorator is still running.

    The class is also inserted into the module-local annotation namespace used
    by :func:`resolve_type_hints_by_owner`, allowing sibling ``@py_class``
    declarations in the same module to reference each other before the second
    class has been assigned to the module global by Python.
    """
    parent_info = next(
        (info for base in cls.__mro__[1:] if (info := _registered_type_info(base)) is not None),
        None,
    )
    if parent_info is None:
        raise TypeError(
            f"{cls.__name__} must inherit from a registered FFI Object type (e.g. tvm_ffi.Object)"
        )
    if type_key is None:
        type_key = f"{cls.__module__}.{cls.__qualname__}"
    info = core._register_py_class(parent_info, type_key, cls)
    setattr(cls, "__tvm_ffi_type_info__", info)
    # Register in resolution namespace so sibling classes can find us.
    _PY_CLASS_BY_MODULE.setdefault(cls.__module__, {})[cls.__name__] = cls
    return info


def rollback_registration(cls: type, type_info: TypeInfo) -> None:
    """Undo Python-side state from :func:`register_type_without_fields`.

    The C-level type index is permanently consumed (cannot be reclaimed),
    but the Python-level registry dicts are cleaned up so a retry with
    the same type key does not hit "already registered".  The dataclass marker
    is also removed because failed phase-2 finalization means the class is not a
    usable FFI dataclass.
    """
    # Remove from the Cython-level registry dicts (TYPE_KEY_TO_INFO,
    # TYPE_CLS_TO_INFO, TYPE_INDEX_TO_INFO, TYPE_INDEX_TO_CLS).
    core._rollback_py_class(type_info)  # ty: ignore[unresolved-attribute]
    # Remove from our own module-level resolution namespace.
    _PY_CLASS_BY_MODULE.get(cls.__module__, {}).pop(cls.__name__, None)
    for attr in ("__tvm_ffi_type_info__", "__ffi_is_dataclass__"):
        try:
            delattr(cls, attr)
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# Annotation resolution
# ---------------------------------------------------------------------------


def own_annotations(cls: type) -> dict[str, Any]:
    """Return annotations declared directly on ``cls`` without MRO merging."""
    # Python 3.14+ (PEP 749): annotations are lazily evaluated via
    # __annotate__ and no longer stored directly in __dict__.  getattr()
    # triggers evaluation and returns per-class annotations correctly.
    # On Python < 3.14, getattr() follows MRO and returns *parent*
    # annotations when the child has none — use __dict__ to avoid that.
    if sys.version_info >= (3, 14):
        return getattr(cls, "__annotations__", {})
    return cls.__dict__.get("__annotations__", {})


def _field_owner_classes(cls: type) -> list[type]:
    """Return local MRO entries whose annotations become fields on ``cls``.

    A Python subclass of a registered FFI parent may include unregistered mixin
    bases before the registered parent.  Those mixin annotations are owned by
    the new Python type and should be registered together with ``cls``.  Fields
    already represented by the nearest registered parent are excluded.
    """
    registered_parent = next(
        (b for b in cls.__mro__[1:] if _registered_type_info(b) is not None), object
    )
    represented = set(registered_parent.__mro__)
    return [
        b
        for b in reversed(cls.__mro__)
        if b is not object and b not in represented and own_annotations(b)
    ]


def _build_localns(cls: type, *, cross_module: bool = False) -> dict[str, Any]:
    """Build the localns dict for resolving ``cls``'s annotations.

    By default, includes only classes from ``cls.__module__``, preserving
    standard Python name resolution semantics.  When ``cross_module=True``,
    also includes classes from all other registered modules as a fallback
    — this is needed when ``cls`` has a forward reference to a class in
    another module that can't appear in ``cls.__module__``'s globals due
    to a circular import (e.g. the target is imported only under
    ``if TYPE_CHECKING:``).

    Cross-module entries are added with ``setdefault`` so same-module
    classes and the class itself always take precedence over foreign
    classes with the same ``__name__``.
    """
    localns = dict(_PY_CLASS_BY_MODULE.get(cls.__module__, {}))
    localns[cls.__name__] = cls
    if cross_module:
        for mod_name, mod_classes in list(_PY_CLASS_BY_MODULE.items()):
            if mod_name == cls.__module__:
                continue
            for name, klass in mod_classes.items():
                localns.setdefault(name, klass)
    return localns


def _resolve_own_type_hints(
    owner: type,
    globalns: dict[str, Any],
    localns: dict[str, Any],
) -> dict[str, Any]:
    """Resolve only annotations declared directly on ``owner``.

    ``typing.get_type_hints(cls)`` merges annotations across the full MRO.
    That is wrong for py_class phase 2 because inherited C++/c_class fields
    are already registered by the parent type.  Resolving only the owner
    annotations also avoids evaluating parent annotations in the child
    module's namespace.

    Python 3.10's ``typing.get_type_hints`` rejects ``dataclasses.KW_ONLY``
    because it is a singleton, not a type.  Replace it in the temporary shim
    annotations, then restore it in the resolved hints.
    """
    annotations = own_annotations(owner)
    if not annotations:
        return {}
    shim_annotations = dict(annotations)
    kw_only_names: list[str] = []
    for name, annotation in annotations.items():
        if _is_kw_only_annotation(annotation):
            shim_annotations[name] = _KWOnlyAnnotation
            kw_only_names.append(name)
    shim = type(
        f"_{owner.__name__}OwnAnnotations",
        (),
        {"__annotations__": shim_annotations, "__module__": owner.__module__},
    )
    kwargs: dict[str, Any] = {"globalns": globalns, "localns": localns}
    if sys.version_info >= (3, 11):
        kwargs["include_extras"] = True
    hints = typing.get_type_hints(shim, **kwargs)
    for name in kw_only_names:
        hints[name] = KW_ONLY
    return hints


def resolve_type_hints_by_owner(
    cls: type,
    globalns: dict[str, Any],
) -> ResolvedFields | None:
    """Resolve field annotations grouped by owner class.

    Returns ``(owners, hints_by_owner)`` when every annotation can be resolved.
    Returns :data:`None` when a forward reference is still unavailable, signaling
    that the caller should defer field registration and retry later.
    """
    # Resolve string annotations to types; return None (defer) on NameError.
    #
    # First try with module-scoped localns (standard Python name resolution).
    # On NameError, retry with a cross-module localns that includes classes
    # from every registered module — this handles circular imports where the
    # target of a forward reference is imported only under TYPE_CHECKING and
    # therefore never enters the declaring module's globals.
    owners = _field_owner_classes(cls)
    localns = _build_localns(cls)
    localns.update({owner.__name__: owner for owner in owners})
    try:
        hints_by_owner = {
            owner: _resolve_own_type_hints(
                owner,
                getattr(sys.modules.get(owner.__module__, None), "__dict__", globalns),
                localns,
            )
            for owner in owners
        }
    except (NameError, AttributeError):
        localns = _build_localns(cls, cross_module=True)
        localns.update({owner.__name__: owner for owner in owners})
        try:
            hints_by_owner = {
                owner: _resolve_own_type_hints(
                    owner,
                    getattr(sys.modules.get(owner.__module__, None), "__dict__", globalns),
                    localns,
                )
                for owner in owners
            }
        except (NameError, AttributeError):
            return None
    return owners, hints_by_owner


# ---------------------------------------------------------------------------
# Deferred resolution
# ---------------------------------------------------------------------------


def flush_pending() -> None:
    """Retry deferred classes until no additional annotations resolve.

    A successful finalization can make another pending class resolvable, so this
    function runs to a fixed point.  The field-registration finalizer is imported
    lazily to keep this module focused on resolution and to avoid an import
    cycle with :mod:`tvm_ffi.dataclasses.py_class`.
    """
    from .py_class import on_fields_resolved  # noqa: PLC0415

    changed = True
    while changed:
        changed = False
        remaining: list[_PendingClass] = []
        for entry in _PENDING_CLASSES:
            resolved = resolve_type_hints_by_owner(entry.cls, entry.globalns)
            if resolved is None:
                remaining.append(entry)
            else:
                on_fields_resolved(entry.type_info, resolved)
                changed = True
        _PENDING_CLASSES[:] = remaining


def _raise_unresolved_forward_reference(cls: type, globalns: dict[str, Any]) -> None:
    """Raise :class:`TypeError` listing the annotations that cannot be resolved."""
    localns = _build_localns(cls, cross_module=True)
    owners = _field_owner_classes(cls)
    localns.update({owner.__name__: owner for owner in owners})
    unresolved: list[str] = []
    for owner in owners:
        for name, ann_str in own_annotations(owner).items():
            if isinstance(ann_str, str):
                try:
                    eval(ann_str, globalns, localns)  # pylint: disable=eval-used
                except Exception as err:
                    unresolved.append(f"{name}: {ann_str} ({err})")
    raise TypeError(
        f"Cannot instantiate {cls.__name__}: unresolved forward references: {unresolved}"
    )


def _remove_from_pending(cls: type) -> None:
    _PENDING_CLASSES[:] = [p for p in _PENDING_CLASSES if p.cls is not cls]


def _make_temporary_init(
    cls: type,
    type_info: TypeInfo,
    globalns: dict[str, Any],
) -> Callable[..., None]:
    """Build the ``__init__`` shim used while field registration is deferred.

    The shim resolves annotations on first construction, invokes the normal
    finalization path, removes the class from the pending queue, and then
    dispatches to the real ``__init__`` that finalization installed or restored.
    If finalization fails, it rolls back phase-1 registration so the type key can
    be reused after the user fixes the annotation error.
    """

    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        if type_info.fields is None:
            try:
                from .py_class import on_fields_resolved  # noqa: PLC0415

                resolved = resolve_type_hints_by_owner(cls, globalns)
                if resolved is None:
                    _raise_unresolved_forward_reference(cls, globalns)
                on_fields_resolved(type_info, resolved)
                # cls stays in _PENDING_CLASSES after phase-2 succeeds; drop it
                # before flush_pending so the loop doesn't hit the Cython-level
                # "_register_fields already called" assertion on a second pass.
                _remove_from_pending(cls)
                flush_pending()
            except Exception:
                # Remove from pending list and roll back so the type key can be reused.
                _remove_from_pending(cls)
                rollback_registration(cls, type_info)
                raise
        # cls.__init__ has been replaced by the real init (or restored user init).
        cls.__init__(self, *args, **kwargs)

    __init__.__qualname__ = f"{cls.__qualname__}.__init__"
    __init__.__module__ = cls.__module__
    return __init__


def _install_deferred_init(
    cls: type,
    type_info: TypeInfo,
    globalns: dict[str, Any],
) -> None:
    """Install a temporary ``__init__`` that completes registration on first call.

    Preserves a user-defined ``__init__`` if present in *cls.__dict__*;
    it is restored by the phase-2 callback after registration completes
    so that ``_install_dataclass_dunders`` sees it and skips auto-generation.
    """
    # Save user-defined __init__ before overwriting.
    user_init = cls.__dict__.get("__init__")
    if user_init is not None:
        cls._py_class_user_init = user_init  # type: ignore[attr-defined]

    cls.__init__ = _make_temporary_init(  # type: ignore[assignment]
        cls,
        type_info,
        globalns,
    )
    cls.__ffi_py_class_is_deferred_init__ = True  # type: ignore[attr-defined]


def defer_field_registration(
    cls: type,
    type_info: TypeInfo,
    globalns: dict[str, Any],
) -> None:
    """Record unresolved field registration and install first-use finalization.

    Deferred entries keep only the class, ``TypeInfo``, and globals needed to
    retry annotation resolution.  Decorator parameters are read later from
    ``type_info._decorator_args`` by ``py_class.on_fields_resolved``.
    """
    _PENDING_CLASSES.append(_PendingClass(cls, type_info, globalns))
    _install_deferred_init(cls, type_info, globalns)
