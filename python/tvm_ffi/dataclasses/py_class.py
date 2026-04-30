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
"""The ``py_class`` decorator: Python-defined FFI classes with dataclass semantics."""

from __future__ import annotations

import sys
import typing
from collections.abc import Callable
from copy import copy
from dataclasses import dataclass
from typing import Any, ClassVar, TypeVar

from typing_extensions import dataclass_transform

from .. import core
from .._dunder import _install_dataclass_dunders
from ..core import MISSING, TypeSchema
from ..registry import _add_class_attrs
from .field import KW_ONLY, Field, field

_T = TypeVar("_T", bound=type)


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
#
# Registration happens in two phases:
#
#   Phase 1 (_register_type_without_fields)
#       Allocates a C-level type index and inserts the class into the
#       global type registry.  This must happen early so that self-
#       referential and mutually-referential annotations can resolve
#       the class via ``TypeSchema.from_annotation()``.  Phase 1 always
#       succeeds (or raises immediately for non-Object parents).
#
#   Phase 2 (_register_fields_into_type)
#       Resolves string annotations via ``typing.get_type_hints``,
#       converts them to ``TypeSchema`` / ``Field`` objects, validates
#       field ordering, registers fields with the Cython layer, and
#       installs ``__init__``, ``__repr__``, ``__eq__``, etc.
#
#       If ``get_type_hints`` raises ``NameError`` (forward reference
#       not yet defined), the class is added to ``_PENDING_CLASSES``
#       and retried after each successful phase-2.  If phase-2 fails
#       for any other reason, ``_rollback_registration`` undoes phase-1
#       so the type key can be reused.
# ---------------------------------------------------------------------------


@dataclass
class _PendingClass:
    """Bookkeeping for a class whose annotations couldn't be resolved yet."""

    cls: type
    type_info: Any  # core.TypeInfo
    globalns: dict[str, Any]
    params: dict[str, Any]


#: Classes whose phase-2 (field registration) was deferred because
#: ``typing.get_type_hints`` raised ``NameError`` on an unresolved
#: forward reference.  Retried after each successful phase-2 via
#: :func:`_flush_pending`.
_PENDING_CLASSES: list[_PendingClass] = []

#: Per-module mapping of ``class.__name__ → class`` for every
#: ``@py_class``-decorated type.  Used as *localns* when resolving
#: annotations so that mutual references between classes in the same
#: module work even before the second class is assigned to the module
#: variable by Python.
_PY_CLASS_BY_MODULE: dict[str, dict[str, type]] = {}

# ---------------------------------------------------------------------------
# Phase 1: type registration
# ---------------------------------------------------------------------------


def _register_type_without_fields(cls: type, type_key: str | None) -> Any:
    """Phase 1: allocate type index and register the type (always succeeds)."""
    parent_info: core.TypeInfo | None = None
    for base in cls.__bases__:
        parent_info = core._type_cls_to_type_info(base)
        if parent_info is None:
            parent_info = getattr(base, "__tvm_ffi_type_info__", None)
        if parent_info is not None:
            break
    if parent_info is None:
        raise TypeError(
            f"{cls.__name__} must inherit from a registered FFI Object type (e.g. tvm_ffi.Object)"
        )
    if type_key is None:
        type_key = f"{cls.__module__}.{cls.__qualname__}"
    info = core._register_py_class(parent_info, type_key, cls)
    setattr(cls, "__tvm_ffi_type_info__", info)
    # Register in resolution namespace so sibling classes can find us
    _PY_CLASS_BY_MODULE.setdefault(cls.__module__, {})[cls.__name__] = cls
    return info


def _rollback_registration(cls: type, type_info: Any) -> None:
    """Undo :func:`_register_type_without_fields` after a phase-2 failure.

    The C-level type index is permanently consumed (cannot be reclaimed),
    but the Python-level registry dicts are cleaned up so a retry with
    the same type key does not hit "already registered".
    """
    # Remove from the Cython-level registry dicts (TYPE_KEY_TO_INFO,
    # TYPE_CLS_TO_INFO, TYPE_INDEX_TO_INFO, TYPE_INDEX_TO_CLS).
    core._rollback_py_class(type_info)  # ty: ignore[unresolved-attribute]
    # Remove from our own module-level resolution namespace.
    _PY_CLASS_BY_MODULE.get(cls.__module__, {}).pop(cls.__name__, None)
    for attr in ("__tvm_ffi_type_info__", "__tvm_ffi_is_dataclass__"):
        try:
            delattr(cls, attr)
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# Phase 2: annotation resolution, field registration, dunder installation
# ---------------------------------------------------------------------------


def _own_annotations(cls: type) -> dict[str, Any]:
    """Return annotations declared directly on *cls*."""
    # Python 3.14+ (PEP 749): annotations are lazily evaluated via
    # __annotate__ and no longer stored directly in __dict__.  getattr()
    # triggers evaluation and returns per-class annotations correctly.
    # On Python < 3.14, getattr() follows MRO and returns *parent*
    # annotations when the child has none — use __dict__ to avoid that.
    if sys.version_info >= (3, 14):
        return getattr(cls, "__annotations__", {})
    return cls.__dict__.get("__annotations__", {})


def _field_owner_classes(cls: type) -> list[type]:
    """Classes whose annotations become this type's own fields."""
    registered_parent = next(
        (b for b in cls.__mro__[1:] if "__tvm_ffi_type_info__" in b.__dict__), object
    )
    represented = set(registered_parent.__mro__)
    return [
        b
        for b in reversed(cls.__mro__)
        if b is not object and b not in represented and _own_annotations(b)
    ]


def _collect_own_fields(  # noqa: PLR0912
    cls: type,
    owner: type,
    hints: dict[str, Any],
    kw_only_active: bool,
    decorator_frozen: bool,
) -> tuple[list[Field], bool]:
    """Parse own annotations into :class:`Field` objects.

    - Skips ``ClassVar`` annotations.
    - Handles ``KW_ONLY`` sentinel.
    - Extracts ``Field`` metadata from class attributes (set via :func:`field`).
    - Handles bare defaults (non-``Field`` values).
    - Converts resolved types to ``TypeSchema``.
    - Resolves ``hash=None`` to follow ``compare``.
    """
    fields: list[Field] = []
    own_annotations = _own_annotations(owner)

    for name in own_annotations:
        resolved_type = hints.get(name)
        # Skip ClassVar
        if (
            resolved_type is None
            or resolved_type is ClassVar
            or typing.get_origin(resolved_type) is ClassVar
        ):
            continue

        # KW_ONLY sentinel
        if resolved_type is KW_ONLY:
            kw_only_active = True
            if owner is cls and name in cls.__dict__:
                try:
                    delattr(cls, name)
                except AttributeError:
                    pass
            continue

        # Extract Field from class dict (inline of _pop_field_from_class)
        class_val = owner.__dict__.get(name, MISSING)
        if isinstance(class_val, Field):
            f = class_val if owner is cls else copy(class_val)
        elif class_val is not MISSING:
            f = field(default=class_val)
        else:
            f = field()
        if owner is cls and class_val is not MISSING:
            try:
                delattr(cls, name)
            except AttributeError:
                pass

        # Fill in name, schema, and resolved type (set by the decorator, not the user)
        f.name = name
        f._ty_schema = TypeSchema.from_annotation(resolved_type)
        f.type = resolved_type

        # Resolve kw_only: None means "inherit from decorator"
        if f.kw_only is None:
            f.kw_only = kw_only_active

        # Apply class-level frozen when the field doesn't explicitly set it
        if decorator_frozen and not f.frozen:
            f.frozen = True

        # Resolve hash=None → follow compare (native dataclass semantics)
        if f.hash is None:
            f.hash = f.compare

        fields.append(f)

    return fields, kw_only_active


def method(fn: Any) -> Any:
    """Mark a ``@py_class`` method for FFI TypeMethod registration.

    Decorate any staticmethod or plain instance method on a ``@py_class``
    body to have it land in the C-level ``TVMFFITypeInfo.methods[]``
    table. Once registered, the method is resolvable by name from any
    FFI consumer — Python-side reflection via ``TypeInfo.methods``,
    C++, Rust — through the same path already used by C++-defined
    methods declared via ``refl::ObjectDef<T>().def(...)``.

    Example::

        from tvm_ffi import Object, method
        from tvm_ffi.dataclasses import py_class


        @py_class("example.Node")
        class Node(Object):
            x: int

            @method
            def label(self) -> str:
                return f"N({self.x})"


        # The method is now in ``TypeInfo.methods`` and FFI-callable:
        info = Node.__tvm_ffi_type_info__
        fn = next(m.func for m in info.methods if m.name == "label")
        fn(Node(x=7))  # -> "N(7)"

    ``staticmethod`` is supported: the marker is written onto the
    underlying function and unwrapped at registration time. Plain
    functions are also accepted — the marker lives on the function
    object directly. ``classmethod`` is rejected at decoration time
    because its ``cls``-first dispatch does not match the
    packed-call convention.
    """
    if isinstance(fn, staticmethod):
        fn.__func__.__ffi_method__ = True
        return fn
    if isinstance(fn, classmethod):
        raise TypeError(
            "@tvm_ffi.method: @classmethod is not supported for FFI "
            "TypeMethod registration — the classmethod's ``cls`` first "
            "arg does not match the packed-call convention. Use "
            "@staticmethod or a plain instance method instead.",
        )
    if not callable(fn):
        raise TypeError(
            f"@tvm_ffi.method: expected a callable, got {type(fn).__name__}.",
        )
    fn.__ffi_method__ = True
    return fn


def _is_method_marked(value: Any) -> bool:
    """Return True when ``value`` is a callable marked by :func:`method`."""
    if isinstance(value, (staticmethod, classmethod)):
        return getattr(value.__func__, "__ffi_method__", False) is True
    if callable(value):
        return getattr(value, "__ffi_method__", False) is True
    return False


def _validate_method_name(cls: type, name: str) -> None:
    """Reject ``@method``-marked names that collide with reserved namespaces.

    Names in :data:`_FFI_TYPE_ATTR_NAMES` and Python-protocol dunders
    are not allowed for ``@method`` — they are routed through the
    TypeAttrColumn / Python-protocol paths instead.
    """
    if name in _FFI_TYPE_ATTR_NAMES:
        raise NameError(
            f"@py_class({cls.__name__!r}): {name!r} is a TypeAttrColumn "
            "name — define it directly on the class body without "
            "``@method``; the FFI system routes it to ``TVMFFITypeRegisterAttr``.",
        )
    if name.startswith("__ffi_"):
        raise NameError(
            f"@py_class({cls.__name__!r}): {name!r} starts with the "
            "reserved ``__ffi_`` prefix. Pick a different name for your "
            "``@method``-decorated method.",
        )
    if name.startswith("__") and name.endswith("__"):
        raise NameError(
            f"@py_class({cls.__name__!r}): {name!r} is a Python protocol "
            "dunder — these are reserved for Python semantics and cannot "
            "be registered as FFI TypeMethods.",
        )


def _collect_py_methods(cls: type) -> list[tuple[str, Any, bool]] | None:
    """Extract FFI-registered entries from a ``@py_class`` body.

    Two sources are collected:

    1. **TypeAttrColumn dunders** — names in :data:`_FFI_RECOGNIZED_METHODS`
       that appear in ``cls.__dict__``. Both callables (e.g.
       ``__ffi_repr__``) and non-callable values flow here; the Cython
       layer routes them to ``TVMFFITypeRegisterAttr`` based on name.
    2. **User TypeMethods** — every callable in ``cls.__dict__`` marked
       with :func:`method`. Registered via ``TVMFFITypeRegisterMethod``
       so the method is resolvable by name from any FFI consumer
       (introspection through ``TypeInfo.methods``, name-based lookup
       from C++ / Rust, etc.). The decorator pattern keeps the
       per-class declaration co-located with the method body; no
       separate allowlist.

    Validation runs at registration time — reserved ``__ffi_*`` names
    and Python protocol dunders cannot be ``@method``-decorated; those
    are reserved by the TypeAttrColumn and Python semantics respectively.

    Returns the ``(name, value, is_static)`` list, or :data:`None` when
    no entries were found.
    """
    methods: list[tuple[str, Any, bool]] = []
    for name, value in cls.__dict__.items():
        marked = _is_method_marked(value)
        if name not in _FFI_RECOGNIZED_METHODS and not marked:
            continue
        if marked:
            _validate_method_name(cls, name)
        # In every case, registering a classmethod as a TypeMethod is
        # wrong: the packed-call convention places ``self`` (an instance)
        # in slot 0, but classmethod's descriptor binds slot 0 to the
        # class.
        if isinstance(value, classmethod):
            raise TypeError(
                f"@py_class({cls.__name__!r}): {name!r} is wrapped by "
                "@classmethod, which is incompatible with FFI "
                "registration — the cls-first arg breaks the packed-call "
                "convention. Use @staticmethod or a plain instance "
                "method. If you wrote ``@classmethod @method``, swap to "
                "``@staticmethod @method`` (or drop @classmethod).",
            )
        is_static = isinstance(value, staticmethod)
        func = value.__func__ if is_static else value
        methods.append((name, func, is_static))
    return methods if methods else None


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


def _register_fields_into_type(
    cls: type,
    type_info: Any,
    globalns: dict[str, Any],
    params: dict[str, Any],
) -> bool:
    """Phase 2: resolve annotations, register fields, install dunders.

    Returns True on success, False if forward references are unresolved.
    """
    # Resolve string annotations to types; return False (defer) on NameError.
    #
    # First try with module-scoped localns (standard Python name resolution).
    # On NameError, retry with a cross-module localns that includes classes
    # from every registered module — this handles circular imports where the
    # target of a forward reference is imported only under TYPE_CHECKING and
    # therefore never enters the declaring module's globals.
    kwargs: dict[str, Any] = {"globalns": globalns, "localns": _build_localns(cls)}
    if sys.version_info >= (3, 11):
        kwargs["include_extras"] = True
    try:
        hints = typing.get_type_hints(cls, **kwargs)
    except (NameError, AttributeError):
        kwargs["localns"] = _build_localns(cls, cross_module=True)
        try:
            hints = typing.get_type_hints(cls, **kwargs)
        except (NameError, AttributeError):
            return False

    fields_map: dict[str, Field] = {}
    kw_only_active = params["kw_only"]
    for owner in _field_owner_classes(cls):
        owner_fields, kw_only_active = _collect_own_fields(
            cls,
            owner,
            hints,
            kw_only_active,
            params["frozen"],
        )
        for f in owner_fields:
            assert f.name is not None
            fields_map[f.name] = f
    own_fields = list(fields_map.values())
    py_methods = _collect_py_methods(cls)

    # Register fields and type-level structural eq/hash kind with the C layer.
    structure_kind = _STRUCTURE_KIND_MAP.get(params.get("structural_eq"))
    type_info._register_fields(own_fields, structure_kind)
    # Attach the user's Field sentinel to each TypeField so the
    # ``tvm_ffi.dataclasses.fields()`` compat layer can recover defaults
    # and default_factory values.  _register_fields preserves order, so
    # own_fields and type_info.fields line up 1:1.
    for py_field, type_field in zip(own_fields, type_info.fields):
        type_field.dataclass_field = py_field
    # Register user-defined dunder methods and read back system-generated ones.
    # Non-callable entries whose names are in _FFI_TYPE_ATTR_NAMES are routed
    # to TVMFFITypeRegisterAttr by the Cython layer.
    type_info._register_py_methods(py_methods, type_attr_names=_FFI_TYPE_ATTR_NAMES)
    _add_class_attrs(cls, type_info)

    # Remove deferred __init__ and restore user-defined __init__ if saved
    if "_py_class_deferred_init" in cls.__dict__:
        # Always remove the deferred wrapper
        if "__init__" in cls.__dict__:
            delattr(cls, "__init__")
        try:
            delattr(cls, "_py_class_deferred_init")
        except AttributeError:
            pass
        # Restore user-defined __init__ if it was saved
        user_init = cls.__dict__.get("_py_class_user_init")
        if user_init is not None:
            cls.__init__ = user_init
            delattr(cls, "_py_class_user_init")

    _install_dataclass_dunders(
        cls,
        init=params["init"],
        repr=params["repr"],
        eq=params["eq"],
        order=params["order"],
        unsafe_hash=params["unsafe_hash"],
        match_args=params["match_args"],
        py_class_mode=True,
    )
    return True


# ---------------------------------------------------------------------------
# Deferred resolution (when phase 2 cannot run at decoration time)
# ---------------------------------------------------------------------------


def _flush_pending() -> None:
    """Retry all pending classes.  Called after each successful phase 2."""
    changed = True
    while changed:
        changed = False
        remaining: list[_PendingClass] = []
        for entry in _PENDING_CLASSES:
            if _register_fields_into_type(entry.cls, entry.type_info, entry.globalns, entry.params):
                changed = True
            else:
                remaining.append(entry)
        _PENDING_CLASSES[:] = remaining


def _raise_unresolved_forward_reference(cls: type, globalns: dict[str, Any]) -> None:
    """Raise :class:`TypeError` listing the annotations that cannot be resolved."""
    localns = _build_localns(cls, cross_module=True)
    owners = _field_owner_classes(cls)
    localns.update({owner.__name__: owner for owner in owners})
    unresolved: list[str] = []
    for owner in owners:
        for name, ann_str in _own_annotations(owner).items():
            if isinstance(ann_str, str):
                try:
                    eval(ann_str, globalns, localns)
                except NameError:
                    unresolved.append(f"{name}: {ann_str}")
    raise TypeError(
        f"Cannot instantiate {cls.__name__}: unresolved forward references: {unresolved}"
    )


def _make_temporary_init(
    cls: type, type_info: Any, globalns: dict[str, Any], params: dict[str, Any]
) -> Callable[[...], None]:
    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        if type_info.fields is None:
            try:
                if not _register_fields_into_type(cls, type_info, globalns, params):
                    _raise_unresolved_forward_reference(cls, globalns)
                # cls stays in _PENDING_CLASSES after phase-2 succeeds; drop it
                # before _flush_pending so the loop doesn't hit the Cython-level
                # "_register_fields already called" assertion on a second pass.
                _PENDING_CLASSES[:] = [p for p in _PENDING_CLASSES if p.cls is not cls]
                _flush_pending()
            except Exception:
                # Remove from pending list and roll back so the type key can be reused.
                _PENDING_CLASSES[:] = [p for p in _PENDING_CLASSES if p.cls is not cls]
                _rollback_registration(cls, type_info)
                raise
        # cls.__init__ has been replaced by the real init (or restored user init)
        cls.__init__(self, *args, **kwargs)

    __init__.__qualname__ = f"{cls.__qualname__}.__init__"
    __init__.__module__ = cls.__module__
    return __init__


def _install_deferred_init(
    cls: type,
    type_info: Any,
    globalns: dict[str, Any],
    params: dict[str, Any],
) -> None:
    """Install a temporary ``__init__`` that completes registration on first call.

    Preserves a user-defined ``__init__`` if present in *cls.__dict__*;
    it is restored by :func:`_register_fields_into_type` after registration
    completes so that ``_install_dataclass_dunders`` sees it and skips
    auto-generation.
    """
    # Save user-defined __init__ before overwriting
    user_init = cls.__dict__.get("__init__")
    if user_init is not None:
        cls._py_class_user_init = user_init  # type: ignore[attr-defined]

    cls.__init__ = _make_temporary_init(cls, type_info, globalns, params)  # type: ignore[assignment]
    cls._py_class_deferred_init = True  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Main decorator
# ---------------------------------------------------------------------------


#: Mapping from Python string names to C-level ``TVMFFISEqHashKind`` enum values.
_STRUCTURE_KIND_MAP: dict[str | None, int] = {
    None: 0,  # kTVMFFISEqHashKindUnsupported (default; no metadata registered)
    "tree": 1,  # kTVMFFISEqHashKindTreeNode
    "var": 2,  # kTVMFFISEqHashKindFreeVar
    "dag": 3,  # kTVMFFISEqHashKindDAGNode
    "const-tree": 4,  # kTVMFFISEqHashKindConstTreeNode
    "singleton": 5,  # kTVMFFISEqHashKindUniqueInstance
}

#: Names that should be registered as TypeAttrColumn entries (for C++
#: dispatch via ``TypeAttrColumn``), NOT as TypeMethod.
#:
#: See ``reflection::type_attr`` in ``accessor.h`` for the C++ constants.
_FFI_TYPE_ATTR_NAMES: frozenset[str] = frozenset(
    {
        "__ffi_repr__",
        "__ffi_hash__",
        "__ffi_eq__",
        "__ffi_compare__",
        "__ffi_convert__",
        "__any_hash__",
        "__any_equal__",
        "__s_equal__",
        "__s_hash__",
        "__data_to_json__",
        "__data_from_json__",
    }
)

#: Allowlist of dunder names that ``_collect_py_methods`` collects from
#: the class body.  Names in ``_FFI_TYPE_ATTR_NAMES`` are registered as
#: TypeAttrColumn entries; all other names are registered as TypeMethod.
#:
#: System-managed names (``__ffi_new__``, ``__ffi_init__``,
#: ``__ffi_shallow_copy__``) are intentionally
#: absent because the C++ runtime generates them.
_FFI_RECOGNIZED_METHODS: frozenset[str] = _FFI_TYPE_ATTR_NAMES


@dataclass_transform(
    eq_default=False,
    order_default=False,
    field_specifiers=(field, Field),
)
def py_class(  # noqa: PLR0913
    cls_or_type_key: type | str | None = None,
    /,
    *,
    type_key: str | None = None,
    frozen: bool = False,
    init: bool = True,
    repr: bool = True,
    eq: bool = False,
    order: bool = False,
    unsafe_hash: bool = False,
    match_args: bool = True,
    kw_only: bool = False,
    structural_eq: str | None = None,
    slots: bool = True,
) -> Callable[[_T], _T] | _T:
    """Register a Python-defined FFI class with dataclass-style semantics.

    Can be used as:

    .. code-block:: python

        @py_class  # bare decorator
        class Point(Object):
            x: float
            y: float


        @py_class("my.Point")  # with explicit type_key
        class Point(Object): ...


        @py_class(eq=True, order=True)  # with options
        class Point(Object): ...


        @py_class("my.Point", eq=True)  # both
        class Point(Object): ...


        @py_class(structural_eq="tree")  # structural eq/hash kind
        class MyNode(Object):
            value: int
            span: Object = field(structural_eq="ignore")

    Parameters
    ----------
    cls_or_type_key
        When a string, used as the FFI type key.  When a type (bare
        decorator usage), the class to decorate.
    type_key
        Explicit FFI type key.  Auto-generated from
        ``{module}.{qualname}`` when omitted.
    frozen
        If True, all fields are read-only after ``__init__`` by default.
        Individual fields can still be marked ``field(frozen=True)`` on a
        non-frozen class.  Use ``type(obj).field_name.set(obj, value)``
        as an escape hatch when mutation is necessary.
    init
        If True (default), generate ``__init__`` from field annotations.
    repr
        If True (default), generate ``__repr__``.
    eq
        If True, generate ``__eq__`` and ``__ne__`` using recursive
        field-wise content equality.  Default False, in which case the
        class inherits the pointer-based ``__eq__`` from ``Object``
        (``a == b`` is equivalent to ``a.same_as(b)``).  If the class
        body defines ``__eq__`` or ``__ne__``, the generator is skipped
        and the user definition is preserved.
    order
        If True, generate ``__lt__``, ``__le__``, ``__gt__``, ``__ge__``.
        Requires ``eq=True``.
    unsafe_hash
        If True, generate ``__hash__`` using recursive field-wise
        content hashing (unsafe for mutable objects).  Default False,
        in which case the class inherits the handle-address ``__hash__``
        from ``Object``.  A user-defined ``__hash__`` in the class body
        is preserved.
    match_args
        If True (default), set ``__match_args__`` to a tuple of the
        positional ``__init__`` field names (``init=True`` and not
        ``kw_only``), enabling ``match`` statements.  Ignored when the
        class body already defines ``__match_args__``.
    kw_only
        If True, all fields are keyword-only in ``__init__`` by default.
    structural_eq
        Structural equality/hashing kind for this type.  Controls how
        instances participate in ``structural_equal`` and
        ``structural_hash``.  Valid values are:

        - ``None`` (default): structural comparison is not supported.
        - ``"tree"``: content-based comparison, the safe default for
          most IR nodes.
        - ``"var"``: compared by binding position, for variable types.
        - ``"dag"``: content + sharing-aware comparison, for dataflow
          graph nodes.
        - ``"const-tree"``: like ``"tree"`` with a pointer-equality
          fast path (only safe for types with no transitive ``"var"``
          children).
        - ``"singleton"``: pointer equality only, for singleton types.

        This parameter is **independent** of ``eq`` / ``unsafe_hash``:
        it only configures how ``structural_equal`` / ``structural_hash``
        walk the object in C++ and never installs or alters Python-level
        ``__eq__`` / ``__hash__``.  See Notes below.
    slots
        Accepted for ``dataclass_transform`` compatibility.  Object
        subclasses always use ``__slots__ = ()`` via the metaclass.

    Returns
    -------
    Callable | type
        A class decorator, or the decorated class (bare usage).

    Notes
    -----
    Three orthogonal equality/hashing mechanisms coexist on a
    ``@py_class`` type, each controlled by an independent knob:

    - ``a == b`` / ``hash(a)`` — selected by ``eq`` / ``unsafe_hash``
      params (or user-defined ``__eq__`` / ``__hash__`` in the class
      body).  Default: pointer-based ``same_as`` and handle-address
      hash, inherited from ``Object``.
    - ``structural_equal(a, b)`` / ``structural_hash(a)`` — selected
      by the ``structural_eq`` param.  Default (``None``): structural
      comparison is unsupported for this type.
    - ``a.same_as(b)`` — always available; always pointer comparison.

    The typical pattern is to leave ``eq`` / ``unsafe_hash`` at their
    defaults so ``==`` and ``hash()`` stay cheap and pointer-based
    (ideal for pass-internal bookkeeping such as visited-set tracking),
    and call ``structural_equal`` / ``structural_hash`` explicitly at
    the points that require the heavy semantic check.

    Combining ``eq=True`` (or ``unsafe_hash=True``) with a
    ``structural_eq`` kind is legal but gives the type two different
    recursive equalities — a Python-level one for ``==`` and a C++
    structural one for ``structural_equal`` — which rarely coincide.
    Prefer setting only one.

    """
    if order and not eq:
        raise ValueError("order=True requires eq=True")
    if structural_eq not in _STRUCTURE_KIND_MAP:
        raise ValueError(
            f"structural_eq must be one of "
            f"{sorted(k for k in _STRUCTURE_KIND_MAP if k is not None)}"
            f" or None, got {structural_eq!r}"
        )

    effective_type_key = type_key
    params: dict[str, Any] = {
        "frozen": frozen,
        "init": init,
        "repr": repr,
        "eq": eq,
        "order": order,
        "unsafe_hash": unsafe_hash,
        "match_args": match_args,
        "kw_only": kw_only,
        "structural_eq": structural_eq,
    }

    def decorator(cls: _T) -> _T:
        nonlocal effective_type_key
        globalns = getattr(sys.modules.get(cls.__module__, None), "__dict__", {})

        info = _register_type_without_fields(cls, effective_type_key)

        try:
            if _register_fields_into_type(cls, info, globalns, params):
                _flush_pending()
            else:
                _PENDING_CLASSES.append(_PendingClass(cls, info, globalns, params))
                _install_deferred_init(cls, info, globalns, params)
        except Exception:
            # Phase-2 failed (bad annotation, field ordering, etc.).
            # Roll back phase-1 so the type key can be reused after
            # the user fixes the error.
            _rollback_registration(cls, info)
            raise

        # Marker: distinguishes @c_class / @py_class types from FFI containers
        # (Array, List, Map, Dict) that also have __tvm_ffi_type_info__ but are
        # not dataclasses.  Used by is_dataclass() in common.py.
        setattr(cls, "__tvm_ffi_is_dataclass__", True)
        return cls

    # Handle different calling conventions:
    #   @py_class                → cls_or_type_key is the class
    #   @py_class("key")         → cls_or_type_key is a string
    #   @py_class()              → cls_or_type_key is None
    #   @py_class(eq=True)       → cls_or_type_key is None
    if cls_or_type_key is None:
        return decorator
    if isinstance(cls_or_type_key, str):
        effective_type_key = cls_or_type_key
        return decorator
    if isinstance(cls_or_type_key, type):
        return decorator(cls_or_type_key)
    raise TypeError(f"py_class: expected str or type, got {type(cls_or_type_key)}")
