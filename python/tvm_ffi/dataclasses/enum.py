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
"""Cross-language enum types: named, frozen, ordinal-indexed singletons.

An ``Enum`` subclass has one of two usage modes, distinguished by whether
its ``type_key`` is already registered in the FFI type system:

* **Closed Python enum** — fresh ``type_key``, variants declared once in
  the class body.  Behavior matches ``enum.Enum``.
* **Cross-language registry** — ``type_key`` also registered in C++ (or
  another Python module).  Python and C++ both contribute variants to
  the same per-class registry, and consumers attach *extensible
  attributes* to variants from any module at any time.

See :class:`Enum` for declaration forms and :meth:`Enum.def_attr` for
extensible attributes.

Storage layout (mirrors ``include/tvm/ffi/enum.h``):

* ``__ffi_enum_entries__`` — ``Dict[str, Enum]``, name → variant.
* ``__ffi_enum_attrs__``   — ``Dict[str, List[Any]]``, extensible-attr
  name → column indexed by each variant's ordinal.
"""

from __future__ import annotations

import sys
import typing
from collections.abc import Callable, Iterator
from typing import Any, ClassVar

from typing_extensions import dataclass_transform

from .. import core
from ..container import Dict, List
from ..core import Object
from .c_class import c_class
from .field import Field, field
from .py_class import py_class

__all__ = [
    "ENUM_ATTRS_ATTR",
    "ENUM_ENTRIES_ATTR",
    "Enum",
    "EnumAttrMap",
    "auto",
    "entry",
]

#: TypeAttr column storing ``Dict[str, Enum]`` (instance name → singleton).
ENUM_ENTRIES_ATTR = "__ffi_enum_entries__"

#: TypeAttr column storing ``Dict[str, List[Any]]`` of per-variant attrs.
ENUM_ATTRS_ATTR = "__ffi_enum_attrs__"


# ---------------------------------------------------------------------------
# entry() sentinel
# ---------------------------------------------------------------------------


class _EnumEntry:
    """Sentinel produced by :func:`entry`; consumed by ``Enum.__init_subclass__``.

    Holds the positional and keyword arguments forwarded to the subclass's
    ``__init__`` when the variant is materialized.  ``value`` and ``name``
    are auto-assigned (dense ordinal and class-body name) and must not
    appear in the captured arguments.
    """

    __slots__ = ("args", "kwargs")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args: tuple[Any, ...] = args
        self.kwargs: dict[str, Any] = kwargs

    def __repr__(self) -> str:
        parts = [repr(a) for a in self.args]
        parts.extend(f"{k}={v!r}" for k, v in self.kwargs.items())
        return f"entry({', '.join(parts)})"


def entry(*args: Any, **kwargs: Any) -> Any:
    """Declare a new enum variant with values for its declared fields.

    ``entry(...)`` is a class-body sentinel; it never produces a real
    instance.  At class creation, :meth:`Enum.__init_subclass__` scans
    for these sentinels and, for each one, constructs a singleton variant
    by forwarding the captured positional and keyword arguments to the
    subclass's ``__init__``, together with an auto-assigned
    :attr:`~Enum.value` (dense ordinal) and :attr:`~Enum.name`
    (class-body name).

    Prefer :func:`auto` when a variant has no declared fields beyond the
    auto-assigned ordinal and name — it expresses intent without the
    empty-arg-list noise.

    Examples
    --------
    Variant with declared fields:

    .. code-block:: python

        from typing import ClassVar


        class Activation(Enum, type_key="my.Activation"):
            output_zero: bool
            is_monotonic: bool

            relu: ClassVar[Activation] = entry(output_zero=True, is_monotonic=True)
            gelu: ClassVar[Activation] = entry(output_zero=False, is_monotonic=False)

    Returns
    -------
    object
        An opaque sentinel.  The declared return type is ``Any`` so that
        ``ClassVar[Cls] = entry(...)`` type-checks even though the sentinel
        is not a real ``Cls``.

    """
    return _EnumEntry(*args, **kwargs)


def auto() -> Any:
    """Declare a new enum variant with no declared fields.

    Semantically equivalent to :func:`entry` called with no arguments but
    reads more clearly for the common case where a variant differs from
    its siblings only by name and ordinal.  The resulting singleton has
    only the auto-assigned :attr:`~Enum.value` and :attr:`~Enum.name`.

    ``auto()`` registers a *new* Python-side variant; it is not the right
    tool for binding to a pre-existing C++-registered entry (use a bare
    ``ClassVar[Cls]`` annotation for that — see :class:`Enum`).

    Examples
    --------
    .. code-block:: python

        class Status(Enum, type_key="my.Status"):
            ok = auto()
            err = auto()
            retry = auto()


        assert Status.ok.value == 0
        assert Status.err.name == "err"

    Returns
    -------
    object
        An opaque sentinel, the same kind returned by :func:`entry`.  The
        declared return type is ``Any`` so that both ``name = auto()`` and
        ``name: ClassVar[Cls] = auto()`` type-check.

    """
    return _EnumEntry()


# ---------------------------------------------------------------------------
# Class-level helpers
# ---------------------------------------------------------------------------


class _ClassProperty:
    """Read-only descriptor whose getter receives the owning class.

    Used for ``by_name``/``by_value``/``attr_dict`` so they work as class-level
    attribute access (e.g., ``Op.attr_dict["has_side_effects"]``) without
    needing a metaclass.
    """

    __slots__ = ("_fget",)

    def __init__(self, fget: Callable[[type], Any]) -> None:
        self._fget = fget

    def __get__(self, instance: Any, owner: type | None = None) -> Any:
        cls = owner if owner is not None else type(instance)
        return self._fget(cls)


# ---------------------------------------------------------------------------
# Enum base + EnumAttrMap
# ---------------------------------------------------------------------------


@dataclass_transform(
    eq_default=False,
    order_default=False,
    frozen_default=True,
    field_specifiers=(Field, field, entry, auto),
)
@c_class("ffi.Enum", init=False)
class Enum(Object):
    """A named-singleton registry with cross-language identity.

    Subclasses declare variants: frozen, named, ordinal-indexed
    singletons — the familiar enum pattern.  Unlike ``enum.Enum``, an
    ``Enum`` subclass bound to an FFI-registered ``type_key`` has an
    **open variant set**: C++ translation units and other Python modules
    binding the same ``type_key`` can contribute variants to the shared
    registry.  Per-variant metadata can also be attached post-hoc via
    :meth:`def_attr` as an *extensible attribute*, outside the class
    definition.

    For **closed, Python-only enums**, use a fresh ``type_key`` with
    :func:`auto` / :func:`entry` — behavior matches ``enum.Enum``.

    Attributes
    ----------
    value : int
        Dense ordinal assigned at registration (0-indexed per class).
    name : str
        Instance name (e.g., ``"Add"`` for ``Op.Add``).

    Closed Python enum
    ------------------
    Pick a fresh ``type_key`` and list variants with :func:`auto` or
    :func:`entry`.  The variant set is fixed at class-definition time.

    .. code-block:: python

        class Priority(Enum, type_key="my.Priority"):
            low = auto()
            medium = auto()
            high = auto()


        # Variants with declared fields — values supplied via entry(...).
        class Activation(Enum, type_key="my.Activation"):
            output_zero: bool
            is_monotonic: bool

            relu: ClassVar[Activation] = entry(output_zero=True, is_monotonic=True)
            gelu: ClassVar[Activation] = entry(output_zero=False, is_monotonic=False)

    Cross-language registry
    -----------------------
    When ``type_key`` is already registered (typically by C++), the
    Python class *binds* to the existing type rather than creating a
    new one.  Bare ``ClassVar[Cls]`` annotations bind to variants
    already registered on the C++ side; :func:`entry` / :func:`auto`
    still register fresh Python variants whose ordinals extend past the
    C++ ones.  All variants — regardless of origin — land in the same
    per-class registry and are visible to every binder of the same
    ``type_key``.

    .. code-block:: python

        # Registered in C++ via refl::EnumDef<VariantObj>("Alpha")... .
        class Variant(Enum, type_key="testing.TestEnumVariant"):
            Alpha: ClassVar[Variant]  # binds to C++-registered "Alpha"
            Beta: ClassVar[Variant]  # binds to C++-registered "Beta"

    Declaration forms
    -----------------
    Four shapes are supported in the class body:

    1. ``name = auto()`` — new variant with no declared fields.
    2. ``name: ClassVar[Cls] = entry(**kwargs)`` — new variant; ``kwargs``
       populate declared fields.
    3. ``name = entry(**kwargs)`` — same as (2), without the ``ClassVar``
       annotation.
    4. ``name: ClassVar[Cls]`` — in cross-language mode, binds to an
       existing C++-registered variant (error if unknown); otherwise
       registers a blank Python variant.

    Integer literals (``ok = 0``) are rejected: :attr:`value` is
    auto-assigned, so a user-supplied ordinal would either silently
    duplicate or conflict.  ``entry(value=...)`` and ``entry(name=...)``
    raise :class:`TypeError` at class-body time.

    Differences from ``enum.Enum``
    ------------------------------
    * **Same**: :attr:`name`, :attr:`value`, iteration, identity
      comparison; closed-set behavior when ``type_key`` is fresh.
    * **Extended**: ``entry(**kwargs)`` replaces the tuple-RHS idiom;
      ``dataclass_transform`` gives native type-checker support; open
      registry when ``type_key`` is shared across languages.
    * **Different**: :attr:`value` is always the ordinal (no
      user-supplied integer values); :meth:`def_attr` adds extensible
      attributes outside the class schema.
    * **Not provided**: ``Flag`` / ``IntFlag``, member aliasing,
      ``_missing_`` hook.

    Subclasses inherit :meth:`get`, :meth:`entries`, :meth:`def_attr`,
    and the ``by_name`` / ``by_value`` / ``attr_dict`` class-level
    views.

    """

    __slots__ = ()

    value: int
    name: str

    def __init_subclass__(
        cls,
        *,
        type_key: str | None = None,
        frozen: bool = True,
        init: bool = True,
        repr: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init_subclass__(**kwargs)
        if type_key is None:
            return

        binders, python_entries = _collect_entry_declarations(cls)

        cxx_backed = core._object_type_key_to_index(type_key) is not None
        if cxx_backed:
            c_class(type_key, init=init, repr=repr)(cls)
        else:
            py_class(type_key, frozen=frozen)(cls)

        _resolve_entries(cls, binders, python_entries, type_key=type_key, cxx_backed=cxx_backed)

    @classmethod
    def get(cls, name: str) -> Enum:
        """Return the variant named *name*, or raise :class:`KeyError`."""
        entries = _entries_dict(cls)
        if entries is not None and name in entries:
            return entries[name]
        raise KeyError(f"{cls.__name__} has no variant named {name!r}")

    @classmethod
    def entries(cls) -> Iterator[Enum]:
        """Iterate over all variants, in ordinal (value) order."""
        entries = _entries_dict(cls)
        if entries is None:
            return iter(())
        values = list(entries.values())
        values.sort(key=lambda v: int(v.value))
        return iter(values)

    @_ClassProperty
    def by_name(cls: type) -> Any:
        """Live ``Dict[str, Enum]`` mapping each variant's name to the variant singleton."""
        return _entries_dict(cls) or Dict({})

    @_ClassProperty
    def by_value(cls: type) -> list[Any]:
        """Return the variants as a list indexed by ordinal (``variant.value``)."""
        entries = _entries_dict(cls)
        if entries is None:
            return []
        ordered: list[Any] = [None] * len(entries)
        for inst in entries.values():
            ordered[int(inst.value)] = inst
        return ordered

    @_ClassProperty
    def attr_dict(cls: type) -> Any:
        """Live ``Dict[str, List[Any]]`` backing every extensible attribute.

        The outer dict is keyed by extensible-attribute name; each value
        is a list indexed by variant ordinal.  Prefer :meth:`def_attr`
        for normal per-variant reads and writes; this property is for
        bulk inspection and for reading values written by C++
        ``EnumDef::set_attr``.
        """
        return _attrs_dict(cls) or Dict({})

    @classmethod
    def def_attr(
        cls,
        name: str,
        *,
        default: Any = core.MISSING,
    ) -> EnumAttrMap:
        """Declare an *extensible attribute* column on this enum.

        Extensible attributes let any consumer associate per-variant data
        outside the enum's class-body schema — a lowering function
        attached to an operator by a code generator, a cost model
        registered only on some targets, a documentation string added
        after the fact.  Writes are last-write-wins for the same
        ``(variant, name)`` pair and visible to every consumer that
        calls :meth:`def_attr` with the same name on the same enum,
        including C++ code writing via ``EnumDef::set_attr``.

        Extensible attributes differ from **declared fields**:

        ====================  ========================  ==========================
        Concept               Lives on                  Added by
        ====================  ========================  ==========================
        Declared field        The variant object        Enum author, in class body
        Extensible attribute  ``__ffi_enum_attrs__``    Any consumer, any time
        ====================  ========================  ==========================

        Rule of thumb: if the data is part of *what a variant is*,
        declare a field in the class body; if it's part of *what a
        consumer wants to know*, attach it with :meth:`def_attr`.

        Parameters
        ----------
        name
            The extensible-attribute name (e.g., ``"has_side_effects"``).
            Writes go to ``attr_dict[name]`` as a list indexed by each
            variant's ordinal.
        default
            Value returned by ``attr[variant]`` when nothing was
            registered for that variant.  Left as ``MISSING`` to raise
            :class:`KeyError` on unset variants instead.  The default
            is a property of *this* :meth:`def_attr` call, not of the
            underlying column — two callers declaring the same name
            with different defaults share the storage but disagree on
            unset reads.

        Returns
        -------
        EnumAttrMap
            Mutable view over the column.  Use ``variant in attr`` to
            distinguish an explicit write from a default-hit.

        Notes
        -----
        :meth:`def_attr` is not a way to add fields to the enum's
        schema, subclass frozen variants, or bypass the frozen-instance
        invariant via ``setattr`` — for that, declare a field in the
        class body instead.

        """
        return EnumAttrMap(cls, name, default=default)


class EnumAttrMap:
    """Mutable per-variant view over an extensible-attribute column.

    Returned by :meth:`Enum.def_attr`.  Writes go to a ``List[Any]``
    column keyed by extensible-attribute name inside the per-class
    ``__ffi_enum_attrs__`` dict.  The list is indexed by each variant's
    ordinal (``variant.value``) and padded with ``None`` as new variants
    are registered.  The column is shared across every consumer —
    including C++ code writing via ``EnumDef::set_attr`` — and the data
    is not a field on the variant object.  See :meth:`Enum.def_attr` for
    full semantics.
    """

    __slots__ = ("_default", "_enum_cls", "_name")

    def __init__(self, enum_cls: type, name: str, *, default: Any = core.MISSING) -> None:
        self._enum_cls = enum_cls
        self._name = name
        self._default = default

    def _ordinal_of(self, variant: object) -> int:
        if not isinstance(variant, self._enum_cls):
            raise TypeError(
                f"{self._enum_cls.__name__}.def_attr({self._name!r}) expects a "
                f"{self._enum_cls.__name__} variant, got {type(variant).__name__}"
            )
        return int(variant.value)  # type: ignore[attr-defined]

    def _column(self, *, create: bool) -> Any | None:
        """Return the ``List[Any]`` column for this attribute; create if missing.

        Returns ``None`` when ``create`` is false and the column doesn't exist.
        """
        attrs = _attrs_dict(self._enum_cls) if not create else _ensure_attrs_dict(self._enum_cls)
        if attrs is None:
            return None
        if self._name in attrs:
            return attrs[self._name]
        if not create:
            return None
        col = List([])
        attrs[self._name] = col
        return col

    def __setitem__(self, variant: object, value: Any) -> None:
        ordinal = self._ordinal_of(variant)
        col = self._column(create=True)
        assert col is not None  # create=True always materialises the column.
        while len(col) <= ordinal:
            col.append(None)
        col[ordinal] = value

    def __getitem__(self, variant: object) -> Any:
        ordinal = self._ordinal_of(variant)
        col = self._column(create=False)
        if col is not None and ordinal < len(col):
            v = col[ordinal]
            if v is not None:
                return v
        if self._default is core.MISSING:
            raise KeyError(
                f"{self._enum_cls.__name__}.{variant.name} has no "  # type: ignore[attr-defined]
                f"extensible attribute {self._name!r} set"
            )
        return self._default

    def __contains__(self, variant: object) -> bool:
        if not isinstance(variant, self._enum_cls):
            return False
        try:
            ordinal = self._ordinal_of(variant)
        except TypeError:
            return False
        col = self._column(create=False)
        return col is not None and ordinal < len(col) and col[ordinal] is not None

    def get(self, variant: object, default: Any = None) -> Any:
        """Return the value for *variant*, or *default* if unset or foreign."""
        if not isinstance(variant, self._enum_cls):
            return default
        try:
            return self[variant]
        except KeyError:
            return default

    @property
    def name(self) -> str:
        """The extensible-attribute name passed to :meth:`Enum.def_attr`."""
        return self._name


# ---------------------------------------------------------------------------
# TypeAttr accessors
# ---------------------------------------------------------------------------


def _entries_dict(cls: type) -> Any:
    type_info = getattr(cls, "__tvm_ffi_type_info__", None)
    if type_info is None:
        return None
    return core._lookup_type_attr(type_info.type_index, ENUM_ENTRIES_ATTR)


def _attrs_dict(cls: type) -> Any:
    type_info = getattr(cls, "__tvm_ffi_type_info__", None)
    if type_info is None:
        return None
    return core._lookup_type_attr(type_info.type_index, ENUM_ATTRS_ATTR)


def _ensure_entries_dict(cls: type) -> Any:
    """Return the live ``__ffi_enum_entries__`` dict, registering it if absent."""
    type_info = cls.__tvm_ffi_type_info__  # ty: ignore[unresolved-attribute]
    entries = core._lookup_type_attr(type_info.type_index, ENUM_ENTRIES_ATTR)
    if entries is not None:
        return entries
    entries = Dict({})
    core._register_type_attr(type_info.type_index, ENUM_ENTRIES_ATTR, entries)
    # Re-read so mutations go through the ref owned by the registry.
    return core._lookup_type_attr(type_info.type_index, ENUM_ENTRIES_ATTR)


def _ensure_attrs_dict(cls: type) -> Any:
    """Return the live ``__ffi_enum_attrs__`` dict, registering it if absent."""
    type_info = cls.__tvm_ffi_type_info__  # ty: ignore[unresolved-attribute]
    attrs = core._lookup_type_attr(type_info.type_index, ENUM_ATTRS_ATTR)
    if attrs is not None:
        return attrs
    attrs = Dict({})
    core._register_type_attr(type_info.type_index, ENUM_ATTRS_ATTR, attrs)
    return core._lookup_type_attr(type_info.type_index, ENUM_ATTRS_ATTR)


# ---------------------------------------------------------------------------
# Class-body scanning + entry materialisation
# ---------------------------------------------------------------------------


def _collect_entry_declarations(
    cls: type,
) -> tuple[list[str], dict[str, _EnumEntry]]:
    """Scan *cls.__dict__* for variant declarations.

    Returns ``(binders, python_entries)`` in declaration order:

    * *binders* — names annotated as ``ClassVar[Cls]`` with no assigned value.
      Each either binds to an existing C++-registered entry with the same
      name or registers a new blank Python entry.
    * *python_entries* — names assigned an ``entry(...)`` sentinel (with or
      without a ``ClassVar`` annotation).  Each registers a new Python entry
      using the captured args/kwargs.

    Matched assignments are removed from ``cls.__dict__`` so that
    ``@c_class`` / ``@py_class`` don't misinterpret them as field defaults or
    class constants.
    """
    annotations = _own_annotations(cls)
    dict_keys = set(cls.__dict__.keys())

    binders: list[str] = []
    for name, ann in annotations.items():
        if name.startswith("_"):
            continue
        if _is_class_var(ann) and name not in dict_keys:
            binders.append(name)

    python_entries: dict[str, _EnumEntry] = {}
    for name, value in list(cls.__dict__.items()):
        if name.startswith("_"):
            continue
        if isinstance(value, _EnumEntry):
            python_entries[name] = value
            try:
                delattr(cls, name)
            except AttributeError:
                pass

    return binders, python_entries


def _resolve_entries(
    cls: type,
    binders: list[str],
    python_entries: dict[str, _EnumEntry],
    *,
    type_key: str,
    cxx_backed: bool,
) -> None:
    """Materialise *binders* and *python_entries* into class-attribute singletons.

    Processing order matches declaration order: ``binders`` first (because
    their annotations appear before any class-body assignments), then
    ``python_entries`` in their class-body order.  Each newly registered
    entry gets a dense ordinal equal to the current entries-dict size, so
    ordinals stay compact and stable across registrations.

    A cxx-backed enum (``type_key`` was already registered in the FFI type
    system before this Python subclass was created) supports mixing C++ and
    Python entries: bare ``ClassVar[Cls]`` binders must name an existing
    C++-registered entry, but ``entry(...)``/``auto()`` sentinels may add
    fresh Python-side entries whose ordinals extend past the C++ entries.
    """
    entries = _ensure_entries_dict(cls)

    for name in binders:
        if name in entries:
            # Already materialised — either C++-registered or previously bound.
            setattr(cls, name, entries[name])
            continue
        if cxx_backed:
            raise _cxx_backed_unknown_binder_error(cls, name, type_key, entries)
        ordinal = len(entries)
        instance = _instantiate(cls, args=(), kwargs={}, ordinal=ordinal, name=name)
        entries[name] = instance
        setattr(cls, name, instance)

    for name, e in python_entries.items():
        if name in entries:
            raise RuntimeError(
                f"Duplicate enum entry {name!r} for {cls.__name__}: already "
                f"registered as ordinal {int(entries[name].value)}."
            )
        if "value" in e.kwargs or "name" in e.kwargs:
            raise TypeError(
                f"{cls.__name__}.{name}: `value` and `name` are auto-assigned "
                f"and must not appear in entry(...) arguments."
            )
        ordinal = len(entries)
        if cxx_backed:
            instance = _instantiate_cxx_backed(
                cls, args=e.args, kwargs=e.kwargs, ordinal=ordinal, name=name
            )
        else:
            instance = _instantiate(cls, args=e.args, kwargs=e.kwargs, ordinal=ordinal, name=name)
        entries[name] = instance
        setattr(cls, name, instance)


def _cxx_backed_unknown_binder_error(
    cls: type,
    name: str,
    type_key: str,
    entries: Any,
) -> RuntimeError:
    """Build a descriptive error for an unbindable bare ``ClassVar`` binder.

    A bare ``ClassVar[Cls]`` annotation on a cxx-backed enum means "bind to
    an existing C++ entry with this name" — if the C++ registry has no such
    entry, the declaration is almost always a typo.  For adding a *new*
    Python-side variant on a cxx-backed enum, use ``entry(...)`` or
    ``auto()`` instead.
    """
    known = list(entries.keys()) if entries is not None else []
    known_str = ", ".join(repr(k) for k in known) if known else "<none>"
    return RuntimeError(
        f"Cannot bind enum variant {name!r} on {cls.__name__}: the FFI "
        f"type {type_key!r} is already registered in C++ with entries "
        f"[{known_str}], but has no C++ entry named {name!r}. "
        f"Bare ``ClassVar[{cls.__name__}]`` binders on a C++-backed enum "
        f"must name an entry already registered in C++; they cannot "
        f"introduce new variants from Python. "
        f"If this was a typo, double-check the spelling against the known "
        f"entries above (`{name}: ClassVar[{cls.__name__}]`); if you meant "
        f"to add a new Python-side variant, use `{name} = auto()` or "
        f"`{name} = entry(...)` instead."
    )


def _instantiate(
    cls: type,
    *,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    ordinal: int,
    name: str,
) -> Any:
    """Construct a subclass instance with auto-assigned ``value``/``name``."""
    merged = dict(kwargs)
    merged["value"] = ordinal
    merged["name"] = name
    return cls(*args, **merged)


def _instantiate_cxx_backed(
    cls: type,
    *,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    ordinal: int,
    name: str,
) -> Any:
    """Construct a new variant of a cxx-backed enum without going through ``__init__``.

    C++-backed enums whose underlying type is registered with
    ``refl::init(false)`` (e.g. any subclass of ``EnumObj`` in C++) have no
    ``__ffi_init__``, so the usual ``cls(value=..., name=...)`` path is not
    available.  Mirror ``reflection::EnumDef`` by allocating a blank instance
    via ``__ffi_new__`` and populating fields through the frozen-setter
    escape hatch exposed on the reflected property descriptors.
    """
    if args:
        raise TypeError(
            f"{cls.__name__}.{name}: positional `entry(...)` args are not "
            f"supported when extending a C++-backed enum; use keyword "
            f"arguments naming reflected fields."
        )
    type_info = cls.__tvm_ffi_type_info__  # ty: ignore[unresolved-attribute]
    ffi_new = core._lookup_type_attr(type_info.type_index, "__ffi_new__")
    if ffi_new is None:
        raise RuntimeError(
            f"Cannot add Python enum variant {name!r} on {cls.__name__}: "
            f"its C++ type has no ``__ffi_new__`` allocator registered, so "
            f"blank instances cannot be created from Python."
        )
    instance = ffi_new()
    for key in ("value", "name", *kwargs.keys()):
        descriptor = getattr(cls, key, None)
        if descriptor is None or not hasattr(descriptor, "set"):
            raise TypeError(
                f"{cls.__name__}.{name}: cannot set field {key!r} on a "
                f"C++-backed enum — no reflected setter is available."
            )
    getattr(cls, "value").set(instance, ordinal)
    getattr(cls, "name").set(instance, name)
    for k, v in kwargs.items():
        getattr(cls, k).set(instance, v)
    return instance


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------


def _own_annotations(cls: type) -> dict[str, Any]:
    """Return *cls*'s own annotations dict (not inherited)."""
    if sys.version_info >= (3, 14):
        return dict(getattr(cls, "__annotations__", {}) or {})
    return dict(cls.__dict__.get("__annotations__", {}))


def _is_class_var(annotation: Any) -> bool:
    """Return True if *annotation* is ``ClassVar`` or ``ClassVar[...]``."""
    if annotation is ClassVar:
        return True
    if typing.get_origin(annotation) is ClassVar:
        return True
    if isinstance(annotation, str):
        stripped = annotation.replace(" ", "")
        return stripped.startswith("ClassVar") or stripped.startswith("typing.ClassVar")
    return False
