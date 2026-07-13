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

import inspect
import json
import sys
import typing
from collections.abc import Callable
from copy import copy
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar, overload

from typing_extensions import dataclass_transform

from .._dunder import _install_dataclass_dunders
from ..core import MISSING, TypeSchema
from ..registry import _add_class_attrs
from . import _resolve_fields
from .field import KW_ONLY, Field, _field_converter, field

if TYPE_CHECKING:
    from ..core import TypeInfo

_T = TypeVar("_T", bound=type)

# Mapping from Python string names to C-level ``TVMFFISEqHashKind`` enum values.
_STRUCTURE_KIND_MAP: dict[str | None, int] = {
    None: 0,  # kTVMFFISEqHashKindUnsupported (explicit opt-out; no metadata registered)
    "tree": 1,  # kTVMFFISEqHashKindTreeNode (default)
    "var": 2,  # kTVMFFISEqHashKindFreeVar
    "dag": 3,  # kTVMFFISEqHashKindDAGNode
    "const-tree": 4,  # kTVMFFISEqHashKindConstTreeNode
    "singleton": 5,  # kTVMFFISEqHashKindUniqueInstance
}

# Names that should be registered as TypeAttrColumn entries (for C++
# dispatch via ``TypeAttrColumn``), NOT as TypeMethod.
# See ``reflection::type_attr`` in ``accessor.h`` for the C++ constants.
_FFI_TYPE_ATTR_NAMES: frozenset[str] = frozenset(
    {
        "__ffi_repr__",
        "__ffi_hash__",
        "__ffi_eq__",
        "__ffi_compare__",
        "__ffi_convert__",
        "__ffi_convert_type_schema__",
        "__any_hash__",
        "__any_equal__",
        "__s_equal__",
        "__s_hash__",
        "__s_visit__",
        "__data_to_json__",
        "__data_from_json__",
    }
)

# Names collected directly from the class body. Names in
# ``_FFI_TYPE_ATTR_NAMES`` are registered as TypeAttrColumn entries; other
# names require explicit ``@method`` marking and register as TypeMethod.
_FFI_RECOGNIZED_METHODS: frozenset[str] = _FFI_TYPE_ATTR_NAMES


@overload
@dataclass_transform(
    eq_default=False,
    order_default=False,
    field_specifiers=(Field, field),
)
def py_class(
    cls_or_type_key: _T,
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
    structural_eq: str | None = "tree",
    slots: bool = True,
) -> _T: ...


@overload
@dataclass_transform(
    eq_default=False,
    order_default=False,
    field_specifiers=(Field, field),
)
def py_class(
    cls_or_type_key: str | None = None,
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
    structural_eq: str | None = "tree",
    slots: bool = True,
) -> Callable[[_T], _T]: ...


def method(fn: Any) -> Any:
    """Mark a ``@py_class`` method for FFI reflection.

    Decorate any staticmethod or plain instance method on a ``@py_class``
    body to have it collected during class registration.  Ordinary names land
    in the C-level ``TVMFFITypeInfo.methods[]`` table.  Names reserved for
    TypeAttrColumn dispatch, such as ``__ffi_repr__``, are routed to the
    type-attribute table instead.

    Once registered as a TypeMethod, the method is resolvable by name from any
    FFI consumer — Python-side reflection via ``TypeInfo.methods``, C++, Rust —
    through the same path already used by C++-defined methods declared via
    ``refl::ObjectDef<T>().def(...)``.

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


def _callable_type_schema_json() -> str:
    return json.dumps({"type_schema": TypeSchema("Callable").to_json()})


def _method_type_schema_json(
    cls: type,
    func: Any,
    is_static: bool,
    globalns: dict[str, Any],
) -> str:
    """Build reflection metadata for a Python-defined FFI TypeMethod."""
    kwargs: dict[str, Any] = {
        "globalns": globalns,
        "localns": _resolve_fields._build_localns(cls),
    }
    if sys.version_info >= (3, 11):
        kwargs["include_extras"] = True
    try:
        hints = typing.get_type_hints(func, **kwargs)
    except (NameError, AttributeError):
        kwargs["localns"] = _resolve_fields._build_localns(cls, cross_module=True)
        try:
            hints = typing.get_type_hints(func, **kwargs)
        except (NameError, AttributeError):
            return _callable_type_schema_json()

    sig = inspect.signature(func)
    params = list(sig.parameters.values())
    if any(
        param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        for param in params
    ):
        return _callable_type_schema_json()

    arg_schemas: list[TypeSchema] = []
    if not is_static:
        arg_schemas.append(TypeSchema.from_annotation(cls))
        params = params[1:]

    for param in params:
        if param.kind is inspect.Parameter.KEYWORD_ONLY:
            return _callable_type_schema_json()
        annotation = hints.get(param.name, Any)
        arg_schemas.append(TypeSchema.from_annotation(annotation))

    ret_annotation = hints.get("return", Any)
    ret_schema = TypeSchema.from_annotation(ret_annotation)
    schema = TypeSchema("Callable", (ret_schema, *arg_schemas))
    return json.dumps({"type_schema": schema.to_json()})


def _collect_py_methods(
    cls: type, globalns: dict[str, Any] | None = None
) -> list[tuple[Any, ...]] | None:
    """Extract Python-defined reflection entries from a ``@py_class`` body.

    Two sources are collected:

    1. **TypeAttrColumn dunders** — names in :data:`_FFI_TYPE_ATTR_NAMES`
       that appear in ``cls.__dict__``. Both callables (e.g.
       ``__ffi_repr__``) and non-callable values flow here; the Cython
       layer routes them to ``TVMFFITypeRegisterAttr`` based on name.
    2. **User TypeMethods** — every callable in ``cls.__dict__`` marked
       with :func:`method`, unless its name is a TypeAttrColumn name.
       Ordinary marked callables are registered via
       ``TVMFFITypeRegisterMethod`` so they are resolvable by name from
       any FFI consumer (introspection through ``TypeInfo.methods``,
       name-based lookup from C++ / Rust, etc.).  Marked callables whose
       names are in :data:`_FFI_TYPE_ATTR_NAMES` are still collected, but
       are routed to ``TVMFFITypeRegisterAttr``.

    Returns ``(name, value, is_static)`` tuples for direct helper use, or
    ``(name, value, is_static, metadata_json)`` tuples when *globalns* is
    provided by the registration path.
    """
    legacy_shape = globalns is None
    if globalns is None:
        globalns = vars(sys.modules[cls.__module__])

    methods: list[tuple[Any, ...]] = []
    for name, value in cls.__dict__.items():
        marked = _is_method_marked(value)
        if name not in _FFI_RECOGNIZED_METHODS and not marked:
            continue
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
        metadata_json = None
        if marked and name not in _FFI_TYPE_ATTR_NAMES:
            metadata_json = _method_type_schema_json(cls, func, is_static, globalns)
        if legacy_shape:
            methods.append((name, func, is_static))
        else:
            methods.append((name, func, is_static, metadata_json))
    return methods if methods else None


def on_fields_resolved(  # noqa: PLR0912, PLR0915
    type_info: TypeInfo,
    resolved_fields: _resolve_fields.ResolvedFields,
) -> None:
    """Finalize a ``@py_class`` after annotation resolution succeeds.

    ``_resolve_fields`` supplies owner classes and their resolved type hints.
    This function turns those hints into :class:`Field` objects, applies
    decorator-level defaults stored on ``type_info._decorator_args``, registers
    field metadata and structural-equality kind with the Cython layer, registers
    Python-defined TypeMethods and TypeAttrColumn values, restores any deferred
    user ``__init__``, and installs the dataclass-style dunder methods.
    """
    from ..utils.descriptors import init_property as _InitProperty  # noqa: PLC0415

    cls = type_info.type_cls
    assert cls is not None
    params = type_info._decorator_args
    owners, hints_by_owner = resolved_fields

    fields_map: dict[str, Field] = {}
    ip_funcs: dict[str, Any] = {}
    kw_only_active = params["kw_only"]
    for owner in owners:
        own_annotations = _resolve_fields.own_annotations(owner)
        for name in own_annotations:
            resolved_type = hints_by_owner[owner].get(name)
            # Skip ClassVar.
            if (
                resolved_type is None
                or resolved_type is ClassVar
                or typing.get_origin(resolved_type) is ClassVar
            ):
                continue

            # KW_ONLY sentinel.
            if resolved_type is KW_ONLY:
                kw_only_active = True
                if owner is cls and name in cls.__dict__:
                    try:
                        delattr(cls, name)
                    except AttributeError:
                        pass
                continue

            # Extract Field from class dict (inline of _pop_field_from_class).
            class_val = owner.__dict__.get(name, MISSING)
            if isinstance(class_val, Field):
                f = class_val if owner is cls else copy(class_val)
            elif isinstance(class_val, _InitProperty):
                # Synthesize a field; record the compute function for __init__.
                f = field(
                    init=False,
                    structural_eq="ignore",
                    repr=False,
                    hash=False,
                    compare=False,
                )
                ip_funcs[name] = class_val.func
            elif class_val is not MISSING:
                f = field(default=class_val)
            else:
                f = field()
            if owner is cls and class_val is not MISSING:
                try:
                    delattr(cls, name)
                except AttributeError:
                    pass

            # Fill in name, schema, and resolved type.
            f.name = name
            f._ty_schema = TypeSchema.from_annotation(resolved_type)
            f.type = resolved_type

            # Resolve kw_only: None means "inherit from decorator".
            if f.kw_only is None:
                f.kw_only = kw_only_active

            # Apply class-level frozen when the field doesn't explicitly set it.
            if params["frozen"] and not f.frozen:
                f.frozen = True

            # Resolve hash=None -> follow compare (native dataclass semantics).
            if f.hash is None:
                f.hash = f.compare

            assert f.name is not None
            fields_map[f.name] = f
    own_fields = list(fields_map.values())
    globalns = getattr(sys.modules.get(cls.__module__, None), "__dict__", {})
    if ip_funcs:
        setattr(cls, "__ffi_init_property_funcs__", ip_funcs)
    py_methods = _collect_py_methods(cls, globalns)

    # Register fields and type-level structural eq/hash kind with the C layer.
    structure_kind = _STRUCTURE_KIND_MAP.get(params.get("structural_eq"))
    type_info._register_fields(own_fields, structure_kind)
    # Attach the user's Field sentinel to each TypeField so the
    # ``tvm_ffi.dataclasses.fields()`` compat layer can recover defaults
    # and default_factory values.  _register_fields preserves order, so
    # own_fields and type_info.fields line up 1:1.
    assert len(own_fields) == len(type_info.fields)
    for py_field, type_field in zip(own_fields, type_info.fields):
        type_field.dataclass_field = py_field
    # Register user-defined dunder methods and read back system-generated ones.
    # Non-callable entries whose names are in _FFI_TYPE_ATTR_NAMES are routed
    # to TVMFFITypeRegisterAttr by the Cython layer.
    type_info._register_py_methods(py_methods, type_attr_names=_FFI_TYPE_ATTR_NAMES)
    _add_class_attrs(cls, type_info, type_attr_names=_FFI_TYPE_ATTR_NAMES)

    # Remove deferred __init__ and restore user-defined __init__ if saved.
    if "__ffi_py_class_is_deferred_init__" in cls.__dict__:
        # Always remove the deferred wrapper.
        if "__init__" in cls.__dict__:
            delattr(cls, "__init__")
        try:
            delattr(cls, "__ffi_py_class_is_deferred_init__")
        except AttributeError:
            pass
        # Restore user-defined __init__ if it was saved.
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


# ---------------------------------------------------------------------------
# Main decorator
# ---------------------------------------------------------------------------


@dataclass_transform(
    eq_default=False,
    order_default=False,
    field_specifiers=(Field, field),
    converter=_field_converter,
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
    structural_eq: str | None = "tree",
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

        - ``None``: structural comparison is not supported.
        - ``"tree"`` (default): content-based comparison, the safe default for
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
      by the ``structural_eq`` param.  Default (``"tree"``): recursive,
      field-wise structural comparison and hashing.
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

        info = _resolve_fields.register_type_without_fields(cls, effective_type_key)
        info._decorator_args = params

        try:
            resolved = _resolve_fields.resolve_type_hints_by_owner(cls, globalns)
            if resolved is not None:
                on_fields_resolved(info, resolved)
                _resolve_fields.flush_pending()
            else:
                _resolve_fields.defer_field_registration(
                    cls,
                    info,
                    globalns,
                )
        except Exception:
            # Phase-2 failed (bad annotation, field ordering, etc.).
            # Roll back phase-1 so the type key can be reused after
            # the user fixes the error.
            _resolve_fields.rollback_registration(cls, info)
            raise

        # Marker: distinguishes @c_class / @py_class types from FFI containers
        # (Array, List, Map, Dict) that also have __tvm_ffi_type_info__ but are
        # not dataclasses.  Used by is_dataclass() in common.py.
        setattr(cls, "__ffi_is_dataclass__", True)
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
