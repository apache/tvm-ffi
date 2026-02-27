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
"""FFI registry to register function and objects."""

from __future__ import annotations

import inspect
import json
import sys
from typing import Any, Callable, Literal, Sequence, TypeVar, overload

from . import core
from .core import TypeInfo

# whether we simplify skip unknown objects regtistration
_SKIP_UNKNOWN_OBJECTS = False


_T = TypeVar("_T", bound=type)


def register_object(type_key: str | None = None) -> Callable[[_T], _T]:
    """Register object type.

    Parameters
    ----------
    type_key
        The type key of the node. It requires ``type_key`` to be registered already
        on the C++ side. If not specified, the class name will be used.

    Examples
    --------
    The following code registers MyObject using type key "test.MyObject", if the
    type key is already registered on the C++ side.

    .. code-block:: python

      @tvm_ffi.register_object("test.MyObject")
      class MyObject(Object):
          pass

    """

    def _register(cls: _T, object_name: str) -> _T:
        """Register the object type with the FFI core."""
        type_index = core._object_type_key_to_index(object_name)
        if type_index is None:
            if _SKIP_UNKNOWN_OBJECTS:
                return cls
            raise ValueError(f"Cannot find object type index for {object_name}")
        info = core._register_object_by_index(type_index, cls)
        _add_class_attrs(type_cls=cls, type_info=info)
        setattr(cls, "__tvm_ffi_type_info__", info)
        return cls

    if isinstance(type_key, str):

        def _decorator_with_name(cls: _T) -> _T:
            return _register(cls, type_key)

        return _decorator_with_name

    def _decorator_default(cls: _T) -> _T:
        return _register(cls, cls.__name__)

    if type_key is None:
        return _decorator_default
    if isinstance(type_key, type):
        return _decorator_default(type_key)
    raise TypeError("type_key must be a string, type, or None")


def register_global_func(
    func_name: str | Callable[..., Any],
    f: Callable[..., Any] | None = None,
    override: bool = False,
) -> Any:
    """Register global function.

    Parameters
    ----------
    func_name
        The function name

    f
        The function to be registered.

    override
        Whether override existing entry.

    Returns
    -------
    fregister
        Register function if f is not specified.

    Examples
    --------
    .. code-block:: python

        import tvm_ffi


        # we can use decorator to register a function
        @tvm_ffi.register_global_func("mytest.echo")
        def echo(x):
            return x


        # After registering, we can get the function by its name
        f = tvm_ffi.get_global_func("mytest.echo")
        assert f(1) == 1

        # we can also directly register a function
        tvm_ffi.register_global_func("mytest.add_one", lambda x: x + 1)
        f = tvm_ffi.get_global_func("mytest.add_one")
        assert f(1) == 2

    See Also
    --------
    :py:func:`tvm_ffi.get_global_func`
    :py:func:`tvm_ffi.remove_global_func`

    """
    if not isinstance(func_name, str):
        f = func_name
        func_name = f.__name__  # ty: ignore[unresolved-attribute]

    if not isinstance(func_name, str):
        raise ValueError("expect string function name")

    def register(myf: Callable[..., Any]) -> Any:
        """Register the global function with the FFI core."""
        return core._register_global_func(func_name, myf, override)

    if f is not None:
        return register(f)
    return register


@overload
def get_global_func(name: str, allow_missing: Literal[True]) -> core.Function | None: ...


@overload
def get_global_func(name: str, allow_missing: Literal[False] = False) -> core.Function: ...


def get_global_func(name: str, allow_missing: bool = False) -> core.Function | None:
    """Get a global function by name.

    Parameters
    ----------
    name
        The name of the global function

    allow_missing
        Whether allow missing function or raise an error.

    Returns
    -------
    func
        The function to be returned, ``None`` if function is missing.

    Examples
    --------
    .. code-block:: python

        import tvm_ffi


        @tvm_ffi.register_global_func("demo.echo")
        def echo(x):
            return x


        f = tvm_ffi.get_global_func("demo.echo")
        assert f(123) == 123

    See Also
    --------
    :py:func:`tvm_ffi.register_global_func`

    """
    return core._get_global_func(name, allow_missing)


def list_global_func_names() -> list[str]:
    """Get list of global functions registered.

    Returns
    -------
    names
       List of global functions names.

    """
    name_functor = get_global_func("ffi.FunctionListGlobalNamesFunctor")()
    num_names = name_functor(-1)
    return [name_functor(i) for i in range(num_names)]


def remove_global_func(name: str) -> None:
    """Remove a global function by name.

    Parameters
    ----------
    name
        The name of the global function.

    Examples
    --------
    .. code-block:: python

        import tvm_ffi


        @tvm_ffi.register_global_func("my.temp")
        def temp():
            return 42


        assert tvm_ffi.get_global_func("my.temp", allow_missing=True) is not None
        tvm_ffi.remove_global_func("my.temp")
        assert tvm_ffi.get_global_func("my.temp", allow_missing=True) is None

    See Also
    --------
    :py:func:`tvm_ffi.register_global_func`
    :py:func:`tvm_ffi.get_global_func`

    """
    get_global_func("ffi.FunctionRemoveGlobal")(name)


def get_global_func_metadata(name: str) -> dict[str, Any]:
    """Get metadata (including type schema) for a global function.

    Parameters
    ----------
    name
        The name of the global function.

    Returns
    -------
    metadata
        A dictionary containing function metadata. The ``type_schema`` field
        encodes the callable signature.

    Examples
    --------
    .. code-block:: python

        import tvm_ffi

        meta = tvm_ffi.get_global_func_metadata("testing.add_one")
        print(meta)

    See Also
    --------
    :py:func:`tvm_ffi.get_global_func`
        Retrieve a callable for an existing global function.
    :py:func:`tvm_ffi.register_global_func`
        Register a Python callable as a global FFI function.

    """
    metadata_json = get_global_func("ffi.GetGlobalFuncMetadata")(name)
    return json.loads(metadata_json) if metadata_json else {}


def init_ffi_api(namespace: str, target_module_name: str | None = None) -> None:
    """Initialize register ffi api  functions into a given module.

    Parameters
    ----------
    namespace
       The namespace of the source registry

    target_module_name
       The target module name if different from namespace

    Examples
    --------
    A typical usage pattern is to create a _ffi_api.py file to register
    the functions under a given module. The following
    code populates all registered global functions
    prefixed with ``mypackage.`` into the current module,
    then we can call the function through ``_ffi_api.func_name(*args)``
    which will call into the registered global function "mypackage.func_name".

    .. code-block:: python

        # _ffi_api.py
        import tvm_ffi

        tvm_ffi.init_ffi_api("mypackage", __name__)

    """
    target_module_name = target_module_name if target_module_name else namespace

    if namespace.startswith("tvm."):
        prefix = namespace[4:]
    else:
        prefix = namespace

    target_module = sys.modules[target_module_name]

    for name in list_global_func_names():
        if not name.startswith(prefix):
            continue

        fname = name[len(prefix) + 1 :]
        if fname.find(".") != -1:
            continue

        f = get_global_func(name)
        setattr(f, "__name__", fname)
        setattr(target_module, fname, f)


__SENTINEL = object()


def _make_init(type_cls: type, type_info: TypeInfo) -> Callable[..., None]:
    """Build a Python ``__init__`` that delegates to the C++ auto-generated ``__ffi_init__``."""
    sig = _make_init_signature(type_info)
    kwargs_obj = core.KWARGS

    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        ffi_args: list[Any] = list(args)
        ffi_args.append(kwargs_obj)
        for key, val in kwargs.items():
            ffi_args.append(key)
            ffi_args.append(val)
        self.__ffi_init__(*ffi_args)

    __init__.__signature__ = sig  # ty: ignore[unresolved-attribute]
    __init__.__qualname__ = f"{type_cls.__qualname__}.__init__"
    __init__.__module__ = type_cls.__module__
    return __init__


def _make_init_signature(type_info: TypeInfo) -> inspect.Signature:
    """Build an ``inspect.Signature`` from reflection field metadata."""
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
            inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=__SENTINEL)
        )

    for name, _has_default in kw_required:
        params.append(inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY))

    for name, _has_default in kw_default:
        params.append(inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=__SENTINEL))

    return inspect.Signature(params)


def _add_class_attrs(type_cls: type, type_info: TypeInfo) -> type:
    for field in type_info.fields:
        name = field.name
        if not hasattr(type_cls, name):  # skip already defined attributes
            setattr(type_cls, name, field.as_property(type_cls))
    has_shallow_copy = False
    for method in type_info.methods:
        name = method.name
        if name == "__ffi_init__":
            # Always override: init is type-specific and must not be inherited
            setattr(type_cls, "__c_ffi_init__", method.as_callable(type_cls))
        elif name == "__ffi_shallow_copy__":
            has_shallow_copy = True
            # Always override: shallow copy is type-specific and must not be inherited
            setattr(type_cls, name, method.as_callable(type_cls))
        elif not hasattr(type_cls, name):
            setattr(type_cls, name, method.as_callable(type_cls))
    is_container = type_info.type_key in (
        "ffi.Array",
        "ffi.Map",
        "ffi.List",
        "ffi.Dict",
        "ffi.Shape",
    )
    _setup_copy_methods(type_cls, has_shallow_copy, is_container=is_container)
    return type_cls


def _setup_copy_methods(
    type_cls: type, has_shallow_copy: bool, *, is_container: bool = False
) -> None:
    """Set up __copy__, __deepcopy__, __replace__ based on copy support."""
    if has_shallow_copy:
        if "__copy__" not in type_cls.__dict__:
            setattr(type_cls, "__copy__", _copy_supported)
        if "__deepcopy__" not in type_cls.__dict__:
            setattr(type_cls, "__deepcopy__", _deepcopy_supported)
        if "__replace__" not in type_cls.__dict__:
            setattr(type_cls, "__replace__", _replace_supported)
    else:
        if "__copy__" not in type_cls.__dict__:
            setattr(type_cls, "__copy__", _copy_unsupported)
        if "__deepcopy__" not in type_cls.__dict__:
            # Containers (Array, Map) support deepcopy via ffi.DeepCopy
            # even without __ffi_shallow_copy__
            if is_container:
                setattr(type_cls, "__deepcopy__", _deepcopy_supported)
            else:
                setattr(type_cls, "__deepcopy__", _deepcopy_unsupported)
        if "__replace__" not in type_cls.__dict__:
            setattr(type_cls, "__replace__", _replace_unsupported)


def _install_init(cls: type, *, enabled: bool) -> None:
    """Install ``__init__`` from C++ reflection metadata, or a guard.

    When *enabled* is True, looks for a ``__ffi_init__`` method in the
    type's C++ reflection metadata.  If the method has ``auto_init=True``
    metadata (set by ``refl::init()`` in C++), a Python ``__init__`` is
    synthesized with an ``inspect.Signature`` derived from the field
    metadata (respecting ``Init()``, ``KwOnly()``, ``Default()`` traits).
    Otherwise the raw ``__ffi_init__`` is exposed as ``__init__`` directly.

    When *enabled* is False, installs a guard that raises ``TypeError``
    on construction.  Skipped entirely if the class body already defines
    ``__init__``.
    """
    if "__init__" in cls.__dict__:
        return
    type_info: TypeInfo | None = getattr(cls, "__tvm_ffi_type_info__", None)
    if type_info is None:
        return
    if enabled:
        ffi_init_method = next((m for m in type_info.methods if m.name == "__ffi_init__"), None)
        if ffi_init_method is not None:
            if ffi_init_method.metadata.get("auto_init", False):
                setattr(cls, "__init__", _make_init(cls, type_info))
            else:
                setattr(cls, "__init__", getattr(cls, "__ffi_init__"))
            return
        if issubclass(cls, core.PyNativeObject):
            return
        msg = (
            f"`{cls.__name__}` (C++ type `{type_info.type_key}`) has no __ffi_init__ "
            f"registered. Either add `refl::init()` to its C++ ObjectDef, "
            f"or pass `init=False` to @c_class."
        )
    else:
        msg = (
            f"`{cls.__name__}` cannot be constructed directly. "
            f"Define a custom __init__ or use a factory method."
        )

    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        raise TypeError(msg)

    __init__.__qualname__ = f"{cls.__qualname__}.__init__"
    __init__.__module__ = cls.__module__
    setattr(cls, "__init__", __init__)


def _copy_supported(self: Any) -> Any:
    return self.__ffi_shallow_copy__()


def _deepcopy_supported(self: Any, memo: Any = None) -> Any:
    from . import _ffi_api  # noqa: PLC0415

    return _ffi_api.DeepCopy(self)


def _replace_supported(self: Any, **kwargs: Any) -> Any:
    import copy  # noqa: PLC0415

    obj = copy.copy(self)
    for key, value in kwargs.items():
        setattr(obj, key, value)
    return obj


def _copy_unsupported(self: Any) -> Any:
    raise TypeError(
        f"Type `{type(self).__name__}` does not support copy. "
        f"The underlying C++ type is not copy-constructible."
    )


def _deepcopy_unsupported(self: Any, memo: Any = None) -> Any:
    raise TypeError(
        f"Type `{type(self).__name__}` does not support deepcopy. "
        f"The underlying C++ type is not copy-constructible."
    )


def _replace_unsupported(self: Any, **kwargs: Any) -> Any:
    raise TypeError(
        f"Type `{type(self).__name__}` does not support replace. "
        f"The underlying C++ type is not copy-constructible."
    )


def _install_dataclass_dunders(
    cls: type,
    *,
    init: bool,
    repr: bool,
    eq: bool,
    order: bool,
    unsafe_hash: bool,
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
        If True, install ``__init__`` from C++ reflection metadata via
        :func:`_install_init`.
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

    """
    _install_init(cls, enabled=init)

    if repr and "__repr__" not in cls.__dict__:
        from .core import object_repr  # noqa: PLC0415

        cls.__repr__ = object_repr  # type: ignore[attr-defined]

    from . import _ffi_api  # noqa: PLC0415

    def _is_comparable(self: Any, other: Any) -> bool:
        """Return True if *self* and *other* share a type hierarchy."""
        return isinstance(other, type(self)) or isinstance(self, type(other))

    dunders: dict[str, Any] = {}

    if eq:
        recursive_eq = _ffi_api.RecursiveEq

        def __eq__(self: Any, other: Any) -> bool:
            if not _is_comparable(self, other):
                return NotImplemented
            return recursive_eq(self, other)

        def __ne__(self: Any, other: Any) -> bool:
            if not _is_comparable(self, other):
                return NotImplemented
            return not recursive_eq(self, other)

        dunders["__eq__"] = __eq__
        dunders["__ne__"] = __ne__

    if unsafe_hash:
        recursive_hash = _ffi_api.RecursiveHash

        def __hash__(self: Any) -> int:
            return recursive_hash(self)

        dunders["__hash__"] = __hash__

    if order:
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

        dunders["__lt__"] = __lt__
        dunders["__le__"] = __le__
        dunders["__gt__"] = __gt__
        dunders["__ge__"] = __ge__

    for name, impl in dunders.items():
        if name not in cls.__dict__:
            setattr(cls, name, impl)


def get_registered_type_keys() -> Sequence[str]:
    """Get the list of valid type keys registered to TVM-FFI.

    Returns
    -------
    type_keys
        List of valid type keys.

    """
    return get_global_func("ffi.GetRegisteredTypeKeys")()


__all__ = [
    "get_global_func",
    "get_global_func_metadata",
    "get_registered_type_keys",
    "init_ffi_api",
    "list_global_func_names",
    "register_global_func",
    "register_object",
    "remove_global_func",
]
