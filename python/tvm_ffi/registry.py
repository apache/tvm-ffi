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

import json
import sys
import warnings
from typing import Any, Callable, Literal, Sequence, TypeVar, overload

from . import core
from .core import Function, TypeInfo

# whether we simplify skip unknown objects regtistration
_SKIP_UNKNOWN_OBJECTS = False


_T = TypeVar("_T", bound=type)


def register_object(
    type_key: str | None = None,
    *,
    init: bool = True,
) -> Callable[[_T], _T]:
    """Register object type.

    Parameters
    ----------
    type_key
        The type key of the node. It requires ``type_key`` to be registered already
        on the C++ side. If not specified, the class name will be used.
    init
        If True (default), install ``__init__`` from the C++ ``__ffi_init__``
        TypeAttrColumn when available, or a TypeError guard for ``Object``
        subclasses that lack one.  Set to False when a subsequent decorator
        (e.g. ``@c_class``) will handle ``__init__`` installation.

    Notes
    -----
    All :class:`Object` subclasses derive ``__slots__`` from their
    annotations via the metaclass (or use ``__slots__ = ()`` when they
    have no annotated fields), preventing a per-instance ``__dict__`` by
    default. To opt out and allow arbitrary instance attributes, declare
    ``__slots__ = ("__dict__",)`` explicitly in the class body::

        @tvm_ffi.register_object("test.MyObject")
        class MyObject(Object):
            __slots__ = ("__dict__",)

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
        if init:
            _install_init(cls, info)
        # Field helpers (_force_set_field, __replace__) are installed
        # after __init__.  For init=False (used by c_class internally),
        # c_class installs field helpers itself after _install_dataclass_dunders.
        if init and info.fields:
            core._install_field_helpers(cls, info)
            core._install_field_getattro(cls, info)
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


def _install_init(cls: type, type_info: TypeInfo) -> None:
    """Install ``__init__`` from ``__ffi_init__`` TypeMethod or TypeAttrColumn.

    Skipped if the class body already defines ``__init__``.
    This ensures that ``register_object`` alone provides a working
    constructor, maintaining the invariant that ``c_class`` is a full
    alias of ``register_object`` + dunder installation.

    When no ``__ffi_init__`` is available and the class is an ``Object``
    subclass, a TypeError guard is installed to prevent segfaults from
    uninitialised handles.
    """
    if "__init__" in cls.__dict__:
        return
    # Look up __ffi_init__ from TypeMethod (preferred) or TypeAttrColumn (fallback).
    ffi_init = None
    for method in type_info.methods:
        if method.name == "__ffi_init__":
            ffi_init = method.func
            break
    if ffi_init is None:
        ffi_init = core._lookup_type_attr(type_info.type_index, "__ffi_init__")
    if ffi_init is not None:
        from ._dunder import _make_init  # noqa: PLC0415

        cls.__init__ = _make_init(  # type: ignore[attr-defined]
            cls,
            type_info,
            ffi_init=ffi_init,
        )
    elif issubclass(cls, core.Object):
        type_name = cls.__name__

        def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
            raise TypeError(
                f"`{type_name}` cannot be constructed directly. "
                f"Define a custom __init__ or use a factory method."
            )

        __init__.__qualname__ = f"{cls.__qualname__}.__init__"
        __init__.__module__ = cls.__module__
        cls.__init__ = __init__  # type: ignore[attr-defined]


def _add_class_attrs(type_cls: type, type_info: TypeInfo) -> type:
    for field in type_info.fields:
        # Install Python property (overwrites member_descriptor from __slots__).
        # Using exactly ``property`` (not a subclass) triggers CPython 3.12+
        # LOAD_ATTR_PROPERTY inline-cache specialization.
        setattr(type_cls, field.name, field.as_property(type_cls))
    has_ffi_init = False
    for method in type_info.methods:
        name = method.name
        if name == "__ffi_init__":
            _install_ffi_init_attr(type_cls, type_info, method.func)
            has_ffi_init = True
            continue
        if not hasattr(type_cls, name):
            setattr(type_cls, name, method.as_callable(type_cls))
    # Also check TypeAttrColumn for auto-generated __ffi_init__.
    if not has_ffi_init:
        ffi_init = core._lookup_type_attr(type_info.type_index, "__ffi_init__")
        if ffi_init is not None:
            _install_ffi_init_attr(type_cls, type_info, ffi_init)
    return type_cls


def _install_ffi_init_attr(cls: type, type_info: TypeInfo, ffi_init: Function) -> None:
    """Install ``__ffi_init__`` as a method that delegates to ``__init_handle_by_constructor__``.

    Custom ``__init__`` methods call ``self.__ffi_init__(*args, **kwargs)`` to
    construct the underlying C++ object. This installs a wrapper that translates
    that call into ``self.__init_handle_by_constructor__(ffi_init, *ffi_args)``
    with kwargs packed using the FFI KWARGS protocol.

    The wrapper includes a type-owner guard (same as ``_make_init``) to prevent
    subclasses from accidentally using a parent's ``__ffi_init__``.
    """
    kwargs_obj = core.KWARGS
    missing = core.MISSING
    type_name = cls.__name__

    def __ffi_init__(self: Any, *args: Any, **kwargs: Any) -> None:
        if type_info is not type(self).__tvm_ffi_type_info__:
            raise TypeError(
                f"Calling `{type_name}.__ffi_init__()` on a `{type(self).__name__}` "
                f"instance is not supported. Define `{type(self).__name__}` with init=True."
            )
        ffi_args: list[Any] = list(args)
        if kwargs:
            ffi_args.append(kwargs_obj)
            for key, val in kwargs.items():
                if val is not missing:
                    ffi_args.append(key)
                    ffi_args.append(val)
        self.__init_handle_by_constructor__(ffi_init, *ffi_args)

    __ffi_init__.__qualname__ = f"{cls.__qualname__}.__ffi_init__"
    __ffi_init__.__module__ = cls.__module__
    cls.__ffi_init__ = __ffi_init__  # type: ignore[attr-defined]


def _warn_missing_field_annotations(cls: type, type_info: TypeInfo, *, stacklevel: int) -> None:
    """Emit a warning if any C++ reflected fields lack Python annotations on *cls*.

    Only checks fields owned by *type_info* (not inherited from parents).
    Only checks annotations defined directly on *cls* (``cls.__dict__``),
    so parent annotations do not suppress warnings for child-level fields.
    """
    reflected_names = {field.name for field in type_info.fields}
    if not reflected_names:
        return
    own_annotations = cls.__dict__.get("__annotations__", {})
    missing = sorted(reflected_names - set(own_annotations))
    if missing:
        missing_str = ", ".join(missing)
        warnings.warn(
            f"@c_class({type_info.type_key!r}): class `{cls.__qualname__}` does not "
            f"annotate the following reflected field(s): {missing_str}. "
            f"Add type annotations (e.g. `field_name: type`) to the class body "
            f"for IDE support and documentation.",
            UserWarning,
            stacklevel=stacklevel,
        )


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
