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
    if callable(func_name):
        f = func_name
        func_name = f.__name__

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
    return json.loads(get_global_func("ffi.GetGlobalFuncMetadata")(name) or "{}")


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


def _add_class_attrs(type_cls: type, type_info: TypeInfo) -> type:
    for field in type_info.fields:
        name = field.name
        if not hasattr(type_cls, name):  # skip already defined attributes
            setattr(type_cls, name, field.as_property(type_cls))
    has_c_init = False
    for method in type_info.methods:
        name = method.name
        if name == "__ffi_init__":
            name = "__c_ffi_init__"
            has_c_init = True
        if not hasattr(type_cls, name):
            setattr(type_cls, name, method.as_callable(type_cls))
    if "__init__" not in type_cls.__dict__:
        if has_c_init:
            setattr(type_cls, "__init__", getattr(type_cls, "__ffi_init__"))
        elif not issubclass(type_cls, core.PyNativeObject):
            setattr(type_cls, "__init__", __init__invalid)
    return type_cls


def __init__invalid(self: Any, *args: Any, **kwargs: Any) -> None:
    raise RuntimeError("The __init__ method of this class is not implemented.")


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
