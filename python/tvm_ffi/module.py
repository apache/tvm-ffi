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
"""Module related objects and functions."""

# pylint: disable=invalid-name
from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum
from os import PathLike, fspath
from typing import TYPE_CHECKING, Any, ClassVar, cast

from . import _ffi_api, core
from .registry import register_object

__all__ = ["Module", "ModulePropertyMask", "load_module", "system_lib"]


class ModulePropertyMask(IntEnum):
    """Runtime Module Property Mask."""

    BINARY_SERIALIZABLE = 0b001
    RUNNABLE = 0b010
    COMPILATION_EXPORTABLE = 0b100


@register_object("ffi.Module")
class Module(core.Object):
    """Module container for dynamically loaded Module.

    Examples
    --------
    .. code-block:: python

        import tvm_ffi

        # load the module from a tvm-ffi shared library
        mod : tvm_ffi.Module = tvm_ffi.load_module("path/to/library.so")
        # you can use mod.func_name to call the exported function
        mod.func_name(*args)

    See Also
    --------
    :py:func:`tvm_ffi.load_module`

    Notes
    -----
    If you load a module within a local scope, be careful when any called function
    creates and returns an object. The memory deallocation routines are part of
    the library's code. If the module is unloaded before the object is destroyed,
    the deleter may call an invalid address. Keep the module loaded until all returned
    objects are deleted. You can safely use returned objects inside a nested function
    that finishes before the module goes out of scope. When possible, consider keeping
    the module alive in a long-lived/global scope (for example, in a global state) to
    avoid premature unloading.

    .. code-block:: python

        def bad_pattern(x):
            # Bad: unload order of `tensor` and `mod` is not guaranteed
            mod = tvm_ffi.load_module("path/to/library.so")
            # ... do something with the tensor
            tensor = mod.func_create_and_return_tensor(x)

        def good_pattern(x):
            # Good: `tensor` is freed before `mod` goes out of scope
            mod = tvm_ffi.load_module("path/to/library.so")
            def run_some_tests():
                tensor = mod.func_create_and_return_tensor(x)
                # ... do something with the tensor
            run_some_tests()

    """

    # tvm-ffi-stubgen(begin): object/ffi.Module
    if TYPE_CHECKING:
        # fmt: off
        imports_: Sequence[Any]
        # fmt: on
    # tvm-ffi-stubgen(end)

    entry_name: ClassVar[str] = "main"  # constant for entry function name

    @property
    def kind(self) -> str:
        """Get type key of the module."""
        return _ffi_api.ModuleGetKind(self)

    @property
    def imports(self) -> list[Module]:
        """Get imported modules.

        Returns
        -------
        modules
            The module

        """
        return self.imports_  # type: ignore[return-value]

    def implements_function(self, name: str, query_imports: bool = False) -> bool:
        """Return True if the module defines a global function.

        Notes
        -----
        ``implements_function(name)`` does not guarantee that
        ``get_function(name)`` will return a callable, because some module
        kinds (e.g. a source-only module) may not provide a packed function
        implementation until further compilation occurs. However, a non-null
        result from ``get_function(name)`` should imply the module implements
        the function.

        Parameters
        ----------
        name
            The name of the function

        query_imports
            Whether to also query modules imported by this module.

        Returns
        -------
        True if module (or one of its imports) has a definition for name.

        """
        return _ffi_api.ModuleImplementsFunction(self, name, query_imports)

    def __getattr__(self, name: str) -> core.Function:
        """Accessor to allow getting functions as attributes."""
        try:
            func = self.get_function(name)
            self.__dict__[name] = func
            return func
        except AttributeError:
            raise AttributeError(f"Module has no function '{name}'")

    def get_function(self, name: str, query_imports: bool = False) -> core.Function:
        """Get function from the module.

        Parameters
        ----------
        name
            The name of the function

        query_imports
            Whether to also query modules imported by this module.

        Returns
        -------
        The result function.

        """
        func = _ffi_api.ModuleGetFunction(self, name, query_imports)
        func = cast(core.Function, func)
        if func is None:
            raise AttributeError(f"Module has no function '{name}'")
        return func

    def import_module(self, module: Module) -> None:
        """Add module to the import list of current one.

        Parameters
        ----------
        module
            The other module.

        """
        _ffi_api.ModuleImportModule(self, module)

    def __getitem__(self, name: str) -> core.Function:
        """Return function by name using item access (module["func"])."""
        if not isinstance(name, str):
            raise ValueError("Can only take string as function name")
        return self.get_function(name)

    def __call__(self, *args: Any) -> Any:
        """Call the module's entry function (`main`)."""
        # pylint: disable=not-callable
        return self.main(*args)

    def inspect_source(self, fmt: str = "") -> str:
        """Get source code from module, if available.

        Parameters
        ----------
        fmt
            The specified format.

        Returns
        -------
        The result source code.

        """
        return _ffi_api.ModuleInspectSource(self, fmt)

    def get_write_formats(self) -> Sequence[str]:
        """Get the format of the module."""
        return _ffi_api.ModuleGetWriteFormats(self)

    def get_property_mask(self) -> int:
        """Get the runtime module property mask. The mapping is stated in ModulePropertyMask.

        Returns
        -------
        Bitmask of runtime module property

        """
        return _ffi_api.ModuleGetPropertyMask(self)

    def is_binary_serializable(self) -> bool:
        """Return whether the module is binary serializable (supports save_to_bytes).

        Returns
        -------
        True if the module is binary serializable.

        """
        return (self.get_property_mask() & ModulePropertyMask.BINARY_SERIALIZABLE) != 0

    def is_runnable(self) -> bool:
        """Return whether the module is runnable (supports get_function).

        Returns
        -------
        True if the module is runnable.

        """
        return (self.get_property_mask() & ModulePropertyMask.RUNNABLE) != 0

    def is_compilation_exportable(self) -> bool:
        """Return whether the module is compilation exportable.

        write_to_file is supported for object or source.

        Returns
        -------
        True if the module is compilation exportable.

        """
        return (self.get_property_mask() & ModulePropertyMask.COMPILATION_EXPORTABLE) != 0

    def clear_imports(self) -> None:
        """Remove all imports of the module."""
        _ffi_api.ModuleClearImports(self)

    def write_to_file(self, file_name: str, fmt: str = "") -> None:
        """Write the current module to file.

        Parameters
        ----------
        file_name
            The name of the file.
        fmt
            The format of the file.

        See Also
        --------
        tvm.runtime.Module.export_library : export the module to shared library.

        """
        _ffi_api.ModuleWriteToFile(self, file_name, fmt)


def system_lib(symbol_prefix: str = "") -> Module:
    """Get system-wide library module singleton with functions prefixed by ``__tvm_ffi_{symbol_prefix}``.

    The library module contains symbols that are registered via :cpp:func:`TVMFFIEnvModRegisterSystemLibSymbol`.

    .. note::
        The system lib is intended to be statically linked and loaded during the entire lifecycle of the program.
        If you want dynamic loading features, use DSO modules instead.

    Parameters
    ----------
    symbol_prefix
        Optional symbol prefix that can be used for search. When we lookup a symbol
        symbol_prefix + name will first be searched, then the name without symbol_prefix.

    Returns
    -------
    The system-wide library module.

    Examples
    --------
    Register the function ``test_symbol_add_one`` in C++ with the name ``__tvm_ffi_test_symbol_add_one``
    via :cpp:func:`TVMFFIEnvModRegisterSystemLibSymbol`.

    .. code-block:: cpp

        // A function to be registered in the system lib
        static int test_symbol_add_one(void*, const TVMFFIAny* args, int32_t num_args, TVMFFIAny* ret) {
          TVM_FFI_SAFE_CALL_BEGIN();
          TVM_FFI_CHECK(num_args == 1, "Expected 1 argument, but got: " + std::to_string(num_args));
          int64_t x = reinterpret_cast<const AnyView*>(args)[0].cast<int64_t>();
          reinterpret_cast<Any*>(ret)[0] = x + 1;
          TVM_FFI_SAFE_CALL_END();
        }

        // Register the function with name `test_symbol_add_one` prefixed by `__tvm_ffi_`
        int _ = TVMFFIEnvModRegisterSystemLibSymbol("__tvm_ffi_testing.add_one", reinterpret_cast<void*>(test_symbol_add_one));

    Look up and call the function from Python:

    .. code-block:: python

        import tvm_ffi

        mod: tvm_ffi.Module = tvm_ffi.system_lib("testing.")  # symbols prefixed with `__tvm_ffi_testing.`
        func: tvm_ffi.Function = mod["add_one"]  # looks up `__tvm_ffi_testing.add_one`
        assert func(10) == 11

    """
    return _ffi_api.SystemLib(symbol_prefix)


def load_module(path: str | PathLike) -> Module:
    """Load module from file.

    Parameters
    ----------
    path
        The path to the module file.

    Returns
    -------
    The loaded module

    Examples
    --------
    .. code-block:: python

        import tvm_ffi
        from pathlib import Path

        # Works with string paths
        mod = tvm_ffi.load_module("path/to/module.so")
        # Also works with pathlib.Path objects
        mod = tvm_ffi.load_module(Path("path/to/module.so"))

        mod.func_name(*args)

    See Also
    --------
    :py:class:`tvm_ffi.Module`

    """
    path = fspath(path)
    return _ffi_api.ModuleLoadFromFile(path)
