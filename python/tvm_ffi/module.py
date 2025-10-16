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

        Note
        ----
        that has_function(name) does not imply get_function(name) is non-null since the module
        that has_function(name) does not imply get_function(name) is non-null since the module
        may be, eg, a CSourceModule which cannot supply a packed-func implementation of the function
        without further compilation. However, get_function(name) non null should always imply
        has_function(name).

        Parameters
        ----------
        name
            The name of the function

        query_imports
            Whether to also query modules imported by this module.

        Returns
        -------
        b
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
            Whether also query modules imported by this module.

        Returns
        -------
        f
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
        source
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
        mask
            Bitmask of runtime module property

        """
        return _ffi_api.ModuleGetPropertyMask(self)

    def is_binary_serializable(self) -> bool:
        """Return whether the module is binary serializable (supports save_to_bytes).

        Returns
        -------
        b
            True if the module is binary serializable.

        """
        return (self.get_property_mask() & ModulePropertyMask.BINARY_SERIALIZABLE) != 0

    def is_runnable(self) -> bool:
        """Return whether the module is runnable (supports get_function).

        Returns
        -------
        b
            True if the module is runnable.

        """
        return (self.get_property_mask() & ModulePropertyMask.RUNNABLE) != 0

    def is_compilation_exportable(self) -> bool:
        """Return whether the module is compilation exportable.

        write_to_file is supported for object or source.

        Returns
        -------
        b
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
        runtime.Module.export_library : export the module to shared library.

        """
        _ffi_api.ModuleWriteToFile(self, file_name, fmt)


def system_lib(symbol_prefix: str = "") -> Module:
    """Get system-wide library module singleton.

    System lib is a global module that contains self register functions in startup.
    Unlike normal dso modules which need to be loaded explicitly.
    It is useful in environments where dynamic loading api like dlopen is banned.

    The system lib is intended to be linked and loaded during the entire life-cyle of the program.
    If you want dynamic loading features, use dso modules instead.

    Parameters
    ----------
    symbol_prefix
        Optional symbol prefix that can be used for search. When we lookup a symbol
        symbol_prefix + name will first be searched, then the name without symbol_prefix.

    Returns
    -------
    module
        The system-wide library module.

    """
    return _ffi_api.SystemLib(symbol_prefix)


def load_module(path: str) -> Module:
    """Load module from file.

    Parameters
    ----------
    path
        The path to the module file.

    Returns
    -------
    module
        The loaded module

    Examples
    --------
    .. code-block:: python

      mod = tvm_ffi.load_module("path/to/module.so")
      mod.func_name(*args)

    See Also
    --------
    :py:class:`tvm_ffi.Module`

    """
    return _ffi_api.ModuleLoadFromFile(path)
