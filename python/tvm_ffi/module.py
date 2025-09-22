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

from enum import IntEnum
from typing import Any

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

    """

    # constant for entry function name
    entry_name = "main"

    @property
    def kind(self) -> str:
        """Get type key of the module."""
        return _ffi_api.ModuleGetKind(self)

    @property
    def imports(self) -> list["Module"]:
        """Get imported modules.

        Returns
        -------
        modules : list of Module
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
        name : str
            The name of the function

        query_imports : bool
            Whether to also query modules imported by this module.

        Returns
        -------
        b : Bool
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
        name : str
            The name of the function

        query_imports : bool
            Whether also query modules imported by this module.

        Returns
        -------
        f : tvm_ffi.Function
            The result function.

        """
        func = _ffi_api.ModuleGetFunction(self, name, query_imports)
        if func is None:
            raise AttributeError(f"Module has no function '{name}'")
        return func

    def import_module(self, module: "Module") -> None:
        """Add module to the import list of current one.

        Parameters
        ----------
        module : tvm.runtime.Module
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
        fmt : str, optional
            The specified format.

        Returns
        -------
        source : str
            The result source code.

        """
        return _ffi_api.ModuleInspectSource(self, fmt)

    def get_write_formats(self) -> list[str]:
        """Get the format of the module."""
        return _ffi_api.ModuleGetWriteFormats(self)

    def get_property_mask(self) -> int:
        """Get the runtime module property mask. The mapping is stated in ModulePropertyMask.

        Returns
        -------
        mask : int
            Bitmask of runtime module property

        """
        return _ffi_api.ModuleGetPropertyMask(self)

    def is_binary_serializable(self) -> bool:
        """Return whether the module is binary serializable (supports save_to_bytes).

        Returns
        -------
        b : Bool
            True if the module is binary serializable.

        """
        return (self.get_property_mask() & ModulePropertyMask.BINARY_SERIALIZABLE) != 0

    def is_runnable(self) -> bool:
        """Return whether the module is runnable (supports get_function).

        Returns
        -------
        b : Bool
            True if the module is runnable.

        """
        return (self.get_property_mask() & ModulePropertyMask.RUNNABLE) != 0

    def is_compilation_exportable(self) -> bool:
        """Return whether the module is compilation exportable.

        write_to_file is supported for object or source.

        Returns
        -------
        b : Bool
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
        file_name : str
            The name of the file.
        fmt : str
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
    symbol_prefix: str = ""
        Optional symbol prefix that can be used for search. When we lookup a symbol
        symbol_prefix + name will first be searched, then the name without symbol_prefix.

    Returns
    -------
    module : runtime.Module
        The system-wide library module.

    """
    return _ffi_api.SystemLib(symbol_prefix)


def load_module(path: str) -> Module:
    """Load module from file.

    Parameters
    ----------
    path : str
        The path to the module file.

    Returns
    -------
    module : :py:class:`tvm_ffi.Module`
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
