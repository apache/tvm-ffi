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
"""Utilities to locate tvm_ffi libraries, headers, and helper include paths.

This module also provides helpers to locate and load platform-specific shared
libraries by a target_name (e.g., ``tvm_ffi`` -> ``libtvm_ffi.so`` on Linux).
"""

from __future__ import annotations

import ctypes
import importlib.metadata as im
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .module import Module


def find_libtvm_ffi() -> str:
    """Find libtvm_ffi.

    Returns
    -------
    path
        The full path to the located library.

    """
    candidate = _find_library_by_basename("apache-tvm-ffi", "tvm_ffi")
    if ret := _resolve_and_validate([candidate], cond=lambda _: True):
        return ret
    raise RuntimeError("Cannot find libtvm_ffi")


def find_windows_implib() -> str:
    """Find and return the Windows import library path for tvm_ffi.lib."""
    # implib = _find_library_by_basename("apache-tvm-ffi", "tvm_ffi").parent / "tvm_ffi.lib"
    # ret = _resolve_to_str(implib)
    candidate = _find_library_by_basename("apache-tvm-ffi", "tvm_ffi").parent / "tvm_ffi.lib"
    if ret := _resolve_and_validate([candidate], cond=lambda _: True):
        return ret
    raise RuntimeError("Cannot find implib tvm_ffi.lib")


def find_source_path() -> str:
    """Find packaged source home path."""
    if ret := _resolve_and_validate(
        paths=[
            _rel_top_directory(),
            _dev_top_directory(),
        ],
        cond=lambda p: (p / "src").is_dir(),
    ):
        return ret
    raise RuntimeError("Cannot find home path.")


def find_cmake_path() -> str:
    """Find the preferred cmake path."""
    if ret := _resolve_and_validate(
        paths=[
            _rel_top_directory() / "share" / "cmake" / "tvm_ffi",  # Standard install
            _dev_top_directory() / "cmake",  # Development mode
        ],
        cond=lambda p: p.is_dir(),
    ):
        return ret
    raise RuntimeError("Cannot find cmake path.")


def find_include_path() -> str:
    """Find header files for C compilation."""
    if ret := _resolve_and_validate(
        paths=[
            _rel_top_directory() / "include",
            _dev_top_directory() / "include",
        ],
        cond=lambda p: p.is_dir(),
    ):
        return ret
    raise RuntimeError("Cannot find include path.")


def find_dlpack_include_path() -> str:
    """Find dlpack header files for C compilation."""
    if ret := _resolve_and_validate(
        paths=[
            _rel_top_directory() / "include",
            _dev_top_directory() / "3rdparty" / "dlpack" / "include",
        ],
        cond=lambda p: (p / "dlpack").is_dir(),
    ):
        return ret
    raise RuntimeError("Cannot find dlpack include path.")


def find_cython_lib() -> str:
    """Find the path to tvm cython."""
    from tvm_ffi import core  # noqa: PLC0415

    try:
        return str(Path(core.__file__).resolve())
    except OSError:
        pass
    raise RuntimeError("Cannot find tvm cython path.")


def find_python_helper_include_path() -> str:
    """Find header files for C compilation."""
    if ret := _resolve_and_validate(
        paths=[
            _rel_top_directory() / "include",
            _dev_top_directory() / "python" / "tvm_ffi" / "cython",
        ],
        cond=lambda p: (p / "tvm_ffi_python_helpers.h").is_file(),
    ):
        return ret
    raise RuntimeError("Cannot find python helper include path.")


def include_paths() -> list[str]:
    """Find all include paths needed for FFI related compilation."""
    return sorted(
        {
            find_include_path(),
            find_dlpack_include_path(),
            find_python_helper_include_path(),
        }
    )


def load_lib_ctypes(package: str, target_name: str, mode: str) -> ctypes.CDLL:
    """Load the tvm_ffi shared library by searching likely paths.

    Parameters
    ----------
    package
        The package name where the library is expected to be found. For example,
        ``"apache-tvm-ffi"`` is the package name of `tvm-ffi`.
    target_name
        Name of the CMake target, e.g., ``"tvm_ffi"``. It is used to derive the platform-specific
        shared library name, e.g., ``"libtvm_ffi.so"`` on Linux, ``"tvm_ffi.dll"`` on Windows.
    mode
        The mode to load the shared library. See `ctypes.${MODE}` for details.
        Usually it is either ``"RTLD_LOCAL"`` or ``"RTLD_GLOBAL"``.

    Returns
    -------
    The loaded shared library.

    """
    lib_path: Path = _find_library_by_basename(package, target_name)
    # The dll search path need to be added explicitly in windows
    if sys.platform.startswith("win32"):
        os.add_dll_directory(str(lib_path.parent))
    return ctypes.CDLL(str(lib_path), getattr(ctypes, mode))


def load_lib_module(package: str, target_name: str, keep_module_alive: bool = True) -> Module:
    """Load the tvm_ffi shared library by searching likely paths.

    Parameters
    ----------
    package
        The package name where the library is expected to be found. For example,
        ``"apache-tvm-ffi"`` is the package name of `tvm-ffi`.

    target_name
        Name of the CMake target, e.g., ``"tvm_ffi"``. It is used to derive the platform-specific
        shared library name, e.g., ``"libtvm_ffi.so"`` on Linux, ``"tvm_ffi.dll"`` on Windows.

    keep_module_alive
        Whether to keep the loaded module alive to prevent it from being unloaded.

    Returns
    -------
    The loaded shared library.

    """
    from .module import load_module  # noqa: PLC0415

    lib_path: Path = _find_library_by_basename(package, target_name)
    # The dll search path need to be added explicitly in windows
    if sys.platform.startswith("win32"):
        os.add_dll_directory(str(lib_path.parent))
    return load_module(lib_path, keep_module_alive=keep_module_alive)


def _find_library_by_basename(package: str, target_name: str) -> Path:  # noqa: PLR0912
    """Find a shared library by target_name name across known directories.

    Parameters
    ----------
    package
        The package name where the library is expected to be found.
    target_name
        Base name (e.g., ``"tvm_ffi"`` or ``"tvm_ffi_testing"``).

    Returns
    -------
    path
        The full path to the located library.

    Raises
    ------
    RuntimeError
        If the library cannot be found in any of the candidate directories.

    """
    if sys.platform.startswith("win32"):
        lib_dll_names = (f"{target_name}.dll",)
    elif sys.platform.startswith("darwin"):
        # Prefer dylib, also allow .so for some toolchains
        lib_dll_names = (f"lib{target_name}.dylib", f"lib{target_name}.so")
    else:  # Linux, FreeBSD, etc
        lib_dll_names = (f"lib{target_name}.so",)

    # Use `importlib.metadata` is the most reliable way to find package data files
    dist: im.PathDistribution = im.distribution(package)  # type: ignore[assignment]
    record = dist.read_text("RECORD") or ""
    for line in record.splitlines():
        partial_path, *_ = line.split(",")
        if partial_path.endswith(lib_dll_names):
            try:
                path = (dist._path.parent / partial_path).resolve()
            except OSError:
                continue
            if path.name in lib_dll_names:
                return path

    # **Fallback**. it's possible that the library is not built as part of Python ecosystem,
    # e.g. Use PYTHONPATH to point to dev package, and CMake + Makefiles to build the shared library.
    dll_paths: list[Path] = []

    # Case 1. It is under $PROJECT_ROOT/build/lib/ or $PROJECT_ROOT/lib/
    dll_paths.append(_rel_top_directory() / "build" / "lib")
    dll_paths.append(_rel_top_directory() / "lib")
    dll_paths.append(_dev_top_directory() / "build" / "lib")
    dll_paths.append(_dev_top_directory() / "lib")

    # Case 2. It is specified in PATH-related environment variables
    if sys.platform.startswith("win32"):
        dll_paths.extend(Path(p) for p in _split_env_var("PATH", ";"))
    elif sys.platform.startswith("darwin"):
        dll_paths.extend(Path(p) for p in _split_env_var("DYLD_LIBRARY_PATH", ":"))
        dll_paths.extend(Path(p) for p in _split_env_var("PATH", ":"))
    else:
        dll_paths.extend(Path(p) for p in _split_env_var("LD_LIBRARY_PATH", ":"))
        dll_paths.extend(Path(p) for p in _split_env_var("PATH", ":"))

    # Search for the library in candidate directories
    for dll_dir in dll_paths:
        for lib_dll_name in lib_dll_names:
            try:
                path = (dll_dir / lib_dll_name).resolve()
                if path.is_file():
                    return path
            except OSError:
                continue
    raise RuntimeError(f"Cannot find library: {', '.join(lib_dll_names)}")


def _split_env_var(env_var: str, split: str) -> list[str]:
    """Split an environment variable string.

    Parameters
    ----------
    env_var
        Name of environment variable.

    split
        String to split env_var on.

    Returns
    -------
    splits
        If env_var exists, split env_var. Otherwise, empty list.

    """
    if os.environ.get(env_var, None):
        return [p.strip() for p in os.environ[env_var].split(split)]
    return []


def _rel_top_directory() -> Path:
    """Get the current directory of this file."""
    return Path(__file__).parent


def _dev_top_directory() -> Path:
    """Get the top-level development directory."""
    return _rel_top_directory() / ".." / ".."


def _resolve_and_validate(
    paths: list[Path],
    cond: Callable[[Path], bool | Path],
) -> str | None:
    """For all paths that resolve properly, find the 1st one that meets the specified condition.

    M. B. This code path gracefully handles broken paths, symlinks, or permission issues,
    and is required for robust library discovery in all public APIs in this file.
    """
    for path in paths:
        try:
            resolved = path.resolve()
            ret = cond(resolved)
        except (OSError, AssertionError):
            continue
        if isinstance(ret, Path):
            return str(ret)
        elif ret is True:
            return str(resolved)
    return None
