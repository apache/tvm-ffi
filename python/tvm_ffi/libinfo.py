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
libraries by a base name (e.g., ``tvm_ffi`` -> ``libtvm_ffi.so`` on Linux).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def split_env_var(env_var: str, split: str) -> list[str]:
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


def get_dll_directories() -> list[str]:
    """Get the possible dll directories."""
    ffi_dir = Path(__file__).expanduser().resolve().parent
    dll_path: list[Path] = [ffi_dir / "lib"]
    dll_path.append(ffi_dir / ".." / ".." / "build" / "lib")
    # in source build from parent if needed
    dll_path.append(ffi_dir / ".." / ".." / ".." / "build" / "lib")
    if sys.platform.startswith("linux") or sys.platform.startswith("freebsd"):
        dll_path.extend(Path(p) for p in split_env_var("LD_LIBRARY_PATH", ":"))
        dll_path.extend(Path(p) for p in split_env_var("PATH", ":"))
    elif sys.platform.startswith("darwin"):
        dll_path.extend(Path(p) for p in split_env_var("DYLD_LIBRARY_PATH", ":"))
        dll_path.extend(Path(p) for p in split_env_var("PATH", ":"))
    elif sys.platform.startswith("win32"):
        dll_path.extend(Path(p) for p in split_env_var("PATH", ";"))

    valid_paths = []
    for path in dll_path:
        try:
            if path.is_dir():
                valid_paths.append(str(path.resolve()))
        except OSError:
            # need to ignore as resolve may fail if
            # we don't have permission to access it
            pass
    return valid_paths


def find_libtvm_ffi() -> str:
    """Find libtvm_ffi.

    Returns
    -------
    path
        The full path to the located library.

    """
    return find_library_by_basename("tvm_ffi")


def find_library_by_basename(base: str) -> str:
    """Find a shared library by base name across known directories.

    Parameters
    ----------
    base
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
    dll_path = [Path(p) for p in get_dll_directories()]
    if sys.platform.startswith("win32"):
        lib_dll_names = [f"{base}.dll"]
    elif sys.platform.startswith("darwin"):
        lib_dll_names = [  # Prefer dylib, also allow .so for some toolchains
            f"lib{base}.dylib",
            f"lib{base}.so",
        ]
    else:  # Linux, FreeBSD, etc
        lib_dll_names = [f"lib{base}.so"]

    lib_dll_path = [p / name for name in lib_dll_names for p in dll_path]
    lib_found = [p for p in lib_dll_path if p.exists() and p.is_file()]

    if not lib_found:
        candidate_list = "\n".join(str(p) for p in lib_dll_path)
        raise RuntimeError(
            f"Cannot find library: {', '.join(lib_dll_names)}\nList of candidates:\n{candidate_list}"
        )

    return str(lib_found[0])


def find_source_path() -> str:
    """Find packaged source home path."""
    candidates = [
        str(Path(__file__).resolve().parent),
        str(Path(__file__).resolve().parent / ".." / ".."),
    ]
    for candidate in candidates:
        if Path(candidate, "cmake").is_dir():
            return candidate
    raise RuntimeError("Cannot find home path.")


def find_cmake_path() -> str:
    """Find the preferred cmake path."""
    candidates = [
        str(Path(__file__).resolve().parent / "share" / "cmake" / "tvm_ffi"),  # Standard install
        str(Path(__file__).resolve().parent / ".." / ".." / "cmake"),  # Development mode
    ]
    for candidate in candidates:
        if Path(candidate).is_dir():
            return candidate
    raise RuntimeError("Cannot find cmake path.")


def find_include_path() -> str:
    """Find header files for C compilation."""
    candidates = [
        str(Path(__file__).resolve().parent / "include"),
        str(Path(__file__).resolve().parent / ".." / ".." / "include"),
    ]
    for candidate in candidates:
        if Path(candidate).is_dir():
            return candidate
    raise RuntimeError("Cannot find include path.")


def find_python_helper_include_path() -> str:
    """Find header files for C compilation."""
    candidates = [
        str(Path(__file__).resolve().parent / "include"),
        str(Path(__file__).resolve().parent / "cython"),
    ]
    for candidate in candidates:
        if Path(candidate, "tvm_ffi_python_helpers.h").is_file():
            return candidate
    raise RuntimeError("Cannot find python helper include path.")


def find_dlpack_include_path() -> str:
    """Find dlpack header files for C compilation."""
    install_include_path = Path(__file__).resolve().parent / "include"
    if (install_include_path / "dlpack").is_dir():
        return str(install_include_path)

    source_include_path = (
        Path(__file__).resolve().parent / ".." / ".." / "3rdparty" / "dlpack" / "include"
    )
    if source_include_path.is_dir():
        return str(source_include_path)

    raise RuntimeError("Cannot find include path.")


def find_cython_lib() -> str:
    """Find the path to tvm cython."""
    path_candidates = [
        Path(__file__).resolve().parent,
        Path(__file__).resolve().parent / ".." / ".." / "build",
    ]
    suffixes = "pyd" if sys.platform.startswith("win32") else "so"
    for candidate in path_candidates:
        for path in Path(candidate).glob(f"core*.{suffixes}"):
            return str(Path(path).resolve())
    raise RuntimeError("Cannot find tvm cython path.")


def include_paths() -> list[str]:
    """Find all include paths needed for FFI related compilation."""
    include_path = find_include_path()
    python_helper_include_path = find_python_helper_include_path()
    dlpack_include_path = find_dlpack_include_path()
    result = [include_path, dlpack_include_path]
    if python_helper_include_path != include_path:
        result.append(python_helper_include_path)
    return result
