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
"""Utilities to locate tvm_ffi libraries, headers, and helper include paths."""

import os
import sys
from pathlib import Path


def split_env_var(env_var, split):
    """Split an environment variable string.

    Parameters
    ----------
    env_var : str
        Name of environment variable.

    split : str
        String to split env_var on.

    Returns
    -------
    splits : list(string)
        If env_var exists, split env_var. Otherwise, empty list.

    """
    if os.environ.get(env_var, None):
        return [p.strip() for p in os.environ[env_var].split(split)]
    return []


def get_dll_directories():
    """Get the possible dll directories."""
    ffi_dir = Path(__file__).expanduser().resolve().parent
    dll_path = [ffi_dir / "lib"]
    dll_path += [ffi_dir / ".." / ".." / "build" / "lib"]
    # in source build from parent if needed
    dll_path += [ffi_dir / ".." / ".." / ".." / "build" / "lib"]

    if sys.platform.startswith("linux") or sys.platform.startswith("freebsd"):
        dll_path.extend(split_env_var("LD_LIBRARY_PATH", ":"))
        dll_path.extend(split_env_var("PATH", ":"))
    elif sys.platform.startswith("darwin"):
        dll_path.extend(split_env_var("DYLD_LIBRARY_PATH", ":"))
        dll_path.extend(split_env_var("PATH", ":"))
    elif sys.platform.startswith("win32"):
        dll_path.extend(split_env_var("PATH", ";"))
    return [str(Path(x).resolve()) for x in dll_path if Path(x).is_dir()]


def find_libtvm_ffi():
    """Find libtvm_ffi."""
    dll_path = get_dll_directories()
    if sys.platform.startswith("win32"):
        lib_dll_names = ["tvm_ffi.dll"]
    elif sys.platform.startswith("darwin"):
        lib_dll_names = ["libtvm_ffi.dylib", "libtvm_ffi.so"]
    else:
        lib_dll_names = ["libtvm_ffi.so"]

    name = lib_dll_names
    lib_dll_path = [str(Path(p) / name) for name in lib_dll_names for p in dll_path]
    lib_found = [p for p in lib_dll_path if Path(p).exists() and Path(p).is_file()]

    if not lib_found:
        raise RuntimeError(f"Cannot find library: {name}\nList of candidates:\n{lib_dll_path}")

    return lib_found[0]


def find_source_path():
    """Find packaged source home path."""
    candidates = [
        str(Path(__file__).resolve().parent),
        str(Path(__file__).resolve().parent / ".." / ".."),
    ]
    for candidate in candidates:
        if Path(candidate, "cmake").is_dir():
            return candidate
    raise RuntimeError("Cannot find home path.")


def find_cmake_path():
    """Find the preferred cmake path."""
    candidates = [
        str(Path(__file__).resolve().parent / "cmake"),
        str(Path(__file__).resolve().parent / ".." / ".." / "cmake"),
    ]
    for candidate in candidates:
        if Path(candidate).is_dir():
            return candidate
    raise RuntimeError("Cannot find cmake path.")


def find_include_path():
    """Find header files for C compilation."""
    candidates = [
        str(Path(__file__).resolve().parent / "include"),
        str(Path(__file__).resolve().parent / ".." / ".." / "include"),
    ]
    for candidate in candidates:
        if Path(candidate).is_dir():
            return candidate
    raise RuntimeError("Cannot find include path.")


def find_python_helper_include_path():
    """Find header files for C compilation."""
    candidates = [
        str(Path(__file__).resolve().parent / "include"),
        str(Path(__file__).resolve().parent / "cython"),
    ]
    for candidate in candidates:
        if Path(candidate, "tvm_ffi_python_helpers.h").is_file():
            return candidate
    raise RuntimeError("Cannot find python helper include path.")


def find_dlpack_include_path():
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


def find_cython_lib():
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


def include_paths():
    """Find all include paths needed for FFI related compilation."""
    include_path = find_include_path()
    python_helper_include_path = find_python_helper_include_path()
    dlpack_include_path = find_dlpack_include_path()
    result = [include_path, dlpack_include_path]
    if python_helper_include_path != include_path:
        result.append(python_helper_include_path)
    return result
