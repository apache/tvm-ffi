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

"""TVM-FFI OrcJIT.

This module provides functionality to load object files (.o) compiled with TVM-FFI
exports using LLVM ORC JIT v2.

Example:
    >>> from tvm_ffi_orcjit import create_session
    >>> session = create_session()
    >>> lib = session.create_library()
    >>> lib.add("example.o")
    >>> func = lib.get_function("my_function")
    >>> result = func(arg1, arg2)

"""

import ctypes
import platform
from pathlib import Path

from tvm_ffi import load_module

# Determine the library extension based on platform
if platform.system() == "Darwin":
    _LIB_EXT = "dylib"
elif platform.system() == "Windows":
    _LIB_EXT = "dll"
else:
    _LIB_EXT = "so"

# Load the orcjit extension library
_LIB_PATH = [
    Path(__file__).parent / "lib" / f"libtvm_ffi_orcjit.{_LIB_EXT}",
    Path(__file__).parent / "build" / f"libtvm_ffi_orcjit.{_LIB_EXT}",
    Path(__file__).parent.parent.parent / "lib" / f"libtvm_ffi_orcjit.{_LIB_EXT}",
    Path(__file__).parent.parent.parent / "build" / f"libtvm_ffi_orcjit.{_LIB_EXT}",
]
_lib_path_str = None
for path in _LIB_PATH:
    if path.exists():
        _ = load_module(str(path))
        _lib_path_str = str(path)
if _lib_path_str is None:
    raise RuntimeError(
        f"Could not find libtvm_ffi_orcjit.{_LIB_EXT}. "
        f"Searched in {_LIB_PATH} and site-packages. "
        f"Please ensure the package is installed correctly."
    )

# Explicitly initialize the library to register functions
# This is needed because static initializers may not run when loaded via dlopen
try:
    # Load the library with ctypes and call the initialization function
    c_lib = ctypes.CDLL(_lib_path_str, mode=ctypes.RTLD_GLOBAL)
    init_func = c_lib.TVMFFIOrcJITInitialize
    init_func.restype = None
    init_func()
except Exception as e:
    import warnings

    warnings.warn(f"Failed to explicitly initialize orcjit library: {e}")

from .dylib import DynamicLibrary
from .session import ExecutionSession

__all__ = ["DynamicLibrary", "ExecutionSession"]
__version__ = "0.1.0"
