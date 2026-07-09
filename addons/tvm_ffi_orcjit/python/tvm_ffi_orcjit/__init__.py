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

Examples
--------
>>> import tvm_ffi_orcjit as oj
>>> session = oj.default_session()
>>> mod = session.load_module("example.o")
>>> result = mod.my_function(arg1, arg2)

"""

import ctypes
import os
import platform
import sys
from pathlib import Path

from tvm_ffi import load_module

# Determine the library name based on platform
if platform.system() == "Windows":
    _LIB_NAME = "tvm_ffi_orcjit.dll"
elif platform.system() == "Darwin":
    _LIB_NAME = "libtvm_ffi_orcjit.dylib"
else:
    _LIB_NAME = "libtvm_ffi_orcjit.so"

# Load the orcjit extension library
# - lib/: normal install (wheel)
# - ../../build/: editable install (cmake build output relative to python/tvm_ffi_orcjit/)
_LIB_PATH = [
    Path(__file__).parent / "lib" / _LIB_NAME,
    Path(__file__).parent.parent.parent / "build" / _LIB_NAME,
]
_lib_dir = None
for path in _LIB_PATH:
    if path.exists():
        _ = load_module(str(path))
        _lib_dir = path.parent
if _lib_dir is None:
    raise RuntimeError(
        f"Could not find {_LIB_NAME}. "
        f"Searched in {_LIB_PATH} and site-packages. "
        f"Please ensure the package is installed correctly."
    )

# Explicitly initialize the library to register functions
# This is needed because static initializers may not run when loaded via dlopen
try:
    # The dll search path need to be added explicitly in windows
    if sys.platform.startswith("win32"):
        os.add_dll_directory(str(_lib_dir))
    # Load the library with ctypes and call the initialization function
    c_lib = ctypes.CDLL(str(_lib_dir / _LIB_NAME), mode=ctypes.RTLD_GLOBAL)
    init_func = c_lib.TVMFFIOrcJITInitialize
    init_func.restype = None
    init_func()
except Exception as e:
    import warnings

    warnings.warn(f"Failed to explicitly initialize orcjit library: {e}")

from .session import ExecutionSession, default_session

__all__ = ["ExecutionSession", "default_session"]

try:
    from importlib.metadata import version

    __version__ = version("apache-tvm_ffi_orcjit")
except Exception:
    __version__ = "0.0.0.dev0"
