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
# specific language governing permissions and limitations.
# Base logic to load library for extension package
"""Utilities to locate and load the example extension shared library."""

import sys
from pathlib import Path

import tvm_ffi


def _load_lib() -> tvm_ffi.Module:
    # first look at the directory of the current file
    file_dir = Path(__file__).resolve().parent

    if sys.platform.startswith("win32"):
        lib_dll_name = "my_ffi_extension.dll"
    elif sys.platform.startswith("darwin"):
        lib_dll_name = "my_ffi_extension.dylib"
    else:
        lib_dll_name = "my_ffi_extension.so"

    lib_path = file_dir / lib_dll_name
    return tvm_ffi.load_module(str(lib_path))


_LIB = _load_lib()
