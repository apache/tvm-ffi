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

import ctypes
from packaging.version import Version
from pathlib import Path
import sys


def get_torch_c_dlpack_lib_path(version_str: str):
    version = Version(version_str)

    if sys.platform.startswith("win32"):
        extension = ".dll"
    elif sys.platform.startswith("darwin"):
        extension = ".dylib"
    else:
        extension = ".so"

    if version >= Version("2.4") and version <= Version("2.9"):
        return (
            Path(__file__).parent
            / f"libtorch_c_dlpack_addon_torch{version.major}{version.minor}{extension}"
        )
    raise ValueError


def load_torch_c_dlpack_lib(version_str: str):
    lib_path = get_torch_c_dlpack_lib_path(version_str)
    lib = ctypes.CDLL(str(lib_path))
    func = lib.TorchDLPackExchangeAPIPtr
    func.restype = ctypes.c_uint64
    func.argtypes = []
    return func
