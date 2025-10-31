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
"""torch c dlpack ext core methods."""

import ctypes
import sys
from pathlib import Path

import torch
from packaging.version import Version


def load_torch_c_dlpack_extension() -> None:
    """Load the torch c dlpack extension based on torch version."""
    if hasattr(torch.Tensor, "__c_dlpack_exchange_api__"):
        return None
    version = Version(torch.__version__)
    if sys.platform.startswith("win32"):
        extension = "dll"
    elif sys.platform.startswith("darwin"):
        extension = "dylib"
    else:
        extension = "so"
    suffix = "cuda" if torch.cuda.is_available() else "cpu"
    lib_path = (
        Path(__file__).parent
        / f"libtorch_c_dlpack_addon_torch{version.major}{version.minor}-{suffix}.{extension}"
    )
    if not lib_path.exists() or not lib_path.is_file():
        raise ImportError("No matching prebuilt torch c dlpack extension")
    lib = ctypes.CDLL(str(lib_path))
    func = lib.TorchDLPackExchangeAPIPtr
    func.restype = ctypes.c_uint64
    func.argtypes = []
    setattr(torch.Tensor, "__c_dlpack_exchange_api__", func())
    return lib


_lib = load_torch_c_dlpack_extension()
