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
from typing import Any

import torch
from packaging.version import Version


def _create_dlpack_exchange_api_capsule(ptr_as_int: int) -> Any:
    """Create a PyCapsule wrapping the DLPack exchange API pointer."""
    capsule_name = b"dlpack_exchange_api"
    pythonapi = ctypes.pythonapi
    pythonapi.PyCapsule_New.restype = ctypes.py_object
    pythonapi.PyCapsule_New.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_void_p,
    ]
    capsule = pythonapi.PyCapsule_New(ctypes.c_void_p(ptr_as_int), capsule_name, None)
    return capsule


def load_torch_c_dlpack_extension() -> None:
    """Load the torch c dlpack extension based on torch version."""
    if hasattr(torch.Tensor, "__dlpack_c_exchange_api__") or hasattr(
        torch.Tensor, "__c_dlpack_exchange_api__"
    ):
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
    # note: we need to keep this behavior for a while
    # to ensure backward compatibility with older versions dependencies
    # that relies on the value being int.
    # We will do eager upgrade to PyCapsule in the tvm-ffi side instead.
    dlpack_exchange_api_ptr_as_int = func()
    setattr(torch.Tensor, "__c_dlpack_exchange_api__", dlpack_exchange_api_ptr_as_int)
    setattr(
        torch.Tensor,
        "__dlpack_c_exchange_api__",
        _create_dlpack_exchange_api_capsule(dlpack_exchange_api_ptr_as_int),
    )

    return lib


_lib = load_torch_c_dlpack_extension()
