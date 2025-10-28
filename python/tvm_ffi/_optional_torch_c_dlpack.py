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
"""Optional module to support faster DLPack conversion.

This is an optional module to support faster DLPack conversion for torch.
Some of the changes are merged but not yet released, so it is used
as a stop gap to support faster DLPack conversion.

This file contains source code from PyTorch:
License: licenses/LICENSE.pytorch.txt

This module only serves as temp measure and will
likely be phased away and deleted after changes landed and released in pytorch.

This module will load slowly at first time due to JITing,
subsequent calls will be much faster.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any


def load_torch_c_dlpack_extension() -> Any:
    try:
        import torch  # noqa: PLC0415

        if hasattr(torch.Tensor, "__c_dlpack_exchange_api__"):
            # skip loading the extension if the __c_dlpack_exchange_api__
            # attribute is already set so we don't have to do it in
            # newer version of PyTorch
            return None
    except ImportError:
        return None

    """Load the torch c dlpack extension."""
    try:
        # todo: check whether a prebuilt package is installed, if so, use it.
        ...

        # check whether a JIT shared library is built in cache
        cache_dir = Path(os.environ.get("TVM_FFI_CACHE_DIR", "~/.cache/tvm-ffi")).expanduser()
        addon_build_dir = cache_dir / "torch_c_dlpack_addon"
        lib_path = addon_build_dir / (
            "libtorch_c_dlpack_addon" + (".dll" if sys.platform == "win32" else ".so")
        )
        if not lib_path.exists():
            build_script_path = Path(__file__).parent / "utils" / "build_optional_c_dlpack.py"
            args = [sys.executable, str(build_script_path), "--build_dir", str(addon_build_dir)]
            if torch.cuda.is_available():
                args.append("--build_with_cuda")
            subprocess.run(
                args,
                check=True,
            )
            assert lib_path.exists(), "Failed to build torch c dlpack addon."

        lib = ctypes.CDLL(str(lib_path))
        func = lib.TorchDLPackExchangeAPIPtr
        func.restype = ctypes.c_uint64
        func.argtypes = []

        # Set the DLPackExchangeAPI pointer on the class
        setattr(torch.Tensor, "__c_dlpack_exchange_api__", func())
    except ImportError:
        pass
    except Exception as e:
        warnings.warn(
            f"Failed to load torch c dlpack extension, EnvTensorAllocator will not be enabled:\n  {e}"
        )
    return None


def patch_torch_cuda_stream_protocol() -> Any:
    """Load the torch cuda stream protocol for older versions of torch."""
    try:
        import torch  # noqa: PLC0415

        if not torch.cuda.is_available():
            return
        if not hasattr(torch.cuda.Stream, "__cuda_stream__"):

            def __torch_cuda_stream__(self: torch.cuda.Stream) -> tuple[int, torch.cuda.Stream]:
                """Return the version number and the cuda stream."""
                return (0, self.cuda_stream)

            setattr(torch.cuda.Stream, "__cuda_stream__", __torch_cuda_stream__)
    except ImportError:
        pass


if os.environ.get("TVM_FFI_DISABLE_TORCH_C_DLPACK", "0") == "0":
    load_torch_c_dlpack_extension()
    patch_torch_cuda_stream_protocol()
