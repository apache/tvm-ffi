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
import logging
import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)  # type: ignore


def _create_dlpack_exchange_api_capsule(ptr_as_int: int) -> Any:
    """Create a PyCapsule wrapping the DLPack exchange API pointer.

    Parameters
    ----------
    ptr_as_int : int
        The pointer to the DLPack exchange API as an integer.

    Returns
    -------
    capsule : PyCapsule
        A PyCapsule object wrapping the pointer with name "dlpack_exchange_api".

    """
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


def _check_and_update_dlpack_c_exchange_api(tensor_cls: object) -> bool:
    """Check if the DLPack exchange API is available and update the __dlpack_c_exchange_api__ attribute."""
    if hasattr(tensor_cls, "__dlpack_c_exchange_api__"):
        return True
    # legacy path compactibility handling
    if hasattr(tensor_cls, "__c_dlpack_exchange_api__"):
        c_dlpack_attribute = tensor_cls.__c_dlpack_exchange_api__
        if isinstance(c_dlpack_attribute, int):
            setattr(
                tensor_cls,
                "__dlpack_c_exchange_api__",
                _create_dlpack_exchange_api_capsule(c_dlpack_attribute),
            )
        else:
            setattr(tensor_cls, "__dlpack_c_exchange_api__", c_dlpack_attribute)
        return True
    return False


def load_torch_c_dlpack_extension() -> Any:  # noqa: PLR0912, PLR0915
    try:
        import torch  # noqa: PLC0415

        if _check_and_update_dlpack_c_exchange_api(torch.Tensor):
            # skip loading the extension if the __dlpack_c_exchange_api__
            # attribute is already set so we don't have to do it in
            # newer version of PyTorch
            return None
    except ImportError:
        return None

    """Load the torch c dlpack extension."""
    try:
        import torch_c_dlpack_ext  # type: ignore  # noqa: PLC0415, F401

        if _check_and_update_dlpack_c_exchange_api(torch.Tensor):
            return None
    except ImportError:
        pass

    try:
        # check whether a JIT shared library is built in cache
        cache_dir = Path(os.environ.get("TVM_FFI_CACHE_DIR", "~/.cache/tvm-ffi")).expanduser()
        addon_output_dir = cache_dir
        major, minor = torch.__version__.split(".")[:2]
        # First use "torch.cuda.is_available()" to check whether GPU environment
        # is available. Then determine the GPU type.
        if torch.cuda.is_available():
            if torch.version.cuda is not None:
                device = "cuda"
            elif torch.version.hip is not None:
                device = "rocm"
            else:
                raise ValueError("Cannot determine whether to build with CUDA or ROCm.")
        else:
            device = "cpu"
        suffix = ".dll" if sys.platform.startswith("win") else ".so"
        libname = f"libtorch_c_dlpack_addon_torch{major}{minor}-{device}{suffix}"
        lib_path = addon_output_dir / libname
        if not lib_path.exists():
            logger.debug("JIT-compiling torch-c-dlpack-ext to cache...")
            build_script_path = (
                Path(__file__).parent / "utils" / "_build_optional_torch_c_dlpack.py"
            )
            args = [
                sys.executable,
                str(build_script_path),
                "--output-dir",
                str(cache_dir),
                "--libname",
                libname,
            ]
            if device == "cuda":
                args.append("--build-with-cuda")
            elif device == "rocm":
                args.append("--build-with-rocm")

            # use capture_output to reduce noise when building the torch c dlpack addon
            result = subprocess.run(args, check=False, capture_output=True)
            if result.returncode != 0:
                msg = [f"Build failed with status {result.returncode}"]
                if result.stdout:
                    msg.append(f"stdout:\n{result.stdout.decode('utf-8')}")
                if result.stderr:
                    msg.append(f"stderr:\n{result.stderr.decode('utf-8')}")
                raise RuntimeError("\n".join(msg))
            if not lib_path.exists():
                raise RuntimeError("Failed to build torch c dlpack addon.")

        lib = ctypes.CDLL(str(lib_path))
        func = lib.TorchDLPackExchangeAPIPtr
        func.restype = ctypes.c_uint64
        func.argtypes = []
        # Create a PyCapsule from the pointer
        capsule = _create_dlpack_exchange_api_capsule(func())
        # Set the DLPackExchangeAPI pointer on the class
        setattr(torch.Tensor, "__dlpack_c_exchange_api__", capsule)
        return lib
    except ImportError:
        pass
    except Exception:
        warnings.warn(
            "Failed to JIT torch c dlpack extension, EnvTensorAllocator will not be enabled.\n"
            "We recommend installing via `pip install torch-c-dlpack-ext`"
        )
    return None


def patch_torch_cuda_stream_protocol() -> None:
    """Load the torch cuda stream protocol for older versions of torch."""
    try:
        import torch  # noqa: PLC0415

        if not torch.cuda.is_available():
            return
        if not hasattr(torch.cuda.Stream, "__cuda_stream__"):

            def __torch_cuda_stream__(self: torch.cuda.Stream) -> tuple[int, int]:
                """Return the version number and the cuda stream."""
                return (0, self.cuda_stream)

            setattr(torch.cuda.Stream, "__cuda_stream__", __torch_cuda_stream__)
    except ImportError:
        pass


if os.environ.get("TVM_FFI_DISABLE_TORCH_C_DLPACK", "0") == "0":
    _LIB = load_torch_c_dlpack_extension()  # keep a reference to the loaded shared library
    patch_torch_cuda_stream_protocol()
