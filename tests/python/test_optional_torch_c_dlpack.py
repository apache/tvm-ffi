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
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import pytest

try:
    import torch
    import torch.version
except ImportError:
    torch = None  # ty: ignore[invalid-assignment]


import tvm_ffi
from tvm_ffi import _optional_torch_c_dlpack

IS_WINDOWS = sys.platform.startswith("win")


def _fake_torch_module(
    *, cuda_available: bool, cuda_version: Optional[str], hip_version: Optional[str]
) -> Any:
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda_available),
        version=SimpleNamespace(cuda=cuda_version, hip=hip_version),
    )


def test_torch_extension_device() -> None:
    assert (
        _optional_torch_c_dlpack._torch_extension_device(
            _fake_torch_module(cuda_available=False, cuda_version=None, hip_version=None)
        )
        == "cpu"
    )
    assert (
        _optional_torch_c_dlpack._torch_extension_device(
            _fake_torch_module(cuda_available=True, cuda_version="12.8", hip_version=None)
        )
        == "cuda"
    )
    assert (
        _optional_torch_c_dlpack._torch_extension_device(
            _fake_torch_module(cuda_available=True, cuda_version=None, hip_version="7.2")
        )
        == "rocm"
    )


@pytest.mark.skipif(torch is None, reason="torch is not installed")
def test_build_torch_c_dlpack_extension() -> None:
    assert torch is not None
    build_script = Path(tvm_ffi.__file__).parent / "utils" / "_build_optional_torch_c_dlpack.py"
    args = [
        sys.executable,
        str(build_script),
        "--output-dir",
        "./output-dir",
        "--libname",
        "libtorch_c_dlpack_addon_test.so",
    ]
    # First use "torch.cuda.is_available()" to check whether GPU environment
    # is available. Then determine the GPU type.
    if torch.cuda.is_available():
        if torch.version.cuda is not None:
            args.append("--build-with-cuda")
        elif torch.version.hip is not None:
            args.append("--build-with-rocm")
        else:
            raise ValueError("Cannot determine whether to build with CUDA or ROCm.")
    subprocess.run(args, check=True)

    lib_path = str(Path("./output-dir/libtorch_c_dlpack_addon_test.so").resolve())
    assert Path(lib_path).exists()

    lib = ctypes.CDLL(lib_path)
    func = lib.TorchDLPackExchangeAPIPtr
    func.restype = ctypes.c_int64
    ptr = func()
    assert ptr != 0


@pytest.mark.skipif(torch is None, reason="torch is not installed")
def test_parallel_build() -> None:
    build_script = Path(tvm_ffi.__file__).parent / "utils" / "_build_optional_torch_c_dlpack.py"
    num_processes = 4
    output_dir = "./output-dir-parallel"
    libname = "libtorch_c_dlpack_addon_test.so"
    processes = []
    for i in range(num_processes):
        p = subprocess.Popen(
            [sys.executable, str(build_script), "--output-dir", output_dir, "--libname", libname]
        )
        processes.append((p, output_dir))

    for p, output_dir in processes:
        p.wait()
        assert p.returncode == 0
    lib_path = str(Path(f"{output_dir}/{libname}").resolve())
    assert Path(lib_path).exists()


if __name__ == "__main__":
    pytest.main([__file__])
