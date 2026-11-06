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

from __future__ import annotations

import builtins
import ctypes
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

try:
    import torch
    import torch.version
except ImportError:
    torch = None  # ty: ignore[invalid-assignment]


import tvm_ffi
from tvm_ffi import _optional_torch_c_dlpack

IS_WINDOWS = sys.platform.startswith("win")


@pytest.mark.parametrize(
    ("has_existing_api", "torch_npu_available", "expected"),
    [
        pytest.param(False, None, False, id="missing-api"),
        pytest.param(True, None, True, id="existing-api"),
        pytest.param(True, True, False, id="torch-npu-override"),
    ],
)
def test_should_use_existing_torch_dlpack_api(
    monkeypatch: pytest.MonkeyPatch,
    has_existing_api: bool,
    torch_npu_available: bool | None,
    expected: bool,
) -> None:
    tensor_cls = type("Tensor", (), {})
    if has_existing_api:
        setattr(tensor_cls, "__dlpack_c_exchange_api__", object())
    torch_module = SimpleNamespace(
        Tensor=tensor_cls,
        cuda=SimpleNamespace(is_available=lambda: False),
        version=SimpleNamespace(cuda=None, hip=None),
    )
    if torch_npu_available is not None:
        torch_module.npu = SimpleNamespace(is_available=lambda: torch_npu_available)
    monkeypatch.setattr(_optional_torch_c_dlpack, "IS_WINDOWS", False)
    monkeypatch.setattr(_optional_torch_c_dlpack, "IS_DARWIN", False)

    assert _optional_torch_c_dlpack._should_use_existing_torch_dlpack_api(torch_module) is expected


@pytest.mark.skipif(torch is None, reason="torch is not installed")
@pytest.mark.parametrize(
    ("device_options", "expected"),
    [
        pytest.param((), (False, False, False), id="cpu"),
        pytest.param(("--build-with-cuda",), (True, False, False), id="cuda"),
        pytest.param(("--build-with-rocm",), (False, True, False), id="rocm"),
        pytest.param(("--build-with-torch-npu",), (False, False, True), id="torch-npu"),
    ],
)
def test_build_device_option_selection(
    monkeypatch: pytest.MonkeyPatch,
    device_options: tuple[str, ...],
    expected: tuple[bool, bool, bool],
) -> None:
    from tvm_ffi.utils import _build_optional_torch_c_dlpack  # noqa: PLC0415

    monkeypatch.setattr(_build_optional_torch_c_dlpack, "IS_WINDOWS", False)
    monkeypatch.setattr(_build_optional_torch_c_dlpack, "IS_DARWIN", False)
    args = _build_optional_torch_c_dlpack._parse_args(device_options)

    assert (
        args.build_with_cuda,
        args.build_with_rocm,
        args.build_with_torch_npu,
    ) == expected


@pytest.mark.skipif(torch is None, reason="torch is not installed")
@pytest.mark.parametrize(
    "device_options",
    [
        pytest.param(("--build-with-cuda", "--build-with-rocm"), id="cuda-rocm"),
        pytest.param(("--build-with-cuda", "--build-with-torch-npu"), id="cuda-torch-npu"),
        pytest.param(("--build-with-rocm", "--build-with-torch-npu"), id="rocm-torch-npu"),
    ],
)
def test_build_device_options_are_mutually_exclusive(
    monkeypatch: pytest.MonkeyPatch,
    device_options: tuple[str, ...],
) -> None:
    from tvm_ffi.utils import _build_optional_torch_c_dlpack  # noqa: PLC0415

    monkeypatch.setattr(_build_optional_torch_c_dlpack, "IS_WINDOWS", False)
    monkeypatch.setattr(_build_optional_torch_c_dlpack, "IS_DARWIN", False)

    with pytest.raises(SystemExit) as exc_info:
        _build_optional_torch_c_dlpack._parse_args(device_options)

    assert exc_info.value.code == 2


@pytest.mark.skipif(torch is None, reason="torch is not installed")
@pytest.mark.parametrize(
    ("is_windows", "is_darwin"),
    [
        pytest.param(True, False, id="windows"),
        pytest.param(False, True, id="macos"),
    ],
)
def test_torch_npu_build_is_rejected_on_unsupported_platforms(
    monkeypatch: pytest.MonkeyPatch,
    is_windows: bool,
    is_darwin: bool,
) -> None:
    from tvm_ffi.utils import _build_optional_torch_c_dlpack  # noqa: PLC0415

    monkeypatch.setattr(_build_optional_torch_c_dlpack, "IS_WINDOWS", is_windows)
    monkeypatch.setattr(_build_optional_torch_c_dlpack, "IS_DARWIN", is_darwin)

    with pytest.raises(SystemExit) as exc_info:
        _build_optional_torch_c_dlpack._parse_args(["--build-with-torch-npu"])

    assert exc_info.value.code == 2


def _fake_torch_module(
    *,
    cuda_available: bool,
    cuda_version: str | None = None,
    hip_version: str | None = None,
    torch_npu_available: bool | None = None,
    include_cuda_attr: bool = True,
    include_hip_attr: bool = True,
) -> Any:
    version = SimpleNamespace()
    if include_cuda_attr:
        version.cuda = cuda_version
    if include_hip_attr:
        version.hip = hip_version
    torch_module = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda_available),
        version=version,
    )
    if torch_npu_available is not None:
        torch_module.npu = SimpleNamespace(is_available=lambda: torch_npu_available)
    return torch_module


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
    assert (
        _optional_torch_c_dlpack._torch_extension_device(
            _fake_torch_module(
                cuda_available=True,
                include_cuda_attr=False,
                include_hip_attr=False,
            )
        )
        == "cuda"
    )


@pytest.mark.parametrize(
    ("cuda_available", "cuda_version", "hip_version", "platform", "expected"),
    [
        pytest.param(False, None, None, (False, False), "torch_npu", id="torch-npu"),
        pytest.param(True, "12.8", None, (False, False), "cuda", id="cuda-before-torch-npu"),
        pytest.param(True, None, "7.2", (False, False), "rocm", id="rocm-before-torch-npu"),
        pytest.param(False, None, None, (True, False), "cpu", id="windows-fallback"),
        pytest.param(False, None, None, (False, True), "cpu", id="macos-fallback"),
    ],
)
def test_torch_extension_device_with_torch_npu(
    monkeypatch: pytest.MonkeyPatch,
    cuda_available: bool,
    cuda_version: str | None,
    hip_version: str | None,
    platform: tuple[bool, bool],
    expected: str,
) -> None:
    is_windows, is_darwin = platform
    monkeypatch.setattr(_optional_torch_c_dlpack, "IS_WINDOWS", is_windows)
    monkeypatch.setattr(_optional_torch_c_dlpack, "IS_DARWIN", is_darwin)

    torch_module = _fake_torch_module(
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        hip_version=hip_version,
        torch_npu_available=True,
    )
    assert _optional_torch_c_dlpack._torch_extension_device(torch_module) == expected


def test_existing_torch_dlpack_api_is_preferred_on_rocm(monkeypatch: pytest.MonkeyPatch) -> None:
    torch_module = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
        version=SimpleNamespace(cuda=None, hip="7.2"),
        Tensor=SimpleNamespace(__dlpack_c_exchange_api__=object()),
    )
    original_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "torch_c_dlpack_ext":
            raise AssertionError("torch_c_dlpack_ext should not be imported")
        return original_import(name, *args, **kwargs)

    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "torch.version", torch_module.version)
    monkeypatch.setattr(builtins, "__import__", guarded_import)

    assert _optional_torch_c_dlpack.load_torch_c_dlpack_extension() is None


def _run_build(args: list[str]) -> None:
    """Run the addon build script, surfacing its output when the build fails.

    The build script reports compiler and linker errors through its own stderr, so
    capture it and attach it to the failure instead of only reporting the exit code.
    """
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise AssertionError(
            f"Build failed with exit status {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


@pytest.mark.skipif(torch is None, reason="torch is not installed")
def test_build_torch_c_dlpack_extension(tmp_path: Path) -> None:
    assert torch is not None
    build_script = Path(tvm_ffi.__file__).parent / "utils" / "_build_optional_torch_c_dlpack.py"
    output_dir = tmp_path / "output-dir"
    libname = "libtorch_c_dlpack_addon_test.so"
    args = [
        sys.executable,
        str(build_script),
        "--output-dir",
        str(output_dir),
        "--libname",
        libname,
    ]
    device = _optional_torch_c_dlpack._torch_extension_device(torch)
    if device == "cuda":
        args.append("--build-with-cuda")
    elif device == "rocm":
        args.append("--build-with-rocm")
    elif device == "torch_npu":
        args.append("--build-with-torch-npu")
    _run_build(args)

    lib_path = str((output_dir / libname).resolve())
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
            [sys.executable, str(build_script), "--output-dir", output_dir, "--libname", libname],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        processes.append((p, output_dir))

    for p, output_dir in processes:
        stdout, stderr = p.communicate()
        if p.returncode != 0:
            raise AssertionError(
                f"Build failed with exit status {p.returncode}\n"
                f"stdout:\n{stdout}\nstderr:\n{stderr}"
            )
    lib_path = str(Path(f"{output_dir}/{libname}").resolve())
    assert Path(lib_path).exists()


if __name__ == "__main__":
    pytest.main([__file__])
