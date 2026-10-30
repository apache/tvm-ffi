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


def _fake_torch_module(
    *,
    cuda_available: bool,
    cuda_version: str | None = None,
    hip_version: str | None = None,
    include_cuda_attr: bool = True,
    include_hip_attr: bool = True,
    xpu_available: bool | None = None,
) -> Any:
    version = SimpleNamespace()
    if include_cuda_attr:
        version.cuda = cuda_version
    if include_hip_attr:
        version.hip = hip_version
    module = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda_available),
        version=version,
    )
    if xpu_available is not None:
        module.xpu = SimpleNamespace(is_available=lambda: xpu_available)
    return module


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


def test_torch_extension_device_xpu() -> None:
    assert (
        _optional_torch_c_dlpack._torch_extension_device(
            _fake_torch_module(cuda_available=False, xpu_available=True)
        )
        == "xpu"
    )
    assert (
        _optional_torch_c_dlpack._torch_extension_device(
            _fake_torch_module(cuda_available=False, xpu_available=False)
        )
        == "cpu"
    )
    assert (
        _optional_torch_c_dlpack._torch_extension_device(
            _fake_torch_module(cuda_available=True, cuda_version="12.8", xpu_available=True)
        )
        == "cuda"
    )


@pytest.mark.skipif(torch is None, reason="torch is not installed")
def test_prebuilt_addon_package_extension_device_xpu() -> None:
    """The torch_c_dlpack_ext package resolves the same device names as tvm_ffi."""
    import importlib.util  # noqa: PLC0415

    core_path = (
        Path(__file__).parents[2]
        / "addons"
        / "torch_c_dlpack_ext"
        / "torch_c_dlpack_ext"
        / "core.py"
    )
    spec = importlib.util.spec_from_file_location("_torch_c_dlpack_ext_core", core_path)
    assert spec is not None and spec.loader is not None
    addon_core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(addon_core)
    for fake in [
        _fake_torch_module(cuda_available=False, xpu_available=True),
        _fake_torch_module(cuda_available=False, xpu_available=False),
        _fake_torch_module(cuda_available=False),
        _fake_torch_module(cuda_available=True, cuda_version="12.8", xpu_available=True),
        _fake_torch_module(cuda_available=True, hip_version="7.2"),
    ]:
        assert addon_core._torch_extension_device(
            fake
        ) == _optional_torch_c_dlpack._torch_extension_device(fake)
    assert (
        addon_core._torch_extension_device(
            _fake_torch_module(cuda_available=False, xpu_available=True)
        )
        == "xpu"
    )


def test_addon_build_flags() -> None:
    assert _optional_torch_c_dlpack._addon_build_flags("cuda") == ["--build-with-cuda"]
    assert _optional_torch_c_dlpack._addon_build_flags("rocm") == ["--build-with-rocm"]
    assert _optional_torch_c_dlpack._addon_build_flags("xpu") == ["--build-with-xpu"]
    assert _optional_torch_c_dlpack._addon_build_flags("cpu") == []


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
        device = _optional_torch_c_dlpack._torch_extension_device(torch)
        if device == "cuda":
            args.append("--build-with-cuda")
        elif device == "rocm":
            args.append("--build-with-rocm")
        else:
            raise ValueError("Cannot determine whether to build with CUDA or ROCm.")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        args.append("--build-with-xpu")
    _run_build(args)

    lib_path = str(Path("./output-dir/libtorch_c_dlpack_addon_test.so").resolve())
    assert Path(lib_path).exists()

    lib = ctypes.CDLL(lib_path)
    func = lib.TorchDLPackExchangeAPIPtr
    func.restype = ctypes.c_int64
    ptr = func()
    assert ptr != 0


class _DLPackExchangeAPI(ctypes.Structure):
    """ctypes mirror of DLPackExchangeAPI in dlpack.h."""

    _fields_ = [  # noqa: RUF012
        ("major", ctypes.c_uint32),
        ("minor", ctypes.c_uint32),
        ("prev_api", ctypes.c_void_p),
        ("managed_tensor_allocator", ctypes.c_void_p),
        ("managed_tensor_from_py_object_no_sync", ctypes.c_void_p),
        ("managed_tensor_to_py_object_no_sync", ctypes.c_void_p),
        ("dltensor_from_py_object_no_sync", ctypes.c_void_p),
        ("current_work_stream", ctypes.c_void_p),
    ]


# PYFUNCTYPE keeps the GIL held and raises the Python error set by the addon on failure
_CurrentWorkStream = ctypes.PYFUNCTYPE(
    ctypes.c_int, ctypes.c_int, ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p)
)


@pytest.mark.skipif(torch is None, reason="torch is not installed")
@pytest.mark.skipif(IS_WINDOWS, reason="addon library name is .so-specific in this test")
def test_build_torch_c_dlpack_extension_xpu(tmp_path: Path) -> None:
    """The XPU variant builds without a SYCL compiler and reports the XPU queue."""
    assert torch is not None
    build_script = Path(tvm_ffi.__file__).parent / "utils" / "_build_optional_torch_c_dlpack.py"
    _run_build(
        [
            sys.executable,
            str(build_script),
            "--output-dir",
            str(tmp_path),
            "--build-with-xpu",
        ]
    )
    major, minor = torch.__version__.split(".")[:2]
    lib_path = tmp_path / f"libtorch_c_dlpack_addon_torch{major}{minor}-xpu.so"
    assert lib_path.exists()

    lib = ctypes.CDLL(str(lib_path))
    lib.TorchDLPackExchangeAPIPtr.restype = ctypes.c_void_p
    api = _DLPackExchangeAPI.from_address(lib.TorchDLPackExchangeAPIPtr())
    current_work_stream = _CurrentWorkStream(api.current_work_stream)

    stream = ctypes.c_void_p(1)
    assert current_work_stream(1, 0, ctypes.byref(stream)) == 0  # kDLCPU
    assert stream.value is None

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        from tvm_ffi.utils.xpu import xpu_global_device_index  # noqa: PLC0415

        xpu_stream = torch.xpu.Stream()
        with torch.xpu.stream(xpu_stream):
            global_idx = xpu_global_device_index(xpu_stream.device.index)
            assert current_work_stream(14, global_idx, ctypes.byref(stream)) == 0
            assert stream.value == xpu_stream.sycl_queue
        # a SYCL device that torch does not enumerate has no torch index
        with pytest.raises(RuntimeError, match="global SYCL device index 10000"):
            current_work_stream(14, 10000, ctypes.byref(stream))
    else:
        with pytest.raises(RuntimeError, match="global SYCL device index"):
            current_work_stream(14, 0, ctypes.byref(stream))


@pytest.mark.skipif(torch is None, reason="torch is not installed")
def test_build_torch_c_dlpack_extension_rejects_multiple_backends(tmp_path: Path) -> None:
    build_script = Path(tvm_ffi.__file__).parent / "utils" / "_build_optional_torch_c_dlpack.py"
    result = subprocess.run(
        [
            sys.executable,
            str(build_script),
            "--output-dir",
            str(tmp_path),
            "--build-with-cuda",
            "--build-with-xpu",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "Only one of CUDA, ROCm, and XPU" in result.stderr


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
