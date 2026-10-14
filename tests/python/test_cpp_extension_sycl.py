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

"""Tests for the ``sycl`` backend of tvm_ffi.cpp and ``tvm/ffi/extra/sycl/base.h``.

Without a oneAPI toolchain, a fake ``icpx`` that forwards to the host C++ compiler and a
stub ``<sycl/sycl.hpp>`` exercise the build pipeline and the header logic. Tests needing
the real compiler or an Intel GPU are skipped when unavailable.
"""

from __future__ import annotations

import shutil
import stat
import sys
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
import tvm_ffi
import tvm_ffi.cpp
from tvm_ffi.cpp import extension

IS_WINDOWS = sys.platform == "win32"
_HOST_CXX = shutil.which("c++")
_REAL_ICPX = shutil.which("icpx")

_requires_fake_icpx = pytest.mark.skipif(
    IS_WINDOWS or _HOST_CXX is None, reason="Requires a Unix host C++ compiler"
)

_SYCL_STUB_HEADER = r"""
#pragma once
#include <exception>
#include <string>
#include <system_error>
#include <vector>

namespace sycl {
class exception : public std::exception {
 public:
  exception(std::error_code code, std::string msg) : code_(code), msg_(std::move(msg)) {}
  const std::error_code& code() const noexcept { return code_; }
  const char* what() const noexcept override { return msg_.c_str(); }

 private:
  std::error_code code_;
  std::string msg_;
};
struct queue {
  int id = 0;
};
struct device {
  int id = 0;
  static std::vector<device> get_devices() { return {device{10}, device{11}}; }
};
}  // namespace sycl
"""

_HEADER_TEST_SOURCE = r"""
#include <tvm/ffi/extra/sycl/base.h>

int64_t make_queue(int64_t id) {
  static sycl::queue queues[4];
  queues[id].id = static_cast<int>(id);
  return reinterpret_cast<int64_t>(&queues[id]);
}

int64_t current_queue_id(DLDevice device) {
  return tvm::ffi::GetCurrentSYCLQueue(device).id;
}

int64_t sycl_device_id(DLDevice device) {
  return tvm::ffi::SYCLDeviceFromDLDevice(device).id;
}

int64_t check_sycl_error(bool should_throw) {
  TVM_FFI_CHECK_SYCL_ERROR({
    if (should_throw) {
      throw sycl::exception(std::make_error_code(std::errc::invalid_argument), "bad kernel");
    }
  });
  return 1;
}
"""


@pytest.fixture(autouse=True)
def _clear_backend_cache() -> Iterator[None]:
    extension._detect_gpu_backend.cache_clear()
    yield
    extension._detect_gpu_backend.cache_clear()


@pytest.fixture
def fake_icpx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Put a fake ``icpx`` first on PATH; returns the file recording its arguments."""
    assert _HOST_CXX is not None
    bin_dir = tmp_path / "fake_oneapi" / "bin"
    bin_dir.mkdir(parents=True)
    log = tmp_path / "icpx_args.log"
    script = bin_dir / "icpx"
    script.write_text(
        "#!/bin/sh\n"
        f'echo "$@" >> "{log}"\n'
        "args=\n"
        'for a in "$@"; do\n'
        '  case "$a" in -fsycl|-fsycl-targets=*) ;; *) args="$args \'$a\'" ;; esac\n'
        "done\n"
        f'eval exec "{_HOST_CXX}" $args\n'
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setenv("PATH", f"{bin_dir}:{Path(sys.executable).parent}:/usr/bin:/bin")
    monkeypatch.delenv("TVM_FFI_SYCL_TARGETS", raising=False)
    return log


@pytest.fixture
def sycl_stub_include(tmp_path: Path) -> str:
    include_dir = tmp_path / "sycl_stub"
    (include_dir / "sycl").mkdir(parents=True)
    (include_dir / "sycl" / "sycl.hpp").write_text(_SYCL_STUB_HEADER)
    return str(include_dir)


def test_resolve_gpu_backend_sycl() -> None:
    assert extension._resolve_gpu_backend("sycl") == "sycl"
    with pytest.raises(ValueError, match="'cuda', 'hip', and 'sycl'"):
        extension._resolve_gpu_backend("xpu")


def test_detect_gpu_backend_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TVM_FFI_GPU_BACKEND", "SYCL")
    assert extension._detect_gpu_backend() == "sycl"


@pytest.mark.parametrize(
    "version, expected",
    [
        (SimpleNamespace(hip=None, cuda=None, xpu="20250201"), "sycl"),
        (SimpleNamespace(hip=None, cuda="12.8", xpu="20250201"), "cuda"),
        (SimpleNamespace(hip="7.2", cuda=None, xpu="20250201"), "hip"),
    ],
)
def test_detect_gpu_backend_from_torch_xpu(
    monkeypatch: pytest.MonkeyPatch, version: SimpleNamespace, expected: str
) -> None:
    monkeypatch.delenv("TVM_FFI_GPU_BACKEND", raising=False)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(version=version))
    assert extension._detect_gpu_backend() == expected


def test_detect_gpu_backend_does_not_infer_sycl_from_icpx(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("TVM_FFI_GPU_BACKEND", raising=False)
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(version=SimpleNamespace(hip=None, cuda=None))
    )
    monkeypatch.setattr(
        extension, "_find_compiler", lambda name, *homes: "/x/icpx" if name == "icpx" else None
    )
    assert extension._detect_gpu_backend() == "cuda"


def test_get_sycl_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TVM_FFI_SYCL_TARGETS", raising=False)
    assert extension._get_sycl_targets() == []
    monkeypatch.setenv("TVM_FFI_SYCL_TARGETS", "intel_gpu_pvc intel_gpu_bmg_g21")
    assert extension._get_sycl_targets() == ["-fsycl-targets=intel_gpu_pvc,intel_gpu_bmg_g21"]
    monkeypatch.setenv("TVM_FFI_SYCL_TARGETS", "spir64,intel_gpu_pvc")
    assert extension._get_sycl_targets() == ["-fsycl-targets=spir64,intel_gpu_pvc"]
    monkeypatch.setenv("TVM_FFI_SYCL_TARGETS", "intel_gpu_pvc;rm")
    with pytest.raises(ValueError, match="Invalid SYCL target"):
        extension._get_sycl_targets()


def _make_executable(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n")
    path.chmod(0o755)


@pytest.mark.skipif(IS_WINDOWS, reason="Unix executable layout")
def test_find_sycl_compiler(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.delenv("CMPLR_ROOT", raising=False)
    monkeypatch.delenv("ONEAPI_ROOT", raising=False)
    with pytest.raises(RuntimeError, match="Could not find the SYCL compiler icpx"):
        extension._find_sycl_compiler()

    oneapi_icpx = tmp_path / "oneapi" / "compiler" / "latest" / "bin" / "icpx"
    _make_executable(oneapi_icpx)
    monkeypatch.setenv("ONEAPI_ROOT", str(tmp_path / "oneapi"))
    assert extension._find_sycl_compiler() == str(oneapi_icpx)

    # CMPLR_ROOT takes priority over ONEAPI_ROOT
    cmplr_icpx = tmp_path / "cmplr" / "bin" / "icpx"
    _make_executable(cmplr_icpx)
    monkeypatch.setenv("CMPLR_ROOT", str(tmp_path / "cmplr"))
    assert extension._find_sycl_compiler() == str(cmplr_icpx)

    # PATH takes priority over both
    path_icpx = tmp_path / "path" / "icpx"
    _make_executable(path_icpx)
    monkeypatch.setenv("PATH", str(path_icpx.parent))
    assert extension._find_sycl_compiler() == str(path_icpx)


def _ninja_vars(ninja: str) -> dict[str, str]:
    result = {}
    for line in ninja.splitlines():
        if " = " in line and not line.startswith(" "):
            key, value = line.split(" = ", 1)
            result[key] = value
    return result


@pytest.mark.skipif(IS_WINDOWS, reason="sycl backend is Unix-only")
def test_generate_ninja_build_sycl(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(extension, "_find_sycl_compiler", lambda: "/opt/intel/bin/icpx")
    monkeypatch.setenv("TVM_FFI_SYCL_TARGETS", "intel_gpu_pvc")
    ninja = extension._generate_ninja_build(
        name="m",
        extra_cflags=["-DHOST_FLAG"],
        extra_cuda_cflags=["-DDEVICE_FLAG"],
        extra_ldflags=[],
        extra_include_paths=[],
        sources=["/src/main.cpp", "/src/sycl.cpp"],
        backend="sycl",
    )
    variables = _ninja_vars(ninja)
    assert variables["cxx"] == "/opt/intel/bin/icpx"
    cxxflags = variables["cxxflags"].split()
    assert "-fsycl" in cxxflags
    assert "-std=c++20" in cxxflags
    assert "-std=c++17" not in cxxflags
    assert "-fsycl-targets=intel_gpu_pvc" in cxxflags
    assert "-DHOST_FLAG" in cxxflags
    assert "-DDEVICE_FLAG" in cxxflags
    ldflags = variables["ldflags"].split()
    assert "-fsycl" in ldflags
    assert "-fsycl-targets=intel_gpu_pvc" in ldflags
    assert "nvcc" not in variables
    assert "rule compile_cuda" not in ninja
    assert "build cpp_0.o: compile /src/main.cpp" in ninja
    assert "build cpp_1.o: compile /src/sycl.cpp" in ninja


def test_generate_ninja_build_non_sycl_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail() -> str:
        raise AssertionError("icpx must not be looked up without the sycl backend")

    monkeypatch.setattr(extension, "_find_sycl_compiler", fail)
    ninja = extension._generate_ninja_build(
        name="m",
        extra_cflags=[],
        extra_cuda_cflags=["-DDEVICE_FLAG"],
        extra_ldflags=[],
        extra_include_paths=[],
        sources=["/src/main.cpp"],
    )
    cxxflags = _ninja_vars(ninja)["cxxflags"]
    assert "-fsycl" not in cxxflags
    assert "-DDEVICE_FLAG" not in cxxflags


def test_generate_ninja_build_sycl_rejects_cuda_sources() -> None:
    with pytest.raises(ValueError, match=r"\.cu\) sources cannot be built with the sycl backend"):
        extension._generate_ninja_build(
            name="m",
            extra_cflags=[],
            extra_cuda_cflags=[],
            extra_ldflags=[],
            extra_include_paths=[],
            sources=["/src/kernel.cu"],
            backend="sycl",
        )


def test_hash_sources_backend_only_changes_sycl_key() -> None:
    args = ("int x;", "int y;", None, None, {"f": ""}, [], [], [], [])
    base = extension._hash_sources(*args)
    assert extension._hash_sources(*args, backend=None) == base
    assert extension._hash_sources(*args, backend="sycl") != base


@_requires_fake_icpx
def test_load_inline_sycl_backend_pipeline(fake_icpx: Path, tmp_path: Path) -> None:
    build_dir = tmp_path / "build"
    mod = tvm_ffi.cpp.load_inline(
        name="sycl_pipeline",
        cpp_sources="int64_t host_add(int64_t a, int64_t b);",
        cuda_sources="int64_t host_add(int64_t a, int64_t b) { return a + b + __cplusplus / 100; }",
        functions=["host_add"],
        extra_cuda_cflags=["-DDEVICE_FLAG"],
        backend="sycl",
        build_directory=str(build_dir),
    )
    # compiled as C++20 (__cplusplus == 202002L)
    assert mod.host_add(1, 2) == 1 + 2 + 2020
    assert (build_dir / "sycl.cpp").exists()
    assert not (build_dir / "cuda.cu").exists()
    calls = fake_icpx.read_text().splitlines()
    compile_calls = [c for c in calls if " -c " in f" {c} "]
    link_calls = [c for c in calls if " -c " not in f" {c} "]
    assert len(compile_calls) == 2
    assert all("-fsycl" in c.split() and "-std=c++20" in c.split() for c in compile_calls)
    assert all("-DDEVICE_FLAG" in c.split() for c in compile_calls)
    assert len(link_calls) == 1
    assert "-fsycl" in link_calls[0].split()


@_requires_fake_icpx
def test_load_inline_sycl_backend_from_env(
    fake_icpx: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TVM_FFI_GPU_BACKEND", "sycl")
    build_dir = tmp_path / "build"
    mod = tvm_ffi.cpp.load_inline(
        name="sycl_env",
        cuda_sources="int64_t one() { return 1; }",
        functions=["one"],
        build_directory=str(build_dir),
    )
    assert mod.one() == 1
    assert (build_dir / "sycl.cpp").exists()
    assert fake_icpx.exists()


@_requires_fake_icpx
def test_build_sycl_backend_rejects_cu_files(fake_icpx: Path, tmp_path: Path) -> None:
    src = tmp_path / "kernel.cu"
    src.write_text("int f() { return 0; }")
    with pytest.raises(ValueError, match="cannot be built with the sycl backend"):
        tvm_ffi.cpp.build(
            name="sycl_cu", cuda_files=[str(src)], backend="sycl", build_directory=str(tmp_path)
        )


@_requires_fake_icpx
def test_sycl_base_header(fake_icpx: Path, sycl_stub_include: str, tmp_path: Path) -> None:
    mod = tvm_ffi.cpp.load_inline(
        name="sycl_base_header",
        cpp_sources=_HEADER_TEST_SOURCE,
        functions=["make_queue", "current_queue_id", "sycl_device_id", "check_sycl_error"],
        extra_include_paths=[sycl_stub_include],
        backend="sycl",
        build_directory=str(tmp_path / "build"),
    )
    dev = tvm_ffi.device("oneapi", 1)

    with pytest.raises(ValueError, match="No SYCL queue is set for oneapi:1"):
        mod.current_queue_id(dev)
    with tvm_ffi.use_raw_stream(dev, mod.make_queue(3)):
        assert mod.current_queue_id(dev) == 3
        with tvm_ffi.use_raw_stream(dev, mod.make_queue(2)):
            assert mod.current_queue_id(dev) == 2
        assert mod.current_queue_id(dev) == 3
    with pytest.raises(ValueError, match="Expected a kDLOneAPI device"):
        mod.current_queue_id(tvm_ffi.device("cpu"))

    assert mod.sycl_device_id(tvm_ffi.device("oneapi", 0)) == 10
    assert mod.sycl_device_id(tvm_ffi.device("xpu", 1)) == 11
    with pytest.raises(ValueError, match="oneapi:2 is out of range, 2 SYCL devices"):
        mod.sycl_device_id(tvm_ffi.device("oneapi", 2))
    with pytest.raises(ValueError, match="Expected a kDLOneAPI device"):
        mod.sycl_device_id(tvm_ffi.device("cuda", 0))

    assert mod.check_sycl_error(False) == 1
    with pytest.raises(RuntimeError, match=r"SYCL Error \(\d+\): bad kernel"):
        mod.check_sycl_error(True)


_REAL_KERNEL_SOURCE = r"""
#include <tvm/ffi/extra/sycl/base.h>

void add_one(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
  sycl::queue& q = tvm::ffi::GetCurrentSYCLQueue(x.device());
  const float* in = static_cast<const float*>(x.data_ptr());
  float* out = static_cast<float*>(y.data_ptr());
  TVM_FFI_CHECK_SYCL_ERROR(
      q.parallel_for(sycl::range<1>(x.size(0)), [=](sycl::id<1> i) { out[i] = in[i] + 1.0f; }));
}

bool queue_device_matches(tvm::ffi::TensorView x) {
  return tvm::ffi::GetCurrentSYCLQueue(x.device()).get_device() ==
         tvm::ffi::SYCLDeviceFromDLDevice(x.device());
}
"""


@pytest.mark.skipif(_REAL_ICPX is None, reason="Requires the oneAPI icpx compiler")
def test_load_inline_sycl_real_compiler(tmp_path: Path) -> None:
    mod = tvm_ffi.cpp.load_inline(
        name="sycl_add_one",
        cuda_sources=_REAL_KERNEL_SOURCE,
        functions=["add_one", "queue_device_matches"],
        backend="sycl",
        build_directory=str(tmp_path / "build"),
    )
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        return
    x = torch.arange(8, dtype=torch.float32, device="xpu")
    y = torch.empty_like(x)
    mod.add_one(x, y)
    torch.xpu.synchronize()
    torch.testing.assert_close(y, x + 1)
    # the DLPack device_id from torch resolves to the device of torch's queue
    assert mod.queue_device_matches(x)
