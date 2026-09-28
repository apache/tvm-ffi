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
"""Tests for the compile and link lines of the build.ninja that tvm_ffi.cpp generates."""

from __future__ import annotations

import shlex
from pathlib import Path

import pytest
from tvm_ffi.cpp import extension

CUDA_TARGET = "-gencode=arch=compute_86,code=sm_86"
WINDOWS_CUDA_HOME = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0"
WINDOWS_LIBTVM_FFI = "C:/Users/Ada Lovelace/venv/Lib/site-packages/tvm_ffi/lib/tvm_ffi.dll"


def _variables(ninja: str) -> dict[str, list[str]]:
    """Return the top-level variables of a build.ninja, each split into its arguments."""
    variables = {}
    for line in ninja.splitlines():
        if not line:
            break
        name, _, value = line.partition(" = ")
        variables[name] = shlex.split(value, posix=False)
    return variables


def _last_with_prefix(flags: list[str], prefix: str) -> str:
    return [flag for flag in flags if flag.startswith(prefix)][-1]


def _cuda_build(
    extra_cflags: list[str] | None = None, extra_cuda_cflags: list[str] | None = None
) -> dict[str, list[str]]:
    return _variables(
        extension._generate_ninja_build(
            name="kernels",
            extra_cflags=extra_cflags or [],
            extra_cuda_cflags=extra_cuda_cflags or [],
            extra_ldflags=[],
            extra_include_paths=[],
            sources=["main.cc", "kernels.cu"],
            backend="cuda",
        )
    )


@pytest.fixture
def windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(extension, "IS_WINDOWS", True)
    monkeypatch.setattr(extension, "find_libtvm_ffi", lambda: WINDOWS_LIBTVM_FFI)
    monkeypatch.setattr(extension, "_find_cuda_home", lambda: WINDOWS_CUDA_HOME)
    monkeypatch.setattr(extension, "_get_cuda_target", lambda: CUDA_TARGET)


@pytest.fixture
def linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(extension, "IS_WINDOWS", False)
    monkeypatch.setattr(extension, "find_libtvm_ffi", lambda: "/venv/tvm_ffi/lib/libtvm_ffi.so")
    monkeypatch.setattr(extension, "_find_cuda_home", lambda: "/usr/local/cuda")
    monkeypatch.setattr(extension, "_get_cuda_target", lambda: CUDA_TARGET)


@pytest.mark.usefixtures("windows")
def test_windows_nvcc_line_holds_no_bare_host_flag() -> None:
    # nvcc forwards its own -std and -O to cl. A host flag outside -Xcompiler is
    # read as a second input file, and a -Xcompiler /std: outranks the caller's -std.
    cuda_cflags = _cuda_build()["cuda_cflags"]
    assert cuda_cflags[2:5] == ["-std=c++17", "-O2", CUDA_TARGET]
    for index, flag in enumerate(cuda_cflags):
        if flag.startswith("/"):
            assert cuda_cflags[index - 1] == "-Xcompiler", flag


@pytest.mark.usefixtures("windows")
def test_windows_cuda_host_code_shares_the_cpp_runtime() -> None:
    # cl defaults to the static runtime (/MT); a module mixing C++ and CUDA
    # objects then fails to link (LNK2038: mismatch detected for 'RuntimeLibrary').
    variables = _cuda_build()
    assert "/MD" in variables["cxxflags"]
    assert variables["cuda_cflags"][:2] == ["-Xcompiler", "/MD"]


@pytest.mark.usefixtures("windows")
def test_windows_links_the_cuda_runtime() -> None:
    ldflags = _cuda_build()["ldflags"]
    assert f'"/LIBPATH:{Path(WINDOWS_CUDA_HOME) / "lib" / "x64"}"' in ldflags
    assert "cudart.lib" in ldflags


@pytest.mark.usefixtures("windows")
def test_windows_library_path_with_a_space_stays_one_argument() -> None:
    assert f'"/LIBPATH:{Path(WINDOWS_LIBTVM_FFI).parent}"' in _cuda_build()["ldflags"]


@pytest.mark.usefixtures("windows")
def test_windows_library_path_without_a_space_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(extension, "find_libtvm_ffi", lambda: "C:/venv/tvm_ffi/lib/tvm_ffi.dll")
    assert f"/LIBPATH:{Path('C:/venv/tvm_ffi/lib')}" in _cuda_build()["ldflags"]


@pytest.mark.usefixtures("windows")
def test_windows_caller_standard_is_the_one_that_applies() -> None:
    # cl and nvcc both apply the last standard they are given.
    variables = _cuda_build(extra_cflags=["/std:c++20"], extra_cuda_cflags=["-std=c++20"])
    assert _last_with_prefix(variables["cxxflags"], "/std:") == "/std:c++20"
    assert _last_with_prefix(variables["cuda_cflags"], "-std=") == "-std=c++20"
    assert not any("/std:" in flag for flag in variables["cuda_cflags"])


@pytest.mark.usefixtures("windows")
def test_windows_cpp_is_optimized() -> None:
    assert "/O2" in _cuda_build()["cxxflags"]


@pytest.mark.usefixtures("linux")
def test_linux_lines_are_unchanged() -> None:
    variables = _cuda_build()
    assert variables["cxxflags"][:3] == ["-std=c++17", "-fPIC", "-O2"]
    assert variables["cuda_cflags"][:5] == ["-Xcompiler", "-fPIC", "-std=c++17", "-O2", CUDA_TARGET]
    assert variables["ldflags"] == [
        "-shared",
        f"-L{Path('/venv/tvm_ffi/lib')}",
        "-ltvm_ffi",
        f"-L{Path('/usr/local/cuda') / 'lib64'}",
        "-lcudart",
    ]
