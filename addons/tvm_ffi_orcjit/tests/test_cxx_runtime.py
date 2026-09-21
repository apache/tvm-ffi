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
"""Tests for compiler-selected C++ runtime discovery."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from tvm_ffi_orcjit import ExecutionSession
from tvm_ffi_orcjit import session as session_module
from utils import build_test_objects


def test_discover_linux_cxx_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The driver command selects both parts of the libstdc++ linker script."""
    shared = tmp_path / "libstdc++.so.6"
    nonshared = tmp_path / "libstdc++_nonshared.a"
    shared.touch()
    nonshared.touch()
    requested = {
        "libstdc++.so.6": shared,
        "libstdc++_nonshared.a": nonshared,
    }
    commands: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        filename = command[-1].split("=", maxsplit=1)[1]
        return subprocess.CompletedProcess(command, 0, stdout=f"{requested[filename]}\n")

    monkeypatch.setattr(session_module.subprocess, "run", fake_run)
    session_module._discover_linux_cxx_runtime.cache_clear()
    try:
        assert session_module._discover_linux_cxx_runtime("ccache g++-14") == (
            str(shared.resolve()),
            str(nonshared.resolve()),
        )
    finally:
        session_module._discover_linux_cxx_runtime.cache_clear()

    assert commands == [
        ["ccache", "g++-14", "-print-file-name=libstdc++.so.6"],
        ["ccache", "g++-14", "-print-file-name=libstdc++_nonshared.a"],
    ]


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="libstdc++ is Linux-only")
def test_load_module_with_explicit_cxx() -> None:
    """A caller can select the compiler runtime for one JITDylib."""
    cxx = shutil.which("c++")
    if cxx is None:
        pytest.skip("default C++ compiler is unavailable")

    obj_path = build_test_objects() / "cc-gcc" / "test_funcs.o"
    if not obj_path.exists():
        pytest.skip("GCC C++ test object is unavailable")

    mod = ExecutionSession().load_module(obj_path, cxx=cxx)
    assert mod.test_add(2, 3) == 5
