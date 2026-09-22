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
"""Shared-library visibility tests for the ORC JIT addon."""

from __future__ import annotations

import ctypes
import platform
import shutil
import subprocess
import sys

import pytest
import tvm_ffi_orcjit
from utils import build_test_objects


@pytest.mark.skipif(platform.system() == "Windows", reason="RTLD_DEFAULT is POSIX-only")
def test_addon_is_not_in_the_process_global_scope() -> None:
    """Loading the addon must not promote its symbols to RTLD_GLOBAL."""
    with pytest.raises(AttributeError):
        getattr(ctypes.CDLL(None), "TVMFFIOrcJITInitialize")


@pytest.mark.skipif(platform.system() == "Windows", reason="nm flags differ on Windows")
def test_addon_exports_only_initializer() -> None:
    """Static LLVM and C++ runtime symbols must stay out of the dynamic API."""
    nm = shutil.which("nm")
    if nm is None:
        pytest.skip("nm is unavailable")

    lib_path = tvm_ffi_orcjit._lib_path
    if platform.system() == "Darwin":
        command = [nm, "-gjU", str(lib_path)]
        # LLVM's PrettyStackTrace object marks this hidden symbol with
        # REFERENCED_DYNAMICALLY, which makes Apple ld export it even when an
        # exported-symbols list is present.
        allowed_runtime_exports = {"___crashreporter_info__"}
        expected = {"_TVMFFIOrcJITInitialize"}
    else:
        command = [nm, "-D", "--defined-only", "--format=posix", str(lib_path)]
        allowed_runtime_exports = set()
        expected = {"TVMFFIOrcJITInitialize"}

    output = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    exported = {line.split()[0].split("@@", 1)[0] for line in output.splitlines() if line.strip()}
    exported -= allowed_runtime_exports
    assert exported == expected


def test_tvm_ffi_loader_survives_process_shutdown() -> None:
    """The TVM-FFI keep-alive registry must retain the addon through shutdown."""
    obj_dir = build_test_objects()
    candidates = [
        "cc-gcc/test_funcs.o",
        "cc/test_funcs.o",
        "cc-appleclang/test_funcs.o",
        "c-msvc/test_funcs.o",
        "c-clang-cl/test_funcs.o",
        "c/test_funcs.o",
        "c-gcc/test_funcs.o",
    ]
    obj_path = next(
        (obj_dir / candidate for candidate in candidates if (obj_dir / candidate).is_file()), None
    )
    if obj_path is None:
        pytest.skip("no test object is available")

    script = """
import sys
from tvm_ffi_orcjit import ExecutionSession

module = ExecutionSession().load_module(sys.argv[1], keep_module_alive=True)
assert module.test_add(2, 3) == 5
del module
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(obj_path.resolve())],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
