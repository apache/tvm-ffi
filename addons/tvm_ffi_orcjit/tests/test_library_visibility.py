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

import pytest
import tvm_ffi_orcjit


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
        expected = {"_TVMFFIOrcJITInitialize"}
    else:
        command = [nm, "-D", "--defined-only", "--format=posix", str(lib_path)]
        expected = {"TVMFFIOrcJITInitialize"}

    output = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    exported = {line.split()[0].split("@@", 1)[0] for line in output.splitlines() if line.strip()}
    assert exported == expected
