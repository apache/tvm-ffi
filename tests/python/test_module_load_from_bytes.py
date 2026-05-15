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
"""End-to-end test of the `ffi.ModuleLoadFromBytes` dispatch API.

The C++ API exposes a dispatching entry point — actual module-format
loaders are registered by consumers via
``ffi.Module.load_from_bytes.<kind>``. This test registers a trivial
Python loader, then drives a round-trip through
``Module::LoadFromBytes`` and verifies the loader was called with the
expected payload.

This is the smallest exercise of the API contract introduced by
``Module::LoadFromBytes`` — it does not depend on libcuda or any
real backend (the cubin_launcher example covers the CUDA case).
"""

from __future__ import annotations

import pytest
import tvm_ffi


def _make_dummy_module() -> tvm_ffi.Module:
    """Build a minimal Module by reusing system_lib(); the loader needs to
    return a real Module, but the test doesn't care what's in it.
    """
    return tvm_ffi.system_lib()


def test_load_from_bytes_dispatches_to_registered_loader() -> None:
    payload_seen: list[bytes] = []
    kind_seen: list[str] = []

    @tvm_ffi.register_global_func("ffi.Module.load_from_bytes.test_echo", override=True)
    def _echo_loader(payload: bytes) -> tvm_ffi.Module:
        payload_seen.append(bytes(payload))
        kind_seen.append("test_echo")
        return _make_dummy_module()

    mod = tvm_ffi.load_module_from_bytes("test_echo", b"hello world", keep_module_alive=False)
    assert isinstance(mod, tvm_ffi.Module)
    assert payload_seen == [b"hello world"]
    assert kind_seen == ["test_echo"]


def test_load_from_bytes_raises_for_unregistered_kind() -> None:
    with pytest.raises(RuntimeError) as exc_info:
        tvm_ffi.load_module_from_bytes("nonexistent_kind_xyz", b"")
    # Error message should name the unregistered loader so the user knows
    # exactly what they need to register.
    assert "ffi.Module.load_from_bytes.nonexistent_kind_xyz" in str(exc_info.value)


def test_load_from_bytes_propagates_loader_exceptions() -> None:
    @tvm_ffi.register_global_func("ffi.Module.load_from_bytes.boom", override=True)
    def _boom(_payload: bytes) -> tvm_ffi.Module:
        raise ValueError("loader rejected the payload")

    with pytest.raises(Exception) as exc_info:
        tvm_ffi.load_module_from_bytes("boom", b"<bad payload>")
    assert "loader rejected the payload" in str(exc_info.value)
