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

"""Tests for Intel XPU (kDLOneAPI) support.

Most tests mock ``torch.xpu`` so they run without an Intel GPU; tests marked
with ``_requires_xpu`` exercise the real device.
"""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest
import tvm_ffi
import tvm_ffi.cpp
from tvm_ffi import DLDeviceType, core
from tvm_ffi import stream as ffi_stream
from tvm_ffi.utils import xpu as ffi_xpu

try:
    import torch
    import torch.utils.dlpack
except ImportError:
    torch = None  # ty: ignore[invalid-assignment]

_HAS_TORCH_XPU_MODULE = torch is not None and hasattr(torch, "xpu")
_HAS_XPU = _HAS_TORCH_XPU_MODULE and bool(torch.xpu.is_available())

_requires_torch_xpu_module = pytest.mark.skipif(
    not _HAS_TORCH_XPU_MODULE, reason="Requires torch with the torch.xpu module"
)
_requires_xpu = pytest.mark.skipif(not _HAS_XPU, reason="Requires an Intel XPU device")

FAKE_QUEUE = 0x1234


@pytest.fixture(autouse=True)
def _clear_index_cache() -> Iterator[None]:
    ffi_xpu.xpu_global_device_index.cache_clear()
    yield
    ffi_xpu.xpu_global_device_index.cache_clear()


def _fake_xpu_stream(index: int, queue: int = FAKE_QUEUE) -> Any:
    assert torch is not None
    return SimpleNamespace(device=torch.device("xpu", index), sycl_queue=queue)


def _noop_xpu_stream_context() -> Any:
    """Create a ``torch.xpu.StreamContext`` that does not touch the (absent) device."""
    assert torch is not None
    ctx = object.__new__(torch.xpu.StreamContext)
    ctx.stream = None
    ctx.idx = -1
    return ctx


def _fresh_tensor_type() -> type:
    """Create a new torch.Tensor subclass, so the per-type arg setter cache is not hit."""
    assert torch is not None
    return type("FreshTensor", (torch.Tensor,), {})


def gen_check_stream_mod() -> tvm_ffi.Module:
    return tvm_ffi.cpp.load_inline(
        name="check_xpu_stream",
        cpp_sources="""
        void check_stream(tvm::ffi::TensorView x, int device_type, int device_id, uint64_t stream) {
            uint64_t cur_stream = reinterpret_cast<uint64_t>(TVMFFIEnvGetStream(device_type, device_id));
            TVM_FFI_ICHECK_EQ(cur_stream, stream);
        }
    """,
        functions=["check_stream"],
    )


@pytest.mark.skipif(torch is None, reason="Requires torch")
def test_torch_dlpack_device_uses_cpp_exporter() -> None:
    assert torch is not None
    assert ffi_xpu._torch_dlpack_device(torch.empty(2)) == (DLDeviceType.kDLCPU, 0)


@_requires_torch_xpu_module
def test_xpu_global_device_index_maps_and_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    assert torch is not None
    probed: list[Any] = []

    def fake_empty(*shape: int, device: Any) -> Any:
        probed.append(device)
        return SimpleNamespace(device=device)

    monkeypatch.setattr(torch, "empty", fake_empty)
    # pretend a non-GPU SYCL device enumerates first, shifting the global index by one
    monkeypatch.setattr(
        ffi_xpu,
        "_torch_dlpack_device",
        lambda t: (int(DLDeviceType.kDLOneAPI), t.device.index + 1),
    )
    assert ffi_xpu.xpu_global_device_index(0) == 1
    assert ffi_xpu.xpu_global_device_index(2) == 3
    assert ffi_xpu.xpu_global_device_index(0) == 1
    assert probed == [torch.device("xpu", 0), torch.device("xpu", 2)]


@_requires_torch_xpu_module
def test_xpu_global_device_index_rejects_wrong_device_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert torch is not None
    monkeypatch.setattr(torch, "empty", lambda *shape, device: None)
    monkeypatch.setattr(ffi_xpu, "_torch_dlpack_device", lambda t: (int(DLDeviceType.kDLCPU), 0))
    with pytest.raises(RuntimeError, match="kDLOneAPI"):
        ffi_xpu.xpu_global_device_index(0)


@_requires_torch_xpu_module
def test_xpu_device_from_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    assert torch is not None
    monkeypatch.setattr(ffi_xpu, "xpu_global_device_index", lambda index: index + 4)
    monkeypatch.setattr(torch.xpu, "current_device", lambda: 1)

    assert ffi_xpu.xpu_device_from_torch(torch.device("xpu", 0)) == tvm_ffi.device("oneapi", 4)
    assert ffi_xpu.xpu_device_from_torch("xpu:2") == tvm_ffi.device("oneapi", 6)
    # no index -> torch's current device
    dev = ffi_xpu.xpu_device_from_torch("xpu")
    assert isinstance(dev, core.Device)
    assert dev == tvm_ffi.device("xpu", 5)


@_requires_torch_xpu_module
def test_xpu_device_from_torch_rejects_other_devices() -> None:
    with pytest.raises(ValueError, match="Expected an xpu device"):
        ffi_xpu.xpu_device_from_torch("cpu")
    with pytest.raises(ValueError, match="Expected an xpu device"):
        ffi_xpu.xpu_device_from_torch("cuda:0")


@_requires_torch_xpu_module
def test_uses_torch_xpu_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    assert torch is not None
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.xpu, "is_available", lambda: True)
    assert ffi_stream._uses_torch_xpu(None)
    assert ffi_stream._uses_torch_xpu(_noop_xpu_stream_context())
    assert not ffi_stream._uses_torch_xpu(object())

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert not ffi_stream._uses_torch_xpu(None)
    assert ffi_stream._uses_torch_xpu(_noop_xpu_stream_context())

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.xpu, "is_available", lambda: False)
    assert not ffi_stream._uses_torch_xpu(None)


@_requires_torch_xpu_module
@pytest.mark.parametrize("use_context", [False, True])
def test_use_torch_stream_xpu_sets_sycl_queue(
    monkeypatch: pytest.MonkeyPatch, use_context: bool
) -> None:
    assert torch is not None
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.xpu, "is_available", lambda: True)
    monkeypatch.setattr(torch.xpu, "current_stream", lambda device=None: _fake_xpu_stream(1))
    # torch xpu:1 is global SYCL device 3
    monkeypatch.setattr(ffi_xpu, "xpu_global_device_index", lambda index: index + 2)

    ffi_dev = tvm_ffi.device("oneapi", 3)
    context = _noop_xpu_stream_context() if use_context else None
    assert tvm_ffi.get_raw_stream(ffi_dev) == 0
    with tvm_ffi.use_torch_stream(context):
        assert tvm_ffi.get_raw_stream(ffi_dev) == FAKE_QUEUE
        # the torch index must not be used as the FFI device id
        assert tvm_ffi.get_raw_stream(tvm_ffi.device("oneapi", 1)) == 0
    assert tvm_ffi.get_raw_stream(ffi_dev) == 0


@_requires_torch_xpu_module
def test_use_torch_stream_xpu_nested(monkeypatch: pytest.MonkeyPatch) -> None:
    assert torch is not None
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.xpu, "is_available", lambda: True)
    monkeypatch.setattr(ffi_xpu, "xpu_global_device_index", lambda index: index)
    queues = iter([111, 222])
    monkeypatch.setattr(
        torch.xpu, "current_stream", lambda device=None: _fake_xpu_stream(0, next(queues))
    )

    ffi_dev = tvm_ffi.device("oneapi", 0)
    with tvm_ffi.use_torch_stream(_noop_xpu_stream_context()):
        assert tvm_ffi.get_raw_stream(ffi_dev) == 111
        with tvm_ffi.use_torch_stream(_noop_xpu_stream_context()):
            assert tvm_ffi.get_raw_stream(ffi_dev) == 222
        assert tvm_ffi.get_raw_stream(ffi_dev) == 111
    assert tvm_ffi.get_raw_stream(ffi_dev) == 0


@_requires_torch_xpu_module
def test_torch_xpu_raw_stream_api_is_resolved() -> None:
    assert torch is not None
    assert core._torch_xpu_get_current_raw_stream is getattr(
        torch._C, "_xpu_getCurrentRawStream", None
    )


@_requires_torch_xpu_module
def test_torch_xpu_current_raw_stream_uses_raw_api(monkeypatch: pytest.MonkeyPatch) -> None:
    assert torch is not None
    seen: list[Any] = []

    def fake_raw_stream(index: int) -> int:
        seen.append(index)
        return FAKE_QUEUE + index

    def fail_current_stream(device: Any = None) -> Any:
        raise AssertionError("the public API must not be used when the raw API exists")

    monkeypatch.setattr(core, "_torch_xpu_get_current_raw_stream", fake_raw_stream)
    monkeypatch.setattr(torch.xpu, "current_stream", fail_current_stream)
    assert core._torch_xpu_current_raw_stream(0) == FAKE_QUEUE
    assert core._torch_xpu_current_raw_stream(2) == FAKE_QUEUE + 2
    assert seen == [0, 2]


@_requires_torch_xpu_module
def test_torch_xpu_current_raw_stream_without_raw_api(monkeypatch: pytest.MonkeyPatch) -> None:
    assert torch is not None
    seen: list[Any] = []

    def fake_current_stream(device: Any = None) -> Any:
        seen.append(device)
        return _fake_xpu_stream(1)

    monkeypatch.setattr(core, "_torch_xpu_get_current_raw_stream", None)
    monkeypatch.setattr(torch.xpu, "current_stream", fake_current_stream)
    assert core._torch_xpu_current_raw_stream(1) == FAKE_QUEUE
    assert seen == [1]


def _fake_xpu_tensor_type(torch_index: int) -> type:
    """Create a CPU tensor type that reports itself as ``xpu:<torch_index>`` to the fallback setter."""
    assert torch is not None

    class FakeXPUTensor(torch.Tensor):
        @property
        def is_xpu(self) -> bool:  # ty: ignore[invalid-method-override]
            return True

        @property
        def device(self) -> torch.device:  # ty: ignore[invalid-method-override]
            return torch.device("xpu", torch_index)

    return FakeXPUTensor


@_requires_torch_xpu_module
@pytest.mark.parametrize("torch_index", [0, 3])
def test_torch_fallback_setter_captures_xpu_stream(
    monkeypatch: pytest.MonkeyPatch, torch_index: int
) -> None:
    assert torch is not None
    # force the Python fallback setter instead of torch's C exchange API; the setter is
    # chosen once per Python type, so the env var applies to the new class below
    monkeypatch.setenv("TVM_FFI_SKIP_DLPACK_C_EXCHANGE_API", "1")
    queried: list[int] = []

    def fake_raw_stream(index: int) -> int:
        queried.append(index)
        return FAKE_QUEUE

    monkeypatch.setattr(core, "_torch_xpu_get_current_raw_stream", fake_raw_stream)

    x = torch.empty(4).as_subclass(_fake_xpu_tensor_type(torch_index))
    mod = gen_check_stream_mod()
    # the stream is recorded under the exported device (cpu:0) but queried with the torch index
    mod.check_stream(x, int(DLDeviceType.kDLCPU), 0, FAKE_QUEUE)
    assert queried == [torch_index]


@pytest.mark.skipif(torch is None, reason="Requires torch")
def test_torch_fallback_setter_cpu_has_no_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    assert torch is not None
    monkeypatch.setenv("TVM_FFI_SKIP_DLPACK_C_EXCHANGE_API", "1")

    def fail_current_stream(device: Any = None) -> Any:
        raise AssertionError("XPU stream must not be queried for CPU tensors")

    if _HAS_TORCH_XPU_MODULE:
        monkeypatch.setattr(torch.xpu, "current_stream", fail_current_stream)
    mod = gen_check_stream_mod()
    x = torch.empty(4).as_subclass(_fresh_tensor_type())
    mod.check_stream(x, int(DLDeviceType.kDLCPU), 0, 0)


@_requires_xpu
def test_xpu_global_device_index_matches_to_dlpack() -> None:
    assert torch is not None
    for index in range(torch.xpu.device_count()):
        x = torch.empty(1, device=torch.device("xpu", index))
        ffi_x = tvm_ffi.from_dlpack(torch.utils.dlpack.to_dlpack(x))
        assert ffi_x.device.type == "oneapi"
        assert ffi_xpu.xpu_global_device_index(index) == ffi_x.device.index


@_requires_xpu
def test_torch_xpu_current_raw_stream_real_device() -> None:
    assert torch is not None
    assert core._torch_xpu_get_current_raw_stream is not None
    for index in range(torch.xpu.device_count()):
        stream = torch.xpu.Stream(device=index)
        with torch.xpu.stream(stream):
            assert core._torch_xpu_current_raw_stream(index) == stream.sycl_queue
        current = torch.xpu.current_stream(index).sycl_queue
        assert core._torch_xpu_current_raw_stream(index) == current


@_requires_xpu
def test_use_torch_stream_xpu_real_device() -> None:
    assert torch is not None
    stream = torch.xpu.Stream()
    ffi_dev = ffi_xpu.xpu_device_from_torch(stream.device)
    with tvm_ffi.use_torch_stream(torch.xpu.stream(stream)):
        assert tvm_ffi.get_raw_stream(ffi_dev) == stream.sycl_queue
    assert tvm_ffi.get_raw_stream(ffi_dev) == 0


@_requires_xpu
@pytest.mark.parametrize("skip_c_exchange_api", ["0", "1"])
def test_xpu_tensor_argument_propagates_stream(
    monkeypatch: pytest.MonkeyPatch, skip_c_exchange_api: str
) -> None:
    assert torch is not None
    # "0" exercises torch's C exchange API (or the addon), "1" the Python fallback
    monkeypatch.setenv("TVM_FFI_SKIP_DLPACK_C_EXCHANGE_API", skip_c_exchange_api)
    mod = gen_check_stream_mod()
    stream = torch.xpu.Stream()
    x = torch.empty(4, device=stream.device).as_subclass(_fresh_tensor_type())
    ffi_dev = ffi_xpu.xpu_device_from_torch(x.device)
    with torch.xpu.stream(stream):
        mod.check_stream(
            x, ffi_dev.dlpack_device_type(), ffi_dev.index, torch.xpu.current_stream().sycl_queue
        )
