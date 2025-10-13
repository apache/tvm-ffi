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

import ctypes
import pickle

import pytest
import tvm_ffi
from tvm_ffi import DLDeviceType


def test_device() -> None:
    device = tvm_ffi.Device("cuda", 0)
    assert device.dlpack_device_type() == tvm_ffi.DLDeviceType.kDLCUDA
    assert device.index == 0
    assert str(device) == "cuda:0"
    assert device.__repr__() == "device(type='cuda', index=0)"


def test_device_from_str() -> None:
    device = tvm_ffi.device("ext_dev:0")
    assert device.dlpack_device_type() == tvm_ffi.DLDeviceType.kDLExtDev
    assert device.index == 0
    assert str(device) == "ext_dev:0"
    assert device.__repr__() == "device(type='ext_dev', index=0)"


@pytest.mark.parametrize(
    "dev_str, expected_device_type, expect_device_id",
    [
        ("cpu", DLDeviceType.kDLCPU, 0),
        ("cuda", DLDeviceType.kDLCUDA, 0),
        ("cuda:0", DLDeviceType.kDLCUDA, 0),
        ("cuda:3", DLDeviceType.kDLCUDA, 3),
        ("metal:2", DLDeviceType.kDLMetal, 2),
    ],
)
def test_device_dlpack_device_type(
    dev_str: str,
    expected_device_type: DLDeviceType,
    expect_device_id: int,
) -> None:
    dev = tvm_ffi.device(dev_str)
    assert dev.dlpack_device_type() == expected_device_type
    assert dev.index == expect_device_id


@pytest.mark.parametrize(
    "dev_type, dev_id, expected_device_type, expect_device_id",
    [
        ("cpu", 0, DLDeviceType.kDLCPU, 0),
        ("cuda", 0, DLDeviceType.kDLCUDA, 0),
        (DLDeviceType.kDLCUDA, 0, DLDeviceType.kDLCUDA, 0),
        ("cuda", 3, DLDeviceType.kDLCUDA, 3),
        (DLDeviceType.kDLMetal, 2, DLDeviceType.kDLMetal, 2),
    ],
)
def test_device_with_dev_id(
    dev_type: str | DLDeviceType,
    dev_id: int,
    expected_device_type: DLDeviceType,
    expect_device_id: int,
) -> None:
    dev = tvm_ffi.device(dev_type, dev_id)
    assert dev.dlpack_device_type() == expected_device_type
    assert dev.index == expect_device_id


@pytest.mark.parametrize("dev_type, dev_id", [("cpu:0:0", None), ("cpu:?", None), ("cpu:", None)])
def test_deive_type_error(dev_type: str, dev_id: int | None) -> None:
    with pytest.raises(ValueError):
        tvm_ffi.device(dev_type, dev_id)


def test_deive_id_error() -> None:
    with pytest.raises(TypeError):
        tvm_ffi.device("cpu", "?")  # type: ignore[arg-type]


def test_device_pickle() -> None:
    device = tvm_ffi.device("cuda", 0)
    device_pickled = pickle.loads(pickle.dumps(device))
    assert device_pickled.dlpack_device_type() == device.dlpack_device_type()
    assert device_pickled.index == device.index


def test_device_class_override() -> None:
    class MyDevice(tvm_ffi.Device):
        pass

    old_device = tvm_ffi.core._CLASS_DEVICE
    tvm_ffi.core._set_class_device(MyDevice)

    device = tvm_ffi.device("cuda", 0)
    assert isinstance(device, MyDevice)
    tvm_ffi.core._set_class_device(old_device)


def test_cuda_stream_handling() -> None:
    class MyDummyStream:
        def __init__(self, stream: int) -> None:
            self.stream = stream

        def __cuda_stream__(self) -> tuple[str, int]:
            return ("cuda", self.stream)

    stream = MyDummyStream(1)
    echo = tvm_ffi.get_global_func("testing.echo")
    y = echo(stream)
    assert isinstance(y, ctypes.c_void_p)
    assert y.value == 1
