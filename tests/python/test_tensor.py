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

from types import ModuleType

import pytest

torch: ModuleType | None
try:
    import torch  # type: ignore[no-redef]
except ImportError:
    torch = None

import numpy as np
import tvm_ffi


def test_tensor_attributes() -> None:
    data = np.zeros((10, 8, 4, 2), dtype="int16")
    if not hasattr(data, "__dlpack__"):
        return
    x = tvm_ffi.from_dlpack(data)
    assert isinstance(x, tvm_ffi.Tensor)
    assert x.shape == (10, 8, 4, 2)
    assert x.strides == (64, 8, 2, 1)
    assert x.dtype == tvm_ffi.dtype("int16")
    assert x.device.dlpack_device_type() == tvm_ffi.DLDeviceType.kDLCPU
    assert x.device.index == 0
    x2 = np.from_dlpack(x)
    np.testing.assert_equal(x2, data)


def test_shape_object() -> None:
    shape = tvm_ffi.Shape((10, 8, 4, 2))
    assert isinstance(shape, tvm_ffi.Shape)
    assert shape == (10, 8, 4, 2)

    fecho = tvm_ffi.convert(lambda x: x)
    shape2: tvm_ffi.Shape = fecho(shape)
    assert shape2._tvm_ffi_cached_object.same_as(shape._tvm_ffi_cached_object)
    assert isinstance(shape2, tvm_ffi.Shape)
    assert isinstance(shape2, tuple)

    shape3: tvm_ffi.Shape = tvm_ffi.convert(shape)
    assert shape3._tvm_ffi_cached_object.same_as(shape._tvm_ffi_cached_object)
    assert isinstance(shape3, tvm_ffi.Shape)


@pytest.mark.skipif(torch is None, reason="Fast torch dlpack importer is not enabled")
def test_tensor_auto_dlpack() -> None:
    assert torch is not None
    x = torch.arange(128)
    fecho = tvm_ffi.get_global_func("testing.echo")
    y = fecho(x)
    assert isinstance(y, torch.Tensor)
    assert y.data_ptr() == x.data_ptr()
    assert y.dtype == x.dtype
    assert y.shape == x.shape
    assert y.device == x.device
    np.testing.assert_equal(y.numpy(), x.numpy())


def test_tensor_class_override() -> None:
    class MyTensor(tvm_ffi.Tensor):
        pass

    old_tensor = tvm_ffi.core._CLASS_TENSOR
    tvm_ffi.core._set_class_tensor(MyTensor)

    data = np.zeros((10, 8, 4, 2), dtype="int16")
    if not hasattr(data, "__dlpack__"):
        return
    x = tvm_ffi.from_dlpack(data)

    fecho = tvm_ffi.get_global_func("testing.echo")
    y = fecho(x)
    assert isinstance(y, MyTensor)
    tvm_ffi.core._set_class_tensor(old_tensor)


def test_tvm_ffi_tensor_compatible() -> None:
    class MyTensor:
        def __init__(self, tensor: tvm_ffi.Tensor) -> None:
            """Initialize the MyTensor."""
            self._tensor = tensor

        def __tvm_ffi_object__(self) -> tvm_ffi.Tensor:
            """Implement __tvm_ffi_object__ protocol."""
            return self._tensor

    data = np.zeros((10, 8, 4, 2), dtype="int32")
    if not hasattr(data, "__dlpack__"):
        return
    x = tvm_ffi.from_dlpack(data)
    y = MyTensor(x)
    fecho = tvm_ffi.get_global_func("testing.echo")
    z = fecho(y)
    assert z.__chandle__() == x.__chandle__()


@pytest.mark.skipif(
    torch is None or not torch.cuda.is_available() or torch.version.hip is None,
    reason="ROCm is not enabled in PyTorch",
)
def test_tensor_from_pytorch_rocm() -> None:
    assert torch is not None

    @tvm_ffi.register_global_func("testing.check_device", override=True)
    def _check_device(x: tvm_ffi.Tensor) -> str:
        return x.device.type

    # PyTorch uses device name "cuda" to represent ROCm device
    x = torch.randn(128, device="cuda")
    device_type = tvm_ffi.get_global_func("testing.check_device")(x)
    assert device_type == "rocm"


if __name__ == "__main__":
    pytest.main([__file__])
