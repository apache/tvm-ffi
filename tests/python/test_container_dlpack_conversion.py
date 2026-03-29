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

"""Tests for recursive container-to-native conversion when DLPack exchange API is active."""

from __future__ import annotations

import numpy as np
import pytest

try:
    import torch
    import torch.version
except ImportError:
    torch = None  # ty: ignore[invalid-assignment]

import tvm_ffi

pytestmark = pytest.mark.skipif(torch is None, reason="torch is not installed")


def test_array_tensor_only() -> None:
    """Array<Tensor> returned as list[torch.Tensor]."""
    assert torch is not None
    x = torch.arange(8, dtype=torch.float32)
    f = tvm_ffi.get_global_func("testing.make_array_with_tensor")
    result = f(x)
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], torch.Tensor)
    assert result[0].data_ptr() == x.data_ptr()


def test_array_mixed() -> None:
    """Array with Tensor + int + string elements."""
    assert torch is not None
    x = torch.arange(4, dtype=torch.float32)
    f = tvm_ffi.get_global_func("testing.make_array_with_mixed")
    result = f(x, 42)
    assert isinstance(result, list)
    assert len(result) == 3
    assert isinstance(result[0], torch.Tensor)
    assert result[0].data_ptr() == x.data_ptr()
    assert result[1] == 42
    assert result[2] == "hello"


def test_array_nested() -> None:
    """Nested Array<Array<Tensor>> -> list[list[...]]."""
    assert torch is not None
    x = torch.arange(4, dtype=torch.float32)
    f = tvm_ffi.get_global_func("testing.make_nested_array_with_tensor")
    result = f(x)
    assert isinstance(result, list)
    assert len(result) == 2
    # First element is inner array [tensor, 42]
    assert isinstance(result[0], list)
    assert len(result[0]) == 2
    assert isinstance(result[0][0], torch.Tensor)
    assert result[0][0].data_ptr() == x.data_ptr()
    assert result[0][1] == 42
    # Second element is a tensor
    assert isinstance(result[1], torch.Tensor)
    assert result[1].data_ptr() == x.data_ptr()


def test_list_with_tensor() -> None:
    """List<Any> with tensor -> list."""
    assert torch is not None
    x = torch.arange(4, dtype=torch.float32)
    f = tvm_ffi.get_global_func("testing.make_list_with_tensor")
    result = f(x, 7)
    assert isinstance(result, list)
    assert len(result) == 2
    assert isinstance(result[0], torch.Tensor)
    assert result[0].data_ptr() == x.data_ptr()
    assert result[1] == 7


def test_map_with_tensor() -> None:
    """Map<String, Any> with tensor value -> dict."""
    assert torch is not None
    x = torch.arange(4, dtype=torch.float32)
    f = tvm_ffi.get_global_func("testing.make_map_with_tensor")
    result = f(x)
    assert isinstance(result, dict)
    assert len(result) == 3
    assert isinstance(result["tensor"], torch.Tensor)
    assert result["tensor"].data_ptr() == x.data_ptr()
    assert result["value"] == 42
    assert result["name"] == "test"


def test_dict_with_tensor() -> None:
    """Dict<String, Any> with tensor value -> dict."""
    assert torch is not None
    x = torch.arange(4, dtype=torch.float32)
    f = tvm_ffi.get_global_func("testing.make_dict_with_tensor")
    result = f(x)
    assert isinstance(result, dict)
    assert len(result) == 2
    assert isinstance(result["tensor"], torch.Tensor)
    assert result["tensor"].data_ptr() == x.data_ptr()
    assert result["value"] == 42


def test_nested_map_with_array() -> None:
    """Nested Map with Array values -> dict with list values."""
    assert torch is not None
    x1 = torch.arange(4, dtype=torch.float32)
    x2 = torch.arange(8, dtype=torch.int32)
    f = tvm_ffi.get_global_func("testing.make_nested_map_with_tensor")
    result = f(x1, x2)
    assert isinstance(result, dict)
    # "array" -> list of tensors
    assert isinstance(result["array"], list)
    assert len(result["array"]) == 2
    assert isinstance(result["array"][0], torch.Tensor)
    assert isinstance(result["array"][1], torch.Tensor)
    # "map" -> nested dict
    assert isinstance(result["map"], dict)
    assert isinstance(result["map"]["t"], torch.Tensor)
    # "scalar" -> int
    assert result["scalar"] == 99


def test_empty_array() -> None:
    """Empty Array with torch input -> empty list."""
    assert torch is not None
    x = torch.arange(4, dtype=torch.float32)
    f = tvm_ffi.get_global_func("testing.make_empty_array_with_tensor_input")
    result = f(x)
    assert isinstance(result, list)
    assert len(result) == 0


def test_no_torch_input_no_conversion() -> None:
    """Without torch tensor input, containers stay as FFI types."""
    x = tvm_ffi.from_dlpack(np.arange(4, dtype="float32"))
    f = tvm_ffi.get_global_func("testing.make_array_with_tensor")
    result = f(x)
    # No torch input, so no dlpack API set -> normal FFI Array return
    assert isinstance(result, tvm_ffi.Array)
    assert isinstance(result[0], tvm_ffi.Tensor)


def test_data_correctness() -> None:
    """Verify tensor data is correct after container conversion."""
    assert torch is not None
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    f = tvm_ffi.get_global_func("testing.make_array_with_tensor")
    result = f(x)
    assert isinstance(result, list)
    assert isinstance(result[0], torch.Tensor)
    np.testing.assert_equal(result[0].numpy(), x.numpy())


def test_echo_bare_tensor_unchanged() -> None:
    """Existing behavior: bare tensor return still works."""
    assert torch is not None
    x = torch.arange(128)
    fecho = tvm_ffi.get_global_func("testing.echo")
    y = fecho(x)
    assert isinstance(y, torch.Tensor)
    assert y.data_ptr() == x.data_ptr()
