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

"""Tests for lazy container DLPack conversion when DLPack exchange API is active."""

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
    """Array<Tensor> stays as Array; element access converts to torch.Tensor."""
    assert torch is not None
    x = torch.arange(8, dtype=torch.float32)
    f = tvm_ffi.get_global_func("testing.make_array_with_tensor")
    result = f(x)
    assert isinstance(result, tvm_ffi.Array)
    assert len(result) == 1
    elem = result[0]
    assert isinstance(elem, torch.Tensor)
    assert elem.data_ptr() == x.data_ptr()


def test_array_mixed() -> None:
    """Array with Tensor + int + string: lazy conversion on access."""
    assert torch is not None
    x = torch.arange(4, dtype=torch.float32)
    f = tvm_ffi.get_global_func("testing.make_array_with_mixed")
    result = f(x, 42)
    assert isinstance(result, tvm_ffi.Array)
    assert len(result) == 3
    assert isinstance(result[0], torch.Tensor)
    assert result[0].data_ptr() == x.data_ptr()
    assert result[1] == 42
    assert result[2] == "hello"


def test_array_nested() -> None:
    """Nested Array<Array<Tensor>>: inner arrays also get tagged."""
    assert torch is not None
    x = torch.arange(4, dtype=torch.float32)
    f = tvm_ffi.get_global_func("testing.make_nested_array_with_tensor")
    result = f(x)
    assert isinstance(result, tvm_ffi.Array)
    assert len(result) == 2
    # First element is inner array
    inner = result[0]
    assert isinstance(inner, tvm_ffi.Array)
    assert len(inner) == 2
    assert isinstance(inner[0], torch.Tensor)
    assert inner[0].data_ptr() == x.data_ptr()
    assert inner[1] == 42
    # Second element is a tensor
    assert isinstance(result[1], torch.Tensor)
    assert result[1].data_ptr() == x.data_ptr()


def test_list_with_tensor() -> None:
    """List<Any> with tensor: stays as List, elements convert on access."""
    assert torch is not None
    x = torch.arange(4, dtype=torch.float32)
    f = tvm_ffi.get_global_func("testing.make_list_with_tensor")
    result = f(x, 7)
    assert isinstance(result, tvm_ffi.List)
    assert len(result) == 2
    assert isinstance(result[0], torch.Tensor)
    assert result[0].data_ptr() == x.data_ptr()
    assert result[1] == 7


def test_map_with_tensor() -> None:
    """Map<String, Any> with tensor value: stays as Map, values convert on access."""
    assert torch is not None
    x = torch.arange(4, dtype=torch.float32)
    f = tvm_ffi.get_global_func("testing.make_map_with_tensor")
    result = f(x)
    assert isinstance(result, tvm_ffi.Map)
    assert len(result) == 3
    assert isinstance(result["tensor"], torch.Tensor)
    assert result["tensor"].data_ptr() == x.data_ptr()
    assert result["value"] == 42
    assert result["name"] == "test"


def test_dict_with_tensor() -> None:
    """Dict<String, Any> with tensor value: stays as Dict, values convert on access."""
    assert torch is not None
    x = torch.arange(4, dtype=torch.float32)
    f = tvm_ffi.get_global_func("testing.make_dict_with_tensor")
    result = f(x)
    assert isinstance(result, tvm_ffi.Dict)
    assert len(result) == 2
    assert isinstance(result["tensor"], torch.Tensor)
    assert result["tensor"].data_ptr() == x.data_ptr()
    assert result["value"] == 42


def test_nested_map_with_array() -> None:
    """Nested Map with Array values: all containers tagged, lazy conversion on access."""
    assert torch is not None
    x1 = torch.arange(4, dtype=torch.float32)
    x2 = torch.arange(8, dtype=torch.int32)
    f = tvm_ffi.get_global_func("testing.make_nested_map_with_tensor")
    result = f(x1, x2)
    assert isinstance(result, tvm_ffi.Map)
    # "array" -> Array with tagged tensors
    arr = result["array"]
    assert isinstance(arr, tvm_ffi.Array)
    assert len(arr) == 2
    assert isinstance(arr[0], torch.Tensor)
    assert isinstance(arr[1], torch.Tensor)
    # "map" -> nested Map
    inner_map = result["map"]
    assert isinstance(inner_map, tvm_ffi.Map)
    assert isinstance(inner_map["t"], torch.Tensor)
    # "scalar" -> int
    assert result["scalar"] == 99


def test_empty_array() -> None:
    """Empty Array with torch input: stays as empty Array."""
    assert torch is not None
    x = torch.arange(4, dtype=torch.float32)
    f = tvm_ffi.get_global_func("testing.make_empty_array_with_tensor_input")
    result = f(x)
    assert isinstance(result, tvm_ffi.Array)
    assert len(result) == 0


def test_no_torch_input_no_conversion() -> None:
    """Without torch tensor input, containers stay as FFI types with no tag."""
    x = tvm_ffi.from_dlpack(np.arange(4, dtype="float32"))
    f = tvm_ffi.get_global_func("testing.make_array_with_tensor")
    result = f(x)
    # No torch input, so no dlpack API set -> normal FFI Array return
    assert isinstance(result, tvm_ffi.Array)
    assert isinstance(result[0], tvm_ffi.Tensor)


def test_data_correctness() -> None:
    """Verify tensor data is correct after lazy container conversion."""
    assert torch is not None
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    f = tvm_ffi.get_global_func("testing.make_array_with_tensor")
    result = f(x)
    assert isinstance(result, tvm_ffi.Array)
    elem = result[0]
    assert isinstance(elem, torch.Tensor)
    np.testing.assert_equal(elem.numpy(), x.numpy())


def test_echo_bare_tensor_unchanged() -> None:
    """Existing behavior: bare tensor return still works."""
    assert torch is not None
    x = torch.arange(128)
    fecho = tvm_ffi.get_global_func("testing.echo")
    y = fecho(x)
    assert isinstance(y, torch.Tensor)
    assert y.data_ptr() == x.data_ptr()


def test_container_preserves_identity() -> None:
    """Lazy conversion preserves container identity (can be passed back to FFI)."""
    assert torch is not None
    x = torch.arange(4, dtype=torch.float32)
    f = tvm_ffi.get_global_func("testing.make_array_with_tensor")
    result = f(x)
    assert isinstance(result, tvm_ffi.Array)
    # Pass container back to FFI (echo)
    fecho = tvm_ffi.get_global_func("testing.echo")
    echoed = fecho(result)
    assert isinstance(echoed, tvm_ffi.Array)
    assert isinstance(echoed[0], torch.Tensor)
    assert echoed[0].data_ptr() == x.data_ptr()


def test_mutable_list_shared_semantics() -> None:
    """Lazy conversion preserves mutable list shared-reference semantics."""
    assert torch is not None
    x = torch.arange(4, dtype=torch.float32)
    f = tvm_ffi.get_global_func("testing.make_list_with_tensor")
    result = f(x, 7)
    assert isinstance(result, tvm_ffi.List)
    # The result is the actual FFI List, not a detached copy
    assert result.same_as(result)
