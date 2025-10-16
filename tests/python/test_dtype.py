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

import pickle
from typing import Any

import numpy as np
import pytest
import tvm_ffi


def test_dtype() -> None:
    float32 = tvm_ffi.dtype("float32")
    assert float32.__repr__() == "dtype('float32')"
    assert type(float32) == tvm_ffi.dtype
    x = np.array([1, 2, 3], dtype=float32)
    assert x.dtype == float32


@pytest.mark.parametrize(
    "dtype_str, expected_size",
    [
        ("float32", 4),
        ("float32x4", 16),
        ("float8_e5m2x4", 4),
        ("float6_e2m3fnx4", 3),
        ("float4_e2m1fnx4", 2),
        ("uint8", 1),
        ("bool", 1),
    ],
)
def test_dtype_itemsize(dtype_str: str, expected_size: int) -> None:
    dtype = tvm_ffi.dtype(dtype_str)
    assert dtype.itemsize == expected_size


@pytest.mark.parametrize("dtype_str", ["int32xvscalex4"])
def test_dtype_itemmize_error(dtype_str: str) -> None:
    with pytest.raises(ValueError):
        tvm_ffi.dtype(dtype_str).itemsize


@pytest.mark.parametrize(
    "dtype_str",
    [
        "float32",
        "float32x4",
        "float8_e5m2x4",
        "float6_e2m3fnx4",
        "float4_e2m1fnx4",
        "uint8",
        "bool",
    ],
)
def test_dtype_pickle(dtype_str: str) -> None:
    dtype = tvm_ffi.dtype(dtype_str)
    dtype_pickled = pickle.loads(pickle.dumps(dtype))
    assert dtype_pickled.type_code == dtype.type_code
    assert dtype_pickled.bits == dtype.bits
    assert dtype_pickled.lanes == dtype.lanes


@pytest.mark.parametrize("dtype_str", ["float32", "bool"])
def test_dtype_with_lanes(dtype_str: str) -> None:
    dtype = tvm_ffi.dtype(dtype_str)
    dtype_with_lanes = dtype.with_lanes(4)
    assert dtype_with_lanes.type_code == dtype.type_code
    assert dtype_with_lanes.bits == dtype.bits
    assert dtype_with_lanes.lanes == 4


_fecho = tvm_ffi.get_global_func("testing.echo")


def _check_dtype(dtype: Any, code: int, bits: int, lanes: int) -> None:
    echo_dtype = _fecho(dtype)
    assert echo_dtype.type_code == code
    assert echo_dtype.bits == bits
    assert echo_dtype.lanes == lanes
    converted_dtype = tvm_ffi.convert(dtype)
    assert converted_dtype.type_code == code
    assert converted_dtype.bits == bits
    assert converted_dtype.lanes == lanes


def test_torch_dtype_conversion() -> None:
    torch = pytest.importorskip("torch")
    _check_dtype(torch.int8, 0, 8, 1)
    _check_dtype(torch.short, 0, 16, 1)
    _check_dtype(torch.int16, 0, 16, 1)
    _check_dtype(torch.int32, 0, 32, 1)
    _check_dtype(torch.int, 0, 32, 1)
    _check_dtype(torch.int64, 0, 64, 1)
    _check_dtype(torch.long, 0, 64, 1)
    _check_dtype(torch.uint8, 1, 8, 1)
    _check_dtype(torch.uint16, 1, 16, 1)
    _check_dtype(torch.uint32, 1, 32, 1)
    _check_dtype(torch.uint64, 1, 64, 1)
    _check_dtype(torch.float16, 2, 16, 1)
    _check_dtype(torch.half, 2, 16, 1)
    _check_dtype(torch.float32, 2, 32, 1)
    _check_dtype(torch.float, 2, 32, 1)
    _check_dtype(torch.float64, 2, 64, 1)
    _check_dtype(torch.double, 2, 64, 1)
    _check_dtype(torch.bfloat16, 4, 16, 1)
    _check_dtype(torch.bool, 6, 8, 1)
    _check_dtype(torch.float8_e4m3fn, 10, 8, 1)
    _check_dtype(torch.float8_e4m3fnuz, 11, 8, 1)
    _check_dtype(torch.float8_e5m2, 12, 8, 1)
    _check_dtype(torch.float8_e5m2fnuz, 13, 8, 1)
    if hasattr(torch, "float8_e8m0fnu"):
        _check_dtype(torch.float8_e8m0fnu, 14, 8, 1)


def test_numpy_dtype_conversion() -> None:
    np = pytest.importorskip("numpy")
    _check_dtype(np.dtype(np.int8), 0, 8, 1)
    _check_dtype(np.dtype(np.int16), 0, 16, 1)
    _check_dtype(np.dtype(np.int32), 0, 32, 1)
    _check_dtype(np.dtype(np.int64), 0, 64, 1)
    _check_dtype(np.dtype(np.uint8), 1, 8, 1)
    _check_dtype(np.dtype(np.uint16), 1, 16, 1)
    _check_dtype(np.dtype(np.uint32), 1, 32, 1)
    _check_dtype(np.dtype(np.uint64), 1, 64, 1)
    _check_dtype(np.dtype(np.float16), 2, 16, 1)
    _check_dtype(np.dtype(np.float32), 2, 32, 1)
    _check_dtype(np.dtype(np.float64), 2, 64, 1)


def test_ml_dtypes_dtype_conversion() -> None:
    np = pytest.importorskip("numpy")
    ml_dtypes = pytest.importorskip("ml_dtypes")
    _check_dtype(np.dtype(ml_dtypes.int2), 0, 2, 1)
    _check_dtype(np.dtype(ml_dtypes.int4), 0, 4, 1)
    _check_dtype(np.dtype(ml_dtypes.uint2), 1, 2, 1)
    _check_dtype(np.dtype(ml_dtypes.uint4), 1, 4, 1)
    _check_dtype(np.dtype(ml_dtypes.bfloat16), 4, 16, 1)
    _check_dtype(np.dtype(ml_dtypes.float8_e3m4), 7, 8, 1)
    _check_dtype(np.dtype(ml_dtypes.float8_e4m3), 8, 8, 1)
    _check_dtype(np.dtype(ml_dtypes.float8_e4m3b11fnuz), 9, 8, 1)
    _check_dtype(np.dtype(ml_dtypes.float8_e4m3fn), 10, 8, 1)
    _check_dtype(np.dtype(ml_dtypes.float8_e4m3fnuz), 11, 8, 1)
    _check_dtype(np.dtype(ml_dtypes.float8_e5m2), 12, 8, 1)
    _check_dtype(np.dtype(ml_dtypes.float8_e5m2fnuz), 13, 8, 1)
    _check_dtype(np.dtype(ml_dtypes.float8_e8m0fnu), 14, 8, 1)
    _check_dtype(np.dtype(ml_dtypes.float6_e2m3fn), 15, 6, 1)
    _check_dtype(np.dtype(ml_dtypes.float6_e3m2fn), 16, 6, 1)
    _check_dtype(np.dtype(ml_dtypes.float4_e2m1fn), 17, 4, 1)
