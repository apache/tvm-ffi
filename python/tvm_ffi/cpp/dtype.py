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
"""Utilities for C++ dtype conversion."""

from __future__ import annotations

import functools
from typing import Any, Literal

CPU_DTYPE_MAP = {
    "int8": "int8_t",
    "int16": "int16_t",
    "int32": "int32_t",
    "int64": "int64_t",
    "uint8": "uint8_t",
    "uint16": "uint16_t",
    "uint32": "uint32_t",
    "uint64": "uint64_t",
    "float32": "float",
    "float64": "double",
    "bool": "bool",
}

CUDA_DTYPE_MAP = {
    "float16": "__half",
    "bfloat16": "__nv_bfloat16",
    "float8_e4m3fn": "__nv_fp8_e4m3",
    # "float8_e4m3fnuz": "__nv_fp8_e4m3",
    "float8_e5m2": "__nv_fp8_e5m2",
    # "float8_e5m2fnuz": "__nv_fp8_e5m2",
    "float8_e8m0fnu": "__nv_fp8_e8m0",
    "float4_e2m1": "__nv_fp4_e2m1",
    "float4_e2m1fn_x2": "__nv_fp4x2_e2m1",
}

ROCM_DTYPE_MAP = {
    "float16": "__half",
    "bfloat16": "__hip_bfloat16",
    "float8_e4m3fn": "__hip_fp8_e4m3",
    "float8_e4m3fnuz": "__hip_fp8_e4m3_fnuz",
    "float8_e5m2": "__hip_fp8_e5m2",
    "float8_e5m2fnuz": "__hip_fp8_e5m2_fnuz",
    "float4_e2m1": "__hip_fp4_e2m1",
    "float4_e2m1fn_x2": "__hip_fp4x2_e2m1",
}


@functools.lru_cache(maxsize=None)
def _determine_backend_once() -> Literal["cpu", "cuda", "rocm"]:
    try:
        import torch  # noqa: PLC0415
        import torch.version  # noqa: PLC0415

        if torch.cuda.is_available():
            if torch.version.cuda is not None:
                return "cuda"
            elif torch.version.hip is not None:
                return "rocm"
    except ImportError:
        pass
    return "cpu"


def to_cpp_dtype(dtype_str: str | Any) -> str:
    """Convert a dtype to its corresponding C++ dtype string.

    Parameters
    ----------
    dtype_str : `str` or `torch.dtype`
        The dtype string or object to convert.

    Returns
    -------
    str
        The corresponding C++ dtype string.

    """
    if not isinstance(dtype_str, str):
        dtype_str = str(dtype_str)
    if dtype_str.startswith("torch."):
        dtype_str = dtype_str[6:]
    cpp_str = CPU_DTYPE_MAP.get(dtype_str)
    if cpp_str is not None:
        return cpp_str
    backend = _determine_backend_once()
    if backend in ("cuda", "rocm"):
        dtype_map = CUDA_DTYPE_MAP if backend == "cuda" else ROCM_DTYPE_MAP
        cpp_str = dtype_map.get(dtype_str)
        if cpp_str is not None:
            return cpp_str
    raise ValueError(f"Unsupported dtype string: {dtype_str} for {backend = }")
