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

import cutlass
import pytest
import torch
from cutlass import cute


@cute.kernel
def device_add_one(a: cute.Tensor) -> None:
    a[0] += cutlass.Float16(1.0)


@cute.jit
def add_one(a: cute.Tensor) -> None:
    device_add_one(a).launch(grid=(1, 1, 1), block=(1, 1, 1))


def test_cute_dsl_compile() -> None:
    dtype = cutlass.Float16
    alignment_bytes = 16
    divisibility = alignment_bytes * 8 // dtype.width

    fake_tensor = cute.runtime.make_fake_compact_tensor(
        dtype,
        (cute.SymInt(), cute.SymInt(divisibility=divisibility)),
        stride_order=(1, 0),
        assumed_align=alignment_bytes,
    )

    compiled_add_one = cute.compile(add_one, fake_tensor, options="--enable-tvm-ffi")
    # Pass in tensor with correct divisibility
    a = torch.zeros((4, 16), device="cuda", dtype=torch.float16)
    compiled_add_one(a)

    # Pass in tensor with incorrect divisibility
    with pytest.raises(ValueError):
        b = torch.zeros((4, 18), device="cuda", dtype=torch.float16)
        compiled_add_one(b)
