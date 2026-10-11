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
"""Check the universal compatibility shim and its native PyTorch contract."""

from __future__ import annotations

import subprocess
import sys

import pytest
import torch
from torch_c_dlpack_ext import core


def test_native_api_and_free_threaded_import() -> None:
    """Keep the native API and GIL state intact across a fresh shim import."""
    subprocess.run(
        [
            sys.executable,
            "-W",
            "error::RuntimeWarning",
            "-c",
            """
import sys, sysconfig, torch
api = torch.Tensor.__dlpack_c_exchange_api__
import torch_c_dlpack_ext
assert torch_c_dlpack_ext.core.load_torch_c_dlpack_extension() is None
assert torch.Tensor.__dlpack_c_exchange_api__ is api
x = torch.arange(4)
assert torch.equal(torch.from_dlpack(x), x)
if sysconfig.get_config_var("Py_GIL_DISABLED"):
    assert not sys._is_gil_enabled()
""",
        ],
        check=True,
    )


def test_missing_native_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject PyTorch builds without the required native API."""
    monkeypatch.delattr(torch.Tensor, "__dlpack_c_exchange_api__", raising=False)
    monkeypatch.delattr(torch.Tensor, "__c_dlpack_exchange_api__", raising=False)
    with pytest.raises(ImportError, match="native DLPack exchange API"):
        core.load_torch_c_dlpack_extension()
