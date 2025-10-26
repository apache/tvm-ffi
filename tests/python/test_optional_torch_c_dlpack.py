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

from pathlib import Path

import pytest
import tvm_ffi
from tvm_ffi._optional_torch_c_dlpack import build_torch_c_dlpack_extension


def test_build_torch_c_dlpack_extension() -> None:
    libpath = build_torch_c_dlpack_extension(build_directory="./build_test_dir")
    assert Path(libpath).exists()

    mod = tvm_ffi.load_module(libpath)
    assert mod is not None

    assert mod.TorchDLPackExchangeAPIPtr() is not None


if __name__ == "__main__":
    pytest.main([__file__])
