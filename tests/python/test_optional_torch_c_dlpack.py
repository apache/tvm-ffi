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

import ctypes
import subprocess
import sys
from pathlib import Path

import pytest
import tvm_ffi

IS_WINDOWS = sys.platform.startswith("win")


def test_build_torch_c_dlpack_extension() -> None:
    build_script = Path(tvm_ffi.__file__).parent / "utils" / "_build_optional_c_dlpack.py"
    subprocess.run(
        [sys.executable, str(build_script), "--build_dir", "./build_test_dir"], check=True
    )

    lib_path = str(
        Path(
            "./build_test_dir/libtorch_c_dlpack_addon.{}".format("dll" if IS_WINDOWS else "so")
        ).resolve()
    )
    assert Path(lib_path).exists()

    lib = ctypes.CDLL(lib_path)
    func = lib.TorchDLPackExchangeAPIPtr
    func.restype = ctypes.c_int64
    ptr = func()
    assert ptr != 0


def test_parallel_build() -> None:
    build_script = Path(tvm_ffi.__file__).parent / "utils" / "_build_optional_c_dlpack.py"
    num_processes = 4
    build_dir = "./build_test_dir_parallel"
    processes = []
    for i in range(num_processes):
        p = subprocess.Popen([sys.executable, str(build_script), "--build_dir", build_dir])
        processes.append((p, build_dir))

    for p, build_dir in processes:
        p.wait()
        assert p.returncode == 0
    lib_path = str(
        Path(
            "{}/libtorch_c_dlpack_addon.{}".format(build_dir, "dll" if IS_WINDOWS else "so")
        ).resolve()
    )
    assert Path(lib_path).exists()


if __name__ == "__main__":
    pytest.main([__file__])
