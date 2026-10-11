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

import subprocess
import sys
import sysconfig

import pytest
import tvm_ffi


def _is_free_threaded_python() -> bool:
    return hasattr(sys, "_is_gil_enabled") and not sys._is_gil_enabled()


@pytest.mark.skipif(not _is_free_threaded_python(), reason="requires free-threaded Python")
def test_pyobject_deleter_handles_last_ref() -> None:
    drop_last_ref = getattr(tvm_ffi.core, "_testing_drop_last_ref_without_thread_state")
    drop_last_ref()


@pytest.mark.skipif(
    not sysconfig.get_config_var("Py_GIL_DISABLED"), reason="requires free-threaded Python"
)
def test_fresh_import_and_concurrent_json() -> None:
    subprocess.run(
        [
            sys.executable,
            "-W",
            "error::RuntimeWarning",
            "-c",
            """
import sys
from concurrent.futures import ThreadPoolExecutor
import tvm_ffi
from tvm_ffi.serialization import from_json_graph_str, to_json_graph_str
assert not sys._is_gil_enabled()
def roundtrip(i):
    value = tvm_ffi.convert([i, str(i)])
    result = from_json_graph_str(to_json_graph_str(value))
    assert list(result) == [i, str(i)]
with ThreadPoolExecutor(max_workers=2) as pool:
    list(pool.map(roundtrip, range(100)))
assert not sys._is_gil_enabled()
""",
        ],
        check=True,
    )
