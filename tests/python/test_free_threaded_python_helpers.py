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

import sys

import pytest
import tvm_ffi


def _is_free_threaded_python() -> bool:
    return hasattr(sys, "_is_gil_enabled") and not sys._is_gil_enabled()


@pytest.mark.skipif(not _is_free_threaded_python(), reason="requires free-threaded Python")
def test_pyobject_deleter_handles_last_ref() -> None:
    drop_last_ref = getattr(tvm_ffi.core, "_testing_drop_last_ref_without_thread_state")
    drop_last_ref()
