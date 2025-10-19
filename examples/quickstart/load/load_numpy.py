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
# fmt: off
# ruff: noqa
# mypy: ignore-errors
# [example.begin]
# File: load/load_numpy.py
import tvm_ffi
mod = tvm_ffi.load_module("build/add_one_cpu.so")

import numpy as np
x = np.array([1, 2, 3, 4, 5], dtype=np.float32)
y = np.empty_like(x)
mod.add_one_cpu(x, y)
print(y)
# [example.end]
