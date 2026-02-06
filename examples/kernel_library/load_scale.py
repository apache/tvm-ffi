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
"""Load and call a scale kernel."""

# [load_and_call.begin]
import torch
import tvm_ffi

# Load the compiled shared library
mod = tvm_ffi.load_module("build/scale_kernel.so")

# Pre-allocate input and output tensors in PyTorch
x = torch.randn(1024, device="cuda", dtype=torch.float32)
y = torch.empty_like(x)

# Call the kernel — PyTorch tensors are auto-converted to TensorView
mod.scale(y, x, 2.0)

assert torch.allclose(y, x * 2.0)
# [load_and_call.end]
