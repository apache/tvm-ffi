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
# File: load/load_jax.py
# Step 1. Load `build/add_one_cuda.so`
import tvm_ffi
mod = tvm_ffi.load_module("build/add_one_cuda.so")

# Step 2. Register `mod.add_one` into JAX
import jax_tvm_ffi
jax_tvm_ffi.register_ffi_target("add_one", mod.add_one_cuda, platform="gpu")

# Step 3. Run `mod.add_one` with JAX
import jax
import jax.numpy as jnp
jax_device, *_ = jax.devices("gpu")
x = jnp.array([1, 2, 3, 4, 5], dtype=jnp.float32, device=jax_device)
y = jax.ffi.ffi_call(
    "add_one",  # name of the registered function
    jax.ShapeDtypeStruct(x.shape, x.dtype),  # shape and dtype of the output
    vmap_method="broadcast_all",
)(x)
print(y)
# [example.end]
