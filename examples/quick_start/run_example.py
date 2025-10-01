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
"""Quick start script to run tvm-ffi examples from prebuilt libraries."""

import numpy
import torch
import tvm_ffi


def run_add_one_cpu() -> None:
    """Load the add_one_cpu module and call the add_one_cpu function."""
    mod = tvm_ffi.load_module("build/add_one_cpu.so")

    x = numpy.array([1, 2, 3, 4, 5], dtype=numpy.float32)
    y = numpy.empty_like(x)
    # tvm-ffi automatically handles DLPack compatible tensors
    # torch tensors can be viewed as ffi::TensorView
    # in the background
    mod.add_one_cpu(x, y)
    print("numpy.result after add_one(x, y)")
    print(x)

    x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
    y = torch.empty_like(x)
    # tvm-ffi automatically handles DLPack compatible tensors
    # torch tensors can be viewed as ffi::TensorView
    # in the background
    mod.add_one_cpu(x, y)
    print("torch.result after add_one(x, y)")
    print(y)


def run_add_one_c() -> None:
    """Load the add_one_c module and call the add_one_c function."""
    mod = tvm_ffi.load_module("build/add_one_c.so")

    x = numpy.array([1, 2, 3, 4, 5], dtype=numpy.float32)
    y = numpy.empty_like(x)
    mod.add_one_c(x, y)
    print("numpy.result after add_one_c(x, y)")
    print(x)

    x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
    y = torch.empty_like(x)
    mod.add_one_c(x, y)
    print("torch.result after add_one_c(x, y)")
    print(y)


def run_add_one_cuda() -> None:
    """Load the add_one_cuda module and call the add_one_cuda function."""
    if not torch.cuda.is_available():
        return

    mod = tvm_ffi.load_module("build/add_one_cuda.so")
    x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32, device="cuda")
    y = torch.empty_like(x)

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        # tvm-ffi automatically handles DLPack compatible tensors
        # it also handles interactions with torch runtime
        # torch.cuda.current_stream() will be set and available via TVMFFIEnvGetStream
        # when calling the function
        mod.add_one_cuda(x, y)
    stream.synchronize()
    print("torch.result after mod.add_one_cuda(x, y)")
    print(y)


def main() -> None:
    """Run the quick start example."""
    run_add_one_cpu()
    run_add_one_c()
    run_add_one_cuda()


if __name__ == "__main__":
    main()
