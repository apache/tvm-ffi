#!/usr/bin/env python3
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
"""Example script for dynamic CUBIN loading.

This example demonstrates using lib_dynamic.so which accepts CUBIN data
dynamically at runtime.
"""

import sys
from pathlib import Path

import torch
from tvm_ffi import load_module


def main() -> int:  # noqa: PLR0915
    """Test the lib_dynamic.so library with dynamic CUBIN loading."""
    print("=" * 60)
    print("Example: Dynamic CUBIN Loading")
    print("=" * 60)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("[ERROR] CUDA is not available")
        return 1

    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch version: {torch.__version__}\n")

    # Load the library
    lib_path = Path(__file__).parent / "build" / "lib_dynamic.so"
    mod = load_module(str(lib_path))
    print(f"Loaded library: {lib_path}")

    # Read CUBIN file into memory
    cubin_path = Path(__file__).parent / "build" / "kernel.cubin"
    cubin_bytes = cubin_path.read_bytes()
    print(f"Read CUBIN from: {cubin_path} ({len(cubin_bytes)} bytes)")

    # Set CUBIN data to the module
    set_cubin = mod["set_cubin"]
    set_cubin(cubin_bytes)
    print("CUBIN module loaded successfully")

    # Get the kernel functions
    add_one = mod["add_one"]
    mul_two = mod["mul_two"]
    print("Loaded functions: add_one, mul_two")

    # Test add_one kernel
    print("\n[Test 1] add_one kernel")
    n = 2048
    x = torch.arange(n, dtype=torch.float32, device="cuda") * 0.1
    y = torch.empty(n, dtype=torch.float32, device="cuda")

    print(f"  Input shape: {x.shape}, device: {x.device}")
    add_one(x, y)

    # Verify results
    expected = x + 1
    if torch.allclose(y, expected):
        print(f"  [PASS] Verified {n} elements correctly")
    else:
        print(f"  [FAIL] Verification failed, max error: {(y - expected).abs().max().item()}")
        return 1

    # Test mul_two kernel
    print("\n[Test 2] mul_two kernel")
    n = 1024
    x = torch.arange(n, dtype=torch.float32, device="cuda") * 0.25
    y = torch.empty(n, dtype=torch.float32, device="cuda")

    print(f"  Input shape: {x.shape}, device: {x.device}")
    mul_two(x, y)

    # Verify results
    expected = x * 2
    if torch.allclose(y, expected):
        print(f"  [PASS] Verified {n} elements correctly")
    else:
        print(f"  [FAIL] Verification failed, max error: {(y - expected).abs().max().item()}")
        return 1

    # Test chained execution
    print("\n[Test 3] Chained execution: (x + 1) * 2")
    n = 512
    x = torch.full((n,), 5.0, dtype=torch.float32, device="cuda")
    temp = torch.empty(n, dtype=torch.float32, device="cuda")
    y = torch.empty(n, dtype=torch.float32, device="cuda")

    print(f"  Initial value: {x[0].item()}")
    add_one(x, temp)  # temp = x + 1 = 6
    mul_two(temp, y)  # y = temp * 2 = 12

    expected = 12.0
    if torch.allclose(y, torch.tensor(expected, device="cuda")):
        print(f"  [PASS] Result: {y[0].item()}")
    else:
        print(f"  [FAIL] Expected {expected}, got {y[0].item()}")
        return 1

    print("\n[PASS] All tests passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
