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

"""Quick Start Example - Compile, load, and call functions via ORC JIT.

This script demonstrates how to:
1. Compile C/C++ source files to object files using tvm_ffi.cpp.build
2. Load them via the shared ORC JIT session (default_session().load_module)
3. Call the resulting module's functions like regular Python functions

Usage:
    python run.py          # Load C++ object file (add.o)
    python run.py --lang c # Load pure C object file (add_c.o)
"""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

import tvm_ffi.cpp
import tvm_ffi_orcjit as oj

SCRIPT_DIR = Path(__file__).resolve().parent


def _extra_cflags() -> list[str]:
    """Return extra compiler flags for the current platform."""
    machine = platform.machine()
    if machine in ("aarch64", "arm64"):
        return ["-mno-outline-atomics"]
    return []


def _build_object(lang: str) -> str:
    """Compile the source file to a relocatable object file."""
    if lang == "c":
        return tvm_ffi.cpp.build(
            name="add_c",
            sources=[str(SCRIPT_DIR / "add_c.c")],
            output="add_c.o",
            extra_cflags=_extra_cflags(),
        )
    else:
        return tvm_ffi.cpp.build(
            name="add",
            sources=[str(SCRIPT_DIR / "add.cc")],
            output="add.o",
            extra_cflags=_extra_cflags(),
        )


def _run_tests(obj_file: str, lang: str) -> None:
    """Load object file and run all test assertions.

    All JIT references (functions, module) are released automatically
    when this function returns.
    """
    print(f"Loading object file: {obj_file} (lang={lang})")

    # Load the object into a module on the shared session.
    mod = oj.default_session().load_module(obj_file)

    print("Object file loaded successfully\n")

    # Get and call the 'add' function
    print("=== Testing add function ===")
    result = mod.add(10, 20)
    print(f"add(10, 20) = {result}")
    assert result == 30, f"Expected 30, got {result}"

    # Get and call the 'multiply' function
    print("\n=== Testing multiply function ===")
    result = mod.multiply(7, 6)
    print(f"multiply(7, 6) = {result}")
    assert result == 42, f"Expected 42, got {result}"

    # Get and call the 'fibonacci' function
    print("\n=== Testing fibonacci function ===")
    result = mod.fibonacci(10)
    print(f"fibonacci(10) = {result}")
    assert result == 55, f"Expected 55, got {result}"

    print("\n" + "=" * 50)
    print("All tests passed successfully!")
    print("=" * 50)


def main() -> int:
    """Run the quick start example."""
    parser = argparse.ArgumentParser(description="Quick Start Example")
    parser.add_argument(
        "--lang",
        choices=["cpp", "c"],
        default="cpp",
        help="Language variant to load: 'cpp' for add.o (default), 'c' for add_c.o",
    )
    args = parser.parse_args()

    obj_file = _build_object(args.lang)
    _run_tests(obj_file, args.lang)
    return 0


if __name__ == "__main__":
    sys.exit(main())
