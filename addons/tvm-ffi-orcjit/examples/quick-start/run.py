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

"""Quick Start Example - Load and call functions from add.o.

This script demonstrates how to:
1. Create an ExecutionSession instance
2. Create a DynamicLibrary
3. Load a compiled object file
4. Get functions by name
5. Call them like regular Python functions
"""

import sys
from pathlib import Path

# Add the parent python directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))


from tvm_ffi_orcjit import ExecutionSession


def main() -> int:
    """Run the quick start example."""
    # Check if object file exists
    obj_file = Path("add.o")
    if not obj_file.exists():
        print(f"Error: {obj_file} not found!")
        print("Please run ./compile.sh first to compile the C++ code.")
        return 1

    print(f"Loading object file: {obj_file}")

    # Create execution session and dynamic library
    session = ExecutionSession()
    lib = session.create_library()
    lib.add(str(obj_file))

    print("✓ Object file loaded successfully\n")

    # Get and call the 'add' function
    print("=== Testing add function ===")
    add = lib.get_function("add")
    result = add(10, 20)
    print(f"add(10, 20) = {result}")
    assert result == 30, f"Expected 30, got {result}"

    # Get and call the 'multiply' function
    print("\n=== Testing multiply function ===")
    multiply = lib.get_function("multiply")
    result = multiply(7, 6)
    print(f"multiply(7, 6) = {result}")
    assert result == 42, f"Expected 42, got {result}"

    # Get and call the 'fibonacci' function
    print("\n=== Testing fibonacci function ===")
    fibonacci = lib.get_function("fibonacci")
    result = fibonacci(10)
    print(f"fibonacci(10) = {result}")
    assert result == 55, f"Expected 55, got {result}"

    # Get and call the 'concat' function
    print("\n=== Testing concat function ===")
    concat = lib.get_function("concat")
    result = concat("Hello, ", "World!")
    print(f"concat('Hello, ', 'World!') = '{result}'")
    assert result == "Hello, World!", f"Expected 'Hello, World!', got '{result}'"

    print("\n" + "=" * 50)
    print("✓ All tests passed successfully!")
    print("=" * 50)

    return 0


if __name__ == "__main__":
    sys.exit(main())
