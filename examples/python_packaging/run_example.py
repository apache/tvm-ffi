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
# specific language governing permissions and limitations.
# Base logic to load library for extension package
"""Run functions from the example packaged tvm-ffi extension."""

import traceback

import my_ffi_extension


def run_add_two() -> None:
    """Invoke add_two from the extension and print the result."""
    print("=========== Example 1: add_two ===========")
    print(my_ffi_extension.LIB.add_two(1))


def run_add_one() -> None:
    """Invoke add_one from the extension and print the result."""
    print("=========== Example 2: add_one ===========")
    print(my_ffi_extension.add_one(3))


def run_raise_error() -> None:
    """Invoke raise_error from the extension to demonstrate error handling."""
    print("=========== Example 3: raise_error ===========")
    try:
        my_ffi_extension.raise_error("This is an error")
    except RuntimeError:
        traceback.print_exc()


def run_int_pair() -> None:
    """Invoke IntPair from the extension to demonstrate object handling."""
    print("=========== Example 4: IntPair ===========")
    pair = my_ffi_extension.IntPair(1, 2)
    print(f"a={pair.a}")
    print(f"b={pair.b}")
    print(f"sum={pair.sum()}")


if __name__ == "__main__":
    run_add_two()
    run_add_one()
    run_raise_error()
    run_int_pair()
