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
"""Basic tests for tvm-ffi-orcjit functionality."""

from __future__ import annotations

from pathlib import Path

import pytest
import tvm_ffi
from tvm_ffi_orcjit import ExecutionSession


def get_test_obj_file(object_file: str) -> Path:
    """Get the path to the pre-built test object file.

    Returns
    -------
    Path
        Path to the test_funcs.o object file.

    """
    # The object file should be built by CMake and located in the tests directory
    test_dir = Path(__file__).parent
    obj_file = test_dir / object_file

    if not obj_file.exists():
        raise FileNotFoundError(
            f"Test object file not found: {obj_file}\n"
            "Please build the test object file first:\n"
            "  cd tests && cmake -B build && cmake --build build"
        )

    return obj_file


def test_create_session() -> None:
    """Test creating an execution session."""
    session = ExecutionSession()
    assert session is not None


def test_create_library() -> None:
    """Test creating a dynamic library."""
    session = ExecutionSession()
    lib = session.create_library()
    assert lib is not None


def test_load_and_execute_function() -> None:
    """Test loading an object file and executing a function."""
    # Get pre-built test object file
    obj_file = get_test_obj_file("test_funcs.o")

    # Create session and library
    session = ExecutionSession()
    lib = session.create_library()

    # Load object file
    lib.add(str(obj_file))

    # Get and call test_add function
    add_func = lib.get_function("test_add")
    result = add_func(10, 20)
    assert result == 30

    # Get and call test_multiply function
    mul_func = lib.get_function("test_multiply")
    result = mul_func(7, 6)
    assert result == 42


def test_multiple_libraries() -> None:
    """Test creating and using multiple libraries."""
    session = ExecutionSession()

    lib1 = session.create_library("lib1")
    lib2 = session.create_library("lib2")

    assert lib1 is not None
    assert lib2 is not None


def test_function_not_found() -> None:
    """Test that getting a non-existent function raises an error."""
    # Get pre-built test object file
    obj_file = get_test_obj_file("test_funcs.o")

    session = ExecutionSession()
    lib = session.create_library()
    lib.add(str(obj_file))

    with pytest.raises(AttributeError, match="Module has no function"):
        lib.get_function("nonexistent_function")


def test_gradually_add_objects_to_same_library() -> None:
    """Test gradually adding multiple object files to the same library."""
    obj_file1 = get_test_obj_file("test_funcs.o")
    obj_file2 = get_test_obj_file("test_funcs2.o")

    session = ExecutionSession()
    lib = session.create_library()

    # Add first object file
    lib.add(str(obj_file1))

    # Test functions from first object
    add_func = lib.get_function("test_add")
    assert add_func(5, 3) == 8

    mul_func = lib.get_function("test_multiply")
    assert mul_func(4, 5) == 20

    # Add second object file to the same library
    lib.add(str(obj_file2))

    # Test functions from second object
    sub_func = lib.get_function("test_subtract")
    assert sub_func(10, 3) == 7

    div_func = lib.get_function("test_divide")
    assert div_func(20, 4) == 5

    # Verify first object's functions still work
    assert add_func(10, 20) == 30
    assert mul_func(7, 6) == 42


def test_two_separate_libraries() -> None:
    """Test creating two separate libraries each with its own object file."""
    obj_file1 = get_test_obj_file("test_funcs.o")
    obj_file2 = get_test_obj_file("test_funcs2.o")

    session = ExecutionSession()

    # Create first library with first object
    lib1 = session.create_library("lib1")
    lib1.add(str(obj_file1))

    # Create second library with second object
    lib2 = session.create_library("lib2")
    lib2.add(str(obj_file2))

    # Test functions from lib1
    add_func = lib1.get_function("test_add")
    assert add_func(5, 3) == 8

    mul_func = lib1.get_function("test_multiply")
    assert mul_func(4, 5) == 20

    # Test functions from lib2
    sub_func = lib2.get_function("test_subtract")
    assert sub_func(10, 3) == 7

    div_func = lib2.get_function("test_divide")
    assert div_func(20, 4) == 5

    # Verify lib1 doesn't have lib2's functions
    with pytest.raises(AttributeError, match="Module has no function"):
        lib1.get_function("test_subtract")

    # Verify lib2 doesn't have lib1's functions
    with pytest.raises(AttributeError, match="Module has no function"):
        lib2.get_function("test_add")


def test_symbol_conflict_same_library() -> None:
    """Test that adding objects with conflicting symbols to same library fails."""
    obj_file1 = get_test_obj_file("test_funcs.o")
    obj_file_conflict = get_test_obj_file("test_funcs_conflict.o")

    session = ExecutionSession()
    lib = session.create_library()

    # Add first object file
    lib.add(str(obj_file1))

    # Verify first object's function works
    add_func = lib.get_function("test_add")
    assert add_func(10, 20) == 30

    # Try to add conflicting object - should raise an error
    with pytest.raises(Exception):  # LLVM will throw an error for duplicate symbols
        lib.add(str(obj_file_conflict))


def test_symbol_conflict_different_libraries() -> None:
    """Test that adding objects with conflicting symbols to different libraries works."""
    obj_file1 = get_test_obj_file("test_funcs.o")
    obj_file_conflict = get_test_obj_file("test_funcs_conflict.o")

    session = ExecutionSession()

    # Create first library with first object
    lib1 = session.create_library("lib1")
    lib1.add(str(obj_file1))

    # Create second library with conflicting object
    lib2 = session.create_library("lib2")
    lib2.add(str(obj_file_conflict))

    # Test that both libraries work with their own versions
    add_func1 = lib1.get_function("test_add")
    result1 = add_func1(10, 20)
    assert result1 == 30  # Original implementation

    add_func2 = lib2.get_function("test_add")
    result2 = add_func2(10, 20)
    assert result2 == 1030  # Conflicting implementation adds 1000

    # Test multiply functions
    mul_func1 = lib1.get_function("test_multiply")
    assert mul_func1(5, 6) == 30  # Original: 5 * 6

    mul_func2 = lib2.get_function("test_multiply")
    assert mul_func2(5, 6) == 60  # Conflict: (5 * 6) * 2


def test_load_and_execute_cuda_function() -> None:
    """Test loading an object file and executing a function."""
    # Get pre-built test object file
    try:
        obj_file = get_test_obj_file("test_funcs_cuda.o")
    except FileNotFoundError:
        return

    # Create session and library
    session = ExecutionSession()
    lib = session.create_library()

    # Load object file
    lib.add(str(obj_file))

    # Get and call test_add function
    add_func = lib.get_function("test_add")
    result = add_func(10, 20)
    assert result == 30

    # Get and call test_multiply function
    mul_func = lib.get_function("test_multiply")
    result = mul_func(7, 6)
    assert result == 42


def test_ctor_dtor() -> None:
    """Test ctor and dtor when loading an object file."""
    log = ""

    @tvm_ffi.register_global_func("append_log")
    def _append_ctor_log(x: str) -> None:
        nonlocal log
        log += x

    # Get pre-built test object file
    obj_file = get_test_obj_file("test_ctor_dtor.o")

    # Create session and library
    session = ExecutionSession()
    lib = session.create_library()

    # Load object file
    lib.add(str(obj_file))

    lib.get_function("main")()
    del lib
    print(log)
    assert (
        log
        == "<init_array.101><init_array.102><init_array.103><init_array><ctors.103><ctors.102><ctors.101><ctors><main><dtors><dtors.101><dtors.102><dtors.103><fini_array><fini_array.103><fini_array.102><fini_array.101>"
    ), log


if __name__ == "__main__":
    # pytest.main([__file__, "-v"])
    test_ctor_dtor()
