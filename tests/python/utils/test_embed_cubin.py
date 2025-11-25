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
"""Tests for the embed_cubin utility."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType

import pytest
from tvm_ffi.utils.embed_cubin import embed_cubin

torch: ModuleType | None
try:
    import torch  # type: ignore[import-not-found,no-redef]
except ImportError:
    torch = None


def _is_cuda_available() -> bool:
    """Check if CUDA is available for testing."""
    if torch is None:
        return False
    return torch.cuda.is_available()


def _create_test_object_file(obj_path: Path) -> None:
    """Create a simple test object file with TVM_FFI_EMBED_CUBIN macro usage.

    We avoid including the full header to prevent CUDA dependency in tests.
    Instead, we manually declare the external symbols that TVM_FFI_EMBED_CUBIN would create.
    """
    cpp_code = """
    // Manually declare the external symbols that will be provided by embed_cubin
    // This is what TVM_FFI_EMBED_CUBIN(test_cubin) would do, but without CUDA headers
    extern "C" {
      extern const unsigned char __tvm_ffi__cubin_test_cubin[];
      extern const unsigned char __tvm_ffi__cubin_test_cubin_end[];
    }

    // Simple function that references the embedded CUBIN symbols
    const void* get_cubin_start() {
      return __tvm_ffi__cubin_test_cubin;
    }

    const void* get_cubin_end() {
      return __tvm_ffi__cubin_test_cubin_end;
    }
    """

    # Write C++ code to a temporary file
    cpp_file = obj_path.with_suffix(".cc")
    cpp_file.write_text(cpp_code)

    # Compile to object file
    # Try to find a C++ compiler
    compilers = ["g++", "clang++", "c++"]
    compiler = None
    for c in compilers:
        try:
            subprocess.run(
                [c, "--version"],
                check=True,
                capture_output=True,
            )
            compiler = c
            break
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue

    if compiler is None:
        pytest.skip("No C++ compiler found (tried g++, clang++, c++)")

    assert isinstance(compiler, str), "Compiler is not a string"

    compile_cmd: list[str] = [compiler, "-c", "-fPIC", "-o", str(obj_path), str(cpp_file)]

    try:
        subprocess.run(compile_cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        pytest.skip(f"Failed to compile test object file: {e.stderr.decode('utf-8')}")
    finally:
        # Clean up temporary C++ file
        cpp_file.unlink(missing_ok=True)


def _create_test_cubin(cubin_path: Path) -> None:
    """Create a dummy CUBIN file for testing."""
    # Create a simple binary file with some recognizable content
    cubin_path.write_bytes(b"DUMMY_CUBIN_DATA_FOR_TESTING" * 10)


def _check_symbols_in_object(obj_path: Path, expected_symbols: list[str]) -> bool:
    """Check if expected symbols exist in the object file using nm."""
    try:
        result = subprocess.run(
            ["nm", str(obj_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        nm_output = result.stdout

        for symbol in expected_symbols:
            if symbol not in nm_output:
                return False
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("nm tool not available")
        return False


@pytest.mark.skipif(sys.platform != "linux", reason="embed_cubin only supported on Linux")
@pytest.mark.skipif(torch is None, reason="PyTorch not installed")
@pytest.mark.skipif(not _is_cuda_available(), reason="CUDA not available")
def test_embed_cubin_basic() -> None:
    """Test basic embed_cubin functionality."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test files
        input_obj = temp_path / "input.o"
        output_obj = temp_path / "output.o"
        cubin_file = temp_path / "test.cubin"

        _create_test_object_file(input_obj)
        _create_test_cubin(cubin_file)

        # Embed CUBIN into object file
        embed_cubin(
            cubin_path=cubin_file,
            input_obj_path=input_obj,
            output_obj_path=output_obj,
            name="test_cubin",
            verbose=False,
        )

        # Check that output file exists
        assert output_obj.exists(), "Output object file was not created"
        assert output_obj.stat().st_size > 0, "Output object file is empty"

        # Check that the output is larger than the input (it should contain both)
        assert output_obj.stat().st_size > input_obj.stat().st_size, (
            "Output object file should be larger than input"
        )

        # Check that expected symbols are present
        expected_symbols = [
            "__tvm_ffi__cubin_test_cubin",
            "__tvm_ffi__cubin_test_cubin_end",
        ]
        assert _check_symbols_in_object(output_obj, expected_symbols), (
            f"Expected symbols {expected_symbols} not found in output object file"
        )


@pytest.mark.skipif(sys.platform != "linux", reason="embed_cubin only supported on Linux")
@pytest.mark.skipif(torch is None, reason="PyTorch not installed")
@pytest.mark.skipif(not _is_cuda_available(), reason="CUDA not available")
def test_embed_cubin_different_names() -> None:
    """Test embedding CUBIN with different names."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test files
        input_obj = temp_path / "input.o"
        output_obj = temp_path / "output.o"
        cubin_file = temp_path / "test.cubin"

        _create_test_object_file(input_obj)
        _create_test_cubin(cubin_file)

        # Test with a different name
        custom_name = "my_custom_kernel"
        embed_cubin(
            cubin_path=cubin_file,
            input_obj_path=input_obj,
            output_obj_path=output_obj,
            name=custom_name,
            verbose=False,
        )

        # Check that symbols with custom name are present
        expected_symbols = [
            f"__tvm_ffi__cubin_{custom_name}",
            f"__tvm_ffi__cubin_{custom_name}_end",
        ]
        assert _check_symbols_in_object(output_obj, expected_symbols), (
            f"Expected symbols {expected_symbols} not found in output object file"
        )


def test_embed_cubin_nonexistent_input() -> None:
    """Test that embed_cubin raises error for nonexistent input files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        input_obj = temp_path / "nonexistent_input.o"
        output_obj = temp_path / "output.o"
        cubin_file = temp_path / "nonexistent.cubin"

        # Test with nonexistent CUBIN file
        with pytest.raises(FileNotFoundError):
            embed_cubin(
                cubin_path=cubin_file,
                input_obj_path=input_obj,
                output_obj_path=output_obj,
                name="test",
                verbose=False,
            )


@pytest.mark.skipif(sys.platform != "linux", reason="embed_cubin only supported on Linux")
@pytest.mark.skipif(torch is None, reason="PyTorch not installed")
@pytest.mark.skipif(not _is_cuda_available(), reason="CUDA not available")
def test_embed_cubin_verbose_mode() -> None:
    """Test that verbose mode doesn't crash."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test files
        input_obj = temp_path / "input.o"
        output_obj = temp_path / "output.o"
        cubin_file = temp_path / "test.cubin"

        _create_test_object_file(input_obj)
        _create_test_cubin(cubin_file)

        # Run with verbose mode (should not crash)
        embed_cubin(
            cubin_path=cubin_file,
            input_obj_path=input_obj,
            output_obj_path=output_obj,
            name="test_cubin",
            verbose=True,  # Test verbose mode
        )

        assert output_obj.exists(), "Output object file was not created in verbose mode"
