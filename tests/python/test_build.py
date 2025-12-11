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
import pathlib

import numpy
import pytest
import tvm_ffi
import tvm_ffi.cpp
from tvm_ffi.core import TypeSchema
from tvm_ffi.module import Module


def test_build_cpp() -> None:
    cpp_path = pathlib.Path(__file__).parent.resolve() / "test_build.cc"
    output_lib_path = tvm_ffi.cpp.build(
        name="hello",
        cpp_files=[str(cpp_path)],
    )

    mod: Module = tvm_ffi.load_module(output_lib_path)

    metadata = mod.get_function_metadata("add_one_cpu")
    assert metadata is not None, "add_one_cpu should have metadata"
    assert "type_schema" in metadata, f"{'add_one_cpu'}: {metadata}"
    schema = TypeSchema.from_json_str(metadata["type_schema"])
    assert str(schema) == "Callable[[Tensor, Tensor], None]", f"{'add_one_cpu'}: {schema}"
    doc = mod.get_function_doc("add_one_cpu")
    assert doc is None

    x = numpy.array([1, 2, 3, 4, 5], dtype=numpy.float32)
    y = numpy.empty_like(x)
    mod.add_one_cpu(x, y)
    numpy.testing.assert_equal(x + 1, y)


def test_build_inline_with_metadata() -> None:  # noqa: PLR0915
    """Test functions with various input and output types."""
    # Keep module alive until all returned objects are destroyed
    mod: Module = tvm_ffi.cpp.load_inline(
        name="test_io_types",
        cpp_sources=r"""
            // int input -> int output
            int square(int x) {
                return x * x;
            }

            // float input -> float output
            float reciprocal(float x) {
                return 1.0f / x;
            }

            // bool input -> bool output
            bool negate(bool x) {
                return !x;
            }

            // String input -> String output
            tvm::ffi::String uppercase_first(tvm::ffi::String s) {
                std::string result(s.c_str());
                if (!result.empty()) {
                    result[0] = std::toupper(result[0]);
                }
                return tvm::ffi::String(result);
            }

            // Multiple inputs: int, float -> float
            float weighted_sum(int count, float weight) {
                return static_cast<float>(count) * weight;
            }

            // Multiple inputs: String, int -> String
            tvm::ffi::String repeat_string(tvm::ffi::String s, int times) {
                std::string result;
                for (int i = 0; i < times; ++i) {
                    result += s.c_str();
                }
                return tvm::ffi::String(result);
            }

            // Mixed types: bool, int, float, String -> String
            tvm::ffi::String format_data(bool flag, int count, float value, tvm::ffi::String label) {
                std::ostringstream oss;
                oss << label.c_str() << ": flag=" << (flag ? "true" : "false")
                    << ", count=" << count << ", value=" << value;
                return tvm::ffi::String(oss.str());
            }

            // Tensor input/output
            void double_tensor(tvm::ffi::TensorView input, tvm::ffi::TensorView output) {
                TVM_FFI_ICHECK(input.ndim() == 1);
                TVM_FFI_ICHECK(output.ndim() == 1);
                TVM_FFI_ICHECK(input.size(0) == output.size(0));
                DLDataType f32_dtype{kDLFloat, 32, 1};
                TVM_FFI_ICHECK(input.dtype() == f32_dtype);
                TVM_FFI_ICHECK(output.dtype() == f32_dtype);

                for (int i = 0; i < input.size(0); ++i) {
                    static_cast<float*>(output.data_ptr())[i] =
                        static_cast<const float*>(input.data_ptr())[i] * 2.0f;
                }
            }
        """,
        functions=[
            "square",
            "reciprocal",
            "negate",
            "uppercase_first",
            "weighted_sum",
            "repeat_string",
            "format_data",
            "double_tensor",
        ],
        extra_cflags=["-DTVM_FFI_DLL_EXPORT_INCLUDE_METADATA=1"],
    )

    # Test square: int -> int
    assert mod.square(5) == 25
    metadata = mod.get_function_metadata("square")
    assert metadata is not None
    schema = TypeSchema.from_json_str(metadata["type_schema"])
    assert str(schema) == "Callable[[int], int]"

    # Test reciprocal: float -> float
    result = mod.reciprocal(2.0)
    assert abs(result - 0.5) < 0.001
    metadata = mod.get_function_metadata("reciprocal")
    assert metadata is not None
    schema = TypeSchema.from_json_str(metadata["type_schema"])
    assert str(schema) == "Callable[[float], float]"

    # Test negate: bool -> bool
    assert mod.negate(True) is False
    assert mod.negate(False) is True
    metadata = mod.get_function_metadata("negate")
    assert metadata is not None
    schema = TypeSchema.from_json_str(metadata["type_schema"])
    assert str(schema) == "Callable[[bool], bool]"

    # Test uppercase_first: String -> String
    result = mod.uppercase_first("hello")
    assert result == "Hello"
    metadata = mod.get_function_metadata("uppercase_first")
    assert metadata is not None
    schema = TypeSchema.from_json_str(metadata["type_schema"])
    assert str(schema) == "Callable[[str], str]"

    # Test weighted_sum: int, float -> float
    result = mod.weighted_sum(10, 2.5)
    assert abs(result - 25.0) < 0.001
    metadata = mod.get_function_metadata("weighted_sum")
    assert metadata is not None
    schema = TypeSchema.from_json_str(metadata["type_schema"])
    assert str(schema) == "Callable[[int, float], float]"

    # Test repeat_string: String, int -> String
    result = mod.repeat_string("ab", 3)
    assert result == "ababab"
    metadata = mod.get_function_metadata("repeat_string")
    assert metadata is not None
    schema = TypeSchema.from_json_str(metadata["type_schema"])
    assert str(schema) == "Callable[[str, int], str]"

    # Test format_data: bool, int, float, String -> String
    result = mod.format_data(True, 42, 3.14, "test")
    assert "test:" in result
    assert "flag=true" in result
    assert "count=42" in result
    assert "value=3.14" in result
    metadata = mod.get_function_metadata("format_data")
    assert metadata is not None
    schema = TypeSchema.from_json_str(metadata["type_schema"])
    assert str(schema) == "Callable[[bool, int, float, str], str]"

    # Test double_tensor: Tensor, Tensor -> None
    x = numpy.array([1.0, 2.0, 3.0], dtype=numpy.float32)
    y = numpy.empty_like(x)
    mod.double_tensor(x, y)
    numpy.testing.assert_allclose(y, x * 2.0)
    metadata = mod.get_function_metadata("double_tensor")
    assert metadata is not None
    schema = TypeSchema.from_json_str(metadata["type_schema"])
    assert str(schema) == "Callable[[Tensor, Tensor], None]"


def test_build_inline_with_docstrings() -> None:
    """Test building functions with documentation using the functions dict."""
    # Keep module alive until all returned objects are destroyed
    add_docstring = (
        "Add two integers and return the sum.\n"
        "\n"
        "Parameters\n"
        "----------\n"
        "a : int\n"
        "    First integer\n"
        "b : int\n"
        "    Second integer\n"
        "\n"
        "Returns\n"
        "-------\n"
        "result : int\n"
        "    Sum of a and b"
    )

    divide_docstring = "Divides two floats. Returns a/b."

    mod: Module = tvm_ffi.cpp.load_inline(
        name="test_docs",
        cpp_sources=r"""
            int add(int a, int b) {
                return a + b;
            }

            int subtract(int a, int b) {
                return a - b;
            }

            float divide(float a, float b) {
                TVM_FFI_ICHECK(b != 0.0f) << "Division by zero";
                return a / b;
            }
        """,
        functions={
            "add": add_docstring,
            "subtract": "",  # No documentation
            "divide": divide_docstring,
        },
        extra_cflags=["-DTVM_FFI_DLL_EXPORT_INCLUDE_METADATA=1"],
    )

    # Test add function with full documentation
    assert mod.add(10, 5) == 15
    metadata = mod.get_function_metadata("add")
    assert metadata is not None
    schema = TypeSchema.from_json_str(metadata["type_schema"])
    assert str(schema) == "Callable[[int, int], int]"

    doc = mod.get_function_doc("add")
    assert doc is not None, "add should have documentation"
    assert doc == add_docstring

    # Test subtract function without documentation
    assert mod.subtract(10, 5) == 5
    metadata = mod.get_function_metadata("subtract")
    assert metadata is not None
    schema = TypeSchema.from_json_str(metadata["type_schema"])
    assert str(schema) == "Callable[[int, int], int]"

    doc = mod.get_function_doc("subtract")
    assert doc is None, "subtract should not have documentation"

    # Test divide function with short documentation
    result = mod.divide(10.0, 2.0)
    assert abs(result - 5.0) < 0.001
    metadata = mod.get_function_metadata("divide")
    assert metadata is not None
    schema = TypeSchema.from_json_str(metadata["type_schema"])
    assert str(schema) == "Callable[[float, float], float]"

    doc = mod.get_function_doc("divide")
    assert doc is not None, "divide should have documentation"
    assert doc == divide_docstring


def test_build_without_metadata() -> None:
    """Test building without metadata export."""
    mod: Module = tvm_ffi.cpp.load_inline(
        name="test_no_meta",
        cpp_sources=r"""
            // Note: NOT defining TVM_FFI_DLL_EXPORT_INCLUDE_METADATA

            int simple_add(int a, int b) {
                return a + b;
            }
        """,
        functions=["simple_add"],
    )

    # Function should still work
    result = mod.simple_add(10, 20)
    assert result == 30

    # But metadata should not be available
    metadata = mod.get_function_metadata("simple_add")
    assert metadata is None, (
        "Metadata should not be available without TVM_FFI_DLL_EXPORT_INCLUDE_METADATA"
    )

    # Doc should also not be available
    doc = mod.get_function_doc("simple_add")
    assert doc is None


if __name__ == "__main__":
    pytest.main([__file__])
