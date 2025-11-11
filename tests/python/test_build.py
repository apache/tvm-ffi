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
    assert metadata is not None and "type_schema" in metadata
    schema = TypeSchema.from_json_str(metadata["type_schema"])
    assert str(schema) == "Callable[[Tensor, Tensor], None]", f"{'add_one_cpu'}: {schema}"
    assert "arg_const" in metadata
    arg_const = metadata["arg_const"]
    assert len(arg_const) == 2, "Should have 2 arguments"
    assert arg_const[0] is False and arg_const[1] is False, f"{'add_one_cpu'}: {arg_const}"

    x = numpy.array([1, 2, 3, 4, 5], dtype=numpy.float32)
    y = numpy.empty_like(x)
    mod.add_one_cpu(x, y)
    numpy.testing.assert_equal(x + 1, y)


if __name__ == "__main__":
    pytest.main([__file__])
