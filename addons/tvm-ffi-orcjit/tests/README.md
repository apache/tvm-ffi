<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# TVM-FFI-OrcJIT Tests

This directory contains tests for the tvm-ffi-orcjit package.

## Building Test Objects

The tests require pre-built object files. To build them:

```bash
cd tests
cmake -B build
cmake --build build --target install
```

This will compile `sources/test_funcs.cc` and generate `test_funcs.o` in the tests directory.

## Running Tests

After building the test objects, run the tests with:

```bash
pytest tests/ -v
```

Or from the repository root:

```bash
cd addons/tvm-ffi-orcjit
pytest tests/ -v
```

## Test Structure

- `sources/` - C++ source files for test functions
- `test_basic.py` - Python test cases
- `CMakeLists.txt` - Build configuration for test objects
- `test_funcs.o` - Generated object file (after building)

## CI/CD

The CI workflow automatically builds the test objects before running tests. See `.github/workflows/tvm-ffi-orcjit/ci_test.yml` for the full workflow.
