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

# Getting Started with TVM FFI

This example demonstrates how to use tvm-ffi to expose a universal function
that can be loaded in different environments.

The example implements a simple "add one" operation that adds 1 to each element
of an input tensor, showing how to create C++ functions callable from Python.

## Prerequisites

Before running the quick start, ensure you have:

- tvm-ffi installed locally (editable installs are convenient while iterating):
- Installation guide: [Installation guide](https://tvm.apache.org/ffi/get_started/install.html)

```bash
# From the quick_start directory
# install and include test dependency(this will install torch and numpy)
pip install -ve "../..[test]"
```

## Run the Quick Start

From `examples/quick_start` you can build and run everything with the helper script:

```bash
cd examples/quick_start
./run_example.sh
```

The script picks an available CMake generator (preferring Ninja), configures a build in `build/`, compiles the C++ libraries and examples,
and finally runs the Python and C++ demos. If the CUDA toolkit is detected it will also build and execute `run_example_cuda`.

If you prefer to drive the build manually, run the following instead:

```bash
# configure (omit -G Ninja if Ninja is not installed)
cmake -G Ninja -B build -S .

# compile the example targets
cmake --build build --parallel

# run the demos
python run_example.py
./build/run_example
./build/run_example_cuda  # optional, requires CUDA toolkit
```

At a high level, the `TVM_FFI_DLL_EXPORT_TYPED_FUNC` macro helps to expose
a C++ function into the TVM FFI C ABI convention for functions.
Then the function can be accessed by different environments and languages
that interface with the TVM FFI. The current example shows how to do so
in Python and C++.

## Key Files

- `src/add_one_cpu.cc` - CPU implementation of the add_one function
- `src/add_one_c.c` - C implementation showing the C ABI workflow
- `src/add_one_cuda.cu` - CUDA implementation for GPU operations
- `src/run_example.cc` - C++ example showing how to call the functions
- `src/run_example_cuda.cc` - C++ example showing how to call the CUDA functions
- `run_example.py` - Python example showing how to call the functions
- `run_example.sh` - Convenience script that builds and runs all examples

## Compile without CMake

You can also compile the modules directly using
flags provided by the `tvm-ffi-config` tool.

```bash
gcc -shared -fPIC `tvm-ffi-config --cflags`  \
    src/add_one_c.c -o build/add_one_c.so \
    `tvm-ffi-config --ldflags` `tvm-ffi-config --libs`
```
