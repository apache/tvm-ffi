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

# Quick Start Code Example

This directory contains all the source code for [tutorial](https://tvm.apache.org/ffi/get_started/quickstart.html).

## Run everything (CPU path)

On Linux/macOS:

```bash
bash run_all_cpu.sh
```

On Windows:

```batch
run_all_cpu.bat
```

## Compile and Distribute `add_one_*` manually

To compile the C++ Example:

```bash
cmake . -B build -DEXAMPLE_NAME="compile_cpu" -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --config RelWithDebInfo
```

This produces `build/add_one_cpu.so`.

To compile CUDA Example (Linux with CUDA toolchain available):

```bash
cmake . -B build -DEXAMPLE_NAME="compile_cuda" -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --config RelWithDebInfo
```

## Load the Distributed `add_one_*`

To run library loading examples across ML frameworks (requires CUDA for the CUDA example):

```bash
python load/load_pytorch.py
python load/load_numpy.py
python load/load_cupy.py
```

To run library loading example in C++:

```bash
cmake . -B build -DEXAMPLE_NAME="load_cpp" -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --config RelWithDebInfo
build/load_cpp
```

The executable is emitted as `build/load_cpp` (`build/load_cpp.exe` on Windows).

For a CUDA end-to-end run, use:

```bash
bash run_all_cuda.sh
```
