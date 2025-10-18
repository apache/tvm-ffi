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

To compile the C++ Example:

```bash
cmake . -B build -DEXAMPLE_NAME="cpp" -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --config RelWithDebInfo
```

To compile CUDA Example:

```bash
cmake . -B build -DEXAMPLE_NAME="cuda" -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --config RelWithDebInfo
```

To run library loading examples across ML frameworks:

```bash
python load_pytorch.py
python load_jax.py
python load_numpy.py
python load_cupy.py
```

To run library loading example in C++:

```bash
g++ -fvisibility=hidden -O3             \
  load_cpp.cc                           \
  `tvm-ffi-config --cxxflags`           \
  `tvm-ffi-config --ldflags`            \
  `tvm-ffi-config --libs`               \
  -Wl,-rpath,`tvm-ffi-config --libdir`  \
  -o build/load_cpp

build/load_cpp
```
