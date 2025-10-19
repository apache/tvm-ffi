#!/bin/bash
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
# shellcheck disable=SC2046
set -ex

BUILD_DIR=build
mkdir -p $BUILD_DIR

# Example 1. Compile C++ `add_one_cpu.cc` to shared library `add_one_cpu.so`
# [cpp_compile.begin]
g++ -shared -O3 compile/add_one_cpu.cc  \
    -fPIC -fvisibility=hidden           \
    $(tvm-ffi-config --cxxflags)        \
    $(tvm-ffi-config --ldflags)         \
    $(tvm-ffi-config --libs)            \
    -o $BUILD_DIR/add_one_cpu.so
# [cpp_compile.end]

# Example 2. Compile CUDA `add_one_cuda.cu` to shared library `add_one_cuda.so`

if command -v nvcc >/dev/null 2>&1; then
# [cuda_compile.begin]
nvcc -shared -O3 compile/add_one_cuda.cu    \
    -Xcompiler -fPIC,-fvisibility=hidden    \
    $(tvm-ffi-config --cxxflags)            \
    $(tvm-ffi-config --ldflags)             \
    $(tvm-ffi-config --libs)                \
    -o $BUILD_DIR/add_one_cuda.so
# [cuda_compile.end]
fi

# Example 3. Load and run `add_one_cpu.so` in C++

if [ -f "$BUILD_DIR/add_one_cpu.so" ]; then
# [load_cpp.begin]
g++ -fvisibility=hidden -O3                 \
    load/load_cpp.cc                        \
    $(tvm-ffi-config --cxxflags)            \
    $(tvm-ffi-config --ldflags)             \
    $(tvm-ffi-config --libs)                \
    -Wl,-rpath,$(tvm-ffi-config --libdir)   \
    -o build/load_cpp

build/load_cpp
# [load_cpp.end]
fi
