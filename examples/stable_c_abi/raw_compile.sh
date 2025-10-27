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
# [kernel.begin]
gcc -shared -O3 -std=c11 src/add_one_cpu.c  \
    -fPIC -fvisibility=hidden               \
    $(tvm-ffi-config --cflags)              \
    $(tvm-ffi-config --ldflags)             \
    $(tvm-ffi-config --libs)                \
    -o $BUILD_DIR/add_one_cpu.so
# [kernel.end]

# Example 2. Load and run `add_one_cpu.so` in C
if [ -f "$BUILD_DIR/add_one_cpu.so" ]; then
# [load.begin]
gcc -fvisibility=hidden -O3 -std=c11        \
    src/load.c                              \
    $(tvm-ffi-config --cflags)              \
    $(tvm-ffi-config --ldflags)             \
    $(tvm-ffi-config --libs)                \
    -Wl,-rpath,$(tvm-ffi-config --libdir)   \
    -o build/load
build/load
# [load.end]
fi
