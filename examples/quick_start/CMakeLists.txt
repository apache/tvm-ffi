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

cmake_minimum_required(VERSION 3.18)
project(tvm_ffi_example)

# Discover the Python interpreter so we can query tvm-ffi for its CMake package path.
find_package(
  Python
  COMPONENTS Interpreter
  REQUIRED
)

# Find tvm-ffi automatically
find_package(tvm_ffi CONFIG REQUIRED)

# Build the CPU and C versions of the simple "add one" function that the examples call.
add_library(add_one_cpu SHARED src/add_one_cpu.cc)
add_library(add_one_c SHARED src/add_one_c.c)
target_link_libraries(add_one_cpu tvm_ffi_header)
target_link_libraries(add_one_cpu tvm_ffi_shared)
target_link_libraries(add_one_c tvm_ffi_shared)
# show as add_one_cpu.so
set_target_properties(add_one_cpu PROPERTIES PREFIX "" SUFFIX ".so")
set_target_properties(add_one_c PROPERTIES PREFIX "" SUFFIX ".so")

# Optionally build the CUDA variant if the CUDA toolkit is present.
if (NOT WIN32)
  find_package(CUDAToolkit QUIET)
  if (CUDAToolkit_FOUND)
    enable_language(CUDA)

    add_library(add_one_cuda SHARED src/add_one_cuda.cu)
    target_link_libraries(add_one_cuda PRIVATE tvm_ffi_shared)

    set_target_properties(add_one_cuda PROPERTIES PREFIX "" SUFFIX ".so")

    add_executable(run_example_cuda src/run_example_cuda.cc)
    set_target_properties(run_example_cuda PROPERTIES CXX_STANDARD 17)
    target_link_libraries(run_example_cuda PRIVATE tvm_ffi_shared CUDA::cudart)
  endif ()
endif ()

# CPU-only C++ driver used in the quick start guide.
add_executable(run_example src/run_example.cc)
set_target_properties(run_example PROPERTIES CXX_STANDARD 17)
target_link_libraries(run_example tvm_ffi_shared)
