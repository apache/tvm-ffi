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
cmake_minimum_required(VERSION 3.20)
project(tvm_ffi_example)

option(EXAMPLE_NAME "Which example to build (cpp/cuda)" "cpp")
message(STATUS "Building example: ${EXAMPLE_NAME}")

find_package(tvm_ffi CONFIG)
if (NOT tvm_ffi_FOUND)
  message(FATAL_ERROR "Unable to find tvm_ffi. Set `tvm_ffi_DIR` with `tvm-ffi-config --cmakedir`")
endif ()

if (EXAMPLE_NAME STREQUAL "cpp")
  # Example 1. C++ `add_one`
  add_library(add_one_cpu SHARED src/add_one_cpu.cc)
  target_link_libraries(add_one_cpu tvm_ffi_header)
  target_link_libraries(add_one_cpu tvm_ffi_shared)
  set_target_properties(
    add_one_cpu
    PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/"
               PREFIX ""
               SUFFIX ".so"
  )
elseif (EXAMPLE_NAME STREQUAL "cuda")
  # Example 2. CUDA `add_one`
  enable_language(CUDA)
  add_library(add_one_cuda SHARED src/add_one_cuda.cu)
  target_link_libraries(add_one_cuda PRIVATE tvm_ffi_shared)
  set_target_properties(
    add_one_cuda
    PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/"
               PREFIX ""
               SUFFIX ".so"
  )
else ()
  message(FATAL_ERROR "Unknown EXAMPLE_NAME option: ${EXAMPLE_NAME}. Expected: 'cpp', 'cuda'")
endif ()
