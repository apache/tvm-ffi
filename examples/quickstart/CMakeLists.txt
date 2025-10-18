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

option(EXAMPLE_NAME "Which example to build ('compile_cpu'/'compile_cuda'/'load_cpp')"
       "compile_cpu"
)
message(STATUS "Building example: ${EXAMPLE_NAME}")

# Run `tvm_ffi.config --cmakedir` to find tvm-ffi package
find_package(
  Python
  COMPONENTS Interpreter
  REQUIRED
)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m tvm_ffi.config --cmakedir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE tvm_ffi_ROOT
)
find_package(tvm_ffi CONFIG REQUIRED)

if (EXAMPLE_NAME STREQUAL "compile_cpu")
  # Example 1. C++ `add_one`
  add_library(add_one_cpu SHARED compile/add_one_cpu.cc)
  target_link_libraries(add_one_cpu tvm_ffi_header)
  target_link_libraries(add_one_cpu tvm_ffi_shared)
  set_target_properties(
    add_one_cpu
    PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/"
               PREFIX ""
               SUFFIX ".so"
  )
elseif (EXAMPLE_NAME STREQUAL "compile_cuda")
  # Example 2. CUDA `add_one`
  enable_language(CUDA)
  add_library(add_one_cuda SHARED compile/add_one_cuda.cu)
  target_link_libraries(add_one_cuda PRIVATE tvm_ffi_shared)
  set_target_properties(
    add_one_cuda
    PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/"
               PREFIX ""
               SUFFIX ".so"
  )
elseif (EXAMPLE_NAME STREQUAL "load_cpp")
  # Example 3. Load C++ shared library
  add_executable(load_cpp load/load_cpp.cc)
  target_link_libraries(load_cpp tvm_ffi_header)
  target_link_libraries(load_cpp tvm_ffi_shared)
  set_target_properties(
    load_cpp
    PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/"
               PREFIX ""
               SUFFIX ""
  )
else ()
  message(FATAL_ERROR "Unknown EXAMPLE_NAME option: ${EXAMPLE_NAME}. "
                      "Expected: 'compile_cpu', 'compile_cuda', 'load_cpp'."
  )
endif ()
