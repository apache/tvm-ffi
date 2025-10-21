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
project(tvm_ffi_example LANGUAGES C)

option(EXAMPLE_NAME "Which example to build ('kernel'/'load')" "kernel")
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

if (EXAMPLE_NAME STREQUAL "kernel")
  # Example 1. `add_one_cpu` in C
  add_library(add_one_cpu SHARED src/add_one_cpu.c)
  target_link_libraries(add_one_cpu PRIVATE tvm_ffi_header)
  target_link_libraries(add_one_cpu PRIVATE tvm_ffi_shared)
  set_target_properties(
    add_one_cpu
    PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/"
               PREFIX ""
               SUFFIX ".so"
               C_STANDARD 11
               C_STANDARD_REQUIRED YES
               C_EXTENSIONS NO
  )
elseif (EXAMPLE_NAME STREQUAL "load")
  # Example 2. Load `add_one_cpu` shared library in C
  add_executable(load src/load.c)
  target_link_libraries(load PRIVATE tvm_ffi_header)
  target_link_libraries(load PRIVATE tvm_ffi_shared)
  set_target_properties(
    load
    PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/"
               PREFIX ""
               SUFFIX ""
               C_STANDARD 11
               C_STANDARD_REQUIRED YES
               C_EXTENSIONS NO
  )
else ()
  message(FATAL_ERROR "Unknown EXAMPLE_NAME option: ${EXAMPLE_NAME}. "
                      "Expected: 'kernel' or 'load'."
  )
endif ()
