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
project(tvm_ffi_orcjit_example)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find tvm-ffi package
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

# Helper: build a source file into an object and copy as <name>.o
function (add_example_object name source)
  get_filename_component(_ext "${source}" EXT)
  add_library(${name}_obj OBJECT "${source}")
  target_link_libraries(${name}_obj PRIVATE tvm_ffi::header)
  if (_ext STREQUAL ".c" AND MSVC)
    target_compile_options(${name}_obj PRIVATE /O2)
  else ()
    target_compile_options(${name}_obj PRIVATE -O2)
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
      target_compile_options(${name}_obj PRIVATE -mno-outline-atomics)
    endif ()
  endif ()
  add_custom_target(
    copy_${name} ALL
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_OBJECTS:${name}_obj>
            ${CMAKE_CURRENT_SOURCE_DIR}/${name}.o
    DEPENDS ${name}_obj
    COMMENT "Copying ${name}.o to example directory"
  )
endfunction ()

# C++ object — skip on Windows (C-only strategy for ORC JIT on Windows)
if (NOT WIN32)
  add_example_object(add add.cc)
endif ()

# Pure C object — built on all platforms
enable_language(C)
add_example_object(add_c add_c.c)
