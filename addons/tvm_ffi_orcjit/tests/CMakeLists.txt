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
project(tvm_ffi_orcjit_tests)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(TEST_OBJ_INSTALL_SUFFIX
    ""
    CACHE STRING "Suffix for install subdirectory (e.g., -gcc)"
)

if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  add_compile_options(-mno-outline-atomics)
endif ()

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

# Helper: build a source file into an object and install as <subdir>/<name>.o Derives the target
# name and install subdirectory from the source path.
function (add_test_object source)
  get_filename_component(_name "${source}" NAME_WE)
  get_filename_component(_ext "${source}" EXT)
  get_filename_component(_dir "${source}" DIRECTORY)
  get_filename_component(_subdir "${_dir}" NAME)
  set(target_name "${_subdir}_${_name}")
  if (_ext STREQUAL ".c")
    set(lang C)
  else ()
    set(lang CXX)
  endif ()
  add_library(${target_name}_obj OBJECT "${source}")
  target_link_libraries(${target_name}_obj PRIVATE tvm_ffi::header)
  if (lang STREQUAL "C" AND MSVC)
    # /GS- disables security cookie checks (__security_cookie) which are CRT symbols the ORC JIT
    # cannot resolve.
    target_compile_options(${target_name}_obj PRIVATE /O2 /GS-)
  else ()
    target_compile_options(${target_name}_obj PRIVATE -O2)
  endif ()
  install(
    FILES $<TARGET_OBJECTS:${target_name}_obj>
    DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/${_subdir}${TEST_OBJ_INSTALL_SUFFIX}
    RENAME ${_name}.o
  )
endfunction ()

# C++ object files — skip on Windows (C-only strategy for ORC JIT on Windows)
if (NOT WIN32)
  add_test_object(sources/cc/test_funcs.cc)
  add_test_object(sources/cc/test_funcs2.cc)
  add_test_object(sources/cc/test_funcs_conflict.cc)
  add_test_object(sources/cc/test_ctor_dtor.cc)
  add_test_object(sources/cc/test_call_global.cc)
  add_test_object(sources/cc/test_types.cc)
  add_test_object(sources/cc/test_link_order_base.cc)
  add_test_object(sources/cc/test_link_order_caller.cc)
  add_test_object(sources/cc/test_error.cc)
  add_test_object(sources/cc/test_containers.cc)
endif ()

# Pure C object files — built on all platforms (no C++ runtime deps)
enablelanguage(C)
add_test_object(sources/c/test_funcs.c)
add_test_object(sources/c/test_funcs2.c)
add_test_object(sources/c/test_funcs_conflict.c)
add_test_object(sources/c/test_call_global.c)
add_test_object(sources/c/test_context.c)
add_test_object(sources/c/test_types.c)
add_test_object(sources/c/test_link_order_base.c)
add_test_object(sources/c/test_link_order_caller.c)
add_test_object(sources/c/test_error.c)
add_test_object(sources/c/test_ctor_dtor.c)

# CUDA object files — optional
find_package(CUDAToolkit)
if (CUDAToolkit_FOUND)
  enablelanguage(CUDA)
  message(STATUS "CUDA found: ${CUDAToolkit_VERSION}")
  add_test_object(sources/cuda/test_funcs.cu)
endif ()
