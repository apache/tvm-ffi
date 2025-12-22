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
project(quickstart LANGUAGES CXX)

option(EXAMPLE_NAME "Which example to build ('compile_cpu'/'compile_cuda'/'load_cpp')"
       "compile_cpu"
)
message(STATUS "Building example: ${EXAMPLE_NAME}")

# cmake-lint: disable=C0111
function (set_flat_output_dirs target_name)
  set(output_dir "${CMAKE_BINARY_DIR}")
  # Ensure multi-config generators (e.g., MSVC) output to the same flat directory.
  foreach (config IN ITEMS "" DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    if (config STREQUAL "")
      set_property(TARGET ${target_name} PROPERTY RUNTIME_OUTPUT_DIRECTORY "${output_dir}")
      set_property(TARGET ${target_name} PROPERTY LIBRARY_OUTPUT_DIRECTORY "${output_dir}")
      set_property(TARGET ${target_name} PROPERTY ARCHIVE_OUTPUT_DIRECTORY "${output_dir}")
    else ()
      set_property(
        TARGET ${target_name} PROPERTY RUNTIME_OUTPUT_DIRECTORY_${config} "${output_dir}"
      )
      set_property(
        TARGET ${target_name} PROPERTY LIBRARY_OUTPUT_DIRECTORY_${config} "${output_dir}"
      )
      set_property(
        TARGET ${target_name} PROPERTY ARCHIVE_OUTPUT_DIRECTORY_${config} "${output_dir}"
      )
    endif ()
  endforeach ()
endfunction ()

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
  target_link_libraries(add_one_cpu PRIVATE tvm_ffi::header tvm_ffi::shared)
  set_target_properties(add_one_cpu PROPERTIES PREFIX "" SUFFIX ".so")
  set_flat_output_dirs(add_one_cpu)
elseif (EXAMPLE_NAME STREQUAL "compile_cuda")
  # Example 2. CUDA `add_one`
  enable_language(CUDA)
  add_library(add_one_cuda SHARED compile/add_one_cuda.cu)
  target_link_libraries(add_one_cuda PRIVATE tvm_ffi::header tvm_ffi::shared)
  set_target_properties(add_one_cuda PROPERTIES PREFIX "" SUFFIX ".so")
  set_flat_output_dirs(add_one_cuda)
elseif (EXAMPLE_NAME STREQUAL "load_cpp")
  # Example 3. Load C++ shared library
  add_executable(load_cpp load/load_cpp.cc)
  target_link_libraries(load_cpp PRIVATE tvm_ffi::header tvm_ffi::shared)
  set_flat_output_dirs(load_cpp)
else ()
  message(FATAL_ERROR "Unknown EXAMPLE_NAME option: ${EXAMPLE_NAME}. "
                      "Expected: 'compile_cpu', 'compile_cuda', 'load_cpp'."
  )
endif ()
