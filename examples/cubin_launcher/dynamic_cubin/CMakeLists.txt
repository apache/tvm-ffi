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
project(dynamic_cubin_example LANGUAGES CXX CUDA)

# Prefer virtualenv when searching for python
set(Python_FIND_VIRTUALENV FIRST) # cmake-lint: disable=C0103

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

# Find CUDA toolkit
find_package(CUDAToolkit REQUIRED)

# Step 1: Compile kernel.cu to CUBIN using tvm_ffi_generate_cubin utility Use -arch=native to
# automatically detect the GPU architecture
tvm_ffi_generate_cubin(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/kernel.cubin SOURCE src/kernel.cu ARCH native
)

# Create a target that depends on the CUBIN
add_custom_target(
  generate_cubin ALL
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/kernel.cubin
  COMMENT "Generating CUBIN file"
)

# Step 2: Build lib_dynamic shared library (loads CUBIN from file at runtime)
add_library(lib_dynamic SHARED src/lib_dynamic.cc)
target_link_libraries(lib_dynamic PRIVATE tvm_ffi::header tvm_ffi::shared CUDA::cudart)
add_dependencies(lib_dynamic generate_cubin)
set_target_properties(
  lib_dynamic
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/"
             PREFIX ""
             SUFFIX ".so"
)
