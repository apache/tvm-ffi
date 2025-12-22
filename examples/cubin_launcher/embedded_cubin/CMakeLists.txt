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
project(embedded_cubin_example LANGUAGES CXX CUDA)

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

# [cmake_example.begin] Step 1: Compile kernel.cu to CUBIN using tvm_ffi_generate_cubin utility Use
# -arch=native to automatically detect the GPU architecture
tvm_ffi_generate_cubin(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/kernel.cubin SOURCE src/kernel.cu ARCH native
)

# Step 2: Embed CUBIN into the object file using tvm_ffi_embed_cubin utility This creates symbols:
# __tvm_ffi__cubin_env, __tvm_ffi__cubin_env_end (local)
tvm_ffi_embed_cubin(
  OUTPUT
  ${CMAKE_CURRENT_BINARY_DIR}/lib_embedded_with_cubin.o
  SOURCE
  src/lib_embedded.cc
  CUBIN
  ${CMAKE_CURRENT_BINARY_DIR}/kernel.cubin
  NAME
  env
)

# Step 3: Build lib_embedded shared library (with embedded CUBIN)
add_library(lib_embedded SHARED ${CMAKE_CURRENT_BINARY_DIR}/lib_embedded_with_cubin.o)
target_link_libraries(lib_embedded PRIVATE tvm_ffi::header tvm_ffi::shared CUDA::cudart)
set_target_properties(
  lib_embedded
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/"
             PREFIX ""
             SUFFIX ".so"
             LINKER_LANGUAGE CXX
)
# [cmake_example.end]
