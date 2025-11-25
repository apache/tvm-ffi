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
project(cubin_launcher LANGUAGES CXX CUDA)

# Prefer virtualenv when searching for python
set(Python_FIND_VIRTUALENV FIRST)

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

# Create a target for lib_dynamic to depend on
add_custom_target(
  generate_cubin ALL
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/kernel.cubin
  COMMENT "Generating CUBIN file"
)

# Step 2: Embed CUBIN into the object file using tvm_ffi_embed_cubin utility This creates symbols:
# __tvm_ffi__cubin_env, __tvm_ffi__cubin_env_end
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
target_link_libraries(
  lib_embedded PRIVATE tvm_ffi_header tvm_ffi_shared CUDA::cuda_driver CUDA::cudart
)
set_target_properties(
  lib_embedded
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/"
             PREFIX ""
             SUFFIX ".so"
             LINKER_LANGUAGE CXX
)
# Mark the embedded object file's stack as non-executable
target_link_options(lib_embedded PRIVATE "LINKER:-z,noexecstack")

# Step 4: Build lib_dynamic shared library (loads CUBIN from file)
add_library(lib_dynamic SHARED src/lib_dynamic.cc)
target_link_libraries(
  lib_dynamic PRIVATE tvm_ffi_header tvm_ffi_shared CUDA::cuda_driver CUDA::cudart
)
add_dependencies(lib_dynamic generate_cubin)
set_target_properties(
  lib_dynamic
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/"
             PREFIX ""
             SUFFIX ".so"
)
