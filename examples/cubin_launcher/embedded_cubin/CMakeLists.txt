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
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(TVM_FFI_CUBIN_LAUNCHER_USE_DRIVER_API
    OFF
    CACHE BOOL "Use driver API in cubin launcher"
)

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
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# [cmake_example.begin] Step 1: Compile kernel.cu to CUBIN using add_tvm_ffi_cubin utility
set(CMAKE_CUDA_ARCHITECTURES 75;80;86;89;90;100;120)
if (CMAKE_VERSION VERSION_LESS "3.27.0")
  add_tvm_ffi_fatbin(kernel_fatbin CUDA src/kernel.cu)
else ()
  add_library(kernel_fatbin OBJECT src/kernel.cu)
  set_target_properties(kernel_fatbin PROPERTY CUDA_FATBIN_COMPILATION ON)
endif ()

add_library(lib_embedded SHARED src/lib_embedded.cc)
target_link_libraries(lib_embedded PRIVATE tvm_ffi_header tvm_ffi_shared)
set_target_properties(lib_embedded PROPERTIES POSITION_INDEPENDENT_CODE ON)

if (TVM_FFI_CUBIN_LAUNCHER_USE_DRIVER_API)
  add_compile_definitions(TVM_FFI_CUDA_USE_DRIVER_API=1)
  target_link_libraries(lib_embedded PRIVATE cuda)
else ()
  target_link_libraries(lib_embedded PRIVATE CUDA::cudart)
endif ()

tvm_ffi_embed_bin_into(lib_embedded env BIN "$<TARGET_OBJECTS:kernel_fatbin>")

set_target_properties(
  lib_embedded
  PROPERTIES PREFIX ""
             SUFFIX ".so"
             LINKER_LANGUAGE CXX
)
# [cmake_example.end]
