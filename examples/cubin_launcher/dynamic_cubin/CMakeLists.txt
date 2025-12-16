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

set(TVM_FFI_CUBIN_LAUNCHER_USE_DRIVER_API OFF CACHE BOOL "Use driver API in cubin launcher")

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

# [cmake_example.begin] Step 1: Compile kernel.cu to CUBIN using add_tvm_ffi_cubin utility Use
# CMAKE_CUDA_ARCHITECTURES=native to automatically detect the GPU architecture
set(CMAKE_CUDA_ARCHITECTURES native)
if(CMAKE_VERSION VERSION_LESS "3.27.0")
  add_tvm_ffi_cubin(kernel_cubin CUDA src/kernel.cu)
else()
  add_library(kernel_cubin OBJECT src/kernel.cu)
  set_property(TARGET kernel_cubin PROPERTY CUDA_CUBIN_COMPILATION ON)
endif()

# Step 2: Build lib_dynamic shared library (loads CUBIN from file at runtime)
add_library(lib_dynamic SHARED src/lib_dynamic.cc)
include_directories(${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(lib_dynamic PRIVATE tvm_ffi_header tvm_ffi_shared)
if(TVM_FFI_CUBIN_LAUNCHER_USE_DRIVER_API)
  target_link_libraries(lib_dynamic PRIVATE cuda)
else()
  target_link_libraries(lib_dynamic PRIVATE CUDA::cudart)
endif()
add_dependencies(lib_dynamic kernel_cubin)
set_target_properties(
  lib_dynamic
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/"
             PREFIX ""
             SUFFIX ".so"
)
