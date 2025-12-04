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

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Run `python -m tvm_ffi.config --cmakedir` to find tvm-ffi package
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

# Create object library for test functions
add_library(test_funcs_obj OBJECT sources/test_funcs.cc)
target_link_libraries(test_funcs_obj PRIVATE tvm_ffi_header)
target_compile_options(test_funcs_obj PRIVATE -fPIC -O2)
install(
  FILES $<TARGET_OBJECTS:test_funcs_obj>
  DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}
  RENAME test_funcs.o
)

# Create object library for second set of test functions
add_library(test_funcs2_obj OBJECT sources/test_funcs2.cc)
target_link_libraries(test_funcs2_obj PRIVATE tvm_ffi_header)
target_compile_options(test_funcs2_obj PRIVATE -fPIC -O2)
install(
  FILES $<TARGET_OBJECTS:test_funcs2_obj>
  DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}
  RENAME test_funcs2.o
)

# Create object library for conflicting test functions
add_library(test_funcs_conflict_obj OBJECT sources/test_funcs_conflict.cc)
target_link_libraries(test_funcs_conflict_obj PRIVATE tvm_ffi_header)
target_compile_options(test_funcs_conflict_obj PRIVATE -fPIC -O2)
install(
  FILES $<TARGET_OBJECTS:test_funcs_conflict_obj>
  DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}
  RENAME test_funcs_conflict.o
)


find_package(CUDAToolkit)

if(CUDAToolkit_FOUND)
  enable_language(CUDA)
  message(STATUS "CUDA found: ${CUDAToolkit_VERSION}")
  add_library(test_funcs_cuda_obj OBJECT sources/test_funcs_cuda.cu)
  target_link_libraries(test_funcs_cuda_obj PRIVATE tvm_ffi_header)
  target_compile_options(test_funcs_cuda_obj PRIVATE -fPIC -O2)
  install(
    FILES $<TARGET_OBJECTS:test_funcs_cuda_obj>
    DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}
    RENAME test_funcs_cuda.o
  )
endif()
