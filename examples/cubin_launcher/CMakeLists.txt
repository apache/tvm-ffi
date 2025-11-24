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

# Step 1: Compile kernel.cu to CUBIN We use a custom command to compile directly to CUBIN with nvcc
# Use -arch=native to automatically detect the GPU architecture
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/kernel.cubin
  COMMAND ${CMAKE_CUDA_COMPILER} --cubin -arch=native ${CMAKE_CURRENT_SOURCE_DIR}/src/kernel.cu -o
          ${CMAKE_CURRENT_BINARY_DIR}/kernel.cubin
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/kernel.cu
  COMMENT "Compiling kernel.cu to CUBIN for native GPU architecture"
)

add_custom_target(
  generate_cubin ALL
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/kernel.cubin
  COMMENT "Generating CUBIN file"
)

# Step 2: Generate object file with embedded CUBIN data for linking This creates symbols:
# __tvm_ffi__cubin_env, __tvm_ffi__cubin_env_end
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/kernel_cubin_data.o
  COMMAND ${CMAKE_LINKER} -r -b binary -o ${CMAKE_CURRENT_BINARY_DIR}/kernel_cubin_data.o
          kernel.cubin
  COMMAND
    ${CMAKE_OBJCOPY} --rename-section .data=.rodata,alloc,load,readonly,data,contents --redefine-sym
    _binary_kernel_cubin_start=__tvm_ffi__cubin_env --redefine-sym
    _binary_kernel_cubin_end=__tvm_ffi__cubin_env_end --redefine-sym
    _binary_kernel_cubin_size=__tvm_ffi__cubin_env_size
    ${CMAKE_CURRENT_BINARY_DIR}/kernel_cubin_data.o
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/kernel.cubin
  COMMENT "Embedding CUBIN as object file"
)

add_custom_target(
  embed_cubin
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/kernel_cubin_data.o
  COMMENT "Embedding CUBIN as object file"
)

# Step 3: Build lib_embedded shared library (with embedded CUBIN)
add_library(lib_embedded SHARED src/lib_embedded.cc ${CMAKE_CURRENT_BINARY_DIR}/kernel_cubin_data.o)
target_link_libraries(
  lib_embedded PRIVATE tvm_ffi_header tvm_ffi_shared CUDA::cuda_driver CUDA::cudart
)
add_dependencies(lib_embedded embed_cubin)
set_target_properties(
  lib_embedded
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/"
             PREFIX ""
             SUFFIX ".so"
)

# Mark the embedded object file's stack as non-executable. This is necessary because the object file
# created by 'ld -r -b binary' does not include a .note.GNU-stack section, which causes the linker
# to warn about an executable stack. The -z,noexecstack flag ensures compatibility with security
# policies and future linker versions that will enforce non-executable stacks by default.
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
