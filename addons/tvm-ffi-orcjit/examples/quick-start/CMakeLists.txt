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

# Create object library (not a shared library, just object files)
add_library(add_obj OBJECT add.cc)
target_link_libraries(add_obj PRIVATE tvm_ffi::header)

# Set compiler flags
target_compile_options(add_obj PRIVATE -fPIC -O2)

# Custom target to copy the object file to the example directory
add_custom_target(
  copy_obj_file ALL
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_OBJECTS:add_obj> ${CMAKE_CURRENT_SOURCE_DIR}/add.o
  COMMAND ${CMAKE_COMMAND} -E echo "Successfully compiled add.o"
  COMMAND ${CMAKE_COMMAND} -E echo ""
  COMMAND ${CMAKE_COMMAND} -E echo "You can now run: python run.py"
  DEPENDS add_obj
  COMMENT "Copying add.o to example directory"
)
