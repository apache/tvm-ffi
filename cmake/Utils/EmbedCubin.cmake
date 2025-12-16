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

set(CMAKE_CUDA_RUNTIME_LIBRARY None)

# ~~~
# add_tvm_ffi_cubin(<target_name> CUDA <source_files>...)
#
# Creates an object library that compiles CUDA sources to CUBIN format.
# This function uses CMake's native CUDA support and respects CMAKE_CUDA_ARCHITECTURES.
# User can use `CUDA_CUBIN_COMPILATION` after cmake 3.27.
#
# Parameters:
#   target_name: Name of the object library target
#   CUDA: List of CUDA source files
#
# Example:
#   add_tvm_ffi_cubin(my_kernel_cubin CUDA kernel.cu)
# ~~~
function (add_tvm_ffi_cubin target_name)
  cmake_parse_arguments(ARG "" "" "CUDA" ${ARGN})
  if (NOT ARG_CUDA)
    message(FATAL_ERROR "add_tvm_ffi_cubin: CUDA sources are required")
  endif ()

  add_library(${target_name} OBJECT ${ARG_CUDA})
  target_compile_options(${target_name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--cubin>)
endfunction ()

# ~~~
# add_tvm_ffi_fatbin(<target_name> CUDA <source_files>...)
#
# Creates an object library that compiles CUDA sources to FATBIN format.
# This function uses CMake's native CUDA support and respects CMAKE_CUDA_ARCHITECTURES.
# User can use `CUDA_FATBIN_COMPILATION` after cmake 3.27.
#
# Parameters:
#   target_name: Name of the object library target
#   CUDA: List of CUDA source files
#
# Example:
#   add_tvm_ffi_fatbin(my_kernel_fatbin CUDA kernel.cu)
# ~~~
function (add_tvm_ffi_fatbin target_name)
  cmake_parse_arguments(ARG "" "" "CUDA" ${ARGN})
  if (NOT ARG_CUDA)
    message(FATAL_ERROR "add_tvm_ffi_fatbin: CUDA sources are required")
  endif ()

  add_library(${target_name} OBJECT ${ARG_CUDA})
  target_compile_options(${target_name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fatbin>)
endfunction ()

# function(add_tvm_ffi_header target)
#   cmake_parse_arguments(ARG "" "BIN;OUTPUT" "" ${ARGN})
#   if (NOT ARG_BIN)
#     message(FATAL_ERROR "add_tvm_ffi_header: BIN (cubin/fatbin) is required")
#   endif()
#   get_filename_component(BIN_ABS "${ARG_BIN}" ABSOLUTE)
#   if (NOT ARG_OUTPUT)
#     string(REGEX REPLACE "\\.([Cc][Uu][Bb][Ii][Nn]|[Ff][Aa][Tt][Bb][Ii][Nn])$" ".h" OUTPUT_HEADER "${BIN_ABS}")
#     if ("${OUTPUT_HEADER}" STREQUAL "${BIN_ABS}")
#       set(OUTPUT_HEADER "${BIN_ABS}.h")
#     endif()
#   else()
#     set(OUTPUT_HEADER "${ARG_OUTPUT}")
#   endif()
#   get_filename_component(OUT_DIR "${OUTPUT_HEADER}" DIRECTORY)
#   file(MAKE_DIRECTORY "${OUT_DIR}")
#   add_custom_command(
#     OUTPUT "${OUTPUT_HEADER}"
#     COMMAND bin2c -c
#             "${BIN_ABS}" > "${OUTPUT_HEADER}"
#     DEPENDS "${BIN_ABS}"
#     COMMENT "Generating header from ${BIN_ABS} -> ${OUTPUT_HEADER}"
#     VERBATIM
#   )
#   add_library("${target}" INTERFACE)
#   target_include_directories("${target}" INTERFACE
#     $<BUILD_INTERFACE:${OUT_DIR}>
#   )
#   add_custom_target("${target}_generate" DEPENDS "${OUTPUT_HEADER}")
#   add_dependencies("${target}" "${target}_generate")
# endfunction()

# ~~~
# tvm_ffi_embed_cubin(
#   OUTPUT <output_object_file>
#   SOURCE <source_file>
#   CUBIN <cubin_file>
#   NAME <symbol_name>
#   [DEPENDS <additional_dependencies>...]
# )
#
# Compiles a C++ source file and embeds a CUBIN file into it, creating a
# combined object file that can be linked into a shared library or executable.
#
# Parameters:
#   OUTPUT: Path to the output object file (e.g., lib_embedded_with_cubin.o)
#   SOURCE: Path to the C++ source file that uses TVM_FFI_EMBED_CUBIN macro
#   CUBIN: Path to the CUBIN file to embed (can be a file path or a custom target output)
#   NAME: Name used in the TVM_FFI_EMBED_CUBIN macro (e.g., "env" for TVM_FFI_EMBED_CUBIN(env))
#   DEPENDS: Optional additional dependencies (e.g., custom targets)
#
# The function will:
#   1. Compile the SOURCE file to an intermediate object file
#   2. Use the tvm_ffi.utils.embed_cubin Python utility to merge the object file
#      with the CUBIN data
#   3. Create symbols: __tvm_ffi__cubin_<NAME> and __tvm_ffi__cubin_<NAME>_end
#
# Example:
#   tvm_ffi_embed_cubin(
#     OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/lib_embedded_with_cubin.o
#     SOURCE src/lib_embedded.cc
#     CUBIN ${CMAKE_CURRENT_BINARY_DIR}/kernel.cubin
#     NAME env
#   )
#
#   add_library(lib_embedded SHARED ${CMAKE_CURRENT_BINARY_DIR}/lib_embedded_with_cubin.o)
#   target_link_libraries(lib_embedded PRIVATE tvm_ffi_header CUDA::cudart)
#
# Note: The .note.GNU-stack section is automatically added to mark the stack as
#       non-executable, so you don't need to add linker options manually
# ~~~

# cmake-lint: disable=C0111,C0103
function (tvm_ffi_embed_cubin)
  # Parse arguments
  set(options "")
  set(oneValueArgs OUTPUT SOURCE CUBIN NAME)
  set(multiValueArgs DEPENDS)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  # Validate required arguments
  if (NOT ARG_OUTPUT)
    message(FATAL_ERROR "tvm_ffi_embed_cubin: OUTPUT is required")
  endif ()
  if (NOT ARG_SOURCE)
    message(FATAL_ERROR "tvm_ffi_embed_cubin: SOURCE is required")
  endif ()
  if (NOT ARG_CUBIN)
    message(FATAL_ERROR "tvm_ffi_embed_cubin: CUBIN is required")
  endif ()
  if (NOT ARG_NAME)
    message(FATAL_ERROR "tvm_ffi_embed_cubin: NAME is required")
  endif ()

  # Ensure Python is found (prefer virtualenv)
  if (NOT Python_EXECUTABLE)
    set(Python_FIND_VIRTUALENV FIRST)
    find_package(
      Python
      COMPONENTS Interpreter
      REQUIRED
    )
  endif ()

  # Get absolute paths
  get_filename_component(ARG_SOURCE_ABS "${ARG_SOURCE}" ABSOLUTE)
  get_filename_component(ARG_OUTPUT_ABS "${ARG_OUTPUT}" ABSOLUTE)

  # Generate intermediate object file path
  get_filename_component(OUTPUT_DIR "${ARG_OUTPUT_ABS}" DIRECTORY)
  get_filename_component(OUTPUT_NAME "${ARG_OUTPUT_ABS}" NAME_WE)
  set(INTERMEDIATE_OBJ "${OUTPUT_DIR}/${OUTPUT_NAME}_intermediate.o")

  # Get include directories from tvm_ffi_header
  get_target_property(TVM_FFI_INCLUDES tvm_ffi_header INTERFACE_INCLUDE_DIRECTORIES)

  # Convert list to -I flags
  set(INCLUDE_FLAGS "")
  foreach (inc_dir ${TVM_FFI_INCLUDES})
    list(APPEND INCLUDE_FLAGS "-I${inc_dir}")
  endforeach ()

  # Add CUDA include directories if CUDAToolkit is found
  if (TARGET CUDA::cudart)
    get_target_property(CUDA_INCLUDES CUDA::cudart INTERFACE_INCLUDE_DIRECTORIES)
    foreach (inc_dir ${CUDA_INCLUDES})
      list(APPEND INCLUDE_FLAGS "-I${inc_dir}")
    endforeach ()
  endif ()

  # Step 1: Compile source file to intermediate object file
  add_custom_command(
    OUTPUT "${INTERMEDIATE_OBJ}"
    COMMAND ${CMAKE_CXX_COMPILER} -c -fPIC -std=c++17 ${INCLUDE_FLAGS} "${ARG_SOURCE_ABS}" -o
            "${INTERMEDIATE_OBJ}"
    DEPENDS "${ARG_SOURCE_ABS}"
    COMMENT "Compiling ${ARG_SOURCE} to intermediate object file"
    VERBATIM
  )

  # Step 2: Embed CUBIN into the object file using Python utility Note: The Python utility
  # automatically adds .note.GNU-stack section
  add_custom_command(
    OUTPUT "${ARG_OUTPUT_ABS}"
    COMMAND ${Python_EXECUTABLE} -m tvm_ffi.utils.embed_cubin --output-obj "${ARG_OUTPUT_ABS}"
            --input-obj "${INTERMEDIATE_OBJ}" --cubin "${ARG_CUBIN}" --name "${ARG_NAME}"
    DEPENDS "${INTERMEDIATE_OBJ}" "${ARG_CUBIN}" ${ARG_DEPENDS}
    COMMENT "Embedding CUBIN into object file (name: ${ARG_NAME})"
    VERBATIM
  )

  # Set a variable in parent scope so users can add dependencies
  set(${ARG_NAME}_EMBEDDED_OBJ
      "${ARG_OUTPUT_ABS}"
      PARENT_SCOPE
  )
endfunction ()
