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

# ~~~
# tvm_ffi_generate_cubin(
#   OUTPUT <output_cubin_file>
#   SOURCE <cuda_source_file>
#   [ARCH <architecture>]
#   [OPTIONS <extra_nvcc_options>...]
#   [DEPENDS <additional_dependencies>...]
# )
#
# Compiles a CUDA source file to CUBIN format using nvcc.
#
# Parameters:
#   OUTPUT: Path to the output CUBIN file (e.g., kernel.cubin)
#   SOURCE: Path to the CUDA source file (e.g., kernel.cu)
#   ARCH: Target GPU architecture (default: native for auto-detection)
#         Examples: sm_75, sm_80, sm_86, compute_80, native
#   OPTIONS: Additional nvcc compiler options (e.g., -O3, --use_fast_math)
#   DEPENDS: Optional additional dependencies
#
# The function will:
#   1. Find the CUDA compiler (nvcc)
#   2. Compile the SOURCE to CUBIN with specified architecture and options
#   3. Create the output CUBIN file
#
# Example:
#   tvm_ffi_generate_cubin(
#     OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/kernel.cubin
#     SOURCE src/kernel.cu
#     ARCH native
#     OPTIONS -O3 --use_fast_math
#   )
# ~~~

# cmake-lint: disable=C0111,C0103
function (tvm_ffi_generate_cubin)
  # Parse arguments
  set(options "")
  set(oneValueArgs OUTPUT SOURCE ARCH)
  set(multiValueArgs OPTIONS DEPENDS)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  # Validate required arguments
  if (NOT ARG_OUTPUT)
    message(FATAL_ERROR "tvm_ffi_generate_cubin: OUTPUT is required")
  endif ()
  if (NOT ARG_SOURCE)
    message(FATAL_ERROR "tvm_ffi_generate_cubin: SOURCE is required")
  endif ()

  # Default architecture to native if not specified
  if (NOT ARG_ARCH)
    set(ARG_ARCH "native")
  endif ()

  # Ensure CUDA compiler is available
  if (NOT CMAKE_CUDA_COMPILER)
    message(
      FATAL_ERROR
        "tvm_ffi_generate_cubin: CMAKE_CUDA_COMPILER not found. Enable CUDA language in project()."
    )
  endif ()

  # Get absolute paths
  get_filename_component(ARG_SOURCE_ABS "${ARG_SOURCE}" ABSOLUTE)
  get_filename_component(ARG_OUTPUT_ABS "${ARG_OUTPUT}" ABSOLUTE)

  # Build nvcc command
  add_custom_command(
    OUTPUT "${ARG_OUTPUT_ABS}"
    COMMAND ${CMAKE_CUDA_COMPILER} --cubin -arch=${ARG_ARCH} ${ARG_OPTIONS} "${ARG_SOURCE_ABS}" -o
            "${ARG_OUTPUT_ABS}"
    DEPENDS "${ARG_SOURCE_ABS}" ${ARG_DEPENDS}
    COMMENT "Compiling ${ARG_SOURCE} to CUBIN (arch: ${ARG_ARCH})"
    VERBATIM
  )
endfunction ()

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

  # Get include directories from tvm_ffi header target
  if (TARGET tvm_ffi::header)
    set(TVM_FFI_HEADER_TARGET tvm_ffi::header)
  elseif (TARGET tvm_ffi_header)
    set(TVM_FFI_HEADER_TARGET tvm_ffi_header)
  else ()
    message(
      FATAL_ERROR
        "tvm_ffi_embed_cubin: required target 'tvm_ffi::header' or 'tvm_ffi_header' does not exist."
    )
  endif ()
  get_target_property(TVM_FFI_INCLUDES ${TVM_FFI_HEADER_TARGET} INTERFACE_INCLUDE_DIRECTORIES)

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
