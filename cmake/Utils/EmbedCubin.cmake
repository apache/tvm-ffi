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

# We need this to simulate `CUDA_{CUBIN,FATBIN}_COMPILATION` in `add_tvm_ffi_{cubin,fatbin}`, to
# copy `a.cu.o` to `a.cubin`/`a.fatbin`.
set(COPY_SCRIPT "${CMAKE_BINARY_DIR}/cuda_copy_utils.cmake")
file(
  WRITE ${COPY_SCRIPT}
  "
# Arguments: OBJECTS (semicolon-separated list), OUT_DIR, EXT
string(REPLACE \"\\\"\" \"\" ext_strip \"\${EXT}\")
string(REPLACE \"\\\"\" \"\" out_dir_strip \"\${OUT_DIR}\")
foreach(obj_raw \${OBJECTS})
  string(REPLACE \"\\\"\" \"\" obj \"\${obj_raw}\")

  # Extract filename: /path/to/kernel.cu.o -> kernel
  # Note: CMake objects are usually named source.cu.o, so we strip extensions twice.
  get_filename_component(fname \${obj} NAME_WE)
  get_filename_component(fname \${fname} NAME_WE)

  # If OUT_DIR is provided, use it. Otherwise, use the object's directory.
  if(NOT out_dir_strip STREQUAL \"\")
      set(final_dir \"\${out_dir_strip}\")
  else()
      get_filename_component(final_dir \${obj} DIRECTORY)
  endif()

  message(\"Copying \${obj} -> \${final_dir}/\${fname}.\${ext_strip}\")
  execute_process(
    COMMAND \${CMAKE_COMMAND} -E copy_if_different
    \"\${obj}\"
    \"\${final_dir}/\${fname}.\${ext_strip}\"
  )
endforeach()
"
)

# ~~~
# add_tvm_ffi_cubin(<target_name> CUDA <source_file>)
#
# Creates an object library that compiles CUDA source to CUBIN format.
# This function uses CMake's native CUDA support and respects CMAKE_CUDA_ARCHITECTURES.
# This is a compatibility util for cmake < 3.27, user can create
# cmake target with `CUDA_CUBIN_COMPILATION` for cmake >= 3.27.
#
# Parameters:
#   target_name: Name of the object library target
#   CUDA: One CUDA source file
#
# Example:
#   add_tvm_ffi_cubin(my_kernel_cubin CUDA kernel.cu)
# ~~~
function (add_tvm_ffi_cubin target_name)
  cmake_parse_arguments(ARG "" "CUDA" "" ${ARGN})
  if (NOT ARG_CUDA)
    message(FATAL_ERROR "add_tvm_ffi_cubin: CUDA source is required")
  endif ()

  add_library(${target_name} OBJECT ${ARG_CUDA})
  target_compile_options(${target_name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--cubin>)

  add_custom_target(
    ${target_name}_bin ALL
    COMMAND ${CMAKE_COMMAND} -DOBJECTS="$<TARGET_OBJECTS:${target_name}>" -DOUT_DIR="" -DEXT="cubin"
            -P "${COPY_SCRIPT}"
    DEPENDS ${target_name}
    COMMENT "Generating .cubin files for ${target_name}"
    VERBATIM
  )
endfunction ()

# ~~~
# add_tvm_ffi_fatbin(<target_name> CUDA <source_file>)
#
# Creates an object library that compiles CUDA source to FATBIN format.
# This function uses CMake's native CUDA support and respects CMAKE_CUDA_ARCHITECTURES.
# This is a compatibility util for cmake < 3.27, user can create
# cmake target with `CUDA_FATBIN_COMPILATION` for cmake >= 3.27.
#
# Parameters:
#   target_name: Name of the object library target
#   CUDA: One CUDA source file
#
# Example:
#   add_tvm_ffi_fatbin(my_kernel_cubin CUDA kernel.cu)
# ~~~
function (add_tvm_ffi_fatbin target_name)
  cmake_parse_arguments(ARG "" "CUDA" "" ${ARGN})
  if (NOT ARG_CUDA)
    message(FATAL_ERROR "add_tvm_ffi_fatbin: CUDA source is required")
  endif ()

  add_library(${target_name} OBJECT ${ARG_CUDA})
  target_compile_options(${target_name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fatbin>)

  add_custom_target(
    ${target_name}_bin ALL
    COMMAND ${CMAKE_COMMAND} -DOBJECTS="$<TARGET_OBJECTS:${target_name}>" -DOUT_DIR="" -DEXT="cubin"
            -P "${COPY_SCRIPT}"
    DEPENDS ${target_name}
    COMMENT "Generating .fatbin files for ${target_name}"
    VERBATIM
  )
endfunction ()

# ~~~
# tvm_ffi_embed_bin_into(<target_name> <library_name>
#                        BIN <cubin_or_fatbin>
#                        INTERMEDIATE_FILE <intermediate_path>)
#
# Embed one cubin/fatbin into given target with specified library name,
# can be loaded with `TVM_FFI_EMBED_CUBIN(library_name)`.
# Can only have one object in target and one cubin/fatbin.
#
# The reason of this design is to integrate with cmake's workflow.
#
# Parameters:
#   target_name: Name of the object library target
#   library_name: Name of the kernel library
#   BIN: CUBIN or FATBIN file
#   INTERMEDIATE_FILE: Optional, location to copy original object file to.
#
# Example:
#   tvm_ffi_embed_bin_into(lib_embedded env BIN "$<TARGET_OBJECTS:kernel_fatbin>")
# ~~~
function (tvm_ffi_embed_bin_into target_name kernel_name)
  cmake_parse_arguments(ARG "" "BIN;INTERMEDIATE_FILE" "" ${ARGN})

  if (NOT ARG_BIN)
    message(FATAL_ERROR "tvm_ffi_embed_object: BIN is required")
  endif ()

  get_filename_component(LIB_ABS "$<TARGET_OBJECTS:${target_name}>" ABSOLUTE)
  if (NOT INTERMEDIATE_FILE)
    get_filename_component(OUTPUT_DIR_ABS "${LIB_ABS}" DIRECTORY)

    set(final_output "${OUTPUT_DIR_ABS}/${kernel_name}_intermediate.o")
  else ()
    get_filename_component(final_output "${ARG_OUTPUT}" ABSOLUTE)
  endif ()

  add_custom_command(
    TARGET ${target_name}
    PRE_LINK
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_OBJECTS:${target_name}>"
            "${final_output}"
    COMMENT "Moving $<TARGET_OBJECTS:${target_name}> -> ${final_output}"
  )

  add_custom_command(
    TARGET ${target_name}
    PRE_LINK
    COMMAND
      ${Python_EXECUTABLE} -m tvm_ffi.utils.embed_cubin --output-obj
      "$<TARGET_OBJECTS:${target_name}>" --name "${kernel_name}" --input-obj "${FINAL_OUTPUT}"
      --cubin "${ARG_BIN}" DEPENDS
    COMMENT "Embedding CUBIN into object file (name: ${kernel_name})"
    VERBATIM
  )
endfunction ()
