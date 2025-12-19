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

# Do not let cmake to link cudart.
set(CMAKE_CUDA_RUNTIME_LIBRARY None)

set(OBJECT_COPY_UTIL "${CMAKE_CURRENT_LIST_DIR}/ObjectCopyUtil.cmake")

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
            -P "${OBJECT_COPY_UTIL}"
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
    COMMAND ${CMAKE_COMMAND} -DOBJECTS="$<TARGET_OBJECTS:${target_name}>" -DOUT_DIR=""
            -DEXT="fatbin" -P "${OBJECT_COPY_UTIL}"
    DEPENDS ${target_name}
    COMMENT "Generating .fatbin files for ${target_name}"
    VERBATIM
  )
endfunction ()

# ~~~
# tvm_ffi_embed_bin_into(<target_name>
#                        SYMBOL <symbol_name>
#                        BIN <cubin_or_fatbin>)
#
# Embed one cubin/fatbin into given target with specified library name,
# can be loaded with `TVM_FFI_EMBED_CUBIN(symbol_name)`.
# Can only have one object in target and one cubin/fatbin.
#
# The reason of this design is to integrate with cmake's workflow.
#
# Parameters:
#   target_name: Name of the object library target
#   symbol_name: Name of the symbol in TVM_FFI_EMBED_CUBIN macro.
#   BIN: CUBIN or FATBIN file
#
# Example:
#   tvm_ffi_embed_bin_into(lib_embedded SYMBOL env BIN "$<TARGET_OBJECTS:kernel_fatbin>")
# ~~~
function (tvm_ffi_embed_bin_into target_name)
  cmake_parse_arguments(ARG "" "SYMBOL;BIN" "" ${ARGN})

  if (NOT ARG_BIN)
    message(FATAL_ERROR "tvm_ffi_embed_object: BIN is required")
  endif ()
  if (NOT ARG_SYMBOL)
    message(FATAL_ERROR "tvm_ffi_embed_object: SYMBOL is required")
  endif ()

  get_filename_component(LIB_ABS "$<TARGET_OBJECTS:${target_name}>" ABSOLUTE)
  get_filename_component(OUTPUT_DIR_ABS "${LIB_ABS}" DIRECTORY)
  set(intermediate_path "${OUTPUT_DIR_ABS}/${kernel_name}_intermediate.o")

  add_custom_command(
    TARGET ${target_name}
    PRE_LINK
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_OBJECTS:${target_name}>"
            "${intermediate_path}"
    COMMENT "Moving $<TARGET_OBJECTS:${target_name}> -> ${intermediate_path}"
  )

  add_custom_command(
    TARGET ${target_name}
    PRE_LINK
    COMMAND
      ${Python_EXECUTABLE} -m tvm_ffi.utils.embed_cubin --output-obj
      "$<TARGET_OBJECTS:${target_name}>" --name "${ARG_SYMBOL}" --input-obj "${intermediate_path}"
      --cubin "${ARG_BIN}" DEPENDS
    COMMENT "Embedding CUBIN into object file (name: ${ARG_SYMBOL})"
    VERBATIM
  )
endfunction ()
