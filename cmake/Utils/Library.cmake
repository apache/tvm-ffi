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
# tvm_ffi_add_prefix_map(target_name, prefix_path)
# Add a compile prefix map so absolute paths under `prefix_path` are remapped to a stable,
# relative form for reproducible builds and cleaner diagnostics.
#
# Parameters:
#   target_name: CMake target to modify
#   prefix_path: Absolute path prefix to remap
# ~~~
function (tvm_ffi_add_prefix_map target_name prefix_path)
  # Add prefix map so the path displayed becomes relative to prefix_path
  if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target_name} PRIVATE "-ffile-prefix-map=${prefix_path}/=")
  endif ()
endfunction ()

# ~~~
# tvm_ffi_add_apple_dsymutil(target_name)
# On Apple platforms, run `dsymutil` post-build to generate debug symbols for better backtraces.
# No-ops on non-Apple platforms.
#
# Parameters:
#   target_name: CMake target to attach post-build step
# ~~~
function (tvm_ffi_add_apple_dsymutil target_name)
  # running dsymutil on macos to generate debugging symbols for backtraces
  if (APPLE)
    find_program(DSYMUTIL dsymutil)
    mark_as_advanced(DSYMUTIL)
    add_custom_command(
      TARGET ${target_name}
      POST_BUILD
      COMMAND ${DSYMUTIL} ARGS $<TARGET_FILE:${target_name}>
      COMMENT "[COMMAND] dsymutil $<TARGET_FILE:${target_name}>"
      VERBATIM
    )
  endif ()
endfunction ()

# ~~~
# tvm_ffi_add_msvc_flags(target_name)
# Apply MSVC-specific definitions and flags to improve build compatibility and warnings behavior
# on Windows.
#
# Parameters:
#   target_name: CMake target to modify
# ~~~
function (tvm_ffi_add_msvc_flags target_name)
  # running if we are under msvc
  if (MSVC)
    target_compile_definitions(${target_name} PUBLIC -DWIN32_LEAN_AND_MEAN)
    target_compile_definitions(${target_name} PUBLIC -D_CRT_SECURE_NO_WARNINGS)
    target_compile_definitions(${target_name} PUBLIC -D_SCL_SECURE_NO_WARNINGS)
    target_compile_definitions(${target_name} PUBLIC -D_ENABLE_EXTENDED_ALIGNED_STORAGE)
    target_compile_definitions(${target_name} PUBLIC -DNOMINMAX)
    target_compile_options(${target_name} PRIVATE "/Zi")
  endif ()
endfunction ()

# ~~~
# tvm_ffi_add_target_from_obj(target_name, obj_target_name)
# Create static and shared library targets from an object library and set output directories
# consistently across platforms. Also runs dsymutil on Apple for the shared target.
#
# Parameters:
#   target_name: Base name for created targets
#   obj_target_name: Object library to link into the outputs
# ~~~
function (tvm_ffi_add_target_from_obj target_name obj_target_name)
  add_library(${target_name}_static STATIC $<TARGET_OBJECTS:${obj_target_name}>)
  set_target_properties(
    ${target_name}_static
    PROPERTIES OUTPUT_NAME "${target_name}_static"
               ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
               LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
               RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  )
  add_library(${target_name}_shared SHARED $<TARGET_OBJECTS:${obj_target_name}>)
  set_target_properties(
    ${target_name}_shared
    PROPERTIES OUTPUT_NAME "${target_name}"
               ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
               LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
               RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  )
  add_library(${target_name}_testing SHARED)
  set_target_properties(
    ${target_name}_testing
    PROPERTIES OUTPUT_NAME "${target_name}_testing"
               ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
               LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
               RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  )
  if (WIN32)
    target_compile_definitions(${obj_target_name} PRIVATE TVM_FFI_EXPORTS)
    # set the output directory for each config type so msbuild also get into lib without appending
    # the config type to the output directory do both Release and RELEASE suffix, since while cmake
    # docs suggest Release is ok. real runs on MSbuild suggest that we might need RELEASE instead
    foreach (config_type Release RELEASE)
      set_target_properties(
        ${target_name}_shared
        PROPERTIES RUNTIME_OUTPUT_DIRECTORY_${config_type} "${CMAKE_BINARY_DIR}/lib"
                   LIBRARY_OUTPUT_DIRECTORY_${config_type} "${CMAKE_BINARY_DIR}/lib"
                   ARCHIVE_OUTPUT_DIRECTORY_${config_type} "${CMAKE_BINARY_DIR}/lib"
      )
      set_target_properties(
        ${target_name}_static
        PROPERTIES RUNTIME_OUTPUT_DIRECTORY_${config_type} "${CMAKE_BINARY_DIR}/lib"
                   LIBRARY_OUTPUT_DIRECTORY_${config_type} "${CMAKE_BINARY_DIR}/lib"
                   ARCHIVE_OUTPUT_DIRECTORY_${config_type} "${CMAKE_BINARY_DIR}/lib"
      )
      set_target_properties(
        ${target_name}_testing
        PROPERTIES RUNTIME_OUTPUT_DIRECTORY_${config_type} "${CMAKE_BINARY_DIR}/lib"
                   LIBRARY_OUTPUT_DIRECTORY_${config_type} "${CMAKE_BINARY_DIR}/lib"
                   ARCHIVE_OUTPUT_DIRECTORY_${config_type} "${CMAKE_BINARY_DIR}/lib"
      )
    endforeach ()
  endif ()
  tvm_ffi_add_apple_dsymutil(${target_name}_shared)
  tvm_ffi_add_apple_dsymutil(${target_name}_testing)
endfunction ()

# cmake-lint: disable=C0301,R0912,R0915
# ~~~
# tvm_ffi_configure_target(
#   target_name
#   [LINK_SHARED ON|OFF] [LINK_HEADER ON|OFF] [DEBUG_SYMBOL ON|OFF] [MSVC_FLAGS ON|OFF]
#   [STUB_INIT ON|OFF] [STUB_DIR <dir>] [STUB_PKG <pkg>] [STUB_PREFIX <prefix>]
# )
# Configure a target to integrate with TVM-FFI CMake utilities:
#   - Link against tvm_ffi::header and/or tvm_ffi::shared
#   - Always apply tvm_ffi_add_prefix_map(target_name <current source dir>)
#   - Enable Apple dSYM generation via tvm_ffi_add_apple_dsymutil(target_name)
#   - Apply MSVC-specific flags via tvm_ffi_add_msvc_flags(target_name)
#   - Add post-build step to generate Python stubs via tvm_ffi.stub.cli
#
# Parameters:
#   target_name: Existing CMake target to modify (positional, required)
#
# Keyword parameters:
#   LINK_SHARED:  Whether to link tvm_ffi::shared into the target (default: ON; ON/OFF-style)
#   LINK_HEADER:  Whether to link tvm_ffi::header into the target (default: ON; ON/OFF-style)
#   DEBUG_SYMBOL: Whether to enable debug symbol post-processing hooks.
#                 On Apple this calls tvm_ffi_add_apple_dsymutil(target_name) (default: ON; ON/OFF-style)
#                 On non-Apple platforms this is currently a no-op unless you extend it. (default: ON)
#   MSVC_FLAGS:   Whether to call tvm_ffi_add_msvc_flags(target_name) to apply MSVC-specific flags (default: ON; ON/OFF-style)
#   STUB_DIR:     Stub generation runs when this is set. Directory to generate Python stubs. Relative paths resolve against CMAKE_CURRENT_SOURCE_DIR.
#   STUB_INIT:    Whether to allow generating new directives. Default: OFF (ON/OFF-style)
#   STUB_PKG:     Package name passed to stub generator (requires STUB_DIR and STUB_INIT=ON; default: ${SKBUILD_PROJECT_NAME} if set, otherwise target name)
#   STUB_PREFIX:  Module prefix passed to stub generator (requires STUB_DIR and STUB_INIT=ON; default: "<STUB_PKG>.")
# ~~~
function (tvm_ffi_configure_target target)
  if (NOT target)
    message(
      FATAL_ERROR
        "tvm_ffi_configure_target: missing target name. "
        "Usage: tvm_ffi_configure_target(<target> [LINK_SHARED ON|OFF] [LINK_HEADER ON|OFF] [DEBUG_SYMBOL ON|OFF] [MSVC_FLAGS ON|OFF] [STUB_INIT ON|OFF] [STUB_DIR <dir>] [STUB_PKG <pkg>] [STUB_PREFIX <prefix>])"
    )
  endif ()

  if (NOT TARGET "${target}")
    message(FATAL_ERROR "tvm_ffi_configure_target: '${target}' is not an existing CMake target.")
  endif ()

  # Parse keyword args after the positional target name.
  set(tvm_ffi_arg_options) # none; require explicit ON/OFF style values
  set(tvm_ffi_arg_oneValueArgs
      LINK_SHARED
      LINK_HEADER
      DEBUG_SYMBOL
      MSVC_FLAGS
      STUB_INIT
      STUB_DIR
      STUB_PKG
      STUB_PREFIX
  )
  set(tvm_ffi_arg_multiValueArgs)

  cmake_parse_arguments(
    tvm_ffi_arg_ "${tvm_ffi_arg_options}" "${tvm_ffi_arg_oneValueArgs}"
    "${tvm_ffi_arg_multiValueArgs}" ${ARGN}
  )

  # Defaults
  foreach (arg IN ITEMS LINK_SHARED LINK_HEADER DEBUG_SYMBOL MSVC_FLAGS)
    if (NOT DEFINED tvm_ffi_arg__${arg})
      set(tvm_ffi_arg__${arg} ON)
    endif ()
  endforeach ()
  if (NOT DEFINED tvm_ffi_arg__STUB_INIT)
    set(tvm_ffi_arg__STUB_INIT OFF)
  endif ()

  # Validation
  if ((NOT DEFINED tvm_ffi_arg__STUB_DIR) OR (NOT tvm_ffi_arg__STUB_DIR))
    if (DEFINED tvm_ffi_arg__STUB_PKG OR DEFINED tvm_ffi_arg__STUB_PREFIX)
      message(
        FATAL_ERROR
          "tvm_ffi_configure_target(${target}): STUB_PKG/STUB_PREFIX require STUB_DIR to be set."
      )
    endif ()
  endif ()
  if (NOT tvm_ffi_arg__STUB_INIT)
    if (DEFINED tvm_ffi_arg__STUB_PKG OR DEFINED tvm_ffi_arg__STUB_PREFIX)
      message(
        FATAL_ERROR
          "tvm_ffi_configure_target(${target}): STUB_PKG/STUB_PREFIX cannot be set when STUB_INIT is OFF."
      )
    endif ()
  else ()
    if (NOT DEFINED tvm_ffi_arg__STUB_DIR OR NOT tvm_ffi_arg__STUB_DIR)
      message(
        FATAL_ERROR "tvm_ffi_configure_target(${target}): STUB_INIT=ON requires STUB_DIR to be set."
      )
    endif ()
  endif ()

  # STUB_PKG and STUB_PREFIX defaults
  if (tvm_ffi_arg__STUB_INIT AND tvm_ffi_arg__STUB_DIR)
    if (NOT DEFINED tvm_ffi_arg__STUB_PKG)
      if (DEFINED SKBUILD_PROJECT_NAME AND SKBUILD_PROJECT_NAME)
        set(tvm_ffi_arg__STUB_PKG "${SKBUILD_PROJECT_NAME}")
      else ()
        set(tvm_ffi_arg__STUB_PKG "${target}")
      endif ()
    endif ()
    if (NOT DEFINED tvm_ffi_arg__STUB_PREFIX)
      set(tvm_ffi_arg__STUB_PREFIX "${tvm_ffi_arg__STUB_PKG}.")
    endif ()
  endif ()

  # Always-on prefix map
  if (COMMAND tvm_ffi_add_prefix_map)
    tvm_ffi_add_prefix_map("${target}" "${CMAKE_CURRENT_SOURCE_DIR}")
  else ()
    message(
      FATAL_ERROR
        "tvm_ffi_configure_target(${target}): required function 'tvm_ffi_add_prefix_map' is not defined/included."
    )
  endif ()

  # LINK_HEADER
  if (tvm_ffi_arg__LINK_HEADER)
    if (TARGET tvm_ffi::header)
      target_link_libraries("${target}" PRIVATE tvm_ffi::header)
    else ()
      message(
        FATAL_ERROR
          "tvm_ffi_configure_target(${target}): LINK_HEADER requested but targets 'tvm_ffi::header' do not exist."
      )
    endif ()
  endif ()

  # LINK_SHARED
  if (tvm_ffi_arg__LINK_SHARED)
    if (TARGET tvm_ffi::shared)
      target_link_libraries("${target}" PRIVATE tvm_ffi::shared)
    else ()
      message(
        FATAL_ERROR
          "tvm_ffi_configure_target(${target}): LINK_SHARED requested but targets 'tvm_ffi::shared' do not exist."
      )
    endif ()
  endif ()

  # DEBUG_SYMBOL (default ON). Apple behavior only (hook only; installation handled by
  # tvm_ffi_install()).
  if (tvm_ffi_arg__DEBUG_SYMBOL)
    if (APPLE)
      if (COMMAND tvm_ffi_add_apple_dsymutil)
        tvm_ffi_add_apple_dsymutil("${target}")
      else ()
        message(
          FATAL_ERROR
            "tvm_ffi_configure_target(${target}): DEBUG_SYMBOL=ON but 'tvm_ffi_add_apple_dsymutil' is not defined/included."
        )
      endif ()
    endif ()
  endif ()

  # Optional: MSVC flags
  if (tvm_ffi_arg__MSVC_FLAGS)
    if (COMMAND tvm_ffi_add_msvc_flags)
      tvm_ffi_add_msvc_flags("${target}")
    else ()
      message(
        FATAL_ERROR
          "tvm_ffi_configure_target(${target}): MSVC_FLAGS=ON but 'tvm_ffi_add_msvc_flags' is not defined/included."
      )
    endif ()
  endif ()

  if (DEFINED tvm_ffi_arg__STUB_DIR AND tvm_ffi_arg__STUB_DIR)
    get_filename_component(
      tvm_ffi_arg__STUB_DIR_ABS "${tvm_ffi_arg__STUB_DIR}" ABSOLUTE BASE_DIR
      "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    find_package(
      Python3
      COMPONENTS Interpreter
      REQUIRED
    )
    set(tvm_ffi_stub_cli_args "${tvm_ffi_arg__STUB_DIR_ABS}" --dlls $<TARGET_FILE:${target}>)
    if (tvm_ffi_arg__STUB_INIT)
      list(
        APPEND
        tvm_ffi_stub_cli_args
        --init-lib
        ${target}
        --init-pypkg
        "${tvm_ffi_arg__STUB_PKG}"
        --init-prefix
        "${tvm_ffi_arg__STUB_PREFIX}"
      )
    endif ()
    add_custom_command(
      TARGET ${target}
      POST_BUILD
      COMMAND ${Python3_EXECUTABLE} -m tvm_ffi.stub.cli ${tvm_ffi_stub_cli_args}
      COMMENT
        "[COMMAND] Running: ${Python3_EXECUTABLE} -m tvm_ffi.stub.cli ${tvm_ffi_stub_cli_args}"
      VERBATIM
    )
  endif ()
endfunction ()

# ~~~
# tvm_ffi_install(target_name [DESTINATION <dir>])
# Install TVM-FFI related artifacts for a configured target.
#
# Parameters:
#   target_name: Existing CMake target whose artifacts should be installed
#
# Keyword parameters:
#   DESTINATION: Install destination directory relative to CMAKE_INSTALL_PREFIX (default: ".")
#
# Behavior:
#   - On Apple, installs the target's dSYM bundle if it exists.
#     This uses generator expressions and OPTIONAL so it does not fail if the dSYM is absent.
#   - On non-Apple platforms, currently no-op (extend as needed for PDB/DWARF packaging).
#
# Notes:
#   - This function does not create dSYMs; it only installs them if present.
#     Pair it with tvm_ffi_configure_target(... DEBUG_SYMBOL ON) to enable dSYM generation hooks.
# ~~~
function (tvm_ffi_install target)
  if (NOT target)
    message(
      FATAL_ERROR
        "tvm_ffi_install: missing target name. Usage: tvm_ffi_install(<target> [DESTINATION <dir>])"
    )
  endif ()

  if (NOT TARGET "${target}")
    message(FATAL_ERROR "tvm_ffi_install: '${target}' is not an existing CMake target.")
  endif ()

  set(tvm_ffi_install_options) # none
  set(tvm_ffi_install_oneValueArgs DESTINATION)
  set(tvm_ffi_install_multiValueArgs)

  cmake_parse_arguments(
    tvm_ffi_install_ "${tvm_ffi_install_options}" "${tvm_ffi_install_oneValueArgs}"
    "${tvm_ffi_install_multiValueArgs}" ${ARGN}
  )

  if (NOT DEFINED tvm_ffi_install__DESTINATION)
    set(tvm_ffi_install__DESTINATION ".")
  endif ()

  if (APPLE)
    # Install target dSYM bundle if present.
    install(
      DIRECTORY "$<TARGET_FILE:${target}>.dSYM"
      DESTINATION "${tvm_ffi_install__DESTINATION}"
      OPTIONAL
    )
  endif ()
endfunction ()
