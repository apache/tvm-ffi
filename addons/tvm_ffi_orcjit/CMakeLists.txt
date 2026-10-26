cmake_minimum_required(VERSION 3.24)
project(
  tvm_ffi_orcjit
  VERSION 0.1.3
  LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(
  Python
  COMPONENTS Interpreter
  REQUIRED
)

# ---- LLVM static SDK ----
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m mlc_llvm_static.config --prefix
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE _llvm_prefix COMMAND_ERROR_IS_FATAL ANY
)
list(PREPEND CMAKE_PREFIX_PATH "${_llvm_prefix}")
set(LLVM_DIR "${_llvm_prefix}/lib/cmake/llvm")
set(zstd_DIR "${_llvm_prefix}/lib/cmake/zstd") # cmake-lint: disable=C0103
set(ZLIB_USE_STATIC_LIBS ON)
# Discard system-zlib results cached by builds using the old manual link setup.
unset(ZLIB_INCLUDE_DIR CACHE)
unset(ZLIB_LIBRARY_RELEASE CACHE)
unset(ZLIB_LIBRARY_DEBUG CACHE)
set(MLC_LLVM_STATIC_PROFILE native)
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION} in ${LLVM_DIR}")

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m tvm_ffi.config --cmakedir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE tvm_ffi_ROOT
)
find_package(tvm_ffi CONFIG REQUIRED)

# ---- Build shared library ----
add_library(
  tvm_ffi_orcjit SHARED src/ffi/orcjit_session.cc src/ffi/orcjit_dylib.cc
                        src/ffi/orcjit_memory_manager.cc src/ffi/orcjit_slab.cc
)
if (APPLE)
  target_sources(
    tvm_ffi_orcjit PRIVATE src/ffi/llvm_patches/init_fini_plugin.cc
                           src/ffi/llvm_patches/macho_cxa_atexit_shim.cc
  )
elseif (WIN32)
  target_sources(
    tvm_ffi_orcjit
    PRIVATE src/ffi/llvm_patches/init_fini_plugin.cc src/ffi/llvm_patches/win_coff_pdata_strip.cc
            src/ffi/llvm_patches/win_dll_import_generator.cc
  )
  if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(ARM64|arm64|aarch64)$")
    target_sources(tvm_ffi_orcjit PRIVATE src/ffi/llvm_patches/win_arm64_rtdyld.cc)
    target_compile_definitions(tvm_ffi_orcjit PRIVATE TVM_FFI_ORCJIT_WINDOWS_ARM64=1)
  endif ()
endif ()
set_target_properties(
  tvm_ffi_orcjit PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON
)
target_include_directories(
  tvm_ffi_orcjit PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${LLVM_INCLUDE_DIRS}
)
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND "${LLVM_DEFINITIONS}")
target_compile_definitions(tvm_ffi_orcjit PRIVATE ${LLVM_DEFINITIONS_LIST})
# LLVM-facing subclasses must use the SDK's RTTI mode: otherwise their typeinfo references LLVM
# base-class typeinfo that a no-RTTI SDK does not provide.
if (NOT LLVM_ENABLE_RTTI)
  if (MSVC)
    target_compile_options(tvm_ffi_orcjit PRIVATE /GR-)
  else ()
    target_compile_options(tvm_ffi_orcjit PRIVATE -fno-rtti)
  endif ()
endif ()

# Imported targets carry the static compression libraries and platform system dependencies.
llvm_map_components_to_libnames(LLVM_LIBS Core OrcJIT Support native)
target_link_libraries(tvm_ffi_orcjit PRIVATE tvm_ffi::header tvm_ffi::shared ${LLVM_LIBS})

# Export only the explicit initialization API. In particular, do not publish symbols from the
# statically linked LLVM/zlib/zstd archives: the addon is loaded into host processes that may carry
# a different LLVM, and interposition between those copies is unsafe. On ELF, --exclude-libs,ALL
# also gives archive members local binding before the version script trims the dynamic symbol table.
if (CMAKE_SYSTEM_NAME MATCHES "Linux|Android|FreeBSD|NetBSD|OpenBSD" AND CMAKE_CXX_COMPILER_ID
                                                                         MATCHES "GNU|Clang"
)
  target_compile_options(tvm_ffi_orcjit PRIVATE -ffunction-sections -fdata-sections)
  target_link_options(
    tvm_ffi_orcjit PRIVATE "LINKER:--exclude-libs,ALL" "LINKER:--gc-sections"
    "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/cmake/orcjit.lds"
  )
elseif (APPLE)
  target_link_options(
    tvm_ffi_orcjit PRIVATE
    "LINKER:-exported_symbols_list,${CMAKE_CURRENT_SOURCE_DIR}/cmake/orcjit.macos.lds"
  )
endif ()

# ---- Find and embed liborc_rt (Linux/ELF only) ----
# The archive bytes are baked into the .so's .rodata (see src/ffi/orc_rt_embed.S.in) and handed to
# ExecutorNativePlatform as a zero-copy MemoryBuffer at runtime, so the wheel ships one artifact
# with no on-disk path lookup. Platform notes:
#
# * Linux: liborc_rt drives ExecutorNativePlatform → ELFNixPlatform.
# * macOS: MachOPlatform is skipped to sidestep the compact-unwind delta bug (see
#   src/ffi/orcjit_session.cc); liborc_rt is unused.
# * Windows: COFFPlatform is not hooked up; liborc_rt is unused.
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
  file(GLOB_RECURSE _orc_rt_candidates
       "${_llvm_prefix}/lib/clang/${LLVM_VERSION_MAJOR}/lib/liborc_rt*.a"
  )
  if (NOT _orc_rt_candidates)
    message(FATAL_ERROR "mlc-llvm-static is missing the Linux liborc_rt archive")
  endif ()
  list(GET _orc_rt_candidates 0 _orc_rt_source)
  # Debug information inside the embedded archive survives stripping the outer shared library. Strip
  # a build-directory copy, preserving the SDK and the archive's linkable symbols.
  set(ORC_RT_PATH "${CMAKE_CURRENT_BINARY_DIR}/liborc_rt-stripped.a")
  add_custom_command(
    OUTPUT "${ORC_RT_PATH}"
    COMMAND "${CMAKE_COMMAND}" -E copy "${_orc_rt_source}" "${ORC_RT_PATH}"
    COMMAND "${CMAKE_STRIP}" --strip-debug "${ORC_RT_PATH}"
    DEPENDS "${_orc_rt_source}"
    COMMENT "Stripping debug information from embedded ORC runtime"
    VERBATIM
  )
  set_source_files_properties("${ORC_RT_PATH}" PROPERTIES HEADER_FILE_ONLY TRUE)
  message(STATUS "Embedding liborc_rt: ${ORC_RT_PATH}")
  enable_language(ASM)
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ffi/orc_rt_embed.S.in"
    "${CMAKE_CURRENT_BINARY_DIR}/orc_rt_embed.S" @ONLY
  )
  target_sources(
    tvm_ffi_orcjit PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/orc_rt_embed.S" "${ORC_RT_PATH}"
  )
  target_compile_definitions(tvm_ffi_orcjit PRIVATE TVM_FFI_ORCJIT_EMBED_ORC_RT=1)
  # Re-run .incbin if the archive itself changes (path is stable across builds).
  set_property(
    SOURCE "${CMAKE_CURRENT_BINARY_DIR}/orc_rt_embed.S" PROPERTY OBJECT_DEPENDS "${ORC_RT_PATH}"
  )
endif ()

# ---- Install ----
install(
  TARGETS tvm_ffi_orcjit
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION lib
)
