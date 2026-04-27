cmake_minimum_required(VERSION 3.20)
project(
  tvm_ffi_orcjit
  VERSION 0.1.0
  LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---- LLVM_PREFIX handling ----
# scikit-build-core overrides CMAKE_PREFIX_PATH via its init cache, so we read LLVM_PREFIX from the
# environment instead.
if (DEFINED ENV{LLVM_PREFIX})
  list(APPEND CMAKE_PREFIX_PATH "$ENV{LLVM_PREFIX}")
  if (EXISTS "$ENV{LLVM_PREFIX}/Library")
    list(APPEND CMAKE_PREFIX_PATH "$ENV{LLVM_PREFIX}/Library")
  endif ()
endif ()

# ---- Find packages ----
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION} in ${LLVM_DIR}")

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

# ---- Build shared library ----
add_library(
  tvm_ffi_orcjit SHARED
  src/ffi/orcjit_session.cc
  src/ffi/orcjit_dylib.cc
  src/ffi/orcjit_memory_manager.cc
  src/ffi/llvm_patches/gotpcrelx_fix.cc
  src/ffi/llvm_patches/init_fini_plugin.cc
)
set_target_properties(
  tvm_ffi_orcjit PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON
)
target_include_directories(
  tvm_ffi_orcjit PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${LLVM_INCLUDE_DIRS}
)
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
target_compile_definitions(tvm_ffi_orcjit PRIVATE ${LLVM_DEFINITIONS_LIST})

# ---- Static LLVM linking via llvm-config ----
set(_llvm_config_hints ${LLVM_TOOLS_BINARY_DIR} "${LLVM_DIR}/../../../bin")
if (DEFINED ENV{LLVM_PREFIX})
  list(APPEND _llvm_config_hints "$ENV{LLVM_PREFIX}/bin" "$ENV{LLVM_PREFIX}/Library/bin")
endif ()
find_program(
  LLVM_CONFIG_EXE
  NAMES llvm-config-${LLVM_VERSION_MAJOR} llvm-config
  HINTS ${_llvm_config_hints}
)
if (NOT LLVM_CONFIG_EXE)
  message(FATAL_ERROR "llvm-config not found")
endif ()

execute_process(
  COMMAND ${LLVM_CONFIG_EXE} --link-static --libs Core OrcJIT Support native
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE _llvm_libs
)
execute_process(
  COMMAND ${LLVM_CONFIG_EXE} --link-static --ldflags
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE _llvm_ldflags
)
separate_arguments(_llvm_libs_list NATIVE_COMMAND "${_llvm_libs}")
separate_arguments(_llvm_ldflags_list NATIVE_COMMAND "${_llvm_ldflags}")

# ---- zlib (static, from LLVM prefix) ----
cmake_path(GET LLVM_DIR PARENT_PATH _llvm_prefix)
cmake_path(GET _llvm_prefix PARENT_PATH _llvm_prefix)
cmake_path(GET _llvm_prefix PARENT_PATH _llvm_prefix)
set(_lib_hints "${_llvm_prefix}/lib" "${_llvm_prefix}/lib64" "${_llvm_prefix}/Library/lib")

if (WIN32)
  find_library(
    ZLIB_STATIC
    NAMES zlibstatic z
    HINTS ${_lib_hints} REQUIRED
  )
else ()
  find_library(
    ZLIB_STATIC
    NAMES libz.a
    HINTS ${_lib_hints} REQUIRED
  )
endif ()

# ---- zstd (static, from LLVM prefix) ----
if (WIN32)
  find_library(
    ZSTD_STATIC
    NAMES zstd_static zstd
    HINTS ${_lib_hints} REQUIRED
  )
else ()
  find_library(
    ZSTD_STATIC
    NAMES libzstd.a
    HINTS ${_lib_hints} REQUIRED
  )
endif ()

# ---- Link everything ----
target_link_libraries(
  tvm_ffi_orcjit PRIVATE tvm_ffi::header tvm_ffi::shared ${_llvm_ldflags_list} ${_llvm_libs_list}
                         ${ZLIB_STATIC} ${ZSTD_STATIC}
)

# LLVM system libs (ntdll/psapi on Windows, rt/dl on Linux), minus zlib/zstd (linked above).
execute_process(
  COMMAND ${LLVM_CONFIG_EXE} --system-libs
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE _sys_libs
  RESULT_VARIABLE _sys_libs_rc
)
if (_sys_libs_rc EQUAL 0 AND _sys_libs)
  separate_arguments(_sys_libs_list NATIVE_COMMAND "${_sys_libs}")
  list(FILTER _sys_libs_list EXCLUDE REGEX "(zstd|^-lz$|^z\\.lib$|xml2)")
  target_link_libraries(tvm_ffi_orcjit PRIVATE ${_sys_libs_list})
endif ()

# ---- Platform-specific fixups ----
if (APPLE)
  add_custom_command(
    TARGET tvm_ffi_orcjit
    POST_BUILD
    COMMAND install_name_tool -change @rpath/libc++.1.dylib /usr/lib/libc++.1.dylib
            $<TARGET_FILE:tvm_ffi_orcjit>
    COMMENT "Fixing libc++ rpath to use system library"
  )
elseif (UNIX AND NOT WIN32)
  target_link_options(tvm_ffi_orcjit PRIVATE -static-libstdc++ -static-libgcc)
endif ()

# ---- Find and bundle liborc_rt ----
set(ORC_RT_PATH "")
if (DEFINED ENV{ORC_RT_PATH})
  set(ORC_RT_PATH "$ENV{ORC_RT_PATH}")
else ()
  if (WIN32)
    file(GLOB_RECURSE _orc_rt_candidates "${_llvm_prefix}/lib/clang/*/lib/orc_rt*.lib")
    if (NOT _orc_rt_candidates)
      file(GLOB_RECURSE _orc_rt_candidates "${_llvm_prefix}/Library/lib/clang/*/lib/orc_rt*.lib")
    endif ()
  else ()
    file(GLOB_RECURSE _orc_rt_candidates "${_llvm_prefix}/lib/clang/*/lib/liborc_rt*.a")
  endif ()
  if (_orc_rt_candidates)
    list(GET _orc_rt_candidates 0 ORC_RT_PATH)
  endif ()
endif ()

if (NOT ORC_RT_PATH)
  message(WARNING "Could not find liborc_rt. ORC runtime features will be disabled.")
endif ()

if (ORC_RT_PATH)
  message(STATUS "Found liborc_rt: ${ORC_RT_PATH}")
  file(COPY "${ORC_RT_PATH}" DESTINATION "${CMAKE_BINARY_DIR}")
  install(FILES "${ORC_RT_PATH}" DESTINATION lib)
endif ()

# ---- Install ----
install(
  TARGETS tvm_ffi_orcjit
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION lib
)
