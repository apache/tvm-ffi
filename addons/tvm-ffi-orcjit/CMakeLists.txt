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

# Find dependencies
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# Find tvm-ffi package using the same method as quickstart example
find_package(
  Python
  COMPONENTS Interpreter
  REQUIRED
)

# Get the cmake directory for tvm-ffi
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m tvm_ffi.config --cmakedir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE tvm_ffi_ROOT
)
message(STATUS "tvm_ffi_ROOT: ${tvm_ffi_ROOT}")

find_package(tvm_ffi CONFIG REQUIRED)

# Source files
set(SOURCES src/ffi/orcjit_session.cc src/ffi/orcjit_dylib.cc)

# Build shared library
add_library(tvm_ffi_orcjit SHARED ${SOURCES})

target_include_directories(
  tvm_ffi_orcjit PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${LLVM_INCLUDE_DIRS}
)

# Compile definitions
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
target_compile_definitions(tvm_ffi_orcjit PRIVATE ${LLVM_DEFINITIONS_LIST})

# ---- Static LLVM Linking via llvm-config ----
# Allow overriding via env: LLVM_CONFIG_PATH
if(DEFINED ENV{LLVM_CONFIG_PATH})
  set(LLVM_CONFIG_EXE "$ENV{LLVM_CONFIG_PATH}")
else()
  find_program(
    LLVM_CONFIG_EXE
    NAMES llvm-config-${LLVM_VERSION_MAJOR} llvm-config
    HINTS ${LLVM_TOOLS_BINARY_DIR} "${LLVM_DIR}/../../../bin"
  )
endif()
if(NOT LLVM_CONFIG_EXE)
  message(FATAL_ERROR "llvm-config not found, required for static LLVM linking")
endif()
message(STATUS "Using llvm-config: ${LLVM_CONFIG_EXE}")

execute_process(
  COMMAND ${LLVM_CONFIG_EXE} --link-static --libs Core OrcJIT Support native
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE _llvm_static_libs
)
execute_process(
  COMMAND ${LLVM_CONFIG_EXE} --link-static --ldflags
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE _llvm_static_ldflags
)
message(STATUS "LLVM static libs: ${_llvm_static_libs}")
message(STATUS "LLVM static ldflags: ${_llvm_static_ldflags}")

separate_arguments(_llvm_static_libs_list NATIVE_COMMAND "${_llvm_static_libs}")
separate_arguments(_llvm_static_ldflags_list NATIVE_COMMAND "${_llvm_static_ldflags}")

# Find static zlib and zstd to avoid runtime dependencies on target systems.
# This enables manylinux_2_28 compatibility.
cmake_path(GET LLVM_DIR PARENT_PATH _llvm_prefix)
cmake_path(GET _llvm_prefix PARENT_PATH _llvm_prefix)
cmake_path(GET _llvm_prefix PARENT_PATH _llvm_prefix)
find_library(ZLIB_STATIC NAMES libz.a HINTS "${_llvm_prefix}/lib" REQUIRED)
find_library(ZSTD_STATIC NAMES libzstd.a HINTS "${_llvm_prefix}/lib" "${_llvm_prefix}/lib64" REQUIRED)
message(STATUS "Static zlib: ${ZLIB_STATIC}")
message(STATUS "Static zstd: ${ZSTD_STATIC}")

target_link_libraries(
  tvm_ffi_orcjit
  PRIVATE tvm_ffi::header tvm_ffi::shared ${_llvm_static_ldflags_list}
          ${_llvm_static_libs_list} ${ZLIB_STATIC} ${ZSTD_STATIC}
)

# Statically link libstdc++ and libgcc to avoid GLIBCXX version requirements
# on the target system, enabling manylinux_2_28 support.
target_link_options(tvm_ffi_orcjit PRIVATE -static-libstdc++ -static-libgcc)

# ---- Find and bundle liborc_rt ----
# Allow overriding via env: CLANG_PATH
if(DEFINED ENV{CLANG_PATH})
  set(CLANG_EXE "$ENV{CLANG_PATH}")
else()
  find_program(
    CLANG_EXE
    NAMES clang-${LLVM_VERSION_MAJOR} clang
    HINTS ${LLVM_TOOLS_BINARY_DIR} "${LLVM_DIR}/../../../bin"
  )
endif()

if(NOT CLANG_EXE)
  message(FATAL_ERROR "clang not found. Required to locate liborc_rt library.")
endif()

execute_process(
  COMMAND ${CLANG_EXE} -print-runtime-dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE _clang_runtime_dir
  ERROR_QUIET
  RESULT_VARIABLE _clang_runtime_dir_rc
)

cmake_host_system_information(RESULT _host_arch QUERY OS_PLATFORM)
set(ORC_RT_PATH "")

if(_clang_runtime_dir_rc EQUAL 0 AND _clang_runtime_dir)
  # Try arch-specific name in the exact directory first
  if(EXISTS "${_clang_runtime_dir}/liborc_rt-${_host_arch}.a")
    set(ORC_RT_PATH "${_clang_runtime_dir}/liborc_rt-${_host_arch}.a")
  elseif(EXISTS "${_clang_runtime_dir}/liborc_rt.a")
    set(ORC_RT_PATH "${_clang_runtime_dir}/liborc_rt.a")
  else()
    # The reported runtime dir may not exist (e.g. clang reports
    # aarch64-unknown-linux-gnu/ but files live in linux/). Resolve
    # the parent and search recursively from there.
    cmake_path(GET _clang_runtime_dir PARENT_PATH _clang_runtime_parent)
    if(EXISTS "${_clang_runtime_parent}")
      file(GLOB_RECURSE _orc_rt_candidates "${_clang_runtime_parent}/liborc_rt*.a")
      if(_orc_rt_candidates)
        list(GET _orc_rt_candidates 0 ORC_RT_PATH)
      endif()
    endif()
  endif()
endif()

if(NOT ORC_RT_PATH)
  message(FATAL_ERROR "Could not find liborc_rt. "
    "Ensure clang runtime libraries are installed (e.g. libclang-rt-dev).")
endif()

message(STATUS "Found liborc_rt: ${ORC_RT_PATH}")
# Copy into build dir for editable installs (where _lib_dir resolves to CMAKE_BINARY_DIR)
file(COPY "${ORC_RT_PATH}" DESTINATION "${CMAKE_BINARY_DIR}")
install(FILES "${ORC_RT_PATH}" DESTINATION lib)

# Installation rules
install(
  TARGETS tvm_ffi_orcjit
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)
