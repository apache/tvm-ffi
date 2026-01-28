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

# LLVM components needed for ORC JIT v2
llvm_map_components_to_libnames(LLVM_LIBS Core OrcJIT Support native)

# Filter out non-existent targets
set(LLVM_LIBS_FILTERED)
foreach (lib ${LLVM_LIBS})
  if (TARGET ${lib})
    list(APPEND LLVM_LIBS_FILTERED ${lib})
  else ()
    message(STATUS "Skipping non-existent LLVM target: ${lib}")
  endif ()
endforeach ()
set(LLVM_LIBS ${LLVM_LIBS_FILTERED})

# Source files
set(SOURCES src/ffi/orcjit_session.cc src/ffi/orcjit_dylib.cc)

# Build shared library
add_library(tvm_ffi_orcjit SHARED ${SOURCES})

target_include_directories(
  tvm_ffi_orcjit PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${LLVM_INCLUDE_DIRS}
)

target_link_libraries(tvm_ffi_orcjit PRIVATE tvm_ffi::header tvm_ffi::shared LLVM)

# Compile definitions
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
target_compile_definitions(tvm_ffi_orcjit PRIVATE ${LLVM_DEFINITIONS_LIST})

# Installation rules
install(
  TARGETS tvm_ffi_orcjit
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)
