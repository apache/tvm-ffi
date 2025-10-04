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

include(ExternalProject)

# ~~~
# _libbacktrace_compile()
# Build and install libbacktrace as an ExternalProject, then add an imported
# static target `libbacktrace` exposing headers and archive for consumers.
# ~~~
function (_libbacktrace_compile)
  set(libbacktrace_source ${CMAKE_CURRENT_LIST_DIR}/../../3rdparty/libbacktrace)
  set(libbacktrace_prefix ${CMAKE_CURRENT_BINARY_DIR}/libbacktrace)
  if (CMAKE_SYSTEM_NAME MATCHES "Darwin" AND (CMAKE_C_COMPILER MATCHES "^/Library"
                                              OR CMAKE_C_COMPILER MATCHES "^/Applications")
  )
    set(cmake_c_compiler "/usr/bin/cc")
  else ()
    set(cmake_c_compiler "${CMAKE_C_COMPILER}")
  endif ()
  message(STATUS CMAKC_C_COMPILER="${CMAKE_C_COMPILER}")

  file(MAKE_DIRECTORY ${libbacktrace_prefix}/include)
  file(MAKE_DIRECTORY ${libbacktrace_prefix}/lib)

  ExternalProject_Add(
    project_libbacktrace
    PREFIX libbacktrace
    SOURCE_DIR ${libbacktrace_source}
    BINARY_DIR ${libbacktrace_prefix}
    CONFIGURE_COMMAND
      "sh" "${libbacktrace_source}/configure" "--prefix=${libbacktrace_prefix}" --with-pic
      "CC=${cmake_c_compiler}" "CPP=${cmake_c_compiler} -E" "CFLAGS=${CMAKE_C_FLAGS}"
      "LDFLAGS=${CMAKE_EXE_LINKER_FLAGS}" "NM=${CMAKE_NM}" "STRIP=${CMAKE_STRIP}"
      "--host=${MACHINE_NAME}"
    INSTALL_DIR ${libbacktrace_prefix}
    BUILD_COMMAND make
    INSTALL_COMMAND make install
    BUILD_BYPRODUCTS "${libbacktrace_prefix}/lib/libbacktrace.a"
                     "${libbacktrace_prefix}/include/backtrace.h"
  )
  ExternalProject_Add_Step(
    project_libbacktrace checkout
    DEPENDERS configure
    DEPENDEES download
  )
  set_target_properties(project_libbacktrace PROPERTIES EXCLUDE_FROM_ALL TRUE)
  add_library(libbacktrace STATIC IMPORTED)
  add_dependencies(libbacktrace project_libbacktrace)
  set_target_properties(
    libbacktrace
    PROPERTIES IMPORTED_LOCATION ${libbacktrace_prefix}/lib/libbacktrace.a
               INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_LIST_DIR}/../../3rdparty/libbacktrace/
  )
endfunction ()

if (NOT MSVC)
  _libbacktrace_compile()
endif ()
