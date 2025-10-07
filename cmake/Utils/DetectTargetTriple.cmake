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
# detect_target_triple(out_var)
# Determine the target machine triple and store it in the variable named by
# `out_var`.
#
# The result is determined by (in order of preference):
# - querying CMake's own configuration,
# - asking the compiler directly via `-dumpmachine` or `--print-target-triple`,
# - using CMake's `CMAKE_LIBRARY_ARCHITECTURE` variable (Debian/Ubuntu multiarch hint),
# - synthesizing from system information.
# ~~~
# machine_triple.cmake
function (detect_target_triple out_var)
  # cmake-lint: disable=R0911,R0912,R0915
  # --- 1) Prefer CMake's own notion (e.g. when --target was used) ---
  foreach (lang C CXX)
    if (CMAKE_${lang}_COMPILER_TARGET)
      set(${out_var}
          "${CMAKE_${lang}_COMPILER_TARGET}"
          PARENT_SCOPE
      )
      return()
    endif ()
  endforeach ()

  # --- 2) Ask the compiler directly (works for Clang/GCC, Android NDK, Emscripten) ---
  set(cc "${CMAKE_C_COMPILER}")
  if (NOT cc AND CMAKE_CXX_COMPILER)
    set(cc "${CMAKE_CXX_COMPILER}")
  endif ()
  if (cc)
    execute_process(
      COMMAND "${cc}" -dumpmachine
      OUTPUT_VARIABLE ret
      OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
    )
    if (NOT ret)
      execute_process(
        COMMAND "${cc}" --print-target-triple
        OUTPUT_VARIABLE ret
        OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
      )
    endif ()
    if (ret)
      set(${out_var}
          "${ret}"
          PARENT_SCOPE
      )
      return()
    endif ()
  endif ()

  # --- 3) Platform-specific construction ---

  # 3a) Emscripten (toolchains usually set CMAKE_SYSTEM_NAME to Emscripten)
  if (CMAKE_SYSTEM_NAME STREQUAL "Emscripten" OR EMSCRIPTEN)
    set(${out_var}
        "wasm32-unknown-emscripten"
        PARENT_SCOPE
    )
    return()
  endif ()

  # 3b) Android (derive from ANDROID_ABI / ANDROID_PLATFORM)
  if (ANDROID OR CMAKE_SYSTEM_NAME STREQUAL "Android")
    set(arch "")
    set(abi "${ANDROID_ABI}")
    if (abi STREQUAL "armeabi-v7a")
      set(arch "armv7a")
      set(base "linux-androideabi")
    elseif (abi STREQUAL "arm64-v8a")
      set(arch "aarch64")
      set(base "linux-android")
    elseif (abi STREQUAL "x86")
      set(arch "i686")
      set(base "linux-android")
    elseif (abi STREQUAL "x86_64")
      set(arch "x86_64")
      set(base "linux-android")
    elseif (abi STREQUAL "riscv64")
      set(arch "riscv64")
      set(base "linux-android")
    else ()
      # Fallback from processor if ABI isn't set
      set(arch "${CMAKE_SYSTEM_PROCESSOR}")
      string(TOLOWER "${arch}" arch)
      if (arch MATCHES "armv7")
        set(arch "armv7a")
        set(base "linux-androideabi")
      elseif (arch MATCHES "aarch64|arm64")
        set(arch "aarch64")
        set(base "linux-android")
      elseif (arch MATCHES "x86_64|amd64")
        set(arch "x86_64")
        set(base "linux-android")
      elseif (arch MATCHES "i[3-6]86|x86")
        set(arch "i686")
        set(base "linux-android")
      elseif (arch MATCHES "riscv64")
        set(arch "riscv64")
        set(base "linux-android")
      endif ()
    endif ()

    # Append API level if we can (e.g. aarch64-linux-android21)
    set(api "")
    if (DEFINED ANDROID_PLATFORM AND NOT "${ANDROID_PLATFORM}" STREQUAL "")
      string(REGEX REPLACE "android-?" "" api "${ANDROID_PLATFORM}")
    endif ()

    if (arch STREQUAL "armv7a")
      set(ret "${arch}-${base}")
    else ()
      set(ret "${arch}-${base}")
    endif ()
    if (api)
      set(ret "${ret}${api}")
    endif ()
    set(${out_var}
        "${ret}"
        PARENT_SCOPE
    )
    return()
  endif ()

  # 3c) Apple iOS (device & simulator). Works for Xcode + toolchains.
  if (APPLE AND (CMAKE_SYSTEM_NAME STREQUAL "iOS" OR CMAKE_OSX_SYSROOT MATCHES "[iI]phone"))
    # Choose first arch if multi-arch is set
    set(archs "${CMAKE_OSX_ARCHITECTURES}")
    if (NOT archs)
      set(archs "${CMAKE_SYSTEM_PROCESSOR}")
    endif ()
    list(GET archs 0 arch)
    string(TOLOWER "${arch}" _arch_l)
    if (_arch_l MATCHES "aarch64|arm64|arm64e") # iOS uses 'arm64' in triples
      set(arch "arm64")
    elseif (_arch_l MATCHES "x86_64|amd64")
      set(arch "x86_64")
    endif ()

    # Simulator?
    set(is_sim OFF)
    if (CMAKE_OSX_SYSROOT MATCHES "simulator" OR CMAKE_XCODE_EFFECTIVE_PLATFORMS MATCHES
                                                 "simulator"
    )
      set(is_sim ON)
    endif ()

    # Deployment target (best-effort)
    set(ios_ver "")
    foreach (maybe_ver CMAKE_OSX_DEPLOYMENT_TARGET CMAKE_IOS_DEPLOYMENT_TARGET
                       IOS_DEPLOYMENT_TARGET
    )
      if (DEFINED ${maybe_ver} AND NOT "${${maybe_ver}}" STREQUAL "")
        set(ios_ver "${${maybe_ver}}")
        break()
      endif ()
    endforeach ()

    if (is_sim)
      set(ret "${arch}-apple-ios${ios_ver}-simulator")
    else ()
      set(ret "${arch}-apple-ios${ios_ver}")
    endif ()
    string(REGEX REPLACE "ios$" "ios" ret "${ret}") # normalize empty version case
    set(${out_var}
        "${ret}"
        PARENT_SCOPE
    )
    return()
  endif ()

  # 3d) Windows + MSVC (cl.exe / clang-cl in MSVC mode)
  if (MSVC AND CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(plat "${CMAKE_GENERATOR_PLATFORM}")
    if (NOT plat AND DEFINED CMAKE_VS_PLATFORM_NAME)
      set(plat "${CMAKE_VS_PLATFORM_NAME}")
    endif ()
    if (plat STREQUAL "Win32")
      set(arch "i686")
    elseif (plat MATCHES "^(x64|X64)$")
      set(arch "x86_64")
    elseif (plat MATCHES "ARM64")
      set(arch "arm64")
    else ()
      # Fallback from pointer size
      if (CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(arch "x86_64")
      else ()
        set(arch "i686")
      endif ()
    endif ()
    set(${out_var}
        "${arch}-pc-windows-msvc"
        PARENT_SCOPE
    )
    return()
  endif ()

  # 3e) MinGW (handy if you ever hit it)
  if (MINGW)
    if (CMAKE_SIZEOF_VOID_P EQUAL 8)
      set(${out_var}
          "x86_64-w64-mingw32"
          PARENT_SCOPE
      )
    else ()
      set(${out_var}
          "i686-w64-mingw32"
          PARENT_SCOPE
      )
      return()
    endif ()
  endif ()

  # --- 4) Debian/Ubuntu multiarch hint provided by CMake ---
  if (CMAKE_LIBRARY_ARCHITECTURE)
    set(${out_var}
        "${CMAKE_LIBRARY_ARCHITECTURE}"
        PARENT_SCOPE
    )
    return()
  endif ()

  # --- 5) Canonical, sensible fallback by OS ---
  set(arch "${CMAKE_SYSTEM_PROCESSOR}")
  if (NOT arch)
    if (CMAKE_SIZEOF_VOID_P EQUAL 8)
      set(arch "x86_64")
    else ()
      set(arch "i686")
    endif ()
  endif ()
  string(TOLOWER "${arch}" _arch_l)

  # Normalize common arch spellings
  if (_arch_l MATCHES "aarch64|arm64|arm64e")
    if (APPLE)
      set(arch "arm64") # Apple uses arm64
    else ()
      set(arch "aarch64")
    endif ()
  elseif (_arch_l MATCHES "x86_64|amd64")
    set(arch "x86_64")
  elseif (_arch_l MATCHES "i[3-6]86|x86")
    set(arch "i686")
  elseif (_arch_l MATCHES "armv7")
    set(arch "armv7")
  elseif (_arch_l MATCHES "riscv64")
    set(arch "riscv64")
  endif ()

  if (APPLE)
    # macOS (Darwin) fallback
    set(${out_var}
        "${arch}-apple-darwin"
        PARENT_SCOPE
    )
    return()
  elseif (CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(${out_var}
        "${arch}-pc-windows-msvc"
        PARENT_SCOPE
    )
    return()
  elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # Default to glibc; adjust to 'musl' in your toolchain if needed
    set(${out_var}
        "${arch}-unknown-linux-gnu"
        PARENT_SCOPE
    )
    return()
  elseif (CMAKE_SYSTEM_NAME MATCHES "FreeBSD")
    set(${out_var}
        "${arch}-unknown-freebsd"
        PARENT_SCOPE
    )
    return()
  endif ()

  # Last-ditch (keeps your old behavior minimally sane)
  set(${out_var}
      "${arch}-${CMAKE_SYSTEM_NAME}"
      PARENT_SCOPE
  )
endfunction ()
