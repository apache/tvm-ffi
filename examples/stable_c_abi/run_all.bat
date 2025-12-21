@echo off
rem Licensed to the Apache Software Foundation (ASF) under one
rem or more contributor license agreements.  See the NOTICE file
rem distributed with this work for additional information
rem regarding copyright ownership.  The ASF licenses this file
rem to you under the Apache License, Version 2.0 (the
rem "License"); you may not use this file except in compliance
rem with the License.  You may obtain a copy of the License at
rem
rem   http://www.apache.org/licenses/LICENSE-2.0
rem
rem Unless required by applicable law or agreed to in writing,
rem software distributed under the License is distributed on an
rem "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
rem KIND, either express or implied.  See the License for the
rem specific language governing permissions and limitations
rem under the License.

setlocal enabledelayedexpansion
cd /d "%~dp0"

if exist build rmdir /s /q build

rem To compile `src/add_one_cpu.c` to shared library `build/add_one_cpu.so`
cmake . -B build -DEXAMPLE_NAME="kernel" -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --config RelWithDebInfo

rem To compile `src/load.c` to executable `build/load`
for /f "delims=" %%i in ('python -c "from tvm_ffi import libinfo; import pathlib; print(pathlib.Path(libinfo.find_libtvm_ffi()).parent)"') do set "TVM_FFI_LIBDIR=%%i"
set "PATH=%TVM_FFI_LIBDIR%;%PATH%"
cmake . -B build -DEXAMPLE_NAME="load" -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --config RelWithDebInfo
build\load.exe

endlocal
