@REM Licensed to the Apache Software Foundation (ASF) under one
@REM or more contributor license agreements.  See the NOTICE file
@REM distributed with this work for additional information
@REM regarding copyright ownership.  The ASF licenses this file
@REM to you under the Apache License, Version 2.0 (the
@REM "License"); you may not use this file except in compliance
@REM with the License.  You may obtain a copy of the License at
@REM
@REM   http://www.apache.org/licenses/LICENSE-2.0
@REM
@REM Unless required by applicable law or agreed to in writing,
@REM software distributed under the License is distributed on an
@REM "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
@REM KIND, either express or implied.  See the License for the
@REM specific language governing permissions and limitations
@REM under the License.
@echo off
setlocal enabledelayedexpansion

set arch=%~1
set python_version=%~2

set tvm_ffi=%cd%
set torch_c_dlpack_ext=%tvm_ffi%\addons\torch_c_dlpack_ext
set wheel_platform=
set uv_python=%python_version%
if /I "%arch%"=="x86_64" set wheel_platform=win_amd64
if /I "%arch%"=="arm64" (
    set wheel_platform=win_arm64
    set uv_python=3.%python_version:~3%-arm64
)
if not defined wheel_platform (
    echo Unsupported architecture: %arch% >&2
    exit /b 1
)

if not exist "%tvm_ffi%\.venv" mkdir "%tvm_ffi%\.venv"
if not exist "%tvm_ffi%\lib" mkdir "%tvm_ffi%\lib"
for %%P in (2.4 2.5 2.6 2.7 2.8 2.9) do (
    call :build_libs %%P
    if errorlevel 1 exit /b 1
)

copy %tvm_ffi%\lib\*.dll %torch_c_dlpack_ext%\torch_c_dlpack_ext
if errorlevel 1 exit /b 1
uv venv %tvm_ffi%\.venv\build --python %uv_python%
if errorlevel 1 exit /b 1
call %tvm_ffi%\.venv\build\Scripts\activate
if errorlevel 1 exit /b 1
uv pip install build wheel
if errorlevel 1 exit /b 1
cd %torch_c_dlpack_ext%
if errorlevel 1 exit /b 1
python -m build -w
if errorlevel 1 exit /b 1
dir dist
for %%f in (dist\*.whl) do (
    python -m wheel tags "%%f" --python-tag=%python_version% ^
        --abi-tag=%python_version% --platform-tag=%wheel_platform%
    if errorlevel 1 exit /b 1
)
dir dist
if not exist wheelhouse mkdir wheelhouse
copy dist\*-%wheel_platform%.whl wheelhouse
if errorlevel 1 exit /b 1
dir wheelhouse
endlocal
exit /b 0

:build_libs
    set torch_version=%1
    call :check_availability
    if %errorlevel%==0 (
        call :get_torch_url
        if errorlevel 1 exit /b 1
        uv venv %tvm_ffi%\.venv\torch%torch_version% --python %uv_python%
        if errorlevel 1 exit /b 1
        call %tvm_ffi%\.venv\torch%torch_version%\Scripts\activate
        if errorlevel 1 exit /b 1
        uv pip install setuptools ninja
        if errorlevel 1 exit /b 1
        uv pip install torch==%torch_version% --index !torch_url! ^
            --default-index https://pypi.org/simple --index-strategy unsafe-best-match
        if errorlevel 1 exit /b 1
        uv pip install -v .
        if errorlevel 1 exit /b 1
        python -m tvm_ffi.utils._build_optional_torch_c_dlpack --output-dir %tvm_ffi%\lib
        if errorlevel 1 exit /b 1
        python -m tvm_ffi.utils._build_optional_torch_c_dlpack ^
            --output-dir %tvm_ffi%\lib --build-with-cuda
        if errorlevel 1 exit /b 1
        call deactivate
        if errorlevel 1 exit /b 1
        rmdir /s /q %tvm_ffi%\.venv\torch%torch_version%
        if errorlevel 1 exit /b 1
    ) else (
        echo Skipping build for torch %torch_version% on %arch% with python %python_version% as it is not available.
    )
    exit /b 0


:check_availability
    if /I "%arch%"=="arm64" (
        if %torch_version%==2.7 if %python_version%==cp312 exit /b 0
        if %torch_version%==2.8 if %python_version%==cp311 exit /b 0
        if %torch_version%==2.8 if %python_version%==cp312 exit /b 0
        if %torch_version%==2.8 if %python_version%==cp313 exit /b 0
        if %torch_version%==2.9 if %python_version%==cp311 exit /b 0
        if %torch_version%==2.9 if %python_version%==cp312 exit /b 0
        if %torch_version%==2.9 if %python_version%==cp313 exit /b 0
        if %torch_version%==2.9 if %python_version%==cp314 exit /b 0
        exit /b 1
    )
    if %torch_version%==2.4 (
        if %python_version%==cp313 exit /b 1
        if %python_version%==cp314 exit /b 1
        exit /b 0
    )
    if %torch_version%==2.5 (
        if %python_version%==cp314 exit /b 1
        exit /b 0
    )
    if %torch_version%==2.6 (
        if %python_version%==cp314 exit /b 1
        exit /b 0
    )
    if %torch_version%==2.7 (
        if %python_version%==cp314 exit /b 1
        exit /b 0
    )
    if %torch_version%==2.8 (
        if %python_version%==cp314 exit /b 1
        exit /b 0
    )
    if %torch_version%==2.9 (
        if %python_version%==cp39 exit /b 1
        exit /b 0
    )
    echo Unknown or unsupported torch version: %torch_version% >&2
    exit /b 1

:get_torch_url
    set cuda_version=
    if %torch_version%==2.4 set cuda_version=cu124
    if %torch_version%==2.5 set cuda_version=cu124
    if %torch_version%==2.6 set cuda_version=cu126
    if %torch_version%==2.7 set cuda_version=cu128
    if %torch_version%==2.8 set cuda_version=cu129
    if %torch_version%==2.9 set cuda_version=cu129
    if defined cuda_version (
        set torch_url=https://download.pytorch.org/whl/%cuda_version%
        exit /b 0
    )
    echo Unknown or unsupported torch version: %torch_version% >&2
    exit /b 1
