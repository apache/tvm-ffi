@echo on

set arch=%~1
set python_version=%~2

echo arch=%arch%
echo python_version=%python_version%

set tvm_ffi=%cd%
set torch_c_dlpack_ext=%tvm_ffi%\addons\torch_c_dlpack_ext

for %%P in (2.4 2.5 2.6 2.7 2.8 2.9) do (
    call :build_libs %%P
)

@REM copy %tvm_ffi%\lib\*.dll %torch_c_dlpack_ext%\torch_c_dlpack_ext
@REM uv venv %tvm_ffi%\.venv\build --python %python_version%
@REM %tvm_ffi%\.venv\build\Scripts\activate
@REM uv pip install build wheel auditwheel
@REM cd %torch_c_dlpack_ext%
@REM python -m build -w
@REM dir dist
@REM python -m wheel tags dist/*.whl --python-tag=%python_version% --abi-tag=%python_version% --remove
@REM dir dist
@REM auditwheel repair --exclude libtorch.so --exclude libtorch_cpu.so --exclude libc10.so --exclude libtorch_python.so dist/*.whl -w wheelhouse
@REM dir wheelhouse

exit /b 0

:build_libs
    set torch_version=%1
    setlocal enabledelayedexpansion
    call :check_availability
    if %errorlevel%==0 (
        call :get_torch_url
        echo fff %arch% %python_version% %torch_version% !torch_url!
        @REM mkdir %tvm_ffi%\.venv
        @REM uv venv %tvm_ffi%\.venv\torch%torch_version% --python %python_version%
        @REM %tvm_ffi%\.venv\torch%torch_version%\Scripts\activate
        @REM uv pip install setuptools ninja
        @REM uv pip install torch==%torch_version% --index-url %torch_url%
        @REM uv pip install -v .
        @REM mkdir %tvm_ffi%\lib
        @REM python -m tvm_ffi.utils._build_optional_torch_c_dlpack --output-dir %tvm_ffi%\lib
        @REM python -m tvm_ffi.utils._build_optional_torch_c_dlpack --output-dir %tvm_ffi%\lib --build-with-cuda
        @REM dir %tvm_ffi%\lib
        @REM deactivate
        @REM rmdir -s -q %tvm_ffi%\.venv\torch%torch_version%
    ) else (
        echo Skipping build for torch %torch_version% on %arch% with python %python_version% as it is not available.
    )
    endlocal
    exit /b 0


:check_availability
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
    if %torch_version%==2.4 (set torch_url=https://download.pytorch.org/whl/cu124 && exit /b 0)
    if %torch_version%==2.5 (set torch_url=https://download.pytorch.org/whl/cu124 && exit /b 0)
    if %torch_version%==2.6 (set torch_url=https://download.pytorch.org/whl/cu126 && exit /b 0)
    if %torch_version%==2.7 (set torch_url=https://download.pytorch.org/whl/cu128 && exit /b 0)
    if %torch_version%==2.8 (set torch_url=https://download.pytorch.org/whl/cu129 && exit /b 0)
    if %torch_version%==2.9 (set torch_url=https://download.pytorch.org/whl/cu129 && exit /b 0)
    echo Unknown or unsupported torch version: %torch_version% >&2
    exit /b 1
