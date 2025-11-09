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

copy %tvm_ffi%\lib\*.dll %torch_c_dlpack_ext%\torch_c_dlpack_ext
uv venv %tvm_ffi%\.venv\build --python %python_version%
%tvm_ffi%\.venv\build\Scripts\activate
uv pip install build wheel auditwheel
cd %torch_c_dlpack_ext%
python -m build -w
dir dist
python -m wheel tags dist/*.whl --python-tag=%python_version% --abi-tag=%python_version% --remove
dir dist
auditwheel repair --exclude libtorch.so --exclude libtorch_cpu.so --exclude libc10.so --exclude libtorch_python.so dist/*.whl -w wheelhouse
dir wheelhouse

exit /b 0

:get_torch_url
    set torch_version=%1
    if %torch_version%==2.4 (set %~2=https://download.pytorch.org/whl/cu124 & exit /b 0)
    if %torch_version%==2.5 (set %~2=https://download.pytorch.org/whl/cu124 & exit /b 0)
    if %torch_version%==2.6 (set %~2=https://download.pytorch.org/whl/cu126 & exit /b 0)
    if %torch_version%==2.7 (set %~2=https://download.pytorch.org/whl/cu128 & exit /b 0)
    if %torch_version%==2.8 (set %~2=https://download.pytorch.org/whl/cu129 & exit /b 0)
    if %torch_version%==2.9 (set %~2=https://download.pytorch.org/whl/cu129 & exit /b 0)
    echo Unknown or unsupported torch version: %torch_version% >&2
    set %~2=
    exit /b 1

:check_availability
    set torch_version=%1
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

:build_libs
    set torch_version=%1
    call :check_availability %torch_version%
    if %errorlevel%==0 (
        call :get_torch_url %torch_version% torch_url
        echo %arch% %python_version% %torch_version% %torch_url%
        mkdir %tvm_ffi%\.venv
        uv venv %tvm_ffi%\.venv\torch%torch_version% --python %python_version%
        %tvm_ffi%\.venv\torch%torch_version%\Scripts\activate
        uv pip install setuptools ninja
        uv pip install torch==%torch_version% --index-url %torch_url%
        uv pip install -v .
        mkdir %tvm_ffi%\lib
        python -m tvm_ffi.utils._build_optional_torch_c_dlpack --output-dir %tvm_ffi%\lib
        python -m tvm_ffi.utils._build_optional_torch_c_dlpack --output-dir %tvm_ffi%\lib --build-with-cuda
        dir %tvm_ffi%\lib
        deactivate
        rmdir -s -q %tvm_ffi%\.venv\torch%torch_version%
    ) else (
        echo Skipping build for torch %torch_version% on %arch% with python %python_version% as it is not available.
    )
    exit /b 0
