@echo on

set arch=%~1
set python_version=%~2

echo arch=%arch%
echo python_version=%python_version%

set tvm_ffi=%cd%
set torch_c_dlpack_ext=%tvm_ffi%/addons/torch_c_dlpack_ext

echo tvm_ffi=%tvm_ffi%
echo torch_c_dlpack_ext=%torch_c_dlpack_ext%

for %%P in ("2.4" "2.5" "2.6" "2.7" "2.8" "2.9") do (
    call :build_libs %%~P
)

:get_torch_url
    setlocal
    set version=%~1
    if "%version%"=="2.4" (
        set url="https://download.pytorch.org/whl/cu124"
    ) else if "%version%"=="2.5" (
        set url="https://download.pytorch.org/whl/cu124"
    ) else if "%version%"=="2.6" (
        set url="https://download.pytorch.org/whl/cu126"
    ) else if "%version%"=="2.7" (
        set url="https://download.pytorch.org/whl/cu128"
    ) else if "%version%"=="2.8" (
        set url="https://download.pytorch.org/whl/cu129"
    ) else if "%version%"=="2.9" (
        set url="https://download.pytorch.org/whl/cu129"
    ) else (
        echo  "Unknown or unsupported torch version: %version%" >&2
        exit /b 1
    )
    endlocal & set "%~2=%url%"
    exit /b 0

:check_availability
    setlocal
    set torch_version=%1
    echo %torch_version%
    set return=0
    if "%torch_version%"=="2.4" (
        if "%python_version%"=="cp313" set return=1
        if "%python_version%"=="cp314" set return=1
    ) else if "%version%"=="2.5" (
        if "%python_version%"=="cp314" set return=1
    ) else if "%version%"=="2.6" (
        if "%python_version%"=="cp314" set return=1
    ) else if "%version%"=="2.7" (
        if "%python_version%"=="cp314" set return=1
    ) else if "%version%"=="2.8" (
        if "%python_version%"=="cp314" set return=1
    ) else if "%version%"=="2.9" (
        if "%python_version%"=="cp39" set return=1
    ) else (
        echo Unknown or unsupported torch version: %version% >&2
        set return=1
    )
    endlocal
    exit /b %return%

:build_libs
    setlocal
    set torch_version=%1
    echo %torch_version%
    call :check_availability %torch_version%
    if %errorlevel%==0 (
        call :get_torch_url %torch_version% torch_url
        echo %arch% %python_version% %torch_version% %torch_url%
    ) else (
        echo Skipping build for torch %torch_version% on %arch% with python %python_version% as it is not available.
    )
    endlocal
    exit /b 0
