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
"""NVRTC (NVIDIA Runtime Compilation) utilities for compiling CUDA source to CUBIN."""

from __future__ import annotations

from typing import Sequence


def nvrtc_compile(  # noqa: PLR0912, PLR0915
    source: str,
    *,
    name: str = "kernel.cu",
    arch: str | None = None,
    extra_opts: Sequence[str] | None = None,
) -> bytes:
    """Compile CUDA source code to CUBIN using NVRTC.

    This function uses the NVIDIA Runtime Compilation (NVRTC) library to compile
    CUDA C++ source code into a CUBIN binary that can be loaded and executed
    using the CUDA Driver API.

    Parameters
    ----------
    source : str
        The CUDA C++ source code to compile.

    name : str, optional
        The name to use for the source file (for error messages). Default: "kernel.cu"

    arch : str, optional
        The target GPU architecture (e.g., "sm_75", "sm_80", "sm_89"). If not specified,
        attempts to auto-detect from the current GPU.

    extra_opts : Sequence[str], optional
        Additional compilation options to pass to NVRTC (e.g., ["-I/path/to/include", "-DDEFINE=1"]).

    Returns
    -------
    bytes
        The compiled CUBIN binary data.

    Raises
    ------
    RuntimeError
        If NVRTC compilation fails or CUDA bindings are not available.

    Example
    -------
    .. code-block:: python

        from tvm_ffi.cpp import nvrtc

        cuda_source = '''
        extern "C" __global__ void add_one(float* x, float* y, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                y[idx] = x[idx] + 1.0f;
            }
        }
        '''

        cubin_bytes = nvrtc.nvrtc_compile(cuda_source)
        # Use cubin_bytes with tvm_ffi.cpp.load_inline and embed_cubin parameter

    """
    try:
        from cuda.bindings import driver, nvrtc  # type: ignore[import-not-found]  # noqa: PLC0415
    except ImportError as e:
        raise RuntimeError(
            "CUDA bindings not available. Install with: pip install cuda-python"
        ) from e

    # Auto-detect architecture if not specified
    if arch is None:
        try:
            # Initialize CUDA driver API
            (result,) = driver.cuInit(0)
            if result != driver.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"Failed to initialize CUDA driver: {result}")

            # Get current device
            result, device = driver.cuCtxGetDevice()
            if result != driver.CUresult.CUDA_SUCCESS:
                # Try to get device 0 if no context exists
                device = 0

            # Get compute capability
            result, major = driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device
            )
            if result != driver.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"Failed to get compute capability major: {result}")

            result, minor = driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device
            )
            if result != driver.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"Failed to get compute capability minor: {result}")

            arch = f"sm_{major}{minor}"
        except Exception as e:
            # Fallback to a reasonable default
            raise RuntimeError(
                f"Failed to auto-detect GPU architecture: {e}. "
                "Please specify 'arch' parameter explicitly."
            ) from e

    # Create program
    result, prog = nvrtc.nvrtcCreateProgram(str.encode(source), str.encode(name), 0, None, None)
    if result != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(f"Failed to create NVRTC program: {result}")

    # Compile options
    opts = [
        b"--gpu-architecture=" + arch.encode(),
        b"-default-device",
    ]

    # Add extra options if provided
    if extra_opts:
        opts.extend([opt.encode() if isinstance(opt, str) else opt for opt in extra_opts])

    # Compile
    (result,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    if result != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        # Get compilation log
        result_log, log_size = nvrtc.nvrtcGetProgramLogSize(prog)
        if result_log == nvrtc.nvrtcResult.NVRTC_SUCCESS and log_size > 0:
            log_buf = b" " * log_size
            (result_log,) = nvrtc.nvrtcGetProgramLog(prog, log_buf)
            if result_log == nvrtc.nvrtcResult.NVRTC_SUCCESS:
                error_msg = f"NVRTC compilation failed:\n{log_buf.decode('utf-8')}"
            else:
                error_msg = f"NVRTC compilation failed (couldn't get log): {result}"
        else:
            error_msg = f"NVRTC compilation failed: {result}"

        nvrtc.nvrtcDestroyProgram(prog)
        raise RuntimeError(error_msg)

    # Get CUBIN
    result, cubin_size = nvrtc.nvrtcGetCUBINSize(prog)
    if result != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        nvrtc.nvrtcDestroyProgram(prog)
        raise RuntimeError(f"Failed to get CUBIN size from NVRTC: {result}")

    cubin_buf = b" " * cubin_size
    (result,) = nvrtc.nvrtcGetCUBIN(prog, cubin_buf)
    if result != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        nvrtc.nvrtcDestroyProgram(prog)
        raise RuntimeError(f"Failed to get CUBIN from NVRTC: {result}")

    # Clean up
    nvrtc.nvrtcDestroyProgram(prog)

    return cubin_buf
