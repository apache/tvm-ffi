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
"""build backend for torch c dlpack ext."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from setuptools import build_meta as orig

_root = Path(__file__).parent.resolve()
_package_path = _root / "torch_c_dlpack_ext"


get_requires_for_build_sdist = orig.get_requires_for_build_sdist
get_requires_for_build_editable = orig.get_requires_for_build_editable
prepare_metadata_for_build_wheel = orig.prepare_metadata_for_build_wheel
prepare_metadata_for_build_editable = orig.prepare_metadata_for_build_editable
build_sdist = orig.build_sdist
build_editable = orig.build_editable


def _is_lib_prebuilt() -> bool:
    if sys.platform.startswith("win32"):
        extension = "dll"
    else:
        extension = "so"
    return next(_package_path.rglob(f"*.{extension}"), None) is not None


def get_requires_for_build_wheel(
    config_settings: orig._ConfigSettings = None,
) -> list[str]:
    """Get build requirements for wheel, conditionally including torch and apache-tvm-ffi."""
    requires = orig.get_requires_for_build_wheel(config_settings)
    if not _is_lib_prebuilt():
        # build wheel from sdist package, requires apache-tvm-ffi>=0.1.1 to build lib
        requires.append("apache-tvm-ffi>=0.1.1")
    return requires


def build_wheel(
    wheel_directory: orig.StrPath,
    config_settings: orig._ConfigSettings = None,
    metadata_directory: orig.StrPath | None = None,
) -> str:
    """Build wheel."""
    if not _is_lib_prebuilt():
        # build wheel from sdist package, compile the torch c dlpack ext library locally.
        import torch  # noqa: PLC0415

        if hasattr(torch.Tensor, "__c_dlpack_exchange_api__"):
            print(
                "torch.Tensor already has attribute __c_dlpack_exchange_api__. "
                "No need to build any torch c dlpackc libs."
            )
        else:
            extra_args = []
            # First use "torch.cuda.is_available()" to check whether GPU environment
            # is available. Then determine the GPU type.
            if torch.cuda.is_available():
                if torch.version.cuda is not None:
                    extra_args.append("--build-with-cuda")
                elif torch.version.hip is not None:
                    extra_args.append("--build-with-rocm")
                else:
                    raise ValueError("Cannot determine whether to build with CUDA or ROCm.")
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tvm_ffi.utils._build_optional_torch_c_dlpack",
                    "--output-dir",
                    str(_package_path),
                    *extra_args,
                ],
                check=True,
                env={**os.environ, "TVM_FFI_DISABLE_TORCH_C_DLPACK": "1"},
            )
    return orig.build_wheel(wheel_directory, config_settings, metadata_directory)
