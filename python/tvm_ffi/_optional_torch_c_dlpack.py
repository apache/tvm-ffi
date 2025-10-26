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
"""Optional module to support faster DLPack conversion.

This is an optional module to support faster DLPack conversion for torch.
Some of the changes are merged but not yet released, so it is used
as a stop gap to support faster DLPack conversion.

This file contains source code from PyTorch:
License: licenses/LICENSE.pytorch.txt

This module only serves as temp measure and will
likely be phased away and deleted after changes landed and released in pytorch.

This module will load slowly at first time due to JITing,
subsequent calls will be much faster.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

from . import libinfo


def load_torch_c_dlpack_extension() -> Any:
    try:
        import torch  # noqa: PLC0415

        if hasattr(torch.Tensor, "__c_dlpack_exchange_api__"):
            # skip loading the extension if the __c_dlpack_exchange_api__
            # attribute is already set so we don't have to do it in
            # newer version of PyTorch
            return None
    except ImportError:
        return None

    try:
        import torch_c_dlpack_ext  # type: ignore # noqa: PLC0415

        setattr(
            torch.Tensor,
            "__c_dlpack_exchange_api__",
            torch_c_dlpack_ext.TorchDLPackExchangeAPIPtr(),
        )
        return torch_c_dlpack_ext
    except ImportError:
        pass

    """Load the torch c dlpack extension."""
    cpp_source = (
        Path(__file__).parent.parent.parent
        / "addons"
        / "torch_c_dlpack_ext"
        / "torch_c_dlpack_ext.cc"
    )
    try:
        # optionally import torch
        import torch  # noqa: PLC0415
        from torch.utils import cpp_extension  # noqa: PLC0415

        include_paths = libinfo.include_paths()
        extra_cflags = ["-O3"]

        if torch.cuda.is_available():
            include_paths += cpp_extension.include_paths("cuda")
            extra_cflags += ["-DBUILD_WITH_CUDA"]
        mod = cpp_extension.load(
            name="torch_c_dlpack_ext",
            sources=[str(cpp_source)],
            extra_cflags=extra_cflags,
            extra_include_paths=include_paths,
        )
        # Set the DLPackExchangeAPI pointer on the class
        setattr(torch.Tensor, "__c_dlpack_exchange_api__", mod.TorchDLPackExchangeAPIPtr())
        return mod
    except ImportError:
        pass
    except Exception as e:
        warnings.warn(
            f"Failed to load torch c dlpack extension: {e},EnvTensorAllocator will not be enabled."
        )
    return None


# keep alive
_mod = load_torch_c_dlpack_extension()


def patch_torch_cuda_stream_protocol() -> Any:
    """Load the torch cuda stream protocol for older versions of torch."""
    try:
        import torch  # noqa: PLC0415

        if not torch.cuda.is_available():
            return
        if not hasattr(torch.cuda.Stream, "__cuda_stream__"):

            def __torch_cuda_stream__(self: torch.cuda.Stream) -> tuple[int, torch.cuda.Stream]:
                """Return the version number and the cuda stream."""
                return (0, self.cuda_stream)

            setattr(torch.cuda.Stream, "__cuda_stream__", __torch_cuda_stream__)
    except ImportError:
        pass


patch_torch_cuda_stream_protocol()
