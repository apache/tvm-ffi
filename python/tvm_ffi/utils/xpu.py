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

"""Helpers for Intel XPU (oneAPI / SYCL) devices.

For ``kDLOneAPI``, DLPack defines ``device_id`` as the index of the device in
``sycl::device::get_devices()``. This global index may differ from the torch
``xpu:<n>`` index, since torch only enumerates Level Zero GPUs (and, since 2.7, hides
integrated GPUs when a discrete GPU is present). The mapping is therefore always
obtained from torch itself.
"""

from __future__ import annotations

import functools
from typing import Any

from .. import core
from .._tensor import device


def _torch_dlpack_device(tensor: Any) -> tuple[int, int]:
    """Return the DLPack device produced by torch's C++ exporter for ``tensor``.

    ``torch.Tensor.__dlpack_device__`` reports the torch index for XPU, while the C++
    exporter (``torch.utils.dlpack.to_dlpack``) reports the global SYCL index. Tensors
    reach TVM FFI through the latter, so it is the source of truth.
    """
    import torch.utils.dlpack  # noqa: PLC0415

    dev = core.from_dlpack(torch.utils.dlpack.to_dlpack(tensor)).device
    return dev.dlpack_device_type(), dev.index


@functools.cache
def xpu_global_device_index(torch_index: int) -> int:
    """Map a torch XPU device index to the DLPack (global SYCL) device index.

    Parameters
    ----------
    torch_index
        The index ``n`` of the torch device ``xpu:<n>``.

    Returns
    -------
    index
        The ``device_id`` that TVM FFI uses for the same device.

    """
    import torch  # noqa: PLC0415

    probe = torch.empty(0, device=torch.device("xpu", torch_index))
    device_type, device_id = _torch_dlpack_device(probe)
    if device_type != core.DLDeviceType.kDLOneAPI:
        raise RuntimeError(
            f"Expected torch xpu:{torch_index} to export as kDLOneAPI, got device type {device_type}"
        )
    return device_id


def xpu_device_from_torch(torch_device: Any) -> core.Device:
    """Convert a torch XPU device to the TVM FFI device referring to the same GPU.

    Parameters
    ----------
    torch_device
        A ``torch.device`` (or device string) of type ``xpu``. When no index is
        given, the current torch XPU device is used.

    Returns
    -------
    device
        The ``oneapi`` device whose index is the global SYCL device index.

    Examples
    --------
    .. code-block:: python

        import torch
        from tvm_ffi.utils.xpu import xpu_device_from_torch

        dev = xpu_device_from_torch(torch.device("xpu:0"))
        assert dev.type == "oneapi"

    """
    import torch  # noqa: PLC0415

    torch_device = torch.device(torch_device)
    if torch_device.type != "xpu":
        raise ValueError(f"Expected an xpu device, got {torch_device}")
    index = torch_device.index
    if index is None:
        index = torch.xpu.current_device()
    return device(core.DLDeviceType.kDLOneAPI, xpu_global_device_index(index))
