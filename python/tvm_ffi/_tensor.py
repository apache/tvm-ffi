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
"""Tensor related objects and functions."""

from __future__ import annotations

# we name it as _tensor.py to avoid potential future case
# if we also want to expose a tensor function in the root namespace
from numbers import Integral
from typing import Any

from . import _ffi_api, core, registry
from .core import (
    Device,
    DLDeviceType,
    PyNativeObject,
    Tensor,
    _shape_obj_get_py_tuple,
    from_dlpack,
)


@registry.register_object("ffi.Shape")
class Shape(tuple, PyNativeObject):
    """Shape tuple that represents :cpp:class:`tvm::ffi::Shape` returned by an FFI call.

    Notes
    -----
    This class subclasses :class:`tuple` so it can be used in most places where
    :class:`tuple` is used in Python array APIs.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import tvm_ffi

        x = tvm_ffi.from_dlpack(np.arange(6, dtype="int32").reshape(2, 3))
        assert x.shape == (2, 3)

    """

    _tvm_ffi_cached_object: Any

    def __new__(cls, content: tuple[int, ...]) -> Shape:
        if any(not isinstance(x, Integral) for x in content):
            raise ValueError("Shape must be a tuple of integers")
        val: Shape = tuple.__new__(cls, content)
        val.__init_cached_object_by_constructor__(_ffi_api.Shape, *content)
        return val

    # pylint: disable=no-self-argument
    def __from_tvm_ffi_object__(cls, obj: Any) -> Shape:
        """Construct from a given tvm object."""
        content = _shape_obj_get_py_tuple(obj)
        val: Shape = tuple.__new__(cls, content)  # type: ignore[arg-type]
        val._tvm_ffi_cached_object = obj  # type: ignore[attr-defined]
        return val


def device(device_type: str | int | DLDeviceType, index: int | None = None) -> Device:
    """Construct a TVM FFI device with given device type and index.

    Parameters
    ----------
    device_type: str or int
        The device type or name.

    index: int, optional
        The device index.

    Returns
    -------
    device: tvm_ffi.Device

    Examples
    --------
    Device can be used to create reflection of device by
    string representation of the device type.

    .. code-block:: python

      import tvm_ffi
      assert tvm_ffi.device("cuda:0") == tvm_ffi.device("cuda", 0)
      assert tvm_ffi.device("cpu:0") == tvm_ffi.device("cpu", 0)

    """
    # must refer to core._CLASS_DEVICE so we pick up override here
    return core._CLASS_DEVICE(device_type, index)


__all__ = ["DLDeviceType", "Device", "Tensor", "device", "from_dlpack"]
