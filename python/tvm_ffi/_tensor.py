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
# we name it as _tensor.py to avoid potential future case
# if we also want to expose a tensor function in the root namespace

from numbers import Integral
from typing import Any, Optional, Union

from . import _ffi_api, registry
from .core import (
    _CLASS_DEVICE,
    Device,
    DLDeviceType,
    PyNativeObject,
    Tensor,
    _shape_obj_get_py_tuple,
    from_dlpack,
)


@registry.register_object("ffi.Shape")
class Shape(tuple, PyNativeObject):
    """Shape tuple that represents `ffi::Shape` returned by a ffi call.

    Note:
    ----
    This class subclasses `tuple` so it can be used in most places where
    tuple is used in python array apis.

    """

    def __new__(cls, content: tuple[int, ...]) -> "Shape":
        if any(not isinstance(x, Integral) for x in content):
            raise ValueError("Shape must be a tuple of integers")
        val = tuple.__new__(cls, content)
        val.__init_tvm_ffi_object_by_constructor__(_ffi_api.Shape, *content)
        return val

    # pylint: disable=no-self-argument
    def __from_tvm_ffi_object__(cls, obj: Any) -> "Shape":
        """Construct from a given tvm object."""
        content = _shape_obj_get_py_tuple(obj)
        val = tuple.__new__(cls, content)
        val.__tvm_ffi_object__ = obj
        return val


def device(device_type: Union[str, int, DLDeviceType], index: Optional[int] = None) -> Device:
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

      assert tvm_ffi.device("cuda:0") == tvm_ffi.device("cuda", 0)
      assert tvm_ffi.device("cpu:0") == tvm_ffi.device("cpu", 0)

    """
    return _CLASS_DEVICE(device_type, index)


__all__ = ["DLDeviceType", "Device", "Tensor", "device", "from_dlpack"]
