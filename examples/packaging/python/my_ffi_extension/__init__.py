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
# specific language governing permissions and limitations.
# order matters here so we need to skip isort here
# isort: skip_file
"""Public Python API for the example tvm-ffi extension package."""

from typing import Any, TYPE_CHECKING

import tvm_ffi

from .base import _LIB
from . import _ffi_api


@tvm_ffi.register_object("my_ffi_extension.IntPair")
class IntPair(tvm_ffi.Object):
    """IntPair object."""

    def __init__(self, a: int, b: int) -> None:
        """Construct the object."""
        # __ffi_init__ call into the refl::init<> registered
        # in the static initialization block of the extension library
        self.__ffi_init__(a, b)


def add_one(x: Any, y: Any) -> None:
    """Add one to the input tensor.

    Parameters
    ----------
    x : Tensor
      The input tensor.
    y : Tensor
      The output tensor.

    """
    return _LIB.add_one(x, y)


def raise_error(msg: str) -> None:
    """Raise an error with the given message.

    Parameters
    ----------
    msg : str
        The message to raise the error with.

    Raises
    ------
    RuntimeError
        The error raised by the function.

    """
    return _ffi_api.raise_error(msg)
