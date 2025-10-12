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
"""Conversion utilities to bring python objects into ffi values."""

from __future__ import annotations

from numbers import Number
from types import ModuleType
from typing import Any

from . import container, core

torch: ModuleType | None = None
try:
    import torch  # type: ignore[no-redef]
except ImportError:
    pass

numpy: ModuleType | None = None
try:
    import numpy
except ImportError:
    pass


def convert(value: Any) -> Any:  # noqa: PLR0911,PLR0912
    """Convert a python object to ffi values.

    Parameters
    ----------
    value
        The python object to be converted.

    Returns
    -------
    ffi_obj
        The converted TVM FFI object.

    Note
    ----
    Function arguments to ffi function calls are
    automatically converted. So this function is mainly
    only used in internal or testing scenarios.

    """
    if isinstance(value, (core.Object, core.PyNativeObject, bool, Number)):
        return value
    elif isinstance(value, (tuple, list)):
        return container.Array(value)
    elif isinstance(value, dict):
        return container.Map(value)
    elif isinstance(value, str):
        return core.String(value)
    elif isinstance(value, (bytes, bytearray)):
        return core.Bytes(value)
    elif isinstance(value, core.ObjectConvertible):
        return value.asobject()
    elif callable(value):
        return core._convert_to_ffi_func(value)
    elif value is None:
        return None
    elif hasattr(value, "__dlpack__"):
        return core.from_dlpack(value)
    elif torch is not None and isinstance(value, torch.dtype):
        return core._convert_torch_dtype_to_ffi_dtype(value)
    elif numpy is not None and isinstance(value, numpy.dtype):
        return core._convert_numpy_dtype_to_ffi_dtype(value)
    elif isinstance(value, Exception):
        return core._convert_to_ffi_error(value)
    else:
        # in this case, it is an opaque python object
        return core._convert_to_opaque_object(value)
