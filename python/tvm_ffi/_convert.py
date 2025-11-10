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
"""Conversion utilities to convert Python objects into TVM FFI values."""

from __future__ import annotations

import ctypes
from numbers import Number
from types import ModuleType
from typing import Any

from . import _dtype, container, core

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
    """Convert a Python object into TVM FFI values.

    This helper mirrors the automatic argument conversion that happens when
    calling FFI functions. It is primarily useful in tests or places where
    an explicit conversion is desired.

    Parameters
    ----------
    value
        The Python object to be converted.

    Returns
    -------
    ffi_obj
        The converted TVM FFI object.

    Examples
    --------
    .. code-block:: python

        import tvm_ffi

        # Lists and tuples become tvm_ffi.Array
        a = tvm_ffi.convert([1, 2, 3])
        assert isinstance(a, tvm_ffi.Array)

        # Dicts become tvm_ffi.Map
        m = tvm_ffi.convert({"a": 1, "b": 2})
        assert isinstance(m, tvm_ffi.Map)

        # Strings and bytes become zero-copy FFI-aware types
        s = tvm_ffi.convert("hello")
        b = tvm_ffi.convert(b"bytes")
        assert isinstance(s, tvm_ffi.core.String)
        assert isinstance(b, tvm_ffi.core.Bytes)

        # Callables are wrapped as tvm_ffi.Function
        f = tvm_ffi.convert(lambda x: x + 1)
        assert isinstance(f, tvm_ffi.Function)

        # Array libraries that support DLPack export can be converted to Tensor
        import numpy as np
        x = tvm_ffi.convert(np.arange(4, dtype="int32"))
        assert isinstance(x, tvm_ffi.Tensor)

    Note
    ----
    Function arguments to ffi function calls are
    automatically converted. So this function is mainly
    only used in internal or testing scenarios.

    """
    if isinstance(value, (core.Object, core.PyNativeObject, bool, Number, ctypes.c_void_p)):
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
    elif hasattr(value, "__dlpack_data_type__"):
        cdtype = core._create_cdtype_from_tuple(core.DataType, *value.__dlpack_data_type__())
        dtype = str.__new__(_dtype.dtype, str(cdtype))
        dtype._tvm_ffi_dtype = cdtype
        return dtype
    elif isinstance(value, Exception):
        return core._convert_to_ffi_error(value)
    elif hasattr(value, "__tvm_ffi_object__"):
        return value.__tvm_ffi_object__()
    # keep rest protocol values as it is as they can be handled by ffi function
    elif hasattr(value, "__cuda_stream__"):
        return value
    elif hasattr(value, "__tvm_ffi_opaque_ptr__"):
        return value
    elif hasattr(value, "__dlpack_device__"):
        return value
    elif hasattr(value, "__tvm_ffi_int__"):
        return value
    elif hasattr(value, "__tvm_ffi_float__"):
        return value
    else:
        # in this case, it is an opaque python object
        return core._convert_to_opaque_object(value)
