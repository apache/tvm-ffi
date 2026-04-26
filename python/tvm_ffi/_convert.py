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
from typing import Any, Callable

from . import _dtype, container, core

try:
    import torch
except ImportError:
    torch = None  # ty: ignore[invalid-assignment]

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
    if isinstance(
        value, (core.Object, core.PyNativeObject, bool, Number, ctypes.c_void_p, _dtype.dtype)
    ):
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


def convert_func(
    pyfunc: Callable[..., Any],
    tensor_cls: type | None = None,
) -> Any:
    """Convert a Python callable to an FFI :py:class:`~tvm_ffi.Function`.

    This is the callable-specific sibling of :py:func:`tvm_ffi.convert`.
    It accepts one extra argument, ``tensor_cls``, that lets the caller
    specify how tensor arguments should be delivered to the Python
    callable when the resulting :py:class:`Function` is invoked from C++.
    :py:func:`tvm_ffi.convert` has no such knob — it always produces a
    :py:class:`Function` whose callback receives ``tvm_ffi.Tensor`` for
    tensor args.

    Parameters
    ----------
    pyfunc : Callable
        The Python callable to wrap.
    tensor_cls : type, optional
        The class whose instances the callback should receive for tensor
        args. The class must expose a ``__dlpack_c_exchange_api__``
        :py:class:`PyCapsule`; its capsule is threaded into the callback
        closure so tensor args are converted at the C level (via the
        DLPack exchange API) before the Python callback body runs — this
        is significantly faster than calling ``torch.from_dlpack(x)`` (or
        equivalent) inside the callback. Raises :py:class:`TypeError` if
        ``tensor_cls`` does not expose the attribute.

        When ``tensor_cls`` is ``None``, ``convert_func`` behaves like the
        callable branch of :py:func:`tvm_ffi.convert`.

    Returns
    -------
    Function
        The wrapped FFI function.

    Examples
    --------
    .. code-block:: python

        import torch
        import tvm_ffi

        # Without tensor_cls: same as tvm_ffi.convert(pyfunc) — the callback
        # receives tvm_ffi.Tensor for tensor args.
        f = tvm_ffi.convert_func(lambda x: x + 1)
        assert isinstance(f, tvm_ffi.Function)


        # With tensor_cls=torch.Tensor: the callback receives torch.Tensor
        # directly; the DLPack conversion happens in C before the body runs.
        def callback(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return a + b


        g = tvm_ffi.convert_func(callback, tensor_cls=torch.Tensor)

    See Also
    --------
    :py:func:`tvm_ffi.convert` :
        Generic value-to-FFI conversion. Use this when you don't need to
        specify ``tensor_cls``.

    """
    return core._convert_to_ffi_func(pyfunc, tensor_cls=tensor_cls)
