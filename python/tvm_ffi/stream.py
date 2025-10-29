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
# pylint: disable=invalid-name
"""Stream context."""

from __future__ import annotations

from ctypes import c_void_p
from typing import Any

from . import core
from ._tensor import device


class StreamContext:
    """Represent a stream context in the FFI system.

    StreamContext helps setup ffi environment stream by python `with` statement.
    When entering `with` scope, it caches the current environment stream and
    setup the given new stream.
    When exiting `with` scope, it recovers the stream to the cached environment stream.

    Parameters
    ----------
    device
        The device to which the stream belongs.

    stream
        The stream handle.

    See Also
    --------
    :py:func:`tvm_ffi.use_raw_stream`, :py:func:`tvm_ffi.use_torch_stream`

    """

    def __init__(self, device: core.Device, stream: int | c_void_p) -> None:
        """Initialize a stream context with a device and stream handle."""
        self.device_type = device.dlpack_device_type()
        self.device_id = device.index
        self.stream = stream

    def __enter__(self) -> StreamContext:
        """Enter the context and set the current stream."""
        self.prev_stream = core._env_set_current_stream(
            self.device_type, self.device_id, self.stream
        )
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit the context and restore the previous stream."""
        self.prev_stream = core._env_set_current_stream(
            self.device_type, self.device_id, self.prev_stream
        )


try:
    import torch

    class TorchStreamContext:
        """Context manager that syncs Torch and FFI stream contexts."""

        def __init__(self, context: Any) -> None:
            """Initialize with an optional Torch stream/graph context wrapper."""
            self.torch_context = context

        def __enter__(self) -> TorchStreamContext:
            """Enter both Torch and FFI stream contexts."""
            if self.torch_context:
                self.torch_context.__enter__()
            current_stream = torch.cuda.current_stream()
            self.ffi_context = StreamContext(
                device(str(current_stream.device)), current_stream.cuda_stream
            )
            self.ffi_context.__enter__()
            return self

        def __exit__(self, *args: Any) -> None:
            """Exit both Torch and FFI stream contexts."""
            if self.torch_context:
                self.torch_context.__exit__(*args)
            self.ffi_context.__exit__(*args)

    def use_torch_stream(context: Any = None) -> TorchStreamContext:
        """Create an FFI stream context with a Torch stream or graph.

        cuda graph or current stream if `None` provided.

        Parameters
        ----------
        context
            The wrapped torch stream or cuda graph.

        Returns
        -------
        context
            The ffi stream context wrapping torch stream context.

        Examples
        --------
        .. code-block:: python

            s = torch.cuda.Stream()
            with tvm_ffi.use_torch_stream(torch.cuda.stream(s)):
                ...

            g = torch.cuda.CUDAGraph()
            with tvm_ffi.use_torch_stream(torch.cuda.graph(g)):
                ...

        Note
        ----
        When working with a raw ``cudaStream_t`` handle, use
        :py:func:`tvm_ffi.use_raw_stream` instead.

        """
        return TorchStreamContext(context)

except ImportError:

    def use_torch_stream(context: Any = None) -> TorchStreamContext:
        """Raise an informative error when Torch is unavailable."""
        raise ImportError("Cannot import torch")


def use_raw_stream(device: core.Device, stream: int | c_void_p) -> StreamContext:
    """Create an FFI stream context with the given device and stream handle.

    Parameters
    ----------
    device
        The device to which the stream belongs.

    stream
        The stream handle (for example, a CUDA ``cudaStream_t`` as an integer, or ``0``).

    Returns
    -------
    context
        The FFI stream context.

    Examples
    --------
    The example below uses a CPU device and a dummy stream handle. On CUDA, pass a
    real ``cudaStream_t`` integer.

    .. code-block:: python

        import tvm_ffi

        dev = tvm_ffi.device("cpu:0")
        with tvm_ffi.use_raw_stream(dev, 0):
            # Within the context, the current stream for this device is set
            assert tvm_ffi.get_raw_stream(dev) == 0

    See Also
    --------
    :py:func:`tvm_ffi.use_torch_stream`
        Use a Torch stream or CUDA graph as the source of truth.
    :py:func:`tvm_ffi.get_raw_stream`
        Query the current FFI stream for a device.

    """
    if not isinstance(stream, (int, c_void_p)):
        raise ValueError(
            "use_raw_stream only accepts int or c_void_p as stream input, "
            "try use_torch_stream when using torch.cuda.Stream or torch.cuda.graph"
        )
    return StreamContext(device, stream)


def get_raw_stream(device: core.Device) -> int:
    """Get the current FFI stream of a given device.

    Parameters
    ----------
    device
        The device to which the stream belongs.

    Returns
    -------
    stream
        The current FFI stream as an integer handle.

    Examples
    --------
    .. code-block:: python

        import tvm_ffi

        dev = tvm_ffi.device("cpu:0")
        # Default stream is implementation-defined; set it explicitly
        with tvm_ffi.use_raw_stream(dev, 0):
            assert tvm_ffi.get_raw_stream(dev) == 0

    See Also
    --------
    :py:func:`tvm_ffi.use_raw_stream`
        Set the current stream for a device.

    """
    return core._env_get_current_stream(device.dlpack_device_type(), device.index)
