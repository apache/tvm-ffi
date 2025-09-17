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

from ctypes import c_void_p
from typing import Any, Optional, Union

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
    device : Device
        The device to which the stream belongs.

    stream : Union[int, c_void_p]
        The stream handle.

    See Also
    --------
    :py:func:`tvm_ffi.use_raw_stream`, :py:func:`tvm_ffi.use_torch_stream`

    """

    def __init__(self, device: core.Device, stream: Union[int, c_void_p]):
        """Initialize a stream context with a device and stream handle."""
        self.device_type = device.dlpack_device_type()
        self.device_id = device.index
        self.stream = stream

    def __enter__(self):
        """Enter the context and set the current stream."""
        self.prev_stream = core._env_set_current_stream(
            self.device_type, self.device_id, self.stream
        )

    def __exit__(self, *args):
        """Exit the context and restore the previous stream."""
        self.prev_stream = core._env_set_current_stream(
            self.device_type, self.device_id, self.prev_stream
        )


try:
    import torch

    class TorchStreamContext:
        """Context manager that syncs Torch and FFI stream contexts."""

        def __init__(self, context: Optional[Any]):
            """Initialize with an optional Torch stream/graph context wrapper."""
            self.torch_context = context

        def __enter__(self):
            """Enter both Torch and FFI stream contexts."""
            if self.torch_context:
                self.torch_context.__enter__()
            current_stream = torch.cuda.current_stream()
            self.ffi_context = StreamContext(
                device(str(current_stream.device)), current_stream.cuda_stream
            )
            self.ffi_context.__enter__()

        def __exit__(self, *args):
            """Exit both Torch and FFI stream contexts."""
            if self.torch_context:
                self.torch_context.__exit__(*args)
            self.ffi_context.__exit__(*args)

    def use_torch_stream(context: Optional[Any] = None):
        """Create an FFI stream context with a Torch stream or graph.

        cuda graph or current stream if `None` provided.

        Parameters
        ----------
        context : Optional[Any]
            The wrapped torch stream or cuda graph.

        Returns
        -------
        context : tvm_ffi.TorchStreamContext
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
        When working with raw cudaStream_t handle, using :py:func:`tvm_ffi.use_raw_stream` instead.

        """
        return TorchStreamContext(context)

except ImportError:

    def use_torch_stream(context: Optional[Any] = None):
        """Raise an informative error when Torch is unavailable."""
        raise ImportError("Cannot import torch")


def use_raw_stream(device: core.Device, stream: Union[int, c_void_p]):
    """Create a ffi stream context with given device and stream handle.

    Parameters
    ----------
    device : tvm_ffi.Device
        The device to which the stream belongs.

    stream : Union[int, c_void_p]
        The stream handle.

    Returns
    -------
    context : tvm_ffi.StreamContext
        The ffi stream context.

    Note
    ----
    When working with torch stram or cuda graph, using :py:func:`tvm_ffi.use_torch_stream` instead.

    """
    if not isinstance(stream, (int, c_void_p)):
        raise ValueError(
            "use_raw_stream only accepts int or c_void_p as stram input, "
            "try use_torch_stream when using torch.cuda.Stream or torch.cuda.graph"
        )
    return StreamContext(device, stream)
