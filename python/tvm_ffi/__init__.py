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
"""TVM FFI Python package."""

# order matters here so we need to skip isort here
# isort: skip_file
__version__ = "0.1.0"

# HACK: try importing torch first, to avoid a potential
# symbol conflict when both torch and tvm_ffi are imported.
# This conflict can be reproduced in a very narrow scenario:
# 1. GitHub action on Windows X64
# 2. Python 3.12
# 3. torch 2.9.0
try:
    import torch  # type: ignore
except ImportError:
    pass

# base always go first to load the libtvm_ffi
from . import base
from . import libinfo

# package init part
from .registry import (
    register_object,
    register_global_func,
    get_global_func,
    get_global_func_metadata,
    remove_global_func,
    init_ffi_api,
)
from ._dtype import dtype
from .core import Object, ObjectConvertible, Function
from ._convert import convert
from .error import register_error
from ._tensor import Device, device, DLDeviceType
from ._tensor import from_dlpack, Tensor, Shape
from .container import Array, Map
from .module import Module, system_lib, load_module
from .stream import StreamContext, get_raw_stream, use_raw_stream, use_torch_stream
from . import serialization
from . import access_path
from . import dataclasses

# optional module to speedup dlpack conversion
from . import _optional_torch_c_dlpack

__all__ = [
    "Array",
    "DLDeviceType",
    "Device",
    "Device",
    "Function",
    "Map",
    "Module",
    "Object",
    "Object",
    "ObjectConvertible",
    "Shape",
    "Tensor",
    "__version__",
    "access_path",
    "convert",
    "dataclasses",
    "device",
    "dtype",
    "from_dlpack",
    "get_global_func",
    "get_global_func_metadata",
    "init_ffi_api",
    "load_module",
    "register_error",
    "register_global_func",
    "register_object",
    "remove_global_func",
    "serialization",
    "system_lib",
]
