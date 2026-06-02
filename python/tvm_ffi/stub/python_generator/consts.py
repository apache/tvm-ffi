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
"""Python-specific constants for the ``tvm-ffi-stubgen`` Python generator.

These tables map FFI-origin names and module prefixes onto Python typing /
import syntax. They are intentionally kept out of the language-agnostic
:mod:`tvm_ffi.stub.consts` so that a non-Python generator never inherits Python
typing assumptions.
"""

from __future__ import annotations

#: Default FFI-origin -> Python-type name map used to seed a render.
TY_MAP_DEFAULTS = {
    "Any": "typing.Any",
    "Callable": "typing.Callable",
    "Array": "collections.abc.Sequence",
    "List": "collections.abc.MutableSequence",
    "Map": "collections.abc.Mapping",
    "Dict": "collections.abc.MutableMapping",
    "Object": "ffi.Object",
    "Tensor": "ffi.Tensor",
    "dtype": "ffi.dtype",
    "Device": "ffi.Device",
}

# TODO(@junrushao): Make it configurable
#: Module-prefix rewrites applied when constructing a Python ``import`` path.
MOD_MAP = {
    "testing": "tvm_ffi.testing",
    "ffi": "tvm_ffi",
}
