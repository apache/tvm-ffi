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
"""FFI API bindings for my_ffi_extension."""

# tvm-ffi-stubgen(begin): import-section
# fmt: off
# isort: off
from __future__ import annotations
from tvm_ffi import Object as _ffi_Object, init_ffi_api as _FFI_INIT_FUNC, register_object as _FFI_REG_OBJ
from tvm_ffi.libinfo import load_lib_module as _FFI_LOAD_LIB
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from tvm_ffi import Object
# isort: on
# fmt: on
# tvm-ffi-stubgen(end)
# tvm-ffi-stubgen(import-object): tvm_ffi.libinfo.load_lib_module;False;_FFI_LOAD_LIB
LIB = _FFI_LOAD_LIB("my_ffi_extension", "my_ffi_extension")
# tvm-ffi-stubgen(begin): global/my_ffi_extension
# fmt: off
_FFI_INIT_FUNC("my_ffi_extension", __name__)
if TYPE_CHECKING:
    def add_one(_0: int, /) -> int: ...
    def raise_error(_0: str, /) -> None: ...
# fmt: on
# tvm-ffi-stubgen(end)
# tvm-ffi-stubgen(import-object): tvm_ffi.register_object;False;_FFI_REG_OBJ
# tvm-ffi-stubgen(import-object): ffi.Object;False;_ffi_Object
@_FFI_REG_OBJ("my_ffi_extension.IntPair")
class IntPair(_ffi_Object):
    """FFI binding for `my_ffi_extension.IntPair`."""

    # tvm-ffi-stubgen(begin): object/my_ffi_extension.IntPair
    # fmt: off
    a: int
    b: int
    if TYPE_CHECKING:
        @staticmethod
        def __c_ffi_init__(_0: int, _1: int, /) -> Object: ...
        def sum(self, /) -> int: ...
    # fmt: on
    # tvm-ffi-stubgen(end)


__all__ = [
    # tvm-ffi-stubgen(begin): __all__
    "LIB",
    "IntPair",
    "add_one",
    "raise_error",
    # tvm-ffi-stubgen(end)
]
