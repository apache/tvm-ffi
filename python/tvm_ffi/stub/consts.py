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
"""Constants used in stub generation."""

STUB_PREFIX = "# tvm-ffi-stubgen("
STUB_BEGIN = f"{STUB_PREFIX}begin):"
STUB_END = f"{STUB_PREFIX}end)"
STUB_TY_MAP = f"{STUB_PREFIX}ty-map):"
STUB_SKIP_FILE = f"{STUB_PREFIX}skip-file)"

TERM_RESET = "\033[0m"
TERM_BOLD = "\033[1m"
TERM_BLACK = "\033[30m"
TERM_RED = "\033[31m"
TERM_GREEN = "\033[32m"
TERM_YELLOW = "\033[33m"
TERM_BLUE = "\033[34m"
TERM_MAGENTA = "\033[35m"
TERM_CYAN = "\033[36m"
TERM_WHITE = "\033[37m"

DEFAULT_SOURCE_EXTS = {".py", ".pyi"}
TY_MAP_DEFAULTS = {
    "list": "collections.abc.Sequence",
    "dict": "collections.abc.Mapping",
}

TY_TO_IMPORT = {
    "Any": "typing.Any",
    "Callable": "typing.Callable",
    "Mapping": "typing.Mapping",
    "Object": "tvm_ffi.Object",
    "Tensor": "tvm_ffi.Tensor",
    "dtype": "tvm_ffi.dtype",
    "Device": "tvm_ffi.Device",
}

# TODO(@junrushao): Make it configurable
MOD_MAP = {
    "testing": "tvm_ffi.testing",
    "ffi": "tvm_ffi",
}

FN_NAME_MAP = {
    "__ffi_init__": "__c_ffi_init__",
}
