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

from typing import Literal

from typing_extensions import TypeAlias

STUB_PREFIX = "# tvm-ffi-stubgen("
STUB_BEGIN = f"{STUB_PREFIX}begin):"
STUB_END = f"{STUB_PREFIX}end)"
STUB_TY_MAP = f"{STUB_PREFIX}ty-map):"
STUB_IMPORT_OBJECT = f"{STUB_PREFIX}import-object):"
STUB_SKIP_FILE = f"{STUB_PREFIX}skip-file)"
STUB_BLOCK_KINDS: TypeAlias = Literal[
    "global",
    "object",
    "ty-map",
    "import-section",
    "import-object",
    "export",
    "__all__",
    None,
]

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
DOC_URL = "https://tvm.apache.org/ffi/packaging/python_packaging.html#stub-generation-tool"

DEFAULT_SOURCE_EXTS = {".py", ".pyi"}
TY_MAP_DEFAULTS = {
    "Any": "typing.Any",
    "Callable": "typing.Callable",
    "Mapping": "typing.Mapping",
    "list": "collections.abc.Sequence",
    "dict": "collections.abc.Mapping",
    "Object": "ffi.Object",
    "Tensor": "ffi.Tensor",
    "dtype": "ffi.dtype",
    "Device": "ffi.Device",
}

# TODO(@junrushao): Make it configurable
MOD_MAP = {
    "testing": "tvm_ffi.testing",
    "ffi": "tvm_ffi",
}

FN_NAME_MAP = {
    "__ffi_init__": "__c_ffi_init__",
}

BUILTIN_TYPE_KEYS = {
    "ffi.Bytes",
    "ffi.Error",
    "ffi.Function",
    "ffi.Object",
    "ffi.OpaquePyObject",
    "ffi.SmallBytes",
    "ffi.SmallStr",
    "ffi.String",
    "ffi.Tensor",
}


def _prompt_globals(mod: str) -> str:
    return f"""{STUB_BEGIN} global/{mod}
{STUB_END}
"""


def _prompt_class_def(type_name: str, type_key: str, parent_type_name: str) -> str:
    return f'''@_FFI_REG_OBJ("{type_key}")
class {type_name}({parent_type_name}):
    """FFI binding for `{type_key}`."""

    {STUB_BEGIN} object/{type_key}
    {STUB_END}\n\n'''


def _prompt_import_object(type_key: str, type_name: str) -> str:
    return f"""{STUB_IMPORT_OBJECT} {type_key};False;{type_name}\n"""


PROMPT_IMPORT_SECTION = f"""
{STUB_BEGIN} import-section
{STUB_END}
"""

PROMPT_ALL_SECTION = f"""
__all__ = [
    {STUB_BEGIN} __all__
    {STUB_END}
]
"""
