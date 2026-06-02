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

from __future__ import annotations

import dataclasses
from typing import Literal

from typing_extensions import TypeAlias


@dataclasses.dataclass(frozen=True)
class MarkerSyntax:
    """Comment-syntax-specific stub directive markers.

    All stub directives are embedded inside single-line comments. The comment
    token (currently ``#`` for Python sources) parameterizes the marker set,
    while the directive grammar (``tvm-ffi-stubgen(begin): ...`` etc.) stays
    uniform.
    """

    comment: str
    """The line-comment token for the target language."""

    @property
    def prefix(self) -> str:
        """Common prefix shared by every stub directive on a line."""
        return f"{self.comment} tvm-ffi-stubgen("

    @property
    def begin(self) -> str:
        """Marker that opens a generated block: ``<comment> tvm-ffi-stubgen(begin):``."""
        return f"{self.prefix}begin):"

    @property
    def end(self) -> str:
        """Marker that closes a generated block: ``<comment> tvm-ffi-stubgen(end)``."""
        return f"{self.prefix}end)"

    @property
    def ty_map(self) -> str:
        """One-line type-map directive: ``<comment> tvm-ffi-stubgen(ty-map):``."""
        return f"{self.prefix}ty-map):"

    @property
    def import_object(self) -> str:
        """One-line import-object directive: ``<comment> tvm-ffi-stubgen(import-object):``."""
        return f"{self.prefix}import-object):"

    @property
    def skip_file(self) -> str:
        """Whole-file opt-out directive: ``<comment> tvm-ffi-stubgen(skip-file)``."""
        return f"{self.prefix}skip-file)"


PYTHON_SYNTAX = MarkerSyntax(comment="#")

#: Map a source-file extension to the marker syntax used inside it. The block
#: parser selects the syntax per file.
SYNTAX_BY_EXT: dict[str, MarkerSyntax] = {
    ".py": PYTHON_SYNTAX,
    ".pyi": PYTHON_SYNTAX,
}

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
DOC_URL = "https://tvm.apache.org/ffi/packaging/stubgen.html"

DEFAULT_SOURCE_EXTS = set(SYNTAX_BY_EXT)

# Language-neutral metadata transform applied while building `ObjectInfo` from
# the FFI reflection registry (see `utils.ObjectInfo.from_type_info`).
FN_NAME_MAP: dict[str, str] = {}

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
