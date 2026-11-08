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

from .directives import Key, KeyTable


@dataclasses.dataclass(frozen=True)
class MarkerSyntax:
    """Comment-syntax-specific stub directive markers.

    All stub directives are embedded inside single-line comments. The comment
    token (``#`` for Python sources, ``//`` for Rust) parameterizes the marker
    set, while the directive grammar (``tvm-ffi-stubgen(begin): ...`` etc.)
    stays uniform.
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
    def skip_file(self) -> str:
        """Whole-file opt-out directive: ``<comment> tvm-ffi-stubgen(skip-file)``."""
        return f"{self.prefix}skip-file)"

    def directive(self, subject: str) -> str:
        """One-line directive marker: ``<comment> tvm-ffi-stubgen(<subject>):``."""
        return f"{self.prefix}{subject}):"


PYTHON_SYNTAX = MarkerSyntax(comment="#")
RUST_SYNTAX = MarkerSyntax(comment="//")

#: Map a source-file extension to the marker syntax used inside it. The block
#: parser selects the syntax per file. Which extensions a run actually visits is
#: decided by the active generator (``Generator.source_exts``), not by this map.
SYNTAX_BY_EXT: dict[str, MarkerSyntax] = {
    ".py": PYTHON_SYNTAX,
    ".pyi": PYTHON_SYNTAX,
    ".rs": RUST_SYNTAX,
}

#: Directive keys the language-neutral pipeline reads (see :mod:`.directives`): an
#: ``object`` maps its type key (``ty_map``) or opts out of ``--init`` and ``prefix``
#: (``skip``); ``file: mod(prefix = <p>)`` asks for a block per object under ``p``.
#: The active generator declares every other key (``Generator.directive_keys``).
PIPELINE_DIRECTIVE_KEYS: KeyTable = {
    "object": {"skip": Key(), "ty_map": Key(takes_value=True)},
    "file": {"prefix": Key(takes_value=True)},
}

STUB_BLOCK_KINDS: TypeAlias = Literal[
    "global",
    "object",
    "import-section",
    "directive",
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

# Language-neutral metadata transform applied while building `ObjectInfo` from
# the FFI reflection registry (see `utils.ObjectInfo.from_type_info`).
FN_NAME_MAP: dict[str, str] = {}

BUILTIN_TYPE_KEYS = {
    "ffi.BigInt",
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
