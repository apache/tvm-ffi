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
"""Pluggable code generators for ``tvm-ffi-stubgen``.

The stub generator separates two concerns:

1. *Language-agnostic* infrastructure — reading the FFI reflection registry
   (:mod:`.lib_state`), parsing/writing marker blocks (:mod:`.file_utils`), and
   the abstract object/function metadata (:class:`.utils.ObjectInfo`,
   :class:`.utils.FuncInfo`). None of this knows or cares about the target
   language.
2. *Language-specific* rendering — turning that metadata into concrete source
   text (Python ``def``/``class`` vs Rust ``fn``/``struct``/``impl``), rendering
   a :class:`~tvm_ffi.core.TypeSchema` into a target-language type expression
   (``T | None`` vs ``Option<T>``), and modelling that language's imports.

A :class:`Generator` encapsulates concern (2); ``cli.py`` drives concern (1) and
delegates every act of emitting text — and every act of collecting imports — to
the active generator. The import collector is opaque to the pipeline: ``cli.py``
asks the generator to create one, seed it from ``import-object`` directives, and
later render it, but never reaches inside. Adding a language is therefore
"implement one more :class:`Generator`" rather than forking the pipeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from . import consts as C
from .python_generator import PythonGenerator
from .rust_generator import RustGenerator

if TYPE_CHECKING:
    from pathlib import Path

    from .file_utils import CodeBlock
    from .utils import FuncInfo, InitConfig, ObjectInfo, Options


class Generator(Protocol):
    """Language-specific rendering surface used by the stub-generation pipeline.

    Each method that ends in ``_block`` mutates ``code.lines`` in place to hold
    the freshly generated text between the ``begin``/``end`` markers. The
    ``*_file`` methods return whole-file scaffolding text used by ``--init``
    mode. Implementations must be stateless with respect to a single file so the
    pipeline can process files in any order.

    The ``imports`` parameter threaded through the ``_block`` methods is an
    opaque import collector created by :meth:`new_imports`; only the generator
    that created it understands its contents.
    """

    #: Short identifier, e.g. ``"python"`` or ``"rust"``.
    name: str

    #: Comment-marker syntax for the files this generator emits.
    syntax: C.MarkerSyntax

    def default_ty_map(self) -> dict[str, str]:
        """Return the default FFI-origin -> target-type name map for this language."""
        ...

    # --- import collection (representation is generator-private) ------------

    def new_imports(self) -> Any:
        """Create a fresh, empty import collector for one file."""
        ...

    def add_imported_object(
        self, imports: Any, name: str, type_checking_only: str, alias: str
    ) -> None:
        """Record an ``import-object`` directive (raw directive fields) into ``imports``."""
        ...

    def canonical_type_name(self, type_key: str) -> str:
        """Return the canonical identifier for a locally-defined type key.

        Used to suppress importing a type the file itself defines, and to feed
        the public-export list. Must be comparable to the names produced while
        collecting imports.
        """
        ...

    def extra_export_names(self, imports: Any) -> set[str]:
        """Return extra public-export names implied by the collected imports."""
        ...

    # --- per-block generation (mutates `code.lines`) ------------------------

    def generate_global_funcs_block(
        self,
        code: CodeBlock,
        global_funcs: list[FuncInfo],
        ty_map: dict[str, str],
        imports: Any,
        opt: Options,
    ) -> None:
        """Emit free function signatures for a ``global/<prefix>`` block."""
        ...

    def generate_object_block(
        self,
        code: CodeBlock,
        ty_map: dict[str, str],
        imports: Any,
        opt: Options,
        obj_info: ObjectInfo,
    ) -> None:
        """Emit a type definition (fields + methods + init) for an ``object/<key>`` block."""
        ...

    def generate_import_section_block(
        self, code: CodeBlock, imports: Any, opt: Options, defined_types: set[str]
    ) -> None:
        """Emit the import/`use` statements collected while rendering other blocks.

        ``defined_types`` holds the canonical names defined in this file so the
        generator can drop imports that would shadow a local definition.
        """
        ...

    def generate_all_block(self, code: CodeBlock, names: set[str], opt: Options) -> None:
        """Emit the public-export list for this generator."""
        ...

    def generate_export_block(self, code: CodeBlock) -> None:
        """Emit a submodule re-export for an ``export/<submodule>`` block."""
        ...

    # --- whole-file scaffolding (used by `--init` mode) ---------------------

    def api_filename(self) -> str:
        """File name of the scaffolded API file (Python ``_ffi_api.py``; Rust ``mod.rs``)."""
        ...

    def init_filename(self) -> str:
        """File name of the scaffolded package entry (Python ``__init__.py``)."""
        ...

    def generate_api_file(
        self,
        code_blocks: list[CodeBlock],
        ty_map: dict[str, str],
        module_name: str,
        object_infos: list[ObjectInfo],
        init_cfg: InitConfig,
        is_root: bool,
    ) -> str:
        """Return text appended to a scaffolded API file (Python ``_ffi_api.py``).

        Whole-file headers (e.g. Rust's ``#![allow]`` inner attribute, which must
        precede every item) may be emitted only when ``code_blocks`` is empty,
        i.e. the target file did not exist or was empty.
        """
        ...

    def generate_init_file(
        self, code_blocks: list[CodeBlock], module_name: str, submodule: str
    ) -> str:
        """Return text appended to a freshly scaffolded package entry (Python ``__init__.py``)."""
        ...

    def finalize_init(self, init_path: Path, generated_prefixes: set[str]) -> None:
        """Post-``--init`` hook to stitch the generated tree (after all files exist).

        Python is a no-op (packages need no parent declarations). Rust writes the
        ``pub mod <child>;`` declarations that wire the module tree together.
        """
        ...


_GENERATORS: dict[str, Generator] = {
    "python": PythonGenerator(),
    "rust": RustGenerator(),
}


def get_generator(target: str) -> Generator:
    """Resolve a generator by target name."""
    if target not in _GENERATORS:
        known = ", ".join(sorted(_GENERATORS))
        raise ValueError(f"Unknown stubgen generator: {target!r}. Known generators: {known}")
    return _GENERATORS[target]
