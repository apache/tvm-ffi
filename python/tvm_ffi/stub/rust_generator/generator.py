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
"""The Rust code generator for ``tvm-ffi-stubgen``.

:class:`RustGenerator` implements the :class:`tvm_ffi.stub.generator.Generator`
protocol, delegating the actual rendering to ``rust_generator.codegen``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .. import consts as C
from . import consts as C_RUST
from .codegen import (
    finalize_rust_module_tree,
    generate_rust_api_file,
    generate_rust_import_section,
    generate_rust_object,
)
from .utils import RustImports, RustUse

if TYPE_CHECKING:
    from pathlib import Path

    from ..file_utils import CodeBlock
    from ..utils import FuncInfo, InitConfig, ObjectInfo, Options


class RustGenerator:
    """Generator that emits opaque Rust bindings for reflected objects.

    Every object gets a wrapper whose fields are read through the C ABI
    getters (see :mod:`.codegen`); global functions and ``__all__`` / ``export``
    blocks are not generated.
    """

    name = "rust"
    syntax = C.RUST_SYNTAX
    source_exts = frozenset({".rs"})
    directive_kinds = C_RUST.RUST_DIRECTIVE_KINDS

    def default_ty_map(self) -> dict[str, str]:
        """Return the default FFI-origin -> Rust-type name map."""
        return C_RUST.RUST_TY_MAP_DEFAULTS.copy()

    # --- per-file collection --------------------------------------------------

    def new_imports(self) -> RustImports:
        """Create an empty per-file collector (``use`` items and directives)."""
        return RustImports()

    def add_directive(self, imports: RustImports, name: str, payload: str, lineno: int) -> None:
        """Record ``import-object`` as a ``use``; parse the Rust directives into the collector.

        For ``import-object`` only the first ``;``-separated field (the path) is
        meaningful: Rust has no ``TYPE_CHECKING`` split and no ``use .. as``.
        """
        if name == "import-object":
            imports.record(payload.split(";", 1)[0].strip())
        else:
            imports.directives.add(name, payload, lineno)

    def canonical_type_name(self, type_key: str) -> str:
        """Return the Rust path for a defined type key (matches :attr:`RustUse.path`)."""
        return RustUse(type_key).path

    def extra_export_names(self, imports: RustImports) -> set[str]:
        """No extra export names for Rust."""
        return set()

    # --- per-block generation -------------------------------------------------

    def generate_global_funcs_block(
        self,
        code: CodeBlock,
        global_funcs: list[FuncInfo],
        ty_map: dict[str, str],
        imports: RustImports,
        opt: Options,
    ) -> None:
        """No-op: Rust reaches global functions through ``cached_global_func!``."""

    def generate_object_block(
        self,
        code: CodeBlock,
        ty_map: dict[str, str],
        imports: RustImports,
        opt: Options,
        obj_info: ObjectInfo,
    ) -> None:
        """Emit the opaque Rust binding for an ``object/<key>`` block."""
        generate_rust_object(code, ty_map, imports, opt, obj_info)

    def generate_import_section_block(
        self, code: CodeBlock, imports: RustImports, opt: Options, defined_types: set[str]
    ) -> None:
        """Emit Rust ``use`` statements for the collected imports."""
        generate_rust_import_section(code, imports, opt, defined_types)

    def generate_all_block(self, code: CodeBlock, names: set[str], opt: Options) -> None:
        """No-op: Rust re-exports are not generated."""

    def generate_export_block(self, code: CodeBlock) -> None:
        """No-op: submodule re-export is not generated."""

    def generate_helpers_block(self, code: CodeBlock, opt: Options) -> None:
        """No-op: the generated bindings need no per-file support code."""

    # --- whole-file scaffolding (`--init` mode) --------------------------------

    def api_filename(self) -> str:
        """One Rust file per module prefix."""
        return "mod.rs"

    def init_filename(self) -> str:
        """No separate entry file for Rust; the API file is the module."""
        return "mod.rs"

    def generate_api_file(
        self,
        code_blocks: list[CodeBlock],
        ty_map: dict[str, str],
        module_name: str,
        object_infos: list[ObjectInfo],
        init_cfg: InitConfig,
        is_root: bool,
    ) -> str:
        """Scaffold a Rust binding file: header + import/object markers."""
        return generate_rust_api_file(
            code_blocks, ty_map, module_name, object_infos, init_cfg, is_root, self.syntax
        )

    def generate_init_file(
        self, code_blocks: list[CodeBlock], module_name: str, submodule: str
    ) -> str:
        """No-op: the API file is the module entry."""
        return ""

    def finalize_init(self, init_path: Path, generated_prefixes: set[str]) -> None:
        """Auto-form the module tree: write ``pub mod <child>;`` declarations."""
        finalize_rust_module_tree(init_path, generated_prefixes)
