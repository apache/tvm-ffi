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
"""The Python code generator for ``tvm-ffi-stubgen``.

:class:`PythonGenerator` implements the :class:`tvm_ffi.stub.generator.Generator`
protocol by delegating to :mod:`.codegen`. It owns the Python notion of an
import (:class:`.utils.ImportItem` / :class:`.utils.PythonImports`); the
language-agnostic pipeline only ever sees the opaque collector.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .. import consts as C
from . import codegen as G
from . import consts as PC
from .utils import ImportItem, PythonImports

if TYPE_CHECKING:
    from pathlib import Path

    from ..file_utils import CodeBlock
    from ..utils import FuncInfo, InitConfig, ObjectInfo, Options


class PythonGenerator:
    """Generator that emits Python type stubs by delegating to :mod:`.codegen`."""

    name = "python"
    syntax = C.PYTHON_SYNTAX

    def default_ty_map(self) -> dict[str, str]:
        """Return the default FFI-origin -> Python-type name map."""
        return PC.TY_MAP_DEFAULTS.copy()

    # --- import collection (Python representation is private) ---------------

    def new_imports(self) -> PythonImports:
        """Create an empty import collector."""
        return PythonImports()

    def add_imported_object(
        self, imports: PythonImports, name: str, type_checking_only: str, alias: str
    ) -> None:
        """Record an ``import-object`` directive into the collector."""
        tco = type_checking_only.lower() == "true"
        imports.items.append(ImportItem(name, type_checking_only=tco, alias=alias or None))
        if alias == "_FFI_LOAD_LIB" or name.endswith("libinfo.load_lib_module"):
            imports.has_lib_load = True

    def canonical_type_name(self, type_key: str) -> str:
        """Return the canonical (import-comparable) full name for a defined type key."""
        return ImportItem(type_key).full_name

    def extra_export_names(self, imports: PythonImports) -> set[str]:
        """Return extra ``__all__`` names implied by the collected imports."""
        return {"LIB"} if imports.has_lib_load else set()

    # --- per-block generation (mutates `code.lines`) ------------------------

    def generate_global_funcs_block(
        self,
        code: CodeBlock,
        global_funcs: list[FuncInfo],
        ty_map: dict[str, str],
        imports: PythonImports,
        opt: Options,
    ) -> None:
        """Emit Python free-function signatures for a ``global/<prefix>`` block."""
        G.generate_python_global_funcs(code, global_funcs, ty_map, imports.items, opt)

    def generate_object_block(
        self,
        code: CodeBlock,
        ty_map: dict[str, str],
        imports: PythonImports,
        opt: Options,
        obj_info: ObjectInfo,
    ) -> None:
        """Emit a Python class definition for an ``object/<key>`` block."""
        G.generate_python_object(code, ty_map, imports.items, opt, obj_info)

    def generate_import_section_block(
        self,
        code: CodeBlock,
        imports: PythonImports,
        opt: Options,
        defined_types: set[str],
    ) -> None:
        """Emit Python ``import`` statements for the collected imports.

        Imports whose full name is a type defined in this same file are dropped
        (you don't import what you define locally).
        """
        filtered = [i for i in imports.items if i.full_name not in defined_types]
        G.generate_python_import_section(code, filtered, opt)

    def generate_all_block(self, code: CodeBlock, names: set[str], opt: Options) -> None:
        """Emit a Python ``__all__`` list."""
        G.generate_python_all(code, names, opt)

    def generate_export_block(self, code: CodeBlock) -> None:
        """Emit a Python submodule re-export for an ``export/<submodule>`` block."""
        G.generate_python_export(code)

    # --- whole-file scaffolding (used by `--init` mode) ---------------------

    def api_filename(self) -> str:
        """Return the Python API file name."""
        return "_ffi_api.py"

    def init_filename(self) -> str:
        """Return the Python package entry file name."""
        return "__init__.py"

    def generate_api_file(
        self,
        code_blocks: list[CodeBlock],
        ty_map: dict[str, str],
        module_name: str,
        object_infos: list[ObjectInfo],
        init_cfg: InitConfig,
        is_root: bool,
    ) -> str:
        """Return text appended to a scaffolded ``_ffi_api.py``."""
        return G.generate_python_ffi_api(
            code_blocks, ty_map, module_name, object_infos, init_cfg, is_root, self.syntax
        )

    def generate_init_file(
        self, code_blocks: list[CodeBlock], module_name: str, submodule: str
    ) -> str:
        """Return text appended to a scaffolded ``__init__.py``."""
        return G.generate_python_init(code_blocks, module_name, submodule, self.syntax)

    def finalize_init(self, init_path: Path, generated_prefixes: set[str]) -> None:
        """No-op: Python packages need no parent-declares-child wiring."""
