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
"""Code generation logic for the `tvm-ffi-stubgen` tool."""

from __future__ import annotations

from typing import Callable

from . import consts as C
from .file_utils import CodeBlock
from .utils import FuncInfo, ImportItem, InitConfig, ObjectInfo, Options


def _type_suffix_and_record(
    ty_map: dict[str, str], imports: list[ImportItem]
) -> Callable[[str], str]:
    def _run(name: str) -> str:
        nonlocal ty_map, imports
        name = ty_map.get(name, name)
        if "." in name:
            imports.append(ImportItem(name, type_checking_only=True, alias=None))
        return name.rsplit(".", 1)[-1]

    return _run


def generate_global_funcs(
    code: CodeBlock,
    global_funcs: list[FuncInfo],
    ty_map: dict[str, str],
    imports: list[ImportItem],
    opt: Options,
) -> None:
    """Generate function signatures for global functions.

    It processes: global/${prefix}@${import_from="tvm_ffi")
    """
    assert len(code.lines) >= 2
    if not global_funcs:
        return
    assert isinstance(code.param, tuple)
    prefix, import_from = code.param
    if not import_from:
        import_from = "tvm_ffi"
    imports.extend(
        [
            ImportItem(
                f"{import_from}.init_ffi_api",
                type_checking_only=False,
                alias="_FFI_INIT_FUNC",
            ),
            ImportItem(
                "typing.TYPE_CHECKING",
                type_checking_only=False,
            ),
        ]
    )
    fn_ty_map = _type_suffix_and_record(ty_map, imports)
    results: list[str] = [
        "# fmt: off",
        f'_FFI_INIT_FUNC("{prefix}", __name__)',
        "if TYPE_CHECKING:",
        *[func.gen(fn_ty_map, indent=opt.indent) for func in global_funcs],
        "# fmt: on",
    ]
    indent = " " * code.indent
    code.lines = [
        code.lines[0],
        *[indent + line for line in results],
        code.lines[-1],
    ]


def generate_object(
    code: CodeBlock,
    ty_map: dict[str, str],
    imports: list[ImportItem],
    opt: Options,
    obj_info: ObjectInfo,
) -> None:
    """Generate a class definition for an object type.

    It processes: object/${type_key}
    """
    assert len(code.lines) >= 2
    info = obj_info
    fn_ty_map = _type_suffix_and_record(ty_map, imports)
    if info.methods:
        imports.append(
            ImportItem(
                "typing.TYPE_CHECKING",
                type_checking_only=False,
            )
        )
        results = [
            "# fmt: off",
            *info.gen_fields(fn_ty_map, indent=0),
            "if TYPE_CHECKING:",
            *info.gen_methods(fn_ty_map, indent=opt.indent),
            "# fmt: on",
        ]
    else:
        results = [
            "# fmt: off",
            *info.gen_fields(fn_ty_map, indent=0),
            "# fmt: on",
        ]
    indent = " " * code.indent
    code.lines = [
        code.lines[0],
        *[indent + line for line in results],
        code.lines[-1],
    ]


def generate_import_section(
    code: CodeBlock,
    imports: list[ImportItem],
    opt: Options,
) -> None:
    """Generate import statements for the types used in the stub.

    It processes: import-section
    """
    imports_concrete: dict[str, list[ImportItem]] = {}
    imports_ty_check: dict[str, list[ImportItem]] = {}
    for item in imports:
        if item.type_checking_only:
            imports_ty_check.setdefault(item.mod, []).append(item)
        else:
            imports_concrete.setdefault(item.mod, []).append(item)
    if imports_ty_check:
        imports_concrete.setdefault("typing", []).append(
            ImportItem("typing.TYPE_CHECKING", type_checking_only=True)
        )

    def _make_line(mod: str, items: list[ImportItem], indent: int) -> str:
        items.sort(key=lambda item: item.name)
        names = ", ".join(sorted(set(item.name_with_alias for item in items)))
        indent_str = " " * indent
        if mod:
            return f"{indent_str}from {mod} import {names}"
        else:
            return f"{indent_str}import {names}"

    results: list[str] = []
    if imports_concrete:
        results.extend(
            _make_line(mod, imports_concrete[mod], indent=0) for mod in sorted(imports_concrete)
        )
    if imports_ty_check:
        results.append("if TYPE_CHECKING:")
        results.extend(
            _make_line(mod, imports_ty_check[mod], opt.indent) for mod in sorted(imports_ty_check)
        )
    if results:
        code.lines = [
            code.lines[0],
            "# fmt: off",
            "# isort: off",
            "from __future__ import annotations",
            *results,
            "# isort: on",
            "# fmt: on",
            code.lines[-1],
        ]


def generate_all(code: CodeBlock, names: set[str], opt: Options) -> None:
    """Generate an `__all__` variable for the given names."""
    assert len(code.lines) >= 2
    if not names:
        return

    indent = " " * code.indent
    names = {f.rsplit(".", 1)[-1] for f in names}

    def _sort_key(name: str) -> tuple[int, str]:
        if name.isupper():
            return (0, name)
        if name and name[0].isupper() and not "_" in name:
            return (1, name)
        return (2, name)

    code.lines = [
        code.lines[0],
        *[f'{indent}"{name}",' for name in sorted(names, key=_sort_key)],
        code.lines[-1],
    ]


def generate_export(code: CodeBlock) -> None:
    """Generate an `__all__` variable for the given names."""
    assert len(code.lines) >= 2

    mod = code.param
    code.lines = [
        code.lines[0],
        "# fmt: off",
        "# isort: off",
        f"from .{mod} import *  # noqa: F403",
        f"from .{mod} import __all__ as {mod}__all__",
        'if "__all__" not in globals():',
        "    __all__ = []",
        f"__all__.extend({mod}__all__)",
        "# isort: on",
        "# fmt: on",
        code.lines[-1],
    ]


def generate_ffi_api(
    code_blocks: list[CodeBlock],
    ty_map: dict[str, str],
    module_name: str,
    object_infos: list[ObjectInfo],
    init_cfg: InitConfig,
    is_root: bool,
) -> str:
    """Generate the initial FFI API stub code for a given module."""
    # TODO(@junrus): New code is appended to the end of the file.
    # We should consider a more sophisticated approach.
    append = ""

    # Part 0. Imports
    if not code_blocks:
        append += f"""\"\"\"FFI API bindings for {module_name}.\"\"\"\n"""
    if not any(code.kind == "import-section" for code in code_blocks):
        append += C.PROMPT_IMPORT_SECTION

    # Part 1. Library loading
    if is_root:
        append += C._prompt_import_object("tvm_ffi.libinfo.load_lib_module", "_FFI_LOAD_LIB")
        append += f"""LIB = _FFI_LOAD_LIB("{init_cfg.pkg}", "{init_cfg.shared_target}")\n"""

    # Part 2. Global functions
    if not any(code.kind == "global" for code in code_blocks):
        append += C._prompt_globals(module_name)

    # Part 3. Object types
    if object_infos:
        append += C._prompt_import_object("tvm_ffi.register_object", "_FFI_REG_OBJ")

    defined_type_keys = {info.type_key for info in object_infos if info.type_key}
    for info in object_infos:
        type_key = info.type_key
        parent_type_key = info.parent_type_key
        if type_key is None:
            continue
        # Canonicalize type key names
        type_key = ty_map.get(type_key, type_key)
        type_name = type_key.rsplit(".", 1)[-1]
        parent_type_key = (
            ty_map.get(parent_type_key, parent_type_key) if parent_type_key else parent_type_key
        )
        parent_type_name = parent_type_key.rsplit(".", 1)[-1] if parent_type_key else "Object"
        # Import parent type keys if they are not defined in the current module
        if parent_type_key and parent_type_key not in defined_type_keys:
            parent_type_name = "_" + parent_type_key.replace(".", "_")
            append += C._prompt_import_object(parent_type_key, parent_type_name)
        # Generate class definition
        append += C._prompt_class_def(
            type_name,
            type_key,
            parent_type_name,
        )
    # Part 4. __all__
    if not any(code.kind == "__all__" for code in code_blocks):
        append += C.PROMPT_ALL_SECTION
    return append


def generate_init(
    code_blocks: list[CodeBlock],
    module_name: str,
    submodule: str = "_ffi_api",
) -> str:
    """Generate the `__init__.py` file for the `tvm_ffi` package."""
    code = f"""
{C.STUB_BEGIN} export/{submodule}
{C.STUB_END}
"""
    if not code_blocks:
        return f"""\"\"\"Package {module_name}.\"\"\"\n""" + code
    if not any(code.kind == "export" for code in code_blocks):
        return code
    return ""
