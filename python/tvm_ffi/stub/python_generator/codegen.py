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
"""Python code generation for the ``tvm-ffi-stubgen`` tool.

This module owns the Python codegen orchestration for language-agnostic FFI
metadata (:class:`tvm_ffi.stub.utils.FuncInfo` /
:class:`~tvm_ffi.stub.utils.ObjectInfo`). Rendering helpers live in
``python_generator.utils`` so the per-block generation pipeline here stays focused
on directive handling and source assembly.
"""

from __future__ import annotations

from typing import Callable

from .. import consts as C
from ..file_utils import CodeBlock
from ..utils import FuncInfo, InitConfig, ObjectInfo, Options
from . import consts as PC
from .utils import (
    ImportItem,
    render_func_signature,
    render_object_ffi_init,
    render_object_fields,
    render_object_init,
    render_object_methods,
)

# --- Python scaffolding templates (init mode) -------------------------------
# These emit Python source plus stub-directive markers. The marker comment token
# comes from the supplied `MarkerSyntax`, so the directive structure stays
# language-aware even though the surrounding code is Python-specific.


def _prompt_globals(mod: str, syntax: C.MarkerSyntax) -> str:
    return f"""{syntax.begin} global/{mod}
{syntax.end}
"""


def _prompt_class_def(
    type_name: str, type_key: str, parent_type_name: str, syntax: C.MarkerSyntax
) -> str:
    return f'''@_FFI_REG_OBJ("{type_key}")
class {type_name}({parent_type_name}):
    """FFI binding for `{type_key}`."""

    {syntax.begin} object/{type_key}
    {syntax.end}\n\n'''


def _prompt_import_object(type_key: str, type_name: str, syntax: C.MarkerSyntax) -> str:
    return f"""{syntax.import_object} {type_key};False;{type_name}\n"""


def _prompt_import_section(syntax: C.MarkerSyntax) -> str:
    return f"""
{syntax.begin} import-section
{syntax.end}
"""


def _prompt_all_section(syntax: C.MarkerSyntax) -> str:
    return f"""
__all__ = [
    {syntax.begin} __all__
    {syntax.end}
]
"""


def _type_suffix_and_record(
    ty_map: dict[str, str],
    imports: list[ImportItem],
    func_names: set[str] | None = None,
) -> Callable[[str], str]:
    def _run(name: str) -> str:
        nonlocal ty_map, imports
        name = ty_map.get(name, name)
        suffix = name.rsplit(".", 1)[-1]
        if "." in name:
            alias = None
            if func_names and suffix in func_names:
                alias = f"_{suffix}"
            imports.append(ImportItem(name, type_checking_only=True, alias=alias))
            if alias:
                return alias
        return suffix

    return _run


def _make_input_ty_map(ty_map: dict[str, str]) -> dict[str, str]:
    """Derive input-side defaults without overriding explicit ty-map entries."""
    input_ty_map = ty_map.copy()
    for key, input_default in PC.TY_MAP_INPUT_DEFAULTS.items():
        output_default = PC.TY_MAP_DEFAULTS.get(key)
        if ty_map.get(key, output_default) == output_default:
            input_ty_map[key] = input_default
    return input_ty_map


def generate_python_global_funcs(
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
    func_names = {f.schema.name.rsplit(".", 1)[-1] for f in global_funcs}
    fn_ty_map = _type_suffix_and_record(ty_map, imports, func_names=func_names)
    input_fn_ty_map = _type_suffix_and_record(
        _make_input_ty_map(ty_map), imports, func_names=func_names
    )
    results: list[str] = [
        "# fmt: off",
        f'_FFI_INIT_FUNC("{prefix}", __name__)',
        "if TYPE_CHECKING:",
        *[
            render_func_signature(func, fn_ty_map, opt.indent, input_ty_map=input_fn_ty_map)
            for func in global_funcs
        ],
        "# fmt: on",
    ]
    indent = " " * code.indent
    code.lines = [
        code.lines[0],
        *[indent + line for line in results],
        code.lines[-1],
    ]


def generate_python_object(
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
    method_names = {m.schema.name.rsplit(".", 1)[-1] for m in info.methods}
    fn_ty_map = _type_suffix_and_record(ty_map, imports, func_names=method_names)
    input_fn_ty_map = _type_suffix_and_record(
        _make_input_ty_map(ty_map), imports, func_names=method_names
    )
    init_lines = render_object_init(info, fn_ty_map, opt.indent, input_ty_map=input_fn_ty_map)
    ffi_init_lines = render_object_ffi_init(
        info, fn_ty_map, opt.indent, input_ty_map=input_fn_ty_map
    )
    type_checking_lines = [
        *init_lines,
        *ffi_init_lines,
        *render_object_methods(info, fn_ty_map, opt.indent, input_ty_map=input_fn_ty_map),
    ]
    if type_checking_lines:
        imports.append(
            ImportItem(
                "typing.TYPE_CHECKING",
                type_checking_only=False,
            )
        )
        results = [
            "# fmt: off",
            *render_object_fields(info, fn_ty_map, 0),
            "if TYPE_CHECKING:",
            *type_checking_lines,
            "# fmt: on",
        ]
    else:
        results = [
            "# fmt: off",
            *render_object_fields(info, fn_ty_map, 0),
            "# fmt: on",
        ]
    indent = " " * code.indent
    code.lines = [
        code.lines[0],
        *[indent + line for line in results],
        code.lines[-1],
    ]


def generate_python_import_section(
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


def generate_python_all(code: CodeBlock, names: set[str], opt: Options) -> None:
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


def generate_python_export(code: CodeBlock) -> None:
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


def generate_python_ffi_api(
    code_blocks: list[CodeBlock],
    ty_map: dict[str, str],
    module_name: str,
    object_infos: list[ObjectInfo],
    init_cfg: InitConfig,
    is_root: bool,
    syntax: C.MarkerSyntax,
) -> str:
    """Generate the initial FFI API stub code for a given module."""
    # TODO(@junrus): New code is appended to the end of the file.
    # We should consider a more sophisticated approach.
    append = ""

    # Part 0. Imports
    if not code_blocks:
        append += f"""\"\"\"FFI API bindings for {module_name}.\"\"\"\n"""
    if not any(code.kind == "import-section" for code in code_blocks):
        append += _prompt_import_section(syntax)

    # Part 1. Library loading
    if is_root:
        append += _prompt_import_object("tvm_ffi.libinfo.load_lib_module", "_FFI_LOAD_LIB", syntax)
        append += f"""LIB = _FFI_LOAD_LIB("{init_cfg.pkg}", "{init_cfg.shared_target}")\n"""

    # Part 2. Global functions
    if not any(code.kind == "global" for code in code_blocks):
        append += _prompt_globals(module_name, syntax)

    # Part 3. Object types
    if object_infos:
        append += _prompt_import_object("tvm_ffi.register_object", "_FFI_REG_OBJ", syntax)

    defined_type_keys = {
        ty_map.get(info.type_key, info.type_key) for info in object_infos if info.type_key
    }
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
            append += _prompt_import_object(parent_type_key, parent_type_name, syntax)
        # Generate class definition
        append += _prompt_class_def(
            type_name,
            type_key,
            parent_type_name,
            syntax,
        )
    # Part 4. __all__
    if not any(code.kind == "__all__" for code in code_blocks):
        append += _prompt_all_section(syntax)
    return append


def generate_python_init(
    code_blocks: list[CodeBlock],
    module_name: str,
    submodule: str,
    syntax: C.MarkerSyntax,
) -> str:
    """Generate the `__init__.py` file for the `tvm_ffi` package."""
    code = f"""
{syntax.begin} export/{submodule}
{syntax.end}
"""
    if not code_blocks:
        return f"""\"\"\"Package {module_name}.\"\"\"\n""" + code
    if not any(code.kind == "export" for code in code_blocks):
        return code
    return ""
