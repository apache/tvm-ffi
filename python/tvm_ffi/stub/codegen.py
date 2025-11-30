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
from .utils import FuncInfo, ObjectInfo, Options


def generate_global_funcs(
    code: CodeBlock,
    global_funcs: list[FuncInfo],
    fn_ty_map: Callable[[str], str],
    opt: Options,
) -> None:
    """Generate function signatures for global functions."""
    assert len(code.lines) >= 2
    if not global_funcs:
        return
    assert isinstance(code.param, tuple)
    prefix, import_from = code.param
    if not import_from:
        import_from = "tvm_ffi"
    results: list[str] = [
        "# fmt: off",
        "# isort: off",
        f"from {import_from} import init_ffi_api as _INIT",
        f'_INIT("{prefix}", __name__)',
        "# isort: on",
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


def generate_object(code: CodeBlock, fn_ty_map: Callable[[str], str], opt: Options) -> None:
    """Generate a class definition for an object type."""
    assert len(code.lines) >= 2
    assert isinstance(code.param, str)
    info = ObjectInfo.from_type_key(code.param)
    if info.methods:
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


def generate_imports(code: CodeBlock, ty_used: set[str], opt: Options) -> None:
    """Generate import statements for the types used in the stub."""
    ty_collected: dict[str, list[str]] = {}
    for ty in ty_used:
        assert "." in ty
        module, name = ty.rsplit(".", 1)
        for mod_prefix, mod_replacement in C.MOD_MAP.items():
            if module.startswith(mod_prefix):
                module = module.replace(mod_prefix, mod_replacement, 1)
                break
        ty_collected.setdefault(module, []).append(name)

    def _make_line(module: str, names: list[str], indent: int) -> str:
        names = ", ".join(sorted(set(names)))
        indent_str = " " * indent
        return f"{indent_str}from {module} import {names}"

    results: list[str] = [
        "from __future__ import annotations",
        _make_line(
            "typing",
            [*ty_collected.pop("typing", []), "TYPE_CHECKING"],
            indent=0,
        ),
    ]
    if ty_collected:
        results.append("if TYPE_CHECKING:")
        for module in sorted(ty_collected):
            names = ty_collected[module]
            results.append(_make_line(module, names, indent=opt.indent))
    if results:
        code.lines = [
            code.lines[0],
            "# fmt: off",
            "# isort: off",
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
    code.lines = [
        code.lines[0],
        *[f'{indent}"{name}",' for name in sorted(names)],
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
        'if "__all__" not in globals(): __all__ = []',
        f"__all__.extend({mod}__all__)",
        "# isort: on",
        "# fmt: on",
        code.lines[-1],
    ]


def generate_ffi_api(
    code_blocks: list[CodeBlock],
    module_name: str,
    type_keys: list[str],
) -> str:
    """Generate the initial FFI API stub code for a given module."""
    append = ""
    if not code_blocks:
        append += f"""\"\"\"FFI API bindings for {module_name}.\"\"\"
"""
    # Part 1. Imports
    if not any(code.kind == "import" for code in code_blocks):
        append += f"""
{C.STUB_BEGIN} import
{C.STUB_END}
"""
    # Part 2. Global functions
    if not any(code.kind == "global" for code in code_blocks):
        append += f"""
{C.STUB_BEGIN} global/{module_name}
{C.STUB_END}
"""
    # Part 3. __all__
    if not any(code.kind == "all" for code in code_blocks):
        append += f"""
__all__ = [
    {C.STUB_BEGIN} __all__
    {C.STUB_END}
]
"""
    # Part 4. Object types
    if type_keys:
        append += """

# isort: off
import tvm_ffi
# isort: on

"""
    for type_key in sorted(type_keys):
        type_cls_name = type_key.rsplit(".", 1)[-1]
        append += f"""
@tvm_ffi.register_object("{type_key}")
class {type_cls_name}(tvm_ffi.Object):
    \"\"\"FFI binding for `{type_key}`.\"\"\"

    {C.STUB_BEGIN} object/{type_key}
    {C.STUB_END}
"""
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
