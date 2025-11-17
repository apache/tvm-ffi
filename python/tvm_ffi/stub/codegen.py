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
    code: CodeBlock, global_funcs: list[FuncInfo], fn_ty_map: Callable[[str], str], opt: Options
) -> None:
    """Generate function signatures for global functions."""
    assert len(code.lines) >= 2
    if not global_funcs:
        return
    results: list[str] = [
        "# fmt: off",
        "if TYPE_CHECKING:",
        *[
            func.gen(
                fn_ty_map,
                indent=opt.indent,
            )
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


def generate_object(code: CodeBlock, fn_ty_map: Callable[[str], str], opt: Options) -> None:
    """Generate a class definition for an object type."""
    assert len(code.lines) >= 2
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
    if not ty_collected:
        return

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
