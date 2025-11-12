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

from io import StringIO
from typing import Callable

from tvm_ffi.core import TypeSchema, _lookup_or_register_type_info_from_type_key
from tvm_ffi.registry import get_global_func_metadata

from . import consts as C
from .file_utils import CodeBlock
from .utils import Options


def generate_func_signature(
    schema: TypeSchema,
    func_name: str,
    ty_map: Callable[[str], str],
    is_member: bool,
) -> str:
    """Generate a function signature string from a TypeSchema."""
    buf = StringIO()
    buf.write(f"def {func_name}(")
    if schema.origin != "Callable":
        raise ValueError(f"Expected Callable type schema, but got: {schema}")
    if not schema.args:
        ty_map("Any")
        buf.write("*args: Any) -> Any: ...")
        return buf.getvalue()
    arg_ret = schema.args[0]
    arg_args = schema.args[1:]
    for i, arg in enumerate(arg_args):
        if is_member and i == 0:
            buf.write("self, ")
        else:
            buf.write(f"_{i}: ")
            buf.write(arg.repr(ty_map))
            buf.write(", ")
    if arg_args:
        buf.write("/")
    buf.write(") -> ")
    buf.write(arg_ret.repr(ty_map))
    buf.write(": ...")
    return buf.getvalue()


def generate_global_funcs(
    code: CodeBlock,
    global_funcs: dict[str, list[str]],
    fn_ty_map: Callable[[str], str],
    opt: Options,
) -> None:
    """Generate function signatures for global functions."""
    assert len(code.lines) >= 2
    indent = " " * code.indent
    indent_long = " " * (code.indent + opt.indent)
    prefix = code.param
    results: list[str] = [
        generate_func_signature(
            TypeSchema.from_json_str(get_global_func_metadata(f"{prefix}.{name}")["type_schema"]),
            name,
            ty_map=fn_ty_map,
            is_member=False,
        )
        for name in global_funcs.get(prefix, [])
    ]
    if not results:
        return
    code.lines = [
        code.lines[0],
        f"{indent}# fmt: off",
        f"{indent}if TYPE_CHECKING:",
        *[indent_long + sig for sig in results],
        f"{indent}# fmt: on",
        code.lines[-1],
    ]


def generate_object(code: CodeBlock, fn_ty_map: Callable[[str], str], opt: Options) -> None:
    """Generate a class definition for an object type."""
    assert len(code.lines) >= 2
    type_key = code.param
    type_info = _lookup_or_register_type_info_from_type_key(type_key)
    indent = " " * code.indent
    indent_long = " " * (code.indent + opt.indent)

    fields: list[str] = []
    for field in type_info.fields:
        fields.append(
            f"{indent}{field.name}: "
            + TypeSchema.from_json_str(field.metadata["type_schema"]).repr(fn_ty_map)
        )

    methods: list[str] = []
    if type_info.methods:
        methods = [f"{indent}if TYPE_CHECKING:"]
    for method in type_info.methods:
        if method.is_static:
            methods.append(f"{indent_long}@staticmethod")
        methods.append(
            indent_long
            + generate_func_signature(
                TypeSchema.from_json_str(method.metadata["type_schema"]),
                {
                    "__ffi_init__": "__c_ffi_init__",
                }.get(method.name, method.name),
                fn_ty_map,
                is_member=not method.is_static,
            )
        )
    code.lines = [
        code.lines[0],
        f"{indent}# fmt: off",
        *fields,
        *methods,
        f"{indent}# fmt: on",
        code.lines[-1],
    ]


def generate_imports(
    code: CodeBlock,
    ty_used: set[str],
    opt: Options,
) -> None:
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
