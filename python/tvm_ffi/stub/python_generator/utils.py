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
"""Python generator helpers for ``tvm-ffi-stubgen``.

This module groups two Python-specific concerns:

- import modelling (:class:`ImportItem`, :class:`PythonImports`)
- stub rendering helpers for function/object signatures
"""

from __future__ import annotations

import dataclasses
from io import StringIO
from typing import Callable

from ..utils import FuncInfo, ObjectInfo
from . import consts as C


@dataclasses.dataclass(frozen=True, eq=True)
class ImportItem:
    """An import statement item."""

    mod: str
    name: str
    type_checking_only: bool = False
    alias: str | None = None

    def __init__(
        self,
        full_name: str,
        type_checking_only: bool = False,
        alias: str | None = None,
    ) -> None:
        """Initialize an `ImportItem` from a dotted ``module.symbol`` name and optional alias."""
        if "." in full_name:
            mod, name = full_name.rsplit(".", 1)
            for mod_prefix, mod_replacement in C.MOD_MAP.items():
                if mod == mod_prefix or mod.startswith(mod_prefix + "."):
                    mod = mod.replace(mod_prefix, mod_replacement, 1)
                    break
        else:
            mod, name = "", full_name
        object.__setattr__(self, "mod", mod)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "type_checking_only", type_checking_only)
        object.__setattr__(self, "alias", alias)

    @property
    def name_with_alias(self) -> str:
        """Generate a string of the form `name as alias` if an alias is set, otherwise just `name`."""
        return f"{self.name} as {self.alias}" if self.alias else self.name

    @property
    def full_name(self) -> str:
        """Generate a string of the form `mod.name` or `name` if no module is set."""
        return f"{self.mod}.{self.name}" if self.mod else self.name

    def __repr__(self) -> str:
        """Generate an import statement string for this item."""
        return str(self)

    def __str__(self) -> str:
        """Generate an import statement string for this item."""
        if self.mod:
            ret = f"from {self.mod} import {self.name_with_alias}"
        else:
            ret = f"import {self.name_with_alias}"
        return ret


@dataclasses.dataclass
class PythonImports:
    """Opaque import collector threaded through the Python generation pipeline.

    The language-agnostic ``cli`` treats this as an opaque token: it asks the
    generator to create one, seed it from ``import-object`` directives, and later
    render it. Only the Python generator reaches inside.
    """

    items: list[ImportItem] = dataclasses.field(default_factory=list)
    has_lib_load: bool = False
    """Whether an FFI library-loading import was seen (adds ``LIB`` to ``__all__``)."""


def render_func_signature(
    func: FuncInfo,
    ty_map: Callable[[str], str],
    indent: int,
    input_ty_map: Callable[[str], str] | None = None,
) -> str:
    """Render a function signature string for ``func``."""
    if input_ty_map is None:
        input_ty_map = ty_map
    func_name = func.schema.name.rsplit(".", 1)[-1]
    buf = StringIO()
    buf.write(" " * indent)
    buf.write(f"def {func_name}(")
    if func.schema.origin != "Callable":
        raise ValueError(f"Expected Callable type schema, but got: {func.schema}")
    if not func.schema.args:
        ty_map("Any")
        buf.write("*args: Any) -> Any: ...")
        return buf.getvalue()
    arg_ret = func.schema.args[0]
    arg_args = func.schema.args[1:]
    for i, arg in enumerate(arg_args):
        if func.is_member and i == 0:
            buf.write("self, ")
        else:
            buf.write(f"_{i}: ")
            buf.write(arg.input_repr(input_ty_map))
            buf.write(", ")
    if arg_args:
        buf.write("/")
    buf.write(") -> ")
    buf.write(arg_ret.output_repr(ty_map))
    buf.write(": ...")
    return buf.getvalue()


def render_object_fields(
    info: ObjectInfo,
    ty_map: Callable[[str], str],
    indent: int,
) -> list[str]:
    """Render field definitions for ``info``."""
    indent_str = " " * indent
    return [f"{indent_str}{field.name}: {field.output_repr(ty_map)}" for field in info.fields]


def render_object_methods(
    info: ObjectInfo,
    ty_map: Callable[[str], str],
    indent: int,
    input_ty_map: Callable[[str], str] | None = None,
) -> list[str]:
    """Render method definitions for ``info``."""
    if input_ty_map is None:
        input_ty_map = ty_map
    indent_str = " " * indent
    ret = []
    groups: list[list[FuncInfo]] = []
    seen: dict[tuple[str, bool], list[FuncInfo]] = {}
    for method in info.methods:
        key = (method.schema.name, method.is_member)
        if key not in seen:
            group: list[FuncInfo] = []
            groups.append(group)
            seen[key] = group
        seen[key].append(method)

    for group in groups:
        for candidate in group:
            func_name = candidate.schema.name.rsplit(".", 1)[-1]
            if len(group) > 1:
                ret.append(f"{indent_str}@overload")
            if func_name == "__ffi_init__":
                # __ffi_init__ is installed as an instance method (self, *args, **kwargs) -> None
                # by _install_ffi_init_attr, regardless of the C++ static registration.
                ret.append(_render_ffi_init_from_method(candidate, ty_map, indent, input_ty_map))
                continue
            if not candidate.is_member:
                ret.append(f"{indent_str}@staticmethod")
            ret.append(render_func_signature(candidate, ty_map, indent, input_ty_map))
    return ret


def _render_ffi_init_from_method(
    method: FuncInfo,
    ty_map: Callable[[str], str],
    indent: int,
    input_ty_map: Callable[[str], str] | None = None,
) -> str:
    """Render ``__ffi_init__`` TypeMethod as an instance method returning None."""
    if input_ty_map is None:
        input_ty_map = ty_map
    indent_str = " " * indent
    schema = method.schema
    if schema.origin != "Callable" or not schema.args:
        ty_map("Any")
        return f"{indent_str}def __ffi_init__(self, *args: Any) -> None: ..."
    # schema.args[0] is return type, schema.args[1:] are param types.
    parts: list[str] = []
    for i, arg in enumerate(schema.args[1:]):
        parts.append(f"_{i}: {arg.input_repr(input_ty_map)}")
    if parts:
        params = ", ".join(parts)
        return f"{indent_str}def __ffi_init__(self, {params}, /) -> None: ..."
    return f"{indent_str}def __ffi_init__(self) -> None: ..."


def render_object_ffi_init(
    info: ObjectInfo,
    ty_map: Callable[[str], str],
    indent: int,
    input_ty_map: Callable[[str], str] | None = None,
) -> list[str]:
    """Render a ``__ffi_init__`` stub when it's not already in TypeMethod.

    For types whose ``__ffi_init__`` is auto-generated by ``RegisterFFIInit``
    (TypeAttrColumn only), synthesize a static-method stub from field metadata.
    Types that already have ``__ffi_init__`` in TypeMethod (from explicit
    ``refl::init<>``) get it via ``render_object_methods`` instead.
    """
    if not info.has_init:
        return []
    # If __ffi_init__ is already in methods (from TypeMethod), methods render it.
    if any(m.schema.name.rsplit(".", 1)[-1] == "__ffi_init__" for m in info.methods):
        return []
    return _render_ffi_init_from_fields(info, ty_map, indent, input_ty_map)


def render_object_init(
    info: ObjectInfo,
    ty_map: Callable[[str], str],
    indent: int,
    input_ty_map: Callable[[str], str] | None = None,
) -> list[str]:
    """Render an ``__init__`` stub from init-eligible field metadata."""
    if not info.has_init:
        return []
    return _render_init_from_fields(info, ty_map, indent, input_ty_map)


def _format_field_params(
    info: ObjectInfo,
    ty_map: Callable[[str], str],
    input_ty_map: Callable[[str], str] | None = None,
) -> str:
    """Format init-eligible fields as a parameter string with defaults and kw_only."""
    if input_ty_map is None:
        input_ty_map = ty_map
    positional = [f for f in info.init_fields if not f.kw_only]
    kw_only = [f for f in info.init_fields if f.kw_only]

    pos_required = [f for f in positional if not f.has_default]
    pos_default = [f for f in positional if f.has_default]
    kw_required = [f for f in kw_only if not f.has_default]
    kw_default = [f for f in kw_only if f.has_default]

    parts: list[str] = []
    for f in pos_required:
        parts.append(f"{f.name}: {f.schema.input_repr(input_ty_map)}")
    for f in pos_default:
        parts.append(f"{f.name}: {f.schema.input_repr(input_ty_map)} = ...")
    if kw_required or kw_default:
        parts.append("*")
        for f in kw_required:
            parts.append(f"{f.name}: {f.schema.input_repr(input_ty_map)}")
        for f in kw_default:
            parts.append(f"{f.name}: {f.schema.input_repr(input_ty_map)} = ...")

    return ", ".join(parts)


def _render_init_from_fields(
    info: ObjectInfo,
    ty_map: Callable[[str], str],
    indent: int,
    input_ty_map: Callable[[str], str] | None = None,
) -> list[str]:
    """Render ``__init__`` from init-eligible field metadata (auto-generated init)."""
    indent_str = " " * indent
    params = _format_field_params(info, ty_map, input_ty_map)
    if params:
        return [f"{indent_str}def __init__(self, {params}) -> None: ..."]
    return [f"{indent_str}def __init__(self) -> None: ..."]


def _render_ffi_init_from_fields(
    info: ObjectInfo,
    ty_map: Callable[[str], str],
    indent: int,
    input_ty_map: Callable[[str], str] | None = None,
) -> list[str]:
    """Render ``__ffi_init__`` stub from field metadata for auto-generated init."""
    indent_str = " " * indent
    params = _format_field_params(info, ty_map, input_ty_map)
    if params:
        return [f"{indent_str}def __ffi_init__(self, {params}) -> None: ..."]
    return [f"{indent_str}def __ffi_init__(self) -> None: ..."]
