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
"""Common utilities for the `tvm-ffi-stubgen` tool."""

from __future__ import annotations

import dataclasses
from io import StringIO
from typing import Callable

from tvm_ffi.core import TypeInfo, TypeSchema

from . import consts as C


@dataclasses.dataclass
class InitConfig:
    """Configuration for generating new stubs.

    Examples
    --------
    If we are generating type stubs for Python package `my-ffi-extension`,
    and the CMake target that generates the shared library is `my_ffi_extension_shared`,
    then we can run the following command to generate the stubs:

    --init-pypkg my-ffi-extension --init-lib my_ffi_extension_shared --init-prefix my_ffi_extension.

    """

    pkg: str
    """Name of the Python package to generate stubs for, e.g. apache-tvm-ffi (instead of tvm_ffi)"""

    shared_target: str
    """Name of CMake target that generates the shared library, e.g. tvm_ffi_shared

    This is used to determine the name of the shared library file.
    - macOS: lib{shared_target}.dylib or lib{shared_target}.so
    - Linux: lib{shared_target}.so
    - Windows: {shared_target}.dll
    """

    prefix: str
    """Only generate stubs for global function and objects with the given prefix, e.g. `tvm_ffi.`"""


@dataclasses.dataclass
class Options:
    """Command line options for stub generation."""

    imports: list[str] = dataclasses.field(default_factory=list)
    dlls: list[str] = dataclasses.field(default_factory=list)
    init: InitConfig | None = None
    indent: int = 4
    files: list[str] = dataclasses.field(default_factory=list)
    verbose: bool = False
    dry_run: bool = False


@dataclasses.dataclass(frozen=True, eq=True)
class ImportItem:
    """An import statement item."""

    mod: str
    name: str
    type_checking_only: bool = False
    alias: str | None = None

    def __init__(
        self,
        name: str,
        type_checking_only: bool = False,
        alias: str | None = None,
    ) -> None:
        """Initialize an `ImportItem` with the given module name and optional alias."""
        if "." in name:
            mod, name = name.rsplit(".", 1)
            for mod_prefix, mod_replacement in C.MOD_MAP.items():
                if mod.startswith(mod_prefix):
                    mod = mod.replace(mod_prefix, mod_replacement, 1)
                    break
        else:
            mod = ""
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


@dataclasses.dataclass(init=False)
class NamedTypeSchema(TypeSchema):
    """A type schema with an associated name."""

    name: str

    def __init__(self, name: str, schema: TypeSchema) -> None:
        """Initialize a `NamedTypeSchema` with the given name and type schema."""
        super().__init__(origin=schema.origin, args=schema.args)
        self.name = name


@dataclasses.dataclass
class FuncInfo:
    """Information of a function."""

    schema: NamedTypeSchema
    is_member: bool

    @staticmethod
    def from_schema(name: str, schema: TypeSchema, *, is_member: bool = False) -> FuncInfo:
        """Construct a `FuncInfo` from a name and its type schema."""
        return FuncInfo(schema=NamedTypeSchema(name=name, schema=schema), is_member=is_member)

    def gen(self, ty_map: Callable[[str], str], indent: int) -> str:
        """Generate a function signature string for this function."""
        try:
            _, func_name = self.schema.name.rsplit(".", 1)
        except ValueError:
            func_name = self.schema.name
        buf = StringIO()
        buf.write(" " * indent)
        buf.write(f"def {func_name}(")
        if self.schema.origin != "Callable":
            raise ValueError(f"Expected Callable type schema, but got: {self.schema}")
        if not self.schema.args:
            ty_map("Any")
            buf.write("*args: Any) -> Any: ...")
            return buf.getvalue()
        arg_ret = self.schema.args[0]
        arg_args = self.schema.args[1:]
        for i, arg in enumerate(arg_args):
            if self.is_member and i == 0:
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


@dataclasses.dataclass
class ObjectInfo:
    """Information of an object type, including its fields and methods."""

    fields: list[NamedTypeSchema]
    methods: list[FuncInfo]
    type_key: str | None = None
    parent_type_key: str | None = None

    @staticmethod
    def from_type_info(type_info: TypeInfo) -> ObjectInfo:
        """Construct an `ObjectInfo` from a `TypeInfo` instance."""
        parent_type_key: str | None = None
        if type_info.parent_type_info is not None:
            parent_type_key = type_info.parent_type_info.type_key
        return ObjectInfo(
            fields=[
                NamedTypeSchema(
                    name=field.name,
                    schema=TypeSchema.from_json_str(field.metadata["type_schema"]),
                )
                for field in type_info.fields
            ],
            methods=[
                FuncInfo(
                    schema=NamedTypeSchema(
                        name=C.FN_NAME_MAP.get(method.name, method.name),
                        schema=TypeSchema.from_json_str(method.metadata["type_schema"]),
                    ),
                    is_member=not method.is_static,
                )
                for method in type_info.methods
            ],
            type_key=type_info.type_key,
            parent_type_key=parent_type_key,
        )

    def gen_fields(self, ty_map: Callable[[str], str], indent: int) -> list[str]:
        """Generate field definitions for this object."""
        indent_str = " " * indent
        return [f"{indent_str}{field.name}: {field.repr(ty_map)}" for field in self.fields]

    def gen_methods(self, ty_map: Callable[[str], str], indent: int) -> list[str]:
        """Generate method definitions for this object."""
        indent_str = " " * indent
        ret = []
        for method in self.methods:
            if not method.is_member:
                ret.append(f"{indent_str}@staticmethod")
            ret.append(method.gen(ty_map, indent))
        return ret
