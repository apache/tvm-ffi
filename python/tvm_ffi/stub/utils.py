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

from tvm_ffi.core import TypeSchema

from . import consts as C


@dataclasses.dataclass
class Options:
    """Command line options for stub generation."""

    dlls: list[str] = dataclasses.field(default_factory=list)
    indent: int = 4
    files: list[str] = dataclasses.field(default_factory=list)
    verbose: bool = False
    dry_run: bool = False


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
    def from_global_name(name: str) -> FuncInfo:
        """Construct a `FuncInfo` from a string name of this global function."""
        from tvm_ffi.registry import get_global_func_metadata  # noqa: PLC0415

        return FuncInfo(
            schema=NamedTypeSchema(
                name=name,
                schema=TypeSchema.from_json_str(get_global_func_metadata(name)["type_schema"]),
            ),
            is_member=False,
        )

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

    @staticmethod
    def from_type_key(type_key: str) -> ObjectInfo:
        """Construct an `ObjectInfo` from a type key."""
        from tvm_ffi.core import _lookup_or_register_type_info_from_type_key  # noqa: PLC0415

        type_info = _lookup_or_register_type_info_from_type_key(type_key)
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
