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
"""Language-agnostic data model for the `tvm-ffi-stubgen` tool.

These dataclasses describe the FFI reflection metadata (functions, object
fields/methods, init signatures) without committing to any target language.
Turning this metadata into source text is the job of a target language
generator (e.g. :mod:`tvm_ffi.stub.python_generator.codegen`).
"""

from __future__ import annotations

import dataclasses
from typing import Any

from tvm_ffi.core import TypeInfo, TypeSchema, _lookup_type_attr

from . import consts as C


def _parse_type_schema(raw: str | dict[str, Any]) -> TypeSchema:
    """Parse a type schema from either a JSON string or an already-parsed dict."""
    if isinstance(raw, dict):
        return TypeSchema.from_json_obj(raw)
    return TypeSchema.from_json_str(raw)


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
    target: str = "python"
    """Code generator target to use."""


@dataclasses.dataclass(init=False)
class NamedTypeSchema(TypeSchema):
    """A type schema with an associated name."""

    name: str

    def __init__(self, name: str, schema: TypeSchema) -> None:
        """Initialize a `NamedTypeSchema` with the given name and schema."""
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


@dataclasses.dataclass
class InitFieldInfo:
    """A field that participates in the auto-generated ``__init__``."""

    name: str
    schema: NamedTypeSchema
    kw_only: bool
    has_default: bool


@dataclasses.dataclass
class ObjectInfo:
    """Information of an object type, including its fields and methods."""

    fields: list[NamedTypeSchema]
    methods: list[FuncInfo]
    type_key: str | None = None
    parent_type_key: str | None = None
    init_fields: list[InitFieldInfo] = dataclasses.field(default_factory=list)
    has_init: bool = False

    @staticmethod
    def from_type_info(type_info: TypeInfo) -> ObjectInfo:
        """Construct an `ObjectInfo` from a `TypeInfo` instance."""
        parent_type_key: str | None = None
        if type_info.parent_type_info is not None:
            parent_type_key = type_info.parent_type_info.type_key

        # Detect __ffi_init__ from TypeMethod or TypeAttrColumn.
        has_init = any(m.name == "__ffi_init__" for m in type_info.methods)
        if not has_init:
            has_init = _lookup_type_attr(type_info.type_index, "__ffi_init__") is not None

        # Walk parent chain (parent-first) to collect all init-eligible fields.
        init_fields: list[InitFieldInfo] = []
        if has_init:
            ti: TypeInfo | None = type_info
            chain: list[TypeInfo] = []
            while ti is not None:
                chain.append(ti)
                ti = ti.parent_type_info
            for ancestor_info in reversed(chain):
                for field in ancestor_info.fields:
                    if not field.c_init:
                        continue
                    init_fields.append(
                        InitFieldInfo(
                            name=field.name,
                            schema=NamedTypeSchema(
                                name=field.name,
                                schema=_parse_type_schema(field.metadata["type_schema"]),
                            ),
                            kw_only=field.c_kw_only,
                            has_default=field.c_has_default,
                        )
                    )

        return ObjectInfo(
            fields=[
                NamedTypeSchema(
                    name=field.name,
                    schema=_parse_type_schema(field.metadata["type_schema"]),
                )
                for field in type_info.fields
            ],
            methods=[
                FuncInfo(
                    schema=NamedTypeSchema(
                        name=C.FN_NAME_MAP.get(method.name, method.name),
                        schema=_parse_type_schema(method.metadata["type_schema"]),
                    ),
                    is_member=not method.is_static,
                )
                for method in type_info.methods
            ],
            type_key=type_info.type_key,
            parent_type_key=parent_type_key,
            init_fields=init_fields,
            has_init=has_init,
        )
