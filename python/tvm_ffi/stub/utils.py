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
from typing import TYPE_CHECKING, Any

from tvm_ffi.core import MISSING, TypeInfo, TypeSchema, _lookup_type_attr

from . import consts as C

if TYPE_CHECKING:
    from tvm_ffi.core import TypeField


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
    """A type schema with an associated name.

    For a reflected object field, the schema also carries the facts the C++
    registry recorded about the native field, so a generator can reason about
    memory layout and defaults without a second reflection pass:

    - ``size`` / ``alignment`` / ``offset``: the byte facts of the native field.
      ``offset`` is measured from the start of the ``TVMFFIObject`` header, the
      same base the C ABI field getters use. All three are ``None`` for function
      parameters and synthetic schemas.
    - ``default``: the registered static default value (:data:`MISSING` when
      none). ``default_is_factory`` marks a ``default_factory`` registration,
      whose value only exists by calling the factory through the FFI.
    - ``structural_eq``: the decoded structural-equality flag
      (``"ignore"``, ``"def-recursive"``, ``"def-non-recursive"`` or ``None``).
    - ``frozen``: ``True`` for read-only (``def_ro``) fields.
    """

    name: str
    size: int | None = None
    alignment: int | None = None
    offset: int | None = None
    default: Any = MISSING
    default_is_factory: bool = False
    structural_eq: str | None = None
    frozen: bool = False

    def __init__(
        self,
        name: str,
        schema: TypeSchema,
        *,
        size: int | None = None,
        alignment: int | None = None,
        offset: int | None = None,
        default: Any = MISSING,
        default_is_factory: bool = False,
        structural_eq: str | None = None,
        frozen: bool = False,
    ) -> None:
        """Initialize a `NamedTypeSchema` with the given name, schema and field facts."""
        super().__init__(origin=schema.origin, args=schema.args)
        self.name = name
        self.size = size
        self.alignment = alignment
        self.offset = offset
        self.default = default
        self.default_is_factory = default_is_factory
        self.structural_eq = structural_eq
        self.frozen = frozen

    @staticmethod
    def from_type_field(field: TypeField) -> NamedTypeSchema:
        """Construct a `NamedTypeSchema` from a reflected :class:`~tvm_ffi.core.TypeField`."""
        is_factory = field.c_default_factory is not MISSING
        return NamedTypeSchema(
            name=field.name,
            schema=_parse_type_schema(field.metadata["type_schema"]),
            size=field.size,
            alignment=field.alignment,
            offset=field.offset,
            default=field.c_default_factory if is_factory else field.c_default,
            default_is_factory=is_factory,
            structural_eq=field.c_structural_eq,
            frozen=field.frozen,
        )


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
    """Information of an object type, including its fields and methods.

    ``fields`` lists only the fields declared by this type; inherited fields are
    reached through ``parent_type_key`` / ``ancestors``. ``total_size`` is the
    native ``sizeof`` of the object (header included) when the type registered
    its own metadata, and ``None`` otherwise: a type without its own
    ``ObjectDef`` inherits its parent's metadata entry, whose size says nothing
    about the type itself.
    """

    fields: list[NamedTypeSchema]
    methods: list[FuncInfo]
    type_key: str | None = None
    parent_type_key: str | None = None
    init_fields: list[InitFieldInfo] = dataclasses.field(default_factory=list)
    has_init: bool = False
    ancestors: list[str] = dataclasses.field(default_factory=list)
    """Type keys of every ancestor, root first (``["ffi.Object", "ir.Expr"]`` for ``tirx.Add``)."""
    total_size: int | None = None
    """Native ``sizeof`` in bytes, or ``None`` when the type has no metadata of its own."""

    def has_overloaded_methods(self) -> bool:
        """Return whether reflection exposed multiple signatures for a method."""
        seen: set[tuple[str, bool]] = set()
        for method in self.methods:
            key = (method.schema.name, method.is_member)
            if key in seen:
                return True
            seen.add(key)
        return False

    @staticmethod
    def from_type_info(type_info: TypeInfo) -> ObjectInfo:
        """Construct an `ObjectInfo` from a `TypeInfo` instance."""
        # Ancestor chain, root first (`ancestor_infos[-1]` is the direct parent).
        ancestor_infos: list[TypeInfo] = []
        ancestor_info: TypeInfo | None = type_info.parent_type_info
        while ancestor_info is not None:
            ancestor_infos.append(ancestor_info)
            ancestor_info = ancestor_info.parent_type_info
        ancestor_infos.reverse()
        parent_type_key = ancestor_infos[-1].type_key if ancestor_infos else None

        # Detect __ffi_init__ from TypeMethod or TypeAttrColumn.
        has_init = any(m.name == "__ffi_init__" for m in type_info.methods)
        if not has_init:
            has_init = _lookup_type_attr(type_info.type_index, "__ffi_init__") is not None

        # Collect init-eligible fields from the whole chain, inherited fields first.
        init_fields: list[InitFieldInfo] = []
        if has_init:
            for declaring_info in [*ancestor_infos, type_info]:
                for field in declaring_info.fields:
                    if not field.c_init:
                        continue
                    init_fields.append(
                        InitFieldInfo(
                            name=field.name,
                            schema=NamedTypeSchema.from_type_field(field),
                            kw_only=field.c_kw_only,
                            has_default=field.c_has_default,
                        )
                    )

        return ObjectInfo(
            fields=[NamedTypeSchema.from_type_field(field) for field in type_info.fields],
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
            ancestors=[info.type_key for info in ancestor_infos],
            total_size=type_info.total_size if type_info._has_type_metadata else None,
        )
