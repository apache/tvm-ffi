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
"""Generate lookup-only C++ ABI views from TVM-FFI reflection data."""

from __future__ import annotations

import fnmatch
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import groupby
from typing import cast

from ..core import (
    TypeField,
    TypeInfo,
    TypeSchema,
    _lookup_or_register_type_info_from_type_key,
    _object_type_key_to_index,
)
from ..registry import get_registered_type_keys

_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_ESCAPE_PREFIX = "__ffi_escape_"
_OBJECT_SIZE = 24
_OBJECT_ALIGNMENT = 8
_OBJECT_TYPE_INDEX = 64
_DYNAMIC_OBJECT_TYPE_INDEX_BEGIN = 128


@dataclass(frozen=True)
class _CppName:
    namespaces: tuple[str, ...]
    object_name: str

    @property
    def qualified(self) -> str:
        return "::" + "::".join((*self.namespaces, self.object_name))


@dataclass(frozen=True)
class _BuiltinType:
    object_type: str
    value_type: str | None
    size: int
    alignment: int


@dataclass(frozen=True)
class _Carrier:
    cpp_type: str
    size: int
    alignment: int


_STATIC_CARRIERS = {
    -1: _Carrier("::tvm::ffi::Any", 16, 8),
    1: _Carrier("int64_t", 8, 8),
    2: _Carrier("bool", 1, 1),
    3: _Carrier("double", 8, 8),
    4: _Carrier("void*", 8, 8),
    5: _Carrier("DLDataType", 4, 2),
    6: _Carrier("DLDevice", 8, 4),
    _OBJECT_TYPE_INDEX: _Carrier("::tvm::ffi::ObjectPtr<::tvm::ffi::Object>", 8, 8),
}


def _static_carrier(type_index: int) -> _Carrier | None:
    carrier = _STATIC_CARRIERS.get(type_index)
    if carrier is None and type_index >= _OBJECT_TYPE_INDEX:
        return _Carrier("::tvm::ffi::ObjectPtr<::tvm::ffi::Object>", 8, 8)
    return carrier


@dataclass(frozen=True)
class _FieldModel:
    reflected_name: str
    member_name: str
    carrier: _Carrier
    offset: int


@dataclass(frozen=True)
class _ClassModel:
    info: TypeInfo
    cpp_name: _CppName
    base_cpp_type: str
    alignment: int
    total_size: int
    fields: tuple[_FieldModel, ...]


def _cpp_identifier(value: str) -> str:
    if _IDENTIFIER_RE.fullmatch(value) and not value.startswith(_ESCAPE_PREFIX):
        return value
    return _ESCAPE_PREFIX + value.encode("utf-8").hex()


def _cpp_string_literal(value: str) -> str:
    """Encode UTF-8 bytes without depending on the compiler execution charset."""
    chunks: list[str] = []
    for byte in value.encode("utf-8"):
        if byte == ord('"'):
            chunks.append(r"\"")
        elif byte == ord("\\"):
            chunks.append(r"\\")
        elif 0x20 <= byte <= 0x7E:
            chunks.append(chr(byte))
        else:
            chunks.append(f"\\{byte:03o}")
    return '"' + "".join(chunks) + '"'


def _cpp_name(type_key: str) -> _CppName:
    parts = type_key.split(".")
    return _CppName(
        namespaces=tuple(_cpp_identifier(part) for part in parts[:-1]),
        object_name=f"{_cpp_identifier(parts[-1])}Obj",
    )


def _cpp_name_sort_key(info: TypeInfo) -> tuple[tuple[str, ...], str, str]:
    name = _cpp_name(info.type_key)
    return name.namespaces, name.object_name, info.type_key


def _lineage(type_info: TypeInfo) -> list[TypeInfo]:
    result: list[TypeInfo] = []
    current = type_info
    while current is not None:
        result.append(current)
        current = current.parent_type_info
    result.reverse()
    return result


def _namespace_lines(
    blocks: Sequence[tuple[tuple[str, ...], list[str]]],
) -> list[str]:
    lines: list[str] = []
    for namespaces, group in groupby(blocks, key=lambda block: block[0]):
        if lines:
            lines.append("")
        body: list[str] = []
        for _, block in group:
            if body:
                body.append("")
            body.extend(block)
        lines.extend(f"namespace {namespace} {{" for namespace in namespaces)
        if namespaces:
            lines.append("")
        lines.extend(body)
        if namespaces:
            lines.append("")
        lines.extend(f"}}  // namespace {namespace}" for namespace in reversed(namespaces))
    if lines:
        lines.append("")
    return lines


def _validate_native_schema(schema: TypeSchema, raw: dict[str, object]) -> None:
    """Reject normalized aliases whose native bytes have different semantics."""
    origin = schema.origin
    raw_origin = raw["type"]
    canonical_origins = {
        "int": {"int"},
        "float": {"float"},
        "bool": {"bool"},
        "ctypes.c_void_p": {"void*"},
        "dtype": {"DataType"},
        "DataType": {"DataType"},
        "Device": {"Device"},
        "Any": {"Any"},
        "str": {"ffi.String", "ffi.SmallStr"},
        "bytes": {"ffi.Bytes", "ffi.SmallBytes"},
        "Callable": {"ffi.Function"},
        "Tensor": {"ffi.Tensor"},
    }
    allowed = canonical_origins.get(origin)
    if allowed is not None:
        if raw_origin not in allowed:
            raise ValueError(
                f"native schema origin {raw_origin!r} normalizes to {origin!r} but does not "
                "have the canonical owning representation"
            )
        return

    structural_origins = {
        "Array": "ffi.Array",
        "List": "ffi.List",
        "Map": "ffi.Map",
        "Dict": "ffi.Dict",
        "tuple": "Tuple",
        "Optional": "Optional",
        "Union": "Variant",
    }
    expected_raw_origin = structural_origins.get(origin)
    if expected_raw_origin is not None:
        if raw_origin != expected_raw_origin:
            raise ValueError(
                f"native schema origin {raw_origin!r} normalizes to {origin!r} but is not "
                f"the canonical {expected_raw_origin!r} carrier"
            )
        normalized_args = schema.args
        raw_args = raw.get("args", ())
        if not isinstance(raw_args, list) or len(raw_args) != len(normalized_args):
            raise ValueError(f"raw schema {raw!r} does not match normalized schema {schema!r}")
        for normalized_arg, raw_arg in zip(normalized_args, raw_args):
            if not isinstance(raw_arg, dict) or not isinstance(raw_arg.get("type"), str):
                raise ValueError(f"invalid nested raw schema {raw_arg!r}")
            _validate_native_schema(normalized_arg, cast(dict[str, object], raw_arg))
        return

    if schema.origin_type_index >= _OBJECT_TYPE_INDEX:
        expected = "ffi.Object" if origin == "Object" else origin
        if raw_origin != expected:
            raise ValueError(
                f"native object schema origin {raw_origin!r} does not match {expected!r}"
            )
        return
    raise ValueError(f"unsupported native raw schema {raw!r}")


def _builtin_table() -> dict[int, _BuiltinType]:
    specs = {
        "ffi.Object": _BuiltinType("::tvm::ffi::Object", None, 8, 8),
        "ffi.String": _BuiltinType("::tvm::ffi::details::StringObj", "::tvm::ffi::String", 16, 8),
        "ffi.Bytes": _BuiltinType("::tvm::ffi::details::BytesObj", "::tvm::ffi::Bytes", 16, 8),
        "ffi.Error": _BuiltinType("::tvm::ffi::ErrorObj", "::tvm::ffi::Error", 16, 8),
        "ffi.Function": _BuiltinType("::tvm::ffi::FunctionObj", "::tvm::ffi::Function", 8, 8),
        "ffi.Shape": _BuiltinType("::tvm::ffi::ShapeObj", "::tvm::ffi::Shape", 8, 8),
        "ffi.Tensor": _BuiltinType("::tvm::ffi::TensorObj", "::tvm::ffi::Tensor", 8, 8),
        "ffi.Array": _BuiltinType(
            "::tvm::ffi::ArrayObj", "::tvm::ffi::Array<::tvm::ffi::Any>", 8, 8
        ),
        "ffi.Map": _BuiltinType(
            "::tvm::ffi::MapObj", "::tvm::ffi::Map<::tvm::ffi::Any, ::tvm::ffi::Any>", 8, 8
        ),
        "ffi.Module": _BuiltinType(
            "::tvm::ffi::ModuleObj", "::tvm::ffi::ObjectPtr<::tvm::ffi::ModuleObj>", 8, 8
        ),
        # OpaquePyObject intentionally has no public C++ owning wrapper.
        "ffi.List": _BuiltinType("::tvm::ffi::ListObj", "::tvm::ffi::List<::tvm::ffi::Any>", 8, 8),
        "ffi.Dict": _BuiltinType(
            "::tvm::ffi::DictObj",
            "::tvm::ffi::Dict<::tvm::ffi::Any, ::tvm::ffi::Any>",
            8,
            8,
        ),
        "ffi.VisitInterrupt": _BuiltinType(
            "::tvm::ffi::VisitInterruptObj",
            "::tvm::ffi::ObjectPtr<::tvm::ffi::VisitInterruptObj>",
            8,
            8,
        ),
    }
    result: dict[int, _BuiltinType] = {}
    for type_key, spec in specs.items():
        type_index = _object_type_key_to_index(type_key)
        if type_index is not None:
            result[type_index] = spec
    return result


def _select_type_infos(type_keys: str | Sequence[str]) -> list[TypeInfo]:
    registered = sorted({str(key) for key in get_registered_type_keys()})
    selected_keys: set[str] = set()
    for selector in [type_keys] if isinstance(type_keys, str) else type_keys:
        matches = [key for key in registered if fnmatch.fnmatchcase(key, selector)]
        if not matches:
            raise ValueError(f"Type-key selector {selector!r} did not match any registered type")
        object_matches = []
        for key in matches:
            info = _lookup_or_register_type_info_from_type_key(key)
            if info.type_index >= _OBJECT_TYPE_INDEX:
                object_matches.append(key)
        if not object_matches:
            raise ValueError(
                f"Type-key selector {selector!r} did not match any registered object type"
            )
        selected_keys.update(object_matches)

    builtins = _builtin_table()
    closure: dict[int, TypeInfo] = {}
    for type_key in sorted(selected_keys):
        info = _lookup_or_register_type_info_from_type_key(type_key)
        if info.type_index < _DYNAMIC_OBJECT_TYPE_INDEX_BEGIN:
            if info.type_index not in builtins:
                raise ValueError(f"Static TVM-FFI type {info.type_key!r} is not supported")
            continue
        for ancestor in _lineage(info):
            if ancestor.type_index >= _DYNAMIC_OBJECT_TYPE_INDEX_BEGIN:
                closure[ancestor.type_index] = ancestor

    return sorted(closure.values(), key=lambda info: (len(info.type_ancestors), info.type_key))


class _Generator:
    def __init__(self, emitted_infos: list[TypeInfo]) -> None:
        self.emitted_infos = emitted_infos
        self.builtins = _builtin_table()
        self.dependencies: dict[int, TypeInfo] = {info.type_index: info for info in emitted_infos}

    def _lower_object(self, schema: TypeSchema) -> _Carrier:
        type_index = schema.origin_type_index
        if type_index == _OBJECT_TYPE_INDEX or schema.origin == "Object":
            return _Carrier("::tvm::ffi::ObjectPtr<::tvm::ffi::Object>", 8, 8)
        builtin = self.builtins.get(schema.origin_type_index)
        if builtin is not None:
            if builtin.value_type is None:
                raise ValueError(
                    f"Static TVM-FFI type {schema.origin!r} has no supported C++ value wrapper"
                )
            return _Carrier(builtin.value_type, builtin.size, builtin.alignment)
        if schema.origin_type_index < _DYNAMIC_OBJECT_TYPE_INDEX_BEGIN:
            raise ValueError(f"Schema {schema!r} is not a registered object type")
        info = _lookup_or_register_type_info_from_type_key(schema.origin)
        self.dependencies[info.type_index] = info
        object_type = _cpp_name(info.type_key).qualified
        return _Carrier(f"::tvm::ffi::ObjectPtr<{object_type}>", 8, 8)

    def _lower_value(
        self,
        schema: TypeSchema,
        *,
        container_argument: bool = False,
        is_native_field: bool = False,
    ) -> _Carrier:
        origin = schema.origin
        scalar = {
            "int": _Carrier("int64_t", 8, 8),
            "float": _Carrier("double", 8, 8),
            "bool": _Carrier("bool", 1, 1),
            "ctypes.c_void_p": _Carrier("void*", 8, 8),
            "dtype": _Carrier("DLDataType", 4, 2),
            "DataType": _Carrier("DLDataType", 4, 2),
            "Device": _Carrier("DLDevice", 8, 4),
            "Any": _Carrier("::tvm::ffi::Any", 16, 8),
            "str": _Carrier("::tvm::ffi::String", 16, 8),
            "bytes": _Carrier("::tvm::ffi::Bytes", 16, 8),
            "Callable": _Carrier("::tvm::ffi::Function", 8, 8),
            "Tensor": _Carrier("::tvm::ffi::Tensor", 8, 8),
        }
        if origin in scalar:
            return scalar[origin]
        args = schema.args
        if (
            origin == "Optional"
            and is_native_field
            and container_argument
            and args[0].origin_type_index >= _OBJECT_TYPE_INDEX
        ):
            return self._lower_object(args[0])
        if origin in ("Optional", "Union"):
            # Container elements are already stored as Any cells.  Keeping the
            # view erased avoids depending on the unstable native wrapper
            # representation while preserving the outer container layout.
            return _Carrier("::tvm::ffi::Any", 16, 8)

        container_origins = {
            "Array": "::tvm::ffi::Array",
            "List": "::tvm::ffi::List",
            "Map": "::tvm::ffi::Map",
            "Dict": "::tvm::ffi::Dict",
        }
        if origin in container_origins:
            lowered = [
                self._lower_value(
                    arg,
                    container_argument=True,
                    is_native_field=is_native_field,
                ).cpp_type
                for arg in args
            ]
            return _Carrier(f"{container_origins[origin]}<{', '.join(lowered)}>", 8, 8)

        if origin == "tuple":
            lowered = [
                self._lower_value(
                    arg,
                    container_argument=True,
                    is_native_field=is_native_field,
                ).cpp_type
                for arg in args
            ]
            return _Carrier(f"::tvm::ffi::Tuple<{', '.join(lowered)}>", 8, 8)

        if schema.origin_type_index >= _OBJECT_TYPE_INDEX:
            return self._lower_object(schema)
        raise ValueError(f"Unsupported TVM-FFI type schema {schema!r}")

    def _lower_field(self, field: TypeField, owner: TypeInfo) -> _Carrier:  # noqa: PLR0912
        schema = field.ty
        is_python_field = hasattr(owner, "_decorator_args")
        if schema is not None and not is_python_field:
            raw_schema: object = field.metadata.get("type_schema")
            if isinstance(raw_schema, str):
                try:
                    raw_schema = json.loads(raw_schema)
                except json.JSONDecodeError as err:
                    raise ValueError(
                        f"Invalid raw type schema for {owner.type_key}.{field.name}: {raw_schema!r}"
                    ) from err
            if not isinstance(raw_schema, dict) or not isinstance(raw_schema.get("type"), str):
                raise ValueError(
                    f"Missing raw type schema for native field {owner.type_key}.{field.name}"
                )
            _validate_native_schema(schema, cast(dict[str, object], raw_schema))
        if schema is None:
            carrier = _static_carrier(field.field_static_type_index)
            if carrier is None:
                raise ValueError(
                    f"Cannot determine carrier for {owner.type_key}.{field.name}: "
                    f"missing type schema and static type index {field.field_static_type_index}"
                )
        elif schema.origin in ("Optional", "Union"):
            if is_python_field:
                carrier = _Carrier("::tvm::ffi::Any", 16, 8)
            elif schema.origin == "Optional":
                value_schema = schema.args[0]
                if value_schema.origin in ("str", "bytes") and (field.size, field.alignment) == (
                    16,
                    8,
                ):
                    carrier = self._lower_value(value_schema)
                elif value_schema.origin_type_index >= _OBJECT_TYPE_INDEX and (
                    field.size,
                    field.alignment,
                ) == (
                    8,
                    8,
                ):
                    carrier = self._lower_object(value_schema)
                    if carrier.size != 8:
                        builtin = self.builtins.get(value_schema.origin_type_index)
                        if builtin is None:
                            raise ValueError(
                                f"Cannot prove pointer carrier for {owner.type_key}.{field.name}"
                            )
                        carrier = _Carrier(f"::tvm::ffi::ObjectPtr<{builtin.object_type}>", 8, 8)
                else:
                    raise ValueError(
                        f"Ambiguous native Optional carrier for {owner.type_key}.{field.name} "
                        f"with layout ({field.size}, {field.alignment})"
                    )
            elif (
                schema.origin == "Union"
                and (field.size, field.alignment) == (8, 8)
                and all(arg.origin_type_index >= _OBJECT_TYPE_INDEX for arg in schema.args)
            ):
                carrier = _Carrier("::tvm::ffi::ObjectPtr<::tvm::ffi::Object>", 8, 8)
            else:
                raise ValueError(
                    f"Ambiguous native {schema.origin} carrier for {owner.type_key}.{field.name} "
                    f"with layout ({field.size}, {field.alignment})"
                )
        else:
            carrier = self._lower_value(schema, is_native_field=not is_python_field)

        if schema is not None:
            builtin = self.builtins.get(schema.origin_type_index)
            if builtin is not None and builtin.value_type == "::tvm::ffi::Error":
                # Error also derives from std::exception, whose size is a C++
                # library ABI detail.  Reflection supplies the dimensions and
                # the generated static assertions verify the local wrapper.
                carrier = _Carrier(builtin.value_type, field.size, field.alignment)

        actual = (field.size, field.alignment)
        expected = (carrier.size, carrier.alignment)
        if (
            actual != expected
            and actual == (8, 8)
            and schema is not None
            and schema.origin_type_index >= _OBJECT_TYPE_INDEX
            and (builtin := self.builtins.get(schema.origin_type_index)) is not None
        ):
            # A few canonical reference wrappers (notably Error, which also
            # derives from std::exception) contain more than one pointer.
            # A reflected one-pointer field uses the canonical object class
            # instead of pretending that the larger wrapper is layout-compatible.
            carrier = _Carrier(f"::tvm::ffi::ObjectPtr<{builtin.object_type}>", 8, 8)
            expected = (8, 8)
        if actual != expected:
            raise ValueError(
                f"Carrier {carrier.cpp_type} for {owner.type_key}.{field.name} has layout "
                f"{expected}, but reflection reports {actual} at offset {field.offset}"
            )
        return carrier

    def _build_class(self, info: TypeInfo) -> _ClassModel:
        if not info._has_type_metadata and not hasattr(info, "_decorator_args"):
            raise ValueError(
                f"Native type {info.type_key!r} does not expose fixed total-size metadata"
            )
        parent = info.parent_type_info
        if parent is None:
            raise ValueError(f"Object type {info.type_key!r} does not expose its parent type")
        if parent.type_index >= _DYNAMIC_OBJECT_TYPE_INDEX_BEGIN:
            base_cpp_type = _cpp_name(parent.type_key).qualified
        elif (builtin := self.builtins.get(parent.type_index)) is not None:
            base_cpp_type = builtin.object_type
        else:
            raise ValueError(
                f"Parent type {parent.type_key!r} of {info.type_key!r} is not supported"
            )

        total_size = int(info.total_size)
        fields = tuple(
            _FieldModel(
                reflected_name=field.name,
                member_name=_cpp_identifier(field.name),
                carrier=self._lower_field(field, info),
                offset=field.offset,
            )
            for field in sorted(info.fields or (), key=lambda field: field.offset)
        )
        alignment = max(
            [_OBJECT_ALIGNMENT]
            + [field.alignment for owner in _lineage(info) for field in (owner.fields or ())]
        )
        return _ClassModel(
            info=info,
            cpp_name=_cpp_name(info.type_key),
            base_cpp_type=base_cpp_type,
            alignment=alignment,
            total_size=total_size,
            fields=fields,
        )

    def build(self) -> str:
        classes = [self._build_class(info) for info in self.emitted_infos]
        lines = [
            "#pragma once",
            "",
            "#include <tvm/ffi/tvm_ffi.h>",
            "",
        ]

        dependencies = sorted(
            self.dependencies.values(),
            key=_cpp_name_sort_key,
        )
        class_indices = {model.info.type_index for model in classes}
        opaque_dependencies = [
            info for info in dependencies if info.type_index not in class_indices
        ]
        forward_blocks = []
        for info in dependencies:
            name = _cpp_name(info.type_key)
            forward_blocks.append((name.namespaces, [f"struct {name.object_name};"]))
        lines.extend(_namespace_lines(forward_blocks))
        for info in dependencies:
            name = _cpp_name(info.type_key)
            lines.extend(
                [
                    "template <>",
                    "inline constexpr bool "
                    f"tvm::ffi::is_object_subclass_v<{name.qualified}> = true;",
                ]
            )
        lines.append("")

        opaque_blocks = []
        for info in opaque_dependencies:
            name = _cpp_name(info.type_key)
            opaque_blocks.append(
                (
                    name.namespaces,
                    [
                        f"struct {name.object_name} : public ::tvm::ffi::Object {{",
                        "  TVM_FFI_DECLARE_OBJECT_INFO_LOOKUP("
                        f"{_cpp_string_literal(info.type_key)}, {len(info.type_ancestors)});",
                        "};",
                    ],
                )
            )
        lines.extend(_namespace_lines(opaque_blocks))

        lines.append(f"static_assert(sizeof(::tvm::ffi::Object) == {_OBJECT_SIZE});")
        lines.append(f"static_assert(alignof(::tvm::ffi::Object) == {_OBJECT_ALIGNMENT});")
        lines.append("")
        lines.extend(
            [
                "#if defined(__clang__) || defined(__GNUC__)",
                "#pragma GCC diagnostic push",
                '#pragma GCC diagnostic ignored "-Winvalid-offsetof"',
                "#elif defined(_MSC_VER)",
                "#pragma warning(push)",
                "#pragma warning(disable : 4749)",
                "#endif",
                "",
            ]
        )
        class_blocks = []
        for model in classes:
            body = [
                f"struct alignas({model.alignment}) {model.cpp_name.object_name} "
                f": public {model.base_cpp_type} {{",
                "  TVM_FFI_DECLARE_OBJECT_INFO_LOOKUP("
                f"{_cpp_string_literal(model.info.type_key)}, "
                f"{len(model.info.type_ancestors)});",
            ]
            if model.fields:
                body.append("")
            for field in model.fields:
                comment = ""
                if field.member_name != field.reflected_name:
                    comment = f", reflected name={field.reflected_name!r}"
                body.append(
                    f"  {field.carrier.cpp_type} {field.member_name};  "
                    f"// offset={field.offset}, size={field.carrier.size}, "
                    f"align={field.carrier.alignment}{comment}"
                )
            body.append("};")
            body.append("")
            body.append(
                f"static_assert(sizeof({model.cpp_name.object_name}) == {model.total_size});"
            )
            body.append(
                f"static_assert(alignof({model.cpp_name.object_name}) == {model.alignment});"
            )
            for field in model.fields:
                body.extend(
                    [
                        f"static_assert(sizeof(decltype({model.cpp_name.object_name}::{field.member_name})) "
                        f"== {field.carrier.size});",
                        f"static_assert(alignof(decltype({model.cpp_name.object_name}::{field.member_name})) "
                        f"== {field.carrier.alignment});",
                        f"static_assert(offsetof({model.cpp_name.object_name}, "
                        f"{field.member_name}) == {field.offset});",
                    ]
                )
            class_blocks.append((model.cpp_name.namespaces, body))
        lines.extend(_namespace_lines(class_blocks))
        lines.extend(
            [
                "#if defined(__clang__) || defined(__GNUC__)",
                "#pragma GCC diagnostic pop",
                "#elif defined(_MSC_VER)",
                "#pragma warning(pop)",
                "#endif",
                "",
            ]
        )
        return "\n".join(lines)


def gen_abi_cpp(type_keys: str | Sequence[str]) -> str:
    """Generate one C++ header containing inheritance-preserving ABI views.

    Parameters
    ----------
    type_keys
        An exact registered type key, a shell-style pattern, or a sequence of
        exact keys and patterns.

    Returns
    -------
    str
        Deterministic C++17 header source.  The source performs lookup only;
        it does not register, compile, load, or allocate any type.

    """
    return _Generator(_select_type_infos(type_keys)).build()


__all__ = ["gen_abi_cpp"]
