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
"""The Rust backend's one-line directives: payload grammar and per-file storage.

Three address one reflected field as ``<type_key>.<field>``, three address a type::

    // tvm-ffi-stubgen(field): tirx.Add.a -> PrimExpr
    // tvm-ffi-stubgen(nullable): ir.Expr.span
    // tvm-ffi-stubgen(enum): tirx.For.kind -> ForKind(i32) { Serial=0, Parallel=1 }
    // tvm-ffi-stubgen(opaque): ir.SourceName
    // tvm-ffi-stubgen(upcast): tirx.Add -> PrimExpr
    // tvm-ffi-stubgen(custom-new): tirx.Add

``field`` sets the field's Rust type (a name in scope, or a ``::`` path to
``use``); on a field inherited from an ancestor it narrows the allocator
parameter instead. ``nullable`` wraps it in ``Option``; ``enum`` declares an open
integer newtype for it; ``opaque`` keeps a type opaque even when its layout is
reproducible. ``upcast`` adds a typed view outside the ancestor chain;
``custom-new`` says the wrapper's ``new`` is hand-written, so none is generated.
"""

from __future__ import annotations

import dataclasses
import re

_ENUM_RE = re.compile(
    r"^(?P<target>\S+)\s*->\s*(?P<name>[A-Za-z_]\w*)\((?P<repr>[iu](?:8|16|32|64))\)"
    r"\s*(?:\{(?P<body>[^{}]*)\})?$"
)
_MEMBER_RE = re.compile(r"^(?P<name>[A-Za-z_]\w*)\s*=\s*(?P<value>-?\d+)$")


@dataclasses.dataclass(frozen=True)
class EnumSpec:
    """An ``enum`` directive: the newtype's name, its integer repr, and its members."""

    name: str
    repr: str
    members: tuple[tuple[str, int], ...]


@dataclasses.dataclass
class Directives:
    """The Rust directives of one file, keyed by ``<type_key>.<field>`` or ``<type_key>``."""

    field_types: dict[str, str] = dataclasses.field(default_factory=dict)
    nullable: set[str] = dataclasses.field(default_factory=set)
    enums: dict[str, EnumSpec] = dataclasses.field(default_factory=dict)
    opaque: set[str] = dataclasses.field(default_factory=set)
    upcasts: dict[str, list[str]] = dataclasses.field(default_factory=dict)
    custom_new: set[str] = dataclasses.field(default_factory=set)

    def add(self, name: str, payload: str, lineno: int) -> None:
        """Parse and store one directive; raise ``ValueError`` on a malformed payload."""
        if name == "field":
            lhs, rust_type = _split_arrow(name, payload, lineno, "<type_key>.<field> -> <RustType>")
            self.field_types[_field_target(name, lhs, lineno)] = rust_type
        elif name == "nullable":
            self.nullable.add(_field_target(name, payload, lineno))
        elif name == "enum":
            target, spec = _parse_enum(payload, lineno)
            self.enums[target] = spec
        elif name == "opaque":
            self.opaque.add(_type_target(name, payload, lineno))
        elif name == "upcast":
            lhs, rust_type = _split_arrow(name, payload, lineno, "<type_key> -> <RustType>")
            self.upcasts.setdefault(_type_target(name, lhs, lineno), []).append(rust_type)
        elif name == "custom-new":
            self.custom_new.add(_type_target(name, payload, lineno))
        else:
            raise ValueError(f"Unknown directive `{name}` at line {lineno}")


def _invalid(name: str, lineno: int, expected: str) -> ValueError:
    return ValueError(f"Invalid `{name}` directive at line {lineno}. Expected `{expected}`")


def _type_target(name: str, text: str, lineno: int) -> str:
    """Validate a ``<type_key>`` reference."""
    target = text.strip()
    if not target or " " in target:
        raise _invalid(name, lineno, "<type_key>")
    return target


def _field_target(name: str, text: str, lineno: int) -> str:
    """Validate a ``<type_key>.<field>`` reference."""
    target = text.strip()
    if not target or " " in target or "." not in target.strip("."):
        raise _invalid(name, lineno, "<type_key>.<field>")
    return target


def _split_arrow(name: str, payload: str, lineno: int, expected: str) -> tuple[str, str]:
    """Split ``<target> -> <rust type>``; the caller validates the target."""
    lhs, arrow, rhs = payload.partition("->")
    if not arrow or not rhs.strip():
        raise _invalid(name, lineno, expected)
    return lhs, rhs.strip()


def _parse_enum(payload: str, lineno: int) -> tuple[str, EnumSpec]:
    """Parse ``<type_key>.<field> -> Name(i32) { A=0, B=1 }`` (the member list is optional)."""
    expected = "<type_key>.<field> -> Name(i32) { A=0, B=1 }"
    match = _ENUM_RE.match(payload.strip())
    if match is None:
        raise _invalid("enum", lineno, expected)
    members: list[tuple[str, int]] = []
    for item in (match.group("body") or "").split(","):
        if not item.strip():
            continue
        member = _MEMBER_RE.match(item.strip())
        if member is None:
            raise _invalid("enum", lineno, expected)
        members.append((member.group("name"), int(member.group("value"))))
    target = _field_target("enum", match.group("target"), lineno)
    return target, EnumSpec(match.group("name"), match.group("repr"), tuple(members))
