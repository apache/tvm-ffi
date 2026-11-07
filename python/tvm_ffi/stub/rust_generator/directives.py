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

Four address one reflected field as ``<type_key>.<field>``, three address a type,
and one applies to the whole file::

    // tvm-ffi-stubgen(field): tirx.Add.a -> PrimExpr
    // tvm-ffi-stubgen(nullable): ir.Expr.span
    // tvm-ffi-stubgen(visibility): ir.IRModule.global_var_map_ -> pub(crate)
    // tvm-ffi-stubgen(enum): tirx.For.kind -> ForKind(i32) { Serial=0, Parallel=1 }
    // tvm-ffi-stubgen(opaque): ir.SourceName
    // tvm-ffi-stubgen(upcast): tirx.Add -> PrimExpr
    // tvm-ffi-stubgen(custom-new): tirx.Add
    // tvm-ffi-stubgen(missing-docs): allow

``field`` sets the field's Rust type (a name in scope, or a ``::`` path to
``use``); on a field inherited from an ancestor it narrows the allocator
parameter instead. ``nullable`` wraps it in ``Option``; ``visibility`` makes the
field and its accessor ``pub(crate)`` instead of ``pub``; ``enum`` declares an open
integer newtype for it, or with ``closed`` before the member list a
``#[repr(<int>)]`` enum of exactly those members; ``opaque`` keeps a type opaque
even when its layout is reproducible. ``upcast`` adds an upcast to a reference
outside the ancestor chain; ``custom-new`` says the wrapper's ``new`` is
hand-written; the generated one is named ``from_fields`` instead.
``missing-docs: allow`` puts ``#[allow(missing_docs)]`` on the file's generated
types and ``impl`` blocks.
"""

from __future__ import annotations

import dataclasses
import re

_ENUM_RE = re.compile(
    r"^(?P<target>\S+)\s*->\s*(?P<name>[A-Za-z_]\w*)\((?P<repr>[iu](?:8|16|32|64))\)"
    r"\s*(?:(?P<closed>closed\s*)?\{(?P<body>[^{}]*)\})?$"
)
_MEMBER_RE = re.compile(r"^(?P<name>[A-Za-z_]\w*)\s*=\s*(?P<value>-?\d+)$")


@dataclasses.dataclass(frozen=True)
class EnumSpec:
    """An ``enum`` directive: the type's name, its integer repr, and its members.

    ``closed`` emits a Rust ``enum``: reading any other value from a mirrored field is
    undefined behavior, so it is for value sets fixed by the ABI that include 0, the
    value of a field an object zero-filled at creation never set.
    """

    name: str
    repr: str
    members: tuple[tuple[str, int], ...]
    closed: bool = False


@dataclasses.dataclass
class Directives:
    """The Rust directives of one file, keyed by ``<type_key>.<field>`` or ``<type_key>``."""

    field_types: dict[str, str] = dataclasses.field(default_factory=dict)
    nullable: set[str] = dataclasses.field(default_factory=set)
    visibility: dict[str, str] = dataclasses.field(default_factory=dict)
    enums: dict[str, EnumSpec] = dataclasses.field(default_factory=dict)
    opaque: set[str] = dataclasses.field(default_factory=set)
    upcasts: dict[str, list[str]] = dataclasses.field(default_factory=dict)
    custom_new: set[str] = dataclasses.field(default_factory=set)
    allow_missing_docs: bool = False

    def add(self, name: str, payload: str, lineno: int) -> None:
        """Parse and store one directive; raise ``ValueError`` on a malformed payload."""
        if name == "field":
            lhs, rust_type = _split_arrow(name, payload, lineno, "<type_key>.<field> -> <RustType>")
            self.field_types[_field_target(name, lhs, lineno)] = rust_type
        elif name == "nullable":
            self.nullable.add(_field_target(name, payload, lineno))
        elif name == "visibility":
            lhs, vis = _split_arrow(name, payload, lineno, "<type_key>.<field> -> pub(crate)")
            if vis != "pub(crate)":  # the one non-default level so far
                raise _invalid(name, lineno, "<type_key>.<field> -> pub(crate)")
            self.visibility[_field_target(name, lhs, lineno)] = vis
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
        elif name == "missing-docs":
            if payload.strip() != "allow":  # the one level so far
                raise _invalid(name, lineno, "allow")
            self.allow_missing_docs = True
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
    """Parse ``<type_key>.<field> -> Name(i32) [closed] { A=0, B=1 }``.

    The member list is optional unless ``closed``, which needs a member equal to 0.
    """
    expected = "<type_key>.<field> -> Name(i32) [closed] { A=0, B=1 }"
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
    closed = match.group("closed") is not None
    if closed and not members:
        raise _invalid("enum", lineno, expected)
    if closed and all(value != 0 for _, value in members):
        raise ValueError(
            f"Closed `enum` directive at line {lineno} has no member equal to 0, the value of "
            "a field an object zero-filled at creation never set"
        )
    target = _field_target("enum", match.group("target"), lineno)
    spec = EnumSpec(match.group("name"), match.group("repr"), tuple(members), closed)
    return target, spec
