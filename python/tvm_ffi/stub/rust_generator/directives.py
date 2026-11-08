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
"""The Rust backend's directive keys, and the settings they give one file.

Each subject's keys (see :mod:`tvm_ffi.stub.directives` for the grammar)::

    // tvm-ffi-stubgen(object): tirx.Add(custom_new, upcast = PrimExpr)
    // tvm-ffi-stubgen(object): ir.SourceName(opaque)
    // tvm-ffi-stubgen(object): ir.Expr(no_new)
    // tvm-ffi-stubgen(field): tirx.Add.a(type = PrimExpr)
    // tvm-ffi-stubgen(field): ir.Expr.span(nullable)
    // tvm-ffi-stubgen(field): ir.IRModule.global_var_map_(visibility = pub(crate))
    // tvm-ffi-stubgen(enum): ForKind(repr = i32, Serial = 0, Parallel = 1)
    // tvm-ffi-stubgen(field): tirx.For.kind(type = ForKind)
    // tvm-ffi-stubgen(file): mod(missing_docs = allow)
    // tvm-ffi-stubgen(import): tvm_ffi::AnyView

``object`` keys: ``opaque`` keeps a type opaque even when its layout is
reproducible; ``upcast`` (repeatable) adds an upcast to a reference outside the
ancestor chain; ``custom_new`` says the wrapper's ``new`` is hand-written, so the
generated one is named ``from_fields``; ``no_new`` skips the wrapper's generated
allocator, and for a final type ``<Leaf>Obj::new`` too, which only that allocator
uses. ``field`` keys: ``type`` sets the field's Rust type (a name in scope, a
``::`` path to ``use``, or an ``enum`` of the file); on a field inherited from an
ancestor it narrows the allocator parameter instead. ``nullable`` wraps it in
``Option``; ``visibility = pub(crate)`` makes the field and its accessor
``pub(crate)`` instead of ``pub``. ``enum`` declares an open integer newtype of
``repr`` with the members ``Name = <int>``, or with ``closed`` a ``#[repr(<int>)]``
enum of exactly those members. ``file: mod(missing_docs = allow)`` puts
``#[allow(missing_docs)]`` on the file's generated types and ``impl`` blocks.
``import`` keeps a path ``use``d in the import section.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from ..directives import Key, KeyTable

if TYPE_CHECKING:
    from ..directives import Directives

_FLAG, _VALUE = Key(), Key(takes_value=True)
_INT_TYPES = ("i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64")

#: The keys the Rust backend reads; an ``enum``'s other keys are its members.
RUST_DIRECTIVE_KEYS: KeyTable = {
    "object": {
        "custom_new": _FLAG,
        "no_new": _FLAG,
        "opaque": _FLAG,
        "upcast": Key(takes_value=True, repeatable=True),
    },
    "field": {"type": _VALUE, "nullable": _FLAG, "visibility": Key(True, values=("pub(crate)",))},
    "enum": {"repr": Key(True, values=_INT_TYPES), "closed": _FLAG, "*": _VALUE},
    "file": {"missing_docs": Key(True, values=("allow",))},
    "import": {},
}


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
class RustDirectives:
    """The Rust settings of one file, keyed by ``<type_key>.<field>`` or ``<type_key>``."""

    field_types: dict[str, str] = dataclasses.field(default_factory=dict)
    nullable: set[str] = dataclasses.field(default_factory=set)
    visibility: dict[str, str] = dataclasses.field(default_factory=dict)
    enums: dict[str, EnumSpec] = dataclasses.field(default_factory=dict)
    opaque: set[str] = dataclasses.field(default_factory=set)
    upcasts: dict[str, list[str]] = dataclasses.field(default_factory=dict)
    custom_new: set[str] = dataclasses.field(default_factory=set)
    no_new: set[str] = dataclasses.field(default_factory=set)
    allow_missing_docs: bool = False

    @staticmethod
    def from_directives(directives: Directives) -> RustDirectives:
        """Collect the settings of ``directives``; raise ``ValueError`` on a malformed ``enum``."""
        settings, out = directives.settings, RustDirectives()
        specs = {name: _enum_spec(name, keys) for name, keys in settings["enum"].items()}
        for target, keys in settings["field"].items():
            rust_type = str(keys.get("type", ""))
            if rust_type in specs:
                out.enums[target] = specs[rust_type]
            elif rust_type:
                out.field_types[target] = rust_type
            if "nullable" in keys:
                out.nullable.add(target)
            if "visibility" in keys:
                out.visibility[target] = str(keys["visibility"])
        for target, keys in settings["object"].items():
            for name, chosen in (
                ("opaque", out.opaque),
                ("custom_new", out.custom_new),
                ("no_new", out.no_new),
            ):
                if name in keys:
                    chosen.add(target)
            upcasts = keys.get("upcast", [])
            if isinstance(upcasts, list) and upcasts:
                out.upcasts[target] = upcasts
        out.allow_missing_docs = "missing_docs" in settings["file"].get("mod", {})
        return out


def _enum_spec(name: str, keys: dict[str, bool | str | list[str]]) -> EnumSpec:
    """Build the ``enum`` ``name``: a ``repr``, members ``Name = <int>``, and ``closed``."""
    members = tuple((k, v) for k, v in keys.items() if k not in ("repr", "closed"))
    if "repr" not in keys:
        raise ValueError(f"`enum` directive `{name}` needs `repr = <{'|'.join(_INT_TYPES)}>`")
    try:
        values = tuple((member, int(str(value))) for member, value in members)
    except ValueError:
        raise ValueError(f"`enum` directive `{name}`: a member value is not an integer") from None
    closed = "closed" in keys
    if closed and all(value != 0 for _, value in values):
        raise ValueError(
            f"Closed `enum` directive `{name}` has no member equal to 0, the value of "
            "a field an object zero-filled at creation never set"
        )
    return EnumSpec(name, str(keys["repr"]), values, closed)
