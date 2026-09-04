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
"""Rust generator helpers: ``use`` modelling, type rendering, identifier spelling."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Callable

from . import consts as C
from .directives import Directives

if TYPE_CHECKING:
    from tvm_ffi.core import TypeSchema


@dataclasses.dataclass(frozen=True, eq=True)
class RustUse:
    """A Rust ``use`` item: ``use <path>;``.

    Dotted FFI names become ``::`` paths via :data:`~.consts.RUST_MOD_MAP`
    (``ffi.String -> tvm_ffi::String``); bare names (``i64``) need no ``use``.
    """

    path: str

    def __init__(self, name: str) -> None:
        """Normalize ``name`` into a Rust ``use`` path and store it."""
        if "::" not in name and "." in name:
            head, _, tail = name.partition(".")
            head = C.RUST_MOD_MAP.get(head, head)
            name = f"{head}.{tail}"
        object.__setattr__(self, "path", name.replace(".", "::"))

    @property
    def leaf(self) -> str:
        """The final path segment (the in-scope name), e.g. ``Array`` for ``tvm_ffi::Array``."""
        return self.path.rsplit("::", 1)[-1]

    def as_use_line(self) -> str:
        """Render the ``use`` statement, or ``""`` for a bare prelude/primitive type."""
        if "::" not in self.path:
            return ""
        return f"use {self.path};"


def builtin_mirror_name(type_key: str) -> str:
    """Name of the header-only stand-in for a builtin type (``ffi.IntEnum -> FfiIntEnumObj``).

    ``derive(Object)`` takes ``TYPE_DEPTH`` from the embedded base, so a type
    under a builtin needs a base at the builtin's depth; the crate has none.
    """
    head, _, leaf = type_key.rpartition(".")
    return f"{head.replace('.', '_').capitalize()}{leaf}Obj"


@dataclasses.dataclass
class RustImports:
    """The per-file collector: the ``use`` items and the Rust directives of one file."""

    items: list[RustUse] = dataclasses.field(default_factory=list)
    directives: Directives = dataclasses.field(default_factory=Directives)
    builtin_mirrors: dict[str, str] = dataclasses.field(default_factory=dict)
    """Builtin ancestors mirrored in this file, root first: type key -> the base each embeds."""

    def record_builtin_base(self, chain: list[str]) -> str:
        """Record the stand-ins for the builtin ``chain`` (root side first); return the last name.

        An empty chain (parent ``ffi.Object``) yields the crate's ``Object``.
        """
        base = self.record("tvm_ffi::Object")
        for key in chain:
            self.builtin_mirrors.setdefault(key, base)
            base = builtin_mirror_name(key)
        return base

    def record(self, name: str) -> str:
        """Record a ``use`` (deduped) and return the name to spell in code.

        The leaf, or the full path when another path already claimed that leaf.
        """
        probe = RustUse(name)
        if not probe.as_use_line():
            return probe.leaf
        for item in self.items:
            if item.path == probe.path:
                return item.leaf
        if any(item.leaf == probe.leaf for item in self.items):
            return probe.path
        self.items.append(probe)
        return probe.leaf


def render_rust_type(schema: TypeSchema, ty_render: Callable[[str], str | None]) -> str | None:
    """Render ``schema`` as a Rust value type via ``ty_render`` (leaf origin -> name), or ``None``."""
    origin, args = schema.origin, schema.args
    if origin in C.RUST_UNSUPPORTED_ORIGINS:
        return None
    if origin == "Array":
        assert args  # TypeSchema's post_init fills a missing element type.
        return _generic(ty_render("Array"), render_rust_type(args[0], ty_render))
    if origin == "Map":
        assert len(args) == 2  # TypeSchema's post_init fills a bare Map to (Any, Any).
        key = render_rust_type(args[0], ty_render)
        value = render_rust_type(args[1], ty_render)
        return _generic(ty_render("Map"), key, value)
    if origin == "Optional":
        (payload,) = args  # TypeSchema's post_init enforces exactly one argument.
        return _generic("Option", render_rust_type(payload, ty_render))
    if origin == "Callable":
        return ty_render("Callable")  # the crate's Function is type-erased
    return ty_render(origin)


def _generic(base: str | None, *params: str | None) -> str | None:
    if base is None or any(p is None for p in params):
        return None
    return f"{base}<{', '.join(p for p in params if p is not None)}>"


def rust_ident(name: str) -> str:
    """Spell a reflected field name in Rust: the C++ trailing underscore is dropped."""
    return name.rstrip("_") or name
