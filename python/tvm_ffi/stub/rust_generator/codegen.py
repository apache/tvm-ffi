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
"""Rust code generation for ``tvm-ffi-stubgen``.

Every reflected object gets a ``#[repr(C)]`` object struct, a reference wrapper,
read-only ``Deref``, and the upcasts along its ancestor chain. What the object
struct holds depends on the verdict of :mod:`tvm_ffi.stub.layout`:

- *complete*: the layout is reproducible, so the struct mirrors every physical
  field at its real offset and width, public, borrowed directly. A ``const``
  assertion pins the struct's ``size_of`` / ``align_of`` to the reflected facts,
  so a mirror rustc lays out differently fails to compile.
- *opaque*: the struct embeds only its parent, and one accessor per reflected
  field reads through the C ABI getter. The bytes are never reproduced, so the
  binding is correct for every registered type.

The two target-language rules the classifier leaves to its caller live here: a
field without a Rust mirror (``Optional<Any>``, a ``Union``, ``void*``, ...)
makes the type opaque and is read as ``Any``; an ``opaque`` directive vetoes a
reproducible layout. ``field`` / ``nullable`` / ``enum`` directives shape the
field types of both forms; where a directive names a scalar width, it is checked
against the reflected field size at generation time.

Construction and behaviour go through the registered global functions,
hand-written outside the markers. A builtin parent (``ffi.IntEnum``, say) has
no ``<Leaf>Obj`` in the crate: the import section defines a header-only
stand-in per builtin ancestor, so ``derive(Object)`` computes the registry's
``TYPE_DEPTH``.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from .. import consts as C
from ..layout import Verdict, classify
from ..lib_state import object_info_from_type_key
from ..utils import DirectiveError
from . import consts as C_RUST
from .utils import RustImports, builtin_mirror_name, render_rust_type, rust_ident

if TYPE_CHECKING:
    from pathlib import Path

    from ..file_utils import CodeBlock
    from ..utils import InitConfig, NamedTypeSchema, ObjectInfo, Options
    from .directives import EnumSpec


def _check_width(target: str, field: NamedTypeSchema, rust_type: str, width: int) -> None:
    """Reject a directive whose scalar type does not match the reflected field size."""
    if field.size is not None and field.size != width:
        raise DirectiveError(
            f"Directive on `{target}` maps a {field.size}-byte field to `{rust_type}` "
            f"({width} bytes)"
        )


@dataclasses.dataclass
class _ObjectRenderer:
    """Renders one ``object/<key>`` block into Rust source lines."""

    info: ObjectInfo
    imports: RustImports
    ty_map: dict[str, str]
    #: Module segments of the file this object lands in (``tirx.transform.X`` -> ``("tirx", "transform")``).
    mod_segments: tuple[str, ...]

    @property
    def type_key(self) -> str:
        """The object's type key."""
        assert self.info.type_key is not None
        return self.info.type_key

    @property
    def leaf(self) -> str:
        """The reference wrapper's name (``IterVar``)."""
        return self.type_key.rsplit(".", 1)[-1]

    @property
    def obj_struct(self) -> str:
        """The object struct's name (``IterVarObj``)."""
        return f"{self.leaf}Obj"

    # --- name resolution ---------------------------------------------------

    def _resolve(self, origin: str, imports: RustImports) -> str | None:
        """Resolve a leaf origin to its in-scope Rust name (recording its ``use``), or ``None``."""
        mapped = self.ty_map.get(origin)
        if mapped is None:
            if "." not in origin or origin.startswith("ctypes."):
                return None
            mapped = self._generated_type_path(origin)
        return imports.record(mapped)

    def _ty_render(self, origin: str) -> str | None:
        return self._resolve(origin, self.imports)

    def _generated_type_path(self, type_key: str) -> str:
        """Spell a generated type key from this file.

        Same module: the bare leaf. Elsewhere: ``super::`` per segment of this
        file's module, then the full path (edition 2021 rejects ``use ir::Expr``).
        """
        head, _, _ = type_key.partition(".")
        if head in C_RUST.RUST_MOD_MAP:
            return type_key
        mod, _, type_leaf = type_key.rpartition(".")
        if tuple(mod.split(".")) == self.mod_segments:
            return type_leaf
        supers = "super::" * len(self.mod_segments)
        return f"{supers or 'self::'}{type_key.replace('.', '::')}"

    def _generated(self, type_key: str) -> bool:
        """Whether ``type_key`` has a generated binding (builtin ``ffi.*`` types live in the crate)."""
        return type_key.partition(".")[0] not in C_RUST.RUST_MOD_MAP

    def _base_type(self) -> tuple[str, bool]:
        """Resolve the ``base`` struct and whether it is a generated parent.

        A builtin parent below ``ffi.Object`` is embedded as its header-only
        stand-in (see :meth:`RustImports.record_builtin_base`).
        """
        parent = self.info.parent_type_key
        if parent is not None and self._generated(parent):
            return self.imports.record(self._generated_type_path(parent) + "Obj"), True
        chain = [key for key in self.info.ancestors if key != C_RUST.RUST_ROOT_TYPE_KEY]
        if parent not in (None, C_RUST.RUST_ROOT_TYPE_KEY, *chain):
            chain.append(parent)
        assert not any(self._generated(key) for key in chain), (self.type_key, chain)
        return self.imports.record_builtin_base(chain), False

    # --- classification ----------------------------------------------------

    def classify(self) -> Verdict:
        """Classify this object with its ancestors, under the file's directives."""
        infos = {key: object_info_from_type_key(key) for key in self.info.ancestors}
        infos[self.type_key] = self.info
        owner_of = {id(f): key for key, owner in infos.items() for f in owner.fields}
        scratch = RustImports()

        def renderable(field: NamedTypeSchema) -> bool:
            return self._field_mirror(owner_of[id(field)], field, scratch) is not None

        verdicts = classify(
            infos, forced_opaque=self.imports.directives.opaque, field_renderable=renderable
        )
        return verdicts[self.type_key]

    # --- field types ---------------------------------------------------------

    def _field_mirror(self, owner: str, field: NamedTypeSchema, imports: RustImports) -> str | None:
        """Render the type of ``field`` in a ``#[repr(C)]`` mirror; ``None`` when it has none.

        Scalars take the width the registry recorded; ``Optional`` fields take
        the in-place mirror of their C++ layout; directives override the rest.
        """
        directives = self.imports.directives
        target = f"{owner}.{field.name}"
        enum = directives.enums.get(target)
        if enum is not None:
            _check_width(target, field, enum.repr, C_RUST.RUST_SCALAR_WIDTHS[enum.repr])
            return enum.name
        override = directives.field_types.get(target)
        if override is not None:
            width = C_RUST.RUST_SCALAR_WIDTHS.get(override)
            if width is not None:
                _check_width(target, field, override, width)
            mirror: str | None = imports.record(override) if "::" in override else override
        elif field.origin == "Optional":
            mirror = self._optional_mirror(field, imports)
        else:
            narrowed = C_RUST.RUST_SCALAR_BY_SIZE.get((field.origin, field.size))
            mirror = narrowed or render_rust_type(field, lambda o: self._resolve(o, imports))
        if mirror is None:
            return None
        if target in directives.nullable and not mirror.startswith("Option<"):
            if field.size not in (None, C_RUST.RUST_POINTER_SIZE):
                raise DirectiveError(
                    f"`nullable` directive on `{target}`: the field is {field.size} bytes, "
                    "not a pointer-sized object reference"
                )
            mirror = f"Option<{mirror}>"
        return mirror

    def _optional_mirror(self, field: NamedTypeSchema, imports: RustImports) -> str | None:
        """Mirror an ``Optional<T>`` field in place.

        An ``ObjectRef``-derived payload is a pointer-sized nullable pointer in
        C++, mirrored by Rust's niche-optimized ``Option<T>``. Every other
        payload stays a 16-byte ``TVMFFIAny`` cell, mirrored by
        ``tvm_ffi::Optional<T>``. ``Optional<Any>`` has no mirror, and neither
        does a field whose size disagrees with its payload kind.
        """
        (payload,) = field.args  # TypeSchema's post_init enforces exactly one argument.
        if payload.origin == "Any":
            return None
        inner = render_rust_type(payload, lambda o: self._resolve(o, imports))
        if inner is None:
            return None
        any_backed = (
            payload.origin in C_RUST.RUST_ANY_BACKED_OPTIONAL_PAYLOADS
            or payload.origin == "Optional"
        )
        expected = (
            C_RUST.RUST_OPTIONAL_FIELD_SIZE
            if any_backed
            else C_RUST.RUST_OBJECT_OPTIONAL_FIELD_SIZE
        )
        if field.size not in (None, expected):
            return None
        if any_backed:
            return f"{imports.record(C_RUST.RUST_OPTIONAL_PATH)}<{inner}>"
        return f"Option<{inner}>"

    def _accessor_lines(self, field: NamedTypeSchema) -> list[str]:
        """One ``pub fn <field>(&self) -> Result<T>`` through the C ABI getter.

        ``T`` comes from the directives, else the schema; without a Rust type, ``Any``.
        """
        directives = self.imports.directives
        target = f"{self.type_key}.{field.name}"
        name = rust_ident(field.name)
        getter = f'FieldGetter::new(Self::type_index(), "{field.name}")?'

        enum = directives.enums.get(target)
        if enum is not None:
            return [
                f"pub fn {name}(&self) -> Result<{enum.name}> {{",
                f"    let raw: i64 = {getter}.get(self)?;",
                f"    {enum.name}::try_from(raw)",
                "}",
            ]
        override = directives.field_types.get(target)
        if override is not None:
            rust_type = self.imports.record(override) if "::" in override else override
        else:
            rust_type = render_rust_type(field, self._ty_render)
        if rust_type is None:
            any_type = self.imports.record("tvm_ffi::Any")
            return [
                f"pub fn {name}(&self) -> Result<{any_type}> {{",
                f"    {getter}.get_any(self)",
                "}",
            ]
        if target in directives.nullable and not rust_type.startswith("Option<"):
            rust_type = f"Option<{rust_type}>"
        return [
            f"pub fn {name}(&self) -> Result<{rust_type}> {{",
            f"    {getter}.get(self)",
            "}",
        ]

    # --- pieces ------------------------------------------------------------

    def _enum_lines(self, spec: EnumSpec) -> list[str]:
        """Render the open integer newtype an ``enum`` directive declares."""
        error = self.imports.record("tvm_ffi::Error")
        value_error = self.imports.record("tvm_ffi::VALUE_ERROR")
        return [
            "#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]",
            "#[repr(transparent)]",
            f"pub struct {spec.name}({spec.repr});",
            "",
            "#[allow(non_upper_case_globals)]",
            f"impl {spec.name} {{",
            *[f"    pub const {member}: Self = Self({value});" for member, value in spec.members],
            f"    pub const fn from_raw(value: {spec.repr}) -> Self {{",
            "        Self(value)",
            "    }",
            f"    pub const fn as_raw(self) -> {spec.repr} {{",
            "        self.0",
            "    }",
            "}",
            "",
            f"impl TryFrom<i64> for {spec.name} {{",
            f"    type Error = {error};",
            "    fn try_from(value: i64) -> Result<Self> {",
            f"        {spec.repr}::try_from(value).map(Self).map_err(|_| {{",
            f'            {error}::new({value_error}, &format!("{spec.name} value {{value}} does not fit '
            f'{spec.repr}"), "")',
            "        })",
            "    }",
            "}",
        ]

    def _deref_lines(self, source: str, target: str, member: str) -> list[str]:
        return [
            f"impl Deref for {source} {{",
            f"    type Target = {target};",
            f"    fn deref(&self) -> &{target} {{",
            f"        &self.{member}",
            "    }",
            "}",
        ]

    def _upcast_lines(self) -> list[str]:
        """``impl_object_upcast!`` from the wrapper to every ancestor's wrapper."""
        targets = [
            self.imports.record(self._generated_type_path(key))
            for key in self.info.ancestors
            if self._generated(key)
        ]
        if not targets:
            return []
        pairs = ", ".join(f"{self.leaf} => {target}" for target in targets)
        return [f"tvm_ffi::impl_object_upcast!({pairs});"]

    def _struct_lines(self, verdict: Verdict, base: str) -> list[str]:
        """Render the object struct: every field when complete, the parent alone when opaque."""
        header = [
            "#[repr(C)]",
            "#[derive(tvm_ffi::derive::Object)]",
            f'#[type_key = "{self.type_key}"]',
            *(["#[type_final]"] if self.info.is_final else []),
            f"pub struct {self.obj_struct} {{",
            f"    base: {base},",
        ]
        if not verdict.is_complete:
            return [
                f"/// Opaque: {verdict.detail}. Fields are read through the C ABI getters.",
                *header,
                "}",
            ]
        members = []
        for field in sorted(self.info.fields, key=lambda f: f.offset or 0):
            mirror = self._field_mirror(self.type_key, field, self.imports)
            assert mirror is not None  # the verdict already ran the renderability check
            members.append(f"    pub {rust_ident(field.name)}: {mirror},")
        return [
            f"/// Complete: {verdict.detail}.",
            *header,
            *members,
            "}",
            "",
            "const _: () = {",
            f"    assert!(::core::mem::size_of::<{self.obj_struct}>() == {verdict.total_size});",
            f"    assert!(::core::mem::align_of::<{self.obj_struct}>() == {verdict.alignment});",
            "};",
        ]

    def body(self) -> list[str]:
        """Build the Rust source lines for the object."""
        verdict = self.classify()
        # Derive macros are spelled by full path: their leaves collide with `Object` / `ObjectRef`.
        self.imports.record("std::ops::Deref")
        self.imports.record("tvm_ffi::ObjectArc")
        base, has_parent = self._base_type()
        fields = self.info.fields
        accessors = bool(fields) and not verdict.is_complete
        if accessors:
            self.imports.record("tvm_ffi::ObjectCore")  # `Self::type_index()`
            self.imports.record("tvm_ffi::FieldGetter")
            self.imports.record("tvm_ffi::Result")

        sections: list[list[str]] = []
        enums = self.imports.directives.enums
        sections += [
            self._enum_lines(enums[f"{self.type_key}.{f.name}"])
            for f in fields
            if f"{self.type_key}.{f.name}" in enums
        ]
        sections.append(self._struct_lines(verdict, base))
        sections.append(
            [
                "#[repr(C)]",
                "#[derive(tvm_ffi::derive::ObjectRef, Clone)]",
                f"pub struct {self.leaf} {{",
                f"    data: ObjectArc<{self.obj_struct}>,",
                "}",
            ]
        )
        sections.append(self._deref_lines(self.leaf, self.obj_struct, "data"))
        if has_parent:
            sections.append(self._deref_lines(self.obj_struct, base, "base"))
        if accessors:
            lines_: list[str] = []
            for i, field in enumerate(fields):
                if i:
                    lines_.append("")
                lines_ += self._accessor_lines(field)
            sections.append(
                [
                    f"impl {self.obj_struct} {{",
                    *[f"    {line}" if line else "" for line in lines_],
                    "}",
                ]
            )
        upcasts = self._upcast_lines()
        if upcasts:
            sections.append(upcasts)

        lines: list[str] = []
        for i, section in enumerate(sections):
            if i:
                lines.append("")
            lines += section
        return lines


def generate_rust_object(
    code: CodeBlock,
    ty_map: dict[str, str],
    imports: RustImports,
    opt: Options,
    obj_info: ObjectInfo,
) -> None:
    """Emit the Rust binding of ``obj_info`` into an ``object/<key>`` block."""
    assert len(code.lines) >= 2
    assert isinstance(obj_info.type_key, str)
    renderer = _ObjectRenderer(
        info=obj_info,
        imports=imports,
        ty_map=ty_map,
        mod_segments=tuple(obj_info.type_key.split(".")[:-1]),
    )
    body = renderer.body()
    indent = " " * code.indent
    code.lines = [
        code.lines[0],
        *[(indent + line) if line else "" for line in body],
        code.lines[-1],
    ]
    _ = opt  # accepted for protocol parity


# --- import section (`use` statements) --------------------------------------


def _builtin_mirror_lines(type_key: str, base: str) -> list[str]:
    """Render the header-only stand-in for one builtin ancestor."""
    return [
        f"/// Header-only stand-in for the builtin `{type_key}`; it only carries the ancestor depth.",
        "#[allow(dead_code)]",
        "#[repr(C)]",
        "#[derive(tvm_ffi::derive::Object)]",
        f'#[type_key = "{type_key}"]',
        f"struct {builtin_mirror_name(type_key)} {{",
        f"    base: {base},",
        "}",
    ]


def generate_rust_import_section(
    code: CodeBlock,
    imports: RustImports,
    opt: Options,
    defined_types: set[str],
) -> None:
    """Render the ``use`` lines, then the builtin stand-ins, into an ``import-section`` block.

    Imports of types defined in this file are dropped; the rest are deduped and sorted.
    """
    assert len(code.lines) >= 2
    body = sorted({item.as_use_line() for item in imports.items if item.path not in defined_types})
    for type_key, base in imports.builtin_mirrors.items():
        body += ["", *_builtin_mirror_lines(type_key, base)]
    indent = " " * code.indent
    code.lines = [
        code.lines[0],
        *[(indent + line) if line else "" for line in body],
        code.lines[-1],
    ]
    _ = opt  # accepted for protocol parity


# --- whole-file scaffolding (`--init` mode) ---------------------------------


def generate_rust_api_file(
    code_blocks: list[CodeBlock],
    ty_map: dict[str, str],
    module_name: str,
    object_infos: list[ObjectInfo],
    init_cfg: InitConfig,
    is_root: bool,
    syntax: C.MarkerSyntax,
) -> str:
    """Scaffold a single Rust binding file (one file per module prefix)."""
    append = ""
    if not code_blocks:
        append += "#![allow(dead_code, unused_imports)]\n"
        append += f"\n//! FFI bindings for `{module_name}` (generated by tvm-ffi-stubgen).\n\n"
    if not any(c.kind == "import-section" for c in code_blocks):
        append += f"{syntax.begin} import-section\n{syntax.end}\n\n"
    defined = {c.param for c in code_blocks if c.kind == "object"}
    for info in object_infos:
        type_key = info.type_key
        if type_key is None or type_key in defined:
            continue
        append += f"{syntax.begin} object/{type_key}\n{syntax.end}\n\n"
    _ = (ty_map, init_cfg, is_root)  # unused for the Rust single-file layout
    return append


# --- module-tree stitching (auto-form `pub mod` declarations) ----------------


def finalize_rust_module_tree(init_path: Path, prefixes: set[str]) -> None:
    """Declare each generated prefix with ``pub mod`` in its parent's ``mod.rs``.

    Missing ``mod.rs`` files are created; the user mounts ``init_path`` with one ``mod`` line.
    """
    children: dict[Path, set[str]] = {}
    for prefix in prefixes:
        segs = [s for s in prefix.split(".") if s]
        for i, seg in enumerate(segs):
            parent = init_path.joinpath(*segs[:i])
            children.setdefault(parent, set()).add(seg)

    for parent, names in children.items():
        parent.mkdir(parents=True, exist_ok=True)
        mod_rs = parent / "mod.rs"
        existing = mod_rs.read_text(encoding="utf-8") if mod_rs.exists() else ""
        to_add = [f"pub mod {n};" for n in sorted(names) if f"pub mod {n};" not in existing]
        if not to_add:
            continue
        text = existing
        if text and not text.endswith("\n"):
            text += "\n"
        if text.strip():  # separate from any existing bindings
            text += "\n"
        text += "\n".join(to_add) + "\n"
        mod_rs.write_text(text, encoding="utf-8")
