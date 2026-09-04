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
"""Rust code generation for ``tvm-ffi-stubgen``: the opaque binding form.

Every reflected object renders as an *opaque* binding: a ``#[repr(C)]`` struct
that embeds only its parent, the reference wrapper, read-only ``Deref``, the
upcasts along the ancestor chain, and one accessor per reflected field that
reads through the C ABI getter. The bytes of the object are never reproduced,
so the binding is correct for every registered type; a later step upgrades the
types whose layout the registry can prove.

Per object, for ``tirx.IterVar`` deriving from ``ir.PrimExprConvertible``::

    #[repr(C)]
    #[derive(tvm_ffi::derive::Object)]
    #[type_key = "tirx.IterVar"]
    #[type_final]
    pub struct IterVarObj {
        base: PrimExprConvertibleObj,
    }

    #[repr(C)]
    #[derive(tvm_ffi::derive::ObjectRef, Clone)]
    pub struct IterVar {
        data: ObjectArc<IterVarObj>,
    }

    impl Deref for IterVar { ... }              // IterVar -> IterVarObj
    impl Deref for IterVarObj { ... }           // IterVarObj -> PrimExprConvertibleObj

    impl IterVarObj {
        pub fn dom(&self) -> Result<Option<Range>> {
            FieldGetter::new(Self::type_index(), "dom")?.get(self)
        }
        ...
    }

    tvm_ffi::impl_object_upcast!(IterVar => PrimExprConvertible);

Nothing here constructs objects or calls methods: construction and behaviour
go through the registered global functions, hand-written outside the markers.

A parent among the ``ffi.*`` builtins (``ffi.IntEnum``, say) has no
``<Leaf>Obj`` in the crate. Embedding the bare ``Object`` header instead would
be wrong: ``derive(Object)`` derives ``TYPE_DEPTH`` from the embedded base and
the runtime instance check indexes the ancestor table with it, so every
subtype of such a type would fail to cast. The import section therefore
defines a header-only mirror per builtin ancestor (``FfiEnumObj``,
``FfiIntEnumObj``, ...) and the object embeds the last one; no ``Deref`` or
upcast leads to a mirror.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from .. import consts as C
from . import consts as C_RUST
from .utils import RustImports, builtin_mirror_name, render_rust_type, rust_ident

if TYPE_CHECKING:
    from pathlib import Path

    from ..file_utils import CodeBlock
    from ..utils import InitConfig, NamedTypeSchema, ObjectInfo, Options
    from .directives import EnumSpec


@dataclasses.dataclass
class _ObjectRenderer:
    """Renders one ``object/<key>`` block into Rust source lines."""

    info: ObjectInfo
    imports: RustImports
    ty_map: dict[str, str]
    #: Module segments of the file this object lands in (its type key minus the
    #: leaf; ``tirx.transform.X`` -> ``("tirx", "transform")``): one file per
    #: prefix, mounted at ``<out>/<seg>/.../mod.rs`` (see :func:`finalize_rust_module_tree`).
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

    def _ty_render(self, origin: str) -> str | None:
        """Resolve a leaf origin to its in-scope Rust name, recording its ``use``.

        Unmapped dotted names (object type keys) resolve against the generated
        module tree. An unmapped bare origin (``const char*``) or a
        ``ctypes.*`` sentinel (``void*``) has no Rust type.
        """
        mapped = self.ty_map.get(origin)
        if mapped is None:
            if "." not in origin or origin.startswith("ctypes."):
                return None
            mapped = self._generated_type_path(origin)
        return self.imports.record(mapped)

    def _generated_type_path(self, type_key: str) -> str:
        """Resolve a generated-tree type key to a path valid from this file.

        A bare ``use ir::Expr;`` is broken in edition 2021 (it resolves to an
        extern crate ``ir``), so cross-module references anchor at the shared
        generated root: ``super::`` once per segment of this file's own module
        path, then the referenced key's full path (``super::ir::Expr`` from
        ``tirx/mod.rs``). A key in *this* file's module is a local item: bare
        leaf, no ``use``. A builtin ``ffi.*`` key lives in the crate and passes
        through for :class:`~.utils.RustUse` to rewrite.
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
        """Whether ``type_key`` has a generated binding to refer to.

        Builtin ``ffi.*`` types live in the crate, which has no ``<Leaf>Obj``
        struct for them: a type under such a parent embeds a header-only
        mirror (see :meth:`_base_type`) and upcasts only to generated ancestors.
        """
        return type_key.partition(".")[0] not in C_RUST.RUST_MOD_MAP

    def _base_type(self) -> tuple[str, bool]:
        """Resolve the struct embedded as ``base`` and whether it is a generated parent.

        A generated parent is embedded by name. Otherwise the parent is a
        builtin: ``ffi.Object`` itself, embedded as the crate's ``Object``
        header, or a deeper ``ffi.*`` type, embedded as the header-only mirror
        the import section defines for every builtin ancestor on the way (see
        :meth:`RustImports.record_builtin_base`), so that ``derive(Object)``
        computes the same ``TYPE_DEPTH`` the registry holds.
        """
        parent = self.info.parent_type_key
        if parent is not None and self._generated(parent):
            return self.imports.record(self._generated_type_path(parent) + "Obj"), True
        chain = [key for key in self.info.ancestors if key != C_RUST.RUST_ROOT_TYPE_KEY]
        if parent not in (None, C_RUST.RUST_ROOT_TYPE_KEY, *chain):
            chain.append(parent)
        # A builtin type only ever derives from builtin types.
        assert not any(self._generated(key) for key in chain), (self.type_key, chain)
        return self.imports.record_builtin_base(chain), False

    # --- pieces ------------------------------------------------------------

    def _accessor_lines(self, field: NamedTypeSchema) -> list[str]:
        """One ``pub fn <field>(&self) -> Result<T>`` reading through the C ABI getter.

        ``T`` comes from the ``enum`` / ``field`` / ``nullable`` directives, else
        from the schema; a field without a Rust type is read as ``Any``.
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
        return [f"pub fn {name}(&self) -> Result<{rust_type}> {{", f"    {getter}.get(self)", "}"]

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

    def body(self) -> list[str]:
        """Build the Rust source lines for the object."""
        # Boilerplate `use`s go through the same collector as field types. The
        # derive macros are spelled by full path, never imported: their leaves
        # collide with `tvm_ffi::Object` / `ObjectRef`.
        self.imports.record("std::ops::Deref")
        self.imports.record("tvm_ffi::ObjectArc")
        base, has_parent = self._base_type()
        fields = self.info.fields
        if fields:
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
        sections.append(
            [
                "#[repr(C)]",
                "#[derive(tvm_ffi::derive::Object)]",
                f'#[type_key = "{self.type_key}"]',
                *(["#[type_final]"] if self.info.is_final else []),
                f"pub struct {self.obj_struct} {{",
                f"    base: {base},",
                "}",
            ]
        )
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
        if fields:
            accessors: list[str] = []
            for i, field in enumerate(fields):
                if i:
                    accessors.append("")
                accessors += self._accessor_lines(field)
            sections.append(
                [
                    f"impl {self.obj_struct} {{",
                    *[f"    {line}" if line else "" for line in accessors],
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
    """Emit the opaque Rust binding of ``obj_info`` into an ``object/<key>`` block."""
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
    """Render the header-only stand-in for one builtin ancestor (see :func:`builtin_mirror_name`)."""
    return [
        f"/// Header-only stand-in for the builtin `{type_key}`, which the crate does not",
        "/// mirror: it only gives `derive(Object)` the ancestor depth of the types below it.",
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
    """Render the ``use`` statements and the builtin mirrors into an ``import-section`` block.

    Imports for types defined in this same file are dropped; the rest are
    deduped and sorted. The header-only mirrors of the builtin ancestors some
    object of this file derives from follow, root first (see
    :meth:`RustImports.record_builtin_base`).
    """
    assert len(code.lines) >= 2
    # `record` never admits bare types, so every `as_use_line()` is non-empty.
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
    """Stitch the generated tree under ``init_path`` into a valid Rust module tree.

    Ensures every generated prefix is declared via ``pub mod`` in its parent's
    ``mod.rs``, creating intermediate ``mod.rs`` files as needed; declarations
    are appended only when absent. The user still mounts ``init_path`` with one
    ``mod`` line at the crate root (stubgen does not edit ``lib.rs``/``main.rs``).
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
