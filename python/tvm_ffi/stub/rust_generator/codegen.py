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
``Deref``, and the upcasts along its ancestor chain. The struct's contents follow
the verdict of :mod:`tvm_ffi.stub.layout`:

- *complete*: the struct mirrors every physical field at its real offset and
  width, pinned by a ``const`` size/alignment assertion. ``<Leaf>Obj::new``
  (crate-private) and the wrapper's ``new`` take every field root to leaf;
  ``custom-new`` leaves the wrapper's ``new`` to hand-written code and names
  the generated one ``from_complete_fields``.
- *opaque*: the struct embeds only its parent, and each field is read through
  the C ABI getter. Nothing allocates it.

A field without a Rust mirror (``Optional<Any>``, ``Union``, ``void*``, ...)
makes the type opaque and is read as ``Any``; an ``opaque`` directive vetoes a
reproducible layout; a scalar width named by a directive is checked against the
reflected field size. A builtin parent (``ffi.IntEnum``, say) has no
``<Leaf>Obj`` in the crate: the import section defines a header-only stand-in
per builtin ancestor so ``derive(Object)`` computes the registry's
``TYPE_DEPTH``, and everything under such a parent stays opaque (``no-mirror``).
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from .. import consts as C
from ..layout import Verdict, classify
from ..lib_state import object_info_from_type_key
from . import consts as C_RUST
from .utils import RustImports, builtin_mirror_name, render_rust_type, rust_ident

if TYPE_CHECKING:
    from pathlib import Path

    from ..file_utils import CodeBlock
    from ..utils import InitConfig, NamedTypeSchema, ObjectInfo, Options
    from .directives import EnumSpec


def _call_lines(open_: str, items: list[str], close: str) -> list[str]:
    """``open_ + items + close`` on one line, or one item per line when it would overflow."""
    line = f"{open_}{', '.join(items)}{close}"
    if len(line) <= C_RUST.RUST_MAX_WIDTH:
        return [line]
    indent = open_[: len(open_) - len(open_.lstrip())]
    return [open_, *[f"{indent}    {item}," for item in items], f"{indent}{close}"]


def _check_width(target: str, field: NamedTypeSchema, rust_type: str, width: int) -> None:
    """Reject a directive whose scalar type does not match the reflected field size."""
    if field.size is not None and field.size != width:
        raise ValueError(
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

        # Builtin ancestors are header-only stand-ins (`_base_type`): none below is complete.
        unmirrored = {
            key
            for key in self.info.ancestors
            if key != C_RUST.RUST_ROOT_TYPE_KEY and not self._generated(key)
        }
        verdicts = classify(
            infos,
            forced_opaque=self.imports.directives.opaque,
            unmirrored=unmirrored,
            field_renderable=renderable,
        )
        return verdicts[self.type_key]

    # --- field types -------------------------------------------------------

    def _field_mirror(self, owner: str, field: NamedTypeSchema, imports: RustImports) -> str | None:
        """Render the ``#[repr(C)]`` mirror type of ``field``, or ``None``.

        Directives win; then scalars by reflected width, ``Optional`` by C++ layout, else schema.
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
                raise ValueError(
                    f"`nullable` directive on `{target}`: the field is {field.size} bytes, "
                    "not a pointer-sized object reference"
                )
            mirror = f"Option<{mirror}>"
        return mirror

    def _optional_mirror(self, field: NamedTypeSchema, imports: RustImports) -> str | None:
        """Mirror an ``Optional<T>`` field in place.

        An object payload is a nullable pointer (``Option<T>``); any other payload
        is a 16-byte ``TVMFFIAny`` cell (``tvm_ffi::Optional<T>``). ``Optional<Any>``
        and a size mismatch have no mirror.
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

    # --- pieces ------------------------------------------------------------

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
        return [f"pub fn {name}(&self) -> Result<{rust_type}> {{", f"    {getter}.get(self)", "}"]

    def _enum_lines(self, spec: EnumSpec) -> list[str]:
        """Render the open integer newtype an ``enum`` directive declares."""
        error = self.imports.record("tvm_ffi::Error")
        value_error = self.imports.record("tvm_ffi::VALUE_ERROR")
        result = self.imports.record("tvm_ffi::Result")
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
            f"    fn try_from(value: i64) -> {result}<Self> {{",
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
        """``impl_object_upcast!`` to every ancestor's wrapper, then the ``upcast`` directives."""
        targets = [
            self.imports.record(self._generated_type_path(key))
            for key in self.info.ancestors
            if self._generated(key)
        ]
        for view in self.imports.directives.upcasts.get(self.type_key, []):
            targets.append(self.imports.record(view) if "::" in view else view)
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
            f"    base: {base},",  # a reflected `base` field becomes `base_` (`rust_ident`)
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

    # --- allocators --------------------------------------------------------

    def _allocator_params(self, key: str, info: ObjectInfo) -> list[tuple[str, str]]:
        """``(field, type)`` of every physical field root to leaf, as ``<key>Obj::new`` takes."""
        parent = info.parent_type_key
        inherited: list[tuple[str, str]] = []
        if parent is not None and self._generated(parent):
            inherited = self._allocator_params(parent, object_info_from_type_key(parent))
        return self._level_params(key, info, inherited)

    def _level_params(
        self, key: str, info: ObjectInfo, inherited: list[tuple[str, str]]
    ) -> list[tuple[str, str]]:
        """Extend the parent's parameters with ``key``'s own fields by offset.

        A ``field`` directive on an inherited field narrows that parameter; the
        body hands it to the parent with ``.into()``.
        """
        params = [(name, self._narrowed(key, name, rust_type)) for name, rust_type in inherited]
        for field in sorted(info.fields, key=lambda f: f.offset or 0):
            mirror = self._field_mirror(key, field, self.imports)
            assert mirror is not None  # complete: every field along the chain has a mirror
            params.append((field.name, mirror))
        return params

    def _narrowed(self, key: str, field_name: str, rust_type: str) -> str:
        override = self.imports.directives.field_types.get(f"{key}.{field_name}")
        if override is None:
            return rust_type
        return self.imports.record(override) if "::" in override else override

    def _fn_lines(
        self, head: str, params: list[tuple[str, str]], call: tuple[str, list[str]], result: str
    ) -> list[str]:
        """Render ``<head>(<params>) -> Self { let <call>; <result> }`` inside an ``impl`` block."""
        plist = [f"{rust_ident(name)}: {rust_type}" for name, rust_type in params]
        binding, args = call
        return [
            *_call_lines(f"    {head}(", plist, ") -> Self {"),
            *_call_lines(f"        let {binding}(", args, ");"),
            f"        {result}",
            "    }",
        ]

    def _allocator_sections(self, base: str, has_parent: bool) -> list[list[str]]:
        """``<Leaf>Obj::new`` and the wrapper's ``new``.

        ``custom-new`` names the wrapper's allocator ``from_complete_fields`` instead.
        """
        inherited: list[tuple[str, str]] = []
        if has_parent:
            parent = self.info.parent_type_key
            assert parent is not None
            inherited = self._allocator_params(parent, object_info_from_type_key(parent))
        params = self._level_params(self.type_key, self.info, inherited)
        to_parent = [
            f"{rust_ident(name)}.into()" if rust_type != parent_type else rust_ident(name)
            for (name, rust_type), (_, parent_type) in zip(params, inherited)
        ]
        own = [rust_ident(f.name) for f in sorted(self.info.fields, key=lambda f: f.offset or 0)]
        forward = [rust_ident(name) for name, _ in params]
        sections = [
            [
                f"impl {self.obj_struct} {{",
                *self._fn_lines(
                    "pub(crate) fn new",
                    params,
                    (f"base = {base}::new", to_parent),
                    f"Self {{ {', '.join(['base', *own])} }}",
                ),
                "}",
            ]
        ]
        custom = self.type_key in self.imports.directives.custom_new
        sections.append(
            [
                f"impl {self.leaf} {{",
                "    /// Lossless complete-field allocation.",
                *self._fn_lines(
                    "pub fn from_complete_fields" if custom else "pub fn new",
                    params,
                    (f"obj = {self.obj_struct}::new", forward),
                    "Self { base: ObjectArc::new(obj) }",
                ),
                "}",
            ]
        )
        return sections

    def body(self) -> list[str]:
        """Build the Rust source lines for the object."""
        verdict = self.classify()
        # Derive macros are spelled by full path: their leaves collide with `Object` / `ObjectRef`.
        self.imports.record("std::ops::Deref")
        self.imports.record("tvm_ffi::ObjectArc")
        base, has_parent = self._base_type()
        fields = self.info.fields
        has_accessors = bool(fields) and not verdict.is_complete
        if has_accessors:
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
                f"    base: ObjectArc<{self.obj_struct}>,",
                "}",
            ]
        )
        sections.append(self._deref_lines(self.leaf, self.obj_struct, "base"))
        if has_parent:
            sections.append(self._deref_lines(self.obj_struct, base, "base"))
        if has_accessors:
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
        elif verdict.is_complete:
            sections += self._allocator_sections(base, has_parent)
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
