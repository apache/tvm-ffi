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
"""Rust code generation for the ``tvm-ffi-stubgen`` tool.

Codegen orchestration lives here; low-level rendering helpers live in
``rust_generator.utils``.
"""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING

from tvm_ffi.core import MISSING

from .. import consts as C
from ..lib_state import object_info_from_type_key
from . import consts as C_RUST
from .utils import (
    RustImports,
    UnsupportedTypeError,
    _deref_impl,
    _packed_args_expr,
    _packed_call_lines,
    render_rust_type,
)

if TYPE_CHECKING:
    from pathlib import Path

    from tvm_ffi.core import TypeSchema

    from ..file_utils import CodeBlock
    from ..utils import FuncInfo, InitConfig, NamedTypeSchema, ObjectInfo, Options


# --- native (FFI-free) construction eligibility ------------------------------


def _rust_string_literal(s: str) -> str:
    """Escape ``s`` as a double-quoted Rust string literal."""
    out = ['"']
    for ch in s:
        if ch in ('"', "\\"):
            out.append("\\" + ch)
        elif ch.isprintable():
            out.append(ch)
        else:
            out.append(f"\\u{{{ord(ch):x}}}")
    out.append('"')
    return "".join(out)


def _default_expr(field: NamedTypeSchema) -> str | None:
    """Render ``field``'s registered default as a Rust expression (``None``: can't).

    Only values whose Rust spelling is self-evident are supported: ``bool`` /
    ``int`` / finite ``float`` literals (which coerce to the field's possibly
    narrowed scalar type in the struct-literal position) and ``str`` (which
    becomes a ``tvm_ffi::String``). Anything else -- objects, containers,
    non-finite floats, factories -- has no native materialization.
    """
    value = field.default
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return repr(value)
    if isinstance(value, float):
        return repr(value) if math.isfinite(value) else None
    if isinstance(value, str):
        return f"tvm_ffi::String::from({_rust_string_literal(value)})"
    return None


def _native_blocker(info: ObjectInfo) -> str | None:
    """Why ``info`` cannot be constructed natively; ``None`` when it can.

    The native builder allocates the struct directly, binding every own
    field from its setter or a stubgen-rendered default and silently
    bypassing any C++ constructor logic -- that is the opted-in behavior, so
    native is used whenever possible. There is no FFI fallback: a blocked
    type gets no generated constructor at all (the user hand-writes one).
    """
    if not info.has_init:
        return "the type has no reflected constructor"
    for field in info.fields:
        if field.default_is_factory:
            return f"field {field.name!r} uses a default factory (FFI-only)"
        if field.default is not MISSING and _default_expr(field) is None:
            return f"the default value of field {field.name!r} has no Rust rendering"
    parent = info.parent_type_key
    if parent in (None, "ffi.Object") or _native_eligible(parent):
        return None
    return f"parent {parent!r} is not natively constructible"


def _info_native_eligible(info: ObjectInfo) -> bool:
    """Whether ``info`` can be constructed natively (see :func:`_native_blocker`)."""
    return _native_blocker(info) is None


def _native_eligible(type_key: str) -> bool:
    """Type-key wrapper of :func:`_info_native_eligible` (parent recursion).

    A type that cannot be resolved is warned about and treated as non-native.
    Deliberately uncached: a cache would go stale across registry changes.
    """
    try:
        info = object_info_from_type_key(type_key)
    except Exception as e:  # any failure means "cannot prove native-safe"
        print(
            f"{C.TERM_YELLOW}[Warning] cannot resolve type {type_key!r} for native "
            f"construction ({type(e).__name__}: {e}); treating it as non-native"
            f"{C.TERM_RESET}"
        )
        return False
    return _info_native_eligible(info)


def _layout_fields(fields: list[NamedTypeSchema]) -> list[NamedTypeSchema]:
    """Sort own fields by reflection ``offset`` (C++ memory order).

    Registration order need not match memory order, but the ``#[repr(C)]``
    struct is positional. Fields without an offset (synthetic ``ObjectInfo``s
    in tests) keep registration order.
    """
    if any(f.offset is None for f in fields):
        return list(fields)
    return sorted(fields, key=lambda f: f.offset)


def _warn_offset_mismatch(type_key: str | None, fields: list[NamedTypeSchema]) -> None:
    """Warn when ``#[repr(C)]`` cannot reproduce the recorded field offsets.

    Recomputes each field's ``#[repr(C)]`` placement from the previous field's
    end. Reflection has no ``alignof``, so alignment is approximated from
    ``size`` (largest power of two, capped at 8) -- exact for scalars, but
    composite FFI structs like ``DLDevice`` can trigger a false positive. A
    mismatch only warns; the binding is still emitted. Fields without
    offset/size metadata are skipped and reset the running position.
    """
    prev_end: int | None = None
    for field in fields:
        if field.offset is None or field.size is None:
            prev_end = None
            continue
        if prev_end is not None:
            align = min(8, field.size & -field.size)
            placed = (prev_end + align - 1) // align * align
            if placed != field.offset:
                print(
                    f"{C.TERM_YELLOW}[Warning] object {type_key}: field "
                    f"{field.name!r} is at C++ offset {field.offset}, but the "
                    f"generated #[repr(C)] layout places it at offset {placed}; "
                    f"the Rust struct may not match the C++ object layout"
                    f"{C.TERM_RESET}"
                )
        # Resync to the recorded offset so one hole yields one warning.
        prev_end = field.offset + field.size


@dataclasses.dataclass
class _ObjectRenderer:
    """Renders one ``object/<key>`` block into Rust source lines.

    Holds the per-object rendering context (imports, ``ty_map``, resolved
    names) so helper methods don't have to thread it through.
    """

    info: ObjectInfo
    leaf: str
    obj_struct: str
    base_type: str
    is_root: bool
    imports: RustImports
    ty_map: dict[str, str]

    def _ty_render(self, origin: str) -> str:
        """Resolve a leaf origin to its Rust name and record its ``use``.

        Unmapped dotted names (object type keys) pass through; an unmapped bare
        origin (e.g. ``const char*``) has no Rust rendering and raises, which
        skips the enclosing object.
        """
        mapped = self.ty_map.get(origin)
        if mapped is None:
            if "." not in origin:
                raise UnsupportedTypeError(origin)
            mapped = origin
        return self.imports.record(mapped)

    def render_struct_field(self, schema: NamedTypeSchema) -> str:
        """Render a directly-laid-out struct field type, width-correct for scalars.

        An ``int32_t`` field must render as ``i32``, not the schema-erased
        default ``i64``; the width comes from reflection's per-field ``size``.
        Non-scalar origins (or schemas without a size) render plainly.
        """
        narrowed = C_RUST.RUST_SCALAR_BY_SIZE.get((schema.origin, schema.size))
        return narrowed if narrowed is not None else render_rust_type(schema, self._ty_render)

    def render_param(self, schema: TypeSchema) -> str:
        """Render an argument type (a top-level ``Any`` is the non-owning ``AnyView``)."""
        if schema.origin == "Any":
            return self.imports.record("tvm_ffi::AnyView")
        return render_rust_type(schema, self._ty_render)

    def body(self) -> list[str]:
        """Build the Rust source lines for the object (raises on unsupported types)."""
        # Boilerplate `use`s, recorded through the same collector as field types
        # so leaf collisions raise and skip the object. The derive macros are
        # spelled by full path in the attribute, never imported: their leaves
        # collide with `tvm_ffi::Object`/`ObjectRef`.
        self.imports.record("std::ops::Deref")
        # `ObjectCore` must be in scope for the generated `type_index()` calls.
        self.imports.record("tvm_ffi::ObjectCore")
        self.imports.record("tvm_ffi::ObjectArc")
        if self.is_root:
            # Same path the ty_map uses for `Object` fields, so they dedup
            # instead of colliding.
            self.base_type = self.imports.record("tvm_ffi::Object")
        # C++ `_type_mutable`: class-level mutability dominates per-field `def_ro`.
        if self.info.mutable:
            self.imports.record("std::ops::DerefMut")

        leaf, obj_struct, base_type = self.leaf, self.obj_struct, self.base_type
        lines: list[str] = []
        lines += [
            "#[repr(C)]",
            "#[derive(tvm_ffi::derive::Object)]",
            f'#[type_key = "{self.info.type_key}"]',
            f"pub struct {obj_struct} {{",
            f"    base: {base_type},",
        ]
        for field in _layout_fields(self.info.fields):
            lines.append(f"    pub {field.name}: {self.render_struct_field(field)},")
        lines += ["}", ""]

        lines += [
            "#[repr(C)]",
            "#[derive(tvm_ffi::derive::ObjectRef, Clone)]",
            f"pub struct {leaf} {{",
            f"    data: ObjectArc<{obj_struct}>,",
            "}",
            "",
        ]

        lines += _deref_impl(leaf, obj_struct, "data", self.info.mutable)
        if not self.is_root:
            lines += _deref_impl(obj_struct, base_type, "base", self.info.mutable)

        # Native (FFI-free) construction whenever the whole chain is eligible;
        # there is no FFI fallback -- a blocked constructor is skipped loudly.
        blocker = _native_blocker(self.info)
        native = blocker is None
        if self.info.has_init and not native:
            print(
                f"{C.TERM_YELLOW}[Warning] object {self.info.type_key}: skipping "
                f"`ffi_new` because {blocker}; hand-write a constructor outside "
                f"the generated markers{C.TERM_RESET}"
            )
        lines += self._impl_block(native)
        if native:
            lines += self._builder_lines()

        lines.pop()  # every section above ends with a `""` separator
        return lines

    def _impl_block(self, native: bool) -> list[str]:
        """Emit `impl <T> { ffi_new; methods }`; empty list when there's nothing to emit."""
        methods = [
            m for m in self.info.methods if m.schema.name.rsplit(".", 1)[-1] != "__ffi_init__"
        ]
        if not native and not methods:
            return []

        inner: list[str] = []
        if native:  # `native` implies `has_init` (see `_native_blocker`)
            inner += self._new_fn_native()
            if methods:
                inner.append("")
        for i, method in enumerate(methods):
            inner += self._method_fn(method)
            if i != len(methods) - 1:
                inner.append("")

        return [
            f"impl {self.leaf} {{",
            *[f"    {line}" if line else "" for line in inner],
            "}",
            "",
        ]

    def _obj_literal_lines(self) -> list[str]:
        """Render the ``<Obj> { .. }`` literal moving the builder's fields in.

        Defaulted fields move straight from the builder; the rest bind the
        like-named locals that :meth:`_unwrap_lines` just checked (on derived
        types ``base`` binds the local :meth:`_base_resolve_lines` produced).
        """
        base_entry = "    base: self.base," if self.is_root else "    base,"
        lines = [f"{self.obj_struct} {{", base_entry]
        # Entries bind by name; memory order just mirrors the struct definition.
        for field in _layout_fields(self.info.fields):
            if field.default is MISSING:
                lines.append(f"    {field.name},")  # the unwrapped local
            else:
                lines.append(f"    {field.name}: self.{field.name},")
        lines.append("}")
        return lines

    def _base_resolve_lines(self) -> list[str]:
        """``let base = ..`` resolving a derived builder's base (empty for roots).

        An unset ``base`` falls back to the parent's all-default builder. Its
        error is re-contextualized: the parent's bare "field `x` is not set"
        would point at a field this type does not have.
        """
        if self.is_root:
            return []
        parent_ref = (self.info.parent_type_key or "").rsplit(".", 1)[-1]
        return [
            "let base = match self.base {",
            "    Some(base) => base,",
            f"    None => {parent_ref}::ffi_new().build_obj().map_err(|e| tvm_ffi::Error::new(",
            "        tvm_ffi::VALUE_ERROR,",
            f'        &format!("field `base` is not set and default `{parent_ref}` '
            'construction failed: {}", e.message()),',
            '        "",',
            "    ))?,",
            "};",
        ]

    def _unwrap_lines(self) -> list[str]:
        """``let <f> = self.<f>.ok_or_else(..)?;`` for every field without a default."""
        return [
            f"let {field.name} = self.{field.name}.ok_or_else(|| tvm_ffi::Error::new("
            f'tvm_ffi::VALUE_ERROR, "field `{field.name}` is not set", ""))?;'
            for field in _layout_fields(self.info.fields)
            if field.default is MISSING
        ]

    def _new_fn_native(self) -> list[str]:
        """Emit ``fn ffi_new() -> <T>Builder``, opening the builder chain.

        Uniformly nullary: every input -- own fields and a derived type's
        ``base`` alike -- is set through its like-named builder setter.
        Defaulted fields start prefilled with their stubgen-rendered default,
        the rest start unset and ``build()`` errors on any still missing (an
        unset ``base`` is default-constructed through the parent's builder
        instead; see :meth:`_base_resolve_lines`). Named ``ffi_new`` (not
        ``new``); a user who needs the faithful C++ constructor semantics
        hand-writes ``new`` (outside the markers) delegating to the builder.
        """
        builder = f"{self.leaf}Builder"
        lines = [f"pub fn ffi_new() -> {builder} {{", f"    {builder} {{"]
        if self.is_root:
            lines.append(f"        base: {self.base_type}::new(),")
        else:
            lines.append("        base: None,")
        for field in _layout_fields(self.info.fields):
            if field.default is MISSING:
                lines.append(f"        {field.name}: None,")
            else:
                # `_native_blocker` already guaranteed the default renders.
                lines.append(f"        {field.name}: {_default_expr(field)},")
        lines += ["    }", "}"]
        return lines

    def _builder_lines(self) -> list[str]:
        """Emit ``pub struct <T>Builder`` + its ``impl`` (setters, ``build``, ``build_obj``).

        One consuming setter per own field, plus ``base`` on derived types
        (stored ``Option<ParentObj>``; left unset it is default-constructed
        through the parent's builder at build time). Defaulted fields are
        stored prefilled; fields without a default are stored as ``Option<T>``
        and checked by ``build_obj``, which returns ``Err`` when one is still
        unset. ``build_obj`` is public -- it returns the bare struct value a
        derived type's ``base`` setter takes -- and ``build`` delegates to it,
        wrapping the struct in the allocated ref type.
        """
        builder = f"{self.leaf}Builder"
        fields = _layout_fields(self.info.fields)
        base_store = self.base_type if self.is_root else f"Option<{self.base_type}>"
        lines = [f"pub struct {builder} {{", f"    base: {base_store},"]
        for field in fields:
            ty = self.render_struct_field(field)
            store = ty if field.default is not MISSING else f"Option<{ty}>"
            lines.append(f"    {field.name}: {store},")
        lines += ["}", ""]

        inner: list[str] = []
        if not self.is_root:
            inner += [
                f"pub fn base(mut self, base: {self.base_type}) -> Self {{",
                "    self.base = Some(base);",
                "    self",
                "}",
                "",
            ]
        for field in fields:
            ty = self.render_struct_field(field)
            value = field.name if field.default is not MISSING else f"Some({field.name})"
            inner += [
                f"pub fn {field.name}(mut self, {field.name}: {ty}) -> Self {{",
                f"    self.{field.name} = {value};",
                "    self",
                "}",
                "",
            ]
        self.imports.record("tvm_ffi::Result")
        prelude = [*self._base_resolve_lines(), *self._unwrap_lines()]
        literal = self._obj_literal_lines()
        inner += [
            f"pub fn build(self) -> Result<{self.leaf}> {{",
            f"    Ok({self.leaf} {{",
            "        data: ObjectArc::new(self.build_obj()?),",
            "    })",
            "}",
            "",
            f"pub fn build_obj(self) -> Result<{self.obj_struct}> {{",
            *[f"    {line}" for line in prelude],
            f"    Ok({literal[0]}",
            *[f"    {line}" for line in literal[1:-1]],
            f"    {literal[-1]})",
            "}",
        ]
        lines += [
            f"impl {builder} {{",
            *[f"    {line}" if line else "" for line in inner],
            "}",
            "",
        ]
        return lines

    def _cached_getter_lines(self, fvar: str, ffi_name: str) -> list[str]:
        """Body lines binding ``fvar`` to the reflected method, cached per call site.

        A ``thread_local!`` ``OnceCell`` makes the crate's method-table scan run
        once per thread (``Function`` is not ``Sync``, ruling out a ``OnceLock``).
        """
        cell = fvar.upper()
        return [
            f"    thread_local!(static {cell}: std::cell::OnceCell<tvm_ffi::Function> = "
            "const { std::cell::OnceCell::new() });",
            f"    let {fvar} = tvm_ffi::Function::from_type_method_cached(&{cell}, "
            f'{self.obj_struct}::type_index(), "{ffi_name}")?;',
        ]

    def _method_fn(self, method: FuncInfo) -> list[str]:
        """Emit one reflected method (instance or static) on `impl <T>`."""
        ffi_name = method.schema.name.rsplit(".", 1)[-1]
        args = method.schema.args or ()
        # The return type stays owning (a top-level `Any` is `Any`, not `AnyView`).
        ret = render_rust_type(args[0], self._ty_render) if args else self._ty_render("Any")
        rest = args[2:] if method.is_member else args[1:]
        params = [(f"_{i}", self.render_param(p)) for i, p in enumerate(rest)]

        self_recv = "&mut self" if self.info.mutable else "&self"
        if method.is_member:
            sig_parts = [self_recv, *[f"{n}: {t}" for n, t in params]]
        else:
            sig_parts = [f"{n}: {t}" for n, t in params]
        self.imports.record("tvm_ffi::Result")
        if method.is_member or params:
            self.imports.record("tvm_ffi::AnyView")
        packed = _packed_args_expr(params, method.is_member)
        getter = self._cached_getter_lines("f", ffi_name)
        header = f"pub fn {ffi_name}({', '.join(sig_parts)}) -> Result<{ret}> {{"
        return [header, *_packed_call_lines("f", getter, packed, ret), "}"]


def generate_rust_object(
    code: CodeBlock,
    ty_map: dict[str, str],
    imports: RustImports,
    opt: Options,
    obj_info: ObjectInfo,
) -> None:
    """Emit a Rust ``struct``/``impl`` binding for an ``object/<key>`` block.

    Emits ``<T>Obj`` (``#[repr(C)]``, parent embedded as ``base``), the ``<T>``
    ref wrapper, ``Deref``/``DerefMut``, ``impl <T>`` with ``ffi_new`` plus the
    reflected methods, and the ``<T>Builder`` (when natively constructible).
    Raises :class:`UnsupportedTypeError` for types the crate cannot represent;
    ``cli`` catches it and skips the block (any ``use``s already recorded are
    harmless -- generated files allow unused imports).
    """
    assert len(code.lines) >= 2
    type_key = obj_info.type_key
    assert isinstance(type_key, str)
    leaf = type_key.rsplit(".", 1)[-1]
    obj_struct = f"{leaf}Obj"
    parent_key = obj_info.parent_type_key
    is_root = parent_key in (None, "ffi.Object")
    if is_root:
        base_type = "Object"
    else:
        assert isinstance(parent_key, str)
        base_type = f"{parent_key.rsplit('.', 1)[-1]}Obj"
    renderer = _ObjectRenderer(
        info=obj_info,
        leaf=leaf,
        obj_struct=obj_struct,
        base_type=base_type,
        is_root=is_root,
        imports=imports,
        ty_map=ty_map,
    )

    body = renderer.body()

    _warn_offset_mismatch(type_key, _layout_fields(obj_info.fields))

    indent = " " * code.indent
    code.lines = [
        code.lines[0],
        *[(indent + line) if line else "" for line in body],
        code.lines[-1],
    ]
    _ = opt  # accepted for protocol parity; Rust object layout needs no `opt`


# --- import section (`use` statements) --------------------------------------


def generate_rust_import_section(
    code: CodeBlock,
    imports: RustImports,
    opt: Options,
    defined_types: set[str],
) -> None:
    """Render the collected ``use`` statements into an ``import-section`` block.

    Imports for types defined in this same file are dropped; the rest are
    deduped and sorted.
    """
    assert len(code.lines) >= 2
    # `record` never admits bare types, so every `as_use_line()` is non-empty.
    use_lines = sorted(
        {item.as_use_line() for item in imports.items if item.path not in defined_types}
    )
    indent = " " * code.indent
    code.lines = [
        code.lines[0],
        *[indent + line for line in use_lines],
        code.lines[-1],
    ]
    _ = opt  # accepted for protocol parity; Rust needs no indent/TYPE_CHECKING handling


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
