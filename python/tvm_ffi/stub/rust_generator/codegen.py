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
  and the wrapper's ``new`` take every field root to leaf: the former each as
  its own type, the latter a scalar as its own type and anything else as
  ``impl Into``. ``custom-new`` leaves the wrapper's ``new`` to hand-written
  code and names the generated one ``from_fields``. A type that is not final
  can be the parent of a type defined in Rust (``tvm_ffi::rs_class``): its
  ``<Leaf>Obj::new`` is public, and its object implements ``ShallowCopy`` when
  every ancestor's is generated too (a ``ty-map``'d one is hand-written and
  may not). ``<Leaf>Obj::new`` of a final type is crate-private.
- *opaque*: the struct embeds only its parent, and each field is read through
  the C ABI getter. Nothing allocates it.

A field without a Rust mirror (``Optional<Any>``, ``Union``, ``void*``, an
``ffi.*`` type the default ``ty-map`` does not list, ...) makes the type opaque
and is read as ``Any``; an ``opaque`` directive vetoes a reproducible layout; a
scalar width named by a directive is checked against the reflected field size.
A builtin parent (``ffi.IntEnum``, say) has no
``<Leaf>Obj`` in the crate: the import section defines a header-only stand-in
per builtin ancestor so ``derive(Object)`` computes the registry's
``TYPE_DEPTH``, and everything under such a parent stays opaque (``no-mirror``).

Every type key an object refers to (parent, ancestors, field types) must be
provided in the same run: ``ffi.*`` by the crate, a ``ty-map`` by a hand-written
binding whose object struct is ``<Name>Obj``, anything else by an ``object/``
block in one of the processed files. Otherwise the block is an error naming the
missing keys, so a partial binding never references a module that does not exist.
A generated type is named from the module of the file its block is in (see
:func:`~.utils.rust_module_path`), not from its type key: the bare name in that
module, a ``crate::`` path anywhere else.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from .. import consts as C
from ..layout import Verdict, classify
from ..lib_state import object_info_from_type_key
from . import consts as C_RUST
from .utils import (
    RustImports,
    builtin_mirror_name,
    is_crate_type_key,
    render_rust_type,
    rust_ident,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from pathlib import Path

    from ..file_utils import CodeBlock
    from ..utils import InitConfig, NamedTypeSchema, ObjectInfo, Options
    from .directives import EnumSpec


def _call_lines(open_: str, items: list[str], close: str) -> list[str]:
    """``open_ + items + close`` on one line, or wrapped the way rustfmt does when it would overflow.

    A ``let`` first moves its call to the next line; anything else, and a call that
    still does not fit, takes one item per line.
    """
    line = f"{open_}{', '.join(items)}{close}"
    if len(line) <= C_RUST.RUST_MAX_WIDTH:
        return [line]
    indent = open_[: len(open_) - len(open_.lstrip())]
    let, eq, call = open_.partition(" = ")
    moved = f"{indent}    {call}{', '.join(items)}{close}"
    if eq and len(moved) <= C_RUST.RUST_MAX_WIDTH:
        return [f"{let} =", moved]
    return [
        open_.rstrip(),
        *[f"{indent}    {item}," for item in items],
        f"{indent}{close.lstrip()}",
    ]


def _param_type(rust_type: str) -> str:
    """Type a parameter of the wrapper's allocator: a scalar as itself, anything else as ``impl Into``."""
    return rust_type if rust_type in C_RUST.RUST_EXACT_PARAMS else f"impl Into<{rust_type}>"


def _forward(ident: str, rust_type: str) -> str:
    """Pass a parameter of the wrapper's allocator to ``<Leaf>Obj::new``, which takes ``rust_type``."""
    return ident if rust_type in C_RUST.RUST_EXACT_PARAMS else f"{ident}.into()"


def _check_width(target: str, field: NamedTypeSchema, rust_type: str, width: int) -> None:
    """Reject a directive whose scalar type does not match the reflected field size."""
    if field.size is not None and field.size != width:
        raise ValueError(
            f"Directive on `{target}` maps a {field.size}-byte field to `{rust_type}` "
            f"({width} bytes)"
        )


def _allow_non_snake_case(idents: Iterable[str]) -> list[str]:
    """``#[allow(non_snake_case)]`` when rustc would warn on one of the Rust ``idents``."""
    if any(ident.removeprefix("r#") != ident.removeprefix("r#").lower() for ident in idents):
        return ["#[allow(non_snake_case)]"]
    return []


@dataclasses.dataclass
class _ObjectRenderer:
    """Renders one ``object/<key>`` block into Rust source lines."""

    info: ObjectInfo
    imports: RustImports
    ty_map: dict[str, str]
    #: Module of the file this object lands in (``src/tirx/transform.rs`` -> ``("crate", "tirx", "transform")``).
    module: tuple[str, ...]
    #: Type keys with an ``object/`` block somewhere in this run, to the module of the block's file.
    declared: Mapping[str, tuple[str, ...]]
    #: Referenced type keys nobody provides; reported together once the body is built.
    missing: set[str] = dataclasses.field(default_factory=set)

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

    def _resolve(
        self, origin: str, imports: RustImports, type_key: str | None = None
    ) -> str | None:
        """Resolve a leaf origin to its in-scope Rust name (recording its ``use``), or ``None``.

        A ``ty-map`` entry for the type key the schema was parsed from wins over the origin.
        """
        key = type_key if type_key in self.ty_map else origin
        mapped = self.ty_map.get(key)
        if mapped is None:
            if "." not in origin or origin.startswith("ctypes."):
                return None
            if self._provider(origin) == "crate":  # a key nobody provides is recorded
                return None  # an `ffi.*` key off the default `ty-map`: no crate type, read as `Any`
            return imports.record(self._generated_type_path(origin))
        return self._record_mapped(key, mapped, imports)

    def _record_mapped(self, type_key: str, path: str, imports: RustImports) -> str:
        """Record the ``ty-map`` target of ``type_key``; one in this file's own module is in scope.

        A hand-written binding next to the generated ones (``ir.Op -> crate::ir::Op``
        in ``crate::ir``) must not import itself.
        """
        module, _, name = path.rpartition("::")
        if module == "::".join(self.module):
            return name
        return imports.record(path)

    def _ty_render(self, origin: str, type_key: str | None = None) -> str | None:
        return self._resolve(origin, self.imports, type_key)

    def _generated_type_path(self, type_key: str) -> str:
        """Spell a generated type key from this file.

        Same module: the bare leaf. Elsewhere: the ``crate::`` path of the module its block is in.
        """
        head, _, _ = type_key.partition(".")
        if head in C_RUST.RUST_MOD_MAP:
            return type_key
        type_leaf = type_key.rpartition(".")[2]
        if type_key not in self.declared:  # missing: reported once the body is built
            return type_leaf
        module = self.declared[type_key]
        if module == self.module:
            return type_leaf
        return "::".join((*module, type_leaf))

    def _provider(self, type_key: str) -> str | None:
        """Who provides the binding of ``type_key``; a key nobody does is recorded in :attr:`missing`.

        ``"crate"`` for builtin ``ffi.*`` types, ``"mapped"`` for a ``ty-map`` to a
        hand-written binding, ``"generated"`` for an ``object/`` block in this run.
        """
        if is_crate_type_key(type_key):
            return "crate"
        if type_key in self.ty_map:
            return "mapped"
        if type_key in self.declared:
            return "generated"
        self.missing.add(type_key)
        return None

    def _base_type(self) -> tuple[str, bool]:
        """Resolve the ``base`` struct and whether it is a generated or ``ty-map``'d parent.

        A builtin parent below ``ffi.Object`` is embedded as its header-only
        stand-in (see :meth:`RustImports.record_builtin_base`).
        """
        parent = self.info.parent_type_key
        if parent is not None:
            provider = self._provider(parent)
            if provider == "mapped":
                return self._record_mapped(parent, self.ty_map[parent] + "Obj", self.imports), True
            if provider != "crate":  # generated in this run, or missing (reported after the body)
                return self.imports.record(self._generated_type_path(parent) + "Obj"), True
        chain = [key for key in self.info.ancestors if key != C_RUST.RUST_ROOT_TYPE_KEY]
        if parent not in (None, C_RUST.RUST_ROOT_TYPE_KEY, *chain):
            chain.append(parent)
        assert all(self._provider(key) == "crate" for key in chain), (self.type_key, chain)
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
            if key != C_RUST.RUST_ROOT_TYPE_KEY and self._provider(key) == "crate"
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
            mirror = narrowed or render_rust_type(field, lambda o, k: self._resolve(o, imports, k))
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
        inner = render_rust_type(payload, lambda o, k: self._resolve(o, imports, k))
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
        allow = _allow_non_snake_case([name])

        enum = directives.enums.get(target)
        if enum is not None:
            return [
                *allow,
                f"pub fn {name}(&self) -> Result<{enum.name}> {{",
                f"    let raw: i64 = {getter}.get(self)?;",
                f"    {enum.name}::try_from(raw)",
                "}",
            ]
        override = directives.field_types.get(target)
        if override is not None:
            rust_type = self.imports.record(override) if "::" in override else override
        elif field.origin == "Any":
            rust_type = None  # `FieldGetter::get` does not take the infallible `Any` -> `Any`
        else:
            rust_type = render_rust_type(field, self._ty_render)
        if rust_type is None:
            any_type = self.imports.record("tvm_ffi::Any")
            return [
                *allow,
                f"pub fn {name}(&self) -> Result<{any_type}> {{",
                f"    {getter}.get_any(self)",
                "}",
            ]
        if target in directives.nullable and not rust_type.startswith("Option<"):
            rust_type = f"Option<{rust_type}>"
        return [
            *allow,
            f"pub fn {name}(&self) -> Result<{rust_type}> {{",
            f"    {getter}.get(self)",
            "}",
        ]

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
        targets: list[str] = []
        for key in self.info.ancestors:
            provider = self._provider(key)
            if provider == "mapped":
                targets.append(self._record_mapped(key, self.ty_map[key], self.imports))
            elif provider != "crate":
                targets.append(self.imports.record(self._generated_type_path(key)))
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
            *_allow_non_snake_case(rust_ident(f.name) for f in self.info.fields),
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
        if parent is not None and self._provider(parent) != "crate":
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
        self,
        head: str,
        params: list[tuple[str, str]],
        call: tuple[str, list[str]],
        init: list[str],
        *,
        generic: bool = False,
    ) -> list[str]:
        """Render ``<head>(<params>) -> Self { let <call>; Self { <init> } }`` in an ``impl`` block.

        ``generic`` takes a parameter that is not a scalar as ``impl Into`` (:func:`_param_type`).
        """
        plist = [
            f"{rust_ident(name)}: {_param_type(rust_type) if generic else rust_type}"
            for name, rust_type in params
        ]
        binding, args = call
        return [
            *[f"    {line}" for line in _allow_non_snake_case(rust_ident(n) for n, _ in params)],
            *_call_lines(f"    {head}(", plist, ") -> Self {"),
            *_call_lines(f"        let {binding}(", args, ");"),
            *_call_lines("        Self { ", init, " }"),
            "    }",
        ]

    def _allocator_sections(self, base: str, has_parent: bool) -> list[list[str]]:
        """``<Leaf>Obj::new`` and the wrapper's ``new``.

        ``<Leaf>Obj::new`` takes every field as its own type, so the parent's, generated
        or hand-written, takes what it hands on. The wrapper's ``new`` takes a scalar
        field as its own type and any other as ``impl Into`` (:func:`_param_type`), as
        ``#[rs_class]``'s ``new`` does, and converts it on the way to ``<Leaf>Obj::new``.
        ``custom-new`` names the wrapper's allocator ``from_fields`` instead.
        ``<Leaf>Obj::new`` of a type that is not final is public: a subtype defined in Rust
        builds its parent's object with it.
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
        own = [rust_ident(name) for name, _ in params[len(inherited) :]]
        forward = [_forward(rust_ident(name), rust_type) for name, rust_type in params]
        sections = [
            [
                f"impl {self.obj_struct} {{",
                *self._fn_lines(
                    "pub(crate) fn new" if self.info.is_final else "pub fn new",
                    params,
                    (f"base = {base}::new", to_parent),
                    ["base", *own],
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
                    "pub fn from_fields" if custom else "pub fn new",
                    params,
                    (f"obj = {self.obj_struct}::new", forward),
                    ["base: ObjectArc::new(obj)"],
                    generic=True,
                ),
                "}",
            ]
        )
        return sections

    def _shallow_copy_lines(self) -> list[str] | None:
        """``ShallowCopy``, which a subtype defined in Rust needs of its parent's object.

        The parent's shallow copy and a clone of every field, as ``derive(Object)``
        implements it, for a type that is not final and whose ancestors below
        ``ffi.Object`` are all generated: each of those implements it by this rule,
        while a ``ty-map``'d one is hand-written and may not.
        """
        if self.info.is_final:
            return None
        ancestors = [key for key in self.info.ancestors if key != C_RUST.RUST_ROOT_TYPE_KEY]
        if not all(self._provider(key) == "generated" for key in ancestors):
            return None
        shallow_copy = self.imports.record("tvm_ffi::ShallowCopy")
        own = [rust_ident(f.name) for f in sorted(self.info.fields, key=lambda f: f.offset or 0)]
        init = [
            f"base: {shallow_copy}::shallow_copy(&self.base)",
            *[f"{name}: Clone::clone(&self.{name})" for name in own],
        ]
        return [
            f"impl {shallow_copy} for {self.obj_struct} {{",
            "    fn shallow_copy(&self) -> Self {",
            *_call_lines("        Self { ", init, " }"),
            "    }",
            "}",
        ]

    def body(self) -> list[str]:
        """Build the Rust source lines for the object."""
        self.imports.reserved.update((self.leaf, self.obj_struct))
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
            shallow_copy = self._shallow_copy_lines()
            if shallow_copy:
                sections.append(shallow_copy)
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
    declared: Mapping[str, tuple[str, ...]],
) -> None:
    """Emit the Rust binding of ``obj_info`` into an ``object/<key>`` block.

    ``declared`` maps each type key with an ``object/`` block in this run, ``obj_info``'s
    own among them, to the module of the file the block is in. Raises ``ValueError`` when
    the binding refers to a type key this run does not provide.
    """
    assert len(code.lines) >= 2
    assert isinstance(obj_info.type_key, str)
    renderer = _ObjectRenderer(
        info=obj_info,
        imports=imports,
        ty_map=ty_map,
        module=declared[obj_info.type_key],
        declared=declared,
    )
    body = renderer.body()
    if renderer.missing:
        keys = ", ".join(f"`{key}`" for key in sorted(renderer.missing))
        raise ValueError(
            f"`{obj_info.type_key}` (line {code.lineno_start}) depends on {keys}, which this run "
            "does not ask for: add an `object/<key>` block or a `ty-map` directive for each"
        )
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
