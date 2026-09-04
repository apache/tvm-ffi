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
"""Tests for the Rust backend of ``tvm-ffi-stubgen``: complete and opaque bindings."""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import pytest
import tvm_ffi.stub.cli as stub_cli
import tvm_ffi.testing  # noqa: F401  (loads the `testing.*` fixture types)
from tvm_ffi.core import TypeSchema
from tvm_ffi.stub import consts as C
from tvm_ffi.stub.cli import _stage_3
from tvm_ffi.stub.file_utils import CodeBlock, FileInfo
from tvm_ffi.stub.generator import get_generator
from tvm_ffi.stub.lib_state import object_info_from_type_key
from tvm_ffi.stub.rust_generator import codegen
from tvm_ffi.stub.rust_generator import consts as RC
from tvm_ffi.stub.rust_generator.codegen import (
    finalize_rust_module_tree,
    generate_rust_api_file,
    generate_rust_import_section,
    generate_rust_object,
)
from tvm_ffi.stub.rust_generator.directives import Directives, EnumSpec
from tvm_ffi.stub.rust_generator.utils import RustImports, RustUse, render_rust_type, rust_ident
from tvm_ffi.stub.utils import DirectiveError, InitConfig, NamedTypeSchema, ObjectInfo, Options

RUST = get_generator("rust")
HEADER = 24  # sizeof(TVMFFIObject)


def _field(
    name: str,
    schema: str | TypeSchema,
    offset: int | None = None,
    size: int | None = None,
    alignment: int | None = None,
) -> NamedTypeSchema:
    if isinstance(schema, str):
        schema = TypeSchema(schema)
    if alignment is None and size is not None:
        alignment = min(size, 8)  # a 16-byte `TVMFFIAny` cell is 8-aligned
    return NamedTypeSchema(name, schema, offset=offset, size=size, alignment=alignment)


def _info(
    type_key: str,
    fields: tuple[NamedTypeSchema, ...] = (),
    *,
    parent: str | None = "ffi.Object",
    ancestors: list[str] | None = None,
    is_final: bool | None = None,
    total_size: int | None = None,
) -> ObjectInfo:
    if ancestors is None:
        ancestors = ["ffi.Object"] if parent in (None, "ffi.Object") else ["ffi.Object", parent]
    return ObjectInfo(
        fields=list(fields),
        methods=[],
        type_key=type_key,
        parent_type_key=parent,
        ancestors=ancestors,
        is_final=is_final,
        total_size=total_size,
    )


#: Ancestors the tests define with byte facts; anything else a test names but
#: does not define resolves to a type without metadata of its own.
_SYNTHETIC: dict[str, ObjectInfo] = {}


def _register(*infos: ObjectInfo) -> None:
    for info in infos:
        assert info.type_key is not None
        _SYNTHETIC[info.type_key] = info


@pytest.fixture(autouse=True)
def _synthetic_registry(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    def lookup(type_key: str) -> ObjectInfo:
        if type_key in _SYNTHETIC:
            return _SYNTHETIC[type_key]
        if type_key.startswith(("ffi.", "testing.")):
            return object_info_from_type_key(type_key)
        return _info(type_key, total_size=None)

    monkeypatch.setattr(codegen, "object_info_from_type_key", lookup)
    yield
    _SYNTHETIC.clear()


def _object_block(type_key: str) -> CodeBlock:
    return CodeBlock(
        kind="object",
        param=type_key,
        lineno_start=1,
        lineno_end=2,
        lines=[f"{C.RUST_SYNTAX.begin} object/{type_key}", C.RUST_SYNTAX.end],
    )


def _render(info: ObjectInfo, imports: RustImports | None = None) -> tuple[str, RustImports]:
    """Render ``info`` into a fresh object block; return the body text and the collector."""
    imports = RustImports() if imports is None else imports
    assert info.type_key is not None
    block = _object_block(info.type_key)
    generate_rust_object(block, RUST.default_ty_map(), imports, Options(), info)
    return "\n".join(block.lines[1:-1]), imports


def _uses(imports: RustImports) -> set[str]:
    return {item.path for item in imports.items}


NO_METADATA = "/// Opaque: no metadata of its own: total_size is unknown. Fields are read through the C ABI getters."


# ---------------------------------------------------------------------------
# `use` modelling and type rendering
# ---------------------------------------------------------------------------


def test_rustuse_paths() -> None:
    assert RustUse("tvm_ffi::Array").path == "tvm_ffi::Array"
    assert RustUse("tvm_ffi::Array").leaf == "Array"
    assert RustUse("tvm_ffi::Array").as_use_line() == "use tvm_ffi::Array;"
    # A dotted FFI name becomes a path; the builtin `ffi` module maps to the crate root.
    assert RustUse("ffi.String").path == "tvm_ffi::String"
    assert RustUse("my_pkg.sub.Foo").as_use_line() == "use my_pkg::sub::Foo;"
    # Bare names need no `use`.
    assert RustUse("i64").as_use_line() == ""


def test_rustimports_record_dedups_and_resolves_collisions() -> None:
    imports = RustImports()
    assert imports.record("i64") == "i64"
    assert imports.record("tvm_ffi::Array") == "Array"
    assert imports.record("tvm_ffi::Array") == "Array"
    assert imports.items == [RustUse("tvm_ffi::Array")]
    # A second path wanting an already-claimed leaf is spelled in full, and not recorded.
    assert imports.record("other::Array") == "other::Array"
    assert imports.items == [RustUse("tvm_ffi::Array")]


def _render_type(schema: TypeSchema) -> tuple[str | None, RustImports]:
    imports = RustImports()
    ty_map = RC.RUST_TY_MAP_DEFAULTS

    def ty_render(origin: str) -> str | None:
        return imports.record(ty_map[origin]) if origin in ty_map else None

    return render_rust_type(schema, ty_render), imports


def test_render_rust_type_value_positions() -> None:
    assert _render_type(TypeSchema("int"))[0] == "i64"
    assert _render_type(TypeSchema("str"))[0] == "String"
    assert _render_type(TypeSchema("Any"))[0] == "Any"
    assert _render_type(TypeSchema("Callable", (TypeSchema("int"),)))[0] == "Function"
    assert _render_type(TypeSchema("Optional", (TypeSchema("str"),)))[0] == "Option<String>"
    text, imports = _render_type(TypeSchema("Map", (TypeSchema("str"), TypeSchema("Array"))))
    assert text == "Map<String, Array<Any>>"
    assert _uses(imports) == {"tvm_ffi::Map", "tvm_ffi::String", "tvm_ffi::Array", "tvm_ffi::Any"}


@pytest.mark.parametrize(
    "schema",
    [
        TypeSchema("Union", (TypeSchema("int"), TypeSchema("str"))),
        TypeSchema("Dict", (TypeSchema("str"), TypeSchema("int"))),
        TypeSchema("tuple"),
        TypeSchema("Array", (TypeSchema("List", (TypeSchema("int"),)),)),
        TypeSchema("Optional", (TypeSchema("ctypes.c_void_p"),)),
    ],
)
def test_render_rust_type_without_mirror(schema: TypeSchema) -> None:
    assert _render_type(schema)[0] is None


def test_rust_ident() -> None:
    assert rust_ident("value") == "value"
    assert rust_ident("imports_") == "imports"
    assert rust_ident("type") == "r#type"
    assert rust_ident("self") == "self_"
    assert rust_ident("crate") == "crate_"


# ---------------------------------------------------------------------------
# Directives
# ---------------------------------------------------------------------------


def test_directives_parse() -> None:
    directives = Directives()
    directives.add("field", " tirx.Add.a ->  PrimExpr ", 1)
    directives.add("nullable", "ir.Expr.span", 2)
    directives.add("enum", "tirx.For.kind -> ForKind(i32) { Serial=0, Parallel = 1 }", 3)
    directives.add("enum", "tirx.For.mode -> Mode(u8)", 4)
    directives.add("opaque", " ir.SourceName ", 5)
    assert directives.field_types == {"tirx.Add.a": "PrimExpr"}
    assert directives.nullable == {"ir.Expr.span"}
    assert directives.enums == {
        "tirx.For.kind": EnumSpec("ForKind", "i32", (("Serial", 0), ("Parallel", 1))),
        "tirx.For.mode": EnumSpec("Mode", "u8", ()),
    }
    assert directives.opaque == {"ir.SourceName"}


@pytest.mark.parametrize(
    ("name", "payload", "expected"),
    [
        ("field", "tirx.Add.a", "-> <RustType>"),
        ("field", "tirx.Add.a ->", "-> <RustType>"),
        ("field", "Add -> PrimExpr", "<type_key>.<field>"),
        ("nullable", "ir.Expr.span extra", "<type_key>.<field>"),
        ("enum", "tirx.For.kind -> ForKind", "Name(i32)"),
        ("enum", "tirx.For.kind -> ForKind(i128)", "Name(i32)"),
        ("enum", "tirx.For.kind -> ForKind(i32) { Serial }", "Name(i32)"),
        ("opaque", "ir.SourceName ir.Source", "<type_key>"),
        ("upcast", "tirx.Add -> PrimExpr", "Unknown directive"),
    ],
)
def test_directives_reject_malformed(name: str, payload: str, expected: str) -> None:
    with pytest.raises(DirectiveError, match=re.escape(expected)) as exc:
        Directives().add(name, payload, 7)
    assert "at line 7" in str(exc.value)


def test_generator_declares_its_directives_and_records_imports() -> None:
    assert RUST.directive_kinds == {"import-object", "field", "nullable", "enum", "opaque"}
    imports = RUST.new_imports()
    RUST.add_directive(imports, "import-object", "tvm_ffi.libinfo.Foo;False;_Foo", 1)
    RUST.add_directive(imports, "nullable", "demo.Node.span", 2)
    assert imports.items == [RustUse("tvm_ffi::libinfo::Foo")]
    assert imports.directives.nullable == {"demo.Node.span"}


# ---------------------------------------------------------------------------
# Opaque rendering (no byte facts: the layout cannot be proven)
# ---------------------------------------------------------------------------

ROOT_EXPECTED = f"""\
{NO_METADATA}
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "demo.Pair"]
pub struct PairObj {{
    base: Object,
}}

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Pair {{
    data: ObjectArc<PairObj>,
}}

impl Deref for Pair {{
    type Target = PairObj;
    fn deref(&self) -> &PairObj {{
        &self.data
    }}
}}

impl PairObj {{
    pub fn a(&self) -> Result<i64> {{
        FieldGetter::new(Self::type_index(), "a")?.get(self)
    }}

    pub fn tag(&self) -> Result<Option<String>> {{
        FieldGetter::new(Self::type_index(), "tag")?.get(self)
    }}

    pub fn items(&self) -> Result<Array<Any>> {{
        FieldGetter::new(Self::type_index(), "items")?.get(self)
    }}

    pub fn owner(&self) -> Result<ObjectRef> {{
        FieldGetter::new(Self::type_index(), "owner")?.get(self)
    }}

    pub fn r#type(&self) -> Result<Any> {{
        FieldGetter::new(Self::type_index(), "type")?.get_any(self)
    }}
}}"""


def test_render_root_object() -> None:
    """A root type embeds the header, gets no parent `Deref` and no upcast."""
    info = _info(
        "demo.Pair",
        (
            _field("a", "int"),
            _field("tag", TypeSchema("Optional", (TypeSchema("str"),))),
            _field("items", "Array"),
            _field("owner", "Object"),
            _field("type", TypeSchema("Union", (TypeSchema("int"), TypeSchema("str")))),
        ),
    )
    text, imports = _render(info)
    assert text == ROOT_EXPECTED
    assert _uses(imports) == {
        "std::ops::Deref",
        "tvm_ffi::Object",
        "tvm_ffi::ObjectArc",
        "tvm_ffi::ObjectCore",
        "tvm_ffi::FieldGetter",
        "tvm_ffi::Result",
        "tvm_ffi::String",
        "tvm_ffi::Array",
        "tvm_ffi::Any",
        "tvm_ffi::object::ObjectRef",
    }


def test_render_object_without_fields_has_no_impl() -> None:
    text, imports = _render(_info("demo.Marker", is_final=True))
    assert "#[type_final]" in text
    assert "impl MarkerObj" not in text
    assert "FieldGetter" not in _uses(imports)


def test_render_derived_object_same_module() -> None:
    """A generated parent is embedded, dereferenced to, and upcast to along the chain."""
    info = _info(
        "demo.Add",
        (_field("a", "demo.Expr"),),
        parent="demo.Expr",
        ancestors=["ffi.Object", "demo.BaseExpr", "demo.Expr"],
        is_final=True,
    )
    text, imports = _render(info)
    assert "#[type_final]\npub struct AddObj {\n    base: ExprObj,\n}" in text
    assert "impl Deref for AddObj {\n    type Target = ExprObj;" in text
    assert "pub fn a(&self) -> Result<Expr> {" in text
    assert text.endswith("tvm_ffi::impl_object_upcast!(Add => BaseExpr, Add => Expr);")
    # Same-module names are local items: nothing to `use`.
    assert not any(path.startswith("demo") for path in _uses(imports))


def test_render_derived_object_cross_module() -> None:
    """A parent in another module is reached through the generated root."""
    info = _info("tirx.Add", parent="ir.Expr", ancestors=["ffi.Object", "ir.Expr"])
    text, imports = _render(info)
    assert "    base: ExprObj," in text
    assert text.endswith("tvm_ffi::impl_object_upcast!(Add => Expr);")
    assert {"super::ir::ExprObj", "super::ir::Expr"} <= _uses(imports)


def test_render_object_under_builtin_parent() -> None:
    """A builtin parent is embedded via header-only stand-ins so `TYPE_DEPTH` matches the registry."""
    info = _info(
        "demo.Color",
        (_field("value", "int"),),
        parent="ffi.IntEnum",
        ancestors=["ffi.Object", "ffi.Enum", "ffi.IntEnum"],
    )
    text, imports = _render(info)
    assert "    base: FfiIntEnumObj," in text
    # One stand-in per builtin ancestor below `ffi.Object`, chained through `base`.
    assert imports.builtin_mirrors == {"ffi.Enum": "Object", "ffi.IntEnum": "FfiEnumObj"}
    assert "tvm_ffi::Object" in _uses(imports)
    # No Deref, upcast or crate import for the stand-ins.
    assert "impl Deref for ColorObj" not in text
    assert "impl_object_upcast" not in text
    assert not any("Enum" in path for path in _uses(imports))
    # A child of `ffi.Object` embeds the crate's header.
    text, imports = _render(_info("demo.Root"))
    assert "    base: Object," in text
    assert imports.builtin_mirrors == {}
    # A generated parent is embedded by name.
    red = _info(
        "demo.Red",
        parent="demo.Color",
        ancestors=["ffi.Object", "ffi.Enum", "ffi.IntEnum", "demo.Color"],
    )
    text, imports = _render(red)
    assert "    base: ColorObj," in text
    assert imports.builtin_mirrors == {}


BUILTIN_MIRRORS_EXPECTED = """\
use std::ops::Deref;
use tvm_ffi::Object;
use tvm_ffi::ObjectArc;

/// Header-only stand-in for the builtin `ffi.Enum`; it only carries the ancestor depth.
#[allow(dead_code)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ffi.Enum"]
struct FfiEnumObj {
    base: Object,
}

/// Header-only stand-in for the builtin `ffi.IntEnum`; it only carries the ancestor depth.
#[allow(dead_code)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ffi.IntEnum"]
struct FfiIntEnumObj {
    base: FfiEnumObj,
}

/// Header-only stand-in for the builtin `ffi.StrEnum`; it only carries the ancestor depth.
#[allow(dead_code)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ffi.StrEnum"]
struct FfiStrEnumObj {
    base: FfiEnumObj,
}"""


def test_import_section_defines_builtin_mirrors_once() -> None:
    """Objects sharing builtin ancestors share one mirror chain, rendered after the `use`s."""
    imports = RustImports()
    enum_chain = ["ffi.Object", "ffi.Enum"]
    for type_key, parent in (
        ("demo.Color", "ffi.IntEnum"),
        ("demo.Mode", "ffi.IntEnum"),
        ("demo.Flag", "ffi.Enum"),
        ("demo.Op", "ffi.StrEnum"),
    ):
        ancestors = enum_chain if parent == "ffi.Enum" else [*enum_chain, parent]
        _render(_info(type_key, parent=parent, ancestors=ancestors), imports)
    block = CodeBlock(
        kind="import-section",
        param="",
        lineno_start=1,
        lineno_end=2,
        lines=[f"{C.RUST_SYNTAX.begin} import-section", C.RUST_SYNTAX.end],
    )
    generate_rust_import_section(block, imports, Options(), defined_types=set())
    assert "\n".join(block.lines[1:-1]) == BUILTIN_MIRRORS_EXPECTED


ITER_VAR_EXPECTED = f"""\
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct IterVarType(i32);

#[allow(non_upper_case_globals)]
impl IterVarType {{
    pub const kDataPar: Self = Self(0);
    pub const kThreadIndex: Self = Self(1);
    pub const fn from_raw(value: i32) -> Self {{
        Self(value)
    }}
    pub const fn as_raw(self) -> i32 {{
        self.0
    }}
}}

impl TryFrom<i64> for IterVarType {{
    type Error = Error;
    fn try_from(value: i64) -> Result<Self> {{
        i32::try_from(value).map(Self).map_err(|_| {{
            Error::new(VALUE_ERROR, &format!("IterVarType value {{value}} does not fit i32"), "")
        }})
    }}
}}

{NO_METADATA}
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.IterVar"]
#[type_final]
pub struct IterVarObj {{
    base: PrimExprConvertibleObj,
}}

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct IterVar {{
    data: ObjectArc<IterVarObj>,
}}

impl Deref for IterVar {{
    type Target = IterVarObj;
    fn deref(&self) -> &IterVarObj {{
        &self.data
    }}
}}

impl Deref for IterVarObj {{
    type Target = PrimExprConvertibleObj;
    fn deref(&self) -> &PrimExprConvertibleObj {{
        &self.base
    }}
}}

impl IterVarObj {{
    pub fn dom(&self) -> Result<Option<Range>> {{
        FieldGetter::new(Self::type_index(), "dom")?.get(self)
    }}

    pub fn var(&self) -> Result<PrimVar> {{
        FieldGetter::new(Self::type_index(), "var")?.get(self)
    }}

    pub fn iter_type(&self) -> Result<IterVarType> {{
        let raw: i64 = FieldGetter::new(Self::type_index(), "iter_type")?.get(self)?;
        IterVarType::try_from(raw)
    }}

    pub fn thread_tag(&self) -> Result<String> {{
        FieldGetter::new(Self::type_index(), "thread_tag")?.get(self)
    }}

    pub fn span(&self) -> Result<Option<Span>> {{
        FieldGetter::new(Self::type_index(), "span")?.get(self)
    }}
}}

tvm_ffi::impl_object_upcast!(IterVar => PrimExprConvertible);"""


def test_render_iter_var_golden() -> None:
    """The shape tvm-rust-ext hand-writes for its polymorphic `IterVar`, driven by directives."""
    info = _info(
        "tirx.IterVar",
        (
            _field("dom", "ir.Range"),
            _field("var", "ir.Var"),
            _field("iter_type", "int"),
            _field("thread_tag", "str"),
            _field("span", "ir.Span"),
        ),
        parent="ir.PrimExprConvertible",
        is_final=True,
    )
    imports = RUST.new_imports()
    RUST.add_directive(imports, "nullable", "tirx.IterVar.dom", 1)
    RUST.add_directive(imports, "field", "tirx.IterVar.var -> PrimVar", 2)
    RUST.add_directive(imports, "nullable", "tirx.IterVar.span", 3)
    RUST.add_directive(
        imports,
        "enum",
        "tirx.IterVar.iter_type -> IterVarType(i32) { kDataPar=0, kThreadIndex=1 }",
        4,
    )
    text, imports = _render(info, imports)
    assert text == ITER_VAR_EXPECTED
    assert {
        "tvm_ffi::Error",
        "tvm_ffi::VALUE_ERROR",
        "super::ir::Range",
        "super::ir::Span",
    } <= _uses(imports)


def test_field_directive_with_path_records_use_and_nullable_does_not_double_wrap() -> None:
    info = _info(
        "demo.Node",
        (
            _field("buffer", "demo.Var"),
            _field("dom", TypeSchema("Optional", (TypeSchema("int"),))),
        ),
    )
    imports = RUST.new_imports()
    RUST.add_directive(imports, "field", "demo.Node.buffer -> crate::typed::BufferVar", 1)
    RUST.add_directive(imports, "nullable", "demo.Node.dom", 2)
    text, imports = _render(info, imports)
    assert "pub fn buffer(&self) -> Result<BufferVar> {" in text
    assert "pub fn dom(&self) -> Result<Option<i64>> {" in text
    assert "crate::typed::BufferVar" in _uses(imports)


# ---------------------------------------------------------------------------
# Complete rendering (byte facts prove the layout)
# ---------------------------------------------------------------------------


def _span() -> ObjectInfo:
    return _info(
        "ir.Span",
        (
            _field("source_name", "ir.SourceName", 24, 8),
            _field("line", "int", 32, 4),
            _field("column", "int", 36, 4),
            _field("end_line", "int", 40, 4),
            _field("end_column", "int", 44, 4),
        ),
        total_size=48,
    )


def _expr() -> ObjectInfo:
    return _info(
        "ir.Expr",
        (_field("span", "ir.Span", 24, 8), _field("ty", "ir.Type", 32, 8)),
        total_size=40,
    )


def _add() -> ObjectInfo:
    return _info(
        "tirx.Add",
        (_field("a", "ir.Expr", 40, 8), _field("b", "ir.Expr", 48, 8)),
        parent="ir.Expr",
        total_size=56,
        is_final=True,
    )


EXPR_EXPECTED = """\
/// Complete: reflected fields fill [24, 40) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.Expr"]
pub struct ExprObj {
    base: Object,
    pub span: Option<Span>,
    pub ty: Type,
}

const _: () = {
    assert!(::core::mem::size_of::<ExprObj>() == 40);
    assert!(::core::mem::align_of::<ExprObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Expr {
    data: ObjectArc<ExprObj>,
}

impl Deref for Expr {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.data
    }
}"""

ADD_EXPECTED = """\
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.Add"]
#[type_final]
pub struct AddObj {
    base: ExprObj,
    pub a: PrimExpr,
    pub b: PrimExpr,
}

const _: () = {
    assert!(::core::mem::size_of::<AddObj>() == 56);
    assert!(::core::mem::align_of::<AddObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Add {
    data: ObjectArc<AddObj>,
}

impl Deref for Add {
    type Target = AddObj;
    fn deref(&self) -> &AddObj {
        &self.data
    }
}

impl Deref for AddObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

tvm_ffi::impl_object_upcast!(Add => Expr);"""


def test_render_complete_expr_golden() -> None:
    """`ir.Expr` as tvm-rust-ext hand-writes it: `span: Option<Span>`, `ty: Type`, no getters."""
    imports = RUST.new_imports()
    RUST.add_directive(imports, "nullable", "ir.Expr.span", 1)
    text, imports = _render(_expr(), imports)
    assert text == EXPR_EXPECTED
    assert _uses(imports) == {"std::ops::Deref", "tvm_ffi::Object", "tvm_ffi::ObjectArc"}


def test_render_complete_add_golden() -> None:
    """`tirx.Add` on top of a complete `ir.Expr`, with `field` directives narrowing `a` / `b`."""
    _register(_expr())
    imports = RUST.new_imports()
    RUST.add_directive(imports, "field", "tirx.Add.a -> PrimExpr", 1)
    RUST.add_directive(imports, "field", "tirx.Add.b -> PrimExpr", 2)
    text, imports = _render(_add(), imports)
    assert text == ADD_EXPECTED
    assert _uses(imports) == {
        "std::ops::Deref",
        "tvm_ffi::ObjectArc",
        "super::ir::ExprObj",
        "super::ir::Expr",
    }


def test_render_complete_span_and_prim_type() -> None:
    """Scalars take their reflected width; alignment padding is reported, not mirrored."""
    text, _ = _render(_span())
    assert (
        "pub struct SpanObj {\n"
        "    base: Object,\n"
        "    pub source_name: SourceName,\n"
        "    pub line: i32,\n"
        "    pub column: i32,\n"
        "    pub end_line: i32,\n"
        "    pub end_column: i32,\n"
        "}"
    ) in text
    assert "/// Complete: reflected fields fill [24, 48) exactly." in text

    _register(_info("ir.Type", (_field("span", "ir.Span", 24, 8),), total_size=32))
    prim_type = _info(
        "ir.PrimType",
        (_field("dtype", "dtype", 32, 4, 2),),
        parent="ir.Type",
        total_size=40,
        is_final=True,
    )
    text, imports = _render(prim_type)
    assert (
        "/// Complete: reflected fields fill [32, 40) exactly (alignment padding [36, 40))." in text
    )
    assert "pub struct PrimTypeObj {\n    base: TypeObj,\n    pub dtype: DLDataType,\n}" in text
    assert "assert!(::core::mem::size_of::<PrimTypeObj>() == 40);" in text
    assert "tvm_ffi::DLDataType" in _uses(imports)


def test_complete_optional_field_mirrors() -> None:
    """`Optional<T>` fields mirror their C++ layout: a 16-byte cell or a nullable pointer."""
    info = _info(
        "demo.Opt",
        (
            _field("count", TypeSchema("Optional", (TypeSchema("int"),)), 24, 16),
            _field("name", TypeSchema("Optional", (TypeSchema("str"),)), 40, 16),
            _field("items", TypeSchema("Optional", (TypeSchema("Array"),)), 56, 8),
        ),
        total_size=64,
    )
    text, imports = _render(info)
    assert "pub count: Optional<i64>," in text
    assert "pub name: Optional<String>," in text
    assert "pub items: Option<Array<Any>>," in text
    assert "tvm_ffi::Optional" in _uses(imports)


@pytest.mark.parametrize(
    "field",
    [
        _field("x", TypeSchema("Optional", (TypeSchema("Any"),)), 24, 16),
        _field("x", TypeSchema("Optional", (TypeSchema("int"),)), 24, 8),  # not the 16-byte cell
        _field("x", TypeSchema("Union", (TypeSchema("int"), TypeSchema("str"))), 24, 16),
        _field("x", "ctypes.c_void_p", 24, 8),
    ],
)
def test_unrenderable_field_keeps_the_type_opaque(field: NamedTypeSchema) -> None:
    """A field without a mirror demotes an otherwise complete type; it is read as `Any`."""
    assert field.size is not None
    info = _info("demo.Holder", (field,), total_size=HEADER + field.size)
    text, _ = _render(info)
    assert "/// Opaque: field 'x'" in text
    assert "has no native mirror" in text
    assert "impl HolderObj {\n    pub fn x(&self) -> Result<" in text
    assert "const _: () =" not in text


def test_opaque_directive_vetoes_a_complete_type() -> None:
    imports = RUST.new_imports()
    RUST.add_directive(imports, "opaque", "ir.Span", 1)
    text, _ = _render(_span(), imports)
    assert "/// Opaque: vetoed by directive although the layout is reproducible." in text
    assert "pub fn line(&self) -> Result<i64> {" in text


def test_opaque_parent_keeps_the_child_opaque() -> None:
    """A child of an opaque type cannot embed a mirror of its parent: it stays opaque."""
    _register(_info("ir.Expr", (_field("span", "ir.Span", 24, 8),), total_size=40))  # hole
    text, _ = _render(_add())
    assert "/// Opaque: parent 'ir.Expr' is opaque (uncovered-bytes)." in text
    assert "    base: ExprObj," in text
    assert "pub fn a(&self) -> Result<Expr> {" in text


@pytest.mark.parametrize(
    ("name", "payload", "message"),
    [
        ("field", "demo.Pair.count -> i64", "maps a 4-byte field to `i64` (8 bytes)"),
        ("enum", "demo.Pair.count -> Kind(i64)", "maps a 4-byte field to `i64` (8 bytes)"),
        ("nullable", "demo.Pair.count", "the field is 4 bytes, not a pointer-sized"),
    ],
)
def test_directive_disagreeing_with_bytes_is_an_error(
    name: str, payload: str, message: str
) -> None:
    info = _info("demo.Pair", (_field("count", "int", 24, 4),), total_size=32)
    imports = RUST.new_imports()
    RUST.add_directive(imports, name, payload, 1)
    with pytest.raises(DirectiveError, match=re.escape(message)):
        _render(info, imports)


# ---------------------------------------------------------------------------
# File scaffolding
# ---------------------------------------------------------------------------


def test_import_section_dedups_sorts_and_filters_defined_types() -> None:
    imports = RustImports()
    for name in ("tvm_ffi::ObjectArc", "std::ops::Deref", "tvm_ffi::ObjectArc", "super::ir::Expr"):
        imports.record(name)
    block = CodeBlock(
        kind="import-section",
        param="",
        lineno_start=1,
        lineno_end=2,
        lines=[f"{C.RUST_SYNTAX.begin} import-section", C.RUST_SYNTAX.end],
    )
    generate_rust_import_section(block, imports, Options(), defined_types={"super::ir::Expr"})
    assert block.lines[1:-1] == ["use std::ops::Deref;", "use tvm_ffi::ObjectArc;"]


def test_api_file_scaffold() -> None:
    infos = [_info("demo.A"), _info("demo.B")]
    cfg = InitConfig(pkg="demo", shared_target="demo_shared", prefix="demo.")
    text = generate_rust_api_file([], {}, "demo", infos, cfg, True, C.RUST_SYNTAX)
    assert text.startswith("#![allow(dead_code, unused_imports)]\n")
    assert "//! FFI bindings for `demo` (generated by tvm-ffi-stubgen)." in text
    assert f"{C.RUST_SYNTAX.begin} import-section\n{C.RUST_SYNTAX.end}" in text
    assert text.count(f"{C.RUST_SYNTAX.begin} object/demo.") == 2
    # Existing blocks are not re-scaffolded.
    again = generate_rust_api_file(
        [_object_block("demo.A")], {}, "demo", infos, cfg, True, C.RUST_SYNTAX
    )
    assert "object/demo.A" not in again and "object/demo.B" in again
    assert RUST.api_filename() == RUST.init_filename() == "mod.rs"
    assert RUST.generate_init_file([], "demo", "mod") == ""


def test_finalize_module_tree(tmp_path: Path) -> None:
    (tmp_path / "ir").mkdir()
    (tmp_path / "ir" / "mod.rs").write_text("pub struct Existing;\n", encoding="utf-8")
    finalize_rust_module_tree(tmp_path, {"ir", "tirx.transform"})
    assert (tmp_path / "mod.rs").read_text(encoding="utf-8") == "pub mod ir;\npub mod tirx;\n"
    assert (tmp_path / "tirx" / "mod.rs").read_text(encoding="utf-8") == "pub mod transform;\n"
    assert (tmp_path / "ir" / "mod.rs").read_text(encoding="utf-8") == "pub struct Existing;\n"
    finalize_rust_module_tree(tmp_path, {"ir", "tirx.transform"})  # idempotent
    assert (tmp_path / "mod.rs").read_text(encoding="utf-8") == "pub mod ir;\npub mod tirx;\n"


# ---------------------------------------------------------------------------
# The real registry, and the pipeline end to end
# ---------------------------------------------------------------------------


def test_registry_complete_chain_is_mirrored() -> None:
    base, _ = _render(object_info_from_type_key("testing.TestCxxClassBase"))
    assert (
        "/// Complete: reflected fields fill [24, 40) exactly (alignment padding [36, 40))." in base
    )
    assert (
        "pub struct TestCxxClassBaseObj {\n    base: Object,\n    pub v_i64: i64,\n    pub v_i32: i32,\n}"
        in base
    )
    assert "assert!(::core::mem::size_of::<TestCxxClassBaseObj>() == 40);" in base
    assert "FieldGetter" not in base

    dd, imports = _render(object_info_from_type_key("testing.TestCxxClassDerivedDerived"))
    assert (
        "    base: TestCxxClassDerivedObj,\n    pub v_str: String,\n    pub v_bool: bool,\n}" in dd
    )
    assert dd.endswith(
        "tvm_ffi::impl_object_upcast!(TestCxxClassDerivedDerived => TestCxxClassBase, "
        "TestCxxClassDerivedDerived => TestCxxClassDerived);"
    )
    assert "tvm_ffi::String" in _uses(imports)


def test_registry_hidden_field_and_vptr_stay_opaque() -> None:
    hidden, _ = _render(object_info_from_type_key("testing.TestCxxClassHiddenField"))
    assert (
        "/// Opaque: bytes [32, 40) of [24, 48) are not accounted for by reflected fields. "
        "Fields are read through the C ABI getters."
    ) in hidden
    assert "pub fn v_i32(&self) -> Result<i64> {" in hidden
    poly, _ = _render(object_info_from_type_key("testing.TestCxxClassPolymorphic"))
    assert "/// Opaque: bytes [32, 40) of [24, 40)" in poly


def test_stage_3_applies_directives_to_a_registered_type(tmp_path: Path) -> None:
    src = tmp_path / "mod.rs"
    src.write_text(
        "\n".join(
            [
                f"{C.RUST_SYNTAX.begin} import-section",
                C.RUST_SYNTAX.end,
                f"{C.RUST_SYNTAX.directive('enum')} testing.TestCxxClassBase.v_i32 -> Kind(i32) {{ A=0 }}",
                f"{C.RUST_SYNTAX.begin} object/testing.TestCxxClassBase",
                C.RUST_SYNTAX.end,
                "",
            ]
        ),
        encoding="utf-8",
    )
    info = FileInfo.from_file(src)
    assert info is not None
    _stage_3(info, Options(dry_run=True), RUST.default_ty_map(), {}, RUST)
    text = "\n".join(line for block in info.code_blocks for line in block.lines)
    assert "pub struct Kind(i32);" in text
    assert "    pub v_i32: Kind,\n" in text
    assert "use tvm_ffi::Error;" in text


def test_cli_init_generates_a_module_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "tvm-ffi-stubgen",
            "--target",
            "rust",
            "--init-pypkg",
            "demo",
            "--init-lib",
            "demo_shared",
            "--init-prefix",
            "testing.",
            str(tmp_path),
        ],
    )
    assert stub_cli.__main__() == 0
    assert (tmp_path / "mod.rs").read_text(encoding="utf-8") == "pub mod testing;\n"
    text = (tmp_path / "testing" / "mod.rs").read_text(encoding="utf-8")
    assert text.startswith("#![allow(dead_code, unused_imports)]\n")
    assert "use tvm_ffi::FieldGetter;" in text  # the opaque fixtures read through getters
    assert '#[type_key = "testing.TestCxxClassDerivedDerived"]' in text
    assert "    base: TestCxxClassDerivedObj,\n    pub v_str: String," in text
    assert (
        "tvm_ffi::impl_object_upcast!(TestCxxClassDerivedDerived => TestCxxClassBase, "
        "TestCxxClassDerivedDerived => TestCxxClassDerived);"
    ) in text
    # Builtin ancestors are mirrored once, in the import section; enum fixtures embed the last one.
    assert text.count("struct FfiEnumObj {\n    base: Object,\n}") == 1
    assert text.count("struct FfiIntEnumObj {\n    base: FfiEnumObj,\n}") == 1
    assert text.count("struct FfiStrEnumObj {\n    base: FfiEnumObj,\n}") == 1
    assert text.index("struct FfiIntEnumObj") < text.index(f"{C.RUST_SYNTAX.begin} object/")
    assert "pub struct TestEnumVariantObj {\n    base: FfiEnumObj,\n}" in text
    assert "pub struct TestCxxIntEnumObj {\n    base: FfiIntEnumObj,\n}" in text
    assert "pub struct TestCxxStrEnumObj {\n    base: FfiStrEnumObj,\n}" in text
    # Running again over the generated tree is a no-op.
    assert stub_cli.__main__() == 0
    assert (tmp_path / "testing" / "mod.rs").read_text(encoding="utf-8") == text


def test_cli_exits_non_zero_on_a_directive_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "mod.rs"
    src.write_text(
        "\n".join(
            [
                f"{C.RUST_SYNTAX.directive('field')} testing.TestCxxClassBase.v_i32 -> i64",
                f"{C.RUST_SYNTAX.begin} object/testing.TestCxxClassBase",
                C.RUST_SYNTAX.end,
                "",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("sys.argv", ["tvm-ffi-stubgen", "--target", "rust", str(src)])
    assert stub_cli.__main__() == 1
    src.write_text(f"{C.RUST_SYNTAX.directive('upcast')} testing.TestCxxClassBase -> X\n")
    assert stub_cli.__main__() == 1
