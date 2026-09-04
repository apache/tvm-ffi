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
"""Tests for the Rust backend of ``tvm-ffi-stubgen``: opaque bindings."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import tvm_ffi.stub.cli as stub_cli
import tvm_ffi.testing  # noqa: F401  (loads the `testing.*` fixture types)
from tvm_ffi.core import TypeSchema
from tvm_ffi.stub import consts as C
from tvm_ffi.stub.cli import _stage_3
from tvm_ffi.stub.file_utils import CodeBlock, FileInfo
from tvm_ffi.stub.generator import get_generator
from tvm_ffi.stub.rust_generator import consts as RC
from tvm_ffi.stub.rust_generator.codegen import (
    finalize_rust_module_tree,
    generate_rust_api_file,
    generate_rust_import_section,
    generate_rust_object,
)
from tvm_ffi.stub.rust_generator.directives import Directives, EnumSpec
from tvm_ffi.stub.rust_generator.utils import RustImports, RustUse, render_rust_type, rust_ident
from tvm_ffi.stub.utils import InitConfig, NamedTypeSchema, ObjectInfo, Options

RUST = get_generator("rust")


def _info(
    type_key: str,
    fields: tuple[tuple[str, TypeSchema], ...] = (),
    *,
    parent: str | None = "ffi.Object",
    ancestors: list[str] | None = None,
    is_final: bool | None = None,
) -> ObjectInfo:
    if ancestors is None:
        ancestors = ["ffi.Object"] if parent in (None, "ffi.Object") else ["ffi.Object", parent]
    return ObjectInfo(
        fields=[NamedTypeSchema(name, schema) for name, schema in fields],
        methods=[],
        type_key=type_key,
        parent_type_key=parent,
        ancestors=ancestors,
        is_final=is_final,
    )


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


# ---------------------------------------------------------------------------
# Directives
# ---------------------------------------------------------------------------


def test_directives_parse() -> None:
    directives = Directives()
    directives.add("field", " tirx.Add.a ->  PrimExpr ", 1)
    directives.add("nullable", "ir.Expr.span", 2)
    directives.add("enum", "tirx.For.kind -> ForKind(i32) { Serial=0, Parallel = 1 }", 3)
    directives.add("enum", "tirx.For.mode -> Mode(u8)", 4)
    assert directives.field_types == {"tirx.Add.a": "PrimExpr"}
    assert directives.nullable == {"ir.Expr.span"}
    assert directives.enums == {
        "tirx.For.kind": EnumSpec("ForKind", "i32", (("Serial", 0), ("Parallel", 1))),
        "tirx.For.mode": EnumSpec("Mode", "u8", ()),
    }


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
        ("upcast", "tirx.Add -> PrimExpr", "Unknown directive"),
    ],
)
def test_directives_reject_malformed(name: str, payload: str, expected: str) -> None:
    with pytest.raises(ValueError, match=re.escape(expected)) as exc:
        Directives().add(name, payload, 7)
    assert "at line 7" in str(exc.value)


def test_generator_declares_its_directives_and_records_imports() -> None:
    assert RUST.directive_kinds == {"import-object", "field", "nullable", "enum"}
    imports = RUST.new_imports()
    RUST.add_directive(imports, "import-object", "tvm_ffi.libinfo.Foo;False;_Foo", 1)
    RUST.add_directive(imports, "nullable", "demo.Node.span", 2)
    assert imports.items == [RustUse("tvm_ffi::libinfo::Foo")]
    assert imports.directives.nullable == {"demo.Node.span"}


# ---------------------------------------------------------------------------
# Object rendering
# ---------------------------------------------------------------------------

ROOT_EXPECTED = """\
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "demo.Pair"]
pub struct PairObj {
    base: Object,
}

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Pair {
    data: ObjectArc<PairObj>,
}

impl Deref for Pair {
    type Target = PairObj;
    fn deref(&self) -> &PairObj {
        &self.data
    }
}

impl PairObj {
    pub fn a(&self) -> Result<i64> {
        FieldGetter::new(Self::type_index(), "a")?.get(self)
    }

    pub fn tag(&self) -> Result<Option<String>> {
        FieldGetter::new(Self::type_index(), "tag")?.get(self)
    }

    pub fn items(&self) -> Result<Array<Any>> {
        FieldGetter::new(Self::type_index(), "items")?.get(self)
    }

    pub fn owner(&self) -> Result<ObjectRef> {
        FieldGetter::new(Self::type_index(), "owner")?.get(self)
    }

    pub fn extra(&self) -> Result<Any> {
        FieldGetter::new(Self::type_index(), "extra")?.get_any(self)
    }
}"""


def test_render_root_object() -> None:
    """A root type embeds the header, gets no parent `Deref` and no upcast."""
    info = _info(
        "demo.Pair",
        (
            ("a", TypeSchema("int")),
            ("tag", TypeSchema("Optional", (TypeSchema("str"),))),
            ("items", TypeSchema("Array")),
            ("owner", TypeSchema("Object")),
            ("extra", TypeSchema("Union", (TypeSchema("int"), TypeSchema("str")))),
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
        (("a", TypeSchema("demo.Expr")),),
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
        (("value", TypeSchema("int")),),
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


ITER_VAR_EXPECTED = """\
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct IterVarType(i32);

#[allow(non_upper_case_globals)]
impl IterVarType {
    pub const kDataPar: Self = Self(0);
    pub const kThreadIndex: Self = Self(1);
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }
    pub const fn as_raw(self) -> i32 {
        self.0
    }
}

impl TryFrom<i64> for IterVarType {
    type Error = Error;
    fn try_from(value: i64) -> Result<Self> {
        i32::try_from(value).map(Self).map_err(|_| {
            Error::new(VALUE_ERROR, &format!("IterVarType value {value} does not fit i32"), "")
        })
    }
}

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

impl Deref for IterVar {
    type Target = IterVarObj;
    fn deref(&self) -> &IterVarObj {
        &self.data
    }
}

impl Deref for IterVarObj {
    type Target = PrimExprConvertibleObj;
    fn deref(&self) -> &PrimExprConvertibleObj {
        &self.base
    }
}

impl IterVarObj {
    pub fn dom(&self) -> Result<Option<Range>> {
        FieldGetter::new(Self::type_index(), "dom")?.get(self)
    }

    pub fn var(&self) -> Result<PrimVar> {
        FieldGetter::new(Self::type_index(), "var")?.get(self)
    }

    pub fn iter_type(&self) -> Result<IterVarType> {
        let raw: i64 = FieldGetter::new(Self::type_index(), "iter_type")?.get(self)?;
        IterVarType::try_from(raw)
    }

    pub fn thread_tag(&self) -> Result<String> {
        FieldGetter::new(Self::type_index(), "thread_tag")?.get(self)
    }

    pub fn span(&self) -> Result<Option<Span>> {
        FieldGetter::new(Self::type_index(), "span")?.get(self)
    }
}

tvm_ffi::impl_object_upcast!(IterVar => PrimExprConvertible);"""


def test_render_iter_var_golden() -> None:
    """The shape tvm-rust-ext hand-writes for its polymorphic `IterVar`, driven by directives."""
    info = _info(
        "tirx.IterVar",
        (
            ("dom", TypeSchema("ir.Range")),
            ("var", TypeSchema("ir.Var")),
            ("iter_type", TypeSchema("int")),
            ("thread_tag", TypeSchema("str")),
            ("span", TypeSchema("ir.Span")),
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
        (("buffer", TypeSchema("demo.Var")), ("dom", TypeSchema("Optional", (TypeSchema("int"),)))),
    )
    imports = RUST.new_imports()
    RUST.add_directive(imports, "field", "demo.Node.buffer -> crate::typed::BufferVar", 1)
    RUST.add_directive(imports, "nullable", "demo.Node.dom", 2)
    text, imports = _render(info, imports)
    assert "pub fn buffer(&self) -> Result<BufferVar> {" in text
    assert "pub fn dom(&self) -> Result<Option<i64>> {" in text
    assert "crate::typed::BufferVar" in _uses(imports)


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
# The pipeline end to end
# ---------------------------------------------------------------------------


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
    assert "pub fn v_i32(&self) -> Result<Kind> {" in text
    assert "pub fn v_i64(&self) -> Result<i64> {" in text
    assert "use tvm_ffi::FieldGetter;" in text


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
    assert "use tvm_ffi::FieldGetter;" in text
    assert '#[type_key = "testing.TestCxxClassDerivedDerived"]' in text
    assert "    base: TestCxxClassDerivedObj," in text
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
