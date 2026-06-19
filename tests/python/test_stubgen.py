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
from __future__ import annotations

import itertools
import typing
from pathlib import Path

import pytest
import tvm_ffi.stub.cli as stub_cli
from tvm_ffi import Object, method
from tvm_ffi.core import TypeSchema
from tvm_ffi.dataclasses import py_class
from tvm_ffi.stub import consts as C
from tvm_ffi.stub.cli import _stage_2, _stage_3
from tvm_ffi.stub.file_utils import CodeBlock, FileInfo
from tvm_ffi.stub.generator import get_generator
from tvm_ffi.stub.python_generator import consts as PC
from tvm_ffi.stub.python_generator.codegen import (
    generate_python_all,
    generate_python_export,
    generate_python_ffi_api,
    generate_python_global_funcs,
    generate_python_import_section,
    generate_python_init,
    generate_python_object,
    render_func_signature,
    render_object_ffi_init,
    render_object_fields,
    render_object_init,
    render_object_methods,
)
from tvm_ffi.stub.python_generator.utils import ImportItem
from tvm_ffi.stub.utils import (
    FuncInfo,
    InitConfig,
    InitFieldInfo,
    NamedTypeSchema,
    ObjectInfo,
    Options,
)

_counter = itertools.count()


def _identity_ty_map(name: str) -> str:
    return name


def _unique_type_key(base: str) -> str:
    return f"testing.stubgen.{base}_{next(_counter)}"


def _default_ty_map() -> dict[str, str]:
    return PC.TY_MAP_DEFAULTS.copy()


def _type_suffix(name: str) -> str:
    return PC.TY_MAP_DEFAULTS.get(name, name).rsplit(".", 1)[-1]


def _input_type_suffix(name: str) -> str:
    return PC.TY_MAP_INPUT_DEFAULTS.get(name, PC.TY_MAP_DEFAULTS.get(name, name)).rsplit(".", 1)[-1]


def test_codeblock_from_begin_line_variants() -> None:
    cases = [
        (f"{C.PYTHON_SYNTAX.begin} global/demo", "global", ("demo", "")),
        (f"{C.PYTHON_SYNTAX.begin} global/demo@.registry", "global", ("demo", ".registry")),
        (f"{C.PYTHON_SYNTAX.begin} object/demo.TypeBase", "object", "demo.TypeBase"),
        (f"{C.PYTHON_SYNTAX.begin} ty-map/custom", "ty-map", "custom"),
        (f"{C.PYTHON_SYNTAX.begin} import-section", "import-section", ""),
    ]
    for lineno, (line, kind, param) in enumerate(cases, start=1):
        block = CodeBlock.from_begin_line(lineno, line, C.PYTHON_SYNTAX)
        assert block.kind == kind
        assert block.param == param
        assert block.lineno_start == lineno
        assert block.lineno_end is None
        assert block.lines == []


def test_codeblock_from_begin_line_ty_map_and_unknown() -> None:
    line = f"{C.PYTHON_SYNTAX.ty_map} custom -> mapped"
    block = CodeBlock.from_begin_line(5, line, C.PYTHON_SYNTAX)
    assert block.kind == "ty-map"
    assert block.param == "custom -> mapped"
    assert block.lineno_start == 5
    assert block.lineno_end == 5

    with pytest.raises(ValueError):
        CodeBlock.from_begin_line(1, f"{C.PYTHON_SYNTAX.begin} unsupported/kind", C.PYTHON_SYNTAX)


def test_fileinfo_from_file_skip_and_missing_markers(tmp_path: Path) -> None:
    skip = tmp_path / "skip.py"
    skip.write_text(f"print('hi')\n{C.PYTHON_SYNTAX.skip_file}\n", encoding="utf-8")
    assert FileInfo.from_file(skip) is None

    plain = tmp_path / "plain.py"
    plain.write_text("print('plain')\n", encoding="utf-8")
    assert FileInfo.from_file(plain) is None


def test_fileinfo_from_file_parses_blocks(tmp_path: Path) -> None:
    content = "\n".join(
        [
            "first = 1",
            f"{C.PYTHON_SYNTAX.begin} global/demo.func",
            "in_stub = True",
            C.PYTHON_SYNTAX.end,
            f"{C.PYTHON_SYNTAX.ty_map} x -> y",
        ]
    )
    path = tmp_path / "demo.py"
    path.write_text(content, encoding="utf-8")

    info = FileInfo.from_file(path)
    assert info is not None
    assert info.path == path.resolve()
    assert len(info.code_blocks) == 3

    first, stub, ty_map = info.code_blocks
    assert first.kind is None and first.lines == ["first = 1"]

    assert stub.kind == "global"
    assert stub.param == ("demo.func", "")
    assert stub.lineno_start == 2
    assert stub.lineno_end == 4
    assert stub.lines == [
        f"{C.PYTHON_SYNTAX.begin} global/demo.func",
        "in_stub = True",
        C.PYTHON_SYNTAX.end,
    ]

    assert ty_map.kind == "ty-map"
    assert ty_map.param == "x -> y"
    assert ty_map.lineno_start == ty_map.lineno_end == 5
    assert ty_map.lines == [f"{C.PYTHON_SYNTAX.ty_map} x -> y"]


def test_fileinfo_from_file_error_paths(tmp_path: Path) -> None:
    nested = tmp_path / "nested.py"
    nested.write_text(
        "\n".join(
            [
                f"{C.PYTHON_SYNTAX.begin} global/outer",
                f"{C.PYTHON_SYNTAX.begin} global/inner",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Nested stub not permitted"):
        FileInfo.from_file(nested)

    unmatched_end = tmp_path / "unmatched.py"
    unmatched_end.write_text(C.PYTHON_SYNTAX.end + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unmatched"):
        FileInfo.from_file(unmatched_end)

    unclosed = tmp_path / "unclosed.py"
    unclosed.write_text(f"{C.PYTHON_SYNTAX.begin} global/method\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unclosed stub block"):
        FileInfo.from_file(unclosed)


def test_funcinfo_gen_variants() -> None:
    called: list[str] = []

    def ty_map(name: str) -> str:
        called.append(name)
        return name

    schema_no_args = NamedTypeSchema("demo.no_args", TypeSchema("Callable", ()))
    func = FuncInfo(schema=schema_no_args, is_member=False)
    assert render_func_signature(func, ty_map, indent=2) == "  def no_args(*args: Any) -> Any: ..."
    assert called == ["Any"]

    schema_member = NamedTypeSchema(
        "pkg.Class.method",
        TypeSchema(
            "Callable",
            (
                TypeSchema("str"),
                TypeSchema("int"),
                TypeSchema("float"),
            ),
        ),
    )
    member_func = FuncInfo(schema=schema_member, is_member=True)
    assert (
        render_func_signature(member_func, _identity_ty_map, indent=0)
        == "def method(self, _1: float, /) -> str: ..."
    )

    schema_bad = NamedTypeSchema("bad", TypeSchema("int"))
    with pytest.raises(ValueError):
        render_func_signature(
            FuncInfo(schema=schema_bad, is_member=False), _identity_ty_map, indent=0
        )


def test_objectinfo_gen_fields_and_methods() -> None:
    ty_calls: list[str] = []

    def ty_map(name: str) -> str:
        ty_calls.append(name)
        return {"list": "Sequence", "dict": "Mapping"}.get(name, name)

    info = ObjectInfo(
        fields=[
            NamedTypeSchema("field_a", TypeSchema("list", (TypeSchema("int"),))),
            NamedTypeSchema(
                "field_b", TypeSchema("dict", (TypeSchema("str"), TypeSchema("float")))
            ),
        ],
        methods=[
            FuncInfo(
                schema=NamedTypeSchema("demo.static", TypeSchema("Callable", (TypeSchema("int"),))),
                is_member=False,
            ),
            FuncInfo(
                schema=NamedTypeSchema(
                    "demo.member",
                    TypeSchema("Callable", (TypeSchema("str"), TypeSchema("bytes"))),
                ),
                is_member=True,
            ),
        ],
    )

    assert render_object_fields(info, ty_map, indent=2) == [
        "  field_a: Sequence[int]",
        "  field_b: Mapping[str, float]",
    ]
    assert ty_calls.count("list") == 1 and ty_calls.count("dict") == 1

    methods = render_object_methods(info, _identity_ty_map, indent=2)
    assert methods == [
        "  @staticmethod",
        "  def static() -> int: ...",
        "  def member(self, /) -> str: ...",
    ]


def test_type_schema_container_origins() -> None:
    """Test that Array/List/Map/Dict origins are distinct and validated correctly."""
    # Array and List: 0 or 1 arg, default to (Any,)
    for origin in ("Array", "List"):
        s = TypeSchema(origin)
        assert s.args == (TypeSchema("Any"),), f"{origin} should default to (Any,)"
        s = TypeSchema(origin, (TypeSchema("int"),))
        assert s.repr() == f"{origin}[int]"

    # Map and Dict: 0 or 2 args, default to (Any, Any)
    for origin in ("Map", "Dict"):
        s = TypeSchema(origin)
        assert s.args == (TypeSchema("Any"), TypeSchema("Any")), (
            f"{origin} should default to (Any, Any)"
        )
        s = TypeSchema(origin, (TypeSchema("str"), TypeSchema("float")))
        assert s.repr() == f"{origin}[str, float]"

    # from_json_str round-trip through _TYPE_SCHEMA_ORIGIN_CONVERTER
    s = TypeSchema.from_json_str('{"type":"ffi.Array","args":[{"type":"int"}]}')
    assert s.origin == "Array"
    assert s.repr() == "Array[int]"

    s = TypeSchema.from_json_str('{"type":"ffi.List","args":[{"type":"str"}]}')
    assert s.origin == "List"
    assert s.repr() == "List[str]"

    s = TypeSchema.from_json_str('{"type":"ffi.Map","args":[{"type":"str"},{"type":"int"}]}')
    assert s.origin == "Map"
    assert s.repr() == "Map[str, int]"

    s = TypeSchema.from_json_str('{"type":"ffi.Dict","args":[{"type":"str"},{"type":"float"}]}')
    assert s.origin == "Dict"
    assert s.repr() == "Dict[str, float]"

    # Backward compat: "list" and "dict" origins still work
    s = TypeSchema("list", (TypeSchema("int"),))
    assert s.repr() == "list[int]"
    s = TypeSchema("dict", (TypeSchema("str"), TypeSchema("int")))
    assert s.repr() == "dict[str, int]"


def test_objectinfo_gen_fields_container_types() -> None:
    """Test that ObjectInfo fields render distinct container annotations."""
    info = ObjectInfo(
        fields=[
            NamedTypeSchema("arr", TypeSchema("Array", (TypeSchema("int"),))),
            NamedTypeSchema("lst", TypeSchema("List", (TypeSchema("str"),))),
            NamedTypeSchema("mp", TypeSchema("Map", (TypeSchema("str"), TypeSchema("int")))),
            NamedTypeSchema("dt", TypeSchema("Dict", (TypeSchema("str"), TypeSchema("float")))),
        ],
        methods=[],
    )
    assert render_object_fields(info, _type_suffix, indent=0) == [
        "arr: Sequence[int]",
        "lst: MutableSequence[str]",
        "mp: Mapping[str, int]",
        "dt: MutableMapping[str, float]",
    ]


def test_funcinfo_gen_uses_input_annotations_for_parameters() -> None:
    info = FuncInfo(
        schema=NamedTypeSchema(
            "demo.echo_list",
            TypeSchema(
                "Callable",
                (
                    TypeSchema("List", (TypeSchema("int"),)),
                    TypeSchema("List", (TypeSchema("int"),)),
                ),
            ),
        ),
        is_member=False,
    )

    assert (
        render_func_signature(info, _type_suffix, indent=0, input_ty_map=_input_type_suffix)
        == "def echo_list(_0: Sequence[int], /) -> MutableSequence[int]: ..."
    )


def test_generate_global_funcs_populates_input_defaults_for_partial_ty_map() -> None:
    code = CodeBlock(
        kind="global",
        param=("demo", "mockpkg"),
        lineno_start=1,
        lineno_end=2,
        lines=[f"{C.PYTHON_SYNTAX.begin} global/demo@mockpkg", C.PYTHON_SYNTAX.end],
    )
    funcs = [
        FuncInfo(
            schema=NamedTypeSchema(
                "demo.echo_list",
                TypeSchema(
                    "Callable",
                    (
                        TypeSchema("List", (TypeSchema("int"),)),
                        TypeSchema("List", (TypeSchema("int"),)),
                    ),
                ),
            ),
            is_member=False,
        )
    ]
    imports: list[ImportItem] = []

    generate_python_global_funcs(
        code, funcs, {"List": "collections.abc.MutableSequence"}, imports, Options()
    )

    assert code.lines == [
        f"{C.PYTHON_SYNTAX.begin} global/demo@mockpkg",
        "# fmt: off",
        '_FFI_INIT_FUNC("demo", __name__)',
        "if TYPE_CHECKING:",
        "    def echo_list(_0: Sequence[int], /) -> MutableSequence[int]: ...",
        "# fmt: on",
        C.PYTHON_SYNTAX.end,
    ]


def test_objectinfo_gen_init_uses_input_annotations() -> None:
    info = ObjectInfo(
        fields=[NamedTypeSchema("items", TypeSchema("List", (TypeSchema("int"),)))],
        methods=[],
        init_fields=[
            InitFieldInfo(
                name="items",
                schema=NamedTypeSchema("items", TypeSchema("List", (TypeSchema("int"),))),
                kw_only=False,
                has_default=False,
            )
        ],
        has_init=True,
    )

    assert render_object_fields(info, _type_suffix, indent=0) == ["items: MutableSequence[int]"]
    assert render_object_init(info, _type_suffix, indent=0, input_ty_map=_input_type_suffix) == [
        "def __init__(self, items: Sequence[int]) -> None: ..."
    ]
    assert render_object_ffi_init(
        info, _type_suffix, indent=0, input_ty_map=_input_type_suffix
    ) == [
        "def __ffi_init__(self, items: Sequence[int]) -> None: ...  # ty: "
        "ignore[invalid-method-override]"
    ]


def test_py_class_method_metadata_renders_stub_signature() -> None:
    @py_class(_unique_type_key("MethodMetadata"))
    class MethodMetadata(Object):
        value: int

        @method
        def describe(self, values: typing.List[int], prefix: str) -> str:  # noqa: UP006
            return f"{prefix}:{self.value}:{len(values)}"

        @method
        @staticmethod
        def normalize(values: typing.List[int]) -> typing.List[int]:  # noqa: UP006
            return values

    info = ObjectInfo.from_type_info(MethodMetadata.__tvm_ffi_type_info__)  # ty: ignore[unresolved-attribute]
    methods = {method.schema.name: method for method in info.methods}
    describe_schema = methods["describe"].schema

    assert describe_schema.origin == "Callable"
    assert [arg.origin for arg in describe_schema.args] == [
        "str",
        MethodMetadata.__tvm_ffi_type_info__.type_key,  # ty: ignore[unresolved-attribute]
        "List",
        "str",
    ]
    assert render_object_methods(info, _type_suffix, indent=0, input_ty_map=_input_type_suffix) == [
        "def describe(self, _1: Sequence[int], _2: str, /) -> str: ...",
        "@staticmethod",
        "def normalize(_0: Sequence[int], /) -> MutableSequence[int]: ...",
    ]


def test_generate_global_funcs_updates_block() -> None:
    code = CodeBlock(
        kind="global",
        param=("demo", "mockpkg"),
        lineno_start=1,
        lineno_end=2,
        lines=[f"{C.PYTHON_SYNTAX.begin} global/demo@mockpkg", C.PYTHON_SYNTAX.end],
    )
    funcs = [
        FuncInfo(
            schema=NamedTypeSchema(
                "demo.add_one",
                TypeSchema("Callable", (TypeSchema("int"), TypeSchema("int"))),
            ),
            is_member=False,
        )
    ]
    opts = Options(indent=2)
    imports: list[ImportItem] = []
    generate_python_global_funcs(code, funcs, _default_ty_map(), imports, opts)
    assert imports == [
        ImportItem("mockpkg.init_ffi_api", alias="_FFI_INIT_FUNC"),
        ImportItem("typing.TYPE_CHECKING"),
    ]
    assert code.lines == [
        f"{C.PYTHON_SYNTAX.begin} global/demo@mockpkg",
        "# fmt: off",
        '_FFI_INIT_FUNC("demo", __name__)',
        "if TYPE_CHECKING:",
        "  def add_one(_0: int, /) -> int: ...",
        "# fmt: on",
        C.PYTHON_SYNTAX.end,
    ]


def test_generate_global_funcs_noop_on_empty_list() -> None:
    code = CodeBlock(
        kind="global",
        param=("empty", ""),
        lineno_start=1,
        lineno_end=2,
        lines=[f"{C.PYTHON_SYNTAX.begin} global/empty", C.PYTHON_SYNTAX.end],
    )
    imports: list[ImportItem] = []
    generate_python_global_funcs(code, [], _default_ty_map(), imports, Options())
    assert code.lines == [f"{C.PYTHON_SYNTAX.begin} global/empty", C.PYTHON_SYNTAX.end]
    assert imports == []


def test_generate_global_funcs_respects_custom_import_from() -> None:
    code = CodeBlock(
        kind="global",
        param=("demo", "custom.mod"),
        lineno_start=1,
        lineno_end=2,
        lines=[f"{C.PYTHON_SYNTAX.begin} global/demo@custom.mod", C.PYTHON_SYNTAX.end],
    )
    funcs = [
        FuncInfo(
            schema=NamedTypeSchema(
                "demo.add_one",
                TypeSchema("Callable", (TypeSchema("int"), TypeSchema("int"))),
            ),
            is_member=False,
        )
    ]
    imports: list[ImportItem] = []
    generate_python_global_funcs(code, funcs, _default_ty_map(), imports, Options(indent=0))
    assert ImportItem("custom.mod.init_ffi_api", alias="_FFI_INIT_FUNC") in imports


def test_generate_global_funcs_aliases_colliding_type() -> None:
    """When a function name matches a type name, the type import gets an alias."""
    code = CodeBlock(
        kind="global",
        param=("demo", "mockpkg"),
        lineno_start=1,
        lineno_end=2,
        lines=[f"{C.PYTHON_SYNTAX.begin} global/demo@mockpkg", C.PYTHON_SYNTAX.end],
    )
    # Function "demo.Foo" returns type "demo.Foo" — name collision
    funcs = [
        FuncInfo(
            schema=NamedTypeSchema(
                "demo.Foo",
                TypeSchema("Callable", (TypeSchema("demo.Foo"), TypeSchema("Any"))),
            ),
            is_member=False,
        )
    ]
    ty_map = _default_ty_map()
    ty_map["demo.Foo"] = "somepkg.Foo"
    imports: list[ImportItem] = []
    generate_python_global_funcs(code, funcs, ty_map, imports, Options(indent=4))
    # The type import should use an alias to avoid shadowing the function
    assert ImportItem("somepkg.Foo", type_checking_only=True, alias="_Foo") in imports
    # The function annotation should use the alias
    assert any("-> _Foo:" in line for line in code.lines)


def test_generate_object_fields_only_block() -> None:
    code = CodeBlock(
        kind="object",
        param="demo.TypeDerived",
        lineno_start=1,
        lineno_end=2,
        lines=[f"{C.PYTHON_SYNTAX.begin} object/demo.TypeDerived", C.PYTHON_SYNTAX.end],
    )
    opts = Options(indent=4)
    imports: list[ImportItem] = []
    info = ObjectInfo(
        fields=[
            NamedTypeSchema("field_a", TypeSchema("int")),
            NamedTypeSchema("field_b", TypeSchema("float")),
        ],
        methods=[],
        type_key="demo.TypeDerived",
        parent_type_key="demo.Parent",
    )
    generate_python_object(
        code,
        _default_ty_map(),
        imports,
        opts,
        info,
    )
    assert imports == []

    expected = [
        f"{C.PYTHON_SYNTAX.begin} object/demo.TypeDerived",
        " " * code.indent + "# fmt: off",
        *[
            (" " * code.indent) + line
            for line in render_object_fields(info, _type_suffix, indent=0)
        ],
        " " * code.indent + "# fmt: on",
        C.PYTHON_SYNTAX.end,
    ]
    assert code.lines == expected


def test_generate_object_with_methods() -> None:
    code = CodeBlock(
        kind="object",
        param="demo.IntPair",
        lineno_start=1,
        lineno_end=2,
        lines=[f"{C.PYTHON_SYNTAX.begin} object/demo.IntPair", C.PYTHON_SYNTAX.end],
    )
    opts = Options(indent=4)
    imports: list[ImportItem] = []
    info = ObjectInfo(
        fields=[],
        methods=[
            FuncInfo.from_schema(
                "demo.IntPair.__ffi_init__",
                TypeSchema("Callable", (TypeSchema("None"), TypeSchema("int"), TypeSchema("int"))),
                is_member=True,
            ),
            FuncInfo.from_schema(
                "demo.IntPair.sum",
                TypeSchema("Callable", (TypeSchema("int"),)),
                is_member=True,
            ),
        ],
        type_key="demo.IntPair",
        parent_type_key="demo.Parent",
    )
    generate_python_object(code, _default_ty_map(), imports, opts, info)
    assert set(imports) == {ImportItem("typing.TYPE_CHECKING")}

    assert code.lines[0] == f"{C.PYTHON_SYNTAX.begin} object/demo.IntPair"
    assert code.lines[-1] == C.PYTHON_SYNTAX.end
    assert "# fmt: off" in code.lines[1]
    assert any("if TYPE_CHECKING:" in line for line in code.lines)
    method_lines = [line for line in code.lines if "def __ffi_init__" in line or "def sum" in line]
    # __ffi_init__ from TypeMethod is rendered as an instance method (self, ...) -> None
    assert any(line.strip().startswith("def __ffi_init__(self") for line in method_lines)
    assert any(line.strip().startswith("def sum") for line in method_lines)


def test_import_item_mod_map_prefix_rewrite() -> None:
    # MOD_MAP rewrites must respect module-path boundaries.
    assert ImportItem("ffi.Object").mod == "tvm_ffi"
    assert ImportItem("testing.TestIntPair").mod == "tvm_ffi.testing"
    assert ImportItem("testing.sub.Thing").mod == "tvm_ffi.testing.sub"
    # A module that merely starts with a mapped prefix is NOT rewritten.
    assert ImportItem("testingfoo.Thing").mod == "testingfoo"
    assert ImportItem("ffi2.Thing").mod == "ffi2"


def test_generate_import_section_groups_modules() -> None:
    code = CodeBlock(
        kind="import-section",
        param="",
        lineno_start=1,
        lineno_end=2,
        lines=[f"{C.PYTHON_SYNTAX.begin} import", C.PYTHON_SYNTAX.end],
    )
    imports = [
        ImportItem("typing.Any", type_checking_only=True),
        ImportItem("demo_pkg.Tensor", type_checking_only=True),
        ImportItem("demo.TestObjectBase", type_checking_only=True),
        ImportItem("custom.mod.Type", type_checking_only=True),
    ]
    opts = Options(indent=4)
    generate_python_import_section(code, imports, opts)

    expected_prefix = [
        f"{C.PYTHON_SYNTAX.begin} import",
        "# fmt: off",
        "# isort: off",
        "from __future__ import annotations",
        "from typing import TYPE_CHECKING",
        "if TYPE_CHECKING:",
    ]
    assert code.lines[: len(expected_prefix)] == expected_prefix
    assert "    from demo import TestObjectBase" in code.lines
    assert "    from demo_pkg import Tensor" in code.lines
    assert "    from custom.mod import Type" in code.lines
    assert "    from typing import Any" in code.lines
    assert code.lines[-2:] == ["# fmt: on", C.PYTHON_SYNTAX.end]


def test_generate_import_section_no_imports_noop() -> None:
    code = CodeBlock(
        kind="import-section",
        param="",
        lineno_start=1,
        lineno_end=2,
        lines=[f"{C.PYTHON_SYNTAX.begin} import", C.PYTHON_SYNTAX.end],
    )
    before = list(code.lines)
    generate_python_import_section(code, [], Options())
    assert code.lines == before


def test_generate_all_builds_sorted_and_deduped_list() -> None:
    code = CodeBlock(
        kind="global",
        param="all",
        lineno_start=1,
        lineno_end=2,
        lines=["    " + C.PYTHON_SYNTAX.begin + " global/all", C.PYTHON_SYNTAX.end],
    )
    generate_python_all(
        code,
        names={"tvm_ffi.foo", "bar", "pkg.baz", "bar"},  # duplicates stripped
        opt=Options(indent=2),
    )
    assert code.lines == [
        "    " + C.PYTHON_SYNTAX.begin + " global/all",
        '    "bar",',
        '    "baz",',
        '    "foo",',
        C.PYTHON_SYNTAX.end,
    ]


def test_generate_all_noop_on_empty_names() -> None:
    code = CodeBlock(
        kind="global",
        param="all-empty",
        lineno_start=1,
        lineno_end=2,
        lines=[C.PYTHON_SYNTAX.begin + " global/all-empty", C.PYTHON_SYNTAX.end],
    )
    before = list(code.lines)
    generate_python_all(code, names=set(), opt=Options())
    assert code.lines == before


def test_generate_all_uses_isort_style_ordering() -> None:
    code = CodeBlock(
        kind="global",
        param="all-mixed",
        lineno_start=1,
        lineno_end=2,
        lines=[C.PYTHON_SYNTAX.begin + " global/all-mixed", C.PYTHON_SYNTAX.end],
    )
    names = {"foo", "Bar", "LIB", "baz", "Alpha", "CONST"}
    generate_python_all(code, names=names, opt=Options(indent=0))
    assert code.lines == [
        C.PYTHON_SYNTAX.begin + " global/all-mixed",
        '"CONST",',
        '"LIB",',
        '"Alpha",',
        '"Bar",',
        '"baz",',
        '"foo",',
        C.PYTHON_SYNTAX.end,
    ]


def test_stage_3_adds_LIB_when_load_lib_imported(tmp_path: Path) -> None:
    path = tmp_path / "demo.py"
    global_block = CodeBlock(
        kind="global",
        param=("testing", ""),
        lineno_start=2,
        lineno_end=3,
        lines=[f"{C.PYTHON_SYNTAX.begin} global/testing", C.PYTHON_SYNTAX.end],
    )
    import_obj_block = CodeBlock(
        kind="import-object",
        param=("tvm_ffi.libinfo.load_lib_module", "False", "_FFI_LOAD_LIB"),
        lineno_start=1,
        lineno_end=1,
        lines=[
            f"{C.PYTHON_SYNTAX.import_object} tvm_ffi.libinfo.load_lib_module;False;_FFI_LOAD_LIB"
        ],
    )
    all_block = CodeBlock(
        kind="__all__",
        param="",
        lineno_start=4,
        lineno_end=5,
        lines=[f"{C.PYTHON_SYNTAX.begin} __all__", C.PYTHON_SYNTAX.end],
    )
    file_info = FileInfo(
        path=path,
        lines=tuple(
            line for block in (import_obj_block, global_block, all_block) for line in block.lines
        ),
        code_blocks=[import_obj_block, global_block, all_block],
        syntax=C.PYTHON_SYNTAX,
    )
    funcs = [
        FuncInfo.from_schema(
            "testing.add_one",
            TypeSchema("Callable", (TypeSchema("int"), TypeSchema("int"))),
        )
    ]
    _stage_3(
        file_info,
        Options(dry_run=True),
        _default_ty_map(),
        {"testing": funcs},
        get_generator("python"),
    )
    lib_lines = [line for line in all_block.lines if "LIB" in line]
    assert any("LIB" in line for line in lib_lines)


def test_generate_export_builds_all_extension() -> None:
    code = CodeBlock(
        kind="export",
        param="ffi_api",
        lineno_start=1,
        lineno_end=2,
        lines=[f"{C.PYTHON_SYNTAX.begin} export/ffi_api", C.PYTHON_SYNTAX.end],
    )
    generate_python_export(code)
    full_text = "\n".join(code.lines)
    assert "from .ffi_api import *" in full_text
    assert "ffi_api__all__" in full_text


def test_generate_init_with_and_without_existing_export_block() -> None:
    code_no_blocks = generate_python_init([], "demo", "_ffi_api", C.PYTHON_SYNTAX)
    assert "Package demo." in code_no_blocks
    assert f"{C.PYTHON_SYNTAX.begin} export/_ffi_api" in code_no_blocks

    code_with_export = generate_python_init(
        [
            CodeBlock(
                kind="export",
                param="_ffi_api",
                lineno_start=1,
                lineno_end=2,
                lines=["", ""],
            )
        ],
        "demo",
        "_ffi_api",
        C.PYTHON_SYNTAX,
    )
    assert code_with_export == ""


def test_generate_ffi_api_without_objects_includes_sections() -> None:
    init_cfg = InitConfig(pkg="pkg", shared_target="pkg_shared", prefix="pkg.")
    code = generate_python_ffi_api(
        [],
        _default_ty_map(),
        "demo.mod",
        [],
        init_cfg,
        is_root=False,
        syntax=C.PYTHON_SYNTAX,
    )
    assert f"{C.PYTHON_SYNTAX.begin} import-section" in code
    assert f"{C.PYTHON_SYNTAX.begin} global/demo.mod" in code
    assert C.PYTHON_SYNTAX.begin + " __all__" in code
    assert "LIB =" not in code


def test_generate_ffi_api_with_objects_imports_parents() -> None:
    init_cfg = InitConfig(pkg="pkg", shared_target="pkg_shared", prefix="pkg.")
    obj_info = ObjectInfo(
        fields=[],
        methods=[],
        type_key="demo.TypeDerived",
        parent_type_key="demo.Parent",
    )
    parent_key = obj_info.parent_type_key
    code = generate_python_ffi_api(
        [],
        _default_ty_map(),
        "demo",
        [obj_info],
        init_cfg,
        is_root=False,
        syntax=C.PYTHON_SYNTAX,
    )
    assert C.PYTHON_SYNTAX.import_object in code  # register_object prompt
    assert f"{C.PYTHON_SYNTAX.begin} object/{obj_info.type_key}" in code
    assert parent_key is not None
    parent_import_prompt = (
        f"{C.PYTHON_SYNTAX.import_object} {parent_key};False;_{parent_key.replace('.', '_')}"
    )
    assert parent_import_prompt in code


def test_stage_2_filters_prefix_and_marks_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prefixes: dict[str, list[FuncInfo]] = {"demo.sub": [], "demo": [], "other": []}
    monkeypatch.setattr(stub_cli, "collect_type_keys", lambda: prefixes)
    monkeypatch.setattr(stub_cli, "toposort_objects", lambda objs: [])

    global_funcs = {
        "demo.sub": [
            FuncInfo.from_schema(
                "demo.sub.add_one",
                TypeSchema("Callable", (TypeSchema("int"), TypeSchema("int"))),
            )
        ],
        "demo": [
            FuncInfo.from_schema(
                "demo.add_one",
                TypeSchema("Callable", (TypeSchema("int"), TypeSchema("int"))),
            )
        ],
        "other": [
            FuncInfo.from_schema(
                "other.add_one",
                TypeSchema("Callable", (TypeSchema("int"), TypeSchema("int"))),
            )
        ],
    }
    _stage_2(
        files=[],
        ty_map=_default_ty_map(),
        init_cfg=InitConfig(pkg="demo-pkg", shared_target="demo_shared", prefix="demo."),
        init_path=tmp_path,
        global_funcs=global_funcs,
        generator=get_generator("python"),
    )

    root_api = tmp_path / "demo" / "_ffi_api.py"
    sub_api = tmp_path / "demo" / "sub" / "_ffi_api.py"
    other_api = tmp_path / "other" / "_ffi_api.py"
    assert root_api.exists()
    assert sub_api.exists()
    assert not other_api.exists()
    root_text = root_api.read_text(encoding="utf-8")
    sub_text = sub_api.read_text(encoding="utf-8")
    assert 'LIB = _FFI_LOAD_LIB("demo-pkg", "demo_shared")' in root_text
    assert "LIB =" not in sub_text
