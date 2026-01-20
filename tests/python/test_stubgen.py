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

from pathlib import Path

import pytest
import tvm_ffi.stub.cli as stub_cli
from tvm_ffi.core import TypeSchema
from tvm_ffi.stub import consts as C
from tvm_ffi.stub.cli import _stage_2, _stage_3
from tvm_ffi.stub.codegen import (
    generate_all,
    generate_export,
    generate_ffi_api,
    generate_global_funcs,
    generate_import_section,
    generate_init,
    generate_object,
)
from tvm_ffi.stub.file_utils import CodeBlock, FileInfo
from tvm_ffi.stub.utils import (
    FuncInfo,
    ImportItem,
    InitConfig,
    NamedTypeSchema,
    ObjectInfo,
    Options,
)


def _identity_ty_map(name: str) -> str:
    return name


def _default_ty_map() -> dict[str, str]:
    return C.TY_MAP_DEFAULTS.copy()


def _type_suffix(name: str) -> str:
    return C.TY_MAP_DEFAULTS.get(name, name).rsplit(".", 1)[-1]


def test_codeblock_from_begin_line_variants() -> None:
    cases = [
        (f"{C.STUB_BEGIN} global/demo", "global", ("demo", "")),
        (f"{C.STUB_BEGIN} global/demo@.registry", "global", ("demo", ".registry")),
        (f"{C.STUB_BEGIN} object/demo.TypeBase", "object", "demo.TypeBase"),
        (f"{C.STUB_BEGIN} ty-map/custom", "ty-map", "custom"),
        (f"{C.STUB_BEGIN} import-section", "import-section", ""),
    ]
    for lineno, (line, kind, param) in enumerate(cases, start=1):
        block = CodeBlock.from_begin_line(lineno, line)
        assert block.kind == kind
        assert block.param == param
        assert block.lineno_start == lineno
        assert block.lineno_end is None
        assert block.lines == []


def test_codeblock_from_begin_line_ty_map_and_unknown() -> None:
    line = f"{C.STUB_TY_MAP} custom -> mapped"
    block = CodeBlock.from_begin_line(5, line)
    assert block.kind == "ty-map"
    assert block.param == "custom -> mapped"
    assert block.lineno_start == 5
    assert block.lineno_end == 5

    with pytest.raises(ValueError):
        CodeBlock.from_begin_line(1, f"{C.STUB_BEGIN} unsupported/kind")


def test_fileinfo_from_file_skip_and_missing_markers(tmp_path: Path) -> None:
    skip = tmp_path / "skip.py"
    skip.write_text(f"print('hi')\n{C.STUB_SKIP_FILE}\n", encoding="utf-8")
    assert FileInfo.from_file(skip) is None

    plain = tmp_path / "plain.py"
    plain.write_text("print('plain')\n", encoding="utf-8")
    assert FileInfo.from_file(plain) is None


def test_fileinfo_from_file_parses_blocks(tmp_path: Path) -> None:
    content = "\n".join(
        [
            "first = 1",
            f"{C.STUB_BEGIN} global/demo.func",
            "in_stub = True",
            C.STUB_END,
            f"{C.STUB_TY_MAP} x -> y",
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
        f"{C.STUB_BEGIN} global/demo.func",
        "in_stub = True",
        C.STUB_END,
    ]

    assert ty_map.kind == "ty-map"
    assert ty_map.param == "x -> y"
    assert ty_map.lineno_start == ty_map.lineno_end == 5
    assert ty_map.lines == [f"{C.STUB_TY_MAP} x -> y"]


def test_fileinfo_from_file_error_paths(tmp_path: Path) -> None:
    nested = tmp_path / "nested.py"
    nested.write_text(
        "\n".join(
            [
                f"{C.STUB_BEGIN} global/outer",
                f"{C.STUB_BEGIN} global/inner",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Nested stub not permitted"):
        FileInfo.from_file(nested)

    unmatched_end = tmp_path / "unmatched.py"
    unmatched_end.write_text(C.STUB_END + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unmatched"):
        FileInfo.from_file(unmatched_end)

    unclosed = tmp_path / "unclosed.py"
    unclosed.write_text(f"{C.STUB_BEGIN} global/method\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unclosed stub block"):
        FileInfo.from_file(unclosed)


def test_funcinfo_gen_variants() -> None:
    called: list[str] = []

    def ty_map(name: str) -> str:
        called.append(name)
        return name

    schema_no_args = NamedTypeSchema("demo.no_args", TypeSchema("Callable", ()))
    func = FuncInfo(schema=schema_no_args, is_member=False)
    assert func.gen(ty_map, indent=2) == "  def no_args(*args: Any) -> Any: ..."
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
        member_func.gen(_identity_ty_map, indent=0) == "def method(self, _1: float, /) -> str: ..."
    )

    schema_bad = NamedTypeSchema("bad", TypeSchema("int"))
    with pytest.raises(ValueError):
        FuncInfo(schema=schema_bad, is_member=False).gen(_identity_ty_map, indent=0)


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

    assert info.gen_fields(ty_map, indent=2) == [
        "  field_a: Sequence[int]",
        "  field_b: Mapping[str, float]",
    ]
    assert ty_calls.count("list") == 1 and ty_calls.count("dict") == 1

    methods = info.gen_methods(_identity_ty_map, indent=2)
    assert methods == [
        "  @staticmethod",
        "  def static() -> int: ...",
        "  def member(self, /) -> str: ...",
    ]


def test_generate_global_funcs_updates_block() -> None:
    code = CodeBlock(
        kind="global",
        param=("demo", "mockpkg"),
        lineno_start=1,
        lineno_end=2,
        lines=[f"{C.STUB_BEGIN} global/demo@mockpkg", C.STUB_END],
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
    generate_global_funcs(code, funcs, _default_ty_map(), imports, opts)
    assert imports == [
        ImportItem("mockpkg.init_ffi_api", alias="_FFI_INIT_FUNC"),
        ImportItem("typing.TYPE_CHECKING"),
    ]
    assert code.lines == [
        f"{C.STUB_BEGIN} global/demo@mockpkg",
        "# fmt: off",
        '_FFI_INIT_FUNC("demo", __name__)',
        "if TYPE_CHECKING:",
        "  def add_one(_0: int, /) -> int: ...",
        "# fmt: on",
        C.STUB_END,
    ]


def test_generate_global_funcs_noop_on_empty_list() -> None:
    code = CodeBlock(
        kind="global",
        param=("empty", ""),
        lineno_start=1,
        lineno_end=2,
        lines=[f"{C.STUB_BEGIN} global/empty", C.STUB_END],
    )
    imports: list[ImportItem] = []
    generate_global_funcs(code, [], _default_ty_map(), imports, Options())
    assert code.lines == [f"{C.STUB_BEGIN} global/empty", C.STUB_END]
    assert imports == []


def test_generate_global_funcs_respects_custom_import_from() -> None:
    code = CodeBlock(
        kind="global",
        param=("demo", "custom.mod"),
        lineno_start=1,
        lineno_end=2,
        lines=[f"{C.STUB_BEGIN} global/demo@custom.mod", C.STUB_END],
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
    generate_global_funcs(code, funcs, _default_ty_map(), imports, Options(indent=0))
    assert ImportItem("custom.mod.init_ffi_api", alias="_FFI_INIT_FUNC") in imports


def test_generate_object_fields_only_block() -> None:
    code = CodeBlock(
        kind="object",
        param="demo.TypeDerived",
        lineno_start=1,
        lineno_end=2,
        lines=[f"{C.STUB_BEGIN} object/demo.TypeDerived", C.STUB_END],
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
    generate_object(
        code,
        _default_ty_map(),
        imports,
        opts,
        info,
    )
    assert imports == []

    expected = [
        f"{C.STUB_BEGIN} object/demo.TypeDerived",
        " " * code.indent + "# fmt: off",
        *[(" " * code.indent) + line for line in info.gen_fields(_type_suffix, indent=0)],
        " " * code.indent + "# fmt: on",
        C.STUB_END,
    ]
    assert code.lines == expected


def test_generate_object_with_methods() -> None:
    code = CodeBlock(
        kind="object",
        param="demo.IntPair",
        lineno_start=1,
        lineno_end=2,
        lines=[f"{C.STUB_BEGIN} object/demo.IntPair", C.STUB_END],
    )
    opts = Options(indent=4)
    imports: list[ImportItem] = []
    info = ObjectInfo(
        fields=[],
        methods=[
            FuncInfo.from_schema(
                "demo.IntPair.__c_ffi_init__",
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
    generate_object(code, _default_ty_map(), imports, opts, info)
    assert set(imports) == {ImportItem("typing.TYPE_CHECKING")}

    assert code.lines[0] == f"{C.STUB_BEGIN} object/demo.IntPair"
    assert code.lines[-1] == C.STUB_END
    assert "# fmt: off" in code.lines[1]
    assert any("if TYPE_CHECKING:" in line for line in code.lines)
    method_lines = [
        line for line in code.lines if "def __c_ffi_init__" in line or "def sum" in line
    ]
    assert any(line.strip().startswith("def __c_ffi_init__") for line in method_lines)
    assert any(line.strip().startswith("def sum") for line in method_lines)


def test_generate_import_section_groups_modules() -> None:
    code = CodeBlock(
        kind="import-section",
        param="",
        lineno_start=1,
        lineno_end=2,
        lines=[f"{C.STUB_BEGIN} import", C.STUB_END],
    )
    imports = [
        ImportItem("typing.Any", type_checking_only=True),
        ImportItem("demo_pkg.Tensor", type_checking_only=True),
        ImportItem("demo.TestObjectBase", type_checking_only=True),
        ImportItem("custom.mod.Type", type_checking_only=True),
    ]
    opts = Options(indent=4)
    generate_import_section(code, imports, opts)

    expected_prefix = [
        f"{C.STUB_BEGIN} import",
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
    assert code.lines[-2:] == ["# fmt: on", C.STUB_END]


def test_generate_import_section_no_imports_noop() -> None:
    code = CodeBlock(
        kind="import-section",
        param="",
        lineno_start=1,
        lineno_end=2,
        lines=[f"{C.STUB_BEGIN} import", C.STUB_END],
    )
    before = list(code.lines)
    generate_import_section(code, [], Options())
    assert code.lines == before


def test_generate_all_builds_sorted_and_deduped_list() -> None:
    code = CodeBlock(
        kind="global",
        param="all",
        lineno_start=1,
        lineno_end=2,
        lines=["    " + C.STUB_BEGIN + " global/all", C.STUB_END],
    )
    generate_all(
        code,
        names={"tvm_ffi.foo", "bar", "pkg.baz", "bar"},  # duplicates stripped
        opt=Options(indent=2),
    )
    assert code.lines == [
        "    " + C.STUB_BEGIN + " global/all",
        '    "bar",',
        '    "baz",',
        '    "foo",',
        C.STUB_END,
    ]


def test_generate_all_noop_on_empty_names() -> None:
    code = CodeBlock(
        kind="global",
        param="all-empty",
        lineno_start=1,
        lineno_end=2,
        lines=[C.STUB_BEGIN + " global/all-empty", C.STUB_END],
    )
    before = list(code.lines)
    generate_all(code, names=set(), opt=Options())
    assert code.lines == before


def test_generate_all_uses_isort_style_ordering() -> None:
    code = CodeBlock(
        kind="global",
        param="all-mixed",
        lineno_start=1,
        lineno_end=2,
        lines=[C.STUB_BEGIN + " global/all-mixed", C.STUB_END],
    )
    names = {"foo", "Bar", "LIB", "baz", "Alpha", "CONST"}
    generate_all(code, names=names, opt=Options(indent=0))
    assert code.lines == [
        C.STUB_BEGIN + " global/all-mixed",
        '"CONST",',
        '"LIB",',
        '"Alpha",',
        '"Bar",',
        '"baz",',
        '"foo",',
        C.STUB_END,
    ]


def test_stage_3_adds_LIB_when_load_lib_imported(tmp_path: Path) -> None:
    path = tmp_path / "demo.py"
    global_block = CodeBlock(
        kind="global",
        param=("testing", ""),
        lineno_start=2,
        lineno_end=3,
        lines=[f"{C.STUB_BEGIN} global/testing", C.STUB_END],
    )
    import_obj_block = CodeBlock(
        kind="import-object",
        param=("tvm_ffi.libinfo.load_lib_module", "False", "_FFI_LOAD_LIB"),
        lineno_start=1,
        lineno_end=1,
        lines=[f"{C.STUB_IMPORT_OBJECT} tvm_ffi.libinfo.load_lib_module;False;_FFI_LOAD_LIB"],
    )
    all_block = CodeBlock(
        kind="__all__",
        param="",
        lineno_start=4,
        lineno_end=5,
        lines=[f"{C.STUB_BEGIN} __all__", C.STUB_END],
    )
    file_info = FileInfo(
        path=path,
        lines=tuple(
            line for block in (import_obj_block, global_block, all_block) for line in block.lines
        ),
        code_blocks=[import_obj_block, global_block, all_block],
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
    )
    lib_lines = [line for line in all_block.lines if "LIB" in line]
    assert any("LIB" in line for line in lib_lines)


def test_generate_export_builds_all_extension() -> None:
    code = CodeBlock(
        kind="export",
        param="ffi_api",
        lineno_start=1,
        lineno_end=2,
        lines=[f"{C.STUB_BEGIN} export/ffi_api", C.STUB_END],
    )
    generate_export(code)
    full_text = "\n".join(code.lines)
    assert "from .ffi_api import *" in full_text
    assert "ffi_api__all__" in full_text


def test_generate_init_with_and_without_existing_export_block() -> None:
    code_no_blocks = generate_init([], "demo")
    assert "Package demo." in code_no_blocks
    assert f"{C.STUB_BEGIN} export/_ffi_api" in code_no_blocks

    code_with_export = generate_init(
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
    )
    assert code_with_export == ""


def test_generate_ffi_api_without_objects_includes_sections() -> None:
    init_cfg = InitConfig(pkg="pkg", shared_target="pkg_shared", prefix="pkg.")
    code = generate_ffi_api(
        [],
        _default_ty_map(),
        "demo.mod",
        [],
        init_cfg,
        is_root=False,
    )
    assert f"{C.STUB_BEGIN} import-section" in code
    assert f"{C.STUB_BEGIN} global/demo.mod" in code
    assert C.STUB_BEGIN + " __all__" in code
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
    code = generate_ffi_api(
        [],
        _default_ty_map(),
        "demo",
        [obj_info],
        init_cfg,
        is_root=False,
    )
    assert C.STUB_IMPORT_OBJECT in code  # register_object prompt
    assert f"{C.STUB_BEGIN} object/{obj_info.type_key}" in code
    assert parent_key is not None
    parent_import_prompt = (
        f"{C.STUB_IMPORT_OBJECT} {parent_key};False;_{parent_key.replace('.', '_')}"
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
