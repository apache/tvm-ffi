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
from pathlib import Path

import pytest
from tvm_ffi.core import TypeSchema
from tvm_ffi.stub import consts as C
from tvm_ffi.stub.codegen import (
    generate_all,
    generate_global_funcs,
    generate_imports,
    generate_object,
)
from tvm_ffi.stub.file_utils import CodeBlock, FileInfo
from tvm_ffi.stub.utils import FuncInfo, NamedTypeSchema, ObjectInfo, Options


def _identity_ty_map(name: str) -> str:
    return name


def test_codeblock_from_begin_line_variants() -> None:
    cases = [
        (f"{C.STUB_BEGIN} global/example", "global", "example"),
        (f"{C.STUB_BEGIN} object/testing.TestObjectBase", "object", "testing.TestObjectBase"),
        (f"{C.STUB_BEGIN} ty-map/custom", "ty-map", "custom"),
        (f"{C.STUB_BEGIN} import", "import", ""),
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
    assert stub.param == "demo.func"
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
        param="testing",
        lineno_start=1,
        lineno_end=2,
        lines=[f"{C.STUB_BEGIN} global/testing", C.STUB_END],
    )
    funcs = [
        FuncInfo(
            schema=NamedTypeSchema(
                "testing.add_one",
                TypeSchema("Callable", (TypeSchema("int"), TypeSchema("int"))),
            ),
            is_member=False,
        )
    ]
    opts = Options(indent=2)
    generate_global_funcs(code, funcs, _identity_ty_map, opts)
    assert code.lines == [
        f"{C.STUB_BEGIN} global/testing",
        "# fmt: off",
        "if TYPE_CHECKING:",
        "  def add_one(_0: int, /) -> int: ...",
        "# fmt: on",
        C.STUB_END,
    ]


def test_generate_global_funcs_noop_on_empty_list() -> None:
    code = CodeBlock(
        kind="global",
        param="empty",
        lineno_start=1,
        lineno_end=2,
        lines=[f"{C.STUB_BEGIN} global/empty", C.STUB_END],
    )
    generate_global_funcs(code, [], _identity_ty_map, Options())
    assert code.lines == [f"{C.STUB_BEGIN} global/empty", C.STUB_END]


def test_generate_object_fields_only_block() -> None:
    code = CodeBlock(
        kind="object",
        param="testing.TestObjectDerived",
        lineno_start=1,
        lineno_end=2,
        lines=[f"{C.STUB_BEGIN} object/testing.TestObjectDerived", C.STUB_END],
    )
    opts = Options(indent=4)
    generate_object(code, _identity_ty_map, opts)

    info = ObjectInfo.from_type_key("testing.TestObjectDerived")
    expected = [
        f"{C.STUB_BEGIN} object/testing.TestObjectDerived",
        " " * code.indent + "# fmt: off",
        *[(" " * code.indent) + line for line in info.gen_fields(_identity_ty_map, indent=0)],
        " " * code.indent + "# fmt: on",
        C.STUB_END,
    ]
    assert code.lines == expected


def test_generate_object_with_methods() -> None:
    code = CodeBlock(
        kind="object",
        param="testing.TestIntPair",
        lineno_start=1,
        lineno_end=2,
        lines=[f"{C.STUB_BEGIN} object/testing.TestIntPair", C.STUB_END],
    )
    opts = Options(indent=4)
    generate_object(code, _identity_ty_map, opts)

    assert code.lines[0] == f"{C.STUB_BEGIN} object/testing.TestIntPair"
    assert code.lines[-1] == C.STUB_END
    assert "# fmt: off" in code.lines[1]
    assert any("if TYPE_CHECKING:" in line for line in code.lines)
    method_lines = [
        line for line in code.lines if "def __c_ffi_init__" in line or "def sum" in line
    ]
    assert any(line.strip().startswith("def __c_ffi_init__") for line in method_lines)
    assert any(line.strip().startswith("def sum") for line in method_lines)


def test_generate_imports_groups_modules() -> None:
    code = CodeBlock(
        kind="import",
        param="",
        lineno_start=1,
        lineno_end=2,
        lines=[f"{C.STUB_BEGIN} import", C.STUB_END],
    )
    ty_used = {
        "typing.Any",
        "tvm_ffi.Tensor",
        "testing.TestObjectBase",
        "custom.mod.Type",
    }
    opts = Options(indent=4)
    generate_imports(code, ty_used, opts)

    expected_prefix = [
        f"{C.STUB_BEGIN} import",
        "# fmt: off",
        "# isort: off",
        "from __future__ import annotations",
        "from typing import Any, TYPE_CHECKING",
        "if TYPE_CHECKING:",
    ]
    assert code.lines[: len(expected_prefix)] == expected_prefix
    assert "    from tvm_ffi.testing import TestObjectBase" in code.lines
    assert "    from tvm_ffi import Tensor" in code.lines
    assert "    from custom.mod import Type" in code.lines
    assert code.lines[-2:] == ["# fmt: on", C.STUB_END]


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
