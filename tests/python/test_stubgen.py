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

import re
from pathlib import Path

import pytest
from tvm_ffi.stub import stubgen


def test_stubgen_skip_file(tmp_path: Path) -> None:
    p: Path = tmp_path / "dummy.py"
    src = (
        "# tvm-ffi-stubgen(skip-file)\n"
        "from typing import TYPE_CHECKING\n\n"
        "# tvm-ffi-stubgen(begin): global/ffi\n"
        "if TYPE_CHECKING:\n"
        "    pass\n"
        "# tvm-ffi-stubgen(end)\n"
    )
    p.write_text(src, encoding="utf-8")
    # Run the generator; it should skip without trying to query the registry
    stubgen._main(p, stubgen.Options(indent=4, suppress_print=True))
    # File must be unchanged
    assert p.read_text(encoding="utf-8") == src


def test_stubgen_global_block_generates_and_indents(tmp_path: Path) -> None:
    p: Path = tmp_path / "gen_global.py"
    # Indent begin by 2 spaces; inner indent is begin-indent + opt.indent (3)
    src = (
        "from typing import TYPE_CHECKING\n\n"
        "  # tvm-ffi-stubgen(begin): global/ffi\n"
        "  # tvm-ffi-stubgen(end)\n"
    )
    p.write_text(src, encoding="utf-8")

    stubgen._main(p, stubgen.Options(indent=3, suppress_print=True))

    out = p.read_text(encoding="utf-8").splitlines()

    # Expect TYPE_CHECKING guard with the same begin indentation
    assert any(line == "  if TYPE_CHECKING:" for line in out)
    # Expect formatting guards
    assert any(line == "     # fmt: off" for line in out)
    assert any(line == "     # fmt: on" for line in out)
    # Expect at least one known ffi function signature, e.g. String(...)-> str
    string_lines = [ln for ln in out if re.search(r"\bdef\s+String\(.*\)\s*->\s*str:\s*\.\.\.", ln)]
    assert string_lines, "Expected stub for ffi.String"
    # Check inner indent equals begin (2) + opt.indent (3) = 5 spaces
    assert all(ln.startswith(" " * 5) for ln in string_lines)

    # Idempotency: second run should keep file unchanged
    before = "\n".join(out) + "\n"
    stubgen._main(p, stubgen.Options(indent=3, suppress_print=True))
    assert p.read_text(encoding="utf-8") == before


def test_stubgen_global_block_no_matches_is_noop(tmp_path: Path) -> None:
    p: Path = tmp_path / "gen_global_empty.py"
    src = "# tvm-ffi-stubgen(begin): global/this_prefix_does_not_exist\n# tvm-ffi-stubgen(end)\n"
    p.write_text(src, encoding="utf-8")
    stubgen._main(p, stubgen.Options(indent=4, suppress_print=True))
    assert p.read_text(encoding="utf-8") == src


def test_stubgen_object_block_generates_fields_and_methods(tmp_path: Path) -> None:
    # Ensure object type registrations are loaded

    p: Path = tmp_path / "gen_object_pair.py"
    src = (
        "class _C:\n"
        "    # tvm-ffi-stubgen(begin): object/testing.TestIntPair\n"
        "    # tvm-ffi-stubgen(end)\n"
    )
    p.write_text(src, encoding="utf-8")

    stubgen._main(p, stubgen.Options(indent=4, suppress_print=True))
    out = p.read_text(encoding="utf-8").splitlines()

    # Fields a and b should be generated
    assert any("        a: int" == ln for ln in out)
    assert any("        b: int" == ln for ln in out)
    # __ffi_init__ should be exposed as __c_ffi_init__ and marked staticmethod
    init_idx = next(i for i, ln in enumerate(out) if "def __c_ffi_init__(" in ln)
    assert out[init_idx - 1].strip() == "@staticmethod"


def test_stubgen_object_block_with_ty_map_and_collections(tmp_path: Path) -> None:
    # Ensure type info for SchemaAllTypes is available
    p: Path = tmp_path / "gen_object_schema.py"
    src = (
        "# tvm-ffi-stubgen(begin): object/testing.SchemaAllTypes\n"
        "# tvm-ffi-stubgen(ty_map): testing.SchemaAllTypes -> _SchemaAllTypes\n"
        "# tvm-ffi-stubgen(end)\n"
    )
    p.write_text(src, encoding="utf-8")

    stubgen._main(p, stubgen.Options(indent=4, suppress_print=True))
    text = p.read_text(encoding="utf-8")

    # Mapped container aliases should appear
    assert "Sequence[int]" in text
    assert "Mapping[str, Sequence[int]]" in text
    # Method types reflect mapping of the object type
    assert re.search(r"def\s+add_int\(_0: _SchemaAllTypes, _1: int, /\)\s*->\s*int:\s*\.\.\.", text)
    # Static factory returns the mapped type
    assert re.search(
        r"@staticmethod\s*\n\s*def\s+make_with\(.*\)\s*->\s*_SchemaAllTypes:\s*\.\.\.", text
    )


def test_stubgen_errors_for_invalid_directives(tmp_path: Path) -> None:
    # ty_map outside a block
    p1 = tmp_path / "invalid_ty_map_outside.py"
    p1.write_text("# tvm-ffi-stubgen(ty_map): A.B -> C\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Stub ty_map outside stub block"):
        stubgen._main(p1, stubgen.Options(suppress_print=True))

    # invalid ty_map format inside a block
    p2 = tmp_path / "invalid_ty_map_format.py"
    p2.write_text(
        (
            "# tvm-ffi-stubgen(begin): object/testing.TestObjectBase\n"
            "# tvm-ffi-stubgen(ty_map): not_a_map_line\n"
            "# tvm-ffi-stubgen(end)\n"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"Invalid ty_map format"):
        stubgen._main(p2, stubgen.Options(suppress_print=True))


def test_stubgen_errors_for_block_structure(tmp_path: Path) -> None:
    # Nested stub blocks are not allowed
    p_nested = tmp_path / "nested.py"
    p_nested.write_text(
        (
            "# tvm-ffi-stubgen(begin): global/ffi\n"
            "    # tvm-ffi-stubgen(begin): global/ffi\n"
            "# tvm-ffi-stubgen(end)\n"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"Nested stub not permitted"):
        stubgen._main(p_nested, stubgen.Options(suppress_print=True))

    # Unmatched end
    p_unmatched_end = tmp_path / "unmatched_end.py"
    p_unmatched_end.write_text("# tvm-ffi-stubgen(end)\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"Unmatched stub end"):
        stubgen._main(p_unmatched_end, stubgen.Options(suppress_print=True))

    # Unknown stub type
    p_unknown = tmp_path / "unknown.py"
    p_unknown.write_text(
        ("# tvm-ffi-stubgen(begin): unknown/foo\n# tvm-ffi-stubgen(end)\n"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"Unknown stub type"):
        stubgen._main(p_unknown, stubgen.Options(suppress_print=True))

    # Unclosed block
    p_unclosed = tmp_path / "unclosed.py"
    p_unclosed.write_text("# tvm-ffi-stubgen(begin): global/ffi\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"Unclosed stub block"):
        stubgen._main(p_unclosed, stubgen.Options(suppress_print=True))
