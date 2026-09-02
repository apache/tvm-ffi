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
"""Tests for the native-layout classifier behind ``tvm-ffi-stubgen --coverage-out``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import tvm_ffi.stub.cli as stub_cli
import tvm_ffi.testing  # noqa: F401  (loads the `testing.*` fixture types)
from tvm_ffi import core
from tvm_ffi.core import TypeSchema
from tvm_ffi.stub.layout import (
    OBJECT_HEADER_ALIGNMENT,
    ByteRange,
    FieldBytes,
    Verdict,
    classify,
    coverage_report,
    write_coverage_report,
)
from tvm_ffi.stub.lib_state import collect_type_keys, object_info_from_type_key
from tvm_ffi.stub.utils import NamedTypeSchema, ObjectInfo

HEADER = 24  # sizeof(TVMFFIObject)


# ---------------------------------------------------------------------------
# Synthetic fixtures: the criterion is a pure function of the byte facts.
# ---------------------------------------------------------------------------


def _field(
    name: str, offset: int | None, size: int | None, alignment: int | None = None
) -> NamedTypeSchema:
    if alignment is None:
        alignment = size
    return NamedTypeSchema(name, TypeSchema("int"), size=size, alignment=alignment, offset=offset)


_ANCESTORS: dict[str, list[str]] = {}


def _info(
    type_key: str,
    *,
    parent: str | None = None,
    total_size: int | None,
    fields: tuple[NamedTypeSchema, ...] = (),
    is_final: bool | None = None,
) -> ObjectInfo:
    ancestors = [] if parent is None else [*_ANCESTORS[parent], parent]
    _ANCESTORS[type_key] = ancestors
    return ObjectInfo(
        fields=list(fields),
        methods=[],
        type_key=type_key,
        parent_type_key=parent,
        ancestors=ancestors,
        total_size=total_size,
        is_final=is_final,
    )


ROOT = _info("ffi.Object", total_size=HEADER)
# int64 @24, int32 @32, tail padding [36, 40): the shape of `testing.TestCxxClassBase`.
BASE = _info(
    "t.Base",
    parent="ffi.Object",
    total_size=40,
    fields=(_field("v_i64", 24, 8), _field("v_i32", 32, 4)),
    is_final=False,
)


def _ranges(ranges: list[ByteRange]) -> list[tuple[int, int]]:
    return [(r.start, r.end) for r in ranges]


def test_root_is_the_object_header() -> None:
    """The root of the hierarchy is the C ABI header: complete, nothing to fill."""
    verdict = classify({"ffi.Object": ROOT})["ffi.Object"]
    assert verdict.is_complete
    assert verdict.reason is None
    assert verdict.own_bytes == ByteRange(HEADER, HEADER)
    assert verdict.alignment == OBJECT_HEADER_ALIGNMENT
    assert "object header" in verdict.detail


def test_complete_with_alignment_padding() -> None:
    verdict = classify({"ffi.Object": ROOT, "t.Base": BASE})["t.Base"]
    assert verdict.is_complete
    assert verdict.reason is None
    assert verdict.own_bytes == ByteRange(HEADER, 40)
    assert verdict.alignment == 8
    assert verdict.fields == [FieldBytes("v_i64", 24, 8, 8), FieldBytes("v_i32", 32, 4, 4)]
    assert _ranges(verdict.padding) == [(36, 40)]
    assert verdict.uncovered == []
    assert "[24, 40)" in verdict.detail and "[36, 40)" in verdict.detail


def test_fields_are_visited_by_offset_not_declaration_order() -> None:
    info = _info(
        "t.Shuffled",
        parent="ffi.Object",
        total_size=40,
        fields=(_field("late", 32, 8), _field("early", 24, 8)),
    )
    verdict = classify({"ffi.Object": ROOT, "t.Shuffled": info})["t.Shuffled"]
    assert verdict.is_complete
    assert [f.name for f in verdict.fields] == ["early", "late"]


def test_uncovered_gap_in_the_middle() -> None:
    """A member the registry never saw: [32, 40) is neither a field nor alignment."""
    info = _info(
        "t.Hidden",
        parent="ffi.Object",
        total_size=48,
        fields=(_field("v_i64", 24, 8), _field("v_i32", 40, 4)),
    )
    verdict = classify({"ffi.Object": ROOT, "t.Hidden": info})["t.Hidden"]
    assert verdict.verdict == "opaque"
    assert verdict.reason == "uncovered-bytes"
    assert _ranges(verdict.uncovered) == [(32, 40)]
    assert _ranges(verdict.padding) == [(44, 48)]  # the tail is still explained by alignment
    assert "[32, 40)" in verdict.detail


def test_uncovered_tail() -> None:
    """The polymorphic shape: offsets are header-relative, `sizeof` is absolute."""
    info = _info("t.Poly", parent="ffi.Object", total_size=40, fields=(_field("v_i64", 24, 8),))
    verdict = classify({"ffi.Object": ROOT, "t.Poly": info})["t.Poly"]
    assert verdict.reason == "uncovered-bytes"
    assert _ranges(verdict.uncovered) == [(32, 40)]
    assert verdict.padding == []


def test_tail_gap_is_padding_only_when_alignment_forces_it() -> None:
    padded = _info("t.Pad", parent="ffi.Object", total_size=32, fields=(_field("v_i8", 24, 1),))
    hole = _info("t.Hole", parent="ffi.Object", total_size=40, fields=(_field("v_i8", 24, 1),))
    verdicts = classify({"ffi.Object": ROOT, "t.Pad": padded, "t.Hole": hole})
    assert verdicts["t.Pad"].is_complete
    assert _ranges(verdicts["t.Pad"].padding) == [(25, 32)]
    assert verdicts["t.Hole"].reason == "uncovered-bytes"
    assert _ranges(verdicts["t.Hole"].uncovered) == [(25, 40)]


def test_alignment_propagates_along_the_chain() -> None:
    base = _info("t.A", parent="ffi.Object", total_size=32, fields=(_field("a", 24, 8),))
    wide = _info("t.B", parent="t.A", total_size=48, fields=(_field("b", 32, 16, 16),))
    leaf = _info("t.C", parent="t.B", total_size=64, fields=(_field("c", 48, 4),))
    verdicts = classify({"ffi.Object": ROOT, "t.A": base, "t.B": wide, "t.C": leaf})
    assert [verdicts[k].alignment for k in ("t.A", "t.B", "t.C")] == [8, 16, 16]
    assert verdicts["t.C"].is_complete
    assert _ranges(verdicts["t.C"].padding) == [(52, 64)]  # the 16-byte alignment explains the tail


def test_no_own_metadata_is_layout_unknown_and_propagates() -> None:
    inherited = _info("t.NoMeta", parent="ffi.Object", total_size=None)
    child = _info("t.Child", parent="t.NoMeta", total_size=48, fields=(_field("x", 40, 8),))
    verdicts = classify({"ffi.Object": ROOT, "t.NoMeta": inherited, "t.Child": child})
    assert verdicts["t.NoMeta"].reason == "layout-unknown"
    assert verdicts["t.NoMeta"].own_bytes is None
    assert verdicts["t.Child"].reason == "parent-opaque"
    assert "'t.NoMeta'" in verdicts["t.Child"].detail
    assert "layout-unknown" in verdicts["t.Child"].detail
    # The child's own byte facts are still recorded as evidence.
    assert verdicts["t.Child"].fields == [FieldBytes("x", 40, 8, 8)]


def test_field_without_byte_facts_is_layout_unknown() -> None:
    info = _info("t.NoFacts", parent="ffi.Object", total_size=32, fields=(_field("x", None, None),))
    verdict = classify({"ffi.Object": ROOT, "t.NoFacts": info})["t.NoFacts"]
    assert verdict.reason == "layout-unknown"
    assert "'x'" in verdict.detail


def test_field_overlap() -> None:
    """A field placed in the parent's tail padding (Itanium ABI) cannot be nested by value."""
    reuse = _info("t.TailReuse", parent="t.Base", total_size=40, fields=(_field("v_tail", 36, 4),))
    overflow = _info("t.Overflow", parent="ffi.Object", total_size=24, fields=(_field("a", 24, 8),))
    verdicts = classify(
        {"ffi.Object": ROOT, "t.Base": BASE, "t.TailReuse": reuse, "t.Overflow": overflow}
    )
    assert verdicts["t.TailReuse"].reason == "field-overlap"
    assert "'v_tail' at [36, 40) starts before byte 40" in verdicts["t.TailReuse"].detail
    assert verdicts["t.Overflow"].reason == "field-overlap"
    assert "past total_size 24" in verdicts["t.Overflow"].detail


def test_forced_opaque_only_demotes_a_complete_type() -> None:
    inherited = _info("t.NoMeta", parent="ffi.Object", total_size=None)
    infos = {"ffi.Object": ROOT, "t.Base": BASE, "t.NoMeta": inherited}
    verdicts = classify(infos, forced_opaque={"t.Base", "t.NoMeta"})
    assert verdicts["t.Base"].reason == "by-directive"
    assert verdicts["t.Base"].own_bytes == ByteRange(HEADER, 40)  # the evidence is still there
    assert verdicts["t.NoMeta"].reason == "layout-unknown"  # a layout reason wins over the veto


def test_field_renderable_predicate() -> None:
    seen: list[str] = []

    def renderable(field: NamedTypeSchema) -> bool:
        seen.append(field.name)
        return field.name != "v_i32"

    hidden = _info("t.Hidden", parent="ffi.Object", total_size=48, fields=(_field("v_i64", 24, 8),))
    infos = {"ffi.Object": ROOT, "t.Base": BASE, "t.Hidden": hidden}
    verdicts = classify(infos, field_renderable=renderable)
    assert verdicts["t.Base"].reason == "unrenderable-field"
    assert "'v_i32'" in verdicts["t.Base"].detail
    # A type whose bytes already fail is not asked about renderability.
    assert verdicts["t.Hidden"].reason == "uncovered-bytes"
    assert seen == ["v_i64", "v_i32"]


def test_children_of_any_opaque_parent_are_parent_opaque() -> None:
    child = _info("t.Child", parent="t.Base", total_size=48, fields=(_field("x", 40, 8),))
    infos = {"ffi.Object": ROOT, "t.Base": BASE, "t.Child": child}
    verdicts = classify(infos, forced_opaque={"t.Base"})
    assert verdicts["t.Child"].reason == "parent-opaque"
    assert "by-directive" in verdicts["t.Child"].detail


def test_order_does_not_matter_but_ancestors_must_be_present() -> None:
    child = _info("t.Child", parent="t.Base", total_size=48, fields=(_field("x", 40, 8),))
    verdicts = classify({"t.Child": child, "t.Base": BASE, "ffi.Object": ROOT})
    assert verdicts["t.Child"].is_complete
    with pytest.raises(KeyError, match=r"t\.Base"):
        classify({"t.Child": child})


def test_coverage_report_shape(tmp_path: Path) -> None:
    hidden = _info("t.Hidden", parent="ffi.Object", total_size=48, fields=(_field("v_i64", 24, 8),))
    verdicts = classify({"t.Hidden": hidden, "t.Base": BASE, "ffi.Object": ROOT})
    report = coverage_report(verdicts)
    assert list(report) == ["ffi.Object", "t.Base", "t.Hidden"]  # sorted
    entry = report["t.Base"]
    assert entry == {
        "verdict": "complete",
        "reason": None,
        "detail": entry["detail"],
        "parent": "ffi.Object",
        "ancestors": ["ffi.Object"],
        "total_size": 40,
        "is_final": False,
        "own_bytes": [24, 40],
        "fields": [
            {"name": "v_i64", "offset": 24, "size": 8, "alignment": 8},
            {"name": "v_i32", "offset": 32, "size": 4, "alignment": 4},
        ],
        "uncovered": [],
        "padding": [[36, 40]],
    }
    assert report["t.Hidden"]["reason"] == "uncovered-bytes"
    assert report["t.Hidden"]["uncovered"] == [[32, 48]]

    out = tmp_path / "coverage.json"
    write_coverage_report(out, verdicts)
    assert json.loads(out.read_text(encoding="utf-8")) == report


# ---------------------------------------------------------------------------
# The real registry: the `testing.*` fixtures exercise every path in CI.
# ---------------------------------------------------------------------------


def _registry_verdicts() -> dict[str, Verdict]:
    infos = {
        type_key: object_info_from_type_key(type_key)
        for type_keys in collect_type_keys().values()
        for type_key in type_keys
    }
    return classify(infos)


def test_registry_complete_chain() -> None:
    verdicts = _registry_verdicts()
    assert verdicts["ffi.Object"].is_complete
    base = verdicts["testing.TestCxxClassBase"]
    derived = verdicts["testing.TestCxxClassDerived"]
    dd = verdicts["testing.TestCxxClassDerivedDerived"]
    assert base.is_complete and derived.is_complete and dd.is_complete
    assert (base.own_bytes, derived.own_bytes, dd.own_bytes) == (
        ByteRange(24, 40),
        ByteRange(40, 56),
        ByteRange(56, 80),
    )
    assert _ranges(base.padding) == [(36, 40)]
    assert _ranges(derived.padding) == [(52, 56)]
    assert _ranges(dd.padding) == [(73, 80)]
    # `ffi.Function` never registered an `ObjectDef`: nothing to check against.
    assert verdicts["ffi.Function"].reason == "layout-unknown"


def test_registry_hidden_field() -> None:
    """A member the registry never saw shows up as an exact byte range."""
    verdict = _registry_verdicts()["testing.TestCxxClassHiddenField"]
    assert verdict.reason == "uncovered-bytes"
    assert verdict.total_size == 48
    assert verdict.fields == [FieldBytes("v_i64", 24, 8, 8), FieldBytes("v_i32", 40, 4, 4)]
    assert _ranges(verdict.uncovered) == [(32, 40)]
    assert _ranges(verdict.padding) == [(44, 48)]
    assert verdict.is_final is True


def test_registry_polymorphic() -> None:
    """A vptr shifts `sizeof` by a pointer, and no polymorphism flag is needed to see it."""
    verdicts = _registry_verdicts()
    verdict = verdicts["testing.TestCxxClassPolymorphic"]
    assert verdict.reason == "uncovered-bytes"
    assert verdict.total_size == 40  # vptr + header + int64
    assert verdict.fields == [FieldBytes("v_i64", 24, 8, 8)]
    assert _ranges(verdict.uncovered) == [(32, 40)]
    # `ffi.Module` is the same shape in production code.
    assert verdicts["ffi.Module"].reason == "uncovered-bytes"


def test_registry_tolerates_pending_py_class_registration() -> None:
    """A ``py_class`` whose registration never completed has no fields yet; the sweep must not crash."""
    parent_info = core._type_cls_to_type_info(core.Object)
    assert parent_info is not None
    cls = type("PendingLayout", (core.Object,), {"__slots__": ()})
    core._register_py_class(parent_info, "testing.stub_layout.PendingLayout", cls)
    verdict = _registry_verdicts()["testing.stub_layout.PendingLayout"]
    assert verdict.reason == "layout-unknown"
    assert verdict.fields == []


def test_cli_coverage_out_without_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "coverage.json"
    monkeypatch.setattr("sys.argv", ["tvm-ffi-stubgen", "--coverage-out", str(out)])
    assert stub_cli.__main__() == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["ffi.Object"]["verdict"] == "complete"
    assert report["testing.TestCxxClassBase"]["verdict"] == "complete"
    assert report["testing.TestCxxClassHiddenField"]["uncovered"] == [[32, 40]]
    assert report["testing.TestCxxClassPolymorphic"]["reason"] == "uncovered-bytes"
    assert report["ffi.Function"]["reason"] == "layout-unknown"


def test_cli_still_requires_files_without_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["tvm-ffi-stubgen"])
    with pytest.raises(SystemExit):
        stub_cli.__main__()
