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
"""Native-layout classification for ``tvm-ffi-stubgen``.

A generator for a language with native structs can hold a reflected object *by
value* only when it can reproduce the object's memory layout byte for byte. The
registry publishes what the compiler computed: each type's own ``sizeof``
(:attr:`.ObjectInfo.total_size`) and every reflected field's size, alignment and
offset. One criterion decides the question:

    The reflected fields must fill ``[parent.total_size, total_size)`` exactly,
    allowing only the gaps alignment forces.

The cursor starts at the parent's size; each field, by ascending offset, must
start at ``align_up(cursor, field.alignment)``; at the end
``align_up(cursor, alignment)`` must equal ``total_size``, where ``alignment`` is
the largest alignment along the chain, the header's included. Two facts must
exist before the criterion can be evaluated: the type needs metadata of its own
(a type without an ``ObjectDef`` inherits its parent's entry, whose size says
nothing about it) and its parent must itself be complete. Nothing else is a
rule: a class with a vptr, for instance, reports offsets relative to the
``TVMFFIObject`` header while its ``sizeof`` is absolute, so its fields can never
fill the region.

``opaque`` is a normal, final verdict: a generator still emits a wrapper and
reaches the fields through the C ABI. Two target-language rules are injected by
the caller instead of living here: a field-renderability predicate, and a set of
semantic vetoes for types that must never be allocated outside their runtime.

The coverage report fixes the language-neutral keys of each entry; a generator
adds its own facts under a per-type ``"target"`` key.
"""

from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING, Any, Callable, Literal

from typing_extensions import TypeAlias

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Set as AbstractSet
    from pathlib import Path

    from .utils import NamedTypeSchema, ObjectInfo

#: Natural alignment of the ``TVMFFIObject`` header (it holds a 64-bit reference
#: count); every object embeds the header first, so none is aligned less strictly.
OBJECT_HEADER_ALIGNMENT = 8

OpaqueReason: TypeAlias = Literal[
    "layout-unknown",
    "parent-opaque",
    "field-overlap",
    "uncovered-bytes",
    "unrenderable-field",
    "by-directive",
]
"""Why a type is opaque.

- ``layout-unknown``: no metadata of its own, or a field without byte facts.
- ``parent-opaque``: the parent is opaque, so the region to fill has no
  trustworthy start.
- ``field-overlap``: a field starts before the cursor (e.g. inside the parent's
  tail padding, which the Itanium ABI may reuse) or ends past ``total_size``.
- ``uncovered-bytes``: a gap alignment does not explain (an unreflected member,
  a vptr, ...).
- ``unrenderable-field``: the caller's predicate rejected a field.
- ``by-directive``: the caller vetoed a type whose layout is reproducible.
"""


def align_up(value: int, alignment: int) -> int:
    """Round ``value`` up to the next multiple of ``alignment``."""
    return (value + alignment - 1) // alignment * alignment


@dataclasses.dataclass(frozen=True)
class ByteRange:
    """A half-open byte interval ``[start, end)`` inside an object."""

    start: int
    end: int

    def __str__(self) -> str:
        return f"[{self.start}, {self.end})"

    def to_json_obj(self) -> list[int]:
        """Render as ``[start, end]``."""
        return [self.start, self.end]


@dataclasses.dataclass(frozen=True)
class FieldBytes:
    """The byte facts of one reflected field."""

    name: str
    offset: int
    size: int
    alignment: int

    @property
    def end(self) -> int:
        """One past the last byte of the field."""
        return self.offset + self.size

    def to_json_obj(self) -> dict[str, Any]:
        """Render as a JSON object."""
        return dataclasses.asdict(self)


@dataclasses.dataclass
class Verdict:
    """The layout verdict of one type, with the byte evidence behind it."""

    type_key: str
    verdict: Literal["complete", "opaque"]
    reason: OpaqueReason | None
    """``None`` for a complete type; otherwise which rule demoted it."""
    detail: str
    """Human-readable explanation, quoting the byte evidence."""
    parent_type_key: str | None
    ancestors: list[str]
    total_size: int | None
    is_final: bool | None
    alignment: int | None = None
    """Natural alignment of the object, derived along the chain (drives the tail check)."""
    own_bytes: ByteRange | None = None
    """The region this type's own fields must fill: ``[parent.total_size, total_size)``."""
    fields: list[FieldBytes] = dataclasses.field(default_factory=list)
    """This type's own reflected fields by ascending offset; empty when the layout is unknown."""
    uncovered: list[ByteRange] = dataclasses.field(default_factory=list)
    """Bytes inside ``own_bytes`` that neither a field nor an alignment gap accounts for."""
    padding: list[ByteRange] = dataclasses.field(default_factory=list)
    """Alignment-forced gaps, recorded so a reviewer can check them against the C++ declaration."""

    @property
    def is_complete(self) -> bool:
        """Whether the native layout is reproducible from the reflected fields."""
        return self.verdict == "complete"

    def to_json_obj(self) -> dict[str, Any]:
        """Render the language-neutral report entry for this type."""
        return {
            "verdict": self.verdict,
            "reason": self.reason,
            "detail": self.detail,
            "parent": self.parent_type_key,
            "ancestors": list(self.ancestors),
            "total_size": self.total_size,
            "is_final": self.is_final,
            "own_bytes": None if self.own_bytes is None else self.own_bytes.to_json_obj(),
            "fields": [f.to_json_obj() for f in self.fields],
            "uncovered": [r.to_json_obj() for r in self.uncovered],
            "padding": [r.to_json_obj() for r in self.padding],
        }


def classify(
    infos: Mapping[str, ObjectInfo],
    *,
    forced_opaque: AbstractSet[str] = frozenset(),
    field_renderable: Callable[[NamedTypeSchema], bool] | None = None,
) -> dict[str, Verdict]:
    """Classify every type in ``infos``, parents before children.

    Parameters
    ----------
    infos
        The types to classify, keyed by type key, in any order. Every ancestor
        of a type must be present as well.
    forced_opaque
        Type keys vetoed by the caller (semantic blockers such as interned
        identities). The veto only demotes a type that would otherwise be
        complete; a type that is opaque for a layout reason keeps that reason.
    field_renderable
        Predicate deciding whether a reflected field has a native mirror in the
        target language. A rejected field demotes its type to opaque with
        reason ``unrenderable-field``. ``None`` accepts every field.

    """
    verdicts: dict[str, Verdict] = {}

    def _classify(type_key: str) -> Verdict:
        if type_key in verdicts:
            return verdicts[type_key]
        if type_key not in infos:
            raise KeyError(f"Ancestor {type_key!r} is not among the types to classify")
        info = infos[type_key]
        parent = None if info.parent_type_key is None else _classify(info.parent_type_key)
        verdicts[type_key] = _classify_one(info, parent, forced_opaque, field_renderable)
        return verdicts[type_key]

    for type_key in infos:
        _classify(type_key)
    return verdicts


def _classify_one(
    info: ObjectInfo,
    parent: Verdict | None,
    forced_opaque: AbstractSet[str],
    field_renderable: Callable[[NamedTypeSchema], bool] | None,
) -> Verdict:
    assert info.type_key is not None, "cannot classify an ObjectInfo without a type key"
    verdict = Verdict(
        type_key=info.type_key,
        verdict="opaque",
        reason=None,
        detail="",
        parent_type_key=info.parent_type_key,
        ancestors=list(info.ancestors),
        total_size=info.total_size,
        is_final=info.is_final,
    )
    outcome = _prove_layout(info, parent, verdict)
    if outcome is None:
        outcome = _apply_target_rules(info, forced_opaque, field_renderable)
    if outcome is not None:
        verdict.reason, verdict.detail = outcome
        return verdict
    verdict.verdict = "complete"
    if parent is None:
        verdict.detail = f"object header [0, {info.total_size}): owned by the C ABI"
    else:
        verdict.detail = f"reflected fields fill {verdict.own_bytes} exactly"
        if verdict.padding:
            gaps = ", ".join(str(r) for r in verdict.padding)
            verdict.detail += f" (alignment padding {gaps})"
    return verdict


def _prove_layout(
    info: ObjectInfo, parent: Verdict | None, verdict: Verdict
) -> tuple[OpaqueReason, str] | None:
    """Check the prerequisites and the fill criterion, recording the evidence on ``verdict``.

    Returns ``None`` when the layout is reproducible, else the reason it is not.
    """
    # Prerequisite 1: the type must have metadata of its own, and its fields byte facts.
    if info.total_size is None:
        return "layout-unknown", "no metadata of its own: total_size is unknown"
    fields = _field_bytes(info)
    if isinstance(fields, str):
        return "layout-unknown", fields
    verdict.fields = fields

    # Prerequisite 2: the parent must be complete, so that its size is a trustworthy start.
    if parent is None:
        # The root of the hierarchy is the `TVMFFIObject` header itself: its bytes belong to
        # the C ABI, not to reflected fields, and children build on its size.
        verdict.alignment = OBJECT_HEADER_ALIGNMENT
        verdict.own_bytes = ByteRange(info.total_size, info.total_size)
        return None
    if not parent.is_complete:
        return "parent-opaque", f"parent {parent.type_key!r} is opaque ({parent.reason})"
    assert parent.total_size is not None and parent.alignment is not None

    # The criterion: fields must fill [parent.total_size, total_size) exactly.
    verdict.own_bytes = ByteRange(parent.total_size, info.total_size)
    verdict.alignment = max([parent.alignment, *(f.alignment for f in fields)])
    return _fill(verdict)


def _field_bytes(info: ObjectInfo) -> list[FieldBytes] | str:
    """Return the type's own fields by ascending offset, or why their bytes are unknown."""
    fields: list[FieldBytes] = []
    for f in info.fields:
        if f.size is None or f.alignment is None or f.offset is None:
            return f"field {f.name!r} carries no native layout facts"
        fields.append(FieldBytes(name=f.name, offset=f.offset, size=f.size, alignment=f.alignment))
    return sorted(fields, key=lambda f: f.offset)


def _fill(verdict: Verdict) -> tuple[OpaqueReason, str] | None:
    """Walk ``verdict.fields`` over ``verdict.own_bytes``, sorting every gap into padding or hole.

    Returns why the fill fails (the first overlap, else the uncovered bytes), or ``None``.
    """
    assert verdict.own_bytes is not None and verdict.alignment is not None
    end = verdict.own_bytes.end
    overlap: str | None = None
    cursor = verdict.own_bytes.start
    for field in verdict.fields:
        if field.offset < cursor:
            where = f"[{field.offset}, {field.end}) starts before byte {cursor}"
            overlap = overlap or f"field {field.name!r} at {where}"
            cursor = max(cursor, field.end)
            continue
        _record_gap(verdict, cursor, align_up(cursor, field.alignment), field.offset)
        cursor = field.end
    if cursor > end:
        overlap = overlap or f"fields extend to byte {cursor}, past total_size {end}"
    else:
        _record_gap(verdict, cursor, align_up(cursor, verdict.alignment), end)
    if overlap is not None:
        return "field-overlap", overlap
    if verdict.uncovered:
        ranges = ", ".join(str(r) for r in verdict.uncovered)
        return "uncovered-bytes", (
            f"bytes {ranges} of {verdict.own_bytes} are not accounted for by reflected fields"
        )
    return None


def _record_gap(verdict: Verdict, cursor: int, expected: int, actual: int) -> None:
    """Classify the gap between ``cursor`` and the next boundary ``actual``.

    ``expected`` is where alignment alone would put that boundary: a gap that
    ends exactly there is padding, any other gap is a hole.
    """
    if expected != actual:
        verdict.uncovered.append(ByteRange(cursor, actual))
    elif expected != cursor:
        verdict.padding.append(ByteRange(cursor, expected))


def _apply_target_rules(
    info: ObjectInfo,
    forced_opaque: AbstractSet[str],
    field_renderable: Callable[[NamedTypeSchema], bool] | None,
) -> tuple[OpaqueReason, str] | None:
    """Apply the caller-injected, target-language rules to a type whose layout is proven."""
    if field_renderable is not None:
        for field in info.fields:
            if not field_renderable(field):
                return "unrenderable-field", (
                    f"field {field.name!r} ({field.repr()}) has no native mirror"
                )
    if info.type_key in forced_opaque:
        return "by-directive", "vetoed by directive although the layout is reproducible"
    return None


def coverage_report(verdicts: Mapping[str, Verdict]) -> dict[str, Any]:
    """Build the JSON-serialisable coverage report: one entry per type key, sorted.

    Each entry holds the language-neutral keys rendered by
    :meth:`Verdict.to_json_obj`; a generator may add a ``"target"`` key with
    its own facts.
    """
    return {key: verdicts[key].to_json_obj() for key in sorted(verdicts)}


def write_coverage_report(path: Path, verdicts: Mapping[str, Verdict]) -> None:
    """Write :func:`coverage_report` to ``path`` as indented JSON."""
    path.write_text(json.dumps(coverage_report(verdicts), indent=2) + "\n", encoding="utf-8")
