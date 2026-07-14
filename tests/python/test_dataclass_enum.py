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
"""Tests for FFI enum singletons and their canonical indices."""

from __future__ import annotations

import copy
import itertools
import json
from typing import Any, ClassVar

import pytest
import tvm_ffi
from tvm_ffi import Object, core
from tvm_ffi.dataclasses import (
    Enum,
    EnumAttrMap,
    EnumState,
    IntEnum,
    StrEnum,
    auto,
    entry,
    py_class,
)
from tvm_ffi.dataclasses.enum import ENUM_STATE_ATTR, _EnumEntry
from tvm_ffi.serialization import from_json_graph_str, to_json_graph_str
from tvm_ffi.testing import _TestCxxEnumHolder, _TestCxxIntEnum, _TestCxxStrEnum

_counter = itertools.count()
_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1


def _key(name: str) -> str:
    return f"testing.py_enum.{name}.{next(_counter)}"


def _state(cls: type) -> Any:
    info = cls.__tvm_ffi_type_info__  # ty: ignore[unresolved-attribute]
    return core._lookup_type_attr(info.type_index, ENUM_STATE_ATTR)


def _assert_json_roundtrip(member: Enum, data: int | str) -> None:
    encoded = to_json_graph_str(member)
    graph = json.loads(encoded)
    assert graph["nodes"][graph["root_index"]]["data"] == data
    assert from_json_graph_str(encoded).same_as(member)


def test_plain_enum_indices_fields_and_order() -> None:
    class Activation(Enum, type_key=_key("Activation")):
        arity: int = 0
        relu: ClassVar[Activation] = entry(arity=1)
        add = entry(arity=2)
        noop = auto()

    assert (Activation.relu._int_index, Activation.relu._str_index) == (0, "relu")
    assert (Activation.add._int_index, Activation.add._str_index) == (1, "add")
    assert Activation.relu.arity == 1
    assert Activation.noop.arity == 0
    assert list(Activation) == [Activation.relu, Activation.add, Activation.noop]
    assert len(Activation) == 3
    assert Activation.get(0).same_as(Activation.relu)
    assert Activation.get("relu").same_as(Activation.relu)
    assert Activation(1).same_as(Activation.add)
    assert Activation("add").same_as(Activation.add)

    with pytest.raises(AttributeError):
        Activation.relu.arity = 3


def test_bare_classvar_and_entry_sentinels() -> None:
    class Status(Enum, type_key=_key("Status")):
        ready: ClassVar[Status]
        failed: ClassVar[Status]

    assert list(Status) == [Status.ready, Status.failed]
    assert Status.ready._str_index == "ready"
    assert Status.failed._int_index == 1

    first, second = auto(), entry(code=2)
    assert isinstance(first, _EnumEntry)
    assert isinstance(second, _EnumEntry)
    assert first is not second
    assert repr(second) == "entry(code=2)"


def test_int_enum_uses_full_signed_int64_indices() -> None:
    class Limit(IntEnum, type_key=_key("Limit")):
        MIN = _INT64_MIN
        MAX = _INT64_MAX

    assert Limit.MIN.value == Limit.MIN._int_index == _INT64_MIN
    assert Limit.MAX.value == Limit.MAX._int_index == _INT64_MAX
    assert Limit.MIN.name == Limit.MIN._str_index == "MIN"
    assert Limit(_INT64_MIN).same_as(Limit.MIN)
    assert Limit(_INT64_MAX).same_as(Limit.MAX)
    assert Limit("MIN").same_as(Limit.MIN)
    assert Limit.get("MAX").same_as(Limit.MAX)
    assert list(Limit) == [Limit.MIN, Limit.MAX]

    for value in (_INT64_MIN - 1, _INT64_MAX + 1):
        with pytest.raises(OverflowError):
            Limit(value)
        with pytest.raises(OverflowError):
            Limit.get(value)


def test_str_enum_name_and_value_are_string_index_aliases() -> None:
    class MemSpace(StrEnum, type_key=_key("MemSpace")):
        SMEM = "smem"
        GMEM = "gmem"

    assert MemSpace.SMEM._str_index == "smem"
    assert MemSpace.SMEM.name == "smem"
    assert MemSpace.SMEM.value == "smem"
    assert MemSpace("smem").same_as(MemSpace.SMEM)
    assert MemSpace.get(0).same_as(MemSpace.SMEM)
    assert repr(MemSpace.SMEM) == "MemSpace.smem"
    assert str(MemSpace.SMEM) == "smem"
    assert MemSpace.SMEM == "smem"

    with pytest.raises(ValueError):
        MemSpace("SMEM")
    with pytest.raises(KeyError):
        MemSpace.get("SMEM")


def test_multiple_class_aliases_share_one_canonical_variant() -> None:
    class Opcode(StrEnum, type_key=_key("OpcodeAlias")):
        PLUS = "+"
        ADD = "+"

    assert Opcode.PLUS.same_as(Opcode.ADD)
    assert list(Opcode) == [Opcode.PLUS]
    assert Opcode.PLUS.name == Opcode.PLUS.value == "+"


def test_payload_behavior_and_custom_repr() -> None:
    class Priority(IntEnum, type_key=_key("Priority")):
        LOW = 1
        HIGH = 10

        def __repr__(self) -> str:
            return f"Priority({self.value})"

    assert Priority.LOW == 1
    assert Priority.LOW != 10
    assert hash(Priority.HIGH) == hash(10)
    assert str(Priority.HIGH) == "10"
    assert repr(Priority.HIGH) == "Priority(10)"


def test_enum_has_exactly_one_registry_typeattr() -> None:
    class Flag(Enum, type_key=_key("State")):
        OFF = auto()
        ON = auto()

    state = _state(Flag)
    assert isinstance(state, EnumState)
    assert {field.name for field in tvm_ffi.dataclasses.fields(state)} == {
        "entries",
        "indexes",
        "attrs",
    }
    assert list(state.entries) == [Flag.OFF, Flag.ON]
    assert state.indexes[0].same_as(Flag.OFF)
    assert state.indexes["ON"].same_as(Flag.ON)

    info = Flag.__tvm_ffi_type_info__  # ty: ignore[unresolved-attribute]
    for removed in (
        "__ffi_enum_entries__",
        "__ffi_enum_index_entries__",
        "__ffi_enum_attrs__",
        "__ffi_enum_value_entries__",
        "__ffi_enum_data_from_json_factory__",
    ):
        assert core._lookup_type_attr(info.type_index, removed) is None


def test_extensible_attrs_are_singleton_keyed() -> None:
    class Limit(IntEnum, type_key=_key("Attrs")):
        MIN = _INT64_MIN
        MAX = _INT64_MAX

    cost = Limit.def_attr("cost", default=-1)
    assert isinstance(cost, EnumAttrMap)
    assert cost[Limit.MIN] == -1
    assert Limit.MIN not in cost

    cost[Limit.MIN] = None
    cost[Limit.MAX] = 7
    assert Limit.MIN in cost
    assert cost[Limit.MIN] is None
    assert cost.get(Limit.MAX) == 7

    column = _state(Limit).attrs["cost"]
    assert column[Limit.MIN] is None
    assert column[Limit.MAX] == 7


def test_extensible_attrs_validate_variant_type() -> None:
    class Left(Enum, type_key=_key("Left")):
        ONE = auto()

    class Right(Enum, type_key=_key("Right")):
        ONE = auto()

    attr = Left.def_attr("x")
    with pytest.raises(TypeError):
        attr[Right.ONE] = 1
    with pytest.raises(KeyError):
        _ = attr[Left.ONE]
    assert attr.get(Left.ONE, 3) == 3


def test_json_roundtrips_use_canonical_indices() -> None:
    class Status(Enum, type_key=_key("JsonStatus")):
        READY = auto()

    class Priority(IntEnum, type_key=_key("JsonPriority")):
        LOW = -9

    class Opcode(StrEnum, type_key=_key("JsonOpcode")):
        ADD = "+"

    _assert_json_roundtrip(Status.READY, "READY")
    _assert_json_roundtrip(Priority.LOW, -9)
    _assert_json_roundtrip(Opcode.ADD, "+")


def test_enum_fields_json_roundtrip_to_same_singletons() -> None:
    class Priority(IntEnum, type_key=_key("HolderPriority")):
        MIN = _INT64_MIN
        MAX = _INT64_MAX

    class Opcode(StrEnum, type_key=_key("HolderOpcode")):
        ADD = "+"
        MUL = "*"

    @py_class(_key("Holder"))
    class Holder(Object):
        priority: Priority
        opcode: Opcode

    holder = Holder(priority=_INT64_MIN, opcode="*")  # ty: ignore[invalid-argument-type]
    assert holder.priority.same_as(Priority.MIN)
    assert holder.opcode.same_as(Opcode.MUL)

    restored = from_json_graph_str(to_json_graph_str(holder))
    assert restored.priority.same_as(Priority.MIN)
    assert restored.opcode.same_as(Opcode.MUL)
    assert tvm_ffi.structural_equal(holder, restored)

    for value in (_INT64_MIN - 1, _INT64_MAX + 1):
        with pytest.raises(TypeError, match="int64 range"):
            Holder(priority=value, opcode="+")  # ty: ignore[invalid-argument-type]


def test_structural_semantics_are_singleton_based() -> None:
    class Status(Enum, type_key=_key("Structural")):
        READY = auto()

    duplicate = copy.copy(Status.READY)
    assert not duplicate.same_as(Status.READY)
    assert tvm_ffi.structural_equal(Status.READY, Status.READY)
    assert not tvm_ffi.structural_equal(Status.READY, duplicate)


def test_payload_value_field_is_reserved() -> None:
    with pytest.raises(TypeError, match="reserves `value`"):

        class BadInt(IntEnum, type_key=_key("BadInt")):
            value: int
            ONE = 1

    with pytest.raises(TypeError, match="reserves `value`"):

        class BadStr(StrEnum, type_key=_key("BadStr")):
            value: str
            ONE = "one"


def test_payload_literals_are_type_checked() -> None:
    with pytest.raises(TypeError):

        class BadInt(IntEnum, type_key=_key("BadIntLiteral")):
            BAD = "bad"

    with pytest.raises(TypeError):

        class BadStr(StrEnum, type_key=_key("BadStrLiteral")):
            BAD = 1


def test_cxx_registered_plain_enum_shares_state_and_attrs() -> None:
    class Variant(Enum, type_key="testing.TestEnumVariant"):
        Alpha: ClassVar[Variant]
        Beta: ClassVar[Variant]

    cxx_get = tvm_ffi.get_global_func("testing.enum_variant_get")
    assert cxx_get("Alpha").same_as(Variant.Alpha)
    assert cxx_get("Beta").same_as(Variant.Beta)
    assert Variant.get(0).same_as(Variant.Alpha)
    assert Variant.get("Beta").same_as(Variant.Beta)
    assert Variant.attr_dict["code"][Variant.Alpha] == 10
    assert Variant.attr_dict["code"][Variant.Beta] == 20
    assert isinstance(_state(Variant), EnumState)

    with pytest.raises(ValueError):
        cxx_get("missing")


def test_cxx_backed_binder_must_use_canonical_string_index() -> None:
    with pytest.raises(RuntimeError, match="string index"):

        class Typo(Enum, type_key="testing.TestEnumVariant"):
            Missing: ClassVar[Typo]


def test_cxx_backed_payload_enums_and_converters() -> None:
    assert _TestCxxIntEnum.high.value == 20
    assert _TestCxxIntEnum.high.name == "high"
    assert _TestCxxStrEnum.mul.value == _TestCxxStrEnum.mul.name == "*"
    assert _TestCxxIntEnum(20).same_as(_TestCxxIntEnum.high)
    assert _TestCxxStrEnum("*").same_as(_TestCxxStrEnum.mul)

    holder = _TestCxxEnumHolder(priority=20, opcode="*")  # ty: ignore[invalid-argument-type]
    assert holder.priority.same_as(_TestCxxIntEnum.high)
    assert holder.opcode.same_as(_TestCxxStrEnum.mul)
    restored = from_json_graph_str(to_json_graph_str(holder))
    assert restored.priority.same_as(_TestCxxIntEnum.high)
    assert restored.opcode.same_as(_TestCxxStrEnum.mul)


def test_plain_default_repr_uses_string_index() -> None:
    type_key = _key("Repr")

    class Status(Enum, type_key=type_key):
        READY = auto()

    assert repr(Status.READY) == f"{type_key}.READY"
