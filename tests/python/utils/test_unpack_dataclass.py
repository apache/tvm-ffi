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

import dataclasses
import sys
from typing import Any

import pytest
from tvm_ffi.utils.unpack_dataclass import (
    _extract_dataclass_to_tuple_schema,
    _validate_dataclass_to_tuple_schema,
    unpack_dataclass_to_tuple,
)


# Module-level dataclass definitions for schema extraction tests.
# Must be at module level so typing.get_type_hints() can resolve
# cross-references with PEP 563 (from __future__ import annotations).
@dataclasses.dataclass
class _ExtractConfig:
    x: int
    y: int


@dataclasses.dataclass
class _ExtractNested:
    value: int
    cfg: _ExtractConfig


@dataclasses.dataclass
class _ExtractWithAny:
    data: Any
    scale: int


@dataclasses.dataclass
class _ExtractWithList:
    items: list[_ExtractConfig]
    scale: int


@dataclasses.dataclass
class _ExtractWithLeafList:
    values: list[int]
    name: str


@dataclasses.dataclass
class _ExtractWithDict:
    mapping: dict[str, _ExtractConfig]
    count: int


@dataclasses.dataclass
class _ExtractWithLeafDict:
    mapping: dict[str, int]
    count: int


@dataclasses.dataclass
class _ExtractWithTuple:
    pair: tuple[_ExtractConfig, int]
    flag: bool


@dataclasses.dataclass
class _ExtractWithOptional:
    value: int | None
    name: str


@dataclasses.dataclass
class _ExtractWithLeafListInt:
    items: list[int]
    scale: int


def test_unpack_dataclass_to_tuple() -> None:
    """Test unpack_dataclass_to_tuple JIT-compiled unpacking."""

    @dataclasses.dataclass
    class Config:
        x: int
        y: int

    @dataclasses.dataclass
    class Nested:
        value: int
        cfg: Config

    @dataclasses.dataclass
    class Deep:
        nested: Nested
        flag: bool

    # Flat dataclass
    assert unpack_dataclass_to_tuple(Config(x=1, y=2)) == (1, 2)

    # Nested dataclass (auto-recurses based on type annotation)
    assert unpack_dataclass_to_tuple(Nested(value=5, cfg=Config(x=10, y=20))) == (5, (10, 20))

    # Deep nesting
    assert unpack_dataclass_to_tuple(
        Deep(nested=Nested(value=5, cfg=Config(x=10, y=20)), flag=True)
    ) == ((5, (10, 20)), True)

    # Leaf passthrough
    assert unpack_dataclass_to_tuple(42) == 42
    assert unpack_dataclass_to_tuple("hello") == "hello"
    assert unpack_dataclass_to_tuple(None) is None

    # List recursion: list of dataclasses -> tuple of tuples
    assert unpack_dataclass_to_tuple([Config(x=1, y=2), Config(x=3, y=4)]) == [(1, 2), (3, 4)]

    # Tuple recursion
    assert unpack_dataclass_to_tuple((Config(x=1, y=2), 5)) == ((1, 2), 5)

    # Dict recursion (recurses values)
    assert unpack_dataclass_to_tuple({"a": Config(x=1, y=2), "b": 3}) == {"a": (1, 2), "b": 3}

    # Leaf values are NOT copied (no deep copy)
    class Holder:
        pass

    @dataclasses.dataclass
    class WithObj:
        obj: Any
        val: int

    h = Holder()
    result = unpack_dataclass_to_tuple(WithObj(obj=h, val=1))
    assert result == (h, 1)
    assert result[0] is h  # same object reference, no copy

    # Dynamic dispatch: Any-typed field receives a dataclass at runtime
    @dataclasses.dataclass
    class WithAnyField:
        data: Any
        scale: int

    # data is Any -> schema marks it as "unpack" -> __dispatch called at runtime
    # When data is a dataclass, it should be recursively unpacked
    result = unpack_dataclass_to_tuple(WithAnyField(data=Config(x=1, y=2), scale=3))
    assert result == ((1, 2), 3)

    # When data is a plain value, passthrough
    result = unpack_dataclass_to_tuple(WithAnyField(data=42, scale=3))
    assert result == (42, 3)

    # When data is a list of dataclasses, recurse each element
    result = unpack_dataclass_to_tuple(
        WithAnyField(data=[Config(x=1, y=2), Config(x=3, y=4)], scale=5)
    )
    assert result == ([(1, 2), (3, 4)], 5)

    # When data is a nested dataclass
    result = unpack_dataclass_to_tuple(
        WithAnyField(data=Nested(value=10, cfg=Config(x=1, y=2)), scale=5)
    )
    assert result == ((10, (1, 2)), 5)

    # When data is a dict with dataclass values
    result = unpack_dataclass_to_tuple(
        WithAnyField(data={"a": Config(x=1, y=2), "b": Config(x=3, y=4)}, scale=5)
    )
    assert result == ({"a": (1, 2), "b": (3, 4)}, 5)

    # Self-referential dataclass (linked list): should not infinite recurse
    @dataclasses.dataclass
    class Node:
        value: int
        next: Node | None

    # Build a short linked list
    node = Node(value=1, next=Node(value=2, next=None))
    result = unpack_dataclass_to_tuple(node)
    # The 'next' field is typed as Node|None, which on 3.10+ resolves to
    # a UnionType. The self-reference is caught by memo -> UNPACK -> dynamic dispatch.
    # Dynamic dispatch recursively unpacks the nested Node.
    assert result == (1, (2, None))


def test_validate_dataclass_to_tuple_schema() -> None:
    """Test internal schema validation."""
    # Valid schemas
    _validate_dataclass_to_tuple_schema({"x": None, "y": None})
    _validate_dataclass_to_tuple_schema({"cfg": {"x": None, "y": None}, "scale": None})

    # Invalid: not a dict
    with pytest.raises(TypeError, match="must be a dict"):
        _validate_dataclass_to_tuple_schema([1, 2])  # type: ignore[arg-type]

    # Invalid: non-string key
    with pytest.raises(TypeError, match="must be a string"):
        _validate_dataclass_to_tuple_schema({123: None})

    # Invalid: not a valid identifier
    with pytest.raises(ValueError, match="not a valid Python identifier"):
        _validate_dataclass_to_tuple_schema({"not-valid": None})

    # Invalid: Python keyword
    with pytest.raises(ValueError, match="is a Python keyword"):
        _validate_dataclass_to_tuple_schema({"class": None})

    # Invalid: nested schema with bad key
    with pytest.raises(ValueError, match="not a valid Python identifier"):
        _validate_dataclass_to_tuple_schema({"cfg": {"x": None, "y!": None}})


@pytest.mark.xfail(
    sys.version_info < (3, 10),
    reason="list[X]/dict[X,Y]/int|None not evaluable by get_type_hints on Python < 3.10",
)
def test_extract_dataclass_to_tuple_schema() -> None:
    """Test schema extraction from dataclass types."""
    # Flat: all known leaf types -> None
    schema = _extract_dataclass_to_tuple_schema(_ExtractConfig)
    assert schema == {"x": None, "y": None}

    # Nested: known dataclass field -> nested schema
    schema = _extract_dataclass_to_tuple_schema(_ExtractNested)
    assert schema == {"value": None, "cfg": {"x": None, "y": None}}

    # Any field -> "unpack" (dynamic dispatch)
    schema = _extract_dataclass_to_tuple_schema(_ExtractWithAny)
    assert schema == {"data": "unpack", "scale": None}

    # list[Config] -> "unpack" (container with dataclass element)
    schema = _extract_dataclass_to_tuple_schema(_ExtractWithList)
    assert schema == {"items": "unpack", "scale": None}

    # list[int] -> "unpack" (list must be converted to tuple per contract)
    schema = _extract_dataclass_to_tuple_schema(_ExtractWithLeafList)
    assert schema == {"values": "unpack", "name": None}

    # list[int] standalone field also gets UNPACK
    schema = _extract_dataclass_to_tuple_schema(_ExtractWithLeafListInt)
    assert schema == {"items": "unpack", "scale": None}

    # dict[str, Config] -> "unpack" (container with dataclass value type)
    schema = _extract_dataclass_to_tuple_schema(_ExtractWithDict)
    assert schema == {"mapping": "unpack", "count": None}

    # dict[str, int] -> None (container with only known leaf types)
    schema = _extract_dataclass_to_tuple_schema(_ExtractWithLeafDict)
    assert schema == {"mapping": None, "count": None}

    # tuple[Config, int] -> "unpack" (tuple containing a dataclass)
    schema = _extract_dataclass_to_tuple_schema(_ExtractWithTuple)
    assert schema == {"pair": "unpack", "flag": None}

    # Optional[int] (int | None) -> None (Union of known leaves)
    schema = _extract_dataclass_to_tuple_schema(_ExtractWithOptional)
    assert schema == {"value": None, "name": None}

    # Non-dataclass raises
    with pytest.raises(TypeError, match="Expected a dataclass class"):
        _extract_dataclass_to_tuple_schema(int)

    # Locally-defined classes with built-in type annotations resolve fine
    @dataclasses.dataclass
    class LocalConfig:
        x: int
        y: int

    schema = _extract_dataclass_to_tuple_schema(LocalConfig)
    assert schema == {"x": None, "y": None}

    # Locally-defined classes referencing other local classes can't be resolved
    # by get_type_hints — all fields fall back to UNPACK
    @dataclasses.dataclass
    class LocalNested:
        val: int
        cfg: LocalConfig

    schema = _extract_dataclass_to_tuple_schema(LocalNested)
    # get_type_hints fails for the class -> all fields become "unpack"
    assert schema == {"val": "unpack", "cfg": "unpack"}
