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
"""Regression tests for init=False with custom __init__ (Bug #2)."""

from __future__ import annotations

from tvm_ffi import Object
from tvm_ffi.dataclasses import c_class

# ---- Define test classes with init=False and custom __init__ ----


@c_class("testing.TestB2Var", init=False)
class _TestB2Var(Object):
    """init=False type with a String field and custom __init__."""

    __test__ = False

    name: str

    def __init__(self, name: str) -> None:
        self.name = name


@c_class("testing.TestB2Pair", init=False)
class _TestB2Pair(Object):
    """init=False type with two int fields and custom __init__."""

    __test__ = False

    first: int
    second: int

    def __init__(self, a: int, b: int) -> None:
        self.first = a
        self.second = b


@c_class("testing.TestB2Base")
class _TestB2Base(Object):
    """Auto-init base type for inheritance test."""

    __test__ = False

    x: int


@c_class("testing.TestB2Child", init=False)
class _TestB2Child(_TestB2Base):
    """init=False child that calls super().__init__()."""

    __test__ = False

    y: int

    def __init__(self, x: int, y: int) -> None:
        super().__init__(x=x)
        self.y = y


@c_class("testing.TestB2EmptyInit", init=False)
class _TestB2EmptyInit(Object):
    """init=False type with custom __init__ that sets no fields."""

    __test__ = False

    val: int

    def __init__(self) -> None:
        pass  # leave val at default (0)


# ---- Regression tests ----


class TestBug2InitFalseSegfault:
    """Regression test for Bug #2: init=False with custom __init__ that sets
    annotated fields must not segfault.
    """

    def test_init_false_set_field_no_segfault(self) -> None:
        """Setting annotated field in custom __init__ with init=False should work."""
        v = _TestB2Var("x")
        assert v.name == "x"

    def test_init_false_multiple_fields(self) -> None:
        """Multiple field assignments in custom init with init=False."""
        p = _TestB2Pair(10, 20)
        assert p.first == 10
        assert p.second == 20

    def test_init_false_with_super(self) -> None:
        """init=False subclass calling super().__init__() should work."""
        c = _TestB2Child(1, 2)
        assert c.x == 1
        assert c.y == 2

    def test_init_false_no_fields(self) -> None:
        """init=False with custom __init__ that doesn't set fields."""
        obj = _TestB2EmptyInit()
        assert obj.val == 0  # calloc'd to zero
