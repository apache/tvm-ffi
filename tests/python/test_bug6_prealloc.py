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
"""Regression tests for Bug #6: c_class with __slots__ = ('__dict__',)
and custom __init__ must not segfault when stored in another c_class field.
"""

from __future__ import annotations

from tvm_ffi.core import Object
from tvm_ffi.dataclasses import c_class
from tvm_ffi.testing import _TestCxxClassBase

# -- Test helper classes (registered against C++ testing.TestBug6* types) ------


@c_class("testing.TestBug6Node")
class _Bug6Node(Object):
    """Base node class (no fields)."""

    __slots__ = ("__dict__",)


@c_class("testing.TestBug6Container")
class _Bug6Container(Object):
    """Container with an Any-typed field to hold arbitrary objects."""

    val: object  # maps to C++ Any


@c_class("testing.TestBug6Hybrid")
class _Bug6Hybrid(Object):
    """Has both a reflected int field and a __dict__ slot."""

    __slots__ = ("__dict__",)
    x: int

    def __init__(self, x: int, extra_name: str) -> None:
        self.x = x
        self._extra = extra_name  # goes to __dict__


@c_class("testing.TestBug6Prealloc")
class _Bug6Prealloc(Object):
    """Has a reflected string field and a custom init (init=True default)."""

    name: str

    def __init__(self, name: str) -> None:
        self.name = name  # must not segfault


# -- Tests --------------------------------------------------------------------


class TestBug6SlotsDictSegfault:
    """Regression test for Bug #6: c_class with __slots__ = ('__dict__',)
    and custom __init__ must not segfault when stored in another c_class field.
    """

    def test_dict_slots_stored_in_field(self) -> None:
        """Object with __dict__ slot and custom init can be stored in a field.

        The key assertion is that storing and retrieving the object does
        not segfault (the pre-allocated chandle is valid).  The Python
        ``__dict__`` is per-wrapper and is NOT preserved across FFI
        round-trips; only the underlying C++ handle identity is checked.
        """

        @c_class("testing.TestBug6Node")
        class DataType(_Bug6Node):
            def __init__(self, tag: str) -> None:
                self._tag = tag

        dt = DataType("float32")
        assert dt._tag == "float32"
        c = _Bug6Container(val=dt)
        # Must not segfault; the retrieved object should share the handle.
        retrieved = c.val
        assert isinstance(retrieved, Object)
        assert retrieved.same_as(dt)

    def test_custom_init_with_fields_and_dict(self) -> None:
        """c_class with both annotated fields and __dict__ via custom init."""
        h = _Bug6Hybrid(42, "hello")
        assert h.x == 42
        assert h._extra == "hello"

    def test_init_true_custom_init_preallocates(self) -> None:
        """With init=True (default), user __init__ should still get pre-allocation."""
        obj = _Bug6Prealloc("test")
        assert obj.name == "test"

    def test_custom_init_still_supports_ffi_init(self) -> None:
        """Existing classes that call __ffi_init__ in custom __init__ still work."""
        obj = _TestCxxClassBase(v_i64=10, v_i32=20)
        assert obj.v_i64 == 11  # +1 from custom __init__
        assert obj.v_i32 == 22  # +2 from custom __init__
