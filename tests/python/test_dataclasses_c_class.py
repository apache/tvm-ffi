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
import inspect
from dataclasses import MISSING

import pytest
from tvm_ffi.dataclasses import KW_ONLY, field
from tvm_ffi.dataclasses.field import _KW_ONLY_TYPE, Field
from tvm_ffi.testing import (
    _TestCxxClassBase,
    _TestCxxClassDerived,
    _TestCxxClassDerivedDerived,
    _TestCxxInitSubset,
    _TestCxxKwOnly,
)


def test_cxx_class_base() -> None:
    obj = _TestCxxClassBase(v_i64=123, v_i32=456)
    assert obj.v_i64 == 123 + 1
    assert obj.v_i32 == 456 + 2


def test_cxx_class_derived() -> None:
    obj = _TestCxxClassDerived(v_i64=123, v_i32=456, v_f64=4.00, v_f32=8.00)
    assert obj.v_i64 == 123
    assert obj.v_i32 == 456
    assert obj.v_f64 == 4.00
    assert obj.v_f32 == 8.00


def test_cxx_class_derived_default() -> None:
    obj = _TestCxxClassDerived(v_i64=123, v_i32=456, v_f64=4.00)
    assert obj.v_i64 == 123
    assert obj.v_i32 == 456
    assert obj.v_f64 == 4.00
    assert isinstance(obj.v_f32, float) and obj.v_f32 == 8.00  # default value


def test_cxx_class_derived_derived() -> None:
    obj = _TestCxxClassDerivedDerived(
        v_i64=123,
        v_i32=456,
        v_f64=4.00,
        v_f32=8.00,
        v_str="hello",
        v_bool=True,
    )
    assert obj.v_i64 == 123
    assert obj.v_i32 == 456
    assert obj.v_f64 == 4.00
    assert obj.v_f32 == 8.00
    assert obj.v_str == "hello"
    assert obj.v_bool is True


def test_cxx_class_derived_derived_default() -> None:
    obj = _TestCxxClassDerivedDerived(123, 456, 4, True)  # ty: ignore[missing-argument]
    assert obj.v_i64 == 123
    assert obj.v_i32 == 456
    assert isinstance(obj.v_f64, float) and obj.v_f64 == 4
    assert isinstance(obj.v_f32, float) and obj.v_f32 == 8
    assert obj.v_str == "default"
    assert isinstance(obj.v_bool, bool) and obj.v_bool is True


def test_cxx_class_init_subset_signature() -> None:
    sig = inspect.signature(_TestCxxInitSubset.__init__)
    params = tuple(sig.parameters)
    assert "required_field" in params
    assert "optional_field" not in params
    assert "note" not in params


def test_cxx_class_init_subset_defaults() -> None:
    obj = _TestCxxInitSubset(required_field=42)
    assert obj.required_field == 42
    assert obj.optional_field == -1
    assert obj.note == "py-default"


def test_cxx_class_init_subset_positional() -> None:
    obj = _TestCxxInitSubset(7)
    assert obj.required_field == 7
    assert obj.optional_field == -1
    obj.optional_field = 11
    assert obj.optional_field == 11


def test_kw_only_class_level_signature() -> None:
    sig = inspect.signature(_TestCxxKwOnly.__init__)
    params = sig.parameters
    assert params["x"].kind == inspect.Parameter.KEYWORD_ONLY
    assert params["y"].kind == inspect.Parameter.KEYWORD_ONLY
    assert params["z"].kind == inspect.Parameter.KEYWORD_ONLY
    assert params["w"].kind == inspect.Parameter.KEYWORD_ONLY


def test_kw_only_class_level_call() -> None:
    obj = _TestCxxKwOnly(x=1, y=2, z=3, w=4)
    assert obj.x == 1
    assert obj.y == 2
    assert obj.z == 3
    assert obj.w == 4


def test_kw_only_class_level_with_default() -> None:
    obj = _TestCxxKwOnly(x=1, y=2, z=3)
    assert obj.w == 100


def test_kw_only_class_level_rejects_positional() -> None:
    with pytest.raises(TypeError, match="positional"):
        _TestCxxKwOnly(1, 2, 3, 4)  # ty: ignore[missing-argument, too-many-positional-arguments]


def test_field_kw_only_parameter() -> None:
    f1: Field = field(kw_only=True)
    assert isinstance(f1, Field)
    assert f1.kw_only is True

    f2: Field = field(kw_only=False)
    assert f2.kw_only is False

    f3: Field = field()
    assert f3.kw_only is MISSING


def test_field_kw_only_with_default() -> None:
    f = field(default=42, kw_only=True)
    assert isinstance(f, Field)
    assert f.kw_only is True
    assert f.default_factory() == 42


def test_kw_only_sentinel_exists() -> None:
    assert isinstance(KW_ONLY, _KW_ONLY_TYPE)
