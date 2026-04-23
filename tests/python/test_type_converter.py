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
"""Tests for TypeSchema type conversion to CAny."""

from __future__ import annotations

import collections.abc
import ctypes
import itertools
import os
import sys
import typing
from numbers import Integral
from typing import Callable, Iterator, Optional, Union

import pytest
import tvm_ffi
from tvm_ffi.core import (
    CAny,
    ObjectConvertible,
    TypeSchema,
    _lookup_type_attr,
    _object_type_key_to_index,
    _to_py_class_value,
)
from tvm_ffi.dataclasses import IntEnum, StrEnum, entry

# Python 3.9+ supports list[int], dict[str, int], tuple[int, ...] at runtime.
# On 3.8, these raise TypeError("'type' object is not subscriptable").
_PY39 = sys.version_info >= (3, 9)
from tvm_ffi.testing import (
    TestIntPair,
    TestObjectBase,
    TestObjectDerived,
    _TestCxxClassBase,
    _TestCxxClassDerived,
    _TestCxxClassDerivedDerived,
)
from tvm_ffi.testing.testing import requires_py39, requires_py310

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_TYPE_KEY_COUNTER = itertools.count()


def S(origin: str, *args: TypeSchema) -> TypeSchema:
    """Shorthand constructor for TypeSchema (string-based)."""
    return TypeSchema(origin, tuple(args))


def _unique_type_key(base: str) -> str:
    return f"testing.type_converter.{base}_{next(_TYPE_KEY_COUNTER)}"


def _make_int_enum_type() -> typing.Any:
    class Colors(IntEnum, type_key=_unique_type_key("IntEnum")):
        red = entry(value=10)
        blue = entry(value=20)

    return Colors


def _make_str_enum_type() -> typing.Any:
    class Tokens(StrEnum, type_key=_unique_type_key("StrEnum")):
        add = entry(value="+")
        mul = entry(value="*")

    return Tokens


# Annotation-based constructor — the main subject under test.
A = TypeSchema.from_annotation


# ---------------------------------------------------------------------------
# Category 1: POD type exact match (check_value)
# ---------------------------------------------------------------------------
class TestPODExactMatch:
    def test_int(self) -> None:
        """Test int."""
        A(int).check_value(42)

    def test_float(self) -> None:
        """Test float."""
        A(float).check_value(3.14)

    def test_bool_true(self) -> None:
        """Test bool true."""
        A(bool).check_value(True)

    def test_bool_false(self) -> None:
        """Test bool false."""
        A(bool).check_value(False)

    def test_str(self) -> None:
        """Test str."""
        A(str).check_value("hello")

    def test_bytes(self) -> None:
        """Test bytes."""
        A(bytes).check_value(b"data")

    def test_none(self) -> None:
        """Test none."""
        A(type(None)).check_value(None)


# ---------------------------------------------------------------------------
# Category 2: Implicit conversions (mirrors TryCastFromAnyView)
# ---------------------------------------------------------------------------
class TestImplicitConversions:
    def test_bool_to_int(self) -> None:
        """Bool -> int is OK (C++: int accepts bool)."""
        A(int).check_value(True)

    def test_int_to_float(self) -> None:
        """Int -> float is OK (C++: float accepts int)."""
        A(float).check_value(42)

    def test_bool_to_float(self) -> None:
        """Bool -> float is OK (C++: float accepts bool)."""
        A(float).check_value(True)

    def test_int_to_bool(self) -> None:
        """Int -> bool is OK (C++: bool accepts int)."""
        A(bool).check_value(1)


# ---------------------------------------------------------------------------
# Category 3: Rejection cases
# ---------------------------------------------------------------------------
class TestRejections:
    def test_str_not_int(self) -> None:
        """Test str not int."""
        with pytest.raises(TypeError, match="expected int"):
            A(int).check_value("hello")

    def test_float_not_int(self) -> None:
        """Test float not int."""
        with pytest.raises(TypeError):
            A(int).check_value(3.14)

    def test_none_not_int(self) -> None:
        """Test none not int."""
        with pytest.raises(TypeError):
            A(int).check_value(None)

    def test_int_not_str(self) -> None:
        """Test int not str."""
        with pytest.raises(TypeError):
            A(str).check_value(42)

    def test_str_not_bool(self) -> None:
        """Test str not bool."""
        with pytest.raises(TypeError):
            A(bool).check_value("hello")

    def test_none_not_str(self) -> None:
        """Test none not str."""
        with pytest.raises(TypeError):
            A(str).check_value(None)

    def test_int_not_bytes(self) -> None:
        """Test int not bytes."""
        with pytest.raises(TypeError):
            A(bytes).check_value(42)

    def test_int_not_none(self) -> None:
        """Test int not none."""
        with pytest.raises(TypeError):
            A(type(None)).check_value(42)


# ---------------------------------------------------------------------------
# Category 4: Special types
# ---------------------------------------------------------------------------
class TestSpecialTypes:
    def test_device_pass(self) -> None:
        """Test device pass."""
        dev = tvm_ffi.Device("cpu", 0)
        A(tvm_ffi.Device).check_value(dev)

    def test_device_fail(self) -> None:
        """Test device fail."""
        with pytest.raises(TypeError):
            A(tvm_ffi.Device).check_value(42)

    def test_dtype_pass(self) -> None:
        """Test dtype pass."""
        dt = tvm_ffi.core.DataType("float32")
        A(tvm_ffi.core.DataType).check_value(dt)

    def test_dtype_str_pass(self) -> None:
        """Str accepted as dtype (will be parsed)."""
        A(tvm_ffi.core.DataType).check_value("float32")

    def test_dtype_fail(self) -> None:
        """Test dtype fail."""
        with pytest.raises(TypeError):
            A(tvm_ffi.core.DataType).check_value(42)

    def test_opaque_ptr_pass(self) -> None:
        """Test opaque ptr pass."""
        A(ctypes.c_void_p).check_value(ctypes.c_void_p(0))

    def test_opaque_ptr_none_pass(self) -> None:
        """Test opaque ptr none pass."""
        A(ctypes.c_void_p).check_value(None)

    def test_opaque_ptr_fail(self) -> None:
        """Test opaque ptr fail."""
        with pytest.raises(TypeError):
            A(ctypes.c_void_p).check_value(42)

    def test_callable_pass_function(self) -> None:
        """Test callable pass function."""
        A(Callable).check_value(lambda x: x)

    def test_callable_pass_builtin(self) -> None:
        """Test callable pass builtin."""
        A(Callable).check_value(len)

    def test_callable_fail(self) -> None:
        """Test callable fail."""
        with pytest.raises(TypeError):
            A(Callable).check_value(42)

    def test_collections_abc_callable_pass_function(self) -> None:
        """collections.abc.Callable accepts Python functions."""
        A(collections.abc.Callable).check_value(lambda x: x)

    def test_collections_abc_callable_pass_builtin(self) -> None:
        """collections.abc.Callable accepts builtins."""
        A(collections.abc.Callable).check_value(len)

    def test_collections_abc_callable_fail(self) -> None:
        """collections.abc.Callable rejects non-callables."""
        with pytest.raises(TypeError, match="expected Callable"):
            A(collections.abc.Callable).check_value(42)

    def test_callable_cobject_wraps_to_function(self) -> None:
        """Callable CObjects are wrapped instead of asserting."""

        class CallableObj(TestObjectBase):
            def __call__(self, x: int) -> int:
                return x + 1

        obj = CallableObj(v_i64=1, v_f64=2.0, v_str="s")
        with pytest.raises(TypeError, match=r"expected Callable, got .*TestObjectBase"):
            A(Callable).check_value(obj)


# ---------------------------------------------------------------------------
# Category 5: Object types
# ---------------------------------------------------------------------------
class TestObjectTypes:
    def test_object_pass(self) -> None:
        """Any CObject passes TypeSchema('Object')."""
        f = tvm_ffi.get_global_func("testing.echo")
        A(tvm_ffi.core.Object).check_value(f)

    def test_object_fail(self) -> None:
        """Test object fail."""
        with pytest.raises(TypeError):
            A(tvm_ffi.core.Object).check_value(42)

    def test_specific_object_pass(self) -> None:
        """A Function object should pass its own type schema."""
        f = tvm_ffi.get_global_func("testing.echo")
        A(Callable).check_value(f)

    def test_function_from_extern_c_exists(self) -> None:
        """ffi.FunctionFromExternC should be registered."""
        fn = tvm_ffi.get_global_func("ffi.FunctionFromExternC", allow_missing=True)
        assert fn is not None, "ffi.FunctionFromExternC not registered"


# ---------------------------------------------------------------------------
# Category 6: Payload enums
# ---------------------------------------------------------------------------
class TestPayloadEnums:
    def test_int_enum_convert_from_int(self) -> None:
        """IntEnum accepts its user-visible integer payload."""
        Colors = _make_int_enum_type()
        result = _to_py_class_value(A(Colors).convert(20))
        assert result.same_as(Colors.blue)

    def test_int_enum_passthrough_existing_object(self) -> None:
        """IntEnum keeps the object passthrough path for existing enum objects."""
        Colors = _make_int_enum_type()
        result = _to_py_class_value(A(Colors).convert(Colors.red))
        assert result.same_as(Colors.red)

    def test_int_enum_rejects_unknown_payload(self) -> None:
        """IntEnum still rejects unmatched integer payloads."""
        Colors = _make_int_enum_type()
        with pytest.raises(TypeError, match="expected"):
            A(Colors).check_value(99)

    def test_str_enum_convert_from_str(self) -> None:
        """StrEnum accepts its user-visible string payload."""
        Tokens = _make_str_enum_type()
        result = _to_py_class_value(A(Tokens).convert("*"))
        assert result.same_as(Tokens.mul)

    def test_str_enum_rejects_unknown_payload(self) -> None:
        """StrEnum still rejects unmatched string payloads."""
        Tokens = _make_str_enum_type()
        with pytest.raises(TypeError, match="expected"):
            A(Tokens).check_value("/")


# ---------------------------------------------------------------------------
# Category 7: Optional
# ---------------------------------------------------------------------------
class TestOptional:
    def test_none_passes(self) -> None:
        """Test none passes."""
        A(Optional[int]).check_value(None)

    def test_inner_type_passes(self) -> None:
        """Test inner type passes."""
        A(Optional[int]).check_value(42)

    def test_wrong_type_fails(self) -> None:
        """Test wrong type fails."""
        with pytest.raises(TypeError, match="expected int"):
            A(Optional[int]).check_value("hello")

    def test_nested_optional(self) -> None:
        """Test nested optional."""
        schema = A(Optional[Optional[int]])
        schema.check_value(None)
        schema.check_value(42)


# ---------------------------------------------------------------------------
# Category 8: Union / Variant
# ---------------------------------------------------------------------------
class TestUnion:
    def test_first_alt_passes(self) -> None:
        """Test first alt passes."""
        A(Union[int, str]).check_value(42)

    def test_second_alt_passes(self) -> None:
        """Test second alt passes."""
        A(Union[int, str]).check_value("hello")

    def test_no_alt_matches(self) -> None:
        """Test no alt matches."""
        with pytest.raises(TypeError, match="got float"):
            A(Union[int, str]).check_value(3.14)

    def test_bool_matches_int_alt(self) -> None:
        """Bool is accepted by the int alternative."""
        A(Union[int, str]).check_value(True)


# ---------------------------------------------------------------------------
# Category 9: Containers
# ---------------------------------------------------------------------------
class TestContainers:
    @requires_py39
    def test_array_list_pass(self) -> None:
        """Test array list pass."""
        A(tuple[int, ...]).check_value([1, 2, 3])

    @requires_py39
    def test_array_tuple_pass(self) -> None:
        """Test array tuple pass."""
        A(tuple[int, ...]).check_value((1, 2, 3))

    @requires_py39
    def test_array_wrong_element(self) -> None:
        """Test array wrong element."""
        with pytest.raises(TypeError, match=r"element \[1\].*expected int"):
            A(tuple[int, ...]).check_value([1, "x"])

    @requires_py39
    def test_array_empty_pass(self) -> None:
        """Test array empty pass."""
        A(tuple[int, ...]).check_value([])

    @requires_py39
    def test_array_any_pass(self) -> None:
        """Test array any pass."""
        A(tuple[typing.Any, ...]).check_value([1, "x", None])

    @requires_py39
    def test_array_wrong_container_type(self) -> None:
        """Test array wrong container type."""
        with pytest.raises(TypeError, match="expected Array"):
            A(tuple[int, ...]).check_value(42)

    @requires_py39
    def test_array_rejects_generator(self) -> None:
        """Generators are not accepted by Array schemas."""

        def gen() -> Iterator[int]:
            yield 1
            yield 2

        with pytest.raises(TypeError, match="expected Array"):
            A(tuple[int, ...]).check_value(gen())

    @requires_py39
    def test_array_rejects_string(self) -> None:
        """Strings are not accepted by Array schemas."""
        with pytest.raises(TypeError, match="expected Array"):
            A(tuple[int, ...]).check_value("hello")

    @requires_py39
    def test_list_pass(self) -> None:
        """Test list pass."""
        A(list[str]).check_value(["a", "b"])

    @requires_py39
    def test_map_pass(self) -> None:
        """Test map pass."""
        A(tvm_ffi.Map[str, int]).check_value({"a": 1, "b": 2})

    @requires_py39
    def test_map_wrong_key(self) -> None:
        """Test map wrong key."""
        with pytest.raises(TypeError, match="expected str"):
            A(tvm_ffi.Map[str, int]).check_value({1: 2})

    @requires_py39
    def test_map_wrong_value(self) -> None:
        """Test map wrong value."""
        with pytest.raises(TypeError, match="expected int"):
            A(tvm_ffi.Map[str, int]).check_value({"a": "b"})

    @requires_py39
    def test_map_empty_pass(self) -> None:
        """Test map empty pass."""
        A(tvm_ffi.Map[str, int]).check_value({})

    @requires_py39
    def test_dict_pass(self) -> None:
        """Test dict pass."""
        A(dict[str, int]).check_value({"a": 1})

    @requires_py39
    def test_map_wrong_container(self) -> None:
        """Test map wrong container."""
        with pytest.raises(TypeError, match="expected Map"):
            A(tvm_ffi.Map[str, int]).check_value([1, 2])

    @requires_py39
    def test_map_rejects_non_mapping_pairs(self) -> None:
        """Lists of pairs are not accepted by Map schemas."""
        with pytest.raises(TypeError, match="expected Map"):
            A(tvm_ffi.Map[str, int]).check_value([("a", 1)])


# ---------------------------------------------------------------------------
# Category 9: Nested types
# ---------------------------------------------------------------------------
class TestNestedTypes:
    @requires_py39
    def test_array_optional_int(self) -> None:
        """Test array optional int."""
        A(tuple[Optional[int], ...]).check_value([1, None, 2])

    @requires_py39
    def test_map_str_array_int(self) -> None:
        """Test map str array int."""
        A(tvm_ffi.Map[str, tuple[int, ...]]).check_value({"a": [1, 2]})

    @requires_py39
    def test_map_str_array_int_nested_fail(self) -> None:
        """Test map str array int nested fail."""
        with pytest.raises(TypeError, match="expected int"):
            A(tvm_ffi.Map[str, tuple[int, ...]]).check_value({"a": [1, "x"]})

    @requires_py39
    def test_union_with_containers(self) -> None:
        """Test union with containers."""
        schema = A(Union[int, tuple[str, ...]])
        schema.check_value(42)
        schema.check_value(["a", "b"])
        with pytest.raises(TypeError):
            schema.check_value(3.14)


# ---------------------------------------------------------------------------
# Category 10: Any
# ---------------------------------------------------------------------------
class TestAny:
    def test_int(self) -> None:
        """Test int."""
        A(typing.Any).check_value(42)

    def test_none(self) -> None:
        """Test none."""
        A(typing.Any).check_value(None)

    def test_str(self) -> None:
        """Test str."""
        A(typing.Any).check_value("hello")

    def test_list(self) -> None:
        """Test list."""
        A(typing.Any).check_value([1, 2, 3])

    def test_object(self) -> None:
        """Test object."""
        A(typing.Any).check_value(object())

    def test_object_convertible_convert(self) -> None:
        """Any eagerly unwraps ObjectConvertible via asobject()."""
        inner = TestIntPair(1, 2)

        class Convertible(ObjectConvertible):
            def asobject(self) -> tvm_ffi.core.Object:
                return inner

        result = _to_py_class_value(A(typing.Any).convert(Convertible()))
        assert result.same_as(inner)

    def test_object_convertible_error(self) -> None:
        """Any surfaces asobject() failures during eager normalization."""

        class BadConvertible(ObjectConvertible):
            def asobject(self) -> tvm_ffi.core.Object:
                raise RuntimeError("broken")

        with pytest.raises(TypeError, match=r"asobject\(\) failed"):
            A(typing.Any).check_value(BadConvertible())

    def test_object_protocol_convert(self) -> None:
        """Any eagerly unwraps __tvm_ffi_object__ before dispatch."""
        inner = TestIntPair(1, 2)

        class ObjProto:
            def __tvm_ffi_object__(self) -> object:
                return inner

        result = _to_py_class_value(A(typing.Any).convert(ObjProto()))
        assert result.same_as(inner)

    def test_object_protocol_error(self) -> None:
        """Any surfaces __tvm_ffi_object__ failures during eager normalization."""

        class BadProto:
            def __tvm_ffi_object__(self) -> object:
                raise RuntimeError("broken")

        with pytest.raises(TypeError, match=r"__tvm_ffi_object__\(\) failed"):
            A(typing.Any).check_value(BadProto())


# ---------------------------------------------------------------------------
# Category 11: Error message quality
# ---------------------------------------------------------------------------
class TestErrorMessages:
    def test_basic_type_mismatch(self) -> None:
        """Test basic type mismatch."""
        with pytest.raises(TypeError, match=r"expected int, got str"):
            A(int).check_value("hello")

    @requires_py39
    def test_nested_array_error(self) -> None:
        """Test nested array error."""
        with pytest.raises(TypeError, match=r"element \[2\].*expected int, got str"):
            A(tuple[int, ...]).check_value([1, 2, "x"])

    @requires_py39
    def test_nested_map_error(self) -> None:
        """Test nested map error."""
        with pytest.raises(TypeError, match=r"value for key 'b'.*expected int, got str"):
            A(tvm_ffi.Map[str, int]).check_value({"a": 1, "b": "x"})

    def test_union_error_lists_alternatives(self) -> None:
        """Test union error lists alternatives."""
        with pytest.raises(TypeError, match="got float") as exc_info:
            A(Union[int, str]).check_value(3.14)
        err = str(exc_info.value)
        assert "int" in err
        assert "str" in err

    def test_schema_in_error_message(self) -> None:
        """check_value includes the schema repr in the TypeError."""
        with pytest.raises(TypeError, match=r"type check failed for"):
            A(int).check_value("hello")

    def test_convert_error_message(self) -> None:
        """Convert includes the schema repr in the TypeError."""
        with pytest.raises(TypeError, match=r"type conversion failed for"):
            A(int).convert("hello")


# ---------------------------------------------------------------------------
# Category 12: from_type_index factory
# ---------------------------------------------------------------------------
class TestFromTypeIndex:
    def test_int(self) -> None:
        """Test int."""
        schema = TypeSchema.from_type_index(1)  # kTVMFFIInt
        assert schema.origin == "int"
        assert schema.origin_type_index == 1

    def test_float(self) -> None:
        """Test float."""
        schema = TypeSchema.from_type_index(3)  # kTVMFFIFloat
        assert schema.origin == "float"

    def test_bool(self) -> None:
        """Test bool."""
        schema = TypeSchema.from_type_index(2)  # kTVMFFIBool
        assert schema.origin == "bool"

    def test_array_with_args(self) -> None:
        """Test array with args."""
        schema = TypeSchema.from_type_index(71, (A(int),))  # kTVMFFIArray
        assert schema.origin == "Array"
        assert len(schema.args) == 1
        assert schema.args[0].origin == "int"

    def test_roundtrip_check(self) -> None:
        """from_type_index then check_value works correctly."""
        schema = TypeSchema.from_type_index(1)  # int
        schema.check_value(42)
        with pytest.raises(TypeError):
            schema.check_value("hello")

    def test_none(self) -> None:
        """Test none."""
        schema = TypeSchema.from_type_index(0)  # kTVMFFINone
        assert schema.origin == "None"
        schema.check_value(None)

    def test_any(self) -> None:
        """Test any."""
        schema = TypeSchema.from_type_index(-1)  # kTVMFFIAny
        assert schema.origin == "Any"
        schema.check_value("anything")

    def test_str(self) -> None:
        """Test str."""
        schema = TypeSchema.from_type_index(65)  # kTVMFFIStr
        assert schema.origin == "str"
        schema.check_value("hello")

    def test_map_with_args(self) -> None:
        """Test map with args."""
        schema = TypeSchema.from_type_index(72, (A(str), A(int)))  # kTVMFFIMap
        assert schema.origin == "Map"
        schema.check_value({"a": 1})


# ---------------------------------------------------------------------------
# Category 13: Edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_bytearray_passes_bytes(self) -> None:
        """Test bytearray passes bytes."""
        A(bytes).check_value(bytearray(b"data"))

    @requires_py39
    def test_tuple_passes_array(self) -> None:
        """Tuple is accepted as a sequence type for Array."""
        A(tuple[int, ...]).check_value((1, 2, 3))

    def test_empty_union_is_rejected(self) -> None:
        """Union requires at least 2 args."""
        with pytest.raises(ValueError, match="at least two"):
            TypeSchema("Union", ())

    def test_origin_type_index_auto_computed(self) -> None:
        """origin_type_index is automatically computed from origin string."""
        schema = A(int)
        assert schema.origin_type_index == 1  # kTVMFFIInt
        schema = A(float)
        assert schema.origin_type_index == 3  # kTVMFFIFloat
        schema = A(Optional[int])
        assert schema.origin_type_index == -2  # structural

    def test_check_value_succeeds_on_valid(self) -> None:
        """Test check value succeeds on valid input."""
        A(int).check_value(42)

    def test_check_value_raises_on_failure(self) -> None:
        """Test check value raises TypeError on failure."""
        with pytest.raises(TypeError, match="expected int"):
            A(int).check_value("hello")

    @requires_py39
    def test_tuple_type_schema(self) -> None:
        """Test tuple type schema."""
        schema = A(tuple[int, str])
        schema.check_value((1, "a"))
        with pytest.raises(TypeError):
            schema.check_value((1, 2))
        with pytest.raises(TypeError):
            schema.check_value((1,))

    def test_numpy_int_passes_int(self) -> None:
        """Numpy integer types should pass int check via Integral."""
        np = pytest.importorskip("numpy")
        A(int).check_value(np.int64(42))
        A(float).check_value(np.float64(3.14))


# ===========================================================================
# Type Converter Tests (convert)
# ===========================================================================


# ---------------------------------------------------------------------------
# Category 14: POD conversion results
# ---------------------------------------------------------------------------
class TestConvertPOD:
    def test_int_passthrough(self) -> None:
        """Int -> int returns the same value."""
        result = _to_py_class_value(A(int).convert(42))
        assert result == 42
        assert type(result) is int

    def test_bool_to_int(self) -> None:
        """Bool -> int actually converts to int."""
        result = _to_py_class_value(A(int).convert(True))
        assert result == 1
        assert type(result) is int

    def test_bool_false_to_int(self) -> None:
        """Test bool false to int."""
        result = _to_py_class_value(A(int).convert(False))
        assert result == 0
        assert type(result) is int

    def test_float_passthrough(self) -> None:
        """Test float passthrough."""
        result = _to_py_class_value(A(float).convert(3.14))
        assert result == 3.14
        assert type(result) is float

    def test_int_to_float(self) -> None:
        """Int -> float actually converts."""
        result = _to_py_class_value(A(float).convert(42))
        assert result == 42.0
        assert type(result) is float

    def test_bool_to_float(self) -> None:
        """Bool -> float actually converts."""
        result = _to_py_class_value(A(float).convert(True))
        assert result == 1.0
        assert type(result) is float

    def test_bool_passthrough(self) -> None:
        """Test bool passthrough."""
        result = _to_py_class_value(A(bool).convert(True))
        assert result is True
        assert type(result) is bool

    def test_int_to_bool(self) -> None:
        """Int -> bool actually converts."""
        result = _to_py_class_value(A(bool).convert(1))
        assert result is True
        assert type(result) is bool

    def test_int_zero_to_bool(self) -> None:
        """Test int zero to bool."""
        result = _to_py_class_value(A(bool).convert(0))
        assert result is False
        assert type(result) is bool

    def test_str_passthrough(self) -> None:
        """Test str passthrough — returns tvm_ffi.String (subclass of str)."""
        result = _to_py_class_value(A(str).convert("hello"))
        assert result == "hello"
        assert isinstance(result, str)
        assert isinstance(result, tvm_ffi.core.String)

    def test_bytes_passthrough(self) -> None:
        """Test bytes passthrough — returns tvm_ffi.Bytes (subclass of bytes)."""
        result = _to_py_class_value(A(bytes).convert(b"data"))
        assert result == b"data"
        assert isinstance(result, bytes)
        assert isinstance(result, tvm_ffi.core.Bytes)

    def test_bytearray_to_bytes(self) -> None:
        """Bytearray -> bytes converts to tvm_ffi.Bytes."""
        result = _to_py_class_value(A(bytes).convert(bytearray(b"data")))
        assert result == b"data"
        assert isinstance(result, bytes)
        assert isinstance(result, tvm_ffi.core.Bytes)


# ---------------------------------------------------------------------------
# Category 15: None disambiguation (critical design point)
# ---------------------------------------------------------------------------
class TestNoneDisambiguation:
    def test_none_converts_successfully_for_none_schema(self) -> None:
        """TypeSchema('None').convert(None) returns None as a valid result."""
        result = _to_py_class_value(A(type(None)).convert(None))
        assert result is None

    def test_none_converts_successfully_for_optional(self) -> None:
        """Optional[int].convert(None) returns None as a valid result."""
        result = _to_py_class_value(A(Optional[int]).convert(None))
        assert result is None

    def test_none_fails_for_int(self) -> None:
        """TypeSchema('int').convert(None) raises TypeError."""
        with pytest.raises(TypeError, match="expected int, got None"):
            A(int).convert(None)

    def test_convert_none_success(self) -> None:
        """Convert returns None for Optional[int] with None input."""
        result = _to_py_class_value(A(Optional[int]).convert(None))
        assert result is None

    def test_convert_none_failure(self) -> None:
        """Convert raises TypeError for failed conversion."""
        with pytest.raises(TypeError, match="expected int"):
            A(int).convert(None)

    def test_convert_success_with_value(self) -> None:
        """Convert returns converted value on success."""
        result = _to_py_class_value(A(int).convert(True))
        assert result == 1
        assert type(result) is int

    def test_opaque_ptr_none_converts(self) -> None:
        """ctypes.c_void_p accepts None and converts it to a null opaque pointer."""
        result = _to_py_class_value(A(ctypes.c_void_p).convert(None))
        assert isinstance(result, ctypes.c_void_p)
        assert result.value is None

    def test_convert_opaque_ptr_none(self) -> None:
        """Test convert opaque ptr none."""
        result = _to_py_class_value(A(ctypes.c_void_p).convert(None))
        assert isinstance(result, ctypes.c_void_p)
        assert result.value is None


# ---------------------------------------------------------------------------
# Category 16: Special type conversions
# ---------------------------------------------------------------------------
class TestConvertSpecialTypes:
    def test_dtype_str_converts(self) -> None:
        """Str -> dtype actually creates a DataType object."""
        result = _to_py_class_value(A(tvm_ffi.core.DataType).convert("float32"))
        assert isinstance(result, tvm_ffi.core.DataType)
        assert str(result) == "float32"

    def test_dtype_passthrough(self) -> None:
        """Test dtype passthrough."""
        dt = tvm_ffi.core.DataType("int32")
        result = _to_py_class_value(A(tvm_ffi.core.DataType).convert(dt))
        assert str(result) == str(dt)

    def test_device_passthrough(self) -> None:
        """Test device passthrough."""
        dev = tvm_ffi.Device("cpu", 0)
        result = _to_py_class_value(A(tvm_ffi.Device).convert(dev))
        assert str(result) == str(dev)

    def test_callable_passthrough(self) -> None:
        """Test callable passthrough."""
        fn = lambda x: x
        result = _to_py_class_value(A(Callable).convert(fn))
        assert callable(result)

    def test_opaque_ptr_passthrough(self) -> None:
        """Test opaque ptr passthrough."""
        ptr = ctypes.c_void_p(42)
        result = _to_py_class_value(A(ctypes.c_void_p).convert(ptr))
        assert result is not None


# ---------------------------------------------------------------------------
# Category 17: Container conversion results
# ---------------------------------------------------------------------------
class TestConvertContainers:
    @requires_py39
    def test_array_converts_bool_elements_to_int(self) -> None:
        """Array[int] with bool elements converts them to int."""
        result = _to_py_class_value(A(tuple[int, ...]).convert([True, False, 1]))
        assert list(result) == [1, 0, 1]
        assert all(type(x) is int for x in result)

    @requires_py39
    def test_array_int_passthrough(self) -> None:
        """Array[int] with int elements returns ffi.Array."""
        result = _to_py_class_value(A(tuple[int, ...]).convert([1, 2, 3]))
        assert list(result) == [1, 2, 3]

    @requires_py39
    def test_array_any_passthrough(self) -> None:
        """Array[Any] wraps into ffi.Array."""
        original = [1, "x", None]
        result = _to_py_class_value(A(tuple[typing.Any, ...]).convert(original))
        assert isinstance(result, tvm_ffi.Array)

    @requires_py39
    def test_map_converts_values(self) -> None:
        """Map[str, float] converts int values to float."""
        result = _to_py_class_value(A(tvm_ffi.Map[str, float]).convert({"a": 1, "b": 2}))
        assert isinstance(result, tvm_ffi.Map)
        assert type(result["a"]) is float
        assert type(result["b"]) is float
        assert result["a"] == 1.0
        assert result["b"] == 2.0

    @requires_py39
    def test_map_any_float_converts_values(self) -> None:
        """Map[Any, float] still converts values when keys are Any."""
        result = _to_py_class_value(A(tvm_ffi.Map[typing.Any, float]).convert({"a": 1, "b": 2}))
        assert isinstance(result, tvm_ffi.Map)
        assert type(result["a"]) is float

    @requires_py39
    def test_map_any_any_passthrough(self) -> None:
        """Map[Any, Any] wraps into ffi.Map."""
        original = {"a": 1}
        result = _to_py_class_value(A(tvm_ffi.Map[typing.Any, typing.Any]).convert(original))
        assert isinstance(result, tvm_ffi.Map)

    @requires_py39
    def test_map_empty_dict_convert(self) -> None:
        """Empty dict converts to Map[str, int]."""
        result = _to_py_class_value(A(tvm_ffi.Map[str, int]).convert({}))
        assert len(result) == 0

    @requires_py39
    def test_dict_empty_dict_convert(self) -> None:
        """Empty dict converts to Dict[str, int]."""
        result = _to_py_class_value(A(dict[str, int]).convert({}))
        assert len(result) == 0

    @requires_py39
    def test_tuple_converts_elements(self) -> None:
        """tuple[int, float] converts elements positionally."""
        result = _to_py_class_value(A(tuple[int, float]).convert((True, 42)))
        assert list(result) == [1, 42.0]
        assert type(result[0]) is int
        assert type(result[1]) is float

    @requires_py39
    def test_nested_array_in_map(self) -> None:
        """Map[str, Array[int]] recursively converts elements."""
        result = _to_py_class_value(
            A(tvm_ffi.Map[str, tuple[int, ...]]).convert({"a": [True, False]})
        )
        assert isinstance(result, tvm_ffi.Map)
        assert list(result["a"]) == [1, 0]
        assert all(type(x) is int for x in result["a"])

    @requires_py39
    def test_array_optional_int_all_none(self) -> None:
        """Array[Optional[int]] accepts an all-None payload."""
        result = _to_py_class_value(A(tuple[Optional[int], ...]).convert([None, None, None]))
        assert list(result) == [None, None, None]


# ---------------------------------------------------------------------------
# Category 18: Optional/Union conversion results
# ---------------------------------------------------------------------------
class TestConvertComposite:
    def test_optional_converts_inner(self) -> None:
        """Optional[float].convert(42) converts int -> float."""
        result = _to_py_class_value(A(Optional[float]).convert(42))
        assert result == 42.0
        assert type(result) is float

    def test_optional_none(self) -> None:
        """Test optional none."""
        result = _to_py_class_value(A(Optional[float]).convert(None))
        assert result is None

    def test_union_picks_first_match(self) -> None:
        """Union[int, str] converts bool via int alternative."""
        result = _to_py_class_value(A(Union[int, str]).convert(True))
        assert result == 1
        assert type(result) is int

    def test_union_second_match(self) -> None:
        """Test union second match."""
        result = _to_py_class_value(A(Union[int, str]).convert("hello"))
        assert result == "hello"

    def test_any_passthrough(self) -> None:
        """Any returns value as-is."""
        result = _to_py_class_value(A(typing.Any).convert(42))
        assert result == 42
        result = _to_py_class_value(A(typing.Any).convert(None))
        assert result is None


# ---------------------------------------------------------------------------
# Category 19: Convert rejection cases
# ---------------------------------------------------------------------------
class TestConvertRejections:
    def test_int_rejects_str(self) -> None:
        """Test int rejects str."""
        with pytest.raises(TypeError, match="expected int, got str"):
            A(int).convert("hello")

    def test_int_rejects_float(self) -> None:
        """Test int rejects float."""
        with pytest.raises(TypeError, match="expected int, got float"):
            A(int).convert(3.14)

    def test_str_rejects_int(self) -> None:
        """Test str rejects int."""
        with pytest.raises(TypeError, match="expected str, got int"):
            A(str).convert(42)

    @requires_py39
    def test_array_rejects_wrong_element(self) -> None:
        """Test array rejects wrong element."""
        with pytest.raises(TypeError, match=r"element \[1\].*expected int, got str"):
            A(tuple[int, ...]).convert([1, "x"])

    @requires_py39
    def test_map_rejects_wrong_value(self) -> None:
        """Test map rejects wrong value."""
        with pytest.raises(TypeError, match=r"value for key 'a'.*expected int, got str"):
            A(tvm_ffi.Map[str, int]).convert({"a": "x"})

    @requires_py39
    def test_tuple_rejects_wrong_length(self) -> None:
        """Test tuple rejects wrong length."""
        with pytest.raises(TypeError, match=r"expected tuple of length 2"):
            A(tuple[int, str]).convert((1,))

    def test_convert_failure_raises(self) -> None:
        """Test convert failure raises TypeError."""
        with pytest.raises(TypeError, match="expected int"):
            A(int).convert("hello")


# ---------------------------------------------------------------------------
# Category 20: Numpy conversion
# ---------------------------------------------------------------------------
class TestConvertNumpy:
    def test_numpy_int_to_int(self) -> None:
        """Test numpy int to int."""
        np = pytest.importorskip("numpy")
        result = _to_py_class_value(A(int).convert(np.int64(42)))
        assert result == 42
        assert type(result) is int

    def test_numpy_float_to_float(self) -> None:
        """Test numpy float to float."""
        np = pytest.importorskip("numpy")
        result = _to_py_class_value(A(float).convert(np.float64(3.14)))
        assert result == pytest.approx(3.14)
        # np.float64 is a subclass of float, so isinstance check passes
        # and the value is returned as-is (no forced conversion to plain float)
        assert isinstance(result, float)


# ===========================================================================
# Nested Conversion Tests (with inner-level conversions)
# ===========================================================================


# ---------------------------------------------------------------------------
# Category 21: Array nested with Optional/Union (inner conversion)
# ---------------------------------------------------------------------------
class TestNestedArrayComposite:
    @requires_py39
    def test_array_optional_float_with_bool(self) -> None:
        """Array[Optional[float]] converts bool elements to float."""
        result = _to_py_class_value(A(tuple[Optional[float], ...]).convert([True, None, 3]))
        assert list(result) == [1.0, None, 3.0]
        assert type(result[0]) is float
        assert result[1] is None
        assert type(result[2]) is float

    @requires_py39
    def test_array_optional_int_with_bool(self) -> None:
        """Array[Optional[int]] converts bool elements to int."""
        result = _to_py_class_value(A(tuple[Optional[int], ...]).convert([True, None, 2]))
        assert list(result) == [1, None, 2]
        assert type(result[0]) is int
        assert result[1] is None

    @requires_py39
    def test_array_union_int_str_with_bool(self) -> None:
        """Array[Union[int, str]] converts bool via int alternative."""
        result = _to_py_class_value(A(tuple[Union[int, str], ...]).convert([True, "hello", False]))
        assert list(result) == [1, "hello", 0]
        assert type(result[0]) is int
        assert type(result[1]) is str
        assert type(result[2]) is int

    @requires_py39
    def test_array_union_float_str_with_int(self) -> None:
        """Array[Union[float, str]] converts int via float alternative."""
        result = _to_py_class_value(A(tuple[Union[float, str], ...]).convert([42, "hi", True]))
        assert list(result) == [42.0, "hi", 1.0]
        assert type(result[0]) is float
        assert type(result[2]) is float

    @requires_py39
    def test_array_optional_float_all_none(self) -> None:
        """Array[Optional[float]] with all None elements."""
        result = _to_py_class_value(A(tuple[Optional[float], ...]).convert([None, None]))
        assert list(result) == [None, None]

    @requires_py39
    def test_array_optional_float_empty(self) -> None:
        """Array[Optional[float]] with empty list."""
        result = _to_py_class_value(A(tuple[Optional[float], ...]).convert([]))
        assert list(result) == []

    @requires_py39
    def test_array_union_failure_in_element(self) -> None:
        """Array[Union[int, str]] fails when element matches no alternative."""
        with pytest.raises(TypeError, match=r"element \[1\].*got float"):
            A(tuple[Union[int, str], ...]).check_value([1, 3.14])


# ---------------------------------------------------------------------------
# Category 22: Map/Dict nested with Optional/Union (inner conversion)
# ---------------------------------------------------------------------------
class TestNestedMapComposite:
    @requires_py39
    def test_map_str_optional_float_with_int(self) -> None:
        """Map[str, Optional[float]] converts int values to float."""
        result = _to_py_class_value(
            A(tvm_ffi.Map[str, Optional[float]]).convert({"a": 1, "b": None})
        )
        assert type(result["a"]) is float
        assert result["a"] == 1.0
        assert result["b"] is None

    @requires_py39
    def test_map_str_union_int_str(self) -> None:
        """Map[str, Union[int, str]] converts bool values via int."""
        result = _to_py_class_value(
            A(tvm_ffi.Map[str, Union[int, str]]).convert({"x": True, "y": "hello"})
        )
        assert result["x"] == 1
        assert result["y"] == "hello"
        assert type(result["x"]) is int

    @requires_py39
    def test_dict_str_optional_int(self) -> None:
        """Dict[str, Optional[int]] with bool conversion."""
        result = _to_py_class_value(
            A(dict[str, Optional[int]]).convert({"a": True, "b": None, "c": 42})
        )
        assert result["a"] == 1
        assert result["b"] is None
        assert result["c"] == 42
        assert type(result["a"]) is int

    @requires_py39
    def test_map_str_optional_float_failure(self) -> None:
        """Map[str, Optional[float]] fails for non-float non-None value."""
        with pytest.raises(TypeError, match="expected float"):
            A(tvm_ffi.Map[str, Optional[float]]).check_value({"a": "bad"})


# ---------------------------------------------------------------------------
# Category 23: Nested containers (container inside container)
# ---------------------------------------------------------------------------
class TestNestedContainerInContainer:
    @requires_py39
    def test_array_of_array_int(self) -> None:
        """Array[Array[int]] with inner bool->int conversion."""
        result = _to_py_class_value(A(tuple[tuple[int, ...], ...]).convert([[True, False], [1, 2]]))
        assert [list(row) for row in result] == [[1, 0], [1, 2]]
        assert all(type(x) is int for row in result for x in row)

    @requires_py39
    def test_array_of_array_float(self) -> None:
        """Array[Array[float]] with inner int->float conversion."""
        result = _to_py_class_value(A(tuple[tuple[float, ...], ...]).convert([[1, 2], [True, 3]]))
        assert [list(row) for row in result] == [[1.0, 2.0], [1.0, 3.0]]
        assert all(type(x) is float for row in result for x in row)

    @requires_py39
    def test_map_str_array_float(self) -> None:
        """Map[str, Array[float]] with int->float conversion in arrays."""
        result = _to_py_class_value(
            A(tvm_ffi.Map[str, tuple[float, ...]]).convert({"a": [1, 2], "b": [True, 3]})
        )
        assert list(result["a"]) == [1.0, 2.0]
        assert list(result["b"]) == [1.0, 3.0]
        assert all(type(x) is float for x in result["a"])
        assert all(type(x) is float for x in result["b"])

    @requires_py39
    def test_dict_str_array_int(self) -> None:
        """Dict[str, Array[int]] with bool->int conversion."""
        result = _to_py_class_value(A(dict[str, tuple[int, ...]]).convert({"a": [True, False]}))
        assert list(result["a"]) == [1, 0]
        assert all(type(x) is int for x in result["a"])

    @requires_py39
    def test_array_of_map_str_int(self) -> None:
        """Array[Map[str, int]] with bool->int value conversion."""
        result = _to_py_class_value(
            A(tuple[tvm_ffi.Map[str, int], ...]).convert([{"x": True}, {"y": 2}])
        )
        assert result[0]["x"] == 1
        assert result[1]["y"] == 2
        assert type(result[0]["x"]) is int

    @requires_py39
    def test_map_str_map_str_float(self) -> None:
        """Map[str, Map[str, float]] double nested with int->float."""
        result = _to_py_class_value(
            A(tvm_ffi.Map[str, tvm_ffi.Map[str, float]]).convert({"outer": {"inner": 42}})
        )
        assert result["outer"]["inner"] == 42.0
        assert type(result["outer"]["inner"]) is float

    @requires_py39
    def test_list_of_list_int(self) -> None:
        """List[List[int]] with bool->int conversion."""
        result = _to_py_class_value(A(list[list[int]]).convert([[True, 1], [False, 2]]))
        assert [list(row) for row in result] == [[1, 1], [0, 2]]
        assert all(type(x) is int for row in result for x in row)

    @requires_py39
    def test_nested_failure_array_of_array(self) -> None:
        """Array[Array[int]] error propagation through nested arrays."""
        with pytest.raises(TypeError, match="expected int"):
            A(tuple[tuple[int, ...], ...]).check_value([[1, 2], [3, "bad"]])

    @requires_py39
    def test_empty_inner_containers(self) -> None:
        """Map[str, Array[int]] with empty inner arrays."""
        result = _to_py_class_value(
            A(tvm_ffi.Map[str, tuple[int, ...]]).convert({"a": [], "b": []})
        )
        assert list(result["a"]) == []
        assert list(result["b"]) == []

    @requires_py39
    def test_array_of_array_of_array_int(self) -> None:
        """Three-level nested Array[int] conversion still works."""
        schema = A(tuple[tuple[tuple[int, ...], ...], ...])
        data = [[[1, 2], [True, False]], [[3], [4, 5, 6]]]
        result = _to_py_class_value(schema.convert(data))
        assert list(result[0][0]) == [1, 2]
        assert list(result[0][1]) == [1, 0]
        assert type(result[0][1][0]) is int

    @requires_py39
    def test_map_of_map_of_array_float(self) -> None:
        """Nested map-to-array conversion still coerces inner values."""
        schema = A(tvm_ffi.Map[str, tvm_ffi.Map[str, tuple[float, ...]]])
        data = {"outer": {"inner": [1, 2, True]}}
        result = _to_py_class_value(schema.convert(data))
        assert list(result["outer"]["inner"]) == [1.0, 2.0, 1.0]
        assert type(result["outer"]["inner"][0]) is float


# ---------------------------------------------------------------------------
# Category 24: Optional/Union wrapping containers
# ---------------------------------------------------------------------------
class TestOptionalUnionWrappingContainers:
    @requires_py39
    def test_optional_array_int_with_conversion(self) -> None:
        """Optional[Array[int]] converts inner bool elements."""
        schema = A(Optional[tuple[int, ...]])
        result = _to_py_class_value(schema.convert([True, 2]))
        assert list(result) == [1, 2]
        assert type(result[0]) is int

    @requires_py39
    def test_optional_array_int_none(self) -> None:
        """Optional[Array[int]] accepts None."""
        result = _to_py_class_value(A(Optional[tuple[int, ...]]).convert(None))
        assert result is None

    @requires_py39
    def test_optional_map_str_float(self) -> None:
        """Optional[Map[str, float]] converts inner int values."""
        result = _to_py_class_value(A(Optional[tvm_ffi.Map[str, float]]).convert({"a": 1}))
        assert result["a"] == 1.0
        assert type(result["a"]) is float

    @requires_py39
    def test_optional_map_str_float_none(self) -> None:
        """Optional[Map[str, float]] accepts None."""
        result = _to_py_class_value(A(Optional[tvm_ffi.Map[str, float]]).convert(None))
        assert result is None

    @requires_py39
    def test_union_array_int_or_map_str_int(self) -> None:
        """Union[Array[int], Map[str, int]] matches first with conversion."""
        schema = A(Union[tuple[int, ...], tvm_ffi.Map[str, int]])
        # list matches Array alternative
        result = _to_py_class_value(schema.convert([True, 2]))
        assert list(result) == [1, 2]
        assert type(result[0]) is int

    @requires_py39
    def test_union_array_int_or_map_str_int_dict(self) -> None:
        """Union[Array[int], Map[str, int]] matches Map for dict input."""
        schema = A(Union[tuple[int, ...], tvm_ffi.Map[str, int]])
        result = _to_py_class_value(schema.convert({"a": True}))
        assert result["a"] == 1
        assert type(result["a"]) is int

    @requires_py39
    def test_union_int_or_array_optional_float(self) -> None:
        """Union[int, Array[Optional[float]]] matches array with nested conversions."""
        schema = A(Union[int, tuple[Optional[float], ...]])
        result = _to_py_class_value(schema.convert([True, None, 1]))
        assert list(result) == [1.0, None, 1.0]
        assert type(result[0]) is float
        assert result[1] is None

    @requires_py39
    def test_optional_optional_array_int(self) -> None:
        """Optional[Optional[Array[int]]] with inner conversion."""
        schema = A(Optional[Optional[tuple[int, ...]]])
        assert _to_py_class_value(schema.convert(None)) is None
        result = _to_py_class_value(schema.convert([True, 2]))
        assert list(result) == [1, 2]
        assert type(result[0]) is int


# ---------------------------------------------------------------------------
# Category 25: Tuple nested with other types
# ---------------------------------------------------------------------------
class TestNestedTuple:
    @requires_py39
    def test_array_of_tuple_int_float(self) -> None:
        """Array[tuple[int, float]] with element-wise conversion."""
        result = _to_py_class_value(
            A(tuple[tuple[int, float], ...]).convert([(True, 1), (2, True)])
        )
        # Check element values; FFI storage may normalize float 1.0 to int 1
        # when stored inside an ffi.Array, so we only check values not types.
        assert result[0][0] == 1
        assert result[0][1] == 1.0
        assert result[1][0] == 2
        assert result[1][1] == 1.0

    @requires_py39
    def test_map_str_tuple_int_str(self) -> None:
        """Map[str, tuple[int, str]] with inner bool->int conversion."""
        result = _to_py_class_value(
            A(tvm_ffi.Map[str, tuple[int, str]]).convert({"a": (True, "hello")})
        )
        assert result["a"][0] == 1
        assert str(result["a"][1]) == "hello"
        assert type(result["a"][0]) is int

    @requires_py39
    def test_tuple_of_array_int_and_map(self) -> None:
        """tuple[Array[int], Map[str, float]] nested conversion."""
        schema = A(tuple[tuple[int, ...], tvm_ffi.Map[str, float]])
        result = _to_py_class_value(schema.convert(([True, 2], {"k": 3})))
        assert list(result[0]) == [1, 2]
        assert result[1]["k"] == 3.0
        assert type(result[0][0]) is int
        assert type(result[1]["k"]) is float

    @requires_py39
    def test_tuple_of_optional_int_and_optional_float(self) -> None:
        """tuple[Optional[int], Optional[float]] with conversions."""
        schema = A(tuple[Optional[int], Optional[float]])
        result = _to_py_class_value(schema.convert((True, None)))
        assert list(result) == [1, None]
        assert type(result[0]) is int
        assert result[1] is None

    @requires_py39
    def test_tuple_nested_failure(self) -> None:
        """tuple[Array[int], str] error propagation from inner array."""
        with pytest.raises(TypeError, match=r"element .0..*element .1..*expected int"):
            A(tuple[tuple[int, ...], str]).check_value(([1, "bad"], "ok"))


# ---------------------------------------------------------------------------
# Category 26: Deep nesting (3+ levels)
# ---------------------------------------------------------------------------
class TestDeepNesting:
    @requires_py39
    def test_map_str_array_optional_int(self) -> None:
        """Map[str, Array[Optional[int]]] with 3-level nesting and conversion."""
        result = _to_py_class_value(
            A(tvm_ffi.Map[str, tuple[Optional[int], ...]]).convert({"a": [1, None, True]})
        )
        assert list(result["a"]) == [1, None, 1]
        assert type(result["a"][0]) is int
        assert result["a"][1] is None
        assert type(result["a"][2]) is int

    @requires_py39
    def test_array_map_str_optional_float(self) -> None:
        """Array[Map[str, Optional[float]]] with 3-level nesting."""
        result = _to_py_class_value(
            A(tuple[tvm_ffi.Map[str, Optional[float]], ...]).convert(
                [{"x": 1, "y": None}, {"z": True}]
            )
        )
        assert result[0]["x"] == 1.0
        assert result[0]["y"] is None
        assert result[1]["z"] == 1.0
        assert type(result[0]["x"]) is float
        assert type(result[1]["z"]) is float

    @requires_py39
    def test_optional_array_map_str_int(self) -> None:
        """Optional[Array[Map[str, int]]] 3 levels deep."""
        schema = A(Optional[tuple[tvm_ffi.Map[str, int], ...]])
        result = _to_py_class_value(schema.convert([{"a": True}, {"b": 2}]))
        assert result[0]["a"] == 1
        assert result[1]["b"] == 2
        assert type(result[0]["a"]) is int

        assert _to_py_class_value(schema.convert(None)) is None

    @requires_py39
    def test_map_str_array_array_int(self) -> None:
        """Map[str, Array[Array[int]]] 3-level container nesting."""
        result = _to_py_class_value(
            A(tvm_ffi.Map[str, tuple[tuple[int, ...], ...]]).convert({"m": [[True, 1], [False, 2]]})
        )
        assert [list(row) for row in result["m"]] == [[1, 1], [0, 2]]
        assert all(type(x) is int for row in result["m"] for x in row)

    @requires_py39
    def test_array_array_optional_float(self) -> None:
        """Array[Array[Optional[float]]] deep nesting with None and conversion."""
        result = _to_py_class_value(
            A(tuple[tuple[Optional[float], ...], ...]).convert([[1, None], [True, 3.14]])
        )
        assert list(result[0]) == [1.0, None]
        assert list(result[1]) == [1.0, 3.14]
        assert type(result[0][0]) is float
        assert result[0][1] is None
        assert type(result[1][0]) is float

    @requires_py39
    def test_deep_nesting_failure_propagation(self) -> None:
        """Error from deepest level propagates with full path info."""
        with pytest.raises(TypeError, match=r"value for key 'key'.*element .1..*expected int"):
            A(tvm_ffi.Map[str, tuple[Optional[int], ...]]).check_value({"key": [1, "bad"]})


# ---------------------------------------------------------------------------
# Category 27: FFI container inputs (tvm_ffi.Array/List/Map/Dict)
# ---------------------------------------------------------------------------
class TestFFIContainerInputs:
    @requires_py39
    def test_ffi_array_with_element_conversion(self) -> None:
        """tvm_ffi.Array([True, 2]) passes Array[int] with bool->int conversion."""
        arr = tvm_ffi.Array([True, 2, 3])
        result = _to_py_class_value(A(tuple[int, ...]).convert(arr))
        assert list(result) == [1, 2, 3]
        assert type(result[0]) is int

    @requires_py39
    def test_ffi_array_any_passthrough(self) -> None:
        """tvm_ffi.Array passes Array[Any] as-is."""
        arr = tvm_ffi.Array([1, "x", None])
        result = _to_py_class_value(A(tuple[typing.Any, ...]).convert(arr))
        assert result.same_as(arr)

    @requires_py39
    def test_ffi_list_with_list_schema(self) -> None:
        """tvm_ffi.List passes List[int] with conversion."""
        lst = tvm_ffi.List([True, 2])
        result = _to_py_class_value(A(list[int]).convert(lst))
        assert list(result) == [1, 2]
        assert type(result[0]) is int

    @requires_py39
    def test_ffi_list_accepted_by_array_schema(self) -> None:
        """tvm_ffi.List passes Array schema (C++ allows cross-type via kOtherTypeIndex)."""
        lst = tvm_ffi.List([1, 2])
        A(tuple[int, ...]).check_value(lst)

    @requires_py39
    def test_ffi_array_accepted_by_list_schema(self) -> None:
        """tvm_ffi.Array passes List schema (C++ allows cross-type via kOtherTypeIndex)."""
        arr = tvm_ffi.Array([1, 2])
        A(list[int]).check_value(arr)

    @requires_py39
    def test_ffi_map_with_value_conversion(self) -> None:
        """tvm_ffi.Map passes Map[str, int] with bool->int conversion."""
        m = tvm_ffi.Map({"a": True, "b": 2})
        result = _to_py_class_value(A(tvm_ffi.Map[str, int]).convert(m))
        assert result["a"] == 1
        assert result["b"] == 2
        assert type(result["a"]) is int

    @requires_py39
    def test_ffi_map_any_any_passthrough(self) -> None:
        """tvm_ffi.Map passes Map[Any, Any] as-is."""
        m = tvm_ffi.Map({"a": 1})
        result = _to_py_class_value(A(tvm_ffi.Map[typing.Any, typing.Any]).convert(m))
        assert result.same_as(m)

    @requires_py39
    def test_ffi_dict_with_dict_schema(self) -> None:
        """tvm_ffi.Dict passes Dict[str, float] with int->float conversion."""
        d = tvm_ffi.Dict({"x": 1, "y": 2})
        result = _to_py_class_value(A(dict[str, float]).convert(d))
        assert result["x"] == 1.0
        assert result["y"] == 2.0
        assert type(result["x"]) is float

    @requires_py39
    def test_ffi_dict_accepted_by_map_schema(self) -> None:
        """tvm_ffi.Dict passes Map schema (C++ allows cross-type via kOtherTypeIndex)."""
        d = tvm_ffi.Dict({"a": 1})
        A(tvm_ffi.Map[str, int]).check_value(d)

    @requires_py39
    def test_ffi_map_accepted_by_dict_schema(self) -> None:
        """tvm_ffi.Map passes Dict schema (C++ allows cross-type via kOtherTypeIndex)."""
        m = tvm_ffi.Map({"a": 1})
        A(dict[str, int]).check_value(m)

    @requires_py39
    def test_ffi_array_nested_optional_float(self) -> None:
        """tvm_ffi.Array with nested Optional[float] conversion."""
        arr = tvm_ffi.Array([1, None, True])
        result = _to_py_class_value(A(tuple[Optional[float], ...]).convert(arr))
        assert list(result) == [1.0, None, 1.0]
        assert type(result[0]) is float
        assert result[1] is None

    @requires_py39
    def test_ffi_map_nested_array_int(self) -> None:
        """tvm_ffi.Map with value being a Python list, converted as Array[int]."""
        # Map values are already stored; create a map with array values
        m = tvm_ffi.Map({"k": tvm_ffi.Array([True, 2])})
        result = _to_py_class_value(A(tvm_ffi.Map[str, tuple[int, ...]]).convert(m))
        assert list(result["k"]) == [1, 2]
        assert type(result["k"][0]) is int

    @requires_py39
    def test_ffi_array_wrong_element_type(self) -> None:
        """tvm_ffi.Array with wrong element type gives clear error."""
        arr = tvm_ffi.Array([1, "bad", 3])
        with pytest.raises(TypeError, match=r"element \[1\].*expected int"):
            A(tuple[int, ...]).check_value(arr)

    @requires_py39
    def test_ffi_map_wrong_value_type(self) -> None:
        """tvm_ffi.Map with wrong value type gives clear error."""
        m = tvm_ffi.Map({"a": 1, "b": "bad"})
        with pytest.raises(TypeError, match=r"value for key.*expected int"):
            A(tvm_ffi.Map[str, int]).check_value(m)

    def test_ffi_array_object_schema(self) -> None:
        """tvm_ffi.Array passes Object schema (it is a CObject)."""
        arr = tvm_ffi.Array([1, 2])
        A(tvm_ffi.core.Object).check_value(arr)

    def test_ffi_map_object_schema(self) -> None:
        """tvm_ffi.Map passes Object schema (it is a CObject)."""
        m = tvm_ffi.Map({"a": 1})
        A(tvm_ffi.core.Object).check_value(m)


# ---------------------------------------------------------------------------
# Category 28: Mixed Python and FFI containers in nesting
# ---------------------------------------------------------------------------
class TestMixedPythonFFIContainers:
    @requires_py39
    def test_python_list_of_ffi_arrays(self) -> None:
        """Python list containing tvm_ffi.Array elements, Array[Array[int]]."""
        inner1 = tvm_ffi.Array([True, 2])
        inner2 = tvm_ffi.Array([3, False])
        result = _to_py_class_value(A(tuple[tuple[int, ...], ...]).convert([inner1, inner2]))
        assert [list(row) for row in result] == [[1, 2], [3, 0]]

    @requires_py39
    def test_python_dict_with_ffi_array_values(self) -> None:
        """Python dict with tvm_ffi.Array values, Map[str, Array[float]]."""
        val = tvm_ffi.Array([1, True])
        result = _to_py_class_value(A(tvm_ffi.Map[str, tuple[float, ...]]).convert({"k": val}))
        assert list(result["k"]) == [1.0, 1.0]
        assert all(type(x) is float for x in result["k"])

    @requires_py39
    def test_ffi_map_with_python_list_in_union(self) -> None:
        """Union[Map[str, int], Array[int]] with tvm_ffi.Map input."""
        schema = A(Union[tvm_ffi.Map[str, int], tuple[int, ...]])
        m = tvm_ffi.Map({"a": True})
        result = _to_py_class_value(schema.convert(m))
        assert result["a"] == 1
        assert type(result["a"]) is int

    @requires_py39
    def test_ffi_array_in_optional(self) -> None:
        """Optional[Array[int]] with tvm_ffi.Array input."""
        arr = tvm_ffi.Array([True, 2])
        result = _to_py_class_value(A(Optional[tuple[int, ...]]).convert(arr))
        assert list(result) == [1, 2]
        assert type(result[0]) is int


# ---------------------------------------------------------------------------
# Category 29: Error propagation through deeply nested FFI containers
# ---------------------------------------------------------------------------
class TestNestedErrorPropagation:
    @requires_py39
    def test_array_array_int_inner_failure(self) -> None:
        """Error path: Array[Array[int]] -> element [1] -> element [0]."""
        with pytest.raises(TypeError, match=r"element \[1\].*element \[0\].*expected int, got str"):
            A(tuple[tuple[int, ...], ...]).convert([[1], ["bad"]])

    @requires_py39
    def test_map_array_int_inner_failure(self) -> None:
        """Error path: Map -> value for key 'k' -> element [2]."""
        with pytest.raises(
            TypeError,
            match=r"value for key 'k'.*element \[2\].*expected int, got str",
        ):
            A(tvm_ffi.Map[str, tuple[int, ...]]).convert({"k": [1, 2, "bad"]})

    @requires_py39
    def test_array_map_int_inner_failure(self) -> None:
        """Error path: Array -> element [0] -> value for key 'x'."""
        with pytest.raises(
            TypeError,
            match=r"element \[0\].*value for key 'x'.*expected int, got str",
        ):
            A(tuple[tvm_ffi.Map[str, int], ...]).convert([{"x": "bad"}])

    @requires_py39
    def test_optional_array_int_inner_failure(self) -> None:
        """Error path through Optional -> Array -> element."""
        with pytest.raises(TypeError, match=r"element \[1\].*expected int, got str"):
            A(Optional[tuple[int, ...]]).convert([1, "bad"])

    @requires_py39
    def test_tuple_array_int_inner_failure(self) -> None:
        """Error path: tuple -> element [0] -> element [1]."""
        with pytest.raises(TypeError, match=r"element \[0\].*element \[1\].*expected int, got str"):
            A(tuple[tuple[int, ...], str]).convert(([1, "bad"], "ok"))

    @requires_py39
    def test_deep_3_level_error(self) -> None:
        """Error at 3 levels deep: Map -> Array -> Optional -> type mismatch."""
        with pytest.raises(TypeError, match=r"value for key 'key'.*element .1..*expected int"):
            A(tvm_ffi.Map[str, tuple[Optional[int], ...]]).check_value({"key": [1, "bad"]})

    @requires_py39
    def test_ffi_array_nested_error(self) -> None:
        """Error from tvm_ffi.Array in nested context."""
        arr = tvm_ffi.Array([1, "bad", 3])
        with pytest.raises(TypeError, match=r"element \[1\].*expected int"):
            A(tuple[int, ...]).convert(arr)


# ---------------------------------------------------------------------------
# Category 30: Custom object type exact match
# ---------------------------------------------------------------------------
class TestCustomObjectExactMatch:
    def test_test_int_pair_pass(self) -> None:
        """TestIntPair passes TypeSchema('testing.TestIntPair')."""
        obj = TestIntPair(1, 2)
        A(TestIntPair).check_value(obj)

    def test_test_object_base_pass(self) -> None:
        """TestObjectBase passes its own schema."""
        obj = TestObjectBase(v_i64=10, v_f64=1.5, v_str="hi")
        A(TestObjectBase).check_value(obj)

    def test_test_object_derived_pass(self) -> None:
        """TestObjectDerived passes its own schema."""
        obj = TestObjectDerived(v_map={"a": 1}, v_array=[1], v_i64=0, v_f64=0.0, v_str="")
        A(TestObjectDerived).check_value(obj)

    def test_cxx_class_base_pass(self) -> None:
        """_TestCxxClassBase passes its own schema."""
        obj = _TestCxxClassBase(v_i64=1, v_i32=2)
        A(_TestCxxClassBase).check_value(obj)

    def test_cxx_class_derived_pass(self) -> None:
        """_TestCxxClassDerived passes its own schema."""
        obj = _TestCxxClassDerived(v_i64=1, v_i32=2, v_f64=3.0)
        A(_TestCxxClassDerived).check_value(obj)

    def test_cxx_class_derived_derived_pass(self) -> None:
        """_TestCxxClassDerivedDerived passes its own schema."""
        obj = _TestCxxClassDerivedDerived(v_i64=1, v_i32=2, v_f64=3.0, v_bool=True)
        A(_TestCxxClassDerivedDerived).check_value(obj)


# ---------------------------------------------------------------------------
# Category 31: Custom object type hierarchy (subclass passes parent schema)
# ---------------------------------------------------------------------------
class TestCustomObjectHierarchy:
    def test_derived_passes_base_schema(self) -> None:
        """TestObjectDerived passes TypeSchema('testing.TestObjectBase')."""
        obj = TestObjectDerived(v_map={"a": 1}, v_array=[1], v_i64=0, v_f64=0.0, v_str="")
        A(TestObjectBase).check_value(obj)

    def test_derived_passes_object_schema(self) -> None:
        """TestObjectDerived passes TypeSchema('Object')."""
        obj = TestObjectDerived(v_map={"a": 1}, v_array=[1], v_i64=0, v_f64=0.0, v_str="")
        A(tvm_ffi.core.Object).check_value(obj)

    def test_cxx_derived_passes_base(self) -> None:
        """_TestCxxClassDerived passes TestCxxClassBase schema."""
        obj = _TestCxxClassDerived(v_i64=1, v_i32=2, v_f64=3.0)
        A(_TestCxxClassBase).check_value(obj)

    def test_cxx_derived_derived_passes_base(self) -> None:
        """_TestCxxClassDerivedDerived passes TestCxxClassBase schema (2-level up)."""
        obj = _TestCxxClassDerivedDerived(v_i64=1, v_i32=2, v_f64=3.0, v_bool=True)
        A(_TestCxxClassBase).check_value(obj)

    def test_cxx_derived_derived_passes_derived(self) -> None:
        """_TestCxxClassDerivedDerived passes TestCxxClassDerived schema (1-level up)."""
        obj = _TestCxxClassDerivedDerived(v_i64=1, v_i32=2, v_f64=3.0, v_bool=True)
        A(_TestCxxClassDerived).check_value(obj)

    def test_all_custom_objects_pass_object_schema(self) -> None:
        """Every custom object passes the generic Object schema."""
        objs = [
            TestIntPair(1, 2),
            TestObjectBase(v_i64=10, v_f64=1.5, v_str="hi"),
            _TestCxxClassBase(v_i64=1, v_i32=2),
            _TestCxxClassDerived(v_i64=1, v_i32=2, v_f64=3.0),
            _TestCxxClassDerivedDerived(v_i64=1, v_i32=2, v_f64=3.0, v_bool=True),
        ]
        schema = A(tvm_ffi.core.Object)
        for obj in objs:
            schema.check_value(obj)


# ---------------------------------------------------------------------------
# Category 32: Custom object type rejection
# ---------------------------------------------------------------------------
class TestCustomObjectRejection:
    def test_wrong_object_type(self) -> None:
        """TestIntPair fails TypeSchema('testing.TestObjectBase')."""
        obj = TestIntPair(1, 2)
        with pytest.raises(TypeError, match=r"testing.TestIntPair"):
            A(TestObjectBase).check_value(obj)

    def test_base_fails_derived_schema(self) -> None:
        """Parent object fails child schema (TestObjectBase fails TestObjectDerived)."""
        obj = TestObjectBase(v_i64=10, v_f64=1.5, v_str="hi")
        with pytest.raises(TypeError, match=r"testing.TestObjectBase"):
            A(TestObjectDerived).check_value(obj)

    def test_non_object_fails_custom_schema(self) -> None:
        """Plain int fails custom object schema."""
        with pytest.raises(TypeError, match=r"expected testing\.TestIntPair.*got int"):
            A(TestIntPair).check_value(42)

    def test_none_fails_custom_schema(self) -> None:
        """None fails custom object schema."""
        with pytest.raises(TypeError, match="got None"):
            A(TestIntPair).check_value(None)

    def test_string_fails_custom_schema(self) -> None:
        """String fails custom object schema."""
        with pytest.raises(TypeError, match="got str"):
            A(TestIntPair).check_value("hello")

    def test_cxx_base_fails_derived_schema(self) -> None:
        """_TestCxxClassBase fails _TestCxxClassDerived schema."""
        obj = _TestCxxClassBase(v_i64=1, v_i32=2)
        with pytest.raises(TypeError):
            A(_TestCxxClassDerived).check_value(obj)

    def test_sibling_types_reject_each_other(self) -> None:
        """TestIntPair and TestCxxClassBase are unrelated -- reject each other."""
        pair = TestIntPair(1, 2)
        base = _TestCxxClassBase(v_i64=1, v_i32=2)
        with pytest.raises(TypeError):
            A(_TestCxxClassBase).check_value(pair)
        with pytest.raises(TypeError):
            A(TestIntPair).check_value(base)


# ---------------------------------------------------------------------------
# Category 33: Custom objects in containers
# ---------------------------------------------------------------------------
class TestCustomObjectInContainers:
    @requires_py39
    def test_array_of_custom_objects(self) -> None:
        """Array[testing.TestIntPair] with matching elements."""
        objs = [TestIntPair(1, 2), TestIntPair(3, 4)]
        A(tuple[TestIntPair, ...]).check_value(objs)

    @requires_py39
    def test_array_of_custom_objects_wrong_type(self) -> None:
        """Array[testing.TestIntPair] with wrong element type fails."""
        objs = [TestIntPair(1, 2), _TestCxxClassBase(v_i64=1, v_i32=2)]
        with pytest.raises(TypeError, match=r"element \[1\]"):
            A(tuple[TestIntPair, ...]).check_value(objs)

    @requires_py39
    def test_array_of_base_with_derived_elements(self) -> None:
        """Array[testing.TestObjectBase] accepts derived elements via hierarchy."""
        base = TestObjectBase(v_i64=1, v_f64=1.0, v_str="a")
        derived = TestObjectDerived(v_map={"a": 1}, v_array=[1], v_i64=0, v_f64=0.0, v_str="")
        A(tuple[TestObjectBase, ...]).check_value([base, derived])

    @requires_py39
    def test_map_str_to_custom_object(self) -> None:
        """Map[str, testing.TestIntPair] pass."""
        objs = {"a": TestIntPair(1, 2), "b": TestIntPair(3, 4)}
        A(tvm_ffi.Map[str, TestIntPair]).check_value(objs)

    @requires_py39
    def test_map_str_to_custom_object_wrong_value(self) -> None:
        """Map[str, testing.TestIntPair] with int value fails."""
        data = {"a": TestIntPair(1, 2), "b": 42}
        with pytest.raises(TypeError, match="value for key 'b'"):
            A(tvm_ffi.Map[str, TestIntPair]).check_value(data)

    @requires_py39
    def test_ffi_array_of_custom_objects(self) -> None:
        """tvm_ffi.Array of custom objects passes Array[Object]."""
        arr = tvm_ffi.Array([TestIntPair(1, 2), TestObjectBase(v_i64=1, v_f64=2.0, v_str="s")])
        A(tuple[tvm_ffi.core.Object, ...]).check_value(arr)

    @requires_py39
    def test_ffi_array_of_custom_objects_specific_type(self) -> None:
        """tvm_ffi.Array of TestIntPair passes Array[testing.TestIntPair]."""
        arr = tvm_ffi.Array([TestIntPair(1, 2), TestIntPair(3, 4)])
        A(tuple[TestIntPair, ...]).check_value(arr)

    @requires_py39
    def test_ffi_map_with_custom_object_values(self) -> None:
        """tvm_ffi.Map with custom object values passes."""
        m = tvm_ffi.Map({"x": TestIntPair(1, 2), "y": TestIntPair(3, 4)})
        A(tvm_ffi.Map[str, TestIntPair]).check_value(m)


# ---------------------------------------------------------------------------
# Category 34: Optional/Union with custom objects
# ---------------------------------------------------------------------------
class TestCustomObjectOptionalUnion:
    def test_optional_custom_object_with_value(self) -> None:
        """Optional[testing.TestIntPair] with actual object."""
        obj = TestIntPair(1, 2)
        A(Optional[TestIntPair]).check_value(obj)

    def test_optional_custom_object_with_none(self) -> None:
        """Optional[testing.TestIntPair] with None."""
        A(Optional[TestIntPair]).check_value(None)

    def test_optional_custom_object_wrong_type(self) -> None:
        """Optional[testing.TestIntPair] with wrong object type."""
        obj = _TestCxxClassBase(v_i64=1, v_i32=2)
        with pytest.raises(TypeError):
            A(Optional[TestIntPair]).check_value(obj)

    def test_union_custom_object_and_int(self) -> None:
        """Union[testing.TestIntPair, int] with object."""
        obj = TestIntPair(1, 2)
        A(Union[TestIntPair, int]).check_value(obj)

    def test_union_custom_object_and_int_with_int(self) -> None:
        """Union[testing.TestIntPair, int] with int."""
        A(Union[TestIntPair, int]).check_value(42)

    def test_union_custom_object_and_int_with_wrong(self) -> None:
        """Union[testing.TestIntPair, int] with str fails."""
        with pytest.raises(TypeError):
            A(Union[TestIntPair, int]).check_value("bad")

    def test_union_two_custom_objects(self) -> None:
        """Union of two custom types accepts both."""
        pair = TestIntPair(1, 2)
        base = _TestCxxClassBase(v_i64=1, v_i32=2)
        schema = A(Union[TestIntPair, _TestCxxClassBase])
        schema.check_value(pair)
        schema.check_value(base)

    def test_union_two_custom_objects_rejects_third(self) -> None:
        """Union of two custom types rejects a third."""
        obj = TestObjectBase(v_i64=1, v_f64=2.0, v_str="s")
        with pytest.raises(TypeError):
            A(Union[TestIntPair, _TestCxxClassBase]).check_value(obj)


# ---------------------------------------------------------------------------
# Category 35: Custom objects with from_type_index
# ---------------------------------------------------------------------------
class TestCustomObjectFromTypeIndex:
    def test_from_type_index_custom_object(self) -> None:
        """from_type_index resolves a custom object type and validates."""
        obj = TestIntPair(1, 2)
        tindex = tvm_ffi.core._object_type_key_to_index("testing.TestIntPair")
        assert tindex is not None
        schema = TypeSchema.from_type_index(tindex)
        assert schema.origin == "testing.TestIntPair"
        schema.check_value(obj)

    def test_from_type_index_rejects_wrong_object(self) -> None:
        """from_type_index schema rejects wrong object type."""
        tindex = tvm_ffi.core._object_type_key_to_index("testing.TestIntPair")
        assert tindex is not None
        schema = TypeSchema.from_type_index(tindex)
        with pytest.raises(TypeError):
            schema.check_value(_TestCxxClassBase(v_i64=1, v_i32=2))

    def test_from_type_index_hierarchy(self) -> None:
        """from_type_index for base type accepts derived objects."""
        tindex = tvm_ffi.core._object_type_key_to_index("testing.TestObjectBase")
        assert tindex is not None
        schema = TypeSchema.from_type_index(tindex)
        derived = TestObjectDerived(v_map={"a": 1}, v_array=[1], v_i64=0, v_f64=0.0, v_str="")
        schema.check_value(derived)


# ---------------------------------------------------------------------------
# Category 36: Custom objects in nested containers
# ---------------------------------------------------------------------------
class TestCustomObjectNestedContainers:
    @requires_py39
    def test_array_of_optional_custom_object(self) -> None:
        """Array[Optional[testing.TestIntPair]] with mix of objects and None."""
        data = [TestIntPair(1, 2), None, TestIntPair(3, 4)]
        A(tuple[Optional[TestIntPair], ...]).check_value(data)

    @requires_py39
    def test_map_str_to_array_of_custom_objects(self) -> None:
        """Map[str, Array[testing.TestIntPair]] with nested objects."""
        data = {
            "group1": [TestIntPair(1, 2), TestIntPair(3, 4)],
            "group2": [TestIntPair(5, 6)],
        }
        A(tvm_ffi.Map[str, tuple[TestIntPair, ...]]).check_value(data)

    @requires_py39
    def test_array_of_union_custom_objects(self) -> None:
        """Array[Union[testing.TestIntPair, testing.TestCxxClassBase]]."""
        data = [TestIntPair(1, 2), _TestCxxClassBase(v_i64=1, v_i32=2), TestIntPair(5, 6)]
        A(tuple[Union[TestIntPair, _TestCxxClassBase], ...]).check_value(data)

    @requires_py39
    def test_optional_array_of_custom_objects(self) -> None:
        """Optional[Array[testing.TestIntPair]] with array."""
        data = [TestIntPair(1, 2)]
        A(Optional[tuple[TestIntPair, ...]]).check_value(data)

    @requires_py39
    def test_optional_array_of_custom_objects_none(self) -> None:
        """Optional[Array[testing.TestIntPair]] with None."""
        A(Optional[tuple[TestIntPair, ...]]).check_value(None)

    @requires_py39
    def test_nested_error_with_custom_object(self) -> None:
        """Array[testing.TestIntPair] error message includes type keys."""
        data = [TestIntPair(1, 2), _TestCxxClassBase(v_i64=1, v_i32=2)]
        with pytest.raises(
            TypeError, match=r"element \[1\].*testing.TestIntPair.*testing.TestCxxClassBase"
        ):
            A(tuple[TestIntPair, ...]).check_value(data)

    @requires_py39
    def test_map_nested_error_with_custom_object(self) -> None:
        """Map value error for custom object includes key and type info."""
        data = {"ok": TestIntPair(1, 2), "bad": 42}
        with pytest.raises(
            TypeError, match=r"value for key 'bad'.*expected testing\.TestIntPair.*got int"
        ):
            A(tvm_ffi.Map[str, TestIntPair]).check_value(data)

    @requires_py39
    def test_deep_nested_custom_objects(self) -> None:
        """Map[str, Array[Optional[testing.TestIntPair]]] deep nesting."""
        data = {
            "a": [TestIntPair(1, 2), None],
            "b": [None, TestIntPair(3, 4), TestIntPair(5, 6)],
        }
        A(tvm_ffi.Map[str, tuple[Optional[TestIntPair], ...]]).check_value(data)

    @requires_py39
    def test_deep_nested_custom_objects_error(self) -> None:
        """Map[str, Array[testing.TestIntPair]] error at 3 levels."""
        data = {"k": [TestIntPair(1, 2), "bad"]}
        with pytest.raises(TypeError, match=r"value for key 'k'.*element .1."):
            A(tvm_ffi.Map[str, tuple[TestIntPair, ...]]).check_value(data)

    @requires_py39
    def test_tuple_with_custom_object(self) -> None:
        """tuple[testing.TestIntPair, int, str] with custom object."""
        data = (TestIntPair(1, 2), 42, "hello")
        A(tuple[TestIntPair, int, str]).check_value(data)

    @requires_py39
    def test_tuple_with_custom_object_wrong(self) -> None:
        """tuple[testing.TestIntPair, int] with wrong object in first position."""
        data = (_TestCxxClassBase(v_i64=1, v_i32=2), 42)
        with pytest.raises(TypeError, match=r"element \[0\]"):
            A(tuple[TestIntPair, int]).check_value(data)


# ---------------------------------------------------------------------------
# Category 37: Lowercase Python-native origins ("list", "dict")
# ---------------------------------------------------------------------------
class TestLowercaseOrigins:
    def test_list_origin_accepts_python_list(self) -> None:
        """TypeSchema("list", ...) should validate elements, not passthrough."""
        S("list", S("int")).check_value([1, 2, 3])  # S: lowercase "list" is an internal origin

    def test_list_origin_rejects_bad_elements(self) -> None:
        """TypeSchema("list", (int,)).check_value(["x"]) should fail."""
        with pytest.raises(TypeError, match=r"element \[0\]"):
            S("list", S("int")).check_value(["x"])  # S: lowercase "list" is an internal origin

    def test_list_origin_converts_elements(self) -> None:
        """TypeSchema("list", (float,)).convert([1, True]) does int->float."""
        # S: lowercase "list" is an internal origin
        result = _to_py_class_value(S("list", S("float")).convert([1, True]))
        assert isinstance(result, tvm_ffi.List)
        assert list(result) == [1.0, 1.0]
        assert all(type(x) is float for x in result)

    def test_dict_origin_accepts_python_dict(self) -> None:
        """TypeSchema("dict", ...) should validate key/value types."""
        S("dict", S("str"), S("int")).check_value(
            {"a": 1}
        )  # S: lowercase "dict" is an internal origin

    def test_dict_origin_rejects_bad_values(self) -> None:
        """TypeSchema("dict", (str, int)).check_value({"a": "x"}) should fail."""
        with pytest.raises(TypeError, match="value for key 'a'"):
            S("dict", S("str"), S("int")).check_value(
                {"a": "x"}
            )  # S: lowercase "dict" is an internal origin

    def test_dict_origin_converts_values(self) -> None:
        """TypeSchema("dict", (str, float)).convert({"a": 1}) does int->float."""
        # S: lowercase "dict" is an internal origin
        result = _to_py_class_value(S("dict", S("str"), S("float")).convert({"a": 1, "b": True}))
        assert isinstance(result, tvm_ffi.Dict)
        assert result["a"] == 1.0
        assert result["b"] == 1.0
        assert all(type(v) is float for v in result.values())

    def test_list_origin_no_args_accepts_anything(self) -> None:
        """TypeSchema("list") with no args accepts any list (element type is Any)."""
        S("list").check_value([1, "a", None])  # S: lowercase "list" is an internal origin

    def test_dict_origin_no_args_accepts_anything(self) -> None:
        """TypeSchema("dict") with no args accepts any dict."""
        S("dict").check_value({"a": 1, 2: "b"})  # S: lowercase "dict" is an internal origin

    def test_list_origin_rejects_non_list(self) -> None:
        """TypeSchema("list") rejects non-sequence types."""
        with pytest.raises(TypeError, match="got int"):
            S("list").check_value(42)  # S: lowercase "list" is an internal origin

    def test_dict_origin_rejects_non_dict(self) -> None:
        """TypeSchema("dict") rejects non-dict types."""
        with pytest.raises(TypeError):
            S("dict").check_value([1, 2])  # S: lowercase "dict" is an internal origin


# ---------------------------------------------------------------------------
# Category 38: Cross-type container conversions (Array<->List, Map<->Dict)
# ---------------------------------------------------------------------------
class TestCrossTypeContainers:
    @requires_py39
    def test_array_schema_accepts_ffi_list(self) -> None:
        """Array[int] schema accepts tvm_ffi.List (C++ kOtherTypeIndex)."""
        lst = tvm_ffi.List([1, 2, 3])
        A(tuple[int, ...]).check_value(lst)

    @requires_py39
    def test_list_schema_accepts_ffi_array(self) -> None:
        """List[int] schema accepts tvm_ffi.Array (C++ kOtherTypeIndex)."""
        arr = tvm_ffi.Array([1, 2, 3])
        A(list[int]).check_value(arr)

    @requires_py39
    def test_map_schema_accepts_ffi_dict(self) -> None:
        """Map[str, int] schema accepts tvm_ffi.Dict (C++ kOtherTypeIndex)."""
        d = tvm_ffi.Dict({"a": 1, "b": 2})
        A(tvm_ffi.Map[str, int]).check_value(d)

    @requires_py39
    def test_dict_schema_accepts_ffi_map(self) -> None:
        """Dict[str, int] schema accepts tvm_ffi.Map (C++ kOtherTypeIndex)."""
        m = tvm_ffi.Map({"a": 1, "b": 2})
        A(dict[str, int]).check_value(m)

    @requires_py39
    def test_array_schema_converts_list_elements(self) -> None:
        """Array[float] converts elements from tvm_ffi.List[int]."""
        lst = tvm_ffi.List([1, 2, True])
        result = _to_py_class_value(A(tuple[float, ...]).convert(lst))
        assert list(result) == [1.0, 2.0, 1.0]
        assert all(type(x) is float for x in result)

    @requires_py39
    def test_list_schema_converts_array_elements(self) -> None:
        """List[float] converts elements from tvm_ffi.Array[int]."""
        arr = tvm_ffi.Array([1, 2, True])
        result = _to_py_class_value(A(list[float]).convert(arr))
        assert list(result) == [1.0, 2.0, 1.0]
        assert all(type(x) is float for x in result)

    @requires_py39
    def test_map_schema_converts_dict_values(self) -> None:
        """Map[str, float] converts values from tvm_ffi.Dict."""
        d = tvm_ffi.Dict({"a": 1, "b": True})
        result = _to_py_class_value(A(tvm_ffi.Map[str, float]).convert(d))
        assert result["a"] == 1.0
        assert result["b"] == 1.0

    @requires_py39
    def test_dict_schema_converts_map_values(self) -> None:
        """Dict[str, float] converts values from tvm_ffi.Map."""
        m = tvm_ffi.Map({"a": 1, "b": True})
        result = _to_py_class_value(A(dict[str, float]).convert(m))
        assert result["a"] == 1.0
        assert result["b"] == 1.0

    @requires_py39
    def test_cross_type_still_rejects_wrong_container(self) -> None:
        """Array schema still rejects non-sequence CObjects (e.g. Map)."""
        m = tvm_ffi.Map({"a": 1})
        with pytest.raises(TypeError, match="expected Array"):
            A(tuple[int, ...]).check_value(m)

    @requires_py39
    def test_cross_type_map_rejects_array(self) -> None:
        """Map schema still rejects sequence CObjects (e.g. Array)."""
        arr = tvm_ffi.Array([1, 2])
        with pytest.raises(TypeError, match="expected Map"):
            A(tvm_ffi.Map[str, int]).check_value(arr)


# ---------------------------------------------------------------------------
# Category 39: tuple accepts list and CObject Array
# ---------------------------------------------------------------------------
class TestTupleAcceptsListAndArray:
    @requires_py39
    def test_tuple_accepts_python_list(self) -> None:
        """tuple[int, str] accepts Python list input."""
        result = _to_py_class_value(A(tuple[int, str]).convert([42, "hello"]))
        assert list(result) == [42, "hello"]

    @requires_py39
    def test_tuple_list_with_conversion(self) -> None:
        """tuple[float, int] converts list elements (bool->float, bool->int)."""
        result = _to_py_class_value(A(tuple[float, int]).convert([True, False]))
        assert list(result) == [1.0, 0]
        assert type(result[0]) is float
        assert type(result[1]) is int

    @requires_py39
    def test_tuple_rejects_wrong_length_list(self) -> None:
        """tuple[int, str] rejects list of wrong length."""
        with pytest.raises(TypeError, match="length"):
            A(tuple[int, str]).check_value([1, "a", "b"])

    @requires_py39
    def test_tuple_accepts_ffi_array(self) -> None:
        """tuple[int, int] accepts tvm_ffi.Array (C++ Tuple accepts kTVMFFIArray)."""
        arr = tvm_ffi.Array([1, 2])
        A(tuple[int, int]).check_value(arr)

    @requires_py39
    def test_tuple_ffi_array_with_conversion(self) -> None:
        """tuple[float, float] converts tvm_ffi.Array elements."""
        arr = tvm_ffi.Array([1, True])
        result = _to_py_class_value(A(tuple[float, float]).convert(arr))
        assert list(result) == [1.0, 1.0]
        assert all(type(x) is float for x in result)

    @requires_py39
    def test_tuple_ffi_array_wrong_length(self) -> None:
        """tuple[int, int] rejects tvm_ffi.Array of wrong length."""
        arr = tvm_ffi.Array([1, 2, 3])
        with pytest.raises(TypeError, match="length"):
            A(tuple[int, int]).check_value(arr)

    @requires_py39
    def test_tuple_rejects_ffi_map(self) -> None:
        """Tuple schema rejects Map CObject."""
        m = tvm_ffi.Map({"a": 1})
        with pytest.raises(TypeError, match="expected tuple"):
            A(tuple[int]).check_value(m)

    def test_untyped_tuple_accepts_list(self) -> None:
        """Tuple (no args) accepts any list as-is."""
        # Untyped tuple has tuple_len=0, so it just checks the container type
        # but doesn't validate elements
        A(tuple).check_value([1, "a", None])

    def test_untyped_tuple_accepts_ffi_array(self) -> None:
        """Tuple (no args) accepts tvm_ffi.Array as-is."""
        arr = tvm_ffi.Array([1, 2, 3])
        A(tuple).check_value(arr)

    def test_typed_empty_tuple_rejects_non_empty_list(self) -> None:
        """Explicit empty tuple schema enforces length 0."""
        schema = TypeSchema("tuple", ())
        with pytest.raises(TypeError, match="length 0"):
            schema.check_value([1])

    def test_untyped_tuple_converts_ffi_list_to_array(self) -> None:
        """Tuple (no args) normalizes tvm_ffi.List input to tvm_ffi.Array."""
        lst = tvm_ffi.List([1, 2, 3])
        result = _to_py_class_value(A(tuple).convert(lst))
        assert isinstance(result, tvm_ffi.Array)
        assert list(result) == [1, 2, 3]
        assert not result.same_as(lst)


# ---------------------------------------------------------------------------
# Category 40: dtype string parse errors
# ---------------------------------------------------------------------------
class TestDtypeParseErrors:
    def test_check_value_bad_dtype_raises_error(self) -> None:
        """check_value should raise TypeError for invalid dtype."""
        with pytest.raises(TypeError, match="dtype"):
            A(tvm_ffi.core.DataType).check_value("not_a_valid_dtype_xyz")

    def test_convert_bad_dtype_raises_type_error_2(self) -> None:
        """Convert should raise TypeError for invalid dtype string."""
        with pytest.raises(TypeError, match="dtype"):
            A(tvm_ffi.core.DataType).convert("not_a_valid_dtype_xyz")

    def test_convert_bad_dtype_raises_type_error(self) -> None:
        """Convert should raise TypeError for invalid dtype string."""
        with pytest.raises(TypeError, match="dtype"):
            A(tvm_ffi.core.DataType).convert("not_a_valid_dtype_xyz")

    def test_valid_dtype_string_still_works(self) -> None:
        """Valid dtype strings should still convert successfully."""
        result = _to_py_class_value(A(tvm_ffi.core.DataType).convert("float32"))
        assert str(result) == "float32"

    def test_convert_valid_dtype(self) -> None:
        """Convert with valid dtype returns DataType."""
        result = _to_py_class_value(A(tvm_ffi.core.DataType).convert("int8"))
        assert str(result) == "int8"


# ---------------------------------------------------------------------------
# Category 41: int64 boundary checking
# ---------------------------------------------------------------------------
class TestInt64Boundaries:
    """Verify int converter rejects values outside int64 range.

    The FFI marshals Python int to C++ int64_t. Values outside
    [-2^63, 2^63-1] would silently overflow at marshal time, so
    the converter rejects them early.
    """

    def test_int64_max_accepted(self) -> None:
        """2^63-1 (INT64_MAX) is the largest valid int."""
        A(int).check_value(2**63 - 1)

    def test_int64_min_accepted(self) -> None:
        """-2^63 (INT64_MIN) is the smallest valid int."""
        A(int).check_value(-(2**63))

    def test_int64_max_plus_one_rejected(self) -> None:
        """2^63 exceeds int64 range."""
        with pytest.raises(TypeError, match="int64 range"):
            A(int).check_value(2**63)

    def test_int64_min_minus_one_rejected(self) -> None:
        """-2^63-1 exceeds int64 range."""
        with pytest.raises(TypeError, match="int64 range"):
            A(int).check_value(-(2**63) - 1)

    def test_very_large_positive_rejected(self) -> None:
        """Very large positive integer rejected."""
        with pytest.raises(TypeError, match="int64 range"):
            A(int).check_value(10**100)

    def test_very_large_negative_rejected(self) -> None:
        """Very large negative integer rejected."""
        with pytest.raises(TypeError, match="int64 range"):
            A(int).check_value(-(10**100))

    def test_convert_raises_type_error_for_overflow(self) -> None:
        """Convert raises TypeError for overflow."""
        with pytest.raises(TypeError, match="int64 range"):
            A(int).convert(2**63)

    def test_bool_to_int_no_range_issue(self) -> None:
        """Bool -> int conversion (0 or 1) always fits."""
        assert _to_py_class_value(A(int).convert(True)) == 1
        assert _to_py_class_value(A(int).convert(False)) == 0

    def test_int64_boundaries_in_float_conversion(self) -> None:
        """Float schema accepts large ints (float64 has wider range)."""
        # float64 can represent integers up to 2^53 exactly,
        # and larger values with precision loss (but no range error)
        A(float).check_value(2**63)
        A(float).check_value(-(2**63))

    def test_int64_overflow_in_optional_int(self) -> None:
        """Optional[int] propagates int64 range check."""
        with pytest.raises(TypeError, match="int64 range"):
            A(Optional[int]).check_value(2**63)

    @requires_py39
    def test_int64_overflow_in_array_element(self) -> None:
        """Array[int] element overflow is caught with path."""
        with pytest.raises(TypeError, match="int64 range"):
            A(tuple[int, ...]).check_value([1, 2**63, 3])


# ---------------------------------------------------------------------------
# Category 42: Unknown origin errors (lazy converter construction)
# ---------------------------------------------------------------------------
class TestUnknownOriginErrors:
    """Converter is built lazily via cached_property. Unknown origins
    construct fine but raise TypeError on first convert/check_value.
    """

    def test_unknown_origin_constructs_ok(self) -> None:
        """TypeSchema with unknown origin can be constructed."""
        schema = S("not_a_real_type")  # S: intentionally invalid origin
        assert schema.origin == "not_a_real_type"

    def test_unknown_origin_errors_on_check_value(self) -> None:
        """Unknown origin raises TypeError on check_value."""
        schema = S("not_a_real_type")  # S: intentionally invalid origin
        with pytest.raises(TypeError, match="unknown TypeSchema origin"):
            schema.check_value(42)

    def test_unknown_origin_errors_on_convert(self) -> None:
        """Unknown origin raises TypeError on convert."""
        schema = S("not_a_real_type")  # S: intentionally invalid origin
        with pytest.raises(TypeError, match="unknown TypeSchema origin"):
            schema.convert(42)

    def test_unknown_origin_errors_on_convert_2(self) -> None:
        """Unknown origin raises TypeError on convert (duplicate check)."""
        schema = S("not_a_real_type")  # S: intentionally invalid origin
        with pytest.raises(TypeError, match="unknown TypeSchema origin"):
            schema.convert(42)

    def test_unknown_origin_errors_on_check_value_2(self) -> None:
        """Unknown origin raises TypeError on check_value (duplicate check)."""
        schema = S("not_a_real_type")  # S: intentionally invalid origin
        with pytest.raises(TypeError, match="unknown TypeSchema origin"):
            schema.check_value(42)

    def test_typo_origin_errors(self) -> None:
        """Common typos are caught, not silently passed through."""
        for typo in ("innt", "floot", "strr", "Int", "Float"):
            schema = S(typo)  # S: intentionally invalid origin
            with pytest.raises(TypeError, match="unknown TypeSchema origin"):
                schema.check_value(42)

    def test_unknown_nested_in_optional_errors(self) -> None:
        """Unknown origin nested inside Optional errors on use."""
        schema = S("Optional", S("not_a_real_type"))  # S: intentionally invalid origin
        with pytest.raises(TypeError, match="unknown TypeSchema origin"):
            schema.check_value(42)


# ---------------------------------------------------------------------------
# Category 43: convert/check_value raise TypeError on errors
# ---------------------------------------------------------------------------
class TestConvertCheckValueErrors:
    """Verify convert and check_value raise TypeError on errors."""

    def test_convert_catches_custom_integral_error(self) -> None:
        """Custom Integral whose __int__ raises is caught by convert."""

        class BadInt:
            """Registered as Integral via ABC but __int__ raises."""

            def __int__(self) -> int:
                raise RuntimeError("broken __int__")

        Integral.register(BadInt)
        with pytest.raises(TypeError, match="broken __int__"):
            A(int).convert(BadInt())

    def test_check_value_catches_custom_integral_error(self) -> None:
        """Custom Integral whose __int__ raises is caught by check_value."""

        class BadInt2:
            def __int__(self) -> int:
                raise ValueError("bad int conversion")

        Integral.register(BadInt2)
        with pytest.raises(TypeError, match="bad int conversion"):
            A(int).check_value(BadInt2())

    def test_convert_unknown_origin_raises(self) -> None:
        """Convert with unknown origin raises TypeError."""
        with pytest.raises(TypeError, match="unknown TypeSchema origin"):
            S("bogus_type").convert("anything")  # S: intentionally invalid origin

    def test_check_value_unknown_origin_raises(self) -> None:
        """check_value with unknown origin raises TypeError."""
        with pytest.raises(TypeError, match="unknown TypeSchema origin"):
            S("bogus_type").check_value("anything")  # S: intentionally invalid origin


# ---------------------------------------------------------------------------
# Category 44: Schema arity validation (ValueError, not assert)
# ---------------------------------------------------------------------------
class TestSchemaArityValidation:
    """Verify arity checks use ValueError (not assert) so they work under -O."""

    def test_union_too_few_args(self) -> None:
        """Union with < 2 args raises ValueError."""
        with pytest.raises(ValueError, match="at least two"):
            S("Union", A(int))

    def test_optional_wrong_arity(self) -> None:
        """Optional with != 1 arg raises ValueError."""
        with pytest.raises(ValueError, match="exactly one"):
            S("Optional")
        with pytest.raises(ValueError, match="exactly one"):
            S("Optional", A(int), A(str))

    def test_array_too_many_args(self) -> None:
        """Array with > 1 arg raises ValueError."""
        with pytest.raises(ValueError, match="0 or 1"):
            S("Array", A(int), A(str))

    def test_list_too_many_args(self) -> None:
        """List with > 1 arg raises ValueError."""
        with pytest.raises(ValueError, match="0 or 1"):
            S("List", A(int), A(str))

    def test_map_wrong_arity(self) -> None:
        """Map with 1 or 3 args raises ValueError."""
        with pytest.raises(ValueError, match="0 or 2"):
            S("Map", A(str))
        with pytest.raises(ValueError, match="0 or 2"):
            S("Map", A(str), A(int), A(float))

    def test_dict_wrong_arity(self) -> None:
        """Dict with 1 or 3 args raises ValueError."""
        with pytest.raises(ValueError, match="0 or 2"):
            S("Dict", A(str))

    def test_lowercase_list_too_many_args(self) -> None:
        """Lowercase 'list' with > 1 arg raises ValueError."""
        with pytest.raises(ValueError, match="0 or 1"):
            S("list", A(int), A(str))  # S: lowercase "list" is an internal origin

    def test_lowercase_dict_wrong_arity(self) -> None:
        """Lowercase 'dict' with 1 arg raises ValueError."""
        with pytest.raises(ValueError, match="0 or 2"):
            S("dict", A(str))  # S: lowercase "dict" is an internal origin


# ---------------------------------------------------------------------------
# Category 45: from_type_index edge cases
# ---------------------------------------------------------------------------
class TestFromTypeIndexEdgeCases:
    """Verify from_type_index behavior for valid indices.

    Note: Unregistered type indices trigger a fatal C++ assertion
    (TVMFFIGetTypeInfo CHECK failure) that cannot be caught from Python.
    Only valid indices obtained from the type registry should be passed.
    """

    def test_valid_pod_index_roundtrip(self) -> None:
        """POD type_index from TypeSchema.origin_type_index round-trips."""
        int_schema = A(int)
        schema = TypeSchema.from_type_index(int_schema.origin_type_index)
        assert schema.origin == "int"
        schema.check_value(42)

    def test_valid_object_index_works(self) -> None:
        """Valid registered object type_index constructs fine."""
        tindex = tvm_ffi.core._object_type_key_to_index("testing.TestIntPair")
        assert tindex is not None
        schema = TypeSchema.from_type_index(tindex)
        assert schema.origin == "testing.TestIntPair"

    def test_from_type_index_with_args(self) -> None:
        """from_type_index with type arguments creates parameterized schema."""
        arr_schema = A(tvm_ffi.Array)
        schema = TypeSchema.from_type_index(arr_schema.origin_type_index, (A(int),))
        assert schema.origin == "Array"
        schema.check_value([1, 2, 3])


# ===========================================================================
# Protocol-based conversion tests (matching Python FFI marshal path)
# ===========================================================================


# ---------------------------------------------------------------------------
# Category 46: __tvm_ffi_int__ protocol
# ---------------------------------------------------------------------------
class TestIntProtocol:
    """int schema accepts values with __tvm_ffi_int__ protocol."""

    def test_int_protocol_accepted(self) -> None:
        """Object with __tvm_ffi_int__ passes int schema."""

        class IntProto:
            def __tvm_ffi_int__(self) -> int:
                return 42

        A(int).check_value(IntProto())

    def test_int_protocol_check_value(self) -> None:
        """check_value succeeds for __tvm_ffi_int__ value."""

        class IntProto:
            def __tvm_ffi_int__(self) -> int:
                return 10

        A(int).check_value(IntProto())

    def test_int_protocol_convert_returns_value(self) -> None:
        """Convert returns the protocol value as-is (marshal handles conversion)."""

        class IntProto:
            def __tvm_ffi_int__(self) -> int:
                return 99

        obj = IntProto()
        result = _to_py_class_value(A(int).convert(obj))
        assert result is not None

    def test_without_protocol_still_rejected(self) -> None:
        """Object without __tvm_ffi_int__ is still rejected by int schema."""

        class NoProto:
            pass

        with pytest.raises(TypeError, match="expected int"):
            A(int).check_value(NoProto())


# ---------------------------------------------------------------------------
# Category 47: __tvm_ffi_float__ protocol
# ---------------------------------------------------------------------------
class TestFloatProtocol:
    """float schema accepts values with __tvm_ffi_float__ protocol."""

    def test_float_protocol_accepted(self) -> None:
        """Object with __tvm_ffi_float__ passes float schema."""

        class FloatProto:
            def __tvm_ffi_float__(self) -> float:
                return 3.14

        A(float).check_value(FloatProto())

    def test_float_protocol_convert(self) -> None:
        """Convert returns protocol value as-is."""

        class FloatProto:
            def __tvm_ffi_float__(self) -> float:
                return 2.0

        obj = FloatProto()
        result = _to_py_class_value(A(float).convert(obj))
        assert result is not None

    def test_without_protocol_still_rejected(self) -> None:
        """Object without __tvm_ffi_float__ is still rejected."""

        class NoProto:
            pass

        with pytest.raises(TypeError, match="expected float"):
            A(float).check_value(NoProto())


# ---------------------------------------------------------------------------
# Category 48: __tvm_ffi_opaque_ptr__ protocol
# ---------------------------------------------------------------------------
class TestOpaquePtrProtocol:
    """ctypes.c_void_p schema accepts __tvm_ffi_opaque_ptr__ protocol."""

    def test_opaque_ptr_protocol_accepted(self) -> None:
        """Object with __tvm_ffi_opaque_ptr__ passes ctypes.c_void_p schema."""

        class PtrProto:
            def __tvm_ffi_opaque_ptr__(self) -> int:
                return 0xDEAD

        A(ctypes.c_void_p).check_value(PtrProto())

    def test_opaque_ptr_protocol_convert(self) -> None:
        """Convert returns protocol value as-is."""

        class PtrProto:
            def __tvm_ffi_opaque_ptr__(self) -> int:
                return 0

        obj = PtrProto()
        result = _to_py_class_value(A(ctypes.c_void_p).convert(obj))
        assert result is not None


# ---------------------------------------------------------------------------
# Category 49: __dlpack_device__ protocol
# ---------------------------------------------------------------------------
class TestDeviceProtocol:
    """Device schema accepts __dlpack_device__ protocol."""

    def test_dlpack_device_protocol_accepted(self) -> None:
        """Object with __dlpack_device__ passes Device schema."""

        class DevProto:
            def __dlpack_device__(self) -> tuple[int, int]:
                return (1, 0)

        A(tvm_ffi.Device).check_value(DevProto())

    def test_dlpack_device_protocol_convert(self) -> None:
        """Convert returns protocol value as-is."""

        class DevProto:
            def __dlpack_device__(self) -> tuple[int, int]:
                return (2, 1)

        obj = DevProto()
        result = _to_py_class_value(A(tvm_ffi.Device).convert(obj))
        assert result is not None

    def test_without_protocol_still_rejected(self) -> None:
        """Object without __dlpack_device__ is still rejected."""

        class NoProto:
            pass

        with pytest.raises(TypeError, match="expected Device"):
            A(tvm_ffi.Device).check_value(NoProto())


# ---------------------------------------------------------------------------
# Category 50: dtype protocols (torch.dtype, numpy.dtype, __dlpack_data_type__)
# ---------------------------------------------------------------------------
class TestDtypeProtocols:
    """dtype schema accepts torch.dtype, numpy.dtype, __dlpack_data_type__."""

    def test_dlpack_data_type_protocol_accepted(self) -> None:
        """Object with __dlpack_data_type__ passes dtype schema."""

        class DTypeProto:
            def __dlpack_data_type__(self) -> tuple[int, int, int]:
                return (2, 32, 1)  # float32

        A(tvm_ffi.core.DataType).check_value(DTypeProto())

    def test_dlpack_data_type_protocol_convert(self) -> None:
        """Convert returns protocol value as-is."""

        class DTypeProto:
            def __dlpack_data_type__(self) -> tuple[int, int, int]:
                return (0, 32, 1)

        obj = DTypeProto()
        result = _to_py_class_value(A(tvm_ffi.core.DataType).convert(obj))
        assert result is not None

    def test_numpy_dtype_accepted(self) -> None:
        """numpy.dtype passes dtype schema (if numpy installed)."""
        numpy = pytest.importorskip("numpy")
        A(tvm_ffi.core.DataType).check_value(numpy.dtype("float32"))

    def test_numpy_dtype_convert(self) -> None:
        """Convert returns numpy.dtype as-is."""
        numpy = pytest.importorskip("numpy")
        dt = numpy.dtype("int32")
        result = _to_py_class_value(A(tvm_ffi.core.DataType).convert(dt))
        assert result is not None

    def test_torch_dtype_accepted(self) -> None:
        """torch.dtype passes dtype schema (if torch installed)."""
        torch = pytest.importorskip("torch")
        A(tvm_ffi.core.DataType).check_value(torch.float32)

    def test_torch_dtype_convert(self) -> None:
        """Convert returns torch.dtype as-is."""
        torch = pytest.importorskip("torch")
        dt = torch.int64
        result = _to_py_class_value(A(tvm_ffi.core.DataType).convert(dt))
        assert result is not None


# ---------------------------------------------------------------------------
# Category 51: __dlpack_c_exchange_api__ protocol (Tensor)
# ---------------------------------------------------------------------------
class TestTensorProtocol:
    """Tensor schema accepts __dlpack_c_exchange_api__ protocol."""

    def test_dlpack_c_exchange_api_accepted(self) -> None:
        """Object with a valid __dlpack_c_exchange_api__ passes Tensor schema."""
        np = pytest.importorskip("numpy")
        tensor = tvm_ffi.from_dlpack(np.arange(4, dtype="int32"))
        wrapper = tvm_ffi.core.DLTensorTestWrapper(tensor)
        A(tvm_ffi.Tensor).check_value(wrapper)

    def test_dlpack_c_exchange_api_convert(self) -> None:
        """Valid exchange-api wrappers can be converted to Tensor."""
        np = pytest.importorskip("numpy")
        tensor = tvm_ffi.from_dlpack(np.arange(4, dtype="int32"))
        wrapper = tvm_ffi.core.DLTensorTestWrapper(tensor)
        result = _to_py_class_value(A(tvm_ffi.Tensor).convert(wrapper))
        assert isinstance(result, tvm_ffi.Tensor)

    def test_dlpack_still_accepted(self) -> None:
        """Object with __dlpack__ still accepted (existing behavior)."""
        np = pytest.importorskip("numpy")
        A(tvm_ffi.Tensor).check_value(np.arange(4, dtype="int32"))


# ---------------------------------------------------------------------------
# Category 52: __tvm_ffi_object__ protocol
# ---------------------------------------------------------------------------
class TestObjectProtocol:
    """Object schemas accept __tvm_ffi_object__ protocol."""

    def test_object_protocol_generic_object(self) -> None:
        """__tvm_ffi_object__ returning a CObject passes generic Object schema."""
        inner = TestIntPair(1, 2)

        class ObjProto:
            def __tvm_ffi_object__(self) -> object:
                return inner

        A(tvm_ffi.core.Object).check_value(ObjProto())

    def test_object_protocol_specific_type(self) -> None:
        """__tvm_ffi_object__ returning TestIntPair passes TestIntPair schema."""
        inner = TestIntPair(3, 4)

        class ObjProto:
            def __tvm_ffi_object__(self) -> object:
                return inner

        A(TestIntPair).check_value(ObjProto())

    def test_object_protocol_convert_returns_cobject(self) -> None:
        """Convert returns the CObject from __tvm_ffi_object__()."""
        inner = TestIntPair(5, 6)

        class ObjProto:
            def __tvm_ffi_object__(self) -> object:
                return inner

        result = _to_py_class_value(A(TestIntPair).convert(ObjProto()))
        assert result.same_as(inner)

    def test_object_protocol_wrong_type_rejected(self) -> None:
        """__tvm_ffi_object__ returning wrong type is rejected."""
        inner = TestIntPair(1, 2)

        class ObjProto:
            def __tvm_ffi_object__(self) -> object:
                return inner

        with pytest.raises(
            TypeError, match=r"expected testing\.TestCxxClassBase, got testing\.TestIntPair"
        ):
            A(_TestCxxClassBase).check_value(ObjProto())

    def test_object_protocol_raises_caught(self) -> None:
        """__tvm_ffi_object__ that raises produces _ConvertError."""

        class BadProto:
            def __tvm_ffi_object__(self) -> object:
                raise RuntimeError("broken")

        with pytest.raises(TypeError, match=r"__tvm_ffi_object__\(\) failed"):
            A(tvm_ffi.core.Object).check_value(BadProto())

    def test_object_protocol_hierarchy(self) -> None:
        """__tvm_ffi_object__ returning derived passes base schema."""
        derived = _TestCxxClassDerived(v_i64=1, v_i32=2, v_f64=3.0)

        class ObjProto:
            def __tvm_ffi_object__(self) -> object:
                return derived

        A(_TestCxxClassBase).check_value(ObjProto())


# ---------------------------------------------------------------------------
# Category 53: ObjectConvertible protocol
# ---------------------------------------------------------------------------
class TestObjectConvertibleProtocol:
    """Object schemas accept ObjectConvertible subclass."""

    def test_object_convertible_accepted(self) -> None:
        """ObjectConvertible with asobject() passes Object schema."""
        inner = TestIntPair(10, 20)

        class MyConvertible(ObjectConvertible):
            def asobject(self) -> tvm_ffi.core.Object:
                return inner

        A(tvm_ffi.core.Object).check_value(MyConvertible())

    def test_object_convertible_specific_type(self) -> None:
        """ObjectConvertible passes specific type schema."""
        inner = TestIntPair(1, 2)

        class MyConvertible(ObjectConvertible):
            def asobject(self) -> tvm_ffi.core.Object:
                return inner

        A(TestIntPair).check_value(MyConvertible())

    def test_object_convertible_convert_returns_cobject(self) -> None:
        """Convert returns the CObject from asobject()."""
        inner = TestIntPair(7, 8)

        class MyConvertible(ObjectConvertible):
            def asobject(self) -> tvm_ffi.core.Object:
                return inner

        result = _to_py_class_value(A(TestIntPair).convert(MyConvertible()))
        assert result.same_as(inner)

    def test_object_convertible_in_union(self) -> None:
        """Union dispatch unwraps ObjectConvertible before trying alternatives."""
        inner = TestIntPair(9, 10)

        class MyConvertible(ObjectConvertible):
            def asobject(self) -> tvm_ffi.core.Object:
                return inner

        result = _to_py_class_value(A(Union[TestIntPair, int]).convert(MyConvertible()))
        assert result.same_as(inner)

    def test_object_convertible_wrong_type(self) -> None:
        """ObjectConvertible returning wrong type is rejected."""
        inner = TestIntPair(1, 2)

        class MyConvertible(ObjectConvertible):
            def asobject(self) -> tvm_ffi.core.Object:
                return inner

        with pytest.raises(
            TypeError,
            match=r"type check failed for testing\.TestCxxClassBase: expected testing\.TestCxxClassBase, got testing\.TestIntPair",
        ):
            A(_TestCxxClassBase).check_value(MyConvertible())

    def test_object_convertible_raises_caught(self) -> None:
        """asobject() that raises produces error, not exception."""

        class BadConvertible(ObjectConvertible):
            def asobject(self) -> tvm_ffi.core.Object:
                raise RuntimeError("broken asobject")

        with pytest.raises(TypeError, match=r"asobject\(\) failed"):
            A(tvm_ffi.core.Object).check_value(BadConvertible())


# ---------------------------------------------------------------------------
# Category 54: __tvm_ffi_value__ protocol (recursive fallback)
# ---------------------------------------------------------------------------
class TestValueProtocol:
    """__tvm_ffi_value__ provides recursive conversion fallback."""

    def test_value_protocol_int(self) -> None:
        """__tvm_ffi_value__ returning int passes int schema."""

        class ValProto:
            def __tvm_ffi_value__(self) -> object:
                return 42

        A(int).check_value(ValProto())

    def test_value_protocol_float(self) -> None:
        """__tvm_ffi_value__ returning float passes float schema."""

        class ValProto:
            def __tvm_ffi_value__(self) -> object:
                return 3.14

        A(float).check_value(ValProto())

    def test_value_protocol_convert(self) -> None:
        """Convert returns the unwrapped value from __tvm_ffi_value__."""

        class ValProto:
            def __tvm_ffi_value__(self) -> object:
                return 42

        result = _to_py_class_value(A(int).convert(ValProto()))
        assert result == 42

    def test_value_protocol_nested(self) -> None:
        """Nested __tvm_ffi_value__ is recursively unwrapped."""

        class ValProto:
            def __init__(self, v: object) -> None:
                self.v = v

            def __tvm_ffi_value__(self) -> object:
                return self.v

        # ValProto(ValProto(ValProto(10))) should unwrap to 10
        wrapped = ValProto(ValProto(ValProto(10)))
        assert _to_py_class_value(A(int).convert(wrapped)) == 10

    def test_value_protocol_object(self) -> None:
        """__tvm_ffi_value__ returning a CObject passes object schema."""
        inner = TestIntPair(1, 2)

        class ValProto:
            def __tvm_ffi_value__(self) -> object:
                return inner

        A(TestIntPair).check_value(ValProto())

    def test_value_protocol_still_fails_on_mismatch(self) -> None:
        """__tvm_ffi_value__ returning wrong type still fails."""

        class ValProto:
            def __tvm_ffi_value__(self) -> object:
                return "not_an_int"

        with pytest.raises(TypeError, match="expected int"):
            A(int).check_value(ValProto())

    def test_value_protocol_raises_uses_original_error(self) -> None:
        """If __tvm_ffi_value__ raises, the original error is returned."""

        class BadValProto:
            def __tvm_ffi_value__(self) -> object:
                raise RuntimeError("broken")

        with pytest.raises(TypeError, match="expected int"):
            A(int).check_value(BadValProto())

    def test_nested_optional_value_protocol_stall(self) -> None:
        """Optional[Optional[float]] reports the unwrapped target type."""

        class SelfRef:
            def __tvm_ffi_value__(self) -> object:
                return self

        with pytest.raises(TypeError, match="expected float"):
            A(Optional[Optional[float]]).check_value(SelfRef())

    def test_value_protocol_eventually_resolves(self) -> None:
        """Short __tvm_ffi_value__ chains still resolve successfully."""

        class ChainStep:
            def __init__(self, remaining: int) -> None:
                self.remaining = remaining

            def __tvm_ffi_value__(self) -> object:
                if self.remaining > 0:
                    return ChainStep(self.remaining - 1)
                return 42

        assert _to_py_class_value(A(int).convert(ChainStep(5))) == 42


# ---------------------------------------------------------------------------
# Category 55: Protocol values in containers
# ---------------------------------------------------------------------------
class TestProtocolsInContainers:
    """Protocol-accepting values work inside containers and composites."""

    @requires_py39
    def test_int_protocol_in_array(self) -> None:
        """Array[int] accepts elements with __tvm_ffi_int__."""

        class IntProto:
            def __tvm_ffi_int__(self) -> int:
                return 1

        A(tuple[int, ...]).check_value([1, IntProto(), 3])

    def test_float_protocol_in_optional(self) -> None:
        """Optional[float] accepts __tvm_ffi_float__ value."""

        class FloatProto:
            def __tvm_ffi_float__(self) -> float:
                return 1.0

        A(Optional[float]).check_value(FloatProto())
        A(Optional[float]).check_value(None)

    def test_object_protocol_in_union(self) -> None:
        """Union[testing.TestIntPair, int] accepts __tvm_ffi_object__ value."""
        inner = TestIntPair(1, 2)

        class ObjProto:
            def __tvm_ffi_object__(self) -> object:
                return inner

        A(Union[TestIntPair, int]).check_value(ObjProto())

    @requires_py39
    def test_value_protocol_in_array(self) -> None:
        """Array[int] elements use __tvm_ffi_value__ fallback (recursive)."""

        class ValProto:
            def __tvm_ffi_value__(self) -> object:
                return 42

        # __tvm_ffi_value__ fallback is applied recursively at every level,
        # matching the marshal path where TVMFFIPyArgSetterFactory_ is
        # called per-element.
        A(tuple[int, ...]).check_value([ValProto()])

    @requires_py39
    def test_object_convertible_in_array(self) -> None:
        """Array[Object] elements unwrap ObjectConvertible before element dispatch."""
        inner = TestIntPair(3, 4)

        class Convertible(ObjectConvertible):
            def asobject(self) -> tvm_ffi.core.Object:
                return inner

        result = _to_py_class_value(A(tuple[tvm_ffi.core.Object, ...]).convert([Convertible()]))
        assert result[0].same_as(inner)

    @requires_py39
    def test_device_protocol_in_map_value(self) -> None:
        """Map[str, Device] accepts __dlpack_device__ values."""

        class DevProto:
            def __dlpack_device__(self) -> tuple[int, int]:
                return (1, 0)

        A(tvm_ffi.Map[str, tvm_ffi.Device]).check_value({"gpu": DevProto()})


# ---------------------------------------------------------------------------
# Category 56: Nested __tvm_ffi_value__ in containers (recursive fallback)
# ---------------------------------------------------------------------------
class TestNestedValueProtocol:
    """__tvm_ffi_value__ fallback works recursively inside containers."""

    @requires_py39
    def test_value_in_array_elements(self) -> None:
        """Array[int] elements with __tvm_ffi_value__ are accepted."""

        class VP:
            def __tvm_ffi_value__(self) -> object:
                return 42

        A(tuple[int, ...]).check_value([1, VP(), 3])

    @requires_py39
    def test_value_in_map_values(self) -> None:
        """Map[str, int] values with __tvm_ffi_value__ are accepted."""

        class VP:
            def __tvm_ffi_value__(self) -> object:
                return 99

        A(tvm_ffi.Map[str, int]).check_value({"a": VP()})

    @requires_py39
    def test_value_in_map_keys(self) -> None:
        """Map[str, int] keys with __tvm_ffi_value__ are accepted."""

        class VP:
            def __tvm_ffi_value__(self) -> object:
                return "key"

        A(tvm_ffi.Map[str, int]).check_value({VP(): 1})

    @requires_py39
    def test_value_in_tuple_positions(self) -> None:
        """tuple[int, str] positions with __tvm_ffi_value__ are accepted."""

        class IntVP:
            def __tvm_ffi_value__(self) -> object:
                return 42

        class StrVP:
            def __tvm_ffi_value__(self) -> object:
                return "hello"

        A(tuple[int, str]).check_value((IntVP(), StrVP()))

    def test_value_in_optional_inner(self) -> None:
        """Optional[int] inner with __tvm_ffi_value__ is accepted."""

        class VP:
            def __tvm_ffi_value__(self) -> object:
                return 42

        A(Optional[int]).check_value(VP())

    def test_value_in_union_alternatives(self) -> None:
        """Union[int, str] with __tvm_ffi_value__ is accepted."""

        class VP:
            def __tvm_ffi_value__(self) -> object:
                return "hello"

        A(Union[int, str]).check_value(VP())

    @requires_py39
    def test_multi_hop_value_in_container(self) -> None:
        """Nested __tvm_ffi_value__ unwrapping inside containers."""

        class VP:
            def __init__(self, v: object) -> None:
                self.v = v

            def __tvm_ffi_value__(self) -> object:
                return self.v

        A(tuple[int, ...]).check_value([VP(VP(10))])

    @requires_py39
    def test_value_convert_in_array(self) -> None:
        """Convert returns unwrapped values in container."""

        class VP:
            def __tvm_ffi_value__(self) -> object:
                return 42

        result = _to_py_class_value(A(tuple[int, ...]).convert([VP()]))
        assert list(result) == [42]


# ---------------------------------------------------------------------------
# Category 57: __tvm_ffi_value__ cycle protection
# ---------------------------------------------------------------------------
class TestValueProtocolCycles:
    """Cycle protection in __tvm_ffi_value__ fallback."""

    def test_self_cycle_returns_error(self) -> None:
        """__tvm_ffi_value__() returning self doesn't infinite-loop."""

        class SelfCycle:
            def __tvm_ffi_value__(self) -> object:
                return self

        with pytest.raises(TypeError, match="expected int"):
            A(int).check_value(SelfCycle())

    def test_any_self_cycle_returns_original_error(self) -> None:
        """Any also routes __tvm_ffi_value__ through the bounded fallback loop."""
        call_count = 0

        class SelfCycle:
            def __tvm_ffi_value__(self) -> object:
                nonlocal call_count
                call_count += 1
                return self

        with pytest.raises(TypeError, match=r"failed to convert Any from .*SelfCycle"):
            A(typing.Any).convert(SelfCycle())
        assert call_count == 1

    def test_mutual_cycle_bounded(self) -> None:
        """Mutual cycle is bounded by explicit depth limit."""

        class A:
            def __init__(self) -> None:
                self.other: object = None

            def __tvm_ffi_value__(self) -> object:
                return self.other

        class B:
            def __init__(self) -> None:
                self.other: object = None

            def __tvm_ffi_value__(self) -> object:
                return self.other

        a, b = A(), B()
        a.other = b
        b.other = a

        # Should not hang — bounded by depth limit in the fallback loop
        with pytest.raises(TypeError, match="cycle"):
            S("int").check_value(a)

    def test_any_mutual_cycle_bounded(self) -> None:
        """Any reports a bounded cycle instead of recursing in raw CAny packing."""

        class Left:
            def __init__(self) -> None:
                self.other: object = None

            def __tvm_ffi_value__(self) -> object:
                return self.other

        class Right:
            def __init__(self) -> None:
                self.other: object = None

            def __tvm_ffi_value__(self) -> object:
                return self.other

        a, b = Left(), Right()
        a.other = b
        b.other = a

        with pytest.raises(TypeError, match="cycle"):
            A(typing.Any).convert(a)

    def test_value_protocol_deep_chain_hits_cycle_limit(self) -> None:
        """Long __tvm_ffi_value__ chains trip the explicit depth guard."""

        class DeepChain:
            def __init__(self, depth: int) -> None:
                self.depth = depth

            def __tvm_ffi_value__(self) -> object:
                return DeepChain(self.depth + 1) if self.depth < 100 else 42

        with pytest.raises(TypeError, match="cycle"):
            A(int).check_value(DeepChain(0))


# ---------------------------------------------------------------------------
# Category 58: Object marshal fallback
# ---------------------------------------------------------------------------
class TestObjectConvertAttrRegistration:
    """Object targets register __ffi_convert__ consistently."""

    def test_core_object_types_register_convert_attr(self) -> None:
        """Core object types register ``__ffi_convert__``."""
        for type_key in (
            "ffi.Object",
            "ffi.Function",
            "ffi.Error",
            "ffi.String",
            "ffi.Bytes",
            "ffi.Array",
            "ffi.List",
            "ffi.Map",
            "ffi.Dict",
            "ffi.Shape",
            "ffi.Tensor",
        ):
            type_index = _object_type_key_to_index(type_key)
            assert type_index is not None
            assert _lookup_type_attr(type_index, "__ffi_convert__") is not None

    def test_explicit_ref_registration_registers_convert_attr(self) -> None:
        """Explicit ``.ref<TObjectRef>()`` registration adds ``__ffi_convert__``."""
        type_index = _object_type_key_to_index("testing.TestIntPair")
        assert type_index is not None
        assert _lookup_type_attr(type_index, "__ffi_convert__") is not None

    def test_reflected_object_without_ref_does_not_register_convert_attr(self) -> None:
        """Reflected object classes without ``.ref<TObjectRef>()`` do not auto-register."""
        type_index = _object_type_key_to_index("testing.TestObjectBase")
        assert type_index is not None
        assert _lookup_type_attr(type_index, "__ffi_convert__") is None


class TestObjectMarshalFallback:
    """Object schema accepts values that the marshal path converts to Objects."""

    def test_exception_accepted_by_object_schema(self) -> None:
        """TypeSchema('Object') accepts Exception (-> ffi.Error)."""
        A(tvm_ffi.core.Object).check_value(RuntimeError("test"))

    def test_exception_accepted_by_error_schema(self) -> None:
        """TypeSchema('ffi.Error') accepts Exception."""
        A(tvm_ffi.core.Error).check_value(ValueError("oops"))

    def test_shape_accepted_from_python_list_via_convert(self) -> None:
        """TypeSchema('ffi.Shape') converts Python lists via __ffi_convert__."""
        result = _to_py_class_value(S("ffi.Shape").convert([1, 2, 3]))
        assert isinstance(result, tvm_ffi.Shape)
        assert tuple(result) == (1, 2, 3)

    def test_shape_accepted_from_ffi_list_via_convert(self) -> None:
        """TypeSchema('ffi.Shape') converts ffi.List via __ffi_convert__."""
        result = _to_py_class_value(S("ffi.Shape").convert(tvm_ffi.List([1, 2, 3])))
        assert isinstance(result, tvm_ffi.Shape)
        assert tuple(result) == (1, 2, 3)

    def test_shape_accepted_from_ffi_array_via_convert(self) -> None:
        """TypeSchema('ffi.Shape') converts ffi.Array via __ffi_convert__."""
        result = _to_py_class_value(S("ffi.Shape").convert(tvm_ffi.Array([1, 2, 3])))
        assert isinstance(result, tvm_ffi.Shape)
        assert tuple(result) == (1, 2, 3)

    def test_shape_convert_repeated_conversions(self) -> None:
        """Repeated __ffi_convert__ conversions keep returning live owning objects."""
        result1 = _to_py_class_value(S("ffi.Shape").convert([1, 2, 3]))
        result2 = _to_py_class_value(S("ffi.Shape").convert(tvm_ffi.List([1, 2, 3])))
        assert isinstance(result1, tvm_ffi.Shape)
        assert isinstance(result2, tvm_ffi.Shape)
        assert tuple(result1) == (1, 2, 3)
        assert tuple(result2) == (1, 2, 3)

    def test_exception_rejected_by_array_schema(self) -> None:
        """Exception is NOT accepted by Array schema (Error !IS-A Array)."""
        with pytest.raises(TypeError, match="expected Array"):
            A(tvm_ffi.Array).check_value(RuntimeError("x"))

    def test_opaque_object_accepted_by_object_schema(self) -> None:
        """TypeSchema('Object') accepts arbitrary Python objects (-> OpaquePyObject)."""

        class Custom:
            pass

        A(tvm_ffi.core.Object).check_value(Custom())

    def test_plain_object_accepted_by_object_schema(self) -> None:
        """TypeSchema('Object') accepts object()."""
        A(tvm_ffi.core.Object).check_value(object())

    def test_opaque_rejected_by_specific_schema(self) -> None:
        """Specific schema rejects arbitrary Python object."""

        class Custom:
            pass

        with pytest.raises(TypeError, match=r"got .*Custom"):
            A(TestIntPair).check_value(Custom())

    @pytest.mark.xfail(reason="SmallStr -> ObjectRef conversion is not supported yet")
    def test_str_accepted_by_object_schema(self) -> None:
        """TypeSchema('Object') accepts str (-> ffi.String IS-A Object)."""
        A(tvm_ffi.core.Object).check_value("hello")

    @pytest.mark.xfail(reason="SmallBytes -> ObjectRef conversion is not supported yet")
    def test_bytes_accepted_by_object_schema(self) -> None:
        """TypeSchema('Object') accepts bytes (-> ffi.Bytes IS-A Object)."""
        A(tvm_ffi.core.Object).check_value(b"hello")

    def test_list_accepted_by_object_schema(self) -> None:
        """TypeSchema('Object') accepts list (-> ffi.Array IS-A Object)."""
        A(tvm_ffi.core.Object).check_value([1, 2, 3])

    def test_dict_accepted_by_object_schema(self) -> None:
        """TypeSchema('Object') accepts dict (-> ffi.Map IS-A Object)."""
        A(tvm_ffi.core.Object).check_value({"a": 1})

    def test_callable_accepted_by_object_schema(self) -> None:
        """TypeSchema('Object') accepts callable (-> ffi.Function IS-A Object)."""
        A(tvm_ffi.core.Object).check_value(lambda: None)

    def test_int_rejected_by_object_schema(self) -> None:
        """TypeSchema('Object') rejects int (int is a POD type, not Object)."""
        with pytest.raises(TypeError):
            A(tvm_ffi.core.Object).check_value(42)

    def test_float_rejected_by_object_schema(self) -> None:
        """TypeSchema('Object') rejects float (float is a POD, not Object)."""
        with pytest.raises(TypeError):
            A(tvm_ffi.core.Object).check_value(3.14)

    def test_none_rejected_by_object_schema(self) -> None:
        """TypeSchema('Object') rejects None (None is a POD, not Object)."""
        with pytest.raises(TypeError):
            A(tvm_ffi.core.Object).check_value(None)


# ---------------------------------------------------------------------------
# Category 59: __cuda_stream__ for ctypes.c_void_p
# ---------------------------------------------------------------------------
class TestCudaStreamProtocol:
    """ctypes.c_void_p schema accepts __cuda_stream__ protocol."""

    def test_cuda_stream_accepted(self) -> None:
        """Object with __cuda_stream__ passes ctypes.c_void_p schema."""

        class CUStream:
            def __cuda_stream__(self) -> tuple[int, int]:
                return (0, 0)

        A(ctypes.c_void_p).check_value(CUStream())

    def test_cuda_stream_convert(self) -> None:
        """Convert returns __cuda_stream__ value as-is."""

        class CUStream:
            def __cuda_stream__(self) -> tuple[int, int]:
                return (0, 123)

        obj = CUStream()
        result = _to_py_class_value(A(ctypes.c_void_p).convert(obj))
        assert result is not None

    def test_cuda_stream_and_opaque_ptr(self) -> None:
        """Object with both __cuda_stream__ and __tvm_ffi_opaque_ptr__ accepted."""

        class DualProto:
            def __cuda_stream__(self) -> tuple[int, int]:
                return (0, 0)

            def __tvm_ffi_opaque_ptr__(self) -> int:
                return 0

        A(ctypes.c_void_p).check_value(DualProto())


# ---------------------------------------------------------------------------
# Category 60: Device __dlpack__ guard
# ---------------------------------------------------------------------------
class TestDeviceDlpackGuard:
    """Device schema respects __dlpack__ precedence."""

    def test_both_dlpack_and_device_rejected_by_device(self) -> None:
        """Object with both __dlpack__ and __dlpack_device__ rejected by Device."""

        class TensorLike:
            def __dlpack__(self) -> object:
                return None

            def __dlpack_device__(self) -> tuple[int, int]:
                return (1, 0)

        with pytest.raises(TypeError):
            A(tvm_ffi.Device).check_value(TensorLike())

    def test_both_dlpack_and_device_accepted_by_tensor(self) -> None:
        """Object with both __dlpack__ and __dlpack_device__ accepted by Tensor."""
        np = pytest.importorskip("numpy")

        class TensorLike:
            def __init__(self) -> None:
                self.array = np.arange(4, dtype="int32")

            def __dlpack__(self) -> object:
                return self.array.__dlpack__()

            def __dlpack_device__(self) -> tuple[int, int]:
                return self.array.__dlpack_device__()

        A(tvm_ffi.Tensor).check_value(TensorLike())

    def test_device_only_accepted_by_device(self) -> None:
        """Object with only __dlpack_device__ still accepted by Device."""

        class DevOnly:
            def __dlpack_device__(self) -> tuple[int, int]:
                return (1, 0)

        A(tvm_ffi.Device).check_value(DevOnly())

    def test_dlpack_only_rejected_by_device(self) -> None:
        """Object with only __dlpack__ rejected by Device schema."""

        class DLPackOnly:
            def __dlpack__(self) -> object:
                return None

        with pytest.raises(TypeError):
            A(tvm_ffi.Device).check_value(DLPackOnly())


# ---------------------------------------------------------------------------
# Category 61: SKIP_DLPACK_C_EXCHANGE_API env gate
# ---------------------------------------------------------------------------
class TestSkipDlpackEnvGate:
    """Tensor schema respects TVM_FFI_SKIP_DLPACK_C_EXCHANGE_API."""

    def test_exchange_api_accepted_by_default(self) -> None:
        """__dlpack_c_exchange_api__ accepted when env not set."""
        os.environ.pop("TVM_FFI_SKIP_DLPACK_C_EXCHANGE_API", None)
        np = pytest.importorskip("numpy")
        tensor = tvm_ffi.from_dlpack(np.arange(4, dtype="int32"))
        wrapper = tvm_ffi.core.DLTensorTestWrapper(tensor)
        A(tvm_ffi.Tensor).check_value(wrapper)

    def test_exchange_api_rejected_when_skipped(self) -> None:
        """__dlpack_c_exchange_api__ rejected when env=1."""
        os.environ["TVM_FFI_SKIP_DLPACK_C_EXCHANGE_API"] = "1"
        try:

            class ExchangeAPI:
                def __dlpack_c_exchange_api__(self) -> int:
                    return 0

            with pytest.raises(TypeError):
                A(tvm_ffi.Tensor).check_value(ExchangeAPI())
        finally:
            del os.environ["TVM_FFI_SKIP_DLPACK_C_EXCHANGE_API"]


# ---------------------------------------------------------------------------
# Category 62: from_type_index low-level indices
# ---------------------------------------------------------------------------
class TestFromTypeIndexLowLevel:
    """from_type_index handles all built-in type indices."""

    def test_dl_tensor_ptr(self) -> None:
        """KTVMFFIDLTensorPtr maps to Tensor."""
        s = TypeSchema.from_type_index(7)  # kTVMFFIDLTensorPtr
        assert s.origin == "Tensor"

    def test_raw_str(self) -> None:
        """KTVMFFIRawStr maps to str."""
        s = TypeSchema.from_type_index(8)  # kTVMFFIRawStr
        assert s.origin == "str"

    def test_byte_array_ptr(self) -> None:
        """KTVMFFIByteArrayPtr maps to bytes."""
        s = TypeSchema.from_type_index(9)  # kTVMFFIByteArrayPtr
        assert s.origin == "bytes"

    def test_object_rvalue_ref(self) -> None:
        """KTVMFFIObjectRValueRef maps to Object."""
        s = TypeSchema.from_type_index(10)  # kTVMFFIObjectRValueRef
        assert s.origin == "Object"

    def test_small_str(self) -> None:
        """KTVMFFISmallStr maps to str."""
        s = TypeSchema.from_type_index(11)  # kTVMFFISmallStr
        assert s.origin == "str"

    def test_small_bytes(self) -> None:
        """KTVMFFISmallBytes maps to bytes."""
        s = TypeSchema.from_type_index(12)  # kTVMFFISmallBytes
        assert s.origin == "bytes"

    def test_all_low_level_schemas_usable(self) -> None:
        """Schemas from low-level indices can be used for conversion."""
        for idx in (7, 8, 9, 11, 12):
            s = TypeSchema.from_type_index(idx)
            # Trigger converter build; some schemas raise TypeError for None
            try:
                s.convert(None)
            except TypeError:
                pass


# ---------------------------------------------------------------------------
# Category 63: STL origin parsing
# ---------------------------------------------------------------------------
class TestSTLOriginParsing:
    """C++ STL schema origins are correctly parsed."""

    def test_std_vector(self) -> None:
        """std::vector maps to Array."""
        s = TypeSchema.from_json_str('{"type":"std::vector","args":[{"type":"int"}]}')
        assert s.origin == "Array"

    def test_std_optional(self) -> None:
        """std::optional maps to Optional."""
        s = TypeSchema.from_json_str('{"type":"std::optional","args":[{"type":"int"}]}')
        assert s.origin == "Optional"
        assert repr(s) == "int | None"

    def test_std_variant(self) -> None:
        """std::variant maps to Union."""
        s = TypeSchema.from_json_str(
            '{"type":"std::variant","args":[{"type":"int"},{"type":"str"}]}'
        )
        assert s.origin == "Union"
        assert repr(s) == "int | str"

    def test_std_tuple(self) -> None:
        """std::tuple maps to tuple."""
        s = TypeSchema.from_json_str('{"type":"std::tuple","args":[{"type":"int"},{"type":"str"}]}')
        assert s.origin == "tuple"

    def test_std_map(self) -> None:
        """std::map maps to Map."""
        s = TypeSchema.from_json_str('{"type":"std::map","args":[{"type":"str"},{"type":"int"}]}')
        assert s.origin == "Map"

    def test_std_unordered_map(self) -> None:
        """std::unordered_map maps to Map."""
        s = TypeSchema.from_json_str(
            '{"type":"std::unordered_map","args":[{"type":"str"},{"type":"int"}]}'
        )
        assert s.origin == "Map"

    def test_std_function(self) -> None:
        """std::function maps to Callable."""
        s = TypeSchema.from_json_str(
            '{"type":"std::function","args":[{"type":"int"},{"type":"str"}]}'
        )
        assert s.origin == "Callable"

    def test_object_rvalue_ref_origin(self) -> None:
        """ObjectRValueRef maps to Object."""
        s = TypeSchema.from_json_str('{"type":"ObjectRValueRef","args":[]}')
        assert s.origin == "Object"


# ---------------------------------------------------------------------------
# Category 64: Zero-copy container conversion
# ---------------------------------------------------------------------------
class TestZeroCopyConversion:
    """Typed container conversion preserves identity when no elements change."""

    @requires_py39
    def test_array_int_exact_list(self) -> None:
        """Array[int] on exact Python list converts successfully."""
        original = [1, 2, 3]
        result = _to_py_class_value(A(tuple[int, ...]).convert(original))
        assert list(result) == original

    @requires_py39
    def test_array_int_needs_conversion(self) -> None:
        """Array[int] on list needing bool->int returns converted list."""
        original = [1, True, 3]
        result = _to_py_class_value(A(tuple[int, ...]).convert(original))
        assert list(result) == [1, 1, 3]

    @requires_py39
    def test_map_str_int_exact_dict(self) -> None:
        """Map[str, int] on exact dict converts successfully."""
        original = {"a": 1, "b": 2}
        result = _to_py_class_value(A(tvm_ffi.Map[str, int]).convert(original))
        assert dict(result) == original

    @requires_py39
    def test_map_str_int_needs_conversion(self) -> None:
        """Map[str, int] on dict needing conversion returns converted dict."""
        original = {"a": True, "b": 2}
        result = _to_py_class_value(A(tvm_ffi.Map[str, int]).convert(original))
        assert result is not None

    @requires_py39
    def test_tuple_exact_match(self) -> None:
        """tuple[int, str] on exact tuple converts successfully."""
        original = (42, "hello")
        result = _to_py_class_value(A(tuple[int, str]).convert(original))
        assert tuple(result) == original

    @requires_py39
    def test_tuple_needs_conversion(self) -> None:
        """tuple[int, str] on tuple needing conversion returns converted tuple."""
        original = (True, "hello")
        result = _to_py_class_value(A(tuple[int, str]).convert(original))
        assert tuple(result) == (1, "hello")

    @requires_py39
    def test_list_int_exact(self) -> None:
        """List[int] on exact list converts successfully."""
        original = [10, 20]
        result = _to_py_class_value(A(list[int]).convert(original))
        assert list(result) == original


# ---------------------------------------------------------------------------
# Category 65: Exception normalization in check_value/convert
# ---------------------------------------------------------------------------
class TestExceptionNormalization:
    """check_value/convert normalize custom __int__/__float__ failures."""

    def test_broken_integral_convert(self) -> None:
        """Integral with broken __int__ caught by convert."""

        class BadIntegral:
            def __int__(self) -> int:
                raise OverflowError("too big")

        Integral.register(BadIntegral)

        with pytest.raises(TypeError, match="too big"):
            A(int).convert(BadIntegral())

    def test_broken_integral_check_value(self) -> None:
        """Integral with broken __int__ handled by check_value."""

        class BrokenInt:
            def __int__(self) -> int:
                raise ValueError("broken")

        Integral.register(BrokenInt)

        # check_value should raise TypeError (wrapping the ValueError)
        with pytest.raises(TypeError, match="broken"):
            A(int).check_value(BrokenInt())

    def test_broken_integral_bool_check_value(self) -> None:
        """Integral with broken __bool__ is normalized to TypeError."""

        class BrokenBoolInt:
            def __int__(self) -> int:
                return 1

            def __bool__(self) -> bool:
                raise RuntimeError("broken bool")

        Integral.register(BrokenBoolInt)

        with pytest.raises(TypeError, match="broken bool"):
            A(bool).check_value(BrokenBoolInt())

    def test_union_falls_back_after_broken_bool(self) -> None:
        """Union keeps trying alternatives when bool conversion fails.

        A str subclass registered as Integral is a pathological case;
        the C packer may dispatch it as Integral rather than str, so
        we only guarantee that Union dispatch does not silently swallow
        the error — either a successful conversion or a raised error
        is acceptable.
        """

        class BrokenBoolStr(str):
            def __bool__(self) -> bool:
                raise RuntimeError("broken bool")

        Integral.register(BrokenBoolStr)

        try:
            result = _to_py_class_value(A(Union[bool, str]).convert(BrokenBoolStr("hello")))
            assert result == "hello"
        except (ValueError, RuntimeError, TypeError):
            pass


# ---------------------------------------------------------------------------
# Category 66: __tvm_ffi_value__ eager normalization
# ---------------------------------------------------------------------------
class TestValueProtocolPrecedence:
    """__tvm_ffi_value__ runs before schema-specific dispatch."""

    def test_value_protocol_runs_before_int_protocol(self) -> None:
        """__tvm_ffi_value__ is applied before __tvm_ffi_int__."""

        class Dual:
            def __tvm_ffi_int__(self) -> int:
                return 42

            def __tvm_ffi_value__(self) -> object:
                return TestIntPair(1, 2)

        with pytest.raises(TypeError):
            A(int).check_value(Dual())
        A(tvm_ffi.core.Object).check_value(Dual())

    def test_value_protocol_runs_before_float_protocol(self) -> None:
        """__tvm_ffi_value__ is applied before __tvm_ffi_float__."""

        class Dual:
            def __tvm_ffi_float__(self) -> float:
                return 1.0

            def __tvm_ffi_value__(self) -> object:
                return TestIntPair(1, 2)

        with pytest.raises(TypeError):
            A(float).check_value(Dual())
        A(tvm_ffi.core.Object).check_value(Dual())

    def test_pure_value_protocol_still_works(self) -> None:
        """Class with ONLY __tvm_ffi_value__ still converts eagerly."""

        class PureVP:
            def __tvm_ffi_value__(self) -> object:
                return 42

        A(int).check_value(PureVP())

    def test_value_protocol_runs_before_callable_dispatch(self) -> None:
        """Callable classes are normalized through __tvm_ffi_value__ first."""

        class CallableVP:
            def __call__(self) -> None:
                pass

            def __tvm_ffi_value__(self) -> object:
                return 42

        with pytest.raises(TypeError):
            A(Callable).check_value(CallableVP())
        A(int).check_value(CallableVP())

    def test_object_protocol_precedes_value_and_convertible(self) -> None:
        """__tvm_ffi_object__ wins over value and ObjectConvertible hooks."""
        inner = TestIntPair(10, 20)

        class Both(ObjectConvertible):
            def __tvm_ffi_object__(self) -> object:
                return inner

            def __tvm_ffi_value__(self) -> object:
                return 999

            def asobject(self) -> tvm_ffi.core.Object:
                return TestIntPair(99, 99)

        result = _to_py_class_value(A(tvm_ffi.core.Object).convert(Both()))
        assert result.same_as(inner)


# ---------------------------------------------------------------------------
# Category 67: Union single-call __tvm_ffi_value__
# ---------------------------------------------------------------------------
class TestUnionValueProtocol:
    """Union dispatches __tvm_ffi_value__ once, not per-alternative."""

    def test_union_value_protocol_once(self) -> None:
        """__tvm_ffi_value__ called once for Union."""
        call_count = 0

        class CountingVP:
            def __tvm_ffi_value__(self) -> object:
                nonlocal call_count
                call_count += 1
                return 42

        A(Union[str, int]).check_value(CountingVP())
        assert call_count == 1

    def test_union_value_protocol_mismatch(self) -> None:
        """__tvm_ffi_value__ returning wrong type fails Union."""
        call_count = 0

        class WrongVP:
            def __tvm_ffi_value__(self) -> object:
                nonlocal call_count
                call_count += 1
                return object()

        with pytest.raises(TypeError):
            A(Union[int, str]).check_value(WrongVP())
        assert call_count == 1


# ---------------------------------------------------------------------------
# Category 68: from_json_obj robustness
# ---------------------------------------------------------------------------
class TestFromJsonObjRobustness:
    """from_json_obj handles non-dict args and malformed input."""

    def test_non_dict_args_skipped(self) -> None:
        """Non-dict elements in args list are silently skipped."""
        obj = {"type": "std::vector", "args": [{"type": "int"}, 42]}
        s = TypeSchema.from_json_obj(obj)
        assert s.origin == "Array"
        assert len(s.args) == 1
        assert s.args[0].origin == "int"

    def test_malformed_input_raises_type_error(self) -> None:
        """Non-dict top-level raises TypeError, not AssertionError."""
        with pytest.raises(TypeError, match="expected schema dict"):
            TypeSchema.from_json_obj("not_a_dict")  # type: ignore[arg-type]

    def test_missing_type_key_raises_type_error(self) -> None:
        """Dict without 'type' key raises TypeError."""
        with pytest.raises(TypeError, match="expected schema dict"):
            TypeSchema.from_json_obj({"args": []})


# ---------------------------------------------------------------------------
# Category 69: Mutual-cycle RecursionError normalized
# ---------------------------------------------------------------------------
class TestMutualCycleNormalization:
    """Mutual __tvm_ffi_value__ cycles produce TypeError, not RecursionError."""

    def test_mutual_cycle_check_value(self) -> None:
        """check_value normalizes mutual-cycle RecursionError to TypeError."""

        class A:
            def __init__(self) -> None:
                self.other: object = None

            def __tvm_ffi_value__(self) -> object:
                return self.other

        class B:
            def __init__(self) -> None:
                self.other: object = None

            def __tvm_ffi_value__(self) -> object:
                return self.other

        a, b = A(), B()
        a.other = b
        b.other = a

        with pytest.raises(TypeError, match="cycle"):
            S("int").check_value(a)

    def test_mutual_cycle_convert(self) -> None:
        """Convert normalizes mutual-cycle RecursionError to TypeError."""

        class A:
            def __init__(self) -> None:
                self.other: object = None

            def __tvm_ffi_value__(self) -> object:
                return self.other

        class B:
            def __init__(self) -> None:
                self.other: object = None

            def __tvm_ffi_value__(self) -> object:
                return self.other

        a, b = A(), B()
        a.other = b
        b.other = a

        with pytest.raises(TypeError, match="cycle"):
            S("int").convert(a)


# ---------------------------------------------------------------------------
# Category 70: ObjectConvertible vs __tvm_ffi_value__ precedence
# ---------------------------------------------------------------------------
class TestObjectConvertiblePrecedence:
    """__tvm_ffi_value__ takes precedence over ObjectConvertible."""

    def test_value_protocol_wins_over_convertible(self) -> None:
        """Class with both __tvm_ffi_value__ and ObjectConvertible uses fallback."""
        pair = TestIntPair(10, 20)

        class DualProtocol(ObjectConvertible):
            def asobject(self) -> tvm_ffi.core.Object:
                return pair

            def __tvm_ffi_value__(self) -> object:
                return 42

        # int schema: __tvm_ffi_value__ returns 42, accepted
        A(int).check_value(DualProtocol())
        # Object schema: __tvm_ffi_value__ returns 42 (POD int, not Object),
        # should REJECT (not accept via ObjectConvertible)
        with pytest.raises(TypeError):
            A(tvm_ffi.core.Object).check_value(DualProtocol())

    def test_pure_convertible_still_works(self) -> None:
        """ObjectConvertible without __tvm_ffi_value__ still accepted."""
        pair = TestIntPair(1, 2)

        class PureConvertible(ObjectConvertible):
            def asobject(self) -> tvm_ffi.core.Object:
                return pair

        A(tvm_ffi.core.Object).check_value(PureConvertible())
        A(TestIntPair).check_value(PureConvertible())


# ---------------------------------------------------------------------------
# Category 71: from_json_obj non-iterable args
# ---------------------------------------------------------------------------
class TestFromJsonObjNonIterableArgs:
    """from_json_obj handles non-iterable args values gracefully."""

    def test_non_iterable_args_treated_as_empty(self) -> None:
        """Non-list/tuple args value (e.g., int) treated as empty args."""
        s = TypeSchema.from_json_obj({"type": "int", "args": 42})
        assert s.origin == "int"
        assert s.args == ()

    def test_string_args_treated_as_empty(self) -> None:
        """String args value treated as empty (not iterated char-by-char)."""
        s = TypeSchema.from_json_obj({"type": "int", "args": "bad"})
        assert s.origin == "int"
        assert s.args == ()


# ---------------------------------------------------------------------------
# CAny class tests
# ---------------------------------------------------------------------------
class TestCAny:
    """Tests for the CAny owned-value container."""

    def test_cany_from_int(self) -> None:
        """convert(int) returns CAny with correct type_index."""
        cany = A(int).convert(42)
        assert isinstance(cany, CAny)
        assert cany.type_index == 1  # kTVMFFIInt

    def test_cany_from_float(self) -> None:
        """convert(float) returns CAny with correct type_index."""
        cany = A(float).convert(3.14)
        assert isinstance(cany, CAny)
        assert cany.type_index == 3  # kTVMFFIFloat

    def test_cany_from_bool(self) -> None:
        """convert(bool) returns CAny with correct type_index."""
        cany = A(bool).convert(True)
        assert isinstance(cany, CAny)
        assert cany.type_index == 2  # kTVMFFIBool

    def test_cany_from_none(self) -> None:
        """convert(None) returns CAny with type_index 0."""
        cany = A(None).convert(None)
        assert isinstance(cany, CAny)
        assert cany.type_index == 0  # kTVMFFINone

    def test_cany_from_str(self) -> None:
        """convert(str) returns CAny."""
        cany = A(str).convert("hello")
        assert isinstance(cany, CAny)
        # Short strings have type_index=11 (SmallStr), longer ones have 65 (Str)
        assert cany.type_index in (11, 65)

    @requires_py39
    def test_cany_from_array(self) -> None:
        """convert(Array) returns CAny with array type_index."""
        cany = A(tuple[int, ...]).convert([1, 2, 3])
        assert isinstance(cany, CAny)
        assert cany.type_index >= 64  # object type

    def test_to_py_int(self) -> None:
        """to_py() round-trips int correctly."""
        result = _to_py_class_value(A(int).convert(42))
        assert result == 42
        assert type(result) is int

    def test_to_py_float(self) -> None:
        """to_py() round-trips float correctly."""
        result = _to_py_class_value(A(float).convert(3.14))
        assert result == 3.14
        assert type(result) is float

    def test_to_py_bool(self) -> None:
        """to_py() round-trips bool correctly."""
        assert _to_py_class_value(A(bool).convert(True)) is True
        assert _to_py_class_value(A(bool).convert(False)) is False

    def test_to_py_none(self) -> None:
        """to_py() round-trips None correctly."""
        assert _to_py_class_value(A(None).convert(None)) is None

    def test_to_py_str(self) -> None:
        """to_py() round-trips str correctly."""
        assert _to_py_class_value(A(str).convert("hello")) == "hello"

    @requires_py39
    def test_to_py_array(self) -> None:
        """to_py() returns ffi.Array for Array convert."""
        result = _to_py_class_value(A(tuple[int, ...]).convert([1, 2, 3]))
        assert isinstance(result, tvm_ffi.Array)
        assert list(result) == [1, 2, 3]

    @requires_py39
    def test_to_py_list(self) -> None:
        """to_py() returns ffi.List for List convert."""
        result = _to_py_class_value(A(list[int]).convert([1, 2, 3]))
        assert isinstance(result, tvm_ffi.List)
        assert list(result) == [1, 2, 3]

    @requires_py39
    def test_to_py_map(self) -> None:
        """to_py() returns ffi.Map for Map convert."""
        result = _to_py_class_value(A(tvm_ffi.Map[str, int]).convert({"a": 1}))
        assert isinstance(result, tvm_ffi.Map)

    @requires_py39
    def test_to_py_dict(self) -> None:
        """to_py() returns ffi.Dict for Dict convert."""
        result = _to_py_class_value(A(dict[str, int]).convert({"a": 1}))
        assert isinstance(result, tvm_ffi.Dict)

    def test_multiple_to_py_calls(self) -> None:
        """to_py() can be called multiple times safely."""
        cany = A(int).convert(42)
        assert _to_py_class_value(cany) == 42
        assert _to_py_class_value(cany) == 42
        assert _to_py_class_value(cany) == 42

    @requires_py39
    def test_object_refcount_safety(self) -> None:
        """to_py() for objects properly IncRefs — no double-free."""
        cany = A(tuple[int, ...]).convert([1, 2, 3])
        py1 = _to_py_class_value(cany)
        py2 = _to_py_class_value(cany)
        del cany  # CAny.__dealloc__ runs
        assert list(py1) == [1, 2, 3]
        assert list(py2) == [1, 2, 3]

    def test_repr_int(self) -> None:
        """Repr shows type and value for int."""
        cany = A(int).convert(42)
        assert "int" in repr(cany)
        assert "42" in repr(cany)

    def test_repr_none(self) -> None:
        """Repr shows None."""
        cany = A(None).convert(None)
        assert "None" in repr(cany)

    def test_repr_float(self) -> None:
        """Repr shows float value."""
        cany = A(float).convert(3.14)
        assert "float" in repr(cany)

    @requires_py39
    def test_repr_object(self) -> None:
        """Repr shows type_index for objects."""
        cany = A(tuple[int, ...]).convert([1, 2, 3])
        assert "type_index" in repr(cany)

    def test_convert_raises_type_error(self) -> None:
        """Convert still raises TypeError for incompatible values."""
        with pytest.raises(TypeError):
            A(int).convert("hello")

    def test_check_value_does_not_return_cany(self) -> None:
        """check_value returns None (not CAny)."""
        result = A(int).check_value(42)
        assert result is None


# ---------------------------------------------------------------------------
# from_annotation structural equality tests
# ---------------------------------------------------------------------------
class TestFromAnnotationScalars:
    """Scalar types — from_annotation produces correct TypeSchema."""

    def test_int(self) -> None:
        """Int annotation."""
        assert A(int) == S("int")

    def test_float(self) -> None:
        """Float annotation."""
        assert A(float) == S("float")

    def test_bool(self) -> None:
        """Bool annotation."""
        assert A(bool) == S("bool")

    def test_str(self) -> None:
        """Str annotation."""
        assert A(str) == S("str")

    def test_bytes(self) -> None:
        """Bytes annotation."""
        assert A(bytes) == S("bytes")

    def test_none_type(self) -> None:
        """type(None) annotation."""
        assert A(type(None)) == S("None")

    def test_none_literal(self) -> None:
        """None annotation."""
        assert A(None) == S("None")

    def test_any(self) -> None:
        """typing.Any annotation."""
        assert A(typing.Any) == S("Any")

    def test_tvm_ffi_string(self) -> None:
        """tvm_ffi.String maps to str schema."""
        assert A(tvm_ffi.core.String) == S("str")

    def test_tvm_ffi_bytes(self) -> None:
        """tvm_ffi.Bytes maps to bytes schema."""
        assert A(tvm_ffi.core.Bytes) == S("bytes")


class TestFromAnnotationFFITypes:
    """FFI container and object types."""

    def test_array(self) -> None:
        """tvm_ffi.Array → canonical origin 'Array'."""
        assert A(tvm_ffi.Array) == S("Array")

    def test_list(self) -> None:
        """tvm_ffi.List → same as A(list)."""
        assert A(tvm_ffi.List) == A(list)

    def test_map(self) -> None:
        """tvm_ffi.Map → canonical origin 'Map'."""
        assert A(tvm_ffi.Map) == S("Map")

    def test_dict(self) -> None:
        """tvm_ffi.Dict → same as A(dict)."""
        assert A(tvm_ffi.Dict) == A(dict)

    def test_function(self) -> None:
        """tvm_ffi.Function → same as A(Callable)."""
        assert A(tvm_ffi.core.Function) == A(Callable)

    def test_object(self) -> None:
        """tvm_ffi.Object → canonical origin 'Object'."""
        assert A(tvm_ffi.core.Object) == S("Object")

    def test_tensor(self) -> None:
        """tvm_ffi.Tensor → canonical origin 'Tensor'."""
        assert A(tvm_ffi.Tensor) == S("Tensor")

    def test_dtype(self) -> None:
        """tvm_ffi.core.DataType → canonical origin 'dtype'."""
        assert A(tvm_ffi.core.DataType) == S("dtype")

    def test_device(self) -> None:
        """tvm_ffi.Device → canonical origin 'Device'."""
        assert A(tvm_ffi.Device) == S("Device")

    def test_ctypes_c_void_p(self) -> None:
        """ctypes.c_void_p → canonical origin 'ctypes.c_void_p'."""
        assert A(ctypes.c_void_p) == S("ctypes.c_void_p")

    @requires_py39
    def test_array_parameterized(self) -> None:
        """tvm_ffi.Array[int] cross-equivalent to tuple[int, ...]."""
        assert A(tvm_ffi.Array[int]) == A(tuple[int, ...])

    @requires_py39
    def test_list_parameterized(self) -> None:
        """tvm_ffi.List[str] cross-equivalent to list[str]."""
        assert A(tvm_ffi.List[str]) == A(list[str])

    @requires_py39
    def test_map_parameterized(self) -> None:
        """tvm_ffi.Map[str, float]."""
        assert A(tvm_ffi.Map[str, float]) == S("Map", S("str"), S("float"))

    @requires_py39
    def test_dict_parameterized(self) -> None:
        """tvm_ffi.Dict[str, int] cross-equivalent to dict[str, int]."""
        assert A(tvm_ffi.Dict[str, int]) == A(dict[str, int])

    @requires_py39
    def test_array_too_many_args(self) -> None:
        """tvm_ffi.Array[int, str] raises TypeError."""
        with pytest.raises(TypeError, match="requires 1"):
            A(tvm_ffi.Array[int, str])  # type: ignore[type-arg]

    @requires_py39
    def test_list_too_many_args(self) -> None:
        """tvm_ffi.List[int, str] raises TypeError."""
        with pytest.raises(TypeError, match="requires 1"):
            A(tvm_ffi.List[int, str])  # type: ignore[type-arg]

    @requires_py39
    def test_dict_one_arg(self) -> None:
        """tvm_ffi.Dict[str] raises TypeError."""
        with pytest.raises(TypeError, match="requires 2"):
            A(tvm_ffi.Dict[str])  # type: ignore[type-arg]

    @requires_py39
    def test_dict_three_args(self) -> None:
        """tvm_ffi.Dict[str, int, float] raises TypeError."""
        with pytest.raises(TypeError, match="requires 2"):
            A(tvm_ffi.Dict[str, int, float])  # type: ignore[type-arg]

    @requires_py39
    def test_map_one_arg(self) -> None:
        """tvm_ffi.Map[str] raises TypeError."""
        with pytest.raises(TypeError, match="requires 2"):
            A(tvm_ffi.Map[str])  # type: ignore[type-arg]

    def test_unregistered_cobject_errors(self) -> None:
        """Unregistered CObject subclass raises TypeError."""
        with pytest.raises(TypeError, match="not registered"):
            A(tvm_ffi.core.CObject)


class TestFromAnnotationCallable:
    """Callable annotation tests."""

    def test_bare(self) -> None:
        """Bare Callable."""
        assert A(Callable) == S("Callable")

    def test_bare_collections_abc(self) -> None:
        """Bare collections.abc.Callable."""
        assert A(collections.abc.Callable) == S("Callable")

    def test_params(self) -> None:
        """Callable[[int, str], bool]."""
        assert A(Callable[[int, str], bool]) == S("Callable", S("bool"), S("int"), S("str"))

    def test_ellipsis(self) -> None:
        """Callable[..., int]."""
        assert A(Callable[..., int]) == S("Callable", S("int"))

    def test_no_params(self) -> None:
        """Callable[[], int]."""
        assert A(Callable[[], int]) == S("Callable", S("int"))


class TestFromAnnotationList:
    """list[T] → List tests."""

    def test_bare(self) -> None:
        """Bare list."""
        assert A(list).origin == "List"

    @requires_py39
    def test_int(self) -> None:
        """list[int]."""
        assert A(list[int]) == S("List", S("int"))

    @requires_py39
    def test_nested(self) -> None:
        """list[list[int]]."""
        assert A(list[list[int]]) == S("List", S("List", S("int")))


class TestFromAnnotationDict:
    """dict[K, V] → Dict tests."""

    def test_bare(self) -> None:
        """Bare dict."""
        assert A(dict).origin == "Dict"

    @requires_py39
    def test_str_int(self) -> None:
        """dict[str, int]."""
        assert A(dict[str, int]) == S("Dict", S("str"), S("int"))


class TestFromAnnotationArray:
    """tuple[T, ...] → Array tests."""

    @requires_py39
    def test_int(self) -> None:
        """tuple[int, ...]."""
        assert A(tuple[int, ...]) == S("Array", S("int"))

    @requires_py39
    def test_float(self) -> None:
        """tuple[float, ...]."""
        assert A(tuple[float, ...]) == S("Array", S("float"))


class TestFromAnnotationTuple:
    """tuple[T1, T2] (fixed) tests."""

    def test_bare(self) -> None:
        """Bare tuple."""
        assert A(tuple).origin == "tuple"

    @requires_py39
    def test_int_str(self) -> None:
        """tuple[int, str]."""
        assert A(tuple[int, str]) == S("tuple", S("int"), S("str"))

    @requires_py39
    def test_empty(self) -> None:
        """tuple[()] stays distinct from bare tuple."""
        assert A(tuple[()]) == TypeSchema("tuple", ())


class TestFromAnnotationOptional:
    """Optional[T] tests."""

    def test_int(self) -> None:
        """Optional[int]."""
        assert A(Optional[int]) == S("Optional", S("int"))

    def test_union_with_none_becomes_optional(self) -> None:
        """Union[int, None] normalizes to Optional[int]."""
        assert A(Union[int, None]) == S("Optional", S("int"))

    @requires_py310
    def test_pipe_syntax(self) -> None:
        """Int | None."""
        assert A(eval("int | None")) == S("Optional", S("int"))


class TestFromAnnotationUnion:
    """Union[T1, T2] tests."""

    def test_int_str(self) -> None:
        """Union[int, str]."""
        assert A(Union[int, str]) == S("Union", S("int"), S("str"))

    def test_nested_union_flattening(self) -> None:
        """Nested unions flatten to a single Union schema."""
        assert A(Union[int, Union[str, float]]) == S("Union", S("int"), S("str"), S("float"))

    @requires_py310
    def test_pipe_syntax(self) -> None:
        """Int | str."""
        assert A(eval("int | str")) == S("Union", S("int"), S("str"))


class TestFromAnnotationObject:
    """Registered CObject subclasses."""

    def test_test_int_pair(self) -> None:
        """TestIntPair annotation."""
        assert A(TestIntPair) == S("testing.TestIntPair")

    def test_cxx_class_base(self) -> None:
        """_TestCxxClassBase annotation."""
        assert A(_TestCxxClassBase) == S("testing.TestCxxClassBase")


class TestFromAnnotationErrors:
    """from_annotation raises TypeError for unsupported annotations."""

    def test_unsupported_type(self) -> None:
        """Complex is not supported."""
        with pytest.raises(TypeError, match="Cannot convert"):
            A(complex)

    @requires_py39
    def test_list_too_many_args(self) -> None:
        """list[int, int, float] raises."""
        with pytest.raises(TypeError, match="list takes at most 1"):
            A(list[int, int, float])  # type: ignore[type-arg]

    @requires_py39
    def test_dict_one_arg(self) -> None:
        """dict[str] raises."""
        with pytest.raises(TypeError, match="dict requires 0 or 2"):
            A(dict[str])  # type: ignore[type-arg]


# ---------------------------------------------------------------------------
# Convert returns FFI containers
# ---------------------------------------------------------------------------
import tvm_ffi as _tvm_ffi


class TestConvertReturnFFIContainers:
    """convert() returns ffi.Array/List/Map/Dict."""

    @requires_py39
    def test_array_from_list(self) -> None:
        """Array convert from Python list."""
        result = _to_py_class_value(A(tuple[float, ...]).convert([1, 2, 3]))
        assert isinstance(result, _tvm_ffi.Array)
        assert list(result) == [1.0, 2.0, 3.0]

    @requires_py39
    def test_list_from_list(self) -> None:
        """List convert from Python list."""
        result = _to_py_class_value(A(list[int]).convert([1, 2, 3]))
        assert isinstance(result, _tvm_ffi.List)
        assert list(result) == [1, 2, 3]

    @requires_py39
    def test_dict_from_dict(self) -> None:
        """Dict convert from Python dict."""
        result = _to_py_class_value(A(dict[str, int]).convert({"a": 1}))
        assert isinstance(result, _tvm_ffi.Dict)

    @requires_py39
    def test_map_from_dict(self) -> None:
        """Map convert from Python dict."""
        result = _to_py_class_value(A(tvm_ffi.Map[str, int]).convert({"a": 1}))
        assert isinstance(result, _tvm_ffi.Map)

    @requires_py39
    def test_array_passthrough(self) -> None:
        """ffi.Array input passes through unchanged."""
        arr = _tvm_ffi.Array([1, 2, 3])
        result = _to_py_class_value(A(tuple[int, ...]).convert(arr))
        assert result.same_as(arr)

    @requires_py39
    def test_list_passthrough(self) -> None:
        """ffi.List input passes through unchanged."""
        lst = _tvm_ffi.List([1, 2, 3])
        result = _to_py_class_value(A(list[int]).convert(lst))
        assert result.same_as(lst)

    @requires_py39
    def test_array_subclass_passthrough(self) -> None:
        """ffi.Array subclasses pass through unchanged."""

        class MyArray(_tvm_ffi.Array):
            pass

        arr = MyArray([1, 2, 3])
        result = _to_py_class_value(A(tuple[int, ...]).convert(arr))
        assert result.same_as(arr)

    @requires_py39
    def test_list_subclass_passthrough(self) -> None:
        """ffi.List subclasses pass through unchanged."""

        class MyList(_tvm_ffi.List):
            pass

        lst = MyList([1, 2, 3])
        result = _to_py_class_value(A(list[int]).convert(lst))
        assert result.same_as(lst)

    @requires_py39
    def test_map_subclass_passthrough(self) -> None:
        """ffi.Map subclasses pass through unchanged for Map[Any, Any]."""

        class MyMap(_tvm_ffi.Map):
            pass

        m = MyMap({"a": 1})
        result = _to_py_class_value(A(tvm_ffi.Map[typing.Any, typing.Any]).convert(m))
        assert result.same_as(m)

    @requires_py39
    def test_dict_subclass_passthrough(self) -> None:
        """ffi.Dict subclasses pass through unchanged for Dict[Any, Any]."""

        class MyDict(_tvm_ffi.Dict):
            pass

        d = MyDict({"a": 1})
        result = _to_py_class_value(A(dict[typing.Any, typing.Any]).convert(d))
        assert result.same_as(d)

    @requires_py39
    def test_nested_array_convert(self) -> None:
        """Nested array conversion."""
        result = _to_py_class_value(A(tuple[tuple[int, ...], ...]).convert([[1, 2], [3, 4]]))
        assert isinstance(result, _tvm_ffi.Array)
        assert isinstance(result[0], _tvm_ffi.Array)


# ---------------------------------------------------------------------------
# FFI type guarantees: convert() always returns tvm_ffi types
# ---------------------------------------------------------------------------
class TestConvertToFFITypes:
    """convert() returns canonical FFI types for all value kinds."""

    def test_short_str_is_string(self) -> None:
        """Short str (SmallStr) promotes to tvm_ffi.String."""
        result = _to_py_class_value(A(str).convert("hi"))
        assert isinstance(result, tvm_ffi.core.String)
        assert result == "hi"

    def test_long_str_is_string(self) -> None:
        """Long str (kTVMFFIStr object) is tvm_ffi.String."""
        long_s = "x" * 200
        result = _to_py_class_value(A(str).convert(long_s))
        assert isinstance(result, tvm_ffi.core.String)
        assert result == long_s

    def test_empty_str_is_string(self) -> None:
        """Empty str is tvm_ffi.String."""
        result = _to_py_class_value(A(str).convert(""))
        assert isinstance(result, tvm_ffi.core.String)
        assert result == ""

    def test_short_bytes_is_bytes(self) -> None:
        """Short bytes (SmallBytes) promotes to tvm_ffi.Bytes."""
        result = _to_py_class_value(A(bytes).convert(b"hi"))
        assert isinstance(result, tvm_ffi.core.Bytes)
        assert result == b"hi"

    def test_long_bytes_is_bytes(self) -> None:
        """Long bytes (kTVMFFIBytes object) is tvm_ffi.Bytes."""
        long_b = b"x" * 200
        result = _to_py_class_value(A(bytes).convert(long_b))
        assert isinstance(result, tvm_ffi.core.Bytes)
        assert result == long_b

    def test_empty_bytes_is_bytes(self) -> None:
        """Empty bytes is tvm_ffi.Bytes."""
        result = _to_py_class_value(A(bytes).convert(b""))
        assert isinstance(result, tvm_ffi.core.Bytes)
        assert result == b""

    def test_bytearray_converts_to_ffi_bytes(self) -> None:
        """Bytearray converts to tvm_ffi.Bytes."""
        result = _to_py_class_value(A(bytes).convert(bytearray(b"hello")))
        assert isinstance(result, tvm_ffi.core.Bytes)
        assert result == b"hello"

    def test_callable_is_function(self) -> None:
        """Callable converts to tvm_ffi.Function."""
        result = _to_py_class_value(A(Callable).convert(lambda x: x))
        assert isinstance(result, tvm_ffi.core.Function)

    @requires_py39
    def test_array_is_ffi_array(self) -> None:
        """Array[int] converts to tvm_ffi.Array."""
        result = _to_py_class_value(A(tuple[int, ...]).convert([1, 2]))
        assert isinstance(result, _tvm_ffi.Array)

    @requires_py39
    def test_list_is_ffi_list(self) -> None:
        """List[int] converts to tvm_ffi.List."""
        result = _to_py_class_value(A(list[int]).convert([1, 2]))
        assert isinstance(result, _tvm_ffi.List)

    @requires_py39
    def test_map_is_ffi_map(self) -> None:
        """Map[str, int] converts to tvm_ffi.Map."""
        result = _to_py_class_value(A(tvm_ffi.Map[str, int]).convert({"a": 1}))
        assert isinstance(result, _tvm_ffi.Map)

    @requires_py39
    def test_dict_is_ffi_dict(self) -> None:
        """Dict[str, int] converts to tvm_ffi.Dict."""
        result = _to_py_class_value(A(dict[str, int]).convert({"a": 1}))
        assert isinstance(result, _tvm_ffi.Dict)

    def test_int_is_int(self) -> None:
        """Int stays as int."""
        result = _to_py_class_value(A(int).convert(42))
        assert type(result) is int
        assert result == 42

    def test_float_is_float(self) -> None:
        """Float stays as float."""
        result = _to_py_class_value(A(float).convert(3.14))
        assert type(result) is float
        assert result == 3.14

    def test_bool_is_bool(self) -> None:
        """Bool stays as bool."""
        result = _to_py_class_value(A(bool).convert(True))
        assert result is True

    def test_none_is_none(self) -> None:
        """None stays as None."""
        result = _to_py_class_value(A(None).convert(None))
        assert result is None

    def test_object_is_cobject(self) -> None:
        """Object converts to CObject subclass."""
        obj = TestIntPair(1, 2)
        result = _to_py_class_value(A(TestIntPair).convert(obj))
        assert isinstance(result, tvm_ffi.core.CObject)
        assert result.same_as(obj)


# ---------------------------------------------------------------------------
# Small string/bytes optimization: verify CAny uses inline storage
# ---------------------------------------------------------------------------
# SmallStr/SmallBytes threshold is sizeof(int64_t) - 1 = 7 bytes.
_SMALL_STR_TYPE_INDEX = 11  # kTVMFFISmallStr
_SMALL_BYTES_TYPE_INDEX = 12  # kTVMFFISmallBytes
_STR_TYPE_INDEX = 65  # kTVMFFIStr (heap-allocated String object)
_BYTES_TYPE_INDEX = 66  # kTVMFFIBytes (heap-allocated Bytes object)


class TestSmallStringOptimization:
    """Verify that short str/bytes use SmallStr/SmallBytes inline storage."""

    def test_short_str_is_small_str(self) -> None:
        """A 2-byte string should be stored inline as SmallStr."""
        cany = A(str).convert("hi")
        assert cany.type_index == _SMALL_STR_TYPE_INDEX

    def test_empty_str_is_small_str(self) -> None:
        """An empty string should be stored inline as SmallStr."""
        cany = A(str).convert("")
        assert cany.type_index == _SMALL_STR_TYPE_INDEX

    def test_7_byte_str_is_small_str(self) -> None:
        """A 7-byte ASCII string (max) should be stored inline as SmallStr."""
        cany = A(str).convert("abcdefg")
        assert cany.type_index == _SMALL_STR_TYPE_INDEX

    def test_8_byte_str_is_heap_str(self) -> None:
        """An 8-byte string exceeds SmallStr capacity and uses heap String."""
        cany = A(str).convert("abcdefgh")
        assert cany.type_index == _STR_TYPE_INDEX

    def test_long_str_is_heap_str(self) -> None:
        """A long string uses heap-allocated String object."""
        cany = A(str).convert("x" * 200)
        assert cany.type_index == _STR_TYPE_INDEX

    def test_short_bytes_is_small_bytes(self) -> None:
        """A 2-byte value should be stored inline as SmallBytes."""
        cany = A(bytes).convert(b"hi")
        assert cany.type_index == _SMALL_BYTES_TYPE_INDEX

    def test_empty_bytes_is_small_bytes(self) -> None:
        """Empty bytes should be stored inline as SmallBytes."""
        cany = A(bytes).convert(b"")
        assert cany.type_index == _SMALL_BYTES_TYPE_INDEX

    def test_7_byte_bytes_is_small_bytes(self) -> None:
        """7-byte value (max) should be stored inline as SmallBytes."""
        cany = A(bytes).convert(b"abcdefg")
        assert cany.type_index == _SMALL_BYTES_TYPE_INDEX

    def test_8_byte_bytes_is_heap_bytes(self) -> None:
        """An 8-byte value exceeds SmallBytes capacity and uses heap Bytes."""
        cany = A(bytes).convert(b"abcdefgh")
        assert cany.type_index == _BYTES_TYPE_INDEX

    def test_long_bytes_is_heap_bytes(self) -> None:
        """A long bytes value uses heap-allocated Bytes object."""
        cany = A(bytes).convert(b"x" * 200)
        assert cany.type_index == _BYTES_TYPE_INDEX

    def test_bytearray_short_is_small_bytes(self) -> None:
        """A short bytearray should produce SmallBytes."""
        cany = A(bytes).convert(bytearray(b"hi"))
        assert cany.type_index == _SMALL_BYTES_TYPE_INDEX

    def test_bytearray_long_is_heap_bytes(self) -> None:
        """A long bytearray should produce heap Bytes."""
        cany = A(bytes).convert(bytearray(b"x" * 200))
        assert cany.type_index == _BYTES_TYPE_INDEX

    def test_small_str_roundtrips_as_string(self) -> None:
        """SmallStr round-trips to tvm_ffi.String via _to_py_class_value."""
        cany = A(str).convert("hi")
        assert cany.type_index == _SMALL_STR_TYPE_INDEX
        result = _to_py_class_value(cany)
        assert isinstance(result, tvm_ffi.core.String)
        assert result == "hi"

    def test_small_bytes_roundtrips_as_bytes(self) -> None:
        """SmallBytes round-trips to tvm_ffi.Bytes via _to_py_class_value."""
        cany = A(bytes).convert(b"hi")
        assert cany.type_index == _SMALL_BYTES_TYPE_INDEX
        result = _to_py_class_value(cany)
        assert isinstance(result, tvm_ffi.core.Bytes)
        assert result == b"hi"

    def test_ffi_string_short_repacks_as_small(self) -> None:
        """A short tvm_ffi.String is re-packed as SmallStr by CAny."""
        s = tvm_ffi.core.String("hi")
        cany = A(str).convert(s)
        assert cany.type_index == _SMALL_STR_TYPE_INDEX

    def test_ffi_string_long_stays_heap(self) -> None:
        """A long tvm_ffi.String stays as heap kTVMFFIStr."""
        s = tvm_ffi.core.String("x" * 200)
        cany = A(str).convert(s)
        assert cany.type_index == _STR_TYPE_INDEX

    def test_ffi_bytes_short_repacks_as_small(self) -> None:
        """A short tvm_ffi.Bytes is re-packed as SmallBytes by CAny."""
        b = tvm_ffi.core.Bytes(b"hi")
        cany = A(bytes).convert(b)
        assert cany.type_index == _SMALL_BYTES_TYPE_INDEX

    def test_ffi_bytes_long_stays_heap(self) -> None:
        """A long tvm_ffi.Bytes stays as heap kTVMFFIBytes."""
        b = tvm_ffi.core.Bytes(b"x" * 200)
        cany = A(bytes).convert(b)
        assert cany.type_index == _BYTES_TYPE_INDEX
