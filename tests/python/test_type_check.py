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
"""Tests for TypeSchema type conversion (convert/check_value)."""

from __future__ import annotations

import ctypes
import os
import sys
import typing
from numbers import Integral
from typing import Callable, Optional, Union

import pytest
import tvm_ffi
from tvm_ffi.core import CAny, ObjectConvertible, TypeSchema
from tvm_ffi.testing import (
    TestIntPair,
    TestObjectBase,
    TestObjectDerived,
    _TestCxxClassBase,
    _TestCxxClassDerived,
    _TestCxxClassDerivedDerived,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def S(origin: str, *args: TypeSchema) -> TypeSchema:
    """Shorthand constructor for TypeSchema (string-based)."""
    return TypeSchema(origin, tuple(args))


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
        S("Device").check_value(dev)

    def test_device_fail(self) -> None:
        """Test device fail."""
        with pytest.raises(TypeError):
            S("Device").check_value(42)

    def test_dtype_pass(self) -> None:
        """Test dtype pass."""
        dt = tvm_ffi.core.DataType("float32")
        S("dtype").check_value(dt)

    def test_dtype_str_pass(self) -> None:
        """Str accepted as dtype (will be parsed)."""
        S("dtype").check_value("float32")

    def test_dtype_fail(self) -> None:
        """Test dtype fail."""
        with pytest.raises(TypeError):
            S("dtype").check_value(42)

    def test_opaque_ptr_pass(self) -> None:
        """Test opaque ptr pass."""
        S("ctypes.c_void_p").check_value(ctypes.c_void_p(0))

    def test_opaque_ptr_none_pass(self) -> None:
        """Test opaque ptr none pass."""
        S("ctypes.c_void_p").check_value(None)

    def test_opaque_ptr_fail(self) -> None:
        """Test opaque ptr fail."""
        with pytest.raises(TypeError):
            S("ctypes.c_void_p").check_value(42)

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


# ---------------------------------------------------------------------------
# Category 5: Object types
# ---------------------------------------------------------------------------
class TestObjectTypes:
    def test_object_pass(self) -> None:
        """Any CObject passes TypeSchema('Object')."""
        f = tvm_ffi.get_global_func("testing.echo")
        S("Object").check_value(f)

    def test_object_fail(self) -> None:
        """Test object fail."""
        with pytest.raises(TypeError):
            S("Object").check_value(42)

    def test_specific_object_pass(self) -> None:
        """A Function object should pass its own type schema."""
        f = tvm_ffi.get_global_func("testing.echo")
        A(Callable).check_value(f)


# ---------------------------------------------------------------------------
# Category 6: Optional
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
        schema = S("Optional", S("Optional", S("int")))
        schema.check_value(None)
        schema.check_value(42)


# ---------------------------------------------------------------------------
# Category 7: Union / Variant
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
# Category 8: Containers
# ---------------------------------------------------------------------------
class TestContainers:
    def test_array_list_pass(self) -> None:
        """Test array list pass."""
        A(tuple[int, ...]).check_value([1, 2, 3])

    def test_array_tuple_pass(self) -> None:
        """Test array tuple pass."""
        A(tuple[int, ...]).check_value((1, 2, 3))

    def test_array_wrong_element(self) -> None:
        """Test array wrong element."""
        with pytest.raises(TypeError, match=r"element \[1\].*expected int"):
            A(tuple[int, ...]).check_value([1, "x"])

    def test_array_empty_pass(self) -> None:
        """Test array empty pass."""
        A(tuple[int, ...]).check_value([])

    def test_array_any_pass(self) -> None:
        """Test array any pass."""
        A(tuple[typing.Any, ...]).check_value([1, "x", None])

    def test_array_wrong_container_type(self) -> None:
        """Test array wrong container type."""
        with pytest.raises(TypeError, match="expected Array"):
            A(tuple[int, ...]).check_value(42)

    def test_list_pass(self) -> None:
        """Test list pass."""
        A(list[str]).check_value(["a", "b"])

    def test_map_pass(self) -> None:
        """Test map pass."""
        S("Map", S("str"), S("int")).check_value({"a": 1, "b": 2})

    def test_map_wrong_key(self) -> None:
        """Test map wrong key."""
        with pytest.raises(TypeError, match="expected str"):
            S("Map", S("str"), S("int")).check_value({1: 2})

    def test_map_wrong_value(self) -> None:
        """Test map wrong value."""
        with pytest.raises(TypeError, match="expected int"):
            S("Map", S("str"), S("int")).check_value({"a": "b"})

    def test_map_empty_pass(self) -> None:
        """Test map empty pass."""
        S("Map", S("str"), S("int")).check_value({})

    def test_dict_pass(self) -> None:
        """Test dict pass."""
        A(dict[str, int]).check_value({"a": 1})

    def test_map_wrong_container(self) -> None:
        """Test map wrong container."""
        with pytest.raises(TypeError, match="expected Map"):
            S("Map", S("str"), S("int")).check_value([1, 2])


# ---------------------------------------------------------------------------
# Category 9: Nested types
# ---------------------------------------------------------------------------
class TestNestedTypes:
    def test_array_optional_int(self) -> None:
        """Test array optional int."""
        A(tuple[Optional[int], ...]).check_value([1, None, 2])

    def test_map_str_array_int(self) -> None:
        """Test map str array int."""
        S("Map", S("str"), S("Array", S("int"))).check_value({"a": [1, 2]})

    def test_map_str_array_int_nested_fail(self) -> None:
        """Test map str array int nested fail."""
        with pytest.raises(TypeError, match="expected int"):
            S("Map", S("str"), S("Array", S("int"))).check_value({"a": [1, "x"]})

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


# ---------------------------------------------------------------------------
# Category 11: Error message quality
# ---------------------------------------------------------------------------
class TestErrorMessages:
    def test_basic_type_mismatch(self) -> None:
        """Test basic type mismatch."""
        with pytest.raises(TypeError, match=r"expected int, got str"):
            A(int).check_value("hello")

    def test_nested_array_error(self) -> None:
        """Test nested array error."""
        with pytest.raises(TypeError, match=r"element \[2\].*expected int, got str"):
            A(tuple[int, ...]).check_value([1, 2, "x"])

    def test_nested_map_error(self) -> None:
        """Test nested map error."""
        with pytest.raises(TypeError, match=r"value for key 'b'.*expected int, got str"):
            S("Map", S("str"), S("int")).check_value({"a": 1, "b": "x"})

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
        schema = TypeSchema.from_type_index(71, (S("int"),))  # kTVMFFIArray
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
        schema = TypeSchema.from_type_index(72, (S("str"), S("int")))  # kTVMFFIMap
        assert schema.origin == "Map"
        schema.check_value({"a": 1})


# ---------------------------------------------------------------------------
# Category 13: Edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_bytearray_passes_bytes(self) -> None:
        """Test bytearray passes bytes."""
        A(bytes).check_value(bytearray(b"data"))

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
        result = A(int).convert(42).to_py()
        assert result == 42
        assert type(result) is int

    def test_bool_to_int(self) -> None:
        """Bool -> int actually converts to int."""
        result = A(int).convert(True).to_py()
        assert result == 1
        assert type(result) is int

    def test_bool_false_to_int(self) -> None:
        """Test bool false to int."""
        result = A(int).convert(False).to_py()
        assert result == 0
        assert type(result) is int

    def test_float_passthrough(self) -> None:
        """Test float passthrough."""
        result = A(float).convert(3.14).to_py()
        assert result == 3.14
        assert type(result) is float

    def test_int_to_float(self) -> None:
        """Int -> float actually converts."""
        result = A(float).convert(42).to_py()
        assert result == 42.0
        assert type(result) is float

    def test_bool_to_float(self) -> None:
        """Bool -> float actually converts."""
        result = A(float).convert(True).to_py()
        assert result == 1.0
        assert type(result) is float

    def test_bool_passthrough(self) -> None:
        """Test bool passthrough."""
        result = A(bool).convert(True).to_py()
        assert result is True
        assert type(result) is bool

    def test_int_to_bool(self) -> None:
        """Int -> bool actually converts."""
        result = A(bool).convert(1).to_py()
        assert result is True
        assert type(result) is bool

    def test_int_zero_to_bool(self) -> None:
        """Test int zero to bool."""
        result = A(bool).convert(0).to_py()
        assert result is False
        assert type(result) is bool

    def test_str_passthrough(self) -> None:
        """Test str passthrough — returns tvm_ffi.String (subclass of str)."""
        result = A(str).convert("hello").to_py()
        assert result == "hello"
        assert isinstance(result, str)
        assert isinstance(result, tvm_ffi.core.String)

    def test_bytes_passthrough(self) -> None:
        """Test bytes passthrough — returns tvm_ffi.Bytes (subclass of bytes)."""
        result = A(bytes).convert(b"data").to_py()
        assert result == b"data"
        assert isinstance(result, bytes)
        assert isinstance(result, tvm_ffi.core.Bytes)

    def test_bytearray_to_bytes(self) -> None:
        """Bytearray -> bytes converts to tvm_ffi.Bytes."""
        result = A(bytes).convert(bytearray(b"data")).to_py()
        assert result == b"data"
        assert isinstance(result, bytes)
        assert isinstance(result, tvm_ffi.core.Bytes)


# ---------------------------------------------------------------------------
# Category 15: None disambiguation (critical design point)
# ---------------------------------------------------------------------------
class TestNoneDisambiguation:
    def test_none_converts_successfully_for_none_schema(self) -> None:
        """TypeSchema('None').convert(None) returns None as a valid result."""
        result = A(type(None)).convert(None).to_py()
        assert result is None

    def test_none_converts_successfully_for_optional(self) -> None:
        """Optional[int].convert(None) returns None as a valid result."""
        result = A(Optional[int]).convert(None).to_py()
        assert result is None

    def test_none_fails_for_int(self) -> None:
        """TypeSchema('int').convert(None) raises TypeError."""
        with pytest.raises(TypeError, match="expected int, got None"):
            A(int).convert(None)

    def test_convert_none_success(self) -> None:
        """Convert returns None for Optional[int] with None input."""
        result = A(Optional[int]).convert(None).to_py()
        assert result is None

    def test_convert_none_failure(self) -> None:
        """Convert raises TypeError for failed conversion."""
        with pytest.raises(TypeError, match="expected int"):
            A(int).convert(None)

    def test_convert_success_with_value(self) -> None:
        """Convert returns converted value on success."""
        result = A(int).convert(True).to_py()
        assert result == 1
        assert type(result) is int

    def test_opaque_ptr_none_converts(self) -> None:
        """ctypes.c_void_p accepts None and returns None as valid result."""
        result = S("ctypes.c_void_p").convert(None).to_py()
        assert result is None

    def test_convert_opaque_ptr_none(self) -> None:
        """Test convert opaque ptr none."""
        result = S("ctypes.c_void_p").convert(None).to_py()
        assert result is None


# ---------------------------------------------------------------------------
# Category 16: Special type conversions
# ---------------------------------------------------------------------------
class TestConvertSpecialTypes:
    def test_dtype_str_converts(self) -> None:
        """Str -> dtype actually creates a DataType object."""
        result = S("dtype").convert("float32").to_py()
        assert isinstance(result, tvm_ffi.core.DataType)
        assert str(result) == "float32"

    def test_dtype_passthrough(self) -> None:
        """Test dtype passthrough."""
        dt = tvm_ffi.core.DataType("int32")
        result = S("dtype").convert(dt).to_py()
        assert str(result) == str(dt)

    def test_device_passthrough(self) -> None:
        """Test device passthrough."""
        dev = tvm_ffi.Device("cpu", 0)
        result = S("Device").convert(dev).to_py()
        assert str(result) == str(dev)

    def test_callable_passthrough(self) -> None:
        """Test callable passthrough."""
        fn = lambda x: x
        result = A(Callable).convert(fn).to_py()
        assert callable(result)

    def test_opaque_ptr_passthrough(self) -> None:
        """Test opaque ptr passthrough."""
        ptr = ctypes.c_void_p(42)
        result = S("ctypes.c_void_p").convert(ptr).to_py()
        assert result is not None


# ---------------------------------------------------------------------------
# Category 17: Container conversion results
# ---------------------------------------------------------------------------
class TestConvertContainers:
    def test_array_converts_bool_elements_to_int(self) -> None:
        """Array[int] with bool elements converts them to int."""
        result = A(tuple[int, ...]).convert([True, False, 1]).to_py()
        assert list(result) == [1, 0, 1]
        assert all(type(x) is int for x in result)

    def test_array_int_passthrough(self) -> None:
        """Array[int] with int elements returns ffi.Array."""
        result = A(tuple[int, ...]).convert([1, 2, 3]).to_py()
        assert list(result) == [1, 2, 3]

    def test_array_any_passthrough(self) -> None:
        """Array[Any] wraps into ffi.Array."""
        original = [1, "x", None]
        result = A(tuple[typing.Any, ...]).convert(original).to_py()
        assert isinstance(result, tvm_ffi.Array)

    def test_map_converts_values(self) -> None:
        """Map[str, float] converts int values to float."""
        result = S("Map", S("str"), S("float")).convert({"a": 1, "b": 2}).to_py()
        assert isinstance(result, tvm_ffi.Map)
        assert type(result["a"]) is float
        assert type(result["b"]) is float
        assert result["a"] == 1.0
        assert result["b"] == 2.0

    def test_map_any_any_passthrough(self) -> None:
        """Map[Any, Any] wraps into ffi.Map."""
        original = {"a": 1}
        result = S("Map", S("Any"), S("Any")).convert(original).to_py()
        assert isinstance(result, tvm_ffi.Map)

    def test_tuple_converts_elements(self) -> None:
        """tuple[int, float] converts elements positionally."""
        result = A(tuple[int, float]).convert((True, 42)).to_py()
        assert list(result) == [1, 42.0]
        assert type(result[0]) is int
        assert type(result[1]) is float

    def test_nested_array_in_map(self) -> None:
        """Map[str, Array[int]] recursively converts elements."""
        result = S("Map", S("str"), S("Array", S("int"))).convert({"a": [True, False]}).to_py()
        assert isinstance(result, tvm_ffi.Map)
        assert list(result["a"]) == [1, 0]
        assert all(type(x) is int for x in result["a"])


# ---------------------------------------------------------------------------
# Category 18: Optional/Union conversion results
# ---------------------------------------------------------------------------
class TestConvertComposite:
    def test_optional_converts_inner(self) -> None:
        """Optional[float].convert(42) converts int -> float."""
        result = A(Optional[float]).convert(42).to_py()
        assert result == 42.0
        assert type(result) is float

    def test_optional_none(self) -> None:
        """Test optional none."""
        result = A(Optional[float]).convert(None).to_py()
        assert result is None

    def test_union_picks_first_match(self) -> None:
        """Union[int, str] converts bool via int alternative."""
        result = A(Union[int, str]).convert(True).to_py()
        assert result == 1
        assert type(result) is int

    def test_union_second_match(self) -> None:
        """Test union second match."""
        result = A(Union[int, str]).convert("hello").to_py()
        assert result == "hello"

    def test_any_passthrough(self) -> None:
        """Any returns value as-is."""
        result = A(typing.Any).convert(42).to_py()
        assert result == 42
        result = A(typing.Any).convert(None).to_py()
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

    def test_array_rejects_wrong_element(self) -> None:
        """Test array rejects wrong element."""
        with pytest.raises(TypeError, match=r"element \[1\].*expected int, got str"):
            A(tuple[int, ...]).convert([1, "x"])

    def test_map_rejects_wrong_value(self) -> None:
        """Test map rejects wrong value."""
        with pytest.raises(TypeError, match=r"value for key 'a'.*expected int, got str"):
            S("Map", S("str"), S("int")).convert({"a": "x"})

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
        result = A(int).convert(np.int64(42)).to_py()
        assert result == 42
        assert type(result) is int

    def test_numpy_float_to_float(self) -> None:
        """Test numpy float to float."""
        np = pytest.importorskip("numpy")
        result = A(float).convert(np.float64(3.14)).to_py()
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
    def test_array_optional_float_with_bool(self) -> None:
        """Array[Optional[float]] converts bool elements to float."""
        result = A(tuple[Optional[float], ...]).convert([True, None, 3]).to_py()
        assert list(result) == [1.0, None, 3.0]
        assert type(result[0]) is float
        assert result[1] is None
        assert type(result[2]) is float

    def test_array_optional_int_with_bool(self) -> None:
        """Array[Optional[int]] converts bool elements to int."""
        result = A(tuple[Optional[int], ...]).convert([True, None, 2]).to_py()
        assert list(result) == [1, None, 2]
        assert type(result[0]) is int
        assert result[1] is None

    def test_array_union_int_str_with_bool(self) -> None:
        """Array[Union[int, str]] converts bool via int alternative."""
        result = A(tuple[Union[int, str], ...]).convert([True, "hello", False]).to_py()
        assert list(result) == [1, "hello", 0]
        assert type(result[0]) is int
        assert type(result[1]) is str
        assert type(result[2]) is int

    def test_array_union_float_str_with_int(self) -> None:
        """Array[Union[float, str]] converts int via float alternative."""
        result = A(tuple[Union[float, str], ...]).convert([42, "hi", True]).to_py()
        assert list(result) == [42.0, "hi", 1.0]
        assert type(result[0]) is float
        assert type(result[2]) is float

    def test_array_optional_float_all_none(self) -> None:
        """Array[Optional[float]] with all None elements."""
        result = A(tuple[Optional[float], ...]).convert([None, None]).to_py()
        assert list(result) == [None, None]

    def test_array_optional_float_empty(self) -> None:
        """Array[Optional[float]] with empty list."""
        result = A(tuple[Optional[float], ...]).convert([]).to_py()
        assert list(result) == []

    def test_array_union_failure_in_element(self) -> None:
        """Array[Union[int, str]] fails when element matches no alternative."""
        with pytest.raises(TypeError, match=r"element \[1\].*got float"):
            A(tuple[Union[int, str], ...]).check_value([1, 3.14])


# ---------------------------------------------------------------------------
# Category 22: Map/Dict nested with Optional/Union (inner conversion)
# ---------------------------------------------------------------------------
class TestNestedMapComposite:
    def test_map_str_optional_float_with_int(self) -> None:
        """Map[str, Optional[float]] converts int values to float."""
        result = S("Map", S("str"), S("Optional", S("float"))).convert({"a": 1, "b": None}).to_py()
        assert type(result["a"]) is float
        assert result["a"] == 1.0
        assert result["b"] is None

    def test_map_str_union_int_str(self) -> None:
        """Map[str, Union[int, str]] converts bool values via int."""
        result = (
            S("Map", S("str"), S("Union", S("int"), S("str")))
            .convert({"x": True, "y": "hello"})
            .to_py()
        )
        assert result["x"] == 1
        assert result["y"] == "hello"
        assert type(result["x"]) is int

    def test_dict_str_optional_int(self) -> None:
        """Dict[str, Optional[int]] with bool conversion."""
        result = A(dict[str, Optional[int]]).convert({"a": True, "b": None, "c": 42}).to_py()
        assert result["a"] == 1
        assert result["b"] is None
        assert result["c"] == 42
        assert type(result["a"]) is int

    def test_map_str_optional_float_failure(self) -> None:
        """Map[str, Optional[float]] fails for non-float non-None value."""
        with pytest.raises(TypeError, match="expected float"):
            S("Map", S("str"), S("Optional", S("float"))).check_value({"a": "bad"})


# ---------------------------------------------------------------------------
# Category 23: Nested containers (container inside container)
# ---------------------------------------------------------------------------
class TestNestedContainerInContainer:
    def test_array_of_array_int(self) -> None:
        """Array[Array[int]] with inner bool->int conversion."""
        result = A(tuple[tuple[int, ...], ...]).convert([[True, False], [1, 2]]).to_py()
        assert [list(row) for row in result] == [[1, 0], [1, 2]]
        assert all(type(x) is int for row in result for x in row)

    def test_array_of_array_float(self) -> None:
        """Array[Array[float]] with inner int->float conversion."""
        result = A(tuple[tuple[float, ...], ...]).convert([[1, 2], [True, 3]]).to_py()
        assert [list(row) for row in result] == [[1.0, 2.0], [1.0, 3.0]]
        assert all(type(x) is float for row in result for x in row)

    def test_map_str_array_float(self) -> None:
        """Map[str, Array[float]] with int->float conversion in arrays."""
        result = (
            S("Map", S("str"), S("Array", S("float")))
            .convert({"a": [1, 2], "b": [True, 3]})
            .to_py()
        )
        assert list(result["a"]) == [1.0, 2.0]
        assert list(result["b"]) == [1.0, 3.0]
        assert all(type(x) is float for x in result["a"])
        assert all(type(x) is float for x in result["b"])

    def test_dict_str_array_int(self) -> None:
        """Dict[str, Array[int]] with bool->int conversion."""
        result = A(dict[str, tuple[int, ...]]).convert({"a": [True, False]}).to_py()
        assert list(result["a"]) == [1, 0]
        assert all(type(x) is int for x in result["a"])

    def test_array_of_map_str_int(self) -> None:
        """Array[Map[str, int]] with bool->int value conversion."""
        result = S("Array", S("Map", S("str"), S("int"))).convert([{"x": True}, {"y": 2}]).to_py()
        assert result[0]["x"] == 1
        assert result[1]["y"] == 2
        assert type(result[0]["x"]) is int

    def test_map_str_map_str_float(self) -> None:
        """Map[str, Map[str, float]] double nested with int->float."""
        result = (
            S("Map", S("str"), S("Map", S("str"), S("float")))
            .convert({"outer": {"inner": 42}})
            .to_py()
        )
        assert result["outer"]["inner"] == 42.0
        assert type(result["outer"]["inner"]) is float

    def test_list_of_list_int(self) -> None:
        """List[List[int]] with bool->int conversion."""
        result = A(list[list[int]]).convert([[True, 1], [False, 2]]).to_py()
        assert [list(row) for row in result] == [[1, 1], [0, 2]]
        assert all(type(x) is int for row in result for x in row)

    def test_nested_failure_array_of_array(self) -> None:
        """Array[Array[int]] error propagation through nested arrays."""
        with pytest.raises(TypeError, match="expected int"):
            A(tuple[tuple[int, ...], ...]).check_value([[1, 2], [3, "bad"]])

    def test_empty_inner_containers(self) -> None:
        """Map[str, Array[int]] with empty inner arrays."""
        result = S("Map", S("str"), S("Array", S("int"))).convert({"a": [], "b": []}).to_py()
        assert list(result["a"]) == []
        assert list(result["b"]) == []


# ---------------------------------------------------------------------------
# Category 24: Optional/Union wrapping containers
# ---------------------------------------------------------------------------
class TestOptionalUnionWrappingContainers:
    def test_optional_array_int_with_conversion(self) -> None:
        """Optional[Array[int]] converts inner bool elements."""
        schema = A(Optional[tuple[int, ...]])
        result = schema.convert([True, 2]).to_py()
        assert list(result) == [1, 2]
        assert type(result[0]) is int

    def test_optional_array_int_none(self) -> None:
        """Optional[Array[int]] accepts None."""
        result = A(Optional[tuple[int, ...]]).convert(None).to_py()
        assert result is None

    def test_optional_map_str_float(self) -> None:
        """Optional[Map[str, float]] converts inner int values."""
        result = S("Optional", S("Map", S("str"), S("float"))).convert({"a": 1}).to_py()
        assert result["a"] == 1.0
        assert type(result["a"]) is float

    def test_optional_map_str_float_none(self) -> None:
        """Optional[Map[str, float]] accepts None."""
        result = S("Optional", S("Map", S("str"), S("float"))).convert(None).to_py()
        assert result is None

    def test_union_array_int_or_map_str_int(self) -> None:
        """Union[Array[int], Map[str, int]] matches first with conversion."""
        schema = S("Union", S("Array", S("int")), S("Map", S("str"), S("int")))
        # list matches Array alternative
        result = schema.convert([True, 2]).to_py()
        assert list(result) == [1, 2]
        assert type(result[0]) is int

    def test_union_array_int_or_map_str_int_dict(self) -> None:
        """Union[Array[int], Map[str, int]] matches Map for dict input."""
        schema = S("Union", S("Array", S("int")), S("Map", S("str"), S("int")))
        result = schema.convert({"a": True}).to_py()
        assert result["a"] == 1
        assert type(result["a"]) is int

    def test_union_int_or_array_optional_float(self) -> None:
        """Union[int, Array[Optional[float]]] matches array with nested conversions."""
        schema = S("Union", S("int"), S("Array", S("Optional", S("float"))))
        result = schema.convert([True, None, 1]).to_py()
        assert list(result) == [1.0, None, 1.0]
        assert type(result[0]) is float
        assert result[1] is None

    def test_optional_optional_array_int(self) -> None:
        """Optional[Optional[Array[int]]] with inner conversion."""
        schema = S("Optional", S("Optional", S("Array", S("int"))))
        assert schema.convert(None).to_py() is None
        result = schema.convert([True, 2]).to_py()
        assert list(result) == [1, 2]
        assert type(result[0]) is int


# ---------------------------------------------------------------------------
# Category 25: Tuple nested with other types
# ---------------------------------------------------------------------------
class TestNestedTuple:
    def test_array_of_tuple_int_float(self) -> None:
        """Array[tuple[int, float]] with element-wise conversion."""
        result = (
            S("Array", S("tuple", S("int"), S("float"))).convert([(True, 1), (2, True)]).to_py()
        )
        # Check element values; FFI storage may normalize float 1.0 to int 1
        # when stored inside an ffi.Array, so we only check values not types.
        assert result[0][0] == 1
        assert result[0][1] == 1.0
        assert result[1][0] == 2
        assert result[1][1] == 1.0

    def test_map_str_tuple_int_str(self) -> None:
        """Map[str, tuple[int, str]] with inner bool->int conversion."""
        result = (
            S("Map", S("str"), S("tuple", S("int"), S("str")))
            .convert({"a": (True, "hello")})
            .to_py()
        )
        assert result["a"][0] == 1
        assert str(result["a"][1]) == "hello"
        assert type(result["a"][0]) is int

    def test_tuple_of_array_int_and_map(self) -> None:
        """tuple[Array[int], Map[str, float]] nested conversion."""
        schema = S("tuple", S("Array", S("int")), S("Map", S("str"), S("float")))
        result = schema.convert(([True, 2], {"k": 3})).to_py()
        assert list(result[0]) == [1, 2]
        assert result[1]["k"] == 3.0
        assert type(result[0][0]) is int
        assert type(result[1]["k"]) is float

    def test_tuple_of_optional_int_and_optional_float(self) -> None:
        """tuple[Optional[int], Optional[float]] with conversions."""
        schema = S("tuple", S("Optional", S("int")), S("Optional", S("float")))
        result = schema.convert((True, None)).to_py()
        assert list(result) == [1, None]
        assert type(result[0]) is int
        assert result[1] is None

    def test_tuple_nested_failure(self) -> None:
        """tuple[Array[int], str] error propagation from inner array."""
        with pytest.raises(TypeError, match=r"element .0..*element .1..*expected int"):
            S("tuple", S("Array", S("int")), S("str")).check_value(([1, "bad"], "ok"))


# ---------------------------------------------------------------------------
# Category 26: Deep nesting (3+ levels)
# ---------------------------------------------------------------------------
class TestDeepNesting:
    def test_map_str_array_optional_int(self) -> None:
        """Map[str, Array[Optional[int]]] with 3-level nesting and conversion."""
        result = (
            S("Map", S("str"), S("Array", S("Optional", S("int"))))
            .convert({"a": [1, None, True]})
            .to_py()
        )
        assert list(result["a"]) == [1, None, 1]
        assert type(result["a"][0]) is int
        assert result["a"][1] is None
        assert type(result["a"][2]) is int

    def test_array_map_str_optional_float(self) -> None:
        """Array[Map[str, Optional[float]]] with 3-level nesting."""
        result = (
            S("Array", S("Map", S("str"), S("Optional", S("float"))))
            .convert([{"x": 1, "y": None}, {"z": True}])
            .to_py()
        )
        assert result[0]["x"] == 1.0
        assert result[0]["y"] is None
        assert result[1]["z"] == 1.0
        assert type(result[0]["x"]) is float
        assert type(result[1]["z"]) is float

    def test_optional_array_map_str_int(self) -> None:
        """Optional[Array[Map[str, int]]] 3 levels deep."""
        schema = S("Optional", S("Array", S("Map", S("str"), S("int"))))
        result = schema.convert([{"a": True}, {"b": 2}]).to_py()
        assert result[0]["a"] == 1
        assert result[1]["b"] == 2
        assert type(result[0]["a"]) is int

        assert schema.convert(None).to_py() is None

    def test_map_str_array_array_int(self) -> None:
        """Map[str, Array[Array[int]]] 3-level container nesting."""
        result = (
            S("Map", S("str"), S("Array", S("Array", S("int"))))
            .convert({"m": [[True, 1], [False, 2]]})
            .to_py()
        )
        assert [list(row) for row in result["m"]] == [[1, 1], [0, 2]]
        assert all(type(x) is int for row in result["m"] for x in row)

    def test_array_array_optional_float(self) -> None:
        """Array[Array[Optional[float]]] deep nesting with None and conversion."""
        result = (
            S("Array", S("Array", S("Optional", S("float"))))
            .convert([[1, None], [True, 3.14]])
            .to_py()
        )
        assert list(result[0]) == [1.0, None]
        assert list(result[1]) == [1.0, 3.14]
        assert type(result[0][0]) is float
        assert result[0][1] is None
        assert type(result[1][0]) is float

    def test_deep_nesting_failure_propagation(self) -> None:
        """Error from deepest level propagates with full path info."""
        with pytest.raises(TypeError, match=r"value for key 'key'.*element .1..*expected int"):
            S("Map", S("str"), S("Array", S("Optional", S("int")))).check_value({"key": [1, "bad"]})


# ---------------------------------------------------------------------------
# Category 27: FFI container inputs (tvm_ffi.Array/List/Map/Dict)
# ---------------------------------------------------------------------------
class TestFFIContainerInputs:
    def test_ffi_array_with_element_conversion(self) -> None:
        """tvm_ffi.Array([True, 2]) passes Array[int] with bool->int conversion."""
        arr = tvm_ffi.Array([True, 2, 3])
        result = S("Array", S("int")).convert(arr).to_py()
        assert list(result) == [1, 2, 3]
        assert type(result[0]) is int

    def test_ffi_array_any_passthrough(self) -> None:
        """tvm_ffi.Array passes Array[Any] as-is."""
        arr = tvm_ffi.Array([1, "x", None])
        result = S("Array", S("Any")).convert(arr).to_py()
        assert result.same_as(arr)

    def test_ffi_list_with_list_schema(self) -> None:
        """tvm_ffi.List passes List[int] with conversion."""
        lst = tvm_ffi.List([True, 2])
        result = S("List", S("int")).convert(lst).to_py()
        assert list(result) == [1, 2]
        assert type(result[0]) is int

    def test_ffi_list_accepted_by_array_schema(self) -> None:
        """tvm_ffi.List passes Array schema (C++ allows cross-type via kOtherTypeIndex)."""
        lst = tvm_ffi.List([1, 2])
        S("Array", S("int")).check_value(lst)

    def test_ffi_array_accepted_by_list_schema(self) -> None:
        """tvm_ffi.Array passes List schema (C++ allows cross-type via kOtherTypeIndex)."""
        arr = tvm_ffi.Array([1, 2])
        S("List", S("int")).check_value(arr)

    def test_ffi_map_with_value_conversion(self) -> None:
        """tvm_ffi.Map passes Map[str, int] with bool->int conversion."""
        m = tvm_ffi.Map({"a": True, "b": 2})
        result = S("Map", S("str"), S("int")).convert(m).to_py()
        assert result["a"] == 1
        assert result["b"] == 2
        assert type(result["a"]) is int

    def test_ffi_map_any_any_passthrough(self) -> None:
        """tvm_ffi.Map passes Map[Any, Any] as-is."""
        m = tvm_ffi.Map({"a": 1})
        result = S("Map", S("Any"), S("Any")).convert(m).to_py()
        assert result.same_as(m)

    def test_ffi_dict_with_dict_schema(self) -> None:
        """tvm_ffi.Dict passes Dict[str, float] with int->float conversion."""
        d = tvm_ffi.Dict({"x": 1, "y": 2})
        result = S("Dict", S("str"), S("float")).convert(d).to_py()
        assert result["x"] == 1.0
        assert result["y"] == 2.0
        assert type(result["x"]) is float

    def test_ffi_dict_accepted_by_map_schema(self) -> None:
        """tvm_ffi.Dict passes Map schema (C++ allows cross-type via kOtherTypeIndex)."""
        d = tvm_ffi.Dict({"a": 1})
        S("Map", S("str"), S("int")).check_value(d)

    def test_ffi_map_accepted_by_dict_schema(self) -> None:
        """tvm_ffi.Map passes Dict schema (C++ allows cross-type via kOtherTypeIndex)."""
        m = tvm_ffi.Map({"a": 1})
        S("Dict", S("str"), S("int")).check_value(m)

    def test_ffi_array_nested_optional_float(self) -> None:
        """tvm_ffi.Array with nested Optional[float] conversion."""
        arr = tvm_ffi.Array([1, None, True])
        result = S("Array", S("Optional", S("float"))).convert(arr).to_py()
        assert list(result) == [1.0, None, 1.0]
        assert type(result[0]) is float
        assert result[1] is None

    def test_ffi_map_nested_array_int(self) -> None:
        """tvm_ffi.Map with value being a Python list, converted as Array[int]."""
        # Map values are already stored; create a map with array values
        m = tvm_ffi.Map({"k": tvm_ffi.Array([True, 2])})
        result = S("Map", S("str"), S("Array", S("int"))).convert(m).to_py()
        assert list(result["k"]) == [1, 2]
        assert type(result["k"][0]) is int

    def test_ffi_array_wrong_element_type(self) -> None:
        """tvm_ffi.Array with wrong element type gives clear error."""
        arr = tvm_ffi.Array([1, "bad", 3])
        with pytest.raises(TypeError, match=r"element \[1\].*expected int"):
            S("Array", S("int")).check_value(arr)

    def test_ffi_map_wrong_value_type(self) -> None:
        """tvm_ffi.Map with wrong value type gives clear error."""
        m = tvm_ffi.Map({"a": 1, "b": "bad"})
        with pytest.raises(TypeError, match=r"value for key.*expected int"):
            S("Map", S("str"), S("int")).check_value(m)

    def test_ffi_array_object_schema(self) -> None:
        """tvm_ffi.Array passes Object schema (it is a CObject)."""
        arr = tvm_ffi.Array([1, 2])
        S("Object").check_value(arr)

    def test_ffi_map_object_schema(self) -> None:
        """tvm_ffi.Map passes Object schema (it is a CObject)."""
        m = tvm_ffi.Map({"a": 1})
        S("Object").check_value(m)


# ---------------------------------------------------------------------------
# Category 28: Mixed Python and FFI containers in nesting
# ---------------------------------------------------------------------------
class TestMixedPythonFFIContainers:
    def test_python_list_of_ffi_arrays(self) -> None:
        """Python list containing tvm_ffi.Array elements, Array[Array[int]]."""
        inner1 = tvm_ffi.Array([True, 2])
        inner2 = tvm_ffi.Array([3, False])
        result = S("Array", S("Array", S("int"))).convert([inner1, inner2]).to_py()
        assert [list(row) for row in result] == [[1, 2], [3, 0]]

    def test_python_dict_with_ffi_array_values(self) -> None:
        """Python dict with tvm_ffi.Array values, Map[str, Array[float]]."""
        val = tvm_ffi.Array([1, True])
        result = S("Map", S("str"), S("Array", S("float"))).convert({"k": val}).to_py()
        assert list(result["k"]) == [1.0, 1.0]
        assert all(type(x) is float for x in result["k"])

    def test_ffi_map_with_python_list_in_union(self) -> None:
        """Union[Map[str, int], Array[int]] with tvm_ffi.Map input."""
        schema = S("Union", S("Map", S("str"), S("int")), S("Array", S("int")))
        m = tvm_ffi.Map({"a": True})
        result = schema.convert(m).to_py()
        assert result["a"] == 1
        assert type(result["a"]) is int

    def test_ffi_array_in_optional(self) -> None:
        """Optional[Array[int]] with tvm_ffi.Array input."""
        arr = tvm_ffi.Array([True, 2])
        result = S("Optional", S("Array", S("int"))).convert(arr).to_py()
        assert list(result) == [1, 2]
        assert type(result[0]) is int


# ---------------------------------------------------------------------------
# Category 29: Error propagation through deeply nested FFI containers
# ---------------------------------------------------------------------------
class TestNestedErrorPropagation:
    def test_array_array_int_inner_failure(self) -> None:
        """Error path: Array[Array[int]] -> element [1] -> element [0]."""
        with pytest.raises(TypeError, match=r"element \[1\].*element \[0\].*expected int, got str"):
            S("Array", S("Array", S("int"))).convert([[1], ["bad"]])

    def test_map_array_int_inner_failure(self) -> None:
        """Error path: Map -> value for key 'k' -> element [2]."""
        with pytest.raises(
            TypeError,
            match=r"value for key 'k'.*element \[2\].*expected int, got str",
        ):
            S("Map", S("str"), S("Array", S("int"))).convert({"k": [1, 2, "bad"]})

    def test_array_map_int_inner_failure(self) -> None:
        """Error path: Array -> element [0] -> value for key 'x'."""
        with pytest.raises(
            TypeError,
            match=r"element \[0\].*value for key 'x'.*expected int, got str",
        ):
            S("Array", S("Map", S("str"), S("int"))).convert([{"x": "bad"}])

    def test_optional_array_int_inner_failure(self) -> None:
        """Error path through Optional -> Array -> element."""
        with pytest.raises(TypeError, match=r"element \[1\].*expected int, got str"):
            S("Optional", S("Array", S("int"))).convert([1, "bad"])

    def test_tuple_array_int_inner_failure(self) -> None:
        """Error path: tuple -> element [0] -> element [1]."""
        with pytest.raises(TypeError, match=r"element \[0\].*element \[1\].*expected int, got str"):
            S("tuple", S("Array", S("int")), S("str")).convert(([1, "bad"], "ok"))

    def test_deep_3_level_error(self) -> None:
        """Error at 3 levels deep: Map -> Array -> Optional -> type mismatch."""
        with pytest.raises(TypeError, match=r"value for key 'key'.*element .1..*expected int"):
            S("Map", S("str"), S("Array", S("Optional", S("int")))).check_value({"key": [1, "bad"]})

    def test_ffi_array_nested_error(self) -> None:
        """Error from tvm_ffi.Array in nested context."""
        arr = tvm_ffi.Array([1, "bad", 3])
        with pytest.raises(TypeError, match=r"element \[1\].*expected int"):
            S("Array", S("int")).convert(arr)


# ---------------------------------------------------------------------------
# Category 30: Custom object type exact match
# ---------------------------------------------------------------------------
class TestCustomObjectExactMatch:
    def test_test_int_pair_pass(self) -> None:
        """TestIntPair passes TypeSchema('testing.TestIntPair')."""
        obj = TestIntPair(1, 2)
        S("testing.TestIntPair").check_value(obj)

    def test_test_object_base_pass(self) -> None:
        """TestObjectBase passes its own schema."""
        obj = TestObjectBase(v_i64=10, v_f64=1.5, v_str="hi")
        S("testing.TestObjectBase").check_value(obj)

    def test_test_object_derived_pass(self) -> None:
        """TestObjectDerived passes its own schema."""
        obj = TestObjectDerived(v_map={"a": 1}, v_array=[1], v_i64=0, v_f64=0.0, v_str="")
        S("testing.TestObjectDerived").check_value(obj)

    def test_cxx_class_base_pass(self) -> None:
        """_TestCxxClassBase passes its own schema."""
        obj = _TestCxxClassBase(v_i64=1, v_i32=2)
        S("testing.TestCxxClassBase").check_value(obj)

    def test_cxx_class_derived_pass(self) -> None:
        """_TestCxxClassDerived passes its own schema."""
        obj = _TestCxxClassDerived(v_i64=1, v_i32=2, v_f64=3.0)
        S("testing.TestCxxClassDerived").check_value(obj)

    def test_cxx_class_derived_derived_pass(self) -> None:
        """_TestCxxClassDerivedDerived passes its own schema."""
        obj = _TestCxxClassDerivedDerived(v_i64=1, v_i32=2, v_f64=3.0, v_bool=True)
        S("testing.TestCxxClassDerivedDerived").check_value(obj)


# ---------------------------------------------------------------------------
# Category 31: Custom object type hierarchy (subclass passes parent schema)
# ---------------------------------------------------------------------------
class TestCustomObjectHierarchy:
    def test_derived_passes_base_schema(self) -> None:
        """TestObjectDerived passes TypeSchema('testing.TestObjectBase')."""
        obj = TestObjectDerived(v_map={"a": 1}, v_array=[1], v_i64=0, v_f64=0.0, v_str="")
        S("testing.TestObjectBase").check_value(obj)

    def test_derived_passes_object_schema(self) -> None:
        """TestObjectDerived passes TypeSchema('Object')."""
        obj = TestObjectDerived(v_map={"a": 1}, v_array=[1], v_i64=0, v_f64=0.0, v_str="")
        S("Object").check_value(obj)

    def test_cxx_derived_passes_base(self) -> None:
        """_TestCxxClassDerived passes TestCxxClassBase schema."""
        obj = _TestCxxClassDerived(v_i64=1, v_i32=2, v_f64=3.0)
        S("testing.TestCxxClassBase").check_value(obj)

    def test_cxx_derived_derived_passes_base(self) -> None:
        """_TestCxxClassDerivedDerived passes TestCxxClassBase schema (2-level up)."""
        obj = _TestCxxClassDerivedDerived(v_i64=1, v_i32=2, v_f64=3.0, v_bool=True)
        S("testing.TestCxxClassBase").check_value(obj)

    def test_cxx_derived_derived_passes_derived(self) -> None:
        """_TestCxxClassDerivedDerived passes TestCxxClassDerived schema (1-level up)."""
        obj = _TestCxxClassDerivedDerived(v_i64=1, v_i32=2, v_f64=3.0, v_bool=True)
        S("testing.TestCxxClassDerived").check_value(obj)

    def test_all_custom_objects_pass_object_schema(self) -> None:
        """Every custom object passes the generic Object schema."""
        objs = [
            TestIntPair(1, 2),
            TestObjectBase(v_i64=10, v_f64=1.5, v_str="hi"),
            _TestCxxClassBase(v_i64=1, v_i32=2),
            _TestCxxClassDerived(v_i64=1, v_i32=2, v_f64=3.0),
            _TestCxxClassDerivedDerived(v_i64=1, v_i32=2, v_f64=3.0, v_bool=True),
        ]
        schema = S("Object")
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
            S("testing.TestObjectBase").check_value(obj)

    def test_base_fails_derived_schema(self) -> None:
        """Parent object fails child schema (TestObjectBase fails TestObjectDerived)."""
        obj = TestObjectBase(v_i64=10, v_f64=1.5, v_str="hi")
        with pytest.raises(TypeError, match=r"testing.TestObjectBase"):
            S("testing.TestObjectDerived").check_value(obj)

    def test_non_object_fails_custom_schema(self) -> None:
        """Plain int fails custom object schema."""
        with pytest.raises(TypeError, match=r"expected testing\.TestIntPair.*got int"):
            S("testing.TestIntPair").check_value(42)

    def test_none_fails_custom_schema(self) -> None:
        """None fails custom object schema."""
        with pytest.raises(TypeError, match="got None"):
            S("testing.TestIntPair").check_value(None)

    def test_string_fails_custom_schema(self) -> None:
        """String fails custom object schema."""
        with pytest.raises(TypeError, match="got str"):
            S("testing.TestIntPair").check_value("hello")

    def test_cxx_base_fails_derived_schema(self) -> None:
        """_TestCxxClassBase fails _TestCxxClassDerived schema."""
        obj = _TestCxxClassBase(v_i64=1, v_i32=2)
        with pytest.raises(TypeError):
            S("testing.TestCxxClassDerived").check_value(obj)

    def test_sibling_types_reject_each_other(self) -> None:
        """TestIntPair and TestCxxClassBase are unrelated -- reject each other."""
        pair = TestIntPair(1, 2)
        base = _TestCxxClassBase(v_i64=1, v_i32=2)
        with pytest.raises(TypeError):
            S("testing.TestCxxClassBase").check_value(pair)
        with pytest.raises(TypeError):
            S("testing.TestIntPair").check_value(base)


# ---------------------------------------------------------------------------
# Category 33: Custom objects in containers
# ---------------------------------------------------------------------------
class TestCustomObjectInContainers:
    def test_array_of_custom_objects(self) -> None:
        """Array[testing.TestIntPair] with matching elements."""
        objs = [TestIntPair(1, 2), TestIntPair(3, 4)]
        S("Array", S("testing.TestIntPair")).check_value(objs)

    def test_array_of_custom_objects_wrong_type(self) -> None:
        """Array[testing.TestIntPair] with wrong element type fails."""
        objs = [TestIntPair(1, 2), _TestCxxClassBase(v_i64=1, v_i32=2)]
        with pytest.raises(TypeError, match=r"element \[1\]"):
            S("Array", S("testing.TestIntPair")).check_value(objs)

    def test_array_of_base_with_derived_elements(self) -> None:
        """Array[testing.TestObjectBase] accepts derived elements via hierarchy."""
        base = TestObjectBase(v_i64=1, v_f64=1.0, v_str="a")
        derived = TestObjectDerived(v_map={"a": 1}, v_array=[1], v_i64=0, v_f64=0.0, v_str="")
        S("Array", S("testing.TestObjectBase")).check_value([base, derived])

    def test_map_str_to_custom_object(self) -> None:
        """Map[str, testing.TestIntPair] pass."""
        objs = {"a": TestIntPair(1, 2), "b": TestIntPair(3, 4)}
        S("Map", S("str"), S("testing.TestIntPair")).check_value(objs)

    def test_map_str_to_custom_object_wrong_value(self) -> None:
        """Map[str, testing.TestIntPair] with int value fails."""
        data = {"a": TestIntPair(1, 2), "b": 42}
        with pytest.raises(TypeError, match="value for key 'b'"):
            S("Map", S("str"), S("testing.TestIntPair")).check_value(data)

    def test_ffi_array_of_custom_objects(self) -> None:
        """tvm_ffi.Array of custom objects passes Array[Object]."""
        arr = tvm_ffi.Array([TestIntPair(1, 2), TestObjectBase(v_i64=1, v_f64=2.0, v_str="s")])
        S("Array", S("Object")).check_value(arr)

    def test_ffi_array_of_custom_objects_specific_type(self) -> None:
        """tvm_ffi.Array of TestIntPair passes Array[testing.TestIntPair]."""
        arr = tvm_ffi.Array([TestIntPair(1, 2), TestIntPair(3, 4)])
        S("Array", S("testing.TestIntPair")).check_value(arr)

    def test_ffi_map_with_custom_object_values(self) -> None:
        """tvm_ffi.Map with custom object values passes."""
        m = tvm_ffi.Map({"x": TestIntPair(1, 2), "y": TestIntPair(3, 4)})
        S("Map", S("str"), S("testing.TestIntPair")).check_value(m)


# ---------------------------------------------------------------------------
# Category 34: Optional/Union with custom objects
# ---------------------------------------------------------------------------
class TestCustomObjectOptionalUnion:
    def test_optional_custom_object_with_value(self) -> None:
        """Optional[testing.TestIntPair] with actual object."""
        obj = TestIntPair(1, 2)
        S("Optional", S("testing.TestIntPair")).check_value(obj)

    def test_optional_custom_object_with_none(self) -> None:
        """Optional[testing.TestIntPair] with None."""
        S("Optional", S("testing.TestIntPair")).check_value(None)

    def test_optional_custom_object_wrong_type(self) -> None:
        """Optional[testing.TestIntPair] with wrong object type."""
        obj = _TestCxxClassBase(v_i64=1, v_i32=2)
        with pytest.raises(TypeError):
            S("Optional", S("testing.TestIntPair")).check_value(obj)

    def test_union_custom_object_and_int(self) -> None:
        """Union[testing.TestIntPair, int] with object."""
        obj = TestIntPair(1, 2)
        S("Union", S("testing.TestIntPair"), S("int")).check_value(obj)

    def test_union_custom_object_and_int_with_int(self) -> None:
        """Union[testing.TestIntPair, int] with int."""
        S("Union", S("testing.TestIntPair"), S("int")).check_value(42)

    def test_union_custom_object_and_int_with_wrong(self) -> None:
        """Union[testing.TestIntPair, int] with str fails."""
        with pytest.raises(TypeError):
            S("Union", S("testing.TestIntPair"), S("int")).check_value("bad")

    def test_union_two_custom_objects(self) -> None:
        """Union of two custom types accepts both."""
        pair = TestIntPair(1, 2)
        base = _TestCxxClassBase(v_i64=1, v_i32=2)
        schema = S("Union", S("testing.TestIntPair"), S("testing.TestCxxClassBase"))
        schema.check_value(pair)
        schema.check_value(base)

    def test_union_two_custom_objects_rejects_third(self) -> None:
        """Union of two custom types rejects a third."""
        obj = TestObjectBase(v_i64=1, v_f64=2.0, v_str="s")
        with pytest.raises(TypeError):
            S("Union", S("testing.TestIntPair"), S("testing.TestCxxClassBase")).check_value(obj)


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
    def test_array_of_optional_custom_object(self) -> None:
        """Array[Optional[testing.TestIntPair]] with mix of objects and None."""
        data = [TestIntPair(1, 2), None, TestIntPair(3, 4)]
        S("Array", S("Optional", S("testing.TestIntPair"))).check_value(data)

    def test_map_str_to_array_of_custom_objects(self) -> None:
        """Map[str, Array[testing.TestIntPair]] with nested objects."""
        data = {
            "group1": [TestIntPair(1, 2), TestIntPair(3, 4)],
            "group2": [TestIntPair(5, 6)],
        }
        S("Map", S("str"), S("Array", S("testing.TestIntPair"))).check_value(data)

    def test_array_of_union_custom_objects(self) -> None:
        """Array[Union[testing.TestIntPair, testing.TestCxxClassBase]]."""
        data = [TestIntPair(1, 2), _TestCxxClassBase(v_i64=1, v_i32=2), TestIntPair(5, 6)]
        S("Array", S("Union", S("testing.TestIntPair"), S("testing.TestCxxClassBase"))).check_value(
            data
        )

    def test_optional_array_of_custom_objects(self) -> None:
        """Optional[Array[testing.TestIntPair]] with array."""
        data = [TestIntPair(1, 2)]
        S("Optional", S("Array", S("testing.TestIntPair"))).check_value(data)

    def test_optional_array_of_custom_objects_none(self) -> None:
        """Optional[Array[testing.TestIntPair]] with None."""
        S("Optional", S("Array", S("testing.TestIntPair"))).check_value(None)

    def test_nested_error_with_custom_object(self) -> None:
        """Array[testing.TestIntPair] error message includes type keys."""
        data = [TestIntPair(1, 2), _TestCxxClassBase(v_i64=1, v_i32=2)]
        with pytest.raises(
            TypeError, match=r"element \[1\].*testing.TestIntPair.*testing.TestCxxClassBase"
        ):
            S("Array", S("testing.TestIntPair")).check_value(data)

    def test_map_nested_error_with_custom_object(self) -> None:
        """Map value error for custom object includes key and type info."""
        data = {"ok": TestIntPair(1, 2), "bad": 42}
        with pytest.raises(
            TypeError, match=r"value for key 'bad'.*expected testing\.TestIntPair.*got int"
        ):
            S("Map", S("str"), S("testing.TestIntPair")).check_value(data)

    def test_deep_nested_custom_objects(self) -> None:
        """Map[str, Array[Optional[testing.TestIntPair]]] deep nesting."""
        data = {
            "a": [TestIntPair(1, 2), None],
            "b": [None, TestIntPair(3, 4), TestIntPair(5, 6)],
        }
        S("Map", S("str"), S("Array", S("Optional", S("testing.TestIntPair")))).check_value(data)

    def test_deep_nested_custom_objects_error(self) -> None:
        """Map[str, Array[testing.TestIntPair]] error at 3 levels."""
        data = {"k": [TestIntPair(1, 2), "bad"]}
        with pytest.raises(TypeError, match=r"value for key 'k'.*element .1."):
            S("Map", S("str"), S("Array", S("testing.TestIntPair"))).check_value(data)

    def test_tuple_with_custom_object(self) -> None:
        """tuple[testing.TestIntPair, int, str] with custom object."""
        data = (TestIntPair(1, 2), 42, "hello")
        S("tuple", S("testing.TestIntPair"), S("int"), S("str")).check_value(data)

    def test_tuple_with_custom_object_wrong(self) -> None:
        """tuple[testing.TestIntPair, int] with wrong object in first position."""
        data = (_TestCxxClassBase(v_i64=1, v_i32=2), 42)
        with pytest.raises(TypeError, match=r"element \[0\]"):
            S("tuple", S("testing.TestIntPair"), S("int")).check_value(data)


# ---------------------------------------------------------------------------
# Category 37: Lowercase Python-native origins ("list", "dict")
# ---------------------------------------------------------------------------
class TestLowercaseOrigins:
    def test_list_origin_accepts_python_list(self) -> None:
        """TypeSchema("list", ...) should validate elements, not passthrough."""
        S("list", S("int")).check_value([1, 2, 3])

    def test_list_origin_rejects_bad_elements(self) -> None:
        """TypeSchema("list", (int,)).check_value(["x"]) should fail."""
        with pytest.raises(TypeError, match=r"element \[0\]"):
            S("list", S("int")).check_value(["x"])

    def test_list_origin_converts_elements(self) -> None:
        """TypeSchema("list", (float,)).convert([1, True]) does int->float."""
        result = S("list", S("float")).convert([1, True]).to_py()
        assert list(result) == [1.0, 1.0]
        assert all(type(x) is float for x in result)

    def test_dict_origin_accepts_python_dict(self) -> None:
        """TypeSchema("dict", ...) should validate key/value types."""
        S("dict", S("str"), S("int")).check_value({"a": 1})

    def test_dict_origin_rejects_bad_values(self) -> None:
        """TypeSchema("dict", (str, int)).check_value({"a": "x"}) should fail."""
        with pytest.raises(TypeError, match="value for key 'a'"):
            S("dict", S("str"), S("int")).check_value({"a": "x"})

    def test_dict_origin_converts_values(self) -> None:
        """TypeSchema("dict", (str, float)).convert({"a": 1}) does int->float."""
        result = S("dict", S("str"), S("float")).convert({"a": 1, "b": True}).to_py()
        assert result["a"] == 1.0
        assert result["b"] == 1.0
        assert all(type(v) is float for v in result.values())

    def test_list_origin_no_args_accepts_anything(self) -> None:
        """TypeSchema("list") with no args accepts any list (element type is Any)."""
        S("list").check_value([1, "a", None])

    def test_dict_origin_no_args_accepts_anything(self) -> None:
        """TypeSchema("dict") with no args accepts any dict."""
        S("dict").check_value({"a": 1, 2: "b"})

    def test_list_origin_rejects_non_list(self) -> None:
        """TypeSchema("list") rejects non-sequence types."""
        with pytest.raises(TypeError, match="got int"):
            S("list").check_value(42)

    def test_dict_origin_rejects_non_dict(self) -> None:
        """TypeSchema("dict") rejects non-dict types."""
        with pytest.raises(TypeError):
            S("dict").check_value([1, 2])


# ---------------------------------------------------------------------------
# Category 38: Cross-type container conversions (Array<->List, Map<->Dict)
# ---------------------------------------------------------------------------
class TestCrossTypeContainers:
    def test_array_schema_accepts_ffi_list(self) -> None:
        """Array[int] schema accepts tvm_ffi.List (C++ kOtherTypeIndex)."""
        lst = tvm_ffi.List([1, 2, 3])
        S("Array", S("int")).check_value(lst)

    def test_list_schema_accepts_ffi_array(self) -> None:
        """List[int] schema accepts tvm_ffi.Array (C++ kOtherTypeIndex)."""
        arr = tvm_ffi.Array([1, 2, 3])
        S("List", S("int")).check_value(arr)

    def test_map_schema_accepts_ffi_dict(self) -> None:
        """Map[str, int] schema accepts tvm_ffi.Dict (C++ kOtherTypeIndex)."""
        d = tvm_ffi.Dict({"a": 1, "b": 2})
        S("Map", S("str"), S("int")).check_value(d)

    def test_dict_schema_accepts_ffi_map(self) -> None:
        """Dict[str, int] schema accepts tvm_ffi.Map (C++ kOtherTypeIndex)."""
        m = tvm_ffi.Map({"a": 1, "b": 2})
        S("Dict", S("str"), S("int")).check_value(m)

    def test_array_schema_converts_list_elements(self) -> None:
        """Array[float] converts elements from tvm_ffi.List[int]."""
        lst = tvm_ffi.List([1, 2, True])
        result = S("Array", S("float")).convert(lst).to_py()
        assert list(result) == [1.0, 2.0, 1.0]
        assert all(type(x) is float for x in result)

    def test_list_schema_converts_array_elements(self) -> None:
        """List[float] converts elements from tvm_ffi.Array[int]."""
        arr = tvm_ffi.Array([1, 2, True])
        result = S("List", S("float")).convert(arr).to_py()
        assert list(result) == [1.0, 2.0, 1.0]
        assert all(type(x) is float for x in result)

    def test_map_schema_converts_dict_values(self) -> None:
        """Map[str, float] converts values from tvm_ffi.Dict."""
        d = tvm_ffi.Dict({"a": 1, "b": True})
        result = S("Map", S("str"), S("float")).convert(d).to_py()
        assert result["a"] == 1.0
        assert result["b"] == 1.0

    def test_dict_schema_converts_map_values(self) -> None:
        """Dict[str, float] converts values from tvm_ffi.Map."""
        m = tvm_ffi.Map({"a": 1, "b": True})
        result = S("Dict", S("str"), S("float")).convert(m).to_py()
        assert result["a"] == 1.0
        assert result["b"] == 1.0

    def test_cross_type_still_rejects_wrong_container(self) -> None:
        """Array schema still rejects non-sequence CObjects (e.g. Map)."""
        m = tvm_ffi.Map({"a": 1})
        with pytest.raises(TypeError, match="expected Array"):
            S("Array", S("int")).check_value(m)

    def test_cross_type_map_rejects_array(self) -> None:
        """Map schema still rejects sequence CObjects (e.g. Array)."""
        arr = tvm_ffi.Array([1, 2])
        with pytest.raises(TypeError, match="expected Map"):
            S("Map", S("str"), S("int")).check_value(arr)


# ---------------------------------------------------------------------------
# Category 39: tuple accepts list and CObject Array
# ---------------------------------------------------------------------------
class TestTupleAcceptsListAndArray:
    def test_tuple_accepts_python_list(self) -> None:
        """tuple[int, str] accepts Python list input."""
        result = S("tuple", S("int"), S("str")).convert([42, "hello"]).to_py()
        assert list(result) == [42, "hello"]

    def test_tuple_list_with_conversion(self) -> None:
        """tuple[float, int] converts list elements (bool->float, bool->int)."""
        result = S("tuple", S("float"), S("int")).convert([True, False]).to_py()
        assert list(result) == [1.0, 0]
        assert type(result[0]) is float
        assert type(result[1]) is int

    def test_tuple_rejects_wrong_length_list(self) -> None:
        """tuple[int, str] rejects list of wrong length."""
        with pytest.raises(TypeError, match="length"):
            S("tuple", S("int"), S("str")).check_value([1, "a", "b"])

    def test_tuple_accepts_ffi_array(self) -> None:
        """tuple[int, int] accepts tvm_ffi.Array (C++ Tuple accepts kTVMFFIArray)."""
        arr = tvm_ffi.Array([1, 2])
        S("tuple", S("int"), S("int")).check_value(arr)

    def test_tuple_ffi_array_with_conversion(self) -> None:
        """tuple[float, float] converts tvm_ffi.Array elements."""
        arr = tvm_ffi.Array([1, True])
        result = S("tuple", S("float"), S("float")).convert(arr).to_py()
        assert list(result) == [1.0, 1.0]
        assert all(type(x) is float for x in result)

    def test_tuple_ffi_array_wrong_length(self) -> None:
        """tuple[int, int] rejects tvm_ffi.Array of wrong length."""
        arr = tvm_ffi.Array([1, 2, 3])
        with pytest.raises(TypeError, match="length"):
            S("tuple", S("int"), S("int")).check_value(arr)

    def test_tuple_rejects_ffi_map(self) -> None:
        """Tuple schema rejects Map CObject."""
        m = tvm_ffi.Map({"a": 1})
        with pytest.raises(TypeError, match="expected tuple"):
            S("tuple", S("int")).check_value(m)

    def test_untyped_tuple_accepts_list(self) -> None:
        """Tuple (no args) accepts any list as-is."""
        # Untyped tuple has tuple_len=0, so it just checks the container type
        # but doesn't validate elements
        S("tuple").check_value([1, "a", None])

    def test_untyped_tuple_accepts_ffi_array(self) -> None:
        """Tuple (no args) accepts tvm_ffi.Array as-is."""
        arr = tvm_ffi.Array([1, 2, 3])
        S("tuple").check_value(arr)


# ---------------------------------------------------------------------------
# Category 40: dtype string parse errors
# ---------------------------------------------------------------------------
class TestDtypeParseErrors:
    def test_check_value_bad_dtype_raises_error(self) -> None:
        """check_value should raise TypeError for invalid dtype."""
        with pytest.raises(TypeError, match="dtype"):
            S("dtype").check_value("not_a_valid_dtype_xyz")

    def test_convert_bad_dtype_raises_type_error_2(self) -> None:
        """Convert should raise TypeError for invalid dtype string."""
        with pytest.raises(TypeError, match="dtype"):
            S("dtype").convert("not_a_valid_dtype_xyz")

    def test_convert_bad_dtype_raises_type_error(self) -> None:
        """Convert should raise TypeError for invalid dtype string."""
        with pytest.raises(TypeError, match="dtype"):
            S("dtype").convert("not_a_valid_dtype_xyz")

    def test_valid_dtype_string_still_works(self) -> None:
        """Valid dtype strings should still convert successfully."""
        result = S("dtype").convert("float32").to_py()
        assert str(result) == "float32"

    def test_convert_valid_dtype(self) -> None:
        """Convert with valid dtype returns DataType."""
        result = S("dtype").convert("int8").to_py()
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
        S("int").check_value(2**63 - 1)

    def test_int64_min_accepted(self) -> None:
        """-2^63 (INT64_MIN) is the smallest valid int."""
        S("int").check_value(-(2**63))

    def test_int64_max_plus_one_rejected(self) -> None:
        """2^63 exceeds int64 range."""
        with pytest.raises(TypeError, match="int64 range"):
            S("int").check_value(2**63)

    def test_int64_min_minus_one_rejected(self) -> None:
        """-2^63-1 exceeds int64 range."""
        with pytest.raises(TypeError, match="int64 range"):
            S("int").check_value(-(2**63) - 1)

    def test_very_large_positive_rejected(self) -> None:
        """Very large positive integer rejected."""
        with pytest.raises(TypeError, match="int64 range"):
            S("int").check_value(10**100)

    def test_very_large_negative_rejected(self) -> None:
        """Very large negative integer rejected."""
        with pytest.raises(TypeError, match="int64 range"):
            S("int").check_value(-(10**100))

    def test_convert_raises_type_error_for_overflow(self) -> None:
        """Convert raises TypeError for overflow."""
        with pytest.raises(TypeError, match="int64 range"):
            S("int").convert(2**63)

    def test_bool_to_int_no_range_issue(self) -> None:
        """Bool -> int conversion (0 or 1) always fits."""
        assert S("int").convert(True).to_py() == 1
        assert S("int").convert(False).to_py() == 0

    def test_int64_boundaries_in_float_conversion(self) -> None:
        """Float schema accepts large ints (float64 has wider range)."""
        # float64 can represent integers up to 2^53 exactly,
        # and larger values with precision loss (but no range error)
        S("float").check_value(2**63)
        S("float").check_value(-(2**63))

    def test_int64_overflow_in_optional_int(self) -> None:
        """Optional[int] propagates int64 range check."""
        with pytest.raises(TypeError, match="int64 range"):
            S("Optional", S("int")).check_value(2**63)

    def test_int64_overflow_in_array_element(self) -> None:
        """Array[int] element overflow is caught with path."""
        with pytest.raises(TypeError, match="int64 range"):
            S("Array", S("int")).check_value([1, 2**63, 3])


# ---------------------------------------------------------------------------
# Category 42: Unknown origin errors (lazy converter construction)
# ---------------------------------------------------------------------------
class TestUnknownOriginErrors:
    """Converter is built lazily via cached_property. Unknown origins
    construct fine but raise TypeError on first convert/check_value.
    """

    def test_unknown_origin_constructs_ok(self) -> None:
        """TypeSchema with unknown origin can be constructed."""
        schema = S("not_a_real_type")
        assert schema.origin == "not_a_real_type"

    def test_unknown_origin_errors_on_check_value(self) -> None:
        """Unknown origin raises TypeError on check_value."""
        schema = S("not_a_real_type")
        with pytest.raises(TypeError, match="unknown TypeSchema origin"):
            schema.check_value(42)

    def test_unknown_origin_errors_on_convert(self) -> None:
        """Unknown origin raises TypeError on convert."""
        schema = S("not_a_real_type")
        with pytest.raises(TypeError, match="unknown TypeSchema origin"):
            schema.convert(42)

    def test_unknown_origin_errors_on_convert_2(self) -> None:
        """Unknown origin raises TypeError on convert (duplicate check)."""
        schema = S("not_a_real_type")
        with pytest.raises(TypeError, match="unknown TypeSchema origin"):
            schema.convert(42)

    def test_unknown_origin_errors_on_check_value_2(self) -> None:
        """Unknown origin raises TypeError on check_value (duplicate check)."""
        schema = S("not_a_real_type")
        with pytest.raises(TypeError, match="unknown TypeSchema origin"):
            schema.check_value(42)

    def test_typo_origin_errors(self) -> None:
        """Common typos are caught, not silently passed through."""
        for typo in ("innt", "floot", "strr", "Int", "Float"):
            schema = S(typo)
            with pytest.raises(TypeError, match="unknown TypeSchema origin"):
                schema.check_value(42)

    def test_unknown_nested_in_optional_errors(self) -> None:
        """Unknown origin nested inside Optional errors on use."""
        schema = S("Optional", S("not_a_real_type"))
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
            S("bogus_type").convert("anything")

    def test_check_value_unknown_origin_raises(self) -> None:
        """check_value with unknown origin raises TypeError."""
        with pytest.raises(TypeError, match="unknown TypeSchema origin"):
            S("bogus_type").check_value("anything")


# ---------------------------------------------------------------------------
# Category 44: Schema arity validation (ValueError, not assert)
# ---------------------------------------------------------------------------
class TestSchemaArityValidation:
    """Verify arity checks use ValueError (not assert) so they work under -O."""

    def test_union_too_few_args(self) -> None:
        """Union with < 2 args raises ValueError."""
        with pytest.raises(ValueError, match="at least two"):
            S("Union", S("int"))

    def test_optional_wrong_arity(self) -> None:
        """Optional with != 1 arg raises ValueError."""
        with pytest.raises(ValueError, match="exactly one"):
            S("Optional")
        with pytest.raises(ValueError, match="exactly one"):
            S("Optional", S("int"), S("str"))

    def test_array_too_many_args(self) -> None:
        """Array with > 1 arg raises ValueError."""
        with pytest.raises(ValueError, match="0 or 1"):
            S("Array", S("int"), S("str"))

    def test_list_too_many_args(self) -> None:
        """List with > 1 arg raises ValueError."""
        with pytest.raises(ValueError, match="0 or 1"):
            S("List", S("int"), S("str"))

    def test_map_wrong_arity(self) -> None:
        """Map with 1 or 3 args raises ValueError."""
        with pytest.raises(ValueError, match="0 or 2"):
            S("Map", S("str"))
        with pytest.raises(ValueError, match="0 or 2"):
            S("Map", S("str"), S("int"), S("float"))

    def test_dict_wrong_arity(self) -> None:
        """Dict with 1 or 3 args raises ValueError."""
        with pytest.raises(ValueError, match="0 or 2"):
            S("Dict", S("str"))

    def test_lowercase_list_too_many_args(self) -> None:
        """Lowercase 'list' with > 1 arg raises ValueError."""
        with pytest.raises(ValueError, match="0 or 1"):
            S("list", S("int"), S("str"))

    def test_lowercase_dict_wrong_arity(self) -> None:
        """Lowercase 'dict' with 1 arg raises ValueError."""
        with pytest.raises(ValueError, match="0 or 2"):
            S("dict", S("str"))


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
        int_schema = S("int")
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
        arr_schema = S("Array")
        schema = TypeSchema.from_type_index(arr_schema.origin_type_index, (TypeSchema("int"),))
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

        S("int").check_value(IntProto())

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
        result = S("int").convert(obj).to_py()
        assert result is not None

    def test_without_protocol_still_rejected(self) -> None:
        """Object without __tvm_ffi_int__ is still rejected by int schema."""

        class NoProto:
            pass

        with pytest.raises(TypeError, match="expected int"):
            S("int").check_value(NoProto())


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

        S("float").check_value(FloatProto())

    def test_float_protocol_convert(self) -> None:
        """Convert returns protocol value as-is."""

        class FloatProto:
            def __tvm_ffi_float__(self) -> float:
                return 2.0

        obj = FloatProto()
        result = S("float").convert(obj).to_py()
        assert result is not None

    def test_without_protocol_still_rejected(self) -> None:
        """Object without __tvm_ffi_float__ is still rejected."""

        class NoProto:
            pass

        with pytest.raises(TypeError, match="expected float"):
            S("float").check_value(NoProto())


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

        S("ctypes.c_void_p").check_value(PtrProto())

    def test_opaque_ptr_protocol_convert(self) -> None:
        """Convert returns protocol value as-is."""

        class PtrProto:
            def __tvm_ffi_opaque_ptr__(self) -> int:
                return 0

        obj = PtrProto()
        result = S("ctypes.c_void_p").convert(obj).to_py()
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

        S("Device").check_value(DevProto())

    def test_dlpack_device_protocol_convert(self) -> None:
        """Convert returns protocol value as-is."""

        class DevProto:
            def __dlpack_device__(self) -> tuple[int, int]:
                return (2, 1)

        obj = DevProto()
        result = S("Device").convert(obj).to_py()
        assert result is not None

    def test_without_protocol_still_rejected(self) -> None:
        """Object without __dlpack_device__ is still rejected."""

        class NoProto:
            pass

        with pytest.raises(TypeError, match="expected Device"):
            S("Device").check_value(NoProto())


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

        S("dtype").check_value(DTypeProto())

    def test_dlpack_data_type_protocol_convert(self) -> None:
        """Convert returns protocol value as-is."""

        class DTypeProto:
            def __dlpack_data_type__(self) -> tuple[int, int, int]:
                return (0, 32, 1)

        obj = DTypeProto()
        result = S("dtype").convert(obj).to_py()
        assert result is not None

    def test_numpy_dtype_accepted(self) -> None:
        """numpy.dtype passes dtype schema (if numpy installed)."""
        numpy = pytest.importorskip("numpy")
        S("dtype").check_value(numpy.dtype("float32"))

    def test_numpy_dtype_convert(self) -> None:
        """Convert returns numpy.dtype as-is."""
        numpy = pytest.importorskip("numpy")
        dt = numpy.dtype("int32")
        result = S("dtype").convert(dt).to_py()
        assert result is not None

    def test_torch_dtype_accepted(self) -> None:
        """torch.dtype passes dtype schema (if torch installed)."""
        torch = pytest.importorskip("torch")
        S("dtype").check_value(torch.float32)

    def test_torch_dtype_convert(self) -> None:
        """Convert returns torch.dtype as-is."""
        torch = pytest.importorskip("torch")
        dt = torch.int64
        result = S("dtype").convert(dt).to_py()
        assert result is not None


# ---------------------------------------------------------------------------
# Category 51: __dlpack_c_exchange_api__ protocol (Tensor)
# ---------------------------------------------------------------------------
class TestTensorProtocol:
    """Tensor schema accepts __dlpack_c_exchange_api__ protocol."""

    def test_dlpack_c_exchange_api_accepted(self) -> None:
        """Object with __dlpack_c_exchange_api__ passes Tensor schema."""

        class ExchangeAPI:
            def __dlpack_c_exchange_api__(self) -> int:
                return 0

        S("Tensor").check_value(ExchangeAPI())

    def test_dlpack_c_exchange_api_convert(self) -> None:
        """Convert returns protocol value as-is."""

        class ExchangeAPI:
            def __dlpack_c_exchange_api__(self) -> int:
                return 0

        obj = ExchangeAPI()
        S("Tensor").check_value(obj)

    def test_dlpack_still_accepted(self) -> None:
        """Object with __dlpack__ still accepted (existing behavior)."""

        class DLPackProto:
            def __dlpack__(self) -> object:
                return None

        S("Tensor").check_value(DLPackProto())


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

        S("Object").check_value(ObjProto())

    def test_object_protocol_specific_type(self) -> None:
        """__tvm_ffi_object__ returning TestIntPair passes TestIntPair schema."""
        inner = TestIntPair(3, 4)

        class ObjProto:
            def __tvm_ffi_object__(self) -> object:
                return inner

        S("testing.TestIntPair").check_value(ObjProto())

    def test_object_protocol_convert_returns_cobject(self) -> None:
        """Convert returns the CObject from __tvm_ffi_object__()."""
        inner = TestIntPair(5, 6)

        class ObjProto:
            def __tvm_ffi_object__(self) -> object:
                return inner

        result = S("testing.TestIntPair").convert(ObjProto()).to_py()
        assert result.same_as(inner)

    def test_object_protocol_wrong_type_rejected(self) -> None:
        """__tvm_ffi_object__ returning wrong type is rejected."""
        inner = TestIntPair(1, 2)

        class ObjProto:
            def __tvm_ffi_object__(self) -> object:
                return inner

        with pytest.raises(TypeError, match="__tvm_ffi_object__"):
            S("testing.TestCxxClassBase").check_value(ObjProto())

    def test_object_protocol_raises_caught(self) -> None:
        """__tvm_ffi_object__ that raises produces _ConvertError."""

        class BadProto:
            def __tvm_ffi_object__(self) -> object:
                raise RuntimeError("broken")

        with pytest.raises(TypeError, match=r"__tvm_ffi_object__\(\) failed"):
            S("Object").check_value(BadProto())

    def test_object_protocol_hierarchy(self) -> None:
        """__tvm_ffi_object__ returning derived passes base schema."""
        derived = _TestCxxClassDerived(v_i64=1, v_i32=2, v_f64=3.0)

        class ObjProto:
            def __tvm_ffi_object__(self) -> object:
                return derived

        S("testing.TestCxxClassBase").check_value(ObjProto())


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

        S("Object").check_value(MyConvertible())

    def test_object_convertible_specific_type(self) -> None:
        """ObjectConvertible passes specific type schema."""
        inner = TestIntPair(1, 2)

        class MyConvertible(ObjectConvertible):
            def asobject(self) -> tvm_ffi.core.Object:
                return inner

        S("testing.TestIntPair").check_value(MyConvertible())

    def test_object_convertible_convert_returns_cobject(self) -> None:
        """Convert returns the CObject from asobject()."""
        inner = TestIntPair(7, 8)

        class MyConvertible(ObjectConvertible):
            def asobject(self) -> tvm_ffi.core.Object:
                return inner

        result = S("testing.TestIntPair").convert(MyConvertible()).to_py()
        assert result.same_as(inner)

    def test_object_convertible_wrong_type(self) -> None:
        """ObjectConvertible returning wrong type is rejected."""
        inner = TestIntPair(1, 2)

        class MyConvertible(ObjectConvertible):
            def asobject(self) -> tvm_ffi.core.Object:
                return inner

        with pytest.raises(TypeError, match=r"asobject\(\)"):
            S("testing.TestCxxClassBase").check_value(MyConvertible())

    def test_object_convertible_raises_caught(self) -> None:
        """asobject() that raises produces error, not exception."""

        class BadConvertible(ObjectConvertible):
            def asobject(self) -> tvm_ffi.core.Object:
                raise RuntimeError("broken asobject")

        with pytest.raises(TypeError, match=r"asobject\(\) failed"):
            S("Object").check_value(BadConvertible())


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

        S("int").check_value(ValProto())

    def test_value_protocol_float(self) -> None:
        """__tvm_ffi_value__ returning float passes float schema."""

        class ValProto:
            def __tvm_ffi_value__(self) -> object:
                return 3.14

        S("float").check_value(ValProto())

    def test_value_protocol_convert(self) -> None:
        """Convert returns the unwrapped value from __tvm_ffi_value__."""

        class ValProto:
            def __tvm_ffi_value__(self) -> object:
                return 42

        result = S("int").convert(ValProto()).to_py()
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
        assert S("int").convert(wrapped).to_py() == 10

    def test_value_protocol_object(self) -> None:
        """__tvm_ffi_value__ returning a CObject passes object schema."""
        inner = TestIntPair(1, 2)

        class ValProto:
            def __tvm_ffi_value__(self) -> object:
                return inner

        S("testing.TestIntPair").check_value(ValProto())

    def test_value_protocol_still_fails_on_mismatch(self) -> None:
        """__tvm_ffi_value__ returning wrong type still fails."""

        class ValProto:
            def __tvm_ffi_value__(self) -> object:
                return "not_an_int"

        with pytest.raises(TypeError, match="expected int"):
            S("int").check_value(ValProto())

    def test_value_protocol_raises_uses_original_error(self) -> None:
        """If __tvm_ffi_value__ raises, the original error is returned."""

        class BadValProto:
            def __tvm_ffi_value__(self) -> object:
                raise RuntimeError("broken")

        with pytest.raises(TypeError, match="expected int"):
            S("int").check_value(BadValProto())


# ---------------------------------------------------------------------------
# Category 55: Protocol values in containers
# ---------------------------------------------------------------------------
class TestProtocolsInContainers:
    """Protocol-accepting values work inside containers and composites."""

    def test_int_protocol_in_array(self) -> None:
        """Array[int] accepts elements with __tvm_ffi_int__."""

        class IntProto:
            def __tvm_ffi_int__(self) -> int:
                return 1

        S("Array", S("int")).check_value([1, IntProto(), 3])

    def test_float_protocol_in_optional(self) -> None:
        """Optional[float] accepts __tvm_ffi_float__ value."""

        class FloatProto:
            def __tvm_ffi_float__(self) -> float:
                return 1.0

        S("Optional", S("float")).check_value(FloatProto())
        S("Optional", S("float")).check_value(None)

    def test_object_protocol_in_union(self) -> None:
        """Union[testing.TestIntPair, int] accepts __tvm_ffi_object__ value."""
        inner = TestIntPair(1, 2)

        class ObjProto:
            def __tvm_ffi_object__(self) -> object:
                return inner

        S("Union", S("testing.TestIntPair"), S("int")).check_value(ObjProto())

    def test_value_protocol_in_array(self) -> None:
        """Array[int] elements use __tvm_ffi_value__ fallback (recursive)."""

        class ValProto:
            def __tvm_ffi_value__(self) -> object:
                return 42

        # __tvm_ffi_value__ fallback is applied recursively at every level,
        # matching the marshal path where TVMFFIPyArgSetterFactory_ is
        # called per-element.
        S("Array", S("int")).check_value([ValProto()])

    def test_device_protocol_in_map_value(self) -> None:
        """Map[str, Device] accepts __dlpack_device__ values."""

        class DevProto:
            def __dlpack_device__(self) -> tuple[int, int]:
                return (1, 0)

        S("Map", S("str"), S("Device")).check_value({"gpu": DevProto()})


# ---------------------------------------------------------------------------
# Category 56: Nested __tvm_ffi_value__ in containers (recursive fallback)
# ---------------------------------------------------------------------------
class TestNestedValueProtocol:
    """__tvm_ffi_value__ fallback works recursively inside containers."""

    def test_value_in_array_elements(self) -> None:
        """Array[int] elements with __tvm_ffi_value__ are accepted."""

        class VP:
            def __tvm_ffi_value__(self) -> object:
                return 42

        S("Array", S("int")).check_value([1, VP(), 3])

    def test_value_in_map_values(self) -> None:
        """Map[str, int] values with __tvm_ffi_value__ are accepted."""

        class VP:
            def __tvm_ffi_value__(self) -> object:
                return 99

        S("Map", S("str"), S("int")).check_value({"a": VP()})

    def test_value_in_map_keys(self) -> None:
        """Map[str, int] keys with __tvm_ffi_value__ are accepted."""

        class VP:
            def __tvm_ffi_value__(self) -> object:
                return "key"

        S("Map", S("str"), S("int")).check_value({VP(): 1})

    def test_value_in_tuple_positions(self) -> None:
        """tuple[int, str] positions with __tvm_ffi_value__ are accepted."""

        class IntVP:
            def __tvm_ffi_value__(self) -> object:
                return 42

        class StrVP:
            def __tvm_ffi_value__(self) -> object:
                return "hello"

        S("tuple", S("int"), S("str")).check_value((IntVP(), StrVP()))

    def test_value_in_optional_inner(self) -> None:
        """Optional[int] inner with __tvm_ffi_value__ is accepted."""

        class VP:
            def __tvm_ffi_value__(self) -> object:
                return 42

        S("Optional", S("int")).check_value(VP())

    def test_value_in_union_alternatives(self) -> None:
        """Union[int, str] with __tvm_ffi_value__ is accepted."""

        class VP:
            def __tvm_ffi_value__(self) -> object:
                return "hello"

        S("Union", S("int"), S("str")).check_value(VP())

    def test_multi_hop_value_in_container(self) -> None:
        """Nested __tvm_ffi_value__ unwrapping inside containers."""

        class VP:
            def __init__(self, v: object) -> None:
                self.v = v

            def __tvm_ffi_value__(self) -> object:
                return self.v

        S("Array", S("int")).check_value([VP(VP(10))])

    def test_value_convert_in_array(self) -> None:
        """Convert returns unwrapped values in container."""

        class VP:
            def __tvm_ffi_value__(self) -> object:
                return 42

        result = S("Array", S("int")).convert([VP()]).to_py()
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
            S("int").check_value(SelfCycle())

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


# ---------------------------------------------------------------------------
# Category 58: Object marshal fallback
# ---------------------------------------------------------------------------
class TestObjectMarshalFallback:
    """Object schema accepts values that the marshal path converts to Objects."""

    def test_exception_accepted_by_object_schema(self) -> None:
        """TypeSchema('Object') accepts Exception (-> ffi.Error)."""
        S("Object").check_value(RuntimeError("test"))

    def test_exception_accepted_by_error_schema(self) -> None:
        """TypeSchema('ffi.Error') accepts Exception."""
        S("ffi.Error").check_value(ValueError("oops"))

    def test_exception_rejected_by_array_schema(self) -> None:
        """Exception is NOT accepted by Array schema (Error !IS-A Array)."""
        with pytest.raises(TypeError, match=r"ffi\.Error"):
            S("ffi.Array").check_value(RuntimeError("x"))

    def test_opaque_object_accepted_by_object_schema(self) -> None:
        """TypeSchema('Object') accepts arbitrary Python objects (-> OpaquePyObject)."""

        class Custom:
            pass

        S("Object").check_value(Custom())

    def test_plain_object_accepted_by_object_schema(self) -> None:
        """TypeSchema('Object') accepts object()."""
        S("Object").check_value(object())

    def test_opaque_rejected_by_specific_schema(self) -> None:
        """Specific schema rejects arbitrary Python object."""

        class Custom:
            pass

        with pytest.raises(TypeError, match="OpaquePyObject"):
            S("testing.TestIntPair").check_value(Custom())

    def test_str_accepted_by_object_schema(self) -> None:
        """TypeSchema('Object') accepts str (-> ffi.String IS-A Object)."""
        S("Object").check_value("hello")

    def test_list_accepted_by_object_schema(self) -> None:
        """TypeSchema('Object') accepts list (-> ffi.Array IS-A Object)."""
        S("Object").check_value([1, 2, 3])

    def test_dict_accepted_by_object_schema(self) -> None:
        """TypeSchema('Object') accepts dict (-> ffi.Map IS-A Object)."""
        S("Object").check_value({"a": 1})

    def test_callable_accepted_by_object_schema(self) -> None:
        """TypeSchema('Object') accepts callable (-> ffi.Function IS-A Object)."""
        S("Object").check_value(lambda: None)

    def test_int_rejected_by_object_schema(self) -> None:
        """TypeSchema('Object') rejects int (int is a POD type, not Object)."""
        with pytest.raises(TypeError):
            S("Object").check_value(42)

    def test_float_rejected_by_object_schema(self) -> None:
        """TypeSchema('Object') rejects float (float is a POD, not Object)."""
        with pytest.raises(TypeError):
            S("Object").check_value(3.14)

    def test_none_rejected_by_object_schema(self) -> None:
        """TypeSchema('Object') rejects None (None is a POD, not Object)."""
        with pytest.raises(TypeError):
            S("Object").check_value(None)


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

        S("ctypes.c_void_p").check_value(CUStream())

    def test_cuda_stream_convert(self) -> None:
        """Convert returns __cuda_stream__ value as-is."""

        class CUStream:
            def __cuda_stream__(self) -> tuple[int, int]:
                return (0, 123)

        obj = CUStream()
        result = S("ctypes.c_void_p").convert(obj).to_py()
        assert result is not None

    def test_cuda_stream_and_opaque_ptr(self) -> None:
        """Object with both __cuda_stream__ and __tvm_ffi_opaque_ptr__ accepted."""

        class DualProto:
            def __cuda_stream__(self) -> tuple[int, int]:
                return (0, 0)

            def __tvm_ffi_opaque_ptr__(self) -> int:
                return 0

        S("ctypes.c_void_p").check_value(DualProto())


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
            S("Device").check_value(TensorLike())

    def test_both_dlpack_and_device_accepted_by_tensor(self) -> None:
        """Object with both __dlpack__ and __dlpack_device__ accepted by Tensor."""

        class TensorLike:
            def __dlpack__(self) -> object:
                return None

            def __dlpack_device__(self) -> tuple[int, int]:
                return (1, 0)

        S("Tensor").check_value(TensorLike())

    def test_device_only_accepted_by_device(self) -> None:
        """Object with only __dlpack_device__ still accepted by Device."""

        class DevOnly:
            def __dlpack_device__(self) -> tuple[int, int]:
                return (1, 0)

        S("Device").check_value(DevOnly())

    def test_dlpack_only_rejected_by_device(self) -> None:
        """Object with only __dlpack__ rejected by Device schema."""

        class DLPackOnly:
            def __dlpack__(self) -> object:
                return None

        with pytest.raises(TypeError):
            S("Device").check_value(DLPackOnly())


# ---------------------------------------------------------------------------
# Category 61: SKIP_DLPACK_C_EXCHANGE_API env gate
# ---------------------------------------------------------------------------
class TestSkipDlpackEnvGate:
    """Tensor schema respects TVM_FFI_SKIP_DLPACK_C_EXCHANGE_API."""

    def test_exchange_api_accepted_by_default(self) -> None:
        """__dlpack_c_exchange_api__ accepted when env not set."""
        os.environ.pop("TVM_FFI_SKIP_DLPACK_C_EXCHANGE_API", None)

        class ExchangeAPI:
            def __dlpack_c_exchange_api__(self) -> int:
                return 0

        S("Tensor").check_value(ExchangeAPI())

    def test_exchange_api_rejected_when_skipped(self) -> None:
        """__dlpack_c_exchange_api__ rejected when env=1."""
        os.environ["TVM_FFI_SKIP_DLPACK_C_EXCHANGE_API"] = "1"
        try:

            class ExchangeAPI:
                def __dlpack_c_exchange_api__(self) -> int:
                    return 0

            with pytest.raises(TypeError):
                S("Tensor").check_value(ExchangeAPI())
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

    def test_array_int_exact_list(self) -> None:
        """Array[int] on exact Python list converts successfully."""
        original = [1, 2, 3]
        result = S("Array", S("int")).convert(original).to_py()
        assert list(result) == original

    def test_array_int_needs_conversion(self) -> None:
        """Array[int] on list needing bool->int returns converted list."""
        original = [1, True, 3]
        result = S("Array", S("int")).convert(original).to_py()
        assert list(result) == [1, 1, 3]

    def test_map_str_int_exact_dict(self) -> None:
        """Map[str, int] on exact dict converts successfully."""
        original = {"a": 1, "b": 2}
        result = S("Map", S("str"), S("int")).convert(original).to_py()
        assert dict(result) == original

    def test_map_str_int_needs_conversion(self) -> None:
        """Map[str, int] on dict needing conversion returns converted dict."""
        original = {"a": True, "b": 2}
        result = S("Map", S("str"), S("int")).convert(original).to_py()
        assert result is not None

    def test_tuple_exact_match(self) -> None:
        """tuple[int, str] on exact tuple converts successfully."""
        original = (42, "hello")
        result = S("tuple", S("int"), S("str")).convert(original).to_py()
        assert tuple(result) == original

    def test_tuple_needs_conversion(self) -> None:
        """tuple[int, str] on tuple needing conversion returns converted tuple."""
        original = (True, "hello")
        result = S("tuple", S("int"), S("str")).convert(original).to_py()
        assert tuple(result) == (1, "hello")

    def test_list_int_exact(self) -> None:
        """List[int] on exact list converts successfully."""
        original = [10, 20]
        result = S("List", S("int")).convert(original).to_py()
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
            S("int").check_value(BrokenInt())


# ---------------------------------------------------------------------------
# Category 66: __tvm_ffi_value__ precedence gate
# ---------------------------------------------------------------------------
class TestValueProtocolPrecedence:
    """__tvm_ffi_value__ fallback is gated by marshal precedence."""

    def test_int_protocol_takes_precedence(self) -> None:
        """Class with __tvm_ffi_int__ + __tvm_ffi_value__ dispatches as int."""

        class Dual:
            def __tvm_ffi_int__(self) -> int:
                return 42

            def __tvm_ffi_value__(self) -> object:
                return TestIntPair(1, 2)

        # Int schema: accepts via __tvm_ffi_int__ (direct)
        S("int").check_value(Dual())
        # Object schema: rejects — marshal dispatches as int, not Object
        with pytest.raises(TypeError):
            S("Object").check_value(Dual())

    def test_float_protocol_takes_precedence(self) -> None:
        """Class with __tvm_ffi_float__ + __tvm_ffi_value__ dispatches as float."""

        class Dual:
            def __tvm_ffi_float__(self) -> float:
                return 1.0

            def __tvm_ffi_value__(self) -> object:
                return TestIntPair(1, 2)

        S("float").check_value(Dual())
        with pytest.raises(TypeError):
            S("Object").check_value(Dual())

    def test_pure_value_protocol_still_works(self) -> None:
        """Class with ONLY __tvm_ffi_value__ still uses fallback."""

        class PureVP:
            def __tvm_ffi_value__(self) -> object:
                return 42

        S("int").check_value(PureVP())

    def test_callable_takes_precedence(self) -> None:
        """Callable class with __tvm_ffi_value__ dispatches as callable."""

        class CallableVP:
            def __call__(self) -> None:
                pass

            def __tvm_ffi_value__(self) -> object:
                return 42

        # Callable schema accepts (direct)
        S("Callable").check_value(CallableVP())
        # Int schema: __tvm_ffi_value__ NOT applied (callable has precedence)
        with pytest.raises(TypeError):
            S("int").check_value(CallableVP())


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

        S("Union", S("str"), S("int")).check_value(CountingVP())
        assert call_count == 1

    def test_union_value_protocol_mismatch(self) -> None:
        """__tvm_ffi_value__ returning wrong type fails Union."""

        class WrongVP:
            def __tvm_ffi_value__(self) -> object:
                return object()

        with pytest.raises(TypeError):
            S("Union", S("int"), S("str")).check_value(WrongVP())


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
        S("int").check_value(DualProtocol())
        # Object schema: __tvm_ffi_value__ returns 42 (POD int, not Object),
        # should REJECT (not accept via ObjectConvertible)
        with pytest.raises(TypeError):
            S("Object").check_value(DualProtocol())

    def test_pure_convertible_still_works(self) -> None:
        """ObjectConvertible without __tvm_ffi_value__ still accepted."""
        pair = TestIntPair(1, 2)

        class PureConvertible(ObjectConvertible):
            def asobject(self) -> tvm_ffi.core.Object:
                return pair

        S("Object").check_value(PureConvertible())
        S("testing.TestIntPair").check_value(PureConvertible())


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
        cany = S("None").convert(None)
        assert isinstance(cany, CAny)
        assert cany.type_index == 0  # kTVMFFINone

    def test_cany_from_str(self) -> None:
        """convert(str) returns CAny."""
        cany = A(str).convert("hello")
        assert isinstance(cany, CAny)
        # Short strings have type_index=11 (SmallStr), longer ones have 65 (Str)
        assert cany.type_index in (11, 65)

    def test_cany_from_array(self) -> None:
        """convert(Array) returns CAny with array type_index."""
        cany = S("Array", S("int")).convert([1, 2, 3])
        assert isinstance(cany, CAny)
        assert cany.type_index >= 64  # object type

    def test_to_py_int(self) -> None:
        """to_py() round-trips int correctly."""
        result = A(int).convert(42).to_py()
        assert result == 42
        assert type(result) is int

    def test_to_py_float(self) -> None:
        """to_py() round-trips float correctly."""
        result = A(float).convert(3.14).to_py()
        assert result == 3.14
        assert type(result) is float

    def test_to_py_bool(self) -> None:
        """to_py() round-trips bool correctly."""
        assert A(bool).convert(True).to_py() is True
        assert A(bool).convert(False).to_py() is False

    def test_to_py_none(self) -> None:
        """to_py() round-trips None correctly."""
        assert S("None").convert(None).to_py() is None

    def test_to_py_str(self) -> None:
        """to_py() round-trips str correctly."""
        assert A(str).convert("hello").to_py() == "hello"

    def test_to_py_array(self) -> None:
        """to_py() returns ffi.Array for Array convert."""
        result = S("Array", S("int")).convert([1, 2, 3]).to_py()
        assert isinstance(result, tvm_ffi.Array)
        assert list(result) == [1, 2, 3]

    def test_to_py_list(self) -> None:
        """to_py() returns ffi.List for List convert."""
        result = S("List", S("int")).convert([1, 2, 3]).to_py()
        assert isinstance(result, tvm_ffi.List)
        assert list(result) == [1, 2, 3]

    def test_to_py_map(self) -> None:
        """to_py() returns ffi.Map for Map convert."""
        result = S("Map", S("str"), S("int")).convert({"a": 1}).to_py()
        assert isinstance(result, tvm_ffi.Map)

    def test_to_py_dict(self) -> None:
        """to_py() returns ffi.Dict for Dict convert."""
        result = S("Dict", S("str"), S("int")).convert({"a": 1}).to_py()
        assert isinstance(result, tvm_ffi.Dict)

    def test_multiple_to_py_calls(self) -> None:
        """to_py() can be called multiple times safely."""
        cany = A(int).convert(42)
        assert cany.to_py() == 42
        assert cany.to_py() == 42
        assert cany.to_py() == 42

    def test_object_refcount_safety(self) -> None:
        """to_py() for objects properly IncRefs — no double-free."""
        cany = S("Array", S("int")).convert([1, 2, 3])
        py1 = cany.to_py()
        py2 = cany.to_py()
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
        cany = S("None").convert(None)
        assert "None" in repr(cany)

    def test_repr_float(self) -> None:
        """Repr shows float value."""
        cany = A(float).convert(3.14)
        assert "float" in repr(cany)

    def test_repr_object(self) -> None:
        """Repr shows type_index for objects."""
        cany = S("Array", S("int")).convert([1, 2, 3])
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


class TestFromAnnotationCallable:
    """Callable annotation tests."""

    def test_bare(self) -> None:
        """Bare Callable."""
        assert A(Callable) == S("Callable")

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

    def test_int(self) -> None:
        """list[int]."""
        assert A(list[int]) == S("List", S("int"))

    def test_nested(self) -> None:
        """list[list[int]]."""
        assert A(list[list[int]]) == S("List", S("List", S("int")))


class TestFromAnnotationDict:
    """dict[K, V] → Dict tests."""

    def test_bare(self) -> None:
        """Bare dict."""
        assert A(dict).origin == "Dict"

    def test_str_int(self) -> None:
        """dict[str, int]."""
        assert A(dict[str, int]) == S("Dict", S("str"), S("int"))


class TestFromAnnotationArray:
    """tuple[T, ...] → Array tests."""

    def test_int(self) -> None:
        """tuple[int, ...]."""
        assert A(tuple[int, ...]) == S("Array", S("int"))

    def test_float(self) -> None:
        """tuple[float, ...]."""
        assert A(tuple[float, ...]) == S("Array", S("float"))


class TestFromAnnotationTuple:
    """tuple[T1, T2] (fixed) tests."""

    def test_bare(self) -> None:
        """Bare tuple."""
        assert A(tuple).origin == "tuple"

    def test_int_str(self) -> None:
        """tuple[int, str]."""
        assert A(tuple[int, str]) == S("tuple", S("int"), S("str"))


class TestFromAnnotationOptional:
    """Optional[T] tests."""

    def test_int(self) -> None:
        """Optional[int]."""
        assert A(Optional[int]) == S("Optional", S("int"))

    @pytest.mark.skipif(sys.version_info < (3, 10), reason="X | Y requires 3.10+")
    def test_pipe_syntax(self) -> None:
        """Int | None."""
        assert A(eval("int | None")) == S("Optional", S("int"))


class TestFromAnnotationUnion:
    """Union[T1, T2] tests."""

    def test_int_str(self) -> None:
        """Union[int, str]."""
        assert A(Union[int, str]) == S("Union", S("int"), S("str"))

    @pytest.mark.skipif(sys.version_info < (3, 10), reason="X | Y requires 3.10+")
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

    def test_list_too_many_args(self) -> None:
        """list[int, int, float] raises."""
        with pytest.raises(TypeError, match="list takes at most 1"):
            A(list[int, int, float])  # type: ignore[type-arg]

    def test_dict_one_arg(self) -> None:
        """dict[str] raises."""
        with pytest.raises(TypeError, match="dict requires 0 or 2"):
            A(dict[str])  # type: ignore[type-arg]


# ---------------------------------------------------------------------------
# Convert returns FFI containers
# ---------------------------------------------------------------------------
import tvm_ffi as _tvm_ffi


class TestConvertReturnFFIContainers:
    """convert().to_py() returns ffi.Array/List/Map/Dict."""

    def test_array_from_list(self) -> None:
        """Array convert from Python list."""
        result = A(tuple[float, ...]).convert([1, 2, 3]).to_py()
        assert isinstance(result, _tvm_ffi.Array)
        assert list(result) == [1.0, 2.0, 3.0]

    def test_list_from_list(self) -> None:
        """List convert from Python list."""
        result = A(list[int]).convert([1, 2, 3]).to_py()
        assert isinstance(result, _tvm_ffi.List)
        assert list(result) == [1, 2, 3]

    def test_dict_from_dict(self) -> None:
        """Dict convert from Python dict."""
        result = A(dict[str, int]).convert({"a": 1}).to_py()
        assert isinstance(result, _tvm_ffi.Dict)

    def test_map_from_dict(self) -> None:
        """Map convert from Python dict."""
        result = S("Map", S("str"), S("int")).convert({"a": 1}).to_py()
        assert isinstance(result, _tvm_ffi.Map)

    def test_array_passthrough(self) -> None:
        """ffi.Array input passes through unchanged."""
        arr = _tvm_ffi.Array([1, 2, 3])
        result = A(tuple[int, ...]).convert(arr).to_py()
        assert result.same_as(arr)

    def test_list_passthrough(self) -> None:
        """ffi.List input passes through unchanged."""
        lst = _tvm_ffi.List([1, 2, 3])
        result = A(list[int]).convert(lst).to_py()
        assert result.same_as(lst)

    def test_nested_array_convert(self) -> None:
        """Nested array conversion."""
        result = A(tuple[tuple[int, ...], ...]).convert([[1, 2], [3, 4]]).to_py()
        assert isinstance(result, _tvm_ffi.Array)
        assert isinstance(result[0], _tvm_ffi.Array)


# ---------------------------------------------------------------------------
# FFI type guarantees: convert().to_py() always returns tvm_ffi types
# ---------------------------------------------------------------------------
class TestConvertToFFITypes:
    """convert().to_py() returns canonical FFI types for all value kinds."""

    def test_short_str_is_string(self) -> None:
        """Short str (SmallStr) promotes to tvm_ffi.String."""
        result = A(str).convert("hi").to_py()
        assert isinstance(result, tvm_ffi.core.String)
        assert result == "hi"

    def test_long_str_is_string(self) -> None:
        """Long str (kTVMFFIStr object) is tvm_ffi.String."""
        long_s = "x" * 200
        result = A(str).convert(long_s).to_py()
        assert isinstance(result, tvm_ffi.core.String)
        assert result == long_s

    def test_empty_str_is_string(self) -> None:
        """Empty str is tvm_ffi.String."""
        result = A(str).convert("").to_py()
        assert isinstance(result, tvm_ffi.core.String)
        assert result == ""

    def test_short_bytes_is_bytes(self) -> None:
        """Short bytes (SmallBytes) promotes to tvm_ffi.Bytes."""
        result = A(bytes).convert(b"hi").to_py()
        assert isinstance(result, tvm_ffi.core.Bytes)
        assert result == b"hi"

    def test_long_bytes_is_bytes(self) -> None:
        """Long bytes (kTVMFFIBytes object) is tvm_ffi.Bytes."""
        long_b = b"x" * 200
        result = A(bytes).convert(long_b).to_py()
        assert isinstance(result, tvm_ffi.core.Bytes)
        assert result == long_b

    def test_empty_bytes_is_bytes(self) -> None:
        """Empty bytes is tvm_ffi.Bytes."""
        result = A(bytes).convert(b"").to_py()
        assert isinstance(result, tvm_ffi.core.Bytes)
        assert result == b""

    def test_bytearray_converts_to_ffi_bytes(self) -> None:
        """Bytearray converts to tvm_ffi.Bytes."""
        result = A(bytes).convert(bytearray(b"hello")).to_py()
        assert isinstance(result, tvm_ffi.core.Bytes)
        assert result == b"hello"

    def test_callable_is_function(self) -> None:
        """Callable converts to tvm_ffi.Function."""
        result = A(Callable).convert(lambda x: x).to_py()
        assert isinstance(result, tvm_ffi.core.Function)

    def test_array_is_ffi_array(self) -> None:
        """Array[int] converts to tvm_ffi.Array."""
        result = S("Array", S("int")).convert([1, 2]).to_py()
        assert isinstance(result, _tvm_ffi.Array)

    def test_list_is_ffi_list(self) -> None:
        """List[int] converts to tvm_ffi.List."""
        result = S("List", S("int")).convert([1, 2]).to_py()
        assert isinstance(result, _tvm_ffi.List)

    def test_map_is_ffi_map(self) -> None:
        """Map[str, int] converts to tvm_ffi.Map."""
        result = S("Map", S("str"), S("int")).convert({"a": 1}).to_py()
        assert isinstance(result, _tvm_ffi.Map)

    def test_dict_is_ffi_dict(self) -> None:
        """Dict[str, int] converts to tvm_ffi.Dict."""
        result = S("Dict", S("str"), S("int")).convert({"a": 1}).to_py()
        assert isinstance(result, _tvm_ffi.Dict)

    def test_int_is_int(self) -> None:
        """Int stays as int."""
        result = A(int).convert(42).to_py()
        assert type(result) is int
        assert result == 42

    def test_float_is_float(self) -> None:
        """Float stays as float."""
        result = A(float).convert(3.14).to_py()
        assert type(result) is float
        assert result == 3.14

    def test_bool_is_bool(self) -> None:
        """Bool stays as bool."""
        result = A(bool).convert(True).to_py()
        assert result is True

    def test_none_is_none(self) -> None:
        """None stays as None."""
        result = S("None").convert(None).to_py()
        assert result is None

    def test_object_is_cobject(self) -> None:
        """Object converts to CObject subclass."""
        obj = TestIntPair(1, 2)
        result = S("testing.TestIntPair").convert(obj).to_py()
        assert isinstance(result, tvm_ffi.core.CObject)
        assert result.same_as(obj)
