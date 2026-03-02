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
"""Tests for TypeSchema type conversion (convert/try_convert/check_value)."""

from __future__ import annotations

import ctypes
import os
from numbers import Integral

import pytest
import tvm_ffi
from tvm_ffi.core import ObjectConvertible, TypeSchema
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
    """Shorthand constructor for TypeSchema."""
    return TypeSchema(origin, tuple(args))


# ---------------------------------------------------------------------------
# Category 1: POD type exact match (check_value)
# ---------------------------------------------------------------------------
class TestPODExactMatch:
    def test_int(self) -> None:
        """Test int."""
        S("int").check_value(42)

    def test_float(self) -> None:
        """Test float."""
        S("float").check_value(3.14)

    def test_bool_true(self) -> None:
        """Test bool true."""
        S("bool").check_value(True)

    def test_bool_false(self) -> None:
        """Test bool false."""
        S("bool").check_value(False)

    def test_str(self) -> None:
        """Test str."""
        S("str").check_value("hello")

    def test_bytes(self) -> None:
        """Test bytes."""
        S("bytes").check_value(b"data")

    def test_none(self) -> None:
        """Test none."""
        S("None").check_value(None)


# ---------------------------------------------------------------------------
# Category 2: Implicit conversions (mirrors TryCastFromAnyView)
# ---------------------------------------------------------------------------
class TestImplicitConversions:
    def test_bool_to_int(self) -> None:
        """Bool -> int is OK (C++: int accepts bool)."""
        S("int").check_value(True)

    def test_int_to_float(self) -> None:
        """Int -> float is OK (C++: float accepts int)."""
        S("float").check_value(42)

    def test_bool_to_float(self) -> None:
        """Bool -> float is OK (C++: float accepts bool)."""
        S("float").check_value(True)

    def test_int_to_bool(self) -> None:
        """Int -> bool is OK (C++: bool accepts int)."""
        S("bool").check_value(1)


# ---------------------------------------------------------------------------
# Category 3: Rejection cases
# ---------------------------------------------------------------------------
class TestRejections:
    def test_str_not_int(self) -> None:
        """Test str not int."""
        assert S("int").try_check_value("hello") is not None
        with pytest.raises(TypeError, match="expected int"):
            S("int").check_value("hello")

    def test_float_not_int(self) -> None:
        """Test float not int."""
        assert S("int").try_check_value(3.14) is not None

    def test_none_not_int(self) -> None:
        """Test none not int."""
        assert S("int").try_check_value(None) is not None

    def test_int_not_str(self) -> None:
        """Test int not str."""
        assert S("str").try_check_value(42) is not None

    def test_str_not_bool(self) -> None:
        """Test str not bool."""
        assert S("bool").try_check_value("hello") is not None

    def test_none_not_str(self) -> None:
        """Test none not str."""
        assert S("str").try_check_value(None) is not None

    def test_int_not_bytes(self) -> None:
        """Test int not bytes."""
        assert S("bytes").try_check_value(42) is not None

    def test_int_not_none(self) -> None:
        """Test int not none."""
        assert S("None").try_check_value(42) is not None


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
        assert S("Device").try_check_value(42) is not None

    def test_dtype_pass(self) -> None:
        """Test dtype pass."""
        dt = tvm_ffi.core.DataType("float32")
        S("dtype").check_value(dt)

    def test_dtype_str_pass(self) -> None:
        """Str accepted as dtype (will be parsed)."""
        S("dtype").check_value("float32")

    def test_dtype_fail(self) -> None:
        """Test dtype fail."""
        assert S("dtype").try_check_value(42) is not None

    def test_opaque_ptr_pass(self) -> None:
        """Test opaque ptr pass."""
        S("ctypes.c_void_p").check_value(ctypes.c_void_p(0))

    def test_opaque_ptr_none_pass(self) -> None:
        """Test opaque ptr none pass."""
        S("ctypes.c_void_p").check_value(None)

    def test_opaque_ptr_fail(self) -> None:
        """Test opaque ptr fail."""
        assert S("ctypes.c_void_p").try_check_value(42) is not None

    def test_callable_pass_function(self) -> None:
        """Test callable pass function."""
        S("Callable").check_value(lambda x: x)

    def test_callable_pass_builtin(self) -> None:
        """Test callable pass builtin."""
        S("Callable").check_value(len)

    def test_callable_fail(self) -> None:
        """Test callable fail."""
        assert S("Callable").try_check_value(42) is not None


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
        assert S("Object").try_check_value(42) is not None

    def test_specific_object_pass(self) -> None:
        """A Function object should pass its own type schema."""
        f = tvm_ffi.get_global_func("testing.echo")
        S("Callable").check_value(f)


# ---------------------------------------------------------------------------
# Category 6: Optional
# ---------------------------------------------------------------------------
class TestOptional:
    def test_none_passes(self) -> None:
        """Test none passes."""
        S("Optional", S("int")).check_value(None)

    def test_inner_type_passes(self) -> None:
        """Test inner type passes."""
        S("Optional", S("int")).check_value(42)

    def test_wrong_type_fails(self) -> None:
        """Test wrong type fails."""
        err = S("Optional", S("int")).try_check_value("hello")
        assert err is not None
        assert "expected int" in err

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
        S("Union", S("int"), S("str")).check_value(42)

    def test_second_alt_passes(self) -> None:
        """Test second alt passes."""
        S("Union", S("int"), S("str")).check_value("hello")

    def test_no_alt_matches(self) -> None:
        """Test no alt matches."""
        err = S("Union", S("int"), S("str")).try_check_value(3.14)
        assert err is not None
        assert "got float" in err

    def test_bool_matches_int_alt(self) -> None:
        """Bool is accepted by the int alternative."""
        S("Union", S("int"), S("str")).check_value(True)


# ---------------------------------------------------------------------------
# Category 8: Containers
# ---------------------------------------------------------------------------
class TestContainers:
    def test_array_list_pass(self) -> None:
        """Test array list pass."""
        S("Array", S("int")).check_value([1, 2, 3])

    def test_array_tuple_pass(self) -> None:
        """Test array tuple pass."""
        S("Array", S("int")).check_value((1, 2, 3))

    def test_array_wrong_element(self) -> None:
        """Test array wrong element."""
        err = S("Array", S("int")).try_check_value([1, "x"])
        assert err is not None
        assert "element [1]" in err
        assert "expected int" in err

    def test_array_empty_pass(self) -> None:
        """Test array empty pass."""
        S("Array", S("int")).check_value([])

    def test_array_any_pass(self) -> None:
        """Test array any pass."""
        S("Array", S("Any")).check_value([1, "x", None])

    def test_array_wrong_container_type(self) -> None:
        """Test array wrong container type."""
        err = S("Array", S("int")).try_check_value(42)
        assert err is not None
        assert "expected Array" in err

    def test_list_pass(self) -> None:
        """Test list pass."""
        S("List", S("str")).check_value(["a", "b"])

    def test_map_pass(self) -> None:
        """Test map pass."""
        S("Map", S("str"), S("int")).check_value({"a": 1, "b": 2})

    def test_map_wrong_key(self) -> None:
        """Test map wrong key."""
        err = S("Map", S("str"), S("int")).try_check_value({1: 2})
        assert err is not None
        assert "key" in err
        assert "expected str" in err

    def test_map_wrong_value(self) -> None:
        """Test map wrong value."""
        err = S("Map", S("str"), S("int")).try_check_value({"a": "b"})
        assert err is not None
        assert "value for key" in err
        assert "expected int" in err

    def test_map_empty_pass(self) -> None:
        """Test map empty pass."""
        S("Map", S("str"), S("int")).check_value({})

    def test_dict_pass(self) -> None:
        """Test dict pass."""
        S("Dict", S("str"), S("int")).check_value({"a": 1})

    def test_map_wrong_container(self) -> None:
        """Test map wrong container."""
        err = S("Map", S("str"), S("int")).try_check_value([1, 2])
        assert err is not None
        assert "expected Map" in err


# ---------------------------------------------------------------------------
# Category 9: Nested types
# ---------------------------------------------------------------------------
class TestNestedTypes:
    def test_array_optional_int(self) -> None:
        """Test array optional int."""
        S("Array", S("Optional", S("int"))).check_value([1, None, 2])

    def test_map_str_array_int(self) -> None:
        """Test map str array int."""
        S("Map", S("str"), S("Array", S("int"))).check_value({"a": [1, 2]})

    def test_map_str_array_int_nested_fail(self) -> None:
        """Test map str array int nested fail."""
        err = S("Map", S("str"), S("Array", S("int"))).try_check_value({"a": [1, "x"]})
        assert err is not None
        assert "value for key 'a'" in err
        assert "element [1]" in err
        assert "expected int" in err

    def test_union_with_containers(self) -> None:
        """Test union with containers."""
        schema = S("Union", S("int"), S("Array", S("str")))
        schema.check_value(42)
        schema.check_value(["a", "b"])
        err = schema.try_check_value(3.14)
        assert err is not None


# ---------------------------------------------------------------------------
# Category 10: Any
# ---------------------------------------------------------------------------
class TestAny:
    def test_int(self) -> None:
        """Test int."""
        S("Any").check_value(42)

    def test_none(self) -> None:
        """Test none."""
        S("Any").check_value(None)

    def test_str(self) -> None:
        """Test str."""
        S("Any").check_value("hello")

    def test_list(self) -> None:
        """Test list."""
        S("Any").check_value([1, 2, 3])

    def test_object(self) -> None:
        """Test object."""
        S("Any").check_value(object())


# ---------------------------------------------------------------------------
# Category 11: Error message quality
# ---------------------------------------------------------------------------
class TestErrorMessages:
    def test_basic_type_mismatch(self) -> None:
        """Test basic type mismatch."""
        with pytest.raises(TypeError, match=r"expected int, got str"):
            S("int").check_value("hello")

    def test_nested_array_error(self) -> None:
        """Test nested array error."""
        with pytest.raises(TypeError, match=r"element \[2\].*expected int, got str"):
            S("Array", S("int")).check_value([1, 2, "x"])

    def test_nested_map_error(self) -> None:
        """Test nested map error."""
        with pytest.raises(TypeError, match=r"value for key 'b'.*expected int, got str"):
            S("Map", S("str"), S("int")).check_value({"a": 1, "b": "x"})

    def test_union_error_lists_alternatives(self) -> None:
        """Test union error lists alternatives."""
        err = S("Union", S("int"), S("str")).try_check_value(3.14)
        assert err is not None
        assert "int" in err
        assert "str" in err
        assert "got float" in err

    def test_schema_in_error_message(self) -> None:
        """check_value includes the schema repr in the TypeError."""
        with pytest.raises(TypeError, match=r"type check failed for"):
            S("int").check_value("hello")

    def test_convert_error_message(self) -> None:
        """Convert includes the schema repr in the TypeError."""
        with pytest.raises(TypeError, match=r"type conversion failed for"):
            S("int").convert("hello")


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
        assert schema.try_check_value("hello") is not None

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
        S("bytes").check_value(bytearray(b"data"))

    def test_tuple_passes_array(self) -> None:
        """Tuple is accepted as a sequence type for Array."""
        S("Array", S("int")).check_value((1, 2, 3))

    def test_empty_union_is_rejected(self) -> None:
        """Union requires at least 2 args."""
        with pytest.raises(ValueError, match="at least two"):
            TypeSchema("Union", ())

    def test_origin_type_index_auto_computed(self) -> None:
        """origin_type_index is automatically computed from origin string."""
        schema = S("int")
        assert schema.origin_type_index == 1  # kTVMFFIInt
        schema = S("float")
        assert schema.origin_type_index == 3  # kTVMFFIFloat
        schema = S("Optional", S("int"))
        assert schema.origin_type_index == -2  # structural

    def test_try_check_value_returns_none_on_success(self) -> None:
        """Test try check value returns none on success."""
        assert S("int").try_check_value(42) is None

    def test_try_check_value_returns_string_on_failure(self) -> None:
        """Test try check value returns string on failure."""
        result = S("int").try_check_value("hello")
        assert isinstance(result, str)
        assert "expected int" in result

    def test_tuple_type_schema(self) -> None:
        """Test tuple type schema."""
        schema = S("tuple", S("int"), S("str"))
        schema.check_value((1, "a"))
        assert schema.try_check_value((1, 2)) is not None
        assert schema.try_check_value((1,)) is not None

    def test_numpy_int_passes_int(self) -> None:
        """Numpy integer types should pass int check via Integral."""
        np = pytest.importorskip("numpy")
        S("int").check_value(np.int64(42))
        S("float").check_value(np.float64(3.14))


# ===========================================================================
# Type Converter Tests (convert / try_convert)
# ===========================================================================


# ---------------------------------------------------------------------------
# Category 14: POD conversion results
# ---------------------------------------------------------------------------
class TestConvertPOD:
    def test_int_passthrough(self) -> None:
        """Int -> int returns the same value."""
        result = S("int").convert(42)
        assert result == 42
        assert type(result) is int

    def test_bool_to_int(self) -> None:
        """Bool -> int actually converts to int."""
        result = S("int").convert(True)
        assert result == 1
        assert type(result) is int

    def test_bool_false_to_int(self) -> None:
        """Test bool false to int."""
        result = S("int").convert(False)
        assert result == 0
        assert type(result) is int

    def test_float_passthrough(self) -> None:
        """Test float passthrough."""
        result = S("float").convert(3.14)
        assert result == 3.14
        assert type(result) is float

    def test_int_to_float(self) -> None:
        """Int -> float actually converts."""
        result = S("float").convert(42)
        assert result == 42.0
        assert type(result) is float

    def test_bool_to_float(self) -> None:
        """Bool -> float actually converts."""
        result = S("float").convert(True)
        assert result == 1.0
        assert type(result) is float

    def test_bool_passthrough(self) -> None:
        """Test bool passthrough."""
        result = S("bool").convert(True)
        assert result is True
        assert type(result) is bool

    def test_int_to_bool(self) -> None:
        """Int -> bool actually converts."""
        result = S("bool").convert(1)
        assert result is True
        assert type(result) is bool

    def test_int_zero_to_bool(self) -> None:
        """Test int zero to bool."""
        result = S("bool").convert(0)
        assert result is False
        assert type(result) is bool

    def test_str_passthrough(self) -> None:
        """Test str passthrough."""
        result = S("str").convert("hello")
        assert result == "hello"
        assert type(result) is str

    def test_bytes_passthrough(self) -> None:
        """Test bytes passthrough."""
        result = S("bytes").convert(b"data")
        assert result == b"data"
        assert type(result) is bytes

    def test_bytearray_to_bytes(self) -> None:
        """Bytearray -> bytes actually converts."""
        result = S("bytes").convert(bytearray(b"data"))
        assert result == b"data"
        assert type(result) is bytes


# ---------------------------------------------------------------------------
# Category 15: None disambiguation (critical design point)
# ---------------------------------------------------------------------------
class TestNoneDisambiguation:
    def test_none_converts_successfully_for_none_schema(self) -> None:
        """TypeSchema('None').convert(None) returns None as a valid result."""
        result = S("None").convert(None)
        assert result is None

    def test_none_converts_successfully_for_optional(self) -> None:
        """Optional[int].convert(None) returns None as a valid result."""
        result = S("Optional", S("int")).convert(None)
        assert result is None

    def test_none_fails_for_int(self) -> None:
        """TypeSchema('int').convert(None) raises TypeError."""
        with pytest.raises(TypeError, match="expected int, got None"):
            S("int").convert(None)

    def test_try_convert_none_success(self) -> None:
        """try_convert distinguishes None-as-result from failure."""
        success, result = S("Optional", S("int")).try_convert(None)
        assert success is True
        assert result is None

    def test_try_convert_none_failure(self) -> None:
        """try_convert returns (False, error_msg) for failed conversion."""
        success, result = S("int").try_convert(None)
        assert success is False
        assert isinstance(result, str)
        assert "expected int" in result

    def test_try_convert_success_with_value(self) -> None:
        """try_convert returns (True, converted_value) on success."""
        success, result = S("int").try_convert(True)
        assert success is True
        assert result == 1
        assert type(result) is int

    def test_opaque_ptr_none_converts(self) -> None:
        """ctypes.c_void_p accepts None and returns None as valid result."""
        result = S("ctypes.c_void_p").convert(None)
        assert result is None

    def test_try_convert_opaque_ptr_none(self) -> None:
        """Test try convert opaque ptr none."""
        success, result = S("ctypes.c_void_p").try_convert(None)
        assert success is True
        assert result is None


# ---------------------------------------------------------------------------
# Category 16: Special type conversions
# ---------------------------------------------------------------------------
class TestConvertSpecialTypes:
    def test_dtype_str_converts(self) -> None:
        """Str -> dtype actually creates a DataType object."""
        result = S("dtype").convert("float32")
        assert isinstance(result, tvm_ffi.core.DataType)
        assert str(result) == "float32"

    def test_dtype_passthrough(self) -> None:
        """Test dtype passthrough."""
        dt = tvm_ffi.core.DataType("int32")
        result = S("dtype").convert(dt)
        assert result is dt

    def test_device_passthrough(self) -> None:
        """Test device passthrough."""
        dev = tvm_ffi.Device("cpu", 0)
        result = S("Device").convert(dev)
        assert result is dev

    def test_callable_passthrough(self) -> None:
        """Test callable passthrough."""
        fn = lambda x: x
        result = S("Callable").convert(fn)
        assert result is fn

    def test_opaque_ptr_passthrough(self) -> None:
        """Test opaque ptr passthrough."""
        ptr = ctypes.c_void_p(42)
        result = S("ctypes.c_void_p").convert(ptr)
        assert result is ptr


# ---------------------------------------------------------------------------
# Category 17: Container conversion results
# ---------------------------------------------------------------------------
class TestConvertContainers:
    def test_array_converts_bool_elements_to_int(self) -> None:
        """Array[int] with bool elements converts them to int."""
        result = S("Array", S("int")).convert([True, False, 1])
        assert result == [1, 0, 1]
        assert all(type(x) is int for x in result)

    def test_array_int_passthrough(self) -> None:
        """Array[int] with int elements returns new list."""
        result = S("Array", S("int")).convert([1, 2, 3])
        assert result == [1, 2, 3]

    def test_array_any_passthrough(self) -> None:
        """Array[Any] returns the original value."""
        original = [1, "x", None]
        result = S("Array", S("Any")).convert(original)
        assert result is original

    def test_map_converts_values(self) -> None:
        """Map[str, float] converts int values to float."""
        result = S("Map", S("str"), S("float")).convert({"a": 1, "b": 2})
        assert result == {"a": 1.0, "b": 2.0}
        assert all(type(v) is float for v in result.values())

    def test_map_any_any_passthrough(self) -> None:
        """Map[Any, Any] returns the original value."""
        original = {"a": 1}
        result = S("Map", S("Any"), S("Any")).convert(original)
        assert result is original

    def test_tuple_converts_elements(self) -> None:
        """tuple[int, float] converts elements positionally."""
        result = S("tuple", S("int"), S("float")).convert((True, 42))
        assert result == (1, 42.0)
        assert type(result[0]) is int
        assert type(result[1]) is float

    def test_nested_array_in_map(self) -> None:
        """Map[str, Array[int]] recursively converts elements."""
        result = S("Map", S("str"), S("Array", S("int"))).convert({"a": [True, False]})
        assert result == {"a": [1, 0]}
        assert all(type(x) is int for x in result["a"])


# ---------------------------------------------------------------------------
# Category 18: Optional/Union conversion results
# ---------------------------------------------------------------------------
class TestConvertComposite:
    def test_optional_converts_inner(self) -> None:
        """Optional[float].convert(42) converts int -> float."""
        result = S("Optional", S("float")).convert(42)
        assert result == 42.0
        assert type(result) is float

    def test_optional_none(self) -> None:
        """Test optional none."""
        result = S("Optional", S("float")).convert(None)
        assert result is None

    def test_union_picks_first_match(self) -> None:
        """Union[int, str] converts bool via int alternative."""
        result = S("Union", S("int"), S("str")).convert(True)
        assert result == 1
        assert type(result) is int

    def test_union_second_match(self) -> None:
        """Test union second match."""
        result = S("Union", S("int"), S("str")).convert("hello")
        assert result == "hello"

    def test_any_passthrough(self) -> None:
        """Any returns value as-is."""
        result = S("Any").convert(42)
        assert result == 42
        result = S("Any").convert(None)
        assert result is None


# ---------------------------------------------------------------------------
# Category 19: Convert rejection cases
# ---------------------------------------------------------------------------
class TestConvertRejections:
    def test_int_rejects_str(self) -> None:
        """Test int rejects str."""
        with pytest.raises(TypeError, match="expected int, got str"):
            S("int").convert("hello")

    def test_int_rejects_float(self) -> None:
        """Test int rejects float."""
        with pytest.raises(TypeError, match="expected int, got float"):
            S("int").convert(3.14)

    def test_str_rejects_int(self) -> None:
        """Test str rejects int."""
        with pytest.raises(TypeError, match="expected str, got int"):
            S("str").convert(42)

    def test_array_rejects_wrong_element(self) -> None:
        """Test array rejects wrong element."""
        with pytest.raises(TypeError, match=r"element \[1\].*expected int, got str"):
            S("Array", S("int")).convert([1, "x"])

    def test_map_rejects_wrong_value(self) -> None:
        """Test map rejects wrong value."""
        with pytest.raises(TypeError, match=r"value for key 'a'.*expected int, got str"):
            S("Map", S("str"), S("int")).convert({"a": "x"})

    def test_tuple_rejects_wrong_length(self) -> None:
        """Test tuple rejects wrong length."""
        with pytest.raises(TypeError, match=r"expected tuple of length 2"):
            S("tuple", S("int"), S("str")).convert((1,))

    def test_try_convert_failure(self) -> None:
        """Test try convert failure."""
        success, result = S("int").try_convert("hello")
        assert success is False
        assert "expected int" in result


# ---------------------------------------------------------------------------
# Category 20: Numpy conversion
# ---------------------------------------------------------------------------
class TestConvertNumpy:
    def test_numpy_int_to_int(self) -> None:
        """Test numpy int to int."""
        np = pytest.importorskip("numpy")
        result = S("int").convert(np.int64(42))
        assert result == 42
        assert type(result) is int

    def test_numpy_float_to_float(self) -> None:
        """Test numpy float to float."""
        np = pytest.importorskip("numpy")
        result = S("float").convert(np.float64(3.14))
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
        result = S("Array", S("Optional", S("float"))).convert([True, None, 3])
        assert result == [1.0, None, 3.0]
        assert type(result[0]) is float
        assert result[1] is None
        assert type(result[2]) is float

    def test_array_optional_int_with_bool(self) -> None:
        """Array[Optional[int]] converts bool elements to int."""
        result = S("Array", S("Optional", S("int"))).convert([True, None, 2])
        assert result == [1, None, 2]
        assert type(result[0]) is int
        assert result[1] is None

    def test_array_union_int_str_with_bool(self) -> None:
        """Array[Union[int, str]] converts bool via int alternative."""
        result = S("Array", S("Union", S("int"), S("str"))).convert([True, "hello", False])
        assert result == [1, "hello", 0]
        assert type(result[0]) is int
        assert type(result[1]) is str
        assert type(result[2]) is int

    def test_array_union_float_str_with_int(self) -> None:
        """Array[Union[float, str]] converts int via float alternative."""
        result = S("Array", S("Union", S("float"), S("str"))).convert([42, "hi", True])
        assert result == [42.0, "hi", 1.0]
        assert type(result[0]) is float
        assert type(result[2]) is float

    def test_array_optional_float_all_none(self) -> None:
        """Array[Optional[float]] with all None elements."""
        result = S("Array", S("Optional", S("float"))).convert([None, None])
        assert result == [None, None]

    def test_array_optional_float_empty(self) -> None:
        """Array[Optional[float]] with empty list."""
        result = S("Array", S("Optional", S("float"))).convert([])
        assert result == []

    def test_array_union_failure_in_element(self) -> None:
        """Array[Union[int, str]] fails when element matches no alternative."""
        err = S("Array", S("Union", S("int"), S("str"))).try_check_value([1, 3.14])
        assert err is not None
        assert "element [1]" in err
        assert "got float" in err


# ---------------------------------------------------------------------------
# Category 22: Map/Dict nested with Optional/Union (inner conversion)
# ---------------------------------------------------------------------------
class TestNestedMapComposite:
    def test_map_str_optional_float_with_int(self) -> None:
        """Map[str, Optional[float]] converts int values to float."""
        result = S("Map", S("str"), S("Optional", S("float"))).convert({"a": 1, "b": None})
        assert result == {"a": 1.0, "b": None}
        assert type(result["a"]) is float
        assert result["b"] is None

    def test_map_str_union_int_str(self) -> None:
        """Map[str, Union[int, str]] converts bool values via int."""
        result = S("Map", S("str"), S("Union", S("int"), S("str"))).convert(
            {"x": True, "y": "hello"}
        )
        assert result == {"x": 1, "y": "hello"}
        assert type(result["x"]) is int

    def test_dict_str_optional_int(self) -> None:
        """Dict[str, Optional[int]] with bool conversion."""
        result = S("Dict", S("str"), S("Optional", S("int"))).convert(
            {"a": True, "b": None, "c": 42}
        )
        assert result == {"a": 1, "b": None, "c": 42}
        assert type(result["a"]) is int
        assert result["b"] is None

    def test_map_str_optional_float_failure(self) -> None:
        """Map[str, Optional[float]] fails for non-float non-None value."""
        err = S("Map", S("str"), S("Optional", S("float"))).try_check_value({"a": "bad"})
        assert err is not None
        assert "value for key 'a'" in err
        assert "expected float" in err


# ---------------------------------------------------------------------------
# Category 23: Nested containers (container inside container)
# ---------------------------------------------------------------------------
class TestNestedContainerInContainer:
    def test_array_of_array_int(self) -> None:
        """Array[Array[int]] with inner bool->int conversion."""
        result = S("Array", S("Array", S("int"))).convert([[True, False], [1, 2]])
        assert result == [[1, 0], [1, 2]]
        assert all(type(x) is int for row in result for x in row)

    def test_array_of_array_float(self) -> None:
        """Array[Array[float]] with inner int->float conversion."""
        result = S("Array", S("Array", S("float"))).convert([[1, 2], [True, 3]])
        assert result == [[1.0, 2.0], [1.0, 3.0]]
        assert all(type(x) is float for row in result for x in row)

    def test_map_str_array_float(self) -> None:
        """Map[str, Array[float]] with int->float conversion in arrays."""
        result = S("Map", S("str"), S("Array", S("float"))).convert({"a": [1, 2], "b": [True, 3]})
        assert result == {"a": [1.0, 2.0], "b": [1.0, 3.0]}
        assert all(type(x) is float for x in result["a"])
        assert all(type(x) is float for x in result["b"])

    def test_dict_str_array_int(self) -> None:
        """Dict[str, Array[int]] with bool->int conversion."""
        result = S("Dict", S("str"), S("Array", S("int"))).convert({"a": [True, False]})
        assert result == {"a": [1, 0]}
        assert all(type(x) is int for x in result["a"])

    def test_array_of_map_str_int(self) -> None:
        """Array[Map[str, int]] with bool->int value conversion."""
        result = S("Array", S("Map", S("str"), S("int"))).convert([{"x": True}, {"y": 2}])
        assert result == [{"x": 1}, {"y": 2}]
        assert type(result[0]["x"]) is int

    def test_map_str_map_str_float(self) -> None:
        """Map[str, Map[str, float]] double nested with int->float."""
        result = S("Map", S("str"), S("Map", S("str"), S("float"))).convert(
            {"outer": {"inner": 42}}
        )
        assert result == {"outer": {"inner": 42.0}}
        assert type(result["outer"]["inner"]) is float

    def test_list_of_list_int(self) -> None:
        """List[List[int]] with bool->int conversion."""
        result = S("List", S("List", S("int"))).convert([[True, 1], [False, 2]])
        assert result == [[1, 1], [0, 2]]
        assert all(type(x) is int for row in result for x in row)

    def test_nested_failure_array_of_array(self) -> None:
        """Array[Array[int]] error propagation through nested arrays."""
        err = S("Array", S("Array", S("int"))).try_check_value([[1, 2], [3, "bad"]])
        assert err is not None
        assert "element [1]" in err
        assert "expected int" in err

    def test_empty_inner_containers(self) -> None:
        """Map[str, Array[int]] with empty inner arrays."""
        result = S("Map", S("str"), S("Array", S("int"))).convert({"a": [], "b": []})
        assert result == {"a": [], "b": []}


# ---------------------------------------------------------------------------
# Category 24: Optional/Union wrapping containers
# ---------------------------------------------------------------------------
class TestOptionalUnionWrappingContainers:
    def test_optional_array_int_with_conversion(self) -> None:
        """Optional[Array[int]] converts inner bool elements."""
        schema = S("Optional", S("Array", S("int")))
        result = schema.convert([True, 2])
        assert result == [1, 2]
        assert type(result[0]) is int

    def test_optional_array_int_none(self) -> None:
        """Optional[Array[int]] accepts None."""
        result = S("Optional", S("Array", S("int"))).convert(None)
        assert result is None

    def test_optional_map_str_float(self) -> None:
        """Optional[Map[str, float]] converts inner int values."""
        result = S("Optional", S("Map", S("str"), S("float"))).convert({"a": 1})
        assert result == {"a": 1.0}
        assert type(result["a"]) is float

    def test_optional_map_str_float_none(self) -> None:
        """Optional[Map[str, float]] accepts None."""
        result = S("Optional", S("Map", S("str"), S("float"))).convert(None)
        assert result is None

    def test_union_array_int_or_map_str_int(self) -> None:
        """Union[Array[int], Map[str, int]] matches first with conversion."""
        schema = S("Union", S("Array", S("int")), S("Map", S("str"), S("int")))
        # list matches Array alternative
        result = schema.convert([True, 2])
        assert result == [1, 2]
        assert type(result[0]) is int

    def test_union_array_int_or_map_str_int_dict(self) -> None:
        """Union[Array[int], Map[str, int]] matches Map for dict input."""
        schema = S("Union", S("Array", S("int")), S("Map", S("str"), S("int")))
        result = schema.convert({"a": True})
        assert result == {"a": 1}
        assert type(result["a"]) is int

    def test_union_int_or_array_optional_float(self) -> None:
        """Union[int, Array[Optional[float]]] matches array with nested conversions."""
        schema = S("Union", S("int"), S("Array", S("Optional", S("float"))))
        result = schema.convert([True, None, 1])
        assert result == [1.0, None, 1.0]
        assert type(result[0]) is float
        assert result[1] is None

    def test_optional_optional_array_int(self) -> None:
        """Optional[Optional[Array[int]]] with inner conversion."""
        schema = S("Optional", S("Optional", S("Array", S("int"))))
        assert schema.convert(None) is None
        result = schema.convert([True, 2])
        assert result == [1, 2]
        assert type(result[0]) is int


# ---------------------------------------------------------------------------
# Category 25: Tuple nested with other types
# ---------------------------------------------------------------------------
class TestNestedTuple:
    def test_array_of_tuple_int_float(self) -> None:
        """Array[tuple[int, float]] with element-wise conversion."""
        result = S("Array", S("tuple", S("int"), S("float"))).convert([(True, 1), (2, True)])
        assert result == [(1, 1.0), (2, 1.0)]
        assert type(result[0][0]) is int
        assert type(result[0][1]) is float
        assert type(result[1][1]) is float

    def test_map_str_tuple_int_str(self) -> None:
        """Map[str, tuple[int, str]] with inner bool->int conversion."""
        result = S("Map", S("str"), S("tuple", S("int"), S("str"))).convert({"a": (True, "hello")})
        assert result == {"a": (1, "hello")}
        assert type(result["a"][0]) is int

    def test_tuple_of_array_int_and_map(self) -> None:
        """tuple[Array[int], Map[str, float]] nested conversion."""
        schema = S("tuple", S("Array", S("int")), S("Map", S("str"), S("float")))
        result = schema.convert(([True, 2], {"k": 3}))
        assert result == ([1, 2], {"k": 3.0})
        assert type(result[0][0]) is int
        assert type(result[1]["k"]) is float

    def test_tuple_of_optional_int_and_optional_float(self) -> None:
        """tuple[Optional[int], Optional[float]] with conversions."""
        schema = S("tuple", S("Optional", S("int")), S("Optional", S("float")))
        result = schema.convert((True, None))
        assert result == (1, None)
        assert type(result[0]) is int
        assert result[1] is None

    def test_tuple_nested_failure(self) -> None:
        """tuple[Array[int], str] error propagation from inner array."""
        err = S("tuple", S("Array", S("int")), S("str")).try_check_value(([1, "bad"], "ok"))
        assert err is not None
        assert "element [0]" in err
        assert "element [1]" in err
        assert "expected int" in err


# ---------------------------------------------------------------------------
# Category 26: Deep nesting (3+ levels)
# ---------------------------------------------------------------------------
class TestDeepNesting:
    def test_map_str_array_optional_int(self) -> None:
        """Map[str, Array[Optional[int]]] with 3-level nesting and conversion."""
        result = S("Map", S("str"), S("Array", S("Optional", S("int")))).convert(
            {"a": [1, None, True]}
        )
        assert result == {"a": [1, None, 1]}
        assert type(result["a"][0]) is int
        assert result["a"][1] is None
        assert type(result["a"][2]) is int

    def test_array_map_str_optional_float(self) -> None:
        """Array[Map[str, Optional[float]]] with 3-level nesting."""
        result = S("Array", S("Map", S("str"), S("Optional", S("float")))).convert(
            [{"x": 1, "y": None}, {"z": True}]
        )
        assert result == [{"x": 1.0, "y": None}, {"z": 1.0}]
        assert type(result[0]["x"]) is float
        assert result[0]["y"] is None
        assert type(result[1]["z"]) is float

    def test_optional_array_map_str_int(self) -> None:
        """Optional[Array[Map[str, int]]] 3 levels deep."""
        schema = S("Optional", S("Array", S("Map", S("str"), S("int"))))
        result = schema.convert([{"a": True}, {"b": 2}])
        assert result == [{"a": 1}, {"b": 2}]
        assert type(result[0]["a"]) is int

        assert schema.convert(None) is None

    def test_map_str_array_array_int(self) -> None:
        """Map[str, Array[Array[int]]] 3-level container nesting."""
        result = S("Map", S("str"), S("Array", S("Array", S("int")))).convert(
            {"m": [[True, 1], [False, 2]]}
        )
        assert result == {"m": [[1, 1], [0, 2]]}
        assert all(type(x) is int for row in result["m"] for x in row)

    def test_array_array_optional_float(self) -> None:
        """Array[Array[Optional[float]]] deep nesting with None and conversion."""
        result = S("Array", S("Array", S("Optional", S("float")))).convert(
            [[1, None], [True, 3.14]]
        )
        assert result == [[1.0, None], [1.0, 3.14]]
        assert type(result[0][0]) is float
        assert result[0][1] is None
        assert type(result[1][0]) is float

    def test_deep_nesting_failure_propagation(self) -> None:
        """Error from deepest level propagates with full path info."""
        err = S("Map", S("str"), S("Array", S("Optional", S("int")))).try_check_value(
            {"key": [1, "bad"]}
        )
        assert err is not None
        assert "value for key 'key'" in err
        assert "element [1]" in err
        assert "expected int" in err


# ---------------------------------------------------------------------------
# Category 27: FFI container inputs (tvm_ffi.Array/List/Map/Dict)
# ---------------------------------------------------------------------------
class TestFFIContainerInputs:
    def test_ffi_array_with_element_conversion(self) -> None:
        """tvm_ffi.Array([True, 2]) passes Array[int] with bool->int conversion."""
        arr = tvm_ffi.Array([True, 2, 3])
        result = S("Array", S("int")).convert(arr)
        assert result == [1, 2, 3]
        assert type(result[0]) is int

    def test_ffi_array_any_passthrough(self) -> None:
        """tvm_ffi.Array passes Array[Any] as-is."""
        arr = tvm_ffi.Array([1, "x", None])
        result = S("Array", S("Any")).convert(arr)
        assert result is arr

    def test_ffi_list_with_list_schema(self) -> None:
        """tvm_ffi.List passes List[int] with conversion."""
        lst = tvm_ffi.List([True, 2])
        result = S("List", S("int")).convert(lst)
        assert result == [1, 2]
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
        result = S("Map", S("str"), S("int")).convert(m)
        assert result == {"a": 1, "b": 2}
        assert type(result["a"]) is int

    def test_ffi_map_any_any_passthrough(self) -> None:
        """tvm_ffi.Map passes Map[Any, Any] as-is."""
        m = tvm_ffi.Map({"a": 1})
        result = S("Map", S("Any"), S("Any")).convert(m)
        assert result is m

    def test_ffi_dict_with_dict_schema(self) -> None:
        """tvm_ffi.Dict passes Dict[str, float] with int->float conversion."""
        d = tvm_ffi.Dict({"x": 1, "y": 2})
        result = S("Dict", S("str"), S("float")).convert(d)
        assert result == {"x": 1.0, "y": 2.0}
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
        result = S("Array", S("Optional", S("float"))).convert(arr)
        assert result == [1.0, None, 1.0]
        assert type(result[0]) is float
        assert result[1] is None

    def test_ffi_map_nested_array_int(self) -> None:
        """tvm_ffi.Map with value being a Python list, converted as Array[int]."""
        # Map values are already stored; create a map with array values
        m = tvm_ffi.Map({"k": tvm_ffi.Array([True, 2])})
        result = S("Map", S("str"), S("Array", S("int"))).convert(m)
        assert result == {"k": [1, 2]}
        assert type(result["k"][0]) is int

    def test_ffi_array_wrong_element_type(self) -> None:
        """tvm_ffi.Array with wrong element type gives clear error."""
        arr = tvm_ffi.Array([1, "bad", 3])
        err = S("Array", S("int")).try_check_value(arr)
        assert err is not None
        assert "element [1]" in err
        assert "expected int" in err

    def test_ffi_map_wrong_value_type(self) -> None:
        """tvm_ffi.Map with wrong value type gives clear error."""
        m = tvm_ffi.Map({"a": 1, "b": "bad"})
        err = S("Map", S("str"), S("int")).try_check_value(m)
        assert err is not None
        assert "value for key" in err
        assert "expected int" in err

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
        result = S("Array", S("Array", S("int"))).convert([inner1, inner2])
        assert result == [[1, 2], [3, 0]]

    def test_python_dict_with_ffi_array_values(self) -> None:
        """Python dict with tvm_ffi.Array values, Map[str, Array[float]]."""
        val = tvm_ffi.Array([1, True])
        result = S("Map", S("str"), S("Array", S("float"))).convert({"k": val})
        assert result == {"k": [1.0, 1.0]}
        assert all(type(x) is float for x in result["k"])

    def test_ffi_map_with_python_list_in_union(self) -> None:
        """Union[Map[str, int], Array[int]] with tvm_ffi.Map input."""
        schema = S("Union", S("Map", S("str"), S("int")), S("Array", S("int")))
        m = tvm_ffi.Map({"a": True})
        result = schema.convert(m)
        assert result == {"a": 1}
        assert type(result["a"]) is int

    def test_ffi_array_in_optional(self) -> None:
        """Optional[Array[int]] with tvm_ffi.Array input."""
        arr = tvm_ffi.Array([True, 2])
        result = S("Optional", S("Array", S("int"))).convert(arr)
        assert result == [1, 2]
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
        err = S("Map", S("str"), S("Array", S("Optional", S("int")))).try_check_value(
            {"key": [1, "bad"]}
        )
        assert err is not None
        assert "value for key 'key'" in err
        assert "element [1]" in err
        assert "expected int" in err

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
        err = S("testing.TestObjectBase").try_check_value(obj)
        assert err is not None
        assert "testing.TestIntPair" in err

    def test_base_fails_derived_schema(self) -> None:
        """Parent object fails child schema (TestObjectBase fails TestObjectDerived)."""
        obj = TestObjectBase(v_i64=10, v_f64=1.5, v_str="hi")
        err = S("testing.TestObjectDerived").try_check_value(obj)
        assert err is not None
        assert "testing.TestObjectBase" in err

    def test_non_object_fails_custom_schema(self) -> None:
        """Plain int fails custom object schema."""
        err = S("testing.TestIntPair").try_check_value(42)
        assert err is not None
        assert "expected testing.TestIntPair" in err
        assert "got int" in err

    def test_none_fails_custom_schema(self) -> None:
        """None fails custom object schema."""
        err = S("testing.TestIntPair").try_check_value(None)
        assert err is not None
        assert "got None" in err

    def test_string_fails_custom_schema(self) -> None:
        """String fails custom object schema."""
        err = S("testing.TestIntPair").try_check_value("hello")
        assert err is not None
        assert "got str" in err

    def test_cxx_base_fails_derived_schema(self) -> None:
        """_TestCxxClassBase fails _TestCxxClassDerived schema."""
        obj = _TestCxxClassBase(v_i64=1, v_i32=2)
        err = S("testing.TestCxxClassDerived").try_check_value(obj)
        assert err is not None

    def test_sibling_types_reject_each_other(self) -> None:
        """TestIntPair and TestCxxClassBase are unrelated -- reject each other."""
        pair = TestIntPair(1, 2)
        base = _TestCxxClassBase(v_i64=1, v_i32=2)
        assert S("testing.TestCxxClassBase").try_check_value(pair) is not None
        assert S("testing.TestIntPair").try_check_value(base) is not None


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
        err = S("Array", S("testing.TestIntPair")).try_check_value(objs)
        assert err is not None
        assert "element [1]" in err

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
        err = S("Map", S("str"), S("testing.TestIntPair")).try_check_value(data)
        assert err is not None
        assert "value for key 'b'" in err

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
        err = S("Optional", S("testing.TestIntPair")).try_check_value(obj)
        assert err is not None

    def test_union_custom_object_and_int(self) -> None:
        """Union[testing.TestIntPair, int] with object."""
        obj = TestIntPair(1, 2)
        S("Union", S("testing.TestIntPair"), S("int")).check_value(obj)

    def test_union_custom_object_and_int_with_int(self) -> None:
        """Union[testing.TestIntPair, int] with int."""
        S("Union", S("testing.TestIntPair"), S("int")).check_value(42)

    def test_union_custom_object_and_int_with_wrong(self) -> None:
        """Union[testing.TestIntPair, int] with str fails."""
        err = S("Union", S("testing.TestIntPair"), S("int")).try_check_value("bad")
        assert err is not None

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
        err = S("Union", S("testing.TestIntPair"), S("testing.TestCxxClassBase")).try_check_value(
            obj
        )
        assert err is not None


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
        err = schema.try_check_value(_TestCxxClassBase(v_i64=1, v_i32=2))
        assert err is not None

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
        err = S("Array", S("testing.TestIntPair")).try_check_value(data)
        assert err is not None
        assert "element [1]" in err
        assert "testing.TestIntPair" in err
        assert "testing.TestCxxClassBase" in err

    def test_map_nested_error_with_custom_object(self) -> None:
        """Map value error for custom object includes key and type info."""
        data = {"ok": TestIntPair(1, 2), "bad": 42}
        err = S("Map", S("str"), S("testing.TestIntPair")).try_check_value(data)
        assert err is not None
        assert "value for key 'bad'" in err
        assert "expected testing.TestIntPair" in err
        assert "got int" in err

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
        err = S("Map", S("str"), S("Array", S("testing.TestIntPair"))).try_check_value(data)
        assert err is not None
        assert "value for key 'k'" in err
        assert "element [1]" in err

    def test_tuple_with_custom_object(self) -> None:
        """tuple[testing.TestIntPair, int, str] with custom object."""
        data = (TestIntPair(1, 2), 42, "hello")
        S("tuple", S("testing.TestIntPair"), S("int"), S("str")).check_value(data)

    def test_tuple_with_custom_object_wrong(self) -> None:
        """tuple[testing.TestIntPair, int] with wrong object in first position."""
        data = (_TestCxxClassBase(v_i64=1, v_i32=2), 42)
        err = S("tuple", S("testing.TestIntPair"), S("int")).try_check_value(data)
        assert err is not None
        assert "element [0]" in err


# ---------------------------------------------------------------------------
# Category 37: Lowercase Python-native origins ("list", "dict")
# ---------------------------------------------------------------------------
class TestLowercaseOrigins:
    def test_list_origin_accepts_python_list(self) -> None:
        """TypeSchema("list", ...) should validate elements, not passthrough."""
        S("list", S("int")).check_value([1, 2, 3])

    def test_list_origin_rejects_bad_elements(self) -> None:
        """TypeSchema("list", (int,)).check_value(["x"]) should fail."""
        err = S("list", S("int")).try_check_value(["x"])
        assert err is not None
        assert "element [0]" in err

    def test_list_origin_converts_elements(self) -> None:
        """TypeSchema("list", (float,)).convert([1, True]) does int->float."""
        result = S("list", S("float")).convert([1, True])
        assert result == [1.0, 1.0]
        assert all(type(x) is float for x in result)

    def test_dict_origin_accepts_python_dict(self) -> None:
        """TypeSchema("dict", ...) should validate key/value types."""
        S("dict", S("str"), S("int")).check_value({"a": 1})

    def test_dict_origin_rejects_bad_values(self) -> None:
        """TypeSchema("dict", (str, int)).check_value({"a": "x"}) should fail."""
        err = S("dict", S("str"), S("int")).try_check_value({"a": "x"})
        assert err is not None
        assert "value for key 'a'" in err

    def test_dict_origin_converts_values(self) -> None:
        """TypeSchema("dict", (str, float)).convert({"a": 1}) does int->float."""
        result = S("dict", S("str"), S("float")).convert({"a": 1, "b": True})
        assert result == {"a": 1.0, "b": 1.0}
        assert all(type(v) is float for v in result.values())

    def test_list_origin_no_args_accepts_anything(self) -> None:
        """TypeSchema("list") with no args accepts any list (element type is Any)."""
        S("list").check_value([1, "a", None])

    def test_dict_origin_no_args_accepts_anything(self) -> None:
        """TypeSchema("dict") with no args accepts any dict."""
        S("dict").check_value({"a": 1, 2: "b"})

    def test_list_origin_rejects_non_list(self) -> None:
        """TypeSchema("list") rejects non-sequence types."""
        err = S("list").try_check_value(42)
        assert err is not None
        assert "got int" in err

    def test_dict_origin_rejects_non_dict(self) -> None:
        """TypeSchema("dict") rejects non-dict types."""
        err = S("dict").try_check_value([1, 2])
        assert err is not None


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
        result = S("Array", S("float")).convert(lst)
        assert result == [1.0, 2.0, 1.0]
        assert all(type(x) is float for x in result)

    def test_list_schema_converts_array_elements(self) -> None:
        """List[float] converts elements from tvm_ffi.Array[int]."""
        arr = tvm_ffi.Array([1, 2, True])
        result = S("List", S("float")).convert(arr)
        assert result == [1.0, 2.0, 1.0]
        assert all(type(x) is float for x in result)

    def test_map_schema_converts_dict_values(self) -> None:
        """Map[str, float] converts values from tvm_ffi.Dict."""
        d = tvm_ffi.Dict({"a": 1, "b": True})
        result = S("Map", S("str"), S("float")).convert(d)
        assert result == {"a": 1.0, "b": 1.0}

    def test_dict_schema_converts_map_values(self) -> None:
        """Dict[str, float] converts values from tvm_ffi.Map."""
        m = tvm_ffi.Map({"a": 1, "b": True})
        result = S("Dict", S("str"), S("float")).convert(m)
        assert result == {"a": 1.0, "b": 1.0}

    def test_cross_type_still_rejects_wrong_container(self) -> None:
        """Array schema still rejects non-sequence CObjects (e.g. Map)."""
        m = tvm_ffi.Map({"a": 1})
        err = S("Array", S("int")).try_check_value(m)
        assert err is not None
        assert "expected Array" in err

    def test_cross_type_map_rejects_array(self) -> None:
        """Map schema still rejects sequence CObjects (e.g. Array)."""
        arr = tvm_ffi.Array([1, 2])
        err = S("Map", S("str"), S("int")).try_check_value(arr)
        assert err is not None
        assert "expected Map" in err


# ---------------------------------------------------------------------------
# Category 39: tuple accepts list and CObject Array
# ---------------------------------------------------------------------------
class TestTupleAcceptsListAndArray:
    def test_tuple_accepts_python_list(self) -> None:
        """tuple[int, str] accepts Python list input."""
        result = S("tuple", S("int"), S("str")).convert([42, "hello"])
        assert result == (42, "hello")
        assert isinstance(result, tuple)

    def test_tuple_list_with_conversion(self) -> None:
        """tuple[float, int] converts list elements (bool->float, bool->int)."""
        result = S("tuple", S("float"), S("int")).convert([True, False])
        assert result == (1.0, 0)
        assert type(result[0]) is float
        assert type(result[1]) is int

    def test_tuple_rejects_wrong_length_list(self) -> None:
        """tuple[int, str] rejects list of wrong length."""
        err = S("tuple", S("int"), S("str")).try_check_value([1, "a", "b"])
        assert err is not None
        assert "length" in err

    def test_tuple_accepts_ffi_array(self) -> None:
        """tuple[int, int] accepts tvm_ffi.Array (C++ Tuple accepts kTVMFFIArray)."""
        arr = tvm_ffi.Array([1, 2])
        S("tuple", S("int"), S("int")).check_value(arr)

    def test_tuple_ffi_array_with_conversion(self) -> None:
        """tuple[float, float] converts tvm_ffi.Array elements."""
        arr = tvm_ffi.Array([1, True])
        result = S("tuple", S("float"), S("float")).convert(arr)
        assert result == (1.0, 1.0)
        assert all(type(x) is float for x in result)

    def test_tuple_ffi_array_wrong_length(self) -> None:
        """tuple[int, int] rejects tvm_ffi.Array of wrong length."""
        arr = tvm_ffi.Array([1, 2, 3])
        err = S("tuple", S("int"), S("int")).try_check_value(arr)
        assert err is not None
        assert "length" in err

    def test_tuple_rejects_ffi_map(self) -> None:
        """Tuple schema rejects Map CObject."""
        m = tvm_ffi.Map({"a": 1})
        err = S("tuple", S("int")).try_check_value(m)
        assert err is not None
        assert "expected tuple" in err

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
# Category 40: dtype string parse errors in try_convert/try_check_value
# ---------------------------------------------------------------------------
class TestDtypeParseErrors:
    def test_try_check_value_bad_dtype_returns_error(self) -> None:
        """try_check_value should return error message, not raise, for invalid dtype."""
        err = S("dtype").try_check_value("not_a_valid_dtype_xyz")
        assert err is not None
        assert "dtype" in err

    def test_try_convert_bad_dtype_returns_false(self) -> None:
        """try_convert should return (False, msg), not raise, for invalid dtype."""
        ok, msg = S("dtype").try_convert("not_a_valid_dtype_xyz")
        assert ok is False
        assert "dtype" in msg

    def test_convert_bad_dtype_raises_type_error(self) -> None:
        """Convert should raise TypeError for invalid dtype string."""
        with pytest.raises(TypeError, match="dtype"):
            S("dtype").convert("not_a_valid_dtype_xyz")

    def test_valid_dtype_string_still_works(self) -> None:
        """Valid dtype strings should still convert successfully."""
        result = S("dtype").convert("float32")
        assert str(result) == "float32"

    def test_try_convert_valid_dtype(self) -> None:
        """try_convert with valid dtype returns (True, DataType)."""
        ok, result = S("dtype").try_convert("int8")
        assert ok is True
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
        err = S("int").try_check_value(2**63)
        assert err is not None
        assert "int64 range" in err

    def test_int64_min_minus_one_rejected(self) -> None:
        """-2^63-1 exceeds int64 range."""
        err = S("int").try_check_value(-(2**63) - 1)
        assert err is not None
        assert "int64 range" in err

    def test_very_large_positive_rejected(self) -> None:
        """Very large positive integer rejected."""
        err = S("int").try_check_value(10**100)
        assert err is not None
        assert "int64 range" in err

    def test_very_large_negative_rejected(self) -> None:
        """Very large negative integer rejected."""
        err = S("int").try_check_value(-(10**100))
        assert err is not None
        assert "int64 range" in err

    def test_convert_returns_error_not_raises_for_overflow(self) -> None:
        """try_convert returns (False, msg) for overflow, not an exception."""
        ok, msg = S("int").try_convert(2**63)
        assert ok is False
        assert "int64 range" in msg

    def test_convert_raises_type_error_for_overflow(self) -> None:
        """Convert raises TypeError for overflow."""
        with pytest.raises(TypeError, match="int64 range"):
            S("int").convert(2**63)

    def test_bool_to_int_no_range_issue(self) -> None:
        """Bool -> int conversion (0 or 1) always fits."""
        assert S("int").convert(True) == 1
        assert S("int").convert(False) == 0

    def test_int64_boundaries_in_float_conversion(self) -> None:
        """Float schema accepts large ints (float64 has wider range)."""
        # float64 can represent integers up to 2^53 exactly,
        # and larger values with precision loss (but no range error)
        S("float").check_value(2**63)
        S("float").check_value(-(2**63))

    def test_int64_overflow_in_optional_int(self) -> None:
        """Optional[int] propagates int64 range check."""
        err = S("Optional", S("int")).try_check_value(2**63)
        assert err is not None
        assert "int64 range" in err

    def test_int64_overflow_in_array_element(self) -> None:
        """Array[int] element overflow is caught with path."""
        err = S("Array", S("int")).try_check_value([1, 2**63, 3])
        assert err is not None
        assert "element [1]" in err
        assert "int64 range" in err


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

    def test_unknown_origin_returns_error_on_try_convert(self) -> None:
        """Unknown origin returns (False, msg) on try_convert (never raises)."""
        schema = S("not_a_real_type")
        ok, msg = schema.try_convert(42)
        assert ok is False
        assert "unknown TypeSchema origin" in msg

    def test_unknown_origin_returns_error_on_try_check_value(self) -> None:
        """Unknown origin returns error msg on try_check_value (never raises)."""
        schema = S("not_a_real_type")
        err = schema.try_check_value(42)
        assert err is not None
        assert "unknown TypeSchema origin" in err

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
# Category 43: try_* methods never raise (robustness)
# ---------------------------------------------------------------------------
class TestTryMethodsNeverRaise:
    """Verify try_convert and try_check_value catch all exceptions."""

    def test_try_convert_catches_custom_integral_error(self) -> None:
        """Custom Integral whose __int__ raises is caught by try_convert."""

        class BadInt:
            """Registered as Integral via ABC but __int__ raises."""

            def __int__(self) -> int:
                raise RuntimeError("broken __int__")

        Integral.register(BadInt)
        ok, msg = S("int").try_convert(BadInt())
        assert ok is False
        assert "broken __int__" in msg

    def test_try_check_value_catches_custom_integral_error(self) -> None:
        """Custom Integral whose __int__ raises is caught by try_check_value."""

        class BadInt2:
            def __int__(self) -> int:
                raise ValueError("bad int conversion")

        Integral.register(BadInt2)
        err = S("int").try_check_value(BadInt2())
        assert err is not None
        assert "bad int conversion" in err

    def test_try_convert_unknown_origin_no_raise(self) -> None:
        """try_convert with unknown origin returns error, never raises."""
        ok, msg = S("bogus_type").try_convert("anything")
        assert ok is False
        assert "unknown TypeSchema origin" in msg

    def test_try_check_value_unknown_origin_no_raise(self) -> None:
        """try_check_value with unknown origin returns error, never raises."""
        err = S("bogus_type").try_check_value("anything")
        assert err is not None
        assert "unknown TypeSchema origin" in err


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

    def test_int_protocol_try_check(self) -> None:
        """try_check_value returns None for __tvm_ffi_int__ value."""

        class IntProto:
            def __tvm_ffi_int__(self) -> int:
                return 10

        assert S("int").try_check_value(IntProto()) is None

    def test_int_protocol_convert_returns_value(self) -> None:
        """Convert returns the protocol value as-is (marshal handles conversion)."""

        class IntProto:
            def __tvm_ffi_int__(self) -> int:
                return 99

        obj = IntProto()
        assert S("int").convert(obj) is obj

    def test_without_protocol_still_rejected(self) -> None:
        """Object without __tvm_ffi_int__ is still rejected by int schema."""

        class NoProto:
            pass

        err = S("int").try_check_value(NoProto())
        assert err is not None
        assert "expected int" in err


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
        assert S("float").convert(obj) is obj

    def test_without_protocol_still_rejected(self) -> None:
        """Object without __tvm_ffi_float__ is still rejected."""

        class NoProto:
            pass

        err = S("float").try_check_value(NoProto())
        assert err is not None
        assert "expected float" in err


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
        assert S("ctypes.c_void_p").convert(obj) is obj


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
        assert S("Device").convert(obj) is obj

    def test_without_protocol_still_rejected(self) -> None:
        """Object without __dlpack_device__ is still rejected."""

        class NoProto:
            pass

        err = S("Device").try_check_value(NoProto())
        assert err is not None
        assert "expected Device" in err


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
        assert S("dtype").convert(obj) is obj

    def test_numpy_dtype_accepted(self) -> None:
        """numpy.dtype passes dtype schema (if numpy installed)."""
        numpy = pytest.importorskip("numpy")
        S("dtype").check_value(numpy.dtype("float32"))

    def test_numpy_dtype_convert(self) -> None:
        """Convert returns numpy.dtype as-is."""
        numpy = pytest.importorskip("numpy")
        dt = numpy.dtype("int32")
        assert S("dtype").convert(dt) is dt

    def test_torch_dtype_accepted(self) -> None:
        """torch.dtype passes dtype schema (if torch installed)."""
        torch = pytest.importorskip("torch")
        S("dtype").check_value(torch.float32)

    def test_torch_dtype_convert(self) -> None:
        """Convert returns torch.dtype as-is."""
        torch = pytest.importorskip("torch")
        dt = torch.int64
        assert S("dtype").convert(dt) is dt


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
        assert S("Tensor").convert(obj) is obj

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

        result = S("testing.TestIntPair").convert(ObjProto())
        assert result is inner

    def test_object_protocol_wrong_type_rejected(self) -> None:
        """__tvm_ffi_object__ returning wrong type is rejected."""
        inner = TestIntPair(1, 2)

        class ObjProto:
            def __tvm_ffi_object__(self) -> object:
                return inner

        err = S("testing.TestCxxClassBase").try_check_value(ObjProto())
        assert err is not None
        assert "__tvm_ffi_object__" in err

    def test_object_protocol_raises_caught(self) -> None:
        """__tvm_ffi_object__ that raises produces _ConvertError."""

        class BadProto:
            def __tvm_ffi_object__(self) -> object:
                raise RuntimeError("broken")

        err = S("Object").try_check_value(BadProto())
        assert err is not None
        assert "__tvm_ffi_object__() failed" in err

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

        result = S("testing.TestIntPair").convert(MyConvertible())
        assert result is inner

    def test_object_convertible_wrong_type(self) -> None:
        """ObjectConvertible returning wrong type is rejected."""
        inner = TestIntPair(1, 2)

        class MyConvertible(ObjectConvertible):
            def asobject(self) -> tvm_ffi.core.Object:
                return inner

        err = S("testing.TestCxxClassBase").try_check_value(MyConvertible())
        assert err is not None
        assert "asobject()" in err

    def test_object_convertible_raises_caught(self) -> None:
        """asobject() that raises produces error, not exception."""

        class BadConvertible(ObjectConvertible):
            def asobject(self) -> tvm_ffi.core.Object:
                raise RuntimeError("broken asobject")

        err = S("Object").try_check_value(BadConvertible())
        assert err is not None
        assert "asobject() failed" in err


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

        result = S("int").convert(ValProto())
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
        assert S("int").convert(wrapped) == 10

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

        err = S("int").try_check_value(ValProto())
        assert err is not None
        assert "expected int" in err

    def test_value_protocol_raises_uses_original_error(self) -> None:
        """If __tvm_ffi_value__ raises, the original error is returned."""

        class BadValProto:
            def __tvm_ffi_value__(self) -> object:
                raise RuntimeError("broken")

        err = S("int").try_check_value(BadValProto())
        assert err is not None
        assert "expected int" in err


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

        result = S("Array", S("int")).convert([VP()])
        assert result == [42]


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

        err = S("int").try_check_value(SelfCycle())
        assert err is not None
        assert "expected int" in err

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
        err = S("int").try_check_value(a)
        assert err is not None
        assert "cycle" in err


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
        err = S("ffi.Array").try_check_value(RuntimeError("x"))
        assert err is not None
        assert "ffi.Error" in err

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

        err = S("testing.TestIntPair").try_check_value(Custom())
        assert err is not None
        assert "OpaquePyObject" in err or "expected" in err

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
        err = S("Object").try_check_value(42)
        assert err is not None

    def test_float_rejected_by_object_schema(self) -> None:
        """TypeSchema('Object') rejects float (float is a POD, not Object)."""
        err = S("Object").try_check_value(3.14)
        assert err is not None

    def test_none_rejected_by_object_schema(self) -> None:
        """TypeSchema('Object') rejects None (None is a POD, not Object)."""
        err = S("Object").try_check_value(None)
        assert err is not None


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
        assert S("ctypes.c_void_p").convert(obj) is obj

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

        err = S("Device").try_check_value(TensorLike())
        assert err is not None

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

        err = S("Device").try_check_value(DLPackOnly())
        assert err is not None


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

            err = S("Tensor").try_check_value(ExchangeAPI())
            assert err is not None
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
            # Should not raise on try_convert (triggers converter build)
            s.try_convert(None)


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
        """Array[int] on exact Python list returns original."""
        original = [1, 2, 3]
        result = S("Array", S("int")).convert(original)
        assert result is original

    def test_array_int_needs_conversion(self) -> None:
        """Array[int] on list needing bool->int returns new list."""
        original = [1, True, 3]
        result = S("Array", S("int")).convert(original)
        assert result is not original
        assert result == [1, 1, 3]

    def test_map_str_int_exact_dict(self) -> None:
        """Map[str, int] on exact dict returns original."""
        original = {"a": 1, "b": 2}
        result = S("Map", S("str"), S("int")).convert(original)
        assert result is original

    def test_map_str_int_needs_conversion(self) -> None:
        """Map[str, int] on dict needing conversion returns new dict."""
        original = {"a": True, "b": 2}
        result = S("Map", S("str"), S("int")).convert(original)
        assert result is not original

    def test_tuple_exact_match(self) -> None:
        """tuple[int, str] on exact tuple returns original."""
        original = (42, "hello")
        result = S("tuple", S("int"), S("str")).convert(original)
        assert result is original

    def test_tuple_needs_conversion(self) -> None:
        """tuple[int, str] on tuple needing conversion returns new tuple."""
        original = (True, "hello")
        result = S("tuple", S("int"), S("str")).convert(original)
        assert result is not original
        assert result == (1, "hello")

    def test_list_int_exact(self) -> None:
        """List[int] on exact list returns original."""
        original = [10, 20]
        result = S("List", S("int")).convert(original)
        assert result is original


# ---------------------------------------------------------------------------
# Category 65: Exception normalization in check_value/convert
# ---------------------------------------------------------------------------
class TestExceptionNormalization:
    """check_value/convert normalize custom __int__/__float__ failures."""

    def test_broken_integral_try_convert(self) -> None:
        """Integral with broken __int__ caught by try_convert."""

        class BadIntegral:
            def __int__(self) -> int:
                raise OverflowError("too big")

        Integral.register(BadIntegral)

        ok, msg = S("int").try_convert(BadIntegral())
        assert not ok
        assert "too big" in msg

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
        err = S("Object").try_check_value(Dual())
        assert err is not None

    def test_float_protocol_takes_precedence(self) -> None:
        """Class with __tvm_ffi_float__ + __tvm_ffi_value__ dispatches as float."""

        class Dual:
            def __tvm_ffi_float__(self) -> float:
                return 1.0

            def __tvm_ffi_value__(self) -> object:
                return TestIntPair(1, 2)

        S("float").check_value(Dual())
        err = S("Object").try_check_value(Dual())
        assert err is not None

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
        err = S("int").try_check_value(CallableVP())
        assert err is not None


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

        err = S("Union", S("int"), S("str")).try_check_value(WrongVP())
        assert err is not None


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
        err = S("Object").try_check_value(DualProtocol())
        assert err is not None

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
