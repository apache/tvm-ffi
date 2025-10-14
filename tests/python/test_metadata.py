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
from typing import Any

import pytest
from tvm_ffi import get_global_func_metadata
from tvm_ffi.core import TypeInfo, TypeSchema
from tvm_ffi.testing import _SchemaAllTypes


def _replace_list_dict(ty: str) -> str:
    return {
        "list": "Sequence",
        "dict": "Mapping",
    }.get(ty, ty)


@pytest.mark.parametrize(
    "func_name,expected",
    [
        ("testing.schema_id_int", "Callable[[int], int]"),
        ("testing.schema_id_float", "Callable[[float], float]"),
        ("testing.schema_id_bool", "Callable[[bool], bool]"),
        ("testing.schema_id_device", "Callable[[Device], Device]"),
        ("testing.schema_id_dtype", "Callable[[dtype], dtype]"),
        ("testing.schema_id_string", "Callable[[str], str]"),
        ("testing.schema_id_bytes", "Callable[[bytes], bytes]"),
        ("testing.schema_id_func", "Callable[[Callable[..., Any]], Callable[..., Any]]"),
        (
            "testing.schema_id_func_typed",
            "Callable[[Callable[[int, float, Callable[..., Any]], None]], Callable[[int, float, Callable[..., Any]], None]]",
        ),
        ("testing.schema_id_any", "Callable[[Any], Any]"),
        ("testing.schema_id_object", "Callable[[Object], Object]"),
        ("testing.schema_id_dltensor", "Callable[[Tensor], Tensor]"),
        ("testing.schema_id_tensor", "Callable[[Tensor], Tensor]"),
        ("testing.schema_tensor_view_input", "Callable[[Tensor], None]"),
        ("testing.schema_id_opt_int", "Callable[[int | None], int | None]"),
        ("testing.schema_id_opt_str", "Callable[[str | None], str | None]"),
        ("testing.schema_id_opt_obj", "Callable[[Object | None], Object | None]"),
        ("testing.schema_id_arr_int", "Callable[[list[int]], list[int]]"),
        ("testing.schema_id_arr_str", "Callable[[list[str]], list[str]]"),
        ("testing.schema_id_arr_obj", "Callable[[list[Object]], list[Object]]"),
        ("testing.schema_id_arr", "Callable[[list[Any]], list[Any]]"),
        ("testing.schema_id_map_str_int", "Callable[[dict[str, int]], dict[str, int]]"),
        ("testing.schema_id_map_str_str", "Callable[[dict[str, str]], dict[str, str]]"),
        ("testing.schema_id_map_str_obj", "Callable[[dict[str, Object]], dict[str, Object]]"),
        ("testing.schema_id_map", "Callable[[dict[Any, Any]], dict[Any, Any]]"),
        ("testing.schema_id_variant_int_str", "Callable[[int | str], int | str]"),
        ("testing.schema_packed", "Callable[..., Any]"),
        (
            "testing.schema_arr_map_opt",
            "Callable[[list[int | None], dict[str, list[int]], str | None], dict[str, list[int]]]",
        ),
        ("testing.schema_variant_mix", "Callable[[int | str | list[int]], int | str | list[int]]"),
        ("testing.schema_no_args", "Callable[[], int]"),
        ("testing.schema_no_return", "Callable[[int], None]"),
        ("testing.schema_no_args_no_return", "Callable[[], None]"),
    ],
)
def test_schema_global_func(func_name: str, expected: str) -> None:
    metadata: dict[str, Any] = get_global_func_metadata(func_name)
    actual: TypeSchema = TypeSchema.from_json_str(metadata["type_schema"])
    assert str(actual) == expected, f"{func_name}: {actual}"
    assert actual.repr(_replace_list_dict) == expected.replace(
        "list",
        "Sequence",
    ).replace(
        "dict",
        "Mapping",
    )


@pytest.mark.parametrize(
    "field_name,expected",
    [
        ("v_bool", "bool"),
        ("v_int", "int"),
        ("v_float", "float"),
        ("v_device", "Device"),
        ("v_dtype", "dtype"),
        ("v_string", "str"),
        ("v_bytes", "bytes"),
        ("v_opt_int", "int | None"),
        ("v_opt_str", "str | None"),
        ("v_arr_int", "list[int]"),
        ("v_arr_str", "list[str]"),
        ("v_map_str_int", "dict[str, int]"),
        ("v_map_str_arr_int", "dict[str, list[int]]"),
        ("v_variant", "str | list[int] | dict[str, int]"),
        ("v_opt_arr_variant", "list[int | str] | None"),
    ],
)
def test_schema_field(field_name: str, expected: str) -> None:
    type_info: TypeInfo = getattr(_SchemaAllTypes, "__tvm_ffi_type_info__")
    for field in type_info.fields:
        if field.name == field_name:
            actual: TypeSchema = TypeSchema.from_json_str(field.metadata["type_schema"])
            assert str(actual) == expected, f"{field_name}: {actual}"
            assert actual.repr(_replace_list_dict) == expected.replace(
                "list",
                "Sequence",
            ).replace(
                "dict",
                "Mapping",
            )
            break
    else:
        raise ValueError(f"Field not found: {field_name}")


@pytest.mark.parametrize(
    "method_name,expected",
    [
        ("add_int", "Callable[[testing.SchemaAllTypes, int], int]"),
        ("append_int", "Callable[[testing.SchemaAllTypes, list[int], int], list[int]]"),
        ("maybe_concat", "Callable[[testing.SchemaAllTypes, str | None, str | None], str | None]"),
        (
            "merge_map",
            "Callable[[testing.SchemaAllTypes, dict[str, list[int]], dict[str, list[int]]], dict[str, list[int]]]",
        ),
        ("make_with", "Callable[[int, float, str], testing.SchemaAllTypes]"),
    ],
)
def test_schema_member_method(method_name: str, expected: str) -> None:
    type_info: TypeInfo = getattr(_SchemaAllTypes, "__tvm_ffi_type_info__")
    for method in type_info.methods:
        if method.name == method_name:
            actual: TypeSchema = TypeSchema.from_json_str(method.metadata["type_schema"])
            assert str(actual) == expected, f"{method_name}: {actual}"
            assert actual.repr(_replace_list_dict) == expected.replace(
                "list",
                "Sequence",
            ).replace(
                "dict",
                "Mapping",
            )
            break
    else:
        raise ValueError(f"Method not found: {method_name}")


def test_metadata_global_func() -> None:
    metadata: dict[str, Any] = get_global_func_metadata("testing.schema_id_int")
    assert len(metadata) == 4
    assert "type_schema" in metadata
    assert metadata["bool_attr"] is True
    assert metadata["int_attr"] == 1
    assert metadata["str_attr"] == "hello"


def test_metadata_field() -> None:
    type_info: TypeInfo = getattr(_SchemaAllTypes, "__tvm_ffi_type_info__")
    for field in type_info.fields:
        if field.name == "v_bool":
            assert len(field.metadata) == 4
            assert "type_schema" in field.metadata
            assert field.metadata["bool_attr"] is True
            assert field.metadata["int_attr"] == 1
            assert field.metadata["str_attr"] == "hello"
            break
    else:
        raise ValueError("Field not found: v_bool")


def test_metadata_member_method() -> None:
    type_info: TypeInfo = getattr(_SchemaAllTypes, "__tvm_ffi_type_info__")
    for method in type_info.methods:
        if method.name == "add_int":
            assert len(method.metadata) == 4
            assert "type_schema" in method.metadata
            assert method.metadata["bool_attr"] is True
            assert method.metadata["int_attr"] == 1
            assert method.metadata["str_attr"] == "hello"
            break
    else:
        raise ValueError("Method not found: add_int")


def test_mem_fn_as_global_func() -> None:
    metadata: dict[str, Any] = get_global_func_metadata("testing.TestIntPairSum")
    type_schema: TypeSchema = TypeSchema.from_json_str(metadata["type_schema"])
    assert str(type_schema) == "Callable[[testing.TestIntPair], int]"
