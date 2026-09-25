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

import sys

import pytest
import tvm_ffi
import tvm_ffi.cpp
from tvm_ffi.core import TypeSchema

# Distinct words and an interior zero exercise word order and sign padding.
_MULTIWORD_VALUE = (1 << 255) | (0x0123456789ABCDEF << 128) | 0xFEDCBA9876543210


@pytest.fixture(scope="module")
def bigint_module() -> tvm_ffi.Module:
    return tvm_ffi.cpp.load_inline(
        name="test_big_int_transport",
        cpp_sources=r"""
            #include <tvm/ffi/big_int.h>
            #include <tvm/ffi/rvalue_ref.h>

            tvm::ffi::BigInt identity(tvm::ffi::BigInt value) { return value; }
            tvm::ffi::BigInt add(tvm::ffi::BigInt a, tvm::ffi::BigInt b) { return a + b; }
            bool is_int64(tvm::ffi::Any value) {
              return value.type_index() == kTVMFFIInt;
            }
            int64_t int64_only(int64_t value) { return value; }
            tvm::ffi::BigInt invoke(tvm::ffi::Function callback, tvm::ffi::BigInt value) {
              return callback(value).cast<tvm::ffi::BigInt>();
            }
            tvm::ffi::BigInt invoke_rvalue(tvm::ffi::Function callback, tvm::ffi::BigInt value) {
              auto object = tvm::ffi::Any(value).cast<tvm::ffi::ObjectRef>();
              return callback(tvm::ffi::RValueRef(std::move(object))).cast<tvm::ffi::BigInt>();
            }
            int64_t rvalue_ref_count_delta(tvm::ffi::Function callback, tvm::ffi::BigInt value) {
              auto owner = tvm::ffi::Any(value).cast<tvm::ffi::ObjectRef>();
              int64_t before = owner.use_count();
              try {
                auto moved = owner;
                callback(tvm::ffi::RValueRef(std::move(moved)));
              } catch (const tvm::ffi::Error&) {
              }
              return owner.use_count() - before;
            }
            tvm::ffi::Bytes content(tvm::ffi::BigInt value) {
              TVMFFIAny view = tvm::ffi::AnyView(value).CopyToTVMFFIAny();
              TVMFFIByteArray bytes = TVMFFIBigIntGetContentByteArray(&view);
              return tvm::ffi::Bytes(bytes.data, bytes.size);
            }
            tvm::ffi::Any from_content(tvm::ffi::Bytes bytes) {
              TVMFFIByteArray input{bytes.data(), bytes.size()};
              TVMFFIAny result;
              if (TVMFFIBigIntFromByteArray(&input, &result) != 0) {
                throw tvm::ffi::details::MoveFromSafeCallRaised();
              }
              return tvm::ffi::details::AnyUnsafe::MoveTVMFFIAnyRawToAny(result);
            }
        """,
        functions=[
            "identity",
            "add",
            "is_int64",
            "int64_only",
            "invoke",
            "invoke_rvalue",
            "rvalue_ref_count_delta",
            "content",
            "from_content",
        ],
        extra_cflags=["-DTVM_FFI_DLL_EXPORT_INCLUDE_METADATA=1"],
    )


def test_integer_round_trip(bigint_module: tvm_ffi.Module) -> None:
    echo = tvm_ffi.get_global_func("testing.echo")
    values = [0, -1, 0x13579BDF2468ACE0, -0x13579BDF2468ACE0, _MULTIWORD_VALUE, -_MULTIWORD_VALUE]
    for value in values:
        assert echo(value) == value
        result = bigint_module.identity(value)
        assert type(result) is int
        assert result == value
        assert bigint_module.is_int64(value) == (-(1 << 63) <= value < (1 << 63))


def test_integer_int64_boundaries(bigint_module: tvm_ffi.Module) -> None:
    echo = tvm_ffi.get_global_func("testing.echo")
    for value in [-(1 << 63) - 1, -(1 << 63), (1 << 63) - 1, 1 << 63]:
        assert echo(value) == value
        assert bigint_module.identity(value) == value
        fits = -(1 << 63) <= value < (1 << 63)
        assert bigint_module.is_int64(value) == fits
        if fits:
            assert bigint_module.int64_only(value) == value


def test_integer_content_format(bigint_module: tvm_ffi.Module) -> None:
    assert bigint_module.from_content(b"") == 0
    with pytest.raises(ValueError, match="whole 64-bit words"):
        bigint_module.from_content(b"\x00" * 7)
    values = [0, -1, (1 << 63) - 1, -(1 << 63), 1 << 63, -(1 << 63) - 1]
    values.extend([_MULTIWORD_VALUE, -_MULTIWORD_VALUE])
    for value in values:
        signed_bits = (value if value >= 0 else ~value).bit_length() + 1
        word_count = (signed_bits + 63) // 64
        expected = b"".join(
            ((value >> (64 * i)) & ((1 << 64) - 1)).to_bytes(8, sys.byteorder)
            for i in range(word_count)
        )
        content = bigint_module.content(value)
        assert content == expected
        assert bigint_module.from_content(content) == value
        padding = (b"\xff" if value < 0 else b"\x00") * 16
        assert bigint_module.from_content(content + padding) == value


def test_integer_promotion_and_demotion(bigint_module: tvm_ffi.Module) -> None:
    assert bigint_module.add((1 << 63) - 1, 1) == 1 << 63
    assert bigint_module.add(-(1 << 63), -1) == -(1 << 63) - 1
    for value in [1 << 63, -(1 << 63) - 1, _MULTIWORD_VALUE, -_MULTIWORD_VALUE]:
        assert bigint_module.add(value, -value) == 0
        assert bigint_module.is_int64(bigint_module.add(value, 17 - value))
        with pytest.raises(TypeError):
            bigint_module.int64_only(value)


def test_integer_type_schema(bigint_module: tvm_ffi.Module) -> None:
    metadata = bigint_module.get_function_metadata("identity")
    assert metadata is not None
    assert "ffi.BigInt" in metadata["type_schema"]
    schema = TypeSchema.from_json_str(metadata["type_schema"])
    assert str(schema) == "Callable[[int], int]"
    assert [arg.type_key for arg in schema.args] == ["ffi.BigInt", "ffi.BigInt"]


def test_integer_callbacks(bigint_module: tvm_ffi.Module) -> None:
    value = _MULTIWORD_VALUE
    received = []

    def callback(arg: int) -> int:
        assert type(arg) is int
        received.append(arg)
        return -arg - 17

    expected = -value - 17
    assert bigint_module.invoke(callback, value) == expected
    assert tvm_ffi.convert_func(callback)(value) == expected
    assert received == [value, value]
    assert tvm_ffi.convert_func(lambda: value)() == value


def test_integer_rvalue_callbacks(bigint_module: tvm_ffi.Module) -> None:
    def identity(value: int) -> int:
        assert type(value) is int
        return value

    def failing(value: int) -> int:
        assert type(value) is int
        raise ValueError("callback failed")

    value = _MULTIWORD_VALUE
    assert bigint_module.invoke_rvalue(identity, value) == value
    assert bigint_module.rvalue_ref_count_delta(identity, value) == 0
    with pytest.raises(ValueError, match="callback failed"):
        bigint_module.invoke_rvalue(failing, value)
    assert bigint_module.rvalue_ref_count_delta(failing, value) == 0


def test_integer_containers() -> None:
    values = [_MULTIWORD_VALUE, -_MULTIWORD_VALUE, 0, (1 << 63) - 1]
    echo = tvm_ffi.get_global_func("testing.echo")
    array = echo(values)
    assert list(array) == values
    mapping = echo({value: -value for value in values})
    assert len(mapping) == len(values)
    for value in values:
        assert mapping[value] == -value


def test_numpy_unsigned_integer() -> None:
    np = pytest.importorskip("numpy")
    echo = tvm_ffi.get_global_func("testing.echo")
    assert echo(np.uint64((1 << 64) - 1)) == (1 << 64) - 1
