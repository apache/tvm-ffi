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
"""Tests for C++ ABI-view generation from reflected dataclasses."""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional, Union, cast

import pytest
import tvm_ffi
from tvm_ffi import Array, Device, Map, Object
from tvm_ffi.core import DataType, TypeField, TypeInfo, _lookup_or_register_type_info_from_type_key
from tvm_ffi.dataclasses import gen_abi_cpp, py_class
from tvm_ffi.dataclasses.gen_abi_cpp import _Generator


@py_class("testing.gen_abi_cpp.Dependency")
class _Dependency(Object):
    value: int


@py_class("testing.gen_abi_cpp.Base")
class _Base(Object):
    base_flag: bool
    base_value: int


@py_class("testing.gen_abi_cpp.empty.Empty")
class _Empty(_Base):
    pass


@py_class("testing.gen_abi_cpp.other.Child")
class _Child(_Empty):
    child_flag: bool
    dependency: _Dependency


@py_class("testing.gen_abi_cpp.other.Sibling")
class _Sibling(_Empty):
    sibling_value: float


@py_class("testing.gen_abi_cpp.tail.Base")
class _TailBase(Object):
    parent_flag: bool


@py_class("testing.gen_abi_cpp.tail.Empty")
class _TailEmpty(_TailBase):
    pass


@py_class("testing.gen_abi_cpp.tail.Child")
class _TailChild(_TailEmpty):
    child_flag: bool


@py_class("testing.gen_abi_cpp.Mixed")
class _Mixed(Object):
    ready: bool
    sequence: int
    ratio: float
    pointer: ctypes.c_void_p
    dtype: DataType
    device: Device
    anything: Any
    title: str
    payload: bytes
    callback: Callable[[int], str]
    dependency: _Dependency
    array_items: Array[_Dependency]
    list_items: list[_Dependency]
    mapping: Map[str, _Dependency]
    dictionary: dict[str, _Dependency]
    # py_class resolves annotations at runtime, where PEP 604 unions are unavailable on Python 3.9.
    optional: Optional[int]  # noqa: UP045
    choice: Union[int, float]  # noqa: UP007


@py_class("testing.gen_abi_cpp.NestedStructural")
class _NestedStructural(Object):
    optionals: list[Optional[int]]  # noqa: UP045
    unions: dict[str, Union[int, float]]  # noqa: UP007
    optional_objects: list[Optional[_Dependency]]  # noqa: UP045
    union_objects: dict[str, Union[_Dependency, _Sibling]]  # noqa: UP007


@py_class("testing.gen_abi_cpp.ExtraObjects")
class _ExtraObjects(Object):
    module: tvm_ffi.Module
    interrupt: tvm_ffi.VisitInterrupt


@py_class("testing.gen_abi_cpp.ΔNode")
class _UnicodeType(Object):
    value: int


@py_class("testing.gen_abi_cpp.Recursive")
class _Recursive(Object):
    children: list[_Recursive]


@py_class("testing.gen_abi_cpp.MutualLeft")
class _MutualLeft(Object):
    rights: list[_MutualRight]


@py_class("testing.gen_abi_cpp.MutualRight")
class _MutualRight(Object):
    lefts: list[_MutualLeft]


@py_class("testing.gen_abi_cpp.Alpha")
class _SortAlpha(Object):
    pass


@py_class("testing.gen_abi_cpp.beta.Dependency")
class _SortNested(Object):
    pass


@py_class("testing.gen_abi_cpp.zulu")
class _SortZulu(Object):
    pass


@py_class("testing.gen_abi_cpp.SortOrder")
class _SortOrder(Object):
    alpha: _SortAlpha
    nested: _SortNested
    zulu: _SortZulu


@pytest.fixture(scope="module")
def native_layout_probes(tmp_path_factory: pytest.TempPathFactory) -> tvm_ffi.Module:
    source = r"""
        #include <tvm/ffi/container/dict.h>
        #include <tvm/ffi/container/list.h>
        #include <tvm/ffi/reflection/registry.h>

        namespace tvm {
        namespace ffi {
        namespace testing {

        struct ABIRawTensorObj : public Object {
          static constexpr bool _type_mutable = true;
          DLTensor* value{nullptr};
          TVM_FFI_DECLARE_OBJECT_INFO_FINAL(
              "testing.gen_abi_cpp.native.RawTensor", ABIRawTensorObj, Object);
        };

        struct ABIObjectContainersObj : public Object {
          Array<ABIRawTensorObj*> array_items;
          List<ABIRawTensorObj*> list_items;
          Map<String, ABIRawTensorObj*> mapping;
          Dict<String, ABIRawTensorObj*> dictionary;
          TVM_FFI_DECLARE_OBJECT_INFO_FINAL(
              "testing.gen_abi_cpp.native.ObjectContainers",
              ABIObjectContainersObj,
              Object);
        };

        struct ABINoMetadataObj : public Object {
          int64_t hidden[8];
          TVM_FFI_DECLARE_OBJECT_INFO_FINAL(
              "testing.gen_abi_cpp.native.NoMetadata", ABINoMetadataObj, Object);
        };

        TVM_FFI_STATIC_INIT_BLOCK() {
          namespace refl = ::tvm::ffi::reflection;
          refl::ObjectDef<ABIRawTensorObj>().def_ro("value", &ABIRawTensorObj::value);
          refl::ObjectDef<ABIObjectContainersObj>()
              .def_ro("array_items", &ABIObjectContainersObj::array_items)
              .def_ro("list_items", &ABIObjectContainersObj::list_items)
              .def_ro("mapping", &ABIObjectContainersObj::mapping)
              .def_ro("dictionary", &ABIObjectContainersObj::dictionary);
        }

        }  // namespace testing
        }  // namespace ffi
        }  // namespace tvm

        int32_t gen_abi_cpp_register_no_metadata() {
          return ::tvm::ffi::testing::ABINoMetadataObj::RuntimeTypeIndex();
        }
    """
    module = tvm_ffi.cpp.load_inline(
        name="test_dataclass_gen_abi_cpp_native_probes",
        cpp_sources=source,
        functions=["gen_abi_cpp_register_no_metadata"],
        build_directory=str(tmp_path_factory.mktemp("gen_abi_cpp_native_probes")),
    )
    module.gen_abi_cpp_register_no_metadata()
    return module


_GENERATED_TYPE_KEYS = [
    "ffi.Function",
    "testing.gen_abi_cpp.Mixed",
    "testing.gen_abi_cpp.Mutual*",
    "testing.gen_abi_cpp.NestedStructural",
    "testing.gen_abi_cpp.Recursive",
    "testing.gen_abi_cpp.other.*",
    "testing.gen_abi_cpp.tail.Child",
    "testing.gen_abi_cpp.ΔNode",
]

_EXPECTED_PROGRAM = r"""#pragma once

#include <tvm/ffi/tvm_ffi.h>

namespace testing {
namespace gen_abi_cpp {

struct BaseObj;

struct DependencyObj;

struct MixedObj;

struct MutualLeftObj;

struct MutualRightObj;

struct NestedStructuralObj;

struct RecursiveObj;

struct __ffi_escape_ce944e6f6465Obj;

}  // namespace gen_abi_cpp
}  // namespace testing

namespace testing {
namespace gen_abi_cpp {
namespace empty {

struct EmptyObj;

}  // namespace empty
}  // namespace gen_abi_cpp
}  // namespace testing

namespace testing {
namespace gen_abi_cpp {
namespace other {

struct ChildObj;

struct SiblingObj;

}  // namespace other
}  // namespace gen_abi_cpp
}  // namespace testing

namespace testing {
namespace gen_abi_cpp {
namespace tail {

struct BaseObj;

struct ChildObj;

struct EmptyObj;

}  // namespace tail
}  // namespace gen_abi_cpp
}  // namespace testing

template <>
inline constexpr bool tvm::ffi::is_object_subclass_v<::testing::gen_abi_cpp::BaseObj> = true;
template <>
inline constexpr bool tvm::ffi::is_object_subclass_v<::testing::gen_abi_cpp::DependencyObj> = true;
template <>
inline constexpr bool tvm::ffi::is_object_subclass_v<::testing::gen_abi_cpp::MixedObj> = true;
template <>
inline constexpr bool tvm::ffi::is_object_subclass_v<::testing::gen_abi_cpp::MutualLeftObj> = true;
template <>
inline constexpr bool tvm::ffi::is_object_subclass_v<::testing::gen_abi_cpp::MutualRightObj> = true;
template <>
inline constexpr bool tvm::ffi::is_object_subclass_v<::testing::gen_abi_cpp::NestedStructuralObj> = true;
template <>
inline constexpr bool tvm::ffi::is_object_subclass_v<::testing::gen_abi_cpp::RecursiveObj> = true;
template <>
inline constexpr bool tvm::ffi::is_object_subclass_v<::testing::gen_abi_cpp::__ffi_escape_ce944e6f6465Obj> = true;
template <>
inline constexpr bool tvm::ffi::is_object_subclass_v<::testing::gen_abi_cpp::empty::EmptyObj> = true;
template <>
inline constexpr bool tvm::ffi::is_object_subclass_v<::testing::gen_abi_cpp::other::ChildObj> = true;
template <>
inline constexpr bool tvm::ffi::is_object_subclass_v<::testing::gen_abi_cpp::other::SiblingObj> = true;
template <>
inline constexpr bool tvm::ffi::is_object_subclass_v<::testing::gen_abi_cpp::tail::BaseObj> = true;
template <>
inline constexpr bool tvm::ffi::is_object_subclass_v<::testing::gen_abi_cpp::tail::ChildObj> = true;
template <>
inline constexpr bool tvm::ffi::is_object_subclass_v<::testing::gen_abi_cpp::tail::EmptyObj> = true;

namespace testing {
namespace gen_abi_cpp {

struct DependencyObj : public ::tvm::ffi::Object {
  TVM_FFI_DECLARE_OBJECT_INFO_LOOKUP("testing.gen_abi_cpp.Dependency", 1);
};

}  // namespace gen_abi_cpp
}  // namespace testing

static_assert(sizeof(::tvm::ffi::Object) == 24);
static_assert(alignof(::tvm::ffi::Object) == 8);

#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winvalid-offsetof"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4749)
#endif

namespace testing {
namespace gen_abi_cpp {

struct alignas(8) BaseObj : public ::tvm::ffi::Object {
  TVM_FFI_DECLARE_OBJECT_INFO_LOOKUP("testing.gen_abi_cpp.Base", 1);

  bool base_flag;  // offset=24, size=1, align=1
  int64_t base_value;  // offset=32, size=8, align=8
};

static_assert(sizeof(BaseObj) == 40);
static_assert(alignof(BaseObj) == 8);
static_assert(sizeof(decltype(BaseObj::base_flag)) == 1);
static_assert(alignof(decltype(BaseObj::base_flag)) == 1);
static_assert(offsetof(BaseObj, base_flag) == 24);
static_assert(sizeof(decltype(BaseObj::base_value)) == 8);
static_assert(alignof(decltype(BaseObj::base_value)) == 8);
static_assert(offsetof(BaseObj, base_value) == 32);

struct alignas(8) MixedObj : public ::tvm::ffi::Object {
  TVM_FFI_DECLARE_OBJECT_INFO_LOOKUP("testing.gen_abi_cpp.Mixed", 1);

  bool ready;  // offset=24, size=1, align=1
  int64_t sequence;  // offset=32, size=8, align=8
  double ratio;  // offset=40, size=8, align=8
  void* pointer;  // offset=48, size=8, align=8
  DLDataType dtype;  // offset=56, size=4, align=2
  DLDevice device;  // offset=60, size=8, align=4
  ::tvm::ffi::Any anything;  // offset=72, size=16, align=8
  ::tvm::ffi::String title;  // offset=88, size=16, align=8
  ::tvm::ffi::Bytes payload;  // offset=104, size=16, align=8
  ::tvm::ffi::Function callback;  // offset=120, size=8, align=8
  ::tvm::ffi::ObjectPtr<::testing::gen_abi_cpp::DependencyObj> dependency;  // offset=128, size=8, align=8
  ::tvm::ffi::Array<::tvm::ffi::ObjectPtr<::testing::gen_abi_cpp::DependencyObj>> array_items;  // offset=136, size=8, align=8
  ::tvm::ffi::List<::tvm::ffi::ObjectPtr<::testing::gen_abi_cpp::DependencyObj>> list_items;  // offset=144, size=8, align=8
  ::tvm::ffi::Map<::tvm::ffi::String, ::tvm::ffi::ObjectPtr<::testing::gen_abi_cpp::DependencyObj>> mapping;  // offset=152, size=8, align=8
  ::tvm::ffi::Dict<::tvm::ffi::String, ::tvm::ffi::ObjectPtr<::testing::gen_abi_cpp::DependencyObj>> dictionary;  // offset=160, size=8, align=8
  ::tvm::ffi::Any optional;  // offset=168, size=16, align=8
  ::tvm::ffi::Any choice;  // offset=184, size=16, align=8
};

static_assert(sizeof(MixedObj) == 200);
static_assert(alignof(MixedObj) == 8);
static_assert(sizeof(decltype(MixedObj::ready)) == 1);
static_assert(alignof(decltype(MixedObj::ready)) == 1);
static_assert(offsetof(MixedObj, ready) == 24);
static_assert(sizeof(decltype(MixedObj::sequence)) == 8);
static_assert(alignof(decltype(MixedObj::sequence)) == 8);
static_assert(offsetof(MixedObj, sequence) == 32);
static_assert(sizeof(decltype(MixedObj::ratio)) == 8);
static_assert(alignof(decltype(MixedObj::ratio)) == 8);
static_assert(offsetof(MixedObj, ratio) == 40);
static_assert(sizeof(decltype(MixedObj::pointer)) == 8);
static_assert(alignof(decltype(MixedObj::pointer)) == 8);
static_assert(offsetof(MixedObj, pointer) == 48);
static_assert(sizeof(decltype(MixedObj::dtype)) == 4);
static_assert(alignof(decltype(MixedObj::dtype)) == 2);
static_assert(offsetof(MixedObj, dtype) == 56);
static_assert(sizeof(decltype(MixedObj::device)) == 8);
static_assert(alignof(decltype(MixedObj::device)) == 4);
static_assert(offsetof(MixedObj, device) == 60);
static_assert(sizeof(decltype(MixedObj::anything)) == 16);
static_assert(alignof(decltype(MixedObj::anything)) == 8);
static_assert(offsetof(MixedObj, anything) == 72);
static_assert(sizeof(decltype(MixedObj::title)) == 16);
static_assert(alignof(decltype(MixedObj::title)) == 8);
static_assert(offsetof(MixedObj, title) == 88);
static_assert(sizeof(decltype(MixedObj::payload)) == 16);
static_assert(alignof(decltype(MixedObj::payload)) == 8);
static_assert(offsetof(MixedObj, payload) == 104);
static_assert(sizeof(decltype(MixedObj::callback)) == 8);
static_assert(alignof(decltype(MixedObj::callback)) == 8);
static_assert(offsetof(MixedObj, callback) == 120);
static_assert(sizeof(decltype(MixedObj::dependency)) == 8);
static_assert(alignof(decltype(MixedObj::dependency)) == 8);
static_assert(offsetof(MixedObj, dependency) == 128);
static_assert(sizeof(decltype(MixedObj::array_items)) == 8);
static_assert(alignof(decltype(MixedObj::array_items)) == 8);
static_assert(offsetof(MixedObj, array_items) == 136);
static_assert(sizeof(decltype(MixedObj::list_items)) == 8);
static_assert(alignof(decltype(MixedObj::list_items)) == 8);
static_assert(offsetof(MixedObj, list_items) == 144);
static_assert(sizeof(decltype(MixedObj::mapping)) == 8);
static_assert(alignof(decltype(MixedObj::mapping)) == 8);
static_assert(offsetof(MixedObj, mapping) == 152);
static_assert(sizeof(decltype(MixedObj::dictionary)) == 8);
static_assert(alignof(decltype(MixedObj::dictionary)) == 8);
static_assert(offsetof(MixedObj, dictionary) == 160);
static_assert(sizeof(decltype(MixedObj::optional)) == 16);
static_assert(alignof(decltype(MixedObj::optional)) == 8);
static_assert(offsetof(MixedObj, optional) == 168);
static_assert(sizeof(decltype(MixedObj::choice)) == 16);
static_assert(alignof(decltype(MixedObj::choice)) == 8);
static_assert(offsetof(MixedObj, choice) == 184);

struct alignas(8) MutualLeftObj : public ::tvm::ffi::Object {
  TVM_FFI_DECLARE_OBJECT_INFO_LOOKUP("testing.gen_abi_cpp.MutualLeft", 1);

  ::tvm::ffi::List<::tvm::ffi::ObjectPtr<::testing::gen_abi_cpp::MutualRightObj>> rights;  // offset=24, size=8, align=8
};

static_assert(sizeof(MutualLeftObj) == 32);
static_assert(alignof(MutualLeftObj) == 8);
static_assert(sizeof(decltype(MutualLeftObj::rights)) == 8);
static_assert(alignof(decltype(MutualLeftObj::rights)) == 8);
static_assert(offsetof(MutualLeftObj, rights) == 24);

struct alignas(8) MutualRightObj : public ::tvm::ffi::Object {
  TVM_FFI_DECLARE_OBJECT_INFO_LOOKUP("testing.gen_abi_cpp.MutualRight", 1);

  ::tvm::ffi::List<::tvm::ffi::ObjectPtr<::testing::gen_abi_cpp::MutualLeftObj>> lefts;  // offset=24, size=8, align=8
};

static_assert(sizeof(MutualRightObj) == 32);
static_assert(alignof(MutualRightObj) == 8);
static_assert(sizeof(decltype(MutualRightObj::lefts)) == 8);
static_assert(alignof(decltype(MutualRightObj::lefts)) == 8);
static_assert(offsetof(MutualRightObj, lefts) == 24);

struct alignas(8) NestedStructuralObj : public ::tvm::ffi::Object {
  TVM_FFI_DECLARE_OBJECT_INFO_LOOKUP("testing.gen_abi_cpp.NestedStructural", 1);

  ::tvm::ffi::List<::tvm::ffi::Any> optionals;  // offset=24, size=8, align=8
  ::tvm::ffi::Dict<::tvm::ffi::String, ::tvm::ffi::Any> unions;  // offset=32, size=8, align=8
  ::tvm::ffi::List<::tvm::ffi::Any> optional_objects;  // offset=40, size=8, align=8
  ::tvm::ffi::Dict<::tvm::ffi::String, ::tvm::ffi::Any> union_objects;  // offset=48, size=8, align=8
};

static_assert(sizeof(NestedStructuralObj) == 56);
static_assert(alignof(NestedStructuralObj) == 8);
static_assert(sizeof(decltype(NestedStructuralObj::optionals)) == 8);
static_assert(alignof(decltype(NestedStructuralObj::optionals)) == 8);
static_assert(offsetof(NestedStructuralObj, optionals) == 24);
static_assert(sizeof(decltype(NestedStructuralObj::unions)) == 8);
static_assert(alignof(decltype(NestedStructuralObj::unions)) == 8);
static_assert(offsetof(NestedStructuralObj, unions) == 32);
static_assert(sizeof(decltype(NestedStructuralObj::optional_objects)) == 8);
static_assert(alignof(decltype(NestedStructuralObj::optional_objects)) == 8);
static_assert(offsetof(NestedStructuralObj, optional_objects) == 40);
static_assert(sizeof(decltype(NestedStructuralObj::union_objects)) == 8);
static_assert(alignof(decltype(NestedStructuralObj::union_objects)) == 8);
static_assert(offsetof(NestedStructuralObj, union_objects) == 48);

struct alignas(8) RecursiveObj : public ::tvm::ffi::Object {
  TVM_FFI_DECLARE_OBJECT_INFO_LOOKUP("testing.gen_abi_cpp.Recursive", 1);

  ::tvm::ffi::List<::tvm::ffi::ObjectPtr<::testing::gen_abi_cpp::RecursiveObj>> children;  // offset=24, size=8, align=8
};

static_assert(sizeof(RecursiveObj) == 32);
static_assert(alignof(RecursiveObj) == 8);
static_assert(sizeof(decltype(RecursiveObj::children)) == 8);
static_assert(alignof(decltype(RecursiveObj::children)) == 8);
static_assert(offsetof(RecursiveObj, children) == 24);

}  // namespace gen_abi_cpp
}  // namespace testing

namespace testing {
namespace gen_abi_cpp {
namespace tail {

struct alignas(8) BaseObj : public ::tvm::ffi::Object {
  TVM_FFI_DECLARE_OBJECT_INFO_LOOKUP("testing.gen_abi_cpp.tail.Base", 1);

  bool parent_flag;  // offset=24, size=1, align=1
};

static_assert(sizeof(BaseObj) == 32);
static_assert(alignof(BaseObj) == 8);
static_assert(sizeof(decltype(BaseObj::parent_flag)) == 1);
static_assert(alignof(decltype(BaseObj::parent_flag)) == 1);
static_assert(offsetof(BaseObj, parent_flag) == 24);

}  // namespace tail
}  // namespace gen_abi_cpp
}  // namespace testing

namespace testing {
namespace gen_abi_cpp {

struct alignas(8) __ffi_escape_ce944e6f6465Obj : public ::tvm::ffi::Object {
  TVM_FFI_DECLARE_OBJECT_INFO_LOOKUP("testing.gen_abi_cpp.\316\224Node", 1);

  int64_t value;  // offset=24, size=8, align=8
};

static_assert(sizeof(__ffi_escape_ce944e6f6465Obj) == 32);
static_assert(alignof(__ffi_escape_ce944e6f6465Obj) == 8);
static_assert(sizeof(decltype(__ffi_escape_ce944e6f6465Obj::value)) == 8);
static_assert(alignof(decltype(__ffi_escape_ce944e6f6465Obj::value)) == 8);
static_assert(offsetof(__ffi_escape_ce944e6f6465Obj, value) == 24);

}  // namespace gen_abi_cpp
}  // namespace testing

namespace testing {
namespace gen_abi_cpp {
namespace empty {

struct alignas(8) EmptyObj : public ::testing::gen_abi_cpp::BaseObj {
  TVM_FFI_DECLARE_OBJECT_INFO_LOOKUP("testing.gen_abi_cpp.empty.Empty", 2);
};

static_assert(sizeof(EmptyObj) == 40);
static_assert(alignof(EmptyObj) == 8);

}  // namespace empty
}  // namespace gen_abi_cpp
}  // namespace testing

namespace testing {
namespace gen_abi_cpp {
namespace tail {

struct alignas(8) EmptyObj : public ::testing::gen_abi_cpp::tail::BaseObj {
  TVM_FFI_DECLARE_OBJECT_INFO_LOOKUP("testing.gen_abi_cpp.tail.Empty", 2);
};

static_assert(sizeof(EmptyObj) == 32);
static_assert(alignof(EmptyObj) == 8);

}  // namespace tail
}  // namespace gen_abi_cpp
}  // namespace testing

namespace testing {
namespace gen_abi_cpp {
namespace other {

struct alignas(8) ChildObj : public ::testing::gen_abi_cpp::empty::EmptyObj {
  TVM_FFI_DECLARE_OBJECT_INFO_LOOKUP("testing.gen_abi_cpp.other.Child", 3);

  bool child_flag;  // offset=40, size=1, align=1
  ::tvm::ffi::ObjectPtr<::testing::gen_abi_cpp::DependencyObj> dependency;  // offset=48, size=8, align=8
};

static_assert(sizeof(ChildObj) == 56);
static_assert(alignof(ChildObj) == 8);
static_assert(sizeof(decltype(ChildObj::child_flag)) == 1);
static_assert(alignof(decltype(ChildObj::child_flag)) == 1);
static_assert(offsetof(ChildObj, child_flag) == 40);
static_assert(sizeof(decltype(ChildObj::dependency)) == 8);
static_assert(alignof(decltype(ChildObj::dependency)) == 8);
static_assert(offsetof(ChildObj, dependency) == 48);

struct alignas(8) SiblingObj : public ::testing::gen_abi_cpp::empty::EmptyObj {
  TVM_FFI_DECLARE_OBJECT_INFO_LOOKUP("testing.gen_abi_cpp.other.Sibling", 3);

  double sibling_value;  // offset=40, size=8, align=8
};

static_assert(sizeof(SiblingObj) == 48);
static_assert(alignof(SiblingObj) == 8);
static_assert(sizeof(decltype(SiblingObj::sibling_value)) == 8);
static_assert(alignof(decltype(SiblingObj::sibling_value)) == 8);
static_assert(offsetof(SiblingObj, sibling_value) == 40);

}  // namespace other
}  // namespace gen_abi_cpp
}  // namespace testing

namespace testing {
namespace gen_abi_cpp {
namespace tail {

struct alignas(8) ChildObj : public ::testing::gen_abi_cpp::tail::EmptyObj {
  TVM_FFI_DECLARE_OBJECT_INFO_LOOKUP("testing.gen_abi_cpp.tail.Child", 3);

  bool child_flag;  // offset=25, size=1, align=1
};

static_assert(sizeof(ChildObj) == 32);
static_assert(alignof(ChildObj) == 8);
static_assert(sizeof(decltype(ChildObj::child_flag)) == 1);
static_assert(alignof(decltype(ChildObj::child_flag)) == 1);
static_assert(offsetof(ChildObj, child_flag) == 25);

}  // namespace tail
}  // namespace gen_abi_cpp
}  // namespace testing

#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif
"""

if sys.platform == "win32":
    # The Microsoft C++ ABI starts direct-subclass fields after the complete
    # base object instead of reusing its tail padding.
    _EXPECTED_PROGRAM = _EXPECTED_PROGRAM.replace(
        r"""struct alignas(8) ChildObj : public ::testing::gen_abi_cpp::tail::EmptyObj {
  TVM_FFI_DECLARE_OBJECT_INFO_LOOKUP("testing.gen_abi_cpp.tail.Child", 3);

  bool child_flag;  // offset=25, size=1, align=1
};

static_assert(sizeof(ChildObj) == 32);
static_assert(alignof(ChildObj) == 8);
static_assert(sizeof(decltype(ChildObj::child_flag)) == 1);
static_assert(alignof(decltype(ChildObj::child_flag)) == 1);
static_assert(offsetof(ChildObj, child_flag) == 25);""",
        r"""struct alignas(8) ChildObj : public ::testing::gen_abi_cpp::tail::EmptyObj {
  TVM_FFI_DECLARE_OBJECT_INFO_LOOKUP("testing.gen_abi_cpp.tail.Child", 3);

  bool child_flag;  // offset=32, size=1, align=1
};

static_assert(sizeof(ChildObj) == 40);
static_assert(alignof(ChildObj) == 8);
static_assert(sizeof(decltype(ChildObj::child_flag)) == 1);
static_assert(alignof(decltype(ChildObj::child_flag)) == 1);
static_assert(offsetof(ChildObj, child_flag) == 32);""",
    )

_EXPECTED_NATIVE_PROGRAM = r"""#pragma once

#include <tvm/ffi/tvm_ffi.h>

namespace testing {
namespace gen_abi_cpp {
namespace native {

struct ObjectContainersObj;

struct RawTensorObj;

}  // namespace native
}  // namespace gen_abi_cpp
}  // namespace testing

template <>
inline constexpr bool tvm::ffi::is_object_subclass_v<::testing::gen_abi_cpp::native::ObjectContainersObj> = true;
template <>
inline constexpr bool tvm::ffi::is_object_subclass_v<::testing::gen_abi_cpp::native::RawTensorObj> = true;

namespace testing {
namespace gen_abi_cpp {
namespace native {

struct RawTensorObj : public ::tvm::ffi::Object {
  TVM_FFI_DECLARE_OBJECT_INFO_LOOKUP("testing.gen_abi_cpp.native.RawTensor", 1);
};

}  // namespace native
}  // namespace gen_abi_cpp
}  // namespace testing

static_assert(sizeof(::tvm::ffi::Object) == 24);
static_assert(alignof(::tvm::ffi::Object) == 8);

#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winvalid-offsetof"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4749)
#endif

namespace testing {
namespace gen_abi_cpp {
namespace native {

struct alignas(8) ObjectContainersObj : public ::tvm::ffi::Object {
  TVM_FFI_DECLARE_OBJECT_INFO_LOOKUP("testing.gen_abi_cpp.native.ObjectContainers", 1);

  ::tvm::ffi::Array<::tvm::ffi::ObjectPtr<::testing::gen_abi_cpp::native::RawTensorObj>> array_items;  // offset=24, size=8, align=8
  ::tvm::ffi::List<::tvm::ffi::ObjectPtr<::testing::gen_abi_cpp::native::RawTensorObj>> list_items;  // offset=32, size=8, align=8
  ::tvm::ffi::Map<::tvm::ffi::String, ::tvm::ffi::ObjectPtr<::testing::gen_abi_cpp::native::RawTensorObj>> mapping;  // offset=40, size=8, align=8
  ::tvm::ffi::Dict<::tvm::ffi::String, ::tvm::ffi::ObjectPtr<::testing::gen_abi_cpp::native::RawTensorObj>> dictionary;  // offset=48, size=8, align=8
};

static_assert(sizeof(ObjectContainersObj) == 56);
static_assert(alignof(ObjectContainersObj) == 8);
static_assert(sizeof(decltype(ObjectContainersObj::array_items)) == 8);
static_assert(alignof(decltype(ObjectContainersObj::array_items)) == 8);
static_assert(offsetof(ObjectContainersObj, array_items) == 24);
static_assert(sizeof(decltype(ObjectContainersObj::list_items)) == 8);
static_assert(alignof(decltype(ObjectContainersObj::list_items)) == 8);
static_assert(offsetof(ObjectContainersObj, list_items) == 32);
static_assert(sizeof(decltype(ObjectContainersObj::mapping)) == 8);
static_assert(alignof(decltype(ObjectContainersObj::mapping)) == 8);
static_assert(offsetof(ObjectContainersObj, mapping) == 40);
static_assert(sizeof(decltype(ObjectContainersObj::dictionary)) == 8);
static_assert(alignof(decltype(ObjectContainersObj::dictionary)) == 8);
static_assert(offsetof(ObjectContainersObj, dictionary) == 48);

}  // namespace native
}  // namespace gen_abi_cpp
}  // namespace testing

#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif
"""

_EXPECTED_EXTRA_PROGRAM = r"""#pragma once

#include <tvm/ffi/tvm_ffi.h>

namespace testing {
namespace gen_abi_cpp {

struct ExtraObjectsObj;

}  // namespace gen_abi_cpp
}  // namespace testing

template <>
inline constexpr bool tvm::ffi::is_object_subclass_v<::testing::gen_abi_cpp::ExtraObjectsObj> = true;

static_assert(sizeof(::tvm::ffi::Object) == 24);
static_assert(alignof(::tvm::ffi::Object) == 8);

#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winvalid-offsetof"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4749)
#endif

namespace testing {
namespace gen_abi_cpp {

struct alignas(8) ExtraObjectsObj : public ::tvm::ffi::Object {
  TVM_FFI_DECLARE_OBJECT_INFO_LOOKUP("testing.gen_abi_cpp.ExtraObjects", 1);

  ::tvm::ffi::ObjectPtr<::tvm::ffi::ModuleObj> module;  // offset=24, size=8, align=8
  ::tvm::ffi::ObjectPtr<::tvm::ffi::VisitInterruptObj> interrupt;  // offset=32, size=8, align=8
};

static_assert(sizeof(ExtraObjectsObj) == 40);
static_assert(alignof(ExtraObjectsObj) == 8);
static_assert(sizeof(decltype(ExtraObjectsObj::module)) == 8);
static_assert(alignof(decltype(ExtraObjectsObj::module)) == 8);
static_assert(offsetof(ExtraObjectsObj, module) == 24);
static_assert(sizeof(decltype(ExtraObjectsObj::interrupt)) == 8);
static_assert(alignof(decltype(ExtraObjectsObj::interrupt)) == 8);
static_assert(offsetof(ExtraObjectsObj, interrupt) == 32);

}  // namespace gen_abi_cpp
}  // namespace testing

#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif
"""

_EXPECTED_EMPTY_PROGRAM = r"""#pragma once

#include <tvm/ffi/tvm_ffi.h>


static_assert(sizeof(::tvm::ffi::Object) == 24);
static_assert(alignof(::tvm::ffi::Object) == 8);

#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winvalid-offsetof"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4749)
#endif

#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif
"""


def test_generated_program() -> None:
    assert gen_abi_cpp(_GENERATED_TYPE_KEYS) == _EXPECTED_PROGRAM
    assert gen_abi_cpp([*_GENERATED_TYPE_KEYS, "testing.gen_abi_cpp.Mix*"]) == _EXPECTED_PROGRAM


def test_native_generated_program(native_layout_probes: tvm_ffi.Module) -> None:
    del native_layout_probes
    assert gen_abi_cpp("testing.gen_abi_cpp.native.ObjectContainers") == _EXPECTED_NATIVE_PROGRAM


def test_builtin_generated_program() -> None:
    assert gen_abi_cpp("ffi.Function") == _EXPECTED_EMPTY_PROGRAM
    with pytest.raises(ValueError):
        gen_abi_cpp("ffi.OpaquePyObject")


@pytest.mark.parametrize("selector", ["testing.gen_abi_cpp.DoesNotExist", "no.match.*"])
def test_unmatched_selector_rejected(selector: str) -> None:
    with pytest.raises(ValueError, match="did not match"):
        gen_abi_cpp(selector)


def test_dependencies_are_grouped_by_namespace() -> None:
    header = gen_abi_cpp("testing.gen_abi_cpp.SortOrder")
    forward_declarations = header.split("template <>\ninline constexpr bool", maxsplit=1)[0]
    assert forward_declarations.count("namespace gen_abi_cpp {") == 2
    assert forward_declarations.index("struct zuluObj;") < forward_declarations.index(
        "namespace beta {"
    )


def test_schema_less_custom_object_uses_object_ptr_carrier() -> None:
    field = cast(
        TypeField,
        SimpleNamespace(
            name="value",
            size=8,
            alignment=8,
            offset=24,
            field_static_type_index=128,
            ty=None,
        ),
    )
    owner = cast(
        TypeInfo,
        SimpleNamespace(type_key="testing.gen_abi_cpp.SchemaLess"),
    )

    carrier = _Generator([])._lower_field(field, owner)

    assert carrier.cpp_type == "::tvm::ffi::ObjectPtr<::tvm::ffi::Object>"
    assert (carrier.size, carrier.alignment) == (8, 8)


def test_extra_object_carriers_remain_typed(tmp_path: Path) -> None:
    header = gen_abi_cpp("testing.gen_abi_cpp.ExtraObjects")
    assert header == _EXPECTED_EXTRA_PROGRAM

    header_path = tmp_path / "extra_objects.h"
    source_path = tmp_path / "extra_objects_test.cc"
    header_path.write_text(header)
    source_path.write_text(
        "#include <tvm/ffi/extra/module.h>\n"
        "#include <tvm/ffi/extra/structural_visit.h>\n"
        '#include "extra_objects.h"\n'
    )
    object_path = tvm_ffi.cpp.build(
        name="test_dataclass_gen_abi_cpp_extra_objects",
        sources=str(source_path),
        extra_include_paths=[str(tmp_path)],
        build_directory=str(tmp_path / "build"),
        output="extra_objects_test.o",
    )
    assert Path(object_path).is_file()


def test_non_object_registered_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="did not match any registered object type"):
        gen_abi_cpp("int")


def test_unsafe_native_schema_alias_is_rejected(
    native_layout_probes: tvm_ffi.Module,
) -> None:
    del native_layout_probes
    with pytest.raises(ValueError, match=r"DLTensor\*"):
        gen_abi_cpp("testing.gen_abi_cpp.native.RawTensor")


def test_native_type_without_size_metadata_is_rejected(
    native_layout_probes: tvm_ffi.Module,
) -> None:
    del native_layout_probes
    with pytest.raises(ValueError, match="does not expose fixed total-size metadata"):
        gen_abi_cpp("testing.gen_abi_cpp.native.NoMetadata")


def test_ambiguous_native_optional_is_rejected() -> None:
    with pytest.raises(ValueError, match="Ambiguous native Optional carrier"):
        gen_abi_cpp("testing.SchemaAllTypes")


def test_generated_header_is_self_contained(tmp_path: Path) -> None:
    header_path = tmp_path / "abi.h"
    source_path = tmp_path / "abi_header_test.cc"
    header_path.write_text(gen_abi_cpp(_GENERATED_TYPE_KEYS))
    source_path.write_text('#include "abi.h"\n')

    object_path = tvm_ffi.cpp.build(
        name="test_dataclass_gen_abi_cpp_header",
        sources=str(source_path),
        extra_include_paths=[str(tmp_path)],
        build_directory=str(tmp_path / "build"),
        output="abi_header_test.o",
    )
    assert Path(object_path).is_file()


def test_generated_header_compiles_and_reads_live_objects(tmp_path: Path) -> None:
    header = gen_abi_cpp(_GENERATED_TYPE_KEYS)
    source = (
        header
        + r"""
        static_assert(std::is_base_of_v<
                      ::tvm::ffi::Object,
                      ::testing::gen_abi_cpp::DependencyObj>);
        static_assert(std::is_base_of_v<
                      ::tvm::ffi::Object,
                      ::testing::gen_abi_cpp::MixedObj>);
        static_assert(std::is_base_of_v<
                      ::testing::gen_abi_cpp::BaseObj,
                      ::testing::gen_abi_cpp::other::ChildObj>);
        static_assert(std::is_base_of_v<
                      ::testing::gen_abi_cpp::empty::EmptyObj,
                      ::testing::gen_abi_cpp::other::ChildObj>);
        static_assert(std::is_constructible_v<
                      ::tvm::ffi::ObjectPtr<::testing::gen_abi_cpp::BaseObj>,
                      ::tvm::ffi::ObjectPtr<
                          ::testing::gen_abi_cpp::other::ChildObj>>);

        int64_t gen_abi_cpp_read_sequence(
            const ::testing::gen_abi_cpp::MixedObj* value) {
          return value->sequence;
        }

        int64_t gen_abi_cpp_read_inherited_value(
            const ::testing::gen_abi_cpp::other::ChildObj* value) {
          const ::testing::gen_abi_cpp::BaseObj* base = value;
          return base->base_value;
        }

        bool gen_abi_cpp_upcast_identity(
            ::tvm::ffi::ObjectPtr<
                ::testing::gen_abi_cpp::other::ChildObj> child) {
          ::tvm::ffi::ObjectPtr<::testing::gen_abi_cpp::BaseObj> base = child;
          return reinterpret_cast<const void*>(child.get()) ==
                 reinterpret_cast<const void*>(base.get());
        }

        int32_t gen_abi_cpp_unicode_type_index() {
          return ::testing::gen_abi_cpp::__ffi_escape_ce944e6f6465Obj::RuntimeTypeIndex();
        }
        """
    )
    module = tvm_ffi.cpp.load_inline(
        name="test_dataclass_gen_abi_cpp",
        cpp_sources=source,
        functions=[
            "gen_abi_cpp_read_sequence",
            "gen_abi_cpp_read_inherited_value",
            "gen_abi_cpp_upcast_identity",
            "gen_abi_cpp_unicode_type_index",
        ],
        build_directory=str(tmp_path / "build"),
    )

    mixed = _Mixed(
        ready=True,
        sequence=123,
        ratio=1.5,
        pointer=ctypes.c_void_p(),
        dtype=DataType("float32"),
        device=Device("cpu", 0),
        anything=None,
        title="title",
        payload=b"payload",
        callback=lambda value: str(value),
        dependency=_Dependency(value=1),
        array_items=Array[_Dependency]([]),
        list_items=[],
        mapping=Map[str, _Dependency]({}),
        dictionary={},
        optional=None,
        choice=2.0,
    )
    child = _Child(base_flag=True, base_value=4, child_flag=False, dependency=_Dependency(value=2))
    assert module.gen_abi_cpp_read_sequence(mixed) == 123
    assert module.gen_abi_cpp_read_inherited_value(child) == 4
    assert module.gen_abi_cpp_upcast_identity(child) is True
    unicode_info = _lookup_or_register_type_info_from_type_key("testing.gen_abi_cpp.ΔNode")
    assert module.gen_abi_cpp_unicode_type_index() == unicode_info.type_index
