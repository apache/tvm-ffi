/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
// This file is used for testing the FFI API.
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>

#include <chrono>
#include <iostream>
#include <thread>

namespace tvm {
namespace ffi {

// Step 1: Define the object class (stores the actual data)
class TestIntPairObj : public tvm::ffi::Object {
 public:
  int64_t a;
  int64_t b;

  TestIntPairObj() = default;
  TestIntPairObj(int64_t a, int64_t b) : a(a), b(b) {}

  // Required: declare type information
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("testing.TestIntPair", TestIntPairObj, tvm::ffi::Object);
};

// Step 2: Define the reference wrapper (user-facing interface)
class TestIntPair : public tvm::ffi::ObjectRef {
 public:
  // Constructor
  explicit TestIntPair(int64_t a, int64_t b) {
    data_ = tvm::ffi::make_object<TestIntPairObj>(a, b);
  }

  // Required: define object reference methods
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TestIntPair, tvm::ffi::ObjectRef, TestIntPairObj);
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TestIntPairObj>()
      .def_ro("a", &TestIntPairObj::a, "Field `a`")
      .def_ro("b", &TestIntPairObj::b, "Field `b`")
      .def_static("__ffi_init__", refl::init<TestIntPairObj, int64_t, int64_t>);
}

class TestObjectBase : public Object {
 public:
  int64_t v_i64;
  double v_f64;
  String v_str;

  int64_t AddI64(int64_t other) const { return v_i64 + other; }

  // declare as one slot, with float as overflow
  static constexpr bool _type_mutable = true;
  static constexpr uint32_t _type_child_slots = 1;
  TVM_FFI_DECLARE_OBJECT_INFO("testing.TestObjectBase", TestObjectBase, Object);
};

class TestObjectDerived : public TestObjectBase {
 public:
  Map<Any, Any> v_map;
  Array<Any> v_array;

  // declare as one slot, with float as overflow
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("testing.TestObjectDerived", TestObjectDerived, TestObjectBase);
};

class TestCxxClassBase : public Object {
 public:
  int64_t v_i64;
  int32_t v_i32;

  TestCxxClassBase(int64_t v_i64, int32_t v_i32) : v_i64(v_i64), v_i32(v_i32) {}

  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("testing.TestCxxClassBase", TestCxxClassBase, Object);
};

class TestCxxClassDerived : public TestCxxClassBase {
 public:
  double v_f64;
  float v_f32;

  TestCxxClassDerived(int64_t v_i64, int32_t v_i32, double v_f64, float v_f32)
      : TestCxxClassBase(v_i64, v_i32), v_f64(v_f64), v_f32(v_f32) {}

  TVM_FFI_DECLARE_OBJECT_INFO("testing.TestCxxClassDerived", TestCxxClassDerived, TestCxxClassBase);
};

class TestCxxClassDerivedDerived : public TestCxxClassDerived {
 public:
  String v_str;
  bool v_bool;

  TestCxxClassDerivedDerived(int64_t v_i64, int32_t v_i32, double v_f64, float v_f32, String v_str,
                             bool v_bool)
      : TestCxxClassDerived(v_i64, v_i32, v_f64, v_f32), v_str(v_str), v_bool(v_bool) {}

  TVM_FFI_DECLARE_OBJECT_INFO("testing.TestCxxClassDerivedDerived", TestCxxClassDerivedDerived,
                              TestCxxClassDerived);
};

class TestCxxInitSubsetObj : public Object {
 public:
  int64_t required_field;
  int64_t optional_field;
  String note;

  explicit TestCxxInitSubsetObj(int64_t value, String note)
      : required_field(value), optional_field(-1), note(note) {}

  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("testing.TestCxxInitSubset", TestCxxInitSubsetObj, Object);
};

class TestUnregisteredBaseObject : public Object {
 public:
  int64_t v1;
  explicit TestUnregisteredBaseObject(int64_t v1) : v1(v1) {}
  int64_t GetV1PlusOne() const { return v1 + 1; }
  TVM_FFI_DECLARE_OBJECT_INFO("testing.TestUnregisteredBaseObject", TestUnregisteredBaseObject,
                              Object);
};

class TestUnregisteredObject : public TestUnregisteredBaseObject {
 public:
  int64_t v2;
  explicit TestUnregisteredObject(int64_t v1, int64_t v2)
      : TestUnregisteredBaseObject(v1), v2(v2) {}
  int64_t GetV2PlusTwo() const { return v2 + 2; }
  TVM_FFI_DECLARE_OBJECT_INFO("testing.TestUnregisteredObject", TestUnregisteredObject,
                              TestUnregisteredBaseObject);
};

TVM_FFI_NO_INLINE void TestRaiseError(String kind, String msg) {
  // keep name and no liner for testing backtrace
  throw ffi::Error(kind, msg, TVMFFIBacktrace(__FILE__, __LINE__, TVM_FFI_FUNC_SIG, 0));
}

TVM_FFI_NO_INLINE void TestApply(PackedArgs args, Any* ret) {
  // keep name and no liner for testing backtrace
  auto f = args[0].cast<Function>();
  f.CallPacked(args.Slice(1), ret);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;

  refl::ObjectDef<TestObjectBase>()
      .def_rw("v_i64", &TestObjectBase::v_i64, refl::DefaultValue(10), "i64 field")
      .def_ro("v_f64", &TestObjectBase::v_f64, refl::DefaultValue(10.0))
      .def_rw("v_str", &TestObjectBase::v_str, refl::DefaultValue("hello"))
      .def("add_i64", &TestObjectBase::AddI64, "add_i64 method");

  refl::ObjectDef<TestObjectDerived>()
      .def_ro("v_map", &TestObjectDerived::v_map)
      .def_ro("v_array", &TestObjectDerived::v_array);

  refl::ObjectDef<TestCxxClassBase>()
      .def_static("__ffi_init__", refl::init<TestCxxClassBase, int64_t, int32_t>)
      .def_rw("v_i64", &TestCxxClassBase::v_i64)
      .def_rw("v_i32", &TestCxxClassBase::v_i32);

  refl::ObjectDef<TestCxxClassDerived>()
      .def_static("__ffi_init__", refl::init<TestCxxClassDerived, int64_t, int32_t, double, float>)
      .def_rw("v_f64", &TestCxxClassDerived::v_f64)
      .def_rw("v_f32", &TestCxxClassDerived::v_f32);

  refl::ObjectDef<TestCxxClassDerivedDerived>()
      .def_static(
          "__ffi_init__",
          refl::init<TestCxxClassDerivedDerived, int64_t, int32_t, double, float, String, bool>)
      .def_rw("v_str", &TestCxxClassDerivedDerived::v_str)
      .def_rw("v_bool", &TestCxxClassDerivedDerived::v_bool);

  refl::ObjectDef<TestCxxInitSubsetObj>()
      .def_static("__ffi_init__", refl::init<TestCxxInitSubsetObj, int64_t, String>)
      .def_rw("required_field", &TestCxxInitSubsetObj::required_field)
      .def_rw("optional_field", &TestCxxInitSubsetObj::optional_field)
      .def_rw("note", &TestCxxInitSubsetObj::note);

  refl::ObjectDef<TestUnregisteredBaseObject>()
      .def_ro("v1", &TestUnregisteredBaseObject::v1)
      .def_static("__ffi_init__", refl::init<TestUnregisteredBaseObject, int64_t>,
                  "Constructor of TestUnregisteredBaseObject")
      .def("get_v1_plus_one", &TestUnregisteredBaseObject::GetV1PlusOne,
           "Get (v1 + 1) from TestUnregisteredBaseObject");

  refl::ObjectDef<TestUnregisteredObject>()
      .def_ro("v1", &TestUnregisteredObject::v1)
      .def_ro("v2", &TestUnregisteredObject::v2)
      .def_static("__ffi_init__", refl::init<TestUnregisteredObject, int64_t, int64_t>,
                  "Constructor of TestUnregisteredObject")
      .def("get_v2_plus_two", &TestUnregisteredObject::GetV2PlusTwo,
           "Get (v2 + 2) from TestUnregisteredObject");

  refl::GlobalDef()
      .def("testing.test_raise_error", TestRaiseError)
      .def("testing.add_one", [](int x) { return x + 1; })
      .def_packed("testing.nop", [](PackedArgs args, Any* ret) {})
      .def_packed("testing.echo", [](PackedArgs args, Any* ret) { *ret = args[0]; })
      .def_packed("testing.apply", TestApply)
      .def("testing.run_check_signal",
           [](int nsec) {
             for (int i = 0; i < nsec; ++i) {
               if (TVMFFIEnvCheckSignals() != 0) {
                 throw ffi::EnvErrorAlreadySet();
               }
               std::this_thread::sleep_for(std::chrono::seconds(1));
             }
             std::cout << "Function finished without catching signal" << std::endl;
           })
      .def("testing.object_use_count", [](const Object* obj) { return obj->use_count(); })
      .def("testing.make_unregistered_object",
           []() { return ObjectRef(make_object<TestUnregisteredObject>(41, 42)); });
}

}  // namespace ffi
}  // namespace tvm
