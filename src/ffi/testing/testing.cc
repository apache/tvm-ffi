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
#include <dlpack/dlpack.h>
#include <tvm/ffi/any.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/container/variant.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/optional.h>
#include <tvm/ffi/reflection/accessor.h>
#include <tvm/ffi/reflection/registry.h>

#include <chrono>
#include <iostream>
#include <thread>
#include <utility>

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

  int64_t Sum() const { return get()->a + get()->b; }

  // Required: define object reference methods
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TestIntPair, tvm::ffi::ObjectRef, TestIntPairObj);
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TestIntPairObj>()
      .def(refl::init<int64_t, int64_t>())
      .def_ro("a", &TestIntPairObj::a, "Field `a`")
      .def_ro("b", &TestIntPairObj::b, "Field `b`")
      .def("sum", &TestIntPair::Sum, "Method to compute sum of a and b");
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
      : TestCxxClassDerived(v_i64, v_i32, v_f64, v_f32), v_str(std::move(v_str)), v_bool(v_bool) {}

  TVM_FFI_DECLARE_OBJECT_INFO("testing.TestCxxClassDerivedDerived", TestCxxClassDerivedDerived,
                              TestCxxClassDerived);
};

class TestCxxInitSubsetObj : public Object {
 public:
  int64_t required_field;
  int64_t optional_field = -1;
  String note;

  explicit TestCxxInitSubsetObj(int64_t value, String note)
      : required_field(value), note(std::move(note)) {}

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

// NOLINTNEXTLINE(performance-unnecessary-value-param)
TVM_FFI_NO_INLINE void TestRaiseError(String kind, String msg) {
  // keep name and no liner for testing backtrace
  throw ffi::Error(kind, msg, TVMFFIBacktrace(__FILE__, __LINE__, TVM_FFI_FUNC_SIG, 0));
}

TVM_FFI_NO_INLINE void TestApply(PackedArgs args, Any* ret) {
  // keep name and no liner for testing backtrace
  auto f = args[0].cast<Function>();
  f.CallPacked(args.Slice(1), ret);
}

// NOLINTNEXTLINE(bugprone-reserved-identifier)
int __add_one_c_symbol(void*, const TVMFFIAny* args, int32_t num_args, TVMFFIAny* ret) {
  TVM_FFI_SAFE_CALL_BEGIN();
  int x = reinterpret_cast<const AnyView*>(args)[0].cast<int>();
  reinterpret_cast<Any*>(ret)[0] = x + 1;
  TVM_FFI_SAFE_CALL_END();
}

void _mlir_add_one_c_symbol(void** packed_args) {
  void* handle = *reinterpret_cast<void**>(packed_args[0]);
  const TVMFFIAny* args = *reinterpret_cast<const TVMFFIAny**>(packed_args[1]);
  int32_t num_args = *reinterpret_cast<int32_t*>(packed_args[2]);
  TVMFFIAny* rv = *reinterpret_cast<TVMFFIAny**>(packed_args[3]);
  int* ret_code = reinterpret_cast<int*>(packed_args[4]);
  *ret_code = __add_one_c_symbol(handle, args, num_args, rv);
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
      .def(refl::init<int64_t, int32_t>())
      .def_rw("v_i64", &TestCxxClassBase::v_i64)
      .def_rw("v_i32", &TestCxxClassBase::v_i32);

  refl::ObjectDef<TestCxxClassDerived>()
      .def(refl::init<int64_t, int32_t, double, float>())
      .def_rw("v_f64", &TestCxxClassDerived::v_f64)
      .def_rw("v_f32", &TestCxxClassDerived::v_f32);

  refl::ObjectDef<TestCxxClassDerivedDerived>()
      .def(refl::init<int64_t, int32_t, double, float, String, bool>())
      .def_rw("v_str", &TestCxxClassDerivedDerived::v_str)
      .def_rw("v_bool", &TestCxxClassDerivedDerived::v_bool);

  refl::ObjectDef<TestCxxInitSubsetObj>()
      .def(refl::init<int64_t, String>())
      .def_rw("required_field", &TestCxxInitSubsetObj::required_field)
      .def_rw("optional_field", &TestCxxInitSubsetObj::optional_field)
      .def_rw("note", &TestCxxInitSubsetObj::note);

  refl::ObjectDef<TestUnregisteredBaseObject>()
      .def(refl::init<int64_t>(), "Constructor of TestUnregisteredBaseObject")
      .def_ro("v1", &TestUnregisteredBaseObject::v1)
      .def("get_v1_plus_one", &TestUnregisteredBaseObject::GetV1PlusOne,
           "Get (v1 + 1) from TestUnregisteredBaseObject");

  refl::ObjectDef<TestUnregisteredObject>()
      .def(refl::init<int64_t, int64_t>(), "Constructor of TestUnregisteredObject")
      .def_ro("v1", &TestUnregisteredObject::v1)
      .def_ro("v2", &TestUnregisteredObject::v2)
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
           []() { return ObjectRef(make_object<TestUnregisteredObject>(41, 42)); })
      .def("testing.get_add_one_c_symbol",
           []() {
             TVMFFISafeCallType symbol = __add_one_c_symbol;
             // NOLINTNEXTLINE(bugprone-casting-through-void)
             return reinterpret_cast<int64_t>(reinterpret_cast<void*>(symbol));
           })
      .def("testing.get_mlir_add_one_c_symbol",
           []() {
             // NOLINTNEXTLINE(bugprone-casting-through-void)
             return reinterpret_cast<int64_t>(reinterpret_cast<void*>(_mlir_add_one_c_symbol));
           })
      .def_method("testing.TestIntPairSum", &TestIntPair::Sum, "Get sum of the pair");
}

}  // namespace ffi
}  // namespace tvm

// -----------------------------------------------------------------------------
// Additional comprehensive schema coverage
// -----------------------------------------------------------------------------
namespace tvm {
namespace ffi {

// A class with a wide variety of field types and method signatures
class SchemaAllTypesObj : public Object {
 public:
  // POD and builtin types
  bool v_bool{true};
  int64_t v_int;
  double v_float;
  DLDevice v_device;
  DLDataType v_dtype;

  // Atomic object types
  String v_string;
  Bytes v_bytes;

  // Containers and combinations
  Optional<int64_t> v_opt_int;
  Optional<String> v_opt_str;
  Array<int64_t> v_arr_int;
  Array<String> v_arr_str;
  Map<String, int64_t> v_map_str_int;
  Map<String, Array<int64_t>> v_map_str_arr_int;
  Variant<String, Array<int64_t>, Map<String, int64_t>> v_variant;
  Optional<Array<Variant<int64_t, String>>> v_opt_arr_variant;

  // Constructor used by refl::init in make_with
  SchemaAllTypesObj(int64_t vi, double vf, String s)  // NOLINT(*): explicit not necessary here
      : v_int(vi),
        v_float(vf),
        v_device(TVMFFIDLDeviceFromIntPair(kDLCPU, 0)),
        v_dtype(StringToDLDataType("float32")),
        v_string(std::move(s)),
        v_variant(String("v")) {}

  // Some methods to exercise RegisterMethod
  int64_t AddInt(int64_t x) const { return v_int + x; }
  Array<int64_t> AppendInt(Array<int64_t> xs, int64_t y) const {
    xs.push_back(y);
    return xs;
  }
  Optional<String> MaybeConcat(Optional<String> a, Optional<String> b) const {
    if (a.has_value() && b.has_value()) return String(a.value() + b.value());
    if (a.has_value()) return a;
    if (b.has_value()) return b;
    return Optional<String>(std::nullopt);
  }
  Map<String, Array<int64_t>> MergeMap(Map<String, Array<int64_t>> lhs,
                                       // NOLINTNEXTLINE(performance-unnecessary-value-param)
                                       Map<String, Array<int64_t>> rhs) const {
    for (const auto& kv : rhs) {
      if (!lhs.count(kv.first)) {
        lhs.Set(kv.first, kv.second);
      } else {
        Array<int64_t> arr = lhs[kv.first];
        for (const auto& v : kv.second) arr.push_back(v);
        lhs.Set(kv.first, arr);
      }
    }
    return lhs;
  }

  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("testing.SchemaAllTypes", SchemaAllTypesObj, Object);
};

class SchemaAllTypes : public ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(SchemaAllTypes, ObjectRef, SchemaAllTypesObj);
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  // Register fields of various types (RegisterField usage)
  refl::ObjectDef<SchemaAllTypesObj>()
      .def_rw("v_bool", &SchemaAllTypesObj::v_bool, "bool field",
              refl::Metadata{{"bool_attr", true},  //
                             {"int_attr", 1},      //
                             {"str_attr", "hello"}})
      .def_rw("v_int", &SchemaAllTypesObj::v_int, refl::DefaultValue(0), "int field")
      .def_rw("v_float", &SchemaAllTypesObj::v_float, refl::DefaultValue(0.0), "float field")
      .def_rw("v_device", &SchemaAllTypesObj::v_device, "device field")
      .def_rw("v_dtype", &SchemaAllTypesObj::v_dtype, "dtype field")
      .def_rw("v_string", &SchemaAllTypesObj::v_string, refl::DefaultValue("s"), "string field")
      .def_rw("v_bytes", &SchemaAllTypesObj::v_bytes, "bytes field")
      .def_rw("v_opt_int", &SchemaAllTypesObj::v_opt_int, "optional int")
      .def_rw("v_opt_str", &SchemaAllTypesObj::v_opt_str, "optional str")
      .def_rw("v_arr_int", &SchemaAllTypesObj::v_arr_int, "array<int>")
      .def_rw("v_arr_str", &SchemaAllTypesObj::v_arr_str, "array<str>")
      .def_rw("v_map_str_int", &SchemaAllTypesObj::v_map_str_int, "map<str,int>")
      .def_rw("v_map_str_arr_int", &SchemaAllTypesObj::v_map_str_arr_int, "map<str,array<int>>")
      .def_rw("v_variant", &SchemaAllTypesObj::v_variant, "variant<str,array<int>,map<str,int>>")
      .def_rw("v_opt_arr_variant", &SchemaAllTypesObj::v_opt_arr_variant,
              "optional<array<variant<int,str>>>")
      // Register methods (RegisterMethod usage)
      .def("add_int", &SchemaAllTypesObj::AddInt, "add int method",
           refl::Metadata{{"bool_attr", true},  //
                          {"int_attr", 1},      //
                          {"str_attr", "hello"}})
      .def("append_int", &SchemaAllTypesObj::AppendInt, "append int to array")
      .def("maybe_concat", &SchemaAllTypesObj::MaybeConcat, "optional concat")
      .def("merge_map", &SchemaAllTypesObj::MergeMap, "merge maps")
      // Register static creator (also a static method)
      .def_static(
          "make_with",
          [](int64_t vi, double vf, String s) {
            return SchemaAllTypes(make_object<SchemaAllTypesObj>(vi, vf, std::move(s)));
          },
          "Constructor with subset of fields");

  // Global typed functions to exercise RegisterFunc with various schemas
  refl::GlobalDef()
      .def(
          "testing.schema_id_int", [](int64_t x) { return x; },
          refl::Metadata{{"bool_attr", true},  //
                         {"int_attr", 1},      //
                         {"str_attr", "hello"}})
      .def("testing.schema_id_float", [](double x) { return x; })
      .def("testing.schema_id_bool", [](bool x) { return x; })
      .def("testing.schema_id_device", [](DLDevice d) { return d; })
      .def("testing.schema_id_dtype", [](DLDataType dt) { return dt; })
      .def("testing.schema_id_string", [](String s) { return s; })
      .def("testing.schema_id_bytes", [](Bytes b) { return b; })
      .def("testing.schema_id_func", [](Function f) -> Function { return f; })
      .def("testing.schema_id_func_typed",
           [](TypedFunction<void(int64_t, float, Function)> f)
               -> TypedFunction<void(int64_t, float, Function)> { return f; })
      .def("testing.schema_id_any", [](Any a) { return a; })
      .def("testing.schema_id_object", [](ObjectRef o) { return o; })
      .def("testing.schema_id_dltensor", [](DLTensor* t) { return t; })
      .def("testing.schema_id_tensor", [](Tensor t) { return t; })
      .def("testing.schema_tensor_view_input", [](TensorView t) -> void {})
      .def("testing.schema_id_opt_int", [](Optional<int64_t> o) { return o; })
      .def("testing.schema_id_opt_str", [](Optional<String> o) { return o; })
      .def("testing.schema_id_opt_obj", [](Optional<ObjectRef> o) { return o; })
      .def("testing.schema_id_arr_int", [](Array<int64_t> arr) { return arr; })
      .def("testing.schema_id_arr_str", [](Array<String> arr) { return arr; })
      .def("testing.schema_id_arr_obj", [](Array<ObjectRef> arr) { return arr; })
      .def("testing.schema_id_arr", [](const ArrayObj* arr) { return arr; })
      .def("testing.schema_id_map_str_int", [](Map<String, int64_t> m) { return m; })
      .def("testing.schema_id_map_str_str", [](Map<String, String> m) { return m; })
      .def("testing.schema_id_map_str_obj", [](Map<String, ObjectRef> m) { return m; })
      .def("testing.schema_id_map", [](const MapObj* m) { return m; })
      .def("testing.schema_id_variant_int_str", [](Variant<int64_t, String> v) { return v; })
      .def_packed("testing.schema_packed", [](PackedArgs args, Any* ret) {})
      .def("testing.schema_arr_map_opt",
           // NOLINTNEXTLINE(performance-unnecessary-value-param)
           [](Array<Optional<int64_t>> arr, Map<String, Array<int64_t>> mp,
              // NOLINTNEXTLINE(performance-unnecessary-value-param)
              Optional<String> os) -> Map<String, Array<int64_t>> {
             // no-op combine
             if (os.has_value()) {
               Array<int64_t> extra;
               for (const auto& i : arr) {
                 if (i.has_value()) extra.push_back(i.value());
               }
               mp.Set(os.value(), extra);
             }
             return mp;
           })
      .def(
          "testing.schema_variant_mix",
          [](Variant<int64_t, String, Array<int64_t>> v) { return v; }, "variant passthrough")
      .def("testing.schema_no_args", []() { return 1; })
      .def("testing.schema_no_return", [](int64_t x) {})
      .def("testing.schema_no_args_no_return", []() {});
  TVMFFIEnvModRegisterSystemLibSymbol("__tvm_ffi_testing.add_one",
                                      reinterpret_cast<void*>(__add_one_c_symbol));
}

}  // namespace ffi
}  // namespace tvm

extern "C" TVM_FFI_DLL_EXPORT int TVMFFITestingDummyTarget() { return 0; }
