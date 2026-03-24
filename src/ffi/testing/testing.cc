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
// NOTE: TVM_FFI_DLL_EXPORT_INCLUDE_METADATA=1 is set via CMake target_compile_definitions
#include <dlpack/dlpack.h>
#include <tvm/ffi/any.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/dict.h>
#include <tvm/ffi/container/list.h>
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
  refl::TypeAttrDef<TestIntPairObj>().def(
      refl::type_attr::kConvert, &refl::details::FFIConvertFromAnyViewToObjectRef<TestIntPair>);
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
  int64_t v_i64 = 0;
  int32_t v_i32 = 0;

  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("testing.TestCxxClassBase", TestCxxClassBase, Object);
};

class TestCxxClassDerived : public TestCxxClassBase {
 public:
  double v_f64 = 0.0;
  float v_f32 = 0.0f;

  TVM_FFI_DECLARE_OBJECT_INFO("testing.TestCxxClassDerived", TestCxxClassDerived, TestCxxClassBase);
};

class TestCxxClassDerivedDerived : public TestCxxClassDerived {
 public:
  String v_str;
  bool v_bool = false;

  TVM_FFI_DECLARE_OBJECT_INFO("testing.TestCxxClassDerivedDerived", TestCxxClassDerivedDerived,
                              TestCxxClassDerived);
};

class TestCxxInitSubsetObj : public Object {
 public:
  int64_t required_field = 0;
  int64_t optional_field = -1;
  String note;

  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("testing.TestCxxInitSubset", TestCxxInitSubsetObj, Object);
};

class TestCxxKwOnly : public Object {
 public:
  int64_t x = 0;
  int64_t y = 0;
  int64_t z = 0;
  int64_t w = 0;

  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("testing.TestCxxKwOnly", TestCxxKwOnly, Object);
};

// Test: auto-generated __ffi_init__ with init(false) and KwOnly(true) per-field.
// No explicit refl::init<>() — the auto-init is generated in ObjectDef's destructor.
class TestCxxAutoInitObj : public Object {
 public:
  int64_t a;  ///< init=True, positional (default)
  int64_t b;  ///< init=False, has default
  int64_t c;  ///< init=True, kw_only
  int64_t d;  ///< init=True, positional, has default

  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("testing.TestCxxAutoInit", TestCxxAutoInitObj, Object);
};

// Test: auto-generated init with all fields init=True (no Init/KwOnly traits).
class TestCxxAutoInitSimpleObj : public Object {
 public:
  int64_t x;
  int64_t y;

  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("testing.TestCxxAutoInitSimple", TestCxxAutoInitSimpleObj, Object);
};

// Test: auto-generated init with all fields excluded from init.
class TestCxxAutoInitAllInitOffObj : public Object {
 public:
  int64_t x = 7;     ///< init=False, has reflection default
  int64_t y = 0;     ///< init=False, has reflection default
  int64_t z = 1234;  ///< init=False, no reflection default (creator default is kept)

  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("testing.TestCxxAutoInitAllInitOff", TestCxxAutoInitAllInitOffObj,
                              Object);
};

// Test: mixed positional + kw_only + defaults + init=False field.
class TestCxxAutoInitKwOnlyDefaultsObj : public Object {
 public:
  int64_t p_required;  ///< init=True, positional, required
  int64_t p_default;   ///< init=True, positional, default=11
  int64_t k_required;  ///< init=True, kw_only, required
  int64_t k_default;   ///< init=True, kw_only, default=22
  int64_t hidden;      ///< init=False, default=33

  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("testing.TestCxxAutoInitKwOnlyDefaults",
                              TestCxxAutoInitKwOnlyDefaultsObj, Object);
};

// Test: inheritance + auto-generated init.
class TestCxxAutoInitParentObj : public Object {
 public:
  int64_t parent_required;
  int64_t parent_default;

  static constexpr bool _type_mutable = true;
  static constexpr uint32_t _type_child_slots = 1;
  TVM_FFI_DECLARE_OBJECT_INFO("testing.TestCxxAutoInitParent", TestCxxAutoInitParentObj, Object);
};

class TestCxxAutoInitChildObj : public TestCxxAutoInitParentObj {
 public:
  int64_t child_required;
  int64_t child_kw_only;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("testing.TestCxxAutoInitChild", TestCxxAutoInitChildObj,
                                    TestCxxAutoInitParentObj);
};

// Test: init(false) at class level suppresses auto-generated __ffi_init__.
class TestCxxNoAutoInitObj : public Object {
 public:
  int64_t x;
  int64_t y;

  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("testing.TestCxxNoAutoInit", TestCxxNoAutoInitObj, Object);
};

class TestDeepCopyEdgesObj : public Object {
 public:
  Any v_any;
  ObjectRef v_obj;

  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("testing.TestDeepCopyEdges", TestDeepCopyEdgesObj, Object);
};

class TestNonCopyable : public Object {
 public:
  int64_t value;

  explicit TestNonCopyable(int64_t value) : value(value) {}
  TestNonCopyable(const TestNonCopyable&) = delete;
  TestNonCopyable& operator=(const TestNonCopyable&) = delete;

  TVM_FFI_DECLARE_OBJECT_INFO("testing.TestNonCopyable", TestNonCopyable, Object);
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

class TestCompareObj : public Object {
 public:
  int64_t key;
  String name;
  int64_t ignored_field;

  TestCompareObj() = default;
  TestCompareObj(int64_t key, String name, int64_t ignored_field)
      : key(key), name(std::move(name)), ignored_field(ignored_field) {}

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("testing.TestCompare", TestCompareObj, Object);
};

class TestCompare : public ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TestCompare, ObjectRef, TestCompareObj);
};

class TestHashObj : public Object {
 public:
  int64_t key;
  String name;
  int64_t hash_ignored;

  TestHashObj() = default;
  TestHashObj(int64_t key, String name, int64_t hash_ignored)
      : key(key), name(std::move(name)), hash_ignored(hash_ignored) {}

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("testing.TestHash", TestHashObj, Object);
};

class TestHash : public ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TestHash, ObjectRef, TestHashObj);
};

class TestCustomHashObj : public Object {
 public:
  int64_t key;
  String label;

  TestCustomHashObj() = default;
  TestCustomHashObj(int64_t key, String label) : key(key), label(std::move(label)) {}

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("testing.TestCustomHash", TestCustomHashObj, Object);
};

class TestCustomHash : public ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TestCustomHash, ObjectRef, TestCustomHashObj);
};

class TestCustomCompareObj : public Object {
 public:
  int64_t key;
  String label;

  TestCustomCompareObj() = default;
  TestCustomCompareObj(int64_t key, String label) : key(key), label(std::move(label)) {}

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("testing.TestCustomCompare", TestCustomCompareObj, Object);
};

class TestCustomCompare : public ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TestCustomCompare, ObjectRef, TestCustomCompareObj);
};

// Test object with __ffi_eq__ but deliberately no __ffi_hash__.
// Used to verify the RecursiveHash guard that rejects eq-without-hash types.
class TestEqWithoutHashObj : public Object {
 public:
  int64_t key;
  String label;

  TestEqWithoutHashObj() = default;
  TestEqWithoutHashObj(int64_t key, String label) : key(key), label(std::move(label)) {}

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("testing.TestEqWithoutHash", TestEqWithoutHashObj, Object);
};

class TestEqWithoutHash : public ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TestEqWithoutHash, ObjectRef, TestEqWithoutHashObj);
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
      .def_rw("v_i64", &TestObjectBase::v_i64, refl::default_value(10), "i64 field")
      .def_ro("v_f64", &TestObjectBase::v_f64, refl::default_value(10.0))
      .def_rw("v_str", &TestObjectBase::v_str, refl::default_value("hello"))
      .def("add_i64", &TestObjectBase::AddI64, "add_i64 method");

  refl::ObjectDef<TestObjectDerived>()
      .def_rw("v_map", &TestObjectDerived::v_map)
      .def_rw("v_array", &TestObjectDerived::v_array);

  refl::ObjectDef<TestCxxClassBase>()
      .def_rw("v_i64", &TestCxxClassBase::v_i64, refl::repr(false))
      .def_rw("v_i32", &TestCxxClassBase::v_i32, refl::repr(false));

  refl::ObjectDef<TestCxxClassDerived>()
      .def_rw("v_f64", &TestCxxClassDerived::v_f64)
      .def_rw("v_f32", &TestCxxClassDerived::v_f32, refl::default_value(float{8.0f}));

  refl::ObjectDef<TestCxxClassDerivedDerived>()
      .def_rw("v_str", &TestCxxClassDerivedDerived::v_str, refl::default_value(String("default")))
      .def_rw("v_bool", &TestCxxClassDerivedDerived::v_bool);

  refl::ObjectDef<TestCxxInitSubsetObj>()
      .def_rw("required_field", &TestCxxInitSubsetObj::required_field)
      .def_rw("optional_field", &TestCxxInitSubsetObj::optional_field, refl::init(false),
              refl::default_value(int64_t{-1}))
      .def_rw("note", &TestCxxInitSubsetObj::note, refl::init(false),
              refl::default_value(String("default")));

  refl::ObjectDef<TestCxxKwOnly>()
      .def_rw("x", &TestCxxKwOnly::x, refl::kw_only(true))
      .def_rw("y", &TestCxxKwOnly::y, refl::kw_only(true))
      .def_rw("z", &TestCxxKwOnly::z, refl::kw_only(true))
      .def_rw("w", &TestCxxKwOnly::w, refl::kw_only(true), refl::default_value(int64_t{100}));

  // No refl::init<>() — auto-generates __ffi_init__ in ObjectDef destructor.
  refl::ObjectDef<TestCxxAutoInitObj>()
      .def_rw("a", &TestCxxAutoInitObj::a)
      .def_rw("b", &TestCxxAutoInitObj::b, refl::init(false), refl::default_value(int64_t{42}))
      .def_rw("c", &TestCxxAutoInitObj::c, refl::kw_only(true))
      .def_rw("d", &TestCxxAutoInitObj::d, refl::default_value(int64_t{99}));

  refl::ObjectDef<TestCxxAutoInitSimpleObj>()
      .def_rw("x", &TestCxxAutoInitSimpleObj::x)
      .def_rw("y", &TestCxxAutoInitSimpleObj::y);

  refl::ObjectDef<TestCxxAutoInitAllInitOffObj>()
      .def_rw("x", &TestCxxAutoInitAllInitOffObj::x, refl::init(false),
              refl::default_value(int64_t{7}))
      .def_rw("y", &TestCxxAutoInitAllInitOffObj::y, refl::init(false),
              refl::default_value(int64_t{9}))
      .def_rw("z", &TestCxxAutoInitAllInitOffObj::z, refl::init(false));

  refl::ObjectDef<TestCxxAutoInitKwOnlyDefaultsObj>()
      .def_rw("p_required", &TestCxxAutoInitKwOnlyDefaultsObj::p_required)
      .def_rw("p_default", &TestCxxAutoInitKwOnlyDefaultsObj::p_default,
              refl::default_value(int64_t{11}))
      .def_rw("k_required", &TestCxxAutoInitKwOnlyDefaultsObj::k_required, refl::kw_only(true))
      .def_rw("k_default", &TestCxxAutoInitKwOnlyDefaultsObj::k_default, refl::kw_only(true),
              refl::default_value(int64_t{22}))
      .def_rw("hidden", &TestCxxAutoInitKwOnlyDefaultsObj::hidden, refl::init(false),
              refl::default_value(int64_t{33}));

  refl::ObjectDef<TestCxxAutoInitParentObj>()
      .def_rw("parent_required", &TestCxxAutoInitParentObj::parent_required)
      .def_rw("parent_default", &TestCxxAutoInitParentObj::parent_default,
              refl::default_value(int64_t{5}));

  refl::ObjectDef<TestCxxAutoInitChildObj>()
      .def_rw("child_required", &TestCxxAutoInitChildObj::child_required)
      .def_rw("child_kw_only", &TestCxxAutoInitChildObj::child_kw_only, refl::kw_only(true));

  // init(false) at class level: has fields, has creator, but no __ffi_init__.
  refl::ObjectDef<TestCxxNoAutoInitObj>(refl::init(false))
      .def_rw("x", &TestCxxNoAutoInitObj::x)
      .def_rw("y", &TestCxxNoAutoInitObj::y);

  refl::ObjectDef<TestDeepCopyEdgesObj>()
      .def_rw("v_any", &TestDeepCopyEdgesObj::v_any)
      .def_rw("v_obj", &TestDeepCopyEdgesObj::v_obj);

  refl::ObjectDef<TestNonCopyable>()
      .def(refl::init<int64_t>())
      .def_ro("value", &TestNonCopyable::value);

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

  refl::ObjectDef<TestCompareObj>()
      .def(refl::init<int64_t, String, int64_t>())
      .def_ro("key", &TestCompareObj::key)
      .def_ro("name", &TestCompareObj::name)
      .def_ro("ignored_field", &TestCompareObj::ignored_field, refl::compare(false));

  refl::ObjectDef<TestHashObj>()
      .def(refl::init<int64_t, String, int64_t>())
      .def_ro("key", &TestHashObj::key)
      .def_ro("name", &TestHashObj::name)
      .def_ro("hash_ignored", &TestHashObj::hash_ignored, refl::hash(false));

  refl::ObjectDef<TestCustomHashObj>()
      .def(refl::init<int64_t, String>())
      .def_ro("key", &TestCustomHashObj::key)
      .def_ro("label", &TestCustomHashObj::label);

  refl::ObjectDef<TestCustomCompareObj>()
      .def(refl::init<int64_t, String>())
      .def_ro("key", &TestCustomCompareObj::key)
      .def_ro("label", &TestCustomCompareObj::label);

  refl::ObjectDef<TestEqWithoutHashObj>()
      .def(refl::init<int64_t, String>())
      .def_ro("key", &TestEqWithoutHashObj::key)
      .def_ro("label", &TestEqWithoutHashObj::label);

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
      .def("testing.optional_tensor_view_has_value",
           [](const Optional<TensorView>& t) { return t.has_value(); })
      .def_method("testing.TestIntPairSum", &TestIntPair::Sum, "Get sum of the pair");
}

}  // namespace ffi
}  // namespace tvm

// -----------------------------------------------------------------------------
// Additional comprehensive schema coverage
// -----------------------------------------------------------------------------
namespace tvm {
namespace ffi {

// -----------------------------------------------------------------------------
// Implementation functions for schema testing
// -----------------------------------------------------------------------------
namespace schema_test_impl {

// Simple types
int64_t schema_id_int(int64_t x) { return x; }
double schema_id_float(double x) { return x; }
bool schema_id_bool(bool x) { return x; }
DLDevice schema_id_device(DLDevice d) { return d; }
DLDataType schema_id_dtype(DLDataType dt) { return dt; }
String schema_id_string(String s) { return s; }
Bytes schema_id_bytes(Bytes b) { return b; }
Function schema_id_func(Function f) { return f; }
TypedFunction<void(int64_t, float, Function)> schema_id_func_typed(
    TypedFunction<void(int64_t, float, Function)> f) {
  return f;
}
Any schema_id_any(Any a) { return a; }
ObjectRef schema_id_object(ObjectRef o) { return o; }
DLTensor* schema_id_dltensor(DLTensor* t) { return t; }
Tensor schema_id_tensor(Tensor t) { return t; }
void schema_tensor_view_input(TensorView t) {}

// Optional types
Optional<int64_t> schema_id_opt_int(Optional<int64_t> o) { return o; }
Optional<String> schema_id_opt_str(Optional<String> o) { return o; }
Optional<ObjectRef> schema_id_opt_obj(Optional<ObjectRef> o) { return o; }

// Array types
Array<int64_t> schema_id_arr_int(Array<int64_t> arr) { return arr; }
Array<String> schema_id_arr_str(Array<String> arr) { return arr; }
Array<ObjectRef> schema_id_arr_obj(Array<ObjectRef> arr) { return arr; }
const ArrayObj* schema_id_arr(const ArrayObj* arr) { return arr; }

// Map types
Map<String, int64_t> schema_id_map_str_int(Map<String, int64_t> m) { return m; }
Map<String, String> schema_id_map_str_str(Map<String, String> m) { return m; }
Map<String, ObjectRef> schema_id_map_str_obj(Map<String, ObjectRef> m) { return m; }
const MapObj* schema_id_map(const MapObj* m) { return m; }

// Variant types
Variant<int64_t, String> schema_id_variant_int_str(Variant<int64_t, String> v) { return v; }
Variant<int64_t, String, Array<int64_t>> schema_variant_mix(
    Variant<int64_t, String, Array<int64_t>> v) {
  return v;
}

// List types
List<int64_t> schema_id_list_int(List<int64_t> lst) { return lst; }
List<String> schema_id_list_str(List<String> lst) { return lst; }
List<ObjectRef> schema_id_list_obj(List<ObjectRef> lst) { return lst; }

// Dict types
Dict<String, int64_t> schema_id_dict_str_int(Dict<String, int64_t> d) { return d; }
Dict<String, String> schema_id_dict_str_str(Dict<String, String> d) { return d; }

// Complex nested types
Map<String, Array<int64_t>> schema_arr_map_opt(const Array<Optional<int64_t>>& arr,
                                               Map<String, Array<int64_t>> mp,
                                               const Optional<String>& os) {
  // no-op combine
  if (os.has_value()) {
    Array<int64_t> extra;
    for (const auto& i : arr) {
      if (i.has_value()) extra.push_back(i.value());
    }
    mp.Set(os.value(), extra);
  }
  return mp;
}

// Edge cases
int64_t schema_no_args() { return 1; }
void schema_no_return(int64_t x) {}
void schema_no_args_no_return() {}

// Member function pattern
int64_t test_int_pair_sum_wrapper(const TestIntPair& target) { return target.Sum(); }

// Documentation export
int64_t test_add_with_docstring(int64_t a, int64_t b) { return a + b; }

}  // namespace schema_test_impl

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
      .def_rw("v_int", &SchemaAllTypesObj::v_int, refl::default_value(0), "int field")
      .def_rw("v_float", &SchemaAllTypesObj::v_float, refl::default_value(0.0), "float field")
      .def_rw("v_device", &SchemaAllTypesObj::v_device, "device field")
      .def_rw("v_dtype", &SchemaAllTypesObj::v_dtype, "dtype field")
      .def_rw("v_string", &SchemaAllTypesObj::v_string, refl::default_value("s"), "string field")
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
      .def("testing.schema_id_int", schema_test_impl::schema_id_int,
           refl::Metadata{{"bool_attr", true},  //
                          {"int_attr", 1},      //
                          {"str_attr", "hello"}})
      .def("testing.schema_id_float", schema_test_impl::schema_id_float)
      .def("testing.schema_id_bool", schema_test_impl::schema_id_bool)
      .def("testing.schema_id_device", schema_test_impl::schema_id_device)
      .def("testing.schema_id_dtype", schema_test_impl::schema_id_dtype)
      .def("testing.schema_id_string", schema_test_impl::schema_id_string)
      .def("testing.schema_id_bytes", schema_test_impl::schema_id_bytes)
      .def("testing.schema_id_func", schema_test_impl::schema_id_func)
      .def("testing.schema_id_func_typed", schema_test_impl::schema_id_func_typed)
      .def("testing.schema_id_any", schema_test_impl::schema_id_any)
      .def("testing.schema_id_object", schema_test_impl::schema_id_object)
      .def("testing.schema_id_dltensor", schema_test_impl::schema_id_dltensor)
      .def("testing.schema_id_tensor", schema_test_impl::schema_id_tensor)
      .def("testing.schema_tensor_view_input", schema_test_impl::schema_tensor_view_input)
      .def("testing.schema_id_opt_int", schema_test_impl::schema_id_opt_int)
      .def("testing.schema_id_opt_str", schema_test_impl::schema_id_opt_str)
      .def("testing.schema_id_opt_obj", schema_test_impl::schema_id_opt_obj)
      .def("testing.schema_id_arr_int", schema_test_impl::schema_id_arr_int)
      .def("testing.schema_id_arr_str", schema_test_impl::schema_id_arr_str)
      .def("testing.schema_id_arr_obj", schema_test_impl::schema_id_arr_obj)
      .def("testing.schema_id_arr", schema_test_impl::schema_id_arr)
      .def("testing.schema_id_list_int", schema_test_impl::schema_id_list_int)
      .def("testing.schema_id_list_str", schema_test_impl::schema_id_list_str)
      .def("testing.schema_id_list_obj", schema_test_impl::schema_id_list_obj)
      .def("testing.schema_id_map_str_int", schema_test_impl::schema_id_map_str_int)
      .def("testing.schema_id_map_str_str", schema_test_impl::schema_id_map_str_str)
      .def("testing.schema_id_map_str_obj", schema_test_impl::schema_id_map_str_obj)
      .def("testing.schema_id_map", schema_test_impl::schema_id_map)
      .def("testing.schema_id_dict_str_int", schema_test_impl::schema_id_dict_str_int)
      .def("testing.schema_id_dict_str_str", schema_test_impl::schema_id_dict_str_str)
      .def("testing.schema_id_variant_int_str", schema_test_impl::schema_id_variant_int_str)
      .def_packed("testing.schema_packed", [](PackedArgs args, Any* ret) {})
      .def("testing.schema_arr_map_opt", schema_test_impl::schema_arr_map_opt)
      .def("testing.schema_variant_mix", schema_test_impl::schema_variant_mix,
           "variant passthrough")
      .def("testing.schema_no_args", schema_test_impl::schema_no_args)
      .def("testing.schema_no_return", schema_test_impl::schema_no_return)
      .def("testing.schema_no_args_no_return", schema_test_impl::schema_no_args_no_return);
  TVMFFIEnvModRegisterSystemLibSymbol("__tvm_ffi_testing.add_one",
                                      reinterpret_cast<void*>(__add_one_c_symbol));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  // Register __ffi_hash__ for TestCustomHash: only hashes `key`, ignores `label`.
  refl::TypeAttrDef<TestCustomHashObj>().def(
      refl::type_attr::kHash, [](const Object* self, const Function& fn_hash) -> int64_t {
        auto* obj = static_cast<const TestCustomHashObj*>(self);
        return fn_hash(AnyView(obj->key)).cast<int64_t>();
      });
  // Register __ffi_hash__ for TestCustomCompare: only hashes `key`, consistent with eq/compare.
  refl::TypeAttrDef<TestCustomCompareObj>().def(
      refl::type_attr::kHash, [](const Object* self, const Function& fn_hash) -> int64_t {
        auto* obj = static_cast<const TestCustomCompareObj*>(self);
        return fn_hash(AnyView(obj->key)).cast<int64_t>();
      });
  // Register __ffi_eq__ for TestCustomCompare: compares only `key`.
  refl::TypeAttrDef<TestCustomCompareObj>().def(
      refl::type_attr::kEq,
      [](const Object* lhs, const Object* rhs, const Function& fn_eq) -> bool {
        auto* a = static_cast<const TestCustomCompareObj*>(lhs);
        auto* b = static_cast<const TestCustomCompareObj*>(rhs);
        return fn_eq(AnyView(a->key), AnyView(b->key)).cast<bool>();
      });
  // Register __ffi_compare__ for TestCustomCompare: three-way ordering on `key`.
  refl::TypeAttrDef<TestCustomCompareObj>().def(
      refl::type_attr::kCompare,
      [](const Object* lhs, const Object* rhs, const Function& fn_cmp) -> int32_t {
        auto* a = static_cast<const TestCustomCompareObj*>(lhs);
        auto* b = static_cast<const TestCustomCompareObj*>(rhs);
        return fn_cmp(AnyView(a->key), AnyView(b->key)).cast<int32_t>();
      });
  // Register __ffi_eq__ for TestEqWithoutHash: deliberately no __ffi_hash__.
  // This exercises the RecursiveHash guard that rejects eq-without-hash types.
  refl::TypeAttrDef<TestEqWithoutHashObj>().def(
      refl::type_attr::kEq,
      [](const Object* lhs, const Object* rhs, const Function& fn_eq) -> bool {
        auto* a = static_cast<const TestEqWithoutHashObj*>(lhs);
        auto* b = static_cast<const TestEqWithoutHashObj*>(rhs);
        return fn_eq(AnyView(a->key), AnyView(b->key)).cast<bool>();
      });
}

}  // namespace ffi
}  // namespace tvm

// -----------------------------------------------------------------------------
// Exported symbols for metadata testing on DLL-exported functions
// -----------------------------------------------------------------------------
// We keep minimal DLL exports here to verify the export mechanism.
TVM_FFI_DLL_EXPORT_TYPED_FUNC(testing_dll_schema_id_int, tvm::ffi::schema_test_impl::schema_id_int)

// Documentation export
TVM_FFI_DLL_EXPORT_TYPED_FUNC(testing_dll_test_add_with_docstring,
                              tvm::ffi::schema_test_impl::test_add_with_docstring);
TVM_FFI_DLL_EXPORT_TYPED_FUNC_DOC(testing_dll_test_add_with_docstring,
                                  R"(Add two integers and return the sum.

Parameters
----------
a : int
    First integer
b : int
    Second integer

Returns
-------
result : int
    Sum of a and b)");

extern "C" TVM_FFI_DLL_EXPORT int TVMFFITestingDummyTarget() { return 0; }
