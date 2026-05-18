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

#ifndef TVM_FFI_TESTING_OBJECT_H_
#define TVM_FFI_TESTING_OBJECT_H_

#include <tvm/ffi/any.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/memory.h>
#include <tvm/ffi/object.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>

namespace tvm {
namespace ffi {
namespace testing {

// We deliberately pad extra
// in the header to test cases
// where the object subclass address
// do not align with the base object address
// not handling properly will cause buffer overflow
class BasePad {
 public:
  int64_t extra[4];
};

class TNumberObj : public BasePad, public Object {
 public:
  // declare as one slot, with float as overflow
  static constexpr uint32_t _type_child_slots = 1;
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO("test.Number", TNumberObj, Object);
};

class TNumber : public ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TNumber, ObjectRef, TNumberObj);
};

class TIntObj : public TNumberObj {
 public:
  int64_t value;

  TIntObj(int64_t value) : value(value) {}
  explicit TIntObj(UnsafeInit) {}

  int64_t GetValue() const { return value; }

  inline static void RegisterReflection();

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("test.Int", TIntObj, TNumberObj);
};

class TInt : public TNumber {
 public:
  explicit TInt(int64_t value) { data_ = make_object<TIntObj>(value); }

  static TInt StaticAdd(TInt lhs, TInt rhs) { return TInt(lhs->value + rhs->value); }

  static int64_t CustomAnyHash(const Any& src) {
    return static_cast<int64_t>(src.cast<TInt>()->value + 1024);
  }

  static bool CustomAnyEqual(const Any& lhs, const Any& rhs) {
    return lhs.cast<TInt>()->value == rhs.cast<TInt>()->value;
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(TInt, TNumber, TIntObj);
};

inline void TIntObj::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TIntObj>()
      .def_ro("value", &TIntObj::value)
      .def_static("static_add", &TInt::StaticAdd, "static add method");
  // define extra type attributes
  refl::TypeAttrDef<TIntObj>()
      .def("test.GetValue", &TIntObj::GetValue)
      .attr("test.size", sizeof(TIntObj))
      .attr(refl::type_attr::kAnyHash, reinterpret_cast<void*>(&TInt::CustomAnyHash))
      .attr(refl::type_attr::kAnyEqual, reinterpret_cast<void*>(&TInt::CustomAnyEqual));
  // custom json serialization
  refl::TypeAttrDef<TIntObj>()
      .def(refl::type_attr::kDataToJson,
           [](const TIntObj* self) -> Map<String, Any> {
             return Map<String, Any>{{"value", self->value}};
           })
      .def(refl::type_attr::kDataFromJson, [](Map<String, Any> json_obj) -> TInt {
        return TInt(json_obj["value"].cast<int64_t>());
      });
}

class TFloatObj : public TNumberObj {
 public:
  double value;

  TFloatObj(double value) : value(value) {}

  double Add(double other) const { return value + other; }

  static void RegisterReflection();

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("test.Float", TFloatObj, TNumberObj);
};

class TFloat : public TNumber {
 public:
  explicit TFloat(double value) { data_ = make_object<TFloatObj>(value); }

  static uint64_t CustomAnyHash(const Any& src) {
    double value = src.cast<TFloat>()->value;
    return static_cast<int64_t>(value * 10 + 2048);
  }

  static bool CustomAnyEqual(const Any& lhs, const Any& rhs) {
    return lhs.cast<TFloat>()->value == rhs.cast<TFloat>()->value;
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TFloat, TNumber, TFloatObj);
};

inline void TFloatObj::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TFloatObj>()
      .def_ro("value", &TFloatObj::value, "float value field", refl::default_value(10.0))
      .def("sub", [](const TFloatObj* self, double other) -> double { return self->value - other; })
      .def("add", &TFloatObj::Add, "add method");
  refl::TypeAttrDef<TFloatObj>()
      .def(refl::type_attr::kAnyHash, &TFloat::CustomAnyHash)
      .def(refl::type_attr::kAnyEqual, &TFloat::CustomAnyEqual);
}

class TPrimExprObj : public Object {
 public:
  std::string dtype;
  double value;

  TPrimExprObj(std::string dtype, double value) : dtype(dtype), value(value) {}

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TPrimExprObj>()
        .def_rw("dtype", &TPrimExprObj::dtype, "dtype field", refl::default_value("float"))
        .def_ro("value", &TPrimExprObj::value, "value field", refl::default_value(0))
        .def("sub", [](TPrimExprObj* self, double other) -> double {
          // this is ok because TPrimExprObj is declared asmutable
          return self->value - other;
        });
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("test.PrimExpr", TPrimExprObj, Object);
};

class TPrimExpr : public ObjectRef {
 public:
  explicit TPrimExpr(std::string dtype, double value) {
    data_ = make_object<TPrimExprObj>(dtype, value);
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TPrimExpr, ObjectRef, TPrimExprObj);
};

class TVarObj : public Object {
 public:
  std::string name;

  TVarObj(std::string name) : name(name) {}
  // need unsafe init constructor for json serialization
  explicit TVarObj(UnsafeInit) {}

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TVarObj>().def_ro("name", &TVarObj::name,
                                      refl::AttachFieldFlag::SEqHashIgnore());
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindFreeVar;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("test.Var", TVarObj, Object);
};

class TVar : public ObjectRef {
 public:
  explicit TVar(std::string name) { data_ = make_object<TVarObj>(name); }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TVar, ObjectRef, TVarObj);
};

class TPairObj : public Object {
 public:
  ObjectRef lhs;
  ObjectRef rhs;

  TPairObj(ObjectRef lhs, ObjectRef rhs) : lhs(std::move(lhs)), rhs(std::move(rhs)) {}
  explicit TPairObj(UnsafeInit) {}

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TPairObj>().def_ro("lhs", &TPairObj::lhs).def_ro("rhs", &TPairObj::rhs);
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("test.Pair", TPairObj, Object);
};

class TPair : public ObjectRef {
 public:
  TPair(ObjectRef lhs, ObjectRef rhs) {
    data_ = make_object<TPairObj>(std::move(lhs), std::move(rhs));
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TPair, ObjectRef, TPairObj);
};

// FreeVar test object that has a sub-field referencing another FreeVar.
// This models the "var with nested vars" case (analogous to a relax::Var
// whose struct_info contains tir shape vars). It is used to exercise the
// difference between SEqHashDefRecursive and SEqHashDefNonRecursive at the
// FFI layer: under recursive semantics the nested ``dep`` var rebinds
// transitively; under non-recursive semantics it is treated as a use of an
// outer-scope binding and equality fails when no such outer binding exists.
class TVarWithDepObj : public Object {
 public:
  std::string name;
  // Optional dependency var; when null, this object behaves like a plain
  // FreeVar with no nested free vars.
  Optional<ObjectRef> dep;

  TVarWithDepObj(std::string name, Optional<ObjectRef> dep)
      : name(std::move(name)), dep(std::move(dep)) {}
  explicit TVarWithDepObj(UnsafeInit) {}

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TVarWithDepObj>()
        .def_ro("name", &TVarWithDepObj::name, refl::AttachFieldFlag::SEqHashIgnore())
        // ``dep`` participates in structural equality without any def flag,
        // so it is a USE position. Whether the FreeVar in ``dep`` may rebind
        // is decided by the def flag on whichever outer field reaches this
        // object.
        .def_ro("dep", &TVarWithDepObj::dep);
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindFreeVar;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("test.VarWithDep", TVarWithDepObj, Object);
};

class TVarWithDep : public ObjectRef {
 public:
  explicit TVarWithDep(std::string name, Optional<ObjectRef> dep = std::nullopt) {
    data_ = make_object<TVarWithDepObj>(std::move(name), std::move(dep));
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TVarWithDep, ObjectRef, TVarWithDepObj);
};

// Holder with one recursive-def field and one non-recursive-def field.
// Used by StructuralEqualHash.NonRecursiveDef tests below.
class TDefHolderObj : public Object {
 public:
  TVarWithDep def_recursive;
  TVarWithDep def_non_recursive;

  TDefHolderObj(TVarWithDep def_recursive, TVarWithDep def_non_recursive)
      : def_recursive(std::move(def_recursive)), def_non_recursive(std::move(def_non_recursive)) {}
  explicit TDefHolderObj(UnsafeInit) {}

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TDefHolderObj>()
        .def_ro("def_recursive", &TDefHolderObj::def_recursive,
                refl::AttachFieldFlag::SEqHashDefRecursive())
        .def_ro("def_non_recursive", &TDefHolderObj::def_non_recursive,
                refl::AttachFieldFlag::SEqHashDefNonRecursive());
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("test.DefHolder", TDefHolderObj, Object);
};

class TDefHolder : public ObjectRef {
 public:
  explicit TDefHolder(TVarWithDep def_recursive, TVarWithDep def_non_recursive) {
    data_ = make_object<TDefHolderObj>(std::move(def_recursive), std::move(def_non_recursive));
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TDefHolder, ObjectRef, TDefHolderObj);
};

class TFuncObj : public Object {
 public:
  Array<TVar> params;
  Array<ObjectRef> body;
  Optional<String> comment;

  // need unsafe init constructor or default constructor for json serialization
  explicit TFuncObj(UnsafeInit) {}
  TFuncObj(Array<TVar> params, Array<ObjectRef> body, Optional<String> comment)
      : params(params), body(body), comment(comment) {}

  static TVMFFIAny StructuralVisit(StructuralVisitorObj* visitor, AnyView value) noexcept {
    const auto* self = value.cast<const TFuncObj*>();

    auto params_interrupt = visitor->WithDefRegionKind(
        kTVMFFIDefRegionKindRecursive, [&]() { return visitor->VisitExpected(self->params); });
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(params_interrupt);

    auto body_interrupt = visitor->VisitExpected(self->body);
    return details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(body_interrupt));
  }

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TFuncObj>()
        .def_ro("params", &TFuncObj::params, refl::AttachFieldFlag::SEqHashDefRecursive())
        .def_ro("body", &TFuncObj::body)
        .def_ro("comment", &TFuncObj::comment, refl::AttachFieldFlag::SEqHashIgnore());
    refl::EnsureTypeAttrColumn(refl::type_attr::kStructuralVisit);
    refl::TypeAttrDef<TFuncObj>().attr(
        refl::type_attr::kStructuralVisit,
        reinterpret_cast<void*>(static_cast<FStructuralVisit>(&TFuncObj::StructuralVisit)));
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("test.Func", TFuncObj, Object);
};

class TFunc : public ObjectRef {
 public:
  explicit TFunc(Array<TVar> params, Array<ObjectRef> body, Optional<String> comment) {
    data_ = make_object<TFuncObj>(params, body, comment);
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TFunc, ObjectRef, TFuncObj);
};

class TCustomFuncObj : public Object {
 public:
  Array<TVar> params;
  Array<ObjectRef> body;
  String comment;

  TCustomFuncObj(Array<TVar> params, Array<ObjectRef> body, String comment)
      : params(params), body(body), comment(comment) {}

  bool SEqual(const TCustomFuncObj* other,
              ffi::TypedFunction<bool(AnyView, AnyView, bool, AnyView)> cmp) const {
    if (!cmp(params, other->params, true, "params")) {
      return false;
    }
    if (!cmp(body, other->body, false, "body")) {
      return false;
    }
    return true;
  }

  int64_t SHash(int64_t init_hash, ffi::TypedFunction<int64_t(AnyView, int64_t, bool)> hash) const {
    int64_t hash_value = init_hash;
    hash_value = hash(params, hash_value, true);
    hash_value = hash(body, hash_value, false);
    return hash_value;
  }

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TCustomFuncObj>()
        .def_ro("params", &TCustomFuncObj::params)
        .def_ro("body", &TCustomFuncObj::body)
        .def_ro("comment", &TCustomFuncObj::comment);
    refl::TypeAttrDef<TCustomFuncObj>()
        .def(refl::type_attr::kSEqual, &TCustomFuncObj::SEqual)
        .def(refl::type_attr::kSHash, &TCustomFuncObj::SHash);
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("test.CustomFunc", TCustomFuncObj, Object);
};

class TCustomFunc : public ObjectRef {
 public:
  explicit TCustomFunc(Array<TVar> params, Array<ObjectRef> body, String comment) {
    data_ = make_object<TCustomFuncObj>(params, body, comment);
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TCustomFunc, ObjectRef, TCustomFuncObj);
};

// Test object with all POD field types to exercise serialization of every field kind.
class TAllFieldsObj : public Object {
 public:
  bool v_bool;
  int64_t v_int;
  double v_float;
  DLDataType v_dtype;
  DLDevice v_device;
  String v_str;
  Optional<String> v_opt_str;
  Array<Any> v_array;
  Map<String, Any> v_map;

  TAllFieldsObj(bool v_bool, int64_t v_int, double v_float, DLDataType v_dtype, DLDevice v_device,
                String v_str, Optional<String> v_opt_str, Array<Any> v_array,
                Map<String, Any> v_map)
      : v_bool(v_bool),
        v_int(v_int),
        v_float(v_float),
        v_dtype(v_dtype),
        v_device(v_device),
        v_str(v_str),
        v_opt_str(v_opt_str),
        v_array(v_array),
        v_map(v_map) {}
  explicit TAllFieldsObj(UnsafeInit) {}

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TAllFieldsObj>()
        .def_ro("v_bool", &TAllFieldsObj::v_bool)
        .def_ro("v_int", &TAllFieldsObj::v_int)
        .def_ro("v_float", &TAllFieldsObj::v_float)
        .def_ro("v_dtype", &TAllFieldsObj::v_dtype)
        .def_ro("v_device", &TAllFieldsObj::v_device)
        .def_ro("v_str", &TAllFieldsObj::v_str)
        .def_ro("v_opt_str", &TAllFieldsObj::v_opt_str)
        .def_ro("v_array", &TAllFieldsObj::v_array)
        .def_ro("v_map", &TAllFieldsObj::v_map);
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("test.AllFields", TAllFieldsObj, Object);
};

class TAllFields : public ObjectRef {
 public:
  explicit TAllFields(bool v_bool, int64_t v_int, double v_float, DLDataType v_dtype,
                      DLDevice v_device, String v_str, Optional<String> v_opt_str,
                      Array<Any> v_array, Map<String, Any> v_map) {
    data_ = make_object<TAllFieldsObj>(v_bool, v_int, v_float, v_dtype, v_device, v_str, v_opt_str,
                                       v_array, v_map);
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TAllFields, ObjectRef, TAllFieldsObj);
};

// Test object with fields that have default values to test deserialization with missing fields
class TWithDefaultsObj : public Object {
 public:
  int64_t required_val;
  int64_t default_int;
  String default_str;
  bool default_bool;

  TWithDefaultsObj(int64_t required_val, int64_t default_int, String default_str, bool default_bool)
      : required_val(required_val),
        default_int(default_int),
        default_str(default_str),
        default_bool(default_bool) {}
  explicit TWithDefaultsObj(UnsafeInit) {}

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TWithDefaultsObj>()
        .def_ro("required_val", &TWithDefaultsObj::required_val)
        .def_ro("default_int", &TWithDefaultsObj::default_int, refl::default_value(42))
        .def_ro("default_str", &TWithDefaultsObj::default_str, refl::default_value("default"))
        .def_ro("default_bool", &TWithDefaultsObj::default_bool, refl::default_value(true));
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("test.WithDefaults", TWithDefaultsObj, Object);
};

class TWithDefaults : public ObjectRef {
 public:
  explicit TWithDefaults(int64_t required_val, int64_t default_int = 42,
                         String default_str = "default", bool default_bool = true) {
    data_ = make_object<TWithDefaultsObj>(required_val, default_int, default_str, default_bool);
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TWithDefaults, ObjectRef, TWithDefaultsObj);
};

}  // namespace testing

template <>
inline constexpr bool use_default_type_traits_v<testing::TPrimExpr> = true;

template <>
struct TypeTraits<testing::TPrimExpr>
    : public ObjectRefWithFallbackTraitsBase<testing::TPrimExpr, StrictBool, int64_t, double,
                                             String> {
  TVM_FFI_INLINE static testing::TPrimExpr ConvertFallbackValue(StrictBool value) {
    return testing::TPrimExpr("bool", static_cast<double>(value));
  }

  TVM_FFI_INLINE static testing::TPrimExpr ConvertFallbackValue(int64_t value) {
    return testing::TPrimExpr("int64", static_cast<double>(value));
  }

  TVM_FFI_INLINE static testing::TPrimExpr ConvertFallbackValue(double value) {
    return testing::TPrimExpr("float32", static_cast<double>(value));
  }
  // hack into the dtype to store string
  TVM_FFI_INLINE static testing::TPrimExpr ConvertFallbackValue(String value) {
    return testing::TPrimExpr(value, 0);
  }
};

}  // namespace ffi
}  // namespace tvm
#endif  // TVM_FFI_TESTING_OBJECT_H_
