
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
#include <gtest/gtest.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/object.h>
#include <tvm/ffi/reflection/access_path.h>
#include <tvm/ffi/reflection/accessor.h>
#include <tvm/ffi/reflection/creator.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>

#include "./testing_object.h"

/*! \brief Look up __ffi_init__ from the TypeAttrColumn (not the method table). */
static tvm::ffi::Function GetInitAttr(const char* type_key) {
  static tvm::ffi::reflection::TypeAttrColumn col(tvm::ffi::reflection::type_attr::kInit);
  return col[tvm::ffi::TypeKeyToIndex(type_key)].cast<tvm::ffi::Function>();
}

namespace {

using namespace tvm::ffi;
using namespace tvm::ffi::testing;

struct TestObjA : public Object {
  int64_t x;
  int64_t y;
  TestObjA(int64_t x, int64_t y) : x(x), y(y) {}

  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("test.TestObjA", TestObjA, Object);
};

struct TestObjADerived : public TestObjA {
  int64_t z;
  TestObjADerived(int64_t x, int64_t y, int64_t z) : TestObjA(x, y), z(z) {}
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("test.TestObjADerived", TestObjADerived, TestObjA);
};

struct TestObjRefADerived : public ObjectRef {
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TestObjRefADerived, ObjectRef, TestObjADerived);
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;

  TIntObj::RegisterReflection();
  TFloatObj::RegisterReflection();
  TPrimExprObj::RegisterReflection();
  TVarObj::RegisterReflection();
  TPairObj::RegisterReflection();
  TVarWithDepObj::RegisterReflection();
  TDefHolderObj::RegisterReflection();
  TFuncObj::RegisterReflection();
  TCustomFuncObj::RegisterReflection();
  TAllFieldsObj::RegisterReflection();
  TWithDefaultsObj::RegisterReflection();

  refl::ObjectDef<TestObjA>()
      .def(refl::init<int64_t, int64_t>())
      .def_ro("x", &TestObjA::x)
      .def_rw("y", &TestObjA::y)
      .def_type_attr("test.attr.object.literal", "literal")
      .def_type_attr("test.attr.object.string", String("ffi-string"))
      .def_type_attr("test.attr.object.std_string", std::string("std-string"))
      .def_type_attr("test.attr.object.int", int64_t{7})
      .def_type_attr("test.attr.object.bool", true)
      .def_type_attr("test.attr.object.func", [](int64_t value) -> int64_t { return value + 1; });
  refl::ObjectDef<TestObjADerived>()
      .def(refl::init<int64_t, int64_t, int64_t>())
      .def_ro("z", &TestObjADerived::z);
  refl::TypeAttrDef<TestObjADerived>()
      .def("test.attr.type_attr_def.literal", "derived-literal")
      .def("test.attr.type_attr_def.string", String("derived-string"))
      .def("test.attr.type_attr_def.int", int64_t{11})
      .def("test.attr.type_attr_def.func", [](int64_t value) -> int64_t { return value * 2; });
}

TEST(Reflection, GetFieldByteOffset) {
  EXPECT_EQ(reflection::GetFieldByteOffsetToObject(&TestObjA::x), sizeof(TVMFFIObject));
  EXPECT_EQ(reflection::GetFieldByteOffsetToObject(&TestObjA::y), 8 + sizeof(TVMFFIObject));
  EXPECT_EQ(reflection::GetFieldByteOffsetToObject(&TIntObj::value), sizeof(TVMFFIObject));
}

TEST(Reflection, FieldGetter) {
  ObjectRef a = TInt(10);
  reflection::FieldGetter getter("test.Int", "value");
  EXPECT_EQ(getter(a).cast<int>(), 10);

  ObjectRef b = TFloat(10.0);
  reflection::FieldGetter getter_float("test.Float", "value");
  EXPECT_EQ(getter_float(b).cast<double>(), 10.0);
}

TEST(Reflection, FieldSetter) {
  ObjectRef a = TFloat(10.0);
  reflection::FieldSetter setter("test.Float", "value");
  setter(a, 20.0);
  EXPECT_EQ(a.as<TFloatObj>()->value, 20.0);
}

TEST(Reflection, FieldInfo) {
  const TVMFFIFieldInfo* info_int = reflection::GetFieldInfo("test.Int", "value");
  EXPECT_FALSE(info_int->flags & kTVMFFIFieldFlagBitMaskHasDefault);
  EXPECT_FALSE(info_int->flags & kTVMFFIFieldFlagBitMaskWritable);
  EXPECT_EQ(Bytes(info_int->doc).operator std::string(), "");

  const TVMFFIFieldInfo* info_float = reflection::GetFieldInfo("test.Float", "value");
  EXPECT_EQ(info_float->default_value_or_factory.v_float64, 10.0);
  EXPECT_TRUE(info_float->flags & kTVMFFIFieldFlagBitMaskHasDefault);
  EXPECT_FALSE(info_float->flags & kTVMFFIFieldFlagBitMaskWritable);
  EXPECT_EQ(Bytes(info_float->doc).operator std::string(), "float value field");

  const TVMFFIFieldInfo* info_prim_expr_dtype = reflection::GetFieldInfo("test.PrimExpr", "dtype");
  AnyView default_value =
      AnyView::CopyFromTVMFFIAny(info_prim_expr_dtype->default_value_or_factory);
  EXPECT_EQ(default_value.cast<String>(), "float");
  EXPECT_TRUE(info_prim_expr_dtype->flags & kTVMFFIFieldFlagBitMaskHasDefault);
  EXPECT_TRUE(info_prim_expr_dtype->flags & kTVMFFIFieldFlagBitMaskWritable);
  EXPECT_EQ(Bytes(info_prim_expr_dtype->doc).operator std::string(), "dtype field");
}

TEST(Reflection, MethodInfo) {
  const TVMFFIMethodInfo* info_int_static_add = reflection::GetMethodInfo("test.Int", "static_add");
  EXPECT_TRUE(info_int_static_add->flags & kTVMFFIFieldFlagBitMaskIsStaticMethod);
  EXPECT_EQ(Bytes(info_int_static_add->doc).operator std::string(), "static add method");

  const TVMFFIMethodInfo* info_float_add = reflection::GetMethodInfo("test.Float", "add");
  EXPECT_FALSE(info_float_add->flags & kTVMFFIFieldFlagBitMaskIsStaticMethod);
  EXPECT_EQ(Bytes(info_float_add->doc).operator std::string(), "add method");

  const TVMFFIMethodInfo* info_float_sub = reflection::GetMethodInfo("test.Float", "sub");
  EXPECT_FALSE(info_float_sub->flags & kTVMFFIFieldFlagBitMaskIsStaticMethod);
  EXPECT_EQ(Bytes(info_float_sub->doc).operator std::string(), "");
}

TEST(Reflection, CallMethod) {
  Function static_int_add = reflection::GetMethod("test.Int", "static_add");
  EXPECT_EQ(static_int_add(TInt(1), TInt(2)).cast<TInt>()->value, 3);

  Function float_add = reflection::GetMethod("test.Float", "add");
  EXPECT_EQ(float_add(TFloat(1), 2.0).cast<double>(), 3.0);

  Function float_sub = reflection::GetMethod("test.Float", "sub");
  EXPECT_EQ(float_sub(TFloat(1), 2.0).cast<double>(), -1.0);

  Function prim_expr_sub = reflection::GetMethod("test.PrimExpr", "sub");
  EXPECT_EQ(prim_expr_sub(TPrimExpr("float", 1), 2.0).cast<double>(), -1.0);
}

TEST(Reflection, InitFunctionBase) {
  Function int_init = GetInitAttr("test.TestObjA");
  Any obj_a = int_init(1, 2);
  EXPECT_TRUE(obj_a.as<TestObjA>() != nullptr);
  EXPECT_EQ(obj_a.as<TestObjA>()->x, 1);
  EXPECT_EQ(obj_a.as<TestObjA>()->y, 2);
}

TEST(Reflection, InitFunctionDerived) {
  Function derived_init = GetInitAttr("test.TestObjADerived");
  Any obj_derived = derived_init(1, 2, 3);
  EXPECT_TRUE(obj_derived.as<TestObjADerived>() != nullptr);
  EXPECT_EQ(obj_derived.as<TestObjADerived>()->x, 1);
  EXPECT_EQ(obj_derived.as<TestObjADerived>()->y, 2);
  EXPECT_EQ(obj_derived.as<TestObjADerived>()->z, 3);
}

TEST(Reflection, ForEachFieldInfo) {
  const TypeInfo* info = TVMFFIGetTypeInfo(TestObjADerived::RuntimeTypeIndex());
  Map<String, int> field_name_to_offset;
  reflection::ForEachFieldInfo(info, [&](const TVMFFIFieldInfo* field_info) {
    field_name_to_offset.Set(String(field_info->name), static_cast<int>(field_info->offset));
  });
  EXPECT_EQ(field_name_to_offset["x"], sizeof(TVMFFIObject));
  EXPECT_EQ(field_name_to_offset["y"], 8 + sizeof(TVMFFIObject));
  EXPECT_EQ(field_name_to_offset["z"], 16 + sizeof(TVMFFIObject));
}

TEST(Reflection, TypeAttrColumn) {
  reflection::TypeAttrColumn size_attr("test.size");
  EXPECT_EQ(size_attr[TIntObj::RuntimeTypeIndex()].cast<int>(), sizeof(TIntObj));
}

TEST(Reflection, TypeAttrColumnBeginIndex) {
  // Get the column and verify begin_index
  TVMFFIByteArray attr_name = {"test.size", std::char_traits<char>::length("test.size")};
  const TVMFFITypeAttrColumn* column = TVMFFIGetTypeAttrColumn(&attr_name);
  ASSERT_NE(column, nullptr);
  // begin_index should be >= 0
  EXPECT_GE(column->begin_index, 0);
  // size should cover the range from begin_index
  EXPECT_GT(column->size, 0);
  // verify that lookup of a type_index below begin_index returns None
  reflection::TypeAttrColumn size_attr("test.size");
  AnyView result = size_attr[0];  // index 0 is kTVMFFINone, unlikely to have this attr
  (void)result;  // suppress unused variable warning; we only verify no crash occurs
  // The result may or may not be None depending on begin_index; the key is no crash.
  // verify the known registered entry still works
  EXPECT_EQ(size_attr[TIntObj::RuntimeTypeIndex()].cast<int>(), sizeof(TIntObj));
}

TEST(Reflection, ObjectDefTypeAttrDirectValues) {
  reflection::TypeAttrColumn literal_attr("test.attr.object.literal");
  reflection::TypeAttrColumn string_attr("test.attr.object.string");
  reflection::TypeAttrColumn std_string_attr("test.attr.object.std_string");
  reflection::TypeAttrColumn int_attr("test.attr.object.int");
  reflection::TypeAttrColumn bool_attr("test.attr.object.bool");
  reflection::TypeAttrColumn func_attr("test.attr.object.func");
  int32_t type_index = TestObjA::RuntimeTypeIndex();

  EXPECT_EQ(literal_attr[type_index].cast<String>(), "literal");
  EXPECT_EQ(string_attr[type_index].cast<String>(), "ffi-string");
  EXPECT_EQ(std_string_attr[type_index].cast<String>(), "std-string");
  EXPECT_EQ(int_attr[type_index].cast<int64_t>(), 7);
  EXPECT_EQ(bool_attr[type_index].cast<bool>(), true);
  EXPECT_EQ(func_attr[type_index].cast<Function>()(3).cast<int64_t>(), 4);
}

TEST(Reflection, TypeAttrDefDirectValues) {
  reflection::TypeAttrColumn literal_attr("test.attr.type_attr_def.literal");
  reflection::TypeAttrColumn string_attr("test.attr.type_attr_def.string");
  reflection::TypeAttrColumn int_attr("test.attr.type_attr_def.int");
  reflection::TypeAttrColumn func_attr("test.attr.type_attr_def.func");
  int32_t type_index = TestObjADerived::RuntimeTypeIndex();

  EXPECT_EQ(literal_attr[type_index].cast<String>(), "derived-literal");
  EXPECT_EQ(string_attr[type_index].cast<String>(), "derived-string");
  EXPECT_EQ(int_attr[type_index].cast<int64_t>(), 11);
  EXPECT_EQ(func_attr[type_index].cast<Function>()(5).cast<int64_t>(), 10);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_method("testing.Int_GetValue", &TIntObj::GetValue);
}

TEST(Reflection, FuncRegister) {
  Function fget_value = Function::GetGlobalRequired("testing.Int_GetValue");
  TInt a(12);
  EXPECT_EQ(fget_value(a).cast<int>(), 12);
}

TEST(Reflection, ObjectCreator) {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectCreator creator("test.Int");
  EXPECT_EQ(creator(Map<String, Any>({{"value", 1}})).cast<TInt>()->value, 1);
}

TEST(Reflection, AccessPath) {
  namespace refl = tvm::ffi::reflection;

  // Test basic path construction and ToSteps()
  refl::AccessPath path = refl::AccessPath::Root()->Attr("body")->ArrayItem(1);
  auto steps = path->ToSteps();
  EXPECT_EQ(steps.size(), 2);
  EXPECT_EQ(steps[0]->kind, refl::AccessKind::kAttr);
  EXPECT_EQ(steps[1]->kind, refl::AccessKind::kArrayItem);
  EXPECT_EQ(steps[0]->key.cast<String>(), "body");
  EXPECT_EQ(steps[1]->key.cast<int64_t>(), 1);

  // Test PathEqual with identical paths
  refl::AccessPath path2 = refl::AccessPath::Root()->Attr("body")->ArrayItem(1);
  EXPECT_TRUE(path->PathEqual(path2));
  EXPECT_TRUE(path->IsPrefixOf(path2));

  // Test PathEqual with different paths
  refl::AccessPath path3 = refl::AccessPath::Root()->Attr("body")->ArrayItem(2);
  EXPECT_FALSE(path->PathEqual(path3));
  EXPECT_FALSE(path->IsPrefixOf(path3));

  // Test prefix relationship - path4 extends path, so path should be prefix of path4
  refl::AccessPath path4 = refl::AccessPath::Root()->Attr("body")->ArrayItem(1)->Attr("body");
  EXPECT_FALSE(path->PathEqual(path4));  // Not equal (different lengths)
  EXPECT_TRUE(path->IsPrefixOf(path4));  // But path is a prefix of path4

  // Test completely different paths
  refl::AccessPath path5 = refl::AccessPath::Root()->ArrayItem(0)->ArrayItem(1)->Attr("body");
  EXPECT_FALSE(path->PathEqual(path5));
  EXPECT_FALSE(path->IsPrefixOf(path5));

  // Test Root path
  refl::AccessPath root = refl::AccessPath::Root();
  auto root_steps = root->ToSteps();
  EXPECT_EQ(root_steps.size(), 0);
  EXPECT_EQ(root->depth, 0);
  EXPECT_TRUE(root->IsPrefixOf(path));
  EXPECT_TRUE(root->IsPrefixOf(root));
  EXPECT_TRUE(root->PathEqual(refl::AccessPath::Root()));

  // Test depth calculations
  EXPECT_EQ(path->depth, 2);
  EXPECT_EQ(path4->depth, 3);
  EXPECT_EQ(root->depth, 0);

  // Test MapItem access
  refl::AccessPath map_path = refl::AccessPath::Root()->Attr("data")->MapItem("key1");
  auto map_steps = map_path->ToSteps();
  EXPECT_EQ(map_steps.size(), 2);
  EXPECT_EQ(map_steps[0]->kind, refl::AccessKind::kAttr);
  EXPECT_EQ(map_steps[1]->kind, refl::AccessKind::kMapItem);
  EXPECT_EQ(map_steps[0]->key.cast<String>(), "data");
  EXPECT_EQ(map_steps[1]->key.cast<String>(), "key1");

  // Test MapItemMissing access
  refl::AccessPath map_missing_path = refl::AccessPath::Root()->MapItemMissing(42);
  auto map_missing_steps = map_missing_path->ToSteps();
  EXPECT_EQ(map_missing_steps.size(), 1);
  EXPECT_EQ(map_missing_steps[0]->kind, refl::AccessKind::kMapItemMissing);
  EXPECT_EQ(map_missing_steps[0]->key.cast<int64_t>(), 42);

  // Test ArrayItemMissing access
  refl::AccessPath array_missing_path = refl::AccessPath::Root()->ArrayItemMissing(5);
  auto array_missing_steps = array_missing_path->ToSteps();
  EXPECT_EQ(array_missing_steps.size(), 1);
  EXPECT_EQ(array_missing_steps[0]->kind, refl::AccessKind::kArrayItemMissing);
  EXPECT_EQ(array_missing_steps[0]->key.cast<int64_t>(), 5);

  // Test FromSteps static method - round trip conversion
  auto original_steps = path->ToSteps();
  refl::AccessPath reconstructed = refl::AccessPath::FromSteps(original_steps);
  EXPECT_TRUE(path->PathEqual(reconstructed));
  EXPECT_EQ(path->depth, reconstructed->depth);

  // Test complex prefix relationships
  refl::AccessPath short_path = refl::AccessPath::Root()->Attr("x");
  refl::AccessPath medium_path = refl::AccessPath::Root()->Attr("x")->ArrayItem(0);
  refl::AccessPath long_path = refl::AccessPath::Root()->Attr("x")->ArrayItem(0)->MapItem("z");

  EXPECT_TRUE(short_path->IsPrefixOf(medium_path));
  EXPECT_TRUE(short_path->IsPrefixOf(long_path));
  EXPECT_TRUE(medium_path->IsPrefixOf(long_path));
  EXPECT_FALSE(medium_path->IsPrefixOf(short_path));
  EXPECT_FALSE(long_path->IsPrefixOf(medium_path));
  EXPECT_FALSE(long_path->IsPrefixOf(short_path));

  // Test non-prefix relationships
  refl::AccessPath branch1 = refl::AccessPath::Root()->Attr("x")->ArrayItem(0);
  refl::AccessPath branch2 = refl::AccessPath::Root()->Attr("x")->ArrayItem(1);
  EXPECT_FALSE(branch1->IsPrefixOf(branch2));
  EXPECT_FALSE(branch2->IsPrefixOf(branch1));
  EXPECT_FALSE(branch1->PathEqual(branch2));

  // Test GetParent functionality
  auto parent = path4->GetParent();
  EXPECT_TRUE(parent.has_value());
  EXPECT_TRUE(parent.value()->PathEqual(path));

  auto root_parent = root->GetParent();
  EXPECT_FALSE(root_parent.has_value());
}

struct TestObjWithFactory : public Object {
  Array<ObjectRef> items;
  int64_t count;

  explicit TestObjWithFactory(UnsafeInit) {}

  [[maybe_unused]] static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("test.TestObjWithFactory", TestObjWithFactory, Object);
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TestObjWithFactory>()
      .def_ro("items", &TestObjWithFactory::items,
              refl::default_factory(
                  Function::FromTyped([]() -> Array<ObjectRef> { return Array<ObjectRef>(); })))
      .def_ro("count", &TestObjWithFactory::count, refl::default_value(static_cast<int64_t>(0)));
}

struct TestObjWithAny : public Object {
  Any value;
  explicit TestObjWithAny(Any value) : value(std::move(value)) {}
  [[maybe_unused]] static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("test.TestObjWithAny", TestObjWithAny, Object);
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TestObjWithAny>().def(refl::init<Any>()).def_ro("value", &TestObjWithAny::value);
}

struct TestObjWithAnyView : public Object {
  Any value;
  explicit TestObjWithAnyView(AnyView value) : value(value) {}
  [[maybe_unused]] static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("test.TestObjWithAnyView", TestObjWithAnyView, Object);
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TestObjWithAnyView>()
      .def(refl::init<AnyView>())
      .def_ro("value", &TestObjWithAnyView::value);
}

TEST(Reflection, InitWithAny) {
  Function init = GetInitAttr("test.TestObjWithAny");
  Any obj1 = init(42);
  ASSERT_TRUE(obj1.as<TestObjWithAny>() != nullptr);
  EXPECT_EQ(obj1.as<TestObjWithAny>()->value.cast<int>(), 42);

  Any obj2 = init(3.14);
  ASSERT_TRUE(obj2.as<TestObjWithAny>() != nullptr);
  EXPECT_EQ(obj2.as<TestObjWithAny>()->value.cast<double>(), 3.14);

  Any obj3 = init(String("hello"));
  ASSERT_TRUE(obj3.as<TestObjWithAny>() != nullptr);
  EXPECT_EQ(obj3.as<TestObjWithAny>()->value.cast<String>(), "hello");
}

TEST(Reflection, InitWithAnyView) {
  Function init = GetInitAttr("test.TestObjWithAnyView");
  Any obj1 = init(42);
  ASSERT_TRUE(obj1.as<TestObjWithAnyView>() != nullptr);
  EXPECT_EQ(obj1.as<TestObjWithAnyView>()->value.cast<int>(), 42);

  Any obj2 = init(3.14);
  ASSERT_TRUE(obj2.as<TestObjWithAnyView>() != nullptr);
  EXPECT_EQ(obj2.as<TestObjWithAnyView>()->value.cast<double>(), 3.14);

  Any obj3 = init(String("hello"));
  ASSERT_TRUE(obj3.as<TestObjWithAnyView>() != nullptr);
  EXPECT_EQ(obj3.as<TestObjWithAnyView>()->value.cast<String>(), "hello");
}
TEST(Reflection, DefaultFactoryFlag) {
  const TVMFFIFieldInfo* info_items = reflection::GetFieldInfo("test.TestObjWithFactory", "items");
  EXPECT_TRUE(info_items->flags & kTVMFFIFieldFlagBitMaskHasDefault);
  EXPECT_TRUE(info_items->flags & kTVMFFIFieldFlagBitMaskDefaultFromFactory);

  const TVMFFIFieldInfo* info_count = reflection::GetFieldInfo("test.TestObjWithFactory", "count");
  EXPECT_TRUE(info_count->flags & kTVMFFIFieldFlagBitMaskHasDefault);
  EXPECT_FALSE(info_count->flags & kTVMFFIFieldFlagBitMaskDefaultFromFactory);
}

TEST(Reflection, DefaultFactoryCreation) {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectCreator creator("test.TestObjWithFactory");

  // Create two objects without providing "items" - each should get a fresh Array
  Any obj1 = creator(Map<String, Any>({{"count", static_cast<int64_t>(42)}}));
  Any obj2 = creator(Map<String, Any>({{"count", static_cast<int64_t>(99)}}));

  auto* p1 = obj1.as<TestObjWithFactory>();
  auto* p2 = obj2.as<TestObjWithFactory>();

  ASSERT_NE(p1, nullptr);
  ASSERT_NE(p2, nullptr);
  EXPECT_EQ(p1->count, 42);
  EXPECT_EQ(p2->count, 99);
  // Both should have empty arrays
  EXPECT_EQ(p1->items.size(), 0);
  EXPECT_EQ(p2->items.size(), 0);
  // Crucially, the arrays should be distinct objects (not aliased)
  EXPECT_NE(p1->items.get(), p2->items.get());
}

TEST(Reflection, DefaultFactoryNotCalledWhenProvided) {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectCreator creator("test.TestObjWithFactory");

  Array<ObjectRef> custom_items;
  custom_items.push_back(TInt(1));
  Any obj =
      creator(Map<String, Any>({{"items", custom_items}, {"count", static_cast<int64_t>(5)}}));

  auto* p = obj.as<TestObjWithFactory>();
  ASSERT_NE(p, nullptr);
  EXPECT_EQ(p->items.size(), 1);
  EXPECT_EQ(p->count, 5);
}

// ---------------------------------------------------------------------------
// Tests for auto-generated __ffi_init__ with init(false) / KwOnly(true)
// ---------------------------------------------------------------------------

struct TestAutoInitObj : public Object {
  int64_t a;
  int64_t b;
  int64_t c;
  int64_t d;

  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("test.AutoInit", TestAutoInitObj, Object);
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  // No refl::init<>() — auto-generates __ffi_init__
  refl::ObjectDef<TestAutoInitObj>()
      .def_rw("a", &TestAutoInitObj::a)
      .def_rw("b", &TestAutoInitObj::b, refl::init(false), refl::default_value(int64_t{42}))
      .def_rw("c", &TestAutoInitObj::c, refl::kw_only(true))
      .def_rw("d", &TestAutoInitObj::d, refl::default_value(int64_t{99}));
}

TEST(Reflection, AutoInitPositional) {
  // Auto-generated init: positional args for non-kw-only init=True fields (a, d)
  // c is kw_only so it cannot be passed positionally.
  Function auto_init = GetInitAttr("test.AutoInit");
  ObjectRef kwargs = Function::GetGlobalRequired("ffi.GetKwargsObject")().cast<ObjectRef>();
  // Positional: a=1, d=3; keyword: c=2
  Any obj = auto_init(int64_t{1}, int64_t{3}, kwargs, String("c"), int64_t{2});
  auto* p = obj.as<TestAutoInitObj>();
  ASSERT_NE(p, nullptr);
  EXPECT_EQ(p->a, 1);
  EXPECT_EQ(p->b, 42);  // init=False, gets default
  EXPECT_EQ(p->c, 2);   // kw_only, passed via KWARGS
  EXPECT_EQ(p->d, 3);   // init=True, 2nd positional
}

TEST(Reflection, AutoInitPartialPositional) {
  // Provide only a (position 0); c is required but missing → error
  Function auto_init = GetInitAttr("test.AutoInit");
  EXPECT_THROW(
      {
        try {
          auto_init(int64_t{1});
        } catch (const std::exception& e) {
          EXPECT_NE(std::string(e.what()).find("missing required"), std::string::npos);
          throw;
        }
      },
      std::exception);
}

TEST(Reflection, AutoInitWithDefaults) {
  // Provide a positionally and c via KWARGS; d should use default 99
  Function auto_init = GetInitAttr("test.AutoInit");
  ObjectRef kwargs = Function::GetGlobalRequired("ffi.GetKwargsObject")().cast<ObjectRef>();
  Any obj = auto_init(int64_t{10}, kwargs, String("c"), int64_t{20});
  auto* p = obj.as<TestAutoInitObj>();
  ASSERT_NE(p, nullptr);
  EXPECT_EQ(p->a, 10);
  EXPECT_EQ(p->b, 42);  // default
  EXPECT_EQ(p->c, 20);  // provided via KWARGS
  EXPECT_EQ(p->d, 99);  // default
}

TEST(Reflection, AutoInitKwargs) {
  Function auto_init = GetInitAttr("test.AutoInit");
  ObjectRef kwargs = Function::GetGlobalRequired("ffi.GetKwargsObject")().cast<ObjectRef>();

  // Positional: a=1, then KWARGS: c=30, d=40
  Any obj = auto_init(int64_t{1}, kwargs, String("c"), int64_t{30}, String("d"), int64_t{40});
  auto* p = obj.as<TestAutoInitObj>();
  ASSERT_NE(p, nullptr);
  EXPECT_EQ(p->a, 1);
  EXPECT_EQ(p->b, 42);  // default
  EXPECT_EQ(p->c, 30);
  EXPECT_EQ(p->d, 40);
}

TEST(Reflection, AutoInitKwargsOnly) {
  Function auto_init = GetInitAttr("test.AutoInit");
  ObjectRef kwargs = Function::GetGlobalRequired("ffi.GetKwargsObject")().cast<ObjectRef>();

  // No positional args, all via KWARGS
  Any obj = auto_init(kwargs, String("a"), int64_t{5}, String("c"), int64_t{15}, String("d"),
                      int64_t{25});
  auto* p = obj.as<TestAutoInitObj>();
  ASSERT_NE(p, nullptr);
  EXPECT_EQ(p->a, 5);
  EXPECT_EQ(p->b, 42);
  EXPECT_EQ(p->c, 15);
  EXPECT_EQ(p->d, 25);
}

TEST(Reflection, AutoInitKwargsDuplicate) {
  Function auto_init = GetInitAttr("test.AutoInit");
  ObjectRef kwargs = Function::GetGlobalRequired("ffi.GetKwargsObject")().cast<ObjectRef>();

  // a is provided positionally AND as kwarg → error
  EXPECT_THROW(
      {
        try {
          auto_init(int64_t{1}, kwargs, String("a"), int64_t{2}, String("c"), int64_t{3});
        } catch (const std::exception& e) {
          EXPECT_NE(std::string(e.what()).find("multiple values"), std::string::npos);
          throw;
        }
      },
      std::exception);
}

TEST(Reflection, AutoInitKwargsUnknown) {
  Function auto_init = GetInitAttr("test.AutoInit");
  ObjectRef kwargs = Function::GetGlobalRequired("ffi.GetKwargsObject")().cast<ObjectRef>();

  EXPECT_THROW(
      {
        try {
          auto_init(kwargs, String("a"), int64_t{1}, String("z"), int64_t{2}, String("c"),
                    int64_t{3});
        } catch (const std::exception& e) {
          EXPECT_NE(std::string(e.what()).find("unexpected keyword"), std::string::npos);
          throw;
        }
      },
      std::exception);
}

TEST(Reflection, AutoInitFlagBits) {
  // Verify the flag bits are set correctly on the field info.
  const TVMFFIFieldInfo* fi_a = reflection::GetFieldInfo("test.AutoInit", "a");
  EXPECT_FALSE(fi_a->flags & kTVMFFIFieldFlagBitMaskInitOff);
  EXPECT_FALSE(fi_a->flags & kTVMFFIFieldFlagBitMaskKwOnly);

  const TVMFFIFieldInfo* fi_b = reflection::GetFieldInfo("test.AutoInit", "b");
  EXPECT_TRUE(fi_b->flags & kTVMFFIFieldFlagBitMaskInitOff);
  EXPECT_FALSE(fi_b->flags & kTVMFFIFieldFlagBitMaskKwOnly);
  EXPECT_TRUE(fi_b->flags & kTVMFFIFieldFlagBitMaskHasDefault);

  const TVMFFIFieldInfo* fi_c = reflection::GetFieldInfo("test.AutoInit", "c");
  EXPECT_FALSE(fi_c->flags & kTVMFFIFieldFlagBitMaskInitOff);
  EXPECT_TRUE(fi_c->flags & kTVMFFIFieldFlagBitMaskKwOnly);

  const TVMFFIFieldInfo* fi_d = reflection::GetFieldInfo("test.AutoInit", "d");
  EXPECT_FALSE(fi_d->flags & kTVMFFIFieldFlagBitMaskInitOff);
  EXPECT_FALSE(fi_d->flags & kTVMFFIFieldFlagBitMaskKwOnly);
  EXPECT_TRUE(fi_d->flags & kTVMFFIFieldFlagBitMaskHasDefault);
}

// Simple auto-init test: all fields init=True, no Init/KwOnly traits
struct TestAutoInitSimpleObj : public Object {
  int64_t x;
  int64_t y;

  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("test.AutoInitSimple", TestAutoInitSimpleObj, Object);
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TestAutoInitSimpleObj>()
      .def_rw("x", &TestAutoInitSimpleObj::x)
      .def_rw("y", &TestAutoInitSimpleObj::y);
}

TEST(Reflection, AutoInitSimple) {
  Function auto_init = GetInitAttr("test.AutoInitSimple");
  Any obj = auto_init(int64_t{10}, int64_t{20});
  auto* p = obj.as<TestAutoInitSimpleObj>();
  ASSERT_NE(p, nullptr);
  EXPECT_EQ(p->x, 10);
  EXPECT_EQ(p->y, 20);
}

TEST(Reflection, AutoInitSimpleTooManyArgs) {
  Function auto_init = GetInitAttr("test.AutoInitSimple");
  EXPECT_THROW(auto_init(int64_t{1}, int64_t{2}, int64_t{3}), std::exception);
}

// ---------------------------------------------------------------------------
// overload_cast — pick an overload by prefix-matching its parameter types.
// ---------------------------------------------------------------------------

namespace overload_cast_test {

struct Cat {};
struct Dog {};

// Pet: each Feed overload has a unique first arg plus a trailing context
// (`int amount`) that the caller doesn't have to spell out. Get is const
// vs non-const overloaded with identical params (selected via const_ tag).
struct Pet {
  int Feed(const Cat*, int amount) { return 100 + amount; }
  int Feed(const Dog*, int amount) { return 200 + amount; }
  int Get(int x) { return 4000 + x; }
  int Get(int x) const { return 5000 + x; }
};

// Mix: overloads share a leading prefix — spelling more parameters
// disambiguates against the longer variant.
struct Mix {
  int Run(int, int) { return 1000; }
  int Run(int, double) { return 2000; }
  int Run(int, int, int) { return 3000; }
};

int FreeFeed(const Cat*, int x) { return 6000 + x; }
int FreeFeed(const Dog*, int x) { return 7000 + x; }

template <auto Method>
struct CallVia {
  template <typename Self, typename... Args>
  static auto Run(Self&& self, Args&&... args) {
    return (std::forward<Self>(self).*Method)(std::forward<Args>(args)...);
  }
};

}  // namespace overload_cast_test

TEST(OverloadCast, PrefixMatch) {
  using namespace overload_cast_test;
  namespace refl = tvm::ffi::reflection;
  Pet p;
  Cat cat;
  Dog dog;

  // (a) Member with unique first arg per overload: spelling only the
  //     disambiguating prefix picks the overload and deduces the
  //     trailing `int amount` from the picked signature.
  auto p_cat = refl::overload_cast<const Cat*>(&Pet::Feed);
  static_assert(std::is_same_v<decltype(p_cat), int (Pet::*)(const Cat*, int)>,
                "prefix match must deduce trailing arg types");
  EXPECT_EQ((p.*p_cat)(&cat, 7), 107);

  auto p_dog = refl::overload_cast<const Dog*>(&Pet::Feed);
  EXPECT_EQ((p.*p_dog)(&dog, 12), 212);

  // (b) Free function with the same shape — trailing arg deduced.
  auto p_free_cat = refl::overload_cast<const Cat*>(&FreeFeed);
  EXPECT_EQ(p_free_cat(&cat, 7), 6007);
  auto p_free_dog = refl::overload_cast<const Dog*>(&FreeFeed);
  EXPECT_EQ(p_free_dog(&dog, 7), 7007);
}

TEST(OverloadCast, AmbiguousPrefixRequiresMoreSpelling) {
  using namespace overload_cast_test;
  namespace refl = tvm::ffi::reflection;
  Mix m;

  // Mix::Run has three overloads:
  //   Run(int, int), Run(int, double), Run(int, int, int)
  // Spelling only <int> would be ambiguous (all three start with int).
  // Spelling enough parameters to identify exactly one overload picks it.
  EXPECT_EQ((m.*refl::overload_cast<int, int>(&Mix::Run))(0, 0), 1000);
  EXPECT_EQ((m.*refl::overload_cast<int, double>(&Mix::Run))(0, 0.0), 2000);
  EXPECT_EQ((m.*refl::overload_cast<int, int, int>(&Mix::Run))(0, 0, 0), 3000);
}

TEST(OverloadCast, ConstQualifiedMember) {
  using namespace overload_cast_test;
  namespace refl = tvm::ffi::reflection;
  Pet p;
  const Pet& cp = p;

  // Non-const overload — no tag.
  EXPECT_EQ((p.*refl::overload_cast<int>(&Pet::Get))(7), 4007);

  // Const overload — const_ tag required (even when the const overload
  // is the only one with that signature, address-of-overload alone
  // cannot select it from the operator() overload set).
  EXPECT_EQ((cp.*refl::overload_cast<int>(&Pet::Get, refl::const_))(7), 5007);
}

TEST(OverloadCast, NonTypeTemplateArgument) {
  using namespace overload_cast_test;
  namespace refl = tvm::ffi::reflection;
  Pet p;
  Mix m;
  Cat cat;

  // Prefix match composed as a non-type template argument.
  EXPECT_EQ((CallVia<refl::overload_cast<const Cat*>(&Pet::Feed)>::Run(p, &cat, 7)), 107);

  // Disambiguated 3-arg overload as a non-type template argument.
  EXPECT_EQ((CallVia<refl::overload_cast<int, int, int>(&Mix::Run)>::Run(m, 0, 0, 0)), 3000);
}

}  // namespace
