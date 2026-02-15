
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
  TFuncObj::RegisterReflection();
  TCustomFuncObj::RegisterReflection();

  refl::ObjectDef<TestObjA>()
      .def(refl::init<int64_t, int64_t>())
      .def_ro("x", &TestObjA::x)
      .def_rw("y", &TestObjA::y);
  refl::ObjectDef<TestObjADerived>()
      .def(refl::init<int64_t, int64_t, int64_t>())
      .def_ro("z", &TestObjADerived::z);
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
  Function int_init = reflection::GetMethod("test.TestObjA", "__ffi_init__");
  Any obj_a = int_init(1, 2);
  EXPECT_TRUE(obj_a.as<TestObjA>() != nullptr);
  EXPECT_EQ(obj_a.as<TestObjA>()->x, 1);
  EXPECT_EQ(obj_a.as<TestObjA>()->y, 2);
}

TEST(Reflection, InitFunctionDerived) {
  Function derived_init = reflection::GetMethod("test.TestObjADerived", "__ffi_init__");
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
              refl::DefaultFactory(
                  Function::FromTyped([]() -> Array<ObjectRef> { return Array<ObjectRef>(); })))
      .def_ro("count", &TestObjWithFactory::count, refl::DefaultValue(static_cast<int64_t>(0)));
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
  Function init = reflection::GetMethod("test.TestObjWithAny", "__ffi_init__");
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
  Function init = reflection::GetMethod("test.TestObjWithAnyView", "__ffi_init__");
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

}  // namespace
