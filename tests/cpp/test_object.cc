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
#include <tvm/ffi/cast.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/dict.h>
#include <tvm/ffi/container/list.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/container/tuple.h>
#include <tvm/ffi/container/variant.h>
#include <tvm/ffi/expected.h>
#include <tvm/ffi/memory.h>
#include <tvm/ffi/object.h>
#include <tvm/ffi/optional.h>

#include <stdexcept>
#include <type_traits>

#include "./testing_object.h"

namespace tvm {
namespace ffi {
namespace testing {

class TIntOrFloatRef : public ObjectRef {
 public:
  TIntOrFloatRef() = default;
  explicit TIntOrFloatRef(UnsafeInit tag) : ObjectRef(tag) {}

  static constexpr bool _type_is_nullable = true;
  static constexpr bool _type_container_is_exact = false;
  using ContainerType = Object;
};

}  // namespace testing

template <>
struct TypeTraits<testing::TIntOrFloatRef>
    : public ObjectRefTypeTraitsBase<testing::TIntOrFloatRef> {
  TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
    return TypeTraits<testing::TInt>::CheckAnyStrict(src) ||
           TypeTraits<testing::TFloat>::CheckAnyStrict(src);
  }
};

template <typename ObjectType>
inline constexpr bool object_ref_contains_v<testing::TIntOrFloatRef, ObjectType> =
    std::is_base_of_v<testing::TIntObj, ObjectType> ||
    std::is_base_of_v<testing::TFloatObj, ObjectType>;

}  // namespace ffi
}  // namespace tvm

namespace {

using namespace tvm::ffi;
using namespace tvm::ffi::testing;

template <typename RefType, typename ObjectType, typename = void>
inline constexpr bool object_ref_contains_is_enabled_v = false;

template <typename RefType, typename ObjectType>
inline constexpr bool object_ref_contains_is_enabled_v<
    RefType, ObjectType, std::void_t<decltype(object_ref_contains_v<RefType, ObjectType>)>> = true;

static_assert(ObjectRef::_type_container_is_exact);
static_assert(TNumber::_type_container_is_exact);
static_assert(TInt::_type_container_is_exact);
// Optional<T> is uniformly Any-backed and is no longer an ObjectRef, so it does
// not participate in the ObjectRef container concept.
static_assert(!std::is_base_of_v<ObjectRef, Optional<TInt>>);
static_assert(!TIntOrFloatRef::_type_container_is_exact);
static_assert(!Array<TInt>::_type_container_is_exact);
static_assert(!List<TInt>::_type_container_is_exact);
static_assert(!Map<TInt, TFloat>::_type_container_is_exact);
static_assert(!Dict<TInt, TFloat>::_type_container_is_exact);
static_assert(!Tuple<TInt, TFloat>::_type_container_is_exact);
static_assert(!Variant<TInt, TFloat>::_type_container_is_exact);

static_assert(object_ref_contains_v<TNumber, TIntObj>);
static_assert(object_ref_contains_v<TInt, TIntObj>);
// Optional<T> is no longer an ObjectRef, so it is outside the object-ref-contains
// concept entirely (the trait is not even enabled for it).
static_assert(!object_ref_contains_is_enabled_v<Optional<TInt>, TIntObj>);
static_assert(!object_ref_contains_v<TInt, TFloatObj>);
static_assert(!object_ref_contains_v<Array<TInt>, ArrayObj>);
static_assert(object_ref_contains_v<TIntOrFloatRef, TIntObj>);
static_assert(object_ref_contains_v<TIntOrFloatRef, TFloatObj>);
static_assert(!object_ref_contains_v<TIntOrFloatRef, TNumberObj>);
static_assert(object_ref_contains_is_enabled_v<TInt, TIntObj>);
static_assert(object_ref_contains_is_enabled_v<TIntOrFloatRef, TIntObj>);
static_assert(!object_ref_contains_is_enabled_v<int, TIntObj>);
static_assert(!object_ref_contains_is_enabled_v<TIntObj, TIntObj>);
static_assert(!object_ref_contains_is_enabled_v<TInt, int>);
static_assert(!object_ref_contains_is_enabled_v<TIntOrFloatRef, int>);

template <typename T>
class CRTPObject : public Object {
 public:
  static constexpr int _type_child_slots [[maybe_unused]] = 0;
  static constexpr bool _type_final [[maybe_unused]] = true;
  TVM_FFI_DECLARE_OBJECT_INFO_PREDEFINED_TYPE_KEY(T, Object);

 private:
  friend T;
  CRTPObject() = default;
};

class LeafObject : public CRTPObject<LeafObject> {
 public:
  static constexpr const char* _type_key = "test.CRTPLeaf";
};

class ThrowingConstructorObject : public Object {
 public:
  ThrowingConstructorObject() { throw std::runtime_error("constructor failed"); }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("test.ThrowingConstructorObject", ThrowingConstructorObject,
                                    Object);
};

TEST(Object, RefCounter) {
  ObjectPtr<TIntObj> a = make_object<TIntObj>(11);
  ObjectPtr<TIntObj> b = a;

  EXPECT_EQ(a->value, 11);

  EXPECT_EQ(a.use_count(), 2);
  ObjectPtr<TIntObj> aa = make_object<TIntObj>(*a);
  EXPECT_EQ(aa.use_count(), 1);
  EXPECT_EQ(aa->value, 11);

  b.reset();
  EXPECT_EQ(a.use_count(), 1);
  EXPECT_TRUE(b == nullptr);
  EXPECT_EQ(b.use_count(), 0);

  ObjectPtr<TIntObj> c = std::move(a);
  EXPECT_EQ(c.use_count(), 1);
  EXPECT_TRUE(a == nullptr);  // NOLINT(bugprone-use-after-move,clang-analyzer-cplusplus.Move)

  EXPECT_EQ(c->value, 11);
}

TEST(Object, MakeObjectPropagatesConstructorException) {
  EXPECT_THROW(make_object<ThrowingConstructorObject>(), std::runtime_error);
}

TEST(Object, TypeInfo) {
  const TypeInfo* info = TVMFFIGetTypeInfo(TIntObj::RuntimeTypeIndex());
  EXPECT_TRUE(info != nullptr);
  EXPECT_EQ(info->type_index, TIntObj::RuntimeTypeIndex());
  EXPECT_EQ(info->type_depth, 2);
  EXPECT_EQ(info->type_ancestors[0]->type_index, Object::RuntimeTypeIndex());
  EXPECT_EQ(info->type_ancestors[1]->type_index, TNumberObj::RuntimeTypeIndex());
  EXPECT_GE(info->type_index, TypeIndex::kTVMFFIDynObjectBegin);
}

TEST(Object, CRTPObjectInfo) {
  const TypeInfo* info = TVMFFIGetTypeInfo(LeafObject::RuntimeTypeIndex());
  ASSERT_TRUE(info != nullptr);
  EXPECT_EQ(info->type_index, LeafObject::RuntimeTypeIndex());
  EXPECT_EQ(info->type_depth, 1);
  EXPECT_EQ(info->type_ancestors[0]->type_index, Object::RuntimeTypeIndex());
  EXPECT_GE(info->type_index, TypeIndex::kTVMFFIDynObjectBegin);
}

TEST(Object, TypeGetOrAllocIndexQueryRegistered) {
  TVMFFIByteArray type_key{TIntObj::_type_key, std::char_traits<char>::length(TIntObj::_type_key)};
  EXPECT_EQ(TVMFFITypeGetOrAllocIndex(&type_key, -1, 0, 0, 0, -2), TIntObj::RuntimeTypeIndex());
}

TEST(Object, TypeGetOrAllocIndexQueryMissDoesNotRegister) {
  const char* type_key_data = "test.TypeGetOrAllocIndexQueryMiss";
  TVMFFIByteArray type_key{type_key_data, std::char_traits<char>::length(type_key_data)};
  EXPECT_EQ(TVMFFITypeGetOrAllocIndex(&type_key, -1, 0, 0, 0, -2), -2);

  int32_t type_index = -1;
  EXPECT_NE(TVMFFITypeKeyToIndex(&type_key, &type_index), 0);
  EXPECT_EQ(type_index, -1);
}

TEST(Object, InstanceCheck) {
  ObjectPtr<Object> a = make_object<TIntObj>(11);
  ObjectPtr<Object> b = make_object<TFloatObj>(11);

  EXPECT_TRUE(a->IsInstance<Object>());
  EXPECT_TRUE(a->IsInstance<TNumberObj>());
  EXPECT_TRUE(a->IsInstance<TIntObj>());
  EXPECT_TRUE(!a->IsInstance<TFloatObj>());

  EXPECT_TRUE(a->IsInstance<Object>());
  EXPECT_TRUE(b->IsInstance<TNumberObj>());
  EXPECT_TRUE(!b->IsInstance<TIntObj>());
  EXPECT_TRUE(b->IsInstance<TFloatObj>());
}

TEST(ObjectRef, as) {
  ObjectRef a = TInt(10);
  ObjectRef b = TFloat(20);
  // nullable object
  ObjectRef c(nullptr);

  EXPECT_TRUE(a.as<TIntObj>() != nullptr);
  EXPECT_TRUE(a.as<TFloatObj>() == nullptr);
  EXPECT_TRUE(a.as<TNumberObj>() != nullptr);

  EXPECT_TRUE(b.as<TIntObj>() == nullptr);
  EXPECT_TRUE(b.as<TFloatObj>() != nullptr);
  EXPECT_TRUE(b.as<TNumberObj>() != nullptr);

  EXPECT_TRUE(c.as<TIntObj>() == nullptr);
  EXPECT_TRUE(c.as<TFloatObj>() == nullptr);
  EXPECT_TRUE(c.as<TNumberObj>() == nullptr);
  auto null_number = c.as<TNumber>();
  ASSERT_TRUE(null_number.has_value()) << "Expected nullable null ObjectRef cast to succeed";
  EXPECT_TRUE(!(*null_number).defined());  // NOLINT(bugprone-unchecked-optional-access)
  EXPECT_TRUE(!c.as<TInt>().has_value());

  EXPECT_EQ(a.as<TIntObj>()->value, 10);
  EXPECT_EQ(b.as<TFloatObj>()->value, 20);
}

TEST(ObjectRef, AsUsesTypeTraitsCheckAnyStrict) {
  ObjectRef a = TInt(10);
  ObjectRef b = TFloat(20);

  auto int_like = a.as<TIntOrFloatRef>();
  ASSERT_TRUE(int_like.has_value()) << "Expected TIntOrFloatRef cast from TInt to succeed";
  EXPECT_TRUE((*int_like).as<TIntObj>() != nullptr);  // NOLINT(bugprone-unchecked-optional-access)

  auto float_like = b.as<TIntOrFloatRef>();
  ASSERT_TRUE(float_like.has_value()) << "Expected TIntOrFloatRef cast from TFloat to succeed";
  EXPECT_NE((*float_like).as<TFloatObj>(), nullptr);  // NOLINT(bugprone-unchecked-optional-access)
}

TEST(ObjectRef, GetRefUsesObjectRefContainment) {
  ObjectPtr<TIntObj> int_object = make_object<TIntObj>(10);
  TIntOrFloatRef int_or_float = GetRef<TIntOrFloatRef>(int_object.get());

  ASSERT_NE(int_or_float.as<TIntObj>(), nullptr);
  EXPECT_EQ(int_or_float.as<TIntObj>()->value, 10);
}

TEST(ObjectRef, AsOrThrow) {
  ObjectRef a = TInt(10);
  ObjectRef b = TFloat(20);
  ObjectRef c(nullptr);
  const ObjectRef const_a = TInt(30);
  ObjectRef movable_as = TInt(40);
  ObjectRef movable_as_or_throw = TInt(50);

  EXPECT_EQ(a.as<TIntObj>()->value, 10);
  EXPECT_EQ(a.as_or_throw<TInt>()->value, 10);
  EXPECT_EQ(b.as<TFloatObj>()->value, 20);
  EXPECT_TRUE(!c.as_or_throw<TNumber>().defined());
  auto const_as = const_a.as<TInt>();
  ASSERT_TRUE(const_as.has_value()) << "Expected const ObjectRef as<TInt>() to succeed";
  EXPECT_EQ((*const_as).get()->value, 30);  // NOLINT(bugprone-unchecked-optional-access)
  EXPECT_EQ(const_a.as_or_throw<TInt>()->value, 30);

  auto moved_as = std::move(movable_as).as<TInt>();
  ASSERT_TRUE(moved_as.has_value()) << "Expected rvalue ObjectRef as<TInt>() to succeed";
  EXPECT_EQ((*moved_as).get()->value, 40);  // NOLINT(bugprone-unchecked-optional-access)
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  EXPECT_FALSE(movable_as.defined());

  EXPECT_EQ(std::move(movable_as_or_throw).as_or_throw<TInt>()->value, 50);
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  EXPECT_FALSE(movable_as_or_throw.defined());

  EXPECT_THROW(
      {
        try {
          [[maybe_unused]] auto value = a.as_or_throw<TFloat>();
        } catch (const Error& error) {
          EXPECT_EQ(error.kind(), "TypeError");
          std::string what = error.what();
          EXPECT_NE(what.find("Cannot treat type `test.Int` as type `test.Float`"),
                    std::string::npos);
          throw;
        }
      },
      ::tvm::ffi::Error);

  EXPECT_THROW(
      {
        try {
          [[maybe_unused]] auto value = c.as_or_throw<TInt>();
        } catch (const Error& error) {
          EXPECT_EQ(error.kind(), "TypeError");
          std::string what = error.what();
          EXPECT_NE(what.find("Cannot treat type `None` as type `test.Int`"), std::string::npos);
          throw;
        }
      },
      ::tvm::ffi::Error);
}

TEST(ObjectRef, UnsafeInit) {
  ObjectRef a(UnsafeInit{});
  EXPECT_TRUE(a.get() == nullptr);

  TInt b(UnsafeInit{});
  EXPECT_TRUE(b.get() == nullptr);
}

TEST(Object, CAPIAccessor) {
  ObjectRef a = TInt(10);
  TVMFFIObjectHandle obj = details::ObjectUnsafe::RawObjectPtrFromObjectRef(a);
  int32_t type_index = TVMFFIObjectGetTypeIndex(obj);
  EXPECT_EQ(type_index, TIntObj::RuntimeTypeIndex());
}

TEST(Object, WeakObjectPtr) {
  // Test basic construction from ObjectPtr
  ObjectPtr<TIntObj> strong_ptr = make_object<TIntObj>(42);
  WeakObjectPtr<TIntObj> weak_ptr(strong_ptr);

  EXPECT_EQ(strong_ptr.use_count(), 1);
  EXPECT_FALSE(weak_ptr.expired());
  EXPECT_EQ(weak_ptr.use_count(), 1);

  // Test lock() when object is still alive
  ObjectPtr<TIntObj> locked_ptr = weak_ptr.lock();
  EXPECT_TRUE(locked_ptr != nullptr);
  EXPECT_EQ(locked_ptr->value, 42);
  EXPECT_EQ(strong_ptr.use_count(), 2);
  EXPECT_EQ(weak_ptr.use_count(), 2);

  // Test lock() when object is expired
  strong_ptr.reset();
  locked_ptr.reset();
  EXPECT_TRUE(weak_ptr.expired());
  EXPECT_EQ(weak_ptr.use_count(), 0);

  ObjectPtr<TIntObj> expired_lock = weak_ptr.lock();
  EXPECT_TRUE(expired_lock == nullptr);
}

TEST(Object, WeakObjectPtrAssignment) {
  // Test copy construction
  ObjectPtr<TIntObj> new_strong = make_object<TIntObj>(100);
  WeakObjectPtr<TIntObj> weak1(new_strong);
  WeakObjectPtr<TIntObj> weak2(weak1);

  EXPECT_EQ(new_strong.use_count(), 1);
  EXPECT_FALSE(weak1.expired());
  EXPECT_FALSE(weak2.expired());
  EXPECT_EQ(weak1.use_count(), 1);
  EXPECT_EQ(weak2.use_count(), 1);

  // Test move construction
  WeakObjectPtr<TIntObj> weak3(std::move(weak1));
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  EXPECT_TRUE(weak1.expired());  // weak1 should be moved from
  EXPECT_FALSE(weak3.expired());
  EXPECT_EQ(weak3.use_count(), 1);

  // Test assignment
  WeakObjectPtr<TIntObj> weak4;
  weak4 = weak2;
  EXPECT_FALSE(weak2.expired());
  EXPECT_FALSE(weak4.expired());
  EXPECT_EQ(weak2.use_count(), 1);
  EXPECT_EQ(weak4.use_count(), 1);

  // Test move assignment
  WeakObjectPtr<TIntObj> weak5;
  weak5 = std::move(weak2);
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  EXPECT_TRUE(weak2.expired());  // weak2 should be moved from
  EXPECT_FALSE(weak5.expired());
  EXPECT_EQ(weak5.use_count(), 1);

  // Test reset()
  weak3.reset();
  EXPECT_TRUE(weak3.expired());
  EXPECT_EQ(weak3.use_count(), 0);

  // Test swap()
  ObjectPtr<TIntObj> strong_a = make_object<TIntObj>(200);
  ObjectPtr<TIntObj> strong_b = make_object<TIntObj>(300);
  WeakObjectPtr<TIntObj> weak_a(strong_a);
  WeakObjectPtr<TIntObj> weak_b(strong_b);

  weak_a.swap(weak_b);
  EXPECT_EQ(weak_a.lock()->value, 300);
  EXPECT_EQ(weak_b.lock()->value, 200);

  // Test construction from nullptr
  WeakObjectPtr<TIntObj> null_weak(nullptr);
  EXPECT_TRUE(null_weak.expired());
  EXPECT_EQ(null_weak.use_count(), 0);
  EXPECT_TRUE(null_weak.lock() == nullptr);

  // Test inheritance compatibility
  ObjectPtr<TNumberObj> number_ptr = make_object<TIntObj>(500);
  WeakObjectPtr<TNumberObj> number_weak(number_ptr);

  EXPECT_FALSE(number_weak.expired());
  EXPECT_EQ(number_weak.use_count(), 1);

  // Test that weak references don't prevent object deletion
  ObjectPtr<TIntObj> temp_strong = make_object<TIntObj>(999);
  WeakObjectPtr<TIntObj> temp_weak(temp_strong);

  EXPECT_FALSE(temp_weak.expired());
  temp_strong.reset();
  EXPECT_TRUE(temp_weak.expired());
  EXPECT_TRUE(temp_weak.lock() == nullptr);

  // Test multiple weak references
  ObjectPtr<TIntObj> multi_strong = make_object<TIntObj>(777);
  WeakObjectPtr<TIntObj> multi_weak1(multi_strong);
  WeakObjectPtr<TIntObj> multi_weak2(multi_strong);
  WeakObjectPtr<TIntObj> multi_weak3(multi_strong);

  EXPECT_EQ(multi_strong.use_count(), 1);
  EXPECT_FALSE(multi_weak1.expired());
  EXPECT_FALSE(multi_weak2.expired());
  EXPECT_FALSE(multi_weak3.expired());

  // All weak references should be able to lock
  ObjectPtr<TIntObj> lock1 = multi_weak1.lock();
  ObjectPtr<TIntObj> lock2 = multi_weak2.lock();
  ObjectPtr<TIntObj> lock3 = multi_weak3.lock();

  EXPECT_EQ(multi_strong.use_count(), 4);
  EXPECT_EQ(lock1->value, 777);
  EXPECT_EQ(lock2->value, 777);
  EXPECT_EQ(lock3->value, 777);
}

TEST(Object, OpaqueObject) {
  thread_local int deleter_trigger_counter = 0;
  struct DummyOpaqueObject {
    int value;
    explicit DummyOpaqueObject(int value) : value(value) {}

    static void Deleter(void* handle) {
      deleter_trigger_counter++;
      delete static_cast<DummyOpaqueObject*>(handle);
    }
  };
  TVMFFIObjectHandle handle = nullptr;
  TVM_FFI_CHECK_SAFE_CALL(TVMFFIObjectCreateOpaque(new DummyOpaqueObject(10), kTVMFFIOpaquePyObject,
                                                   DummyOpaqueObject::Deleter, &handle));
  ObjectPtr<Object> a =
      details::ObjectUnsafe::ObjectPtrFromOwned<Object>(static_cast<Object*>(handle));
  EXPECT_EQ(a->type_index(), kTVMFFIOpaquePyObject);
  EXPECT_EQ(static_cast<DummyOpaqueObject*>(TVMFFIOpaqueObjectGetCellPtr(a.get())->handle)->value,
            10);
  EXPECT_EQ(a.use_count(), 1);
  EXPECT_EQ(deleter_trigger_counter, 0);
  a.reset();
  EXPECT_EQ(deleter_trigger_counter, 1);
}

using NumberPtr = ObjectPtr<TNumberObj>;

NumberPtr MakeInt(int64_t value) { return make_object<TIntObj>(value); }

NumberPtr MakeFloat(double value) { return make_object<TFloatObj>(value); }

TEST(ObjectPtrStorage, Array) {
  NumberPtr first = MakeInt(1);
  NumberPtr second = MakeFloat(2.0);
  NumberPtr null;
  Array<NumberPtr> array{first, null};

  EXPECT_EQ(array.size(), 2U);
  EXPECT_EQ(array[0], first);
  EXPECT_EQ(array[1], nullptr);

  array.Set(1, second);
  array.push_back(null);
  EXPECT_EQ(array[1], second);
  EXPECT_EQ(array[2], nullptr);

  size_t defined = 0;
  for (NumberPtr item : array) {
    defined += item != nullptr;
  }
  EXPECT_EQ(defined, 2U);

  Array<NumberPtr> roundtrip = Any(array).cast<Array<NumberPtr>>();
  EXPECT_TRUE(roundtrip.same_as(array));
  EXPECT_EQ(roundtrip[0], first);
  EXPECT_EQ(roundtrip[1], second);
  EXPECT_EQ(roundtrip[2], nullptr);

  ObjectPtr<TIntObj> derived = make_object<TIntObj>(3);
  Array<ObjectPtr<TIntObj>> derived_array{derived};
  Array<NumberPtr> upcast_array = derived_array;
  EXPECT_TRUE(upcast_array.same_as(derived_array));
  EXPECT_EQ(upcast_array[0].get(), static_cast<TNumberObj*>(derived.get()));
}

TEST(ObjectPtrStorage, List) {
  NumberPtr first = MakeInt(1);
  NumberPtr second = MakeFloat(2.0);
  NumberPtr null;
  List<NumberPtr> list{first, null};

  list.Set(1, second);
  list.insert(list.begin() + 1, null);
  list.push_back(first);
  EXPECT_EQ(list.size(), 4U);
  EXPECT_EQ(list[0], first);
  EXPECT_EQ(list[1], nullptr);
  EXPECT_EQ(list[2], second);
  EXPECT_EQ(list[3], first);

  List<NumberPtr> roundtrip = Any(list).cast<List<NumberPtr>>();
  EXPECT_TRUE(roundtrip.same_as(list));
  roundtrip.Set(1, second);
  EXPECT_EQ(list[1], second);
}

TEST(ObjectPtrStorage, Map) {
  NumberPtr key = MakeInt(1);
  NumberPtr value = MakeFloat(2.0);
  NumberPtr null;
  Map<NumberPtr, NumberPtr> map{{key, value}};

  EXPECT_EQ(map.count(key), 1U);
  EXPECT_EQ(map[key], value);
  map.Set(key, null);
  map.Set(null, key);
  EXPECT_EQ(map[key], nullptr);
  EXPECT_EQ(map[null], key);

  size_t entries = 0;
  for (const auto& [stored_key, stored_value] : map) {
    if (stored_key == key) {
      EXPECT_EQ(stored_value, nullptr);
    }
    if (stored_key == nullptr) {
      EXPECT_EQ(stored_value, key);
    }
    ++entries;
  }
  EXPECT_EQ(entries, 2U);

  Map<NumberPtr, NumberPtr> roundtrip = Any(map).cast<Map<NumberPtr, NumberPtr>>();
  EXPECT_TRUE(roundtrip.same_as(map));
  EXPECT_EQ(roundtrip[null], key);
}

TEST(ObjectPtrStorage, Dict) {
  NumberPtr key = MakeInt(1);
  NumberPtr value = MakeFloat(2.0);
  NumberPtr null;
  Dict<NumberPtr, NumberPtr> dict{{key, value}, {null, key}};
  Dict<NumberPtr, NumberPtr> alias = dict;

  EXPECT_EQ(dict[key], value);
  EXPECT_EQ(dict[null], key);
  dict.Set(key, null);
  EXPECT_EQ(alias[key], nullptr);

  auto found = dict.Get(null);
  ASSERT_TRUE(found.has_value());
  EXPECT_EQ(found.value(), key);  // NOLINT(bugprone-unchecked-optional-access)

  Dict<NumberPtr, NumberPtr> roundtrip = Any(dict).cast<Dict<NumberPtr, NumberPtr>>();
  EXPECT_TRUE(roundtrip.same_as(dict));
  roundtrip.erase(null);
  EXPECT_EQ(dict.count(null), 0U);
}

TEST(ObjectPtrStorage, Tuple) {
  NumberPtr first = MakeInt(1);
  NumberPtr second = MakeFloat(2.0);
  NumberPtr null;
  Tuple<NumberPtr, NumberPtr> tuple(first, null);

  EXPECT_EQ(tuple.get<0>(), first);
  EXPECT_EQ(tuple.get<1>(), nullptr);
  tuple.Set<1>(second);
  EXPECT_EQ(tuple.get<1>(), second);

  Tuple<NumberPtr, NumberPtr> roundtrip = Any(tuple).cast<Tuple<NumberPtr, NumberPtr>>();
  EXPECT_TRUE(roundtrip.same_as(tuple));
  EXPECT_EQ(roundtrip.get<0>(), first);
  EXPECT_EQ(roundtrip.get<1>(), second);
}

TEST(ObjectPtrStorage, Variant) {
  using NumberOrInt = Variant<NumberPtr, int64_t>;
  NumberPtr first = MakeInt(1);
  NumberPtr null;

  NumberOrInt variant = first;
  EXPECT_EQ(variant.get<NumberPtr>(), first);
  NumberOrInt roundtrip = Any(variant).cast<NumberOrInt>();
  EXPECT_EQ(roundtrip.get<NumberPtr>(), first);

  variant = int64_t{2};
  EXPECT_EQ(variant.get<int64_t>(), 2);
  variant = null;
  EXPECT_EQ(variant.get<NumberPtr>(), nullptr);
}

TEST(ObjectPtrStorage, OptionalAndVariantComposition) {
  using OptionalNumber = Optional<NumberPtr>;
  using NestedOptionalNumber = Optional<OptionalNumber>;
  using OptionalNumberOrInt = Variant<OptionalNumber, int64_t>;
  using OptionalNumberOrIntValue = Optional<Variant<NumberPtr, int64_t>>;

  NumberPtr number = MakeInt(1);
  OptionalNumber optional_number = number;
  Array<OptionalNumber> array{optional_number, std::nullopt};
  EXPECT_TRUE(array[0].has_value());
  EXPECT_EQ(array[0].value(), number);
  EXPECT_FALSE(array[1].has_value());

  OptionalNumberOrInt variant = optional_number;
  OptionalNumber variant_value = variant.get<OptionalNumber>();
  ASSERT_TRUE(variant_value.has_value());
  EXPECT_EQ(variant_value.value(), number);

  OptionalNumberOrIntValue optional_variant = Variant<NumberPtr, int64_t>(number);
  Any encoded = optional_variant;
  OptionalNumberOrIntValue decoded = encoded.cast<OptionalNumberOrIntValue>();
  ASSERT_TRUE(decoded.has_value());
  EXPECT_EQ(decoded.value().get<NumberPtr>(), number);

  OptionalNumber absent = std::nullopt;
  OptionalNumber present_null = NumberPtr();
  EXPECT_FALSE(Any(absent).cast<OptionalNumber>().has_value());
  EXPECT_FALSE(Any(present_null).cast<OptionalNumber>().has_value());

  NestedOptionalNumber nested = optional_number;
  NestedOptionalNumber nested_roundtrip = Any(nested).cast<NestedOptionalNumber>();
  ASSERT_TRUE(nested_roundtrip.has_value());
  ASSERT_TRUE(nested_roundtrip.value().has_value());
  EXPECT_EQ(nested_roundtrip.value().value(), number);

  NestedOptionalNumber present_absent = OptionalNumber(std::nullopt);
  EXPECT_FALSE(Any(present_absent).cast<NestedOptionalNumber>().has_value());

  EXPECT_EQ(TypeTraits<OptionalNumber>::TypeSchema(),
            R"({"type":"Optional","args":[{"type":"Optional","args":[{"type":"test.Number"}]}]})");
  EXPECT_EQ(
      TypeTraits<OptionalNumberOrInt>::TypeSchema(),
      R"({"type":"Variant","args":[{"type":"Optional","args":[{"type":"Optional","args":[{"type":"test.Number"}]}]},{"type":"int"}]})");
}

TEST(ObjectPtrStorage, Expected) {
  NumberPtr number = MakeInt(1);
  Expected<NumberPtr> success = number;
  EXPECT_TRUE(success.is_ok());
  EXPECT_EQ(success.value(), number);

  Expected<NumberPtr> success_roundtrip = Any(success).cast<Expected<NumberPtr>>();
  EXPECT_TRUE(success_roundtrip.is_ok());
  EXPECT_EQ(success_roundtrip.value(), number);

  Expected<NumberPtr> failure = Error("ValueError", "expected failure", "");
  Expected<NumberPtr> failure_roundtrip = Any(failure).cast<Expected<NumberPtr>>();
  EXPECT_TRUE(failure_roundtrip.is_err());
  EXPECT_EQ(failure_roundtrip.error().kind(), "ValueError");
}

using NumberArc = tvm::ffi::Arc<TNumberObj>;

NumberArc MakeArcInt(int64_t value) { return make_arc<TIntObj>(value); }

NumberArc MakeArcFloat(double value) { return make_arc<TFloatObj>(value); }

TEST(ArcStorage, Containers) {
  NumberArc first = MakeArcInt(1);
  NumberArc second = MakeArcFloat(2.0);

  Array<NumberArc> array{first};
  array.push_back(second);
  EXPECT_EQ(array[1], second);

  List<NumberArc> list{first};
  list.push_back(second);
  EXPECT_EQ(list[1], second);

  Map<String, NumberArc> map{{"value", first}};
  map.Set("value", second);
  EXPECT_EQ(map["value"], second);

  Dict<String, NumberArc> dict{{"value", first}};
  dict.Set("value", second);
  EXPECT_EQ(dict["value"], second);

  Tuple<NumberArc, NumberArc> tuple(first, first);
  tuple.Set<1>(second);
  EXPECT_EQ(tuple.get<1>(), second);

  Variant<NumberArc, int64_t> variant = int64_t{3};
  variant = first;
  EXPECT_EQ(variant.get<NumberArc>(), first);

  Expected<NumberArc> expected{first};
  ASSERT_TRUE(expected.is_ok());
  EXPECT_EQ(expected.value(), first);

  EXPECT_EQ(TypeTraits<List<NumberArc>>::TypeSchema(),
            R"({"type":"ffi.List","args":[{"type":"test.Number"}]})");
  EXPECT_EQ((TypeTraits<Map<String, NumberArc>>::TypeSchema()),
            R"({"type":"ffi.Map","args":[{"type":"ffi.String"},{"type":"test.Number"}]})");
  EXPECT_EQ((TypeTraits<Tuple<NumberArc, int64_t>>::TypeSchema()),
            R"({"type":"Tuple","args":[{"type":"test.Number"},{"type":"int"}]})");
}

}  // namespace
