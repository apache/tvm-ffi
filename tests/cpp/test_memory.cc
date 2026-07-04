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
#include <tvm/ffi/memory.h>

#include <cstdint>
#include <stdexcept>

#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#define TVM_FFI_TEST_WITH_ASAN 1
#endif
#elif defined(__SANITIZE_ADDRESS__)
#define TVM_FFI_TEST_WITH_ASAN 1
#endif

#ifdef TVM_FFI_TEST_WITH_ASAN
#include <sanitizer/asan_interface.h>
#endif

namespace {

using namespace tvm::ffi;

class FactoryConstructionError : public std::runtime_error {
 public:
  explicit FactoryConstructionError(const char* message) : std::runtime_error(message) {}
};

class alignas(64) ThrowingObject : public Object {
 public:
  explicit ThrowingObject(bool should_throw) {
    ++constructor_count;
    last_address = this;
    if (should_throw) {
      throw FactoryConstructionError("make_object constructor failure");
    }
  }

  ~ThrowingObject() { ++destructor_count; }

  static void ResetCounters() {
    constructor_count = 0;
    destructor_count = 0;
    last_address = nullptr;
  }

  inline static int constructor_count = 0;
  inline static int destructor_count = 0;
  inline static void* last_address = nullptr;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("test.ThrowingObject", ThrowingObject, Object);
};

class alignas(64) ThrowingInplaceArrayObject : public Object {
 public:
  explicit ThrowingInplaceArrayObject(bool should_throw) {
    ++constructor_count;
    last_address = this;
    if (should_throw) {
      throw FactoryConstructionError("make_inplace_array_object constructor failure");
    }
  }

  ~ThrowingInplaceArrayObject() { ++destructor_count; }

  static void ResetCounters() {
    constructor_count = 0;
    destructor_count = 0;
    last_address = nullptr;
  }

  inline static int constructor_count = 0;
  inline static int destructor_count = 0;
  inline static void* last_address = nullptr;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("test.ThrowingInplaceArrayObject", ThrowingInplaceArrayObject,
                                    Object);
};

void ExpectAllocationWasReleased(void* address) {
  ASSERT_NE(address, nullptr);
#ifdef TVM_FFI_TEST_WITH_ASAN
  EXPECT_NE(__asan_address_is_poisoned(address), 0);
#else
  GTEST_LOG_(INFO) << "Allocation release is checked by the sanitizer-enabled test build";
#endif
}

TEST(Memory, MakeObjectReleasesAllocationWhenConstructorThrows) {
  ThrowingObject::ResetCounters();

  try {
    [[maybe_unused]] ObjectPtr<ThrowingObject> object = make_object<ThrowingObject>(true);
    FAIL() << "Expected constructor failure";
  } catch (const FactoryConstructionError& error) {
    EXPECT_STREQ(error.what(), "make_object constructor failure");
  }

  EXPECT_EQ(ThrowingObject::constructor_count, 1);
  EXPECT_EQ(ThrowingObject::destructor_count, 0);
  ExpectAllocationWasReleased(ThrowingObject::last_address);
}

TEST(Memory, MakeObjectRetainsSuccessfulLifetimeAndAlignment) {
  ThrowingObject::ResetCounters();

  ObjectPtr<ThrowingObject> object = make_object<ThrowingObject>(false);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(object.get()) % alignof(ThrowingObject), 0U);
  EXPECT_EQ(ThrowingObject::constructor_count, 1);
  EXPECT_EQ(ThrowingObject::destructor_count, 0);

  object.reset();
  EXPECT_EQ(ThrowingObject::destructor_count, 1);
}

TEST(Memory, MakeInplaceArrayObjectReleasesAllocationWhenConstructorThrows) {
  ThrowingInplaceArrayObject::ResetCounters();

  try {
    [[maybe_unused]] ObjectPtr<ThrowingInplaceArrayObject> object =
        make_inplace_array_object<ThrowingInplaceArrayObject, std::uint64_t>(4, true);
    FAIL() << "Expected constructor failure";
  } catch (const FactoryConstructionError& error) {
    EXPECT_STREQ(error.what(), "make_inplace_array_object constructor failure");
  }

  EXPECT_EQ(ThrowingInplaceArrayObject::constructor_count, 1);
  EXPECT_EQ(ThrowingInplaceArrayObject::destructor_count, 0);
  ExpectAllocationWasReleased(ThrowingInplaceArrayObject::last_address);
}

TEST(Memory, MakeInplaceArrayObjectRetainsSuccessfulLifetimeAndAlignment) {
  ThrowingInplaceArrayObject::ResetCounters();

  ObjectPtr<ThrowingInplaceArrayObject> object =
      make_inplace_array_object<ThrowingInplaceArrayObject, std::uint64_t>(4, false);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(object.get()) % alignof(ThrowingInplaceArrayObject),
            0U);
  EXPECT_EQ(ThrowingInplaceArrayObject::constructor_count, 1);
  EXPECT_EQ(ThrowingInplaceArrayObject::destructor_count, 0);

  object.reset();
  EXPECT_EQ(ThrowingInplaceArrayObject::destructor_count, 1);
}

}  // namespace
