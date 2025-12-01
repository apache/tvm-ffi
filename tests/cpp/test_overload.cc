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
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/extra/overload.h>
#include <tvm/ffi/object.h>
#include <tvm/ffi/reflection/access_path.h>
#include <tvm/ffi/reflection/accessor.h>
#include <tvm/ffi/reflection/creator.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>

namespace {

using namespace tvm::ffi;

struct TestOverloadObj : public Object {
  explicit TestOverloadObj(int32_t x) : type(Type::INT) {}
  explicit TestOverloadObj(float y) : type(Type::FLOAT) {}
  static int add_one_int(int x) { return x + 1; }
  static float add_one_float(float x) { return x + 1.0f; }

  template <typename T>
  auto holds(T) {
    if constexpr (std::is_same_v<T, int32_t>) {
      return type == Type::INT;
    } else if constexpr (std::is_same_v<T, float>) {
      return type == Type::FLOAT;
    } else {
      static_assert(sizeof(T) == 0, "Unsupported type");
    }
  }

  enum class Type { INT, FLOAT } type;
  TVM_FFI_DECLARE_OBJECT_INFO("test.TestOverloadObj", TestOverloadObj, Object);
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::OverloadObjectDef<TestOverloadObj>()
      .def(refl::init<int32_t>())
      .def(refl::init<float>())
      .def("hold_same_type", &TestOverloadObj::holds<int32_t>)
      .def("hold_same_type", &TestOverloadObj::holds<float>)
      .def_static("add_one_static", &TestOverloadObj::add_one_int)
      .def_static("add_one_static", &TestOverloadObj::add_one_float);
}

TEST(Reflection, CallOverloadedInitMethod) {
  Function init_method = reflection::GetMethod("test.TestOverloadObj", "__ffi_init__");
  Any obj_a = init_method(10);  // choose the int constructor
  EXPECT_TRUE(obj_a.as<TestOverloadObj>() != nullptr);
  EXPECT_EQ(obj_a.as<TestOverloadObj>()->type, TestOverloadObj::Type::INT);
  Any obj_b = init_method(3.14f);  // choose the float constructor
  EXPECT_TRUE(obj_b.as<TestOverloadObj>() != nullptr);
  EXPECT_EQ(obj_b.as<TestOverloadObj>()->type, TestOverloadObj::Type::FLOAT);
}

TEST(Reflection, CallOverloadedMethod) {
  Function init_method = reflection::GetMethod("test.TestOverloadObj", "__ffi_init__");
  Function hold_same_type = reflection::GetMethod("test.TestOverloadObj", "hold_same_type");
  Any obj_a = init_method(10);  // choose the int constructor
  Any res_a = hold_same_type(obj_a, 20);
  EXPECT_EQ(res_a.as<bool>(), true);
  Any res_b = hold_same_type(obj_a, 3.14f);
  EXPECT_EQ(res_b.as<bool>(), false);
}

TEST(Reflection, CallOverloadedStaticMethod) {
  Function add_one = reflection::GetMethod("test.TestOverloadObj", "add_one_static");
  Any res_a = add_one(20);
  EXPECT_EQ(res_a.as<int>(), 21);
  Any res_b = add_one(1.0f);
  static_assert(1.0f + 1.0f == 2.0f);
  EXPECT_EQ(res_b.as<float>(), 2.0f);
}

}  // namespace
