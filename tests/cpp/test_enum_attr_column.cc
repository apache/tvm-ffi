
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
#include <tvm/ffi/enum.h>
#include <tvm/ffi/reflection/enum_def.h>

#include <cstdint>
#include <string>
#include <string_view>
#include <type_traits>

namespace {
using namespace tvm::ffi;
namespace refl = tvm::ffi::reflection;

class AttrEnumObj : public EnumObj {
 public:
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("test.AttrEnum", AttrEnumObj, EnumObj);
};

class LateAttrEnumObj : public EnumObj {
 public:
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("test.LateAttrEnum", LateAttrEnumObj, EnumObj);
};

class OtherAttrEnumObj : public IntEnumObj {
 public:
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("test.OtherAttrEnum", OtherAttrEnumObj, IntEnumObj);
};

constexpr refl::AttrKey<int64_t> kCode{"code"};
constexpr refl::AttrKey<bool> kEnabled{"enabled"};
constexpr refl::AttrKey<String> kDescription{"description"};
constexpr refl::AttrKey<refl::NativeFunctionView<int(int)>> kIncrement{"increment"};

int Increment(int value) { return value + 1; }

bool constructed_during_registration = false;

TVM_FFI_STATIC_INIT_BLOCK() {
  refl::EnumDef<AttrEnumObj>("Alpha")
      .set_attr(kCode, 7)
      .set_attr(kEnabled, false)
      .set_attr(kDescription, "first")
      .set_native_fn_attr<&Increment>(kIncrement);
  refl::EnumAttrColumn column(AttrEnumObj::_GetOrAllocRuntimeTypeIndex(), kCode);
  constructed_during_registration = true;
  refl::EnumDef<AttrEnumObj>("Missing");  // NOLINT(bugprone-unused-raii)
  refl::EnumDef<AttrEnumObj>("None").set_attr(kCode.name, nullptr);
  refl::EnumDef<AttrEnumObj>("WrongType").set_attr(kCode.name, true);
  refl::EnumDef<OtherAttrEnumObj>("Alpha", 0).set_attr(kCode, 99);
}

TEST(EnumAttrColumn, TypedValues) {
  int32_t type_index = AttrEnumObj::RuntimeTypeIndex();
  Enum alpha = EnumObj::_GetByStrIndex<AttrEnumObj>("Alpha");
  const refl::EnumAttrColumn code(type_index, kCode);
  static_assert(std::is_same_v<decltype(code[alpha]), std::optional<int64_t>>);
  EXPECT_TRUE(constructed_during_registration);
  EXPECT_EQ(code[alpha], 7);
  EXPECT_EQ(code[EnumObj::_GetByIntIndex<AttrEnumObj>(0)], 7);
  EXPECT_EQ(refl::EnumAttrColumn(type_index, kEnabled)[alpha], false);
  EXPECT_EQ(refl::EnumAttrColumn(type_index, kDescription)[alpha], String("first"));
  auto increment = refl::EnumAttrColumn(type_index, kIncrement)[alpha];
  ASSERT_TRUE(increment.has_value());
  if (increment) {
    EXPECT_EQ((*increment)(4), 5);
  }
  EXPECT_EQ(refl::EnumAttrColumn<int64_t>(type_index, "code")[alpha], 7);
  EXPECT_EQ(refl::EnumAttrColumn<int64_t>(type_index, std::string_view("code_suffix", 4))[alpha],
            7);
  EXPECT_ANY_THROW(refl::EnumAttrColumn<int64_t>(type_index, "cod"));
}

TEST(EnumAttrColumn, MissingAndWrongTypeEntries) {
  const refl::EnumAttrColumn code(AttrEnumObj::RuntimeTypeIndex(), kCode);
  EXPECT_FALSE(code[EnumObj::_GetByStrIndex<AttrEnumObj>("Missing")].has_value());
  EXPECT_FALSE(code[EnumObj::_GetByStrIndex<AttrEnumObj>("None")].has_value());
  EXPECT_ANY_THROW(code[EnumObj::_GetByStrIndex<AttrEnumObj>("WrongType")]);
  // Matching names and integer indices in a different enum do not alias.
  EXPECT_FALSE(code[EnumObj::_GetByStrIndex<OtherAttrEnumObj>("Alpha")].has_value());
}

TEST(EnumAttrColumn, RegistrationOrderAndUpdates) {
  int32_t type_index = LateAttrEnumObj::RuntimeTypeIndex();
  refl::TypeAttrColumn states(type_attr::kEnumState);
  EXPECT_EQ(states[type_index], nullptr);
  EXPECT_ANY_THROW(refl::EnumAttrColumn(type_index, kCode));
  EXPECT_EQ(states[type_index], nullptr);

  refl::EnumDef<LateAttrEnumObj> first("First");
  Enum first_entry = EnumObj::_GetByStrIndex<LateAttrEnumObj>("First");
  EXPECT_ANY_THROW(refl::EnumAttrColumn(type_index, kCode));
  EXPECT_TRUE(states[type_index].as_or_throw<EnumState>()->attrs.empty());

  first.set_attr(kCode, 1);
  const refl::EnumAttrColumn column(type_index, kCode);
  // Exercise an independent handle sharing the same live dictionary.
  auto copied_column = column;  // NOLINT(performance-unnecessary-copy-initialization)
  EXPECT_EQ(column[first_entry], 1);
  first.set_attr(kCode, 2);
  EXPECT_EQ(column[first_entry], 2);
  EXPECT_EQ(copied_column[first_entry], 2);
  first.set_attr(kCode.name, nullptr);
  EXPECT_FALSE(column[first_entry].has_value());
  first.set_attr(kCode, 2);

  refl::EnumDef<LateAttrEnumObj> second("Second");
  Enum second_entry = EnumObj::_GetByStrIndex<LateAttrEnumObj>("Second");
  EXPECT_FALSE(column[second_entry].has_value());
  second.set_attr(kCode, 3);
  EXPECT_EQ(column[second_entry], 3);

  // Both the outer attribute dictionary and the cached column may grow and rehash.
  for (int64_t i = 0; i < 128; ++i) {
    first.set_attr(("extra_attr_" + std::to_string(i)).c_str(), i);
    refl::EnumDef<LateAttrEnumObj>(("Extra" + std::to_string(i)).c_str()).set_attr(kCode, i);
  }
  EXPECT_EQ(column[first_entry], 2);
  EXPECT_EQ(copied_column[second_entry], 3);
  EXPECT_EQ(column[EnumObj::_GetByStrIndex<LateAttrEnumObj>("Extra127")], 127);
}

}  // namespace
