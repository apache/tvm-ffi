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
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/structural_map.h>
#include <tvm/ffi/extra/visit_error_context.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/object.h>
#include <tvm/ffi/string.h>

#include <cstdint>
#include <string>
#include <utility>

#include "../testing_object.h"

namespace {

using namespace tvm::ffi;
using namespace tvm::ffi::testing;

enum class TransformAction : int32_t {
  kKeep,
  kChange,
  kError,
};

using TestMapper = StructuralMapper;

ObjectRef MakeMapLeaf(int64_t value, TransformAction action = TransformAction::kKeep) {
  static Function make_leaf = Function::GetGlobalRequired("testing.make_structural_map_leaf");
  return make_leaf(value, static_cast<int32_t>(action)).cast<ObjectRef>();
}

ObjectRef MakeFunctionLeaf(int64_t value, TransformAction action = TransformAction::kKeep) {
  static Function make_leaf =
      Function::GetGlobalRequired("testing.make_structural_map_function_leaf");
  return make_leaf(value, static_cast<int32_t>(action)).cast<ObjectRef>();
}

int64_t MapLeafValue(const ObjectRef& value) {
  static Function get_value = Function::GetGlobalRequired("testing.structural_map_leaf_value");
  return get_value(value).cast<int64_t>();
}

Array<String> MapperTrace() {
  static Function get_trace = Function::GetGlobalRequired("testing.structural_mapper_trace");
  return get_trace().cast<Array<String>>();
}

void ClearMapperTrace() {
  static Function clear_trace =
      Function::GetGlobalRequired("testing.structural_mapper_clear_trace");
  clear_trace();
}

TestMapper MakeTestMapper() {
  ClearMapperTrace();
  return StructuralMapper();
}

void ExpectTrace(std::initializer_list<const char*> expected) {
  Array<String> trace = MapperTrace();
  ASSERT_EQ(trace.size(), expected.size());
  size_t i = 0;
  for (const char* item : expected) {
    EXPECT_EQ(trace[i], item);
    ++i;
  }
}

// ---------------------------------------------------------------------------
// StructuralMapper behavior.
// ---------------------------------------------------------------------------

TEST(StructuralMapper, ReturnsPODValuesUnchanged) {
  StructuralMapper mapper;

  Expected<Any> mapped = mapper->MapExpected(int64_t{42});
  ASSERT_TRUE(mapped.is_ok());
  EXPECT_EQ(std::move(mapped).value().cast<int64_t>(), 42);

  Expected<Any> mutated = mapper->InplaceMutateExpected(3.5);
  ASSERT_TRUE(mutated.is_ok());
  EXPECT_EQ(std::move(mutated).value().cast<double>(), 3.5);

  Expected<Any> combined = mapper->MapOrInplaceMutateExpected(true);
  ASSERT_TRUE(combined.is_ok());
  EXPECT_TRUE(std::move(combined).value().cast<bool>());
}

TEST(StructuralMapper, MapReturnsOriginalWhenFieldsUnchanged) {
  ObjectRef lhs = MakeMapLeaf(1);
  TVar rhs("rhs");
  const Object* lhs_addr = lhs.get();
  const Object* rhs_addr = rhs.get();
  TPair root(std::move(lhs), std::move(rhs));
  const Object* root_addr = root.get();
  Any input = std::move(root);
  TestMapper mapper = MakeTestMapper();

  Expected<Any> result = mapper->MapExpected(std::move(input));

  ASSERT_TRUE(result.is_ok());
  TPair mapped = std::move(result).value().cast<TPair>();
  EXPECT_EQ(mapped.get(), root_addr);
  EXPECT_EQ(mapped->lhs.get(), lhs_addr);
  EXPECT_EQ(mapped->rhs.get(), rhs_addr);
  EXPECT_EQ(mapped.use_count(), 1);
  ExpectTrace({"map:1"});
}

TEST(StructuralMapper, MapCreatesShallowCopyWhenFieldChanges) {
  ObjectRef lhs = MakeMapLeaf(1, TransformAction::kChange);
  TVar rhs("rhs");
  TPair root(std::move(lhs), std::move(rhs));
  const Object* root_addr = root.get();
  const Object* lhs_addr = root->lhs.get();
  const Object* rhs_addr = root->rhs.get();
  TestMapper mapper = MakeTestMapper();

  Expected<Any> result = mapper->MapExpected(root);

  ASSERT_TRUE(result.is_ok());
  TPair mapped = std::move(result).value().cast<TPair>();
  EXPECT_NE(mapped.get(), root_addr);
  EXPECT_EQ(root->lhs.get(), lhs_addr);
  EXPECT_EQ(MapLeafValue(root->lhs), 1);
  EXPECT_NE(mapped->lhs.get(), lhs_addr);
  EXPECT_EQ(MapLeafValue(mapped->lhs), 2);
  EXPECT_EQ(mapped->rhs.get(), rhs_addr);
  EXPECT_TRUE(mapped->rhs.same_as(root->rhs));

  EXPECT_EQ(root.use_count(), 1);
  EXPECT_EQ(mapped.use_count(), 1);
  EXPECT_EQ(root->lhs.use_count(), 1);
  EXPECT_EQ(mapped->lhs.use_count(), 1);
  EXPECT_EQ(root->rhs.use_count(), 2);
  ExpectTrace({"map:1"});
}

TEST(StructuralMapper, MapCopiesOnlyChangedPath) {
  ObjectRef leaf = MakeMapLeaf(1, TransformAction::kChange);
  TVar middle_rhs("middle-rhs");
  TPair middle(std::move(leaf), std::move(middle_rhs));
  const TPairObj* middle_addr = middle.get();
  const Object* leaf_addr = middle->lhs.get();
  const Object* middle_rhs_addr = middle->rhs.get();

  TVar outer_rhs("outer-rhs");
  TPair root(std::move(middle), std::move(outer_rhs));
  const Object* root_addr = root.get();
  const Object* outer_rhs_addr = root->rhs.get();
  TestMapper mapper = MakeTestMapper();

  Expected<Any> result = mapper->MapExpected(root);

  ASSERT_TRUE(result.is_ok());
  TPair mapped = std::move(result).value().cast<TPair>();
  const TPairObj* mapped_middle = mapped->lhs.as<TPairObj>();
  ASSERT_NE(mapped_middle, nullptr);

  EXPECT_NE(mapped.get(), root_addr);
  EXPECT_NE(mapped_middle, middle_addr);
  EXPECT_NE(mapped_middle->lhs.get(), leaf_addr);
  EXPECT_EQ(MapLeafValue(mapped_middle->lhs), 2);

  EXPECT_EQ(middle_addr->lhs.get(), leaf_addr);
  EXPECT_EQ(MapLeafValue(middle_addr->lhs), 1);
  EXPECT_EQ(mapped_middle->rhs.get(), middle_rhs_addr);
  EXPECT_EQ(mapped->rhs.get(), outer_rhs_addr);
  ExpectTrace({"map:1"});
}

TEST(StructuralMapper, MapOrInplaceMutateUsesInplaceForUniqueRoot) {
  ObjectRef lhs = MakeMapLeaf(1, TransformAction::kChange);
  ObjectRef rhs = MakeMapLeaf(2);
  TPair root(std::move(lhs), std::move(rhs));
  const Object* root_addr = root.get();
  const Object* lhs_addr = root->lhs.get();
  const Object* rhs_addr = root->rhs.get();
  Any input = std::move(root);
  TestMapper mapper = MakeTestMapper();

  Expected<Any> result = mapper->MapOrInplaceMutateExpected(std::move(input));

  ASSERT_TRUE(result.is_ok());
  TPair mapped = std::move(result).value().cast<TPair>();
  EXPECT_EQ(mapped.get(), root_addr);
  EXPECT_EQ(mapped->lhs.get(), lhs_addr);
  EXPECT_EQ(MapLeafValue(mapped->lhs), 2);
  EXPECT_EQ(mapped->rhs.get(), rhs_addr);
  EXPECT_EQ(mapped.use_count(), 1);
  EXPECT_EQ(mapped->lhs.use_count(), 1);
  EXPECT_EQ(mapped->rhs.use_count(), 1);
  ExpectTrace({"inplace:1", "inplace:2"});
}

TEST(StructuralMapper, MapOrInplaceMutateUsesMapForSharedRoot) {
  ObjectRef lhs = MakeMapLeaf(1, TransformAction::kChange);
  ObjectRef rhs = MakeMapLeaf(2);
  TPair root(std::move(lhs), std::move(rhs));
  const Object* root_addr = root.get();
  const Object* lhs_addr = root->lhs.get();
  TestMapper mapper = MakeTestMapper();

  Expected<Any> result = mapper->MapOrInplaceMutateExpected(root);

  ASSERT_TRUE(result.is_ok());
  TPair mapped = std::move(result).value().cast<TPair>();
  EXPECT_NE(mapped.get(), root_addr);
  EXPECT_EQ(root->lhs.get(), lhs_addr);
  EXPECT_EQ(MapLeafValue(root->lhs), 1);
  EXPECT_NE(mapped->lhs.get(), lhs_addr);
  EXPECT_EQ(MapLeafValue(mapped->lhs), 2);
  EXPECT_TRUE(mapped->rhs.same_as(root->rhs));
  EXPECT_EQ(root.use_count(), 1);
  EXPECT_EQ(mapped.use_count(), 1);
  ExpectTrace({"map:1", "map:2"});
}

TEST(StructuralMapper, ExplicitInplaceMutateUpdatesSharedRoot) {
  ObjectRef lhs = MakeMapLeaf(1, TransformAction::kChange);
  ObjectRef rhs = MakeMapLeaf(2);
  TPair root(std::move(lhs), std::move(rhs));
  TPair alias = root;
  const Object* root_addr = root.get();
  const Object* lhs_addr = root->lhs.get();
  TestMapper mapper = MakeTestMapper();

  Expected<Any> result = mapper->InplaceMutateExpected(std::move(root));

  ASSERT_TRUE(result.is_ok());
  TPair mapped = std::move(result).value().cast<TPair>();
  EXPECT_EQ(mapped.get(), root_addr);
  EXPECT_EQ(alias.get(), root_addr);
  EXPECT_EQ(alias->lhs.get(), lhs_addr);
  EXPECT_EQ(MapLeafValue(alias->lhs), 2);
  EXPECT_EQ(mapped.use_count(), 2);
  ExpectTrace({"inplace:1", "inplace:2"});
}

void CheckInplaceFieldSelection(bool change_lhs, bool change_rhs) {
  SCOPED_TRACE(std::string("change_lhs=") + (change_lhs ? "true" : "false") +
               ", change_rhs=" + (change_rhs ? "true" : "false"));

  // Before the getter runs, lhs is shared by the parent and external_lhs while rhs is owned only
  // by the parent. The getter adds one temporary owner to each field, so lhs must map and rhs may
  // mutate in place.
  ObjectRef external_lhs =
      MakeMapLeaf(10, change_lhs ? TransformAction::kChange : TransformAction::kKeep);
  ObjectRef parent_lhs = external_lhs;
  ObjectRef rhs = MakeMapLeaf(20, change_rhs ? TransformAction::kChange : TransformAction::kKeep);
  const Object* lhs_addr = external_lhs.get();
  const Object* rhs_addr = rhs.get();
  TPair root(std::move(parent_lhs), std::move(rhs));
  const Object* root_addr = root.get();
  ASSERT_EQ(root.use_count(), 1);
  ASSERT_EQ(external_lhs.use_count(), 2);
  ASSERT_EQ(root->rhs.use_count(), 1);
  Any input = std::move(root);
  TestMapper mapper = MakeTestMapper();

  Expected<Any> result = mapper->InplaceMutateExpected(std::move(input));

  ASSERT_TRUE(result.is_ok());
  TPair mapped = std::move(result).value().cast<TPair>();
  EXPECT_EQ(mapped.get(), root_addr);
  EXPECT_EQ(mapped.use_count(), 1);

  if (change_lhs) {
    EXPECT_NE(mapped->lhs.get(), lhs_addr);
    EXPECT_EQ(MapLeafValue(external_lhs), 10);
    EXPECT_EQ(external_lhs.use_count(), 1);
    EXPECT_EQ(MapLeafValue(mapped->lhs), 11);
    EXPECT_EQ(mapped->lhs.use_count(), 1);
  } else {
    EXPECT_EQ(mapped->lhs.get(), lhs_addr);
    EXPECT_EQ(external_lhs.use_count(), 2);
    EXPECT_EQ(MapLeafValue(mapped->lhs), 10);
  }

  EXPECT_EQ(mapped->rhs.get(), rhs_addr);
  EXPECT_EQ(MapLeafValue(mapped->rhs), change_rhs ? 21 : 20);
  EXPECT_EQ(mapped->rhs.use_count(), 1);
  ExpectTrace({"map:10", "inplace:20"});
}

TEST(StructuralMapper, InplaceMutateSelectsEachFieldFromLogicalUniqueness) {
  CheckInplaceFieldSelection(false, false);
  CheckInplaceFieldSelection(true, false);
  CheckInplaceFieldSelection(false, true);
  CheckInplaceFieldSelection(true, true);
}

TEST(StructuralMapper, DispatchesFFIFunctionHooks) {
  TestMapper mapper = MakeTestMapper();

  ObjectRef unchanged_source = MakeFunctionLeaf(5);
  const Object* unchanged_addr = unchanged_source.get();
  Any unchanged_input = std::move(unchanged_source);
  Expected<Any> unchanged_result = mapper->MapExpected(std::move(unchanged_input));

  ASSERT_TRUE(unchanged_result.is_ok());
  ObjectRef unchanged_mapped = std::move(unchanged_result).value().cast<ObjectRef>();
  EXPECT_EQ(unchanged_mapped.get(), unchanged_addr);
  EXPECT_EQ(unchanged_mapped.use_count(), 1);
  ExpectTrace({"function-map:5"});

  ClearMapperTrace();
  ObjectRef source = MakeFunctionLeaf(10, TransformAction::kChange);
  const Object* source_addr = source.get();

  Expected<Any> map_result = mapper->MapExpected(source);

  ASSERT_TRUE(map_result.is_ok());
  ObjectRef mapped = std::move(map_result).value().cast<ObjectRef>();
  EXPECT_NE(mapped.get(), source_addr);
  EXPECT_EQ(MapLeafValue(source), 10);
  EXPECT_EQ(MapLeafValue(mapped), 11);
  EXPECT_EQ(source.use_count(), 1);
  EXPECT_EQ(mapped.use_count(), 1);
  ExpectTrace({"function-map:10"});

  ClearMapperTrace();
  ObjectRef inplace_source = MakeFunctionLeaf(20, TransformAction::kChange);
  const Object* inplace_addr = inplace_source.get();
  Any input = std::move(inplace_source);
  Expected<Any> inplace_result = mapper->InplaceMutateExpected(std::move(input));

  ASSERT_TRUE(inplace_result.is_ok());
  ObjectRef inplace_mapped = std::move(inplace_result).value().cast<ObjectRef>();
  EXPECT_EQ(inplace_mapped.get(), inplace_addr);
  EXPECT_EQ(MapLeafValue(inplace_mapped), 21);
  EXPECT_EQ(inplace_mapped.use_count(), 1);
  ExpectTrace({"function-inplace:20"});
}

TEST(StructuralMapper, RecordsMapErrorContext) {
  ObjectRef failing_leaf = MakeMapLeaf(1, TransformAction::kError);
  TVar rhs("rhs");
  TPair root(std::move(failing_leaf), std::move(rhs));
  const Object* root_addr = root.get();
  const Object* leaf_addr = root->lhs.get();
  TestMapper mapper = MakeTestMapper();

  Expected<Any> result = mapper->MapExpected(root);

  ASSERT_TRUE(result.is_err());
  Error error = result.error();
  EXPECT_EQ(error.kind(), "ValueError");
  EXPECT_EQ(error.message(), "structural map leaf failed");
  Optional<VisitErrorContext> context = VisitErrorContext::TryGetFromError(error);
  ASSERT_TRUE(context.has_value());
  const List<ObjectRef>& chain = context.value()->reverse_visit_pattern;
  ASSERT_EQ(chain.size(), 2U);
  EXPECT_EQ(chain[0].get(), leaf_addr);
  EXPECT_EQ(chain[1].get(), root_addr);
  ExpectTrace({"map:1"});
}

TEST(StructuralMapper, CombinedErrorContextDoesNotDuplicateRoot) {
  ObjectRef failing_leaf = MakeMapLeaf(1, TransformAction::kError);
  TVar rhs("rhs");
  TPair root(std::move(failing_leaf), std::move(rhs));
  const Object* root_addr = root.get();
  const Object* leaf_addr = root->lhs.get();
  Any input = std::move(root);
  TestMapper mapper = MakeTestMapper();

  Expected<Any> result = mapper->MapOrInplaceMutateExpected(std::move(input));

  ASSERT_TRUE(result.is_err());
  Error error = result.error();
  Optional<VisitErrorContext> context = VisitErrorContext::TryGetFromError(error);
  ASSERT_TRUE(context.has_value());
  const List<ObjectRef>& chain = context.value()->reverse_visit_pattern;
  ASSERT_EQ(chain.size(), 2U);
  EXPECT_EQ(chain[0].get(), leaf_addr);
  EXPECT_EQ(chain[1].get(), root_addr);
  ExpectTrace({"inplace:1"});
}

TEST(StructuralMapper, InplaceMutationIsNotRolledBackOnError) {
  ObjectRef lhs = MakeMapLeaf(1, TransformAction::kChange);
  ObjectRef rhs = MakeMapLeaf(2, TransformAction::kError);
  TPair root(std::move(lhs), std::move(rhs));
  const Object* root_addr = root.get();
  const Object* lhs_addr = root->lhs.get();
  const Object* rhs_addr = root->rhs.get();
  Any input = std::move(root);
  TestMapper mapper = MakeTestMapper();

  Expected<Any> result = mapper->InplaceMutateExpected(std::move(input));

  ASSERT_TRUE(result.is_err());
  Error error = result.error();
  Optional<VisitErrorContext> context = VisitErrorContext::TryGetFromError(error);
  ASSERT_TRUE(context.has_value());
  const List<ObjectRef>& chain = context.value()->reverse_visit_pattern;
  ASSERT_EQ(chain.size(), 2U);
  EXPECT_EQ(chain[0].get(), rhs_addr);
  EXPECT_EQ(chain[1].get(), root_addr);

  const TPairObj* partially_mutated = chain[1].as<TPairObj>();
  ASSERT_NE(partially_mutated, nullptr);
  EXPECT_EQ(partially_mutated->lhs.get(), lhs_addr);
  EXPECT_EQ(MapLeafValue(partially_mutated->lhs), 2);
  ExpectTrace({"inplace:1", "inplace:2"});
}

}  // namespace
