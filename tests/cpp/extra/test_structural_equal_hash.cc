
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
#include <tvm/ffi/container/dict.h>
#include <tvm/ffi/container/list.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/extra/structural_hash.h>
#include <tvm/ffi/object.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>

#include "../testing_object.h"

namespace {

using namespace tvm::ffi;
using namespace tvm::ffi::testing;
namespace refl = tvm::ffi::reflection;

// CPU allocator and a helper to build a 1-D float32 tensor filled with one
// value, like test_tensor.cc does.
struct CPUNDAlloc {
  void AllocData(DLTensor* tensor) { tensor->data = malloc(GetDataSize(*tensor)); }
  void FreeData(DLTensor* tensor) { free(tensor->data); }
};

Tensor MakeFilledTensor(const Shape& shape, float value) {
  Tensor t = Tensor::FromNDAlloc(CPUNDAlloc(), shape, DLDataType({kDLFloat, 32, 1}),
                                 DLDevice({kDLCPU, 0}));
  float* dst = reinterpret_cast<float*>(t.data_ptr());
  // dst points at GetDataSize(t) bytes (numel floats); the analyzer cannot infer that
  // size through the allocator, so it wrongly flags this write as out of bounds.
  // NOLINTNEXTLINE(clang-analyzer-security.ArrayBound)
  for (int64_t i = 0; i < t.numel(); ++i) dst[i] = value;
  return t;
}

TEST(StructuralEqualHash, Array) {
  Array<int> a = {1, 2, 3};
  Array<int> b = {1, 2, 3};
  EXPECT_TRUE(StructuralEqual()(a, b));
  EXPECT_EQ(StructuralHash()(a), StructuralHash()(b));

  Array<int> c = {1, 3};
  EXPECT_FALSE(StructuralEqual()(a, c));
  EXPECT_NE(StructuralHash()(a), StructuralHash()(c));
  auto diff_a_c = StructuralEqual::GetFirstMismatch(a, c);

  // first directly interepret diff,
  EXPECT_TRUE(diff_a_c.has_value());
  auto lhs_steps = (*diff_a_c).get<0>()->ToSteps();
  auto rhs_steps = (*diff_a_c).get<1>()->ToSteps();
  EXPECT_EQ(lhs_steps[0]->kind, refl::AccessKind::kArrayItem);
  EXPECT_EQ(rhs_steps[0]->kind, refl::AccessKind::kArrayItem);
  EXPECT_EQ(lhs_steps[0]->key.cast<int64_t>(), 1);
  EXPECT_EQ(rhs_steps[0]->key.cast<int64_t>(), 1);
  EXPECT_EQ(lhs_steps.size(), 1);
  EXPECT_EQ(rhs_steps.size(), 1);

  // use structural equal for checking in future parts
  // given we have done some basic checks above by directly interepret diff,
  Array<int> d = {1, 2};
  auto diff_a_d = StructuralEqual::GetFirstMismatch(a, d);
  auto expected_diff_a_d = refl::AccessPathPair(refl::AccessPath::FromSteps({
                                                    refl::AccessStep::ArrayItem(2),
                                                }),
                                                refl::AccessPath::FromSteps({
                                                    refl::AccessStep::ArrayItemMissing(2),
                                                }));
  // then use structural equal to check it
  EXPECT_TRUE(StructuralEqual()(diff_a_d, expected_diff_a_d));
}

TEST(StructuralEqualHash, Map) {
  // same map but different insertion order
  Map<String, int> a = {{"a", 1}, {"b", 2}, {"c", 3}};
  Map<String, int> b = {{"b", 2}, {"c", 3}, {"a", 1}};
  EXPECT_TRUE(StructuralEqual()(a, b));
  EXPECT_EQ(StructuralHash()(a), StructuralHash()(b));

  Map<String, int> c = {{"a", 1}, {"b", 2}, {"c", 4}};
  EXPECT_FALSE(StructuralEqual()(a, c));
  EXPECT_NE(StructuralHash()(a), StructuralHash()(c));

  auto diff_a_c = StructuralEqual::GetFirstMismatch(a, c);
  auto expected_diff_a_c = refl::AccessPathPair(refl::AccessPath::Root()->MapItem("c"),
                                                refl::AccessPath::Root()->MapItem("c"));
  EXPECT_TRUE(diff_a_c.has_value());
  EXPECT_TRUE(StructuralEqual()(diff_a_c, expected_diff_a_c));
}

TEST(StructuralEqualHash, NestedMapArray) {
  Map<String, Array<Any>> a = {{"a", {1, 2, 3}}, {"b", {4, "hello", 6}}};
  Map<String, Array<Any>> b = {{"a", {1, 2, 3}}, {"b", {4, "hello", 6}}};
  EXPECT_TRUE(StructuralEqual()(a, b));
  EXPECT_EQ(StructuralHash()(a), StructuralHash()(b));

  Map<String, Array<Any>> c = {{"a", {1, 2, 3}}, {"b", {4, "world", 6}}};
  EXPECT_FALSE(StructuralEqual()(a, c));
  EXPECT_NE(StructuralHash()(a), StructuralHash()(c));

  auto diff_a_c = StructuralEqual::GetFirstMismatch(a, c);
  auto expected_diff_a_c =
      refl::AccessPathPair(refl::AccessPath::Root()->MapItem("b")->ArrayItem(1),
                           refl::AccessPath::Root()->MapItem("b")->ArrayItem(1));
  EXPECT_TRUE(diff_a_c.has_value());
  EXPECT_TRUE(StructuralEqual()(diff_a_c, expected_diff_a_c));

  Map<String, Array<Any>> d = {{"a", {1, 2, 3}}};
  auto diff_a_d = StructuralEqual::GetFirstMismatch(a, d);
  auto expected_diff_a_d = refl::AccessPathPair(refl::AccessPath::Root()->MapItem("b"),
                                                refl::AccessPath::Root()->MapItemMissing("b"));
  EXPECT_TRUE(diff_a_d.has_value());
  EXPECT_TRUE(StructuralEqual()(diff_a_d, expected_diff_a_d));

  auto diff_d_a = StructuralEqual::GetFirstMismatch(d, a);
  auto expected_diff_d_a = refl::AccessPathPair(refl::AccessPath::Root()->MapItemMissing("b"),
                                                refl::AccessPath::Root()->MapItem("b"));
}

TEST(StructuralEqualHash, Dict) {
  // same dict but different insertion order
  Dict<String, int> a = {{"a", 1}, {"b", 2}, {"c", 3}};
  Dict<String, int> b = {{"b", 2}, {"c", 3}, {"a", 1}};
  EXPECT_TRUE(StructuralEqual()(a, b));
  EXPECT_EQ(StructuralHash()(a), StructuralHash()(b));

  Dict<String, int> c = {{"a", 1}, {"b", 2}, {"c", 4}};
  EXPECT_FALSE(StructuralEqual()(a, c));
  EXPECT_NE(StructuralHash()(a), StructuralHash()(c));

  auto diff_a_c = StructuralEqual::GetFirstMismatch(a, c);
  auto expected_diff_a_c = refl::AccessPathPair(refl::AccessPath::Root()->MapItem("c"),
                                                refl::AccessPath::Root()->MapItem("c"));
  EXPECT_TRUE(diff_a_c.has_value());
  EXPECT_TRUE(StructuralEqual()(diff_a_c, expected_diff_a_c));
}

TEST(StructuralEqualHash, NestedDictArray) {
  Dict<String, Array<Any>> a = {{"a", {1, 2, 3}}, {"b", {4, "hello", 6}}};
  Dict<String, Array<Any>> b = {{"a", {1, 2, 3}}, {"b", {4, "hello", 6}}};
  EXPECT_TRUE(StructuralEqual()(a, b));
  EXPECT_EQ(StructuralHash()(a), StructuralHash()(b));

  Dict<String, Array<Any>> c = {{"a", {1, 2, 3}}, {"b", {4, "world", 6}}};
  EXPECT_FALSE(StructuralEqual()(a, c));
  EXPECT_NE(StructuralHash()(a), StructuralHash()(c));

  auto diff_a_c = StructuralEqual::GetFirstMismatch(a, c);
  auto expected_diff_a_c =
      refl::AccessPathPair(refl::AccessPath::Root()->MapItem("b")->ArrayItem(1),
                           refl::AccessPath::Root()->MapItem("b")->ArrayItem(1));
  EXPECT_TRUE(diff_a_c.has_value());
  EXPECT_TRUE(StructuralEqual()(diff_a_c, expected_diff_a_c));

  Dict<String, Array<Any>> d = {{"a", {1, 2, 3}}};
  auto diff_a_d = StructuralEqual::GetFirstMismatch(a, d);
  auto expected_diff_a_d = refl::AccessPathPair(refl::AccessPath::Root()->MapItem("b"),
                                                refl::AccessPath::Root()->MapItemMissing("b"));
  EXPECT_TRUE(diff_a_d.has_value());
  EXPECT_TRUE(StructuralEqual()(diff_a_d, expected_diff_a_d));
}

TEST(StructuralEqualHash, DictVsMapDifferentType) {
  Map<String, int> m = {{"a", 1}, {"b", 2}};
  Dict<String, int> d = {{"a", 1}, {"b", 2}};
  // Different type_index => not equal
  EXPECT_FALSE(StructuralEqual()(m, d));
  // Different type_key_hash => different hash (very likely)
  EXPECT_NE(StructuralHash()(m), StructuralHash()(d));
}

TEST(StructuralEqualHash, FreeVar) {
  TVar a = TVar("a");
  TVar b = TVar("b");
  EXPECT_TRUE(StructuralEqual::Equal(a, b, /*map_free_vars=*/true));
  EXPECT_FALSE(StructuralEqual::Equal(a, b));

  EXPECT_NE(StructuralHash()(a), StructuralHash()(b));
  EXPECT_EQ(StructuralHash::Hash(a, /*map_free_vars=*/true),
            StructuralHash::Hash(b, /*map_free_vars=*/true));
}

TEST(StructuralEqualHash, FuncDefAndIgnoreField) {
  TVar x = TVar("x");
  TVar y = TVar("y");
  // comment fields are ignored
  TFunc fa = TFunc({x}, {TInt(1), x}, String("comment a"));
  TFunc fb = TFunc({y}, {TInt(1), y}, String("comment b"));

  TFunc fc = TFunc({x}, {TInt(1), TInt(2)}, String("comment c"));

  EXPECT_TRUE(StructuralEqual()(fa, fb));
  EXPECT_EQ(StructuralHash()(fa), StructuralHash()(fb));

  EXPECT_FALSE(StructuralEqual()(fa, fc));
  auto diff_fa_fc = StructuralEqual::GetFirstMismatch(fa, fc);
  auto expected_diff_fa_fc = refl::AccessPathPair(refl::AccessPath::FromSteps({
                                                      refl::AccessStep::Attr("body"),
                                                      refl::AccessStep::ArrayItem(1),
                                                  }),
                                                  refl::AccessPath::FromSteps({
                                                      refl::AccessStep::Attr("body"),
                                                      refl::AccessStep::ArrayItem(1),
                                                  }));
  EXPECT_TRUE(diff_fa_fc.has_value());
  EXPECT_TRUE(StructuralEqual()(diff_fa_fc, expected_diff_fa_fc));
}

TEST(StructuralEqualHash, CustomTreeNode) {
  TVar x = TVar("x");
  TVar y = TVar("y");
  // comment fields are ignored
  TCustomFunc fa = TCustomFunc({x}, {TInt(1), x}, "comment a");
  TCustomFunc fb = TCustomFunc({y}, {TInt(1), y}, "comment b");

  TCustomFunc fc = TCustomFunc({x}, {TInt(1), TInt(2)}, "comment c");

  EXPECT_TRUE(StructuralEqual()(fa, fb));
  EXPECT_EQ(StructuralHash()(fa), StructuralHash()(fb));

  EXPECT_FALSE(StructuralEqual()(fa, fc));
  auto diff_fa_fc = StructuralEqual::GetFirstMismatch(fa, fc);
  auto expected_diff_fa_fc =
      refl::AccessPathPair(refl::AccessPath::Root()->Attr("body")->ArrayItem(1),
                           refl::AccessPath::Root()->Attr("body")->ArrayItem(1));
  EXPECT_TRUE(diff_fa_fc.has_value());
  EXPECT_TRUE(StructuralEqual()(diff_fa_fc, expected_diff_fa_fc));
}

// Regression tests for the SEqHashDefRecursive vs SEqHashDefNonRecursive
// distinction. ``TDefHolder`` has two sibling fields:
//   - ``def_recursive``     tagged AttachFieldFlag::SEqHashDefRecursive()
//   - ``def_non_recursive`` tagged AttachFieldFlag::SEqHashDefNonRecursive()
// each holding a ``TVarWithDep`` (a FreeVar with a sub-field ``dep`` that
// can itself reference another FreeVar). The four sub-cases below cover
// the observable behaviors of the two flags.
TEST(StructuralEqualHash, NonRecursiveDef) {
  {
    // (a) Recursive flag rebinds nested FreeVars transitively.
    // ``def_non_recursive`` is the same object on both sides so it equates
    // by pointer; the case isolates the recursive field's rebinding.
    SCOPED_TRACE("recursive flag rebinds nested FreeVars");
    TVarWithDep a("a", TVar("m"));
    TVarWithDep b("b", TVar("n"));
    TDefHolder lhs(/*def_recursive=*/a, /*def_non_recursive=*/a);
    TDefHolder rhs(/*def_recursive=*/b, /*def_non_recursive=*/b);
    EXPECT_TRUE(StructuralEqual()(lhs, rhs));
    EXPECT_EQ(StructuralHash::Hash(lhs, /*map_free_vars=*/true),
              StructuralHash::Hash(rhs, /*map_free_vars=*/true));
  }
  {
    // (b) Non-recursive flag does NOT rebind nested FreeVars: the top-level
    // FreeVar binds but the nested ``dep`` is clamped out of the def region.
    // With no outer binding for "p"/"q", equality must fail.
    SCOPED_TRACE("non-recursive flag does not rebind nested FreeVars");
    TVarWithDep shared("shared", std::nullopt);
    TVarWithDep c_with_dep("c", TVar("p"));
    TVarWithDep d_with_dep("d", TVar("q"));
    TDefHolder lhs(/*def_recursive=*/shared, /*def_non_recursive=*/c_with_dep);
    TDefHolder rhs(/*def_recursive=*/shared, /*def_non_recursive=*/d_with_dep);
    EXPECT_FALSE(StructuralEqual::Equal(lhs, rhs, /*map_free_vars=*/false));
  }
  {
    // (c) Non-recursive flag works if nested FreeVars resolve via an outer
    // binding — here we cheat by wiring the same pointer, so the nested
    // FreeVar passes the same-as pointer check without needing the def
    // region to be on inside its sub-field walk.
    SCOPED_TRACE("nested FreeVars resolve via outer pointer identity");
    TVar shared_dep("dep");
    TVarWithDep c_with_dep("c", shared_dep);
    TVarWithDep d_with_dep("d", shared_dep);
    TVarWithDep shared("shared", std::nullopt);
    TDefHolder lhs(/*def_recursive=*/shared, /*def_non_recursive=*/c_with_dep);
    TDefHolder rhs(/*def_recursive=*/shared, /*def_non_recursive=*/d_with_dep);
    EXPECT_TRUE(StructuralEqual()(lhs, rhs));
    EXPECT_EQ(StructuralHash()(lhs), StructuralHash()(rhs));
  }
  {
    // (d) Top-level FreeVar still binds under non-recursive — only the
    // FreeVar's sub-fields are clamped out; the binding step itself for
    // the immediate FreeVar is not suppressed.
    SCOPED_TRACE("top-level FreeVar still binds under non-recursive flag");
    TVarWithDep shared("shared", std::nullopt);
    TVarWithDep c_no_dep("c", std::nullopt);
    TVarWithDep d_no_dep("d", std::nullopt);
    TDefHolder lhs(/*def_recursive=*/shared, /*def_non_recursive=*/c_no_dep);
    TDefHolder rhs(/*def_recursive=*/shared, /*def_non_recursive=*/d_no_dep);
    EXPECT_TRUE(StructuralEqual()(lhs, rhs));
    EXPECT_EQ(StructuralHash::Hash(lhs, /*map_free_vars=*/true),
              StructuralHash::Hash(rhs, /*map_free_vars=*/true));
  }
}

TEST(StructuralEqualHash, List) {
  List<int> a = {1, 2, 3};
  List<int> b = {1, 2, 3};
  EXPECT_TRUE(StructuralEqual()(a, b));
  EXPECT_EQ(StructuralHash()(a), StructuralHash()(b));

  List<int> c = {1, 3};
  EXPECT_FALSE(StructuralEqual()(a, c));
  EXPECT_NE(StructuralHash()(a), StructuralHash()(c));
}

TEST(StructuralEqualHash, ListVsArrayDifferentType) {
  Array<int> arr = {1, 2, 3};
  List<int> lst = {1, 2, 3};
  // Different type_index => not equal
  EXPECT_FALSE(StructuralEqual()(arr, lst));
  // Different type_key_hash => different hash (very likely)
  EXPECT_NE(StructuralHash()(arr), StructuralHash()(lst));
}

TEST(StructuralEqualHash, DISABLED_ListCycleDetection) {
  List<Any> lst;
  lst.push_back(42);
  lst.push_back(lst);  // creates a cycle
  EXPECT_ANY_THROW(StructuralHash()(lst));
  EXPECT_ANY_THROW(StructuralEqual()(lst, lst));
}

TEST(StructuralEqualHash, ArraySelfInsertProducesSnapshot) {
  Array<Any> arr;
  arr.push_back(arr);

  Array<Any> snapshot = arr[0].cast<Array<Any>>();
  EXPECT_TRUE(snapshot.empty());
  EXPECT_FALSE(snapshot.same_as(arr));

  EXPECT_TRUE(StructuralEqual()(arr, arr));
  EXPECT_EQ(StructuralHash()(arr), StructuralHash()(arr));
}

// Regression test for #645. StructuralHash hashes tensor content, so the
// StructuralEqual functor has to compare content too. Otherwise two distinct
// same-shape constants hash differently but compare equal, which breaks the
// constant de-dup map invariant and can silently merge different weights on a
// bucket collision.
TEST(StructuralEqualHash, TensorContent) {
  Tensor zeros = MakeFilledTensor({4}, 0.0f);
  Tensor ones = MakeFilledTensor({4}, 1.0f);
  Tensor zeros_copy = MakeFilledTensor({4}, 0.0f);

  // Different content, same shape and dtype: not equal, and the hash differs.
  EXPECT_FALSE(StructuralEqual()(zeros, ones));
  EXPECT_NE(StructuralHash()(zeros), StructuralHash()(ones));

  // Identical content still compares equal and hashes equal, so real duplicates
  // still get merged.
  EXPECT_TRUE(StructuralEqual()(zeros, zeros_copy));
  EXPECT_EQ(StructuralHash()(zeros), StructuralHash()(zeros_copy));

  // Skipping content is still available as an explicit opt-in.
  EXPECT_TRUE(StructuralEqual::Equal(zeros, ones, /*map_free_vars=*/false,
                                     /*skip_tensor_content=*/true));
}

}  // namespace
