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
#include <tvm/ffi/container/list.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/object.h>
#include <tvm/ffi/string.h>

#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "../testing_object.h"

namespace {

using namespace tvm::ffi;
using namespace tvm::ffi::testing;

class TestVisitorObj : public StructuralVisitorObj {
 public:
  TestVisitorObj() : StructuralVisitorObj(VTable()) {}

  std::vector<ObjectRef> visited;
  std::vector<TVMFFIDefRegionKind> modes;
  ObjectRef interrupt_on;

 private:
  static const StructuralVisitorVTable* VTable() {
    static const StructuralVisitorVTable vtable{
        &TestVisitorObj::DispatchVisit,
    };
    return &vtable;
  }

  static TVMFFIAny DispatchVisit(StructuralVisitorObj* self, AnyView value) noexcept {
    return static_cast<TestVisitorObj*>(self)->VisitImpl(value);
  }

  // NOLINTNEXTLINE(bugprone-exception-escape)
  TVMFFIAny VisitImpl(AnyView value) noexcept {
    if (value.type_index() >= TypeIndex::kTVMFFIStaticObjectBegin) {
      ObjectRef value_ref = value.cast<ObjectRef>();
      visited.push_back(value_ref);
      modes.push_back(def_region_mode_);
      if (interrupt_on.defined() && value_ref.same_as(interrupt_on)) {
        Expected<Optional<VisitInterrupt>> interrupt =
            Optional<VisitInterrupt>(VisitInterrupt(String("stop")));
        return details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(interrupt));
      }
    }
    return details::ExpectedUnsafe::MoveToTVMFFIAny(DefaultVisitExpected(value));
  }
};

struct VisitCount {
  int value = 0;
};

struct VisitTag {
  int value = 7;
};

template <typename Parent = StructuralVisitorObj>
class StructuralWalkWithVisitCount : public Parent {
 public:
  using VisitorObjType = StructuralWalkWithVisitCount;
  using StateTupleType = std::tuple<const VisitCount&, const VisitTag&>;

  explicit StructuralWalkWithVisitCount(const StructuralVisitorVTable* vtable) : Parent(vtable) {}

  int callback_tag() const noexcept { return visit_tag_.value; }

  TVM_FFI_INLINE Expected<Optional<VisitInterrupt>> DefaultVisitExpected(AnyView value) noexcept {
    ++visit_count_.value;
    return Parent::DefaultVisitExpected(value);
  }

 protected:
  TVM_FFI_INLINE TVMFFIAny DefaultVisitRaw(AnyView value) noexcept {
    return details::ExpectedUnsafe::MoveToTVMFFIAny(DefaultVisitExpected(value));
  }

  TVM_FFI_INLINE StateTupleType StateTuple() const noexcept {
    return std::tie(visit_count_, visit_tag_);
  }

 private:
  VisitCount visit_count_;
  VisitTag visit_tag_;
};

template <typename Parent>
class StructuralVisitOuterLayer : public Parent {
 public:
  explicit StructuralVisitOuterLayer(const StructuralVisitorVTable* vtable) : Parent(vtable) {}
};

StructuralVisitor MakeTestVisitor() { return StructuralVisitor(make_object<TestVisitorObj>()); }

TestVisitorObj* AsTestVisitor(const StructuralVisitor& visitor) {
  return static_cast<TestVisitorObj*>(visitor.get());
}

void SetInterrupt(const StructuralVisitor& visitor, const ObjectRef& value) {
  TestVisitorObj* test_visitor = AsTestVisitor(visitor);
  test_visitor->interrupt_on = value;
}

void ExpectTrace(const std::vector<std::string>& actual,
                 std::initializer_list<const char*> expected) {
  ASSERT_EQ(actual.size(), expected.size());
  size_t i = 0;
  for (const char* item : expected) {
    EXPECT_EQ(actual[i], item);
    ++i;
  }
}

// ---------------------------------------------------------------------------
// StructuralVisitor behavior.
// ---------------------------------------------------------------------------

TEST(StructuralVisitor, RecordsNode) {
  ObjectRef leaf = TVar("leaf");
  StructuralVisitor visitor = MakeTestVisitor();

  Expected<Optional<VisitInterrupt>> result = visitor->VisitExpected(leaf);

  ASSERT_TRUE(result.is_ok());
  EXPECT_FALSE(result.value().has_value());
  ASSERT_EQ(AsTestVisitor(visitor)->visited.size(), 1U);
  EXPECT_TRUE(AsTestVisitor(visitor)->visited[0].same_as(leaf));
}

TEST(StructuralVisitor, PropagatesInterrupt) {
  ObjectRef leaf = TVar("leaf");
  StructuralVisitor visitor = MakeTestVisitor();
  SetInterrupt(visitor, leaf);

  Expected<Optional<VisitInterrupt>> result = visitor->VisitExpected(leaf);

  ASSERT_TRUE(result.is_ok());
  ASSERT_TRUE(result.value().has_value());
  EXPECT_EQ(result.value().value()->value.cast<String>(), "stop");
}

TEST(StructuralVisitor, TraversesPair) {
  ObjectRef lhs = TVar("lhs");
  ObjectRef rhs = TVar("rhs");
  ObjectRef root = TPair(lhs, rhs);
  StructuralVisitor visitor = MakeTestVisitor();

  Expected<Optional<VisitInterrupt>> result = visitor->DefaultVisitExpected(root);

  ASSERT_TRUE(result.is_ok());
  EXPECT_FALSE(result.value().has_value());
  ASSERT_EQ(AsTestVisitor(visitor)->visited.size(), 2U);
  EXPECT_TRUE(AsTestVisitor(visitor)->visited[0].same_as(lhs));
  EXPECT_TRUE(AsTestVisitor(visitor)->visited[1].same_as(rhs));
}

TEST(StructuralVisitor, TraversesFunction) {
  TVar param("x");
  ObjectRef body_value = TInt(1);
  Array<TVar> params = {param};
  Array<ObjectRef> body = {body_value};
  ObjectRef root = TFunc(params, body, String("ignored function comment"));
  StructuralVisitor visitor = MakeTestVisitor();

  Expected<Optional<VisitInterrupt>> result = visitor->DefaultVisitExpected(root);

  ASSERT_TRUE(result.is_ok());
  EXPECT_FALSE(result.value().has_value());
  TestVisitorObj* test_visitor = AsTestVisitor(visitor);
  ASSERT_EQ(test_visitor->visited.size(), 4U);
  EXPECT_TRUE(test_visitor->visited[0].same_as(params));
  EXPECT_EQ(test_visitor->modes[0], kTVMFFIDefRegionKindRecursive);
  EXPECT_TRUE(test_visitor->visited[1].same_as(param));
  EXPECT_EQ(test_visitor->modes[1], kTVMFFIDefRegionKindRecursive);
  EXPECT_TRUE(test_visitor->visited[2].same_as(body));
  EXPECT_EQ(test_visitor->modes[2], kTVMFFIDefRegionKindNone);
  EXPECT_TRUE(test_visitor->visited[3].same_as(body_value));
  EXPECT_EQ(test_visitor->modes[3], kTVMFFIDefRegionKindNone);
}

TEST(StructuralVisitor, StopsOnInterrupt) {
  ObjectRef lhs = TVar("lhs");
  ObjectRef rhs = TVar("rhs");
  ObjectRef root = TPair(lhs, rhs);
  StructuralVisitor visitor = MakeTestVisitor();
  SetInterrupt(visitor, lhs);

  Expected<Optional<VisitInterrupt>> result = visitor->DefaultVisitExpected(root);

  ASSERT_TRUE(result.is_ok());
  ASSERT_TRUE(result.value().has_value());
  EXPECT_EQ(result.value().value()->value.cast<String>(), "stop");
  ASSERT_EQ(AsTestVisitor(visitor)->visited.size(), 1U);
  EXPECT_TRUE(AsTestVisitor(visitor)->visited[0].same_as(lhs));
}

TEST(StructuralVisitor, TraversesArray) {
  ObjectRef lhs = TVar("lhs");
  ObjectRef rhs = TVar("rhs");
  Array<ObjectRef> root = {lhs, rhs};
  StructuralVisitor visitor = MakeTestVisitor();

  Expected<Optional<VisitInterrupt>> result = visitor->DefaultVisitExpected(root);

  ASSERT_TRUE(result.is_ok());
  EXPECT_FALSE(result.value().has_value());
  ASSERT_EQ(AsTestVisitor(visitor)->visited.size(), 2U);
  EXPECT_TRUE(AsTestVisitor(visitor)->visited[0].same_as(lhs));
  EXPECT_TRUE(AsTestVisitor(visitor)->visited[1].same_as(rhs));
}

TEST(StructuralVisitor, TraversesMapValuesWithoutVisitingKeys) {
  ObjectRef key = TVar("key");
  ObjectRef value = TVar("value");
  Map<Any, Any> root{{key, value}};
  StructuralVisitor visitor = MakeTestVisitor();

  Expected<Optional<VisitInterrupt>> result = visitor->DefaultVisitExpected(root);

  ASSERT_TRUE(result.is_ok());
  EXPECT_FALSE(result.value().has_value());
  ASSERT_EQ(AsTestVisitor(visitor)->visited.size(), 1U);
  EXPECT_TRUE(AsTestVisitor(visitor)->visited[0].same_as(value));
}

TEST(StructuralVisitor, UsesFuncHook) {
  TVar param("x");
  ObjectRef body_value = TInt(1);
  Array<TVar> params = {param};
  Array<ObjectRef> body = {body_value};
  ObjectRef root = TFunc(params, body, String("ignored function comment"));
  StructuralVisitor visitor = MakeTestVisitor();

  Optional<VisitInterrupt> result = visitor->Visit(root);

  EXPECT_FALSE(result.has_value());
  TestVisitorObj* test_visitor = AsTestVisitor(visitor);
  ASSERT_EQ(test_visitor->visited.size(), 5U);
  EXPECT_TRUE(test_visitor->visited[0].same_as(root));
  EXPECT_EQ(test_visitor->modes[0], kTVMFFIDefRegionKindNone);
  EXPECT_TRUE(test_visitor->visited[1].same_as(params));
  EXPECT_EQ(test_visitor->modes[1], kTVMFFIDefRegionKindRecursive);
  EXPECT_TRUE(test_visitor->visited[2].same_as(param));
  EXPECT_EQ(test_visitor->modes[2], kTVMFFIDefRegionKindRecursive);
  EXPECT_TRUE(test_visitor->visited[3].same_as(body));
  EXPECT_EQ(test_visitor->modes[3], kTVMFFIDefRegionKindNone);
  EXPECT_TRUE(test_visitor->visited[4].same_as(body_value));
  EXPECT_EQ(test_visitor->modes[4], kTVMFFIDefRegionKindNone);
  EXPECT_EQ(test_visitor->def_region_kind(), kTVMFFIDefRegionKindNone);
}

TEST(StructuralVisitor, RestoresFuncDefRegion) {
  TVar param("x");
  Array<TVar> params = {param};
  Array<ObjectRef> body = {TInt(1)};
  ObjectRef root = TFunc(params, body, String("ignored function comment"));
  StructuralVisitor visitor = MakeTestVisitor();
  SetInterrupt(visitor, param);

  Optional<VisitInterrupt> result = visitor->Visit(root);

  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value()->value.cast<String>(), "stop");
  TestVisitorObj* test_visitor = AsTestVisitor(visitor);
  ASSERT_EQ(test_visitor->visited.size(), 3U);
  EXPECT_TRUE(test_visitor->visited[0].same_as(root));
  EXPECT_EQ(test_visitor->modes[0], kTVMFFIDefRegionKindNone);
  EXPECT_TRUE(test_visitor->visited[1].same_as(params));
  EXPECT_EQ(test_visitor->modes[1], kTVMFFIDefRegionKindRecursive);
  EXPECT_TRUE(test_visitor->visited[2].same_as(param));
  EXPECT_EQ(test_visitor->modes[2], kTVMFFIDefRegionKindRecursive);
  EXPECT_EQ(test_visitor->def_region_kind(), kTVMFFIDefRegionKindNone);
}

TEST(StructuralVisitor, ExplicitDefRegionsOverrideFreeVarFieldClamp) {
  TVarWithDep recursive("recursive");
  TVarWithDep non_recursive("non-recursive");
  TDefHolder holder(recursive, non_recursive);
  TVarWithDep root("outer", holder);
  StructuralVisitor visitor = MakeTestVisitor();

  Expected<Optional<VisitInterrupt>> result = visitor->WithDefRegionKind(
      kTVMFFIDefRegionKindNonRecursive, [&]() { return visitor->VisitExpected(root); });

  ASSERT_TRUE(result.is_ok());
  EXPECT_FALSE(result.value().has_value());
  TestVisitorObj* test_visitor = AsTestVisitor(visitor);
  ASSERT_EQ(test_visitor->visited.size(), 4U);
  EXPECT_TRUE(test_visitor->visited[0].same_as(root));
  EXPECT_EQ(test_visitor->modes[0], kTVMFFIDefRegionKindNonRecursive);
  EXPECT_TRUE(test_visitor->visited[1].same_as(holder));
  EXPECT_EQ(test_visitor->modes[1], kTVMFFIDefRegionKindNone);
  EXPECT_TRUE(test_visitor->visited[2].same_as(recursive));
  EXPECT_EQ(test_visitor->modes[2], kTVMFFIDefRegionKindRecursive);
  EXPECT_TRUE(test_visitor->visited[3].same_as(non_recursive));
  EXPECT_EQ(test_visitor->modes[3], kTVMFFIDefRegionKindNonRecursive);
  EXPECT_EQ(test_visitor->def_region_kind(), kTVMFFIDefRegionKindNone);
}

// ---------------------------------------------------------------------------
// StructuralWalk behavior.
// ---------------------------------------------------------------------------

TEST(StructuralVisitor, WalkSkipsChildren) {
  ObjectRef lhs = TVar("lhs");
  ObjectRef rhs = TVar("rhs");
  ObjectRef root = TPair(lhs, rhs);
  std::vector<std::string> visited;

  Optional<VisitInterrupt> result = StructuralWalk<WalkOrder::kPreOrder>(
      root, [&](const ObjectRef& node) -> Expected<WalkResult> {
        if (node.same_as(root)) {
          visited.emplace_back("pair");
          return WalkResult::Skip();
        }
        visited.emplace_back("child");
        return WalkResult::Advance();
      });

  EXPECT_FALSE(result.has_value());
  ExpectTrace(visited, {"pair"});
}

TEST(StructuralVisitor, WalkPostOrder) {
  ObjectRef lhs = TVar("lhs");
  ObjectRef rhs = TVar("rhs");
  ObjectRef root = TPair(lhs, rhs);
  std::vector<std::string> visited;

  Optional<VisitInterrupt> result = StructuralWalk<WalkOrder::kPostOrder>(
      root, [&](const ObjectRef& node) -> Expected<WalkResult> {
        if (node.same_as(lhs)) {
          visited.emplace_back("lhs");
        } else if (node.same_as(rhs)) {
          visited.emplace_back("rhs");
        } else if (node.same_as(root)) {
          visited.emplace_back("pair");
        }
        return WalkResult::Advance();
      });

  EXPECT_FALSE(result.has_value());
  ExpectTrace(visited, {"lhs", "rhs", "pair"});
}

TEST(StructuralVisitor, WalkInterrupts) {
  ObjectRef lhs = TVar("lhs");
  ObjectRef rhs = TVar("rhs");
  ObjectRef root = TPair(lhs, rhs);
  std::vector<std::string> visited;

  Optional<VisitInterrupt> result = StructuralWalk<WalkOrder::kPreOrder>(
      root, [&](const ObjectRef& node) -> Expected<WalkResult> {
        if (node.same_as(lhs)) {
          visited.emplace_back("lhs");
          return WalkResult::Interrupt(VisitInterrupt(String("found lhs")));
        }
        if (node.same_as(rhs)) {
          visited.emplace_back("rhs");
        }
        return WalkResult::Advance();
      });

  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value()->value.cast<String>(), "found lhs");
  ExpectTrace(visited, {"lhs"});
}

TEST(StructuralVisitor, WalkVisitsPOD) {
  int64_t seen = 0;

  Optional<VisitInterrupt> result =
      StructuralWalk<WalkOrder::kPreOrder>(42, [&](int64_t value) -> Expected<WalkResult> {
        seen = value;
        return WalkResult::Advance();
      });

  EXPECT_FALSE(result.has_value());
  EXPECT_EQ(seen, 42);
}

TEST(StructuralVisitor, WalkVisitsObjectPtr) {
  ObjectPtr<TVarObj> root = make_object<TVarObj>("x");
  std::vector<std::string> visited;

  Optional<VisitInterrupt> result =
      StructuralWalk<WalkOrder::kPreOrder>(root, [&](const TVarObj* var) -> Expected<WalkResult> {
        visited.emplace_back(var->name);
        return WalkResult::Advance();
      });

  EXPECT_FALSE(result.has_value());
  ExpectTrace(visited, {"x"});
}

TEST(StructuralVisitor, WalkTraversesArcAndObjectPtrFields) {
  tvm::ffi::Arc<TIntObj> value = make_arc<TIntObj>(42);
  ObjectPtr<TNumberObj> alias = value;
  TObjectPtrHolder root(value, alias);
  size_t num_int_fields = 0;

  Optional<VisitInterrupt> result =
      StructuralWalk<WalkOrder::kPreOrder>(root, [&](const TIntObj* obj) -> Expected<WalkResult> {
        EXPECT_EQ(obj, value.get());
        ++num_int_fields;
        return WalkResult::Advance();
      });

  EXPECT_FALSE(result.has_value());
  EXPECT_EQ(num_int_fields, 2U);
}

TEST(StructuralVisitor, WalkReceivesDefRegionKind) {
  TVar x("x");
  TVar y("y");
  ObjectRef root = TFunc({x}, {x, y}, String("ignored function comment"));
  std::vector<std::string> use_vars;

  Optional<VisitInterrupt> result = StructuralWalk<WalkOrder::kPreOrder>(
      root, [&](const TVarObj* var, TVMFFIDefRegionKind kind) -> Expected<WalkResult> {
        if (kind == kTVMFFIDefRegionKindNone) {
          use_vars.emplace_back(var->name);
        }
        return WalkResult::Advance();
      });

  EXPECT_FALSE(result.has_value());
  ExpectTrace(use_vars, {"x", "y"});
}

TEST(StructuralVisitor, WalkParentStatePreservesDescentSkipAndInterrupt) {
  ObjectRef lhs = TVar("lhs");
  ObjectRef rhs = TVar("rhs");
  ObjectRef root = TPair(lhs, rhs);
  std::vector<int> counts;
  std::vector<int> tags;
  std::vector<TVMFFIDefRegionKind> kinds;

  auto pair_callback = [&](const TPairObj*, const VisitCount& count, const VisitTag& tag,
                           TVMFFIDefRegionKind kind) -> Expected<WalkResult> {
    counts.push_back(count.value);
    tags.push_back(tag.value);
    kinds.push_back(kind);
    return WalkResult::Advance();
  };
  auto fallback_callback = [&](const ObjectRef&, const VisitCount& count, const VisitTag& tag,
                               TVMFFIDefRegionKind kind) -> Expected<WalkResult> {
    counts.push_back(count.value);
    tags.push_back(tag.value);
    kinds.push_back(kind);
    return WalkResult::Advance();
  };
  using PreOrderEngine = StructuralWalkEngine<StructuralWalkWithVisitCount<>, WalkOrder::kPreOrder,
                                              decltype(pair_callback), decltype(fallback_callback)>;
  StructuralVisitor visitor(
      make_object<PreOrderEngine>(std::move(pair_callback), std::move(fallback_callback)));
  Optional<VisitInterrupt> result = visitor->VisitExpected(root).value();

  EXPECT_FALSE(result.has_value());
  EXPECT_EQ(counts, (std::vector<int>{0, 1, 2}));
  EXPECT_EQ(tags, (std::vector<int>{7, 7, 7}));
  EXPECT_EQ(kinds,
            (std::vector<TVMFFIDefRegionKind>{kTVMFFIDefRegionKindNone, kTVMFFIDefRegionKindNone,
                                              kTVMFFIDefRegionKindNone}));
  counts.clear();

  auto interrupt_callback = [&](const ObjectRef& node, const VisitCount& count,
                                const VisitTag&) -> Expected<WalkResult> {
    counts.push_back(count.value);
    if (node.same_as(lhs)) return WalkResult::Skip();
    if (node.same_as(rhs)) return WalkResult::Interrupt(VisitInterrupt(String("done")));
    return WalkResult::Advance();
  };
  using InterruptEngine = StructuralWalkEngine<StructuralWalkWithVisitCount<>, WalkOrder::kPreOrder,
                                               decltype(interrupt_callback)>;
  StructuralVisitor interrupt_visitor(make_object<InterruptEngine>(std::move(interrupt_callback)));
  Optional<VisitInterrupt> interrupt_result = interrupt_visitor->VisitExpected(root).value();

  ASSERT_TRUE(interrupt_result.has_value());
  EXPECT_EQ(interrupt_result.value()->value.cast<String>(), "done");
  EXPECT_EQ(counts, (std::vector<int>{0, 1, 1}));

  counts.clear();
  auto postorder_callback = [&](const ObjectRef&, const VisitCount& count,
                                const VisitTag&) -> Expected<WalkResult> {
    counts.push_back(count.value);
    return WalkResult::Advance();
  };
  using PostOrderEngine = StructuralWalkEngine<StructuralWalkWithVisitCount<>,
                                               WalkOrder::kPostOrder, decltype(postorder_callback)>;
  StructuralVisitor postorder_visitor(make_object<PostOrderEngine>(std::move(postorder_callback)));
  Optional<VisitInterrupt> postorder_result = postorder_visitor->VisitExpected(root).value();
  EXPECT_FALSE(postorder_result.has_value());
  EXPECT_EQ(counts, (std::vector<int>{2, 3, 3}));

  auto error_callback = [&](const ObjectRef& node, const VisitCount&,
                            const VisitTag&) -> Expected<WalkResult> {
    if (node.same_as(lhs)) {
      return Unexpected(Error("ValueError", "stateful walk failed", ""));
    }
    return WalkResult::Advance();
  };
  using ErrorEngine = StructuralWalkEngine<StructuralWalkWithVisitCount<>, WalkOrder::kPreOrder,
                                           decltype(error_callback)>;
  StructuralVisitor error_visitor(make_object<ErrorEngine>(std::move(error_callback)));
  Expected<Optional<VisitInterrupt>> error_result = error_visitor->VisitExpected(root);
  ASSERT_TRUE(error_result.is_err());
  EXPECT_EQ(error_result.error().kind(), "ValueError");
  EXPECT_EQ(error_result.error().message(), "stateful walk failed");
}

TEST(StructuralVisitor, WalkReturnsError) {
  ObjectRef lhs = TVar("lhs");
  ObjectRef rhs = TVar("rhs");
  ObjectRef root = TPair(lhs, rhs);

  Expected<Optional<VisitInterrupt>> result = StructuralWalkExpected<WalkOrder::kPreOrder>(
      root, [&](const ObjectRef& node) -> Expected<WalkResult> {
        if (node.same_as(lhs)) {
          return Unexpected(Error("ValueError", "walk callback failed", ""));
        }
        return WalkResult::Advance();
      });

  ASSERT_TRUE(result.is_err());
  EXPECT_EQ(result.error().kind(), "ValueError");
  EXPECT_EQ(result.error().message(), "walk callback failed");
}

// A callback that fails on a primitive node must surface the error, not abort.
//
// On failure the engine names the node it dispatched on in the visit error context. A visited
// node is not always object-backed -- a container element or reflected field may be a primitive --
// and `AnyView::cast<ObjectRef>()` throws on one. The walk is `noexcept`, so naming the node
// without a type-index guard terminates the process instead of returning the error. Every other
// error test above fails on an ObjectRef node, so only a primitive exercises that guard.
TEST(StructuralVisitor, WalkReturnsErrorOnPrimitiveNode) {
  Array<Any> root = {Any(static_cast<int64_t>(1))};
  auto fail = [](int64_t) -> Expected<WalkResult> {
    return Unexpected(Error("ValueError", "walk callback failed on primitive", ""));
  };

  Expected<Optional<VisitInterrupt>> pre = StructuralWalkExpected<WalkOrder::kPreOrder>(root, fail);
  ASSERT_TRUE(pre.is_err());
  EXPECT_EQ(pre.error().kind(), "ValueError");
  EXPECT_EQ(pre.error().message(), "walk callback failed on primitive");

  Expected<Optional<VisitInterrupt>> post =
      StructuralWalkExpected<WalkOrder::kPostOrder>(root, fail);
  ASSERT_TRUE(post.is_err());
  EXPECT_EQ(post.error().kind(), "ValueError");
  EXPECT_EQ(post.error().message(), "walk callback failed on primitive");
}

TEST(StructuralVisitor, WalkCatchesError) {
  ObjectRef root = TVar("root");

  Expected<Optional<VisitInterrupt>> result = StructuralWalkExpected<WalkOrder::kPreOrder>(
      root, [&](const ObjectRef&) -> Expected<WalkResult> {
        TVM_FFI_THROW(ValueError) << "walk callback threw";
        return WalkResult::Advance();
      });

  ASSERT_TRUE(result.is_err());
  EXPECT_EQ(result.error().kind(), "ValueError");
  EXPECT_EQ(result.error().message(), "walk callback threw");
}

TEST(StructuralVisitor, WalkFirstMatch) {
  ObjectRef lhs = TVar("lhs");
  ObjectRef rhs = TVar("rhs");
  List<ObjectRef> list = {lhs};
  Array<ObjectRef> root = {list, rhs};
  std::vector<std::string> trace;

  Optional<VisitInterrupt> result = StructuralWalk<WalkOrder::kPreOrder>(
      root,
      [&](const Array<ObjectRef>&) -> Expected<WalkResult> {
        trace.emplace_back("array");
        return WalkResult::Advance();
      },
      [&](const List<ObjectRef>&) -> Expected<WalkResult> {
        trace.emplace_back("list");
        return WalkResult::Advance();
      },
      [&](const ObjectRef&) -> Expected<WalkResult> {
        trace.emplace_back("object");
        return WalkResult::Advance();
      });

  EXPECT_FALSE(result.has_value());
  ExpectTrace(trace, {"array", "list", "object", "object"});
}

TEST(StructuralVisitor, WalkFuncProgram) {
  TVar m("m");
  TVar n("n");
  TVar acc("acc");
  ObjectRef func =
      TFunc({m, n}, {TInt(7), acc, TPair(m, TInt(1))}, String("ignored function comment"));
  Array<Any> root = {func, String("metadata"), nullptr};
  std::vector<std::string> trace;

  Optional<VisitInterrupt> result = StructuralWalk<WalkOrder::kPreOrder>(
      root,
      [&](const TFuncObj*) -> Expected<WalkResult> {
        trace.emplace_back("func*");
        return WalkResult::Advance();
      },
      [&](const TVarObj* var) -> Expected<WalkResult> {
        trace.emplace_back("var*:" + var->name);
        return WalkResult::Advance();
      },
      [&](int64_t value) -> Expected<WalkResult> {
        trace.emplace_back("int:" + std::to_string(value));
        return WalkResult::Advance();
      },
      [&](const ObjectRef&) -> Expected<WalkResult> {
        trace.emplace_back("object-ref");
        return WalkResult::Advance();
      },
      [&](AnyView) -> Expected<WalkResult> {
        trace.emplace_back("any-view");
        return WalkResult::Advance();
      });

  EXPECT_FALSE(result.has_value());
  ExpectTrace(trace,
              {"object-ref", "func*", "object-ref", "var*:m", "var*:n", "object-ref", "object-ref",
               "int:7", "var*:acc", "object-ref", "var*:m", "object-ref", "int:1", "object-ref"});
}

TEST(StructuralVisitor, WalkAnyFallback) {
  Array<Any> root = {String("metadata"), nullptr};
  std::vector<std::string> trace;

  Optional<VisitInterrupt> result = StructuralWalk<WalkOrder::kPreOrder>(
      root,
      [&](const ObjectRef&) -> Expected<WalkResult> {
        trace.emplace_back("object-ref");
        return WalkResult::Advance();
      },
      [&](const Any& value) -> Expected<WalkResult> {
        trace.emplace_back(value == nullptr ? "any:none" : "any:value");
        return WalkResult::Advance();
      });

  EXPECT_FALSE(result.has_value());
  ExpectTrace(trace, {"object-ref", "object-ref"});
}

// ---------------------------------------------------------------------------
// StructuralVisit behavior.
// ---------------------------------------------------------------------------

TEST(StructuralVisit, CallbackDrivenTraversal) {
  TVarWithDep lhs("lhs", TVarWithDep("pruned-dependency"));
  TVarWithDep stop("stop");
  TVarWithDep skipped("skipped");
  std::vector<std::pair<std::string, TVMFFIDefRegionKind>> trace;
  Expected<Optional<VisitInterrupt>> result = StructuralVisitExpected(
      TPair(Array<ObjectRef>{lhs}, Array<ObjectRef>{stop, skipped}),
      [](const TPairObj* pair,
         StructuralVisitorObj* visitor) -> Expected<Optional<VisitInterrupt>> {
        TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->WithDefRegionKind(
            kTVMFFIDefRegionKindRecursive, [&] { return visitor->VisitExpected(pair->lhs); }));
        TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->WithDefRegionKind(
            kTVMFFIDefRegionKindNonRecursive, [&] { return visitor->VisitExpected(pair->rhs); }));
        return Optional<VisitInterrupt>(std::nullopt);
      },
      [&](const TVarWithDepObj* var,
          StructuralVisitorObj* visitor) -> Expected<Optional<VisitInterrupt>> {
        trace.emplace_back(var->name, visitor->def_region_kind());
        if (var->name == "stop") {
          return Optional<VisitInterrupt>(VisitInterrupt(String("found stop")));
        }
        return Optional<VisitInterrupt>(std::nullopt);
      });
  ASSERT_TRUE(result.is_ok());
  ASSERT_TRUE(result.value().has_value());
  EXPECT_EQ(result.value().value()->value.cast<String>(), "found stop");
  ASSERT_EQ(trace.size(), 2u);
  EXPECT_EQ(trace[0], std::make_pair(std::string("lhs"), kTVMFFIDefRegionKindRecursive));
  EXPECT_EQ(trace[1], std::make_pair(std::string("stop"), kTVMFFIDefRegionKindNonRecursive));

  using CallbackLayer = StructuralWalkWithVisitCount<>;
  using ComposedLayer = StructuralVisitOuterLayer<CallbackLayer>;
  bool layer_callback_ran = false;
  bool base_callback_ran = false;
  auto layer_callback = [&](const TVarWithDepObj*,
                            CallbackLayer* visitor) -> Expected<Optional<VisitInterrupt>> {
    EXPECT_EQ(visitor->callback_tag(), 7);
    layer_callback_ran = true;
    return Optional<VisitInterrupt>(std::nullopt);
  };
  auto base_callback = [&](const TVarObj*, CallbackLayer*) -> Expected<Optional<VisitInterrupt>> {
    base_callback_ran = true;
    return Optional<VisitInterrupt>(std::nullopt);
  };
  using ComposedEngine =
      StructuralVisitEngine<ComposedLayer, decltype(layer_callback), decltype(base_callback)>;
  static_assert(std::is_same_v<typename ComposedEngine::VisitorObjType, CallbackLayer>);
  StructuralVisitor composed(
      make_object<ComposedEngine>(std::move(layer_callback), std::move(base_callback)));
  ASSERT_FALSE(composed->VisitExpected(Array<ObjectRef>{TVarWithDep("layer"), TVar("base")})
                   .value()
                   .has_value());
  EXPECT_TRUE(layer_callback_ran);
  EXPECT_TRUE(base_callback_ran);

  std::vector<std::string> error_trace;
  result = StructuralVisitExpected(
      Array<ObjectRef>{TVar("throw"), TVar("after")},
      [&](const TVarObj* var, StructuralVisitorObj*) -> Expected<Optional<VisitInterrupt>> {
        error_trace.emplace_back(var->name);
        if (var->name == "throw") {
          TVM_FFI_THROW(ValueError) << "visit callback threw";
        }
        return Optional<VisitInterrupt>(std::nullopt);
      });
  ASSERT_TRUE(result.is_err());
  EXPECT_EQ(result.error().message(), "visit callback threw");
  ExpectTrace(error_trace, {"throw"});
}

TEST(StructuralVisit, CallbackVisitsLhsOnly) {
  TPair root(TPair(TVar("lhs"), TVar("inner-rhs")),
             TPair(TVar("root-rhs-lhs"), TVar("root-rhs-rhs")));
  std::vector<std::string> trace;

  Expected<Optional<VisitInterrupt>> result = StructuralVisitExpected(
      root,
      [](const TPair& pair, StructuralVisitorObj* visitor) -> Expected<Optional<VisitInterrupt>> {
        TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(pair->lhs));
        return Optional<VisitInterrupt>(std::nullopt);
      },
      [&](const TVar& var, StructuralVisitorObj*) -> Expected<Optional<VisitInterrupt>> {
        trace.emplace_back(var->name);
        return Optional<VisitInterrupt>(std::nullopt);
      });

  ASSERT_TRUE(result.is_ok());
  EXPECT_FALSE(result.value().has_value());
  ExpectTrace(trace, {"lhs"});
}

}  // namespace
