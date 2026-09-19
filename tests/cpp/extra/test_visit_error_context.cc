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
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/extra/visit_error_context.h>
#include <tvm/ffi/object.h>
#include <tvm/ffi/reflection/access_path.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>

namespace {

using namespace tvm::ffi;
namespace refl = tvm::ffi::reflection;

// ---------------------------------------------------------------------------
// Test fixture: TPair — a minimal two-field Object with lhs and rhs slots.
//
// Covers kAttr AccessKind via its named fields.
// Array<ObjectRef> and Map<Any, Any> are used alongside TPair in individual
// tests to cover kArrayItem and kMapItem.
// ---------------------------------------------------------------------------

class TPair : public Object {
 public:
  ObjectRef lhs;
  ObjectRef rhs;

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("test.TPair", TPair, Object);
};

class TPairRef : public ObjectRef {
 public:
  TPairRef(ObjectRef lhs, ObjectRef rhs) {
    ObjectPtr<TPair> n = make_object<TPair>();
    n->lhs = std::move(lhs);
    n->rhs = std::move(rhs);
    data_ = std::move(n);
  }
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(TPairRef, ObjectRef, TPair);
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace r = tvm::ffi::reflection;
  r::ObjectDef<TPair>().def_ro("lhs", &TPair::lhs).def_ro("rhs", &TPair::rhs);
}

// ---------------------------------------------------------------------------
// Helper: Visit — recursively walks a tree and throws when it reaches
// throw_target, wrapping each recursion level with the visit
// macros so the error chain accumulates on unwind.
// ---------------------------------------------------------------------------

void Visit(const ObjectRef& node, const ObjectRef& throw_target);

void Visit(const ObjectRef& node, const ObjectRef& throw_target) {
  TVM_FFI_VISIT_BEGIN();

  if (node.same_as(throw_target)) {
    TVM_FFI_THROW(ValueError) << "boom";
  } else if (Optional<TPairRef> pair = node.as<TPairRef>()) {
    Visit(pair.value()->lhs, throw_target);
    Visit(pair.value()->rhs, throw_target);
  } else if (Optional<Array<ObjectRef>> arr = node.as<Array<ObjectRef>>()) {
    for (const ObjectRef& child : *arr) {
      Visit(child, throw_target);
    }
  } else if (Optional<Map<Any, Any>> m = node.as<Map<Any, Any>>()) {
    for (const std::pair<Any, Any>& kv : *m) {
      Visit(kv.second.cast<ObjectRef>(), throw_target);
    }
  }

  TVM_FFI_VISIT_END(node);
}

// ===========================================================================
// Layer A — Macro → Chain
//
// Verify that TVM_FFI_VISIT_BEGIN/_END correctly builds the
// reverse_visit_pattern as an exception unwinds through visit levels.
// ===========================================================================

// ---------------------------------------------------------------------------
// MacroBuildsChain: throw deep in a two-level visit; confirm the chain
// records nodes in innermost-first order (throw site first, root last).
// ---------------------------------------------------------------------------

TEST(VisitErrorContext, MacroBuildsChain) {
  // Tree: root = (lhs=leaf, rhs=empty)
  // Visiting root -> descends into leaf -> throws.
  Array<ObjectRef> empty;
  TPairRef leaf(empty, empty);
  TPairRef root(leaf, empty);

  Error caught("RuntimeError", "", "");
  bool did_catch = false;
  try {
    Visit(root, leaf);
  } catch (Error& err) {
    caught = err;
    did_catch = true;
  }

  EXPECT_TRUE(did_catch);

  Optional<VisitErrorContext> visit_context = VisitErrorContext::TryGetFromError(caught);
  ASSERT_TRUE(visit_context.has_value());

  // reverse_visit_pattern is innermost-first: leaf pushed first (inner catch
  // fires first during unwind), root pushed last.
  const List<ObjectRef>& chain = visit_context.value()->reverse_visit_pattern;
  EXPECT_GE(chain.size(), 2u);
  // Innermost entry is leaf (the throw site).
  EXPECT_TRUE(chain[0].same_as(leaf));
  // Outermost entry is root.
  EXPECT_TRUE(chain[chain.size() - 1].same_as(root));

  // Verify order structurally: [leaf, ..., root].
  List<ObjectRef> expected_chain = {leaf, root};
  EXPECT_TRUE(StructuralEqual::Equal(expected_chain, chain));
}

// ---------------------------------------------------------------------------
// MacroPreExistingPayloadWrap: when UpdateVisitErrorContext is called on
// an error that already has a non-context extra_context payload, the existing
// payload is stashed in prev_error_context and the visit chain starts
// fresh.  Subsequent pushes append to the same context object; prev_error_context
// is not modified.
// ---------------------------------------------------------------------------

TEST(VisitErrorContext, MacroPreExistingPayloadWrap) {
  // Stand-in for a pre-existing extra_context payload.
  Array<ObjectRef> empty;
  TPairRef existing_payload(empty, empty);
  TPairRef leaf(empty, empty);
  TPairRef root(leaf, empty);

  // Construct an error that already carries a non-context extra_context payload.
  Error err("RuntimeError", "test", "", std::nullopt, std::optional<ObjectRef>(existing_payload));
  ASSERT_TRUE(err.extra_context().has_value());

  // First push: existing payload is not a VisitErrorContext → wrapped.
  tvm::ffi::details::UpdateVisitErrorContext(err, leaf);

  Optional<VisitErrorContext> visit_context = VisitErrorContext::TryGetFromError(err);
  ASSERT_TRUE(visit_context.has_value());
  ASSERT_EQ(visit_context.value()->reverse_visit_pattern.size(), 1u);
  EXPECT_TRUE(visit_context.value()->reverse_visit_pattern[0].same_as(leaf));
  ASSERT_TRUE(visit_context.value()->prev_error_context.has_value());
  EXPECT_TRUE(visit_context.value()->prev_error_context.value().same_as(existing_payload));

  // Second push: context already exists → node appended; prev_error_context unchanged.
  tvm::ffi::details::UpdateVisitErrorContext(err, root);

  Optional<VisitErrorContext> visit_context_after = VisitErrorContext::TryGetFromError(err);
  ASSERT_TRUE(visit_context_after.has_value());

  List<ObjectRef> expected_chain = {leaf, root};
  EXPECT_TRUE(
      StructuralEqual::Equal(expected_chain, visit_context_after.value()->reverse_visit_pattern));

  ASSERT_TRUE(visit_context_after.value()->prev_error_context.has_value());
  EXPECT_TRUE(visit_context_after.value()->prev_error_context.value().same_as(existing_payload));
}

// ---------------------------------------------------------------------------
// MacroThrowBuildsChain: TVM_FFI_VISIT_THROW seeds the
// VisitErrorContext with the throw-site node directly (no separate
// UpdateVisitErrorContext call). Enclosing VISIT_BEGIN/END frames then
// append parent nodes on rethrow.
//
// Sub-scenario A — standalone: THROW outside any BEGIN/END wrapper produces
// a context whose reverse_visit_pattern is exactly [throw_site].
//
// Sub-scenario B — nested under BEGIN/END: when THROW fires inside a
// VISIT_BEGIN/END(root) pair, the enclosing END appends root. The throw site
// itself is recorded by THROW. If the throw_target happens to equal `node`
// at the surrounding level (a common case), FindAccessPaths' cleanup
// collapses the consecutive duplicate; tested separately in
// FindAccessPaths.RecordsCleanup.
// ---------------------------------------------------------------------------

TEST(VisitErrorContext, MacroThrowBuildsChain) {
  Array<ObjectRef> empty;

  // Sub-scenario A — standalone THROW (no surrounding BEGIN/END).
  {
    TPairRef throw_site(empty, empty);
    bool did_catch = false;
    Error caught("placeholder", "", "");
    try {
      TVM_FFI_VISIT_THROW(ValueError, throw_site) << "boom";
    } catch (Error& err) {
      caught = err;
      did_catch = true;
    }
    ASSERT_TRUE(did_catch);

    Optional<VisitErrorContext> visit_context = VisitErrorContext::TryGetFromError(caught);
    ASSERT_TRUE(visit_context.has_value());
    const List<ObjectRef>& chain = visit_context.value()->reverse_visit_pattern;
    ASSERT_EQ(chain.size(), 1u);
    EXPECT_TRUE(chain[0].same_as(throw_site));
  }

  // Sub-scenario B — THROW inside a BEGIN/END(root) wrapper.
  {
    TPairRef throw_site(empty, empty);
    TPairRef root(throw_site, empty);
    bool did_catch = false;
    Error caught("placeholder", "", "");
    try {
      TVM_FFI_VISIT_BEGIN();
      TVM_FFI_VISIT_THROW(ValueError, throw_site) << "boom";
      TVM_FFI_VISIT_END(root);
    } catch (Error& err) {
      caught = err;
      did_catch = true;
    }
    ASSERT_TRUE(did_catch);

    Optional<VisitErrorContext> visit_context = VisitErrorContext::TryGetFromError(caught);
    ASSERT_TRUE(visit_context.has_value());
    const List<ObjectRef>& chain = visit_context.value()->reverse_visit_pattern;
    // THROW seeds with throw_site; END appends root.
    List<ObjectRef> expected = {throw_site, root};
    EXPECT_TRUE(StructuralEqual::Equal(expected, chain));
  }
}

// ===========================================================================
// Layer B — Chain → AccessPaths
//
// Verify that FindAccessPaths(root, ctx) resolves a manually-constructed
// VisitErrorContext against a tree, producing AccessPaths that describe
// where matched nodes live in the tree.
//
// Each test:
//   1. Builds a root tree.
//   2. Constructs a VisitErrorContext with a specific reverse_visit_pattern.
//   3. Calls FindAccessPaths(root, ctx).
//   4. Asserts on the result using StructuralEqual::Equal on AccessPath
//      (mirrors the pattern used in test_structural_equal_hash.cc).
// ===========================================================================

// ---------------------------------------------------------------------------
// BasicMatch: unambiguous single-node path via two Attr steps; and CSE
// multi-match when the same pointer appears in two sibling slots.
//
// Sub-scenario A — SingleMatch:
//   Tree: root.lhs = mid, mid.lhs = leaf
//   Pattern: [leaf, mid, root]  (innermost-first)
//   Expected path: Root->Attr("lhs")->Attr("lhs")
//
// Sub-scenario B — CSEMultiMatch:
//   Tree: root.lhs = shared, root.rhs = shared  (identical pointer)
//   Pattern: [shared, root]
//   Expected: two paths, one for each slot (Attr("lhs") and Attr("rhs")).
// ---------------------------------------------------------------------------

TEST(FindAccessPaths, BasicMatch) {
  Array<ObjectRef> empty;

  // Sub-scenario A: single unambiguous path.
  {
    TPairRef leaf(empty, empty);
    TPairRef mid(leaf, empty);
    TPairRef root(mid, empty);

    ObjectPtr<VisitErrorContextObj> ctx_obj = make_object<VisitErrorContextObj>();
    ctx_obj->reverse_visit_pattern = List<ObjectRef>{leaf, mid, root};
    VisitErrorContext ctx(std::move(ctx_obj));

    Array<refl::AccessPath> paths = VisitErrorContext::FindAccessPaths(root, ctx);
    ASSERT_EQ(paths.size(), 1u);

    refl::AccessPath expected = refl::AccessPath::Root()->Attr("lhs")->Attr("lhs");
    EXPECT_TRUE(StructuralEqual::Equal(expected, paths[0]));
  }

  // Sub-scenario B: same pointer in two slots → two paths.
  {
    TPairRef shared(empty, empty);
    TPairRef root(shared, shared);

    ObjectPtr<VisitErrorContextObj> ctx_obj = make_object<VisitErrorContextObj>();
    ctx_obj->reverse_visit_pattern = List<ObjectRef>{shared, root};
    VisitErrorContext ctx(std::move(ctx_obj));

    Array<refl::AccessPath> paths = VisitErrorContext::FindAccessPaths(root, ctx);
    ASSERT_EQ(paths.size(), 2u);

    refl::AccessPath expected_lhs = refl::AccessPath::Root()->Attr("lhs");
    refl::AccessPath expected_rhs = refl::AccessPath::Root()->Attr("rhs");
    bool found_lhs = false;
    bool found_rhs = false;
    for (const refl::AccessPath& p : paths) {
      if (StructuralEqual::Equal(p, expected_lhs)) found_lhs = true;
      if (StructuralEqual::Equal(p, expected_rhs)) found_rhs = true;
    }
    EXPECT_TRUE(found_lhs);
    EXPECT_TRUE(found_rhs);
  }
}

// ---------------------------------------------------------------------------
// AccessKindCoverage: exercises Attr (TPair fields), ArrayItem
// (Array<ObjectRef>), and MapItem (Map<Any, Any>) access kinds.
//
// Sub-scenario A — Attr (kAttr):
//   Tree: root.lhs = mid, mid.lhs = leaf
//   Pattern: [leaf, root]
//   Expected path: Root->Attr("lhs")->Attr("lhs")
//
// Sub-scenario B — ArrayItem (kArrayItem):
//   Tree: root.lhs = [leaf1, leaf2]  (Array in ObjectRef slot)
//   Pattern: [leaf1, root]
//   Expected path: Root->Attr("lhs")->ArrayItem(0)
//
// Sub-scenario C — MapItem (kMapItem):
//   Tree: root.lhs = {"key" -> target}  (Map in ObjectRef slot)
//   Pattern: [target, root]
//   Expected path: Root->Attr("lhs")->MapItem("key")
// ---------------------------------------------------------------------------

TEST(FindAccessPaths, AccessKindCoverage) {
  Array<ObjectRef> empty;

  // Sub-scenario A: Attr steps.
  {
    TPairRef leaf(empty, empty);
    TPairRef mid(leaf, empty);
    TPairRef root(mid, empty);

    ObjectPtr<VisitErrorContextObj> ctx_obj = make_object<VisitErrorContextObj>();
    ctx_obj->reverse_visit_pattern = List<ObjectRef>{leaf, root};
    VisitErrorContext ctx(std::move(ctx_obj));

    Array<refl::AccessPath> paths = VisitErrorContext::FindAccessPaths(root, ctx);
    ASSERT_EQ(paths.size(), 1u);

    refl::AccessPath expected = refl::AccessPath::Root()->Attr("lhs")->Attr("lhs");
    EXPECT_TRUE(StructuralEqual::Equal(expected, paths[0]));
  }

  // Sub-scenario B: ArrayItem step.
  {
    TPairRef leaf1(empty, empty);
    TPairRef leaf2(empty, empty);
    Array<ObjectRef> arr = {leaf1, leaf2};
    TPairRef root(arr, empty);

    ObjectPtr<VisitErrorContextObj> ctx_obj = make_object<VisitErrorContextObj>();
    ctx_obj->reverse_visit_pattern = List<ObjectRef>{leaf1, root};
    VisitErrorContext ctx(std::move(ctx_obj));

    Array<refl::AccessPath> paths = VisitErrorContext::FindAccessPaths(root, ctx);
    ASSERT_EQ(paths.size(), 1u);

    refl::AccessPath expected = refl::AccessPath::Root()->Attr("lhs")->ArrayItem(0);
    EXPECT_TRUE(StructuralEqual::Equal(expected, paths[0]));
  }

  // Sub-scenario C: MapItem step.
  {
    TPairRef target(empty, empty);
    Map<Any, Any> m;
    m.Set(String("key"), target);
    TPairRef root(m, empty);

    ObjectPtr<VisitErrorContextObj> ctx_obj = make_object<VisitErrorContextObj>();
    ctx_obj->reverse_visit_pattern = List<ObjectRef>{target, root};
    VisitErrorContext ctx(std::move(ctx_obj));

    Array<refl::AccessPath> paths = VisitErrorContext::FindAccessPaths(root, ctx);
    ASSERT_EQ(paths.size(), 1u);

    refl::AccessPath expected = refl::AccessPath::Root()->Attr("lhs")->MapItem(String("key"));
    EXPECT_TRUE(StructuralEqual::Equal(expected, paths[0]));
  }
}

// ---------------------------------------------------------------------------
// SparsePatternAnchors: demonstrates that intermediate breadcrumb anchors in
// the pattern disambiguate when an innermost target appears in multiple branches.
//
// Tree layout:
//   root
//   ├─ branch_a: tpa → m_inner_a   (m_inner shared, no m_outer ancestor)
//   └─ branch_b: tpb → m_outer → tpc → m_inner_b  (same m_inner pointer)
//
// m_inner is a POINTER-IDENTICAL ObjectRef (ref-count shared) used as both
// m_inner_a (under branch_a) and m_inner_b (under branch_b via m_outer).
// m_outer appears only under branch_b.
//
// Scenario 1: pattern with only innermost — 2 candidate paths.
//   Pattern: [m_inner, root]
//   Expected: paths.size() == 2
//
// Scenario 2: anchor narrows to branch_b only.
//   Pattern: [m_inner, m_outer, root]
//   Expected: paths.size() == 1, path through branch_b
// ---------------------------------------------------------------------------

TEST(FindAccessPaths, SparsePatternAnchors) {
  Array<ObjectRef> empty;

  // Shared inner node — identical pointer in both branches.
  TPairRef m_inner(empty, empty);

  // branch_a: tpa.lhs = m_inner (no m_outer ancestor)
  TPairRef tpa(m_inner, empty);

  // branch_b: tpb.lhs = m_outer, m_outer.lhs = tpc, tpc.lhs = m_inner
  TPairRef tpc(m_inner, empty);
  TPairRef m_outer(tpc, empty);
  TPairRef tpb(m_outer, empty);

  // root: lhs = tpa (branch_a), rhs = tpb (branch_b)
  TPairRef root(tpa, tpb);

  // Scenario 1: pattern = [m_inner, root] → 2 paths (one per branch).
  {
    ObjectPtr<VisitErrorContextObj> ctx_obj = make_object<VisitErrorContextObj>();
    ctx_obj->reverse_visit_pattern = List<ObjectRef>{m_inner, root};
    VisitErrorContext ctx(std::move(ctx_obj));

    Array<refl::AccessPath> paths = VisitErrorContext::FindAccessPaths(root, ctx);
    EXPECT_EQ(paths.size(), 2u);
  }

  // Scenario 2: pattern = [m_inner, m_outer, root] → 1 path through branch_b.
  {
    ObjectPtr<VisitErrorContextObj> ctx_obj = make_object<VisitErrorContextObj>();
    ctx_obj->reverse_visit_pattern = List<ObjectRef>{m_inner, m_outer, root};
    VisitErrorContext ctx(std::move(ctx_obj));

    Array<refl::AccessPath> paths = VisitErrorContext::FindAccessPaths(root, ctx);
    ASSERT_EQ(paths.size(), 1u);

    // Path must go through branch_b (rhs), then m_outer (lhs), then tpc (lhs), then m_inner (lhs).
    refl::AccessPath expected =
        refl::AccessPath::Root()->Attr("rhs")->Attr("lhs")->Attr("lhs")->Attr("lhs");
    EXPECT_TRUE(StructuralEqual::Equal(expected, paths[0]));
  }
}

// ---------------------------------------------------------------------------
// PartialChain: strict mode returns nothing when the innermost pattern entry
// is unreachable; prefix-match mode returns the deepest matched prefix path.
//
// Sub-scenario A — strict (allow_prefix_match=false):
//   Pattern: [unreachable, child, root]
//   Expected: 0 results (unreachable breaks full match)
//
// Sub-scenario B — prefix match (allow_prefix_match=true):
//   Same pattern, same tree.
//   Expected: at least one result at the path to child (Root->Attr("lhs")).
// ---------------------------------------------------------------------------

TEST(FindAccessPaths, PartialChain) {
  Array<ObjectRef> empty;
  // unreachable is not present anywhere in the tree below root.
  TPairRef unreachable(empty, empty);
  TPairRef child(empty, empty);
  TPairRef root(child, empty);

  ObjectPtr<VisitErrorContextObj> ctx_obj = make_object<VisitErrorContextObj>();
  ctx_obj->reverse_visit_pattern = List<ObjectRef>{unreachable, child, root};
  VisitErrorContext ctx(std::move(ctx_obj));

  // Sub-scenario A: strict mode — no full match.
  {
    Array<refl::AccessPath> paths = VisitErrorContext::FindAccessPaths(root, ctx);
    EXPECT_EQ(paths.size(), 0u);
  }

  // Sub-scenario B: prefix-match mode — path to child reported.
  {
    Array<refl::AccessPath> paths =
        VisitErrorContext::FindAccessPaths(root, ctx, /*allow_prefix_match=*/true);
    ASSERT_GE(paths.size(), 1u);

    refl::AccessPath expected = refl::AccessPath::Root()->Attr("lhs");
    bool found_expected = false;
    for (const refl::AccessPath& p : paths) {
      if (StructuralEqual::Equal(p, expected)) {
        found_expected = true;
        break;
      }
    }
    EXPECT_TRUE(found_expected);
  }
}

// ---------------------------------------------------------------------------
// EdgeCases: empty pattern produces no results; null entries in the pattern
// never match any live node and do not crash.
// ---------------------------------------------------------------------------

TEST(FindAccessPaths, EdgeCases) {
  Array<ObjectRef> empty;

  // Sub-scenario A: empty pattern.
  {
    TPairRef root(empty, empty);

    ObjectPtr<VisitErrorContextObj> ctx_obj = make_object<VisitErrorContextObj>();
    // reverse_visit_pattern left empty.
    VisitErrorContext ctx(std::move(ctx_obj));

    Array<refl::AccessPath> paths = VisitErrorContext::FindAccessPaths(root, ctx);
    EXPECT_EQ(paths.size(), 0u);
  }

  // Sub-scenario A2: root-itself match — pattern is [root], the throw fired
  // before any descent. Expected: a single AccessPath equal to Root() (no
  // crash from materializing on an empty descent stack).
  {
    TPairRef root(empty, empty);

    ObjectPtr<VisitErrorContextObj> ctx_obj = make_object<VisitErrorContextObj>();
    ctx_obj->reverse_visit_pattern = List<ObjectRef>{root};
    VisitErrorContext ctx(std::move(ctx_obj));

    Array<refl::AccessPath> paths = VisitErrorContext::FindAccessPaths(root, ctx);
    ASSERT_EQ(paths.size(), 1u);
    EXPECT_TRUE(StructuralEqual::Equal(refl::AccessPath::Root(), paths[0]));
  }

  // Sub-scenario B: null entries in pattern.
  {
    TPairRef root(empty, empty);

    ObjectPtr<VisitErrorContextObj> ctx_obj = make_object<VisitErrorContextObj>();
    ObjectRef null_ref;
    ctx_obj->reverse_visit_pattern = List<ObjectRef>{null_ref, null_ref};
    VisitErrorContext ctx(std::move(ctx_obj));

    Array<refl::AccessPath> paths;
    ASSERT_NO_THROW({ paths = VisitErrorContext::FindAccessPaths(root, ctx); });
    EXPECT_EQ(paths.size(), 0u);
  }
}

// ---------------------------------------------------------------------------
// RecordsCleanup: FindAccessPaths normalizes reverse_visit_pattern before
// matching by (a) dropping null entries and (b) collapsing runs of
// consecutive duplicates. Both arise naturally — null from stale/torn-down
// nodes; duplicates when TVM_FFI_VISIT_THROW(node) is wrapped by
// a TVM_FFI_VISIT_END(node) at the same level. Without cleanup
// the redundant frame would over-constrain the match and yield zero paths.
// ---------------------------------------------------------------------------

TEST(FindAccessPaths, RecordsCleanup) {
  Array<ObjectRef> empty;
  TPairRef leaf(empty, empty);
  TPairRef root(leaf, empty);
  ObjectRef null_ref;

  refl::AccessPath expected = refl::AccessPath::Root()->Attr("lhs");

  // Sub-scenario A — consecutive duplicates collapse.
  // Raw pattern: [leaf, leaf, root, root]; effective: [leaf, root].
  {
    ObjectPtr<VisitErrorContextObj> ctx_obj = make_object<VisitErrorContextObj>();
    ctx_obj->reverse_visit_pattern = List<ObjectRef>{leaf, leaf, root, root};
    VisitErrorContext ctx(std::move(ctx_obj));

    Array<refl::AccessPath> paths = VisitErrorContext::FindAccessPaths(root, ctx);
    ASSERT_EQ(paths.size(), 1u);
    EXPECT_TRUE(StructuralEqual::Equal(expected, paths[0]));
  }

  // Sub-scenario B — nulls are skipped, surrounding entries still match.
  // Raw pattern: [null, leaf, null, root, null]; effective: [leaf, root].
  {
    ObjectPtr<VisitErrorContextObj> ctx_obj = make_object<VisitErrorContextObj>();
    ctx_obj->reverse_visit_pattern = List<ObjectRef>{null_ref, leaf, null_ref, root, null_ref};
    VisitErrorContext ctx(std::move(ctx_obj));

    Array<refl::AccessPath> paths = VisitErrorContext::FindAccessPaths(root, ctx);
    ASSERT_EQ(paths.size(), 1u);
    EXPECT_TRUE(StructuralEqual::Equal(expected, paths[0]));
  }

  // Sub-scenario C — combined: nulls plus dup-consecutive.
  // Raw pattern: [leaf, null, leaf, root, root, null]; effective: [leaf, root].
  {
    ObjectPtr<VisitErrorContextObj> ctx_obj = make_object<VisitErrorContextObj>();
    ctx_obj->reverse_visit_pattern = List<ObjectRef>{leaf, null_ref, leaf, root, root, null_ref};
    VisitErrorContext ctx(std::move(ctx_obj));

    Array<refl::AccessPath> paths = VisitErrorContext::FindAccessPaths(root, ctx);
    ASSERT_EQ(paths.size(), 1u);
    EXPECT_TRUE(StructuralEqual::Equal(expected, paths[0]));
  }
}

// ---------------------------------------------------------------------------
// TryGetFromError: returns NullOpt when absent, the context when present.
// Composes correctly with FindAccessPaths.
// ---------------------------------------------------------------------------

TEST(VisitErrorContext, TryGetFromError) {
  Array<ObjectRef> empty;
  TPairRef leaf(empty, empty);
  TPairRef root(leaf, empty);

  Error err("RuntimeError", "test", "");

  // No context attached → TryGetFromError returns NullOpt.
  Optional<VisitErrorContext> no_context = VisitErrorContext::TryGetFromError(err);
  EXPECT_FALSE(no_context.has_value());

  // Attach context via UpdateVisitErrorContext.
  tvm::ffi::details::UpdateVisitErrorContext(err, leaf);
  tvm::ffi::details::UpdateVisitErrorContext(err, root);

  Optional<VisitErrorContext> visit_context = VisitErrorContext::TryGetFromError(err);
  ASSERT_TRUE(visit_context.has_value());

  // Compose TryGetFromError + FindAccessPaths → should resolve to root.lhs.
  Array<refl::AccessPath> paths = VisitErrorContext::FindAccessPaths(root, visit_context.value());
  ASSERT_EQ(paths.size(), 1u);

  refl::AccessPath expected = refl::AccessPath::Root()->Attr("lhs");
  EXPECT_TRUE(StructuralEqual::Equal(expected, paths[0]));
}

}  // namespace
