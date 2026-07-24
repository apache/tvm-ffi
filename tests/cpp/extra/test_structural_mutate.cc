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
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/extra/visit_error_context.h>
#include <tvm/ffi/object.h>
#include <tvm/ffi/reflection/accessor.h>

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "../testing_object.h"

namespace {

using namespace tvm::ffi;
using namespace tvm::ffi::testing;

struct VarVisit {
  std::string name;
  TVMFFIDefRegionKind def_region_kind;
};

class TestStructuralMutatorObj final : public StructuralMutatorObj {
 public:
  TestStructuralMutatorObj(std::string var_to_rename, bool increment_integers,
                           bool fail_on_negative)
      : StructuralMutatorObj(VTable()),
        var_to_rename_(std::move(var_to_rename)),
        increment_integers_(increment_integers),
        fail_on_negative_(fail_on_negative) {}

  std::vector<VarVisit> var_visits;

 private:
  static const StructuralMutatorVTable* VTable() {
    static const StructuralMutatorVTable vtable{
        &TestStructuralMutatorObj::DispatchMutate,
        &TestStructuralMutatorObj::DispatchMaybeInplaceMutate,
        &TestStructuralMutatorObj::DispatchVarRemapGet,
        &TestStructuralMutatorObj::DispatchVarRemapSet,
    };
    return &vtable;
  }

  static TVMFFIAny Finish(AnyView value, Expected<Any> result) noexcept {
    if (TVM_FFI_PREDICT_FALSE(result.is_err()) &&
        value.type_index() >= TypeIndex::kTVMFFIStaticObjectBegin) {
      Error error = result.error();
      details::UpdateVisitErrorContext(error, value.cast<ObjectRef>());
    }
    return details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(result));
  }

  // NOLINTNEXTLINE(bugprone-exception-escape)
  static TVMFFIAny DispatchMutate(StructuralMutatorObj* mutator, AnyView value) noexcept {
    auto* self = static_cast<TestStructuralMutatorObj*>(mutator);
    return Finish(value, self->MutateImpl(value, false));
  }

  // NOLINTNEXTLINE(bugprone-exception-escape)
  static TVMFFIAny DispatchMaybeInplaceMutate(StructuralMutatorObj* mutator,
                                              AnyView value) noexcept {
    auto* self = static_cast<TestStructuralMutatorObj*>(mutator);
    return Finish(value, self->MutateImpl(value, true));
  }

  static TVMFFIAny DispatchVarRemapGet(StructuralMutatorObj* mutator, AnyView var) noexcept {
    auto* self = static_cast<TestStructuralMutatorObj*>(mutator);
    return details::ExpectedUnsafe::MoveToTVMFFIAny(self->VarRemapGetImpl(var));
  }

  static TVMFFIAny DispatchVarRemapSet(StructuralMutatorObj* mutator, AnyView var,
                                       AnyView mapped_value) noexcept {
    auto* self = static_cast<TestStructuralMutatorObj*>(mutator);
    return details::ExpectedUnsafe::MoveToTVMFFIAny(self->VarRemapSetImpl(var, mapped_value));
  }

  Expected<Any> VarRemapGetImpl(AnyView var) noexcept {
    if (var.type_index() < TypeIndex::kTVMFFIStaticObjectBegin) {
      return Unexpected(
          Error("TypeError", "Variable-remap key must be an object-backed value", ""));
    }
    try {
      ObjectRef var_ref = var.cast<ObjectRef>();
      std::optional<Any> result = var_remap_.Get(var_ref);
      if (!result.has_value()) {
        return Any(nullptr);
      }
      return *std::move(result);
    } catch (const Error& err) {
      return Unexpected(err);
    }
  }

  Expected<void> VarRemapSetImpl(AnyView var, AnyView mapped_value) noexcept {
    if (var.type_index() < TypeIndex::kTVMFFIStaticObjectBegin) {
      return Unexpected(
          Error("TypeError", "Variable-remap key must be an object-backed value", ""));
    }
    try {
      ObjectRef var_ref = var.cast<ObjectRef>();
      var_remap_.Set(var_ref, Any(mapped_value));
      return Expected<void>();
    } catch (const Error& err) {
      return Unexpected(err);
    }
  }

  Expected<Any> MutateImpl(AnyView value, bool maybe_inplace) {
    if (value.type_index() == TypeIndex::kTVMFFIInt) {
      int64_t integer = value.cast<int64_t>();
      if (fail_on_negative_ && integer < 0) {
        return Unexpected(Error("ValueError", "negative integer", ""));
      }
      return increment_integers_ ? Any(integer + 1) : Any(value);
    }
    if (const auto* var = value.as<TVarObj>()) {
      var_visits.push_back(VarVisit{var->name, def_region_kind()});
      if (var->name == "error") {
        return Unexpected(Error("ValueError", "cannot mutate error variable", ""));
      }
      if (var->name == var_to_rename_) {
        return Any(TVar(var->name + "-mapped"));
      }
      return Any(value);
    }
    return maybe_inplace ? DefaultMaybeInplaceMutateExpected(value) : DefaultMutateExpected(value);
  }

  Map<ObjectRef, Any> var_remap_;
  std::string var_to_rename_;
  bool increment_integers_;
  bool fail_on_negative_;
};

class TestStructuralMutator : public StructuralMutator {
 public:
  explicit TestStructuralMutator(std::string var_to_rename = "", bool increment_integers = false,
                                 bool fail_on_negative = false)
      : StructuralMutator(make_object<TestStructuralMutatorObj>(
            std::move(var_to_rename), increment_integers, fail_on_negative)) {}

  TestStructuralMutatorObj* impl() const {
    return static_cast<TestStructuralMutatorObj*>(StructuralMutator::get());
  }
};

void CheckCustomFreeVarHookRemap(bool maybe_inplace, const std::string& expected_leaf_name) {
  constexpr const char* kTypeKey = "testing.StructuralMutateHookedVar";
  Function init = reflection::GetMethod(kTypeKey, "__ffi_init__");
  reflection::FieldGetter get_name(kTypeKey, "name");
  reflection::FieldGetter get_dependency(kTypeKey, "dependency");

  ObjectRef leaf = init("n", nullptr).cast<ObjectRef>();
  ObjectRef dependent = init("x", leaf).cast<ObjectRef>();
  // [n, x(n), x(n)] maps to [n', x(n'), x(n')], with both x(n') entries sharing identity.
  Array<Any> root{Any(leaf), Any(dependent), Any(dependent)};
  TestStructuralMutator mutator;

  Array<Any> mapped = (maybe_inplace ? mutator->MaybeInplaceMutate(root) : mutator->Mutate(root))
                          .cast<Array<Any>>();

  ObjectRef mapped_leaf = mapped[0].cast<ObjectRef>();
  ObjectRef mapped_dependent = mapped[1].cast<ObjectRef>();
  EXPECT_EQ(get_name(mapped_leaf).cast<String>(), expected_leaf_name);
  EXPECT_EQ(get_name(mapped_dependent).cast<String>(), "x");
  EXPECT_NE(mapped_leaf.get(), leaf.get());
  EXPECT_NE(mapped_dependent.get(), dependent.get());
  EXPECT_TRUE(get_dependency(mapped_dependent).cast<ObjectRef>().same_as(mapped_leaf));
  EXPECT_TRUE(mapped[1].same_as(mapped[2]));
}

TEST(StructuralMutator, HandlesPODAndPerInstanceVariableRemapAPIs) {
  TestStructuralMutator mutator;

  EXPECT_EQ(mutator->Mutate(int64_t{42}).cast<int64_t>(), 42);

  TVar var("x");
  Expected<Any> missing = mutator->VarRemapGetExpected(var);
  ASSERT_TRUE(missing.is_ok());
  EXPECT_EQ(details::ExpectedUnsafe::GetData(missing).type_index(), TypeIndex::kTVMFFINone);

  ASSERT_TRUE(mutator->VarRemapSetExpected(var, int64_t{10}).is_ok());
  EXPECT_EQ(mutator->VarRemapGetExpected(var).value().cast<int64_t>(), 10);
  ASSERT_TRUE(mutator->VarRemapSetExpected(var, int64_t{11}).is_ok());
  EXPECT_EQ(mutator->VarRemapGetExpected(var).value().cast<int64_t>(), 11);

  TestStructuralMutator other_mutator;
  Expected<Any> missing_from_other = other_mutator->VarRemapGetExpected(var);
  ASSERT_TRUE(missing_from_other.is_ok());
  EXPECT_EQ(details::ExpectedUnsafe::GetData(missing_from_other).type_index(),
            TypeIndex::kTVMFFINone);
  ASSERT_TRUE(other_mutator->VarRemapSetExpected(var, int64_t{20}).is_ok());
  EXPECT_EQ(other_mutator->VarRemapGetExpected(var).value().cast<int64_t>(), 20);
  EXPECT_EQ(mutator->VarRemapGetExpected(var).value().cast<int64_t>(), 11);

  Expected<Any> invalid = mutator->VarRemapGetExpected(int64_t{1});
  ASSERT_TRUE(invalid.is_err());
  EXPECT_EQ(invalid.error().kind(), "TypeError");
}

TEST(StructuralMutator, CustomFreeVarMutateHookRemapsDependentOccurrences) {
  CheckCustomFreeVarHookRemap(false, "n-mutated");
}

TEST(StructuralMutator, CustomFreeVarMaybeInplaceHookRemapsDependentOccurrences) {
  CheckCustomFreeVarHookRemap(true, "n-maybe-mutated");
}

TEST(StructuralMutator, MutateReturnsOriginalOrCopiesOnlyChangedPath) {
  {
    TPair root(TVar("lhs"), TVar("rhs"));
    TestStructuralMutator mutator("absent");

    TPair mapped = mutator->Mutate(root).cast<TPair>();

    EXPECT_EQ(mapped.get(), root.get());
    EXPECT_TRUE(mapped->lhs.same_as(root->lhs));
    EXPECT_TRUE(mapped->rhs.same_as(root->rhs));
  }

  TPair middle(TVar("target"), TVar("middle-sibling"));
  const TPairObj* middle_addr = middle.get();
  const Object* target_addr = middle->lhs.get();
  TPair root(std::move(middle), TVar("outer-sibling"));
  const Object* root_addr = root.get();
  TestStructuralMutator mutator("target");

  TPair mapped = mutator->Mutate(root).cast<TPair>();
  const TPairObj* mapped_middle = mapped->lhs.as<TPairObj>();
  ASSERT_NE(mapped_middle, nullptr);
  EXPECT_NE(mapped.get(), root_addr);
  EXPECT_NE(mapped_middle, middle_addr);
  EXPECT_NE(mapped_middle->lhs.get(), target_addr);
  EXPECT_EQ(mapped_middle->lhs.as_or_throw<TVar>()->name, "target-mapped");
  EXPECT_TRUE(mapped_middle->rhs.same_as(middle_addr->rhs));
  EXPECT_TRUE(mapped->rhs.same_as(root->rhs));
  EXPECT_EQ(middle_addr->lhs.as_or_throw<TVar>()->name, "target");
}

TEST(StructuralMutator, MaybeInplaceMutateRequiresExplicitHookForReuse) {
  // A reflected type without __s_maybe_inplace_mutate__ falls back to Mutate, even when unique.
  {
    TPair root(TVar("target"), TVar("sibling"));
    const Object* root_addr = root.get();
    TestStructuralMutator mutator("target");

    TPair mapped = mutator->MaybeInplaceMutate(root).cast<TPair>();

    EXPECT_NE(mapped.get(), root_addr);
    EXPECT_EQ(root->lhs.as_or_throw<TVar>()->name, "target");
    EXPECT_EQ(mapped->lhs.as_or_throw<TVar>()->name, "target-mapped");
  }

  // Array's explicit __s_maybe_inplace_mutate__ hook may reuse unique storage.
  Array<Any> root{Any(int64_t{1})};
  const Object* root_addr = root.get();
  TestStructuralMutator mutator("", true);

  Array<Any> mapped = mutator->MaybeInplaceMutate(root).cast<Array<Any>>();

  EXPECT_EQ(mapped.get(), root_addr);
  EXPECT_EQ(root[0].cast<int64_t>(), 2);
}

TEST(StructuralMutator, ReusesMappedFreeVarIdentity) {
  TVarWithDep var("x", TVar("n"));
  TPair root(var, var);
  TestStructuralMutator mutator("n");

  TPair mapped = mutator->Mutate(root).cast<TPair>();

  ASSERT_TRUE(mapped->lhs.same_as(mapped->rhs));
  TVarWithDep mapped_var = mapped->lhs.as_or_throw<TVarWithDep>();
  ASSERT_TRUE(mapped_var->dep.has_value());
  EXPECT_EQ(mapped_var->dep.value().as_or_throw<TVar>()->name, "n-mapped");
  ASSERT_EQ(mutator.impl()->var_visits.size(), 1U);
  EXPECT_EQ(mutator.impl()->var_visits[0].name, "n");
}

TEST(StructuralMutator, PropagatesAndRestoresDefRegionKind) {
  TDefHolder root(TVarWithDep("recursive", TVar("recursive-dep")),
                  TVarWithDep("non-recursive", TVar("non-recursive-dep")));
  TestStructuralMutator mutator;

  Any mapped = mutator->Mutate(root);

  EXPECT_TRUE(mapped.same_as(root));
  ASSERT_EQ(mutator.impl()->var_visits.size(), 2U);
  EXPECT_EQ(mutator.impl()->var_visits[0].def_region_kind, kTVMFFIDefRegionKindRecursive);
  EXPECT_EQ(mutator.impl()->var_visits[1].def_region_kind, kTVMFFIDefRegionKindNone);
  EXPECT_EQ(mutator->def_region_kind(), kTVMFFIDefRegionKindNone);
}

TEST(StructuralMutator, LeavesPartialInplaceChangesOnError) {
  Array<Any> source{Any(int64_t{1}), Any(int64_t{-1})};
  const Object* source_addr = source.get();
  TestStructuralMutator mutator("", true, true);

  Expected<Any> result = mutator->MaybeInplaceMutateExpected(source);

  ASSERT_TRUE(result.is_err());
  EXPECT_EQ(result.error().kind(), "ValueError");
  EXPECT_EQ(source.get(), source_addr);
  EXPECT_EQ(source[0].cast<int64_t>(), 2);
  EXPECT_EQ(source[1].cast<int64_t>(), -1);
}

TEST(StructuralMap, InvokesFirstMatchingCallback) {
  int any_view_callback_count = 0;
  int typed_callback_count = 0;

  Any result = StructuralMap<WalkOrder::kPostOrder>(
      int64_t{41},
      [&](AnyView value) -> Expected<Any> {
        ++any_view_callback_count;
        return Any(value.cast<int64_t>() + 1);
      },
      [&](int64_t value) -> Expected<Any> {
        ++typed_callback_count;
        return Any(value + 2);
      });

  EXPECT_EQ(result.cast<int64_t>(), 42);
  EXPECT_EQ(any_view_callback_count, 1);
  EXPECT_EQ(typed_callback_count, 0);
}

TEST(StructuralMap, SelectsMutationModeFromCallbackPresence) {
  auto map_var = [](const TVarObj* var) -> Expected<Any> {
    return Any(TVar(var->name + "-mapped"));
  };

  // An unmatched, uniquely owned container may use its explicit in-place hook.
  {
    Array<Any> root{Any(TVar("lhs")), Any(TVar("rhs"))};
    const Object* root_addr = root.get();

    Array<Any> mapped = StructuralMap<WalkOrder::kPostOrder>(root, map_var).cast<Array<Any>>();

    EXPECT_EQ(mapped.get(), root_addr);
    EXPECT_EQ(mapped[0].cast<TVar>()->name, "lhs-mapped");
    EXPECT_EQ(mapped[1].cast<TVar>()->name, "rhs-mapped");
  }

  // A callback on the container forces map semantics instead of invoking its in-place hook.
  {
    Array<Any> root{Any(TVar("lhs")), Any(TVar("rhs"))};
    const Object* root_addr = root.get();

    Array<Any> mapped =
        StructuralMap<WalkOrder::kPostOrder>(root, map_var, [](Array<Any> array) -> Expected<Any> {
          return Any(std::move(array));
        }).cast<Array<Any>>();

    EXPECT_NE(mapped.get(), root_addr);
    EXPECT_EQ(root[0].cast<TVar>()->name, "lhs");
    EXPECT_EQ(root[1].cast<TVar>()->name, "rhs");
    EXPECT_EQ(mapped[0].cast<TVar>()->name, "lhs-mapped");
    EXPECT_EQ(mapped[1].cast<TVar>()->name, "rhs-mapped");
  }
}

TEST(StructuralMap, ReusesFreeVarReplacement) {
  TVarWithDep var("x", TVar("n"));
  TPair root(var, var);
  int callback_count = 0;

  TPair mapped =
      StructuralMap<WalkOrder::kPostOrder>(root, [&](const TVarObj* dep) -> Expected<Any> {
        ++callback_count;
        return Any(TVar(dep->name + "-mapped"));
      }).cast<TPair>();

  ASSERT_EQ(callback_count, 1);
  ASSERT_TRUE(mapped->lhs.same_as(mapped->rhs));
  TVarWithDep mapped_var = mapped->lhs.as_or_throw<TVarWithDep>();
  ASSERT_TRUE(mapped_var->dep.has_value());
  EXPECT_EQ(mapped_var->dep.value().as_or_throw<TVar>()->name, "n-mapped");
}

TEST(StructuralMap, InvokesCallbacksInConfiguredOrder) {
  auto run = [](WalkOrder order) {
    TPair root(TVar("lhs"), TVar("rhs"));
    std::vector<std::string> trace;
    auto map_pair = [&](const TPairObj* pair) -> Expected<Any> {
      trace.emplace_back("pair");
      return Any(GetRef<TPair>(pair));
    };
    auto map_var = [&](const TVarObj* var) -> Expected<Any> {
      trace.emplace_back(var->name);
      return Any(GetRef<TVar>(var));
    };

    if (order == WalkOrder::kPreOrder) {
      StructuralMap<WalkOrder::kPreOrder>(root, map_pair, map_var);
    } else {
      StructuralMap<WalkOrder::kPostOrder>(root, map_pair, map_var);
    }
    return trace;
  };

  EXPECT_EQ(run(WalkOrder::kPreOrder), (std::vector<std::string>{"pair", "lhs", "rhs"}));
  EXPECT_EQ(run(WalkOrder::kPostOrder), (std::vector<std::string>{"lhs", "rhs", "pair"}));
}

TEST(StructuralMap, PropagatesDefRegionKindToCallbacks) {
  TDefHolder root(TVarWithDep("recursive"), TVarWithDep("non-recursive"));
  std::vector<TVMFFIDefRegionKind> kinds;

  Any result = StructuralMap<WalkOrder::kPostOrder>(
      root, [&](const TVarWithDepObj* var, TVMFFIDefRegionKind kind) -> Expected<Any> {
        kinds.push_back(kind);
        return Any(GetRef<TVarWithDep>(var));
      });

  EXPECT_TRUE(result.same_as(root));
  ASSERT_EQ(kinds.size(), 2U);
  EXPECT_EQ(kinds[0], kTVMFFIDefRegionKindRecursive);
  EXPECT_EQ(kinds[1], kTVMFFIDefRegionKindNonRecursive);
}

TEST(StructuralMap, PropagatesCallbackErrorWithContext) {
  TPair root(TVar("lhs"), TVar("rhs"));
  const Object* root_addr = root.get();
  const Object* lhs_addr = root->lhs.get();

  Expected<Any> result =
      StructuralMapExpected<WalkOrder::kPostOrder>(root, [](const TVarObj*) -> Expected<Any> {
        return Unexpected(Error("ValueError", "structural map callback failed", ""));
      });

  ASSERT_TRUE(result.is_err());
  Error error = result.error();
  EXPECT_EQ(error.kind(), "ValueError");
  EXPECT_EQ(error.message(), "structural map callback failed");
  Optional<VisitErrorContext> context = VisitErrorContext::TryGetFromError(error);
  ASSERT_TRUE(context.has_value());
  const List<ObjectRef>& chain = context.value()->reverse_visit_pattern;
  ASSERT_GE(chain.size(), 2U);
  EXPECT_EQ(chain[0].get(), lhs_addr);
  EXPECT_EQ(chain[chain.size() - 1].get(), root_addr);
}

TEST(StructuralMap, MapsSequenceContainers) {
  Array<Any> source{Any(TVar("array-var")), Any(List<Any>{Any(TVar("list-var"))})};
  const Object* source_addr = source.get();
  const ListObj* list_addr = source[1].as<ListObj>();
  int callback_count = 0;

  Array<Any> mapped =
      StructuralMap<WalkOrder::kPostOrder>(source, [&](const TVarObj* var) -> Expected<Any> {
        ++callback_count;
        return Any(TVar(var->name + "-mapped"));
      }).cast<Array<Any>>();

  ASSERT_EQ(mapped.size(), 2U);
  EXPECT_EQ(mapped.get(), source_addr);
  EXPECT_EQ(mapped[0].cast<TVar>()->name, "array-var-mapped");
  List<Any> mapped_list = mapped[1].cast<List<Any>>();
  ASSERT_EQ(mapped_list.size(), 1U);
  EXPECT_EQ(mapped_list.get(), list_addr);
  EXPECT_EQ(mapped_list[0].cast<TVar>()->name, "list-var-mapped");
  EXPECT_EQ(callback_count, 2);
}

TEST(StructuralMap, MapsMapAccordingToOwnershipAndKeyChanges) {
  auto map_var = [](const TVarObj* var) -> Expected<Any> {
    return Any(TVar(var->name + "-mapped"));
  };

  // A unique map reuses its object when only a value changes.
  {
    Map<Any, Any> source{{Any(int64_t{1}), Any(TVar("value"))}};
    const Object* source_addr = source.get();

    Map<Any, Any> mapped =
        StructuralMap<WalkOrder::kPostOrder>(source, map_var).cast<Map<Any, Any>>();

    EXPECT_EQ(mapped.get(), source_addr);
    EXPECT_EQ(source.at(Any(int64_t{1})).cast<TVar>()->name, "value-mapped");
  }

  // A changed key requires a rebuilt map even when the source is unique.
  {
    Map<Any, Any> source{{Any(int64_t{1}), Any(TVar("value"))}};
    const Object* source_addr = source.get();

    Map<Any, Any> mapped =
        StructuralMap<WalkOrder::kPostOrder>(source, [](int64_t key) -> Expected<Any> {
          return Any(key + 10);
        }).cast<Map<Any, Any>>();

    EXPECT_NE(mapped.get(), source_addr);
    EXPECT_EQ(source.at(Any(int64_t{1})).cast<TVar>()->name, "value");
    EXPECT_EQ(mapped.at(Any(int64_t{11})).cast<TVar>()->name, "value");
  }

  // A shared map takes the non-in-place path and returns a mapped container unconditionally.
  {
    Map<Any, Any> source{{Any(int64_t{1}), Any(TVar("value"))}};
    Map<Any, Any> source_alias = source;  // NOLINT(performance-unnecessary-copy-initialization)
    const Object* source_addr = source.get();

    Map<Any, Any> mapped =
        StructuralMap<WalkOrder::kPostOrder>(source, [](const TVarObj* var) -> Expected<Any> {
          return Any(GetRef<TVar>(var));
        }).cast<Map<Any, Any>>();

    EXPECT_NE(mapped.get(), source_addr);
    EXPECT_EQ(source.get(), source_alias.get());
    EXPECT_TRUE(mapped.at(Any(int64_t{1})).same_as(source.at(Any(int64_t{1}))));
  }
}

TEST(StructuralMap, MapsDictKeysAndValues) {
  Dict<Any, Any> source{{Any(int64_t{3}), Any(int64_t{4})}};
  const Object* source_addr = source.get();

  Dict<Any, Any> mapped =
      StructuralMap<WalkOrder::kPostOrder>(source, [](int64_t value) -> Expected<Any> {
        return Any(value + 10);
      }).cast<Dict<Any, Any>>();

  EXPECT_NE(mapped.get(), source_addr);
  EXPECT_EQ(source.at(Any(int64_t{3})).cast<int64_t>(), 4);
  EXPECT_EQ(mapped.at(Any(int64_t{13})).cast<int64_t>(), 14);
}

}  // namespace
