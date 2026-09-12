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
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/string.h>

#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "../testing_object.h"

namespace {

using namespace tvm::ffi;
using namespace tvm::ffi::testing;

using AnyArray = Array<Any>;
using StringMap = Map<String, Any>;

static_assert(std::is_same_v<decltype(std::declval<StructuralMutatorObj&>().DefaultMutateExpected(
                                 std::declval<AnyView>())),
                             Expected<UnchangedOr<Any>>>);
static_assert(
    std::is_same_v<decltype(std::declval<StructuralMutatorObj&>().DefaultMaybeInplaceMutateExpected(
                       std::declval<AnyView>())),
                   Expected<UnchangedOr<Any>>>);

// ---------------------------------------------------------------------------
// Unchanged result protocol.
// ---------------------------------------------------------------------------

Expected<UnchangedOr<String>> ReturnTypedUnchangedExpected() noexcept { return Unchanged(); }

TEST(UnchangedOr, ConversionsAndAssignmentMacro) {
  static_assert(!std::is_convertible_v<UnchangedOr<Any>, UnchangedOr<int>>);
  static_assert(type_subsumes_v<Expected<UnchangedOr<TNumber>>, Expected<UnchangedOr<TInt>>>);
  static_assert(!type_subsumes_v<Expected<UnchangedOr<TInt>>, Expected<UnchangedOr<TNumber>>>);
  static_assert(type_subsumes_v<Expected<Any>, Expected<void>>);
  static_assert(!type_subsumes_v<Expected<void>, Expected<Any>>);

  TInt original(42);
  UnchangedOr<TInt> source = original;
  UnchangedOr<Any> copied = std::as_const(source);
  UnchangedOr<TNumber> moved = std::move(source);
  EXPECT_EQ(original.use_count(), 3);
  EXPECT_TRUE(std::move(copied).ValueUnchecked().same_as(original));
  EXPECT_TRUE(std::move(moved).ValueUnchecked().same_as(original));
  EXPECT_EQ(original.use_count(), 1);

  UnchangedOr<double> numeric = UnchangedOr<int>(42);
  EXPECT_EQ(AnyView(numeric).type_index(), TypeIndex::kTVMFFIFloat);
  EXPECT_DOUBLE_EQ(std::move(numeric).ValueUnchecked(), 42.0);
  UnchangedOr<double> unchanged = UnchangedOr<int>(Unchanged());
  EXPECT_TRUE(unchanged.IsUnchanged());

  // One consumer exercises exact, widening, erased and narrowing source types.
  auto consume = [](auto input, const auto& original) -> Expected<bool> {
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(UnchangedOr<std::decay_t<decltype(original)>>, value,
                                      std::move(input));
    return value.UnchangedOrSameAs(original);
  };
  Expected<UnchangedOr<TInt>> typed = UnchangedOr<TInt>(original);
  EXPECT_TRUE(consume(typed, original).value());
  EXPECT_TRUE(consume(typed, TNumber(original)).value());
  EXPECT_TRUE(consume(Expected<UnchangedOr<Any>>(typed), original).value());
  EXPECT_EQ(consume(Expected<UnchangedOr<Any>>(Any(42)), original).error().kind(), "TypeError");
  EXPECT_EQ(consume(Expected<UnchangedOr<TNumber>>(UnchangedOr<TNumber>(TFloat(1.0))), original)
                .error()
                .kind(),
            "TypeError");
  Error error("ValueError", "child failure", "");
  Expected<UnchangedOr<Any>> failed = Expected<UnchangedOr<TInt>>(error);
  EXPECT_TRUE(consume(std::move(failed), original).error().same_as(error));
}

TEST(UnchangedOr, ErrorRoundTrip) {
  static_assert(std::is_copy_constructible_v<UnchangedOr<String>>);

  UnchangedOr<String> original = String("unchanged-or special-member value");
  UnchangedOr<String> copied_value(original);
  UnchangedOr<String> copy_assigned = Unchanged();
  copy_assigned = original;
  EXPECT_EQ(std::move(copied_value).ValueUnchecked(), "unchanged-or special-member value");
  EXPECT_EQ(std::move(copy_assigned).ValueUnchecked(), "unchanged-or special-member value");

  UnchangedOr<String> moved_value(std::move(original));
  UnchangedOr<String> move_source = String("unchanged-or move-assignment value");
  UnchangedOr<String> move_assigned = Unchanged();
  move_assigned = std::move(move_source);
  EXPECT_EQ(std::move(moved_value).ValueUnchecked(), "unchanged-or special-member value");
  EXPECT_EQ(std::move(move_assigned).ValueUnchecked(), "unchanged-or move-assignment value");

  Expected<UnchangedOr<Any>> failure = Error("ValueError", "expected failure", "");
  const Any copied_storage(failure);
  Expected<UnchangedOr<Any>> copied =
      details::AnyUnsafe::CopyFromAnyViewAfterCheck<Expected<UnchangedOr<Any>>>(copied_storage);

  ASSERT_TRUE(copied.is_err());
  EXPECT_EQ(copied.error().kind(), "ValueError");
  EXPECT_EQ(copied.error().message(), "expected failure");

  Any moved_storage(failure);
  Expected<UnchangedOr<Any>> moved =
      details::AnyUnsafe::MoveFromAnyAfterCheck<Expected<UnchangedOr<Any>>>(
          std::move(moved_storage));

  ASSERT_TRUE(moved.is_err());
  EXPECT_EQ(moved.error().kind(), "ValueError");
  EXPECT_EQ(moved.error().message(), "expected failure");

  Expected<Any> source_error = Error("TypeError", "converted failure", "");
  Expected<UnchangedOr<Any>> converted_error = std::move(source_error);
  ASSERT_TRUE(converted_error.is_err());
  EXPECT_EQ(converted_error.error().kind(), "TypeError");
  EXPECT_EQ(converted_error.error().message(), "converted failure");

  Expected<Any> unexpected_error = Unexpected(Error("IndexError", "unexpected failure", ""));
  ASSERT_TRUE(unexpected_error.is_err());
  EXPECT_EQ(unexpected_error.error().kind(), "IndexError");
  EXPECT_EQ(unexpected_error.error().message(), "unexpected failure");

  Expected<UnchangedOr<Any>> unexpected_result =
      Unexpected(Error("RuntimeError", "unchanged-or unexpected failure", ""));
  ASSERT_TRUE(unexpected_result.is_err());
  EXPECT_EQ(unexpected_result.error().kind(), "RuntimeError");
  EXPECT_EQ(unexpected_result.error().message(), "unchanged-or unexpected failure");
}

TEST(StructuralMutate, UnchangedProtocolResolvesAtThrowingEntryPoints) {
  Expected<Any> raw_tag = []() -> Expected<Any> { return Unchanged(); }();
  ASSERT_TRUE(raw_tag.is_ok());
  EXPECT_EQ(details::ExpectedUnsafe::GetData(raw_tag).type_index(), TypeIndex::kTVMFFIUnchanged);
  EXPECT_TRUE(ReturnTypedUnchangedExpected().value().IsUnchanged());
  auto never_matches = [](int64_t, StructuralMutatorObj*) -> Expected<Any> { return Any(); };
  using Mutator = StructuralMutateEngine<StructuralMapEngineBase, decltype(never_matches)>;
  StructuralMutator mutator(make_object<Mutator>(std::move(never_matches)));
  String value("value longer than small-string storage");

  auto untyped = mutator->MutateExpected(AnyView(value));
  ASSERT_TRUE(untyped.is_ok());
  EXPECT_TRUE(std::move(untyped).value().IsUnchanged());
  auto untyped_throwing = mutator->Mutate(AnyView(value));
  EXPECT_TRUE(untyped_throwing.IsUnchanged());
  EXPECT_TRUE(std::move(untyped_throwing).ValueOrUnchanged(AnyView(value)).same_as(value));

  auto typed = mutator->MutateExpected<String>(value);
  ASSERT_TRUE(typed.is_ok());
  EXPECT_TRUE(std::move(typed).value().IsUnchanged());
  auto typed_throwing = mutator->Mutate<String>(value);
  EXPECT_TRUE(typed_throwing.IsUnchanged());
  String moved_value = value;
  String moved_value_alias = moved_value;
  EXPECT_EQ(std::move(typed_throwing).ValueOrUnchanged(std::move(moved_value)), moved_value_alias);
  EXPECT_TRUE(mutator->MaybeInplaceMutate(AnyView(value)).IsUnchanged());
  EXPECT_TRUE(mutator->MaybeInplaceMutate<String>(value).IsUnchanged());
  auto unique = mutator->MaybeInplaceMutateIfUniqueExpected(AnyView(value));
  ASSERT_TRUE(unique.is_ok());
  EXPECT_TRUE(std::move(unique).value().IsUnchanged());
  auto typed_unique = mutator->MaybeInplaceMutateIfUniqueExpected<String>(value);
  ASSERT_TRUE(typed_unique.is_ok());
  EXPECT_TRUE(std::move(typed_unique).value().IsUnchanged());
  EXPECT_TRUE(StructuralMutateExpected(Any(value), never_matches).value().same_as(value));
  EXPECT_TRUE(StructuralMapExpected<WalkOrder::kPostOrder>(
                  Any(value), [](int64_t item) -> Expected<Any> { return Any(item + 1); })
                  .value()
                  .same_as(value));

  auto replace_int_with_string = [](int64_t value, StructuralMutatorObj*) -> Expected<Any> {
    if (value == -1) return Unexpected(Error("ValueError", "direct-forward failure", ""));
    return String("wrong replacement uses heap storage");
  };
  using WrongTypeMutator =
      StructuralMutateEngine<StructuralMapEngineBase, decltype(replace_int_with_string)>;
  StructuralMutator wrong_type_mutator(
      make_object<WrongTypeMutator>(std::move(replace_int_with_string)));

  auto any_error = wrong_type_mutator->MutateExpected(int64_t{-1});
  ASSERT_TRUE(any_error.is_err());
  EXPECT_EQ(any_error.error().message(), "direct-forward failure");
  auto any_inplace_error = wrong_type_mutator->MaybeInplaceMutateExpected(int64_t{-1});
  ASSERT_TRUE(any_inplace_error.is_err());
  EXPECT_EQ(any_inplace_error.error().message(), "direct-forward failure");
  auto typed_error = wrong_type_mutator->MutateExpected<int64_t>(int64_t{-1});
  ASSERT_TRUE(typed_error.is_err());
  EXPECT_EQ(typed_error.error().message(), "direct-forward failure");
  auto typed_inplace_error = wrong_type_mutator->MaybeInplaceMutateExpected<int64_t>(int64_t{-1});
  ASSERT_TRUE(typed_inplace_error.is_err());
  EXPECT_EQ(typed_inplace_error.error().message(), "direct-forward failure");

  Expected<UnchangedOr<int64_t>> result = wrong_type_mutator->MutateExpected<int64_t>(int64_t{1});
  ASSERT_TRUE(result.is_err());
  EXPECT_EQ(result.error().kind(), "TypeError");
  Expected<UnchangedOr<int64_t>> inplace_result =
      wrong_type_mutator->MaybeInplaceMutateExpected<int64_t>(int64_t{1});
  ASSERT_TRUE(inplace_result.is_err());
  EXPECT_EQ(inplace_result.error().kind(), "TypeError");
  Expected<UnchangedOr<int64_t>> unique_result =
      wrong_type_mutator->MaybeInplaceMutateIfUniqueExpected<int64_t>(int64_t{1});
  ASSERT_TRUE(unique_result.is_err());
  EXPECT_EQ(unique_result.error().kind(), "TypeError");

  for (WalkOrder order : {WalkOrder::kPreOrder, WalkOrder::kPostOrder}) {
    TVar root("n");
    TVar unchanged = (order == WalkOrder::kPreOrder
                          ? StructuralMap<WalkOrder::kPreOrder>(
                                root, [](const TVar&) -> Expected<Any> { return Unchanged(); })
                          : StructuralMap<WalkOrder::kPostOrder>(
                                root, [](const TVar&) -> Expected<Any> { return Unchanged(); }))
                         .cast<TVar>();
    EXPECT_TRUE(unchanged.same_as(root));
  }

  Function dynamic_unchanged = Function::FromTyped([](int64_t) -> Any { return Unchanged(); });
  Array<Tuple<int32_t, Function>> callbacks{
      Tuple<int32_t, Function>(TypeIndex::kTVMFFIInt, dynamic_unchanged)};
  Function structural_map = Function::GetGlobalRequired("ffi.StructuralMap");
  for (WalkOrder order : {WalkOrder::kPreOrder, WalkOrder::kPostOrder}) {
    Any unchanged = structural_map(int64_t{1}, callbacks, Array<Tuple<int32_t, Function>>(),
                                   static_cast<int32_t>(order));
    EXPECT_EQ(unchanged.cast<int64_t>(), 1);
  }
}

class TNestedMapHookObj : public Object {
 public:
  AnyArray field;

  explicit TNestedMapHookObj(AnyArray field) : field(std::move(field)) {}

  static TVMFFIAny StructuralMutate(StructuralMutatorObj* mutator, AnyView value) noexcept {
    const auto* self = value.cast<const TNestedMapHookObj*>();
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(UnchangedOr<Any>, mapped,
                                      mutator->MutateExpected(self->field));
    if (mapped.UnchangedOrSameAs(Any(self->field))) {
      return Unchanged().CopyToTVMFFIAny();
    }
    Any mapped_value = std::move(mapped).ValueOrUnchanged(Any(self->field));
    AnyArray mapped_field = mapped_value.cast<AnyArray>();
    return details::AnyUnsafe::MoveAnyToTVMFFIAny(
        Any(make_object<TNestedMapHookObj>(std::move(mapped_field))));
  }

  static TVMFFIAny MaybeInplaceMutate(StructuralMutatorObj*, AnyView value) noexcept {
    auto* self = value.cast<TNestedMapHookObj*>();
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
        Any, mapped,
        StructuralMapExpected<WalkOrder::kPostOrder>(
            Any(std::move(self->field)),
            [](int64_t item) -> Expected<Any> { return Any(item + 1); }));
    self->field = mapped.cast<AnyArray>();
    return details::AnyUnsafe::MoveAnyToTVMFFIAny(Any(value));
  }

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TNestedMapHookObj>().def_rw("field", &TNestedMapHookObj::field);
    refl::EnsureTypeAttrColumn(refl::type_attr::kStructuralMutate);
    refl::EnsureTypeAttrColumn(refl::type_attr::kStructuralMaybeInplaceMutate);
    refl::TypeAttrDef<TNestedMapHookObj>()
        .attr(refl::type_attr::kStructuralMutate,
              reinterpret_cast<void*>(static_cast<FStructuralMutate>(&StructuralMutate)))
        .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
              reinterpret_cast<void*>(static_cast<FStructuralMutate>(&MaybeInplaceMutate)));
  }

  static constexpr bool _type_mutable = true;
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("test.NestedMapHook", TNestedMapHookObj, Object);
};

class TNestedMapHook : public ObjectRef {
 public:
  explicit TNestedMapHook(AnyArray field) {
    data_ = make_object<TNestedMapHookObj>(std::move(field));
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TNestedMapHook, ObjectRef, TNestedMapHookObj);
};

TVM_FFI_STATIC_INIT_BLOCK() {
  TMutatePairObj::RegisterReflection();
  TNestedMapHookObj::RegisterReflection();
}

Expected<Any> Increment(int64_t value) { return Any(value + 1); }

struct MutateCount {
  int value = 0;
  int mutate_expected = 0;
  int maybe_inplace_expected = 0;
};

class StructuralMapWithMutateCount : public StructuralMapEngineBase {
 public:
  using StateTupleType = std::tuple<const MutateCount&, const int&>;

  explicit StructuralMapWithMutateCount(const StructuralMutatorVTable* vtable)
      : StructuralMapEngineBase(vtable) {}

  const MutateCount& count() const { return count_; }

  Expected<UnchangedOr<Any>> DefaultMutateExpected(AnyView value) noexcept {
    ++count_.value;
    ++count_.mutate_expected;
    return StructuralMapEngineBase::DefaultMutateExpected(value);
  }

  Expected<UnchangedOr<Any>> DefaultMaybeInplaceMutateExpected(AnyView value) noexcept {
    ++count_.value;
    ++count_.maybe_inplace_expected;
    return StructuralMapEngineBase::DefaultMaybeInplaceMutateExpected(value);
  }

 protected:
  StateTupleType StateTuple() const noexcept { return StateTupleType(count_, marker_); }

 private:
  MutateCount count_;
  int marker_ = 17;
};

class StructuralMutateLayer : public StructuralMapEngineBase {
 public:
  using MutatorObjType = StructuralMutateLayer;
  using StructuralMapEngineBase::HasVarRemap;
  using StructuralMapEngineBase::VarRemapGetImpl;
  using StructuralMapEngineBase::VarRemapSetImpl;

  explicit StructuralMutateLayer(const StructuralMutatorVTable* vtable)
      : StructuralMapEngineBase(vtable) {}

  int callback_tag() const { return 23; }
};

TEST(StructuralMap, ParentLayerOwnsBothDescentsAndProvidesState) {
  std::vector<int> callback_counts;
  auto identity = [&](const AnyArray& value, const MutateCount& live_count, const int& live_marker,
                      TVMFFIDefRegionKind kind) -> Expected<Any> {
    EXPECT_EQ(live_marker, 17);
    EXPECT_EQ(kind, kTVMFFIDefRegionKindNone);
    callback_counts.push_back(live_count.value);
    return Any(value);
  };
  int var_callback_count = 0;
  auto map_var = [&](const TVarObj* value, const MutateCount& live_count,
                     const int& live_marker) -> Expected<Any> {
    EXPECT_EQ(live_marker, 17);
    EXPECT_GT(live_count.value, 0);
    ++var_callback_count;
    return Any(TVar(value->name + "-mapped"));
  };
  using Mutator = StructuralMapEngine<StructuralMapWithMutateCount, WalkOrder::kPostOrder,
                                      decltype(identity), decltype(map_var)>;
  auto engine = make_object<Mutator>(std::move(identity), std::move(map_var));
  StructuralMutator mutator(engine);

  ASSERT_FALSE(mutator->MutateExpected(String("unmatched")).is_err());
  AnyArray rebuild_root{int64_t{1}};
  ASSERT_FALSE(mutator->MutateExpected(rebuild_root).is_err());

  ASSERT_FALSE(mutator->MaybeInplaceMutateExpected(String("unmatched")).is_err());
  AnyArray inplace_root{int64_t{1}};
  ASSERT_FALSE(mutator->MaybeInplaceMutateExpected(inplace_root).is_err());

  EXPECT_GT(engine->count().mutate_expected, 0);
  EXPECT_GT(engine->count().maybe_inplace_expected, 0);
  EXPECT_EQ(callback_counts.size(), 2U);
  EXPECT_GT(callback_counts[0], 0);
  EXPECT_GT(callback_counts[1], callback_counts[0]);

  TVar var("n");
  AnyArray repeated{var, var};
  AnyArray mapped =
      std::move(mutator->Mutate<AnyArray>(repeated)).ValueOrUnchanged(std::move(repeated));
  EXPECT_EQ(var_callback_count, 2);
  EXPECT_FALSE(mapped[0].cast<TVar>().same_as(mapped[1].cast<TVar>()));
}

TEST(StructuralMutate, CallbackOwnsMutationAndErrorsStayExpected) {
  std::vector<int64_t> trace;
  auto mutate_array = [&](const AnyArray& value, StructuralMutateLayer* mutator) -> Expected<Any> {
    EXPECT_EQ(mutator->callback_tag(), 23);
    auto first_result = mutator->MutateExpected(value[0]);
    if (TVM_FFI_PREDICT_FALSE(first_result.is_err())) {
      return Unexpected(std::move(first_result).error());
    }
    Any first = std::move(first_result).value().ValueOrUnchanged(AnyView(value[0]));
    return Any(AnyArray{std::move(first), int64_t{10}});
  };
  auto mutate_int = [&](int64_t value, StructuralMutateLayer*) -> Expected<Any> {
    trace.push_back(value);
    return Any(value + 1);
  };
  using Mutator =
      StructuralMutateEngine<StructuralMutateLayer, decltype(mutate_array), decltype(mutate_int)>;
  StructuralMutator mutator(make_object<Mutator>(std::move(mutate_array), std::move(mutate_int)));

  AnyArray root{int64_t{1}, int64_t{2}};
  AnyArray mapped = std::move(mutator->Mutate<AnyArray>(root)).ValueOrUnchanged(std::move(root));
  ASSERT_EQ(mapped.size(), 2U);
  EXPECT_EQ(mapped[0].cast<int64_t>(), 2);
  EXPECT_EQ(mapped[1].cast<int64_t>(), 10);
  EXPECT_EQ(trace, std::vector<int64_t>{1});

  AnyArray default_mapped =
      StructuralMutate(
          AnyArray{int64_t{3}, int64_t{4}},
          [](int64_t value, StructuralMutatorObj*) -> Expected<Any> { return Any(value + 1); })
          .cast<AnyArray>();
  EXPECT_EQ(default_mapped[0].cast<int64_t>(), 4);
  EXPECT_EQ(default_mapped[1].cast<int64_t>(), 5);

  Expected<Any> returned_error =
      StructuralMutateExpected(int64_t{1}, [](int64_t, StructuralMutatorObj*) -> Expected<Any> {
        return Unexpected(Error("ValueError", "returned mutate error", ""));
      });
  ASSERT_TRUE(returned_error.is_err());
  EXPECT_EQ(returned_error.error().message(), "returned mutate error");

  Expected<Any> thrown_error =
      StructuralMutateExpected(int64_t{1}, [](int64_t, StructuralMutatorObj*) -> Expected<Any> {
        TVM_FFI_THROW(ValueError) << "thrown mutate error";
        return Any(nullptr);
      });
  ASSERT_TRUE(thrown_error.is_err());
  EXPECT_EQ(thrown_error.error().message(), "thrown mutate error");
}

TEST(StructuralMutate, CallbackControlsRecursion) {
  TPair root(TPair(TInt(1), TInt(2)), TPair(TInt(3), TInt(4)));
  ObjectRef original_rhs = root->rhs;

  TPair mapped =
      StructuralMutate(
          root,
          [](const TPair& pair, StructuralMutatorObj* mutator) -> Expected<UnchangedOr<ObjectRef>> {
            TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(UnchangedOr<ObjectRef>, lhs_result,
                                              mutator->MutateExpected<ObjectRef>(pair->lhs));
            ObjectRef original_lhs = pair->lhs;
            ObjectRef lhs = std::move(lhs_result).ValueOrUnchanged(std::move(original_lhs));
            return UnchangedOr<ObjectRef>(TPair(std::move(lhs), pair->rhs));
          },
          [](const TInt& value, StructuralMutatorObj*) -> Expected<Any> {
            return Any(TInt(value->value + 100));
          })
          .cast<TPair>();

  TPair mapped_lhs = mapped->lhs.as_or_throw<TPair>();
  TPair mapped_rhs = mapped->rhs.as_or_throw<TPair>();
  EXPECT_EQ(mapped_lhs->lhs.as_or_throw<TInt>()->value, 101);
  EXPECT_EQ(mapped_lhs->rhs.as_or_throw<TInt>()->value, 2);
  EXPECT_EQ(mapped_rhs->lhs.as_or_throw<TInt>()->value, 3);
  EXPECT_EQ(mapped_rhs->rhs.as_or_throw<TInt>()->value, 4);
  EXPECT_TRUE(mapped->rhs.same_as(original_rhs));
}

TEST(StructuralMutate, SingleCallbackCanDelegateToDefault) {
  auto mutate = [](AnyView value, StructuralMutatorObj* mutator) -> Expected<UnchangedOr<Any>> {
    if (auto integer = value.as<int64_t>()) {
      return Any(*integer + 1);
    }
    return mutator->DefaultMutateExpected(value);
  };
  AnyArray root{int64_t{1}, AnyArray{int64_t{2}}};
  AnyArray result = StructuralMutate(root, mutate).cast<AnyArray>();
  EXPECT_EQ(result[0].cast<int64_t>(), 2);
  EXPECT_EQ(result[1].cast<AnyArray>()[0].cast<int64_t>(), 3);
  EXPECT_EQ(root[0].cast<int64_t>(), 1);
}

TEST(StructuralMutate, PreservesUniqueContainerIdentity) {
  AnyArray inner{int64_t{1}};
  const Object* inner_address = inner.get();
  AnyArray root{Any(std::move(inner))};
  const Object* root_address = root.get();

  AnyArray mapped =
      StructuralMutate(std::move(root), [](int64_t value, StructuralMutatorObj*) -> Expected<Any> {
        return Any(value + 1);
      }).cast<AnyArray>();

  AnyArray mapped_inner = mapped[0].cast<AnyArray>();
  EXPECT_EQ(mapped.get(), root_address);
  EXPECT_EQ(mapped_inner.get(), inner_address);
  EXPECT_EQ(mapped_inner[0].cast<int64_t>(), 2);
}

TEST(StructuralMutate, RootByValueProtectsSharedParentSubvalue) {
  AnyArray child{int64_t{1}};
  AnyArray outer{Any(std::move(child))};
  const Object* child_address = outer[0].cast<AnyArray>().get();

  AnyArray mapped =
      StructuralMutate(outer[0], [](int64_t value, StructuralMutatorObj*) -> Expected<Any> {
        return Any(value + 1);
      }).cast<AnyArray>();

  EXPECT_NE(mapped.get(), child_address);
  EXPECT_EQ(outer[0].cast<AnyArray>()[0].cast<int64_t>(), 1);
  EXPECT_EQ(mapped[0].cast<int64_t>(), 2);
}

TEST(StructuralMutate, CallbackArityControlsInplaceMutation) {
  AnyArray inplace_root{int64_t{1}};
  AnyArray copy_on_write_root{int64_t{1}};
  const Object* inplace_root_address = inplace_root.get();
  const Object* copy_on_write_root_address = copy_on_write_root.get();
  std::vector<bool> allow_inplace_trace;

  AnyArray inplace_mapped =
      StructuralMutate(
          std::move(inplace_root),
          [&](const AnyArray& value, StructuralMutatorObj* mutator,
              bool allow_inplace) -> Expected<Any> {
            allow_inplace_trace.push_back(allow_inplace);
            return allow_inplace ? mutator->DefaultMaybeInplaceMutateExpected(value)
                                 : mutator->DefaultMutateExpected(value);
          },
          [&](int64_t value, StructuralMutatorObj*, bool allow_inplace) -> Expected<Any> {
            allow_inplace_trace.push_back(allow_inplace);
            return Any(value + 1);
          })
          .cast<AnyArray>();

  AnyArray copy_on_write_mapped =
      StructuralMutate(
          std::move(copy_on_write_root),
          [](const AnyArray& value, StructuralMutatorObj* mutator) -> Expected<Any> {
            return mutator->DefaultMutateExpected(value);
          },
          [](int64_t value, StructuralMutatorObj*) -> Expected<Any> { return Any(value + 1); })
          .cast<AnyArray>();

  EXPECT_EQ(inplace_mapped.get(), inplace_root_address);
  EXPECT_NE(copy_on_write_mapped.get(), copy_on_write_root_address);
  EXPECT_EQ(inplace_mapped[0].cast<int64_t>(), 2);
  EXPECT_EQ(copy_on_write_mapped[0].cast<int64_t>(), 2);
  EXPECT_EQ(allow_inplace_trace, (std::vector<bool>{true, false}));
}

TEST(StructuralMutate, MatchedVarOwnsRemapConsistency) {
  TVar key("key"), replacement("replacement");
  {
    auto engine = make_object<StructuralMutateLayer>(nullptr);
    EXPECT_FALSE(engine->HasVarRemap());
    EXPECT_TRUE(engine->VarRemapGetImpl(nullptr).is_err());
    EXPECT_TRUE(engine->VarRemapGetImpl(1).is_err());
    EXPECT_EQ(engine->VarRemapGetImpl(key).value(), nullptr);
    engine->VarRemapSetImpl(key, replacement).value();
    EXPECT_TRUE(engine->HasVarRemap());
    EXPECT_FALSE(key.unique());
    EXPECT_TRUE(engine->VarRemapGetImpl(key).value().cast<TVar>().same_as(replacement));
    EXPECT_EQ(engine->VarRemapGetImpl(TVar("missing")).value(), nullptr);
    engine->VarRemapSetImpl(key, Any(Unchanged())).value();
    EXPECT_EQ(engine->VarRemapGetImpl(key).value().type_index(), TypeIndex::kTVMFFIUnchanged);
    EXPECT_TRUE(engine->HasVarRemap());
    EXPECT_TRUE(replacement.unique());
  }
  EXPECT_TRUE(key.unique());

  TVarWithDep var("n", TVar("type"));
  TPair root(TDefHolder(TVarWithDep("pattern"), var), var);
  int type_callback_count = 0;

  TPair mapped = StructuralMap<WalkOrder::kPreOrder>(root, [&](const TVar& value) -> Expected<Any> {
                   if (!value.defined()) return Unchanged();
                   ++type_callback_count;
                   return Any(TVar(value->name + "-mapped"));
                 }).cast<TPair>();
  TDefHolder mapped_defs = mapped->lhs.as_or_throw<TDefHolder>();
  TVarWithDep mapped_def = mapped_defs->def_non_recursive;
  TVarWithDep mapped_use = mapped->rhs.as_or_throw<TVarWithDep>();

  EXPECT_EQ(type_callback_count, 1);
  EXPECT_FALSE(mapped_def.same_as(var));
  EXPECT_TRUE(mapped_def.same_as(mapped_use));
  ASSERT_TRUE(mapped_def->dep.has_value());
  EXPECT_EQ(mapped_def->dep.value().as_or_throw<TVar>()->name, "type-mapped");

  TVarWithDep unchanged("unchanged", TVar("unchanged-type"));
  TPair repeated_definition(TDefHolder(TVarWithDep("first-pattern"), unchanged),
                            TDefHolder(unchanged, TVarWithDep("last-simple")));
  int unchanged_type_callback_count = 0;

  StructuralMap<WalkOrder::kPostOrder>(repeated_definition,
                                       [&](const TVar& value) -> Expected<Any> {
                                         if (!value.defined()) return Unchanged();
                                         ++unchanged_type_callback_count;
                                         return Unchanged();
                                       });

  // The unchanged simple definition leaves no binding, so the later pattern definition descends
  // the same var once and records the unchanged marker.
  EXPECT_EQ(unchanged_type_callback_count, 2);
}

template <WalkOrder order>
void CheckNestedArrayMapOrder(const std::vector<std::string>& expected_trace) {
  AnyArray inner_array{int64_t{1}};
  const Object* inner_array_address = inner_array.get();
  StringMap map{{"value", Any(std::move(inner_array))}};
  const Object* map_address = map.get();
  AnyArray root{Any(std::move(map))};
  const Object* root_address = root.get();
  std::vector<std::string> trace;

  AnyArray mapped =
      StructuralMap<order>(
          std::move(root),
          [&](const AnyArray& array) -> Expected<Any> {
            trace.emplace_back(array.get() == root_address ? "outer-array" : "inner-array");
            return Any(array);
          },
          [&](const StringMap& value) -> Expected<Any> {
            trace.emplace_back("map");
            return Any(value);
          },
          [&](const String&) -> Expected<Any> {
            trace.emplace_back("map-key");
            return Any(String("renamed"));
          },
          [&](int64_t value) -> Expected<Any> {
            trace.emplace_back("int");
            return Any(value + 1);
          })
          .template cast<AnyArray>();

  StringMap mapped_map = mapped[0].cast<StringMap>();
  AnyArray mapped_inner_array = mapped_map["value"].cast<AnyArray>();

  EXPECT_EQ(trace, expected_trace);
  EXPECT_EQ(mapped.get(), root_address);
  EXPECT_EQ(mapped_map.get(), map_address);
  EXPECT_EQ(mapped_inner_array.get(), inner_array_address);
  EXPECT_EQ(mapped_inner_array[0].cast<int64_t>(), 2);
  EXPECT_EQ(mapped_map.count("value"), 1U);
  EXPECT_EQ(mapped_map.count("renamed"), 0U);
}

TEST(StructuralMap, MapsNestedArrayAndMapInConfiguredOrder) {
  CheckNestedArrayMapOrder<WalkOrder::kPreOrder>({"outer-array", "map", "inner-array", "int"});
  CheckNestedArrayMapOrder<WalkOrder::kPostOrder>({"int", "inner-array", "map", "outer-array"});
}

TEST(StructuralMap, RootByValueProtectsSharedParentSubvalue) {
  AnyArray child{int64_t{1}};
  AnyArray outer{Any(std::move(child))};
  const Object* child_address = outer[0].cast<AnyArray>().get();

  AnyArray mapped = StructuralMap<WalkOrder::kPostOrder>(outer[0], Increment).cast<AnyArray>();

  EXPECT_NE(mapped.get(), child_address);
  EXPECT_EQ(outer[0].cast<AnyArray>()[0].cast<int64_t>(), 1);
  EXPECT_EQ(mapped[0].cast<int64_t>(), 2);
}

TEST(StructuralMap, MaybeInplaceHookMovesNestedFieldIntoStructuralMap) {
  TNestedMapHook root(AnyArray{int64_t{1}});
  const Object* root_address = root.get();
  const Object* field_address = root->field.get();

  TNestedMapHook mapped =
      StructuralMap<WalkOrder::kPostOrder>(
          Any(std::move(root)), [](const String& value) -> Expected<Any> { return Any(value); })
          .cast<TNestedMapHook>();

  EXPECT_EQ(mapped.get(), root_address);
  EXPECT_EQ(mapped->field.get(), field_address);
  EXPECT_EQ(mapped->field[0].cast<int64_t>(), 2);
}

TEST(StructuralMap, RegisteredMutateHookUsesAssignOrReturn) {
  TVar lhs("lhs");
  TVar rhs("rhs");
  TMutatePair root(lhs, rhs);
  TMutatePairObj::StructuralMutateCallCount() = 0;

  TMutatePair mapped =
      StructuralMap<WalkOrder::kPostOrder>(root, [](const TVarObj* var) -> Expected<Any> {
        return Any(TVar(var->name + "-mapped"));
      }).cast<TMutatePair>();

  EXPECT_EQ(TMutatePairObj::StructuralMutateCallCount(), 1);
  Optional<TVar> mapped_lhs = mapped->lhs.as<TVar>();
  Optional<TVar> mapped_rhs = mapped->rhs.as<TVar>();
  ASSERT_TRUE(mapped_lhs.has_value());
  ASSERT_TRUE(mapped_rhs.has_value());
  EXPECT_EQ(mapped_lhs.value()->name, "lhs-mapped");
  EXPECT_EQ(mapped_rhs.value()->name, "rhs-mapped");

  Expected<Any> failed =
      StructuralMapExpected<WalkOrder::kPostOrder>(root, [](const TVarObj* var) -> Expected<Any> {
        if (var->name == "lhs") {
          return Unexpected(Error("ValueError", "registered hook child failed", ""));
        }
        return Any(TVar(var->name + "-mapped"));
      });
  ASSERT_TRUE(failed.is_err());
  Optional<VisitErrorContext> context = VisitErrorContext::TryGetFromError(failed.error());
  ASSERT_TRUE(context.has_value());
  const List<ObjectRef>& reverse_pattern = context.value()->reverse_visit_pattern;
  ASSERT_EQ(reverse_pattern.size(), 2U);
  EXPECT_TRUE(reverse_pattern[0].same_as(lhs));
  EXPECT_TRUE(reverse_pattern[1].same_as(root));

  TMutatePair nullable(ObjectRef(nullptr), rhs);
  TMutatePair nullable_mapped =
      StructuralMap<WalkOrder::kPostOrder>(nullable, [](const String& value) -> Expected<Any> {
        return Any(value);
      }).cast<TMutatePair>();
  EXPECT_FALSE(nullable_mapped->lhs.defined());
  EXPECT_TRUE(nullable_mapped->rhs.same_as(rhs));

  int nullable_var_callbacks = 0;
  nullable_mapped =
      StructuralMap<WalkOrder::kPostOrder>(nullable, [&](const TVar& value) -> Expected<Any> {
        ++nullable_var_callbacks;
        return value.defined() ? Any(value) : Any(ObjectRef(nullptr));
      }).cast<TMutatePair>();
  EXPECT_EQ(nullable_var_callbacks, 2);
  EXPECT_FALSE(nullable_mapped->lhs.defined());
  EXPECT_TRUE(nullable_mapped->rhs.same_as(rhs));

  Expected<Any> wrong_type = StructuralMapExpected<WalkOrder::kPostOrder>(
      root, [](const TVarObj*) -> Expected<Any> { return Any(int64_t{1}); });
  ASSERT_TRUE(wrong_type.is_err());
  EXPECT_EQ(wrong_type.error().kind(), "TypeError");
}

TEST(StructuralMap, PreservesSharedArrayAndMapInputs) {
  // A shared Array is copied when one of its elements changes.
  {
    AnyArray child{int64_t{1}};
    const Object* child_address = child.get();
    AnyArray root{Any(std::move(child))};
    AnyArray owner = root;  // NOLINT(performance-unnecessary-copy-initialization)
    const Object* root_address = root.get();

    AnyArray mapped = StructuralMap<WalkOrder::kPostOrder>(root, Increment).cast<AnyArray>();
    AnyArray original_child = root[0].cast<AnyArray>();
    AnyArray mapped_child = mapped[0].cast<AnyArray>();

    EXPECT_NE(mapped.get(), root_address);
    EXPECT_EQ(owner.get(), root_address);
    EXPECT_NE(mapped_child.get(), child_address);
    EXPECT_EQ(original_child[0].cast<int64_t>(), 1);
    EXPECT_EQ(mapped_child[0].cast<int64_t>(), 2);
  }

  // A shared Map and its changed value path are also copied.
  {
    AnyArray value{int64_t{1}};
    const Object* value_address = value.get();
    StringMap root{{"value", Any(std::move(value))}};
    StringMap owner = root;  // NOLINT(performance-unnecessary-copy-initialization)
    const Object* root_address = root.get();

    StringMap mapped = StructuralMap<WalkOrder::kPostOrder>(root, Increment).cast<StringMap>();
    AnyArray original_value = root["value"].cast<AnyArray>();
    AnyArray mapped_value = mapped["value"].cast<AnyArray>();

    EXPECT_NE(mapped.get(), root_address);
    EXPECT_EQ(owner.get(), root_address);
    EXPECT_NE(mapped_value.get(), value_address);
    EXPECT_EQ(original_value[0].cast<int64_t>(), 1);
    EXPECT_EQ(mapped_value[0].cast<int64_t>(), 2);
  }

  // Copy-on-write remains lazy: an unchanged shared Map is returned directly.
  {
    StringMap root{{"value", AnyArray{int64_t{1}}}};
    StringMap owner = root;  // NOLINT(performance-unnecessary-copy-initialization)

    StringMap mapped =
        StructuralMap<WalkOrder::kPostOrder>(root, [](int64_t value) -> Expected<Any> {
          return Any(value);
        }).cast<StringMap>();

    EXPECT_TRUE(mapped.same_as(root));
    EXPECT_TRUE(owner.same_as(root));
    EXPECT_TRUE(mapped["value"].same_as(root["value"]));
  }
}

TEST(StructuralMap, CopyOnWriteProcessesSuffixAfterFirstChange) {
  AnyArray array_root{String("prefix"), int64_t{1}, String("middle"), int64_t{2}};
  AnyArray array_owner = array_root;  // NOLINT(performance-unnecessary-copy-initialization)
  AnyArray mapped_array =
      StructuralMap<WalkOrder::kPostOrder>(array_root, Increment).cast<AnyArray>();

  EXPECT_TRUE(mapped_array[0].same_as(array_root[0]));
  EXPECT_EQ(mapped_array[1].cast<int64_t>(), 2);
  EXPECT_TRUE(mapped_array[2].same_as(array_root[2]));
  EXPECT_EQ(mapped_array[3].cast<int64_t>(), 3);
  EXPECT_TRUE(array_owner.same_as(array_root));

  StringMap map_root{{"prefix", String("unchanged")},
                     {"first", int64_t{1}},
                     {"middle", String("also-unchanged")},
                     {"second", int64_t{2}}};
  StringMap map_owner = map_root;  // NOLINT(performance-unnecessary-copy-initialization)
  StringMap mapped_map =
      StructuralMap<WalkOrder::kPostOrder>(map_root, Increment).cast<StringMap>();

  EXPECT_TRUE(mapped_map["prefix"].same_as(map_root["prefix"]));
  EXPECT_EQ(mapped_map["first"].cast<int64_t>(), 2);
  EXPECT_TRUE(mapped_map["middle"].same_as(map_root["middle"]));
  EXPECT_EQ(mapped_map["second"].cast<int64_t>(), 3);
  EXPECT_TRUE(map_owner.same_as(map_root));

  Expected<Any> suffix_error = StructuralMapExpected<WalkOrder::kPostOrder>(
      AnyArray{int64_t{1}, int64_t{2}}, [](int64_t value) -> Expected<Any> {
        if (value == 2) {
          return Unexpected(Error("ValueError", "suffix mutate failed", ""));
        }
        return Any(value + 1);
      });
  ASSERT_TRUE(suffix_error.is_err());
  EXPECT_EQ(suffix_error.error().message(), "suffix mutate failed");
}

TEST(StructuralMap, PreOrderRecursivelyMapsCallbackResult) {
  StringMap root{{"value", AnyArray{int64_t{1}}}};
  AnyArray replacement{int64_t{10}};

  StringMap mapped =
      StructuralMap<WalkOrder::kPreOrder>(
          root, [&](const AnyArray&) -> Expected<Any> { return Any(replacement); }, Increment)
          .cast<StringMap>();
  AnyArray mapped_value = mapped["value"].cast<AnyArray>();

  EXPECT_FALSE(mapped_value.same_as(replacement));
  EXPECT_EQ(replacement[0].cast<int64_t>(), 10);
  EXPECT_EQ(mapped_value[0].cast<int64_t>(), 11);

  // The changed callback result remains owned while its leaf descent reports unchanged.
  String replacement_leaf("replacement longer than small-string storage");
  String retained =
      StructuralMap<WalkOrder::kPreOrder>(TVar("n"), [&](const TVar&) -> Expected<Any> {
        return Any(replacement_leaf);
      }).cast<String>();
  EXPECT_EQ(retained, replacement_leaf);

  // An unchanged pre-order callback still descends the original node.
  AnyArray unchanged_root{int64_t{1}};
  AnyArray descended_original =
      StructuralMap<WalkOrder::kPreOrder>(
          unchanged_root, [](const AnyArray&) -> Expected<Any> { return Unchanged(); }, Increment)
          .cast<AnyArray>();
  EXPECT_EQ(descended_original[0].cast<int64_t>(), 2);
}

TEST(StructuralMap, AcceptsExpectedCallbackReturnTypes) {
  static_assert(std::is_convertible_v<Expected<TVar>, Expected<Any>>,
                "Expected<T> should implicitly convert to Expected<Any>");
  static_assert(std::is_convertible_v<TVar, Expected<Any>>,
                "Bare values convertible to Any should convert to Expected<Any>");
  TVar root("n");

  TVar bare = StructuralMap<WalkOrder::kPostOrder>(root, [](const TVar&) -> TVar {
                return TVar("bare");
              }).cast<TVar>();
  EXPECT_EQ(bare->name, "bare");

  // Any already converted directly to Expected<Any> before Expected<U> support was added.
  TVar any = StructuralMap<WalkOrder::kPostOrder>(root, [](const TVar&) -> Any {
               return Any(TVar("any"));
             }).cast<TVar>();
  EXPECT_EQ(any->name, "any");

  TVar expected_object =
      StructuralMap<WalkOrder::kPostOrder>(root, [](const TVar&) -> Expected<TVar> {
        return TVar("expected-object");
      }).cast<TVar>();
  EXPECT_EQ(expected_object->name, "expected-object");

  TVar expected_any = StructuralMap<WalkOrder::kPostOrder>(root, [](const TVar&) -> Expected<Any> {
                        return Any(TVar("expected-any"));
                      }).cast<TVar>();
  EXPECT_EQ(expected_any->name, "expected-any");

  auto check_error = [](auto callback, const char* expected_message) {
    Expected<Any> result =
        StructuralMapExpected<WalkOrder::kPostOrder>(TVar("n"), std::move(callback));
    ASSERT_TRUE(result.is_err());
    EXPECT_EQ(result.error().kind(), "ValueError");
    EXPECT_EQ(result.error().message(), expected_message);
  };
  check_error(
      [](const TVar&) -> Expected<TVar> {
        return Error("ValueError", "expected object error", "");
      },
      "expected object error");
  check_error(
      [](const TVar&) -> Expected<Any> { return Error("ValueError", "expected any error", ""); },
      "expected any error");
  check_error([](const TVar&) -> Any { return Any(Error("ValueError", "any error", "")); },
              "any error");
  check_error([](const TVar&) -> Error { return Error("ValueError", "direct error", ""); },
              "direct error");
  check_error(
      [](const TVar&) -> Unexpected<Error> {
        return Unexpected(Error("ValueError", "unexpected error", ""));
      },
      "unexpected error");
  check_error([](const TVar&) -> Expected<TVar> { throw Error("ValueError", "thrown error", ""); },
              "thrown error");
}

template <WalkOrder order>
void CheckContainerCallbackErrorsStayExpected() {
  auto check_error = [](AnyView root, auto callback, const char* expected_message) {
    Expected<Any> result = StructuralMapExpected<order>(root, std::move(callback));
    ASSERT_TRUE(result.is_err());
    EXPECT_EQ(result.error().kind(), "ValueError");
    EXPECT_EQ(result.error().message(), expected_message);
  };
  auto raise = [](const TVar&) -> Expected<Any> { throw Error("ValueError", "raised", ""); };
  auto returns_error = [](const TVar&) -> Expected<Any> {
    return Unexpected(Error("ValueError", "returned", ""));
  };

  AnyArray array_root{Any(TVar("n"))};
  check_error(array_root, raise, "raised");
  check_error(array_root, returns_error, "returned");

  List<Any> list_root{Any(TVar("n"))};
  check_error(list_root, raise, "raised");
  check_error(list_root, returns_error, "returned");

  StringMap map_root{{"value", TVar("n")}};
  check_error(map_root, raise, "raised");
  check_error(map_root, returns_error, "returned");

  Dict<Any, Any> dict_root{{String("value"), Any(TVar("n"))}};
  check_error(dict_root, raise, "raised");
  check_error(dict_root, returns_error, "returned");
}

TEST(StructuralMap, ContainerCallbackErrorsStayExpected) {
  CheckContainerCallbackErrorsStayExpected<WalkOrder::kPreOrder>();
  CheckContainerCallbackErrorsStayExpected<WalkOrder::kPostOrder>();
}

template <WalkOrder order>
void CheckRepeatedFreeVarCallbacks() {
  TVar var("n");
  StringMap use{{"use", var}};
  AnyArray root{var, Any(std::move(use))};
  int callback_count = 0;

  AnyArray mapped = StructuralMap<order>(root, [&](const TVarObj* value) -> Expected<Any> {
                      ++callback_count;
                      return Any(TVar(value->name + "-mapped"));
                    }).template cast<AnyArray>();
  TVar mapped_var = mapped[0].cast<TVar>();
  StringMap mapped_uses = mapped[1].cast<StringMap>();
  TVar mapped_use = mapped_uses["use"].cast<TVar>();

  EXPECT_EQ(callback_count, 2);
  EXPECT_FALSE(mapped_var.same_as(mapped_use));
  EXPECT_EQ(mapped_var->name, "n-mapped");
  EXPECT_EQ(var->name, "n");
}

TEST(StructuralMap, InvokesCallbackForEveryFreeVarOccurrence) {
  CheckRepeatedFreeVarCallbacks<WalkOrder::kPreOrder>();
  CheckRepeatedFreeVarCallbacks<WalkOrder::kPostOrder>();
}

AnyArray MakeStringAndBytesLeaves() {
  return AnyArray{int64_t{1}, String("1234567"), String("12345678"), Bytes("1234567", 7),
                  Bytes("12345678", 8)};
}

template <WalkOrder order>
void CheckStringAndBytesLeaves() {
  // An unmatched callback leaves inline and heap-backed values untouched.
  {
    AnyArray root = MakeStringAndBytesLeaves();
    EXPECT_EQ(root[1].type_index(), TypeIndex::kTVMFFISmallStr);
    EXPECT_EQ(root[2].type_index(), TypeIndex::kTVMFFIStr);
    EXPECT_EQ(root[3].type_index(), TypeIndex::kTVMFFISmallBytes);
    EXPECT_EQ(root[4].type_index(), TypeIndex::kTVMFFIBytes);

    AnyArray unmatched = StructuralMap<order>(root, [](int64_t value) -> Expected<Any> {
                           return Any(value);
                         }).template cast<AnyArray>();

    EXPECT_TRUE(unmatched.same_as(root));
  }

  // Identity callbacks return the original shared Array for both representations.
  {
    AnyArray root = MakeStringAndBytesLeaves();
    AnyArray owner = root;  // NOLINT(performance-unnecessary-copy-initialization)
    int string_callback_count = 0;
    int bytes_callback_count = 0;

    AnyArray identity = StructuralMap<order>(
                            root,
                            [&](const String& value) -> Expected<Any> {
                              ++string_callback_count;
                              return Any(value);
                            },
                            [&](const Bytes& value) -> Expected<Any> {
                              ++bytes_callback_count;
                              return Any(value);
                            })
                            .template cast<AnyArray>();

    EXPECT_TRUE(identity.same_as(root));
    EXPECT_TRUE(owner.same_as(root));
    EXPECT_EQ(string_callback_count, 2);
    EXPECT_EQ(bytes_callback_count, 2);
  }

  // Matching callbacks can replace both representations without traversing into them.
  {
    AnyArray root = MakeStringAndBytesLeaves();
    AnyArray replaced = StructuralMap<order>(
                            root,
                            [](const String& value) -> Expected<Any> {
                              return Any(static_cast<int64_t>(value.size()));
                            },
                            [](const Bytes& value) -> Expected<Any> {
                              return Any(static_cast<int64_t>(value.size()));
                            })
                            .template cast<AnyArray>();

    EXPECT_EQ(replaced[0].cast<int64_t>(), 1);
    EXPECT_EQ(replaced[1].cast<int64_t>(), 7);
    EXPECT_EQ(replaced[2].cast<int64_t>(), 8);
    EXPECT_EQ(replaced[3].cast<int64_t>(), 7);
    EXPECT_EQ(replaced[4].cast<int64_t>(), 8);
  }
}

TEST(StructuralMap, HandlesInlineAndHeapStringAndBytesLeaves) {
  CheckStringAndBytesLeaves<WalkOrder::kPreOrder>();
  CheckStringAndBytesLeaves<WalkOrder::kPostOrder>();
}

// The dynamic mutator duplicates the static one's walk deliberately, so the semantics they
// share need coverage on this copy too. Driven through the same entry point Python uses.
Any CallDynStructuralMap(AnyView root, const Array<Tuple<int32_t, Function>>& callbacks,
                         WalkOrder order) {
  Function fn = Function::GetGlobalRequired("ffi.StructuralMap");
  return fn(root, callbacks, Array<Tuple<int32_t, Function>>(), static_cast<int32_t>(order));
}

TEST(StructuralMapDyn, PreOrderDescendsOriginalAfterUnchangedCallback) {
  AnyArray root{int64_t{1}};
  Function unchanged = Function::FromTyped([](const AnyArray&) -> Any { return Unchanged(); });
  Function increment = Function::FromTyped([](int64_t value) -> Any { return Any(value + 1); });

  AnyArray mapped =
      CallDynStructuralMap(root,
                           {Tuple<int32_t, Function>(TypeIndex::kTVMFFIArray, unchanged),
                            Tuple<int32_t, Function>(TypeIndex::kTVMFFIInt, increment)},
                           WalkOrder::kPreOrder)
          .cast<AnyArray>();
  EXPECT_EQ(mapped[0].cast<int64_t>(), 2);
}

TEST(StructuralMapDyn, InvokesCallbackForEveryFreeVarOccurrence) {
  TVar var("x");
  AnyArray root{Any(var), Any(var)};
  int64_t calls = 0;
  Function remap = Function::FromTyped([&](AnyView v) -> Any {
    ++calls;
    return Any(TVar(v.cast<TVar>()->name + "-mapped"));
  });
  Any mapped = CallDynStructuralMap(
      root, {Tuple<int32_t, Function>(TVarObj::RuntimeTypeIndex(), remap)}, WalkOrder::kPostOrder);
  auto arr = mapped.cast<AnyArray>();
  EXPECT_EQ(calls, 2);
  EXPECT_FALSE(arr[0].cast<TVar>().same_as(arr[1].cast<TVar>()));
}

template <WalkOrder order>
void CheckDynamicParentLayer() {
  int64_t calls = 0;
  Function increment = Function::FromTyped([&](int64_t value) -> Any {
    ++calls;
    return Any(value + 1);
  });
  using Mutator = StructuralMapDynEngine<StructuralMapWithMutateCount, order>;
  auto engine = make_object<Mutator>(
      Array<Tuple<int32_t, Function>>{Tuple<int32_t, Function>(TypeIndex::kTVMFFIInt, increment)},
      Array<Tuple<int32_t, Function>>());
  StructuralMutator mutator(engine);

  AnyArray root{int64_t{1}};
  AnyArray mapped = std::move(mutator->Mutate<AnyArray>(root)).ValueOrUnchanged(std::move(root));
  EXPECT_EQ(mapped[0].cast<int64_t>(), 2);
  EXPECT_EQ(calls, 1);
  EXPECT_GT(engine->count().value, 0);
}

TEST(StructuralMapDyn, ParentLayerRunsThroughHeaderDefinedEngine) {
  CheckDynamicParentLayer<WalkOrder::kPreOrder>();
  CheckDynamicParentLayer<WalkOrder::kPostOrder>();
}

Any CallDynStructuralMutate(Any root,  // NOLINT(performance-unnecessary-value-param)
                            const Array<Tuple<int32_t, Function, bool>>& callbacks) {
  Function fn = Function::GetGlobalRequired("ffi.StructuralMutate");
  return fn(std::move(root), callbacks);
}

TEST(StructuralMutateDyn, PreservesDistinctDefaultDescentPaths) {
  Function increment = Function::FromTyped(
      [](int64_t value, const StructuralMutator&) -> Any { return Any(value + 1); });
  Array<Tuple<int32_t, Function, bool>> callbacks{
      Tuple<int32_t, Function, bool>(TypeIndex::kTVMFFIInt, increment, false)};

  AnyArray unique_root{int64_t{1}};
  AnyArray unique_mapped = CallDynStructuralMutate(unique_root, callbacks).cast<AnyArray>();
  EXPECT_FALSE(unique_mapped.same_as(unique_root));
  EXPECT_EQ(unique_root[0].cast<int64_t>(), 1);
  EXPECT_EQ(unique_mapped[0].cast<int64_t>(), 2);

  AnyArray shared_root{int64_t{1}};
  AnyArray extra_owner = shared_root;  // NOLINT(performance-unnecessary-copy-initialization)
  AnyArray shared_mapped = CallDynStructuralMutate(shared_root, callbacks).cast<AnyArray>();
  EXPECT_FALSE(shared_mapped.same_as(shared_root));
  EXPECT_TRUE(extra_owner.same_as(shared_root));
  EXPECT_EQ(shared_root[0].cast<int64_t>(), 1);
  EXPECT_EQ(shared_mapped[0].cast<int64_t>(), 2);
}

}  // namespace
