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
/*!
 * \file tvm/ffi/extra/structural_visit.h
 * \brief Structural visit API.
 */
#ifndef TVM_FFI_EXTRA_STRUCTURAL_VISIT_H_
#define TVM_FFI_EXTRA_STRUCTURAL_VISIT_H_

#include <tvm/ffi/any.h>
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/cast.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/tuple.h>
#include <tvm/ffi/container/variant.h>
#include <tvm/ffi/expected.h>
#include <tvm/ffi/extra/visit_error_context.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/function_details.h>
#include <tvm/ffi/optional.h>
#include <tvm/ffi/reflection/accessor.h>

#include <cstddef>
#include <exception>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

namespace tvm {
namespace ffi {

/*!
 * \brief Object node carrying the optional payload for an interrupted structural visit.
 */
class VisitInterruptObj : public Object {
 public:
  /*! \brief Payload returned with the interrupt, or FFI None for no payload. */
  Any value;

  VisitInterruptObj() = default;
  /*!
   * \brief Construct a VisitInterruptObj with a payload.
   * \param value The payload carried by the interrupt.
   */
  explicit VisitInterruptObj(Any value) : value(std::move(value)) {}

  /// \cond Doxygen_Suppress
  static constexpr const int32_t _type_index = TypeIndex::kTVMFFIVisitInterrupt;
  static const constexpr bool _type_final = true;
  TVM_FFI_DECLARE_OBJECT_INFO_STATIC(StaticTypeKey::kTVMFFIVisitInterrupt, VisitInterruptObj,
                                     Object);
  /// \endcond
};

/*!
 * \brief ObjectRef wrapper for VisitInterruptObj.
 */
class VisitInterrupt : public ObjectRef {
 public:
  /*! \brief Construct an interrupt with no payload. */
  VisitInterrupt() : VisitInterrupt(Any(nullptr)) {}
  /*!
   * \brief Construct an interrupt with a user-defined payload.
   * \param value The payload carried by the interrupt.
   */
  explicit VisitInterrupt(Any value)
      : ObjectRef(make_object<VisitInterruptObj>(std::move(value))) {}

  /// \cond Doxygen_Suppress
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(VisitInterrupt, ObjectRef, VisitInterruptObj);
  /// \endcond
};

class StructuralVisitorObj;

/*!
 * \brief ABI of structural visit for ``kStructuralVisit`` type attribute and
 * ``StructuralVisitorVTable`` function pointer signature.
 *
 * The callback receives the visitor and the value being visited as an
 * ``AnyView``. It returns a raw ``TVMFFIAny`` storing
 * ``Expected<Optional<VisitInterrupt>>``.
 */
using FStructuralVisit = TVMFFIAny (*)(StructuralVisitorObj* visitor, AnyView value) noexcept;

namespace details {

// Visit reflected structural fields of an object-backed value.
TVM_FFI_INLINE static Expected<Optional<VisitInterrupt>> VisitReflectedFieldsExpected(
    StructuralVisitorObj* visitor, const Object* obj) noexcept;

}  // namespace details

/*!
 * \brief VTable ABI for \ref StructuralVisitor dispatch. This function table provides a stable ABI
 * for the visit method.
 */
struct StructuralVisitorVTable {
  /*!
   * \brief Visit callback.
   * \param visitor The active structural visitor.
   * \param value The value to visit.
   * \return TVMFFIAny carrying Expected<Optional<VisitInterrupt>>.
   *
   * \note The raw ``visitor`` pointer and ``value`` view are non-owning. On
   * failure, the returned ``TVMFFIAny`` stores ``Error``; on success, it stores
   * either None or ``VisitInterrupt``.
   */
  FStructuralVisit visit = nullptr;
};

/*!
 * \brief Object node of a structural visitor.
 *
 * A structural visitor is an active traversal context.  It carries the dispatch
 * table used to visit each object and the current def-region state used by
 * structural equality/hash semantics.  The visitor is ref-counted so it can
 * cross FFI boundaries, but one underlying visitor object should not be shared
 * by overlapping top-level traversals.
 */
class StructuralVisitorObj : public Object {
 public:
  /*! \brief Callback-facing visitor type used by composed callback-driven engines. */
  using VisitorObjType = StructuralVisitorObj;
  /*! \brief State references made available to callback-aware visitor layers. */
  using StateTupleType = std::tuple<>;

  /*!
   * \brief Visit a value, dispatching through this visitor's vtable.
   *
   * \param value The value to visit.
   * \return ``std::nullopt`` to continue traversal, or a \ref VisitInterrupt
   *         to halt the entire visit.
   */
  TVM_FFI_INLINE Optional<VisitInterrupt> Visit(AnyView value) {
    return VisitExpected(value).value();
  }

  /*!
   * \brief Visit a value, propagating error through expected return.
   *
   * \param value The value to visit.
   * \return Expected interrupt state. An error means traversal failed.
   */
  TVM_FFI_INLINE Expected<Optional<VisitInterrupt>> VisitExpected(AnyView value) noexcept {
    return details::ExpectedUnsafe::MoveFromTVMFFIAny<Optional<VisitInterrupt>>(
        (*vtable_->visit)(this, value));
  }

  /*!
   * \brief Return the current def-region context.
   * \return The active def-region kind.
   */
  TVM_FFI_INLINE TVMFFIDefRegionKind def_region_kind() const { return def_region_mode_; }

  /*!
   * \brief Temporarily switch the def-region context while invoking \p callback.
   *
   * \param kind The def-region kind to set during the callback.
   * \param callback A nullary callable that performs recursive visiting.
   * \return The value returned by \p callback.
   */
  template <typename Callback>
  TVM_FFI_INLINE auto WithDefRegionKind(TVMFFIDefRegionKind kind, Callback&& callback) {
    class Scope {
     public:
      Scope(StructuralVisitorObj* visitor, TVMFFIDefRegionKind kind)
          : visitor_(visitor), old_kind_(visitor->def_region_mode_) {
        visitor_->def_region_mode_ = kind;
      }
      ~Scope() { visitor_->def_region_mode_ = old_kind_; }
      Scope(const Scope&) = delete;
      Scope& operator=(const Scope&) = delete;

     private:
      StructuralVisitorObj* visitor_;
      TVMFFIDefRegionKind old_kind_;
    };
    Scope scope(this, kind);
    return std::forward<Callback>(callback)();
  }

  /*!
   * \brief Visit using the structural visit behavior registered by kStructuralVisit for each type,
   * or reflected structural fields when no custom behavior is registered.
   *
   * \note Dispatches to the value type's registered ``__s_visit__`` hook, falling back to
   * reflected fields only when no hook is registered. Call it on a child, or on a matched
   * value from a ``StructuralVisit`` callback to bypass callback dispatch for that value.
   * Called on the value whose own hook is running, it re-enters that hook -- there is no way
   * to request the reflected path from inside a hook.
   *
   * \param value The value to visit.
   * \return Expected interrupt state. An error means traversal failed.
   */
  TVM_FFI_INLINE Expected<Optional<VisitInterrupt>> DefaultVisitExpected(AnyView value) noexcept {
    int32_t type_index = value.type_index();
    static reflection::TypeAttrColumn column(reflection::type_attr::kStructuralVisit);
    AnyView attr = column[type_index];

    // case 1: Type-specific override registered as an opaque ABI visit function pointer.
    if (attr.type_index() == TypeIndex::kTVMFFIOpaquePtr) {
      auto* visit_fn = reinterpret_cast<FStructuralVisit>(attr.cast<void*>());
      TVMFFIAny raw = (*visit_fn)(this, value);
      return details::ExpectedUnsafe::MoveFromTVMFFIAny<Optional<VisitInterrupt>>(raw);
    }

    // case 2: Type-specific override registered as an ffi::Function.
    if (attr.type_index() == TypeIndex::kTVMFFIFunction) {
      return attr.cast<Function>().CallExpected<Optional<VisitInterrupt>>(this, value);
    }

    if (TVM_FFI_PREDICT_FALSE(attr.type_index() != TypeIndex::kTVMFFINone)) {
      return Unexpected(
          Error("TypeError", "__s_visit__ must be an opaque function pointer or ffi.Function", ""));
    }

    if (type_index < TypeIndex::kTVMFFIStaticObjectBegin) {
      return Optional<VisitInterrupt>(std::nullopt);
    }

    return details::VisitReflectedFieldsExpected(this, value.cast<const Object*>());
  }

  /// \cond Doxygen_Suppress
  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("ffi.StructuralVisitor", StructuralVisitorObj, Object);
  /// \endcond

 protected:
  /*!
   * \brief Redirect raw ABI descent to \ref DefaultVisitExpected.
   *
   * A visitor layer that overrides either descent form must redeclare both in
   * the same layer so dependent member lookup reaches the paired override. The
   * raw form is deliberately retained for ABI traversal and permits a layer to
   * forward raw storage without rematerializing a typed ``Expected``.
   *
   * \param value The value to descend into.
   * \return Raw ``Expected<Optional<VisitInterrupt>>`` storage produced by
   *         ``details::ExpectedUnsafe::MoveToTVMFFIAny``. The caller
   *         reinterprets this storage without a runtime type check.
   */
  TVM_FFI_INLINE TVMFFIAny DefaultVisitRaw(AnyView value) noexcept {
    return details::ExpectedUnsafe::MoveToTVMFFIAny(DefaultVisitExpected(value));
  }

  /*! \brief Return the state references maintained by this visitor layer. */
  TVM_FFI_INLINE StateTupleType StateTuple() const noexcept { return {}; }

  /*!
   * \brief Construct a structural visitor from an immutable dispatch vtable.
   * \param vtable The non-null dispatch table for this visitor. It must outlive this object.
   */
  explicit StructuralVisitorObj(const StructuralVisitorVTable* vtable) : vtable_(vtable) {}

  /*!
   * \brief Required ABI dispatch table. \ref StructuralVisitorVTable
   * It must never be null on a constructed visitor.
   */
  const StructuralVisitorVTable* vtable_ = nullptr;

  /*!
   * \brief Current def-region context for structural equality/hash semantics.
   *
   * This is shared mutable traversal state. Be careful when mutating it through
   * multiple references to the same visitor object. Use \ref WithDefRegionKind
   * to scope temporary changes.
   */
  TVMFFIDefRegionKind def_region_mode_ = kTVMFFIDefRegionKindNone;
};

/*!
 * \brief ObjectRef wrapper of \ref StructuralVisitorObj.
 *
 * \sa StructuralVisitorObj
 */
class StructuralVisitor : public ObjectRef {
 public:
  /*!
   * \brief Construct from an existing object pointer.
   * \param n The object pointer to wrap.
   */
  explicit StructuralVisitor(ObjectPtr<StructuralVisitorObj> n) : ObjectRef(std::move(n)) {}

  /// \cond Doxygen_Suppress
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(StructuralVisitor, ObjectRef, StructuralVisitorObj);
  /// \endcond
};

namespace details {

/*!
 * \brief Return true when \p result already carries a traversal-stopping state.
 * \tparam T The Expected success type.
 * \param result The Expected value to inspect.
 * \return Whether \p result stores an Error or VisitInterrupt.
 */
template <typename T>
TVM_FFI_INLINE bool StructuralVisitNeedEarlyReturn(const Expected<T>& result) noexcept {
  int32_t type_index = result.type_index();
  return type_index == TypeIndex::kTVMFFIError || type_index == TypeIndex::kTVMFFIVisitInterrupt;
}

/*!
 * \brief Return true when raw Expected storage carries a traversal-stopping state.
 * \param result Raw ``Expected<Optional<VisitInterrupt>>`` storage returned by
 *               a visitor descent hook. Its type contract is trusted without
 *               a runtime check.
 * \return Whether \p result carries anything other than successful completion.
 */
TVM_FFI_INLINE bool StructuralVisitRawNeedEarlyReturn(const TVMFFIAny& result) noexcept {
  return result.type_index != TypeIndex::kTVMFFINone;
}

/*!
 * \brief Walk reflected structural fields of object-backed \p obj.
 *
 * Fields marked with ``kTVMFFIFieldFlagBitMaskSEqHashIgnore`` are skipped.
 * Def-region field flags are scoped around recursive child visits.
 *
 * \param visitor The active visitor.
 * \param obj The object whose reflected fields should be visited.
 * \return Expected interrupt state. An error means traversal failed.
 */
TVM_FFI_INLINE static Expected<Optional<VisitInterrupt>> VisitReflectedFieldsExpected(
    StructuralVisitorObj* visitor, const Object* obj) noexcept {
  int32_t type_index = obj->type_index();
  const TVMFFITypeInfo* type_info = TVMFFIGetTypeInfo(type_index);
  auto visit_fields = [&]() -> Expected<Optional<VisitInterrupt>> {
    Expected<Optional<VisitInterrupt>> result = Optional<VisitInterrupt>(std::nullopt);
    reflection::ForEachFieldInfoWithEarlyStop(
        type_info, [&](const TVMFFIFieldInfo* field_info) -> bool {
          if (field_info->flags & kTVMFFIFieldFlagBitMaskSEqHashIgnore) {
            return false;
          }

          Any field_value;
          const void* field_addr = reinterpret_cast<const char*>(obj) + field_info->offset;
          int ret_code = field_info->getter(const_cast<void*>(field_addr),
                                            reinterpret_cast<TVMFFIAny*>(&field_value));
          if (TVM_FFI_PREDICT_FALSE(ret_code != 0)) {
            result = Unexpected(details::MoveFromSafeCallRaised());
            return true;
          }

          if (field_info->flags & kTVMFFIFieldFlagBitMaskSEqHashDefNonRecursive) {
            result = visitor->WithDefRegionKind(kTVMFFIDefRegionKindNonRecursive, [&]() {
              return visitor->VisitExpected(field_value);
            });
          } else if (field_info->flags & kTVMFFIFieldFlagBitMaskSEqHashDefRecursive) {
            result = visitor->WithDefRegionKind(kTVMFFIDefRegionKindRecursive, [&]() {
              return visitor->VisitExpected(field_value);
            });
          } else {
            result = visitor->VisitExpected(field_value);
          }
          return StructuralVisitNeedEarlyReturn(result);
        });
    return result;
  };

  // A non-recursive definition applies to the FreeVar itself, but its fields are uses. The
  // complete field traversal are clamped to None, then the definition region is restored.
  if (visitor->def_region_kind() == kTVMFFIDefRegionKindNonRecursive &&
      type_info->metadata != nullptr &&
      type_info->metadata->structural_eq_hash_kind == kTVMFFISEqHashKindFreeVar) {
    return visitor->WithDefRegionKind(kTVMFFIDefRegionKindNone, visit_fields);
  }
  return visit_fields();
}

}  // namespace details

// ---------------------------------------------------------------------------
// Structural Walk API.
// ---------------------------------------------------------------------------

/*!
 * \brief Per-node control signal returned by structural walk callbacks.
 *
 * Walk control result with one of three actions:
 * - ``WalkResult::Advance()``: continue traversal, including this node's children.
 * - ``WalkResult::Skip()``: continue traversal but skip this node's children.
 * - ``WalkResult::Interrupt()``: halt the entire walk, optionally carrying a payload.
 */
class WalkResult : public Variant<VisitInterrupt, int32_t> {
 public:
  /*! \brief Internal tag value carried by ``WalkResult::Advance()``. */
  static constexpr int32_t kAdvanceTag = 0;
  /*! \brief Internal tag value carried by ``WalkResult::Skip()``. */
  static constexpr int32_t kSkipTag = 1;

  /*! \brief The underlying ``Variant`` used as storage. */
  using Storage = Variant<VisitInterrupt, int32_t>;

  /*! \brief Continue traversal and visit this node's children. */
  static WalkResult Advance() { return WalkResult(kAdvanceTag); }

  /*! \brief Continue traversal but skip this node's children. */
  static WalkResult Skip() { return WalkResult(kSkipTag); }

  /*!
   * \brief Halt the walk and propagate an interrupt.
   * \param signal The interrupt to propagate. Defaults to an interrupt with
   *               FFI None payload.
   */
  static WalkResult Interrupt(VisitInterrupt signal = VisitInterrupt()) {
    return WalkResult(Storage(std::move(signal)));
  }

 private:
  // Keep raw storage construction behind the named factories.
  explicit WalkResult(int32_t tag) : Storage(tag) {}
  explicit WalkResult(Storage storage) : Storage(std::move(storage)) {}

  friend struct TypeTraits<WalkResult>;
};

/// \cond Doxygen_Suppress
template <>
inline constexpr bool use_default_type_traits_v<WalkResult> = false;

// Allow WalkResult to round-trip through Any / Expected while reusing Variant storage.
template <>
struct TypeTraits<WalkResult> : public TypeTraits<WalkResult::Storage> {
  using Base = TypeTraits<WalkResult::Storage>;

  TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
    return src->type_index == TypeIndex::kTVMFFINone || Base::CheckAnyStrict(src);
  }
  // Decode from borrowed Any storage after a strict type check.
  TVM_FFI_INLINE static WalkResult CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
    if (src->type_index == TypeIndex::kTVMFFINone) {
      return WalkResult::Advance();
    }
    return WalkResult(Base::CopyFromAnyViewAfterCheck(src));
  }
  // Decode by moving from owned Any storage after a strict type check.
  TVM_FFI_INLINE static WalkResult MoveFromAnyAfterCheck(TVMFFIAny* src) {
    if (src->type_index == TypeIndex::kTVMFFINone) {
      return WalkResult::Advance();
    }
    return WalkResult(Base::MoveFromAnyAfterCheck(src));
  }
  // Try all conversions supported by the underlying Variant storage.
  TVM_FFI_INLINE static std::optional<WalkResult> TryCastFromAnyView(const TVMFFIAny* src) {
    if (src->type_index == TypeIndex::kTVMFFINone) {
      return WalkResult::Advance();
    }
    if (auto opt = Base::TryCastFromAnyView(src)) {
      return WalkResult(*std::move(opt));
    }
    return std::nullopt;
  }
  TVM_FFI_INLINE static std::string TypeStr() { return "WalkResult"; }
};
/// \endcond

/*!
 * \brief Callback order for recursive structural traversal.
 */
enum class WalkOrder : int32_t {
  /*! \brief Invoke the callback before traversing children. */
  kPreOrder = 0,
  /*! \brief Invoke the callback after traversing children. */
  kPostOrder = 1,
};

namespace details {

/*!
 * \brief Return from a visit hook if \p Result stops traversal.
 *
 * Propagates an ``Error`` or a ``VisitInterrupt`` out of the enclosing function
 * and otherwise falls through. Works from a raw ``TVMFFIAny`` hook and from a
 * typed ``Expected`` helper alike; the rvalue-only proxy lets the return type
 * select the representation.
 *
 * A registered ``__s_visit__`` hook is one line per traversed field followed by
 * the terminal return. A field skipped on purpose is guarded by a condition and
 * carries a ``// skips:`` note saying why.
 *
 * \code{.cpp}
 * TVMFFIAny FooVisit(StructuralVisitorObj* visitor, AnyView value) noexcept {
 *   const FooNode* self =
 *       details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const FooNode>(value);
 *   TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->a));
 *   TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->b));
 *   return AnyView(nullptr).CopyToTVMFFIAny();
 * }
 * \endcode
 *
 * \param Result An expression yielding the descent result to inspect.
 */
#define TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Result)                                \
  do {                                                                            \
    auto&& tvm_ffi_res_ = (Result);                                               \
    if (TVM_FFI_PREDICT_FALSE(                                                    \
            ::tvm::ffi::details::StructuralVisitNeedEarlyReturn(tvm_ffi_res_))) { \
      return ::tvm::ffi::details::MaybeReturnHelper(::std::move(tvm_ffi_res_));   \
    }                                                                             \
  } while (0)

}  // namespace details

/*!
 * \brief Callback-dispatched walk engine with a state-carrying Parent layer.
 *
 * Every callback receives all entries of ``Parent::StateTupleType``
 * positionally after the value and may optionally take
 * ``TVMFFIDefRegionKind`` as the final argument.
 *
 * A Parent layer derives from ``StructuralVisitorObj``, declares a public
 * ``StateTupleType``, accepts and forwards ``const StructuralVisitorVTable*``
 * in its constructor, and provides an at-least-protected
 * ``StateTuple() const noexcept`` that returns ``StateTupleType`` by value. The
 * state references in that tuple must outlive the traversal. A layer overriding
 * either ``DefaultVisitExpected`` or ``DefaultVisitRaw`` declares both forms,
 * at least protected, in that same class. The raw form must return raw
 * ``Expected<Optional<VisitInterrupt>>`` storage produced by
 * ``details::ExpectedUnsafe::MoveToTVMFFIAny``; the engine propagates it without
 * a runtime type check. This deliberate pair keeps the raw ABI path available
 * without rematerializing a typed ``Expected``. Engine calls use ``Parent::``
 * qualification; this is static layer dispatch, not virtual dispatch. A layer
 * must still define its own raw boilerplate because boilerplate inherited from
 * a base resolves its unqualified typed call in that base's scope.
 *
 * \tparam Parent Traversal layer extended by the engine. Each layer that
 *                customizes typed descent must define its own ``Default*Raw``
 *                boilerplate; inherited boilerplate resolves its unqualified
 *                typed call in the base layer's scope. Engine protocol and
 *                descent calls are ``Parent::``-qualified.
 * \tparam order Callback placement relative to child traversal.
 * \tparam Callbacks Callback links, tested in declaration order.
 */
template <typename Parent, WalkOrder order, typename... Callbacks>
class StructuralWalkEngine : public Parent {
 public:
  static_assert(std::is_base_of_v<StructuralVisitorObj, Parent>,
                "StructuralWalk Parent must derive from StructuralVisitorObj");
  /*! \brief Tuple of const state references supplied by the Parent layer. */
  using StateTupleType = typename Parent::StateTupleType;

  /*!
   * \brief Construct a structural walk visitor.
   * \param callbacks The typed callback links, tested in declaration order.
   */
  explicit StructuralWalkEngine(Callbacks... callbacks)
      : Parent(VTable()), callbacks_(std::move(callbacks)...) {}

 private:
  /*!
   * \brief Return the shared callback-aware visitor vtable.
   * \return Pointer to the immutable visitor vtable for this specialization.
   */
  static const StructuralVisitorVTable* VTable() {
    static const StructuralVisitorVTable vtable{
        &StructuralWalkEngine::DispatchVisit,
    };
    return &vtable;
  }

  /*!
   * \brief Dispatch from the erased visitor pointer to the concrete walk visitor.
   * \param self The erased structural visitor object.
   * \param value The value to visit.
   * \return Interrupt state, or an error if traversal failed.
   */
  static TVMFFIAny DispatchVisit(StructuralVisitorObj* self, AnyView value) noexcept {
    return static_cast<StructuralWalkEngine*>(self)->VisitImpl(value);
  }

  /*! \brief Invoke one matched callback with its declared state, if any. */
  template <typename Callback, typename Value, size_t... Is>
  TVM_FFI_INLINE Expected<WalkResult> InvokeCallbackLink(Callback& callback, Value&& value,
                                                         std::index_sequence<Is...>) noexcept {
    using FuncInfo = details::FunctionInfo<std::decay_t<Callback>>;
    static_assert(
        FuncInfo::num_args == 1 + sizeof...(Is) || FuncInfo::num_args == 2 + sizeof...(Is),
        "StructuralWalk callback takes (value, state...) with an optional trailing "
        "definition-region kind");
    try {
      static_assert(std::is_same_v<decltype(Parent::StateTuple()), StateTupleType>,
                    "Parent::StateTuple() must return Parent::StateTupleType by value");
      StateTupleType states = Parent::StateTuple();
      if constexpr (FuncInfo::num_args == 1 + sizeof...(Is)) {
        return callback(std::forward<Value>(value), std::get<Is>(states)...);
      } else {
        return callback(std::forward<Value>(value), std::get<Is>(states)...,
                        Parent::def_region_kind());
      }
    } catch (const Error& err) {
      return Unexpected(err);
    }
  }

  /*! \brief Try one callback link, storing its result when the value type matches. */
  template <typename Callback>
  TVM_FFI_INLINE bool TryLink(Callback& callback, AnyView value,
                              Expected<WalkResult>* out) noexcept {
    using FuncInfo = details::FunctionInfo<std::decay_t<Callback>>;
    static_assert(FuncInfo::num_args >= 1, "StructuralWalk callback requires a value argument");
    using FirstArg = std::tuple_element_t<0, typename FuncInfo::ArgType>;
    using TSub = std::remove_cv_t<std::remove_reference_t<FirstArg>>;
    using StateIndices = std::make_index_sequence<std::tuple_size_v<StateTupleType>>;
    if constexpr (std::is_same_v<TSub, AnyView>) {
      *out = InvokeCallbackLink(callback, value, StateIndices{});
      return true;
    } else if constexpr (std::is_same_v<TSub, Any>) {
      *out = InvokeCallbackLink(callback, Any(value), StateIndices{});
      return true;
    } else if (auto matched = value.template as<TSub>()) {
      *out = InvokeCallbackLink(callback, *std::move(matched), StateIndices{});
      return true;
    }
    return false;
  }

  /*! \brief Try callback links in declaration order until the first match. */
  template <size_t... Is>
  TVM_FFI_INLINE bool TryLinks(AnyView value, Expected<WalkResult>* out,
                               std::index_sequence<Is...>) noexcept {
    return (TryLink(std::get<Is>(callbacks_), value, out) || ...);
  }

  /*!
   * \brief Visit one value according to the configured walk order.
   * \param value The value to visit.
   * \return Interrupt state, or an error if traversal failed.
   */
  TVMFFIAny VisitImpl(AnyView value) noexcept {
    if (TVM_FFI_PREDICT_FALSE(value.type_index() == TypeIndex::kTVMFFINone)) {
      return details::ExpectedUnsafe::MoveToTVMFFIAny(
          Expected<Optional<VisitInterrupt>>(std::nullopt));
    }
    if constexpr (order == WalkOrder::kPreOrder) {
      Expected<WalkResult> result = WalkResult::Advance();
      TryLinks(value, &result, std::index_sequence_for<Callbacks...>{});
      if (TVM_FFI_PREDICT_FALSE(details::StructuralVisitNeedEarlyReturn(result))) {
        if (TVM_FFI_PREDICT_FALSE(result.is_err())) {
          Error err = result.error();
          details::UpdateVisitErrorContext(err, value);
        }
        return details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(result));
      }
      // Hoist the call out of TVM_FFI_UNSAFE_ASSUME: clang's -Wassume rejects
      // arguments that contain a call expression (its potential side effects
      // would be discarded), while [[maybe_unused]] keeps -Wunused-variable
      // quiet on configs where the assume macro compiles away.
      [[maybe_unused]] int32_t type_index = result.type_index();
      TVM_FFI_UNSAFE_ASSUME(type_index == TypeIndex::kTVMFFIInt);
      if (TVM_FFI_PREDICT_FALSE(details::ExpectedUnsafe::ValueAs<int32_t>(result) ==
                                WalkResult::kSkipTag)) {
        return details::ExpectedUnsafe::MoveToTVMFFIAny(
            Expected<Optional<VisitInterrupt>>(std::nullopt));
      }
    }

    {
      TVMFFIAny result = Parent::DefaultVisitRaw(value);
      if (TVM_FFI_PREDICT_FALSE(details::StructuralVisitRawNeedEarlyReturn(result))) {
        if (TVM_FFI_PREDICT_FALSE(result.type_index == TypeIndex::kTVMFFIError)) {
          details::UpdateVisitErrorContext(result, value);
        }
        return result;
      }
    }

    if constexpr (order == WalkOrder::kPostOrder) {
      Expected<WalkResult> result = WalkResult::Advance();
      TryLinks(value, &result, std::index_sequence_for<Callbacks...>{});
      if (TVM_FFI_PREDICT_FALSE(details::StructuralVisitNeedEarlyReturn(result))) {
        if (TVM_FFI_PREDICT_FALSE(result.is_err())) {
          Error err = result.error();
          details::UpdateVisitErrorContext(err, value);
        }
        return details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(result));
      }
    }

    return details::ExpectedUnsafe::MoveToTVMFFIAny(
        Expected<Optional<VisitInterrupt>>(std::nullopt));
  }

  /*! \brief The callback links, tested in declaration order. */
  std::tuple<Callbacks...> callbacks_;
};

/*!
 * \brief Walk a structured value graph and invoke typed callbacks on selected values.
 *
 * The callbacks are invoked only for values matching the first argument type of
 * one of the callbacks. The first callback argument may be ``AnyView``, ``Any``,
 * an object reference type, an object pointer type, or another FFI-convertible
 * POD type. It may optionally take ``TVMFFIDefRegionKind`` after the value.
 * Callbacks are tested in order, and the first match is used.
 *
 * Each callback should return ``Expected<WalkResult>``; see ``WalkResult``.
 * - ``WalkResult::Interrupt(...)`` halts traversal.
 * - ``WalkResult::Advance()`` continues traversal.
 * - ``WalkResult::Skip()`` skips children traversal.
 * - ``Error`` indicates traversal failure.
 *
 * \sa WalkOrder, WalkResult
 *
 * \tparam order Whether to invoke the callback before or after visiting children.
 * \tparam Callbacks Callback types.
 * \param root The root value to visit.
 * \param callbacks Callbacks invoked for matching nodes as ``(value)`` or
 *                  ``(value, TVMFFIDefRegionKind)``.
 * \return ``std::nullopt`` if traversal completed, or the interrupt returned by
 *         a callback.
 *
 * \note Return type of each callback should be ``Expected<WalkResult>``.
 */
template <WalkOrder order, typename... Callbacks>
Expected<Optional<VisitInterrupt>> StructuralWalkExpected(AnyView root,
                                                          Callbacks&&... callbacks) noexcept {
  static_assert(sizeof...(Callbacks) != 0, "StructuralWalk requires at least one callback");
  using Visitor = StructuralWalkEngine<StructuralVisitorObj, order, std::decay_t<Callbacks>...>;
  StructuralVisitor visitor(make_object<Visitor>(std::forward<Callbacks>(callbacks)...));
  return visitor->VisitExpected(root);
}

/*!
 * \brief Throwing error over \ref tvm::ffi::StructuralWalkExpected.
 *
 * See \ref tvm::ffi::StructuralWalkExpected for callback semantics and traversal behavior.
 *
 * \tparam order Whether to invoke the callback before or after visiting children.
 * \tparam Callbacks Callback types.
 * \param root The root value to visit.
 * \param callbacks Callbacks invoked for matching nodes as ``(value)`` or
 *                  ``(value, TVMFFIDefRegionKind)``.
 * \return ``std::nullopt`` if traversal completed, or the interrupt returned by
 *         a callback.
 * \throws Error if traversal or a callback returned an error.
 *
 * \note Return type of each callback should be ``Expected<WalkResult>``.
 */
template <WalkOrder order, typename... Callbacks>
Optional<VisitInterrupt> StructuralWalk(AnyView root, Callbacks&&... callbacks) {
  return StructuralWalkExpected<order>(root, std::forward<Callbacks>(callbacks)...).value();
}

// ---------------------------------------------------------------------------
// Structural Visit API.
// ---------------------------------------------------------------------------

/*!
 * \brief Engine of the callback-dispatched \ref tvm::ffi::StructuralVisit.
 *
 * A matched callback owns descent into its value, and its result is final. A
 * value matching no callback uses the Parent layer's default descent. The local
 * typed callback fold preserves declaration-order first match and converts an
 * ``Error`` thrown by a matched callback into the visit result.
 *
 * \tparam Parent Traversal layer extended by the engine. Each layer that
 *                customizes typed descent must define its own ``Default*Raw``
 *                boilerplate; inherited boilerplate resolves its unqualified
 *                typed call in the base layer's scope. Engine protocol and
 *                descent calls are ``Parent::``-qualified.
 * \tparam Callbacks Callable types whose first parameter selects the dispatched value type.
 */
template <typename Parent, typename... Callbacks>
class StructuralVisitEngine : public Parent {
 public:
  static_assert(std::is_base_of_v<StructuralVisitorObj, Parent>,
                "StructuralVisit Parent must derive from StructuralVisitorObj");
  /*!
   * \brief Construct a visit engine over a chain of typed callbacks.
   * \param callbacks Callbacks tested in declaration order; the first match runs.
   */
  explicit StructuralVisitEngine(Callbacks... callbacks)
      : Parent(VTable()), callbacks_(std::move(callbacks)...) {}

 private:
  /*!
   * \brief Return this engine's callback-aware visitor vtable.
   * \return Pointer to the immutable visitor vtable for this specialization.
   */
  static const StructuralVisitorVTable* VTable() {
    static const StructuralVisitorVTable vtable{
        &StructuralVisitEngine::DispatchVisit,
    };
    return &vtable;
  }

  /*!
   * \brief Dispatch from the erased visitor pointer to the concrete engine.
   * \param self The erased structural visitor object.
   * \param value The value to visit.
   * \return Interrupt state, or an error if traversal failed.
   */
  static TVMFFIAny DispatchVisit(StructuralVisitorObj* self, AnyView value) noexcept {
    return static_cast<StructuralVisitEngine*>(self)->VisitImpl(value);
  }

  /*!
   * \brief Visit one value, handing a matched callback ownership of its descent.
   * \param value The value to visit.
   * \return Interrupt state, or an error if traversal failed.
   */
  TVMFFIAny VisitImpl(AnyView value) noexcept {
    if (TVM_FFI_PREDICT_FALSE(value.type_index() == TypeIndex::kTVMFFINone)) {
      return details::ExpectedUnsafe::MoveToTVMFFIAny(
          Expected<Optional<VisitInterrupt>>(std::nullopt));
    }
    if (std::optional<Expected<Optional<VisitInterrupt>>> matched = DispatchCallbacks(value)) {
      // The matched callback already traversed as much of `value` as it wanted, so its
      // result is final and the engine does not descend on its own.
      Expected<Optional<VisitInterrupt>> result = *std::move(matched);
      if (TVM_FFI_PREDICT_FALSE(result.is_err())) {
        Error err = result.error();
        details::UpdateVisitErrorContext(err, value);
      }
      return details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(result));
    }
    // No callback claimed `value`. The Parent layer owns default descent.
    TVMFFIAny result = Parent::DefaultVisitRaw(value);
    if (TVM_FFI_PREDICT_FALSE(result.type_index == TypeIndex::kTVMFFIError)) {
      details::UpdateVisitErrorContext(result, value);
    }
    return result;
  }

  /*! \brief Try one typed callback and preserve Error as an expected result. */
  template <typename Callback>
  TVM_FFI_INLINE std::optional<Expected<Optional<VisitInterrupt>>> TryLink(Callback& callback,
                                                                           AnyView value) noexcept {
    using FuncInfo = details::FunctionInfo<std::decay_t<Callback>>;
    static_assert(FuncInfo::num_args == 2, "StructuralVisit callback takes (value, visitor)");
    using FirstArg = std::tuple_element_t<0, typename FuncInfo::ArgType>;
    using TSub = std::remove_cv_t<std::remove_reference_t<FirstArg>>;
    using SecondArg = std::decay_t<std::tuple_element_t<1, typename FuncInfo::ArgType>>;
    using Second = std::remove_pointer_t<SecondArg>;
    static_assert(std::is_same_v<Second, typename Parent::VisitorObjType>,
                  "second StructuralVisit callback argument must be "
                  "exactly Parent::VisitorObjType*");
    auto* visitor = static_cast<typename Parent::VisitorObjType*>(this);
    try {
      if constexpr (std::is_same_v<TSub, AnyView>) {
        return callback(value, visitor);
      } else if constexpr (std::is_same_v<TSub, Any>) {
        return callback(Any(value), visitor);
      } else if (auto matched = value.template as<TSub>()) {
        return callback(*std::move(matched), visitor);
      }
    } catch (const Error& err) {
      return Unexpected(err);
    }
    return std::nullopt;
  }

  /*! \brief Fold this engine's callback tuple in declaration order. */
  template <size_t... Is>
  TVM_FFI_INLINE std::optional<Expected<Optional<VisitInterrupt>>> TryLinks(
      AnyView value, std::index_sequence<Is...>) noexcept {
    std::optional<Expected<Optional<VisitInterrupt>>> result;
    (... || (result = TryLink(std::get<Is>(callbacks_), value)).has_value());
    return result;
  }

  /*!
   * \brief Run the callback chain on \p value.
   * \param value The value to dispatch on.
   * \return The matched callback's result, or an empty optional when none matched.
   *
   * \note An unmatched value is reported as such rather than folded into a "continue"
   * result: the engine has to tell "the callback chose to stop here" apart from "no
   * callback claimed this value".
   */
  std::optional<Expected<Optional<VisitInterrupt>>> DispatchCallbacks(AnyView value) noexcept {
    return TryLinks(value, std::index_sequence_for<Callbacks...>{});
  }

  /*! \brief Typed callbacks tested in declaration order, first match wins. */
  std::tuple<Callbacks...> callbacks_;
};

/*!
 * \brief Visit a structured value, letting a matched callback own descent.
 *
 * Each callback takes ``(value, StructuralVisitorObj* visitor)`` and returns
 * ``Expected<Optional<VisitInterrupt>>``. The first argument follows the same
 * matching rules as ``StructuralWalk``; callbacks are tested in declaration
 * order and the first match is used.
 *
 * A matched callback owns descent into its value, and its result is final.
 * Returning ``std::nullopt`` completes that subtree, a ``VisitInterrupt`` halts
 * the traversal, and an ``Error`` fails it. A value matching no callback uses
 * the visitor's default descent.
 *
 * \sa StructuralWalkExpected, StructuralVisitorObj, VisitInterrupt
 *
 * \tparam Callbacks Callback types.
 * \param root The root value to visit.
 * \param callbacks Callbacks invoked for matching nodes.
 * \return ``std::nullopt`` if traversal completed, or the interrupt that halted it.
 */
template <typename... Callbacks>
Expected<Optional<VisitInterrupt>> StructuralVisitExpected(AnyView root,
                                                           Callbacks&&... callbacks) noexcept {
  static_assert(sizeof...(Callbacks) != 0, "StructuralVisit requires at least one callback");
  using Engine = StructuralVisitEngine<StructuralVisitorObj, std::decay_t<Callbacks>...>;
  StructuralVisitor visitor(make_object<Engine>(std::forward<Callbacks>(callbacks)...));
  return visitor->VisitExpected(root);
}

/*!
 * \brief Throwing error over \ref tvm::ffi::StructuralVisitExpected.
 *
 * See \ref tvm::ffi::StructuralVisitExpected for callback semantics and traversal behavior.
 *
 * \tparam Callbacks Callback types.
 * \param root The root value to visit.
 * \param callbacks Callbacks invoked for matching nodes. Each callback takes
 *                  ``(value, StructuralVisitorObj*)`` and should return
 *                  ``Expected<Optional<VisitInterrupt>>``.
 * \return ``std::nullopt`` if traversal completed, or the interrupt that halted it.
 * \throws Error if traversal or a callback returned an error.
 */
template <typename... Callbacks>
Optional<VisitInterrupt> StructuralVisit(AnyView root, Callbacks&&... callbacks) {
  return StructuralVisitExpected(root, std::forward<Callbacks>(callbacks)...).value();
}

}  // namespace ffi
}  // namespace tvm
#endif  // TVM_FFI_EXTRA_STRUCTURAL_VISIT_H_
