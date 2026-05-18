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
  /*! \brief Construct the default structural visitor. */
  StructuralVisitorObj() : StructuralVisitorObj(VTable()) {}

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
   * \brief Visit using the structural visit behavior registered by kStructuralVisit for each type,
   * or reflected structural fields when no custom behavior is registered.
   */
  TVM_FFI_INLINE Optional<VisitInterrupt> DefaultVisit(AnyView value) {
    return DefaultVisitExpected(value).value();
  }

  /*!
   * \brief Visit using the registered structural visit behavior by kStructuralVisit, propagating
   * errors by Expected.
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
      return details::ExpectedUnsafe::MoveFromTVMFFIAny<Optional<VisitInterrupt>>(
          (*visit_fn)(this, value));
    }

    // case 2: Type-specific override registered as an ffi::Function.
    if (attr.type_index() == TypeIndex::kTVMFFIFunction) {
      return attr.cast<Function>().CallExpected<Optional<VisitInterrupt>>(this, value);
    }

    if (TVM_FFI_PREDICT_FALSE(attr.type_index() != TypeIndex::kTVMFFINone)) {
      return Unexpected(Error("TypeError",
                              std::string(reflection::type_attr::kStructuralVisit) +
                                  " must be an opaque function pointer or ffi.Function",
                              ""));
    }

    if (type_index < TypeIndex::kTVMFFIStaticObjectBegin) {
      return Optional<VisitInterrupt>(std::nullopt);
    }

    return details::VisitReflectedFieldsExpected(this, value.cast<const Object*>());
  }

  /*!
   * \brief Return the current def-region context.
   * \return The active def-region kind.
   */
  TVM_FFI_INLINE TVMFFIDefRegionKind def_region_kind() const { return def_region_mode_; }

  /*!
   * \brief Temporarily switch the def-region context while invoking \p callback.
   *
   * This helper scopes updates to the traversal state used by def/use-region
   * aware visitors. The previous state is restored when the callback returns
   * or throws.
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

  /// \cond Doxygen_Suppress
  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("ffi.StructuralVisitor", StructuralVisitorObj, Object);
  /// \endcond

 protected:
  /*!
   * \brief Construct a structural visitor subclass with a custom dispatch vtable.
   *
   * \param vtable The non-null dispatch table for this visitor.
   *
   * \note This constructor is for internal subclasses.  The vtable and its
   *       ``visit`` callback must be valid for the lifetime of the visitor.
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

 private:
  /*!
   * \brief Return the vtable used by the default visitor.
   * \return Pointer to the static structural visitor vtable.
   */
  static const StructuralVisitorVTable* VTable() {
    static const StructuralVisitorVTable vtable{&StructuralVisitorObj::DispatchVisit};
    return &vtable;
  }

  /*!
   * \brief Dispatch from the vtable to the default visitor.
   * \param visitor The structural visitor object.
   * \param value The value to visit.
   * \return Interrupt state, or an error if traversal failed.
   */
  static TVMFFIAny DispatchVisit(StructuralVisitorObj* visitor, AnyView value) noexcept {
    auto interrupt = visitor->DefaultVisitExpected(value);
    if (TVM_FFI_PREDICT_FALSE(interrupt.type_index() == TypeIndex::kTVMFFIError)) {
      if (value.type_index() >= TypeIndex::kTVMFFIStaticObjectBegin) {
        Error err = interrupt.error();
        details::UpdateVisitErrorContext(err, value.cast<ObjectRef>());
      }
    }
    return details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(interrupt));
  }
};

/*!
 * \brief ObjectRef wrapper of \ref StructuralVisitorObj.
 *
 * \sa StructuralVisitorObj
 */
class StructuralVisitor : public ObjectRef {
 public:
  /*!
   * \brief Construct the default structural visitor.
   */
  StructuralVisitor() : ObjectRef(make_object<StructuralVisitorObj>()) {}
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

        TVMFFIDefRegionKind kind = kTVMFFIDefRegionKindNone;
        if (field_info->flags & kTVMFFIFieldFlagBitMaskSEqHashDefNonRecursive) {
          kind = kTVMFFIDefRegionKindNonRecursive;
        } else if (field_info->flags & kTVMFFIFieldFlagBitMaskSEqHashDefRecursive) {
          kind = kTVMFFIDefRegionKindRecursive;
        }

        if (kind != kTVMFFIDefRegionKindNone) {
          result = visitor->WithDefRegionKind(
              kind, [&]() { return visitor->VisitExpected(field_value); });
        } else {
          result = visitor->VisitExpected(field_value);
        }
        return StructuralVisitNeedEarlyReturn(result);
      });
  return result;
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
 * \brief Callback order for \ref tvm::ffi::StructuralWalk.
 */
enum class WalkOrder : int32_t {
  /*! \brief Invoke the callback before visiting children. */
  kPreOrder = 0,
  /*! \brief Invoke the callback after visiting children. */
  kPostOrder = 1,
};

namespace details {

/// \cond Doxygen_Suppress
// Return from the current ABI visit function if Result stops traversal.
// Result must be an existing Expected variable whose raw storage can be moved to TVMFFIAny.
#define TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Result)                                          \
  if (TVM_FFI_PREDICT_FALSE(::tvm::ffi::details::StructuralVisitNeedEarlyReturn(Result))) { \
    return ::tvm::ffi::details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(Result));         \
  }

// Return from the current ABI visit function if Result stops traversal.
// If Result is an Error, append Node to the visit error context before returning.
#define TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN_WITH_ERROR_CONTEXT(Result, Node)                    \
  if (TVM_FFI_PREDICT_FALSE(::tvm::ffi::details::StructuralVisitNeedEarlyReturn(Result))) {    \
    if (TVM_FFI_PREDICT_FALSE((Result).type_index() == ::tvm::ffi::TypeIndex::kTVMFFIError)) { \
      if ((Node).type_index() >= ::tvm::ffi::TypeIndex::kTVMFFIStaticObjectBegin) {            \
        ::tvm::ffi::Error _tvm_ffi_visit_err_ = (Result).error();                              \
        ::tvm::ffi::details::UpdateVisitErrorContext(_tvm_ffi_visit_err_,                      \
                                                     (Node).cast<::tvm::ffi::ObjectRef>());    \
      }                                                                                        \
    }                                                                                          \
    return ::tvm::ffi::details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(Result));            \
  }
/// \endcond

/*!
 * \brief Visitor used by callback-dispatched ``StructuralWalk``.
 *
 * \tparam order Callback placement relative to child traversal.
 * \tparam Dispatch Callable returning ``Expected<WalkResult>`` when invoked with ``AnyView`` and
 *                   the active def-region kind. User callbacks wrapped by this dispatcher may
 *                   accept either ``(value)`` or ``(value, def_region_kind)``.
 */
template <WalkOrder order, typename Dispatch>
class StructuralWalkCallbackVisitorObj : public StructuralVisitorObj {
 public:
  /*!
   * \brief Construct a structural walk visitor.
   * \param dispatch The composed dispatcher invoked on each visited node.
   */
  explicit StructuralWalkCallbackVisitorObj(Dispatch dispatch)
      : StructuralVisitorObj(VTable()), dispatch_(std::move(dispatch)) {}

 private:
  /*!
   * \brief Return the vtable used by this visitor.
   * \return Pointer to the static structural visitor vtable.
   */
  static const StructuralVisitorVTable* VTable() {
    static const StructuralVisitorVTable vtable{&StructuralWalkCallbackVisitorObj::DispatchVisit};
    return &vtable;
  }

  /*!
   * \brief Dispatch from the erased visitor pointer to the concrete walk visitor.
   * \param self The erased structural visitor object.
   * \param value The value to visit.
   * \return Interrupt state, or an error if traversal failed.
   */
  static TVMFFIAny DispatchVisit(StructuralVisitorObj* self, AnyView value) noexcept {
    return static_cast<StructuralWalkCallbackVisitorObj*>(self)->VisitImpl(value);
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
      auto result = dispatch_(value, this->def_region_kind());
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN_WITH_ERROR_CONTEXT(result, value);
      int32_t type_index = result.type_index();
      TVM_FFI_UNSAFE_ASSUME(type_index == TypeIndex::kTVMFFIInt);
      if (TVM_FFI_PREDICT_FALSE(details::ExpectedUnsafe::ValueAs<int32_t>(result) ==
                                WalkResult::kSkipTag)) {
        return details::ExpectedUnsafe::MoveToTVMFFIAny(
            Expected<Optional<VisitInterrupt>>(std::nullopt));
      }
    }

    auto interrupt = DefaultVisitExpected(value);
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN_WITH_ERROR_CONTEXT(interrupt, value);

    if constexpr (order == WalkOrder::kPostOrder) {
      auto result = dispatch_(value, this->def_region_kind());
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN_WITH_ERROR_CONTEXT(result, value);
    }

    return details::ExpectedUnsafe::MoveToTVMFFIAny(
        Expected<Optional<VisitInterrupt>>(std::nullopt));
  }

  /*! \brief Composed dispatch closure invoked once per visited node. */
  Dispatch dispatch_;
};

/*!
 * \brief Compose typed callbacks into a single per-node dispatcher.
 *
 * Each callback dispatches on its first parameter's type; callbacks are tested
 * in declaration order and the first match runs. Callbacks may take an optional
 * second ``TVMFFIDefRegionKind`` argument. Nodes that match no callback fall
 * through and traversal continues normally.
 */
struct StructuralWalkCallbackChain {
  /*!
   * \brief Build a dispatcher closure over a chain of typed callbacks.
   * \tparam Callbacks Callable types whose first parameter selects the dispatched
   *                   value type.
   * \param callbacks Callbacks to be tested in order.
   * \return A dispatcher closure of type ``Expected<WalkResult>(AnyView,
   *         TVMFFIDefRegionKind)``. Each user callback may take either
   *         ``(value)`` or ``(value, def_region_kind)``.
   */
  template <typename... Callbacks>
  static auto FromChain(Callbacks... callbacks) {
    return [=](AnyView x, TVMFFIDefRegionKind kind) mutable -> Expected<WalkResult> {
      try {
        Optional<Expected<WalkResult>> result;
        // Fold expression: each TryCallLink returns empty Optional on no-match
        // (falsy) or a result on match (truthy); || short-circuits on first match.
        (... || (result = TryCallLink(callbacks, x, kind)));
        if (result.has_value()) {
          return std::move(result).value();
        }
        return WalkResult::Advance();
      } catch (const Error& err) {
        return Unexpected(err);
      }
    };
  }

 private:
  /*!
   * \brief Invoke ``callback`` when ``x`` matches its first parameter type.
   * \tparam Callback Callable whose first parameter selects the value type and
   *                  whose optional second parameter receives the active def-region kind.
   * \param callback The callback under test.
   * \param x The value to dispatch on.
   * \param kind The active def-region kind.
   * \return The callback result if it matched, empty ``Optional`` otherwise.
   */
  template <typename Callback>
  static Optional<Expected<WalkResult>> TryCallLink(Callback& callback, AnyView x,
                                                    TVMFFIDefRegionKind kind) {
    using FuncInfo = FunctionInfo<std::decay_t<Callback>>;
    static_assert(FuncInfo::num_args == 1 || FuncInfo::num_args == 2,
                  "StructuralWalk callbacks must take one argument (value) or two arguments "
                  "(value, def-region kind)");
    using FirstArg = std::tuple_element_t<0, typename FuncInfo::ArgType>;
    using TSub = std::remove_cv_t<std::remove_reference_t<FirstArg>>;
    if constexpr (std::is_same_v<TSub, AnyView>) {
      // callback on AnyView
      return InvokeCallback(callback, x, kind);
    } else if constexpr (std::is_same_v<TSub, Any>) {
      // callback on Any
      return InvokeCallback(callback, Any(x), kind);
    } else {
      if (auto opt = x.template as<TSub>()) {
        return InvokeCallback(callback, *std::move(opt), kind);
      }
    }
    return std::nullopt;
  }

  /*!
   * \brief Invoke a matched callback with optional def-region context.
   * \tparam Callback Callable returning ``Expected<WalkResult>``.
   * \tparam Value Type of the converted value passed to the callback.
   * \param callback The matched callback to invoke.
   * \param value The converted value.
   * \param kind The active def-region kind.
   * \return The callback result.
   */
  template <typename Callback, typename Value>
  static Expected<WalkResult> InvokeCallback(Callback& callback, Value&& value,
                                             TVMFFIDefRegionKind kind) {
    using FuncInfo = FunctionInfo<std::decay_t<Callback>>;
    if constexpr (FuncInfo::num_args == 1) {
      return callback(std::forward<Value>(value));
    } else {
      return callback(std::forward<Value>(value), kind);
    }
  }
};

}  // namespace details

/*!
 * \brief Walk a structured value graph and invoke typed callbacks on selected values.
 *
 * The callbacks are invoked only for values matching the first argument type of
 * one of the callbacks. The first callback argument may be ``AnyView``, ``Any``,
 * an object reference type, an object pointer type, or another FFI-convertible
 * POD type. A callback may also optionally take a second ``TVMFFIDefRegionKind`` argument
 * to inspect whether the value is being visited in a definition region.
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
 * Example:
 *
 * \code
 * int num_adds = 0;
 *
 * Expected<Optional<VisitInterrupt>> result = StructuralWalkExpected<WalkOrder::kPreOrder>(
 *     root,
 *     [&](const Add& add) -> Expected<WalkResult> {
 *       ++num_adds;
 *       return WalkResult::Advance();
 *     },
 *     [&](const Mul& mul) -> Expected<WalkResult> {
 *       return WalkResult::Skip();
 *     });
 * \endcode
 *
 * \tparam order Whether to invoke the callback before or after visiting children.
 * \tparam Callbacks Callback types.
 * \param root The root value to visit.
 * \param callbacks Callbacks invoked for matching nodes. Each callback may take
 *                  either ``(value)`` or ``(value, def_region_kind)`` and should return
 *                  ``Expected<WalkResult>``.
 * \return ``std::nullopt`` if traversal completed, or the interrupt returned by
 *         a callback.
 *
 * \note Return type of each callback should be ``Expected<WalkResult>``.
 */
template <WalkOrder order, typename... Callbacks>
Expected<Optional<VisitInterrupt>> StructuralWalkExpected(AnyView root,
                                                          Callbacks&&... callbacks) noexcept {
  static_assert(sizeof...(Callbacks) != 0, "StructuralWalk requires at least one callback");
  auto dispatch =
      details::StructuralWalkCallbackChain::FromChain(std::forward<Callbacks>(callbacks)...);
  using Visitor = details::StructuralWalkCallbackVisitorObj<order, decltype(dispatch)>;
  StructuralVisitor visitor(make_object<Visitor>(std::move(dispatch)));
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
 * \param callbacks Callbacks invoked for matching nodes. Each callback may take
 *                  either ``(value)`` or ``(value, def_region_kind)`` and should return
 *                  ``Expected<WalkResult>``.
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

}  // namespace ffi
}  // namespace tvm
#endif  // TVM_FFI_EXTRA_STRUCTURAL_VISIT_H_
