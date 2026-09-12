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
 * \file tvm/ffi/extra/structural_mutate.h
 * \brief Structural mutation API with optional in-place optimization.
 */
#ifndef TVM_FFI_EXTRA_STRUCTURAL_MUTATE_H_
#define TVM_FFI_EXTRA_STRUCTURAL_MUTATE_H_

#include <tvm/ffi/any.h>
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/cast.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/tuple.h>
#include <tvm/ffi/container/variant.h>
#include <tvm/ffi/expected.h>
#include <tvm/ffi/extra/structural_visit.h>
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
#include <unordered_map>
#include <utility>

namespace tvm {
namespace ffi {

class StructuralMutatorObj;
template <typename T>
class UnchangedOr;

/// \cond Doxygen_Suppress
/*! \brief Whether an UnchangedOr replacement can reuse another replacement's storage. */
template <typename T, typename U>
inline constexpr bool type_subsumes_v<UnchangedOr<T>, UnchangedOr<U>> = type_subsumes_v<T, U>;
/// \endcond

template <typename Parent, WalkOrder order, typename... Callbacks>
class StructuralMapEngine;
template <typename Parent, WalkOrder order>
class StructuralMapDynEngine;
template <typename Parent, typename... Callbacks>
class StructuralMutateEngine;

/*!
 * \brief ABI callback type for structural mutation.
 *
 * \param mutator The active structural mutator.
 * \param value The borrowed value to mutate.
 * \return Raw ``TVMFFIAny`` containing a replacement, the unchanged marker, or an Error.
 *
 * \note The hook is exception-free. Representable failures must be returned as an Error. Hook
 *       implementations should use non-throwing accessors when the engine's type dispatch has
 *       already established the type; allocation failure and violated container invariants
 *       remain fatal.
 */
using FStructuralMutate = TVMFFIAny (*)(StructuralMutatorObj* mutator, AnyView value) noexcept;

/*!
 * \brief ABI callback type for looking up an identity substitution.
 *
 * \param mutator The active structural mutator.
 * \param var The borrowed variable identity to look up.
 * \return Raw ``TVMFFIAny`` containing the owning mapped value, FFI None when no mapping exists,
 *         or an Error.
 */
using FStructuralVarRemapGet = TVMFFIAny (*)(StructuralMutatorObj* mutator, AnyView var) noexcept;

/*!
 * \brief ABI callback type for recording an identity substitution.
 *
 * \param mutator The active structural mutator.
 * \param var The borrowed variable identity to bind.
 * \param mapped_value The borrowed replacement value.
 * \return Raw ``TVMFFIAny`` containing FFI None on success or an Error.
 */
using FStructuralVarRemapSet = TVMFFIAny (*)(StructuralMutatorObj* mutator, AnyView var,
                                             AnyView mapped_value) noexcept;

namespace details {

// Copy and structurally mutate the reflected fields of an object-backed value.
TVM_FFI_INLINE static Expected<Any> MutateReflectedFieldsExpected(StructuralMutatorObj* mutator,
                                                                  AnyView value) noexcept;

}  // namespace details

/*!
 * \brief VTable ABI for \ref StructuralMutator dispatch.
 */
struct StructuralMutatorVTable {
  /*!
   * \brief Mutate a value without modifying the source in place.
   *
   * \param mutator The active structural mutator.
   * \param value The borrowed value to mutate.
   * \return Raw ``TVMFFIAny`` carrying a replacement, the unchanged marker, or Error.
   */
  FStructuralMutate mutate = nullptr;
  /*!
   * \brief Mutate a value, permitting an in-place implementation when it is safe.
   *
   * \param mutator The active structural mutator.
   * \param value The borrowed value to mutate.
   * \return Raw ``TVMFFIAny`` carrying a replacement, the unchanged marker, or Error.
   *
   * The returned value may refer to the same object as \p value when the implementation mutates
   * that object in place.
   */
  FStructuralMutate maybe_inplace_mutate = nullptr;
  /*!
   * \brief Look up the replacement for a variable identity.
   *
   * \param mutator The active structural mutator.
   * \param var The borrowed variable identity to look up.
   * \return Raw ``TVMFFIAny`` carrying the owning replacement, FFI None on a miss, or Error.
   */
  FStructuralVarRemapGet var_remap_get = nullptr;
  /*!
   * \brief Record the replacement for a variable identity.
   *
   * \param mutator The active structural mutator.
   * \param var The borrowed variable identity to bind.
   * \param mapped_value The borrowed replacement value.
   * \return Raw ``TVMFFIAny`` carrying None or Error.
   */
  FStructuralVarRemapSet var_remap_set = nullptr;
};

namespace details {
template <typename Parent>
class StructuralMutateDynEngine;

struct UnchangedOrUnsafe;
}  // namespace details

/*! \brief Tag for a mutation result that produced no new value. */
struct Unchanged {
  /*!
   * \brief Copy this tag to its raw marker representation.
   * \return Raw ``TVMFFIAny`` carrying the reserved unchanged type index.
   */
  TVM_FFI_INLINE TVMFFIAny CopyToTVMFFIAny() const noexcept {
    // The marker needs a reserved type index because every ordinary index is a legal mutation
    // result. In particular, kTVMFFINone is a valid replacement and cannot double as the marker.
    TVMFFIAny raw;
    raw.type_index = TypeIndex::kTVMFFIUnchanged;
    // invariance: always set the union padding part to 0
    raw.zero_padding = 0;
    raw.v_int64 = 0;
    return raw;
  }

  /*!
   * \brief Convert this tag to its owning marker representation.
   * \return An owning ``Any`` carrying the reserved unchanged type index.
   */
  // NOLINTNEXTLINE(google-explicit-constructor,runtime/explicit)
  TVM_FFI_INLINE operator Any() const noexcept {
    TVMFFIAny raw = CopyToTVMFFIAny();
    return details::AnyUnsafe::MoveTVMFFIAnyRawToAny(raw);
  }
};

/*!
 * \brief A structural-mutation result containing a replacement or no new value.
 *
 * \tparam T The replacement value type.
 * \note ``UnchangedOr`` is deliberately designed to only have rvalue-qualified value accessors,
 *       so the compiler forces a value to leave the container exactly once, via a move.
 *
 * \code{.cpp}
 * // resolves to the original when the descent reported unchanged
 * copy->a = std::move(a).ValueOrUnchanged(std::move(copy->a));
 * // already known to be changed, so no original is needed
 * copy->b = std::move(b).ValueUnchecked();
 * \endcode
 */
template <typename T>
class UnchangedOr {
 public:
  static_assert(!std::is_base_of_v<Error, std::remove_cv_t<T>>,
                "UnchangedOr<Error> is not supported");

  /*!
   * \brief Construct an unchanged result from its tag.
   * \param unchanged The unchanged tag.
   */
  // NOLINTNEXTLINE(google-explicit-constructor,runtime/explicit)
  TVM_FFI_INLINE UnchangedOr(Unchanged unchanged) noexcept : data_(static_cast<Any>(unchanged)) {}

  /*!
   * \brief Construct a changed result from a replacement value.
   * \param value The replacement value.
   */
  // NOLINTNEXTLINE(google-explicit-constructor,runtime/explicit)
  TVM_FFI_INLINE UnchangedOr(T value) : data_(Any(std::move(value))) {}

  /*!
   * \brief Implicit converting constructor from another replacement type.
   * \tparam U Source replacement type whose storage is subsumed by or implicitly convertible to T.
   * \param other The result to convert, copied from an lvalue or moved from an rvalue.
   */
  template <typename U,
            typename = std::enable_if_t<type_subsumes_v<T, U> || std::is_convertible_v<U, T>>>
  // NOLINTNEXTLINE(google-explicit-constructor,runtime/explicit)
  TVM_FFI_INLINE UnchangedOr(UnchangedOr<U> other)
      : data_([&other]() {
          if constexpr (type_subsumes_v<T, U>) {
            // Reuse materialized storage, including the unchanged marker.
            return details::AnyUnsafe::MoveTVMFFIAnyRawToAny(
                details::AnyUnsafe::MoveAnyToTVMFFIAny(std::move(other.data_)));
          } else {
            return other.IsUnchanged() ? std::move(other.data_)
                                       : Any(T(std::move(other).ValueUnchecked()));
          }
        }()) {}

  /// \cond Doxygen_Suppress
  TVM_FFI_INLINE UnchangedOr(const UnchangedOr&) = default;
  TVM_FFI_INLINE UnchangedOr(UnchangedOr&&) noexcept = default;
  /// \endcond
  TVM_FFI_INLINE ~UnchangedOr() = default;
  TVM_FFI_INLINE UnchangedOr& operator=(const UnchangedOr&) = default;
  TVM_FFI_INLINE UnchangedOr& operator=(UnchangedOr&&) noexcept = default;

  /*!
   * \brief Whether this result asks the caller to preserve the original value.
   * \return Whether the result is unchanged.
   */
  TVM_FFI_INLINE bool IsUnchanged() const& noexcept {
    return data_.type_index() == TypeIndex::kTVMFFIUnchanged;
  }

  /*!
   * \brief Whether this result is unchanged or contains the original object identity.
   * \param original The original value.
   * \return Whether the original identity may be reused.
   */
  TVM_FFI_INLINE bool UnchangedOrSameAs(const T& original) const& noexcept {
    return IsUnchanged() || data_.same_as(original);
  }

  /*!
   * \brief Move the replacement, or move \p original when unchanged.
   * \param original The owned original value.
   * \return The replacement or original value.
   * \note Passing a named lvalue transfers ownership and may leave it moved-from.
   */
  TVM_FFI_INLINE T ValueOrUnchanged(T& original) && {
    return IsUnchanged() ? std::move(original)
                         : details::AnyUnsafe::MoveFromAnyAfterCheck<T>(std::move(data_));
  }

  /*!
   * \brief Move the replacement, or move \p original when unchanged.
   * \param original The owned original value.
   * \return The replacement or original value.
   */
  TVM_FFI_INLINE T ValueOrUnchanged(T&& original) && {
    return IsUnchanged() ? std::move(original)
                         : details::AnyUnsafe::MoveFromAnyAfterCheck<T>(std::move(data_));
  }

  /*!
   * \brief Move the replacement, or materialize \p original when unchanged.
   * \tparam U The replacement type, constrained to ``Any``.
   * \param original The borrowed original value.
   * \return The replacement or original value.
   */
  template <typename U = T,
            typename = std::enable_if_t<std::is_same_v<T, Any> && std::is_same_v<U, T>>>
  TVM_FFI_INLINE Any ValueOrUnchanged(AnyView original) && {
    return IsUnchanged() ? Any(original)
                         : details::AnyUnsafe::MoveFromAnyAfterCheck<Any>(std::move(data_));
  }

  /*!
   * \brief Move the known-changed replacement without checking its state.
   * \return The replacement value.
   * \pre The result is not unchanged.
   */
  TVM_FFI_INLINE T ValueUnchecked() && {
    return details::AnyUnsafe::MoveFromAnyAfterCheck<T>(std::move(data_));
  }

 private:
  template <typename>
  friend class UnchangedOr;
  friend struct details::UnchangedOrUnsafe;
  template <typename, typename>
  friend struct TypeTraits;
  struct UnsafeInit {};
  TVM_FFI_INLINE explicit UnchangedOr(UnsafeInit, Any data) noexcept : data_(std::move(data)) {}
  Any data_;
};

namespace details {
/*! \brief Unsafe moves between UnchangedOr and its single Any storage. */
struct UnchangedOrUnsafe {
  template <typename T>
  TVM_FFI_INLINE static TVMFFIAny MoveToTVMFFIAny(UnchangedOr<T>&& result) noexcept {
    return AnyUnsafe::MoveAnyToTVMFFIAny(std::move(result.data_));
  }
};

}  // namespace details

namespace details {
// Out of line so its strings and Error construction stay out of the hot path of whatever hook
// body TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN expands into. Same reason as
// BadStructuralMutateHookError.
// Takes nothing on purpose. Naming the offending type in the message would keep the result live
// across the predicted-not-taken guard in the hot path. The declared type is already present in
// the source line to which the diagnostic points.
TVM_FFI_COLD_CODE inline UnexpectedReturnHelper SMutateDeclaredTypeError() noexcept {
  return UnexpectedReturnHelper(Unexpected(
      Error("TypeError", "structural mutate result does not match the declared type", "")));
}
}  // namespace details

/*!
 * \brief Object node of a structural mutator.
 */
class StructuralMutatorObj : public Object {
 public:
  /*! \brief Callback-facing mutator type used by composed callback-driven engines. */
  using MutatorObjType = StructuralMutatorObj;

  /*!
   * \brief Mutate a value through the mutator vtable.
   *
   * \param value The value to mutate.
   * \tparam T The declared replacement type.
   * \return The replacement or unchanged marker.
   * \throws Error if mutation fails.
   *
   * This entry point never intentionally mutates \p value in place. Recursive mutations
   * also use \ref Mutate.
   *
   * \code{.cpp}
   * Expr new_node = mutator->Mutate<Expr>(node).ValueOrUnchanged(std::move(node));
   * \endcode
   */
  template <typename T = Any>
  TVM_FFI_INLINE UnchangedOr<T> Mutate(AnyView value) {
    return std::move(MutateExpected<T>(value)).value();
  }

  /*!
   * \brief Exception-free form of \ref Mutate.
   *
   * \param value The value to mutate.
   * \tparam T The declared replacement type.
   * \return The replacement or unchanged marker, or an Error if mutation failed.
   */
  template <typename T = Any>
  TVM_FFI_INLINE Expected<UnchangedOr<T>> MutateExpected(AnyView value) noexcept {
    if constexpr (std::is_same_v<T, Any>) {
      return details::ExpectedUnsafe::MoveFromTVMFFIAny<UnchangedOr<T>>(
          (*vtable_->mutate)(this, value));
    } else {
      TVMFFIAny result = (*vtable_->mutate)(this, value);
      if (TVM_FFI_PREDICT_FALSE(!TypeTraits<Expected<UnchangedOr<T>>>::CheckAnyStrict(&result))) {
        (void)details::AnyUnsafe::MoveTVMFFIAnyRawToAny(result);
        return details::SMutateDeclaredTypeError();
      }
      return details::ExpectedUnsafe::MoveFromTVMFFIAny<UnchangedOr<T>>(result);
    }
  }

  /*!
   * \brief Mutate a value, permitting an in-place implementation when it is safe.
   *
   * \param value The borrowed value to mutate.
   * \tparam T The declared replacement type.
   * \return The replacement or unchanged marker.
   * \throws Error if mutation fails.
   *
   * The returned value may refer to the same object as \p value. Callers must use the return value
   * as the result of the mutation rather than assuming that the input object was reused.
   */
  template <typename T = Any>
  TVM_FFI_INLINE UnchangedOr<T> MaybeInplaceMutate(AnyView value) {
    return std::move(MaybeInplaceMutateExpected<T>(value)).value();
  }

  /*!
   * \brief Exception-free form of \ref MaybeInplaceMutate.
   *
   * \param value The borrowed value to mutate.
   * \tparam T The declared replacement type.
   * \return The replacement or unchanged marker, or an Error if mutation failed.
   *
   * \note Call only from a ``__s_maybe_inplace_mutate__`` hook, which is dispatched
   *       only for a value whose entire path from the root is uniquely owned.
   */
  template <typename T = Any>
  TVM_FFI_INLINE Expected<UnchangedOr<T>> MaybeInplaceMutateExpected(AnyView value) noexcept {
    if constexpr (std::is_same_v<T, Any>) {
      return details::ExpectedUnsafe::MoveFromTVMFFIAny<UnchangedOr<T>>(
          (*vtable_->maybe_inplace_mutate)(this, value));
    } else {
      TVMFFIAny result = (*vtable_->maybe_inplace_mutate)(this, value);
      if (TVM_FFI_PREDICT_FALSE(!TypeTraits<Expected<UnchangedOr<T>>>::CheckAnyStrict(&result))) {
        (void)details::AnyUnsafe::MoveTVMFFIAnyRawToAny(result);
        return details::SMutateDeclaredTypeError();
      }
      return details::ExpectedUnsafe::MoveFromTVMFFIAny<UnchangedOr<T>>(result);
    }
  }

  /*!
   * \brief Mutate a value, using in-place mutation only for a uniquely owned object
   *        and \p allow_inplace set to true.
   * \tparam T The declared replacement type.
   * \param value The borrowed value to mutate.
   * \param allow_inplace Whether in-place mutation is permitted. Defaults to true.
   *        If false, use ordinary mutation without checking uniqueness.
   * \return The replacement or unchanged marker, or an Error if mutation failed.
   *
   * \note When \p allow_inplace is true, the caller must already know the entire
   *       path from the root is uniquely owned, either through an owning moved-in
   *       root or while handling a ``__s_maybe_inplace_mutate__`` hook. This method
   *       checks only \p value itself, not its ancestors.
   */
  template <typename T = Any>
  TVM_FFI_INLINE Expected<UnchangedOr<T>> MaybeInplaceMutateIfUniqueExpected(
      AnyView value, bool allow_inplace = true) noexcept {
    const Object* obj = value.as<Object>();
    if (allow_inplace && obj != nullptr && obj->unique()) {
      return MaybeInplaceMutateExpected<T>(value);
    }
    return MutateExpected<T>(value);
  }

  /*!
   * \brief Apply the default structural mutation with copy-on-write behavior.
   *
   * \param value The value to mutate.
   * \return The replacement or unchanged marker, or an Error if hook dispatch, copying, or field
   *         mutation failed.
   *
   * \note A registered ``__s_mutate__`` hook is dispatched before the reflected fallback. A
   *       FreeVar hook owns the definition-only remap policy for that type; the reflected fallback
   *       applies the same policy automatically.
   */

  TVM_FFI_INLINE Expected<UnchangedOr<Any>> DefaultMutateExpected(AnyView value) noexcept {
    return details::ExpectedUnsafe::MoveFromTVMFFIAny<UnchangedOr<Any>>(DefaultMutateRaw(value));
  }

  /*!
   * \brief Apply custom maybe-in-place mutation, or fall back to non-in-place mutation.
   *
   * \param value The borrowed value to mutate.
   * \return The replacement or unchanged marker, or an Error if mutation failed. In-place
   *         changes completed before an Error are not rolled back.
   *
   * \note In-place mutation is explicitly opt-in. A registered
   *       ``__s_maybe_inplace_mutate__`` hook may rely on its input being safe to mutate and owns
   *       any variable-remap handling. When the hook is absent, this method calls
   *       \ref DefaultMutateExpected.
   */
  TVM_FFI_INLINE Expected<UnchangedOr<Any>> DefaultMaybeInplaceMutateExpected(
      AnyView value) noexcept {
    return details::ExpectedUnsafe::MoveFromTVMFFIAny<UnchangedOr<Any>>(
        DefaultMaybeInplaceMutateRaw(value));
  }

  /*!
   * \brief Look up the replacement recorded for a variable identity.
   *
   * \param var The borrowed variable identity to look up.
   * \return The owning replacement, FFI None if no replacement exists, or an Error if lookup
   *         fails.
   *
   * \note The identity must have ``kTVMFFISEqHashKindFreeVar`` or
   *       ``kTVMFFISEqHashKindDAGNode`` structural-equality metadata.
   */
  TVM_FFI_INLINE Expected<Any> VarRemapGetExpected(AnyView var) noexcept {
    return details::ExpectedUnsafe::MoveFromTVMFFIAny<Any>((*vtable_->var_remap_get)(this, var));
  }

  /*!
   * \brief Record the replacement for a variable identity.
   *
   * \param var The borrowed variable identity to bind.
   * \param mapped_value The borrowed replacement value.
   * \return Successful completion, or an Error if the binding is invalid or cannot be stored.
   *
   * \note The identity must have ``kTVMFFISEqHashKindFreeVar`` or
   *       ``kTVMFFISEqHashKindDAGNode`` structural-equality metadata.
   */
  TVM_FFI_INLINE Expected<void> VarRemapSetExpected(AnyView var, AnyView mapped_value) noexcept {
    return details::ExpectedUnsafe::MoveFromTVMFFIAny<void>(
        (*vtable_->var_remap_set)(this, var, mapped_value));
  }

  /*!
   * \brief Return the current def-region context.
   * \return The active def-region kind.
   * \note A custom mutate hook for a FreeVar type must apply the simple-def clamp itself: when
   *       this is kTVMFFIDefRegionKindSimple, descend the variable's type under
   *       kTVMFFIDefRegionKindNone. The reflected walk does this on its own.
   */
  TVM_FFI_INLINE TVMFFIDefRegionKind def_region_kind() const { return def_region_mode_; }

  /*!
   * \brief Temporarily switch the def-region context while invoking \p callback.
   *
   * \param kind The def-region kind to set during the callback.
   * \param callback A nullary callable that performs recursive mutation.
   * \return The value returned by \p callback.
   * \note Inside a pattern region this is a no-op: the pattern propagates, so \p kind is
   *       ignored and the callback runs under the pattern.
   */
  template <typename Callback>
  TVM_FFI_INLINE auto WithDefRegionKind(TVMFFIDefRegionKind kind, Callback&& callback)
      -> decltype(std::forward<Callback>(callback)()) {
    // Precedence: a pattern region propagates; entering any kind inside it has no effect.
    if (def_region_mode_ == kTVMFFIDefRegionKindPattern) {
      return std::forward<Callback>(callback)();
    }
    class Scope {
     public:
      Scope(StructuralMutatorObj* mutator, TVMFFIDefRegionKind kind)
          : mutator_(mutator), old_kind_(mutator->def_region_mode_) {
        mutator_->def_region_mode_ = kind;
      }
      ~Scope() { mutator_->def_region_mode_ = old_kind_; }
      Scope(const Scope&) = delete;
      Scope& operator=(const Scope&) = delete;

     private:
      StructuralMutatorObj* mutator_;
      TVMFFIDefRegionKind old_kind_;
    };
    Scope scope(this, kind);
    return std::forward<Callback>(callback)();
  }

  /// \cond Doxygen_Suppress
  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("ffi.StructuralMutator", StructuralMutatorObj, Object);
  /// \endcond

 private:
  template <typename Parent>
  friend class details::StructuralMutateDynEngine;

  /*!
   * \brief Diagnostic for a malformed ``__s_mutate__`` registration.
   *
   * Kept out of line and cold: it can only fire for a type whose registered attribute is
   * neither an opaque function pointer nor an ffi.Function, so it is unreachable for any
   * correctly registered type. Inlined, its three string literals and Error construction
   * land in the traversal's hot path for no reason.
   */
  TVM_FFI_COLD_CODE static Expected<Any> BadStructuralMutateHookError() noexcept {
    return Unexpected(
        Error("TypeError", "__s_mutate__ must be an opaque function pointer or ffi.Function", ""));
  }

  // Convention: the ABI boundary is a raw TVMFFIAny; mutation results inside a callback or hook
  // body use Expected<Any> and move out to TVMFFIAny at that boundary. Unchanged converts to an
  // Any carrying kTVMFFIUnchanged.
  //
  // The Raw forms below exist because that boundary is also the default path. A hook is a C-ABI
  // function pointer returning TVMFFIAny, a 16-byte POD that stays in registers; wrapping the
  // result in Expected<Any> would force it to memory because the C++ wrapper is not
  // trivially destructible and is therefore classified MEMORY. Descent through an unmatched node
  // calls a hook and returns its result unchanged, so keeping that path raw removes the round trip
  // entirely. Only a matched callback pays for the Expected wrapper.
  //
  // Engine-internal: subclasses call the Expected forms above.
  /*! \brief Raw default mutation: attr lookup then hook, favouring the fn-ptr case. */
  TVM_FFI_INLINE TVMFFIAny DefaultMutateRaw(AnyView value) noexcept {
    static reflection::TypeAttrColumn column(reflection::type_attr::kStructuralMutate);
    AnyView attr = column[value.type_index()];
    // Exactly one frame per node: hooks propagate errors untouched, and this is the engine
    // dispatching into `value`, so both exits below name it here and nowhere else.
    TVMFFIAny result;
    if (TVM_FFI_PREDICT_TRUE(attr.type_index() == TypeIndex::kTVMFFIOpaquePtr)) {
      result = (*reinterpret_cast<FStructuralMutate>(attr.cast<void*>()))(this, value);
    } else {
      result = DefaultMutateRawTail(value, attr);
    }
    if (TVM_FFI_PREDICT_FALSE(result.type_index == TypeIndex::kTVMFFIError)) {
      return AttachVisitErrorContextRaw(result, value);
    }
    return result;
  }

  /*! \brief Keep address-taking for error decoration off the successful raw-result path. */
  TVM_FFI_COLD_CODE static TVMFFIAny AttachVisitErrorContextRaw(TVMFFIAny result,
                                                                AnyView value) noexcept {
    details::UpdateVisitErrorContext(result, value);
    return result;
  }

  /*! \brief The cold remainder of DefaultMutateRaw: an ffi.Function hook, or no hook at all. */
  TVMFFIAny DefaultMutateRawTail(AnyView value, AnyView attr) noexcept {
    if (attr.type_index() != TypeIndex::kTVMFFINone) {
      // Registered, but as an ffi.Function rather than an opaque pointer.
      if (attr.type_index() == TypeIndex::kTVMFFIFunction) {
        return details::ExpectedUnsafe::MoveToTVMFFIAny(
            attr.cast<Function>().CallExpected<Any>(this, value));
      }
      // Registered as neither: a malformed hook.
      return details::ExpectedUnsafe::MoveToTVMFFIAny(BadStructuralMutateHookError());
    }
    // No hook at all. A POD carries through unchanged; an object walks its reflected fields.
    if (value.type_index() < TypeIndex::kTVMFFIStaticObjectBegin) {
      return details::AnyUnsafe::MoveAnyToTVMFFIAny(Any(value));
    }
    const TVMFFITypeInfo* type_info = TVMFFIGetTypeInfo(value.type_index());
    const int32_t identity_kind = type_info->metadata == nullptr
                                      ? kTVMFFISEqHashKindUnsupported
                                      : type_info->metadata->structural_eq_hash_kind;
    const bool is_free_var = identity_kind == kTVMFFISEqHashKindFreeVar;
    const bool is_dag_node = identity_kind == kTVMFFISEqHashKindDAGNode;
    if (is_free_var || is_dag_node) {
      // Only None means no cached descent result; every other value, including the unchanged
      // marker used by a pattern definition, is returned directly.
      Expected<Any> mapped = VarRemapGetExpected(value);
      if (details::ExpectedUnsafe::GetData(mapped).type_index() != TypeIndex::kTVMFFINone) {
        return details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(mapped));
      }
    }

    // A FreeVar outside a definition region is a use. A miss means its definition was unchanged
    // (or it is free), so there is no field descent and no remap insertion.
    if (is_free_var && def_region_kind() == kTVMFFIDefRegionKindNone) {
      return Unchanged().CopyToTVMFFIAny();
    }

    Expected<Any> result = details::MutateReflectedFieldsExpected(this, value);
    if (TVM_FFI_PREDICT_FALSE(result.is_err())) {
      return details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(result));
    }
    if (is_free_var || is_dag_node) {
      const Any& result_value = details::ExpectedUnsafe::GetData(result);
      // Bind the descent result. The one exception is an unchanged simple definition: its
      // uses resolve to the var itself on a miss, so there is nothing to record.
      if (is_dag_node || def_region_kind() == kTVMFFIDefRegionKindPattern ||
          result_value.type_index() != TypeIndex::kTVMFFIUnchanged) {
        Expected<void> set_result = VarRemapSetExpected(value, result_value);
        if (TVM_FFI_PREDICT_FALSE(set_result.is_err())) {
          return details::ExpectedUnsafe::MoveToTVMFFIAny(
              Expected<Any>(Unexpected(std::move(set_result).error())));
        }
      }
    }
    return details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(result));
  }
  /*!
   * \brief Raw default maybe-in-place mutation.
   *
   * \note A registered opaque hook is the expected case here too, so the attribute is read once
   *       and every other shape is handed to the out-of-line remainder.
   */
  TVM_FFI_INLINE TVMFFIAny DefaultMaybeInplaceMutateRaw(AnyView value) noexcept {
    static reflection::TypeAttrColumn column(reflection::type_attr::kStructuralMaybeInplaceMutate);
    AnyView attr = column[value.type_index()];
    if (TVM_FFI_PREDICT_TRUE(attr.type_index() == TypeIndex::kTVMFFIOpaquePtr)) {
      // This is the engine dispatching into `value`; hooks propagate errors untouched, so the
      // node is named here. The fall-through re-dispatches the same node through
      // DefaultMutateRaw, which names it there instead -- exactly one frame either way.
      TVMFFIAny result = (*reinterpret_cast<FStructuralMutate>(attr.cast<void*>()))(this, value);
      if (TVM_FFI_PREDICT_FALSE(result.type_index == TypeIndex::kTVMFFIError)) {
        return AttachVisitErrorContextRaw(result, value);
      }
      return result;
    }
    return DefaultMaybeInplaceMutateRawTail(value, attr);
  }

  /*!
   * \brief The cold remainder of DefaultMaybeInplaceMutateRaw: an ffi.Function in-place hook, or
   *        no in-place hook at all, in which case the ordinary mutate path runs.
   */
  TVMFFIAny DefaultMaybeInplaceMutateRawTail(AnyView value, AnyView attr) noexcept {
    if (attr.type_index() == TypeIndex::kTVMFFIFunction) {
      TVMFFIAny result = details::ExpectedUnsafe::MoveToTVMFFIAny(
          attr.cast<Function>().CallExpected<Any>(this, value));
      if (TVM_FFI_PREDICT_FALSE(result.type_index == TypeIndex::kTVMFFIError)) {
        return AttachVisitErrorContextRaw(result, value);
      }
      return result;
    }
    return details::ExpectedUnsafe::MoveToTVMFFIAny(DefaultMutateExpected(value));
  }

 protected:
  /*!
   * \brief Construct a structural mutator from an immutable dispatch vtable.
   * \param vtable The non-null dispatch table for this mutator. It must outlive this object.
   */
  explicit StructuralMutatorObj(const StructuralMutatorVTable* vtable) : vtable_(vtable) {}

  /*!
   * \brief Non-owning pointer to the required ABI dispatch table.
   */
  const StructuralMutatorVTable* vtable_ = nullptr;

  /*!
   * \brief Current def-region context for def-region-aware structural mutation.
   */
  TVMFFIDefRegionKind def_region_mode_ = kTVMFFIDefRegionKindNone;
};

/*!
 * \brief ObjectRef wrapper for \ref StructuralMutatorObj.
 *
 * \sa StructuralMutatorObj
 */
class StructuralMutator : public ObjectRef {
 public:
  /*!
   * \brief Construct from an existing mutator object pointer.
   * \param n The object pointer to wrap.
   */
  explicit StructuralMutator(ObjectPtr<StructuralMutatorObj> n) : ObjectRef(std::move(n)) {}

  /// \cond Doxygen_Suppress
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(StructuralMutator, ObjectRef, StructuralMutatorObj);
  /// \endcond
};

namespace details {

/*!
 * \brief Mutate the reflected structural fields of an object-backed value.
 *
 * \param mutator The active structural mutator.
 * \param value The object-backed value to mutate.
 * \return The original value when no field changes, a mutated shallow copy otherwise, or an
 *         Error if copying or mutation failed.
 */
TVM_FFI_INLINE static Expected<Any> MutateReflectedFieldsExpected(StructuralMutatorObj* mutator,
                                                                  AnyView value) noexcept {
  const Object* obj = value.as<Object>();
  int32_t type_index = obj->type_index();

  static reflection::TypeAttrColumn column(reflection::type_attr::kShallowCopy);
  AnyView attr = column[type_index];
  if (TVM_FFI_PREDICT_FALSE(attr.type_index() != TypeIndex::kTVMFFIFunction)) {
    return Unexpected(Error("TypeError", "__ffi_shallow_copy__ must be an ffi.Function", ""));
  }

  Expected<Any> result = attr.cast<Function>().CallExpected<Any>(value);
  if (TVM_FFI_PREDICT_FALSE(result.is_err())) {
    return result;
  }

  const Any& result_value = details::ExpectedUnsafe::GetData(result);
  Object* new_obj = const_cast<Object*>(result_value.as<Object>());
  // Copy-on-write mutation requires a distinct target so partial updates cannot modify the source.
  if (TVM_FFI_PREDICT_FALSE(new_obj == nullptr || result.type_index() != value.type_index() ||
                            new_obj == obj)) {
    return Unexpected(Error(
        "TypeError",
        "Shallow copy callback must return a distinct object with the same type as its input", ""));
  }

  const TVMFFITypeInfo* type_info = TVMFFIGetTypeInfo(new_obj->type_index());
  bool field_changed = false;
  auto mutate_fields = [&]() {
    reflection::ForEachFieldInfoWithEarlyStop(
        type_info, [&](const TVMFFIFieldInfo* field_info) -> bool {
          if (field_info->flags & kTVMFFIFieldFlagBitMaskSEqHashIgnore) {
            return false;
          }

          Any field_value;
          void* field_addr = reinterpret_cast<char*>(new_obj) + field_info->offset;
          int ret_code = field_info->getter(field_addr, reinterpret_cast<TVMFFIAny*>(&field_value));
          if (TVM_FFI_PREDICT_FALSE(ret_code != 0)) {
            result = Unexpected(details::MoveFromSafeCallRaised());
            return true;
          }

          // Reflected fields use the same unchanged-or-value descent protocol.
          Expected<UnchangedOr<Any>> mutated_field = [&]() -> Expected<UnchangedOr<Any>> {
            if (field_info->flags & kTVMFFIFieldFlagBitMaskSEqHashDefSimple) {
              return mutator->WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&]() {
                return mutator->MutateExpected(field_value);
              });
            } else if (field_info->flags & kTVMFFIFieldFlagBitMaskSEqHashDefPattern) {
              return mutator->WithDefRegionKind(kTVMFFIDefRegionKindPattern, [&]() {
                return mutator->MutateExpected(field_value);
              });
            } else {
              return mutator->MutateExpected(field_value);
            }
          }();
          if (TVM_FFI_PREDICT_FALSE(mutated_field.is_err())) {
            result = Unexpected(std::move(mutated_field).error());
            return true;
          }
          const Any& mutated_field_data = details::ExpectedUnsafe::GetData(mutated_field);
          // Unchanged first: it is the common case, and it is one type-index test where the
          // resolved form ran a full same_as against a value it had just been handed back.
          if (mutated_field_data.type_index() == TypeIndex::kTVMFFIUnchanged ||
              field_value.same_as(mutated_field_data)) {
            return false;
          }

          if (TVM_FFI_PREDICT_FALSE(field_info->setter == nullptr)) {
            result = Unexpected(Error(
                "TypeError",
                "Cannot structurally mutate field `" +
                    std::string(field_info->name.data, field_info->name.size) + "` of type `" +
                    std::string(type_info->type_key.data, type_info->type_key.size) +
                    "` because it does not define a setter",
                ""));
            return true;
          }

          ret_code = reflection::CallFieldSetter(
              field_info, field_addr, reinterpret_cast<const TVMFFIAny*>(&mutated_field_data));
          if (TVM_FFI_PREDICT_FALSE(ret_code != 0)) {
            result = Unexpected(details::MoveFromSafeCallRaised());
            return true;
          }
          field_changed = true;
          return false;
        });
  };

  // A simple definition applies to the FreeVar itself, but its fields are uses. The
  // complete field traversal are clamped to None, then the definition region is restored.
  if (mutator->def_region_kind() == kTVMFFIDefRegionKindSimple && type_info->metadata != nullptr &&
      type_info->metadata->structural_eq_hash_kind == kTVMFFISEqHashKindFreeVar) {
    mutator->WithDefRegionKind(kTVMFFIDefRegionKindNone, mutate_fields);
  } else {
    mutate_fields();
  }

  if (TVM_FFI_PREDICT_FALSE(result.is_err())) {
    return result;
  }
  if (!field_changed) {
    return Unchanged();
  }
  return result;
}

}  // namespace details

// ---------------------------------------------------------------------------
// Structural Map API.
// ---------------------------------------------------------------------------

namespace details {
/// \cond Doxygen_Suppress
// Return an error from the current raw or Expected mutation function.
// The rvalue-only helper lets the enclosing return type select the representation.
#define TVM_FFI_S_MUTATE_MAYBE_EARLY_RETURN(Result)                   \
  do {                                                                \
    auto&& tvm_ffi_res_ = (Result);                                   \
    if (TVM_FFI_PREDICT_FALSE(tvm_ffi_res_.is_err())) {               \
      return ::tvm::ffi::details::UnexpectedReturnHelper(             \
          ::tvm::ffi::Unexpected(::std::move(tvm_ffi_res_).error())); \
    }                                                                 \
  } while (0)

/// \endcond

/// \cond Doxygen_Suppress
#define TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN_IMPL_(Result, Type, Name, ResultExpr)               \
  auto Result = (ResultExpr); /* NOLINT(bugprone-macro-parentheses) */                        \
  TVM_FFI_S_MUTATE_MAYBE_EARLY_RETURN(Result);                                                \
  if constexpr (!::tvm::ffi::type_subsumes_v<::tvm::ffi::Expected<Type>, decltype(Result)>) { \
    if (TVM_FFI_PREDICT_FALSE(!::tvm::ffi::details::AnyUnsafe::CheckAnyStrict<Type>(          \
            ::tvm::ffi::details::ExpectedUnsafe::GetData(Result)))) {                         \
      return ::tvm::ffi::details::SMutateDeclaredTypeError();                                 \
    }                                                                                         \
  }                                                                                           \
  Type Name = /* NOLINT(bugprone-macro-parentheses) */                                        \
      ::tvm::ffi::details::AnyUnsafe::MoveFromAnyAfterCheck<Type>(                            \
          ::std::move(::tvm::ffi::details::ExpectedUnsafe::GetData(Result)))
/// \endcond

/*!
 * \brief Unwrap a successful mutation result into a newly declared value or return its error.
 *
 * ``Type`` must be concrete; use a type alias when it contains a top-level comma. A type mismatch
 * returns ``TypeError`` through the surrounding raw or ``Expected`` function without throwing,
 * reported with a fixed string. The check is omitted when the declared type subsumes the result's
 * success type. Its early returns work from either a raw ``TVMFFIAny`` hook or an
 * ``Expected<T>`` helper, including one with a different success type. This macro declares ``Name``
 * into the enclosing scope and must be used in a braced block, never as an unbraced control-flow
 * body.
 *
 * Example:
 * \code{.cpp}
 * TVMFFIAny FooMutate(StructuralMutatorObj* mutator, AnyView value) noexcept {
 *   const FooNode* self =
 *       details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const FooNode>(value);
 *   TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(UnchangedOr<Expr>, a,
 *                                     mutator->MutateExpected(self->a));
 *   TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(UnchangedOr<Expr>, b,
 *                                     mutator->MutateExpected(self->b));
 *   if (a.UnchangedOrSameAs(self->a) && b.UnchangedOrSameAs(self->b)) {
 *     return Unchanged().CopyToTVMFFIAny();
 *   }
 *   ObjectPtr<FooNode> copy = make_object<FooNode>(*self);
 *   copy->a = std::move(a).ValueOrUnchanged(std::move(copy->a));
 *   copy->b = std::move(b).ValueOrUnchanged(std::move(copy->b));
 *   return details::AnyUnsafe::MoveAnyToTVMFFIAny(Any(std::move(copy)));
 * }
 * \endcode
 *
 * Keep one statement per traversed field. A field skipped intentionally must be guarded and carry
 * a ``// skips:`` note.
 *
 * \param Type The concrete successful value type.
 * \param Name The name of the value declared in the enclosing scope.
 * \param ResultExpr An expression producing the ``Expected`` value to unwrap.
 * \sa Unchanged::CopyToTVMFFIAny
 */
#define TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(Type, Name, ResultExpr)                                  \
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN_IMPL_(TVM_FFI_STR_CONCAT(tvm_ffi_mutate_result_, __COUNTER__), \
                                          Type, Name, ResultExpr)

/// \cond Doxygen_Suppress
#define TVM_FFI_S_MUTATE_UNSAFE_ASSIGN_OR_RETURN_UNCHECKED_IMPL_(Result, Type, Name, ResultExpr) \
  auto Result = (ResultExpr); /* NOLINT(bugprone-macro-parentheses) */                           \
  TVM_FFI_S_MUTATE_MAYBE_EARLY_RETURN(Result);                                                   \
  Type Name = /* NOLINT(bugprone-macro-parentheses) */                                           \
      ::tvm::ffi::details::AnyUnsafe::MoveFromAnyAfterCheck<Type>(                               \
          ::std::move(::tvm::ffi::details::ExpectedUnsafe::GetData(Result)))
/// \endcond

/*!
 * \brief Unwrap a successful mutation result when the caller guarantees it to be ``Type``.
 *
 * This is an unsafe form that can only be used when the mutation contract guarantees the result
 * type.
 *
 * \param Type The guaranteed concrete type of the successful value.
 * \param Name The name of the value declared in the enclosing scope.
 * \param ResultExpr An expression producing the ``Expected`` value to unwrap.
 * \sa TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN
 */
#define TVM_FFI_S_MUTATE_UNSAFE_ASSIGN_OR_RETURN_UNCHECKED(Type, Name, ResultExpr) \
  TVM_FFI_S_MUTATE_UNSAFE_ASSIGN_OR_RETURN_UNCHECKED_IMPL_(                        \
      TVM_FFI_STR_CONCAT(tvm_ffi_mutate_result_, __COUNTER__), Type, Name, ResultExpr)

/// \cond Doxygen_Suppress
#define TVM_FFI_UNSAFE_S_MUTATE_ASSIGN_OR_RETURN_SKIP_CHECK_IMPL_(Result, Type, Name, ResultExpr) \
  auto Result = (ResultExpr); /* NOLINT(bugprone-macro-parentheses) */                            \
  TVM_FFI_S_MUTATE_MAYBE_EARLY_RETURN(Result);                                                    \
  TVM_FFI_DCHECK(::tvm::ffi::details::AnyUnsafe::CheckAnyStrict<Type>(                            \
      ::tvm::ffi::details::ExpectedUnsafe::GetData(Result)))                                      \
      << "unchecked structural-mutate assign: result is not of the declared type";                \
  Type Name = /* NOLINT(bugprone-macro-parentheses) */                                            \
      ::tvm::ffi::details::AnyUnsafe::MoveFromAnyAfterCheck<Type>(                                \
          ::std::move(::tvm::ffi::details::ExpectedUnsafe::GetData(Result)))
/// \endcond

/*!
 * \brief \ref TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN without the type check.
 *
 * Same signature, raw-or-same-T early-return support, and error propagation; the difference is
 * only what happens to a successful result that is not of type \p Type.
 *
 * The caller must guarantee the result has the declared type; a mismatch is undefined behavior
 * in a release build, and debug builds catch it with ``TVM_FFI_DCHECK``.
 *
 * \param Type The concrete type of the successful value.
 * \param Name The name of the value declared in the enclosing scope.
 * \param ResultExpr An expression producing the ``Expected`` value to unwrap.
 */
#define TVM_FFI_UNSAFE_S_MUTATE_ASSIGN_OR_RETURN_SKIP_CHECK(Type, Name, ResultExpr) \
  TVM_FFI_UNSAFE_S_MUTATE_ASSIGN_OR_RETURN_SKIP_CHECK_IMPL_(                        \
      TVM_FFI_STR_CONCAT(tvm_ffi_mutate_result_, __COUNTER__), Type, Name, ResultExpr)

}  // namespace details

/*!
 * \brief Shared state and variable-remap dispatch for both structural-map engines.
 *
 * This base owns the identity-substitution environment used for FreeVar and DAG identities.
 * Its variable-remap vtable thunks downcast to this common subobject, allowing the typed and
 * dynamic engines to share the environment even when a Parent layer sits above this base.
 */
class StructuralMapEngineBase : public StructuralMutatorObj {
 public:
  /*! \brief Empty callback-state protocol used when no custom Parent layer is present. */
  using StateTupleType = std::tuple<>;

  /*! \brief Construct the shared engine base with the concrete engine's vtable. */
  explicit StructuralMapEngineBase(const StructuralMutatorVTable* vtable)
      : StructuralMutatorObj(vtable) {}

  ~StructuralMapEngineBase() {
    for (const auto& kv : var_remap_) {
      details::ObjectUnsafe::DecRefObjectHandle(
          reinterpret_cast<TVMFFIObjectHandle>(const_cast<Object*>(kv.first)));
    }
  }

 protected:
  /*! \brief Return the empty state tuple exposed to typed map callbacks. */
  TVM_FFI_INLINE StateTupleType StateTuple() const noexcept { return {}; }

 private:
  /// \cond Doxygen_Suppress
  // Out of line so its strings stay out of the per-node dispatch function, which TryLink inlines
  // into. Shared by the typed and dynamic engines below.
  TVM_FFI_COLD_CODE static Expected<Any> SMutateDescentTypeError() noexcept {
    return Unexpected(Error("TypeError", "structural mutate: descent changed the node type", ""));
  }

  TVM_FFI_COLD_CODE static details::UnexpectedReturnHelper VarRemapKeyTypeError() noexcept {
    return details::UnexpectedReturnHelper(
        Unexpected(Error("TypeError", "Variable-remap key must be an object-backed value", "")));
  }
  /// \endcond

  /*!
   * \brief Dispatch variable-remap lookup through the mutator vtable.
   * \param mutator The erased callback-aware mutator.
   * \param var The borrowed variable identity to look up.
   * \return Raw ``TVMFFIAny`` containing the owning replacement, FFI None, or Error.
   */
  static TVMFFIAny DispatchVarRemapGet(StructuralMutatorObj* mutator, AnyView var) noexcept {
    auto* self = static_cast<StructuralMapEngineBase*>(mutator);
    return details::ExpectedUnsafe::MoveToTVMFFIAny(self->VarRemapGetImpl(var));
  }

  /*!
   * \brief Dispatch variable-remap insertion through the mutator vtable.
   * \param mutator The erased callback-aware mutator.
   * \param var The borrowed variable identity to bind.
   * \param mapped_value The borrowed replacement value.
   * \return Raw ``TVMFFIAny`` containing FFI None or Error.
   */
  static TVMFFIAny DispatchVarRemapSet(StructuralMutatorObj* mutator, AnyView var,
                                       AnyView mapped_value) noexcept {
    auto* self = static_cast<StructuralMapEngineBase*>(mutator);
    return details::ExpectedUnsafe::MoveToTVMFFIAny(self->VarRemapSetImpl(var, mapped_value));
  }

 protected:
  /*!
   * \brief Append \p node to a failed result's mutate error context.
   * \param result The failed result whose Error is annotated.
   * \param node The borrowed node to name in the context.
   */
  TVM_FFI_COLD_CODE static void UpdateVisitErrorContext(const Expected<Any>& result,
                                                        AnyView node) noexcept {
    // The Error is refcounted, so annotating the local handle annotates the object the result
    // holds. A non-object node has no context to add.
    if (node.type_index() >= TypeIndex::kTVMFFIStaticObjectBegin) {
      Error err = result.error();
      ::tvm::ffi::details::UpdateVisitErrorContext(err, node.cast<ObjectRef>());
    }
  }

  /*! \brief Whether the owned variable-remap environment has any entries. */
  TVM_FFI_INLINE bool HasVarRemap() const noexcept { return !var_remap_.empty(); }

  /*!
   * \brief Look up a replacement in the identity-substitution environment.
   * \param var The borrowed variable identity to look up.
   * \return The owning replacement, FFI None on a miss, or an Error.
   */
  Expected<Any> VarRemapGetImpl(AnyView var) noexcept {
    if (TVM_FFI_PREDICT_FALSE(var.type_index() < TypeIndex::kTVMFFIStaticObjectBegin)) {
      return VarRemapKeyTypeError();
    }
    // An empty environment is common for ordinary unchanged traversal. Avoid
    // hashing the identity (including a bucket-count division) on this path.
    if (var_remap_.empty()) return Any(nullptr);
    const Object* var_ptr =
        details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const Object>(var);
    auto it = var_remap_.find(var_ptr);
    return it == var_remap_.end() ? Any(nullptr) : it->second;
  }

  /*!
   * \brief Record a replacement in the identity-substitution environment.
   * \param var The borrowed variable identity to bind.
   * \param mapped_value The borrowed replacement value.
   * \return Successful completion, or an Error if the binding cannot be stored.
   */
  Expected<void> VarRemapSetImpl(AnyView var, AnyView mapped_value) noexcept {
    if (TVM_FFI_PREDICT_FALSE(var.type_index() < TypeIndex::kTVMFFIStaticObjectBegin)) {
      return VarRemapKeyTypeError();
    }
    const Object* var_ptr =
        details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const Object>(var);
    // Allocation in the owned binding must not escape this noexcept boundary.
    // In particular, a failed insertion leaves both the map and borrowed key
    // unchanged, while the local owning replacement releases normally.
    try {
      Any owned_mapped_value(mapped_value);
      auto [it, inserted] = var_remap_.try_emplace(var_ptr, std::move(owned_mapped_value));
      if (inserted) {
        details::ObjectUnsafe::IncRefObjectHandle(
            reinterpret_cast<TVMFFIObjectHandle>(const_cast<Object*>(var_ptr)));
      } else {
        it->second = std::move(owned_mapped_value);
      }
      return Expected<void>();
    } catch (Error& error) {
      return Unexpected(std::move(error));
    } catch (const std::exception& error) {
      return Unexpected(Error("RuntimeError", error.what(), ""));
    }
  }

 private:
  template <typename Parent, WalkOrder order, typename... Callbacks>
  friend class StructuralMapEngine;
  template <typename Parent, WalkOrder order>
  friend class StructuralMapDynEngine;
  template <typename Parent, typename... Callbacks>
  friend class StructuralMutateEngine;
  template <typename Parent>
  friend class details::StructuralMutateDynEngine;

  // Raw-pointer key: IncRef once on first insert, DecRef all keys in the destructor.
  std::unordered_map<const Object*, Any> var_remap_;
};

/*!
 * \brief Callback-dispatched structural mutator with a state-carrying Parent layer.
 *
 * Each callback is an ordinary callable and the engine selects it on its first argument's
 * type, using a compile-time-known ``as<TSub>()`` on the input node in pre-order and on the
 * descended node in post-order.
 *
 * ``Parent`` derives from ``StructuralMapEngineBase``, publishes ``StateTupleType``, accepts
 * and forwards the mutator vtable in its constructor, and provides a protected
 * ``StateTuple() const noexcept``. Engine-internal descent uses the public Expected entry points,
 * so a Parent may override those entries directly. Each callback receives every tuple entry
 * positionally, followed optionally by
 * ``TVMFFIDefRegionKind``. Descent calls use ``this->``, matching
 * ``StructuralWalkEngine``: lookup happens at instantiation in the Parent's class scope and is
 * not virtual dispatch. This deliberately leaves composed deeper-layer declarations eligible;
 * spelling the calls as ``Parent::member`` would instead pin lookup at that qualified layer.
 * A Parent must not hide other engine-internal ``this->`` members because matching ABI-vtable
 * paths deliberately terminate at ``StructuralMapEngineBase``.
 *
 * \tparam Parent Mutator layer that supplies descent and callback state through the complete
 *                protocol above.
 * \tparam order Callback placement relative to child mapping.
 * \tparam Callbacks The callbacks, tested in declaration order.
 */
template <typename Parent, WalkOrder order, typename... Callbacks>
class StructuralMapEngine : public Parent {
 public:
  static_assert(std::is_base_of_v<StructuralMapEngineBase, Parent>,
                "StructuralMap Parent must derive from StructuralMapEngineBase");
  /*! \brief Tuple of state references supplied by the Parent layer. */
  using StateTupleType = typename Parent::StateTupleType;

  /*!
   * \brief Construct a callback-aware mutator that owns its callbacks.
   * \param callbacks The typed callback links, tested in declaration order.
   */
  explicit StructuralMapEngine(Callbacks... callbacks)
      : Parent(VTable()), callbacks_(std::move(callbacks)...) {}

 private:
  using ExpectedUnsafe = details::ExpectedUnsafe;
  using AnyUnsafe = details::AnyUnsafe;

  /*!
   * \brief Return the shared callback-aware mutator vtable.
   * \return Pointer to the immutable mutator vtable for this specialization.
   */
  static const StructuralMutatorVTable* VTable() {
    static const StructuralMutatorVTable vtable{
        &StructuralMapEngine::DispatchMutate,
        &StructuralMapEngine::DispatchMaybeInplaceMutate,
        &StructuralMapEngine::DispatchVarRemapGet,
        &StructuralMapEngine::DispatchVarRemapSet,
    };
    return &vtable;
  }

  /*!
   * \brief Dispatch callback-aware optional in-place mutation through the ABI vtable.
   * \param mutator The erased callback-aware mutator.
   * \param value The borrowed value to mutate.
   * \return Raw ``TVMFFIAny`` containing the mutated value or Error.
   */
  static TVMFFIAny DispatchMaybeInplaceMutate(StructuralMutatorObj* mutator,
                                              AnyView value) noexcept {
    auto* self = static_cast<StructuralMapEngine*>(mutator);
    return self->MaybeInplaceMutateImplRaw(value);
  }

  /*!
   * \brief Dispatch callback-aware mutation through the ABI vtable.
   * \param mutator The erased callback-aware mutator.
   * \param value The borrowed value to mutate.
   * \return Raw ``TVMFFIAny`` containing the mutated value or Error.
   */
  static TVMFFIAny DispatchMutate(StructuralMutatorObj* mutator, AnyView value) noexcept {
    auto* self = static_cast<StructuralMapEngine*>(mutator);
    return self->MutateImplRaw(value);
  }

  template <typename Callback, typename Value, size_t... Is>
  TVM_FFI_INLINE Expected<Any> InvokeTypedCallbackLink(Callback& callback, Value&& value,
                                                       std::index_sequence<Is...>) noexcept {
    using FuncInfo = details::FunctionInfo<std::decay_t<Callback>>;
    static_assert(std::is_convertible_v<typename FuncInfo::RetType, Expected<Any>>,
                  "StructuralMap callbacks must return a replacement value, Error, Unexpected, "
                  "unchanged marker, or Expected<Any>");
    static_assert(
        FuncInfo::num_args == 1 + sizeof...(Is) || FuncInfo::num_args == 2 + sizeof...(Is),
        "StructuralMap callback takes (value, state...) with an optional trailing "
        "definition-region kind");
    try {
      static_assert(std::is_same_v<decltype(this->StateTuple()), StateTupleType>,
                    "Parent::StateTuple() must return Parent::StateTupleType by value");
      StateTupleType states = this->StateTuple();
      if constexpr (FuncInfo::num_args == 1 + sizeof...(Is)) {
        return callback(std::forward<Value>(value), std::get<Is>(states)...);
      } else {
        return callback(std::forward<Value>(value), std::get<Is>(states)...,
                        this->def_region_kind());
      }
    } catch (const Error& err) {
      return Unexpected(err);
    }
  }

  /*!
   * \brief Test one link against \p value and, if it matches, mutate the node through it.
   *
   * \tparam kMaybeInplace Whether the caller may mutate a uniquely owned node in place.
   * \tparam Callback The link's callback type.
   * \param callback The link's callback.
   * \param value The borrowed value to test and mutate.
   * \param out Receives the mutated value or Error when the link matched.
   * \return Whether the link matched, in which case \p out was written.
   */
  template <bool kMaybeInplace, typename Callback>
  TVM_FFI_INLINE bool TryLink(Callback& callback, AnyView value, Expected<Any>* out) noexcept {
    using FuncInfo = details::FunctionInfo<std::decay_t<Callback>>;
    static_assert(FuncInfo::num_args >= 1,
                  "StructuralMap callback must take at least a value argument");
    using FirstArg = std::tuple_element_t<0, typename FuncInfo::ArgType>;
    using TSub = std::remove_cv_t<std::remove_reference_t<FirstArg>>;

    using StateIndices = std::make_index_sequence<std::tuple_size_v<StateTupleType>>;
    if constexpr (order == WalkOrder::kPreOrder) {
      std::optional<TSub> matched;
      if constexpr (!std::is_same_v<TSub, AnyView> && !std::is_same_v<TSub, Any>) {
        matched = value.template as<TSub>();
        if (!matched.has_value()) return false;
      }
      // Pre-order: the callback rewrites this node first, then descent runs over whatever it
      // produced, so a replacement subtree is itself mapped.
      Expected<Any> callback_result = [&]() -> Expected<Any> {
        if constexpr (std::is_same_v<TSub, AnyView>) {
          return InvokeTypedCallbackLink(callback, value, StateIndices{});
        } else if constexpr (std::is_same_v<TSub, Any>) {
          return InvokeTypedCallbackLink(callback, Any(value), StateIndices{});
        } else {
          // Reuses the conversion the match already performed.
          return InvokeTypedCallbackLink(callback, *std::move(matched), StateIndices{});
        }
      }();
      if (TVM_FFI_PREDICT_FALSE(callback_result.is_err())) {
        this->UpdateVisitErrorContext(callback_result, value);
        *out = std::move(callback_result);
        return true;
      }
      Any mapped_value = std::move(ExpectedUnsafe::GetData(callback_result));
      const AnyView descent_view =
          mapped_value.type_index() == TypeIndex::kTVMFFIUnchanged ? value : AnyView(mapped_value);
      // Each descent names the node it actually ran on in the error context.
      *out = [&]() -> Expected<Any> {
        if constexpr (kMaybeInplace) {
          // A pre-order result can be mutated in place if unchanged or uniquely owned.
          if (descent_view.same_as(value)) {
            return this->DefaultMaybeInplaceMutateExpected(value);
          }
          const Object* mapped_obj = descent_view.as<Object>();
          bool can_inplace = mapped_obj != nullptr && mapped_obj->unique();
          return can_inplace ? this->DefaultMaybeInplaceMutateExpected(descent_view)
                             : this->DefaultMutateExpected(descent_view);
        } else {
          return this->DefaultMutateExpected(descent_view);
        }
      }();
      if (TVM_FFI_PREDICT_FALSE(out->is_err())) return true;
      if (ExpectedUnsafe::GetData(*out).type_index() == TypeIndex::kTVMFFIUnchanged) {
        *out = std::move(mapped_value);
      }
      return true;
    } else {
      // Post-order descent is performed once by the engine entry before the callback probes.
      // Descended unchanged uses the original view; otherwise the callback sees the replacement.
      const Any& descended_value = ExpectedUnsafe::GetData(*out);
      const AnyView mapped_view = descended_value.type_index() == TypeIndex::kTVMFFIUnchanged
                                      ? value
                                      : AnyView(descended_value);
      std::optional<TSub> matched;
      if constexpr (!std::is_same_v<TSub, AnyView> && !std::is_same_v<TSub, Any>) {
        matched = mapped_view.template as<TSub>();
        if (!matched.has_value()) return false;
      }
      *out = [&]() -> Expected<Any> {
        if constexpr (std::is_same_v<TSub, AnyView>) {
          return InvokeTypedCallbackLink(callback, mapped_view, StateIndices{});
        } else if constexpr (std::is_same_v<TSub, Any>) {
          return InvokeTypedCallbackLink(callback, Any(mapped_view), StateIndices{});
        } else {
          return InvokeTypedCallbackLink(callback, *std::move(matched), StateIndices{});
        }
      }();
      if (TVM_FFI_PREDICT_FALSE(out->is_err())) {
        this->UpdateVisitErrorContext(*out, mapped_view);
        return true;
      }
      return true;
    }
  }

  /*!
   * \brief Test every link in declaration order, stopping at the first that matches.
   * \return Whether some link matched, in which case \p out was written.
   */
  template <bool kMaybeInplace, size_t... Is>
  TVM_FFI_INLINE bool TryLinks(AnyView value, Expected<Any>* out,
                               std::index_sequence<Is...>) noexcept {
    return (TryLink<kMaybeInplace>(std::get<Is>(callbacks_), value, out) || ...);
  }

  /*!
   * \brief Mutate a value, invoking the first matching callback link.
   * \param value The borrowed value to mutate.
   * \return Raw ``TVMFFIAny`` containing the mutated value or Error.
   */
  TVM_FFI_INLINE TVMFFIAny MutateImplRaw(AnyView value) noexcept {
    Expected<Any> out{Any()};
    if constexpr (order == WalkOrder::kPostOrder) {
      out = this->DefaultMutateExpected(value);
      if (TVM_FFI_PREDICT_FALSE(out.is_err())) {
        return ExpectedUnsafe::MoveToTVMFFIAny(std::move(out));
      }
      TryLinks<false>(value, &out, std::index_sequence_for<Callbacks...>{});
      return ExpectedUnsafe::MoveToTVMFFIAny(std::move(out));
    } else {
      if (TryLinks<false>(value, &out, std::index_sequence_for<Callbacks...>{})) {
        return ExpectedUnsafe::MoveToTVMFFIAny(std::move(out));
      }
      return ExpectedUnsafe::MoveToTVMFFIAny(this->DefaultMutateExpected(value));
    }
  }

  /*!
   * \brief Mutate a value in place when safe, invoking the first matching callback link.
   * \param value The borrowed value to mutate.
   * \return Raw ``TVMFFIAny`` containing the mutated value or Error.
   */
  TVM_FFI_INLINE TVMFFIAny MaybeInplaceMutateImplRaw(AnyView value) noexcept {
    Expected<Any> out{Any()};
    if constexpr (order == WalkOrder::kPostOrder) {
      out = this->DefaultMaybeInplaceMutateExpected(value);
      if (TVM_FFI_PREDICT_FALSE(out.is_err())) {
        return ExpectedUnsafe::MoveToTVMFFIAny(std::move(out));
      }
      TryLinks<true>(value, &out, std::index_sequence_for<Callbacks...>{});
      return ExpectedUnsafe::MoveToTVMFFIAny(std::move(out));
    } else {
      if (TryLinks<true>(value, &out, std::index_sequence_for<Callbacks...>{})) {
        return ExpectedUnsafe::MoveToTVMFFIAny(std::move(out));
      }
      return ExpectedUnsafe::MoveToTVMFFIAny(this->DefaultMaybeInplaceMutateExpected(value));
    }
  }

  /*! \brief The callback links, tested in declaration order. */
  std::tuple<Callbacks...> callbacks_;
};

/*!
 * \brief Structural mutator whose links are runtime ffi.Functions keyed by type index.
 *
 * This is the dynamic counterpart of \ref StructuralMapEngine. Moving it into this header makes
 * the same Parent layering available to downstream dynamic mutators; only its runtime link table
 * and invocation differ from the typed engine.
 * Its ``Parent`` follows the same descent-layer protocol, while runtime ``Function`` callbacks
 * retain their existing ``(value)`` or ``(value, TVMFFIDefRegionKind)`` ABI.
 *
 * \tparam Parent Mutator layer that supplies descent and callback state through the same
 *                ``this->``-bound protocol documented on \ref StructuralMapEngine.
 * \tparam order Callback placement relative to child mapping.
 */
template <typename Parent, WalkOrder order>
class StructuralMapDynEngine : public Parent {
 public:
  static_assert(std::is_base_of_v<StructuralMapEngineBase, Parent>,
                "StructuralMap Parent must derive from StructuralMapEngineBase");
  /*!
   * \brief Construct a dynamic map engine with the default Parent constructor.
   * \param callbacks Runtime links invoked as ``callback(value)``.
   * \param callbacks_with_def_region_kind Runtime links invoked with the active region kind.
   */
  StructuralMapDynEngine(Array<Tuple<int32_t, Function>> callbacks,
                         Array<Tuple<int32_t, Function>> callbacks_with_def_region_kind)
      : Parent(VTable()),
        callbacks_(std::move(callbacks)),
        callbacks_with_def_region_kind_(std::move(callbacks_with_def_region_kind)) {}

 private:
  using ExpectedUnsafe = details::ExpectedUnsafe;
  using AnyUnsafe = details::AnyUnsafe;

  /*! \brief Return the shared dynamic-engine mutator vtable. */
  static const StructuralMutatorVTable* VTable() {
    static const StructuralMutatorVTable vtable{
        &StructuralMapDynEngine::DispatchMutate,
        &StructuralMapDynEngine::DispatchMaybeInplaceMutate,
        &StructuralMapDynEngine::DispatchVarRemapGet,
        &StructuralMapDynEngine::DispatchVarRemapSet,
    };
    return &vtable;
  }

  /*! \brief Dispatch optional in-place mutation through the ABI vtable. */
  static TVMFFIAny DispatchMaybeInplaceMutate(StructuralMutatorObj* mutator,
                                              AnyView value) noexcept {
    return static_cast<StructuralMapDynEngine*>(mutator)->MaybeInplaceMutateImplRaw(value);
  }

  /*! \brief Dispatch non-in-place mutation through the ABI vtable. */
  static TVMFFIAny DispatchMutate(StructuralMutatorObj* mutator, AnyView value) noexcept {
    return static_cast<StructuralMapDynEngine*>(mutator)->MutateImplRaw(value);
  }

  /*!
   * \brief Find the first runtime link registered for \p type_index.
   * \param type_index The candidate node's runtime type index.
   * \param with_kind Set when the matched link also takes a def-region kind.
   * \return The matched Function, or nullopt when no link applies.
   */
  Optional<Function> FindLink(int32_t type_index, bool* with_kind) const noexcept {
    for (const Tuple<int32_t, Function>& entry : callbacks_) {
      if (details::RuntimeTypeIndexMatch(type_index, entry.get<0>())) {
        *with_kind = false;
        return entry.get<1>();
      }
    }
    for (const Tuple<int32_t, Function>& entry : callbacks_with_def_region_kind_) {
      if (details::RuntimeTypeIndexMatch(type_index, entry.get<0>())) {
        *with_kind = true;
        return entry.get<1>();
      }
    }
    return std::nullopt;
  }

  /*!
   * \brief Invoke a matched runtime link with its requested arguments.
   *
   * The caller reads the live def-region kind at invocation time. ``CallExpected`` uses the
   * exception-free safe-call path and represents raised errors as ``Unexpected``.
   */
  TVM_FFI_INLINE static Expected<Any> InvokeLink(const Function& fn, bool with_kind, AnyView target,
                                                 TVMFFIDefRegionKind kind) noexcept {
    return with_kind ? fn.CallExpected<Any>(target, kind) : fn.CallExpected<Any>(target);
  }

  /*!
   * \brief Test the runtime link table against \p value and mutate through the first match.
   * \tparam kMaybeInplace Whether a uniquely owned node may be mutated in place.
   * \param value The borrowed value to test and mutate.
   * \param out Receives the mutated value or Error when a link matched.
   * \return Whether a link matched, in which case \p out was written.
   */
  template <bool kMaybeInplace>
  TVM_FFI_INLINE bool TryLink(AnyView value, Expected<Any>* out) noexcept {
    if constexpr (order == WalkOrder::kPreOrder) {
      bool with_kind = false;
      Optional<Function> matched = FindLink(value.type_index(), &with_kind);
      if (!matched.has_value()) return false;
      // Pre-order: the callback rewrites this node first, then descent runs over what it made.
      Expected<Any> callback_result =
          InvokeLink(*matched, with_kind, value, this->def_region_kind());
      if (TVM_FFI_PREDICT_FALSE(callback_result.is_err())) {
        this->UpdateVisitErrorContext(callback_result, value);
        *out = std::move(callback_result);
        return true;
      }
      Any mapped_value = std::move(ExpectedUnsafe::GetData(callback_result));
      const AnyView descent_view =
          mapped_value.type_index() == TypeIndex::kTVMFFIUnchanged ? value : AnyView(mapped_value);
      *out = [&]() -> Expected<Any> {
        if constexpr (kMaybeInplace) {
          if (descent_view.same_as(value)) {
            return this->DefaultMaybeInplaceMutateExpected(value);
          }
          const Object* mapped_obj = descent_view.as<Object>();
          bool can_inplace = mapped_obj != nullptr && mapped_obj->unique();
          return can_inplace ? this->DefaultMaybeInplaceMutateExpected(descent_view)
                             : this->DefaultMutateExpected(descent_view);
        } else {
          return this->DefaultMutateExpected(descent_view);
        }
      }();
      if (TVM_FFI_PREDICT_FALSE(out->is_err())) return true;
      if (ExpectedUnsafe::GetData(*out).type_index() == TypeIndex::kTVMFFIUnchanged) {
        *out = std::move(mapped_value);
      }
      return true;
    } else {
      // Post-order descent is performed once by the engine entry before link selection.
      // See the typed engine: a borrowed view of the original, never an owning copy of it.
      const Any& descended_value = ExpectedUnsafe::GetData(*out);
      const AnyView mapped_view = descended_value.type_index() == TypeIndex::kTVMFFIUnchanged
                                      ? value
                                      : AnyView(descended_value);
      bool with_kind = false;
      Optional<Function> matched = FindLink(mapped_view.type_index(), &with_kind);
      if (!matched.has_value()) return false;
      // WithDefRegionKind restores its state through RAII, so this late read is equivalent to
      // the typed engine's invocation-time read even after recursive descent.
      *out = InvokeLink(*matched, with_kind, mapped_view, this->def_region_kind());
      if (TVM_FFI_PREDICT_FALSE(out->is_err())) {
        this->UpdateVisitErrorContext(*out, mapped_view);
        return true;
      }
      return true;
    }
  }

  /*! \brief Mutate a value, invoking the first matching runtime link. */
  TVM_FFI_INLINE TVMFFIAny MutateImplRaw(AnyView value) noexcept {
    Expected<Any> out{Any()};
    if constexpr (order == WalkOrder::kPostOrder) {
      out = this->DefaultMutateExpected(value);
      if (TVM_FFI_PREDICT_FALSE(out.is_err())) {
        return ExpectedUnsafe::MoveToTVMFFIAny(std::move(out));
      }
      TryLink<false>(value, &out);
      return ExpectedUnsafe::MoveToTVMFFIAny(std::move(out));
    } else {
      if (TryLink<false>(value, &out)) {
        return ExpectedUnsafe::MoveToTVMFFIAny(std::move(out));
      }
      return ExpectedUnsafe::MoveToTVMFFIAny(this->DefaultMutateExpected(value));
    }
  }

  /*! \brief Optionally mutate a value in place through the first matching runtime link. */
  TVM_FFI_INLINE TVMFFIAny MaybeInplaceMutateImplRaw(AnyView value) noexcept {
    Expected<Any> out{Any()};
    if constexpr (order == WalkOrder::kPostOrder) {
      out = this->DefaultMaybeInplaceMutateExpected(value);
      if (TVM_FFI_PREDICT_FALSE(out.is_err())) {
        return ExpectedUnsafe::MoveToTVMFFIAny(std::move(out));
      }
      TryLink<true>(value, &out);
      return ExpectedUnsafe::MoveToTVMFFIAny(std::move(out));
    } else {
      if (TryLink<true>(value, &out)) {
        return ExpectedUnsafe::MoveToTVMFFIAny(std::move(out));
      }
      return ExpectedUnsafe::MoveToTVMFFIAny(this->DefaultMaybeInplaceMutateExpected(value));
    }
  }

  /*! \brief Runtime links invoked without def-region context. */
  Array<Tuple<int32_t, Function>> callbacks_;
  /*! \brief Runtime links invoked with def-region context. */
  Array<Tuple<int32_t, Function>> callbacks_with_def_region_kind_;
};

/*!
 * \brief Engine of the callback-dispatched \ref tvm::ffi::StructuralMutate.
 *
 * A matched callback owns mutation of its value, so the engine returns the
 * callback result without descending into it. An unmatched value keeps the
 * Parent's default mutation. ``Parent::MutatorObjType`` pins the exact
 * callback-facing mutator view across layer composition.
 *
 * \tparam Parent Mutator layer extended by the engine.
 * \tparam Callbacks Callable types whose first parameter selects the value type.
 */
template <typename Parent, typename... Callbacks>
class StructuralMutateEngine : public Parent {
 public:
  static_assert(std::is_base_of_v<StructuralMapEngineBase, Parent>,
                "StructuralMutate Parent must derive from StructuralMapEngineBase");

  /*! \brief Construct a mutate engine over callbacks tested in declaration order. */
  explicit StructuralMutateEngine(Callbacks... callbacks)
      : Parent(VTable()), callbacks_(std::move(callbacks)...) {}

 private:
  /*! \brief Return this engine's immutable callback-aware mutator vtable. */
  static const StructuralMutatorVTable* VTable() {
    static const StructuralMutatorVTable vtable{
        &StructuralMutateEngine::DispatchMutate,
        &StructuralMutateEngine::DispatchMaybeInplaceMutate,
        &StructuralMutateEngine::DispatchVarRemapGet,
        &StructuralMutateEngine::DispatchVarRemapSet,
    };
    return &vtable;
  }

  /*! \brief Dispatch ordinary mutation from the erased mutator pointer. */
  static TVMFFIAny DispatchMutate(StructuralMutatorObj* mutator, AnyView value) noexcept {
    auto* self = static_cast<StructuralMutateEngine*>(mutator);
    if constexpr (sizeof...(Callbacks) == 1) {
      return self->template MutateSingleCallbackRaw<false>(value);
    } else {
      return self->MutateImplRaw(value);
    }
  }

  /*! \brief Dispatch maybe-in-place mutation from the erased mutator pointer. */
  static TVMFFIAny DispatchMaybeInplaceMutate(StructuralMutatorObj* mutator,
                                              AnyView value) noexcept {
    auto* self = static_cast<StructuralMutateEngine*>(mutator);
    if constexpr (sizeof...(Callbacks) == 1) {
      return self->template MutateSingleCallbackRaw<true>(value);
    } else {
      return self->MaybeInplaceMutateImplRaw(value);
    }
  }

  /*! \brief Mutate one value, handing a matched callback ownership of descent. */
  TVMFFIAny MutateImplRaw(AnyView value) noexcept {
    Expected<Any> result{Any()};
    if (DispatchCallbacks(value, false, &result)) {
      if (TVM_FFI_PREDICT_FALSE(result.is_err())) {
        // Keep callback-boundary context in addition to the default-descent
        // context: a callback may return a rebuilt value, so the two nodes can differ.
        Parent::UpdateVisitErrorContext(result, value);
      }
      return details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(result));
    }
    return details::ExpectedUnsafe::MoveToTVMFFIAny(Parent::DefaultMutateExpected(value));
  }

  /*! \brief Maybe mutate one value in place, with callback-owned descent. */
  TVMFFIAny MaybeInplaceMutateImplRaw(AnyView value) noexcept {
    Expected<Any> result{Any()};
    if (DispatchCallbacks(value, true, &result)) {
      if (TVM_FFI_PREDICT_FALSE(result.is_err())) {
        // Keep callback-boundary context in addition to the default-descent
        // context: a callback may return a rebuilt value, so the two nodes can differ.
        Parent::UpdateVisitErrorContext(result, value);
      }
      return details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(result));
    }
    return details::ExpectedUnsafe::MoveToTVMFFIAny(
        Parent::DefaultMaybeInplaceMutateExpected(value));
  }

  /*! \brief Invoke a matched callback with the Parent view and preserve returned/thrown Error. */
  template <typename Callback, typename Matched>
  TVM_FFI_INLINE Expected<Any> InvokeCallback(Callback& callback, Matched&& matched,
                                              bool allow_inplace) noexcept {
    using FuncInfo = details::FunctionInfo<std::decay_t<Callback>>;
    static_assert(FuncInfo::num_args == 2 || FuncInfo::num_args == 3,
                  "StructuralMutate callback must take (value, mutator) or "
                  "(value, mutator, allow_inplace)");
    using SecondArg = std::decay_t<std::tuple_element_t<1, typename FuncInfo::ArgType>>;
    using Second = std::remove_pointer_t<SecondArg>;
    static_assert(std::is_same_v<Second, typename Parent::MutatorObjType>,
                  "second StructuralMutate callback argument must be exactly "
                  "Parent::MutatorObjType*");
    if constexpr (FuncInfo::num_args == 3) {
      using ThirdArg = std::decay_t<std::tuple_element_t<2, typename FuncInfo::ArgType>>;
      static_assert(std::is_same_v<ThirdArg, bool>,
                    "third StructuralMutate callback argument must be bool");
    }
    auto* mutator = static_cast<typename Parent::MutatorObjType*>(this);
    try {
      if constexpr (FuncInfo::num_args == 3) {
        return callback(std::forward<Matched>(matched), mutator, allow_inplace);
      } else {
        return callback(std::forward<Matched>(matched), mutator);
      }
    } catch (Error& err) {
      return Unexpected(std::move(err));
    }
  }

  /*! \brief Materialize the callback result only on the cold Parent error-decoration path. */
  TVM_FFI_COLD_CODE TVMFFIAny AnnotateCallbackErrorRaw(TVMFFIAny raw, AnyView value) noexcept {
    Expected<Any> result = details::ExpectedUnsafe::MoveFromTVMFFIAny<Any>(raw);
    Parent::UpdateVisitErrorContext(result, value);
    return details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(result));
  }

  // One callback forwards its result directly, without a callback-chain envelope.
  // AnyView/Any always match; only a typed miss needs the Parent's default descent.
  template <bool kMaybeInplace>
  TVM_FFI_INLINE TVMFFIAny MutateSingleCallbackRaw(AnyView value) noexcept {
    auto& callback = std::get<0>(callbacks_);
    using FuncInfo = details::FunctionInfo<std::decay_t<decltype(callback)>>;
    using FirstArg = std::tuple_element_t<0, typename FuncInfo::ArgType>;
    using TSub = std::remove_cv_t<std::remove_reference_t<FirstArg>>;
    TVMFFIAny result;
    if constexpr (std::is_same_v<TSub, AnyView>) {
      result =
          details::ExpectedUnsafe::MoveToTVMFFIAny(InvokeCallback(callback, value, kMaybeInplace));
    } else if constexpr (std::is_same_v<TSub, Any>) {
      result = details::ExpectedUnsafe::MoveToTVMFFIAny(
          InvokeCallback(callback, Any(value), kMaybeInplace));
    } else if (auto matched = value.template as<TSub>()) {
      result = details::ExpectedUnsafe::MoveToTVMFFIAny(
          InvokeCallback(callback, *std::move(matched), kMaybeInplace));
    } else {
      if constexpr (kMaybeInplace) {
        return details::ExpectedUnsafe::MoveToTVMFFIAny(
            Parent::DefaultMaybeInplaceMutateExpected(value));
      } else {
        return details::ExpectedUnsafe::MoveToTVMFFIAny(Parent::DefaultMutateExpected(value));
      }
    }
    // Release any owning match before naming the callback boundary, as in the general path.
    if (TVM_FFI_PREDICT_FALSE(result.type_index == TypeIndex::kTVMFFIError)) {
      return AnnotateCallbackErrorRaw(result, value);
    }
    return result;
  }

  /*! \brief Write a matched callback result, leaving out untouched on a type miss. */
  template <typename Callback>
  TVM_FFI_INLINE bool TryLink(Callback& callback, AnyView value, bool allow_inplace,
                              Expected<Any>* out) noexcept {
    using FuncInfo = details::FunctionInfo<std::decay_t<Callback>>;
    using FirstArg = std::tuple_element_t<0, typename FuncInfo::ArgType>;
    using TSub = std::remove_cv_t<std::remove_reference_t<FirstArg>>;
    if constexpr (std::is_same_v<TSub, AnyView>) {
      *out = InvokeCallback(callback, value, allow_inplace);
      return true;
    } else if constexpr (std::is_same_v<TSub, Any>) {
      *out = InvokeCallback(callback, Any(value), allow_inplace);
      return true;
    } else if (auto matched = value.template as<TSub>()) {
      *out = InvokeCallback(callback, *std::move(matched), allow_inplace);
      return true;
    }
    return false;
  }

  /*! \brief Fold callbacks in declaration order, stopping at the first match. */
  template <size_t... Is>
  TVM_FFI_INLINE bool TryLinks(AnyView value, bool allow_inplace, Expected<Any>* out,
                               std::index_sequence<Is...>) noexcept {
    return (TryLink(std::get<Is>(callbacks_), value, allow_inplace, out) || ...);
  }

  /*! \brief Run the callback chain, returning whether a callback matched. */
  TVM_FFI_INLINE bool DispatchCallbacks(AnyView value, bool allow_inplace,
                                        Expected<Any>* out) noexcept {
    return TryLinks(value, allow_inplace, out, std::index_sequence_for<Callbacks...>{});
  }

  /*! \brief Typed callbacks tested in declaration order, first match wins. */
  std::tuple<Callbacks...> callbacks_;
};

/*!
 * \brief Structural map: mutate(x) = post(D(pre(x))).
 *
 * D is descent with a var remap that keeps the result consistent when a var
 * is rewritten as a cascade effect of its fields changing during descent.
 * Callbacks never read or write D's remap cache, and fire once per occurrence
 * in whichever position they sit.
 *
 * Var policy in default D: each var is descended at most once in def (its
 * first occurrence in a pattern def, its only occurrence in a simple def),
 * and then returns the new rewritten result if any in use. Definitions are
 * assumed to precede uses; a var with no definition is treated as a use.
 *
 * Canonical use cases include:
 * - use pre for var replacement to another value or var
 * - for a tree node, a post rewrite after its children are mapped
 *
 * \note For a DAG type node, post callback fires at every of its occurrences,
 *       so if the intent is to do graph rewrite, post callback needs to have
 *       its own node to value memo; the engine does not dedup callback rewrites.
 *
 * A callback is selected by its first argument type, optionally followed by
 * ``TVMFFIDefRegionKind``; the first match in declaration order runs. It
 * returns a replacement, ``Unchanged``, or ``Expected<U>``, and must not
 * mutate its input. A post-order callback is selected by the type of what
 * descent produced.
 *
 * \tparam order Whether callbacks run before or after mapping children.
 * \tparam Callbacks Callback types whose first parameters select matching values.
 * \param root The owning root value to map; pass ``std::move(root)`` to permit reuse.
 * \param callbacks Callbacks tested in declaration order.
 * \return The mapped owning value, or an Error if mapping or a callback fails.
 */
template <WalkOrder order, typename... Callbacks>
// The owning parameter makes caller ownership visible to the uniqueness check.
Expected<Any> StructuralMapExpected(
    Any root, Callbacks&&... callbacks) noexcept {  // NOLINT(performance-unnecessary-value-param)
  static_assert(sizeof...(Callbacks) != 0, "StructuralMap requires at least one callback");
  using Mutator = StructuralMapEngine<StructuralMapEngineBase, order, std::decay_t<Callbacks>...>;
  StructuralMutator mutator(make_object<Mutator>(std::forward<Callbacks>(callbacks)...));
  auto result = mutator->MaybeInplaceMutateIfUniqueExpected(root);
  if (TVM_FFI_PREDICT_FALSE(result.is_err())) return Unexpected(std::move(result).error());
  UnchangedOr<Any> mapped = details::AnyUnsafe::MoveFromAnyAfterCheck<UnchangedOr<Any>>(
      std::move(details::ExpectedUnsafe::GetData(result)));
  return std::move(mapped).ValueOrUnchanged(std::move(root));
}

/*!
 * \brief Throwing form of \ref tvm::ffi::StructuralMapExpected.
 *
 * See \ref tvm::ffi::StructuralMapExpected for callback dispatch, ordering, and ownership
 * semantics.
 *
 * \tparam order Whether callbacks run before or after recursively mapping children.
 * \tparam Callbacks Callback types whose first parameters select matching values.
 * \param root The owning root value to map.
 * \param callbacks Callbacks tested in declaration order. Each accepts ``(value)`` or
 *        ``(value, def_region_kind)`` and returns a replacement, ``Unchanged``, or
 *        ``Expected<Any>``.
 * \return The mapped owning value.
 * \throws Error if mapping or a callback fails.
 *
 * \note Returning ``Expected<Any>`` expresses errors as values; throwing ``Error`` is also
 *       supported and is rethrown by this interface.
 * \note Pass an owned root with ``std::move(root)`` to permit root reuse.
 */
template <WalkOrder order, typename... Callbacks>
// The owning parameter makes caller ownership visible to the uniqueness check.
Any StructuralMap(Any root,
                  Callbacks&&... callbacks) {  // NOLINT(performance-unnecessary-value-param)
  return StructuralMapExpected<order>(std::move(root), std::forward<Callbacks>(callbacks)...)
      .value();
}

/*!
 * \brief Mutate a structured value with callbacks that own recursion.
 *
 * A callback takes one of two forms, where ``R`` is a supported return type:
 *
 * - ``R(const T& value, StructuralMutatorObj* mutator)``
 * - ``R(const T& value, StructuralMutatorObj* mutator, bool allow_inplace)``
 *
 * A callback returns a replacement, ``Unchanged``, or ``Expected<Any>``. The first argument
 * selects by FFI type; callbacks are tried in declaration order and the first match owns mutation
 * -- it drives its own recursion through the mutator and sets any variable remapping. An unmatched
 * value takes registered or reflected default mutation.
 *
 * \param root The owning root value to mutate.
 * \param callbacks Callbacks tested in declaration order.
 * \return The mutated owning value, or an Error if mutation or a callback fails.
 *
 * \note A two-argument callback descends with ``MutateExpected`` and remains copy-on-write.
 *       A three-argument callback receives ``allow_inplace=true`` only when its value is on a
 *       uniquely owned path and may then explicitly use the maybe-in-place mutator operation.
 * \note Pass an owned root with ``std::move(root)`` to permit root reuse. In a
 *       ``__s_maybe_inplace_mutate__`` hook, the corresponding nested idiom is
 *       ``self->field = StructuralMap(std::move(self->field), callback)``; const-correctness
 *       rejects that ownership transfer outside a mutable maybe-in-place hook.
 */
template <typename... Callbacks>
// The owning parameter makes caller ownership visible to the uniqueness check.
Expected<Any> StructuralMutateExpected(
    Any root, Callbacks&&... callbacks) noexcept {  // NOLINT(performance-unnecessary-value-param)
  static_assert(sizeof...(Callbacks) != 0, "StructuralMutate requires at least one callback");
  using Mutator = StructuralMutateEngine<StructuralMapEngineBase, std::decay_t<Callbacks>...>;
  StructuralMutator mutator(make_object<Mutator>(std::forward<Callbacks>(callbacks)...));
  auto result = mutator->MaybeInplaceMutateIfUniqueExpected(root);
  if (TVM_FFI_PREDICT_FALSE(result.is_err())) return Unexpected(std::move(result).error());
  UnchangedOr<Any> mapped = details::AnyUnsafe::MoveFromAnyAfterCheck<UnchangedOr<Any>>(
      std::move(details::ExpectedUnsafe::GetData(result)));
  return std::move(mapped).ValueOrUnchanged(std::move(root));
}

/*!
 * \brief Throwing form of \ref tvm::ffi::StructuralMutateExpected.
 *
 * \note Pass an owned root with ``std::move(root)`` to permit root reuse.
 */
template <typename... Callbacks>
// The owning parameter makes caller ownership visible to the uniqueness check.
Any StructuralMutate(Any root,
                     Callbacks&&... callbacks) {  // NOLINT(performance-unnecessary-value-param)
  return StructuralMutateExpected(std::move(root), std::forward<Callbacks>(callbacks)...).value();
}

template <typename T>
inline constexpr bool use_default_type_traits_v<UnchangedOr<T>> = false;

template <typename T>
struct TypeTraits<UnchangedOr<T>> : public TypeTraitsBase {
  TVM_FFI_INLINE static void CopyToAnyView(const UnchangedOr<T>& src, TVMFFIAny* result) {
    *result = AnyView(src.data_).CopyToTVMFFIAny();
  }

  TVM_FFI_INLINE static void MoveToAny(UnchangedOr<T> src, TVMFFIAny* result) {
    *result = details::UnchangedOrUnsafe::MoveToTVMFFIAny(std::move(src));
  }

  TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
    if constexpr (std::is_same_v<T, Any>) {
      return src->type_index != TypeIndex::kTVMFFIError;
    } else {
      return src->type_index == TypeIndex::kTVMFFIUnchanged || TypeTraits<T>::CheckAnyStrict(src);
    }
  }

  TVM_FFI_INLINE static UnchangedOr<T> CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
    if (src->type_index == TypeIndex::kTVMFFIUnchanged) return Unchanged();
    if constexpr (std::is_same_v<T, Any>) {
      return UnchangedOr<T>(Any(AnyView::CopyFromTVMFFIAny(*src)));
    } else {
      return UnchangedOr<T>(TypeTraits<T>::CopyFromAnyViewAfterCheck(src));
    }
  }

  TVM_FFI_INLINE static UnchangedOr<T> MoveFromAnyAfterCheck(TVMFFIAny* src) {
    return UnchangedOr<T>(typename UnchangedOr<T>::UnsafeInit{},
                          details::AnyUnsafe::MoveTVMFFIAnyToAny(src));
  }
  TVM_FFI_INLINE static std::string TypeStr() {
    return "UnchangedOr<" + details::Type2Str<T>::v() + ">";
  }
  TVM_FFI_INLINE static std::string TypeSchema() {
    return R"({"type":"UnchangedOr","args":[)" + details::TypeSchema<T>::v() + "]}";
  }
};

}  // namespace ffi
}  // namespace tvm

#endif  // TVM_FFI_EXTRA_STRUCTURAL_MUTATE_H_
