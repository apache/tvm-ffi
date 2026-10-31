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
#include <tvm/ffi/function_detail.h>
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

/*!
 * \brief Permission to mutate a value in place along its ownership path.
 *
 * \note The numeric values 0 and 1 are part of the Python integer binding contract,
 *       matching ``InplaceMode.DISALLOW`` and ``InplaceMode.ALLOW`` respectively.
 */
enum class InplaceMode : int32_t {
  /*! \brief Preserve the source through copy-on-write mutation. */
  kDisallow = 0,
  /*! \brief Permit in-place mutation when the current value is uniquely owned. */
  kAllow = 1,
};

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
using FStructuralMutateABI = TVMFFIAny (*)(StructuralMutatorObj* mutator, AnyView value) noexcept;

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

namespace detail {

// Copy and structurally mutate the reflected fields of an object-backed value.
TVM_FFI_INLINE static Expected<Any> MutateReflectedFieldsExpected(StructuralMutatorObj* mutator,
                                                                  AnyView value) noexcept;

}  // namespace detail

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
  FStructuralMutateABI mutate = nullptr;
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
  FStructuralMutateABI maybe_inplace_mutate = nullptr;
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

namespace detail {
template <typename Parent>
class StructuralMutateDynEngine;

struct UnchangedOrUnsafe;

template <typename T>
inline constexpr bool is_unchanged_or_v = false;

template <typename T>
inline constexpr bool is_unchanged_or_v<UnchangedOr<T>> = true;
}  // namespace detail

/*! \brief Tag indicating that the caller should retain the original value. */
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
    return detail::AnyUnsafe::MoveTVMFFIAnyRawToAny(raw);
  }
};

/*!
 * \brief A structural-mutation result containing a replacement or an instruction
 *        to retain the original value.
 *
 * An Unchanged result indicates that no replacement is needed.
 * A stored replacement may also refer to the original object.
 *
 * \tparam T The replacement value type.
 * \note In-place updates retain the input's identity, so
 *       ``UnchangedOrSameAs`` may return true despite structural changes.
 *       If structural change detection is needed, track changes separately
 *       or disable in-place mutation.
 * \note ``UnchangedOr`` is deliberately designed to only have rvalue-qualified value accessors,
 *       so the compiler forces a value to leave the container exactly once, via a move.
 *
 * \code{.cpp}
 * // resolves to the original when descent returned the Unchanged marker
 * copy->a = std::move(a).ValueOrUnchanged(std::move(copy->a));
 * // already known to contain a replacement, so no original is needed
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
   * \brief Construct a result containing a replacement value.
   * \param value The replacement value.
   */
  // NOLINTNEXTLINE(google-explicit-constructor,runtime/explicit)
  TVM_FFI_INLINE UnchangedOr(T value) : data_(Any(std::move(value))) {}

  /*!
   * \brief Construct a result containing an implicitly convertible replacement value.
   * \tparam U Source value type, implicitly convertible to T.
   * \param value The replacement value to copy or move.
   */
  // Preserve the dedicated tag and wrapper routes. Subsumption applies only to materialized
  // wrapper storage; a bare value must first be implicitly convertible to T.
  template <typename U, typename = std::enable_if_t<!std::is_same_v<std::decay_t<U>, Unchanged> &&
                                                    !detail::is_unchanged_or_v<std::decay_t<U>> &&
                                                    !detail::is_expected_v<std::decay_t<U>> &&
                                                    !detail::is_unexpected_v<std::decay_t<U>> &&
                                                    !std::is_base_of_v<Error, std::decay_t<U>> &&
                                                    std::is_convertible_v<U, T>>>
  // NOLINTNEXTLINE(google-explicit-constructor,runtime/explicit)
  TVM_FFI_INLINE UnchangedOr(U&& value) : data_(Any(T(std::forward<U>(value)))) {}

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
            return detail::AnyUnsafe::MoveTVMFFIAnyRawToAny(
                detail::AnyUnsafe::MoveAnyToTVMFFIAny(std::move(other.data_)));
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
   * \brief Whether this result holds the Unchanged marker, retaining the original value.
   * \return Whether the result holds the Unchanged marker.
   */
  TVM_FFI_INLINE bool IsUnchanged() const& noexcept {
    return data_.type_index() == TypeIndex::kTVMFFIUnchanged;
  }

  /*!
   * \brief Whether this result holds the Unchanged marker or a value that compares
   *        ``same_as`` to \p original.
   * \param original The original value.
   * \return Whether the original value may be retained without replacement.
   * \note For objects, this checks identity, not contents or subtree changes.
   */
  TVM_FFI_INLINE bool UnchangedOrSameAs(const T& original) const& noexcept {
    return IsUnchanged() || data_.same_as(original);
  }

  /*!
   * \brief Move the replacement, or copy \p original when unchanged.
   * \param original The borrowed original value, which is left unmodified.
   * \return The replacement or original value.
   * \note Both mutable and const lvalues are borrowed. Use std::move(original) to transfer
   * ownership.
   */
  TVM_FFI_INLINE T ValueOrUnchanged(const T& original) && {
    return IsUnchanged() ? original : detail::AnyUnsafe::MoveFromAnyAfterCheck<T>(std::move(data_));
  }

  /*!
   * \brief Move the replacement, or move \p original when unchanged.
   * \param original The owned original value.
   * \return The replacement or original value.
   * \note The original is moved from only when the result is unchanged.
   */
  TVM_FFI_INLINE T ValueOrUnchanged(T&& original) && {
    return IsUnchanged() ? std::move(original)
                         : detail::AnyUnsafe::MoveFromAnyAfterCheck<T>(std::move(data_));
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
                         : detail::AnyUnsafe::MoveFromAnyAfterCheck<Any>(std::move(data_));
  }

  /*!
   * \brief Move the stored replacement value without checking its state.
   * \return The replacement value.
   * \pre ``!IsUnchanged()``.
   */
  TVM_FFI_INLINE T ValueUnchecked() && {
    return detail::AnyUnsafe::MoveFromAnyAfterCheck<T>(std::move(data_));
  }

  /*!
   * \brief Strictly cast the stored result, moving it on success.
   * \tparam U The exact target type, including any UnchangedOr wrapper.
   * \return The cast value, or std::nullopt on a type mismatch.
   * \note Unchanged succeeds for an UnchangedOr target. No fallback conversions are run.
   */
  template <typename U,
            typename = std::enable_if_t<TypeTraits<U>::storage_enabled || std::is_same_v<U, Any>>>
  TVM_FFI_INLINE std::optional<U> as() && {
    return std::move(data_).template as<U>();
  }

  /*!
   * \brief Strictly cast the stored result, moving it on success, or throw.
   * \tparam U The exact target type, including any UnchangedOr wrapper.
   * \return The cast value.
   * \throws Error on a type mismatch.
   * \note Unchanged succeeds for an UnchangedOr target. No fallback conversions are run.
   */
  template <typename U,
            typename = std::enable_if_t<TypeTraits<U>::storage_enabled || std::is_same_v<U, Any>>>
  TVM_FFI_INLINE U as_or_throw() && {
    return std::move(data_).template as_or_throw<U>();
  }

 private:
  template <typename>
  friend class UnchangedOr;
  friend struct detail::UnchangedOrUnsafe;
  template <typename, typename>
  friend struct TypeTraits;
  struct UnsafeInit {};
  TVM_FFI_INLINE explicit UnchangedOr(UnsafeInit, Any data) noexcept : data_(std::move(data)) {}
  Any data_;
};

namespace detail {
/*! \brief Unsafe moves between UnchangedOr and its single Any storage. */
struct UnchangedOrUnsafe {
  /*!
   * \brief Adopt an owning raw FFI value into UnchangedOr storage.
   * \tparam T The replacement type.
   * \param raw The raw FFI value whose ownership is transferred to the result.
   * \return UnchangedOr backed by the adopted Any storage.
   * \pre The caller guarantees that raw holds Unchanged or a valid T replacement.
   * \note No type check, reference increment, or value conversion is performed.
   */
  template <typename T>
  TVM_FFI_INLINE static UnchangedOr<T> MoveFromTVMFFIAny(TVMFFIAny raw) {
    return UnchangedOr<T>(typename UnchangedOr<T>::UnsafeInit{},
                          AnyUnsafe::MoveTVMFFIAnyRawToAny(raw));
  }

  template <typename T>
  TVM_FFI_INLINE static TVMFFIAny MoveToTVMFFIAny(UnchangedOr<T>&& result) noexcept {
    return AnyUnsafe::MoveAnyToTVMFFIAny(std::move(result.data_));
  }
};

}  // namespace detail

namespace detail {
// Out of line so its strings and Error construction stay out of the hot path of whatever hook
// body TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN expands into. Same reason as
// BadStructuralMutateHookError.
// Takes nothing on purpose. Naming the offending type in the message would keep the result live
// across the predicted-not-taken guard in the hot path. The declared type is already present in
// the source line to which the diagnostic points.
TVM_FFI_COLD_CODE inline Unexpected<Error> SMutateDeclaredTypeError() noexcept {
  return Unexpected(
      Error("TypeError", "structural mutate result does not match the declared type", ""));
}
}  // namespace detail

/*! \brief Borrowed structural mutation hook at attribute lookup and call. */
using FStructuralMutate =
    reflection::NativeFunctionView<UnchangedOr<Any>(StructuralMutatorObj*, AnyView)>;

namespace type_attr {
/*!
 * \brief Custom structural mutation hook used by ``StructuralMutator``.
 *
 * The hook receives the active mutator and the input value. It should recursively mutate structural
 * children through the mutator, return ``Unchanged`` if no fields are changed, and avoid
 * intentionally mutating the source object. The input is borrowed and remains valid for the
 * duration of the call.
 *
 * Value type: either an opaque native hook pointer returning raw
 * ``Expected<UnchangedOr<Any>>`` storage, or an ``ffi::Function``.
 *
 * This is the canonical mutation hook. A type that defines
 * ``kStructuralMaybeInplaceMutate`` must also define this attribute.
 */
inline constexpr reflection::AttrKey<FStructuralMutate> kStructuralMutate{"__s_mutate__"};
/*!
 * \brief Optional custom mutation hook that may reuse its input object.
 *
 * The hook receives the active mutator and a borrowed input value that is safe to mutate in place.
 * Callers must route shared objects through non-in-place mutation instead. The hook may mutate and
 * return the source object, delegate to non-in-place mutation, or return another replacement.
 *
 * Value type: either an opaque native hook pointer returning raw
 * ``Expected<UnchangedOr<Any>>`` storage, or an ``ffi::Function``.
 *
 * This hook is optional. When it is absent, ``DefaultMutateExpected`` with
 * ``inplace_mode=InplaceMode::kAllow`` falls back to non-in-place mutation through
 * ``kStructuralMutate`` or reflected structural fields. In-place mutation is therefore explicitly
 * opt-in and is never inferred from ownership by the reflected fallback.
 */
inline constexpr reflection::AttrKey<FStructuralMutate> kStructuralMaybeInplaceMutate{
    "__s_maybe_inplace_mutate__"};
}  // namespace type_attr

/*!
 * \brief Object node of a structural mutator.
 */
class StructuralMutatorObj : public Object {
 public:
  /*! \brief Callback-facing mutator type used by composed callback-driven engines. */
  using MutatorObjType = StructuralMutatorObj;

  /*!
   * \brief Throwing form of \ref MutateExpected.
   * \param value The borrowed value to mutate.
   * \param inplace_mode Whether the caller permits mutation along this ownership path.
   * \return The replacement or unchanged marker.
   * \throws Error if mutation fails.
   *
   * \note The default InplaceMode::kDisallow uses copy-on-write. In-place mutation is permitted
   *       only when inplace_mode is InplaceMode::kAllow and the current value is uniquely owned.
   *       See \ref MutateExpected for the full permission and error semantics.
   *
   * Use \ref TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN in Expected-returning hooks and helpers to
   * preserve checked typed mutation results and the fixed mismatch diagnostic. At a throwing
   * boundary, an explicit cast may instead report its own TypeError:
   * \code{.cpp}
   * Expr new_node = mutator->Mutate(node).ValueOrUnchanged(AnyView(node)).cast<Expr>();
   * \endcode
   */
  TVM_FFI_INLINE UnchangedOr<Any> Mutate(AnyView value,
                                         InplaceMode inplace_mode = InplaceMode::kDisallow) {
    return std::move(MutateExpected(value, inplace_mode)).value();
  }

  /*!
   * \brief Mutate a value, permitting in-place mutation only when uniquely owned.
   * \param value The borrowed value to mutate.
   * \param inplace_mode Whether the caller permits mutation along this ownership path.
   * \return The replacement or unchanged marker, or an Error if mutation failed.
   *
   * \note The default InplaceMode::kDisallow uses copy-on-write. In-place mutation is permitted
   *       only when inplace_mode is InplaceMode::kAllow and the current value is uniquely owned.
   *       InplaceMode::kAllow requires permission along the entire path from the root.
   *       Recursive calls must forward their established mode explicitly. Uniqueness is checked
   *       before callback arguments acquire ownership. This permits in-place dispatch but does
   *       not guarantee reuse: a hook may return a replacement. In-place changes completed before
   *       an Error are not rolled back.
   */
  TVM_FFI_INLINE Expected<UnchangedOr<Any>> MutateExpected(
      AnyView value, InplaceMode inplace_mode = InplaceMode::kDisallow) noexcept {
    const Object* object = value.as<Object>();
    // Check uniqueness on the borrowed view before callbacks can acquire owning references.
    if (inplace_mode == InplaceMode::kAllow && object != nullptr && object->unique()) {
      return detail::ExpectedUnsafe::MoveFromTVMFFIAny<UnchangedOr<Any>>(
          (*vtable_->maybe_inplace_mutate)(this, value));
    }
    return detail::ExpectedUnsafe::MoveFromTVMFFIAny<UnchangedOr<Any>>(
        (*vtable_->mutate)(this, value));
  }

  /*!
   * \brief Apply default structural mutation with validated in-place permission.
   * \param value The borrowed current value to mutate.
   * \param inplace_mode The in-place mode already established by the caller for value.
   * \return The replacement or unchanged marker, or an Error if mutation failed.
   *
   * \note The default InplaceMode::kDisallow is a copy-on-write convenience for one-off calls.
   *       Recursive code and hooks must explicitly forward the mode established for the current
   *       value. This method uses that mode without checking uniqueness again, even if a typed
   *       callback argument has acquired another reference. It bypasses the current engine
   *       callback. Permission must cover the entire path from the root.
   *       Without an in-place hook, ordinary mutation runs. In-place changes completed before an
   *       Error are not rolled back. Registered hooks own variable-remap handling; the reflected
   *       fallback applies it automatically and always uses copy-on-write mutation.
   */
  TVM_FFI_INLINE Expected<UnchangedOr<Any>> DefaultMutateExpected(
      AnyView value, InplaceMode inplace_mode = InplaceMode::kDisallow) noexcept {
    return detail::ExpectedUnsafe::MoveFromTVMFFIAny<UnchangedOr<Any>>(
        inplace_mode == InplaceMode::kAllow ? DefaultMaybeInplaceMutateRaw(value)
                                            : DefaultMutateRaw(value));
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
    return detail::ExpectedUnsafe::MoveFromTVMFFIAny<Any>((*vtable_->var_remap_get)(this, var));
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
    return detail::ExpectedUnsafe::MoveFromTVMFFIAny<void>(
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
  friend class detail::StructuralMutateDynEngine;

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
        Error("TypeError", "__s_mutate__ must be a direct hook pointer or ffi.Function", ""));
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
    static reflection::TypeAttrColumn column(type_attr::kStructuralMutate);
    AnyView attr = column[value.type_index()];
    // Exactly one frame per node: hooks propagate errors untouched, and this is the engine
    // dispatching into `value`, so both exits below name it here and nowhere else.
    TVMFFIAny result;
    if (TVM_FFI_PREDICT_TRUE(attr.type_index() == TypeIndex::kTVMFFIOpaquePtr)) {
      result = (*reinterpret_cast<FStructuralMutateABI>(attr.cast<void*>()))(this, value);
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
    detail::UpdateVisitErrorContext(result, value);
    return result;
  }

  /*! \brief The cold remainder of DefaultMutateRaw: an ffi.Function hook, or no hook at all. */
  TVMFFIAny DefaultMutateRawTail(AnyView value, AnyView attr) noexcept {
    if (attr.type_index() != TypeIndex::kTVMFFINone) {
      // Registered as a typed function pointer or ffi.Function.
      if (attr.type_index() == TypeIndex::kTVMFFIFunction) {
        return detail::ExpectedUnsafe::MoveToTVMFFIAny(
            attr.cast<Function>().CallExpected<Any>(this, value));
      }
      // Registered as neither: a malformed hook.
      return detail::ExpectedUnsafe::MoveToTVMFFIAny(BadStructuralMutateHookError());
    }
    // No hook at all. A POD carries through unchanged; an object walks its reflected fields.
    if (value.type_index() < TypeIndex::kTVMFFIStaticObjectBegin) {
      return detail::AnyUnsafe::MoveAnyToTVMFFIAny(Any(value));
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
      if (detail::ExpectedUnsafe::GetData(mapped).type_index() != TypeIndex::kTVMFFINone) {
        return detail::ExpectedUnsafe::MoveToTVMFFIAny(std::move(mapped));
      }
    }

    // A FreeVar outside a definition region is a use. A miss means its definition was unchanged
    // (or it is free), so there is no field descent and no remap insertion.
    if (is_free_var && def_region_kind() == kTVMFFIDefRegionKindNone) {
      return Unchanged().CopyToTVMFFIAny();
    }

    Expected<Any> result = detail::MutateReflectedFieldsExpected(this, value);
    if (TVM_FFI_PREDICT_FALSE(result.is_err())) {
      return detail::ExpectedUnsafe::MoveToTVMFFIAny(std::move(result));
    }
    if (is_free_var || is_dag_node) {
      const Any& result_value = detail::ExpectedUnsafe::GetData(result);
      // Bind the descent result. The one exception is an unchanged simple definition: its
      // uses resolve to the var itself on a miss, so there is nothing to record.
      if (is_dag_node || def_region_kind() == kTVMFFIDefRegionKindPattern ||
          result_value.type_index() != TypeIndex::kTVMFFIUnchanged) {
        Expected<void> set_result = VarRemapSetExpected(value, result_value);
        if (TVM_FFI_PREDICT_FALSE(set_result.is_err())) {
          return detail::ExpectedUnsafe::MoveToTVMFFIAny(
              Expected<Any>(Unexpected(std::move(set_result).error())));
        }
      }
    }
    return detail::ExpectedUnsafe::MoveToTVMFFIAny(std::move(result));
  }
  /*!
   * \brief Raw default maybe-in-place mutation.
   *
   * \note A registered opaque hook is the expected case here too, so the attribute is read once
   *       and every other shape is handed to the out-of-line remainder.
   */
  TVM_FFI_INLINE TVMFFIAny DefaultMaybeInplaceMutateRaw(AnyView value) noexcept {
    static reflection::TypeAttrColumn column(type_attr::kStructuralMaybeInplaceMutate);
    AnyView attr = column[value.type_index()];
    if (TVM_FFI_PREDICT_TRUE(attr.type_index() == TypeIndex::kTVMFFIOpaquePtr)) {
      // This is the engine dispatching into `value`; hooks propagate errors untouched, so the
      // node is named here. The fall-through re-dispatches the same node through
      // DefaultMutateRaw, which names it there instead -- exactly one frame either way.
      TVMFFIAny result = (*reinterpret_cast<FStructuralMutateABI>(attr.cast<void*>()))(this, value);
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
      TVMFFIAny result = detail::ExpectedUnsafe::MoveToTVMFFIAny(
          attr.cast<Function>().CallExpected<Any>(this, value));
      if (TVM_FFI_PREDICT_FALSE(result.type_index == TypeIndex::kTVMFFIError)) {
        return AttachVisitErrorContextRaw(result, value);
      }
      return result;
    }
    return detail::ExpectedUnsafe::MoveToTVMFFIAny(
        DefaultMutateExpected(value, InplaceMode::kDisallow));
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
  explicit StructuralMutator(ObjectPtr<StructuralMutatorObj> n) : ObjectRef(std::move(n)) {
    TVM_FFI_ICHECK_NOTNULL(data_.get());
  }

  /// \cond Doxygen_Suppress
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(StructuralMutator, ObjectRef, StructuralMutatorObj);
  /// \endcond
};

namespace detail {

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

  static reflection::TypeAttrColumn column(type_attr::kShallowCopy);
  AnyView attr = column[type_index];
  if (TVM_FFI_PREDICT_FALSE(attr.type_index() != TypeIndex::kTVMFFIFunction)) {
    return Unexpected(Error("TypeError", "__ffi_shallow_copy__ must be an ffi.Function", ""));
  }

  Expected<Any> result = attr.cast<Function>().CallExpected<Any>(value);
  if (TVM_FFI_PREDICT_FALSE(result.is_err())) {
    return result;
  }

  const Any& result_value = detail::ExpectedUnsafe::GetData(result);
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
            result = Unexpected(detail::MoveFromSafeCallRaised());
            return true;
          }

          // Reflected fields use the same unchanged-or-value descent protocol.
          Expected<UnchangedOr<Any>> mutated_field = [&]() -> Expected<UnchangedOr<Any>> {
            if (field_info->flags & kTVMFFIFieldFlagBitMaskSEqHashDefSimple) {
              return mutator->WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&]() {
                return mutator->MutateExpected(field_value, InplaceMode::kDisallow);
              });
            } else if (field_info->flags & kTVMFFIFieldFlagBitMaskSEqHashDefPattern) {
              return mutator->WithDefRegionKind(kTVMFFIDefRegionKindPattern, [&]() {
                return mutator->MutateExpected(field_value, InplaceMode::kDisallow);
              });
            } else {
              return mutator->MutateExpected(field_value, InplaceMode::kDisallow);
            }
          }();
          if (TVM_FFI_PREDICT_FALSE(mutated_field.is_err())) {
            result = Unexpected(std::move(mutated_field).error());
            return true;
          }
          const Any& mutated_field_data = detail::ExpectedUnsafe::GetData(mutated_field);
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
            result = Unexpected(detail::MoveFromSafeCallRaised());
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

}  // namespace detail

// ---------------------------------------------------------------------------
// Structural Map API.
// ---------------------------------------------------------------------------

namespace detail {
/// \cond Doxygen_Suppress
// Return an error from the current Expected-returning mutation function.
#define TVM_FFI_S_MUTATE_MAYBE_EARLY_RETURN(Result)                     \
  do {                                                                  \
    auto&& tvm_ffi_res_ = (Result);                                     \
    if (TVM_FFI_PREDICT_FALSE(tvm_ffi_res_.is_err())) {                 \
      return ::tvm::ffi::Unexpected(::std::move(tvm_ffi_res_).error()); \
    }                                                                   \
  } while (0)

/// \endcond

/// \cond Doxygen_Suppress
#define TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN_IMPL_(Result, Type, Name, ResultExpr)               \
  auto Result = (ResultExpr); /* NOLINT(bugprone-macro-parentheses) */                        \
  TVM_FFI_S_MUTATE_MAYBE_EARLY_RETURN(Result);                                                \
  if constexpr (!::tvm::ffi::type_subsumes_v<::tvm::ffi::Expected<Type>, decltype(Result)>) { \
    if (TVM_FFI_PREDICT_FALSE(!::tvm::ffi::detail::AnyUnsafe::CheckAnyStrict<Type>(           \
            ::tvm::ffi::detail::ExpectedUnsafe::GetData(Result)))) {                          \
      return ::tvm::ffi::detail::SMutateDeclaredTypeError();                                  \
    }                                                                                         \
  }                                                                                           \
  Type Name = /* NOLINT(bugprone-macro-parentheses) */                                        \
      ::tvm::ffi::detail::AnyUnsafe::MoveFromAnyAfterCheck<Type>(                             \
          ::std::move(::tvm::ffi::detail::ExpectedUnsafe::GetData(Result)))
/// \endcond

/*!
 * \brief Unwrap a successful mutation result into a newly declared value or return its error.
 *
 * ``Type`` must be concrete; use a type alias when it contains a top-level comma. A type mismatch
 * returns ``TypeError`` through the surrounding ``Expected`` function without throwing,
 * reported with a fixed string. The check is omitted when the declared type subsumes the result's
 * success type. Its early returns work from an ``Expected<T>`` hook or helper, including one
 * with a different success type. This macro declares
 * ``Name`` into the enclosing scope and must be used in a braced block, never as an unbraced
 * control-flow body.
 *
 * Example:
 * \code{.cpp}
 * Expected<UnchangedOr<Any>> FooMutate(StructuralMutatorObj* mutator, AnyView value) noexcept {
 *   const FooNode* self =
 *       detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const FooNode>(value);
 *   constexpr InplaceMode inplace_mode = InplaceMode::kDisallow;
 *   TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(UnchangedOr<Expr>, a,
 *                                     mutator->MutateExpected(self->a, inplace_mode));
 *   TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(UnchangedOr<Expr>, b,
 *                                     mutator->MutateExpected(self->b, inplace_mode));
 *   if (a.UnchangedOrSameAs(self->a) && b.UnchangedOrSameAs(self->b)) {
 *     return Unchanged();
 *   }
 *   ObjectPtr<FooNode> copy = make_object<FooNode>(*self);
 *   copy->a = std::move(a).ValueOrUnchanged(std::move(copy->a));
 *   copy->b = std::move(b).ValueOrUnchanged(std::move(copy->b));
 *   return copy;
 * }
 * FStructuralMutate hook =
 * FStructuralMutate::FromNative<&FooMutate>();
 * Expected<UnchangedOr<Any>> result = hook.CallExpected(mutator, value);
 * \endcode
 *
 * Keep one statement per traversed field. A field skipped intentionally must be guarded and carry
 * a ``// skips:`` note.
 *
 * \param Type The concrete successful value type.
 * \param Name The name of the value declared in the enclosing scope.
 * \param ResultExpr An expression producing the ``Expected`` value to unwrap.
 * \sa Unchanged
 */
#define TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(Type, Name, ResultExpr)                                  \
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN_IMPL_(TVM_FFI_STR_CONCAT(tvm_ffi_mutate_result_, __COUNTER__), \
                                          Type, Name, ResultExpr)

/// \cond Doxygen_Suppress
#define TVM_FFI_UNSAFE_S_MUTATE_ASSIGN_OR_RETURN_IMPL_(Result, Type, Name, ResultExpr) \
  auto Result = (ResultExpr); /* NOLINT(bugprone-macro-parentheses) */                 \
  TVM_FFI_S_MUTATE_MAYBE_EARLY_RETURN(Result);                                         \
  Type Name = /* NOLINT(bugprone-macro-parentheses) */                                 \
      ::tvm::ffi::detail::AnyUnsafe::MoveFromAnyAfterCheck<Type>(                      \
          ::std::move(::tvm::ffi::detail::ExpectedUnsafe::GetData(Result)))
/// \endcond

/*!
 * \brief Unwrap a successful mutation result when the caller guarantees it to be ``Type``.
 *
 * The caller must guarantee the successful result has the declared type. No runtime type check
 * is performed, including in debug builds; violating this contract is undefined behavior.
 *
 * \param Type The guaranteed concrete type of the successful value.
 * \param Name The name of the value declared in the enclosing scope.
 * \param ResultExpr An expression producing the ``Expected`` value to unwrap.
 * \sa TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN
 */
#define TVM_FFI_UNSAFE_S_MUTATE_ASSIGN_OR_RETURN(Type, Name, ResultExpr) \
  TVM_FFI_UNSAFE_S_MUTATE_ASSIGN_OR_RETURN_IMPL_(                        \
      TVM_FFI_STR_CONCAT(tvm_ffi_mutate_result_, __COUNTER__), Type, Name, ResultExpr)

}  // namespace detail

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
      detail::ObjectUnsafe::DecRefObjectHandle(
          reinterpret_cast<TVMFFIObjectHandle>(const_cast<Object*>(kv.first)));
    }
  }

 protected:
  /*! \brief Return the empty state tuple exposed to typed map callbacks. */
  TVM_FFI_INLINE StateTupleType StateTuple() const noexcept { return {}; }

  /// \cond Doxygen_Suppress
  // Out of line so its strings stay out of the per-node dispatch function, which TryLink inlines
  // into. Shared by the typed and dynamic engines below.
  TVM_FFI_COLD_CODE static Expected<Any> SMutateDescentTypeError() noexcept {
    return Unexpected(Error("TypeError", "structural mutate: descent changed the node type", ""));
  }

  TVM_FFI_COLD_CODE static Unexpected<Error> VarRemapKeyTypeError() noexcept {
    return Unexpected(Error("TypeError", "Variable-remap key must be an object-backed value", ""));
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
    return detail::ExpectedUnsafe::MoveToTVMFFIAny(self->VarRemapGetImpl(var));
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
    return detail::ExpectedUnsafe::MoveToTVMFFIAny(self->VarRemapSetImpl(var, mapped_value));
  }

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
      ::tvm::ffi::detail::UpdateVisitErrorContext(err, node.cast<ObjectRef>());
    }
  }

  /*!
   * \brief Look up a replacement in the identity-substitution environment.
   * \param var The borrowed variable identity to look up.
   * \return The owning replacement, FFI None on a miss, or an Error.
   */
  Expected<Any> VarRemapGetImpl(AnyView var) noexcept {
    if (TVM_FFI_PREDICT_FALSE(var.type_index() < TypeIndex::kTVMFFIStaticObjectBegin)) {
      return VarRemapKeyTypeError();
    }
    if (var_remap_.empty()) return Any(nullptr);
    const Object* var_ptr = detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const Object>(var);
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
    const Object* var_ptr = detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const Object>(var);
    Any owned_mapped_value(mapped_value);
    auto [it, inserted] = var_remap_.try_emplace(var_ptr, std::move(owned_mapped_value));
    if (inserted) {
      detail::ObjectUnsafe::IncRefObjectHandle(
          reinterpret_cast<TVMFFIObjectHandle>(const_cast<Object*>(var_ptr)));
    } else {
      it->second = std::move(owned_mapped_value);
    }
    return Expected<void>();
  }

  template <typename Parent, WalkOrder order, typename... Callbacks>
  friend class StructuralMapEngine;
  template <typename Parent, WalkOrder order>
  friend class StructuralMapDynEngine;
  template <typename Parent, typename... Callbacks>
  friend class StructuralMutateEngine;
  template <typename Parent>
  friend class detail::StructuralMutateDynEngine;

  /*! \brief Identity-substitution environment keyed by object identity. */
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
  using ExpectedUnsafe = detail::ExpectedUnsafe;
  using AnyUnsafe = detail::AnyUnsafe;

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
    using FuncInfo = detail::FunctionInfo<std::decay_t<Callback>>;
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
   * \tparam kInplaceMode Whether the caller may mutate a uniquely owned node in place.
   * \tparam Callback The link's callback type.
   * \param callback The link's callback.
   * \param value The borrowed value to test and mutate.
   * \param out Receives the mutated value or Error when the link matched.
   * \return Whether the link matched, in which case \p out was written.
   */
  template <InplaceMode kInplaceMode, typename Callback>
  TVM_FFI_INLINE bool TryLink(Callback& callback, AnyView value, Expected<Any>* out) noexcept {
    using FuncInfo = detail::FunctionInfo<std::decay_t<Callback>>;
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
        if constexpr (kInplaceMode == InplaceMode::kAllow) {
          // A pre-order result can be mutated in place if unchanged or uniquely owned.
          if (descent_view.same_as(value)) {
            return this->DefaultMutateExpected(value, InplaceMode::kAllow);
          }
          const Object* mapped_obj = descent_view.as<Object>();
          InplaceMode inplace_mode = mapped_obj != nullptr && mapped_obj->unique()
                                         ? InplaceMode::kAllow
                                         : InplaceMode::kDisallow;
          return this->DefaultMutateExpected(descent_view, inplace_mode);
        } else {
          return this->DefaultMutateExpected(descent_view, InplaceMode::kDisallow);
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
      Expected<Any> callback_result = [&]() -> Expected<Any> {
        if constexpr (std::is_same_v<TSub, AnyView>) {
          return InvokeTypedCallbackLink(callback, mapped_view, StateIndices{});
        } else if constexpr (std::is_same_v<TSub, Any>) {
          return InvokeTypedCallbackLink(callback, Any(mapped_view), StateIndices{});
        } else {
          return InvokeTypedCallbackLink(callback, *std::move(matched), StateIndices{});
        }
      }();
      if (TVM_FFI_PREDICT_FALSE(callback_result.is_err())) {
        this->UpdateVisitErrorContext(callback_result, mapped_view);
        *out = std::move(callback_result);
        return true;
      }
      // Unchanged refers to the descended node, whose owner stays live through the callback.
      if (ExpectedUnsafe::GetData(callback_result).type_index() != TypeIndex::kTVMFFIUnchanged) {
        *out = std::move(callback_result);
      }
      return true;
    }
  }

  /*!
   * \brief Test every link in declaration order, stopping at the first that matches.
   * \return Whether some link matched, in which case \p out was written.
   */
  template <InplaceMode kInplaceMode, size_t... Is>
  TVM_FFI_INLINE bool TryLinks(AnyView value, Expected<Any>* out,
                               std::index_sequence<Is...>) noexcept {
    return (TryLink<kInplaceMode>(std::get<Is>(callbacks_), value, out) || ...);
  }

  /*!
   * \brief Mutate a value, invoking the first matching callback link.
   * \param value The borrowed value to mutate.
   * \return Raw ``TVMFFIAny`` containing the mutated value or Error.
   */
  TVM_FFI_INLINE TVMFFIAny MutateImplRaw(AnyView value) noexcept {
    Expected<Any> out{Any()};
    if constexpr (order == WalkOrder::kPostOrder) {
      out = this->DefaultMutateExpected(value, InplaceMode::kDisallow);
      if (TVM_FFI_PREDICT_FALSE(out.is_err())) {
        return ExpectedUnsafe::MoveToTVMFFIAny(std::move(out));
      }
      TryLinks<InplaceMode::kDisallow>(value, &out, std::index_sequence_for<Callbacks...>{});
      return ExpectedUnsafe::MoveToTVMFFIAny(std::move(out));
    } else {
      if (TryLinks<InplaceMode::kDisallow>(value, &out, std::index_sequence_for<Callbacks...>{})) {
        return ExpectedUnsafe::MoveToTVMFFIAny(std::move(out));
      }
      return ExpectedUnsafe::MoveToTVMFFIAny(
          this->DefaultMutateExpected(value, InplaceMode::kDisallow));
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
      out = this->DefaultMutateExpected(value, InplaceMode::kAllow);
      if (TVM_FFI_PREDICT_FALSE(out.is_err())) {
        return ExpectedUnsafe::MoveToTVMFFIAny(std::move(out));
      }
      TryLinks<InplaceMode::kAllow>(value, &out, std::index_sequence_for<Callbacks...>{});
      return ExpectedUnsafe::MoveToTVMFFIAny(std::move(out));
    } else {
      if (TryLinks<InplaceMode::kAllow>(value, &out, std::index_sequence_for<Callbacks...>{})) {
        return ExpectedUnsafe::MoveToTVMFFIAny(std::move(out));
      }
      return ExpectedUnsafe::MoveToTVMFFIAny(
          this->DefaultMutateExpected(value, InplaceMode::kAllow));
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
  using ExpectedUnsafe = detail::ExpectedUnsafe;
  using AnyUnsafe = detail::AnyUnsafe;

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
      if (detail::RuntimeTypeIndexMatch(type_index, entry.get<0>())) {
        *with_kind = false;
        return entry.get<1>();
      }
    }
    for (const Tuple<int32_t, Function>& entry : callbacks_with_def_region_kind_) {
      if (detail::RuntimeTypeIndexMatch(type_index, entry.get<0>())) {
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
   * \tparam kInplaceMode Whether a uniquely owned node may be mutated in place.
   * \param value The borrowed value to test and mutate.
   * \param out Receives the mutated value or Error when a link matched.
   * \return Whether a link matched, in which case \p out was written.
   */
  template <InplaceMode kInplaceMode>
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
        if constexpr (kInplaceMode == InplaceMode::kAllow) {
          if (descent_view.same_as(value)) {
            return this->DefaultMutateExpected(value, InplaceMode::kAllow);
          }
          const Object* mapped_obj = descent_view.as<Object>();
          InplaceMode inplace_mode = mapped_obj != nullptr && mapped_obj->unique()
                                         ? InplaceMode::kAllow
                                         : InplaceMode::kDisallow;
          return this->DefaultMutateExpected(descent_view, inplace_mode);
        } else {
          return this->DefaultMutateExpected(descent_view, InplaceMode::kDisallow);
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
      Expected<Any> callback_result =
          InvokeLink(*matched, with_kind, mapped_view, this->def_region_kind());
      if (TVM_FFI_PREDICT_FALSE(callback_result.is_err())) {
        this->UpdateVisitErrorContext(callback_result, mapped_view);
        *out = std::move(callback_result);
        return true;
      }
      // Preserve descent and its lifetime just as in the typed engine.
      if (ExpectedUnsafe::GetData(callback_result).type_index() != TypeIndex::kTVMFFIUnchanged) {
        *out = std::move(callback_result);
      }
      return true;
    }
  }

  /*! \brief Mutate a value, invoking the first matching runtime link. */
  TVM_FFI_INLINE TVMFFIAny MutateImplRaw(AnyView value) noexcept {
    Expected<Any> out{Any()};
    if constexpr (order == WalkOrder::kPostOrder) {
      out = this->DefaultMutateExpected(value, InplaceMode::kDisallow);
      if (TVM_FFI_PREDICT_FALSE(out.is_err())) {
        return ExpectedUnsafe::MoveToTVMFFIAny(std::move(out));
      }
      TryLink<InplaceMode::kDisallow>(value, &out);
      return ExpectedUnsafe::MoveToTVMFFIAny(std::move(out));
    } else {
      if (TryLink<InplaceMode::kDisallow>(value, &out)) {
        return ExpectedUnsafe::MoveToTVMFFIAny(std::move(out));
      }
      return ExpectedUnsafe::MoveToTVMFFIAny(
          this->DefaultMutateExpected(value, InplaceMode::kDisallow));
    }
  }

  /*! \brief Optionally mutate a value in place through the first matching runtime link. */
  TVM_FFI_INLINE TVMFFIAny MaybeInplaceMutateImplRaw(AnyView value) noexcept {
    Expected<Any> out{Any()};
    if constexpr (order == WalkOrder::kPostOrder) {
      out = this->DefaultMutateExpected(value, InplaceMode::kAllow);
      if (TVM_FFI_PREDICT_FALSE(out.is_err())) {
        return ExpectedUnsafe::MoveToTVMFFIAny(std::move(out));
      }
      TryLink<InplaceMode::kAllow>(value, &out);
      return ExpectedUnsafe::MoveToTVMFFIAny(std::move(out));
    } else {
      if (TryLink<InplaceMode::kAllow>(value, &out)) {
        return ExpectedUnsafe::MoveToTVMFFIAny(std::move(out));
      }
      return ExpectedUnsafe::MoveToTVMFFIAny(
          this->DefaultMutateExpected(value, InplaceMode::kAllow));
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
      return self->template MutateSingleCallbackRaw<InplaceMode::kDisallow>(value);
    } else {
      return self->MutateImplRaw(value);
    }
  }

  /*! \brief Dispatch maybe-in-place mutation from the erased mutator pointer. */
  static TVMFFIAny DispatchMaybeInplaceMutate(StructuralMutatorObj* mutator,
                                              AnyView value) noexcept {
    auto* self = static_cast<StructuralMutateEngine*>(mutator);
    if constexpr (sizeof...(Callbacks) == 1) {
      return self->template MutateSingleCallbackRaw<InplaceMode::kAllow>(value);
    } else {
      return self->MaybeInplaceMutateImplRaw(value);
    }
  }

  /*! \brief Mutate one value, handing a matched callback ownership of descent. */
  TVMFFIAny MutateImplRaw(AnyView value) noexcept {
    Expected<Any> result{Any()};
    if (DispatchCallbacks(value, InplaceMode::kDisallow, &result)) {
      if (TVM_FFI_PREDICT_FALSE(result.is_err())) {
        // Keep callback-boundary context in addition to the default-descent
        // context: a callback may return a rebuilt value, so the two nodes can differ.
        Parent::UpdateVisitErrorContext(result, value);
      }
      return detail::ExpectedUnsafe::MoveToTVMFFIAny(std::move(result));
    }
    return detail::ExpectedUnsafe::MoveToTVMFFIAny(
        Parent::DefaultMutateExpected(value, InplaceMode::kDisallow));
  }

  /*! \brief Maybe mutate one value in place, with callback-owned descent. */
  TVMFFIAny MaybeInplaceMutateImplRaw(AnyView value) noexcept {
    Expected<Any> result{Any()};
    if (DispatchCallbacks(value, InplaceMode::kAllow, &result)) {
      if (TVM_FFI_PREDICT_FALSE(result.is_err())) {
        // Keep callback-boundary context in addition to the default-descent
        // context: a callback may return a rebuilt value, so the two nodes can differ.
        Parent::UpdateVisitErrorContext(result, value);
      }
      return detail::ExpectedUnsafe::MoveToTVMFFIAny(std::move(result));
    }
    return detail::ExpectedUnsafe::MoveToTVMFFIAny(
        Parent::DefaultMutateExpected(value, InplaceMode::kAllow));
  }

  /*! \brief Invoke a matched callback with the Parent view and preserve returned/thrown Error. */
  template <typename Callback, typename Matched>
  TVM_FFI_INLINE Expected<Any> InvokeCallback(Callback& callback, Matched&& matched,
                                              InplaceMode inplace_mode) noexcept {
    using FuncInfo = detail::FunctionInfo<std::decay_t<Callback>>;
    static_assert(FuncInfo::num_args == 2 || FuncInfo::num_args == 3,
                  "StructuralMutate callback must take (value, mutator) or "
                  "(value, mutator, inplace_mode)");
    using SecondArg = std::decay_t<std::tuple_element_t<1, typename FuncInfo::ArgType>>;
    using Second = std::remove_pointer_t<SecondArg>;
    static_assert(std::is_same_v<Second, typename Parent::MutatorObjType>,
                  "second StructuralMutate callback argument must be exactly "
                  "Parent::MutatorObjType*");
    if constexpr (FuncInfo::num_args == 3) {
      using ThirdArg = std::decay_t<std::tuple_element_t<2, typename FuncInfo::ArgType>>;
      static_assert(std::is_same_v<ThirdArg, InplaceMode>,
                    "third StructuralMutate callback argument must be InplaceMode");
    }
    auto* mutator = static_cast<typename Parent::MutatorObjType*>(this);
    try {
      if constexpr (FuncInfo::num_args == 3) {
        return callback(std::forward<Matched>(matched), mutator, inplace_mode);
      } else {
        return callback(std::forward<Matched>(matched), mutator);
      }
    } catch (Error& err) {
      return Unexpected(std::move(err));
    }
  }

  /*! \brief Materialize the callback result only on the cold Parent error-decoration path. */
  TVM_FFI_COLD_CODE TVMFFIAny AnnotateCallbackErrorRaw(TVMFFIAny raw, AnyView value) noexcept {
    Expected<Any> result = detail::ExpectedUnsafe::MoveFromTVMFFIAny<Any>(raw);
    Parent::UpdateVisitErrorContext(result, value);
    return detail::ExpectedUnsafe::MoveToTVMFFIAny(std::move(result));
  }

  // One callback forwards its result directly, without a callback-chain envelope.
  // AnyView/Any always match; only a typed miss needs the Parent's default descent.
  template <InplaceMode kInplaceMode>
  TVM_FFI_INLINE TVMFFIAny MutateSingleCallbackRaw(AnyView value) noexcept {
    auto& callback = std::get<0>(callbacks_);
    using FuncInfo = detail::FunctionInfo<std::decay_t<decltype(callback)>>;
    using FirstArg = std::tuple_element_t<0, typename FuncInfo::ArgType>;
    using TSub = std::remove_cv_t<std::remove_reference_t<FirstArg>>;
    TVMFFIAny result;
    if constexpr (std::is_same_v<TSub, AnyView>) {
      result =
          detail::ExpectedUnsafe::MoveToTVMFFIAny(InvokeCallback(callback, value, kInplaceMode));
    } else if constexpr (std::is_same_v<TSub, Any>) {
      result = detail::ExpectedUnsafe::MoveToTVMFFIAny(
          InvokeCallback(callback, Any(value), kInplaceMode));
    } else if (auto matched = value.template as<TSub>()) {
      result = detail::ExpectedUnsafe::MoveToTVMFFIAny(
          InvokeCallback(callback, *std::move(matched), kInplaceMode));
    } else {
      if constexpr (kInplaceMode == InplaceMode::kAllow) {
        return detail::ExpectedUnsafe::MoveToTVMFFIAny(
            Parent::DefaultMutateExpected(value, InplaceMode::kAllow));
      } else {
        return detail::ExpectedUnsafe::MoveToTVMFFIAny(
            Parent::DefaultMutateExpected(value, InplaceMode::kDisallow));
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
  TVM_FFI_INLINE bool TryLink(Callback& callback, AnyView value, InplaceMode inplace_mode,
                              Expected<Any>* out) noexcept {
    using FuncInfo = detail::FunctionInfo<std::decay_t<Callback>>;
    using FirstArg = std::tuple_element_t<0, typename FuncInfo::ArgType>;
    using TSub = std::remove_cv_t<std::remove_reference_t<FirstArg>>;
    if constexpr (std::is_same_v<TSub, AnyView>) {
      *out = InvokeCallback(callback, value, inplace_mode);
      return true;
    } else if constexpr (std::is_same_v<TSub, Any>) {
      *out = InvokeCallback(callback, Any(value), inplace_mode);
      return true;
    } else if (auto matched = value.template as<TSub>()) {
      *out = InvokeCallback(callback, *std::move(matched), inplace_mode);
      return true;
    }
    return false;
  }

  /*! \brief Fold callbacks in declaration order, stopping at the first match. */
  template <size_t... Is>
  TVM_FFI_INLINE bool TryLinks(AnyView value, InplaceMode inplace_mode, Expected<Any>* out,
                               std::index_sequence<Is...>) noexcept {
    return (TryLink(std::get<Is>(callbacks_), value, inplace_mode, out) || ...);
  }

  /*! \brief Run the callback chain, returning whether a callback matched. */
  TVM_FFI_INLINE bool DispatchCallbacks(AnyView value, InplaceMode inplace_mode,
                                        Expected<Any>* out) noexcept {
    return TryLinks(value, inplace_mode, out, std::index_sequence_for<Callbacks...>{});
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
  auto result = mutator->MutateExpected(root, InplaceMode::kAllow);
  if (TVM_FFI_PREDICT_FALSE(result.is_err())) {
    if (auto object_root = root.as<ObjectRef>()) {
      return Unexpected(
          VisitErrorContext::EnrichVisitError(std::move(result).error(), *object_root));
    }
    return Unexpected(std::move(result).error());
  }
  UnchangedOr<Any> mapped = detail::AnyUnsafe::MoveFromAnyAfterCheck<UnchangedOr<Any>>(
      std::move(detail::ExpectedUnsafe::GetData(result)));
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
 * - ``R(const T& value, StructuralMutatorObj* mutator, InplaceMode inplace_mode)``
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
 * \note A two-argument callback descends with ``MutateExpected(value)``, using the default
 *       copy-on-write permission. A three-argument callback receives InplaceMode::kAllow only
 *       when its value is on a uniquely owned path. Forward this mode to ``MutateExpected`` for
 *       child values or ``DefaultMutateExpected`` for the current value's default descent.
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
  auto result = mutator->MutateExpected(root, InplaceMode::kAllow);
  if (TVM_FFI_PREDICT_FALSE(result.is_err())) {
    if (auto object_root = root.as<ObjectRef>()) {
      return Unexpected(
          VisitErrorContext::EnrichVisitError(std::move(result).error(), *object_root));
    }
    return Unexpected(std::move(result).error());
  }
  UnchangedOr<Any> mapped = detail::AnyUnsafe::MoveFromAnyAfterCheck<UnchangedOr<Any>>(
      std::move(detail::ExpectedUnsafe::GetData(result)));
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
    *result = detail::UnchangedOrUnsafe::MoveToTVMFFIAny(std::move(src));
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
                          detail::AnyUnsafe::MoveTVMFFIAnyToAny(src));
  }
  TVM_FFI_INLINE static std::string TypeStr() {
    return "UnchangedOr<" + detail::Type2Str<T>::v() + ">";
  }
  TVM_FFI_INLINE static std::string TypeSchema() {
    return R"({"type":"UnchangedOr","args":[)" + detail::TypeSchema<T>::v() + "]}";
  }
};

}  // namespace ffi
}  // namespace tvm

#endif  // TVM_FFI_EXTRA_STRUCTURAL_MUTATE_H_
