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
#include <tvm/ffi/container/map.h>
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
#include <utility>

namespace tvm {
namespace ffi {

class StructuralMutatorObj;

/*!
 * \brief ABI callback type for structural mutation.
 *
 * \param mutator The active structural mutator.
 * \param value The borrowed value to mutate.
 * \return Raw ``TVMFFIAny`` containing the mutated value or an Error.
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
   * \return Raw ``TVMFFIAny`` carrying the mutated value or Error.
   */
  FStructuralMutate mutate = nullptr;
  /*!
   * \brief Mutate a value, permitting an in-place implementation when it is safe.
   *
   * \param mutator The active structural mutator.
   * \param value The borrowed value to mutate.
   * \return Raw ``TVMFFIAny`` carrying the mutated value or Error.
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

/*!
 * \brief Object node of a structural mutator.
 */
class StructuralMutatorObj : public Object {
 public:
  /*!
   * \brief Mutate a value through the mutator vtable.
   *
   * \param value The value to mutate.
   * \return The mutated owning value.
   * \throws Error if mutation fails.
   *
   * This entry point never intentionally mutates \p value in place. Recursive mutations
   * also use \ref Mutate.
   */
  TVM_FFI_INLINE Any Mutate(AnyView value) { return MutateExpected(value).value(); }

  /*!
   * \brief Exception-free form of \ref Mutate.
   *
   * \param value The value to mutate.
   * \return The mutated owning value, or an Error if mutation failed.
   */
  TVM_FFI_INLINE Expected<Any> MutateExpected(AnyView value) noexcept {
    return details::ExpectedUnsafe::MoveFromTVMFFIAny<Any>((*vtable_->mutate)(this, value));
  }

  /*!
   * \brief Mutate a value, permitting an in-place implementation when it is safe.
   *
   * \param value The borrowed value to mutate.
   * \return The mutated owning value.
   * \throws Error if mutation fails.
   *
   * The returned value may refer to the same object as \p value. Callers must use the return value
   * as the result of the mutation rather than assuming that the input object was reused.
   */
  TVM_FFI_INLINE Any MaybeInplaceMutate(AnyView value) {
    return MaybeInplaceMutateExpected(value).value();
  }

  /*!
   * \brief Exception-free form of \ref MaybeInplaceMutate.
   *
   * \param value The borrowed value to mutate.
   * \return The mutated owning value, or an Error if mutation failed.
   */
  TVM_FFI_INLINE Expected<Any> MaybeInplaceMutateExpected(AnyView value) noexcept {
    return details::ExpectedUnsafe::MoveFromTVMFFIAny<Any>(
        (*vtable_->maybe_inplace_mutate)(this, value));
  }

  /*!
   * \brief Mutate a value, using in-place mutation only for a uniquely owned object.
   *
   * \param value The borrowed value to mutate.
   * \return The mutated owning value, or an Error if mutation failed.
   */
  TVM_FFI_INLINE Expected<Any> MaybeInplaceMutateIfUniqueExpected(AnyView value) noexcept {
    const Object* obj = value.as<Object>();
    if (obj != nullptr && obj->unique()) {
      return MaybeInplaceMutateExpected(value);
    }
    return MutateExpected(value);
  }

  /*!
   * \brief Apply the default structural mutation with copy-on-write behavior.
   *
   * \param value The value to mutate.
   * \return The mutated value, or an Error if hook dispatch, copying, or field mutation failed.
   *
   * \note A registered ``__s_mutate__`` hook is dispatched before the reflected fallback and is
   *       responsible for variable-remap lookup and insertion when it represents a FreeVar or DAG
   *       identity. Automatic remapping applies only to the reflected fallback.
   */

  TVM_FFI_INLINE Expected<Any> DefaultMutateExpected(AnyView value) noexcept {
    return details::ExpectedUnsafe::MoveFromTVMFFIAny<Any>(DefaultMutateRaw(value));
  }

  /*!
   * \brief Apply custom maybe-in-place mutation, or fall back to non-in-place mutation.
   *
   * \param value The borrowed value to mutate.
   * \return The mutated owning value, or an Error if mutation failed. In-place changes
   *         completed before an Error are not rolled back.
   *
   * \note In-place mutation is explicitly opt-in. A registered
   *       ``__s_maybe_inplace_mutate__`` hook may rely on its input being safe to mutate and owns
   *       any variable-remap handling. When the hook is absent, this method calls
   *       \ref DefaultMutateExpected.
   */
  TVM_FFI_INLINE Expected<Any> DefaultMaybeInplaceMutateExpected(AnyView value) noexcept {
    return details::ExpectedUnsafe::MoveFromTVMFFIAny<Any>(DefaultMaybeInplaceMutateRaw(value));
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
   */
  TVM_FFI_INLINE TVMFFIDefRegionKind def_region_kind() const { return def_region_mode_; }

  /*!
   * \brief Temporarily switch the def-region context while invoking \p callback.
   *
   * \param kind The def-region kind to set during the callback.
   * \param callback A nullary callable that performs recursive mutation.
   * \return The value returned by \p callback.
   */
  template <typename Callback>
  TVM_FFI_INLINE auto WithDefRegionKind(TVMFFIDefRegionKind kind, Callback&& callback)
      -> decltype(std::forward<Callback>(callback)()) {
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

 protected:
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

  // Convention: the ABI boundary is a raw TVMFFIAny; everything inside a callback or hook body
  // works in Expected<Any> and moves out to TVMFFIAny at that boundary.
  //
  // The Raw forms below exist because that boundary is also the default path. A hook is a C-ABI
  // function pointer returning TVMFFIAny, a 16-byte POD that stays in registers; wrapping the
  // result in Expected<Any> would force it to memory, since Expected<Any> is not trivially
  // destructible and is therefore classified MEMORY. Descent through an unmatched node calls a
  // hook and returns its result unchanged, so keeping that path raw removes the round trip
  // entirely. Only a matched callback pays for an Expected<Any>, and it materializes an Any
  // anyway to record its remap entry.
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
      details::UpdateVisitErrorContext(result, value);
    }
    return result;
  }

  // The cold remainder of DefaultMutateRaw, out of line so that always-inlined caller stays
  // small at every inlining site. `attr` is the attribute the caller already read, so the
  // column is never looked up twice.
  /*!
   * \brief Whether a node's identity is remappable, so it maps once and reuses that result.
   * \param type_index The node's runtime type index.
   * \return True for a FreeVar or DAG node.
   */
  TVM_FFI_INLINE static bool IsRemappableIdentity(int32_t type_index) noexcept {
    if (type_index < TypeIndex::kTVMFFIStaticObjectBegin) return false;
    const TVMFFITypeInfo* type_info = TVMFFIGetTypeInfo(type_index);
    return type_info->metadata != nullptr &&
           (type_info->metadata->structural_eq_hash_kind == kTVMFFISEqHashKindFreeVar ||
            type_info->metadata->structural_eq_hash_kind == kTVMFFISEqHashKindDAGNode);
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
    // A FreeVar or DAG node maps once and every later occurrence reuses that result, so the
    // reflected walk runs under an identity remap.
    const bool remappable = IsRemappableIdentity(value.type_index());
    if (remappable) {
      Expected<Any> mapped = VarRemapGetExpected(value);
      if (TVM_FFI_PREDICT_FALSE(mapped.is_err()) ||
          details::ExpectedUnsafe::GetData(mapped).type_index() != TypeIndex::kTVMFFINone) {
        return details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(mapped));
      }
    }
    Expected<Any> result = details::MutateReflectedFieldsExpected(this, value);
    if (TVM_FFI_PREDICT_FALSE(result.is_err())) {
      return details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(result));
    }
    if (remappable) {
      Expected<void> set_result =
          VarRemapSetExpected(value, details::ExpectedUnsafe::GetData(result));
      if (TVM_FFI_PREDICT_FALSE(set_result.is_err())) {
        return details::ExpectedUnsafe::MoveToTVMFFIAny(
            Expected<Any>(Unexpected(std::move(set_result).error())));
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
        details::UpdateVisitErrorContext(result, value);
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
        details::UpdateVisitErrorContext(result, value);
      }
      return result;
    }
    return DefaultMutateRaw(value);
  }

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

          Expected<Any> mutated_field = [&]() -> Expected<Any> {
            if (field_info->flags & kTVMFFIFieldFlagBitMaskSEqHashDefNonRecursive) {
              return mutator->WithDefRegionKind(kTVMFFIDefRegionKindNonRecursive, [&]() {
                return mutator->MutateExpected(field_value);
              });
            } else if (field_info->flags & kTVMFFIFieldFlagBitMaskSEqHashDefRecursive) {
              return mutator->WithDefRegionKind(kTVMFFIDefRegionKindRecursive, [&]() {
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
          const Any& new_field = details::ExpectedUnsafe::GetData(mutated_field);
          if (field_value.same_as(new_field)) {
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

          ret_code = reflection::CallFieldSetter(field_info, field_addr,
                                                 reinterpret_cast<const TVMFFIAny*>(&new_field));
          if (TVM_FFI_PREDICT_FALSE(ret_code != 0)) {
            result = Unexpected(details::MoveFromSafeCallRaised());
            return true;
          }
          field_changed = true;
          return false;
        });
  };

  // A non-recursive definition applies to the FreeVar itself, but its fields are uses. The
  // complete field traversal are clamped to None, then the definition region is restored.
  if (mutator->def_region_kind() == kTVMFFIDefRegionKindNonRecursive &&
      type_info->metadata != nullptr &&
      type_info->metadata->structural_eq_hash_kind == kTVMFFISEqHashKindFreeVar) {
    mutator->WithDefRegionKind(kTVMFFIDefRegionKindNone, mutate_fields);
  } else {
    mutate_fields();
  }

  if (TVM_FFI_PREDICT_FALSE(result.is_err())) {
    return result;
  }
  if (!field_changed) {
    return Any(value);
  }
  return result;
}

}  // namespace details

// ---------------------------------------------------------------------------
// Structural Map API.
// ---------------------------------------------------------------------------

namespace details {

/// \cond Doxygen_Suppress
// Return from the current mutation function if Result is an Error.
// Append Node to the mutate error context before returning. Node is required: dropping it
// silently degrades every error message produced below this frame.
// A raw pointer Node must be non-null; pass nullable nodes as ObjectRef or Any so None is skipped.
#define TVM_FFI_S_MUTATE_MAYBE_EARLY_RETURN(Result)                                           \
  do {                                                                                        \
    auto&& tvm_ffi_res_ = (Result);                                                           \
    if (TVM_FFI_PREDICT_FALSE(tvm_ffi_res_.is_err())) {                                       \
      return ::tvm::ffi::details::ExpectedUnsafe::MoveToTVMFFIAny(::std::move(tvm_ffi_res_)); \
    }                                                                                         \
  } while (0)

/// \endcond

// Out of line so its strings and Error construction stay out of the hot path of whatever hook
// body TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN expands into. Same reason as
// BadStructuralMutateHookError.
TVM_FFI_COLD_CODE inline TVMFFIAny SMutateDeclaredTypeErrorRaw() noexcept {
  return AnyUnsafe::MoveAnyToTVMFFIAny(
      Any(Error("TypeError", "structural mutate result does not match the declared type", "")));
}

/// \cond Doxygen_Suppress
#define TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN_IMPL_(Result, Type, Name, ResultExpr)    \
  auto Result = (ResultExpr); /* NOLINT(bugprone-macro-parentheses) */             \
  TVM_FFI_S_MUTATE_MAYBE_EARLY_RETURN(Result);                                     \
  if (TVM_FFI_PREDICT_FALSE(!::tvm::ffi::details::AnyUnsafe::CheckAnyStrict<Type>( \
          ::tvm::ffi::details::ExpectedUnsafe::GetData(Result)))) {                \
    return ::tvm::ffi::details::SMutateDeclaredTypeErrorRaw();                     \
  }                                                                                \
  Type Name = /* NOLINT(bugprone-macro-parentheses) */                             \
      ::tvm::ffi::details::AnyUnsafe::MoveFromAnyAfterCheck<Type>(                 \
          ::std::move(::tvm::ffi::details::ExpectedUnsafe::GetData(Result)))
/// \endcond

/*!
 * \brief Unwrap a successful mutation result into a newly declared value or return its error.
 *
 * ``Type`` must be concrete; use a type alias when it contains a top-level comma. A type mismatch
 * returns ``Unexpected(TypeError)`` through the surrounding ``Expected`` function without
 * throwing, reported with a fixed string so a correct hook pays only one predicted-not-taken
 * branch per field. This macro declares ``Name`` into the enclosing scope and must be used in a
 * braced block, never as an unbraced control-flow body. A raw pointer node must be non-null; pass
 * nullable nodes as ``ObjectRef`` or ``Any`` so ``None`` is skipped when constructing error
 * context.
 *
 * Example:
 * \code{.cpp}
 * TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ObjectRef, child, mutator->MutateExpected(self->child), self);
 * \endcode
 *
 * \param Type The concrete type of the successful value.
 * \param Name The name of the value declared in the enclosing scope.
 * \param ResultExpr An expression producing the ``Expected`` value to unwrap.
 */
#define TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(Type, Name, ResultExpr)                                  \
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN_IMPL_(TVM_FFI_STR_CONCAT(tvm_ffi_mutate_result_, __COUNTER__), \
                                          Type, Name, ResultExpr)

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
 * Same signature and same error propagation; the difference is only what happens to a
 * successful result that is not of type \p Type.
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

 protected:
  /*! \brief Return the empty state tuple exposed to typed map callbacks. */
  TVM_FFI_INLINE StateTupleType StateTuple() const noexcept { return {}; }

  /// \cond Doxygen_Suppress
  // Out of line so its strings stay out of the per-node dispatch function, which TryLink inlines
  // into. Shared by the typed and dynamic engines below.
  TVM_FFI_COLD_CODE static Expected<Any> SMutateDescentTypeError() noexcept {
    return Unexpected(Error("TypeError", "structural mutate: descent changed the node type", ""));
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

  /*!
   * \brief Look up a replacement in the identity-substitution environment.
   * \param var The borrowed variable identity to look up.
   * \return The owning replacement, FFI None on a miss, or an Error.
   */
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

  /*!
   * \brief Record a replacement in the identity-substitution environment.
   * \param var The borrowed variable identity to bind.
   * \param mapped_value The borrowed replacement value.
   * \return Successful completion, or an Error if the binding cannot be stored.
   */
  Expected<void> VarRemapSetImpl(AnyView var, AnyView mapped_value) noexcept {
    if (var.type_index() < TypeIndex::kTVMFFIStaticObjectBegin) {
      return Unexpected(
          Error("TypeError", "Variable-remap key must be an object-backed value", ""));
    }
    try {
      ObjectRef var_ref = var.cast<ObjectRef>();
      Any owned_mapped_value(mapped_value);
      var_remap_.Set(var_ref, owned_mapped_value);
      return Expected<void>();
    } catch (const Error& err) {
      return Unexpected(err);
    }
  }

 private:
  /*! \brief Identity-substitution environment, shared across the whole traversal. */
  Map<ObjectRef, Any> var_remap_;
};

/*!
 * \brief Callback-dispatched structural mutator with a state-carrying Parent layer.
 *
 * Each callback is an ordinary callable and the engine selects it on its first argument's
 * type, so selection is a compile-time-known ``as<TSub>()`` on the input node.
 *
 * ``Parent`` derives from ``StructuralMapEngineBase``, publishes ``StateTupleType``, accepts
 * and forwards the mutator vtable in its constructor, and provides a protected
 * ``StateTuple() const noexcept``. A Parent that overrides expected descent must hand-write
 * the matching protected raw redirect so neither entry path silently bypasses the layer. Each
 * callback receives every tuple entry positionally, followed optionally by
 * ``TVMFFIDefRegionKind``. Descent calls use ``this->``, matching
 * ``StructuralWalkEngine``: lookup happens at instantiation in the Parent's class scope and is
 * not virtual dispatch. This deliberately leaves composed deeper-layer declarations eligible;
 * spelling the calls as ``Parent::member`` would instead pin lookup at that qualified layer.
 * A Parent must not hide other engine-internal ``this->`` members because matching ABI-vtable
 * paths deliberately terminate at ``StructuralMapEngineBase``.
 *
 * \code
 * class CountingLayer : public StructuralMapEngineBase {
 *  public:
 *   using StateTupleType = std::tuple<const int&>;
 *   CountingLayer(const StructuralMutatorVTable* vtable, int& count)
 *       : StructuralMapEngineBase(vtable), count_(count) {}
 *   Expected<Any> DefaultMutateExpected(AnyView value) noexcept {
 *     ++count_;
 *     return StructuralMapEngineBase::DefaultMutateExpected(value);
 *   }
 *  protected:
 *   TVMFFIAny DefaultMutateRaw(AnyView value) noexcept {
 *     return details::ExpectedUnsafe::MoveToTVMFFIAny(DefaultMutateExpected(value));
 *   }
 *   StateTupleType StateTuple() const noexcept { return std::tie(std::as_const(count_)); }
 *  private:
 *   int& count_;
 * };
 * int count = 0;
 * auto callback = [](const ObjectRef& value, const int&) -> Expected<Any> {
 *   return Any(value);
 * };
 * using Engine = StructuralMapEngine<CountingLayer, WalkOrder::kPostOrder,
 *                                    decltype(callback)>;
 * StructuralMutator mutator(make_object<Engine>(std::tie(count), std::move(callback)));
 * Expected<Any> result = mutator->MutateExpected(root);
 * \endcode
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

  /*!
   * \brief Construct a callback-aware mutator and forward state into its Parent layer.
   * \param parent_args Arguments following the vtable in the Parent constructor.
   * \param callbacks The typed callback links, tested in declaration order.
   */
  template <typename... ParentArgs>
  explicit StructuralMapEngine(std::tuple<ParentArgs...> parent_args, Callbacks... callbacks)
      : StructuralMapEngine(std::move(parent_args), std::index_sequence_for<ParentArgs...>{},
                            std::move(callbacks)...) {}

 private:
  using ExpectedUnsafe = details::ExpectedUnsafe;
  using AnyUnsafe = details::AnyUnsafe;

  template <typename... ParentArgs, size_t... Is>
  explicit StructuralMapEngine(std::tuple<ParentArgs...> parent_args, std::index_sequence<Is...>,
                               Callbacks... callbacks)
      : Parent(VTable(), std::forward<ParentArgs>(std::get<Is>(parent_args))...),
        callbacks_(std::move(callbacks)...) {}

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
                  "or Expected<U> implicitly convertible to Expected<Any>");
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

    // Deliberately duplicated by StructuralMapDynEngine::TryLink below,
    // which differs only in how a link is found and called; keep the two in step.
    //
    // The match test and the matched-node path live together rather than in separate functions:
    // selecting the link already computes value.as<TSub>(), and a pre-order walk invokes on that
    // very node, so the converted value is reused instead of converted twice.
    //
    // Selection is on the input node, before any descent. A post-order walk must not select on
    // the descended node: the reflected fallback writes its own remap entry, so testing after
    // descent lets that entry swallow the callback.
    std::optional<TSub> matched;
    if constexpr (!std::is_same_v<TSub, AnyView> && !std::is_same_v<TSub, Any>) {
      matched = value.template as<TSub>();
      if (!matched.has_value()) return false;
    }

    // A final statically non-remappable type discards the remap path at optimization time.
    // Every other case uses runtime metadata: nullable refs may match None, non-final subclasses
    // may redeclare the kind, and metadata may be absent.
    const bool remappable = [&]() {
      if constexpr (std::is_base_of_v<ObjectRef, TSub>) {
        using TNode = typename TSub::ContainerType;
        if constexpr (TNode::_type_final &&
                      TNode::_type_s_eq_hash_kind != kTVMFFISEqHashKindFreeVar &&
                      TNode::_type_s_eq_hash_kind != kTVMFFISEqHashKindDAGNode) {
          return false;
        }
      }
      if constexpr (std::is_pointer_v<TSub> &&
                    std::is_base_of_v<Object, std::remove_cv_t<std::remove_pointer_t<TSub>>>) {
        using TNode = std::remove_cv_t<std::remove_pointer_t<TSub>>;
        constexpr bool kFinalNonRemappable =
            TNode::_type_final && TNode::_type_s_eq_hash_kind != kTVMFFISEqHashKindFreeVar &&
            TNode::_type_s_eq_hash_kind != kTVMFFISEqHashKindDAGNode;
        if constexpr (kFinalNonRemappable) return false;
      }
      return this->IsRemappableIdentity(value.type_index());
    }();

    // A FreeVar or DAG node maps once and every later occurrence reuses that result, so if
    // this node already has a cached remap entry, return it instead of mutating it again.
    if (remappable) {
      Expected<Any> mapped = this->VarRemapGetExpected(value);
      if (mapped.is_err()) {
        *out = std::move(mapped);
        return true;
      }
      if (ExpectedUnsafe::GetData(mapped).type_index() != TypeIndex::kTVMFFINone) {
        *out = std::move(mapped);
        return true;
      }
    }

    using StateIndices = std::make_index_sequence<std::tuple_size_v<StateTupleType>>;
    if constexpr (order == WalkOrder::kPreOrder) {
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
      // Own the callback's result: it is the only reference from here on, and moving it out
      // beats holding a reference into an Expected that stays alive across the descent below.
      Any mapped_value = ExpectedUnsafe::GetData(callback_result);
      // Each descent names the node it actually ran on in the error context.
      *out = [&]() -> Expected<Any> {
        if constexpr (kMaybeInplace) {
          // A pre-order result can be mutated in place if unchanged or uniquely owned.
          const TVMFFIAny* mapped_data = AnyUnsafe::TVMFFIAnyPtrFromAny(mapped_value);
          const TVMFFIAny input_data = value.CopyToTVMFFIAny();
          if (mapped_data->type_index == input_data.type_index &&
              mapped_data->zero_padding == input_data.zero_padding &&
              mapped_data->v_int64 == input_data.v_int64) {
            return this->DefaultMaybeInplaceMutateExpected(value);
          }
          const Object* mapped_obj = mapped_value.as<Object>();
          bool can_inplace = mapped_obj != nullptr && mapped_obj->unique();
          return can_inplace ? this->DefaultMaybeInplaceMutateExpected(mapped_value)
                             : this->DefaultMutateExpected(mapped_value);
        } else {
          return this->DefaultMutateExpected(mapped_value);
        }
      }();
      if (TVM_FFI_PREDICT_FALSE(out->is_err())) return true;
    } else {
      // Post-order: children are mapped first and the callback sees the rebuilt node, so it
      // observes its operands already substituted.
      Expected<Any> descended = kMaybeInplace ? this->DefaultMaybeInplaceMutateExpected(value)
                                              : this->DefaultMutateExpected(value);
      if (TVM_FFI_PREDICT_FALSE(descended.is_err())) {
        *out = std::move(descended);
        return true;
      }
      // Held by reference, not moved out: the error path below names this node, so it has to
      // survive the callback. Only a storage-enabled TSub could gain from a move, and that is
      // exactly the case whose move would empty it.
      const Any& mapped_value = ExpectedUnsafe::GetData(descended);
      *out = [&]() -> Expected<Any> {
        if constexpr (std::is_same_v<TSub, AnyView>) {
          return InvokeTypedCallbackLink(callback, AnyView(mapped_value), StateIndices{});
        } else if constexpr (std::is_same_v<TSub, Any>) {
          return InvokeTypedCallbackLink(callback, Any(mapped_value), StateIndices{});
        } else {
          // Re-converted rather than reusing the match: the callback is invoked on the node
          // descent handed back, and must only see the type it asked for. Default mutation is
          // required to preserve the type, so failing here means some hook broke that.
          std::optional<TSub> descended_sub = mapped_value.template as<TSub>();
          if (TVM_FFI_PREDICT_FALSE(!descended_sub.has_value())) {
            return this->SMutateDescentTypeError();
          }
          return InvokeTypedCallbackLink(callback, *std::move(descended_sub), StateIndices{});
        }
      }();
      if (TVM_FFI_PREDICT_FALSE(out->is_err())) {
        this->UpdateVisitErrorContext(*out, mapped_value);
        return true;
      }
    }

    // Bind this node's identity to its final result, so every later occurrence reuses it.
    if (remappable) {
      Expected<void> set_result = this->VarRemapSetExpected(value, ExpectedUnsafe::GetData(*out));
      if (TVM_FFI_PREDICT_FALSE(set_result.is_err())) {
        *out = Unexpected(std::move(set_result).error());
      }
    }
    return true;
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
    if (TryLinks<false>(value, &out, std::index_sequence_for<Callbacks...>{})) {
      return ExpectedUnsafe::MoveToTVMFFIAny(std::move(out));
    }
    return this->DefaultMutateRaw(value);
  }

  /*!
   * \brief Mutate a value in place when safe, invoking the first matching callback link.
   * \param value The borrowed value to mutate.
   * \return Raw ``TVMFFIAny`` containing the mutated value or Error.
   */
  TVM_FFI_INLINE TVMFFIAny MaybeInplaceMutateImplRaw(AnyView value) noexcept {
    Expected<Any> out{Any()};
    if (TryLinks<true>(value, &out, std::index_sequence_for<Callbacks...>{})) {
      return ExpectedUnsafe::MoveToTVMFFIAny(std::move(out));
    }
    return this->DefaultMaybeInplaceMutateRaw(value);
  }

  /*! \brief The callback links, tested in declaration order. */
  std::tuple<Callbacks...> callbacks_;
};

/*!
 * \brief Structural mutator whose links are runtime ffi.Functions keyed by type index.
 *
 * This is the dynamic counterpart of \ref StructuralMapEngine. It remains a distinct
 * straight-line implementation because a post-order dynamic link must remember the runtime
 * type index selected before descent and recheck the rebuilt node against that same index.
 * Moving it into this header makes the same Parent layering available to downstream dynamic
 * mutators; only its runtime link table and invocation differ from the typed engine.
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

  /*!
   * \brief Construct a dynamic map engine and forward state into its Parent layer.
   * \param parent_args Arguments following the vtable in the Parent constructor.
   * \param callbacks Runtime links invoked as ``callback(value)``.
   * \param callbacks_with_def_region_kind Runtime links invoked with the active region kind.
   */
  template <typename... ParentArgs>
  StructuralMapDynEngine(std::tuple<ParentArgs...> parent_args,
                         Array<Tuple<int32_t, Function>> callbacks,
                         Array<Tuple<int32_t, Function>> callbacks_with_def_region_kind)
      : StructuralMapDynEngine(std::move(parent_args), std::index_sequence_for<ParentArgs...>{},
                               std::move(callbacks), std::move(callbacks_with_def_region_kind)) {}

 private:
  using ExpectedUnsafe = details::ExpectedUnsafe;
  using AnyUnsafe = details::AnyUnsafe;

  template <typename... ParentArgs, size_t... Is>
  StructuralMapDynEngine(std::tuple<ParentArgs...> parent_args, std::index_sequence<Is...>,
                         Array<Tuple<int32_t, Function>> callbacks,
                         Array<Tuple<int32_t, Function>> callbacks_with_def_region_kind)
      : Parent(VTable(), std::forward<ParentArgs>(std::get<Is>(parent_args))...),
        callbacks_(std::move(callbacks)),
        callbacks_with_def_region_kind_(std::move(callbacks_with_def_region_kind)) {}

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
   * \param type_index The input node's runtime type index.
   * \param with_kind Set when the matched link also takes a def-region kind.
   * \param link_type_index Set to the registered type index the link matched on, so post-order
   *        traversal can recheck the descended node against the same target.
   * \return The matched Function, or nullopt when no link applies.
   */
  Optional<Function> FindLink(int32_t type_index, bool* with_kind,
                              int32_t* link_type_index) const noexcept {
    for (const Tuple<int32_t, Function>& entry : callbacks_) {
      if (details::RuntimeTypeIndexMatch(type_index, entry.get<0>())) {
        *with_kind = false;
        *link_type_index = entry.get<0>();
        return entry.get<1>();
      }
    }
    for (const Tuple<int32_t, Function>& entry : callbacks_with_def_region_kind_) {
      if (details::RuntimeTypeIndexMatch(type_index, entry.get<0>())) {
        *with_kind = true;
        *link_type_index = entry.get<0>();
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
    // Step for step the same walk as StructuralMapEngine::TryLink, and deliberately so: only
    // link detection and invocation differ. Everything below except finding and calling the
    // link is shared semantics, so a change to either copy belongs in both.
    bool with_kind = false;
    int32_t link_type_index = TypeIndex::kTVMFFINone;
    // A local, so descending into a matching child cannot change what this node invokes.
    Optional<Function> matched = FindLink(value.type_index(), &with_kind, &link_type_index);
    if (!matched.has_value()) return false;

    // --- identity remap, entry half -----------------------------------------
    // A FreeVar or DAG node maps once and every later occurrence reuses that result.
    const bool remappable = this->IsRemappableIdentity(value.type_index());
    if (remappable) {
      Expected<Any> mapped = this->VarRemapGetExpected(value);
      if (mapped.is_err()) {
        *out = std::move(mapped);
        return true;
      }
      if (ExpectedUnsafe::GetData(mapped).type_index() != TypeIndex::kTVMFFINone) {
        *out = std::move(mapped);
        return true;
      }
    }

    // --- callback and descent, in walk order --------------------------------
    if constexpr (order == WalkOrder::kPreOrder) {
      // Pre-order: the callback rewrites this node first, then descent runs over what it made.
      Expected<Any> callback_result =
          InvokeLink(*matched, with_kind, value, this->def_region_kind());
      if (TVM_FFI_PREDICT_FALSE(callback_result.is_err())) {
        this->UpdateVisitErrorContext(callback_result, value);
        *out = std::move(callback_result);
        return true;
      }
      Any mapped_value = ExpectedUnsafe::GetData(callback_result);
      *out = [&]() -> Expected<Any> {
        if constexpr (kMaybeInplace) {
          const TVMFFIAny* mapped_data = AnyUnsafe::TVMFFIAnyPtrFromAny(mapped_value);
          const TVMFFIAny input_data = value.CopyToTVMFFIAny();
          if (mapped_data->type_index == input_data.type_index &&
              mapped_data->zero_padding == input_data.zero_padding &&
              mapped_data->v_int64 == input_data.v_int64) {
            return this->DefaultMaybeInplaceMutateExpected(value);
          }
          const Object* mapped_obj = mapped_value.as<Object>();
          bool can_inplace = mapped_obj != nullptr && mapped_obj->unique();
          return can_inplace ? this->DefaultMaybeInplaceMutateExpected(mapped_value)
                             : this->DefaultMutateExpected(mapped_value);
        } else {
          return this->DefaultMutateExpected(mapped_value);
        }
      }();
      if (TVM_FFI_PREDICT_FALSE(out->is_err())) return true;
    } else {
      // Post-order: children are mapped first, so the callback sees the rebuilt node.
      Expected<Any> descended = kMaybeInplace ? this->DefaultMaybeInplaceMutateExpected(value)
                                              : this->DefaultMutateExpected(value);
      if (TVM_FFI_PREDICT_FALSE(descended.is_err())) {
        *out = std::move(descended);
        return true;
      }
      const Any& mapped_value = ExpectedUnsafe::GetData(descended);
      // Selection used the input node. Recheck the descended node against that same registered
      // target before invoking the saved link.
      if (TVM_FFI_PREDICT_FALSE(
              !details::RuntimeTypeIndexMatch(mapped_value.type_index(), link_type_index))) {
        *out = this->SMutateDescentTypeError();
        this->UpdateVisitErrorContext(*out, mapped_value);
        return true;
      }
      // WithDefRegionKind restores its state through RAII, so this late read is equivalent to
      // the typed engine's invocation-time read even after recursive descent.
      *out = InvokeLink(*matched, with_kind, mapped_value, this->def_region_kind());
      if (TVM_FFI_PREDICT_FALSE(out->is_err())) {
        this->UpdateVisitErrorContext(*out, mapped_value);
        return true;
      }
    }

    // --- identity remap, exit half ------------------------------------------
    // Bind this node's identity to its final result for later occurrences.
    if (remappable) {
      Expected<void> set_result = this->VarRemapSetExpected(value, ExpectedUnsafe::GetData(*out));
      if (TVM_FFI_PREDICT_FALSE(set_result.is_err())) {
        *out = Unexpected(std::move(set_result).error());
      }
    }
    return true;
  }

  /*! \brief Mutate a value, invoking the first matching runtime link. */
  TVM_FFI_INLINE TVMFFIAny MutateImplRaw(AnyView value) noexcept {
    Expected<Any> out{Any()};
    if (TryLink<false>(value, &out)) {
      return ExpectedUnsafe::MoveToTVMFFIAny(std::move(out));
    }
    return this->DefaultMutateRaw(value);
  }

  /*! \brief Optionally mutate a value in place through the first matching runtime link. */
  TVM_FFI_INLINE TVMFFIAny MaybeInplaceMutateImplRaw(AnyView value) noexcept {
    Expected<Any> out{Any()};
    if (TryLink<true>(value, &out)) {
      return ExpectedUnsafe::MoveToTVMFFIAny(std::move(out));
    }
    return this->DefaultMaybeInplaceMutateRaw(value);
  }

  /*! \brief Runtime links invoked without def-region context. */
  Array<Tuple<int32_t, Function>> callbacks_;
  /*! \brief Runtime links invoked with def-region context. */
  Array<Tuple<int32_t, Function>> callbacks_with_def_region_kind_;
};

/*!
 * \brief Map a structured value graph and invoke typed replacement callbacks.
 *
 * Each callback is selected by the type of its first argument. The argument may be ``AnyView``,
 * ``Any``, an object reference type, an object pointer type, or another FFI-convertible POD type. A
 * callback may optionally take a second ``TVMFFIDefRegionKind`` argument. Callbacks are tested in
 * declaration order and only the first strict type match is invoked. An ``AnyView`` callback
 * argument is borrowed and must not be retained after the callback returns.
 *
 * Each callback should follow map semantics: it must not mutate the input in place and should
 * return a bare Any-convertible replacement or ``Expected<U>`` where ``U`` is Any-convertible.
 * A callback may instead return an error value or throw ``Error`` to stop the mapping. In
 * pre-order, an unchanged input or uniquely owned replacement may
 * continue through ``MaybeInplaceMutate``; a shared replacement uses ``Mutate``. In post-order,
 * the callback runs after the node's optional in-place mutation. In-place mutation is available
 * only through an explicit ``__s_maybe_inplace_mutate__`` hook.
 *
 * Objects marked ``kTVMFFISEqHashKindFreeVar`` or ``kTVMFFISEqHashKindDAGNode`` are
 * identity-substituted. A callback is invoked only for the first occurrence of each identity; its
 * final result, including an unchanged result, is reused for every later occurrence in the same
 * structural-map invocation.
 *
 * \sa WalkOrder, StructuralMutator
 *
 * Example:
 *
 * \code{.cpp}
 * Expected<Any> result = StructuralMapExpected<WalkOrder::kPostOrder>(
 *     root,
 *     [](const IntImm& value) -> Expected<Any> {
 *       if (value->value < 0) {
 *         return Unexpected(Error("ValueError", "negative constant", ""));
 *       }
 *       return Any(IntImm(value->value + 1));
 *     },
 *     [](const Add& add, TVMFFIDefRegionKind kind) -> Expected<Any> {
 *       // In post-order, add->lhs and add->rhs have already been mapped.
 *       return Any(add);
 *     });
 * \endcode
 *
 * \tparam order Whether callbacks run before or after recursively mapping children.
 * \tparam Callbacks Callback types whose first parameters select matching values.
 * \param root The borrowed root value to map.
 * \param callbacks Callbacks tested in declaration order. Each accepts ``(value)`` or
 *        ``(value, def_region_kind)`` and returns a bare Any-convertible replacement,
 *        ``Expected<U>`` where ``U`` is Any-convertible, or an error value.
 * \return The mapped owning value, or an Error if mapping or a callback fails.
 *
 * \note Returning ``Expected<U>`` expresses errors as values; throwing ``Error`` is also
 *       supported and is converted to the error state.
 */
template <WalkOrder order, typename... Callbacks>
Expected<Any> StructuralMapExpected(AnyView root, Callbacks&&... callbacks) noexcept {
  static_assert(sizeof...(Callbacks) != 0, "StructuralMap requires at least one callback");
  using Mutator = StructuralMapEngine<StructuralMapEngineBase, order, std::decay_t<Callbacks>...>;
  StructuralMutator mutator(make_object<Mutator>(std::forward<Callbacks>(callbacks)...));
  return mutator->MaybeInplaceMutateIfUniqueExpected(root);
}

/*!
 * \brief Throwing form of \ref tvm::ffi::StructuralMapExpected.
 *
 * See \ref tvm::ffi::StructuralMapExpected for callback dispatch, ordering, and ownership
 * semantics.
 *
 * \tparam order Whether callbacks run before or after recursively mapping children.
 * \tparam Callbacks Callback types whose first parameters select matching values.
 * \param root The borrowed root value to map.
 * \param callbacks Callbacks tested in declaration order. Each accepts ``(value)`` or
 *        ``(value, def_region_kind)`` and returns a bare Any-convertible replacement,
 *        ``Expected<U>`` where ``U`` is Any-convertible, or an error value.
 * \return The mapped owning value.
 * \throws Error if mapping or a callback fails.
 *
 * \note Returning ``Expected<U>`` expresses errors as values; throwing ``Error`` is also
 *       supported and is rethrown by this interface.
 */
template <WalkOrder order, typename... Callbacks>
Any StructuralMap(AnyView root, Callbacks&&... callbacks) {
  return StructuralMapExpected<order>(root, std::forward<Callbacks>(callbacks)...).value();
}

}  // namespace ffi
}  // namespace tvm

#endif  // TVM_FFI_EXTRA_STRUCTURAL_MUTATE_H_
