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
 * \param value The borrowed value to transform.
 * \return Raw ``TVMFFIAny`` containing the transformed value or an Error.
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
   * \return Raw ``TVMFFIAny`` carrying the transformed value or Error.
   */
  FStructuralMutate mutate = nullptr;
  /*!
   * \brief Mutate a value, permitting an in-place implementation when it is safe.
   *
   * \param mutator The active structural mutator.
   * \param value The borrowed value to transform.
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
   * This entry point never intentionally mutates \p value in place. Recursive transformations
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
   * \param value The borrowed value to transform.
   * \return The transformed owning value.
   * \throws Error if transformation fails.
   *
   * The returned value may refer to the same object as \p value. Callers must use the return value
   * as the result of the transformation rather than assuming that the input object was reused.
   */
  TVM_FFI_INLINE Any MaybeInplaceMutate(AnyView value) {
    return MaybeInplaceMutateExpected(value).value();
  }

  /*!
   * \brief Exception-free form of \ref MaybeInplaceMutate.
   *
   * \param value The borrowed value to transform.
   * \return The transformed owning value, or an Error if transformation failed.
   */
  TVM_FFI_INLINE Expected<Any> MaybeInplaceMutateExpected(AnyView value) noexcept {
    return details::ExpectedUnsafe::MoveFromTVMFFIAny<Any>(
        (*vtable_->maybe_inplace_mutate)(this, value));
  }

  /*!
   * \brief Apply the default structural mutation with copy-on-write behavior.
   *
   * \param value The value to mutate.
   * \return The mutated value, or an Error if hook dispatch, copying, or field mutation failed.
   *
   * \note A registered ``__s_mutate__`` hook is dispatched before the reflected fallback and is
   *       responsible for variable-remap lookup and insertion when it represents a FreeVar
   *       identity. Automatic FreeVar remapping applies only to the reflected fallback.
   */
  TVM_FFI_INLINE Expected<Any> DefaultMutateExpected(AnyView value) noexcept {
    int32_t type_index = value.type_index();
    static reflection::TypeAttrColumn column(reflection::type_attr::kStructuralMutate);
    AnyView attr = column[type_index];
    if (attr.type_index() != TypeIndex::kTVMFFINone) {
      if (attr.type_index() == TypeIndex::kTVMFFIOpaquePtr) {
        auto* hook = reinterpret_cast<FStructuralMutate>(attr.cast<void*>());
        return details::ExpectedUnsafe::MoveFromTVMFFIAny<Any>((*hook)(this, value));
      }
      if (attr.type_index() == TypeIndex::kTVMFFIFunction) {
        return attr.cast<Function>().CallExpected<Any>(this, value);
      }
      return Unexpected(Error("TypeError",
                              std::string(reflection::type_attr::kStructuralMutate) +
                                  " must be an opaque function pointer or ffi.Function",
                              ""));
    }
    if (type_index < TypeIndex::kTVMFFIStaticObjectBegin) {
      return Any(value);
    }

    const TVMFFITypeInfo* type_info = TVMFFIGetTypeInfo(type_index);
    bool is_free_var = type_info->metadata != nullptr &&
                       type_info->metadata->structural_eq_hash_kind == kTVMFFISEqHashKindFreeVar;
    if (is_free_var) {
      Expected<Any> mapped_value = VarRemapGetExpected(value);
      if (TVM_FFI_PREDICT_FALSE(mapped_value.is_err())) {
        return Unexpected(std::move(mapped_value).error());
      }
      if (details::ExpectedUnsafe::GetData(mapped_value).type_index() != TypeIndex::kTVMFFINone) {
        return mapped_value;
      }
    }

    Expected<Any> result = details::MutateReflectedFieldsExpected(this, value);
    if (TVM_FFI_PREDICT_FALSE(result.is_err()) || !is_free_var) {
      return result;
    }

    Expected<void> set_result =
        VarRemapSetExpected(value, details::ExpectedUnsafe::GetData(result));
    if (TVM_FFI_PREDICT_FALSE(set_result.is_err())) {
      return Unexpected(std::move(set_result).error());
    }
    return result;
  }

  /*!
   * \brief Apply custom maybe-in-place mutation, or fall back to non-in-place mutation.
   *
   * \param value The borrowed value to transform.
   * \return The transformed owning value, or an Error if transformation failed. In-place changes
   *         completed before an Error are not rolled back.
   *
   * \note In-place mutation is explicitly opt-in. A registered
   *       ``__s_maybe_inplace_mutate__`` hook owns both its in-place safety policy and any
   *       variable-remap handling. When the hook is absent, this method calls \ref MutateExpected.
   */
  TVM_FFI_INLINE Expected<Any> DefaultMaybeInplaceMutateExpected(AnyView value) noexcept {
    int32_t type_index = value.type_index();
    static reflection::TypeAttrColumn maybe_inplace_mutate_column(
        reflection::type_attr::kStructuralMaybeInplaceMutate);
    AnyView maybe_inplace_mutate_attr = maybe_inplace_mutate_column[type_index];
    if (maybe_inplace_mutate_attr.type_index() == TypeIndex::kTVMFFINone) {
      return MutateExpected(value);
    }
    if (maybe_inplace_mutate_attr.type_index() == TypeIndex::kTVMFFIOpaquePtr) {
      auto* hook = reinterpret_cast<FStructuralMutate>(maybe_inplace_mutate_attr.cast<void*>());
      return details::ExpectedUnsafe::MoveFromTVMFFIAny<Any>((*hook)(this, value));
    }
    if (maybe_inplace_mutate_attr.type_index() == TypeIndex::kTVMFFIFunction) {
      return maybe_inplace_mutate_attr.cast<Function>().CallExpected<Any>(this, value);
    }
    return Unexpected(Error("TypeError",
                            std::string(reflection::type_attr::kStructuralMaybeInplaceMutate) +
                                " must be an opaque function pointer or ffi.Function",
                            ""));
  }

  /*!
   * \brief Look up the replacement recorded for a variable identity.
   *
   * \param var The borrowed variable identity to look up.
   * \return The owning replacement, FFI None if no replacement exists, or an Error if lookup
   *         fails.
   *
   * \note The variable identity must have
   *       ``kTVMFFISEqHashKindFreeVar`` structural-equality metadata.
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
   * \note The variable identity must have
   *       ``kTVMFFISEqHashKindFreeVar`` structural-equality metadata.
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
   * \param callback A nullary callable that performs recursive transformation.
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
   * \brief Construct a structural mutator from an immutable dispatch vtable.
   * \param vtable The non-null dispatch table for this mutator. It must outlive this object.
   */
  explicit StructuralMutatorObj(const StructuralMutatorVTable* vtable) : vtable_(vtable) {}

  /*!
   * \brief Non-owning pointer to the required ABI dispatch table.
   */
  const StructuralMutatorVTable* vtable_ = nullptr;

  /*!
   * \brief Current def-region context for def-region-aware structural transformation.
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
 * \return The original value when no field changes, a transformed shallow copy otherwise, or an
 *         Error if copying or mutation failed.
 */
TVM_FFI_INLINE static Expected<Any> MutateReflectedFieldsExpected(StructuralMutatorObj* mutator,
                                                                  AnyView value) noexcept {
  const Object* obj = value.as<Object>();
  int32_t type_index = obj->type_index();

  static reflection::TypeAttrColumn column(reflection::type_attr::kShallowCopy);
  AnyView attr = column[type_index];
  if (TVM_FFI_PREDICT_FALSE(attr.type_index() != TypeIndex::kTVMFFIFunction)) {
    return Unexpected(
        Error("TypeError",
              std::string(reflection::type_attr::kShallowCopy) + " must be an ffi.Function", ""));
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
  // A non-recursive definition applies to a FreeVar itself, but not to its children. All other
  // inherited modes propagate until an explicit field annotation overrides them.
  TVMFFIDefRegionKind inherited_kind = mutator->def_region_kind();
  if (inherited_kind == kTVMFFIDefRegionKindNonRecursive && type_info->metadata != nullptr &&
      type_info->metadata->structural_eq_hash_kind == kTVMFFISEqHashKindFreeVar) {
    inherited_kind = kTVMFFIDefRegionKindNone;
  }

  bool field_changed = false;
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

        TVMFFIDefRegionKind kind = inherited_kind;
        if (field_info->flags & kTVMFFIFieldFlagBitMaskSEqHashDefNonRecursive) {
          kind = kTVMFFIDefRegionKindNonRecursive;
        } else if (field_info->flags & kTVMFFIFieldFlagBitMaskSEqHashDefRecursive) {
          kind = kTVMFFIDefRegionKindRecursive;
        }

        Expected<Any> transformed_field = mutator->WithDefRegionKind(
            kind, [&]() { return mutator->MutateExpected(field_value); });
        if (TVM_FFI_PREDICT_FALSE(transformed_field.is_err())) {
          result = Unexpected(std::move(transformed_field).error());
          return true;
        }
        const Any& new_field = details::ExpectedUnsafe::GetData(transformed_field);
        if (field_value.same_as(new_field)) {
          return false;
        }

        if (TVM_FFI_PREDICT_FALSE(field_info->setter == nullptr)) {
          result = Unexpected(
              Error("TypeError",
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

/*!
 * \brief Structural mutator that invokes typed callbacks during recursive mapping.
 *
 * \tparam order Callback placement relative to child mapping.
 * \tparam Dispatch Callback dispatcher providing ``HasCallbackExpected`` and
 *                  ``InvokeCallbackExpected``. \sa StructuralMapCallbackChain
 */
template <WalkOrder order, typename Dispatch>
class StructuralMapMutatorObj : public StructuralMutatorObj {
 public:
  /*!
   * \brief Construct a callback-aware mutator.
   * \param dispatch The composed callback dispatcher.
   */
  explicit StructuralMapMutatorObj(Dispatch dispatch)
      : StructuralMutatorObj(VTable()), dispatch_(std::move(dispatch)) {}

 private:
  /*!
   * \brief Return the shared callback-aware mutator vtable.
   * \return Pointer to the immutable mutator vtable for this specialization.
   */
  static const StructuralMutatorVTable* VTable() {
    static const StructuralMutatorVTable vtable{
        &StructuralMapMutatorObj::DispatchMutate,
        &StructuralMapMutatorObj::DispatchMaybeInplaceMutate,
        &StructuralMapMutatorObj::DispatchVarRemapGet,
        &StructuralMapMutatorObj::DispatchVarRemapSet,
    };
    return &vtable;
  }

  /*!
   * \brief Dispatch variable-remap lookup through this mutator's vtable.
   * \param mutator The erased callback-aware mutator.
   * \param var The borrowed variable identity to look up.
   * \return Raw ``TVMFFIAny`` containing the owning replacement, FFI None, or Error.
   */
  static TVMFFIAny DispatchVarRemapGet(StructuralMutatorObj* mutator, AnyView var) noexcept {
    auto* self = static_cast<StructuralMapMutatorObj*>(mutator);
    Expected<Any> result = self->VarRemapGetImpl(var);
    return ExpectedUnsafe::MoveToTVMFFIAny(std::move(result));
  }

  /*!
   * \brief Dispatch variable-remap insertion through this mutator's vtable.
   * \param mutator The erased callback-aware mutator.
   * \param var The borrowed variable identity to bind.
   * \param mapped_value The borrowed replacement value.
   * \return Raw ``TVMFFIAny`` containing FFI None or Error.
   */
  static TVMFFIAny DispatchVarRemapSet(StructuralMutatorObj* mutator, AnyView var,
                                       AnyView mapped_value) noexcept {
    auto* self = static_cast<StructuralMapMutatorObj*>(mutator);
    Expected<void> result = self->VarRemapSetImpl(var, mapped_value);
    return ExpectedUnsafe::MoveToTVMFFIAny(std::move(result));
  }

  /*!
   * \brief Look up a replacement in this mutator's identity-substitution environment.
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
   * \brief Record a replacement in this mutator's identity-substitution environment.
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

  /*!
   * \brief Dispatch callback-aware optional in-place mutation through the ABI vtable.
   * \param mutator The erased callback-aware mutator.
   * \param value The borrowed value to transform.
   * \return Raw ``TVMFFIAny`` containing the transformed value or Error.
   */
  static TVMFFIAny DispatchMaybeInplaceMutate(StructuralMutatorObj* mutator,
                                              AnyView value) noexcept {
    auto* self = static_cast<StructuralMapMutatorObj*>(mutator);
    auto result = self->MaybeInplaceMutateImpl(value);
    if (TVM_FFI_PREDICT_FALSE(result.is_err())) {
      if (value.type_index() >= TypeIndex::kTVMFFIStaticObjectBegin) {
        Error err = result.error();
        UpdateVisitErrorContext(err, value.cast<ObjectRef>());
      }
    }
    return ExpectedUnsafe::MoveToTVMFFIAny(std::move(result));
  }

  /*!
   * \brief Dispatch callback-aware mutation through the ABI vtable.
   * \param mutator The erased callback-aware mutator.
   * \param value The borrowed value to mutate.
   * \return Raw ``TVMFFIAny`` containing the mutated value or Error.
   */
  static TVMFFIAny DispatchMutate(StructuralMutatorObj* mutator, AnyView value) noexcept {
    auto* self = static_cast<StructuralMapMutatorObj*>(mutator);
    auto result = self->MutateImpl(value);
    if (TVM_FFI_PREDICT_FALSE(result.is_err())) {
      if (value.type_index() >= TypeIndex::kTVMFFIStaticObjectBegin) {
        Error err = result.error();
        UpdateVisitErrorContext(err, value.cast<ObjectRef>());
      }
    }
    return ExpectedUnsafe::MoveToTVMFFIAny(std::move(result));
  }

  /*!
   * \brief Select mutation for callback-bearing values and otherwise dispatch the optional hook.
   *
   * \param value The borrowed value to transform.
   * \return The transformed value or an Error.
   */
  Expected<Any> MaybeInplaceMutateImpl(AnyView value) noexcept {
    Expected<bool> has_callback = dispatch_.HasCallbackExpected(value);
    if (TVM_FFI_PREDICT_FALSE(has_callback.is_err())) {
      return Unexpected(std::move(has_callback).error());
    }
    if (AnyUnsafe::CopyFromAnyViewAfterCheck<bool>(ExpectedUnsafe::GetData(has_callback))) {
      return MutateExpected(value);
    }
    return DefaultMaybeInplaceMutateExpected(value);
  }

  /*!
   * \brief Mutate one value and invoke its matching callback in the configured order.
   * \param value The borrowed input value. FFI None is returned without callback dispatch.
   * \return The transformed value or an Error.
   */
  Expected<Any> MutateImpl(AnyView value) noexcept {
    if (TVM_FFI_PREDICT_FALSE(value.type_index() == TypeIndex::kTVMFFINone)) {
      return Any(value);
    }

    std::optional<Expected<Any>> pre_order_result;
    if constexpr (order == WalkOrder::kPreOrder) {
      pre_order_result.emplace(dispatch_.InvokeCallbackExpected(value, def_region_kind()));
      if (TVM_FFI_PREDICT_FALSE(pre_order_result->is_err())) {
        return Unexpected(std::move(*pre_order_result).error());
      }
      value = ExpectedUnsafe::GetData(*pre_order_result);
    }

    Expected<Any> result = DefaultMutateExpected(value);
    if (TVM_FFI_PREDICT_FALSE(result.is_err())) {
      return result;
    }

    if constexpr (order == WalkOrder::kPostOrder) {
      const Any& mapped_value = ExpectedUnsafe::GetData(result);
      return dispatch_.InvokeCallbackExpected(mapped_value, def_region_kind());
    }
    return result;
  }

  /*! \brief Composed callback dispatcher owned by this mutator. */
  Dispatch dispatch_;

  /*! \brief Identity-substitution table. */
  Map<ObjectRef, Any> var_remap_;
};

/*!
 * \brief Build a callback dispatcher from a typed callback chain.
 */
struct StructuralMapCallbackChain {
 public:
  /*!
   * \brief Construct a dispatcher owning \p callbacks.
   * \tparam Callbacks Callback types.
   * \param callbacks Callbacks tested in declaration order.
   * \return A callback-aware structural-map dispatcher.
   */
  template <typename... Callbacks>
  static auto FromChain(Callbacks... callbacks) {
    return CallbackDispatch<Callbacks...>(std::move(callbacks)...);
  }

 private:
  /*!
   * \brief Dispatcher over an ordered tuple of typed callbacks.
   * \tparam Callbacks Callback types stored by value.
   */
  template <typename... Callbacks>
  class CallbackDispatch {
   public:
    /*!
     * \brief Construct a dispatcher from a callback chain.
     * \param callbacks Callbacks stored and tested in declaration order.
     */
    explicit CallbackDispatch(Callbacks... callbacks) : callbacks_(std::move(callbacks)...) {}

    /*!
     * \brief Return whether a callback matches \p value.
     * \param value The value to match.
     * \return Whether the callback chain contains a strict type match, or an Error.
     */
    TVM_FFI_INLINE Expected<bool> HasCallbackExpected(AnyView value) const noexcept {
      try {
        return std::apply(
            [&](const auto&... callbacks) {
              return (... || CallbackMatches<std::decay_t<decltype(callbacks)>>(value));
            },
            callbacks_);
      } catch (const Error& err) {
        return Unexpected(err);
      }
    }

    /*!
     * \brief Invoke the first callback matching \p value.
     * \param value The value passed to the callback.
     * \param kind The active def-region kind.
     * \return The first matching callback's result, the original value when nothing matches, or an
     *         Error.
     */
    TVM_FFI_INLINE Expected<Any> InvokeCallbackExpected(AnyView value,
                                                        TVMFFIDefRegionKind kind) noexcept {
      try {
        std::optional<Expected<Any>> result;
        // Fold expression: each TryCallLink returns empty std::optional on no-match
        // (falsy) or a result on match (truthy); || short-circuits on first match.
        std::apply(
            [&](auto&... callbacks) { (... || (result = TryCallLink(callbacks, value, kind))); },
            callbacks_);
        if (result.has_value()) {
          return *std::move(result);
        }
        return Any(value);
      } catch (const Error& err) {
        return Unexpected(err);
      }
    }

    /*!
     * \brief Return whether \p value strictly matches a callback's first argument.
     * \tparam Callback Callback type whose first argument selects the value type.
     * \param value The value to match.
     * \return Whether the callback accepts the value without implicit type conversion.
     */
    template <typename Callback>
    TVM_FFI_INLINE static bool CallbackMatches(AnyView value) {
      using FuncInfo = FunctionInfo<std::decay_t<Callback>>;
      static_assert(FuncInfo::num_args == 1 || FuncInfo::num_args == 2,
                    "StructuralMap callbacks must take one argument (value) or two arguments "
                    "(value, def-region kind)");
      using FirstArg = std::tuple_element_t<0, typename FuncInfo::ArgType>;
      using TSub = std::remove_cv_t<std::remove_reference_t<FirstArg>>;
      if constexpr (std::is_same_v<TSub, AnyView> || std::is_same_v<TSub, Any>) {
        return true;
      } else {
        TVMFFIAny raw_value = value.CopyToTVMFFIAny();
        return TypeTraits<TSub>::CheckAnyStrict(&raw_value);
      }
    }

    /*!
     * \brief Invoke \p callback when \p value matches its first argument.
     * \tparam Callback Callback type whose first argument selects the value type.
     * \param callback The callback under test.
     * \param value The value to match and pass to the callback.
     * \param kind The active def-region kind.
     * \return The callback result on match, or an empty ``std::optional`` otherwise.
     */
    template <typename Callback>
    TVM_FFI_INLINE static std::optional<Expected<Any>> TryCallLink(Callback& callback,
                                                                   AnyView value,
                                                                   TVMFFIDefRegionKind kind) {
      using FuncInfo = FunctionInfo<std::decay_t<Callback>>;
      using FirstArg = std::tuple_element_t<0, typename FuncInfo::ArgType>;
      using TSub = std::remove_cv_t<std::remove_reference_t<FirstArg>>;
      if constexpr (std::is_same_v<TSub, AnyView>) {
        return InvokeCallbackLink(callback, value, kind);
      } else if constexpr (std::is_same_v<TSub, Any>) {
        return InvokeCallbackLink(callback, Any(value), kind);
      } else {
        if (auto opt = value.template as<TSub>()) {
          return InvokeCallbackLink(callback, *std::move(opt), kind);
        }
      }
      return std::nullopt;
    }

    /*!
     * \brief Invoke a matched callback with optional def-region context.
     * \tparam Callback Callable whose result is convertible to ``Expected<Any>``.
     * \tparam Value Type of the converted value passed to the callback.
     * \param callback The matched callback.
     * \param value The converted value passed to the callback.
     * \param kind The active def-region kind.
     * \return The callback result converted to ``Expected<Any>``.
     */
    template <typename Callback, typename Value>
    TVM_FFI_INLINE static Expected<Any> InvokeCallbackLink(Callback& callback, Value&& value,
                                                           TVMFFIDefRegionKind kind) {
      using FuncInfo = FunctionInfo<std::decay_t<Callback>>;
      if constexpr (FuncInfo::num_args == 1) {
        return callback(std::forward<Value>(value));
      } else {
        return callback(std::forward<Value>(value), kind);
      }
    }

    /*! \brief Callbacks tested in declaration order. */
    std::tuple<Callbacks...> callbacks_;
  };
};

}  // namespace details

/*!
 * \brief Transform a structured value graph and invoke typed replacement callbacks.
 *
 * Each callback is selected by the type of its first argument. The argument may be ``AnyView``,
 * ``Any``, an object reference type, an object pointer type, or another FFI-convertible POD type. A
 * callback may optionally take a second ``TVMFFIDefRegionKind`` argument. Callbacks are tested in
 * declaration order and only the first strict type match is invoked. FFI None is skipped.
 * An ``AnyView`` callback argument is borrowed and must not be retained after the callback returns.
 *
 * Each callback should follow map semantics: it must not mutate the input in place and should
 * return ``Expected<Any>`` containing either the unchanged input or its replacement. An ``Error``
 * stops the transformation. A node with a matching callback always uses ``Mutate``. A node
 * without a callback may use in-place mutation only through its explicit
 * ``__s_maybe_inplace_mutate__`` hook.
 *
 * Objects marked ``kTVMFFISEqHashKindFreeVar`` are identity-substituted. A callback is invoked only
 * for the first occurrence of each identity; its final result, including an unchanged result, is
 * reused for every later occurrence in the same structural-map invocation.
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
 * \param root The borrowed root value to transform.
 * \param callbacks Callbacks tested in declaration order. Each accepts ``(value)`` or
 *        ``(value, def_region_kind)`` and should return ``Expected<Any>``.
 * \return The transformed owning value, or an Error if mapping or a callback fails.
 *
 * \note Return type of each callback should be ``Expected<Any>``.
 */
template <WalkOrder order, typename... Callbacks>
Expected<Any> StructuralMapExpected(AnyView root, Callbacks&&... callbacks) noexcept {
  static_assert(sizeof...(Callbacks) != 0, "StructuralMap requires at least one callback");
  auto dispatch =
      details::StructuralMapCallbackChain::FromChain(std::forward<Callbacks>(callbacks)...);
  using Mutator = details::StructuralMapMutatorObj<order, decltype(dispatch)>;
  StructuralMutator mutator(make_object<Mutator>(std::move(dispatch)));
  return mutator->MaybeInplaceMutateExpected(root);
}

/*!
 * \brief Throwing form of \ref tvm::ffi::StructuralMapExpected.
 *
 * See \ref tvm::ffi::StructuralMapExpected for callback dispatch, ordering, and ownership
 * semantics.
 *
 * \tparam order Whether callbacks run before or after recursively mapping children.
 * \tparam Callbacks Callback types whose first parameters select matching values.
 * \param root The borrowed root value to transform.
 * \param callbacks Callbacks tested in declaration order. Each accepts ``(value)`` or
 *        ``(value, def_region_kind)`` and should return ``Expected<Any>``.
 * \return The transformed owning value.
 * \throws Error if mapping or a callback fails.
 *
 * \note Return type of each callback should be ``Expected<Any>``.
 */
template <WalkOrder order, typename... Callbacks>
Any StructuralMap(AnyView root, Callbacks&&... callbacks) {
  return StructuralMapExpected<order>(root, std::forward<Callbacks>(callbacks)...).value();
}

}  // namespace ffi
}  // namespace tvm

#endif  // TVM_FFI_EXTRA_STRUCTURAL_MUTATE_H_
