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
 * \file tvm/ffi/extra/structural_map.h
 * \brief Structural mapping and in-place mutation API.
 */
#ifndef TVM_FFI_EXTRA_STRUCTURAL_MAP_H_
#define TVM_FFI_EXTRA_STRUCTURAL_MAP_H_

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
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

namespace tvm {
namespace ffi {

class StructuralMapperObj;

/*!
 * \brief ABI of structural transformation hooks and \ref StructuralMapperVTable callbacks.
 *
 * The callback receives the active mapper as a non-owning pointer and an owning ``Any`` value. It
 * returns raw ``TVMFFIAny`` storage containing either the transformed value or an Error. The input
 * ownership slot and returned storage are transferred across the ABI boundary by move.
 */
using FStructuralTransform = TVMFFIAny (*)(StructuralMapperObj* mapper, Any value) noexcept;

namespace details {

/*!
 * \brief Move a structural transformation result to raw ABI storage and annotate failures.
 *
 * \param result The transformation result to move into raw ABI storage.
 * \param error_context The value retained by the current dispatch frame for error reporting.
 * \return Raw ``TVMFFIAny`` storing the success value or Error.
 */
TVM_FFI_INLINE static TVMFFIAny MoveStructuralTransformResultToTVMFFIAny(
    Expected<Any> result, const Any& error_context) noexcept {
  if (TVM_FFI_PREDICT_FALSE(result.is_err())) {
    if (error_context.type_index() >= TypeIndex::kTVMFFIStaticObjectBegin) {
      Error err = result.error();
      UpdateVisitErrorContext(err, error_context.cast<ObjectRef>());
    }
  }
  return ExpectedUnsafe::MoveToTVMFFIAny(std::move(result));
}

// Dispatch a type-specific structural map or in-place mutation hook.
TVM_FFI_INLINE static Expected<Any> DispatchTypeAttrHookExpected(
    StructuralMapperObj* mapper, Any value, AnyView attr, std::string_view attr_name) noexcept;

// Copy and structurally map the reflected fields of an object-backed value.
TVM_FFI_INLINE static Expected<Any> MapReflectedFieldsExpected(StructuralMapperObj* mapper,
                                                               Any value) noexcept;

// Structurally transform the reflected fields of an object-backed value in place.
TVM_FFI_INLINE static Expected<Any> InplaceMutateReflectedFieldsExpected(
    StructuralMapperObj* mapper, Any value) noexcept;

}  // namespace details

/*!
 * \brief VTable ABI for \ref StructuralMapper dispatch. This function table provides a stable ABI
 * for the map and in-place mutation methods.
 */
struct StructuralMapperVTable {
  /*!
   * \brief Select mapping or in-place mutation for a value.
   * \param mapper The active structural mapper.
   * \param value The owning value to transform by map or in-place mutation.
   * \return Raw ``TVMFFIAny`` carrying the transformed value or Error.
   */
  FStructuralTransform map_or_inplace_mutate = nullptr;
  /*!
   * \brief Map a value without intentionally mutating the source.
   * \param mapper The active structural mapper.
   * \param value The owning value to map.
   * \return Raw ``TVMFFIAny`` carrying the mapped value or Error.
   */
  FStructuralTransform map = nullptr;
  /*!
   * \brief Transform a value using the explicit in-place mutation path.
   * \param mapper The active structural mapper.
   * \param value The owning value to transform in place.
   * \return Raw ``TVMFFIAny`` carrying the transformed value or Error.
   */
  FStructuralTransform inplace_mutate = nullptr;
};

/*!
 * \brief Object node of a structural mapper.
 *
 * A structural mapper recursively transforms values through a manually supplied ABI vtable.
 * The default behavior supports type-specific hooks and reflected-field fallback. Values
 * are accepted by value so ownership can be transferred without adding persistent references and
 * so the default combined operation can select in-place mutation from logical uniqueness.
 */
class StructuralMapperObj : public Object {
 public:
  /*! \brief Construct the default structural mapper. */
  StructuralMapperObj() : StructuralMapperObj(VTable()) {}

  /*!
   * \brief Transform a value, selecting mapping or in-place mutation through the mapper vtable.
   *
   * \param value The value and one ownership slot to transfer into the transformation.
   * \return The transformed value.
   *
   * \note This method throws the returned Error on failure. Use
   *       \ref MapOrInplaceMutateExpected for exception-free propagation.
   */
  TVM_FFI_INLINE Any MapOrInplaceMutate(Any value) {
    return MapOrInplaceMutateExpected(std::move(value)).value();
  }

  /*!
   * \brief Exception-free form of \ref MapOrInplaceMutate.
   *
   * \param value The value and one ownership slot to transfer into the transformation.
   * \return The transformed value, or an Error if transformation failed.
   */
  TVM_FFI_INLINE Expected<Any> MapOrInplaceMutateExpected(Any value) noexcept {
    return details::ExpectedUnsafe::MoveFromTVMFFIAny<Any>(
        (*vtable_->map_or_inplace_mutate)(this, std::move(value)));
  }

  /*!
   * \brief Apply the default map-or-in-place selection logic.
   *
   * \param value The value to transform by map or in-place mutation.
   * \return The transformed value.
   */
  TVM_FFI_INLINE Any DefaultMapOrInplaceMutate(Any value) {
    return DefaultMapOrInplaceMutateExpected(std::move(value)).value();
  }

  /*!
   * \brief Exception-free form of \ref DefaultMapOrInplaceMutate.
   *
   * \param value The value to transform by map or in-place mutation.
   * \return The transformed value, or an Error if selection or transformation failed.
   */
  TVM_FFI_INLINE Expected<Any> DefaultMapOrInplaceMutateExpected(Any value) noexcept {
    return DefaultMapOrInplaceMutateExpected(std::move(value), false);
  }

  /*!
   * \brief Map a value through the mapper vtable.
   *
   * \param value The value to map.
   * \return The mapped value.
   *
   * \note This method throws the returned Error on failure. Use \ref MapExpected for
   *       exception-free propagation.
   */
  TVM_FFI_INLINE Any Map(Any value) { return MapExpected(std::move(value)).value(); }

  /*!
   * \brief Exception-free form of \ref Map.
   *
   * \param value The value to map.
   * \return The mapped value, or an Error if mapping failed.
   */
  TVM_FFI_INLINE Expected<Any> MapExpected(Any value) noexcept {
    return details::ExpectedUnsafe::MoveFromTVMFFIAny<Any>((*vtable_->map)(this, std::move(value)));
  }

  /*!
   * \brief Apply the default structural map with copy-on-write behavior.
   *
   * \param value The value to map.
   * \return The mapped value.
   */
  TVM_FFI_INLINE Any DefaultMap(Any value) { return DefaultMapExpected(std::move(value)).value(); }

  /*!
   * \brief Exception-free form of \ref DefaultMap.
   *
   * \param value The value to map.
   * \return The mapped value, or an Error if hook dispatch, copying, or field mapping failed.
   */
  TVM_FFI_INLINE Expected<Any> DefaultMapExpected(Any value) noexcept {
    int32_t type_index = value.type_index();
    static reflection::TypeAttrColumn column(reflection::type_attr::kStructuralMap);
    AnyView attr = column[type_index];
    if (attr.type_index() != TypeIndex::kTVMFFINone) {
      return details::DispatchTypeAttrHookExpected(this, std::move(value), attr,
                                                   reflection::type_attr::kStructuralMap);
    }
    if (type_index < TypeIndex::kTVMFFIStaticObjectBegin) {
      return value;
    }
    return details::MapReflectedFieldsExpected(this, std::move(value));
  }

  /*!
   * \brief Transform a value through the explicit in-place mutation vtable entry.
   *
   * \param value The value to transform in place.
   * \return The transformed value.
   */
  TVM_FFI_INLINE Any InplaceMutate(Any value) {
    return InplaceMutateExpected(std::move(value)).value();
  }

  /*!
   * \brief Exception-free form of \ref InplaceMutate.
   *
   * \param value The value to transform in place.
   * \return The transformed value, or an Error if transformation failed.
   */
  TVM_FFI_INLINE Expected<Any> InplaceMutateExpected(Any value) noexcept {
    return details::ExpectedUnsafe::MoveFromTVMFFIAny<Any>(
        (*vtable_->inplace_mutate)(this, std::move(value)));
  }

  /*!
   * \brief Apply the default structural in-place mutation.
   *
   * \param value The value to transform in place.
   * \return The transformed value.
   */
  TVM_FFI_INLINE Any DefaultInplaceMutate(Any value) {
    return DefaultInplaceMutateExpected(std::move(value)).value();
  }

  /*!
   * \brief Exception-free form of \ref DefaultInplaceMutate.
   *
   * \param value The value to transform in place.
   * \return The transformed value, or an Error if hook dispatch or reflected mutation failed.
   */
  TVM_FFI_INLINE Expected<Any> DefaultInplaceMutateExpected(Any value) noexcept {
    int32_t type_index = value.type_index();
    static reflection::TypeAttrColumn column(reflection::type_attr::kStructuralInplaceMutate);
    AnyView attr = column[type_index];
    if (attr.type_index() != TypeIndex::kTVMFFINone) {
      return details::DispatchTypeAttrHookExpected(this, std::move(value), attr,
                                                   reflection::type_attr::kStructuralInplaceMutate);
    }
    if (type_index < TypeIndex::kTVMFFIStaticObjectBegin) {
      return value;
    }
    return details::InplaceMutateReflectedFieldsExpected(this, std::move(value));
  }

  /// \cond Doxygen_Suppress
  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("ffi.StructuralMapper", StructuralMapperObj, Object);
  /// \endcond

 protected:
  /*!
   * \brief Construct a structural mapper subclass with a custom dispatch vtable.
   *
   * \param vtable The non-null dispatch table for this mapper.
   *
   * \note This constructor is for internal subclasses. The vtable and its
   *       ``map_or_inplace_mutate`` callback must be valid for the lifetime of the mapper.
   */
  explicit StructuralMapperObj(const StructuralMapperVTable* vtable) : vtable_(vtable) {}

  /*!
   * \brief Apply default combined transformation with a customized uniqueness decision.
   *
   * \param value The value to transform.
   * \param can_inplace_mutate Whether the caller has already established logical uniqueness before
   *        adding temporary internal owners.
   * \return The transformed value.
   *
   * \note This protected overload is for ownership-aware internal recursion. Passing ``true`` is
   *       invalid when the object is genuinely shared.
   */
  TVM_FFI_INLINE Any DefaultMapOrInplaceMutate(Any value, bool can_inplace_mutate) {
    return DefaultMapOrInplaceMutateExpected(std::move(value), can_inplace_mutate).value();
  }

  /*!
   * \brief Exception-free form of the default transformation with a customized uniqueness decision.
   *
   * \param value The value to transform.
   * \param can_inplace_mutate Whether the caller has already established logical uniqueness before
   *        adding temporary internal owners.
   * \return The transformed value, or an Error if validation or transformation failed.
   */
  TVM_FFI_INLINE Expected<Any> DefaultMapOrInplaceMutateExpected(Any value,
                                                                 bool can_inplace_mutate) noexcept {
    if (const Object* obj = value.as<Object>()) {
      // check both map and inplace_mutate hooks are defined
      static reflection::TypeAttrColumn map_column(reflection::type_attr::kStructuralMap);
      static reflection::TypeAttrColumn inplace_mutate_column(
          reflection::type_attr::kStructuralInplaceMutate);
      int32_t type_index = value.type_index();
      AnyView map_attr = map_column[type_index];
      AnyView inplace_mutate_attr = inplace_mutate_column[type_index];
      bool has_map = map_attr.type_index() != TypeIndex::kTVMFFINone;
      bool has_inplace_mutate = inplace_mutate_attr.type_index() != TypeIndex::kTVMFFINone;
      if (has_map != has_inplace_mutate) {
        return Unexpected(Error("TypeError",
                                "One of " + std::string(reflection::type_attr::kStructuralMap) +
                                    " and " +
                                    std::string(reflection::type_attr::kStructuralInplaceMutate) +
                                    " is undefined, should provide both of them.",
                                ""));
      }
      if (obj->unique() || can_inplace_mutate) {
        return InplaceMutateExpected(std::move(value));
      }
    }
    return MapExpected(std::move(value));
  }

  // Grant reflected in-place recursion access to the ownership-aware protected overload.
  friend Expected<Any> details::InplaceMutateReflectedFieldsExpected(StructuralMapperObj* mapper,
                                                                     Any value) noexcept;

  /*!
   * \brief Required ABI dispatch table. \ref StructuralMapperVTable
   * It must never be null on a constructed mapper.
   */
  const StructuralMapperVTable* vtable_ = nullptr;

 private:
  /*!
   * \brief Return the vtable used by the default structural mapper.
   * \return Pointer to the static default vtable.
   */
  static const StructuralMapperVTable* VTable() {
    static const StructuralMapperVTable vtable{
        &StructuralMapperObj::DispatchMapOrInplaceMutate,
        &StructuralMapperObj::DispatchMap,
        &StructuralMapperObj::DispatchInplaceMutate,
    };
    return &vtable;
  }

  /*!
   * \brief Dispatch the default transformation from the ABI vtable.
   *
   * \param mapper The active structural mapper.
   * \param value The owning value to transform.
   * \return Raw ``TVMFFIAny`` storing the transformed value or Error.
   *
   * \note Validation errors raised before a branch is selected are returned without adding a
   *       visit-context frame at this forwarding layer to avoid duplicated error context.
   */
  static TVMFFIAny DispatchMapOrInplaceMutate(StructuralMapperObj* mapper, Any value) noexcept {
    auto result = mapper->DefaultMapOrInplaceMutateExpected(std::move(value));
    return details::ExpectedUnsafe::MoveToTVMFFIAny(std::move(result));
  }

  /*!
   * \brief Dispatch default mapping from the ABI vtable.
   *
   * \param mapper The active structural mapper.
   * \param value The owning value to map.
   * \return Raw ``TVMFFIAny`` storing the mapped value or Error.
   */
  static TVMFFIAny DispatchMap(StructuralMapperObj* mapper, Any value) noexcept {
    Any error_context = value;
    auto result = mapper->DefaultMapExpected(std::move(value));
    return details::MoveStructuralTransformResultToTVMFFIAny(std::move(result), error_context);
  }

  /*!
   * \brief Dispatch default in-place mutation from the ABI vtable.
   *
   * \param mapper The active structural mapper.
   * \param value The owning value to transform in place.
   * \return Raw ``TVMFFIAny`` storing the transformed value or Error.
   */
  static TVMFFIAny DispatchInplaceMutate(StructuralMapperObj* mapper, Any value) noexcept {
    Any error_context = value;
    auto result = mapper->DefaultInplaceMutateExpected(std::move(value));
    return details::MoveStructuralTransformResultToTVMFFIAny(std::move(result), error_context);
  }
};

/*!
 * \brief ObjectRef wrapper of \ref StructuralMapperObj.
 *
 * \sa StructuralMapperObj
 */
class StructuralMapper : public ObjectRef {
 public:
  /*! \brief Construct the default structural mapper. */
  StructuralMapper() : ObjectRef(make_object<StructuralMapperObj>()) {}

  /*!
   * \brief Construct from an existing mapper object pointer.
   * \param n The object pointer to wrap.
   */
  explicit StructuralMapper(ObjectPtr<StructuralMapperObj> n) : ObjectRef(std::move(n)) {}

  /// \cond Doxygen_Suppress
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(StructuralMapper, ObjectRef, StructuralMapperObj);
  /// \endcond
};

namespace details {

/*!
 * \brief Dispatch a type-specific structural transformation hook.
 *
 * \param mapper The active structural mapper.
 * \param value The value to pass to the hook.
 * \param attr The registered type attribute value.
 * \param attr_name The attribute name used in type errors.
 * \return The hook result, or an Error for hook failure or an invalid attribute value.
 */
TVM_FFI_INLINE static Expected<Any> DispatchTypeAttrHookExpected(
    StructuralMapperObj* mapper, Any value, AnyView attr, std::string_view attr_name) noexcept {
  // case 1: Type-specific override registered as an opaque ABI function pointer.
  if (attr.type_index() == TypeIndex::kTVMFFIOpaquePtr) {
    auto* hook = reinterpret_cast<FStructuralTransform>(attr.cast<void*>());
    return details::ExpectedUnsafe::MoveFromTVMFFIAny<Any>((*hook)(mapper, std::move(value)));
  }

  // case 2: Type-specific override registered as an ffi::Function.
  if (attr.type_index() == TypeIndex::kTVMFFIFunction) {
    return attr.cast<Function>().CallExpected<Any>(mapper, value);
  }

  return Unexpected(
      Error("TypeError",
            std::string(attr_name) + " must be an opaque function pointer or ffi.Function", ""));
}

/*!
 * \brief Transform every reflected structural field of an object.
 *
 * \tparam Callback A callable compatible with ``Expected<Any>(const Any&)``.
 * \param value The original object-backed value.
 * \param result The shallow-copy result in copy-on-write mode; ignored in in-place mode.
 * \param copy_on_write Whether to transform a distinct shallow copy instead of \p value.
 * \param callback The recursive field transformation callback.
 * \return The original value, transformed copy, or in-place result; otherwise an Error.
 */
template <typename Callback>
TVM_FFI_INLINE static Expected<Any> TransformReflectedFieldsExpected(Any value,
                                                                     Expected<Any> result,
                                                                     bool copy_on_write,
                                                                     Callback callback) noexcept {
  const Object* obj = value.as<Object>();
  if (!copy_on_write) {
    // In-place mode transfers the input ownership slot into the result container.
    result = std::exchange(value, Any());
  }
  const Any& result_value = details::ExpectedUnsafe::GetData(result);
  Object* new_obj = const_cast<Object*>(result_value.as<Object>());
  if (copy_on_write) {
    // Mapping requires a distinct target so partially applied updates cannot mutate the source.
    if (TVM_FFI_PREDICT_FALSE(new_obj == nullptr || result.type_index() != value.type_index() ||
                              new_obj == obj)) {
      return Unexpected(
          Error("TypeError",
                "Shallow copy callback must return a distinct object with the same type as its "
                "input",
                ""));
    }
  }
  const TVMFFITypeInfo* type_info = TVMFFIGetTypeInfo(new_obj->type_index());

  bool field_changed = false;
  reflection::ForEachFieldInfoWithEarlyStop(
      type_info, [&](const TVMFFIFieldInfo* field_info) -> bool {
        if (field_info->flags & kTVMFFIFieldFlagBitMaskSEqHashIgnore) {
          return false;
        }

        Any field_value;
        const void* field_addr = reinterpret_cast<const char*>(new_obj) + field_info->offset;
        int ret_code = field_info->getter(const_cast<void*>(field_addr),
                                          reinterpret_cast<TVMFFIAny*>(&field_value));
        if (TVM_FFI_PREDICT_FALSE(ret_code != 0)) {
          result = Unexpected(details::MoveFromSafeCallRaised());
          return true;
        }

        // TODO(kathrync): add WithDefRegionKind
        Expected<Any> transformed_field = callback(field_value);
        if (TVM_FFI_PREDICT_FALSE(transformed_field.is_err())) {
          result = std::move(transformed_field);
          return true;
        }
        const Any& new_field = details::ExpectedUnsafe::GetData(transformed_field);
        if (field_value.same_as(new_field)) {
          return false;
        }

        if (TVM_FFI_PREDICT_FALSE(field_info->setter == nullptr)) {
          result = Unexpected(
              Error("TypeError",
                    "Cannot structurally map field `" +
                        std::string(field_info->name.data, field_info->name.size) + "` of type `" +
                        std::string(type_info->type_key.data, type_info->type_key.size) +
                        "` because it does not define a setter",
                    ""));
          return true;
        }

        void* new_field_addr = reinterpret_cast<char*>(new_obj) + field_info->offset;
        ret_code = reflection::CallFieldSetter(field_info, new_field_addr,
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
  if (copy_on_write && !field_changed) {
    return value;
  }
  return result;
}

/*!
 * \brief Map the reflected structural fields of an object-backed value.
 *
 * \param mapper The active structural mapper.
 * \param value The object-backed value to map.
 * \return The original value when no field changes, a transformed shallow copy otherwise, or an
 *         Error if copying or mapping failed.
 */
TVM_FFI_INLINE static Expected<Any> MapReflectedFieldsExpected(StructuralMapperObj* mapper,
                                                               Any value) noexcept {
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

  return TransformReflectedFieldsExpected(
      std::move(value), std::move(result), true,
      [mapper](const Any& field_value) noexcept -> Expected<Any> {
        return mapper->MapExpected(field_value);
      });
}

/*!
 * \brief Transform the reflected structural fields of an object-backed value in place.
 *
 * \param mapper The active structural mapper.
 * \param value The object-backed value whose fields should be transformed in place.
 * \return The input object after field transformation, or an Error. Mutations made before an Error
 *         are not rolled back.
 */
TVM_FFI_INLINE static Expected<Any> InplaceMutateReflectedFieldsExpected(
    StructuralMapperObj* mapper, Any value) noexcept {
  return TransformReflectedFieldsExpected(
      std::move(value), Any(), false, [mapper](const Any& field_value) noexcept -> Expected<Any> {
        const Object* field_obj = field_value.as<Object>();
        // A logically unique object field has two references here: one from its parent and one
        // from field_value, which was created by the reflection getter.
        bool can_inplace_mutate = field_obj != nullptr && field_obj->use_count() == 2;
        return mapper->DefaultMapOrInplaceMutateExpected(field_value, can_inplace_mutate);
      });
}

}  // namespace details
}  // namespace ffi
}  // namespace tvm

#endif  // TVM_FFI_EXTRA_STRUCTURAL_MAP_H_
