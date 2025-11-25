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
 * \file tvm/ffi/reflection/registry.h
 * \brief Registry of reflection metadata.
 */
#ifndef TVM_FFI_EXTRA_OVERLOAD_H
#define TVM_FFI_EXTRA_OVERLOAD_H

#include <tvm/ffi/any.h>
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/container/variant.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/function_details.h>
#include <tvm/ffi/optional.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>
#include <tvm/ffi/type_traits.h>

#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace tvm {
namespace ffi {
/*! \brief Reflection namespace */
namespace reflection {

/*!
 * \brief Helper to register Object's reflection metadata.
 * \tparam Class The class type.
 *
 * \code
 *  namespace refl = tvm::ffi::reflection;
 *  refl::ObjectDef<MyClass>().def_ro("my_field", &MyClass::my_field);
 * \endcode
 */
template <typename Class>
class OverloadObjectDef : public ReflectionDefBase {
 public:
  /*!
   * \brief Constructor
   * \tparam ExtraArgs The extra arguments.
   * \param extra_args The extra arguments.
   */
  template <typename... ExtraArgs>
  explicit OverloadObjectDef(ExtraArgs&&... extra_args)
      : type_index_(Class::_GetOrAllocRuntimeTypeIndex()), type_key_(Class::_type_key) {
    RegisterExtraInfo(std::forward<ExtraArgs>(extra_args)...);
  }

  /*!
   * \brief Define a readonly field.
   *
   * \tparam Class The class type.
   * \tparam T The field type.
   * \tparam Extra The extra arguments.
   *
   * \param name The name of the field.
   * \param field_ptr The pointer to the field.
   * \param extra The extra arguments that can be docstring or default value.
   *
   * \return The reflection definition.
   */
  template <typename T, typename BaseClass, typename... Extra>
  TVM_FFI_INLINE OverloadObjectDef& def_ro(const char* name, T BaseClass::* field_ptr,
                                           Extra&&... extra) {
    RegisterField(name, field_ptr, false, std::forward<Extra>(extra)...);
    return *this;
  }

  /*!
   * \brief Define a read-write field.
   *
   * \tparam Class The class type.
   * \tparam T The field type.
   * \tparam Extra The extra arguments.
   *
   * \param name The name of the field.
   * \param field_ptr The pointer to the field.
   * \param extra The extra arguments that can be docstring or default value.
   *
   * \return The reflection definition.
   */
  template <typename T, typename BaseClass, typename... Extra>
  TVM_FFI_INLINE OverloadObjectDef& def_rw(const char* name, T BaseClass::* field_ptr,
                                           Extra&&... extra) {
    static_assert(Class::_type_mutable, "Only mutable classes are supported for writable fields");
    RegisterField(name, field_ptr, true, std::forward<Extra>(extra)...);
    return *this;
  }

  /*!
   * \brief Define a method.
   *
   * \tparam Func The function type.
   * \tparam Extra The extra arguments.
   *
   * \param name The name of the method.
   * \param func The function to be registered.
   * \param extra The extra arguments that can be docstring.
   *
   * \return The reflection definition.
   */
  template <typename Func, typename... Extra>
  TVM_FFI_INLINE OverloadObjectDef& def(const char* name, Func&& func, Extra&&... extra) {
    RegisterMethod(name, false, std::forward<Func>(func), std::forward<Extra>(extra)...);
    return *this;
  }

  /*!
   * \brief Define a static method.
   *
   * \tparam Func The function type.
   * \tparam Extra The extra arguments.
   *
   * \param name The name of the method.
   * \param func The function to be registered.
   * \param extra The extra arguments that can be docstring.
   *
   * \return The reflection definition.
   */
  template <typename Func, typename... Extra>
  TVM_FFI_INLINE OverloadObjectDef& def_static(const char* name, Func&& func, Extra&&... extra) {
    RegisterMethod(name, true, std::forward<Func>(func), std::forward<Extra>(extra)...);
    return *this;
  }

  /*!
   * \brief Register a constructor for this object type.
   *
   * This method registers a static `__init__` method that constructs an instance
   * of the object with the specified argument types. The constructor can be invoked
   * from Python or other FFI bindings.
   *
   * \tparam Args The argument types for the constructor.
   * \tparam Extra Additional arguments (e.g., docstring).
   *
   * \param init_func An instance of `init<Args...>` specifying constructor signature.
   * \param extra Optional additional metadata such as docstring.
   *
   * \return Reference to this `ObjectDef` for method chaining.
   *
   * Example:
   * \code
   *   refl::ObjectDef<MyObject>()
   *       .def(refl::init<int64_t, std::string>(), "Constructor docstring");
   * \endcode
   */
  template <typename... Args, typename... Extra>
  TVM_FFI_INLINE OverloadObjectDef& def([[maybe_unused]] init<Args...> init_func,
                                        Extra&&... extra) {
    RegisterMethod(kInitMethodName, true, &init<Args...>::template execute<Class>,
                   std::forward<Extra>(extra)...);
    return *this;
  }

 private:
  template <typename... ExtraArgs>
  void RegisterExtraInfo(ExtraArgs&&... extra_args) {
    TVMFFITypeMetadata info;
    info.total_size = sizeof(Class);
    info.structural_eq_hash_kind = Class::_type_s_eq_hash_kind;
    info.creator = nullptr;
    info.doc = TVMFFIByteArray{nullptr, 0};
    if constexpr (std::is_default_constructible_v<Class>) {
      info.creator = ObjectCreatorDefault<Class>;
    } else if constexpr (std::is_constructible_v<Class, UnsafeInit>) {
      info.creator = ObjectCreatorUnsafeInit<Class>;
    }
    // apply extra info traits
    ((ApplyExtraInfoTrait(&info, std::forward<ExtraArgs>(extra_args)), ...));
    TVM_FFI_CHECK_SAFE_CALL(TVMFFITypeRegisterMetadata(type_index_, &info));
  }

  template <typename T, typename BaseClass, typename... ExtraArgs>
  void RegisterField(const char* name, T BaseClass::* field_ptr, bool writable,
                     ExtraArgs&&... extra_args) {
    static_assert(std::is_base_of_v<BaseClass, Class>, "BaseClass must be a base class of Class");
    FieldInfoBuilder info;
    info.name = TVMFFIByteArray{name, std::char_traits<char>::length(name)};
    info.field_static_type_index = TypeToFieldStaticTypeIndex<T>::value;
    // store byte offset and setter, getter
    // so the same setter can be reused for all the same type
    info.offset = GetFieldByteOffsetToObject<Class, T>(field_ptr);
    info.size = sizeof(T);
    info.alignment = alignof(T);
    info.flags = 0;
    if (writable) {
      info.flags |= kTVMFFIFieldFlagBitMaskWritable;
    }
    info.getter = FieldGetter<T>;
    info.setter = FieldSetter<T>;
    // initialize default value to nullptr
    info.default_value = AnyView(nullptr).CopyToTVMFFIAny();
    info.doc = TVMFFIByteArray{nullptr, 0};
    info.metadata_.emplace_back("type_schema", details::TypeSchema<T>::v());
    // apply field info traits
    ((ApplyFieldInfoTrait(&info, std::forward<ExtraArgs>(extra_args)), ...));
    // call register
    std::string metadata_str = Metadata::ToJSON(info.metadata_);
    info.metadata = TVMFFIByteArray{metadata_str.c_str(), metadata_str.size()};
    TVM_FFI_CHECK_SAFE_CALL(TVMFFITypeRegisterField(type_index_, &info));
  }

  // register a method
  template <typename Func, typename... Extra>
  void RegisterMethod(const char* name, bool is_static, Func&& func, Extra&&... extra) {
    using FuncInfo = details::FunctionInfo<std::decay_t<Func>>;
    MethodInfoBuilder info;
    info.name = TVMFFIByteArray{name, std::char_traits<char>::length(name)};
    info.doc = TVMFFIByteArray{nullptr, 0};
    info.flags = 0;
    if (is_static) {
      info.flags |= kTVMFFIFieldFlagBitMaskIsStaticMethod;
    }

    auto method_name = std::string(type_key_) + "." + name;

    // if an overload method exists, register to existing overload function
    if (const auto overload_it = registered_fields_.find(name);
        overload_it != registered_fields_.end()) {
      details::OverloadBase* overload_ptr = overload_it->second;
      return overload_ptr->Register(NewOverload(std::move(method_name), std::forward<Func>(func)));
    }

    // first time registering overload method
    auto [method, overload_ptr] =
        GetOverloadMethod(std::move(method_name), std::forward<Func>(func));
    registered_fields_.try_emplace(name, overload_ptr);

    info.method = AnyView(method).CopyToTVMFFIAny();
    info.metadata_.emplace_back("type_schema", FuncInfo::TypeSchema());
    // apply method info traits
    ((ApplyMethodInfoTrait(&info, std::forward<Extra>(extra)), ...));
    std::string metadata_str = Metadata::ToJSON(info.metadata_);
    info.metadata = TVMFFIByteArray{metadata_str.c_str(), metadata_str.size()};
    TVM_FFI_CHECK_SAFE_CALL(TVMFFITypeRegisterMethod(type_index_, &info));
  }

  int32_t type_index_;
  const char* type_key_;
  std::unordered_map<std::string, details::OverloadBase*> registered_fields_;
  static constexpr const char* kInitMethodName = "__ffi_init__";
};

}  // namespace reflection
}  // namespace ffi
}  // namespace tvm
#endif  // TVM_FFI_EXTRA_OVERLOAD_H
