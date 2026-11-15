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
 * \file tvm/ffi/reflection/enum_def.h
 * \brief Builder for registering enum instances on ``EnumObj`` subclasses.
 */
#ifndef TVM_FFI_REFLECTION_ENUM_DEF_H_
#define TVM_FFI_REFLECTION_ENUM_DEF_H_

#include <tvm/ffi/any.h>
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/container/dict.h>
#include <tvm/ffi/container/list.h>
#include <tvm/ffi/enum.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/memory.h>
#include <tvm/ffi/object.h>
#include <tvm/ffi/reflection/accessor.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>

#include <cstdint>
#include <optional>
#include <type_traits>
#include <utility>

namespace tvm {
namespace ffi {
namespace reflection {

/*! \brief Register one enum singleton and its canonical indices. */
template <typename EnumClsObj, typename = std::enable_if_t<std::is_base_of_v<EnumObj, EnumClsObj>>>
class EnumDef : public ReflectionDefBase {
 public:
  /*!
   * \brief Check whether a canonical name is defined without registering enum state or entries.
   * \param name The canonical string index.
   * \return Whether the name is defined for this enum type.
   */
  static bool IsDefined(const String& name) {
    constexpr TVMFFIByteArray column_name = AsByteArray(type_attr::kEnumState.name);
    if (TVMFFIGetTypeAttrColumn(&column_name) == nullptr) return false;
    AnyView value =
        TypeAttrColumn(type_attr::kEnumState)[EnumClsObj::_GetOrAllocRuntimeTypeIndex()];
    if (value == nullptr) return false;
    if (value.type_index() != EnumStateObj::_GetOrAllocRuntimeTypeIndex()) {
      TVM_FFI_THROW(TypeError) << "Expected `" << EnumStateObj::_type_key
                               << "` in enum state column, but got `" << value.GetTypeKey() << "`";
    }
    // Avoid cast<EnumState> because its inline type index may not be initialized yet.
    const auto* state = static_cast<const EnumStateObj*>(value.cast<ObjectRef>().get());
    return state->indexes.count(name) != 0;
  }

  /*!
   * \brief Register an enum singleton with an automatically assigned integer index.
   *
   * The integer index is assigned from the singleton's registration order.
   *
   * \param str_index The canonical string index.
   */
  explicit EnumDef(const char* str_index) : type_index_(EnumClsObj::_GetOrAllocRuntimeTypeIndex()) {
    static_assert(!std::is_base_of_v<IntEnumObj, EnumClsObj>,
                  "IntEnum registration requires an explicit integer index");
    EnumState state = EnsureState();
    Register(state, static_cast<int64_t>(state->entries.size()), String(str_index));
  }

  /*!
   * \brief Register an integer enum singleton with explicit canonical indices.
   *
   * \param str_index The canonical string index.
   * \param int_index The canonical integer index.
   */
  EnumDef(const char* str_index, int64_t int_index)
      : type_index_(EnumClsObj::_GetOrAllocRuntimeTypeIndex()) {
    static_assert(std::is_base_of_v<IntEnumObj, EnumClsObj>,
                  "Explicit integer indices are reserved for IntEnum");
    Register(EnsureState(), int_index, String(str_index));
  }

  /*!
   * \brief Set an extensible attribute on the registered enum singleton.
   *
   * \tparam T The attribute value type.
   * \param attr_name The attribute name.
   * \param value The attribute value.
   * \return Reference to this builder for method chaining.
   */
  template <typename T>
  EnumDef& set_attr(const char* attr_name, T value) {
    Dict<String, Dict<ObjectRef, Any>> attrs = EnsureState()->attrs;
    String key(attr_name);
    Dict<ObjectRef, Any> column = attrs.Get(key).value_or(Dict<ObjectRef, Any>());
    attrs.Set(key, column);
    column.Set(instance_, Any(std::move(value)));
    return *this;
  }

  /*! \brief Set an enum attribute using its key's declared value type. */
  template <typename T, typename Value>
  EnumDef& set_attr(AttrKey<T> key, Value&& value) {
    return set_attr(key.name, T(std::forward<Value>(value)));
  }

  /*! \brief Set a named native callback using its key's declared adapter. */
  template <auto Fn, typename T>
  EnumDef& set_native_fn_attr(AttrKey<T> key) {
    return set_attr(key, T::template FromNative<Fn>());
  }

 private:
  void Register(const EnumState& state, int64_t int_index, String str_index) {
    List<ObjectRef> entries = state->entries;
    Dict<Any, ObjectRef> indexes = state->indexes;
    Any int_key(int_index);
    Any str_key(str_index);
    if (indexes.count(int_key) || indexes.count(str_key)) {
      TVM_FFI_THROW(ValueError) << "Duplicate enum index for type `" << EnumClsObj::_type_key
                                << "`";
    }
    ObjectPtr<EnumClsObj> obj = make_object<EnumClsObj>();
    ::tvm::ffi::detail::ObjectUnsafe::GetHeader(obj.get())->type_index = type_index_;
    obj->_int_index = int_index;
    obj->_str_index = std::move(str_index);
    instance_ = ::tvm::ffi::detail::ObjectUnsafe::ObjectRefFromObjectPtr<Enum>(std::move(obj));
    entries.push_back(instance_);
    indexes.Set(int_key, instance_);
    indexes.Set(str_key, instance_);
  }

  EnumState EnsureState() {
    int32_t state_type_index = EnumStateObj::_GetOrAllocRuntimeTypeIndex();
    TypeAttrColumn column(type_attr::kEnumState);
    if (AnyView value = column[type_index_]; value != nullptr) {
      if (value.type_index() != state_type_index) {
        TVM_FFI_THROW(TypeError) << "Expected `" << EnumStateObj::_type_key
                                 << "` in enum state column, but got `" << value.GetTypeKey()
                                 << "`";
      }
      // Avoid cast<EnumState> here because it reads the same deferred inline type index.
      TVMFFIAny value_any = value.CopyToTVMFFIAny();
      return ::tvm::ffi::detail::ObjectUnsafe::ObjectRefFromObjectPtr<EnumState>(
          ::tvm::ffi::detail::ObjectUnsafe::ObjectPtrFromUnowned<Object>(value_any.v_obj));
    }
    ObjectPtr<EnumStateObj> state_obj = make_object<EnumStateObj>();
    // GCC runs attribute constructors before dynamic initialization of inline type indices.
    // Set the header explicitly so the state is not packed as `None` during static init.
    ::tvm::ffi::detail::ObjectUnsafe::GetHeader(state_obj.get())->type_index = state_type_index;
    EnumState state =
        ::tvm::ffi::detail::ObjectUnsafe::ObjectRefFromObjectPtr<EnumState>(std::move(state_obj));
    constexpr TVMFFIByteArray name = AsByteArray(type_attr::kEnumState.name);
    TVMFFIAny value_any = AnyView(state).CopyToTVMFFIAny();
    TVM_FFI_CHECK_SAFE_CALL(TVMFFITypeRegisterAttr(type_index_, &name, &value_any));
    return state;
  }

  int32_t type_index_;
  Enum instance_{UnsafeInit{}};
};

/*!
 * \brief A typed enum attribute map indexed by singleton identity.
 *
 * Construction resolves an existing column once. Like TypeAttrColumn, it throws
 * if the column is missing; construct it after the attribute is registered.
 * The shared column reflects subsequent EnumDef::set_attr calls, including new
 * entries and updates. Registration must not run concurrently with lookups.
 *
 * \tparam EnumClsObj An Object subclass deriving from EnumObj.
 * \tparam T The attribute value type, extracted strictly with AttrKey::ValueFrom.
 * \sa GetEnumAttrMap
 */
template <typename EnumClsObj, typename T>
class EnumAttrMap {
  static_assert(std::is_base_of_v<EnumObj, EnumClsObj>, "EnumClsObj must derive from EnumObj");

 public:
  /*!
   * \brief Resolve an enum type's existing attribute column using a typed key.
   * \param key The attribute key.
   * \throws Error if the enum state or attribute column is missing.
   */
  explicit EnumAttrMap(AttrKey<T> key) : column_(GetColumn(key)), key_(key) {}

  /*!
   * \brief Look up an attribute by enum singleton identity.
   * \param instance The enum singleton; names and integer indices are not used.
   * \return The typed value, or std::nullopt for a missing or None entry.
   * \throws Error if a present value has the wrong type. No fallback conversion is run.
   * \note Borrowed results remain valid only while the stored value remains alive.
   */
  std::optional<T> operator[](const Enum& instance) const {
    if (auto value = column_.Get(instance)) return key_.ValueFrom(*value);
    return std::nullopt;
  }

 private:
  static Dict<ObjectRef, Any> GetColumn(AttrKey<T> key) {
    int32_t type_index = EnumClsObj::_GetOrAllocRuntimeTypeIndex();
    TypeAttrColumn states(type_attr::kEnumState);
    AnyView value = states[type_index];
    if (value != nullptr) {
      if (value.type_index() != EnumStateObj::_GetOrAllocRuntimeTypeIndex()) {
        TVM_FFI_THROW(TypeError) << "Expected `" << EnumStateObj::_type_key
                                 << "` in enum state column, but got `" << value.GetTypeKey()
                                 << "`";
      }
      // Inline type indices may not be initialized yet during static registration.
      const auto* state = static_cast<const EnumStateObj*>(value.cast<ObjectRef>().get());
      if (auto column = state->attrs.Get(String(key.name))) {
        return *column;
      }
    }
    TVM_FFI_THROW(RuntimeError) << "Cannot find enum attribute " << key.name << " for type `"
                                << TypeIndexToTypeKey(type_index) << "`";
    TVM_FFI_UNREACHABLE();
  }

  Dict<ObjectRef, Any> column_;
  AttrKey<T> key_;
};

/*!
 * \brief Resolve a typed enum attribute map, deducing the value type from its key.
 * \tparam EnumClsObj An Object subclass deriving from EnumObj.
 * \tparam T The attribute value type.
 * \param key The attribute key.
 * \return The attribute map for the enum type.
 * \throws Error if the enum state or attribute column is missing.
 */
template <typename EnumClsObj, typename T>
EnumAttrMap<EnumClsObj, T> GetEnumAttrMap(AttrKey<T> key) {
  return EnumAttrMap<EnumClsObj, T>(key);
}

}  // namespace reflection
}  // namespace ffi
}  // namespace tvm

#endif  // TVM_FFI_REFLECTION_ENUM_DEF_H_
