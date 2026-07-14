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
    ::tvm::ffi::details::ObjectUnsafe::GetHeader(obj.get())->type_index = type_index_;
    obj->_int_index = int_index;
    obj->_str_index = std::move(str_index);
    instance_ = Enum(ObjectPtr<EnumObj>(std::move(obj)));
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
      return ::tvm::ffi::details::ObjectUnsafe::ObjectRefFromObjectPtr<EnumState>(
          ::tvm::ffi::details::ObjectUnsafe::ObjectPtrFromUnowned<Object>(value_any.v_obj));
    }
    ObjectPtr<EnumStateObj> state_obj = make_object<EnumStateObj>();
    // GCC runs attribute constructors before dynamic initialization of inline type indices.
    // Set the header explicitly so the state is not packed as `None` during static init.
    ::tvm::ffi::details::ObjectUnsafe::GetHeader(state_obj.get())->type_index = state_type_index;
    EnumState state(std::move(state_obj));
    constexpr TVMFFIByteArray name = AsByteArray(type_attr::kEnumState);
    TVMFFIAny value_any = AnyView(state).CopyToTVMFFIAny();
    TVM_FFI_CHECK_SAFE_CALL(TVMFFITypeRegisterAttr(type_index_, &name, &value_any));
    return state;
  }

  int32_t type_index_;
  Enum instance_;
};

}  // namespace reflection
}  // namespace ffi
}  // namespace tvm

#endif  // TVM_FFI_REFLECTION_ENUM_DEF_H_
