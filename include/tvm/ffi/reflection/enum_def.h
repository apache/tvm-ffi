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
#include <string>
#include <type_traits>
#include <utility>

namespace tvm {
namespace ffi {
namespace reflection {

/*!
 * \brief Builder that registers a single enum instance on ``EnumClsObj``.
 *
 * Each ``EnumDef<EnumClsObj>("Name")`` call allocates a fresh dense ordinal
 * (``= len(existing entries)``), constructs a variant with ``value`` and
 * ``name`` populated, and writes it into the per-class registry stored in
 * the ``type_attr::kEnumEntries`` TypeAttr column.  Subsequent
 * ``.set_attr(...)`` calls write *extensible attributes* — per-variant
 * metadata attached outside the variant's declared fields — into the
 * per-class ``type_attr::kEnumAttrs`` column.  Python bindings of the
 * same ``type_key`` see every C++-registered variant and every extensible
 * attribute through the matching ``Enum.def_attr`` surface.
 *
 * \tparam EnumClsObj An ``Object`` subclass deriving from ``EnumObj``.
 *
 * \code{.cpp}
 * namespace refl = ::tvm::ffi::reflection;
 * refl::EnumDef<OpObj>("Add").set_attr("has_side_effects", false);
 * refl::EnumDef<OpObj>("Mul").set_attr("has_side_effects", false);
 * \endcode
 */
template <typename EnumClsObj, typename = std::enable_if_t<std::is_base_of_v<EnumObj, EnumClsObj>>>
class EnumDef : public ReflectionDefBase {
 public:
  /*!
   * \brief Register a new instance named ``instance_name`` on ``EnumClsObj``.
   * \param instance_name The instance's string name (e.g., ``"Add"``).
   */
  explicit EnumDef(const char* instance_name)
      : type_index_(EnumClsObj::RuntimeTypeIndex()), name_(instance_name) {
    Dict<String, Enum> entries = EnsureEntriesDict();
    String name_str(name_);
    if (entries.count(name_str) != 0) {
      TVM_FFI_THROW(RuntimeError) << "Duplicate enum entry `" << name_ << "` for type `"
                                  << EnumClsObj::_type_key << "`";
    }
    ordinal_ = static_cast<int64_t>(entries.size());
    ObjectPtr<EnumClsObj> obj = make_object<EnumClsObj>();
    obj->value = ordinal_;
    obj->name = name_str;
    instance_ = Enum(ObjectPtr<EnumObj>(std::move(obj)));
    entries.Set(name_str, instance_);
    // Ensure the attrs dict exists so later ``set_attr`` calls can mutate it.
    EnsureAttrsDict();
  }

  /*!
   * \brief Write an *extensible attribute* for this enum variant.
   *
   * Writes land in the per-class ``type_attr::kEnumAttrs`` column and
   * are visible to every binder of the same ``type_key`` — including
   * Python readers via ``Enum.def_attr`` / ``Enum.attr_dict``.  Distinct
   * from declared fields on ``EnumClsObj``: declared fields are part of
   * the variant's schema and set during construction, whereas
   * extensible attributes live outside the variant object and may be
   * attached by any consumer at any time.
   *
   * \tparam T The value type.
   * \param attr_name The extensible-attribute name (e.g.,
   *        ``"has_side_effects"``).
   * \param value The value to store for this variant's ordinal.
   * \return Reference to this builder for chaining.
   */
  template <typename T>
  EnumDef& set_attr(const char* attr_name, T value) {
    Dict<String, List<Any>> attrs = EnsureAttrsDict();
    String attr_key(attr_name);
    List<Any> column;
    auto it = attrs.find(attr_key);
    if (it == attrs.end()) {
      column = List<Any>();
      attrs.Set(attr_key, column);
    } else {
      column = (*it).second;
    }
    while (static_cast<int64_t>(column.size()) <= ordinal_) {
      column.push_back(Any(nullptr));
    }
    column.Set(ordinal_, Any(std::move(value)));
    return *this;
  }

  /*! \brief Return the registered instance (for tests / advanced callers). */
  Enum instance() const { return instance_; }

  /*! \brief Return the ordinal assigned to this instance. */
  int64_t ordinal() const { return ordinal_; }

 private:
  Dict<String, Enum> EnsureEntriesDict() {
    return EnsureDict<Dict<String, Enum>>(type_attr::kEnumEntries);
  }

  Dict<String, List<Any>> EnsureAttrsDict() {
    return EnsureDict<Dict<String, List<Any>>>(type_attr::kEnumAttrs);
  }

  template <typename DictT>
  DictT EnsureDict(const char* attr_name) {
    TVMFFIByteArray name_array = {attr_name, std::char_traits<char>::length(attr_name)};
    const TVMFFITypeAttrColumn* column = TVMFFIGetTypeAttrColumn(&name_array);
    if (column != nullptr) {
      int32_t offset = type_index_ - column->begin_index;
      if (offset >= 0 && offset < column->size) {
        const TVMFFIAny* stored = &column->data[offset];
        if (stored->type_index != kTVMFFINone) {
          return AnyView::CopyFromTVMFFIAny(*stored).cast<DictT>();
        }
      }
    }
    DictT fresh;
    TVMFFIAny value_any = AnyView(fresh).CopyToTVMFFIAny();
    TVM_FFI_CHECK_SAFE_CALL(TVMFFITypeRegisterAttr(type_index_, &name_array, &value_any));
    return fresh;
  }

  int32_t type_index_;
  const char* name_;
  int64_t ordinal_;
  Enum instance_;
};

}  // namespace reflection
}  // namespace ffi
}  // namespace tvm

#endif  // TVM_FFI_REFLECTION_ENUM_DEF_H_
