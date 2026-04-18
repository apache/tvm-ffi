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
 * \file tvm/ffi/enum.h
 * \brief Base class for FFI-registered enum types.
 */
#ifndef TVM_FFI_ENUM_H_
#define TVM_FFI_ENUM_H_

#include <tvm/ffi/any.h>
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/container/dict.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/object.h>
#include <tvm/ffi/reflection/accessor.h>
#include <tvm/ffi/string.h>

#include <cstdint>
#include <type_traits>
#include <utility>

namespace tvm {
namespace ffi {

class Enum;

/*!
 * \brief Base class for FFI-registered enums.
 *
 * Each registered variant is a unique, process-wide singleton with a
 * dense ordinal (``value``) and string ``name``.  Subclasses may add
 * *declared fields* — part of the variant's schema, set at registration
 * time via ``reflection::EnumDef``.  Separately, any consumer may
 * attach *extensible attributes* (per-variant metadata stored outside
 * the variant's fields) via ``EnumDef::set_attr`` or the Python
 * ``Enum.def_attr`` surface, without modifying ``EnumClsObj``.
 *
 * \sa reflection::EnumDef
 */
class EnumObj : public Object {
 public:
  /*! \brief Declared field: dense ordinal assigned at registration time (0-indexed per class). */
  int64_t value;
  /*! \brief Declared field: instance name (e.g., ``"Add"`` for ``Op.Add``). */
  String name;

  EnumObj() = default;
  /*!
   * \brief Construct an EnumObj with an explicit ordinal and name.
   * \param value The dense ordinal (0-indexed per enum class).
   * \param name The instance name key.
   */
  EnumObj(int64_t value, String name) : value(value), name(std::move(name)) {}

  /*!
   * \brief Look up the registered singleton for ``EnumClsObj`` by name.
   *
   * Reads from the per-class ``reflection::type_attr::kEnumEntries``
   * registry populated by ``reflection::EnumDef<EnumClsObj>``.  Instances
   * are unique per ``(type_key, name)`` pair for the life of the process,
   * so the returned ``Enum`` compares equal (by pointer) to every other
   * lookup of the same name.  Throws ``RuntimeError`` if no instance with
   * the given name is registered for ``EnumClsObj``.
   *
   * \tparam EnumClsObj An ``Object`` subclass deriving from ``EnumObj``.
   * \param name The instance name to look up (e.g., ``"Add"``).
   * \return The registered ``Enum`` singleton.
   */
  template <typename EnumClsObj>
  static Enum Get(const String& name);

  /// \cond Doxygen_Suppress
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindUniqueInstance;
  TVM_FFI_DECLARE_OBJECT_INFO("ffi.Enum", EnumObj, Object);
  /// \endcond

 private:
  /*!
   * \brief Return the process-wide ``__ffi_enum_entries__`` column pointer.
   *
   * The column is registered at library init via ``EnsureTypeAttrColumn``
   * and the struct its pointer refers to is stable for the lifetime of the
   * process, so we cache the lookup in a function-local static.
   */
  static const TVMFFITypeAttrColumn* GetEnumEntriesColumn() {
    constexpr TVMFFIByteArray kAttrName =
        reflection::AsByteArray(reflection::type_attr::kEnumEntries);
    static const TVMFFITypeAttrColumn* column = TVMFFIGetTypeAttrColumn(&kAttrName);
    return column;
  }
};

/*!
 * \brief ObjectRef wrapper for ``EnumObj``.
 *
 * Holds a shared reference to a registered singleton.  Two ``Enum``
 * values compare structurally equal if and only if they point at the
 * same underlying object (see ``kTVMFFISEqHashKindUniqueInstance``),
 * which — given the register-once registry — is equivalent to sharing
 * the same ``(type_key, name)`` pair.
 *
 * \sa EnumObj
 * \sa reflection::EnumDef
 */
class Enum : public ObjectRef {
 public:
  /// \cond Doxygen_Suppress
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Enum, ObjectRef, EnumObj);
  /// \endcond
};

template <typename EnumClsObj>
inline Enum EnumObj::Get(const String& name) {
  static_assert(std::is_base_of_v<EnumObj, EnumClsObj>,
                "EnumObj::Get<T> requires T to be a subclass of EnumObj");
  const TVMFFITypeAttrColumn* column = GetEnumEntriesColumn();
  int32_t type_index = EnumClsObj::RuntimeTypeIndex();
  if (column != nullptr) {
    int32_t offset = type_index - column->begin_index;
    if (offset >= 0 && offset < column->size) {
      const TVMFFIAny* stored = &column->data[offset];
      if (stored->type_index != kTVMFFINone) {
        Dict<String, Enum> entries = AnyView::CopyFromTVMFFIAny(*stored).cast<Dict<String, Enum>>();
        auto it = entries.find(name);
        if (it != entries.end()) {
          return (*it).second;
        }
      }
    }
  }
  TVM_FFI_THROW(RuntimeError) << "Enum `" << EnumClsObj::_type_key << "` has no instance named `"
                              << name << "`";
  TVM_FFI_UNREACHABLE();
}

}  // namespace ffi
}  // namespace tvm

#endif  // TVM_FFI_ENUM_H_
