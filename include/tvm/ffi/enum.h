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
#include <tvm/ffi/container/list.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/object.h>
#include <tvm/ffi/reflection/accessor.h>
#include <tvm/ffi/string.h>

#include <cstdint>
#include <optional>
#include <string_view>
#include <type_traits>
#include <utility>

namespace tvm {
namespace ffi {

class Enum;
class IntEnum;
class StrEnum;

/*! \brief Registry state shared by enum definition and lookup. */
class EnumStateObj : public Object {
 public:
  /*! \brief Enum singletons in registration order. */
  List<ObjectRef> entries;
  /*! \brief Canonical integer and string indices to enum singletons. */
  Dict<Any, ObjectRef> indexes;
  /*! \brief Extensible-attribute columns keyed by enum singleton. */
  Dict<String, Dict<ObjectRef, Any>> attrs;

  /// \cond Doxygen_Suppress
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ffi.EnumState", EnumStateObj, Object);
  /// \endcond
};

/*! \brief ObjectRef wrapper for ``EnumStateObj``. */
class EnumState : public ObjectRef {
 public:
  /// \cond Doxygen_Suppress
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(EnumState, ObjectRef, EnumStateObj);
  /// \endcond
};

/*! \brief Base class for FFI-registered enum singletons. */
class EnumObj : public Object {
 public:
  /*! \brief Canonical integer index. */
  int64_t _int_index = 0;
  /*! \brief Canonical string index. */
  String _str_index;

  EnumObj() = default;
  /*!
   * \brief Construct an enum singleton with canonical integer and string indices.
   * \param int_index The canonical integer index.
   * \param str_index The canonical string index.
   */
  EnumObj(int64_t int_index, String str_index)
      : _int_index(int_index), _str_index(std::move(str_index)) {}

  // NOLINTBEGIN(bugprone-reserved-identifier)
  /*!
   * \brief Look up a registered enum singleton by its canonical integer index.
   * \tparam EnumClsObj An ``Object`` subclass deriving from ``EnumObj``.
   * \param index The canonical integer index.
   * \return The registered ``Enum`` singleton.
   */
  template <typename EnumClsObj>
  static Enum _GetByIntIndex(int64_t index);
  /*!
   * \brief Look up a registered enum singleton by its canonical string index.
   * \tparam EnumClsObj An ``Object`` subclass deriving from ``EnumObj``.
   * \param index The canonical string index.
   * \return The registered ``Enum`` singleton.
   */
  template <typename EnumClsObj>
  static Enum _GetByStrIndex(const String& index);
  /*!
   * \brief Look up a registered enum singleton by runtime type and canonical integer index.
   * \param type_index The runtime type index of the registered enum class.
   * \param index The canonical integer index.
   * \return The registered ``Enum`` singleton.
   */
  static Enum _GetByIntIndex(int32_t type_index, int64_t index);
  /*!
   * \brief Look up a registered enum singleton by runtime type and canonical string index.
   * \param type_index The runtime type index of the registered enum class.
   * \param index The canonical string index.
   * \return The registered ``Enum`` singleton.
   */
  static Enum _GetByStrIndex(int32_t type_index, const String& index);
  // NOLINTEND(bugprone-reserved-identifier)

  /// \cond Doxygen_Suppress
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindUniqueInstance;
  TVM_FFI_DECLARE_OBJECT_INFO("ffi.Enum", EnumObj, Object);
  /// \endcond

 private:
  template <typename Index>
  static Enum GetByIndex(int32_t type_index, const Index& index);
};

/*!
 * \brief ObjectRef wrapper for ``EnumObj``.
 *
 * Holds a shared reference to a registered singleton.  Two ``Enum``
 * values compare structurally equal if and only if they point at the
 * same underlying object (see ``kTVMFFISEqHashKindUniqueInstance``).
 *
 * \sa EnumObj
 * \sa reflection::EnumDef
 */
class Enum : public ObjectRef {
 public:
  /// \cond Doxygen_Suppress
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Enum, ObjectRef, EnumObj);
  /// \endcond
};

/*!
 * \brief Base object for payload enums whose public value is an integer.
 */
class IntEnumObj : public EnumObj {
 public:
  /// \cond Doxygen_Suppress
  TVM_FFI_DECLARE_OBJECT_INFO("ffi.IntEnum", IntEnumObj, EnumObj);
  /// \endcond
};

/*!
 * \brief ObjectRef wrapper for ``IntEnumObj``.
 */
class IntEnum : public Enum {
 public:
  /// \cond Doxygen_Suppress
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(IntEnum, Enum, IntEnumObj);
  /// \endcond
};

/*!
 * \brief Base object for payload enums whose public value is a string.
 */
class StrEnumObj : public EnumObj {
 public:
  /// \cond Doxygen_Suppress
  TVM_FFI_DECLARE_OBJECT_INFO("ffi.StrEnum", StrEnumObj, EnumObj);
  /// \endcond
};

/*!
 * \brief ObjectRef wrapper for ``StrEnumObj``.
 */
class StrEnum : public Enum {
 public:
  /// \cond Doxygen_Suppress
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(StrEnum, Enum, StrEnumObj);
  /// \endcond
};

namespace type_attr {
/*! \brief Per-class enum state: ordered entries, canonical indices, and extensible attrs. */
inline constexpr reflection::AttrKey<EnumState> kEnumState{"__ffi_enum__"};
}  // namespace type_attr

namespace reflection {

/*!
 * \brief A typed enum attribute column indexed by singleton identity.
 *
 * Construction resolves an existing column once. Like TypeAttrColumn, it throws
 * if the column is missing; construct it after the attribute is registered.
 * The shared column reflects subsequent EnumDef::set_attr calls, including new
 * entries and updates. Registration must not run concurrently with lookups.
 *
 * \tparam T The attribute value type, extracted strictly as with AttrKey::ValueFrom.
 * \sa EnumDef
 */
template <typename T>
class EnumAttrColumn {
 public:
  /*!
   * \brief Resolve an enum type's existing attribute column.
   * \param type_index The enum's runtime type index.
   * \param attr_name The attribute name.
   * \throws Error if the enum state or attribute column is missing.
   */
  EnumAttrColumn(int32_t type_index, std::string_view attr_name)
      : column_(GetColumn(type_index, attr_name)) {}

  /*! \brief Resolve a column using its key's declared value type. */
  EnumAttrColumn(int32_t type_index, AttrKey<T> key) : EnumAttrColumn(type_index, key.name) {}

  /*!
   * \brief Look up an attribute by enum singleton identity.
   * \param instance The enum singleton; names and integer indices are not used.
   * \return The typed value, or std::nullopt for a missing or None entry.
   * \throws Error if a present value has the wrong type. No fallback conversion is run.
   * \note Borrowed results remain valid only while the stored value remains alive.
   */
  std::optional<T> operator[](const Enum& instance) const {
    auto value = column_.Get(instance);
    if (!value || *value == nullptr) return std::nullopt;
    return value->template as_or_throw<T>();
  }

 private:
  static Dict<ObjectRef, Any> GetColumn(int32_t type_index, std::string_view attr_name) {
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
      if (auto column = state->attrs.Get(String(attr_name.data(), attr_name.size()))) {
        return *column;
      }
    }
    TVM_FFI_THROW(RuntimeError) << "Cannot find enum attribute " << attr_name << " for type `"
                                << TypeIndexToTypeKey(type_index) << "`";
    TVM_FFI_UNREACHABLE();
  }

  Dict<ObjectRef, Any> column_;
};

}  // namespace reflection

template <typename Index>
inline Enum EnumObj::GetByIndex(int32_t type_index, const Index& index) {
  static reflection::TypeAttrColumn state_column(type_attr::kEnumState);
  if (auto value = type_attr::kEnumState.ValueFrom(state_column[type_index])) {
    const EnumState& state = *value;
    if (auto entry = state->indexes.Get(Any(index))) return entry->as_or_throw<Enum>();
  }
  TVM_FFI_THROW(ValueError) << "Enum `" << TypeIndexToTypeKey(type_index)
                            << "` has no instance with index " << index;
  TVM_FFI_UNREACHABLE();
}

// NOLINTBEGIN(bugprone-reserved-identifier)
template <typename EnumClsObj>
inline Enum EnumObj::_GetByIntIndex(int64_t index) {
  static_assert(std::is_base_of_v<EnumObj, EnumClsObj>);
  return _GetByIntIndex(EnumClsObj::_GetOrAllocRuntimeTypeIndex(), index);
}

template <typename EnumClsObj>
inline Enum EnumObj::_GetByStrIndex(const String& index) {
  static_assert(std::is_base_of_v<EnumObj, EnumClsObj>);
  return _GetByStrIndex(EnumClsObj::_GetOrAllocRuntimeTypeIndex(), index);
}

inline Enum EnumObj::_GetByIntIndex(int32_t type_index, int64_t index) {
  return GetByIndex(type_index, index);
}

inline Enum EnumObj::_GetByStrIndex(int32_t type_index, const String& index) {
  return GetByIndex(type_index, index);
}
// NOLINTEND(bugprone-reserved-identifier)

}  // namespace ffi
}  // namespace tvm

#endif  // TVM_FFI_ENUM_H_
