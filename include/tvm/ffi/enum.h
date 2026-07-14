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
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(EnumState, ObjectRef, EnumStateObj);
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
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Enum, ObjectRef, EnumObj);
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
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(IntEnum, Enum, IntEnumObj);
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
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(StrEnum, Enum, StrEnumObj);
  /// \endcond
};

template <typename Index>
inline Enum EnumObj::GetByIndex(int32_t type_index, const Index& index) {
  static reflection::TypeAttrColumn state_column(reflection::type_attr::kEnumState);
  if (AnyView value = state_column[type_index]; value != nullptr) {
    EnumState state = value.cast<EnumState>();
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
