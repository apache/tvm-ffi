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
 * \file tvm/ffi/reflection/accessor.h
 * \brief Reflection-based accessor for object fields and methods.
 */
#ifndef TVM_FFI_REFLECTION_ACCESSOR_H_
#define TVM_FFI_REFLECTION_ACCESSOR_H_

#include <tvm/ffi/any.h>
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/type_traits.h>

#include <string>

namespace tvm {
namespace ffi {
namespace reflection {

/*!
 * \brief helper function to get reflection field info by type key and field name
 */
inline const TVMFFIFieldInfo* GetFieldInfo(std::string_view type_key, const char* field_name) {
  int32_t type_index;
  TVMFFIByteArray type_key_array = {type_key.data(), type_key.size()};
  TVM_FFI_CHECK_SAFE_CALL(TVMFFITypeKeyToIndex(&type_key_array, &type_index));
  const TypeInfo* info = TVMFFIGetTypeInfo(type_index);
  for (int32_t i = 0; i < info->num_fields; ++i) {
    if (std::strncmp(info->fields[i].name.data, field_name, info->fields[i].name.size) == 0) {
      return &(info->fields[i]);
    }
  }
  TVM_FFI_THROW(RuntimeError) << "Cannot find field  `" << field_name << "` in " << type_key;
  TVM_FFI_UNREACHABLE();
}

/*!
 * \brief Call the field setter, dispatching between function pointer and FunctionObj.
 *
 * When kTVMFFIFieldFlagBitSetterIsFunctionObj is off, invokes the setter as a
 * TVMFFIFieldSetter function pointer.  When on, calls via TVMFFIFunctionCall
 * with arguments (field_addr as OpaquePtr, value).
 *
 * \param field_info The field info containing the setter.
 * \param field_addr The address of the field in the object.
 * \param value The value to set (as a TVMFFIAny pointer).
 * \return 0 on success, nonzero on failure.
 */
inline int CallFieldSetter(const TVMFFIFieldInfo* field_info, void* field_addr,
                           const TVMFFIAny* value) {
  if (!(field_info->flags & kTVMFFIFieldFlagBitSetterIsFunctionObj)) {
    auto setter = reinterpret_cast<TVMFFIFieldSetter>(field_info->setter);
    return setter(field_addr, value);
  } else {
    TVMFFIAny args[2]{};
    args[0].type_index = kTVMFFIOpaquePtr;
    args[0].v_ptr = field_addr;
    args[1] = *value;
    TVMFFIAny result{};
    result.type_index = kTVMFFINone;
    return TVMFFIFunctionCall(static_cast<TVMFFIObjectHandle>(field_info->setter), args, 2,
                              &result);
  }
}

/*!
 * \brief helper wrapper class to obtain a getter.
 */
class FieldGetter {
 public:
  /*!
   * \brief Constructor
   * \param field_info The field info.
   */
  explicit FieldGetter(const TVMFFIFieldInfo* field_info) : field_info_(field_info) {}

  /*!
   * \brief Constructor
   * \param type_key The type key.
   * \param field_name The name of the field.
   */
  explicit FieldGetter(std::string_view type_key, const char* field_name)
      : FieldGetter(GetFieldInfo(type_key, field_name)) {}

  /*!
   * \brief Get the value of the field
   * \param obj_ptr The object pointer.
   * \return The value of the field.
   */
  Any operator()(const Object* obj_ptr) const {
    Any result;
    const void* addr = reinterpret_cast<const char*>(obj_ptr) + field_info_->offset;
    TVM_FFI_CHECK_SAFE_CALL(
        field_info_->getter(const_cast<void*>(addr), reinterpret_cast<TVMFFIAny*>(&result)));
    return result;
  }

  Any operator()(const ObjectPtr<Object>& obj_ptr) const { return operator()(obj_ptr.get()); }

  Any operator()(const ObjectRef& obj) const { return operator()(obj.get()); }

 private:
  const TVMFFIFieldInfo* field_info_;
};

/*!
 * \brief helper wrapper class to obtain a setter.
 */
class FieldSetter {
 public:
  /*!
   * \brief Constructor
   * \param field_info The field info.
   */
  explicit FieldSetter(const TVMFFIFieldInfo* field_info) : field_info_(field_info) {}

  /*!
   * \brief Constructor
   * \param type_key The type key.
   * \param field_name The name of the field.
   */
  explicit FieldSetter(std::string_view type_key, const char* field_name)
      : FieldSetter(GetFieldInfo(type_key, field_name)) {}

  /*!
   * \brief Set the value of the field
   * \param obj_ptr The object pointer.
   * \param value The value to be set.
   */
  void operator()(const Object* obj_ptr, AnyView value) const {
    const void* addr = reinterpret_cast<const char*>(obj_ptr) + field_info_->offset;
    TVM_FFI_CHECK_SAFE_CALL(CallFieldSetter(field_info_, const_cast<void*>(addr),
                                            reinterpret_cast<const TVMFFIAny*>(&value)));
  }

  void operator()(const ObjectPtr<Object>& obj_ptr, AnyView value) const {
    operator()(obj_ptr.get(), value);
  }

  void operator()(const ObjectRef& obj, AnyView value) const { operator()(obj.get(), value); }

 private:
  const TVMFFIFieldInfo* field_info_;
};

/*!
 * \brief Helper class to get type attribute column.
 */
class TypeAttrColumn {
 public:
  /*!
   * \brief Constructor
   * \param attr_name The name of the type attribute.
   */
  explicit TypeAttrColumn(std::string_view attr_name) {
    TVMFFIByteArray attr_name_array = {attr_name.data(), attr_name.size()};
    column_ = TVMFFIGetTypeAttrColumn(&attr_name_array);
    if (column_ == nullptr) {
      TVM_FFI_THROW(RuntimeError) << "Cannot find type attribute " << attr_name;
    }
  }
  /*!
   * \brief Get the type attribute column by type index.
   * \param type_index The type index.
   * \return The type attribute column.
   */
  AnyView operator[](int32_t type_index) const {
    int32_t offset = type_index - column_->begin_index;
    if (offset < 0 || offset >= column_->size) {
      return AnyView();
    }
    const AnyView* any_view_data = reinterpret_cast<const AnyView*>(column_->data);
    return any_view_data[offset];
  }

 private:
  const TVMFFITypeAttrColumn* column_;
};

/*!
 * \brief helper function to get reflection method info by type key and method name
 *
 * \param type_key The type key.
 * \param method_name The name of the method.
 * \return The method info.
 */
inline const TVMFFIMethodInfo* GetMethodInfo(std::string_view type_key, const char* method_name) {
  int32_t type_index;
  TVMFFIByteArray type_key_array = {type_key.data(), type_key.size()};
  TVM_FFI_CHECK_SAFE_CALL(TVMFFITypeKeyToIndex(&type_key_array, &type_index));
  const TypeInfo* info = TVMFFIGetTypeInfo(type_index);
  for (int32_t i = 0; i < info->num_methods; ++i) {
    if (std::strncmp(info->methods[i].name.data, method_name, info->methods[i].name.size) == 0) {
      return &(info->methods[i]);
    }
  }
  TVM_FFI_THROW(RuntimeError) << "Cannot find method " << method_name << " in " << type_key;
  TVM_FFI_UNREACHABLE();
}

/*!
 * \brief helper function to get reflection method function by method info
 *
 * \param type_key The type key.
 * \param method_name The name of the method.
 * \return The method function.
 */
inline Function GetMethod(std::string_view type_key, const char* method_name) {
  const TVMFFIMethodInfo* info = GetMethodInfo(type_key, method_name);
  return AnyView::CopyFromTVMFFIAny(info->method).cast<Function>();
}

/*!
 * \brief Set a field to its default value, calling the factory if applicable.
 *
 * When kTVMFFIFieldFlagBitMaskDefaultFromFactory is set, extracts the
 * Function from default_value_or_factory, calls it with no arguments,
 * and uses the result. Otherwise, passes default_value_or_factory directly
 * to the setter.
 *
 * \param field_info The field info (must have kTVMFFIFieldFlagBitMaskHasDefault set).
 * \param field_addr The address of the field in the object.
 */
inline void SetFieldToDefault(const TVMFFIFieldInfo* field_info, void* field_addr) {
  if (field_info->flags & kTVMFFIFieldFlagBitMaskDefaultFromFactory) {
    Function factory =
        AnyView::CopyFromTVMFFIAny(field_info->default_value_or_factory).cast<Function>();
    Any default_val = factory();
    CallFieldSetter(field_info, field_addr, reinterpret_cast<const TVMFFIAny*>(&default_val));
  } else {
    CallFieldSetter(field_info, field_addr, &(field_info->default_value_or_factory));
  }
}

/*!
 * \brief Visit each field info of the type info and run callback.
 *
 * \tparam Callback The callback function type.
 *
 * \param type_info The type info.
 * \param callback The callback function.
 *
 * \note This function calls both the child and parent type info.
 */
template <typename Callback>
inline void ForEachFieldInfo(const TypeInfo* type_info, Callback callback) {
  using ResultType = decltype(callback(type_info->fields));
  static_assert(std::is_same_v<ResultType, void>, "Callback must return void");
  // iterate through acenstors in parent to child order
  // skip the first one since it is always the root object
  for (int i = 1; i < type_info->type_depth; ++i) {
    const TVMFFITypeInfo* parent_info = type_info->type_ancestors[i];
    for (int j = 0; j < parent_info->num_fields; ++j) {
      callback(parent_info->fields + j);
    }
  }
  for (int i = 0; i < type_info->num_fields; ++i) {
    callback(type_info->fields + i);
  }
}

/*!
 * \brief Visit each field info of the type info and run callback which returns bool for early stop.
 *
 * \tparam Callback The callback function type, which returns bool for early stop.
 *
 * \param type_info The type info.
 * \param callback_with_early_stop The callback function.
 * \return true if any of early stop is triggered.
 *
 * \note This function calls both the child and parent type info and can be used for searching.
 */
template <typename Callback>
inline bool ForEachFieldInfoWithEarlyStop(const TypeInfo* type_info,
                                          Callback callback_with_early_stop) {
  // iterate through acenstors in parent to child order
  // skip the first one since it is always the root object
  for (int i = 1; i < type_info->type_depth; ++i) {
    const TVMFFITypeInfo* parent_info = type_info->type_ancestors[i];
    for (int j = 0; j < parent_info->num_fields; ++j) {
      if (callback_with_early_stop(parent_info->fields + j)) return true;
    }
  }
  for (int i = 0; i < type_info->num_fields; ++i) {
    if (callback_with_early_stop(type_info->fields + i)) return true;
  }
  return false;
}

/*!
 * \brief Type attribute names used by the reflection system.
 *
 * Each constant names a TypeAttrColumn — a sparse, type-indexed slot that
 * stores a ``Function`` or any other Any-compliant value.
 *
 *  - ``TypeAttrDef<T>.def(type_attr::kFoo, ...)`` to register a hook,
 *  - ``TypeAttrColumn(type_attr::kFoo)[type_index]`` to look one up,
 *
 */
namespace type_attr {
/*!
 * \brief Zero-arg object allocator.
 *
 * Allocates a zero-initialised object of the registered type.  For C++ types
 * this wraps ``metadata->creator``; for Python ``@py_class`` types it is a
 * ``calloc``-based allocator.
 *
 * Signature: ``() -> TSelf``, where ``TSelf`` is a subclass of ObjectRef.
 */
inline constexpr const char* kNew = "__ffi_new__";
/*!
 * \brief Packed constructor — creates **and** initialises a new object.
 *
 * Used by ``@c_class`` auto-generated ``__init__``: the Python side calls
 * ``self.__init_handle_by_constructor__(ffi_init, *args)`` which invokes
 * this function to produce a fully initialised object.
 *
 * For C++ types with ``refl::init<Args...>()``, the function is the explicit
 * init; otherwise ``ffi._RegisterFFIInit`` generates a default one using reflection.
 *
 * Signature: ``(*args, **kwargs) -> TSelf``, where ``TSelf`` is a subclass of ObjectRef.
 *
 * Keyword arguments are packed as ``[KWARGS, key0, val0, key1, val1, ...]``.
 */
inline constexpr const char* kInit = "__ffi_init__";
/*!
 * \brief In-place init on a pre-allocated object (no allocation).
 *
 * Used by ``@py_class`` auto-generated ``__init__``: ``__new__`` has already
 * allocated the object via ``kNew``, so this function only sets fields.
 * The first argument is ``self`` (the pre-allocated object).
 *
 * Signature: ``(self: TSelf, *args, **kwargs) -> void``, where ``TSelf`` is a subclass of
 * ObjectRef.
 *
 * Keyword arguments are packed as ``[KWARGS, key0, val0, key1, val1, ...]``.
 */
inline constexpr const char* kInitInplace = "__ffi_init_inplace__";
/*!
 * \brief Convert ``AnyView`` to a specific reflected ``TSelf`` type.
 *
 * Registered via ``TypeAttrDef<T>.def(kConvert, &FFIConvertFromAnyViewToObjectRef<T>)``
 * for every type that calls ``.ref<T>()``.  Used by the Python type converter
 * to marshal values into the correct ``TSelf`` subclass.
 *
 * Signature: ``(AnyView src) -> TSelf``, where ``TSelf`` is a subclass of ObjectRef.
 */
inline constexpr const char* kConvert = "__ffi_convert__";
/*!
 * \brief Shallow-copy factory.
 *
 * Allocates a new object and copies all reflected field values from the
 * source.  Used by Python ``copy.copy()`` via ``__copy__`` and by
 * ``copy.replace()`` via ``__replace__``.
 *
 * Signature: ``(TSelf self) -> TSelf``, where ``TSelf`` is a subclass of ObjectRef.
 */
inline constexpr const char* kShallowCopy = "__ffi_shallow_copy__";
/*!
 * \brief Custom recursive repr hook.
 *
 * If registered, ``RecursiveRepr`` (Python ``__repr__``) calls this instead
 * of the default field-by-field formatting.  The hook receives a callback
 * ``fn_repr`` to recursively format nested values, avoiding infinite loops.
 *
 * Signature: ``(TSelf self, fn_repr: FnRepr) -> String``, where ``TSelf`` is a subclass of
 * ObjectRef, and ``FnRepr: (AnyView value) -> String`` formats a nested value.
 */
inline constexpr const char* kRepr = "__ffi_repr__";
/*!
 * \brief Custom recursive hash hook.
 *
 * If registered, ``RecursiveHash`` (Python ``hash()``) calls this instead
 * of the default field-by-field hashing.  The hook receives a callback
 * ``fn_hash`` to recursively hash nested values, with cycle detection.
 *
 * Signature: ``(TSelf self, fn_hash: FnHash) -> int64``, where ``TSelf`` is a subclass of
 * ObjectRef, and ``FnHash: (AnyView value) -> int64`` hashes a nested value.
 */
inline constexpr const char* kHash = "__ffi_hash__";
/*!
 * \brief Custom recursive equality hook.
 *
 * If registered, ``RecursiveEq`` (Python ``==``) calls this instead of the
 * default field-by-field comparison.  Falls back to ``kCompare`` if only the
 * compare hook is present.  The hook receives a callback ``fn_eq`` to
 * recursively compare nested values.
 *
 * Signature: ``(TSelf lhs, TSelf rhs, fn_eq: FnEq) -> bool``, where ``TSelf`` is a subclass of
 * ObjectRef, and ``FnEq: (AnyView lhs, AnyView rhs) -> bool`` compares nested values.
 */
inline constexpr const char* kEq = "__ffi_eq__";
/*!
 * \brief Custom recursive three-way comparison hook.
 *
 * If registered, ``RecursiveCompare`` (Python ``<``, ``<=``, ``>``, ``>=``)
 * calls this.  Also used as fallback for ``RecursiveEq`` when ``kEq`` is not
 * registered.  The hook receives a callback ``fn_cmp`` to recursively
 * compare nested values.
 *
 * Signature: ``(TSelf lhs, TSelf rhs, fn_cmp: FnCmp) -> int32``, where ``TSelf`` is a subclass of
 * ObjectRef, and ``FnCmp: (TSelf lhs, TSelf rhs) -> int32`` returns < 0, == 0, or > 0.
 */
inline constexpr const char* kCompare = "__ffi_compare__";
/*!
 * \brief Custom Any-level hash for use as ``Map``/``Dict`` key.
 *
 * Unlike ``kHash`` (which operates on ``ObjectRef`` and is recursive), this
 * hook operates at the ``Any`` level — it is called by ``AnyHash`` whenever
 * an ``Any`` holding an object of this type is used as a container key.
 *
 * Can be either a raw C function pointer (fast path, no boxing overhead) or
 * a ``Function`` object.
 *
 * Raw pointer signature: ``int64_t (*)(const Any& src)``
 *
 * Function signature: ``(Any src) -> int64``
 */
inline constexpr const char* kAnyHash = "__any_hash__";
/*!
 * \brief Custom Any-level equality for use as ``Map``/``Dict`` key.
 *
 * Unlike ``kEq`` (which operates on ``ObjectRef``), this hook operates at the
 * ``Any`` level — called by ``AnyEqual`` for container key comparison.
 *
 * Can be either a raw C function pointer (fast path) or a ``Function`` object.
 *
 * Raw pointer signature: ``bool (*)(const Any& lhs, const Any& rhs)``
 *
 * Function signature: ``(Any lhs, Any rhs) -> bool``
 */
inline constexpr const char* kAnyEqual = "__any_equal__";
/*!
 * \brief Custom structural hash hook (used by ``StructuralHash``).
 *
 * Unlike ``kHash`` (which is for ``RecursiveHash`` / Python ``hash()``),
 * this is for the ``StructuralHash`` system that supports def/use region
 * semantics (``map_free_vars``).  The hook receives the current accumulated
 * hash and a callback to recursively hash sub-values.
 *
 * Signature: ``(TSelf self, int64 init_hash, Function hash_cb) -> int64``, where
 * ``TSelf`` is a subclass of ``ObjectRef``.
 *
 * ``hash_cb(AnyView val, int64 init_hash, bool def_region) -> int64``
 * recursively hashes a sub-value; ``def_region`` controls free-variable
 * mapping.
 */
inline constexpr const char* kSHash = "__s_hash__";
/*!
 * \brief Custom structural equality hook (used by ``StructuralEqual``).
 *
 * Unlike ``kEq`` (which is for ``RecursiveEq`` / Python ``==``), this is for
 * the ``StructuralEqual`` system that supports def/use region semantics and
 * alpha-equivalence of bound variables.  The hook receives a callback to
 * recursively compare sub-values.
 *
 * Signature: ``(TSelf lhs, TSelf rhs, Function eq_cb) -> bool``, where ``TSelf`` is a
 * subclass of ``ObjectRef``.
 *
 * ``eq_cb(AnyView lhs, AnyView rhs, bool def_region, AnyView field_name) -> bool``
 * recursively compares sub-values; ``def_region`` controls free-variable
 * mapping; ``field_name`` is used for mismatch path reporting.
 */
inline constexpr const char* kSEqual = "__s_equal__";
/*!
 * \brief Serialize object data to a JSON-compatible ``Map``.
 *
 * If registered, ``ToJSONGraph`` calls this instead of the default
 * field-by-field serialization.  Allows types with non-trivial internal
 * state (e.g. ``TInt`` storing a plain ``int64_t``) to define a compact
 * custom JSON representation.
 *
 * Signature: ``(TSelf self) -> Map<String, Any>``, where ``TSelf`` is a subclass of
 * ``ObjectRef``.
 */
inline constexpr const char* kDataToJson = "__data_to_json__";
/*!
 * \brief Deserialize object data from a JSON-compatible ``Map``.
 *
 * Counterpart to ``kDataToJson``.  If registered, ``FromJSONGraph`` calls
 * this to reconstruct the object from its serialized ``Map`` representation
 * instead of using field-by-field deserialization.
 *
 * Signature: ``(Map<String, Any> json_data) -> TSelf``, where ``TSelf`` is a subclass of
 * ``ObjectRef``.
 */
inline constexpr const char* kDataFromJson = "__data_from_json__";
}  // namespace type_attr

/*!
 * \brief C++17 constexpr helper: convert a ``const char*`` to ``TVMFFIByteArray``.
 * \param s A null-terminated string literal.
 * \return A ``TVMFFIByteArray`` whose ``data`` points to *s* and ``size`` equals ``strlen(s)``.
 */
inline constexpr TVMFFIByteArray AsByteArray(const char* s) {
  return {s, std::char_traits<char>::length(s)};
}

}  // namespace reflection
}  // namespace ffi
}  // namespace tvm
#endif  // TVM_FFI_REFLECTION_ACCESSOR_H_
