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
 * \file tvm/ffi/reflection/overload.h
 * \brief Registry of reflection metadata, supporting function overloading.
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

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace tvm {
namespace ffi {

namespace details {

struct OverloadBase {
 public:
  // Try Call function pointer type, return true if matched and called
  using FnPtr = bool (*)(OverloadBase*, const AnyView*, int32_t, Any*);

  explicit OverloadBase(int32_t num_args, std::optional<std::string> name)
      : num_args_(num_args),
        name_(name ? std::move(*name) : ""),
        name_ptr_(name ? &this->name_ : nullptr) {}

  virtual void Register(std::unique_ptr<OverloadBase> overload) = 0;
  virtual FnPtr GetTryCallPtr() = 0;
  virtual void GetMismatchMessage(std::ostringstream& os, const AnyView* args,
                                  int32_t num_args) = 0;

  virtual ~OverloadBase() = default;
  OverloadBase(const OverloadBase&) = delete;
  OverloadBase& operator=(const OverloadBase&) = delete;

 public:
  static constexpr int32_t kAllMatched = -1;

  // a fast cache for last matched arg index
  // on 64-bit platform, this is packed in the same 8 byte with num_args_
  int32_t last_mismatch_index_{kAllMatched};

  // some constant helper args
  const int32_t num_args_;
  const std::string name_;
  const std::string* const name_ptr_;
};

template <typename T>
struct CaptureTupleAux;

template <typename... Args>
struct CaptureTupleAux<std::tuple<Args...>> {
  using type = std::tuple<std::optional<std::decay_t<Args>>...>;
};

template <typename Callable>
struct TypedOverload : OverloadBase {
 public:
  static_assert(std::is_same_v<Callable, std::decay_t<Callable>>, "Callable must be value type");

  using FuncInfo = details::FunctionInfo<Callable>;
  using PackedArgs = typename FuncInfo::ArgType;
  using Ret = typename FuncInfo::RetType;
  using CaptureTuple = typename CaptureTupleAux<PackedArgs>::type;
  using OverloadBase::name_;
  using OverloadBase::name_ptr_;
  using typename OverloadBase::FnPtr;

  static constexpr auto kNumArgs = FuncInfo::num_args;
  static constexpr auto kSeq = std::make_index_sequence<kNumArgs>{};

  explicit TypedOverload(const Callable& f, std::optional<std::string> name = std::nullopt)
      : OverloadBase(kNumArgs, std::move(name)), f_(f) {}
  explicit TypedOverload(Callable&& f, std::optional<std::string> name = std::nullopt)
      : OverloadBase(kNumArgs, std::move(name)), f_(std::move(f)) {}

  bool TryCall(const AnyView* args, int32_t num_args, Any* rv) {
    if (num_args != kNumArgs) return false;
    CaptureTuple captures{};
    if (!TrySetAux(kSeq, captures, args)) return false;
    // now all captures are set
    if constexpr (std::is_same_v<Ret, void>) {
      CallAux(kSeq, captures);
      return true;
    } else {
      *rv = CallAux(kSeq, captures);
      return true;
    }
  }

  void Register(std::unique_ptr<OverloadBase> overload) override {
    TVM_FFI_ICHECK(false) << "This should never be called.";
  }

  FnPtr GetTryCallPtr() final {
    // lambda without a capture can be converted to function pointer
    return [](OverloadBase* base, const AnyView* args, int32_t num_args, Any* rv) -> bool {
      return static_cast<TypedOverload<Callable>*>(base)->TryCall(args, num_args, rv);
    };
  }

  void GetMismatchMessage(std::ostringstream& os, const AnyView* args, int32_t num_args) final {
    FGetFuncSignature f_sig = FuncInfo::Sig;
    if (num_args != kNumArgs) {
      os << "Mismatched number of arguments when calling: `" << name_ << " "
         << (f_sig == nullptr ? "" : (*f_sig)()) << "`. Expected " << kNumArgs << " arguments";
    } else {
      GetMismatchMessageAux<0>(os, args, num_args);
    }
  }

 private:
  template <std::size_t I>
  void GetMismatchMessageAux(std::ostringstream& os, const AnyView* args, int32_t num_args) {
    if constexpr (I < kNumArgs) {
      if (this->last_mismatch_index_ == static_cast<int32_t>(I)) {
        TVMFFIAny any_data = args[I].CopyToTVMFFIAny();
        FGetFuncSignature f_sig = FuncInfo::Sig;
        using Type = std::decay_t<std::tuple_element_t<I, PackedArgs>>;
        os << "Mismatched type on argument #" << I << " when calling: `" << name_ << " "
           << (f_sig == nullptr ? "" : (*f_sig)()) << "`. Expected `" << Type2Str<Type>::v()
           << "` but got `" << TypeTraits<Type>::GetMismatchTypeInfo(&any_data) << '`';
      } else {
        GetMismatchMessageAux<I + 1>(os, args, num_args);
      }
    }
    // end of recursion
  }

  template <std::size_t... I>
  Ret CallAux(std::index_sequence<I...>, CaptureTuple& tuple) {
    /// NOTE: this works for T, const T, const T&, T&& argument types
    return f_(static_cast<std::tuple_element_t<I, PackedArgs>>(std::move(*std::get<I>(tuple)))...);
  }

  template <std::size_t... I>
  bool TrySetAux(std::index_sequence<I...>, CaptureTuple& tuple, const AnyView* args) {
    return (TrySetOne<I>(tuple, args) && ...);
  }

  template <std::size_t I>
  bool TrySetOne(CaptureTuple& tuple, const AnyView* args) {
    using Type = std::decay_t<std::tuple_element_t<I, PackedArgs>>;
    auto& capture = std::get<I>(tuple);
    if constexpr (std::is_same_v<Type, AnyView>) {
      capture = args[I];
      return true;
    } else if constexpr (std::is_same_v<Type, Any>) {
      capture = Any(args[I]);
      return true;
    } else {
      capture = args[I].template try_cast<Type>();
      if (capture.has_value()) return true;
      // slow path: record the last mismatch index
      this->last_mismatch_index_ = static_cast<int32_t>(I);
      return false;
    }
  }

 protected:
  Callable f_;
};

template <typename Callable>
inline auto CreateNewOverload(Callable&& f, std::string name) {
  using Type = TypedOverload<std::decay_t<Callable>>;
  return std::make_unique<Type>(std::forward<Callable>(f), std::move(name));
}

template <typename Callable>
struct OverloadedFunction : TypedOverload<Callable> {
 public:
  using TypedBase = TypedOverload<Callable>;
  using OverloadBase::name_;
  using OverloadBase::name_ptr_;
  using TypedBase::GetTryCallPtr;
  using TypedBase::kNumArgs;
  using TypedBase::kSeq;
  using TypedBase::TypedBase;  // constructors
  using typename OverloadBase::FnPtr;
  using typename TypedBase::Ret;

  void Register(std::unique_ptr<OverloadBase> overload) final {
    const auto fptr = overload->GetTryCallPtr();
    overloads_.emplace_back(std::move(overload), fptr);
  }

  void operator()(const AnyView* args, int32_t num_args, Any* rv) {
    // fast path: only add a little overhead when no overloads
    if (overloads_.size() == 0) {
      return unpack_call<Ret>(kSeq, name_ptr_, f_, args, num_args, rv);
    }

    // this can be inlined by compiler, don't worry
    if (this->TryCall(args, num_args, rv)) return;

    // virtual calls cannot be inlined, so we fast check the num_args first
    // we also de-virtualize the fptr to reduce one more indirection
    for (const auto& [overload, fptr] : overloads_) {
      if (overload->num_args_ != num_args) continue;
      if (fptr(overload.get(), args, num_args, rv)) return;
    }

    this->HandleOverloadFailure(args, num_args);
  }

 private:
  void HandleOverloadFailure(const AnyView* args, int32_t num_args) {
    std::ostringstream oss;
    int32_t i = 0;
    oss << "Overload #" << i++ << ": ";
    this->GetMismatchMessage(oss, args, num_args);
    for (const auto& [overload, _] : overloads_) {
      oss << "\nOverload #" << i++ << ": ";
      overload->GetMismatchMessage(oss, args, num_args);
    }
    TVM_FFI_THROW(TypeError) << "No matching overload found when calling: `" << name_ << "` with "
                             << num_args << " arguments:\n"
                             << std::move(oss).str();
  }
  using TypedBase::f_;
  std::vector<std::pair<std::unique_ptr<OverloadBase>, FnPtr>> overloads_;
};

}  // namespace details

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
class OverloadObjectDef : private ObjectDef<Class> {
 public:
  using Super = ObjectDef<Class>;
  /*!
   * \brief Constructor
   * \tparam ExtraArgs The extra arguments.
   * \param extra_args The extra arguments.
   */
  template <typename... ExtraArgs>
  explicit OverloadObjectDef(ExtraArgs&&... extra_args)
      : Super(std::forward<ExtraArgs>(extra_args)...) {}

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
    /// NOTE: we don't allow properties to be overloaded
    Super::def_ro(name, field_ptr, std::forward<Extra>(extra)...);
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
    /// NOTE: we don't allow properties to be overloaded
    Super::def_rw(name, field_ptr, std::forward<Extra>(extra)...);
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
  using ReflectionDefBase::ApplyExtraInfoTrait;
  using ReflectionDefBase::WrapFunction;
  using Super::kInitMethodName;
  using Super::type_index_;
  using Super::type_key_;

  template <typename Func>
  static auto GetOverloadMethod(std::string name, Func&& func) {
    using WrapFn = decltype(WrapFunction(std::forward<Func>(func)));
    using OverloadFn = details::OverloadedFunction<std::decay_t<WrapFn>>;
    return ffi::Function::FromPackedInplace<OverloadFn>(WrapFunction(std::forward<Func>(func)),
                                                        std::move(name));
  }

  template <typename Func>
  static auto NewOverload(std::string name, Func&& func) {
    return details::CreateNewOverload(WrapFunction(std::forward<Func>(func)), std::move(name));
  }

  template <typename... ExtraArgs>
  void RegisterExtraInfo(ExtraArgs&&... extra_args) {
    TVMFFITypeMetadata info;
    info.total_size = sizeof(Class);
    info.structural_eq_hash_kind = Class::_type_s_eq_hash_kind;
    info.creator = nullptr;
    info.doc = TVMFFIByteArray{nullptr, 0};
    if constexpr (std::is_default_constructible_v<Class>) {
      info.creator = ReflectionDefBase::ObjectCreatorDefault<Class>;
    } else if constexpr (std::is_constructible_v<Class, UnsafeInit>) {
      info.creator = ReflectionDefBase::ObjectCreatorUnsafeInit<Class>;
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
    info.getter = ReflectionDefBase::FieldGetter<T>;
    info.setter = ReflectionDefBase::FieldSetter<T>;
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

  std::unordered_map<std::string, details::OverloadBase*> registered_fields_;
};

}  // namespace reflection
}  // namespace ffi
}  // namespace tvm
#endif  // TVM_FFI_EXTRA_OVERLOAD_H
