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
 * \file tvm/ffi/function_details.h
 * \brief Implements the funciton signature reflection
 */
#ifndef TVM_FFI_FUNCTION_DETAILS_H_
#define TVM_FFI_FUNCTION_DETAILS_H_

#include <tvm/ffi/any.h>
#include <tvm/ffi/base_details.h>
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/error.h>

#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

namespace tvm {
namespace ffi {
namespace details {

template <typename ArgType>
struct Arg2Str {
  template <size_t i>
  TVM_FFI_INLINE static void Apply(std::ostream& os) {
    using Arg = std::tuple_element_t<i, ArgType>;
    if constexpr (i != 0) {
      os << ", ";
    }
    os << i << ": " << Type2Str<Arg>::v();
  }
  template <size_t... I>
  TVM_FFI_INLINE static void Run(std::ostream& os, std::index_sequence<I...>) {
    using TExpander = int[];
    (void)TExpander{0, (Apply<I>(os), 0)...};
  }
};

template <typename T>
static constexpr bool ArgSupported =
    (std::is_same_v<std::remove_const_t<std::remove_reference_t<T>>, Any> ||
     std::is_same_v<std::remove_const_t<std::remove_reference_t<T>>, AnyView> ||
     TypeTraitsNoCR<T>::convert_enabled);

// NOTE: return type can only support non-reference managed returns
template <typename T>
static constexpr bool RetSupported =
    (std::is_same_v<T, Any> || std::is_void_v<T> || TypeTraits<T>::convert_enabled);

template <typename R, typename... Args>
struct FuncFunctorImpl {
  using FType = R(Args...);
  using ArgType = std::tuple<Args...>;
  using RetType = R;
  /*! \brief total number of arguments*/
  static constexpr size_t num_args = sizeof...(Args);
  // MSVC is not that friendly to in-template nested bool evaluation
#ifndef _MSC_VER
  /*! \brief Whether this function can be converted to ffi::Function via FromTyped */
  static constexpr bool unpacked_supported = (ArgSupported<Args> && ...) && (RetSupported<R>);
#endif
  TVM_FFI_INLINE static std::string Sig() {
    using IdxSeq = std::make_index_sequence<sizeof...(Args)>;
    std::ostringstream ss;
    ss << "(";
    Arg2Str<std::tuple<Args...>>::Run(ss, IdxSeq{});
    ss << ") -> " << Type2Str<R>::v();
    return ss.str();
  }
  TVM_FFI_INLINE static std::string TypeSchema() {
    std::ostringstream oss;
    oss << R"({"type":")" << StaticTypeKey::kTVMFFIFunction << R"(","args":[)";
    oss << details::TypeSchema<R>::v();
    ((oss << "," << details::TypeSchema<Args>::v()), ...);
    oss << "]}";
    return oss.str();
  }
};

template <typename T>
struct FunctionInfoHelper;

template <typename T, typename R, typename... Args>
struct FunctionInfoHelper<R (T::*)(Args...)> : FuncFunctorImpl<R, Args...> {};
template <typename T, typename R, typename... Args>
struct FunctionInfoHelper<R (T::*)(Args...) const> : FuncFunctorImpl<R, Args...> {};

/*!
 * \brief Template class to get function signature of a function or functor.
 * \tparam T The function/functor type.
 * \note We need a decltype redirection because this helps lambda types.
 */
template <typename T, typename = void>
struct FunctionInfo : FunctionInfoHelper<decltype(&T::operator())> {};
template <typename R, typename... Args>
struct FunctionInfo<R(Args...), void> : FuncFunctorImpl<R, Args...> {};
template <typename R, typename... Args>
struct FunctionInfo<R (*)(Args...), void> : FuncFunctorImpl<R, Args...> {};
template <typename R, typename... Args>
struct FunctionInfo<R (&)(Args...), void> : FuncFunctorImpl<R, Args...> {};
// Support pointer-to-member functions used in reflection (e.g. &Class::method)
template <typename Class, typename R, typename... Args>
struct FunctionInfo<R (Class::*)(Args...), std::enable_if_t<std::is_base_of_v<Object, Class>>>
    : FuncFunctorImpl<R, Class*, Args...> {};
template <typename Class, typename R, typename... Args>
struct FunctionInfo<R (Class::*)(Args...) const, std::enable_if_t<std::is_base_of_v<Object, Class>>>
    : FuncFunctorImpl<R, const Class*, Args...> {};

template <typename Class, typename R, typename... Args>
struct FunctionInfo<R (Class::*)(Args...), std::enable_if_t<std::is_base_of_v<ObjectRef, Class>>>
    : FuncFunctorImpl<R, Class, Args...> {};
template <typename Class, typename R, typename... Args>
struct FunctionInfo<R (Class::*)(Args...) const,
                    std::enable_if_t<std::is_base_of_v<ObjectRef, Class>>>
    : FuncFunctorImpl<R, const Class, Args...> {};

/*! \brief Using static function to output typed function signature */
using FGetFuncSignature = std::string (*)();

/*!
 * \brief Auxilary argument value with context for error reporting
 */
template <typename Type, bool NeedStorage>
class ArgValueWithContext;

template <typename Type>
class ArgValueWithContext<Type, /*NeedStorage=*/false> {
 public:
  // We only support pure value type to convert to, to avoid any dangling reference issues.
  // C++ will automatically eliminate unnecessary constructor calls by NRVO,
  // so return by value won't cause performance issue.
  static_assert(std::is_same_v<Type, std::remove_const_t<std::remove_reference_t<Type>>>);

  /*!
   * \brief move constructor from another return value.
   * \param args The argument list
   * \param arg_index In a function call, this argument is at index arg_index (0-indexed).
   * \param optional_name Name of the function being called. Can be nullptr if the function is not.
   * \param f_sig Pointer to static function outputting signature of the function being called.
   * named.
   */
  TVM_FFI_INLINE ArgValueWithContext(const AnyView* args, int32_t arg_index,
                                     const std::string* optional_name, FGetFuncSignature f_sig)
      : args_(args), arg_index_(arg_index), optional_name_(optional_name), f_sig_(f_sig) {}

  TVM_FFI_INLINE operator Type() {  // NOLINT(google-explicit-constructor)
    if constexpr (std::is_same_v<Type, AnyView>) {
      return args_[arg_index_];
    } else if constexpr (std::is_same_v<Type, Any>) {
      return Any(args_[arg_index_]);
    } else {
      auto opt = args_[arg_index_].template try_cast<Type>();
      if (!opt.has_value()) {
        TVMFFIAny any_data = args_[arg_index_].CopyToTVMFFIAny();
        TVM_FFI_THROW(TypeError) << "Mismatched type on argument #" << arg_index_
                                 << " when calling: `"
                                 << (optional_name_ == nullptr ? "" : *optional_name_)
                                 << (f_sig_ == nullptr ? "" : (*f_sig_)()) << "`. Expected `"
                                 << Type2Str<Type>::v() << "` but got `"
                                 << TypeTraits<Type>::GetMismatchTypeInfo(&any_data) << '`';
      }
      return *std::move(opt);
    }
  }

 private:
  const AnyView* args_;
  int32_t arg_index_;
  const std::string* optional_name_;
  FGetFuncSignature f_sig_;
};

template <typename Type>
class ArgValueWithContext<Type, /*NeedStorage=*/true> {
 public:
  // We only support pure value type to convert to, to avoid any dangling reference issues.
  // C++ will automatically eliminate unnecessary constructor calls by NRVO,
  // so return by value won't cause performance issue.
  static_assert(std::is_same_v<Type, std::remove_const_t<std::remove_reference_t<Type>>>);
  /*!
   * \brief move constructor from another return value.
   * \param args The argument list
   * \param arg_index In a function call, this argument is at index arg_index (0-indexed).
   * \param optional_name Name of the function being called. Can be nullptr if the function is
   * not.
   * \param f_sig Pointer to static function outputting signature of the function being called.
   * named.
   */
  TVM_FFI_INLINE ArgValueWithContext(const AnyView* args, int32_t arg_index,
                                     const std::string* optional_name, FGetFuncSignature f_sig)
      : args_(args), arg_index_(arg_index), optional_name_(optional_name), f_sig_(f_sig) {}

  // to avoid dangling reference, we only allow one-time conversion to Type&
  TVM_FFI_INLINE operator Type&() {  // NOLINT(google-explicit-constructor)
    TVM_FFI_ICHECK(!temp_storage_.has_value())
        << "ArgValueWithContext<Type> can only be converted to Type& once.";
    if constexpr (std::is_same_v<Type, AnyView>) {
      temp_storage_.emplace(args_[arg_index_]);
    } else if constexpr (std::is_same_v<Type, Any>) {
      temp_storage_.emplace(Any(args_[arg_index_]));
    } else {
      temp_storage_ = args_[arg_index_].template try_cast<Type>();
      if (!temp_storage_.has_value()) {
        TVMFFIAny any_data = args_[arg_index_].CopyToTVMFFIAny();
        TVM_FFI_THROW(TypeError) << "Mismatched type on argument #" << arg_index_
                                 << " when calling: `"
                                 << (optional_name_ == nullptr ? "" : *optional_name_)
                                 << (f_sig_ == nullptr ? "" : (*f_sig_)()) << "`. Expected `"
                                 << Type2Str<Type>::v() << "` but got `"
                                 << TypeTraits<Type>::GetMismatchTypeInfo(&any_data) << '`';
      }
    }
    return temp_storage_.value();
  }

 private:
  const AnyView* args_;
  int32_t arg_index_;
  const std::string* optional_name_;
  FGetFuncSignature f_sig_;
  std::optional<Type> temp_storage_;
};

/**
 * \brief Auxilary argument value with context for error reporting
 * For `Type &` non-const lvalue reference types, we need to
 * store a temporary value to avoid dangling reference issues.
 * For `const Type`, `Type &&`, returning value can be cast to
 * the desired type directly.
 */
template <typename Type>
using ArgValue = ArgValueWithContext<std::remove_const_t<std::remove_reference_t<Type>>,
                                     (!std::is_const_v<std::remove_reference_t<Type>> &&
                                      std::is_lvalue_reference_v<Type>)>;

template <typename R, std::size_t... Is, typename F>
TVM_FFI_INLINE void unpack_call(std::index_sequence<Is...>, const std::string* optional_name,
                                const F& f, [[maybe_unused]] const AnyView* args,
                                [[maybe_unused]] int32_t num_args, [[maybe_unused]] Any* rv) {
  using FuncInfo = FunctionInfo<F>;
  using PackedArgs = typename FuncInfo::ArgType;
  FGetFuncSignature f_sig = FuncInfo::Sig;

  // somehow MSVC does not support the static constexpr member in this case, function is fine
#ifndef _MSC_VER
  static_assert(FuncInfo::unpacked_supported, "The function signature do not support unpacked");
#endif
  constexpr size_t nargs = sizeof...(Is);
  if (nargs != num_args) {
    TVM_FFI_THROW(TypeError) << "Mismatched number of arguments when calling: `"
                             << (optional_name == nullptr ? "" : *optional_name)
                             << (f_sig == nullptr ? "" : (*f_sig)()) << "`. Expected " << nargs
                             << " but got " << num_args << " arguments";
  }
  // use index sequence to do recursive-less unpacking
  if constexpr (std::is_same_v<R, void>) {
    f(ArgValue<std::tuple_element_t<Is, PackedArgs>>{args, Is, optional_name, f_sig}...);
  } else {
    *rv = R(f(ArgValue<std::tuple_element_t<Is, PackedArgs>>{args, Is, optional_name, f_sig}...));
  }
}

/*!
 * \brief Move the safe call raised error to the caller
 * \return The error
 */
TVM_FFI_INLINE static Error MoveFromSafeCallRaised() {
  TVMFFIObjectHandle handle;
  TVMFFIErrorMoveFromRaised(&handle);
  // handle is owned by caller
  return details::ObjectUnsafe::ObjectRefFromObjectPtr<Error>(
      details::ObjectUnsafe::ObjectPtrFromOwned<Object>(static_cast<TVMFFIObject*>(handle)));
}

/*!
 * \brief Set the safe call raised error
 * \param error The error
 */
TVM_FFI_INLINE static void SetSafeCallRaised(const Error& error) {
  TVMFFIErrorSetRaised(details::ObjectUnsafe::TVMFFIObjectPtrFromObjectRef(error));
}

template <typename T>
struct TypeSchemaImpl {
  static std::string v() {
    using U = std::remove_const_t<std::remove_reference_t<T>>;
    return TypeTraits<U>::TypeSchema();
  }
};

template <>
struct TypeSchemaImpl<void> {
  static std::string v() {
    return R"({"type":")" + std::string(StaticTypeKey::kTVMFFINone) + R"("})";
  }
};

template <>
struct TypeSchemaImpl<Any> {
  static std::string v() {
    return R"({"type":")" + std::string(StaticTypeKey::kTVMFFIAny) + R"("})";
  }
};

template <>
struct TypeSchemaImpl<AnyView> {
  static std::string v() {
    return R"({"type":")" + std::string(StaticTypeKey::kTVMFFIAny) + R"("})";
  }
};

}  // namespace details
}  // namespace ffi
}  // namespace tvm
#endif  // TVM_FFI_FUNCTION_DETAILS_H_
