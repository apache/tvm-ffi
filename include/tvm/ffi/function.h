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
 * \file tvm/ffi/function.h
 * \brief A managed function in the TVM FFI.
 */
#ifndef TVM_FFI_FUNCTION_H_
#define TVM_FFI_FUNCTION_H_

#include <tvm/ffi/any.h>
#include <tvm/ffi/base_details.h>
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/function_details.h>

#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace tvm {
namespace ffi {

/**
 * Helper macro to construct a safe call
 *
 * \brief Marks the beginning of the safe call that catches exception explicitly
 * \sa TVM_FFI_SAFE_CALL_END
 *
 * \code
 * int TVMFFICStyleFunction() {
 *   TVM_FFI_SAFE_CALL_BEGIN();
 *   // c++ code region here
 *   TVM_FFI_SAFE_CALL_END();
 * }
 * \endcode
 */
#define TVM_FFI_SAFE_CALL_BEGIN() \
  try {                           \
  (void)0

/*!
 * \brief Marks the end of safe call.
 */
#define TVM_FFI_SAFE_CALL_END()                                                                \
  return 0;                                                                                    \
  }                                                                                            \
  catch (const ::tvm::ffi::Error& err) {                                                       \
    ::tvm::ffi::details::SetSafeCallRaised(err);                                               \
    return -1;                                                                                 \
  }                                                                                            \
  catch (const ::tvm::ffi::EnvErrorAlreadySet&) {                                              \
    return -2;                                                                                 \
  }                                                                                            \
  catch (const std::exception& ex) {                                                           \
    ::tvm::ffi::details::SetSafeCallRaised(::tvm::ffi::Error("InternalError", ex.what(), "")); \
    return -1;                                                                                 \
  }                                                                                            \
  TVM_FFI_UNREACHABLE()

/*!
 * \brief Macro to check a call to TVMFFISafeCallType and raise exception if error happens.
 * \param func The function to check.
 *
 * \code
 * // calls TVMFFIFunctionCall and raises exception if error happens
 * TVM_FFI_CHECK_SAFE_CALL(TVMFFITypeKeyToIndex(&type_key_arr, &type_index));
 * \endcode
 */
#define TVM_FFI_CHECK_SAFE_CALL(func)                      \
  {                                                        \
    int ret_code = (func);                                 \
    if (ret_code != 0) {                                   \
      if (ret_code == -2) {                                \
        throw ::tvm::ffi::EnvErrorAlreadySet();            \
      }                                                    \
      throw ::tvm::ffi::details::MoveFromSafeCallRaised(); \
    }                                                      \
  }

/*!
 * \brief Object container class that backs ffi::Function
 * \note Do not use this class directly, use ffi::Function
 */
class FunctionObj : public Object, public TVMFFIFunctionCell {
 public:
  /*! \brief Typedef for C++ style calling signature that comes with exception propagation */
  using FCall = void (*)(const FunctionObj*, const AnyView*, int32_t, Any*);
  using TVMFFIFunctionCell::cpp_call;
  using TVMFFIFunctionCell::safe_call;
  /*!
   * \brief Call the function in packed format.
   * \param args The arguments
   * \param num_args The number of arguments
   * \param result The return value.
   */
  TVM_FFI_INLINE void CallPacked(const AnyView* args, int32_t num_args, Any* result) const {
    // if cpp_call is set, use it to call the function, otherwise, redirect to safe_call
    // use conditional expression here so the select is branchless
    FCall call_ptr =
        this->cpp_call ? reinterpret_cast<FCall>(this->cpp_call) : CppCallDedirectToSafeCall;
    (*call_ptr)(this, args, num_args, result);
  }
  /// \cond Doxygen_Suppress
  static constexpr const uint32_t _type_index = TypeIndex::kTVMFFIFunction;
  TVM_FFI_DECLARE_OBJECT_INFO_STATIC(StaticTypeKey::kTVMFFIFunction, FunctionObj, Object);
  /// \endcond

 protected:
  /*! \brief Make default constructor protected. */
  FunctionObj() {}
  friend class Function;

 private:
  static void CppCallDedirectToSafeCall(const FunctionObj* func, const AnyView* args,
                                        int32_t num_args, Any* rv) {
    FunctionObj* self = static_cast<FunctionObj*>(const_cast<FunctionObj*>(func));
    TVM_FFI_CHECK_SAFE_CALL(self->safe_call(self, reinterpret_cast<const TVMFFIAny*>(args),
                                            num_args, reinterpret_cast<TVMFFIAny*>(rv)));
  }
};

namespace details {
/*!
 * \brief Derived object class for constructing FunctionObj backed by a TCallable
 *
 * This is a helper class that implements the function call interface.
 */
template <typename TCallable>
class FunctionObjImpl : public FunctionObj {
 public:
  using TStorage = std::remove_cv_t<std::remove_reference_t<TCallable>>;
  /*! \brief The type of derived object class */
  using TSelf = FunctionObjImpl<TCallable>;
  /*!
   * \brief Derived object class for constructing ffi::FunctionObj.
   * \param callable The type-erased callable object.
   */
  explicit FunctionObjImpl(TCallable callable) : callable_(std::move(callable)) {
    this->safe_call = SafeCall;
    this->cpp_call = reinterpret_cast<void*>(CppCall);
  }

 private:
  // implementation of call
  static void CppCall(const FunctionObj* func, const AnyView* args, int32_t num_args, Any* result) {
    (static_cast<const TSelf*>(func))->callable_(args, num_args, result);
  }
  /// \cond Doxygen_Suppress
  // Implementing safe call style
  static int SafeCall(void* func, const TVMFFIAny* args, int32_t num_args, TVMFFIAny* result) {
    TVM_FFI_SAFE_CALL_BEGIN();
    TVM_FFI_ICHECK_LT(result->type_index, TypeIndex::kTVMFFIStaticObjectBegin);
    FunctionObj* self = static_cast<FunctionObj*>(func);
    reinterpret_cast<FCall>(self->cpp_call)(self, reinterpret_cast<const AnyView*>(args), num_args,
                                            reinterpret_cast<Any*>(result));
    TVM_FFI_SAFE_CALL_END();
  }
  /// \endcond
  /*! \brief Type-erased filed for storing callable object*/
  mutable TStorage callable_;
};

/*!
 * \brief FunctionObj specialization for raw C style callback where handle and deleter are null.
 */
class ExternCFunctionObjNullHandleImpl : public FunctionObj {
 public:
  explicit ExternCFunctionObjNullHandleImpl(TVMFFISafeCallType safe_call) {
    this->safe_call = safe_call;
    this->cpp_call = nullptr;
  }
};

/*!
 * \brief FunctionObj specialization that leverages C-style callback definitions.
 */
class ExternCFunctionObjImpl : public FunctionObj {
 public:
  ExternCFunctionObjImpl(void* self, TVMFFISafeCallType safe_call, void (*deleter)(void* self))
      : self_(self), safe_call_(safe_call), deleter_(deleter) {
    this->safe_call = SafeCall;
    this->cpp_call = nullptr;
  }

  ~ExternCFunctionObjImpl() {
    if (deleter_) deleter_(self_);
  }

 private:
  static int32_t SafeCall(void* func, const TVMFFIAny* args, int32_t num_args, TVMFFIAny* rv) {
    ExternCFunctionObjImpl* self = reinterpret_cast<ExternCFunctionObjImpl*>(func);
    return self->safe_call_(self->self_, args, num_args, rv);
  }

  void* self_;
  TVMFFISafeCallType safe_call_;
  void (*deleter_)(void* self);
};

// Helper class to set packed arguments
class PackedArgsSetter {
 public:
  explicit PackedArgsSetter(AnyView* args) : args_(args) {}

  // NOTE: setter needs to be very carefully designed
  // such that we do not have temp variable conversion(eg. convert from lvalue to rvalue)
  // that is why we need T&& and std::forward here
  template <typename T>
  TVM_FFI_INLINE void operator()(size_t i, T&& value) const {
    args_[i].operator=(std::forward<T>(value));
  }

 private:
  AnyView* args_;
};
}  // namespace details

/*!
 * \brief Represents arguments packed in AnyView array
 * \note This class represent packed arguments to ffi::Function
 */
class PackedArgs {
 public:
  /*!
   * \brief Constructor
   * \param data The arguments
   * \param size The number of arguments
   */
  PackedArgs(const AnyView* data, int32_t size) : data_(data), size_(size) {}

  /*! \return size of the arguments */
  int size() const { return size_; }

  /*! \return The arguments */
  const AnyView* data() const { return data_; }

  /*!
   * \brief Slice the arguments
   * \param begin The begin index
   * \param end The end index
   * \return The sliced arguments
   */
  PackedArgs Slice(int begin, int end = -1) const {
    if (end == -1) {
      end = size_;
    }
    return PackedArgs(data_ + begin, end - begin);
  }

  /*!
   * \brief Get i-th argument
   * \param i the index.
   * \return the ith argument.
   */
  AnyView operator[](int i) const { return data_[i]; }

  /*!
   * \brief Fill the arguments into the AnyView array
   * \param data The AnyView array to store the packed arguments
   * \param args The arguments to be packed
   * \note Caller must ensure all args are alive during lifetime of data.
   *       A common pitfall is to pass in local variables that are immediately
   *       destroyed after calling Fill.
   */
  template <typename... Args>
  TVM_FFI_INLINE static void Fill(AnyView* data, Args&&... args) {
    details::for_each(details::PackedArgsSetter(data), std::forward<Args>(args)...);
  }

 private:
  /*! \brief The arguments */
  const AnyView* data_;
  /*! \brief The number of arguments */
  int32_t size_;
};

/*!
 * \brief ffi::Function  is a type-erased function.
 *  The arguments are passed by "packed format" via AnyView
 */
class Function : public ObjectRef {
 public:
  /*! \brief Constructor from null */
  Function(std::nullptr_t) : ObjectRef(nullptr) {}  // NOLINT(*)
  /*!
   * \brief Constructing a packed function from a callable type
   *        whose signature is consistent with `ffi::Function`
   * \param packed_call The packed function signature
   * \note legacy purpose, should change to Function::FromPacked for mostfuture use.
   */
  template <typename TCallable>
  explicit Function(TCallable packed_call) {
    *this = FromPacked(packed_call);
  }
  /*!
   * \brief Constructing a packed function from a callable type
   *        whose signature is consistent with `ffi::Function`
   * \param packed_call The packed function signature
   */
  template <typename TCallable>  // // NOLINTNEXTLINE(performance-unnecessary-value-param)
  static Function FromPacked(TCallable packed_call) {
    static_assert(
        std::is_convertible_v<TCallable, std::function<void(const AnyView*, int32_t, Any*)>> ||
            std::is_convertible_v<TCallable, std::function<void(PackedArgs args, Any*)>>,
        "tvm::ffi::Function::FromPacked requires input function signature to match packed func "
        "format");
    if constexpr (std::is_convertible_v<TCallable, std::function<void(PackedArgs args, Any*)>>) {
      return FromPackedInternal(
          [packed_call](const AnyView* args, int32_t num_args, Any* rv) mutable -> void {
            PackedArgs args_pack(args, num_args);
            packed_call(args_pack, rv);
          });
    } else {
      return FromPackedInternal(packed_call);
    }
  }

  /*!
   * \brief Create ffi::Function from a C style callbacks.
   *
   * self and deleter can be nullptr if the function do not need closure support
   * and corresponds to a raw function pointer.
   *
   * \param self Resource handle to the function
   * \param safe_call The safe_call definition in C.
   * \param deleter The deleter to release the resource of self.
   * \return The created function.
   */
  static Function FromExternC(void* self, TVMFFISafeCallType safe_call,
                              void (*deleter)(void* self)) {
    // the other function coems from a different library
    Function func;
    if (self == nullptr && deleter == nullptr) {
      func.data_ = make_object<details::ExternCFunctionObjNullHandleImpl>(safe_call);
    } else {
      func.data_ = make_object<details::ExternCFunctionObjImpl>(self, safe_call, deleter);
    }
    return func;
  }
  /*!
   * \brief Get global function by name
   * \param name The function name
   * \return The global function.
   * \note This function will return std::nullopt if the function is not found.
   */
  static std::optional<Function> GetGlobal(std::string_view name) {
    TVMFFIObjectHandle handle;
    TVMFFIByteArray name_arr{name.data(), name.size()};
    TVM_FFI_CHECK_SAFE_CALL(TVMFFIFunctionGetGlobal(&name_arr, &handle));
    if (handle != nullptr) {
      return Function(
          details::ObjectUnsafe::ObjectPtrFromOwned<FunctionObj>(static_cast<Object*>(handle)));
    } else {
      return std::nullopt;
    }
  }

  /*!
   * \brief Get global function by name
   * \param name The name of the function
   * \return The global function
   * \note This function will return std::nullopt if the function is not found.
   */
  static std::optional<Function> GetGlobal(const std::string& name) {
    return GetGlobal(std::string_view(name.data(), name.length()));
  }

  /*!
   * \brief Get global function by name
   * \param name The name of the function
   * \return The global function
   * \note This function will return std::nullopt if the function is not found.
   */
  static std::optional<Function> GetGlobal(const String& name) {
    return GetGlobal(std::string_view(name.data(), name.length()));
  }

  /*!
   * \brief Get global function by name
   * \param name The name of the function
   * \return The global function
   * \note This function will return std::nullopt if the function is not found.
   */
  static std::optional<Function> GetGlobal(const char* name) {
    return GetGlobal(std::string_view(name));
  }
  /*!
   * \brief Get global function by name and throw an error if it is not found.
   * \param name The name of the function
   * \return The global function
   * \note This function will throw an error if the function is not found.
   */
  static Function GetGlobalRequired(std::string_view name) {
    std::optional<Function> res = GetGlobal(name);
    if (!res.has_value()) {
      TVM_FFI_THROW(ValueError) << "Function " << name << " not found";
    }
    return *res;
  }

  /*!
   * \brief Get global function by name
   * \param name The name of the function
   * \return The global function
   * \note This function will throw an error if the function is not found.
   */
  static Function GetGlobalRequired(const std::string& name) {
    return GetGlobalRequired(std::string_view(name.data(), name.length()));
  }

  /*!
   * \brief Get global function by name
   * \param name The name of the function
   * \return The global function
   * \note This function will throw an error if the function is not found.
   */
  static Function GetGlobalRequired(const String& name) {
    return GetGlobalRequired(std::string_view(name.data(), name.length()));
  }

  /*!
   * \brief Get global function by name
   * \param name The name of the function
   * \return The global function
   * \note This function will throw an error if the function is not found.
   */
  static Function GetGlobalRequired(const char* name) {
    return GetGlobalRequired(std::string_view(name));
  }
  /*!
   * \brief Set global function by name
   * \param name The name of the function
   * \param func The function
   * \param override Whether to override when there is duplication.
   */
  static void SetGlobal(std::string_view name,
                        Function func,  // NOLINT(performance-unnecessary-value-param)
                        bool override = false) {
    TVMFFIByteArray name_arr{name.data(), name.size()};
    TVM_FFI_CHECK_SAFE_CALL(
        TVMFFIFunctionSetGlobal(&name_arr, details::ObjectUnsafe::GetHeader(func.get()), override));
  }
  /*!
   * \brief List all global names
   * \return A vector of all global names
   * \note This function do not depend on Array so core do not have container dep.
   */
  static std::vector<String> ListGlobalNames() {
    Function fname_functor =
        GetGlobalRequired("ffi.FunctionListGlobalNamesFunctor")().cast<Function>();
    std::vector<String> names;
    int len = fname_functor(-1).cast<int>();
    names.reserve(len);
    for (int i = 0; i < len; ++i) {
      names.push_back(fname_functor(i).cast<String>());
    }
    return names;
  }
  /**
   * \brief Remove a global function by name
   * \param name The name of the function
   */
  static void RemoveGlobal(const String& name) {
    static Function fremove = GetGlobalRequired("ffi.FunctionRemoveGlobal");
    fremove(name);
  }
  /*!
   * \brief Constructing a packed function from a normal function.
   *
   * \param callable the internal container of packed function.
   */
  template <typename TCallable>
  static Function FromTyped(TCallable callable) {
    using FuncInfo = details::FunctionInfo<TCallable>;
    auto call_packed = [callable = std::move(callable)](const AnyView* args, int32_t num_args,
                                                        Any* rv) mutable -> void {
      details::unpack_call<typename FuncInfo::RetType>(
          std::make_index_sequence<FuncInfo::num_args>{}, nullptr, callable, args, num_args, rv);
    };
    return FromPackedInternal(std::move(call_packed));
  }
  /*!
   * \brief Constructing a packed function from a normal function.
   *
   * \param callable the internal container of packed function.
   * \param name optional name attacked to the function.
   */
  template <typename TCallable>
  static Function FromTyped(TCallable callable, std::string name) {
    using FuncInfo = details::FunctionInfo<TCallable>;
    auto call_packed = [callable = std::move(callable), name = std::move(name)](
                           const AnyView* args, int32_t num_args, Any* rv) mutable -> void {
      details::unpack_call<typename FuncInfo::RetType>(
          std::make_index_sequence<FuncInfo::num_args>{}, &name, callable, args, num_args, rv);
    };
    return FromPackedInternal(std::move(call_packed));
  }

  /*!
   * \brief Directly invoke an extern "C" function that follows the TVM FFI SafeCall convention.
   *
   * This function can be useful to turn an existing exported symbol into a typed function.
   *
   * \code
   *
   * // An extern "C" function, matching TVMFFISafeCallType
   * extern "C" int __tvm_ffi_add(
   *   void* handle, const TVMFFIAny* args, int32_t num_args, TVMFFIAny*result
   * );
   *
   * // redirect an existing symbol into a typed function
   * inline int add(int a, int b) {
   *   return tvm::ffi::Function::InvokeExternC(nullptr, __tvm_ffi_add, a, b).cast<int>();
   * }
   *
   * \endcode
   *
   * \tparam Args The types of the arguments to the extern function.
   * \param handle The handle argument, for exported symbols this is usually nullptr.
   * \param safe_call The function pointer to the extern "C" function.
   * \param args The arguments to pass to the function.
   * \return The return value, wrapped in a tvm::ffi::Any.
   */
  template <typename... Args>
  TVM_FFI_INLINE static Any InvokeExternC(void* handle, TVMFFISafeCallType safe_call,
                                          Args&&... args) {
    const int kNumArgs = sizeof...(Args);
    const int kArraySize = kNumArgs > 0 ? kNumArgs : 1;
    AnyView args_pack[kArraySize];
    PackedArgs::Fill(args_pack, std::forward<Args>(args)...);
    Any result;
    TVM_FFI_CHECK_SAFE_CALL(safe_call(handle, reinterpret_cast<const TVMFFIAny*>(args_pack),
                                      kNumArgs, reinterpret_cast<TVMFFIAny*>(&result)));
    return result;
  }
  /*!
   * \brief Call function by directly passing in unpacked arguments.
   *
   * \param args Arguments to be passed.
   * \tparam Args arguments to be passed.
   *
   * \code
   *   // Example code on how to call packed function
   *   void CallFFIFunction(tvm::ffi::Function f) {
   *     // call like normal functions by pass in arguments
   *     // return value is automatically converted back
   *     int rvalue = f(1, 2.0);
   *   }
   * \endcode
   */
  template <typename... Args>
  TVM_FFI_INLINE Any operator()(Args&&... args) const {
    const int kNumArgs = sizeof...(Args);
    const int kArraySize = kNumArgs > 0 ? kNumArgs : 1;
    AnyView args_pack[kArraySize];
    PackedArgs::Fill(args_pack, std::forward<Args>(args)...);
    Any result;
    static_cast<FunctionObj*>(data_.get())->CallPacked(args_pack, kNumArgs, &result);
    return result;
  }
  /*!
   * \brief Call the function in packed format.
   * \param args The arguments
   * \param num_args The number of arguments
   * \param result The return value.
   */
  TVM_FFI_INLINE void CallPacked(const AnyView* args, int32_t num_args, Any* result) const {
    static_cast<FunctionObj*>(data_.get())->CallPacked(args, num_args, result);
  }
  /*!
   * \brief Call the function in packed format.
   * \param args The arguments
   * \param result The return value.
   */
  TVM_FFI_INLINE void CallPacked(PackedArgs args, Any* result) const {
    static_cast<FunctionObj*>(data_.get())->CallPacked(args.data(), args.size(), result);
  }

  /*! \return Whether the packed function is nullptr */
  TVM_FFI_INLINE bool operator==(std::nullptr_t) const { return data_ == nullptr; }
  /*! \return Whether the packed function is not nullptr */
  TVM_FFI_INLINE bool operator!=(std::nullptr_t) const { return data_ != nullptr; }

  /// \cond Doxygen_Suppress
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Function, ObjectRef, FunctionObj);
  /// \endcond

  class Registry;

 private:
  /*!
   * \brief Constructing a packed function from a callable type
   *        whose signature is consistent with `ffi::Function`
   * \param packed_call The packed function signature
   */
  template <typename TCallable>
  static Function FromPackedInternal(TCallable packed_call) {
    using ObjType = typename details::FunctionObjImpl<TCallable>;
    Function func;
    func.data_ = make_object<ObjType>(
        std::forward<TCallable>(packed_call));  // NOLINT(bugprone-chained-comparison)
    return func;
  }
};

/*!
 * \brief Please refer to \ref TypedFunctionAnchor "TypedFunction<R(Args..)>"
 */
template <typename FType>
class TypedFunction;

/*!
 * \anchor TypedFunctionAnchor
 * \brief A ffi::Function wrapper to provide typed function signature.
 * It is backed by a ffi::Function internally.
 *
 * TypedFunction enables compile time type checking.
 * TypedFunction works with the runtime system:
 * - It can be passed as an argument of ffi::Function.
 * - It can be assigned to ffi::Any.
 * - It can be directly converted to a type-erased ffi::Function.
 *
 * Developers should prefer TypedFunction over ffi::Function in C++ code
 * as it enables compile time checking.
 * We can construct a TypedFunction from a lambda function
 * with the same signature.
 *
 * \code
 *  // user defined lambda function.
 *  auto addone = [](int x)->int {
 *    return x + 1;
 *  };
 *  // We can directly convert
 *  // lambda function to TypedFunction
 *  TypedFunction<int(int)> ftyped(addone);
 *  // invoke the function.
 *  int y = ftyped(1);
 *  // Can be directly converted to ffi::Function
 *  ffi::Function packed = ftype;
 * \endcode
 * \tparam R The return value of the function.
 * \tparam Args The argument signature of the function.
 */
template <typename R, typename... Args>
class TypedFunction<R(Args...)> {
 public:
  /*! \brief short hand for this function type */
  using TSelf = TypedFunction<R(Args...)>;
  /*! \brief default constructor */
  TypedFunction() = default;
  /*! \brief constructor from null */
  TypedFunction(std::nullptr_t null) {}  // NOLINT(*)
  /*!
   * \brief constructor from a function
   * \param packed The function
   */
  TypedFunction(Function packed) : packed_(packed) {}  // NOLINT(*)
  /*!
   * \brief construct from a lambda function with the same signature.
   *
   * Example usage:
   * \code
   * auto typed_lambda = [](int x)->int { return x + 1; }
   * // construct from packed function
   * TypedFunction<int(int)> ftyped(typed_lambda, "add_one");
   * // call the typed version.
   * CHECK_EQ(ftyped(1), 2);
   * \endcode
   *
   * \param typed_lambda typed lambda function.
   * \param name the name of the lambda function.
   * \tparam FLambda the type of the lambda function.
   */
  template <typename FLambda,
            typename = std::enable_if_t<std::is_convertible_v<FLambda, std::function<R(Args...)>>>>
  TypedFunction(FLambda typed_lambda, std::string name) {  // NOLINT(*)
    packed_ = Function::FromTyped(typed_lambda, name);
  }
  /*!
   * \brief construct from a lambda function with the same signature.
   *
   * This version does not take a name. It is highly recommend you use the
   * version that takes a name for the lambda.
   *
   * Example usage:
   * \code
   * auto typed_lambda = [](int x)->int { return x + 1; }
   * // construct from packed function
   * TypedFunction<int(int)> ftyped(typed_lambda);
   * // call the typed version.
   * CHECK_EQ(ftyped(1), 2);
   * \endcode
   *
   * \param typed_lambda typed lambda function.
   * \tparam FLambda the type of the lambda function.
   */
  template <typename FLambda,
            typename = std::enable_if_t<std::is_convertible_v<FLambda, std::function<R(Args...)>>>>
  TypedFunction(const FLambda& typed_lambda) {  // NOLINT(*)
    packed_ = Function::FromTyped(typed_lambda);
  }
  /*!
   * \brief copy assignment operator from typed lambda
   *
   * Example usage:
   * \code
   * // construct from packed function
   * TypedFunction<int(int)> ftyped;
   * ftyped = [](int x) { return x + 1; }
   * // call the typed version.
   * CHECK_EQ(ftyped(1), 2);
   * \endcode
   *
   * \param typed_lambda typed lambda function.
   * \tparam FLambda the type of the lambda function.
   * \returns reference to self.
   */
  template <typename FLambda,
            typename = std::enable_if_t<std::is_convertible_v<FLambda, std::function<R(Args...)>>>>
  TSelf& operator=(FLambda typed_lambda) {  // NOLINT(*)
    packed_ = Function::FromTyped(typed_lambda);
    return *this;
  }
  /*!
   * \brief copy assignment operator from ffi::Function.
   * \param packed The packed function.
   * \returns reference to self.
   */
  TSelf& operator=(Function packed) {
    packed_ = std::move(packed);
    return *this;
  }
  /*!
   * \brief Invoke the operator.
   * \param args The arguments
   * \returns The return value.
   */
  TVM_FFI_INLINE R operator()(Args... args) const {  // NOLINT(performance-unnecessary-value-param)
    if constexpr (std::is_same_v<R, void>) {
      packed_(std::forward<Args>(args)...);
    } else {
      Any res = packed_(std::forward<Args>(args)...);
      if constexpr (std::is_same_v<R, Any>) {
        return res;
      } else {
        return std::move(res).cast<R>();
      }
    }
  }
  /*!
   * \brief convert to ffi::Function
   * \return the internal ffi::Function
   */
  operator Function() const { return packed(); }  // NOLINT(google-explicit-constructor)
  /*!
   * \return reference the internal ffi::Function
   */
  const Function& packed() const& { return packed_; }
  /*!
   * \return r-value reference the internal ffi::Function
   */
  constexpr Function&& packed() && { return std::move(packed_); }
  /*! \return Whether the packed function is nullptr */
  bool operator==(std::nullptr_t null) const { return packed_ == nullptr; }
  /*! \return Whether the packed function is not nullptr */
  bool operator!=(std::nullptr_t null) const { return packed_ != nullptr; }
  /*!
   * \brief Get the type schema of `TypedFunction<R(Args...)>` in json format.
   * \return The type schema of the function in json format.
   */
  static std::string TypeSchema() { return details::FuncFunctorImpl<R, Args...>::TypeSchema(); }

 private:
  /*! \brief The internal packed function */
  Function packed_;
};

template <typename FType>
inline constexpr bool use_default_type_traits_v<TypedFunction<FType>> = false;

template <typename FType>
struct TypeTraits<TypedFunction<FType>> : public TypeTraitsBase {
  static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIFunction;

  TVM_FFI_INLINE static void CopyToAnyView(const TypedFunction<FType>& src, TVMFFIAny* result) {
    TypeTraits<Function>::CopyToAnyView(src.packed(), result);
  }

  TVM_FFI_INLINE static void MoveToAny(TypedFunction<FType> src, TVMFFIAny* result) {
    // Move from rvalue to trigger TypedFunction rvalue packed() overload
    TypeTraits<Function>::MoveToAny(std::move(src).packed(), result);
  }

  TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
    return src->type_index == TypeIndex::kTVMFFIFunction;
  }

  TVM_FFI_INLINE static TypedFunction<FType> CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
    return TypedFunction<FType>(TypeTraits<Function>::CopyFromAnyViewAfterCheck(src));
  }

  TVM_FFI_INLINE static std::optional<TypedFunction<FType>> TryCastFromAnyView(
      const TVMFFIAny* src) {
    std::optional<Function> opt = TypeTraits<Function>::TryCastFromAnyView(src);
    if (opt.has_value()) {
      return TypedFunction<FType>(*std::move(opt));
    } else {
      return std::nullopt;
    }
  }

  TVM_FFI_INLINE static std::string TypeStr() { return details::FunctionInfo<FType>::Sig(); }
  TVM_FFI_INLINE static std::string TypeSchema() { return TypedFunction<FType>::TypeSchema(); }
};

/*!
 * \brief helper function to get type index from key
 */
inline int32_t TypeKeyToIndex(std::string_view type_key) {
  int32_t type_index;
  TVMFFIByteArray type_key_array = {type_key.data(), type_key.size()};
  TVM_FFI_CHECK_SAFE_CALL(TVMFFITypeKeyToIndex(&type_key_array, &type_index));
  return type_index;
}

/*!
 * \brief Export typed function as a SafeCallType symbol that follows the FFI ABI.
 *
 * \param ExportName The symbol name to be exported.
 * \param Function The typed function.
 *
 * \sa ffi::TypedFunction
 *
 * \code
 *
 * int AddOne_(int x) {
 *   return x + 1;
 * }
 *
 * // Expose the function as "AddOne"
 * TVM_FFI_DLL_EXPORT_TYPED_FUNC(AddOne, AddOne_);
 *
 * // Expose the function as "SubOne"
 * TVM_FFI_DLL_EXPORT_TYPED_FUNC(SubOne, [](int x) {
 *   return x - 1;
 * });
 * \endcode
 *
 * \note The final symbol name is `__tvm_ffi_<ExportName>`.
 */
#define TVM_FFI_DLL_EXPORT_TYPED_FUNC(ExportName, Function)                            \
  extern "C" {                                                                         \
  TVM_FFI_DLL_EXPORT int __tvm_ffi_##ExportName(void* self, const TVMFFIAny* args,     \
                                                int32_t num_args, TVMFFIAny* result) { \
    TVM_FFI_SAFE_CALL_BEGIN();                                                         \
    using FuncInfo = ::tvm::ffi::details::FunctionInfo<decltype(Function)>;            \
    static std::string name = #ExportName;                                             \
    ::tvm::ffi::details::unpack_call<typename FuncInfo::RetType>(                      \
        std::make_index_sequence<FuncInfo::num_args>{}, &name, Function,               \
        reinterpret_cast<const ::tvm::ffi::AnyView*>(args), num_args,                  \
        reinterpret_cast<::tvm::ffi::Any*>(result));                                   \
    TVM_FFI_SAFE_CALL_END();                                                           \
  }                                                                                    \
  }
}  // namespace ffi
}  // namespace tvm
#endif  // TVM_FFI_FUNCTION_H_
