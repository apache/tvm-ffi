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
 * \file tvm/ffi/reflection/native_function.h
 * \brief Borrowed and owning typed hooks with native ABI dispatch and packed fallback.
 */
#ifndef TVM_FFI_REFLECTION_NATIVE_FUNCTION_H_
#define TVM_FFI_REFLECTION_NATIVE_FUNCTION_H_

#include <tvm/ffi/any.h>
#include <tvm/ffi/expected.h>
#include <tvm/ffi/function.h>

#include <exception>
#include <type_traits>
#include <utility>

namespace tvm {
namespace ffi {
namespace reflection {

template <typename Signature>
class NativeFunctionView;

/*!
 * \brief Borrowed typed function for reflection attributes with native ABI dispatch.
 *
 * Bind a named `Expected<R>(Args...) noexcept` or `R(Args...)` target with `FromNative<&Fn>()`.
 * The view stores its C ABI-compatible `TVMFFIAny(Args...) noexcept` adapter pointer. A target
 * that may throw is adapted to an Expected result at that ABI boundary.
 * A borrowed `ffi::Function` supports frontend registration and packed fallback. A compatible
 * `TypedFunction<R(Args...)>` or `TypedFunction<Expected<R>(Args...)>` lvalue may be borrowed.
 * This approach provides efficient native calls for most natively registered hooks while
 * preserving flexibility in how functions are registered.
 *
 * \code{.cpp}
 * TVM_FFI_INLINE int Increment(int value) { return value + 1; }
 * using FIncrement = tvm::ffi::reflection::NativeFunctionView<int(int)>;
 * FIncrement native = FIncrement::FromNative<&Increment>();
 * int result = native(3);
 * tvm::ffi::Expected<int> checked = native.CallExpected(3);
 * \endcode
 *
 * \note A borrowed packed function must remain alive while its view is used. Native adapters
 * have static lifetime.
 */
template <typename R, typename... Args>
class NativeFunctionView<R(Args...)> {
 public:
  static_assert(((std::is_lvalue_reference_v<Args> ||
                  (std::is_standard_layout_v<Args> && std::is_trivially_copyable_v<Args>)) &&
                 ...),
                "NativeFunctionView parameters require lvalue references or standard-layout, "
                "trivially-copyable values");
  static_assert((ffi::detail::ArgSupported<Args> && ...) && ffi::detail::RetSupported<R>,
                "NativeFunctionView signature must be supported by TypedFunction");
  /*! \brief Raw C ABI-compatible function pointer stored for native hook dispatch. */
  using ABIType = TVMFFIAny (*)(Args...) noexcept;

  /*! \brief A hook view cannot be empty. */
  NativeFunctionView() = delete;
  /*! \brief A null pointer is not a callable hook. */
  NativeFunctionView(std::nullptr_t) = delete;
  /*!
   * \brief Borrow a non-null typed packed function without retaining it.
   *
   * \param packed The function that must outlive this view.
   */
  template <typename PackedR,
            std::enable_if_t<std::is_same_v<PackedR, R> || std::is_same_v<PackedR, Expected<R>>,
                             int> = 0>
  NativeFunctionView(const TypedFunction<PackedR(Args...)>& packed)  // NOLINT(*)
      : data_(AnyView(packed.packed()).CopyToTVMFFIAny()) {
    TVM_FFI_CHECK(packed != nullptr, ValueError)
        << "NativeFunctionView requires a non-null typed function";
  }
  /*! \brief Reject a temporary packed function, which would leave a dangling view. */
  template <
      typename PackedR,
      std::enable_if_t<std::is_same_v<PackedR, R> || std::is_same_v<PackedR, Expected<R>>, int> = 0>
  NativeFunctionView(const TypedFunction<PackedR(Args...)>&&) = delete;

  /*!
   * \brief Bind a named native function through a context-free noexcept ABI adapter.
   *
   * \tparam Fn The native hook function.
   * \return A borrowed view of its static adapter.
   * \note Declare the native target `Fn` with `TVM_FFI_INLINE` when its definition is visible
   * to the adapter. This gives the compiler an opportunity to inline it into the adapter;
   * calls and result handling inside `Fn` may still remain.
   */
  template <auto Fn>
  static NativeFunctionView FromNative() {
    static_assert(Fn != nullptr, "NativeFunctionView requires a non-null native function");
    TVMFFIAny data;
    data.type_index = TypeIndex::kTVMFFIOpaquePtr;
    data.zero_padding = 0;
    TVM_FFI_CLEAR_PTR_PADDING_IN_FFI_ANY(&data);
    data.v_ptr = reinterpret_cast<void*>(&NativeABIAdapter<Fn>);
    return NativeFunctionView(UnsafeInit{}, data);
  }

  /*!
   * \brief Invoke the hook and return a value or an error.
   *
   * \param args The typed hook arguments.
   * \return A value or an error.
   */
  TVM_FFI_INLINE Expected<R> CallExpected(Args... args) const noexcept {
    return ffi::detail::ExpectedUnsafe::MoveFromTVMFFIAny<R>(CallABI(std::forward<Args>(args)...));
  }

  /*! \brief Invoke the hook and throw its error on failure. */
  TVM_FFI_INLINE R operator()(Args... args) const {
    return CallExpected(std::forward<Args>(args)...).value();
  }

 private:
  explicit NativeFunctionView(UnsafeInit, TVMFFIAny data) : data_(data) {}

  template <auto Fn>
  static TVMFFIAny NativeABIAdapter(Args... args) noexcept {
    return NativeABIFuncPtr<Fn>(std::forward<Args>(args)...);
  }

  template <Expected<R> (*Fn)(Args...) noexcept>
  TVM_FFI_INLINE static TVMFFIAny NativeABIFuncPtr(Args... args) noexcept {
    return ffi::detail::ExpectedUnsafe::MoveToTVMFFIAny(Fn(std::forward<Args>(args)...));
  }

  template <R (*Fn)(Args...)>
  TVM_FFI_INLINE static TVMFFIAny NativeABIFuncPtr(Args... args) noexcept {
    try {
      if constexpr (std::is_void_v<R>) {
        Fn(std::forward<Args>(args)...);
        return ffi::detail::ExpectedUnsafe::MoveToTVMFFIAny(Expected<void>());
      } else {
        return ffi::detail::ExpectedUnsafe::MoveToTVMFFIAny(
            Expected<R>(Fn(std::forward<Args>(args)...)));
      }
    } catch (const ffi::Error& error) {
      return ffi::detail::ExpectedUnsafe::MoveToTVMFFIAny(Expected<R>(Unexpected(error)));
    } catch (const std::exception& error) {
      return ffi::detail::ExpectedUnsafe::MoveToTVMFFIAny(
          Expected<R>(Unexpected(ffi::Error("InternalError", error.what(), ""))));
    }
  }

  TVM_FFI_INLINE TVMFFIAny CallABI(Args... args) const noexcept {
    if (TVM_FFI_PREDICT_TRUE(data_.type_index == TypeIndex::kTVMFFIOpaquePtr)) {
      return reinterpret_cast<ABIType>(data_.v_obj)(std::forward<Args>(args)...);
    }
    return CallABITail(std::forward<Args>(args)...);
  }

  TVM_FFI_COLD_CODE TVMFFIAny CallABITail(Args... args) const noexcept {
    auto* cell = TVMFFIFunctionGetCellPtr(data_.v_obj);
    AnyView packed_args[sizeof...(Args) > 0 ? sizeof...(Args) : 1];
    PackedArgs::Fill(packed_args, std::forward<Args>(args)...);
    Any result;
    // The function cell follows the object header in the C ABI allocation.
    // NOLINTNEXTLINE(clang-analyzer-security.ArrayBound)
    int code = cell->safe_call(data_.v_obj, reinterpret_cast<const TVMFFIAny*>(packed_args),
                               sizeof...(Args), reinterpret_cast<TVMFFIAny*>(&result));
    if (TVM_FFI_PREDICT_FALSE(code != 0)) {
      return ffi::detail::ExpectedUnsafe::MoveToTVMFFIAny(
          Expected<R>(Unexpected(ffi::detail::MoveFromSafeCallRaised())));
    }
    if (result.type_index() == TypeIndex::kTVMFFIError) {
      return ffi::detail::ExpectedUnsafe::MoveToTVMFFIAny(Expected<R>(
          Unexpected(ffi::detail::AnyUnsafe::MoveFromAnyAfterCheck<Error>(std::move(result)))));
    }
    if constexpr (std::is_same_v<R, Any>) {
      return ffi::detail::ExpectedUnsafe::MoveToTVMFFIAny(Expected<R>(std::move(result)));
    } else if constexpr (std::is_void_v<R>) {
      if (result.type_index() == TypeIndex::kTVMFFINone) {
        return ffi::detail::ExpectedUnsafe::MoveToTVMFFIAny(Expected<void>());
      }
    } else if (auto value = result.template try_cast<R>()) {
      return ffi::detail::ExpectedUnsafe::MoveToTVMFFIAny(Expected<R>(*std::move(value)));
    }
    return ffi::detail::ExpectedUnsafe::MoveToTVMFFIAny(
        Expected<R>(Unexpected(Error("TypeError", "Packed hook result type mismatch", ""))));
  }

  template <typename, typename>
  friend struct ::tvm::ffi::TypeTraits;
  template <typename>
  friend class NativeFunction;

  TVMFFIAny data_;
};

template <typename Signature>
class NativeFunction;

/*!
 * \brief Owning counterpart of NativeFunctionView for stored hook values.
 *
 * Construct a borrowed view, then copy it into this owner. A packed function remains valid after
 * the original TypedFunction is destroyed; native adapters retain their static function address.
 * The owner stores one `ffi::Any`, which manages the lifetime of packed function objects.
 *
 * \code{.cpp}
 * int Increment(int value) { return value + 1; }
 * using FIncrement = tvm::ffi::reflection::NativeFunction<int(int)>;
 * FIncrement owned = FIncrement::FromNative<&Increment>();
 * int result = owned(3);
 * tvm::ffi::Expected<int> checked = owned.CallExpected(3);
 * \endcode
 */
template <typename R, typename... Args>
class NativeFunction<R(Args...)> {
 public:
  /*! \brief The borrowed view with the same callable signature. */
  using View = NativeFunctionView<R(Args...)>;

  /*!
   * \brief Retain a copy of a borrowed native or packed hook.
   *
   * \param view The borrowed view to copy.
   */
  explicit NativeFunction(View view) : data_(AnyView(view)) {}

  /*!
   * \brief Retain a non-null typed packed function.
   *
   * \param packed The typed function to copy.
   */
  template <typename PackedR,
            std::enable_if_t<std::is_same_v<PackedR, R> || std::is_same_v<PackedR, Expected<R>>,
                             int> = 0>
  NativeFunction(const TypedFunction<PackedR(Args...)>& packed)  // NOLINT(*)
      : NativeFunction(View(packed)) {}

  /*!
   * \brief Copy a borrowed hook into an owning hook.
   *
   * \param view The borrowed view to retain.
   * \return An independent owner of the hook.
   */
  static NativeFunction From(View view) { return NativeFunction(view); }

  /*!
   * \brief Own a named native hook bound through the same ABI adapter as View.
   *
   * \tparam Fn The named native hook function.
   * \return An owner of the generated static adapter.
   * \note As with `View::FromNative`, declaring `Fn` with `TVM_FFI_INLINE` helps the compiler
   * inline a visible target into the adapter, without guaranteeing complete call elimination.
   */
  template <auto Fn>
  static NativeFunction FromNative() {
    return From(View::template FromNative<Fn>());
  }

  /*!
   * \brief Invoke the hook and return a value or an error.
   *
   * \param args The typed hook arguments.
   * \return A value or an error.
   */
  TVM_FFI_INLINE Expected<R> CallExpected(Args... args) const noexcept {
    return View(UnsafeInit{}, AnyView(data_).CopyToTVMFFIAny())
        .CallExpected(std::forward<Args>(args)...);
  }

  /*! \brief Invoke the hook and throw its error on failure. */
  TVM_FFI_INLINE R operator()(Args... args) const {
    return CallExpected(std::forward<Args>(args)...).value();
  }

 private:
  explicit NativeFunction(UnsafeInit, Any data) : data_(std::move(data)) {}

  template <typename, typename>
  friend struct ::tvm::ffi::TypeTraits;

  Any data_;
};

}  // namespace reflection

template <typename Signature>
inline constexpr bool use_default_type_traits_v<reflection::NativeFunctionView<Signature>> = false;

template <typename Signature>
inline constexpr bool use_default_type_traits_v<reflection::NativeFunction<Signature>> = false;

template <typename Signature>
struct TypeTraits<reflection::NativeFunctionView<Signature>> : public TypeTraitsBase {
  using View = reflection::NativeFunctionView<Signature>;
  static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIAny;

  static void CopyToAnyView(const View& src, TVMFFIAny* result) { *result = src.data_; }
  static void MoveToAny(View src, TVMFFIAny* result) {
    *result = detail::AnyUnsafe::MoveAnyToTVMFFIAny(Any(AnyView::CopyFromTVMFFIAny(src.data_)));
  }
  static bool CheckAnyStrict(const TVMFFIAny* src) {
    return (src->type_index == TypeIndex::kTVMFFIOpaquePtr && src->v_ptr != nullptr) ||
           src->type_index == TypeIndex::kTVMFFIFunction;
  }
  static View CopyFromAnyViewAfterCheck(const TVMFFIAny* src) { return View(UnsafeInit{}, *src); }
  static View MoveFromAnyAfterCheck(TVMFFIAny* src) { return CopyFromAnyViewAfterCheck(src); }
  static std::optional<View> TryCastFromAnyView(const TVMFFIAny* src) {
    if (CheckAnyStrict(src)) return CopyFromAnyViewAfterCheck(src);
    return std::nullopt;
  }
  static std::string TypeStr() {
    return "Variant<" + TypeTraits<void*>::TypeStr() + ", " +
           TypeTraits<TypedFunction<Signature>>::TypeStr() + ">";
  }
  static std::string TypeSchema() {
    return R"({"type":"Variant","args":[)" + TypeTraits<void*>::TypeSchema() + "," +
           TypeTraits<TypedFunction<Signature>>::TypeSchema() + "]}";
  }
};

template <typename Signature>
struct TypeTraits<reflection::NativeFunction<Signature>> : public TypeTraitsBase {
  using Owner = reflection::NativeFunction<Signature>;
  using View = typename Owner::View;
  static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIAny;

  static void CopyToAnyView(const Owner& src, TVMFFIAny* result) {
    *result = AnyView(src.data_).CopyToTVMFFIAny();
  }
  static void MoveToAny(Owner src, TVMFFIAny* result) {
    *result = detail::AnyUnsafe::MoveAnyToTVMFFIAny(std::move(src.data_));
  }
  static bool CheckAnyStrict(const TVMFFIAny* src) { return TypeTraits<View>::CheckAnyStrict(src); }
  static Owner CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
    return Owner(TypeTraits<View>::CopyFromAnyViewAfterCheck(src));
  }
  static Owner MoveFromAnyAfterCheck(TVMFFIAny* src) {
    return Owner(UnsafeInit{}, detail::AnyUnsafe::MoveTVMFFIAnyToAny(src));
  }
  static std::optional<Owner> TryCastFromAnyView(const TVMFFIAny* src) {
    if (CheckAnyStrict(src)) return CopyFromAnyViewAfterCheck(src);
    return std::nullopt;
  }
  static std::string TypeStr() { return TypeTraits<View>::TypeStr(); }
  static std::string TypeSchema() { return TypeTraits<View>::TypeSchema(); }
};

}  // namespace ffi
}  // namespace tvm

#endif  // TVM_FFI_REFLECTION_NATIVE_FUNCTION_H_
