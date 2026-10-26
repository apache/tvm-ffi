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
/*! \file tvm/ffi/big_int.h
 *  \brief Integers with inline int64 storage and arbitrary-precision arithmetic.
 */
#ifndef TVM_FFI_BIG_INT_H_
#define TVM_FFI_BIG_INT_H_

#include <tvm/ffi/error.h>
#include <tvm/ffi/memory.h>
#include <tvm/ffi/type_traits.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <istream>
#include <limits>
#include <optional>
#include <ostream>
#include <string>
#include <type_traits>
#include <utility>

#if (defined(_MSVC_LANG) && _MSVC_LANG >= 202002L) || __cplusplus >= 202002L
#include <bit>
#endif

namespace tvm {
namespace ffi {
namespace detail {

struct BigIntUnsafe;
/*!
 * \brief Immutable two's-complement words stored immediately after the header.
 * \note Words MUST form the minimal signed representation, without redundant sign extension
 * and retaining a necessary sign guard. Values fitting int64_t MUST use kTVMFFIInt instead.
 */
class BigIntObj : public Object {
 public:
  /*!
   * \brief Set the logical size; the inplace allocator supplies uninitialized trailing words.
   * \param size Number of logical words. Write every word before reading or normalizing it,
   * and normalize the completed construction before exposing a canonical BigInt.
   */
  explicit BigIntObj(size_t size) : size_(size) {}
  BigIntObj(const BigIntObj&) = delete;

  BigIntObj& operator=(const BigIntObj&) = delete;
  static constexpr uint32_t _type_index = TypeIndex::kTVMFFIBigInt;
  static constexpr bool _type_final = true;
  TVM_FFI_DECLARE_OBJECT_INFO_STATIC(StaticTypeKey::kTVMFFIBigInt, BigIntObj, Object);

 private:
  size_t size_;
  friend struct BigIntUnsafe;
};
}  // namespace detail
/*!
 * \brief An immutable integer, stored inline whenever it fits int64_t.
 *
 * Heap values lie outside the int64_t range, so inline and heap representations cannot be equal.
 *
 * ArrayView borrows least-significant-word-first int64_t words; its storage must outlive the view.
 *
 * Arithmetic has unbounded signed integer semantics, subject to allocation limits.
 * / and % truncate toward zero; floordiv and floormod round the quotient down.
 * Bitwise operations use infinite two's-complement sign extension. Right shifts
 * round down, and negative shift counts are errors. Floating conversions are explicit.
 */
class BigInt {
 public:
  /// \cond Doxygen_Suppress
  /*! \brief Borrowed read-only words, least-significant first. Size counts 64-bit words. */
  class ArrayView {
   public:
    /*!
     * \brief Borrow a least-significant-word-first array without taking ownership.
     * \param data Word storage that must outlive this view and references obtained from it.
     * \param size Number of readable int64_t words starting at data.
     */
    TVM_FFI_INLINE ArrayView(const int64_t* data, size_t size) : data_(data), size_(size) {}
    /*!
     * \brief Borrow the word at index i.
     * \param i Index smaller than size().
     * \return A reference valid while the borrowed storage remains alive.
     */
    TVM_FFI_INLINE const int64_t& operator[](size_t i) const { return data_[i]; }
    /*! \brief Return the number of borrowed 64-bit words. */
    TVM_FFI_INLINE size_t size() const { return size_; }

   private:
    const int64_t* data_;
    size_t size_;
  };
  /// \endcond
  /*! \brief Construct the integer zero in the inline representation. */
  TVM_FFI_INLINE BigInt() { Reset(); }
  /*!
   * \brief Construct an integer, stored inline when representable as int64_t.
   * \param value Integral input of at most 64 bits, preserved exactly.
   */
  template <typename Int, std::enable_if_t<(std::is_integral_v<Int> && sizeof(Int) <= 8), int> = 0>
  TVM_FFI_INLINE BigInt(Int value);  // NOLINT(*)
  /*!
   * \brief Convert a finite double to an integer, truncating toward zero.
   * \param value Finite input; NaN raises ValueError and infinity raises OverflowError.
   */
  TVM_FFI_INLINE explicit BigInt(double value);

  /*! \brief Copy the value, sharing immutable heap storage when present. */
  TVM_FFI_INLINE BigInt(const BigInt& other) : data_(other.data_) {
    if (TVM_FFI_PREDICT_FALSE(data_.type_index != TypeIndex::kTVMFFIInt)) {
      detail::ObjectUnsafe::IncRefObjectHandle(data_.v_obj);
    }
  }

  /*! \brief Transfer the value and reset the source to zero. */
  TVM_FFI_INLINE BigInt(BigInt&& other) noexcept : data_(other.data_) { other.Reset(); }

  TVM_FFI_INLINE ~BigInt() {
    if (TVM_FFI_PREDICT_FALSE(data_.type_index != TypeIndex::kTVMFFIInt)) {
      detail::ObjectUnsafe::DecRefObjectHandle(data_.v_obj);
    }
  }

  TVM_FFI_INLINE BigInt& operator=(const BigInt& other) {
    BigInt(other).swap(*this);
    return *this;
  }

  TVM_FFI_INLINE BigInt& operator=(BigInt&& other) noexcept {
    BigInt(std::move(other)).swap(*this);
    return *this;
  }

  /*! \brief Exchange values and storage ownership without changing reference counts. */
  TVM_FFI_INLINE void swap(BigInt& other) noexcept {
    std::swap(data_.type_index, other.data_.type_index);
    std::swap(data_.zero_padding, other.data_.zero_padding);
    // Supported compilers allow exchanging the payload through its integer union member.
    std::swap(data_.v_int64, other.data_.v_int64);
  }

  TVM_FFI_INLINE explicit operator bool() const noexcept {
    return TVM_FFI_PREDICT_FALSE(data_.type_index != TypeIndex::kTVMFFIInt) || data_.v_int64 != 0;
  }
  /*! \brief Convert to int64_t, throwing OverflowError if it does not fit. */
  TVM_FFI_INLINE explicit operator int64_t() const {
    if (TVM_FFI_PREDICT_FALSE(data_.type_index != TypeIndex::kTVMFFIInt)) {
      TVM_FFI_THROW(OverflowError) << "BigInt does not fit int64_t";
    }
    return data_.v_int64;
  }
  /*! \brief Convert to finite double, rounding nearest with ties to even. */
  TVM_FFI_INLINE explicit operator double() const;
  /*! \brief Return an integer conversion when the value fits the requested type. */
  template <typename Int, std::enable_if_t<(std::is_integral_v<Int> && sizeof(Int) <= 8), int> = 0>
  TVM_FFI_INLINE std::optional<Int> as() const;
  /*! \brief Return a hash of the integer value. */
  TVM_FFI_INLINE uint64_t hash() const;

  // Operator overloads
  TVM_FFI_INLINE BigInt& operator+=(const BigInt& rhs);
  TVM_FFI_INLINE BigInt& operator-=(const BigInt& rhs);
  TVM_FFI_INLINE BigInt& operator*=(const BigInt& rhs);
  TVM_FFI_INLINE BigInt& operator/=(const BigInt& rhs);
  TVM_FFI_INLINE BigInt& operator%=(const BigInt& rhs);
  TVM_FFI_INLINE BigInt& operator&=(const BigInt& rhs);
  TVM_FFI_INLINE BigInt& operator|=(const BigInt& rhs);
  TVM_FFI_INLINE BigInt& operator^=(const BigInt& rhs);
  TVM_FFI_INLINE BigInt& operator<<=(const BigInt& rhs);
  TVM_FFI_INLINE BigInt& operator>>=(const BigInt& rhs);
  TVM_FFI_INLINE BigInt& operator++();
  TVM_FFI_INLINE BigInt& operator--();
  TVM_FFI_INLINE BigInt operator++(int);
  TVM_FFI_INLINE BigInt operator--(int);

 private:
  /*!
   * \brief Take ownership of a canonical multiword integer without an extra reference.
   * \param ptr Owned normalized object with more than one word.
   */
  TVM_FFI_INLINE explicit BigInt(ObjectPtr<detail::BigIntObj> ptr) : BigInt() {
    TVM_FFI_CLEAR_PTR_PADDING_IN_FFI_ANY(&data_);
    data_.v_obj = detail::ObjectUnsafe::MoveObjectPtrToTVMFFIObjectPtr(std::move(ptr));
    data_.type_index = TypeIndex::kTVMFFIBigInt;
  }
  /*! \brief Reset a detached cell to inline zero without releasing ownership. */
  TVM_FFI_INLINE void Reset() noexcept {
    data_.type_index = TypeIndex::kTVMFFIInt;
    data_.zero_padding = 0;
    data_.v_int64 = 0;
  }
  TVMFFIAny data_;
  friend struct detail::BigIntUnsafe;
  friend struct TypeTraits<BigInt>;
};

namespace detail {
struct BigIntUnsafe {
  /*!
   * \brief Test the inline integer representation.
   * \param value Integer to inspect.
   * \return True iff its value is stored inline.
   */
  TVM_FFI_INLINE static bool CheckInt64(const BigInt& value) {
    return value.data_.type_index == TypeIndex::kTVMFFIInt;
  }
  /*!
   * \brief Recognize a primitive int64_t operand.
   * \return Always true.
   */
  TVM_FFI_INLINE static bool CheckInt64(const int64_t&) { return true; }
  /*!
   * \brief Read an inline payload.
   * \param value Integer satisfying CheckInt64(value).
   * \return The exact int64_t value.
   */
  TVM_FFI_INLINE static int64_t GetInt64(const BigInt& value) { return value.data_.v_int64; }
  /*!
   * \brief Read a primitive operand.
   * \param value The int64_t operand.
   * \return The operand unchanged.
   */
  TVM_FFI_INLINE static int64_t GetInt64(const int64_t& value) { return value; }
  /*!
   * \brief Borrow an integer's canonical words.
   * \param value Owner that must outlive the view.
   * \return A nonempty canonical view.
   */
  TVM_FFI_INLINE static BigInt::ArrayView GetArrayView(const BigInt& value) {
    if (value.data_.type_index == TypeIndex::kTVMFFIInt) return {&value.data_.v_int64, 1};
    const auto* obj = ObjectUnsafe::RawObjectPtrFromUnowned<BigIntObj>(value.data_.v_obj);
    return {reinterpret_cast<const int64_t*>(obj + 1), obj->size_};
  }
  /*!
   * \brief Borrow a primitive integer as one word.
   * \param value Storage that must outlive the view.
   * \return A one-word canonical view.
   */
  TVM_FFI_INLINE static BigInt::ArrayView GetArrayView(const int64_t& value) { return {&value, 1}; }
  /*!
   * \brief Borrow construction words, which need not be normalized.
   * \param ptr Non-null object that must outlive the view.
   * \return The current logical words; normalize before requiring canonical form.
   */
  TVM_FFI_INLINE static BigInt::ArrayView GetArrayView(const ObjectPtr<BigIntObj>& ptr) {
    return {GetMutableData(ptr), ptr->size_};
  }
  /*!
   * \brief Borrow the words of an integer ABI cell.
   * \param value Non-null int64 or BigInt cell; its storage must outlive the view.
   * \return A nonempty canonical view.
   */
  TVM_FFI_INLINE static BigInt::ArrayView GetArrayView(const TVMFFIAny* value) {
    if (value->type_index == TypeIndex::kTVMFFIInt) return {&value->v_int64, 1};
    const auto* obj = ObjectUnsafe::RawObjectPtrFromUnowned<BigIntObj>(value->v_obj);
    return {reinterpret_cast<const int64_t*>(obj + 1), obj->size_};
  }
  /*!
   * \brief Access trailing words during construction.
   * \param ptr Non-null, unshared construction object.
   * \return Mutable storage valid while ptr owns the object.
   */
  TVM_FFI_INLINE static int64_t* GetMutableData(const ObjectPtr<BigIntObj>& ptr) {
    return reinterpret_cast<int64_t*>(ptr.get() + 1);
  }
  /*!
   * \brief Shrink construction length without reallocating.
   * \param ptr Non-null, unshared construction object.
   * \param size New logical length, no greater than the existing length.
   */
  TVM_FFI_INLINE static void ShrinkSize(const ObjectPtr<BigIntObj>& ptr, size_t size) {
    ptr->size_ = size;
  }
  /*!
   * \brief Prune redundant sign words and demote fitting values to int64_t.
   * \param ptr Owned, unshared object with every logical word initialized; may have zero words.
   * \return The canonical integer, retaining the allocation when more than one word remains.
   */
  static BigInt Normalize(ObjectPtr<BigIntObj> ptr) {
    size_t size = GetArrayView(ptr).size();
    const int64_t* data = GetMutableData(ptr);
    while (size > 1 && data[size - 1] == (data[size - 2] < 0 ? -1 : 0)) --size;
    if (size <= 1) return BigInt(size ? data[0] : 0);
    // Prune logical length only: keep the allocation that already holds the result.
    ShrinkSize(ptr, size);
    return BigInt(std::move(ptr));
  }
};

// Fallback arithmetic uses unsigned words, with local half-word products for portable C++17.
namespace int_ops {
/*!
 * \brief Whether a nonempty signed-word view represents a negative integer.
 * \param x Borrowed words, least-significant first; the last word carries the sign.
 * \return True iff the represented integer is negative.
 */
TVM_FFI_INLINE bool IsNegative(BigInt::ArrayView x) { return x[x.size() - 1] < 0; }
/*!
 * \brief Whether a normalized signed-word view represents zero.
 * \param x Nonempty borrowed view in canonical form, without redundant sign words.
 * \return True iff the represented integer is zero.
 */
TVM_FFI_INLINE bool IsNormalizedZero(BigInt::ArrayView x) { return x.size() == 1 && x[0] == 0; }

/*!
 * \brief Check the exact signed integer sum for int64_t overflow.
 * \param a First operand.
 * \param b Second operand.
 * \param out Non-null output pointer. Exact result when false; unspecified when true.
 * \return True iff the mathematical sum does not fit int64_t.
 */
TVM_FFI_INLINE bool AddOverflow(int64_t a, int64_t b, int64_t* out) {
#if (defined(__GNUC__) || defined(__clang__)) && !defined(TVM_FFI_BIGINT_FORCE_PORTABLE)
  return TVM_FFI_PREDICT_FALSE(__builtin_add_overflow(a, b, out));
#else
  constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
  constexpr int64_t kMin = std::numeric_limits<int64_t>::min();
  // Test the bound selected by b's sign before adding; both the bound
  // expression and the eventual arithmetic stay representable.
  if (TVM_FFI_PREDICT_FALSE((b > 0 && a > kMax - b) || (b < 0 && a < kMin - b))) return true;
  *out = a + b;
  return false;
#endif
}
/*!
 * \brief Check the exact signed integer difference for int64_t overflow.
 * \param a First operand.
 * \param b Second operand.
 * \param out Non-null output pointer. Exact result when false; unspecified when true.
 * \return True iff the mathematical difference does not fit int64_t.
 */
TVM_FFI_INLINE bool SubOverflow(int64_t a, int64_t b, int64_t* out) {
#if (defined(__GNUC__) || defined(__clang__)) && !defined(TVM_FFI_BIGINT_FORCE_PORTABLE)
  return TVM_FFI_PREDICT_FALSE(__builtin_sub_overflow(a, b, out));
#else
  constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
  constexpr int64_t kMin = std::numeric_limits<int64_t>::min();
  // Test the bound selected by b's sign before subtracting; both the bound
  // expression and the eventual arithmetic stay representable.
  if (TVM_FFI_PREDICT_FALSE((b < 0 && a > kMax + b) || (b > 0 && a < kMin + b))) return true;
  *out = a - b;
  return false;
#endif
}
/*!
 * \brief Check the exact signed integer product for int64_t overflow.
 * \param a First operand.
 * \param b Second operand.
 * \param out Non-null output pointer. Exact result when false; unspecified when true.
 * \return True iff the mathematical product does not fit int64_t.
 */
TVM_FFI_INLINE bool MulOverflow(int64_t a, int64_t b, int64_t* out) {
#if (defined(__GNUC__) || defined(__clang__)) && !defined(TVM_FFI_BIGINT_FORCE_PORTABLE)
  return TVM_FFI_PREDICT_FALSE(__builtin_mul_overflow(a, b, out));
#else
  constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
  constexpr int64_t kMin = std::numeric_limits<int64_t>::min();
  // Safe bounds by sign: ++ a<=kMax/b, +- b>=kMin/a, -+ a>=kMin/b,
  // -- a>=kMax/b (dividing by a negative flips the inequality).
  // Zero cannot overflow; the sign guards avoid zero division and INT64_MIN/-1.
  if (TVM_FFI_PREDICT_FALSE(a > 0 ? (b > 0 ? a > kMax / b : b < kMin / a)
                                  : (a < 0 && (b > 0 ? a < kMin / b : b < 0 && a < kMax / b)))) {
    return true;
  }
  *out = a * b;
  return false;
#endif
}
/*!
 * \brief Check the quotient for division rounded toward zero.
 * \param a Dividend.
 * \param b Divisor; zero raises ZeroDivisionError.
 * \param out Non-null output. Exact result when false; unspecified when true.
 * \return True iff the mathematical quotient does not fit int64_t.
 */
TVM_FFI_INLINE bool TruncDivOverflow(int64_t a, int64_t b, int64_t* out) {
  // Positive division needs neither a signed-overflow guard nor a rounding correction.
  if (TVM_FFI_PREDICT_TRUE(a >= 0 && b > 0)) {
    *out = a / b;
    return false;
  }
  constexpr int64_t kMin = std::numeric_limits<int64_t>::min();
  if (TVM_FFI_PREDICT_FALSE(b == 0)) TVM_FFI_THROW(ZeroDivisionError) << "Division by zero";
  // INT64_MIN / -1 is the only quotient that needs promotion.
  if (TVM_FFI_PREDICT_FALSE(a == kMin && b == -1)) return true;
  *out = a / b;
  return false;
}
/*!
 * \brief Check the remainder for division rounded toward zero.
 * \param a Dividend.
 * \param b Divisor; zero raises ZeroDivisionError.
 * \param out Non-null output. Exact result when false; unspecified when true.
 * \return True iff the mathematical remainder does not fit int64_t.
 */
TVM_FFI_INLINE bool TruncModOverflow(int64_t a, int64_t b, int64_t* out) {
  // Positive division needs neither a signed-overflow guard nor a rounding correction.
  if (TVM_FFI_PREDICT_TRUE(a >= 0 && b > 0)) {
    *out = a % b;
    return false;
  }
  constexpr int64_t kMin = std::numeric_limits<int64_t>::min();
  if (TVM_FFI_PREDICT_FALSE(b == 0)) TVM_FFI_THROW(ZeroDivisionError) << "Division by zero";
  // The remainder is zero here, but evaluating native % would overflow the quotient.
  if (TVM_FFI_PREDICT_FALSE(a == kMin && b == -1)) {
    *out = 0;
    return false;
  }
  *out = a % b;
  return false;
}
/*!
 * \brief Check the quotient for division rounded down.
 * \param a Dividend.
 * \param b Divisor; zero raises ZeroDivisionError.
 * \param out Non-null output. Exact result when false; unspecified when true.
 * \return True iff the mathematical quotient does not fit int64_t.
 */
TVM_FFI_INLINE bool FloorDivOverflow(int64_t a, int64_t b, int64_t* out) {
  // Positive division needs neither a signed-overflow guard nor a rounding correction.
  if (TVM_FFI_PREDICT_TRUE(a >= 0 && b > 0)) {
    *out = a / b;
    return false;
  }
  constexpr int64_t kMin = std::numeric_limits<int64_t>::min();
  if (TVM_FFI_PREDICT_FALSE(b == 0)) TVM_FFI_THROW(ZeroDivisionError) << "Division by zero";
  // INT64_MIN / -1 is the only quotient that needs promotion.
  if (TVM_FFI_PREDICT_FALSE(a == kMin && b == -1)) return true;
  int64_t q = a / b;
  int64_t r = a % b;
  // A nonexact negative quotient rounds one lower; exact division needs no adjustment.
  if (r != 0 && ((a < 0) != (b < 0))) --q;
  *out = q;
  return false;
}
/*!
 * \brief Check the remainder for division rounded down.
 * \param a Dividend.
 * \param b Divisor; zero raises ZeroDivisionError.
 * \param out Non-null output. Exact result when false; unspecified when true.
 * \return True iff the mathematical remainder does not fit int64_t.
 */
TVM_FFI_INLINE bool FloorModOverflow(int64_t a, int64_t b, int64_t* out) {
  // Positive division needs neither a signed-overflow guard nor a rounding correction.
  if (TVM_FFI_PREDICT_TRUE(a >= 0 && b > 0)) {
    *out = a % b;
    return false;
  }
  constexpr int64_t kMin = std::numeric_limits<int64_t>::min();
  if (TVM_FFI_PREDICT_FALSE(b == 0)) TVM_FFI_THROW(ZeroDivisionError) << "Division by zero";
  // The remainder is zero here, but evaluating native % would overflow the quotient.
  if (TVM_FFI_PREDICT_FALSE(a == kMin && b == -1)) {
    *out = 0;
    return false;
  }
  int64_t r = a % b;
  // Compensate for the quotient rounding down, preserving a = q*b + r.
  if (r != 0 && ((a < 0) != (b < 0))) r += b;
  *out = r;
  return false;
}
/*!
 * \brief Interpret an unsigned word as a signed two's-complement integer.
 * \param bits The 64-bit representation.
 * \return The signed value with the same bits.
 */
TVM_FFI_INLINE int64_t BitcastToInt64(uint64_t bits) {
#if defined(__cpp_lib_bit_cast) && __cpp_lib_bit_cast >= 201806L
  return std::bit_cast<int64_t>(bits);
#else
  // The C++17 fallback relies on GCC/Clang's alternate-member union-read extension.
  union {
    uint64_t unsigned_value;
    int64_t signed_value;
  } value{bits};
  return value.signed_value;
#endif
}
/*!
 * \brief Check a signed left shift, rejecting negative counts.
 * \param a Integer to shift.
 * \param b Nonnegative bit count; negative counts raise ValueError.
 * \param out Non-null output. Exact result when false; unspecified when true.
 * \return True iff the mathematical shifted value does not fit int64_t.
 */
TVM_FFI_INLINE bool LeftShiftOverflow(int64_t a, int64_t b, int64_t* out) {
  constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
  if (TVM_FFI_PREDICT_FALSE(b < 0)) TVM_FFI_THROW(ValueError) << "Negative BigInt shift count";
  if (b == 0 || a == 0) {
    *out = a;
    return false;
  }
  // For positive a require a<=INT64_MAX>>b; for negative a require a>=-2^(63-b).
  // Zero/count-zero are handled above; rejecting b>=64 first keeps both bound shifts safe.
  if (TVM_FFI_PREDICT_FALSE(b >= 64 || a > (kMax >> b) || a < -(int64_t{1} << (63 - b)))) {
    return true;
  }
  *out = BitcastToInt64(static_cast<uint64_t>(a) << b);
  return false;
}

/*!
 * \brief Shift right with arithmetic sign extension; this operation cannot overflow.
 * \param a Integer to shift.
 * \param b Nonnegative bit count; negative counts raise ValueError.
 * \return The shifted value, or sign fill (zero or -1) for counts of 64 or more.
 */
TVM_FFI_INLINE int64_t RightShift(int64_t a, int64_t b) {
  if (TVM_FFI_PREDICT_FALSE(b < 0)) TVM_FFI_THROW(ValueError) << "Negative BigInt shift count";
  if (TVM_FFI_PREDICT_FALSE(b >= 64)) return a < 0 ? -1 : 0;
  return a >> b;
}
/*!
 * \brief Test normalized integers for equal values.
 * \param a Nonempty canonical first operand.
 * \param b Nonempty canonical second operand.
 * \return Whether the word counts and every corresponding word are equal.
 */
inline bool EqualFallback(BigInt::ArrayView a, BigInt::ArrayView b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i] != b[i]) return false;
  }
  return true;
}
/*!
 * \brief Compare signed integers.
 * \param a Nonempty canonical first operand.
 * \param b Nonempty canonical second operand.
 * \return -1, 0, or 1 according to signed ordering.
 */
inline int CompareFallback(BigInt::ArrayView a, BigInt::ArrayView b) {
  if (IsNegative(a) != IsNegative(b)) return IsNegative(a) ? -1 : 1;
  uint64_t sign_a = IsNegative(a) ? ~uint64_t{0} : 0;
  uint64_t sign_b = IsNegative(b) ? ~uint64_t{0} : 0;
  for (size_t i = std::max(a.size(), b.size()); i != 0; --i) {
    uint64_t x = i <= a.size() ? static_cast<uint64_t>(a[i - 1]) : sign_a;
    uint64_t y = i <= b.size() ? static_cast<uint64_t>(b[i - 1]) : sign_b;
    if (x != y) return x > y ? 1 : -1;
  }
  return 0;
}
/*!
 * \brief Add signed integers.
 * \param a Nonempty canonical first operand.
 * \param b Nonempty canonical second operand.
 * \return The canonical sum.
 */
inline BigInt AddFallback(BigInt::ArrayView a, BigInt::ArrayView b) {
  // One sign-extension word retains signed overflow until normalization.
  // Unsigned words wrap modulo 2^64 for carry; signed overflow is UB even in C++20.
  // BitcastToInt64 preserves the result bits when storing each signed word.
  size_t size = std::max(a.size(), b.size()) + 1;
  auto ptr = make_inplace_array_object<BigIntObj, int64_t>(size, size);
  int64_t* data = BigIntUnsafe::GetMutableData(ptr);
  uint64_t sign_a = IsNegative(a) ? ~uint64_t{0} : 0;
  uint64_t sign_b = IsNegative(b) ? ~uint64_t{0} : 0;
  uint64_t carry = 0;
  for (size_t i = 0; i < size; ++i) {
    uint64_t x = i < a.size() ? static_cast<uint64_t>(a[i]) : sign_a;
    uint64_t y = i < b.size() ? static_cast<uint64_t>(b[i]) : sign_b;
    uint64_t sum = x + y;
    uint64_t result = sum + carry;
    carry = (sum < x) || (result < sum);
    data[i] = BitcastToInt64(result);
  }
  return BigIntUnsafe::Normalize(std::move(ptr));
}
/*!
 * \brief Subtract signed integers.
 * \param a Nonempty canonical first operand.
 * \param b Nonempty canonical second operand.
 * \return The canonical difference a - b.
 */
inline BigInt SubFallback(BigInt::ArrayView a, BigInt::ArrayView b) {
  // One sign-extension word retains signed overflow until normalization.
  // Unsigned modulo-2^64 addition implements a + ~b + 1 without signed-overflow UB,
  // which persists in C++20; BitcastToInt64 preserves the resulting stored bits.
  size_t size = std::max(a.size(), b.size()) + 1;
  auto ptr = make_inplace_array_object<BigIntObj, int64_t>(size, size);
  int64_t* data = BigIntUnsafe::GetMutableData(ptr);
  uint64_t sign_a = IsNegative(a) ? ~uint64_t{0} : 0;
  uint64_t sign_b = IsNegative(b) ? ~uint64_t{0} : 0;
  uint64_t carry = 1;
  for (size_t i = 0; i < size; ++i) {
    uint64_t x = i < a.size() ? static_cast<uint64_t>(a[i]) : sign_a;
    uint64_t y = i < b.size() ? static_cast<uint64_t>(b[i]) : sign_b;
    y = ~y;  // a - b = a + ~b + 1, after sign extension.
    uint64_t sum = x + y;
    uint64_t result = sum + carry;
    carry = (sum < x) || (result < sum);
    data[i] = BitcastToInt64(result);
  }
  return BigIntUnsafe::Normalize(std::move(ptr));
}
/*!
 * \brief Negate a signed integer.
 * \param x Nonempty canonical operand.
 * \return The canonical negation.
 */
inline BigInt NegateFallback(BigInt::ArrayView x) {
  // -x = ~x + 1. The extra sign word also holds the negation of a minimum value.
  // Unsigned wrap propagates carry without signed minimum-negation UB, even in C++20.
  // BitcastToInt64 then preserves the two's-complement bits in signed storage.
  size_t size = x.size() + 1;
  auto ptr = make_inplace_array_object<BigIntObj, int64_t>(size, size);
  int64_t* data = BigIntUnsafe::GetMutableData(ptr);
  uint64_t extension = IsNegative(x) ? ~uint64_t{0} : 0;
  uint64_t carry = 1;
  for (size_t i = 0; i < size; ++i) {
    uint64_t word = ~(i < x.size() ? static_cast<uint64_t>(x[i]) : extension);
    uint64_t result = word + carry;
    carry = result < word;
    data[i] = BitcastToInt64(result);
  }
  return BigIntUnsafe::Normalize(std::move(ptr));
}
/*!
 * \brief Multiply signed integers using portable full-word arithmetic.
 * \param a Nonempty canonical first operand.
 * \param b Nonempty canonical second operand.
 * \return The canonical product.
 */
inline BigInt MulFallback(BigInt::ArrayView a, BigInt::ArrayView b) {
  size_t size = a.size() + b.size();
  auto ptr = make_inplace_array_object<BigIntObj, int64_t>(size, size);
  int64_t* data = BigIntUnsafe::GetMutableData(ptr);
  size_t na = a.size();
  size_t nb = b.size();
  // Row zero reads this prefix; each row writes its carry before the next row reads it.
  std::fill_n(data, nb, 0);
  for (size_t i = 0; i < na; ++i) {
    uint64_t x = static_cast<uint64_t>(a[i]);
    // Truncate to the low 32 bits, then widen to uint64_t for a full 32x32 product.
    uint64_t x_low = static_cast<uint32_t>(x);
    uint64_t x_high = x >> 32;
    uint64_t carry = 0;
    for (size_t j = 0; j < nb; ++j) {
      uint64_t y = static_cast<uint64_t>(b[j]);
      uint64_t y_low = static_cast<uint32_t>(y);
      uint64_t y_high = y >> 32;
      uint64_t p00 = x_low * y_low;
      uint64_t p01 = x_low * y_high;
      uint64_t p10 = x_high * y_low;
      uint64_t p11 = x_high * y_high;
      // Four 32x32 products form the full unsigned product; middle fits in 34 bits.
      // middle's low 32 bits fill low's upper half; its upper bits carry into high.
      uint64_t middle = (p00 >> 32) + static_cast<uint32_t>(p01) + static_cast<uint32_t>(p10);
      uint64_t low = (middle << 32) | static_cast<uint32_t>(p00);
      uint64_t high = p11 + (p01 >> 32) + (p10 >> 32) + (middle >> 32);
      // Add the existing output word, carrying any overflow into high.
      uint64_t sum_low = low + static_cast<uint64_t>(data[i + j]);
      high += sum_low < low;

      // Add the previous column's carry to the low word.
      sum_low += carry;

      // Comparing against the incoming carry detects overflow from that addition.
      // The total fits in 128 bits; its upper word becomes the next column's carry.
      carry = high + (sum_low < carry);
      data[i + j] = BitcastToInt64(sum_low);
    }
    // Store the remaining high word from this row.
    data[i + nb] = BitcastToInt64(carry);
  }
  // With stored unsigned A/B and 0/1 negative indicators neg_a/neg_b,
  // a = A - neg_a*2^(64*n), b = B - neg_b*2^(64*m). Thus ab is AB minus
  // neg_a*B*2^(64*n) and neg_b*A*2^(64*m); the remaining
  // neg_a*neg_b*2^(64*(n+m)) vanishes modulo the allocated n+m words.
  // Subtract B shifted n words for negative a, and A shifted m for negative b.
  // The exact signed product fits this width; Normalize prunes redundant sign words.
  auto subtract_shifted = [&](BigInt::ArrayView x, size_t shift) {
    uint64_t borrow = 0;
    for (size_t i = 0; i < x.size(); ++i) {
      uint64_t word = static_cast<uint64_t>(x[i]);
      uint64_t rhs = word + borrow;
      uint64_t old = static_cast<uint64_t>(data[shift + i]);
      data[shift + i] = BitcastToInt64(old - rhs);
      borrow = (rhs < word) || (old < rhs);
    }
  };
  if (IsNegative(a)) subtract_shifted(b, na);
  if (IsNegative(b)) subtract_shifted(a, nb);
  return BigIntUnsafe::Normalize(std::move(ptr));
}
/*!
 * \brief Apply sign-extended bitwise AND.
 * \param a Nonempty canonical first operand.
 * \param b Nonempty canonical second operand.
 * \return The canonical bitwise result.
 */
inline BigInt AndFallback(BigInt::ArrayView a, BigInt::ArrayView b) {
  size_t size = std::max(a.size(), b.size());
  auto ptr = make_inplace_array_object<BigIntObj, int64_t>(size, size);
  int64_t* data = BigIntUnsafe::GetMutableData(ptr);
  uint64_t sign_a = IsNegative(a) ? ~uint64_t{0} : 0;
  uint64_t sign_b = IsNegative(b) ? ~uint64_t{0} : 0;
  for (size_t i = 0; i < size; ++i) {
    uint64_t x = i < a.size() ? static_cast<uint64_t>(a[i]) : sign_a;
    uint64_t y = i < b.size() ? static_cast<uint64_t>(b[i]) : sign_b;
    data[i] = BitcastToInt64(x & y);
  }
  return BigIntUnsafe::Normalize(std::move(ptr));
}
/*!
 * \brief Apply sign-extended bitwise OR.
 * \param a Nonempty canonical first operand.
 * \param b Nonempty canonical second operand.
 * \return The canonical bitwise result.
 */
inline BigInt OrFallback(BigInt::ArrayView a, BigInt::ArrayView b) {
  size_t size = std::max(a.size(), b.size());
  auto ptr = make_inplace_array_object<BigIntObj, int64_t>(size, size);
  int64_t* data = BigIntUnsafe::GetMutableData(ptr);
  uint64_t sign_a = IsNegative(a) ? ~uint64_t{0} : 0;
  uint64_t sign_b = IsNegative(b) ? ~uint64_t{0} : 0;
  for (size_t i = 0; i < size; ++i) {
    uint64_t x = i < a.size() ? static_cast<uint64_t>(a[i]) : sign_a;
    uint64_t y = i < b.size() ? static_cast<uint64_t>(b[i]) : sign_b;
    data[i] = BitcastToInt64(x | y);
  }
  return BigIntUnsafe::Normalize(std::move(ptr));
}
/*!
 * \brief Apply sign-extended bitwise XOR.
 * \param a Nonempty canonical first operand.
 * \param b Nonempty canonical second operand.
 * \return The canonical bitwise result.
 */
inline BigInt XorFallback(BigInt::ArrayView a, BigInt::ArrayView b) {
  size_t size = std::max(a.size(), b.size());
  auto ptr = make_inplace_array_object<BigIntObj, int64_t>(size, size);
  int64_t* data = BigIntUnsafe::GetMutableData(ptr);
  uint64_t sign_a = IsNegative(a) ? ~uint64_t{0} : 0;
  uint64_t sign_b = IsNegative(b) ? ~uint64_t{0} : 0;
  for (size_t i = 0; i < size; ++i) {
    uint64_t x = i < a.size() ? static_cast<uint64_t>(a[i]) : sign_a;
    uint64_t y = i < b.size() ? static_cast<uint64_t>(b[i]) : sign_b;
    data[i] = BitcastToInt64(x ^ y);
  }
  return BigIntUnsafe::Normalize(std::move(ptr));
}
/*!
 * \brief Complement all sign-extended bits.
 * \param x Nonempty canonical operand.
 * \return The canonical bitwise complement.
 */
inline BigInt ComplementFallback(BigInt::ArrayView x) {
  auto ptr = make_inplace_array_object<BigIntObj, int64_t>(x.size(), x.size());
  int64_t* data = BigIntUnsafe::GetMutableData(ptr);
  for (size_t i = 0; i < x.size(); ++i) data[i] = ~x[i];
  return BigIntUnsafe::Normalize(std::move(ptr));
}
/*!
 * \brief Convert a nonnegative shift count to size_t.
 * \param count Nonempty canonical integer; negative values raise ValueError.
 * \param out Non-null output, written when the count fits.
 * \return True iff the count fits size_t.
 */
inline bool ShiftCount(BigInt::ArrayView count, size_t* out) {
  if (IsNegative(count)) TVM_FFI_THROW(ValueError) << "Negative BigInt shift count";
  for (size_t i = 1; i < count.size(); ++i) {
    if (count[i] != 0) return false;
  }
  uint64_t low = static_cast<uint64_t>(count[0]);
  if (low > std::numeric_limits<size_t>::max()) return false;
  *out = static_cast<size_t>(low);
  return true;
}
/*!
 * \brief Shift a signed integer left, growing as needed.
 * \param x Nonempty canonical operand.
 * \param count Nonempty canonical count; negative or unrepresentable growth raises an error.
 * \return The canonical shifted integer.
 */
inline BigInt LeftShiftFallback(BigInt::ArrayView x, BigInt::ArrayView count) {
  size_t shift = 0;
  bool fits = ShiftCount(count, &shift);
  if (IsNormalizedZero(x)) return BigInt(0);
  if (!fits) TVM_FFI_THROW(OverflowError) << "BigInt shift count is too large";
  size_t whole = shift / 64;
  size_t size = x.size() + whole + 1;
  unsigned part = static_cast<unsigned>(shift % 64);
  auto ptr = make_inplace_array_object<BigIntObj, int64_t>(size, size);
  int64_t* data = BigIntUnsafe::GetMutableData(ptr);
  std::fill_n(data, whole, 0);
  uint64_t extension = IsNegative(x) ? ~uint64_t{0} : 0;
  for (size_t i = whole; i < size; ++i) {
    size_t j = i - whole;
    uint64_t word = (j < x.size() ? static_cast<uint64_t>(x[j]) : extension) << part;
    if (part && j) word |= static_cast<uint64_t>(x[j - 1]) >> (64 - part);
    data[i] = BitcastToInt64(word);
  }
  return BigIntUnsafe::Normalize(std::move(ptr));
}
/*!
 * \brief Shift a signed integer right with sign extension.
 * \param x Nonempty canonical operand.
 * \param count Nonempty canonical count; negative values raise ValueError.
 * \return The canonical shifted integer; huge counts give zero or -1.
 */
inline BigInt RightShiftFallback(BigInt::ArrayView x, BigInt::ArrayView count) {
  size_t shift = 0;
  bool fits = ShiftCount(count, &shift);
  if (!fits || shift / 64 >= x.size()) return BigInt(IsNegative(x) ? -1 : 0);
  size_t whole = shift / 64;
  size_t size = x.size() - whole;
  unsigned part = static_cast<unsigned>(shift % 64);
  auto ptr = make_inplace_array_object<BigIntObj, int64_t>(size, size);
  int64_t* data = BigIntUnsafe::GetMutableData(ptr);
  uint64_t extension = IsNegative(x) ? ~uint64_t{0} : 0;
  for (size_t i = 0; i < size; ++i) {
    uint64_t word = static_cast<uint64_t>(x[i + whole]) >> part;
    // The last partial word draws its high bits from the sign extension.
    if (part) {
      uint64_t high =
          i + whole + 1 < x.size() ? static_cast<uint64_t>(x[i + whole + 1]) : extension;
      word |= high << (64 - part);
    }
    data[i] = BitcastToInt64(word);
  }
  return BigIntUnsafe::Normalize(std::move(ptr));
}

// Absolute-value access without copying operands: negation carries through the
// low zero words, negates the first nonzero word, then complements higher words.
/*!
 * \brief Locate the first nonzero word.
 * \param x Borrowed words, least-significant first.
 * \return The first nonzero index, or x.size() for zero.
 */
inline size_t FirstNonzero(BigInt::ArrayView x) {
  size_t first = 0;
  while (first < x.size() && x[first] == 0) ++first;
  return first;
}
/*!
 * \brief Read a raw signed word of the whole integer's absolute value without copying.
 * \param x Nonempty signed-word view.
 * \param first_nonzero FirstNonzero(x).
 * \param i Word index; indices beyond the view read as zero.
 * \return Magnitude bits as int64_t; the raw word itself may be negative.
 */
TVM_FFI_INLINE int64_t AbsWord(BigInt::ArrayView x, size_t first_nonzero, size_t i) {
  if (i >= x.size()) return 0;
  int64_t word = x[i];
  if (!IsNegative(x)) return word;
  // Negation maps [0,0,w,v] to [0,0,-w,~v] as modulo-2^64 word patterns:
  // the +1 carries through the original low zeros and stops at the first nonzero word.
  if (i < first_nonzero) return 0;
  if (i == first_nonzero) {
    constexpr int64_t kMin = std::numeric_limits<int64_t>::min();
    return word == kMin ? kMin : -word;
  }
  return ~word;
}
/*!
 * \brief Negate construction words modulo their fixed width.
 * \param data Mutable two's-complement words.
 * \param size Number of words.
 */
inline void NegateInplace(int64_t* data, size_t size) {
  uint64_t carry = 1;
  for (size_t i = 0; i < size; ++i) {
    uint64_t word = ~static_cast<uint64_t>(data[i]);
    uint64_t result = word + carry;
    carry = result < word;
    data[i] = BitcastToInt64(result);
  }
}
/*!
 * \brief Compute truncating quotient and remainder with an int64 divisor.
 * \param a Nonempty canonical dividend.
 * \param b Divisor; zero raises ZeroDivisionError.
 * \return Canonical quotient/remainder satisfying a = q*b + r; r is zero or has a's sign.
 */
inline std::pair<BigInt, BigInt> DivRemFallback(BigInt::ArrayView a, int64_t b) {
  if (b == 0) TVM_FFI_THROW(ZeroDivisionError) << "Division by zero";
  uint64_t divisor = b < 0 ? uint64_t{0} - static_cast<uint64_t>(b) : static_cast<uint64_t>(b);
  size_t first_a = FirstNonzero(a);
  size_t nq = a.size() + 1;
  auto quotient = make_inplace_array_object<BigIntObj, int64_t>(nq, nq);
  int64_t* q = BigIntUnsafe::GetMutableData(quotient);
  q[a.size()] = 0;  // The lower words are assigned by division; this is the sign guard.
  unsigned shift = 0;
  uint64_t normalized_divisor = divisor;
  if (divisor > std::numeric_limits<uint32_t>::max()) {
    while ((normalized_divisor >> 63) == 0) {
      normalized_divisor <<= 1;
      ++shift;
    }
  }
  uint64_t divisor_high = normalized_divisor >> 32;
  uint64_t divisor_low = static_cast<uint32_t>(normalized_divisor);
  uint64_t remainder = 0;
  for (size_t i = a.size(); i != 0; --i) {
    uint64_t word = static_cast<uint64_t>(AbsWord(a, first_a, i - 1));
    uint64_t quotient_word = 0;
    if (divisor <= std::numeric_limits<uint32_t>::max()) {
      // remainder < divisor < 2^32, so each brought-down half fits uint64_t
      // and yields a quotient half below 2^32 while restoring the remainder bound.
      uint64_t high = (remainder << 32) | (word >> 32);
      uint64_t quotient_high = high / divisor;
      remainder = high % divisor;
      uint64_t low = (remainder << 32) | static_cast<uint32_t>(word);
      quotient_word = (quotient_high << 32) | (low / divisor);
      remainder = low % divisor;
    } else {
      // Normalize the carried 128-bit numerator without ever shifting by 64.
      uint64_t high = shift ? (remainder << shift) | (word >> (64 - shift)) : remainder;
      uint64_t low = word << shift;
      // Requires upper < normalized_divisor; returns a radix-2^32 quotient digit
      // and a remainder < normalized_divisor.
      auto digit = [&](uint64_t upper, uint64_t next) {
        constexpr uint64_t kBase = uint64_t{1} << 32;
        uint64_t estimate = upper / divisor_high;
        uint64_t residual = upper % divisor_high;
        // The normalized high digit bounds overestimation by two. Check the
        // base before multiplying, and stop before a residual shift can overflow.
        while (estimate >= kBase || estimate * divisor_low > (residual << 32) + next) {
          --estimate;
          residual += divisor_high;
          if (residual >= kBase) break;
        }
        // Wrapped unsigned subtraction is exact: the true remainder is < normalized divisor.
        return std::make_pair(estimate, (upper << 32) + next - estimate * normalized_divisor);
      };
      // Divide the normalized numerator into two radix digits, high then low.
      auto upper = digit(high, low >> 32);
      auto lower = digit(upper.second, static_cast<uint32_t>(low));
      quotient_word = (upper.first << 32) | lower.first;
      remainder = lower.second >> shift;
    }
    q[i - 1] = BitcastToInt64(quotient_word);
  }
  if (IsNegative(a) != (b < 0)) NegateInplace(q, nq);
  // remainder < |b| <= 2^63 makes the conversion and signed negation safe.
  int64_t signed_remainder = static_cast<int64_t>(remainder);
  if (IsNegative(a)) signed_remainder = -signed_remainder;
  return {BigIntUnsafe::Normalize(std::move(quotient)), BigInt(signed_remainder)};
}
/*!
 * \brief Compute truncating quotient and remainder in one division pass.
 * \param a Nonempty canonical dividend.
 * \param b Nonempty canonical divisor; zero raises ZeroDivisionError.
 * \return Canonical quotient/remainder satisfying a = q*b + r; r is zero or has a's sign.
 */
inline std::pair<BigInt, BigInt> DivRemFallback(BigInt::ArrayView a, BigInt::ArrayView b) {
  if (b.size() == 1) return DivRemFallback(a, b[0]);
  size_t first_a = FirstNonzero(a);
  size_t first_b = FirstNonzero(b);
  // Count magnitude digits, dropping the signed representation's high zero guards.
  auto digit_count = [](BigInt::ArrayView x, size_t first) {
    size_t words = x.size();
    while (words && AbsWord(x, first, words - 1) == 0) --words;
    if (!words) return size_t{0};
    uint64_t top = static_cast<uint64_t>(AbsWord(x, first, words - 1));
    return 2 * words - ((top >> 32) == 0 ? 1 : 0);
  };
  size_t m = digit_count(a, first_a);
  size_t n = digit_count(b, first_b);
  if (m < n) {
    auto remainder = make_inplace_array_object<BigIntObj, int64_t>(a.size(), a.size());
    std::copy_n(&a[0], a.size(), BigIntUnsafe::GetMutableData(remainder));
    return {BigInt(0), BigIntUnsafe::Normalize(std::move(remainder))};
  }

  // Algorithm D uses radix b = 2^32 (kBase), keeping products and estimates in uint64_t.
  // Example: A[0] A[1] A[2] A[3] / B[0] B[1] B[2], with digits most-significant first.
  // Actual u/v digits are least-significant first.
  // The scalar dispatch leaves at least two magnitude digits in the divisor.
  constexpr uint64_t kBase = uint64_t{1} << 32;
  uint64_t top_word = static_cast<uint64_t>(AbsWord(b, first_b, (n - 1) / 2));
  uint32_t top_digit = static_cast<uint32_t>(top_word >> (((n - 1) % 2) * 32));
  unsigned shift = 0;
  // Normalize both inputs until B[0] >= b/2, bounding the quotient estimate error.
  while (top_digit < (uint32_t{1} << 31)) {
    top_digit <<= 1;
    ++shift;
  }
  size_t nq = (m - n) / 2 + 2;
  size_t nr = (n + 1) / 2 + 1;
  constexpr size_t kMaxWords =
      (std::numeric_limits<size_t>::max() - sizeof(BigIntObj) - (alignof(BigIntObj) - 1)) /
      sizeof(int64_t);
  if (TVM_FFI_PREDICT_FALSE(nq > kMaxWords || n > kMaxWords - nq || m >= kMaxWords - nq - n)) {
    TVM_FFI_THROW(OverflowError) << "BigInt division workspace is too large";
  }
  // Scratch follows the logical quotient; normalization excludes it but retains its capacity.
  // int64_t slots hold unsigned radix digits without introducing another storage alias.
  auto quotient = make_inplace_array_object<BigIntObj, int64_t>(nq + m + 1 + n, nq);
  auto remainder = make_inplace_array_object<BigIntObj, int64_t>(nr, nr);
  int64_t* q = BigIntUnsafe::GetMutableData(quotient);
  int64_t* r = BigIntUnsafe::GetMutableData(remainder);
  int64_t* u = q + nq;
  int64_t* v = u + m + 1;
  // Write every scratch digit before use; the dividend also keeps its shifted high carry.
  auto normalize = [&](BigInt::ArrayView x, size_t first, size_t digits, int64_t* out) {
    uint64_t carry = 0;
    for (size_t i = 0; i < digits; ++i) {
      uint64_t word = static_cast<uint64_t>(AbsWord(x, first, i / 2));
      uint64_t part = static_cast<uint32_t>(word >> ((i % 2) * 32));
      uint64_t value = (part << shift) | carry;
      out[i] = static_cast<uint32_t>(value);
      carry = value >> 32;
    }
    return static_cast<uint32_t>(carry);
  };
  u[m] = normalize(a, first_a, m, u);
  normalize(b, first_b, n, v);

  q[nq - 1] = 0;
  r[nr - 1] = 0;
  uint64_t quotient_word = 0;
  for (size_t position = m - n + 1; position != 0; --position) {
    size_t j = position - 1;
    uint64_t estimate;
    uint64_t residual;
    // - Step 1: Estimate q from the leading digits.
    // A[0]=u[j+n], A[1]=u[j+n-1], B[0]=v[n-1]; q=estimate and r=residual below.
    // Estimate q from (A[0]*b+A[1])/B[0], with r=A[0]*b+A[1]-q*B[0].
    // A[0] is at most B[0]; clamp equality so q remains a radix digit.
    if (u[j + n] == v[n - 1]) {
      estimate = kBase - 1;
      residual = static_cast<uint64_t>(u[j + n - 1]) + v[n - 1];
    } else {
      uint64_t numerator = (static_cast<uint64_t>(u[j + n]) << 32) | u[j + n - 1];
      estimate = numerator / v[n - 1];
      residual = numerator % v[n - 1];
    }
    // - Step 2: Refine q using B[1].
    // With A[2]=u[j+n-2] and B[1]=v[n-2], refine until q*B[1] <= r*b+A[2].
    // Decrement q and add B[0] to r at most twice; r < b guards the shift.
    // Refinement changes only q and r; it does not modify the dividend window A[...].
    while (residual < kBase && estimate * v[n - 2] > (residual << 32) + u[j + n - 2]) {
      --estimate;
      residual += v[n - 1];
    }
    // - Step 3: Subtract the full q * B[...] from A[...].
    // Check R=A[...] - q * B[...], including the lower B[2] in the example:
    // R=(r*b+A[2]-q*B[1])*b+A[3]-q*B[2]. A valid digit leaves 0 <= R < divisor.
    // The refined estimate is correct or one high; subtract the complete window to decide.
    uint64_t borrow = 0;
    for (size_t i = 0; i < n; ++i) {
      // A radix-digit product plus the incoming borrow fits uint64_t.
      uint64_t product = estimate * v[i] + borrow;
      uint32_t low = static_cast<uint32_t>(product);
      uint32_t old = u[j + i];
      u[j + i] = static_cast<uint32_t>(old - low);
      borrow = (product >> 32) + (old < low);
    }
    uint32_t old = u[j + n];
    // Only a borrow out of the full window means R < 0; digit borrows merely propagate.
    bool negative = old < borrow;
    u[j + n] = static_cast<uint32_t>(static_cast<uint64_t>(old) - borrow);
    if (negative) {
      // - Step 4: If the result is negative, decrement q and add back all of B[...].
      // This restores 0 <= R < divisor.
      // Add back all of the aligned B[...]:
      // (A[...] - q * B[...]) + B[...] = A[...] - (q - 1) * B[...].
      --estimate;
      uint64_t carry = 0;
      for (size_t i = 0; i < n; ++i) {
        uint64_t sum = static_cast<uint64_t>(u[j + i]) + v[i] + carry;
        u[j + i] = static_cast<uint32_t>(sum);
        carry = sum >> 32;
      }
      u[j + n] = static_cast<uint32_t>(static_cast<uint64_t>(u[j + n]) + carry);
    }
    // Pack descending quotient digits into complete words, without reading raw output storage.
    if (j % 2) {
      quotient_word = estimate << 32;
    } else {
      q[j / 2] = BitcastToInt64(quotient_word | estimate);
    }
  }
  // Extract the remainder before Normalize can demote and release the quotient-owned scratch.
  for (size_t i = 0; i < nr - 1; ++i) {
    size_t digit = 2 * i;
    uint64_t word = u[digit];
    if (digit + 1 < n) word |= static_cast<uint64_t>(u[digit + 1]) << 32;
    word >>= shift;
    // Bring down bits from the next word; shift zero never evaluates a shift by 64.
    if (shift && digit + 2 < n) word |= static_cast<uint64_t>(u[digit + 2]) << (64 - shift);
    r[i] = BitcastToInt64(word);
  }
  // Truncation gives the quotient the operand-sign XOR and a nonzero remainder the dividend's sign.
  if (IsNegative(a) != IsNegative(b)) NegateInplace(q, nq);
  if (IsNegative(a)) NegateInplace(r, nr);
  return {BigIntUnsafe::Normalize(std::move(quotient)),
          BigIntUnsafe::Normalize(std::move(remainder))};
}
/*!
 * \brief Divide with rounding toward zero.
 * \param a Nonempty canonical dividend.
 * \param b Nonempty canonical divisor; zero raises ZeroDivisionError.
 * \return The canonical truncating quotient.
 */
inline BigInt TruncDivFallback(BigInt::ArrayView a, BigInt::ArrayView b) {
  return DivRemFallback(a, b).first;
}
/*!
 * \brief Take the remainder of truncating division.
 * \param a Nonempty canonical dividend.
 * \param b Nonempty canonical divisor; zero raises ZeroDivisionError.
 * \return The canonical remainder, zero or with the dividend's sign.
 */
inline BigInt TruncModFallback(BigInt::ArrayView a, BigInt::ArrayView b) {
  if (b.size() == 1) {
    if (b[0] == 0) TVM_FFI_THROW(ZeroDivisionError) << "Division by zero";
    uint64_t divisor =
        b[0] < 0 ? uint64_t{0} - static_cast<uint64_t>(b[0]) : static_cast<uint64_t>(b[0]);
    if (divisor <= std::numeric_limits<uint32_t>::max()) {
      size_t first = FirstNonzero(a);
      uint64_t remainder = 0;
      // remainder < divisor < 2^32 lets each brought-down half fit uint64_t.
      // The result fits inline, so small-divisor modulo needs no quotient or allocation.
      for (size_t i = a.size(); i != 0; --i) {
        uint64_t word = static_cast<uint64_t>(AbsWord(a, first, i - 1));
        remainder = ((remainder << 32) | (word >> 32)) % divisor;
        remainder = ((remainder << 32) | static_cast<uint32_t>(word)) % divisor;
      }
      int64_t result = static_cast<int64_t>(remainder);
      return BigInt(IsNegative(a) ? -result : result);
    }
  }
  return DivRemFallback(a, b).second;
}
/*!
 * \brief Divide with rounding toward negative infinity.
 * \param a Nonempty canonical dividend.
 * \param b Nonempty canonical divisor; zero raises ZeroDivisionError.
 * \return The canonical floor quotient.
 */
inline BigInt FloorDivFallback(BigInt::ArrayView a, BigInt::ArrayView b) {
  auto qr = DivRemFallback(a, b);
  if (qr.second && IsNegative(a) != IsNegative(b)) {
    // With a nonzero remainder, (q-1)*b + (r+b) preserves a while rounding down.
    const int64_t one = 1;
    qr.first = SubFallback(BigIntUnsafe::GetArrayView(qr.first), BigIntUnsafe::GetArrayView(one));
  }
  return std::move(qr.first);
}
/*!
 * \brief Take the remainder of floor division.
 * \param a Nonempty canonical dividend.
 * \param b Nonempty canonical divisor; zero raises ZeroDivisionError.
 * \return The canonical remainder, zero or with the divisor's sign.
 */
inline BigInt FloorModFallback(BigInt::ArrayView a, BigInt::ArrayView b) {
  BigInt remainder = TruncModFallback(a, b);
  if (remainder && IsNegative(a) != IsNegative(b)) {
    // For a scalar divisor, |r| < |b| and opposite signs make r+b safe and inline.
    if (b.size() == 1) return BigInt(BigIntUnsafe::GetInt64(remainder) + b[0]);
    // The remainder paired with the floor quotient q-1 is r+b.
    remainder = AddFallback(BigIntUnsafe::GetArrayView(remainder), b);
  }
  return remainder;
}
/*!
 * \brief Select the smaller signed integer.
 * \param a Nonempty canonical first operand.
 * \param b Nonempty canonical second operand.
 * \return The canonical minimum.
 */
inline BigInt MinFallback(BigInt::ArrayView a, BigInt::ArrayView b) {
  auto value = CompareFallback(a, b) <= 0 ? a : b;
  auto ptr = make_inplace_array_object<BigIntObj, int64_t>(value.size(), value.size());
  std::copy_n(&value[0], value.size(), BigIntUnsafe::GetMutableData(ptr));
  return BigIntUnsafe::Normalize(std::move(ptr));
}
/*!
 * \brief Select the larger signed integer.
 * \param a Nonempty canonical first operand.
 * \param b Nonempty canonical second operand.
 * \return The canonical maximum.
 */
inline BigInt MaxFallback(BigInt::ArrayView a, BigInt::ArrayView b) {
  auto value = CompareFallback(a, b) >= 0 ? a : b;
  auto ptr = make_inplace_array_object<BigIntObj, int64_t>(value.size(), value.size());
  std::copy_n(&value[0], value.size(), BigIntUnsafe::GetMutableData(ptr));
  return BigIntUnsafe::Normalize(std::move(ptr));
}
/*!
 * \brief Convert a finite double by truncating toward zero.
 * \param value Input; NaN raises ValueError and infinity raises OverflowError.
 * \return The canonical integer.
 */
inline BigInt FromDoubleFallback(double value) {
  if (std::isnan(value)) TVM_FFI_THROW(ValueError) << "Cannot convert NaN to BigInt";
  if (!std::isfinite(value)) TVM_FFI_THROW(OverflowError) << "Cannot convert infinity to BigInt";
  int exponent = 0;
  double fraction = std::frexp(std::fabs(value), &exponent);
  if (exponent <= 0) return BigInt(0);
  uint64_t significand = static_cast<uint64_t>(std::ldexp(fraction, 53));
  if (exponent <= 53) {
    return BigInt((value < 0 ? -1 : 1) * static_cast<int64_t>(significand >> (53 - exponent)));
  }
  size_t shift = static_cast<size_t>(exponent - 53);
  size_t size = (static_cast<size_t>(exponent) + 63) / 64 + 1;
  auto ptr = make_inplace_array_object<BigIntObj, int64_t>(size, size);
  int64_t* data = BigIntUnsafe::GetMutableData(ptr);
  size_t whole = shift / 64;
  std::fill_n(data, whole, 0);
  data[whole] = BitcastToInt64(significand << (shift % 64));
  size_t written = whole + 1;
  if (shift % 64) data[written++] = BitcastToInt64(significand >> (64 - shift % 64));
  std::fill(data + written, data + size, 0);  // Unwritten high words include the sign guard.
  if (value < 0) NegateInplace(data, size);
  return BigIntUnsafe::Normalize(std::move(ptr));
}
/*!
 * \brief Convert an integer with nearest, ties-to-even rounding.
 * \param value Canonical integer; an unrepresentable finite result raises OverflowError.
 * \return The rounded finite double.
 */
inline double ToDoubleFallback(const BigInt& value) {
  auto x = BigIntUnsafe::GetArrayView(value);
  size_t first = FirstNonzero(x);
  size_t size = x.size();
  while (size && AbsWord(x, first, size - 1) == 0) --size;
  if (!size) return 0;
  if (size > 16) TVM_FFI_THROW(OverflowError) << "BigInt does not fit finite double";
  unsigned high_bits = 0;
  for (uint64_t high = static_cast<uint64_t>(AbsWord(x, first, size - 1)); high; high >>= 1) {
    ++high_bits;
  }
  size_t bits = (size - 1) * 64 + high_bits;
  size_t shift = bits > 53 ? bits - 53 : 0;
  auto bit = [&](size_t i) {
    return (static_cast<uint64_t>(AbsWord(x, first, i / 64)) >> (i % 64)) & 1;
  };
  uint64_t significand = 0;
  for (size_t i = bits; i > shift; --i) significand = (significand << 1) | bit(i - 1);
  if (shift) {
    bool sticky = false;
    for (size_t i = 0; i + 1 < shift; ++i) sticky |= bit(i) != 0;
    if (bit(shift - 1) && (sticky || (significand & 1))) ++significand;
  }
  if (bits == 1024 && significand == (uint64_t{1} << 53)) {
    TVM_FFI_THROW(OverflowError) << "BigInt does not fit finite double";
  }
  double result = std::ldexp(static_cast<double>(significand), static_cast<int>(shift));
  return IsNegative(x) ? -result : result;
}
/*!
 * \brief Convert an integer to signed decimal text.
 * \param value Canonical integer.
 * \return The decimal representation.
 */
inline std::string ToStringFallback(const BigInt& value) {
  if (BigIntUnsafe::CheckInt64(value)) return std::to_string(BigIntUnsafe::GetInt64(value));
  auto x = BigIntUnsafe::GetArrayView(value);
  size_t first = FirstNonzero(x);
  size_t size = x.size();
  auto workspace = make_inplace_array_object<BigIntObj, int64_t>(size, size);
  int64_t* data = BigIntUnsafe::GetMutableData(workspace);
  for (size_t i = 0; i < size; ++i) data[i] = AbsWord(x, first, i);
  std::string result;
  do {
    uint64_t remainder = 0;
    // Divide each full word by ten, high half first. The incoming remainder
    // is below ten, so each remainder-plus-half dividend fits in uint64_t.
    for (size_t i = size; i != 0; --i) {
      uint64_t word = static_cast<uint64_t>(data[i - 1]);
      uint64_t current = (remainder << 32) | (word >> 32);
      uint64_t high = current / 10;
      remainder = current % 10;
      current = (remainder << 32) | static_cast<uint32_t>(word);
      data[i - 1] = BitcastToInt64((high << 32) | (current / 10));
      remainder = current % 10;
    }
    result.push_back(static_cast<char>('0' + remainder));
    while (size && data[size - 1] == 0) --size;
  } while (size);
  if (IsNegative(x)) result.push_back('-');
  std::reverse(result.begin(), result.end());
  return result;
}
}  // namespace int_ops
}  // namespace detail

template <typename Int, std::enable_if_t<(std::is_integral_v<Int> && sizeof(Int) <= 8), int>>
TVM_FFI_INLINE BigInt::BigInt(Int value) : BigInt() {
  if constexpr (std::is_unsigned_v<Int> && sizeof(Int) == 8) {
    uint64_t bits = static_cast<uint64_t>(value);
    constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
    if (TVM_FFI_PREDICT_FALSE(bits > static_cast<uint64_t>(kMax))) {
      auto ptr = make_inplace_array_object<detail::BigIntObj, int64_t>(2, 2);
      detail::BigIntUnsafe::GetMutableData(ptr)[0] = detail::int_ops::BitcastToInt64(bits);
      detail::BigIntUnsafe::GetMutableData(ptr)[1] = 0;
      *this = BigInt(std::move(ptr));
      return;
    }
  }
  data_.v_int64 = static_cast<int64_t>(value);
}

template <typename Int, std::enable_if_t<(std::is_integral_v<Int> && sizeof(Int) <= 8), int>>
TVM_FFI_INLINE std::optional<Int> BigInt::as() const {
  if (data_.type_index == TypeIndex::kTVMFFIInt) {
    if constexpr (std::is_same_v<Int, int64_t>) {
      return data_.v_int64;
    } else {
      int64_t value = data_.v_int64;
      if constexpr (std::is_signed_v<Int>) {
        if (value < std::numeric_limits<Int>::min() || value > std::numeric_limits<Int>::max()) {
          return std::nullopt;
        }
      } else {
        if (value < 0 || static_cast<uint64_t>(value) > std::numeric_limits<Int>::max()) {
          return std::nullopt;
        }
      }
      return static_cast<Int>(value);
    }
  }
  if constexpr (std::is_unsigned_v<Int> && sizeof(Int) == 8) {
    auto repr = detail::BigIntUnsafe::GetArrayView(*this);
    if (repr.size() == 2 && repr[1] == 0) return static_cast<Int>(repr[0]);
  }
  return std::nullopt;
}

TVM_FFI_INLINE BigInt::BigInt(double value) : BigInt() {
  // The upper bound is exclusive: INT64_MAX rounds up to 2^63 as a double.
  constexpr double lower = static_cast<double>(std::numeric_limits<int64_t>::min());
  if (TVM_FFI_PREDICT_TRUE(value >= lower && value < -lower)) {
    data_.v_int64 = static_cast<int64_t>(value);
    return;
  }
  *this = detail::int_ops::FromDoubleFallback(value);
}

TVM_FFI_INLINE BigInt::operator double() const {
  if (TVM_FFI_PREDICT_TRUE(data_.type_index == TypeIndex::kTVMFFIInt)) {
    return static_cast<double>(data_.v_int64);
  }
  return detail::int_ops::ToDoubleFallback(*this);
}

TVM_FFI_INLINE uint64_t BigInt::hash() const {
  if (data_.type_index == TypeIndex::kTVMFFIInt) return static_cast<uint64_t>(data_.v_int64);
  auto repr = detail::BigIntUnsafe::GetArrayView(*this);
  return detail::StableHashBytes(&repr[0], repr.size() * sizeof(int64_t));
}

/// \cond Doxygen_Suppress
template <>
inline constexpr bool use_default_type_traits_v<BigInt> = false;
template <>
struct TypeTraits<BigInt> : public TypeTraitsBase {
  static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIAny;
  static void CopyToAnyView(const BigInt& src, TVMFFIAny* result) { *result = src.data_; }

  static void MoveToAny(BigInt src, TVMFFIAny* result) {
    *result = src.data_;
    src.Reset();
  }

  static bool CheckAnyStrict(const TVMFFIAny* src) {
    return src->type_index == TypeIndex::kTVMFFIInt || src->type_index == TypeIndex::kTVMFFIBigInt;
  }

  static BigInt CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
    BigInt result;
    result.data_ = *src;
    if (TVM_FFI_PREDICT_FALSE(result.data_.type_index != TypeIndex::kTVMFFIInt)) {
      detail::ObjectUnsafe::IncRefObjectHandle(result.data_.v_obj);
    }
    return result;
  }

  static BigInt MoveFromAnyAfterCheck(TVMFFIAny* src) {
    BigInt result;
    result.data_ = *src;
    src->type_index = TypeIndex::kTVMFFINone;
    src->zero_padding = 0;
    src->v_int64 = 0;
    return result;
  }

  static std::optional<BigInt> TryCastFromAnyView(const TVMFFIAny* src) {
    if (CheckAnyStrict(src)) return CopyFromAnyViewAfterCheck(src);
    if (src->type_index == TypeIndex::kTVMFFIBool) return BigInt(src->v_int64);
    return std::nullopt;
  }

  static std::string TypeStr() { return "BigInt"; }

  static std::string TypeSchema() { return R"({"type":"ffi.BigInt"})"; }
};
// Keep the same dispatch in all three operand variants; primitive arguments never create a cell.
// clang-format off
#define TVM_FFI_DEFINE_BIGINT_OP_OVERLOAD(Return, Name, Body)                                     \
  TVM_FFI_INLINE Return Name(const BigInt& a, const BigInt& b)                                    \
  Body                                                                                            \
                                                                                                  \
  TVM_FFI_INLINE Return Name(const BigInt& a, int64_t b)                                          \
  Body                                                                                            \
                                                                                                  \
  TVM_FFI_INLINE Return Name(int64_t a, const BigInt& b)                                          \
  Body

// clang-format on

#define TVM_FFI_DEFINE_BIGINT_OP_CHECKED(Name, Fallback, Overflow)               \
  TVM_FFI_DEFINE_BIGINT_OP_OVERLOAD(BigInt, Name, {                              \
    if (TVM_FFI_PREDICT_TRUE(detail::BigIntUnsafe::CheckInt64(a) &&              \
                             detail::BigIntUnsafe::CheckInt64(b))) {             \
      int64_t x = detail::BigIntUnsafe::GetInt64(a);                             \
      int64_t y = detail::BigIntUnsafe::GetInt64(b);                             \
      int64_t result;                                                            \
      if (TVM_FFI_PREDICT_FALSE(Overflow)) {                                     \
        return detail::int_ops::Fallback(detail::BigIntUnsafe::GetArrayView(a),  \
                                         detail::BigIntUnsafe::GetArrayView(b)); \
      }                                                                          \
      return BigInt(result);                                                     \
    }                                                                            \
    return detail::int_ops::Fallback(detail::BigIntUnsafe::GetArrayView(a),      \
                                     detail::BigIntUnsafe::GetArrayView(b));     \
  })

#define TVM_FFI_DEFINE_BIGINT_OP_NO_OVERFLOW(Return, Name, Fallback, Fast)   \
  TVM_FFI_DEFINE_BIGINT_OP_OVERLOAD(Return, Name, {                          \
    if (TVM_FFI_PREDICT_TRUE(detail::BigIntUnsafe::CheckInt64(a) &&          \
                             detail::BigIntUnsafe::CheckInt64(b))) {         \
      int64_t x = detail::BigIntUnsafe::GetInt64(a);                         \
      int64_t y = detail::BigIntUnsafe::GetInt64(b);                         \
      return Fast;                                                           \
    }                                                                        \
    return detail::int_ops::Fallback(detail::BigIntUnsafe::GetArrayView(a),  \
                                     detail::BigIntUnsafe::GetArrayView(b)); \
  })

// Op is an operator token in a function name and expression; parentheses are invalid here.
// NOLINTBEGIN(bugprone-macro-parentheses)
#define TVM_FFI_DEFINE_BIGINT_OP_COMPARE(Op)                                             \
  TVM_FFI_DEFINE_BIGINT_OP_OVERLOAD(bool, operator Op, {                                 \
    if (TVM_FFI_PREDICT_TRUE(detail::BigIntUnsafe::CheckInt64(a) &&                      \
                             detail::BigIntUnsafe::CheckInt64(b))) {                     \
      return detail::BigIntUnsafe::GetInt64(a) Op detail::BigIntUnsafe::GetInt64(b);     \
    }                                                                                    \
    return detail::int_ops::CompareFallback(detail::BigIntUnsafe::GetArrayView(a),       \
                                            detail::BigIntUnsafe::GetArrayView(b)) Op 0; \
  })

// NOLINTEND(bugprone-macro-parentheses)

// ---- BigInt operator overloading: BigInt/BigInt, BigInt/int64_t, int64_t/BigInt ----
// Add signed integers, promoting when the sum exceeds int64_t.
TVM_FFI_DEFINE_BIGINT_OP_CHECKED(operator+, AddFallback,
                                 detail::int_ops::AddOverflow(x, y, &result))
// Subtract signed integers, promoting when the difference exceeds int64_t.
TVM_FFI_DEFINE_BIGINT_OP_CHECKED(operator-, SubFallback,
                                 detail::int_ops::SubOverflow(x, y, &result))
// Multiply signed integers, promoting when the product exceeds int64_t.
TVM_FFI_DEFINE_BIGINT_OP_CHECKED(operator*, MulFallback,
                                 detail::int_ops::MulOverflow(x, y, &result))
// Divide toward zero; zero divisors raise ZeroDivisionError and int64_t overflow promotes.
TVM_FFI_DEFINE_BIGINT_OP_CHECKED(truncdiv, TruncDivFallback,
                                 detail::int_ops::TruncDivOverflow(x, y, &result))
// Truncating remainder is zero or has the dividend sign; zero divisors raise ZeroDivisionError.
TVM_FFI_DEFINE_BIGINT_OP_CHECKED(truncmod, TruncModFallback,
                                 detail::int_ops::TruncModOverflow(x, y, &result))
// Divide toward negative infinity; zero divisors raise ZeroDivisionError and overflow promotes.
TVM_FFI_DEFINE_BIGINT_OP_CHECKED(floordiv, FloorDivFallback,
                                 detail::int_ops::FloorDivOverflow(x, y, &result))
// Floor remainder is zero or has the divisor sign; zero divisors raise ZeroDivisionError.
TVM_FFI_DEFINE_BIGINT_OP_CHECKED(floormod, FloorModFallback,
                                 detail::int_ops::FloorModOverflow(x, y, &result))
// Divide toward zero; zero divisors raise ZeroDivisionError and int64_t overflow promotes.
TVM_FFI_DEFINE_BIGINT_OP_CHECKED(operator/, TruncDivFallback,
                                 detail::int_ops::TruncDivOverflow(x, y, &result))
// Truncating remainder is zero or has the dividend sign; zero divisors raise ZeroDivisionError.
TVM_FFI_DEFINE_BIGINT_OP_CHECKED(operator%, TruncModFallback,
                                 detail::int_ops::TruncModOverflow(x, y, &result))
// Left shift promotes as needed; negative counts raise ValueError, excessive growth OverflowError.
TVM_FFI_DEFINE_BIGINT_OP_CHECKED(operator<<, LeftShiftFallback,
                                 detail::int_ops::LeftShiftOverflow(x, y, &result))
// Bitwise AND uses signed two's-complement sign extension.
TVM_FFI_DEFINE_BIGINT_OP_NO_OVERFLOW(BigInt, operator&, AndFallback, BigInt(x& y))
// Bitwise OR uses signed two's-complement sign extension.
TVM_FFI_DEFINE_BIGINT_OP_NO_OVERFLOW(BigInt, operator|, OrFallback, BigInt(x | y))
// Bitwise XOR uses signed two's-complement sign extension.
TVM_FFI_DEFINE_BIGINT_OP_NO_OVERFLOW(BigInt, operator^, XorFallback, BigInt(x ^ y))
// Return the smaller signed integer.
TVM_FFI_DEFINE_BIGINT_OP_NO_OVERFLOW(BigInt, min, MinFallback, BigInt(x <= y ? x : y))
// Return the larger signed integer.
TVM_FFI_DEFINE_BIGINT_OP_NO_OVERFLOW(BigInt, max, MaxFallback, BigInt(x >= y ? x : y))
// Sign-extending right shift; negative counts raise ValueError, large counts return 0/-1.
TVM_FFI_DEFINE_BIGINT_OP_NO_OVERFLOW(BigInt, operator>>, RightShiftFallback,
                                     BigInt(detail::int_ops::RightShift(x, y)))
// Test signed integer equality.
TVM_FFI_DEFINE_BIGINT_OP_NO_OVERFLOW(bool, operator==, EqualFallback, x == y)
// Test signed integer inequality.
TVM_FFI_DEFINE_BIGINT_OP_OVERLOAD(bool, operator!=, { return !(a == b); })
// Test whether the first signed integer is smaller.
TVM_FFI_DEFINE_BIGINT_OP_COMPARE(<)
// Test whether the first signed integer is smaller or equal.
TVM_FFI_DEFINE_BIGINT_OP_COMPARE(<=)
// Test whether the first signed integer is larger.
TVM_FFI_DEFINE_BIGINT_OP_COMPARE(>)
// Test whether the first signed integer is larger or equal.
TVM_FFI_DEFINE_BIGINT_OP_COMPARE(>=)

#undef TVM_FFI_DEFINE_BIGINT_OP_CHECKED
#undef TVM_FFI_DEFINE_BIGINT_OP_NO_OVERFLOW
#undef TVM_FFI_DEFINE_BIGINT_OP_COMPARE
#undef TVM_FFI_DEFINE_BIGINT_OP_OVERLOAD
/// \endcond

TVM_FFI_INLINE BigInt operator+(const BigInt& value) { return value; }

TVM_FFI_INLINE BigInt operator-(const BigInt& value) {
  if (TVM_FFI_PREDICT_TRUE(detail::BigIntUnsafe::CheckInt64(value))) {
    int64_t x = detail::BigIntUnsafe::GetInt64(value);
    constexpr int64_t kMin = std::numeric_limits<int64_t>::min();
    if (TVM_FFI_PREDICT_TRUE(x != kMin)) return BigInt(-x);
  }
  return detail::int_ops::NegateFallback(detail::BigIntUnsafe::GetArrayView(value));
}

TVM_FFI_INLINE BigInt operator~(const BigInt& value) {
  if (TVM_FFI_PREDICT_TRUE(detail::BigIntUnsafe::CheckInt64(value))) {
    return BigInt(~detail::BigIntUnsafe::GetInt64(value));
  }
  return detail::int_ops::ComplementFallback(detail::BigIntUnsafe::GetArrayView(value));
}

/// \cond Doxygen_Suppress
TVM_FFI_INLINE BigInt& BigInt::operator+=(const BigInt& rhs) { return *this = *this + rhs; }

TVM_FFI_INLINE BigInt& BigInt::operator-=(const BigInt& rhs) { return *this = *this - rhs; }

TVM_FFI_INLINE BigInt& BigInt::operator*=(const BigInt& rhs) { return *this = *this * rhs; }

TVM_FFI_INLINE BigInt& BigInt::operator/=(const BigInt& rhs) { return *this = *this / rhs; }

TVM_FFI_INLINE BigInt& BigInt::operator%=(const BigInt& rhs) { return *this = *this % rhs; }

TVM_FFI_INLINE BigInt& BigInt::operator&=(const BigInt& rhs) { return *this = *this & rhs; }

TVM_FFI_INLINE BigInt& BigInt::operator|=(const BigInt& rhs) { return *this = *this | rhs; }

TVM_FFI_INLINE BigInt& BigInt::operator^=(const BigInt& rhs) { return *this = *this ^ rhs; }

TVM_FFI_INLINE BigInt& BigInt::operator<<=(const BigInt& rhs) { return *this = *this << rhs; }

TVM_FFI_INLINE BigInt& BigInt::operator>>=(const BigInt& rhs) { return *this = *this >> rhs; }

TVM_FFI_INLINE BigInt& BigInt::operator++() { return *this += 1; }

TVM_FFI_INLINE BigInt& BigInt::operator--() { return *this -= 1; }

TVM_FFI_INLINE BigInt BigInt::operator++(int) {
  BigInt old = *this;
  ++*this;
  return old;
}

TVM_FFI_INLINE BigInt BigInt::operator--(int) {
  BigInt old = *this;
  --*this;
  return old;
}
/// \endcond
/*! \brief Read a signed decimal integer, leaving the destination unchanged on failure. */
inline std::istream& operator>>(std::istream& is, BigInt& value) {
  std::istream::sentry sentry(is);
  if (!sentry) return is;
  bool negative = false;
  if (is.peek() == '+' || is.peek() == '-') negative = is.get() == '-';
  BigInt result;
  bool have_digit = false;
  // Simple digit-at-a-time parsing accepts repeated allocation and quadratic work for long inputs.
  while (is.good()) {
    int ch = is.peek();
    if (ch < '0' || ch > '9') break;
    is.get();
    have_digit = true;
    result = result * 10 + (ch - '0');
  }
  if (!have_digit) {
    is.setstate(std::ios::failbit);
  } else {
    value = negative ? -result : std::move(result);
  }
  return is;
}

inline std::ostream& operator<<(std::ostream& os, const BigInt& value) {
  return os << detail::int_ops::ToStringFallback(value);
}

}  // namespace ffi
}  // namespace tvm
namespace std {
// Swap the two owning cells directly, avoiding generic move/reset/destruction work.
template <>
TVM_FFI_INLINE void swap(tvm::ffi::BigInt& a, tvm::ffi::BigInt& b) noexcept {
  a.swap(b);
}

template <>
struct hash<tvm::ffi::BigInt> {
  size_t operator()(const tvm::ffi::BigInt& value) const {
    return static_cast<size_t>(value.hash());
  }
};
}  // namespace std
#endif  // TVM_FFI_BIG_INT_H_
