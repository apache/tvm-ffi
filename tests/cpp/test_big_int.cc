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
#include <gtest/gtest.h>
#include <tvm/ffi/any.h>
#include <tvm/ffi/big_int.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/function.h>

#include <cmath>
#include <limits>
#include <sstream>
#include <type_traits>

namespace {
using namespace tvm::ffi;

constexpr int64_t kMin = std::numeric_limits<int64_t>::min();
constexpr int64_t kMax = std::numeric_limits<int64_t>::max();

static_assert(!std::is_convertible_v<BigInt, bool>);
static_assert(!std::is_convertible_v<double, BigInt>);
static_assert(sizeof(BigInt) == sizeof(TVMFFIAny));
static_assert(!std::is_copy_constructible_v<detail::BigIntObj>);
static_assert(!std::is_default_constructible_v<detail::BigIntObj>);

void ExpectInline(const BigInt& value, int64_t expected) {
  EXPECT_TRUE(detail::BigIntUnsafe::CheckInt64(value));
  EXPECT_EQ(static_cast<int64_t>(value), expected);
}

std::string PrintBigInt(const BigInt& value) {
  std::ostringstream stream;
  stream << value;
  return stream.str();
}

TEST(BigInt, RepresentationAndOwnership) {
  BigInt zero;
  ExpectInline(zero, 0);
  BigInt small(kMin);
  auto repr = detail::BigIntUnsafe::GetArrayView(small);
  ASSERT_EQ(repr.size(), 1);
  EXPECT_EQ(repr[0], kMin);
  EXPECT_EQ(detail::BigIntUnsafe::GetInt64(small), kMin);
  BigInt large = BigInt(1) << 255;
  ASSERT_FALSE(detail::BigIntUnsafe::CheckInt64(large));
  repr = detail::BigIntUnsafe::GetArrayView(large);
  ASSERT_EQ(repr.size(), 5);
  EXPECT_EQ(repr[4], 0);
  EXPECT_EQ(repr[3], kMin);
  EXPECT_EQ(repr[0], 0);
  Any owner(large);
  EXPECT_EQ(owner.type_index(), TypeIndex::kTVMFFIBigInt);
  EXPECT_EQ(owner.GetTypeKey(), "ffi.BigInt");
  BigInt copy = owner.cast<BigInt>();
  EXPECT_TRUE(Any(copy).same_as(owner));
  BigInt moved(std::move(copy));
  ExpectInline(copy, 0);  // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(moved, large);
  copy = moved;
  moved = 9;
  EXPECT_EQ(copy, large);
  EXPECT_EQ(owner.cast<BigInt>(), large);
  AnyView view(copy);
  EXPECT_EQ(view.cast<BigInt>(), large);
  Any transferred(std::move(copy));
  ExpectInline(copy, 0);  // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(transferred.cast<BigInt>(), large);
  EXPECT_EQ(Any(42).cast<BigInt>(), 42);
  EXPECT_FALSE(Any(1.5).try_cast<BigInt>().has_value());
  Function add = Function::FromTyped([](const BigInt& a, const BigInt& b) { return a + b; });
  EXPECT_EQ(add(large, -large).cast<BigInt>(), 0);
  EXPECT_FALSE(static_cast<bool>(zero));
  EXPECT_TRUE(static_cast<bool>(small));
  EXPECT_TRUE(static_cast<bool>(large));
  int calls = 0;
  EXPECT_FALSE(zero && ++calls);
  EXPECT_TRUE(large || ++calls);
  EXPECT_EQ(calls, 0);
}

TEST(BigInt, SwapOwnership) {
  BigInt a = 42;
  BigInt b = BigInt(1) << 256;
  Any owner(b);
  std::swap(a, b);
  EXPECT_TRUE(Any(a).same_as(owner));
  ExpectInline(b, 42);
  std::swap(a, a);
  EXPECT_TRUE(Any(a).same_as(owner));
  BigInt c = -a;
  Any negative_owner(c);
  std::swap(a, c);
  EXPECT_TRUE(Any(a).same_as(negative_owner));
  EXPECT_TRUE(Any(c).same_as(owner));
  std::swap(a, b);
  ExpectInline(a, 42);
  EXPECT_TRUE(Any(b).same_as(negative_owner));
  BigInt small(kMin);
  std::swap(a, small);
  ExpectInline(a, kMin);
  ExpectInline(small, 42);
  std::swap(a, a);
  ExpectInline(a, kMin);
  BigInt* inline_self = &a;
  a = *inline_self;
  ExpectInline(a, kMin);
  a = std::move(*inline_self);
  ExpectInline(a, kMin);
  BigInt* heap_self = &b;
  b = *heap_self;
  EXPECT_TRUE(Any(b).same_as(negative_owner));
  b = std::move(*heap_self);
  EXPECT_TRUE(Any(b).same_as(negative_owner));
}

TEST(BigInt, PromotionDemotionAndMixedOperands) {
  BigInt high = BigInt(kMax) + 1;
  EXPECT_FALSE(detail::BigIntUnsafe::CheckInt64(high));
  EXPECT_EQ(PrintBigInt(high), "9223372036854775808");
  ExpectInline(high - 1, kMax);
  ExpectInline(-high, kMin);
  ExpectInline(high + kMin, 0);
  EXPECT_EQ(BigInt(kMin) - 1, -(high + 1));
  EXPECT_EQ(-BigInt(kMin), high);
  EXPECT_EQ(BigInt(kMin) / -1, high);
  ExpectInline(BigInt(kMin) % -1, 0);
  EXPECT_EQ(PrintBigInt(BigInt(std::numeric_limits<uint64_t>::max())), "18446744073709551615");
  EXPECT_EQ(BigInt(1) + BigInt(std::numeric_limits<uint64_t>::max()), BigInt(1) << 64);
  EXPECT_EQ(BigInt(std::numeric_limits<uint64_t>::max()) - BigInt(1),
            BigInt(std::numeric_limits<uint64_t>::max()) - 1);
  for (const auto& [a, b] : {std::pair<int64_t, int64_t>{kMax, 17}, {-123, 17}}) {
    BigInt x(a), y(b);
    EXPECT_EQ(x + y, x + b);
    EXPECT_EQ(x + y, a + y);
    EXPECT_EQ(x - y, x - b);
    EXPECT_EQ(x - y, a - y);
    EXPECT_EQ(x * y, x * b);
    EXPECT_EQ(x * y, a * y);
    EXPECT_EQ((x * y) / y, x);
    EXPECT_EQ(x / y, a / y);
    EXPECT_EQ(x / y, x / b);
    EXPECT_EQ(x % y, a % y);
    EXPECT_EQ(x % y, x % b);
    EXPECT_EQ(x & y, a & y);
    EXPECT_EQ(x | y, x | b);
    EXPECT_EQ(x ^ y, a ^ y);
    EXPECT_EQ(min(x, y), min(a, y));
    EXPECT_EQ(max(x, y), max(x, b));
  }
  BigInt heap = BigInt(1) << 255;
  EXPECT_EQ(min(heap, -heap), -heap);
  EXPECT_EQ(min(-heap, heap), -heap);
  EXPECT_EQ(max(-heap, heap), heap);
  EXPECT_EQ(max(heap, -heap), heap);
  BigInt counter(kMax);
  EXPECT_EQ(counter++, kMax);
  EXPECT_EQ(counter, high);
  EXPECT_EQ(--counter, kMax);
  EXPECT_EQ(counter--, kMax);
  ExpectInline(counter, kMax - 1);
}

TEST(BigInt, NormalizedEquality) {
  BigInt value = (BigInt(1) << 255) + 13;
  BigInt equal = (BigInt(2) << 254) + 13;
  EXPECT_EQ(value, equal);
  EXPECT_FALSE(value != equal);
  EXPECT_NE(value, value + 1);
  EXPECT_NE(value, value + (BigInt(1) << 128));
  EXPECT_NE(BigInt(1) << 192, BigInt(1) << 193);
  EXPECT_NE(value, BigInt(1) << 128);
  EXPECT_NE(value, -value);

  BigInt above = BigInt(kMax) + 1;
  BigInt below = BigInt(kMin) - 1;
  EXPECT_FALSE(above == kMax);
  EXPECT_TRUE(kMax != above);
  EXPECT_FALSE(kMin == below);
  EXPECT_TRUE(below != kMin);
  EXPECT_EQ(BigInt(kMax), kMax);
  EXPECT_FALSE(kMin != BigInt(kMin));
}

TEST(BigInt, SignedDivision) {
  struct Case {
    int64_t a, b, q, r, floor_q, floor_r;
  };
  for (const auto& row : {Case{7, 3, 2, 1, 2, 1},
                          {-7, 3, -2, -1, -3, 2},
                          {7, -3, -2, 1, -3, -2},
                          {-7, -3, 2, -1, 2, -1},
                          {-6, 3, -2, 0, -2, 0},
                          {0, 3, 0, 0, 0, 0}}) {
    EXPECT_EQ(truncdiv(BigInt(row.a), row.b), row.q);
    EXPECT_EQ(truncmod(row.a, BigInt(row.b)), row.r);
    EXPECT_EQ(floordiv(BigInt(row.a), BigInt(row.b)), row.floor_q);
    EXPECT_EQ(floormod(row.a, BigInt(row.b)), row.floor_r);
  }
  BigInt wide = (BigInt(1) << 255) + 13;
  EXPECT_THROW(wide / 0, Error);
  EXPECT_THROW(floormod(BigInt(0), 0), Error);
}

TEST(BigInt, WideArithmeticAndBitwise) {
  for (int shift : {0, 63, 64, 255, 256}) {
    BigInt x = BigInt(1) << shift;
    EXPECT_EQ(x >> shift, 1);
    EXPECT_EQ((-x) >> shift, -1);
    ExpectInline(x - x, 0);
    ExpectInline((x + 3) - x, 3);
    EXPECT_EQ(x & -1, x);
    EXPECT_EQ(-1 & x, x);
    EXPECT_EQ(x | -1, -1);
    EXPECT_EQ(~x, -x - 1);
    EXPECT_EQ((x - 1) ^ x, 2 * x - 1);
    EXPECT_EQ((-x - 1) >> shift, -2);
    EXPECT_EQ(x << BigInt(2), x * 4);
    EXPECT_EQ(int64_t{1} << BigInt(shift), x);
  }
  BigInt huge = BigInt(1) << 255;
  ExpectInline(huge >> huge, 0);
  ExpectInline(-huge >> huge, -1);
  ExpectInline(BigInt(0) << huge, 0);
  EXPECT_THROW(huge << huge, Error);
  EXPECT_THROW(huge << -1, Error);
  EXPECT_THROW(BigInt(0) >> -1, Error);
  EXPECT_EQ(PrintBigInt((BigInt(1) << 100)), "1267650600228229401496703205376");
}

TEST(BigInt, IndependentWideFixture) {
  // Decimal expectations were computed with Python integers, independently of these operators.
  auto parse = [](const char* text) {
    std::istringstream stream(text);
    BigInt result;
    stream >> result;
    return result;
  };
  BigInt a =
      parse("-57896044618658097711813390734279005840776448950576999782953508207182870753808");
  BigInt b = parse("1361129467683753853854892183719458155744");
  EXPECT_EQ(PrintBigInt(a + b),
            "-57896044618658097711813390734279005839415319482893245929098616023463412598064");
  EXPECT_EQ(PrintBigInt(a * b),
            "-7880401239278895842467674614647861655059524949218434327250421484868722992620012470110"
            "8634385188332729146218745073152");
  EXPECT_EQ(PrintBigInt(floordiv(a, b)), "-42535295865117307932898767499013107221");
  EXPECT_EQ(PrintBigInt(floormod(a, b)), "340250229142126476510862697120718273616");
  EXPECT_EQ(PrintBigInt(a & b), "72903423425023200");
  EXPECT_EQ(PrintBigInt(a ^ b),
            "-57896044618658097711813390734279005839415319482893245929098761830310262644464");
  EXPECT_EQ(PrintBigInt(a >> 97), "-365375409332725729551097270762435786773001928705");
}

TEST(BigInt, Conversions) {
  for (double x : {-7.9, -0.0, 0.0, 0.5, 7.9, std::numeric_limits<double>::denorm_min()}) {
    ExpectInline(BigInt(x), static_cast<int64_t>(x));
  }
  EXPECT_THROW((void)BigInt{std::numeric_limits<double>::quiet_NaN()}, Error);
  EXPECT_THROW((void)BigInt{std::numeric_limits<double>::infinity()}, Error);
  EXPECT_EQ(BigInt(std::ldexp(1.0, 100)), BigInt(1) << 100);
  // shift=64 writes one significand word between a low zero word and a high sign guard.
  EXPECT_EQ(BigInt(std::ldexp(1.0, 116)), BigInt(1) << 116);
  EXPECT_EQ(BigInt(-std::ldexp(1.0, 100)), -(BigInt(1) << 100));
  BigInt x = BigInt(1) << 100;
  EXPECT_EQ(static_cast<double>(x), std::ldexp(1.0, 100));
  EXPECT_EQ(static_cast<double>(x + (BigInt(1) << 47)), std::ldexp(1.0, 100));
  EXPECT_EQ(static_cast<double>(x + (BigInt(1) << 47) + 1),
            std::nextafter(std::ldexp(1.0, 100), std::numeric_limits<double>::infinity()));
  double largest = std::numeric_limits<double>::max();
  EXPECT_EQ(static_cast<double>(BigInt(largest)), largest);
  EXPECT_THROW((void)static_cast<double>(BigInt(largest) + (BigInt(1) << 970)), Error);
  EXPECT_THROW((void)static_cast<double>(BigInt(1) << 1024), Error);
  EXPECT_THROW((void)static_cast<int64_t>(x), Error);
}

TEST(BigInt, ValueHashing) {
  BigInt a = BigInt(1) << 255;
  BigInt b = BigInt(2) << 254;
  EXPECT_FALSE(Any(a).same_as(Any(b)));
  EXPECT_TRUE(AnyEqual()(a, b));
  EXPECT_EQ(AnyHash()(a), AnyHash()(b));
  Map<BigInt, int> map{{a, 3}};
  EXPECT_EQ(map.at(b), 3);
  EXPECT_FALSE(AnyEqual()(a, a + 1));
  EXPECT_EQ(std::hash<BigInt>()(a), std::hash<BigInt>()(b));
}
TEST(BigInt, OptionalIntegerConversionsAndStreams) {
  EXPECT_EQ(BigInt(127).as<int8_t>(), 127);
  EXPECT_FALSE(BigInt(128).as<int8_t>());
  EXPECT_EQ(BigInt(-128).as<int8_t>(), -128);
  EXPECT_FALSE(BigInt(-129).as<int8_t>());
  EXPECT_EQ(BigInt(255).as<uint8_t>(), 255);
  EXPECT_FALSE(BigInt(-1).as<uint8_t>());
  EXPECT_FALSE(BigInt(256).as<uint8_t>());
  EXPECT_EQ(BigInt(kMin).as<int64_t>(), kMin);
  EXPECT_EQ(BigInt(std::numeric_limits<uint64_t>::max()).as<uint64_t>(),
            std::numeric_limits<uint64_t>::max());
  EXPECT_FALSE((BigInt(1) << 64).as<uint64_t>());
  EXPECT_FALSE((BigInt(1) << 63).as<int64_t>());
  for (const BigInt& original : {BigInt(0), BigInt(kMin), BigInt(1) << 255, -(BigInt(1) << 255)}) {
    std::stringstream stream;
    stream << original;
    BigInt roundtrip;
    stream >> roundtrip;
    EXPECT_FALSE(stream.fail());
    EXPECT_EQ(roundtrip, original);
  }
  std::stringstream partial("  +123x -456");
  BigInt value;
  partial >> value;
  EXPECT_EQ(value, 123);
  EXPECT_EQ(partial.peek(), 'x');
  std::stringstream invalid("-x");
  invalid >> value;
  EXPECT_TRUE(invalid.fail());
  EXPECT_EQ(value, 123);
}
TEST(BigInt, NativeWordContent) {
  for (const BigInt& x : {BigInt(0), BigInt(-1), BigInt(kMin), BigInt(kMax), BigInt(1) << 255,
                          -(BigInt(1) << 255), BigInt(std::numeric_limits<uint64_t>::max())}) {
    TVMFFIAny input = AnyView(x).CopyToTVMFFIAny(), decoded;
    TVMFFIByteArray bytes = TVMFFIBigIntGetContentByteArray(&input);
    auto view = detail::BigIntUnsafe::GetArrayView(x);
    ASSERT_EQ(bytes.size, view.size() * sizeof(int64_t));
    EXPECT_EQ(std::memcmp(bytes.data, &view[0], bytes.size), 0);
    if (detail::BigIntUnsafe::CheckInt64(x)) {
      EXPECT_EQ(bytes.data, reinterpret_cast<const char*>(&input.v_int64));
    } else {
      EXPECT_EQ(bytes.data, reinterpret_cast<const char*>(&view[0]));
    }
    ASSERT_EQ(TVMFFIBigIntFromByteArray(&bytes, &decoded), 0);
    Any owner = detail::AnyUnsafe::MoveTVMFFIAnyRawToAny(decoded);
    EXPECT_EQ(owner.cast<BigInt>(), x);
  }
  for (int64_t value : {int64_t{0}, int64_t{-1}, kMin, kMax}) {
    int64_t words[] = {value, value < 0 ? -1 : 0, value < 0 ? -1 : 0};
    TVMFFIByteArray bytes{reinterpret_cast<const char*>(words), sizeof(words)};
    TVMFFIAny decoded;
    ASSERT_EQ(TVMFFIBigIntFromByteArray(&bytes, &decoded), 0);
    EXPECT_EQ(decoded.type_index, TypeIndex::kTVMFFIInt);
    EXPECT_EQ(decoded.v_int64, value);
  }
  TVMFFIByteArray empty{nullptr, 0};
  TVMFFIAny decoded;
  ASSERT_EQ(TVMFFIBigIntFromByteArray(&empty, &decoded), 0);
  EXPECT_EQ(decoded.v_int64, 0);
  TVMFFIByteArray invalid{"x", 1};
  EXPECT_NE(TVMFFIBigIntFromByteArray(&invalid, &decoded), 0);
  TVMFFIObjectHandle error;
  TVMFFIErrorMoveFromRaised(&error);
  TVMFFIObjectDecRef(error);
}

TEST(BigInt, ConstructionAndNormalization) {
  auto zeros = make_inplace_array_object<detail::BigIntObj, int64_t>(4, 4);
  std::fill_n(detail::BigIntUnsafe::GetMutableData(zeros), 4, 0);
  BigInt zero = detail::BigIntUnsafe::Normalize(std::move(zeros));
  ExpectInline(zero, 0);
  for (const BigInt& x : {zero, BigInt(1), BigInt(-1), BigInt(1) << 192, -(BigInt(1) << 192)}) {
    auto view = detail::BigIntUnsafe::GetArrayView(x);
    EXPECT_EQ(detail::int_ops::IsNormalizedZero(view), x == 0);
    EXPECT_EQ(detail::int_ops::IsNegative(view), x < 0);
  }
  auto ptr = make_inplace_array_object<detail::BigIntObj, int64_t>(6, 6);
  auto* original = ptr.get();
  int64_t* data = detail::BigIntUnsafe::GetMutableData(ptr);
  data[0] = 7;
  data[1] = 1;
  std::fill_n(data + 2, 4, 0);
  BigInt value = detail::BigIntUnsafe::Normalize(std::move(ptr));
  auto repr = detail::BigIntUnsafe::GetArrayView(value);
  EXPECT_EQ(repr.size(), 2);
  EXPECT_EQ(&repr[0], reinterpret_cast<const int64_t*>(original + 1));
  EXPECT_EQ(value, (BigInt(1) << 64) + 7);
  auto negative = make_inplace_array_object<detail::BigIntObj, int64_t>(4, 4);
  std::fill_n(detail::BigIntUnsafe::GetMutableData(negative), 4, -1);
  ExpectInline(detail::BigIntUnsafe::Normalize(std::move(negative)), -1);
}

TEST(BigInt, PortableWordMultiply) {
  // Half-product carry, complete-word carry, and unequal multiword lengths.
  for (const auto& [m, n] : {std::pair<int, int>{32, 32}, {64, 64}, {255, 256}}) {
    BigInt a = (BigInt(1) << m) - 1, b = (BigInt(1) << n) - 1;
    BigInt expected = (BigInt(1) << (m + n)) - (BigInt(1) << m) - (BigInt(1) << n) + 1;
    EXPECT_EQ(a * b, expected);
  }
  BigInt a = (BigInt(1) << 255) - 1, b = (BigInt(1) << 256) - 1;
  BigInt product = (BigInt(1) << 511) - (BigInt(1) << 255) - (BigInt(1) << 256) + 1;
  EXPECT_EQ((-a) * b, -product);
  EXPECT_EQ(a * (-b), -product);
  EXPECT_EQ((-a) * (-b), product);
  ExpectInline(a * 0, 0);
  ExpectInline(0 * a, 0);
  ExpectInline((-a) * 0, 0);
  ExpectInline(0 * (-a), 0);
}

TEST(BigInt, DirectSignedMultiplyAndDivRem) {
  BigInt divisor = (BigInt(1) << 130) + 5;
  BigInt quotient = (BigInt(1) << 120) + 3;
  BigInt exact = quotient * divisor;
  BigInt dividend = exact + 7;
  struct Case {
    BigInt a, b, q, r;
  };
  // General zero, all nonexact sign quadrants, and an exact low-zero/minimum-word magnitude.
  for (const auto& row : {Case{0, divisor, 0, 0},
                          {7, divisor, 0, 7},
                          {dividend, divisor, quotient, 7},
                          {-dividend, divisor, -quotient, -7},
                          {dividend, -divisor, -quotient, 7},
                          {-dividend, -divisor, quotient, -7},
                          {BigInt(kMin) << 128, BigInt(1) << 64, -(BigInt(1) << 127), 0}}) {
    auto qr = detail::int_ops::DivRemFallback(detail::BigIntUnsafe::GetArrayView(row.a),
                                              detail::BigIntUnsafe::GetArrayView(row.b));
    EXPECT_EQ(qr.first, row.q);
    EXPECT_EQ(qr.second, row.r);
  }
  EXPECT_EQ(floordiv(-exact, divisor), -quotient);
  EXPECT_EQ(floormod(-exact, divisor), 0);
  EXPECT_EQ(floordiv(-dividend, divisor), -quotient - 1);
  EXPECT_EQ(floormod(-dividend, divisor), divisor - 7);
}

TEST(BigInt, MultiwordDivisionEstimates) {
  BigInt base = BigInt(1) << 32;
  BigInt divisor = (BigInt(1) << 63) + base - 1;
  BigInt clamp_divisor = (BigInt(1) << 63) + 1;
  BigInt addback_divisor = (BigInt(1) << 95) + 1;
  BigInt shifted_divisor = (BigInt(1) << 64) + 1;
  BigInt wide_quotient = (BigInt(1) << 64) + 7;
  BigInt wide_remainder = (BigInt(1) << 63) + base + 5;
  struct Case {
    BigInt a;
    BigInt b;
    BigInt q;
    BigInt r;
  };
  // Clamp, one/two estimate corrections, addback, and maximal normalization shift.
  for (const auto& row :
       {Case{clamp_divisor * base - 1, clamp_divisor, base - 1, BigInt(1) << 63},
        {divisor * 2 - 1, divisor, 1, divisor - 1},
        {divisor * (base - 2) - 1, divisor, base - 3, divisor - 1},
        {addback_divisor * (base + 1) - 1, addback_divisor, base, BigInt(1) << 95},
        {shifted_divisor * wide_quotient + wide_remainder, shifted_divisor, wide_quotient,
         wide_remainder},
        {divisor - 1, divisor, 0, divisor - 1},
        {divisor, divisor, 1, 0}}) {
    auto qr = detail::int_ops::DivRemFallback(detail::BigIntUnsafe::GetArrayView(row.a),
                                              detail::BigIntUnsafe::GetArrayView(row.b));
    EXPECT_EQ(qr.first, row.q);
    EXPECT_EQ(qr.second, row.r);
  }
}

TEST(BigInt, ScalarDivRemBoundaries) {
  BigInt quotient = (BigInt(1) << 192) + 3;
  BigInt dividend = quotient * 3 + 2;
  BigInt maximum_remainder = quotient * (BigInt(1) << 63) + kMax;
  constexpr int64_t kHalfMax = std::numeric_limits<uint32_t>::max();
  BigInt half_boundary = (BigInt(kHalfMax) << 64) - 1;
  BigInt word_quotient = std::numeric_limits<uint64_t>::max();
  constexpr int64_t kCorrectionDivisor = 0x400000007fffffff;
  struct Case {
    BigInt a;
    int64_t b;
    BigInt q;
    int64_t r;
  };
  // Zero, exact/nonexact, signed scalar remainder, and the full |INT64_MIN| bound.
  for (const auto& row :
       {Case{0, 3, 0, 0},
        {2, 3, 0, 2},
        {quotient, 1, quotient, 0},
        {-quotient * 3, 3, -quotient, 0},
        {quotient * 3, 3, quotient, 0},
        {dividend, 3, quotient, 2},
        {-dividend, 3, -quotient, -2},
        {maximum_remainder, kMin, -quotient, kMax},
        {-maximum_remainder, kMin, quotient, -kMax},
        {kMin, -1, BigInt(1) << 63, 0},
        {half_boundary, kHalfMax, word_quotient, kHalfMax - 1},
        {-half_boundary, -kHalfMax, word_quotient, 1 - kHalfMax},
        {(BigInt(kHalfMax + 1) << 64) - 1, kHalfMax + 1, word_quotient, kHalfMax},
        // One quotient correction, then two corrections with both overflow guards.
        {BigInt(1) << 64, kHalfMax + 2, kHalfMax, 1},
        {(BigInt(kCorrectionDivisor) << 64) - 1, kCorrectionDivisor, word_quotient,
         kCorrectionDivisor - 1}}) {
    auto qr = detail::int_ops::DivRemFallback(detail::BigIntUnsafe::GetArrayView(row.a),
                                              detail::BigIntUnsafe::GetArrayView(row.b));
    EXPECT_EQ(qr.first, row.q);
    ExpectInline(qr.second, row.r);
  }
  EXPECT_EQ(truncdiv(dividend, int64_t{3}), quotient);
  EXPECT_EQ(truncmod(-dividend, int64_t{3}), -2);
  EXPECT_EQ(floordiv(-dividend, int64_t{3}), -quotient - 1);
  EXPECT_EQ(floormod(-dividend, int64_t{3}), 1);
  BigInt minimum = kMin;
  const int64_t zero = 0;
  EXPECT_THROW(detail::int_ops::DivRemFallback(detail::BigIntUnsafe::GetArrayView(minimum),
                                               detail::BigIntUnsafe::GetArrayView(zero)),
               Error);
}

TEST(BigInt, SmallDivisorRemainders) {
  BigInt value = (BigInt(1) << 255) | (BigInt(0x12345678) << 128) | BigInt(0xfedcba9876543211ULL);
  struct Case {
    BigInt a;
    int64_t b;
    int64_t trunc;
    int64_t floor;
  };
  // Independent signed int32 boundary answers, plus an exact negative-divisor remainder.
  for (const auto& row : {Case{value, -2147483648LL, 1985229329, -162254319},
                          {-value, 2147483647, -391319368, 1756164279},
                          {-value, -2147483648LL, -1985229329, -1985229329},
                          {value, 2147483647, 391319368, 391319368},
                          {BigInt(1) << 255, -2147483648LL, 0, 0}}) {
    ExpectInline(truncmod(row.a, row.b), row.trunc);
    ExpectInline(floormod(row.a, BigInt(row.b)), row.floor);
  }
  EXPECT_THROW((void)truncmod(value, 0), Error);
  EXPECT_THROW((void)floormod(value, 0), Error);
}
}  // namespace
