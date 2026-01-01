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
 * \file tvm/ffi/extra/dtype.h
 * \brief Type traits to map C++ types to DLPack dtypes.
 */
#ifndef TVM_FFI_EXTRA_DTYPE_H_
#define TVM_FFI_EXTRA_DTYPE_H_

#include <dlpack/dlpack.h>

#include <type_traits>

// Common for both CUDA and HIP
struct __half;

// CUDA
struct __nv_fp8_e4m3;
struct __nv_bfloat16;
struct __nv_fp8_e5m2;
struct __nv_fp8_e8m0;
struct __nv_fp4_e2m1;
struct __nv_fp4x2_e2m1;

// HIP
struct __hip_bfloat16;
struct hip_bfloat16;  // i don't know why this is a struct instead of alias...
struct __hip_fp8_e4m3;
struct __hip_fp8_e4m3_fnuz;
struct __hip_fp8_e5m2;
struct __hip_fp8_e5m2_fnuz;
struct __hip_fp4_e2m1;
struct __hip_fp4x2_e2m1;

namespace tvm_ffi {

template <typename T>
struct dtype_trait {};

namespace details::dtypes {

template <typename T>
struct integer_trait {
  static constexpr DLDataType value = {
      /* code = */ std::is_signed_v<T> ? kDLInt : kDLUInt,
      /* bits = */ static_cast<uint8_t>(sizeof(T) * 8),
      /* lanes = */ 1,
  };
};

template <typename T>
struct float_trait {
  static constexpr DLDataType value = {
      /* code = */ kDLFloat,
      /* bits = */ static_cast<uint8_t>(sizeof(T) * 8),
      /* lanes = */ 1,
  };
};

}  // namespace details::dtypes

template <>
struct dtype_trait<signed char> : details::dtypes::integer_trait<signed char> {};

template <>
struct dtype_trait<unsigned char> : details::dtypes::integer_trait<unsigned char> {};

template <>
struct dtype_trait<signed short> : details::dtypes::integer_trait<signed short> {};

template <>
struct dtype_trait<unsigned short> : details::dtypes::integer_trait<unsigned short> {};

template <>
struct dtype_trait<signed int> : details::dtypes::integer_trait<signed int> {};

template <>
struct dtype_trait<unsigned int> : details::dtypes::integer_trait<unsigned int> {};

template <>
struct dtype_trait<signed long> : details::dtypes::integer_trait<signed long> {};

template <>
struct dtype_trait<unsigned long> : details::dtypes::integer_trait<unsigned long> {};

template <>
struct dtype_trait<signed long long> : details::dtypes::integer_trait<signed long long> {};

template <>
struct dtype_trait<unsigned long long> : details::dtypes::integer_trait<unsigned long long> {};

template <>
struct dtype_trait<float> : details::dtypes::float_trait<float> {};

template <>
struct dtype_trait<double> : details::dtypes::float_trait<double> {};

// Specialization for bool

template <>
struct dtype_trait<bool> {
  static constexpr DLDataType value = {DLDataTypeCode::kDLBool, 8, 1};
};

// Specializations for CUDA

template <>
struct dtype_trait<__half> {
  static constexpr DLDataType value = {DLDataTypeCode::kDLFloat, 16, 1};
};

template <>
struct dtype_trait<__nv_bfloat16> {
  static constexpr DLDataType value = {DLDataTypeCode::kDLBfloat, 16, 1};
};

template <>
struct dtype_trait<__nv_fp8_e4m3> {
  static constexpr DLDataType value = {DLDataTypeCode::kDLFloat8_e4m3fn, 8, 1};
};

template <>
struct dtype_trait<__nv_fp8_e5m2> {
  static constexpr DLDataType value = {DLDataTypeCode::kDLFloat8_e5m2, 8, 1};
};

template <>
struct dtype_trait<__nv_fp8_e8m0> {
  static constexpr DLDataType value = {DLDataTypeCode::kDLFloat8_e8m0fnu, 8, 1};
};

template <>
struct dtype_trait<__nv_fp4_e2m1> {
  static constexpr DLDataType value = {DLDataTypeCode::kDLFloat4_e2m1fn, 4, 1};
};

template <>
struct dtype_trait<__nv_fp4x2_e2m1> {
  static constexpr DLDataType value = {DLDataTypeCode::kDLFloat4_e2m1fn, 4, 2};
};

// Specializations for HIP

template <>
struct dtype_trait<__hip_bfloat16> {
  static constexpr DLDataType value = {DLDataTypeCode::kDLBfloat, 16, 1};
};

template <>
struct dtype_trait<hip_bfloat16> {
  static constexpr DLDataType value = {DLDataTypeCode::kDLBfloat, 16, 1};
};

template <>
struct dtype_trait<__hip_fp8_e4m3> {
  static constexpr DLDataType value = {DLDataTypeCode::kDLFloat8_e4m3fn, 8, 1};
};

template <>
struct dtype_trait<__hip_fp8_e4m3_fnuz> {
  static constexpr DLDataType value = {DLDataTypeCode::kDLFloat8_e4m3fnuz, 8, 1};
};

template <>
struct dtype_trait<__hip_fp8_e5m2> {
  static constexpr DLDataType value = {DLDataTypeCode::kDLFloat8_e5m2, 8, 1};
};

template <>
struct dtype_trait<__hip_fp8_e5m2_fnuz> {
  static constexpr DLDataType value = {DLDataTypeCode::kDLFloat8_e5m2fnuz, 8, 1};
};

template <>
struct dtype_trait<__hip_fp4_e2m1> {
  static constexpr DLDataType value = {DLDataTypeCode::kDLFloat4_e2m1fn, 4, 1};
};

template <>
struct dtype_trait<__hip_fp4x2_e2m1> {
  static constexpr DLDataType value = {DLDataTypeCode::kDLFloat4_e2m1fn, 4, 2};
};

}  // namespace tvm_ffi

#endif  // TVM_FFI_EXTRA_DTYPE_H_
