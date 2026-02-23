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
/*
 * \file src/ffi/extra/recursive_compare.cc
 *
 * \brief Reflection-based recursive comparison (Eq, Lt, Le, Gt, Ge).
 */
#include <tvm/ffi/any.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/dict.h>
#include <tvm/ffi/container/list.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/container/shape.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/reflection/accessor.h>
#include <tvm/ffi/reflection/registry.h>

#include <cmath>

namespace tvm {
namespace ffi {

namespace {

/*!
 * \brief Three-way recursive comparer.
 *
 * Returns int32_t: -1 (lhs < rhs), 0 (equal), +1 (lhs > rhs).
 *
 * When eq_only_ is true, type mismatches return non-zero instead of
 * throwing, and Map/Dict ordering is not attempted.
 */
class RecursiveComparer {
 public:
  explicit RecursiveComparer(bool eq_only) : eq_only_(eq_only) {}

  int32_t CompareAny(const Any& lhs, const Any& rhs) {
    ++depth_;
    struct DepthGuard {
      int32_t& d;
      ~DepthGuard() { --d; }
    } guard{depth_};
    if (depth_ > kMaxDepth) {
      TVM_FFI_THROW(ValueError) << "RecursiveCompare: maximum recursion depth (" << kMaxDepth
                                << ") exceeded; possible cycle";
    }
    using details::AnyUnsafe;
    const TVMFFIAny* lhs_data = AnyUnsafe::TVMFFIAnyPtrFromAny(lhs);
    const TVMFFIAny* rhs_data = AnyUnsafe::TVMFFIAnyPtrFromAny(rhs);
    int32_t lti = lhs_data->type_index;
    int32_t rti = rhs_data->type_index;

    // Handle None specially: None == None, None < any non-None
    if (lti == TypeIndex::kTVMFFINone && rti == TypeIndex::kTVMFFINone) return 0;
    if (lti == TypeIndex::kTVMFFINone) return -1;
    if (rti == TypeIndex::kTVMFFINone) return 1;

    // Handle String (Str/SmallStr cross-variant) before type-mismatch check
    if (IsStringType(lti) && IsStringType(rti)) {
      return CompareString(lhs, rhs, lhs_data, rhs_data, lti, rti);
    }
    // Handle Bytes (Bytes/SmallBytes cross-variant) before type-mismatch check
    if (IsBytesType(lti) && IsBytesType(rti)) {
      return CompareBytes(lhs, rhs, lhs_data, rhs_data, lti, rti);
    }

    // Type mismatch
    if (lti != rti) {
      if (eq_only_) return 1;  // not equal
      TVM_FFI_THROW(TypeError) << "Cannot compare values of different types: " << lhs.GetTypeKey()
                               << " vs " << rhs.GetTypeKey();
    }

    // Same type — POD dispatch
    if (lti < TypeIndex::kTVMFFIStaticObjectBegin) {
      return ComparePOD(lhs, rhs, lhs_data, rhs_data, lti);
    }

    // Object types — check pointer identity first
    const Object* lhs_obj = static_cast<const Object*>(lhs.as<Object>());
    const Object* rhs_obj = static_cast<const Object*>(rhs.as<Object>());
    if (lhs_obj == rhs_obj) return 0;
    if (lhs_obj == nullptr) return -1;
    if (rhs_obj == nullptr) return 1;

    switch (lti) {
      case TypeIndex::kTVMFFIStr:
        return CompareString(lhs, rhs, lhs_data, rhs_data, lti, rti);
      case TypeIndex::kTVMFFIBytes:
        return CompareBytes(lhs, rhs, lhs_data, rhs_data, lti, rti);
      case TypeIndex::kTVMFFIArray:
        return CompareSequence(AnyUnsafe::CopyFromAnyViewAfterCheck<Array<Any>>(lhs),
                               AnyUnsafe::CopyFromAnyViewAfterCheck<Array<Any>>(rhs));
      case TypeIndex::kTVMFFIList:
        return CompareSequence(AnyUnsafe::CopyFromAnyViewAfterCheck<List<Any>>(lhs),
                               AnyUnsafe::CopyFromAnyViewAfterCheck<List<Any>>(rhs));
      case TypeIndex::kTVMFFIMap:
        return CompareMap(AnyUnsafe::CopyFromAnyViewAfterCheck<Map<Any, Any>>(lhs),
                          AnyUnsafe::CopyFromAnyViewAfterCheck<Map<Any, Any>>(rhs));
      case TypeIndex::kTVMFFIDict:
        return CompareMap(AnyUnsafe::CopyFromAnyViewAfterCheck<Dict<Any, Any>>(lhs),
                          AnyUnsafe::CopyFromAnyViewAfterCheck<Dict<Any, Any>>(rhs));
      case TypeIndex::kTVMFFIShape:
        return CompareShape(AnyUnsafe::CopyFromAnyViewAfterCheck<Shape>(lhs),
                            AnyUnsafe::CopyFromAnyViewAfterCheck<Shape>(rhs));
      default:
        return CompareObject(lhs_obj, rhs_obj);
    }
  }

 private:
  static constexpr int32_t kMaxDepth = 128;
  bool eq_only_;
  int32_t depth_ = 0;

  static bool IsStringType(int32_t ti) {
    return ti == TypeIndex::kTVMFFIStr || ti == TypeIndex::kTVMFFISmallStr;
  }

  static bool IsBytesType(int32_t ti) {
    return ti == TypeIndex::kTVMFFIBytes || ti == TypeIndex::kTVMFFISmallBytes;
  }

  static int32_t Sign(int64_t v) { return (v > 0) - (v < 0); }

  // ---------- POD comparison ----------

  int32_t ComparePOD(const Any& lhs, const Any& rhs, const TVMFFIAny* lhs_data,
                     const TVMFFIAny* rhs_data, int32_t ti) {
    switch (ti) {
      case TypeIndex::kTVMFFIBool: {
        bool a = lhs_data->v_int64 != 0;
        bool b = rhs_data->v_int64 != 0;
        return static_cast<int32_t>(a) - static_cast<int32_t>(b);
      }
      case TypeIndex::kTVMFFIInt: {
        int64_t a = lhs_data->v_int64;
        int64_t b = rhs_data->v_int64;
        if (a < b) return -1;
        if (a > b) return 1;
        return 0;
      }
      case TypeIndex::kTVMFFIFloat: {
        double a = lhs_data->v_float64;
        double b = rhs_data->v_float64;
        // NaN == NaN for equality
        if (std::isnan(a) && std::isnan(b)) {
          if (eq_only_) return 0;
          TVM_FFI_THROW(TypeError) << "Cannot order NaN values";
        }
        if (std::isnan(a) || std::isnan(b)) {
          if (eq_only_) return 1;  // not equal
          TVM_FFI_THROW(TypeError) << "Cannot order NaN values";
        }
        if (a < b) return -1;
        if (a > b) return 1;
        return 0;
      }
      case TypeIndex::kTVMFFIDataType: {
        DLDataType a = lhs_data->v_dtype;
        DLDataType b = rhs_data->v_dtype;
        if (a.code != b.code) return (a.code < b.code) ? -1 : 1;
        if (a.bits != b.bits) return (a.bits < b.bits) ? -1 : 1;
        if (a.lanes != b.lanes) return (a.lanes < b.lanes) ? -1 : 1;
        return 0;
      }
      case TypeIndex::kTVMFFIDevice: {
        DLDevice a = lhs_data->v_device;
        DLDevice b = rhs_data->v_device;
        if (a.device_type != b.device_type) return (a.device_type < b.device_type) ? -1 : 1;
        if (a.device_id != b.device_id) return (a.device_id < b.device_id) ? -1 : 1;
        return 0;
      }
      default: {
        // Other POD types: bitwise equality only
        if (lhs_data->zero_padding == rhs_data->zero_padding &&
            lhs_data->v_int64 == rhs_data->v_int64) {
          return 0;
        }
        if (eq_only_) return 1;
        TVM_FFI_THROW(TypeError) << "Cannot order values of type " << lhs.GetTypeKey();
      }
    }
    TVM_FFI_UNREACHABLE();
  }

  // ---------- String comparison (handles SmallStr cross-variant) ----------

  int32_t CompareString(const Any& lhs, const Any& rhs, const TVMFFIAny* lhs_data,
                        const TVMFFIAny* rhs_data, int32_t lti, int32_t rti) {
    const char* lhs_ptr;
    size_t lhs_len;
    const char* rhs_ptr;
    size_t rhs_len;
    GetStringData(lhs, lhs_data, lti, &lhs_ptr, &lhs_len);
    GetStringData(rhs, rhs_data, rti, &rhs_ptr, &rhs_len);
    return SignFromMemncmp(Bytes::memncmp(lhs_ptr, rhs_ptr, lhs_len, rhs_len));
  }

  // ---------- Bytes comparison (handles SmallBytes cross-variant) ----------

  int32_t CompareBytes(const Any& lhs, const Any& rhs, const TVMFFIAny* lhs_data,
                       const TVMFFIAny* rhs_data, int32_t lti, int32_t rti) {
    const char* lhs_ptr;
    size_t lhs_len;
    const char* rhs_ptr;
    size_t rhs_len;
    GetBytesData(lhs, lhs_data, lti, &lhs_ptr, &lhs_len);
    GetBytesData(rhs, rhs_data, rti, &rhs_ptr, &rhs_len);
    return SignFromMemncmp(Bytes::memncmp(lhs_ptr, rhs_ptr, lhs_len, rhs_len));
  }

  static void GetStringData(const Any& val, const TVMFFIAny* data, int32_t ti, const char** out_ptr,
                            size_t* out_len) {
    if (ti == TypeIndex::kTVMFFISmallStr) {
      *out_ptr = data->v_bytes;
      *out_len = data->small_str_len;
    } else {
      const auto* obj =
          details::AnyUnsafe::CopyFromAnyViewAfterCheck<const details::BytesObjBase*>(val);
      *out_ptr = obj->data;
      *out_len = obj->size;
    }
  }

  static void GetBytesData(const Any& val, const TVMFFIAny* data, int32_t ti, const char** out_ptr,
                           size_t* out_len) {
    if (ti == TypeIndex::kTVMFFISmallBytes) {
      *out_ptr = data->v_bytes;
      *out_len = data->small_str_len;
    } else {
      const auto* obj =
          details::AnyUnsafe::CopyFromAnyViewAfterCheck<const details::BytesObjBase*>(val);
      *out_ptr = obj->data;
      *out_len = obj->size;
    }
  }

  static int32_t SignFromMemncmp(int v) {
    if (v < 0) return -1;
    if (v > 0) return 1;
    return 0;
  }

  // ---------- Sequence comparison (Array / List) ----------

  template <typename SeqType>
  int32_t CompareSequence(const SeqType& lhs, const SeqType& rhs) {
    size_t min_len = std::min(lhs.size(), rhs.size());
    for (size_t i = 0; i < min_len; ++i) {
      int32_t cmp = CompareAny(lhs[i], rhs[i]);
      if (cmp != 0) return cmp;
    }
    if (lhs.size() < rhs.size()) return -1;
    if (lhs.size() > rhs.size()) return 1;
    return 0;
  }

  // ---------- Map / Dict comparison (equality only) ----------

  template <typename MapType>
  int32_t CompareMap(const MapType& lhs, const MapType& rhs) {
    if (lhs.size() != rhs.size()) {
      if (eq_only_) return 1;
      TVM_FFI_THROW(TypeError) << "Cannot order Map/Dict values";
    }
    for (const auto& kv : lhs) {
      auto it = rhs.find(kv.first);
      if (it == rhs.end()) {
        if (eq_only_) return 1;
        TVM_FFI_THROW(TypeError) << "Cannot order Map/Dict values";
      }
      int32_t cmp = CompareAny(kv.second, (*it).second);
      if (cmp != 0) {
        if (!eq_only_) {
          TVM_FFI_THROW(TypeError) << "Cannot order Map/Dict values";
        }
        return cmp;
      }
    }
    return 0;
  }

  // ---------- Shape comparison ----------

  int32_t CompareShape(const Shape& lhs, const Shape& rhs) {
    size_t min_len = std::min(lhs.size(), rhs.size());
    for (size_t i = 0; i < min_len; ++i) {
      if (lhs[i] < rhs[i]) return -1;
      if (lhs[i] > rhs[i]) return 1;
    }
    if (lhs.size() < rhs.size()) return -1;
    if (lhs.size() > rhs.size()) return 1;
    return 0;
  }

  // ---------- Reflected Object comparison ----------

  int32_t CompareObject(const Object* lhs, const Object* rhs) {
    // Different type indices
    if (lhs->type_index() != rhs->type_index()) {
      if (eq_only_) return 1;
      const TVMFFITypeInfo* lhs_info = TVMFFIGetTypeInfo(lhs->type_index());
      const TVMFFITypeInfo* rhs_info = TVMFFIGetTypeInfo(rhs->type_index());
      TVM_FFI_THROW(TypeError) << "Cannot compare objects of different types: "
                               << String(lhs_info->type_key) << " vs "
                               << String(rhs_info->type_key);
    }
    const TVMFFITypeInfo* type_info = TVMFFIGetTypeInfo(lhs->type_index());
    int32_t result = 0;
    reflection::ForEachFieldInfoWithEarlyStop(type_info, [&](const TVMFFIFieldInfo* finfo) {
      if (finfo->flags & kTVMFFIFieldFlagBitMaskCompareOff) return false;
      reflection::FieldGetter getter(finfo);
      Any lhs_value = getter(lhs);
      Any rhs_value = getter(rhs);
      result = CompareAny(lhs_value, rhs_value);
      return result != 0;  // early stop on mismatch
    });
    return result;
  }
};

}  // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("ffi.RecursiveEq",
           [](const Any& lhs, const Any& rhs) -> bool {
             RecursiveComparer cmp(/*eq_only=*/true);
             return cmp.CompareAny(lhs, rhs) == 0;
           })
      .def("ffi.RecursiveLt",
           [](const Any& lhs, const Any& rhs) -> bool {
             RecursiveComparer cmp(/*eq_only=*/false);
             return cmp.CompareAny(lhs, rhs) < 0;
           })
      .def("ffi.RecursiveLe",
           [](const Any& lhs, const Any& rhs) -> bool {
             RecursiveComparer cmp(/*eq_only=*/false);
             return cmp.CompareAny(lhs, rhs) <= 0;
           })
      .def("ffi.RecursiveGt",
           [](const Any& lhs, const Any& rhs) -> bool {
             RecursiveComparer cmp(/*eq_only=*/false);
             return cmp.CompareAny(lhs, rhs) > 0;
           })
      .def("ffi.RecursiveGe", [](const Any& lhs, const Any& rhs) -> bool {
        RecursiveComparer cmp(/*eq_only=*/false);
        return cmp.CompareAny(lhs, rhs) >= 0;
      });
}

}  // namespace ffi
}  // namespace tvm
