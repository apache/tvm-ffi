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
 * \file src/ffi/extra/recursive_hash.cc
 *
 * \brief Reflection-based recursive hash (companion to RecursiveEq).
 */
#include <tvm/ffi/any.h>
#include <tvm/ffi/base_details.h>
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

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm {
namespace ffi {

namespace {

/*!
 * \brief Reflection-based recursive hasher.
 *
 * Computes a deterministic hash consistent with RecursiveEq:
 *   RecursiveEq(a, b) => RecursiveHash(a) == RecursiveHash(b)
 */
class RecursiveHasher {
 public:
  uint64_t HashAny(const Any& value) {
    ++depth_;
    struct DepthGuard {
      int32_t& d;
      ~DepthGuard() { --d; }
    } guard{depth_};
    if (depth_ > kMaxDepth) {
      TVM_FFI_THROW(ValueError) << "RecursiveHash: maximum recursion depth (" << kMaxDepth
                                << ") exceeded; possible cycle";
    }
    using details::AnyUnsafe;
    const TVMFFIAny* data = AnyUnsafe::TVMFFIAnyPtrFromAny(value);
    int32_t ti = data->type_index;

    // None
    if (ti == TypeIndex::kTVMFFINone) {
      return details::StableHashCombine(uint64_t{0}, uint64_t{0});
    }

    // String (Str/SmallStr cross-variant)
    if (IsStringType(ti)) {
      return HashString(value, data, ti);
    }
    // Bytes (Bytes/SmallBytes cross-variant)
    if (IsBytesType(ti)) {
      return HashBytes(value, data, ti);
    }

    // POD types
    if (ti < TypeIndex::kTVMFFIStaticObjectBegin) {
      return HashPOD(value, data, ti);
    }

    // Object types — memoization + cycle detection.
    const Object* obj = static_cast<const Object*>(value.as<Object>());
    if (obj == nullptr) {
      return details::StableHashCombine(uint64_t{0}, uint64_t{0});
    }
    // Return memoized hash if this object was already fully hashed.
    auto memo_it = memo_.find(obj);
    if (memo_it != memo_.end()) return memo_it->second;
    // Cycle detection: if this object is currently on the call stack,
    // return a sentinel to break the cycle.
    auto [stack_it, inserted] = on_stack_.insert(obj);
    if (!inserted) {
      return TVMFFIGetTypeInfo(obj->type_index())->type_key_hash;
    }
    struct StackGuard {
      std::unordered_set<const Object*>& s;
      const Object* p;
      ~StackGuard() { s.erase(p); }
    } stack_guard{on_stack_, obj};

    uint64_t result;
    switch (ti) {
      case TypeIndex::kTVMFFIStr:
        result = HashString(value, data, ti);
        break;
      case TypeIndex::kTVMFFIBytes:
        result = HashBytes(value, data, ti);
        break;
      case TypeIndex::kTVMFFIArray:
        result = HashSequence(AnyUnsafe::CopyFromAnyViewAfterCheck<Array<Any>>(value));
        break;
      case TypeIndex::kTVMFFIList:
        result = HashSequence(AnyUnsafe::CopyFromAnyViewAfterCheck<List<Any>>(value));
        break;
      case TypeIndex::kTVMFFIMap:
        result = HashMap(AnyUnsafe::CopyFromAnyViewAfterCheck<Map<Any, Any>>(value));
        break;
      case TypeIndex::kTVMFFIDict:
        result = HashMap(AnyUnsafe::CopyFromAnyViewAfterCheck<Dict<Any, Any>>(value));
        break;
      case TypeIndex::kTVMFFIShape:
        result = HashShape(AnyUnsafe::CopyFromAnyViewAfterCheck<Shape>(value));
        break;
      default:
        result = HashObject(obj);
        break;
    }
    memo_[obj] = result;
    return result;
  }

 private:
  static constexpr int32_t kMaxDepth = 128;
  int32_t depth_ = 0;
  std::unordered_set<const Object*> on_stack_;
  std::unordered_map<const Object*, uint64_t> memo_;

  static bool IsStringType(int32_t ti) {
    return ti == TypeIndex::kTVMFFIStr || ti == TypeIndex::kTVMFFISmallStr;
  }

  static bool IsBytesType(int32_t ti) {
    return ti == TypeIndex::kTVMFFIBytes || ti == TypeIndex::kTVMFFISmallBytes;
  }

  // ---------- POD hashing ----------

  uint64_t HashPOD(const Any& value, const TVMFFIAny* data, int32_t ti) {
    switch (ti) {
      case TypeIndex::kTVMFFIBool: {
        uint64_t v = data->v_int64 != 0 ? 1 : 0;
        return details::StableHashCombine(static_cast<uint64_t>(ti), v);
      }
      case TypeIndex::kTVMFFIInt: {
        return details::StableHashCombine(static_cast<uint64_t>(ti),
                                          static_cast<uint64_t>(data->v_int64));
      }
      case TypeIndex::kTVMFFIFloat: {
        double v = data->v_float64;
        uint64_t bits;
        if (std::isnan(v)) {
          // All NaN payloads hash the same (consistent with RecursiveEq)
          double canonical = std::numeric_limits<double>::quiet_NaN();
          std::memcpy(&bits, &canonical, sizeof(bits));
        } else if (v == 0.0) {
          // +0.0 and -0.0 hash the same (consistent with RecursiveEq)
          double pos_zero = 0.0;
          std::memcpy(&bits, &pos_zero, sizeof(bits));
        } else {
          std::memcpy(&bits, &v, sizeof(bits));
        }
        return details::StableHashCombine(static_cast<uint64_t>(ti), bits);
      }
      case TypeIndex::kTVMFFIDataType: {
        DLDataType dt = data->v_dtype;
        uint64_t h =
            details::StableHashCombine(static_cast<uint64_t>(ti), static_cast<uint64_t>(dt.code));
        h = details::StableHashCombine(h, static_cast<uint64_t>(dt.bits));
        h = details::StableHashCombine(h, static_cast<uint64_t>(dt.lanes));
        return h;
      }
      case TypeIndex::kTVMFFIDevice: {
        DLDevice dev = data->v_device;
        uint64_t h = details::StableHashCombine(static_cast<uint64_t>(ti),
                                                static_cast<uint64_t>(dev.device_type));
        h = details::StableHashCombine(h, static_cast<uint64_t>(dev.device_id));
        return h;
      }
      default: {
        return details::StableHashCombine(static_cast<uint64_t>(ti),
                                          static_cast<uint64_t>(data->v_uint64));
      }
    }
  }

  // ---------- String hashing (handles SmallStr cross-variant) ----------

  uint64_t HashString(const Any& value, const TVMFFIAny* data, int32_t ti) {
    const char* ptr;
    size_t len;
    GetStringData(value, data, ti, &ptr, &len);
    // Use kTVMFFIStr as the type key so SmallStr and Str hash the same
    return details::StableHashCombine(static_cast<uint64_t>(TypeIndex::kTVMFFIStr),
                                      details::StableHashBytes(ptr, len));
  }

  // ---------- Bytes hashing (handles SmallBytes cross-variant) ----------

  uint64_t HashBytes(const Any& value, const TVMFFIAny* data, int32_t ti) {
    const char* ptr;
    size_t len;
    GetBytesData(value, data, ti, &ptr, &len);
    // Use kTVMFFIBytes as the type key so SmallBytes and Bytes hash the same
    return details::StableHashCombine(static_cast<uint64_t>(TypeIndex::kTVMFFIBytes),
                                      details::StableHashBytes(ptr, len));
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

  // ---------- Sequence hashing (Array / List) ----------

  template <typename SeqType>
  uint64_t HashSequence(const SeqType& seq) {
    uint64_t h = details::StableHashCombine(seq->GetTypeKeyHash(), seq.size());
    for (size_t i = 0; i < seq.size(); ++i) {
      // Mix element hash with its index for stronger position-dependent hashing.
      h = details::StableHashCombine(h, details::StableHashCombine(HashAny(seq[i]), i));
    }
    return h;
  }

  // ---------- Map / Dict hashing (order-independent) ----------

  template <typename MapType>
  uint64_t HashMap(const MapType& map) {
    uint64_t h = details::StableHashCombine(map->GetTypeKeyHash(), map.size());
    // Sort per-entry hashes then combine sequentially with StableHashCombine.
    // Sorting makes the result order-independent while sequential combining
    // provides full avalanche mixing (unlike XOR or addition).
    std::vector<uint64_t> entry_hashes;
    entry_hashes.reserve(map.size());
    for (const auto& kv : map) {
      entry_hashes.push_back(details::StableHashCombine(HashAny(kv.first), HashAny(kv.second)));
    }
    std::sort(entry_hashes.begin(), entry_hashes.end());
    for (uint64_t eh : entry_hashes) {
      h = details::StableHashCombine(h, eh);
    }
    return h;
  }

  // ---------- Shape hashing ----------

  uint64_t HashShape(const Shape& shape) {
    uint64_t h = details::StableHashCombine(shape->GetTypeKeyHash(), shape.size());
    for (int64_t dim : shape) {
      h = details::StableHashCombine(h, static_cast<uint64_t>(dim));
    }
    return h;
  }

  // ---------- Reflected Object hashing ----------

  uint64_t HashObject(const Object* obj) {
    const TVMFFITypeInfo* type_info = TVMFFIGetTypeInfo(obj->type_index());
    uint64_t h = type_info->type_key_hash;
    reflection::ForEachFieldInfo(type_info, [&](const TVMFFIFieldInfo* finfo) {
      // Skip fields excluded from hashing or comparison
      if (finfo->flags & (kTVMFFIFieldFlagBitMaskHashOff | kTVMFFIFieldFlagBitMaskCompareOff)) {
        return;
      }
      reflection::FieldGetter getter(finfo);
      Any field_value = getter(obj);
      h = details::StableHashCombine(h, HashAny(field_value));
    });
    return h;
  }
};

}  // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("ffi.RecursiveHash", [](const Any& value) -> int64_t {
    RecursiveHasher hasher;
    // Explicitly bitcast uint64_t -> int64_t to avoid overflow error in Any conversion.
    return static_cast<int64_t>(hasher.HashAny(value));
  });
}

}  // namespace ffi
}  // namespace tvm
