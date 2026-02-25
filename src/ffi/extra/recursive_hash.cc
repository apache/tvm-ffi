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
 * \brief Iterative reflection-based recursive hash with __ffi_hash__ hooks.
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
#include <utility>
#include <vector>

namespace tvm {
namespace ffi {

namespace {

/*!
 * \brief Iterative reflection-based recursive hasher.
 *
 * Uses an explicit heap-allocated stack instead of system stack recursion,
 * so deep object graphs don't overflow the system stack. Supports custom
 * __ffi_hash__ hooks following the __ffi_repr__ pattern.
 *
 * Computes a deterministic hash consistent with RecursiveEq:
 *   RecursiveEq(a, b) => RecursiveHash(a) == RecursiveHash(b)
 */
class RecursiveHasher {
 public:
  uint64_t HashAny(const Any& value) {
    uint64_t h;
    if (TryHashImmediate(value, &h)) return h;
    PushFrame(value);
    return RunLoop();
  }

 private:
  static constexpr size_t kMaxStackDepth = 1 << 20;

  struct HashFrame {
    enum Kind : uint8_t { kSequence, kMap, kObject };
    Kind kind;
    uint64_t hash;
    const Object* obj;
    std::vector<Any> children;
    size_t child_idx = 0;
    size_t seq_index = 0;
    // Map-specific:
    std::vector<uint64_t> entry_hashes;
    bool in_key = true;
    uint64_t key_hash = 0;
  };

  std::vector<HashFrame> stack_;
  std::unordered_set<const Object*> on_stack_;
  std::unordered_map<const Object*, uint64_t> memo_;

  // ---------- Immediate (non-recursive) hashing ----------

  static bool IsStringType(int32_t ti) {
    return ti == TypeIndex::kTVMFFIStr || ti == TypeIndex::kTVMFFISmallStr;
  }

  static bool IsBytesType(int32_t ti) {
    return ti == TypeIndex::kTVMFFIBytes || ti == TypeIndex::kTVMFFISmallBytes;
  }

  /*!
   * \brief Try to hash a value without pushing a frame.
   * \return true if the hash was computed immediately (stored in *out).
   */
  bool TryHashImmediate(const Any& value, uint64_t* out) {
    using details::AnyUnsafe;
    const TVMFFIAny* data = AnyUnsafe::TVMFFIAnyPtrFromAny(value);
    int32_t ti = data->type_index;

    // None
    if (ti == TypeIndex::kTVMFFINone) {
      *out = details::StableHashCombine(uint64_t{0}, uint64_t{0});
      return true;
    }
    // String (Str/SmallStr cross-variant)
    if (IsStringType(ti)) {
      *out = HashString(value, data, ti);
      return true;
    }
    // Bytes (Bytes/SmallBytes cross-variant)
    if (IsBytesType(ti)) {
      *out = HashBytes(value, data, ti);
      return true;
    }
    // POD types
    if (ti < TypeIndex::kTVMFFIStaticObjectBegin) {
      *out = HashPOD(value, data, ti);
      return true;
    }
    // Object types
    const Object* obj = static_cast<const Object*>(value.as<Object>());
    if (obj == nullptr) {
      *out = details::StableHashCombine(uint64_t{0}, uint64_t{0});
      return true;
    }
    // Return memoized hash if already fully hashed.
    auto memo_it = memo_.find(obj);
    if (memo_it != memo_.end()) {
      *out = memo_it->second;
      return true;
    }
    // Cycle detection: if on the call stack, return sentinel.
    if (on_stack_.count(obj)) {
      *out = TVMFFIGetTypeInfo(obj->type_index())->type_key_hash;
      return true;
    }
    // Shape is always immediate (no children)
    if (ti == TypeIndex::kTVMFFIShape) {
      uint64_t h = HashShape(AnyUnsafe::CopyFromAnyViewAfterCheck<Shape>(value));
      memo_[obj] = h;
      *out = h;
      return true;
    }
    // Check for custom __ffi_hash__ hook
    static reflection::TypeAttrColumn hash_column(reflection::type_attr::kHash);
    AnyView custom = hash_column[obj->type_index()];
    if (custom != nullptr) {
      on_stack_.insert(obj);
      Function hook = custom.cast<Function>();
      Function fn_hash = CreateFnHash();
      int64_t r = hook(obj, fn_hash).cast<int64_t>();
      uint64_t h = static_cast<uint64_t>(r);
      memo_[obj] = h;
      on_stack_.erase(obj);
      *out = h;
      return true;
    }
    // For reflected types (not built-in containers), error if the type has
    // __ffi_eq__ or __ffi_compare__ but no __ffi_hash__.  Custom equality
    // that ignores fields makes the default field-by-field hash inconsistent.
    if (ti >= TypeIndex::kTVMFFIStaticObjectEnd) {
      static reflection::TypeAttrColumn eq_column(reflection::type_attr::kEq);
      static reflection::TypeAttrColumn cmp_column(reflection::type_attr::kCompare);
      if (eq_column[obj->type_index()] != nullptr || cmp_column[obj->type_index()] != nullptr) {
        const TVMFFITypeInfo* info = TVMFFIGetTypeInfo(ti);
        TVM_FFI_THROW(ValueError)
            << "RecursiveHash: type '" << String(info->type_key)
            << "' defines __ffi_eq__ or __ffi_compare__ but not __ffi_hash__. "
            << "Add a __ffi_hash__ hook to maintain the invariant "
            << "RecursiveEq(a,b) => RecursiveHash(a)==RecursiveHash(b).";
      }
    }
    // Needs a frame (sequence, map, or reflected object)
    return false;
  }

  // ---------- Frame creation ----------

  void PushFrame(const Any& value) {
    if (stack_.size() >= kMaxStackDepth) {
      TVM_FFI_THROW(ValueError) << "RecursiveHash: maximum stack depth (" << kMaxStackDepth
                                << ") exceeded";
    }
    using details::AnyUnsafe;
    const TVMFFIAny* data = AnyUnsafe::TVMFFIAnyPtrFromAny(value);
    int32_t ti = data->type_index;
    const Object* obj = static_cast<const Object*>(value.as<Object>());
    on_stack_.insert(obj);

    switch (ti) {
      case TypeIndex::kTVMFFIStr: {
        // Already handled in TryHashImmediate — should not reach here.
        // But handle for safety: Str on heap is an object.
        uint64_t h = HashString(value, data, ti);
        memo_[obj] = h;
        on_stack_.erase(obj);
        // Instead of pushing, feed result to parent.
        if (!stack_.empty()) {
          FeedChildHash(stack_.back(), h);
        }
        return;
      }
      case TypeIndex::kTVMFFIBytes: {
        uint64_t h = HashBytes(value, data, ti);
        memo_[obj] = h;
        on_stack_.erase(obj);
        if (!stack_.empty()) {
          FeedChildHash(stack_.back(), h);
        }
        return;
      }
      case TypeIndex::kTVMFFIArray: {
        auto seq = AnyUnsafe::CopyFromAnyViewAfterCheck<Array<Any>>(value);
        HashFrame frame;
        frame.kind = HashFrame::kSequence;
        frame.hash = details::StableHashCombine(seq->GetTypeKeyHash(), seq.size());
        frame.obj = obj;
        frame.children.reserve(seq.size());
        for (const auto& elem : seq) {
          frame.children.push_back(elem);
        }
        stack_.push_back(std::move(frame));
        return;
      }
      case TypeIndex::kTVMFFIList: {
        auto seq = AnyUnsafe::CopyFromAnyViewAfterCheck<List<Any>>(value);
        HashFrame frame;
        frame.kind = HashFrame::kSequence;
        frame.hash = details::StableHashCombine(seq->GetTypeKeyHash(), seq.size());
        frame.obj = obj;
        frame.children.reserve(seq.size());
        for (const auto& elem : seq) {
          frame.children.push_back(elem);
        }
        stack_.push_back(std::move(frame));
        return;
      }
      case TypeIndex::kTVMFFIMap: {
        auto map = AnyUnsafe::CopyFromAnyViewAfterCheck<Map<Any, Any>>(value);
        HashFrame frame;
        frame.kind = HashFrame::kMap;
        frame.hash = details::StableHashCombine(map->GetTypeKeyHash(), map.size());
        frame.obj = obj;
        frame.entry_hashes.reserve(map.size());
        frame.children.reserve(map.size() * 2);
        for (const auto& kv : map) {
          frame.children.push_back(kv.first);
          frame.children.push_back(kv.second);
        }
        stack_.push_back(std::move(frame));
        return;
      }
      case TypeIndex::kTVMFFIDict: {
        auto map = AnyUnsafe::CopyFromAnyViewAfterCheck<Dict<Any, Any>>(value);
        HashFrame frame;
        frame.kind = HashFrame::kMap;
        frame.hash = details::StableHashCombine(map->GetTypeKeyHash(), map.size());
        frame.obj = obj;
        frame.entry_hashes.reserve(map.size());
        frame.children.reserve(map.size() * 2);
        for (const auto& kv : map) {
          frame.children.push_back(kv.first);
          frame.children.push_back(kv.second);
        }
        stack_.push_back(std::move(frame));
        return;
      }
      default: {
        // Reflected object
        PushObjectFrame(obj);
        return;
      }
    }
  }

  void PushObjectFrame(const Object* obj) {
    const TVMFFITypeInfo* type_info = TVMFFIGetTypeInfo(obj->type_index());
    HashFrame frame;
    frame.kind = HashFrame::kObject;
    frame.hash = type_info->type_key_hash;
    frame.obj = obj;
    reflection::ForEachFieldInfo(type_info, [&](const TVMFFIFieldInfo* finfo) {
      if (finfo->flags & (kTVMFFIFieldFlagBitMaskHashOff | kTVMFFIFieldFlagBitMaskCompareOff)) {
        return;
      }
      reflection::FieldGetter getter(finfo);
      frame.children.push_back(getter(obj));
    });
    stack_.push_back(std::move(frame));
  }

  // ---------- Child hash accumulation ----------

  void FeedChildHash(HashFrame& f, uint64_t h) {
    switch (f.kind) {
      case HashFrame::kSequence: {
        f.hash = details::StableHashCombine(f.hash, details::StableHashCombine(h, f.seq_index++));
        break;
      }
      case HashFrame::kMap: {
        if (f.in_key) {
          f.key_hash = h;
          f.in_key = false;
        } else {
          f.entry_hashes.push_back(details::StableHashCombine(f.key_hash, h));
          f.in_key = true;
        }
        break;
      }
      case HashFrame::kObject: {
        f.hash = details::StableHashCombine(f.hash, h);
        break;
      }
    }
  }

  // ---------- Frame finalization ----------

  uint64_t FinalizeFrame(HashFrame& f) {
    if (f.kind == HashFrame::kMap) {
      std::sort(f.entry_hashes.begin(), f.entry_hashes.end());
      for (uint64_t eh : f.entry_hashes) {
        f.hash = details::StableHashCombine(f.hash, eh);
      }
    }
    return f.hash;
  }

  // ---------- Main iterative loop ----------

  uint64_t RunLoop() {
    while (!stack_.empty()) {
      HashFrame& f = stack_.back();
      bool pushed_child = false;
      while (f.child_idx < f.children.size()) {
        Any& child = f.children[f.child_idx++];
        uint64_t h;
        if (TryHashImmediate(child, &h)) {
          FeedChildHash(f, h);
        } else {
          PushFrame(child);
          pushed_child = true;
          break;
        }
      }
      if (pushed_child) continue;
      // Frame completed
      uint64_t result = FinalizeFrame(f);
      const Object* obj = f.obj;
      stack_.pop_back();
      if (obj != nullptr) {
        memo_[obj] = result;
        on_stack_.erase(obj);
      }
      if (stack_.empty()) return result;
      FeedChildHash(stack_.back(), result);
    }
    TVM_FFI_UNREACHABLE();
  }

  // ---------- Custom __ffi_hash__ callback ----------

  Function CreateFnHash() {
    return Function::FromTyped([this](AnyView value) -> int64_t {
      Any v(value);
      uint64_t h;
      if (TryHashImmediate(v, &h)) return static_cast<int64_t>(h);
      // Save/restore isolates the explicit stack for nested hook calls.
      // PushFrame below re-populates stack_ before RunLoop uses it.
      std::vector<HashFrame> saved;
      saved.swap(stack_);
      PushFrame(v);
      h = RunLoop();
      stack_.swap(saved);
      return static_cast<int64_t>(h);
    });
  }

  // ---------- POD hashing ----------

  static uint64_t HashPOD(const Any& value, const TVMFFIAny* data, int32_t ti) {
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
          double canonical = std::numeric_limits<double>::quiet_NaN();
          std::memcpy(&bits, &canonical, sizeof(bits));
        } else if (v == 0.0) {
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

  static uint64_t HashString(const Any& value, const TVMFFIAny* data, int32_t ti) {
    const char* ptr;
    size_t len;
    GetStringData(value, data, ti, &ptr, &len);
    return details::StableHashCombine(static_cast<uint64_t>(TypeIndex::kTVMFFIStr),
                                      details::StableHashBytes(ptr, len));
  }

  // ---------- Bytes hashing (handles SmallBytes cross-variant) ----------

  static uint64_t HashBytes(const Any& value, const TVMFFIAny* data, int32_t ti) {
    const char* ptr;
    size_t len;
    GetBytesData(value, data, ti, &ptr, &len);
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

  // ---------- Shape hashing ----------

  static uint64_t HashShape(const Shape& shape) {
    uint64_t h = details::StableHashCombine(shape->GetTypeKeyHash(), shape.size());
    for (int64_t dim : shape) {
      h = details::StableHashCombine(h, static_cast<uint64_t>(dim));
    }
    return h;
  }
};

}  // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::EnsureTypeAttrColumn(refl::type_attr::kHash);
  refl::GlobalDef().def("ffi.RecursiveHash", [](const Any& value) -> int64_t {
    RecursiveHasher hasher;
    return static_cast<int64_t>(hasher.HashAny(value));
  });
}

}  // namespace ffi
}  // namespace tvm
