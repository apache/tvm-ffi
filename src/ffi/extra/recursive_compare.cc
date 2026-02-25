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
 * \brief Iterative reflection-based recursive comparison (Eq, Lt, Le, Gt, Ge)
 *        with __ffi_eq__/__ffi_compare__ hooks. Raises ValueError on cycles.
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
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace ffi {

namespace {

/*!
 * \brief Iterative three-way recursive comparer.
 *
 * Returns int32_t: -1 (lhs < rhs), 0 (equal), +1 (lhs > rhs).
 *
 * Uses an explicit heap-allocated stack instead of system stack recursion.
 * When eq_only_ is true, type mismatches return non-zero instead of
 * throwing, and Map/Dict ordering is not attempted.
 *
 * Supports custom __ffi_eq__ and __ffi_compare__ hooks.
 */
class RecursiveComparer {
 public:
  explicit RecursiveComparer(bool eq_only) : eq_only_(eq_only) {}

  int32_t CompareAny(const Any& lhs, const Any& rhs) {
    int32_t cmp;
    if (TryCompareImmediate(lhs, rhs, &cmp)) return cmp;
    PushNewFrame(lhs, rhs);
    // Handle early results from eager checks in PushNewFrame (e.g., map size/key mismatch)
    if (has_early_result_) {
      has_early_result_ = false;
      CleanupAllFrames();
      return early_result_;
    }
    return RunLoop();
  }

 private:
  static constexpr size_t kMaxStackDepth = 1 << 20;

  struct PairHash {
    size_t operator()(std::pair<const Object*, const Object*> p) const {
      auto h1 = std::hash<const void*>()(p.first);
      auto h2 = std::hash<const void*>()(p.second);
      return h1 ^ (h2 * 0x9e3779b97f4a7c15ULL + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }
  };

  struct CompareFrame {
    enum Kind : uint8_t { kSequence, kMap, kObject };
    Kind kind;
    std::vector<std::pair<Any, Any>> children;
    size_t child_idx = 0;
    size_t lhs_size = 0;
    size_t rhs_size = 0;
    const Object* lhs_obj = nullptr;
    const Object* rhs_obj = nullptr;
  };

  bool eq_only_;
  std::vector<CompareFrame> stack_;
  std::unordered_set<std::pair<const Object*, const Object*>, PairHash> on_stack_;

  // ---------- Static helpers ----------

  static bool IsStringType(int32_t ti) {
    return ti == TypeIndex::kTVMFFIStr || ti == TypeIndex::kTVMFFISmallStr;
  }

  static bool IsBytesType(int32_t ti) {
    return ti == TypeIndex::kTVMFFIBytes || ti == TypeIndex::kTVMFFISmallBytes;
  }

  // ---------- Immediate (non-recursive) comparison ----------

  /*!
   * \brief Try to compare two values without pushing a frame.
   * \return true if the comparison was done immediately (result stored in *out).
   */
  bool TryCompareImmediate(const Any& lhs, const Any& rhs, int32_t* out) {
    using details::AnyUnsafe;
    const TVMFFIAny* lhs_data = AnyUnsafe::TVMFFIAnyPtrFromAny(lhs);
    const TVMFFIAny* rhs_data = AnyUnsafe::TVMFFIAnyPtrFromAny(rhs);
    int32_t lti = lhs_data->type_index;
    int32_t rti = rhs_data->type_index;

    // Handle None specially
    if (lti == TypeIndex::kTVMFFINone && rti == TypeIndex::kTVMFFINone) {
      *out = 0;
      return true;
    }
    if (lti == TypeIndex::kTVMFFINone) {
      *out = -1;
      return true;
    }
    if (rti == TypeIndex::kTVMFFINone) {
      *out = 1;
      return true;
    }

    // Handle String (Str/SmallStr cross-variant)
    if (IsStringType(lti) && IsStringType(rti)) {
      *out = CompareString(lhs, rhs, lhs_data, rhs_data, lti, rti);
      return true;
    }
    // Handle Bytes (Bytes/SmallBytes cross-variant)
    if (IsBytesType(lti) && IsBytesType(rti)) {
      *out = CompareBytes(lhs, rhs, lhs_data, rhs_data, lti, rti);
      return true;
    }

    // Type mismatch
    if (lti != rti) {
      if (eq_only_) {
        *out = 1;
        return true;
      }
      TVM_FFI_THROW(TypeError) << "Cannot compare values of different types: " << lhs.GetTypeKey()
                               << " vs " << rhs.GetTypeKey();
    }

    // Same type — POD dispatch
    if (lti < TypeIndex::kTVMFFIStaticObjectBegin) {
      *out = ComparePOD(lhs, rhs, lhs_data, rhs_data, lti);
      return true;
    }

    // Object types — check pointer identity first
    const Object* lhs_obj = static_cast<const Object*>(lhs.as<Object>());
    const Object* rhs_obj = static_cast<const Object*>(rhs.as<Object>());
    if (lhs_obj == rhs_obj) {
      *out = 0;
      return true;
    }
    if (lhs_obj == nullptr) {
      *out = -1;
      return true;
    }
    if (rhs_obj == nullptr) {
      *out = 1;
      return true;
    }

    // Shape (no children, always immediate)
    if (lti == TypeIndex::kTVMFFIShape) {
      *out = CompareShape(AnyUnsafe::CopyFromAnyViewAfterCheck<Shape>(lhs),
                          AnyUnsafe::CopyFromAnyViewAfterCheck<Shape>(rhs));
      return true;
    }

    // Cycle detection: pointer pair already being compared → cyclic structure.
    // We raise instead of returning 0 (equal) because the hasher cannot produce
    // consistent hashes for structurally different cyclic graphs, violating
    // the invariant RecursiveEq(a,b) => RecursiveHash(a)==RecursiveHash(b).
    auto pair = std::make_pair(lhs_obj, rhs_obj);
    if (on_stack_.count(pair)) {
      TVM_FFI_THROW(ValueError) << "RecursiveCompare: cyclic reference detected";
    }

    // Check for custom hooks on reflected objects (not containers)
    if (lti >= TypeIndex::kTVMFFIStaticObjectEnd) {
      return TryCustomHook(lhs_obj, rhs_obj, out);
    }

    // Needs a frame (sequence, map)
    return false;
  }

  // ---------- Custom hook dispatch ----------

  bool TryCustomHook(const Object* lhs_obj, const Object* rhs_obj, int32_t* out) {
    // Different type indices
    if (lhs_obj->type_index() != rhs_obj->type_index()) {
      if (eq_only_) {
        *out = 1;
        return true;
      }
      const TVMFFITypeInfo* lhs_info = TVMFFIGetTypeInfo(lhs_obj->type_index());
      const TVMFFITypeInfo* rhs_info = TVMFFIGetTypeInfo(rhs_obj->type_index());
      TVM_FFI_THROW(TypeError) << "Cannot compare objects of different types: "
                               << String(lhs_info->type_key) << " vs "
                               << String(rhs_info->type_key);
    }

    static reflection::TypeAttrColumn eq_column(reflection::type_attr::kEq);
    static reflection::TypeAttrColumn cmp_column(reflection::type_attr::kCompare);

    int32_t ti = lhs_obj->type_index();
    AnyView custom_eq = eq_column[ti];
    AnyView custom_cmp = cmp_column[ti];

    if (eq_only_) {
      if (custom_eq != nullptr) {
        auto pair = std::make_pair(lhs_obj, rhs_obj);
        on_stack_.insert(pair);
        Function hook = custom_eq.cast<Function>();
        Function fn_eq = CreateFnEq();
        bool result = hook(lhs_obj, rhs_obj, fn_eq).cast<bool>();
        on_stack_.erase(pair);
        *out = result ? 0 : 1;
        return true;
      }
      if (custom_cmp != nullptr) {
        auto pair = std::make_pair(lhs_obj, rhs_obj);
        on_stack_.insert(pair);
        Function hook = custom_cmp.cast<Function>();
        Function fn_cmp = CreateFnCompare();
        int32_t result = hook(lhs_obj, rhs_obj, fn_cmp).cast<int32_t>();
        on_stack_.erase(pair);
        *out = result;
        return true;
      }
    } else {
      if (custom_cmp != nullptr) {
        auto pair = std::make_pair(lhs_obj, rhs_obj);
        on_stack_.insert(pair);
        Function hook = custom_cmp.cast<Function>();
        Function fn_cmp = CreateFnCompare();
        int32_t result = hook(lhs_obj, rhs_obj, fn_cmp).cast<int32_t>();
        on_stack_.erase(pair);
        *out = result;
        return true;
      }
    }
    // No hook — will push a frame for reflection
    return false;
  }

  // ---------- Frame creation ----------

  void PushNewFrame(const Any& lhs, const Any& rhs) {
    if (stack_.size() >= kMaxStackDepth) {
      TVM_FFI_THROW(ValueError) << "RecursiveCompare: maximum stack depth (" << kMaxStackDepth
                                << ") exceeded";
    }
    using details::AnyUnsafe;
    const TVMFFIAny* lhs_data = AnyUnsafe::TVMFFIAnyPtrFromAny(lhs);
    int32_t lti = lhs_data->type_index;
    const Object* lhs_obj = static_cast<const Object*>(lhs.as<Object>());
    const Object* rhs_obj = static_cast<const Object*>(rhs.as<Object>());
    auto pair = std::make_pair(lhs_obj, rhs_obj);
    on_stack_.insert(pair);

    switch (lti) {
      case TypeIndex::kTVMFFIStr: {
        int32_t r = CompareString(lhs, rhs, lhs_data, AnyUnsafe::TVMFFIAnyPtrFromAny(rhs), lti,
                                  lhs_data->type_index);
        on_stack_.erase(pair);
        if (!stack_.empty() && r != 0) {
          // propagate
        }
        // This case shouldn't normally happen (strings are immediate)
        return;
      }
      case TypeIndex::kTVMFFIArray: {
        auto lhs_seq = AnyUnsafe::CopyFromAnyViewAfterCheck<Array<Any>>(lhs);
        auto rhs_seq = AnyUnsafe::CopyFromAnyViewAfterCheck<Array<Any>>(rhs);
        PushSequenceFrame(lhs_seq, rhs_seq, lhs_obj, rhs_obj);
        return;
      }
      case TypeIndex::kTVMFFIList: {
        auto lhs_seq = AnyUnsafe::CopyFromAnyViewAfterCheck<List<Any>>(lhs);
        auto rhs_seq = AnyUnsafe::CopyFromAnyViewAfterCheck<List<Any>>(rhs);
        PushSequenceFrame(lhs_seq, rhs_seq, lhs_obj, rhs_obj);
        return;
      }
      case TypeIndex::kTVMFFIMap: {
        auto lhs_map = AnyUnsafe::CopyFromAnyViewAfterCheck<Map<Any, Any>>(lhs);
        auto rhs_map = AnyUnsafe::CopyFromAnyViewAfterCheck<Map<Any, Any>>(rhs);
        PushMapFrame(lhs_map, rhs_map, lhs_obj, rhs_obj);
        return;
      }
      case TypeIndex::kTVMFFIDict: {
        auto lhs_map = AnyUnsafe::CopyFromAnyViewAfterCheck<Dict<Any, Any>>(lhs);
        auto rhs_map = AnyUnsafe::CopyFromAnyViewAfterCheck<Dict<Any, Any>>(rhs);
        PushMapFrame(lhs_map, rhs_map, lhs_obj, rhs_obj);
        return;
      }
      default: {
        PushObjectFrame(lhs_obj, rhs_obj);
        return;
      }
    }
  }

  template <typename SeqType>
  void PushSequenceFrame(const SeqType& lhs, const SeqType& rhs, const Object* lhs_obj,
                         const Object* rhs_obj) {
    CompareFrame frame;
    frame.kind = CompareFrame::kSequence;
    frame.lhs_size = lhs.size();
    frame.rhs_size = rhs.size();
    frame.lhs_obj = lhs_obj;
    frame.rhs_obj = rhs_obj;
    size_t min_len = std::min(lhs.size(), rhs.size());
    frame.children.reserve(min_len);
    for (size_t i = 0; i < min_len; ++i) {
      frame.children.emplace_back(lhs[i], rhs[i]);
    }
    stack_.push_back(std::move(frame));
  }

  template <typename MapType>
  void PushMapFrame(const MapType& lhs, const MapType& rhs, const Object* lhs_obj,
                    const Object* rhs_obj) {
    // Size mismatch check
    if (lhs.size() != rhs.size()) {
      on_stack_.erase(std::make_pair(lhs_obj, rhs_obj));
      if (eq_only_) {
        // Feed non-zero result directly
        FeedNonZeroResult(1);
        return;
      }
      TVM_FFI_THROW(TypeError) << "Cannot order Map/Dict values";
    }
    // Key match check and collect value pairs
    CompareFrame frame;
    frame.kind = CompareFrame::kMap;
    frame.lhs_size = lhs.size();
    frame.rhs_size = rhs.size();
    frame.lhs_obj = lhs_obj;
    frame.rhs_obj = rhs_obj;
    frame.children.reserve(lhs.size());
    for (const auto& kv : lhs) {
      auto it = rhs.find(kv.first);
      if (it == rhs.end()) {
        on_stack_.erase(std::make_pair(lhs_obj, rhs_obj));
        if (eq_only_) {
          FeedNonZeroResult(1);
          return;
        }
        TVM_FFI_THROW(TypeError) << "Cannot order Map/Dict values";
      }
      frame.children.emplace_back(kv.second, (*it).second);
    }
    stack_.push_back(std::move(frame));
  }

  void PushObjectFrame(const Object* lhs, const Object* rhs) {
    if (lhs->type_index() != rhs->type_index()) {
      auto pair = std::make_pair(lhs, rhs);
      on_stack_.erase(pair);
      if (eq_only_) {
        FeedNonZeroResult(1);
        return;
      }
      const TVMFFITypeInfo* lhs_info = TVMFFIGetTypeInfo(lhs->type_index());
      const TVMFFITypeInfo* rhs_info = TVMFFIGetTypeInfo(rhs->type_index());
      TVM_FFI_THROW(TypeError) << "Cannot compare objects of different types: "
                               << String(lhs_info->type_key) << " vs "
                               << String(rhs_info->type_key);
    }
    const TVMFFITypeInfo* type_info = TVMFFIGetTypeInfo(lhs->type_index());
    CompareFrame frame;
    frame.kind = CompareFrame::kObject;
    frame.lhs_obj = lhs;
    frame.rhs_obj = rhs;
    reflection::ForEachFieldInfo(type_info, [&](const TVMFFIFieldInfo* finfo) {
      if (finfo->flags & kTVMFFIFieldFlagBitMaskCompareOff) return;
      reflection::FieldGetter getter(finfo);
      frame.children.emplace_back(getter(lhs), getter(rhs));
    });
    stack_.push_back(std::move(frame));
  }

  // ---------- Propagation ----------

  /*!
   * \brief Feed a non-zero result to the parent frame when a frame creation
   *        determined inequality eagerly (e.g., map size mismatch).
   */
  void FeedNonZeroResult(int32_t result) {
    // This is used during frame creation when we determine the result
    // without actually pushing a frame. We need to propagate this up.
    // We store it by pushing a sentinel empty frame that will immediately
    // resolve in RunLoop. Instead, we handle this by using the early_result_ mechanism.
    early_result_ = result;
    has_early_result_ = true;
  }

  int32_t early_result_ = 0;
  bool has_early_result_ = false;

  /*!
   * \brief When a non-zero comparison result is found, propagate upward.
   *        For ordering mode, check that no ancestor is a map frame.
   */
  int32_t PropagateNonZero(int32_t result) {
    if (!eq_only_) {
      for (auto it = stack_.rbegin(); it != stack_.rend(); ++it) {
        if (it->kind == CompareFrame::kMap) {
          CleanupAllFrames();
          TVM_FFI_THROW(TypeError) << "Cannot order Map/Dict values";
        }
      }
    }
    CleanupAllFrames();
    return result;
  }

  void CleanupAllFrames() {
    for (auto& f : stack_) {
      if (f.lhs_obj != nullptr && f.rhs_obj != nullptr) {
        on_stack_.erase(std::make_pair(f.lhs_obj, f.rhs_obj));
      }
    }
    stack_.clear();
  }

  // ---------- Main iterative loop ----------

  int32_t RunLoop() {
    while (true) {
      // Check for early result from eager frame creation (e.g., map size/key mismatch)
      if (has_early_result_) {
        has_early_result_ = false;
        int32_t r = early_result_;
        if (r != 0) return PropagateNonZero(r);
      }
      if (stack_.empty()) return 0;
      CompareFrame& f = stack_.back();
      bool pushed_child = false;
      while (f.child_idx < f.children.size()) {
        auto& [child_lhs, child_rhs] = f.children[f.child_idx++];
        int32_t cmp;
        if (TryCompareImmediate(child_lhs, child_rhs, &cmp)) {
          if (cmp != 0) return PropagateNonZero(cmp);
        } else {
          PushNewFrame(child_lhs, child_rhs);
          // Check for early result from PushNewFrame (e.g., map size mismatch)
          if (has_early_result_) break;
          pushed_child = true;
          break;
        }
      }
      if (pushed_child) continue;
      if (has_early_result_) continue;
      // Frame completed — check size comparison for sequences
      int32_t size_cmp = 0;
      if (f.kind == CompareFrame::kSequence) {
        if (f.lhs_size < f.rhs_size) {
          size_cmp = -1;
        } else if (f.lhs_size > f.rhs_size) {
          size_cmp = 1;
        }
      }
      // Cleanup
      if (f.lhs_obj != nullptr && f.rhs_obj != nullptr) {
        on_stack_.erase(std::make_pair(f.lhs_obj, f.rhs_obj));
      }
      stack_.pop_back();
      if (size_cmp != 0) {
        if (stack_.empty()) return size_cmp;
        return PropagateNonZero(size_cmp);
      }
    }
    return 0;
  }

  // ---------- Custom hook callbacks ----------

  Function CreateFnEq() {
    return Function::FromTyped([this](AnyView l, AnyView r) -> bool {
      Any lhs(l), rhs(r);
      int32_t cmp;
      if (TryCompareImmediate(lhs, rhs, &cmp)) return cmp == 0;
      // Save/restore isolates the explicit stack for nested hook calls.
      // PushNewFrame below re-populates stack_ before RunLoop uses it.
      std::vector<CompareFrame> saved;
      saved.swap(stack_);
      bool saved_early = has_early_result_;
      int32_t saved_result = early_result_;
      has_early_result_ = false;
      PushNewFrame(lhs, rhs);
      cmp = RunLoop();
      stack_.swap(saved);
      has_early_result_ = saved_early;
      early_result_ = saved_result;
      return cmp == 0;
    });
  }

  Function CreateFnCompare() {
    return Function::FromTyped([this](AnyView l, AnyView r) -> int32_t {
      Any lhs(l), rhs(r);
      int32_t cmp;
      if (TryCompareImmediate(lhs, rhs, &cmp)) return cmp;
      // Save/restore isolates the explicit stack for nested hook calls.
      std::vector<CompareFrame> saved;
      saved.swap(stack_);
      bool saved_early = has_early_result_;
      int32_t saved_result = early_result_;
      has_early_result_ = false;
      PushNewFrame(lhs, rhs);
      cmp = RunLoop();
      stack_.swap(saved);
      has_early_result_ = saved_early;
      early_result_ = saved_result;
      return cmp;
    });
  }

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
        if (std::isnan(a) && std::isnan(b)) {
          if (eq_only_) return 0;
          TVM_FFI_THROW(TypeError) << "Cannot order NaN values";
        }
        if (std::isnan(a) || std::isnan(b)) {
          if (eq_only_) return 1;
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

  static int32_t CompareString(const Any& lhs, const Any& rhs, const TVMFFIAny* lhs_data,
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

  static int32_t CompareBytes(const Any& lhs, const Any& rhs, const TVMFFIAny* lhs_data,
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

  // ---------- Shape comparison ----------

  static int32_t CompareShape(const Shape& lhs, const Shape& rhs) {
    size_t min_len = std::min(lhs.size(), rhs.size());
    for (size_t i = 0; i < min_len; ++i) {
      if (lhs[i] < rhs[i]) return -1;
      if (lhs[i] > rhs[i]) return 1;
    }
    if (lhs.size() < rhs.size()) return -1;
    if (lhs.size() > rhs.size()) return 1;
    return 0;
  }
};

}  // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::EnsureTypeAttrColumn(refl::type_attr::kEq);
  refl::EnsureTypeAttrColumn(refl::type_attr::kCompare);
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
