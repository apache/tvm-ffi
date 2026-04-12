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
 * \file src/ffi/extra/dataclass.cc
 * \brief Reflection-based dataclass operations:
 *        deep copy, repr printing, recursive hash, recursive compare.
 */
#include <tvm/ffi/any.h>
#include <tvm/ffi/base_details.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/dict.h>
#include <tvm/ffi/container/list.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/container/shape.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/dataclass.h>
#include <tvm/ffi/reflection/accessor.h>
#include <tvm/ffi/reflection/creator.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace ffi {

namespace refl = ::tvm::ffi::reflection;

// ============================================================================
// Shared utilities, CRTP base, and all operation classes
// ============================================================================

namespace {

// ---------- Shared traversal utilities ----------

/*! \brief Maximum traversal stack depth for iterative traversals. */
constexpr size_t kMaxTraversalStackDepth = 1 << 20;

/*! \brief Return true for Str or SmallStr type indices. */
bool IsStringType(int32_t ti) {
  return ti == TypeIndex::kTVMFFIStr || ti == TypeIndex::kTVMFFISmallStr;
}

/*! \brief Return true for Bytes or SmallBytes type indices. */
bool IsBytesType(int32_t ti) {
  return ti == TypeIndex::kTVMFFIBytes || ti == TypeIndex::kTVMFFISmallBytes;
}

/*! \brief Extract raw pointer and length from a String or SmallStr value. */
void GetStringData(const Any& val, const TVMFFIAny* data, int32_t ti, const char** out_ptr,
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

/*! \brief Extract raw pointer and length from a Bytes or SmallBytes value. */
void GetBytesData(const Any& val, const TVMFFIAny* data, int32_t ti, const char** out_ptr,
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

// ---------- CRTP base for object-graph DFS ----------

/*!
 * \brief Common frame base for single-value DFS traversals (hash, repr, copy).
 */
struct FrameBase {
  enum Kind : uint8_t { kSequence, kMap, kObject };
  Kind kind;
  int32_t type_index;
  const Object* obj;
  std::vector<Any> children;
  std::vector<const TVMFFIFieldInfo*> field_infos;  // kObject only
  size_t child_idx = 0;
  size_t container_size = 0;
  size_t NumChildren() const { return children.size(); }
};

/*!
 * \brief CRTP base that owns the iterative DFS RunLoop and container dispatch.
 *
 * \tparam Derived   Concrete operation class.
 * \tparam FrameT    Frame type (must provide child_idx and NumChildren()).
 * \tparam ResultT   Result type produced per frame.
 */
template <typename Derived, typename FrameT, typename ResultT>
class ObjectGraphDFS {
  friend Derived;
  ObjectGraphDFS() = default;

 protected:
  std::vector<FrameT> stack_;

  Derived& self() { return static_cast<Derived&>(*this); }

  ResultT RunLoop() {
    while (!stack_.empty()) {
      auto& f = stack_.back();
      bool pushed = false;
      while (f.child_idx < f.NumChildren()) {
        size_t idx = f.child_idx++;
        ResultT r{};
        if (self().TryVisitChild(f, idx, &r)) {
          if (self().FeedChild(f, r)) {
            return self().OnTerminate(std::move(r));
          }
        } else {
          auto maybe = self().PushChildFrame(f, idx);
          if (maybe.has_value()) {
            if (self().FeedChild(f, *maybe)) {
              return self().OnTerminate(std::move(*maybe));
            }
          } else {
            pushed = true;
            break;
          }
        }
      }
      if (pushed) continue;
      ResultT result = self().FinalizeFrame(f);
      self().OnFrameComplete(f);
      stack_.pop_back();
      if (stack_.empty()) return result;
      if (self().FeedChild(stack_.back(), result)) {
        return self().OnTerminate(std::move(result));
      }
    }
    TVM_FFI_UNREACHABLE();
  }

  void EnumerateChildren(FrameBase& frame, const Any& value, const Object* obj, int32_t ti) {
    using details::AnyUnsafe;
    switch (ti) {
      case TypeIndex::kTVMFFIArray: {
        auto seq = AnyUnsafe::CopyFromAnyViewAfterCheck<Array<Any>>(value);
        frame.kind = FrameBase::kSequence;
        frame.container_size = seq.size();
        frame.children.reserve(seq.size());
        for (const auto& elem : seq) frame.children.push_back(elem);
        return;
      }
      case TypeIndex::kTVMFFIList: {
        auto seq = AnyUnsafe::CopyFromAnyViewAfterCheck<List<Any>>(value);
        frame.kind = FrameBase::kSequence;
        frame.container_size = seq.size();
        frame.children.reserve(seq.size());
        for (const auto& elem : seq) frame.children.push_back(elem);
        return;
      }
      case TypeIndex::kTVMFFIMap: {
        auto map = AnyUnsafe::CopyFromAnyViewAfterCheck<Map<Any, Any>>(value);
        frame.kind = FrameBase::kMap;
        frame.container_size = map.size();
        frame.children.reserve(map.size() * 2);
        for (const auto& kv : map) {
          frame.children.push_back(kv.first);
          frame.children.push_back(kv.second);
        }
        return;
      }
      case TypeIndex::kTVMFFIDict: {
        auto map = AnyUnsafe::CopyFromAnyViewAfterCheck<Dict<Any, Any>>(value);
        frame.kind = FrameBase::kMap;
        frame.container_size = map.size();
        frame.children.reserve(map.size() * 2);
        for (const auto& kv : map) {
          frame.children.push_back(kv.first);
          frame.children.push_back(kv.second);
        }
        return;
      }
      default: {
        frame.kind = FrameBase::kObject;
        const TVMFFITypeInfo* type_info = TVMFFIGetTypeInfo(ti);
        uint32_t skip = self().GetFieldSkipMask();
        refl::ForEachFieldInfo(type_info, [&](const TVMFFIFieldInfo* finfo) {
          if (finfo->flags & skip) return;
          refl::FieldGetter getter(finfo);
          frame.children.push_back(getter(obj));
          frame.field_infos.push_back(finfo);
        });
        frame.container_size = frame.field_infos.size();
        return;
      }
    }
  }

  void PushFrame(const Any& value) {
    if (this->stack_.size() >= kMaxTraversalStackDepth) {
      TVM_FFI_THROW(ValueError) << "ObjectGraphDFS: maximum stack depth ("
                                << kMaxTraversalStackDepth << ") exceeded";
    }
    const Object* obj = static_cast<const Object*>(value.as<Object>());
    int32_t ti = obj->type_index();
    self().OnEnter(obj);
    FrameT frame;
    frame.obj = obj;
    frame.type_index = ti;
    EnumerateChildren(frame, value, obj, ti);
    self().OnFrameInit(frame);
    this->stack_.push_back(std::move(frame));
  }
};

// ---------- Deep Copy ----------

struct CopyFrame : FrameBase {
  Any copy;
  std::vector<Any> resolved;  // Array/Map: accumulated resolved children
  bool is_key = true;         // Dict/Map: tracking key vs value
  Any current_key;            // Dict: last resolved key
  size_t feed_idx = 0;        // Object: field setter index
};

/*!
 * \brief Iterative DFS deep copier (CRTP-based).
 *
 * - Mutable containers (List/Dict) and reflected objects register their copy
 *   in copy_map_ before resolving children (enables cyclic back-references).
 * - Immutable containers (Array/Map) mark in_progress_ and build from resolved
 *   children in FinalizeFrame.
 * - Deferred fixup pass replaces stale placeholder references in mutable
 *   containers after all copies are fully constructed.
 */
class ObjectDeepCopier : public ObjectGraphDFS<ObjectDeepCopier, CopyFrame, Any> {
 public:
  explicit ObjectDeepCopier(refl::TypeAttrColumn* column) : column_(column) {}

  Any Run(const Any& value) {
    if (value.type_index() < TypeIndex::kTVMFFIStaticObjectBegin) return value;
    Any result;
    if (TryCopyImmediate(value, &result)) {
      // Immediate (no children)
    } else {
      this->PushFrame(value);
      result = this->RunLoop();
    }
    if (has_deferred_) {
      FixupDeferredReferences();
    }
    return result;
  }

  // ---------- CRTP customization points ----------

  uint32_t GetFieldSkipMask() { return 0; }

  void OnEnter(const Object* obj) {
    // Tracking is done in OnFrameInit per container type.
  }

  void OnFrameInit(CopyFrame& f) {
    int32_t ti = f.type_index;
    switch (ti) {
      case TypeIndex::kTVMFFIArray: {
        in_progress_.insert(f.obj);
        break;
      }
      case TypeIndex::kTVMFFIList: {
        List<Any> new_list;
        new_list.reserve(static_cast<int64_t>(f.container_size));
        f.copy = new_list;
        copy_map_[f.obj] = f.copy;
        break;
      }
      case TypeIndex::kTVMFFIMap: {
        in_progress_.insert(f.obj);
        break;
      }
      case TypeIndex::kTVMFFIDict: {
        Dict<Any, Any> new_dict;
        f.copy = new_dict;
        copy_map_[f.obj] = f.copy;
        break;
      }
      default: {
        // Reflected object: shallow-copy and register
        const TVMFFITypeInfo* type_info = TVMFFIGetTypeInfo(ti);
        TVM_FFI_ICHECK((*column_)[ti] != nullptr)
            << "Cannot deep copy object of type \""
            << std::string_view(type_info->type_key.data, type_info->type_key.size)
            << "\" because it is not copy-constructible";
        Function copy_fn = (*column_)[ti].cast<Function>();
        f.copy = copy_fn(f.obj);
        copy_map_[f.obj] = f.copy;
        break;
      }
    }
  }

  bool TryVisitChild(CopyFrame& f, size_t idx, Any* out) {
    return TryCopyImmediate(f.children[idx], out);
  }

  std::optional<Any> PushChildFrame(CopyFrame& f, size_t idx) {
    this->PushFrame(f.children[idx]);
    return std::nullopt;
  }

  bool FeedChild(CopyFrame& f, Any resolved) {
    int32_t ti = f.type_index;
    switch (ti) {
      case TypeIndex::kTVMFFIList: {
        f.copy.cast<List<Any>>().push_back(resolved);
        break;
      }
      case TypeIndex::kTVMFFIArray: {
        f.resolved.push_back(std::move(resolved));
        break;
      }
      case TypeIndex::kTVMFFIDict: {
        if (f.is_key) {
          f.current_key = std::move(resolved);
          f.is_key = false;
        } else {
          f.copy.cast<Dict<Any, Any>>().Set(f.current_key, resolved);
          f.is_key = true;
        }
        break;
      }
      case TypeIndex::kTVMFFIMap: {
        f.resolved.push_back(std::move(resolved));
        break;
      }
      default: {
        // Reflected object: set field if changed
        const Any& original = f.children[f.feed_idx];
        if (!original.same_as(resolved)) {
          refl::FieldSetter setter(f.field_infos[f.feed_idx]);
          setter(f.copy.as<Object>(), resolved);
        }
        f.feed_idx++;
        break;
      }
    }
    return false;
  }

  Any FinalizeFrame(CopyFrame& f) {
    int32_t ti = f.type_index;
    if (ti == TypeIndex::kTVMFFIArray) {
      Array<Any> new_arr;
      new_arr.reserve(static_cast<int64_t>(f.resolved.size()));
      for (const auto& elem : f.resolved) {
        new_arr.push_back(elem);
      }
      copy_map_[f.obj] = new_arr;
      return new_arr;
    }
    if (ti == TypeIndex::kTVMFFIMap) {
      Map<Any, Any> new_map;
      for (size_t i = 0; i + 1 < f.resolved.size(); i += 2) {
        new_map.Set(f.resolved[i], f.resolved[i + 1]);
      }
      copy_map_[f.obj] = new_map;
      return new_map;
    }
    return f.copy;
  }

  void OnFrameComplete(CopyFrame& f) {
    int32_t ti = f.type_index;
    if (ti == TypeIndex::kTVMFFIArray || ti == TypeIndex::kTVMFFIMap) {
      in_progress_.erase(f.obj);
    }
  }

  Any OnTerminate(Any r) { return r; }

 private:
  refl::TypeAttrColumn* column_;
  std::unordered_map<const Object*, Any> copy_map_;
  std::unordered_set<const Object*> in_progress_;
  bool has_deferred_ = false;

  bool TryCopyImmediate(const Any& value, Any* out) {
    if (value.type_index() < TypeIndex::kTVMFFIStaticObjectBegin) {
      *out = value;
      return true;
    }
    const Object* obj = value.as<Object>();
    if (obj == nullptr) {
      *out = value;
      return true;
    }
    // Already copied
    auto it = copy_map_.find(obj);
    if (it != copy_map_.end()) {
      *out = it->second;
      return true;
    }
    // In-progress immutable container: return original as placeholder
    if (in_progress_.count(obj)) {
      has_deferred_ = true;
      *out = value;
      return true;
    }
    int32_t ti = obj->type_index();
    // Immutable leaf objects
    if (ti == TypeIndex::kTVMFFIStr || ti == TypeIndex::kTVMFFIBytes ||
        ti == TypeIndex::kTVMFFIShape) {
      *out = value;
      return true;
    }
    return false;
  }

  void FixupDeferredReferences() {
    for (auto& [orig_ptr, copy_any] : copy_map_) {
      const Object* copy_obj = copy_any.as<Object>();
      if (!copy_obj) continue;
      int32_t ti = copy_obj->type_index();
      if (ti == TypeIndex::kTVMFFIList) {
        FixupList(copy_any);
      } else if (ti == TypeIndex::kTVMFFIDict) {
        FixupDict(copy_any);
      } else if (ti >= TypeIndex::kTVMFFIStaticObjectEnd) {
        FixupObject(copy_obj, ti);
      }
    }
  }

  void FixupList(const Any& list_any) {
    List<Any> list = list_any.cast<List<Any>>();
    int64_t n = static_cast<int64_t>(list.size());
    for (int64_t i = 0; i < n; ++i) {
      const Any& elem = list[i];
      if (elem.type_index() < TypeIndex::kTVMFFIStaticObjectBegin) continue;
      const Object* elem_obj = elem.as<Object>();
      if (!elem_obj) continue;
      auto it = copy_map_.find(elem_obj);
      if (it != copy_map_.end()) {
        list.Set(i, it->second);
      }
    }
  }

  void FixupDict(const Any& dict_any) {
    Dict<Any, Any> dict = dict_any.cast<Dict<Any, Any>>();
    const DictObj* dict_obj = dict_any.as<DictObj>();
    // Collect entries that need key or value fixup.
    // old_key, new_key, new_val
    std::vector<std::tuple<Any, Any, Any>> fixups;
    for (const auto& [k, v] : *dict_obj) {
      Any new_key = k;
      Any new_val = v;
      bool changed = false;
      if (k.type_index() >= TypeIndex::kTVMFFIStaticObjectBegin) {
        const Object* k_obj = k.as<Object>();
        if (k_obj) {
          auto it = copy_map_.find(k_obj);
          if (it != copy_map_.end()) {
            new_key = it->second;
            changed = true;
          }
        }
      }
      if (v.type_index() >= TypeIndex::kTVMFFIStaticObjectBegin) {
        const Object* v_obj = v.as<Object>();
        if (v_obj) {
          auto it = copy_map_.find(v_obj);
          if (it != copy_map_.end()) {
            new_val = it->second;
            changed = true;
          }
        }
      }
      if (changed) {
        fixups.emplace_back(k, new_key, new_val);
      }
    }
    for (auto& [old_key, new_key, new_val] : fixups) {
      dict.erase(old_key);
      dict.Set(new_key, new_val);
    }
  }

  void FixupObject(const Object* copy_obj, int32_t ti) {
    const TVMFFITypeInfo* type_info = TVMFFIGetTypeInfo(ti);
    refl::ForEachFieldInfo(type_info, [&](const TVMFFIFieldInfo* finfo) {
      refl::FieldGetter getter(finfo);
      Any field_val = getter(copy_obj);
      if (field_val.type_index() < TypeIndex::kTVMFFIStaticObjectBegin) return;
      const Object* field_obj = field_val.as<Object>();
      if (!field_obj) return;
      auto it = copy_map_.find(field_obj);
      if (it != copy_map_.end() && !it->second.same_as(field_val)) {
        refl::FieldSetter setter(finfo);
        setter(copy_obj, it->second);
      }
    });
  }
};

// ---------- Repr helpers ----------

/*!
 * \brief Convert a DLDeviceType to a short name string.
 */
const char* DeviceTypeName(int device_type) {
  switch (device_type) {
    case kDLCPU:
      return "cpu";
    case kDLCUDA:
      return "cuda";
    case kDLCUDAHost:
      return "cuda_host";
    case kDLOpenCL:
      return "opencl";
    case kDLVulkan:
      return "vulkan";
    case kDLMetal:
      return "metal";
    case kDLVPI:
      return "vpi";
    case kDLROCM:
      return "rocm";
    case kDLROCMHost:
      return "rocm_host";
    case kDLExtDev:
      return "ext_dev";
    case kDLCUDAManaged:
      return "cuda_managed";
    case kDLOneAPI:
      return "oneapi";
    case kDLWebGPU:
      return "webgpu";
    case kDLHexagon:
      return "hexagon";
    case kDLMAIA:
      return "maia";
    case kDLTrn:
      return "trn";
    default:
      return "unknown";
  }
}

/*!
 * \brief Format a DLDevice as "device_name:device_id".
 */
std::string DeviceToString(DLDevice device) {
  std::ostringstream os;
  os << DeviceTypeName(device.device_type) << ":" << device.device_id;
  return os.str();
}

/*!
 * \brief Format raw bytes as a Python-style bytes literal: b"...".
 */
std::string FormatBytes(const char* data, size_t size) {
  std::ostringstream os;
  os << "b\"";
  for (size_t i = 0; i < size; ++i) {
    unsigned char c = static_cast<unsigned char>(data[i]);
    if (c >= 32 && c < 127 && c != '\"' && c != '\\') {
      os << static_cast<char>(c);
    } else {
      os << "\\x" << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(c);
    }
  }
  os << "\"";
  return os.str();
}

/*!
 * \brief Format an object address as a hex string.
 */
std::string AddressStr(const Object* obj) {
  std::ostringstream os;
  os << "0x" << std::hex << reinterpret_cast<uintptr_t>(obj);
  return os.str();
}

struct ReprFrame : FrameBase {
  std::string header;
  std::vector<std::string> child_reprs;
};

/*!
 * \brief Iterative DFS-based repr printer (CRTP-based).
 *
 * Uses ObjectGraphDFS to iterate the object graph with an explicit stack.
 * Handles String, Bytes, Tensor, Shape, Array, List, Map, Dict formatting
 * directly (no registered built-in hooks).
 * Custom __ffi_repr__ hooks on user types are still supported.
 *
 * Address display is controlled by the TVM_FFI_REPR_WITH_ADDR environment variable.
 */
class ReprPrinter : public ObjectGraphDFS<ReprPrinter, ReprFrame, std::string> {
 public:
  String Run(const Any& value) {
    const char* env = std::getenv("TVM_FFI_REPR_WITH_ADDR");
    show_addr_ = env != nullptr && std::string_view(env) == "1";
    std::string result;
    if (TryReprImmediate(value, &result)) return String(result);
    this->PushFrame(value);
    return String(this->RunLoop());
  }

  // ---------- CRTP customization points ----------

  uint32_t GetFieldSkipMask() { return kTVMFFIFieldFlagBitMaskReprOff; }

  void OnEnter(const Object* obj) { state_[obj] = State::kInProgress; }

  void OnFrameInit(ReprFrame& f) {
    int32_t ti = f.type_index;
    const TVMFFITypeInfo* type_info = TVMFFIGetTypeInfo(ti);
    std::string type_key(type_info->type_key.data, type_info->type_key.size);
    f.header = show_addr_ ? (type_key + "@" + AddressStr(f.obj)) : type_key;
  }

  bool TryVisitChild(ReprFrame& f, size_t idx, std::string* out) {
    return TryReprImmediate(f.children[idx], out);
  }

  std::optional<std::string> PushChildFrame(ReprFrame& f, size_t idx) {
    this->PushFrame(f.children[idx]);
    return std::nullopt;
  }

  bool FeedChild(ReprFrame& f, std::string repr) {
    f.child_reprs.push_back(std::move(repr));
    return false;
  }

  std::string FinalizeFrame(ReprFrame& f) {
    std::string result;
    switch (f.kind) {
      case FrameBase::kSequence: {
        if (f.type_index == TypeIndex::kTVMFFIArray) {
          result = "(";
          for (size_t i = 0; i < f.child_reprs.size(); ++i) {
            if (i > 0) result += ", ";
            result += f.child_reprs[i];
          }
          if (f.child_reprs.size() == 1) result += ",";
          result += ")";
        } else {
          result = "[";
          for (size_t i = 0; i < f.child_reprs.size(); ++i) {
            if (i > 0) result += ", ";
            result += f.child_reprs[i];
          }
          result += "]";
        }
        break;
      }
      case FrameBase::kMap: {
        result = "{";
        for (size_t i = 0; i + 1 < f.child_reprs.size(); i += 2) {
          if (i > 0) result += ", ";
          result += f.child_reprs[i] + ": " + f.child_reprs[i + 1];
        }
        result += "}";
        break;
      }
      case FrameBase::kObject: {
        if (f.child_reprs.empty()) {
          result = f.header;
        } else {
          result = f.header + "(";
          std::unordered_set<std::string_view> seen_names;
          bool first = true;
          size_t repr_idx = 0;
          for (size_t i = 0; i < f.field_infos.size(); ++i) {
            std::string_view name(f.field_infos[i]->name.data, f.field_infos[i]->name.size);
            if (!seen_names.insert(name).second) {
              repr_idx++;
              continue;
            }
            if (!first) result += ", ";
            first = false;
            result += std::string(name) + "=" + f.child_reprs[repr_idx++];
          }
          result += ")";
        }
        break;
      }
    }
    if (show_addr_ &&
        (f.type_index == TypeIndex::kTVMFFIArray || f.type_index == TypeIndex::kTVMFFIList ||
         f.type_index == TypeIndex::kTVMFFIMap || f.type_index == TypeIndex::kTVMFFIDict)) {
      result += "@" + AddressStr(f.obj);
    }
    state_[f.obj] = State::kDone;
    repr_cache_[f.obj] = result;
    return result;
  }

  void OnFrameComplete(ReprFrame&) {}

  std::string OnTerminate(std::string r) { return r; }

 private:
  enum class State : int8_t { kNotVisited = 0, kInProgress = 1, kDone = 2 };
  std::unordered_map<const Object*, State> state_;
  std::unordered_map<const Object*, std::string> repr_cache_;
  bool show_addr_ = false;

  // ---------- Immediate repr ----------

  bool TryReprImmediate(const Any& value, std::string* out) {
    int32_t ti = value.type_index();
    switch (ti) {
      case TypeIndex::kTVMFFINone:
        *out = "None";
        return true;
      case TypeIndex::kTVMFFIBool:
        *out = value.cast<bool>() ? "True" : "False";
        return true;
      case TypeIndex::kTVMFFIInt:
        *out = std::to_string(value.cast<int64_t>());
        return true;
      case TypeIndex::kTVMFFIFloat: {
        std::ostringstream os;
        os << value.cast<double>();
        *out = os.str();
        return true;
      }
      case TypeIndex::kTVMFFIDataType: {
        String s = DLDataTypeToString(value.cast<DLDataType>());
        *out = std::string(s.data(), s.size());
        return true;
      }
      case TypeIndex::kTVMFFIDevice: {
        *out = DeviceToString(value.cast<DLDevice>());
        return true;
      }
      default:
        break;
    }
    if (ti == TypeIndex::kTVMFFISmallStr) {
      String s = value.cast<String>();
      String escaped = EscapeString(s);
      *out = std::string(escaped.data(), escaped.size());
      return true;
    }
    if (ti == TypeIndex::kTVMFFISmallBytes) {
      Bytes b = value.cast<Bytes>();
      *out = FormatBytes(b.data(), b.size());
      return true;
    }
    if (ti < TypeIndex::kTVMFFIStaticObjectBegin) {
      *out = value.GetTypeKey();
      return true;
    }
    const Object* obj = static_cast<const Object*>(value.as<Object>());
    if (obj == nullptr) {
      *out = "None";
      return true;
    }
    // Check cache / cycle
    auto it = state_.find(obj);
    if (it != state_.end()) {
      if (it->second == State::kDone) {
        *out = repr_cache_[obj];
        return true;
      }
      if (it->second == State::kInProgress) {
        *out = show_addr_ ? ("...@" + AddressStr(obj)) : "...";
        return true;
      }
    }
    // String/Bytes on heap
    if (ti == TypeIndex::kTVMFFIStr) {
      String s = details::AnyUnsafe::CopyFromAnyViewAfterCheck<String>(value);
      String escaped = EscapeString(s);
      *out = std::string(escaped.data(), escaped.size());
      return true;
    }
    if (ti == TypeIndex::kTVMFFIBytes) {
      Bytes b = details::AnyUnsafe::CopyFromAnyViewAfterCheck<Bytes>(value);
      *out = FormatBytes(b.data(), b.size());
      return true;
    }
    // Tensor
    if (ti == TypeIndex::kTVMFFITensor) {
      const TensorObj* t = value.as<TensorObj>();
      std::ostringstream os;
      os << DLDataTypeToString(t->dtype) << "[";
      for (int i = 0; i < t->ndim; ++i) {
        if (i > 0) os << ", ";
        os << t->shape[i];
      }
      os << "]@" << DeviceToString(t->device);
      if (show_addr_) os << "@" << AddressStr(obj);
      *out = os.str();
      return true;
    }
    // Shape
    if (ti == TypeIndex::kTVMFFIShape) {
      const ShapeObj* s = value.as<ShapeObj>();
      std::ostringstream os;
      os << "Shape(";
      for (size_t i = 0; i < s->size; ++i) {
        if (i > 0) os << ", ";
        os << s->data[i];
      }
      os << ")";
      *out = os.str();
      return true;
    }
    // Custom __ffi_repr__ hook
    static refl::TypeAttrColumn repr_column(refl::type_attr::kRepr);
    AnyView custom_repr = repr_column[ti];
    if (custom_repr != nullptr) {
      state_[obj] = State::kInProgress;
      Function repr_fn = custom_repr.cast<Function>();
      Function fn_repr = CreateFnRepr();
      String r = repr_fn(obj, fn_repr).cast<String>();
      std::string result(r.data(), r.size());
      if (show_addr_ && (ti == TypeIndex::kTVMFFIArray || ti == TypeIndex::kTVMFFIList ||
                         ti == TypeIndex::kTVMFFIMap || ti == TypeIndex::kTVMFFIDict)) {
        result += "@" + AddressStr(obj);
      }
      state_[obj] = State::kDone;
      repr_cache_[obj] = result;
      *out = result;
      return true;
    }
    // Needs a frame
    return false;
  }

  // ---------- Custom hook callback ----------

  Function CreateFnRepr() {
    return Function::FromTyped([this](AnyView value) -> String {
      Any v(value);
      std::string result;
      if (TryReprImmediate(v, &result)) return String(result);
      std::vector<ReprFrame> saved;
      saved.swap(this->stack_);
      this->PushFrame(v);
      result = this->RunLoop();
      this->stack_.swap(saved);
      return String(result);
    });
  }
};

// ---------- Section 3: Recursive Hash ----------

/*!
 * \brief Iterative reflection-based recursive hasher (CRTP-based).
 *
 * Uses ObjectGraphDFS to iterate the object graph with an explicit stack.
 * Supports custom __ffi_hash__ hooks.
 *
 * Computes a deterministic hash consistent with RecursiveEq:
 *   RecursiveEq(a, b) => RecursiveHash(a) == RecursiveHash(b)
 */
struct HashFrame : FrameBase {
  uint64_t hash = 0;
  size_t seq_index = 0;
  std::vector<uint64_t> entry_hashes;
  bool in_key = true;
  uint64_t key_hash = 0;
};

class RecursiveHasher : public ObjectGraphDFS<RecursiveHasher, HashFrame, uint64_t> {
 public:
  uint64_t HashAny(const Any& value) {
    uint64_t h;
    if (TryHashImmediate(value, &h)) return h;
    this->PushFrame(value);
    return this->RunLoop();
  }

  // ---------- CRTP customization points ----------

  uint32_t GetFieldSkipMask() {
    return kTVMFFIFieldFlagBitMaskHashOff | kTVMFFIFieldFlagBitMaskCompareOff;
  }

  void OnEnter(const Object* obj) { on_stack_.insert(obj); }

  void OnFrameInit(HashFrame& f) {
    const TVMFFITypeInfo* type_info = TVMFFIGetTypeInfo(f.type_index);
    if (f.kind == FrameBase::kObject) {
      // Reflected objects: use type_key_hash alone (matches old behavior).
      // Containers already combine type_key_hash with container_size.
      f.hash = type_info->type_key_hash;
    } else {
      f.hash = details::StableHashCombine(type_info->type_key_hash, f.container_size);
    }
    if (f.kind == FrameBase::kMap) {
      f.entry_hashes.reserve(f.container_size);
    }
  }

  bool TryVisitChild(HashFrame& f, size_t idx, uint64_t* out) {
    return TryHashImmediate(f.children[idx], out);
  }

  std::optional<uint64_t> PushChildFrame(HashFrame& f, size_t idx) {
    this->PushFrame(f.children[idx]);
    return std::nullopt;
  }

  bool FeedChild(HashFrame& f, uint64_t h) {
    switch (f.kind) {
      case FrameBase::kSequence: {
        f.hash = details::StableHashCombine(f.hash, details::StableHashCombine(h, f.seq_index++));
        break;
      }
      case FrameBase::kMap: {
        if (f.in_key) {
          f.key_hash = h;
          f.in_key = false;
        } else {
          f.entry_hashes.push_back(details::StableHashCombine(f.key_hash, h));
          f.in_key = true;
        }
        break;
      }
      case FrameBase::kObject: {
        f.hash = details::StableHashCombine(f.hash, h);
        break;
      }
    }
    return false;  // never terminates early
  }

  uint64_t FinalizeFrame(HashFrame& f) {
    if (f.kind == FrameBase::kMap) {
      std::sort(f.entry_hashes.begin(), f.entry_hashes.end());
      for (uint64_t eh : f.entry_hashes) {
        f.hash = details::StableHashCombine(f.hash, eh);
      }
    }
    return f.hash;
  }

  void OnFrameComplete(HashFrame& f) {
    if (f.obj != nullptr) {
      memo_[f.obj] = f.hash;
      on_stack_.erase(f.obj);
    }
  }

  uint64_t OnTerminate(uint64_t r) { return r; }

 private:
  std::unordered_set<const Object*> on_stack_;
  std::unordered_map<const Object*, uint64_t> memo_;

  // ---------- Immediate (non-recursive) hashing ----------

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
    static refl::TypeAttrColumn hash_column(refl::type_attr::kHash);
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
    // __ffi_eq__ or __ffi_compare__ but no __ffi_hash__.
    if (ti >= TypeIndex::kTVMFFIStaticObjectEnd) {
      static refl::TypeAttrColumn eq_column(refl::type_attr::kEq);
      static refl::TypeAttrColumn cmp_column(refl::type_attr::kCompare);
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

  // ---------- Custom __ffi_hash__ callback ----------

  Function CreateFnHash() {
    return Function::FromTyped([this](AnyView value) -> int64_t {
      Any v(value);
      uint64_t h;
      if (TryHashImmediate(v, &h)) return static_cast<int64_t>(h);
      std::vector<HashFrame> saved;
      saved.swap(this->stack_);
      this->PushFrame(v);
      h = this->RunLoop();
      this->stack_.swap(saved);
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

  // ---------- String hashing ----------

  static uint64_t HashString(const Any& value, const TVMFFIAny* data, int32_t ti) {
    const char* ptr;
    size_t len;
    GetStringData(value, data, ti, &ptr, &len);
    return details::StableHashCombine(static_cast<uint64_t>(TypeIndex::kTVMFFIStr),
                                      details::StableHashBytes(ptr, len));
  }

  // ---------- Bytes hashing ----------

  static uint64_t HashBytes(const Any& value, const TVMFFIAny* data, int32_t ti) {
    const char* ptr;
    size_t len;
    GetBytesData(value, data, ti, &ptr, &len);
    return details::StableHashCombine(static_cast<uint64_t>(TypeIndex::kTVMFFIBytes),
                                      details::StableHashBytes(ptr, len));
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

// ---------- Recursive Compare ----------

struct CompareFrame {
  enum Kind : uint8_t { kSequence, kMap, kObject };
  Kind kind;
  std::vector<std::pair<Any, Any>> children;
  size_t child_idx = 0;
  size_t lhs_size = 0;
  size_t rhs_size = 0;
  const Object* lhs_obj = nullptr;
  const Object* rhs_obj = nullptr;
  size_t NumChildren() const { return children.size(); }
};

/*!
 * \brief Iterative three-way recursive comparer (CRTP-based).
 *
 * Returns int32_t: -1 (lhs < rhs), 0 (equal), +1 (lhs > rhs).
 * Uses ObjectGraphDFS RunLoop with pair-based CompareFrame (not FrameBase).
 * Supports custom __ffi_eq__ and __ffi_compare__ hooks.
 */
class RecursiveComparer : public ObjectGraphDFS<RecursiveComparer, CompareFrame, int32_t> {
 public:
  explicit RecursiveComparer(bool eq_only) : eq_only_(eq_only) {}

  int32_t CompareAny(const Any& lhs, const Any& rhs) {
    int32_t cmp;
    if (TryCompareImmediate(lhs, rhs, &cmp)) return cmp;
    auto eager = PushPairFrame(lhs, rhs);
    if (eager.has_value()) return *eager;
    return this->RunLoop();
  }

  // ---------- CRTP customization points ----------

  uint32_t GetFieldSkipMask() { return kTVMFFIFieldFlagBitMaskCompareOff; }
  void OnEnter(const Object*) {}
  void OnFrameInit(CompareFrame&) {}

  bool TryVisitChild(CompareFrame& f, size_t idx, int32_t* out) {
    auto& [child_lhs, child_rhs] = f.children[idx];
    return TryCompareImmediate(child_lhs, child_rhs, out);
  }

  std::optional<int32_t> PushChildFrame(CompareFrame& f, size_t idx) {
    auto& [child_lhs, child_rhs] = f.children[idx];
    return PushPairFrame(child_lhs, child_rhs);
  }

  bool FeedChild(CompareFrame&, int32_t result) { return result != 0; }

  int32_t FinalizeFrame(CompareFrame& f) {
    if (f.kind == CompareFrame::kSequence) {
      if (f.lhs_size < f.rhs_size) return -1;
      if (f.lhs_size > f.rhs_size) return 1;
    }
    return 0;
  }

  void OnFrameComplete(CompareFrame& f) {
    if (f.lhs_obj != nullptr && f.rhs_obj != nullptr) {
      on_stack_.erase(std::make_pair(f.lhs_obj, f.rhs_obj));
    }
  }

  int32_t OnTerminate(int32_t result) { return PropagateNonZero(result); }

 private:
  struct PairHash {
    size_t operator()(std::pair<const Object*, const Object*> p) const {
      auto h1 = std::hash<const void*>()(p.first);
      auto h2 = std::hash<const void*>()(p.second);
      return h1 ^ (h2 * 0x9e3779b97f4a7c15ULL + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }
  };

  bool eq_only_;
  std::unordered_set<std::pair<const Object*, const Object*>, PairHash> on_stack_;

  // ---------- Immediate (non-recursive) comparison ----------

  bool TryCompareImmediate(const Any& lhs, const Any& rhs, int32_t* out) {
    using details::AnyUnsafe;
    const TVMFFIAny* lhs_data = AnyUnsafe::TVMFFIAnyPtrFromAny(lhs);
    const TVMFFIAny* rhs_data = AnyUnsafe::TVMFFIAnyPtrFromAny(rhs);
    int32_t lti = lhs_data->type_index;
    int32_t rti = rhs_data->type_index;

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
    if (IsStringType(lti) && IsStringType(rti)) {
      *out = CompareString(lhs, rhs, lhs_data, rhs_data, lti, rti);
      return true;
    }
    if (IsBytesType(lti) && IsBytesType(rti)) {
      *out = CompareBytes(lhs, rhs, lhs_data, rhs_data, lti, rti);
      return true;
    }
    if (lti != rti) {
      if (eq_only_) {
        *out = 1;
        return true;
      }
      TVM_FFI_THROW(TypeError) << "Cannot compare values of different types: " << lhs.GetTypeKey()
                               << " vs " << rhs.GetTypeKey();
    }
    if (lti < TypeIndex::kTVMFFIStaticObjectBegin) {
      *out = ComparePOD(lhs, rhs, lhs_data, rhs_data, lti);
      return true;
    }
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
    if (lti == TypeIndex::kTVMFFIShape) {
      *out = CompareShape(AnyUnsafe::CopyFromAnyViewAfterCheck<Shape>(lhs),
                          AnyUnsafe::CopyFromAnyViewAfterCheck<Shape>(rhs));
      return true;
    }
    auto pair = std::make_pair(lhs_obj, rhs_obj);
    if (on_stack_.count(pair)) {
      if (eq_only_) {
        *out = 0;
        return true;
      }
      TVM_FFI_THROW(ValueError) << "RecursiveCompare: cyclic reference detected in ordering";
    }
    if (lti >= TypeIndex::kTVMFFIStaticObjectEnd) {
      return TryCustomHook(lhs_obj, rhs_obj, out);
    }
    return false;
  }

  // ---------- Custom hook dispatch ----------

  bool TryCustomHook(const Object* lhs_obj, const Object* rhs_obj, int32_t* out) {
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
    static refl::TypeAttrColumn eq_column(refl::type_attr::kEq);
    static refl::TypeAttrColumn cmp_column(refl::type_attr::kCompare);
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
    return false;
  }

  // ---------- Pair-based frame creation ----------

  /*!
   * \brief Push a pair frame or return an eager result (map mismatch).
   */
  std::optional<int32_t> PushPairFrame(const Any& lhs, const Any& rhs) {
    if (this->stack_.size() >= kMaxTraversalStackDepth) {
      TVM_FFI_THROW(ValueError) << "RecursiveCompare: maximum stack depth ("
                                << kMaxTraversalStackDepth << ") exceeded";
    }
    using details::AnyUnsafe;
    const TVMFFIAny* lhs_data = AnyUnsafe::TVMFFIAnyPtrFromAny(lhs);
    int32_t lti = lhs_data->type_index;
    const Object* lhs_obj = static_cast<const Object*>(lhs.as<Object>());
    const Object* rhs_obj = static_cast<const Object*>(rhs.as<Object>());
    auto pair = std::make_pair(lhs_obj, rhs_obj);
    on_stack_.insert(pair);

    switch (lti) {
      case TypeIndex::kTVMFFIArray: {
        auto lhs_seq = AnyUnsafe::CopyFromAnyViewAfterCheck<Array<Any>>(lhs);
        auto rhs_seq = AnyUnsafe::CopyFromAnyViewAfterCheck<Array<Any>>(rhs);
        PushSequenceFrame(lhs_seq, rhs_seq, lhs_obj, rhs_obj);
        return std::nullopt;
      }
      case TypeIndex::kTVMFFIList: {
        auto lhs_seq = AnyUnsafe::CopyFromAnyViewAfterCheck<List<Any>>(lhs);
        auto rhs_seq = AnyUnsafe::CopyFromAnyViewAfterCheck<List<Any>>(rhs);
        PushSequenceFrame(lhs_seq, rhs_seq, lhs_obj, rhs_obj);
        return std::nullopt;
      }
      case TypeIndex::kTVMFFIMap: {
        auto lhs_map = AnyUnsafe::CopyFromAnyViewAfterCheck<Map<Any, Any>>(lhs);
        auto rhs_map = AnyUnsafe::CopyFromAnyViewAfterCheck<Map<Any, Any>>(rhs);
        return PushMapFrame(lhs_map, rhs_map, lhs_obj, rhs_obj);
      }
      case TypeIndex::kTVMFFIDict: {
        auto lhs_map = AnyUnsafe::CopyFromAnyViewAfterCheck<Dict<Any, Any>>(lhs);
        auto rhs_map = AnyUnsafe::CopyFromAnyViewAfterCheck<Dict<Any, Any>>(rhs);
        return PushMapFrame(lhs_map, rhs_map, lhs_obj, rhs_obj);
      }
      default: {
        return PushObjectFrame(lhs_obj, rhs_obj);
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
    this->stack_.push_back(std::move(frame));
  }

  template <typename MapType>
  std::optional<int32_t> PushMapFrame(const MapType& lhs, const MapType& rhs, const Object* lhs_obj,
                                      const Object* rhs_obj) {
    if (lhs.size() != rhs.size()) {
      on_stack_.erase(std::make_pair(lhs_obj, rhs_obj));
      if (eq_only_) return 1;
      TVM_FFI_THROW(TypeError) << "Cannot order Map/Dict values";
    }
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
        if (eq_only_) return 1;
        TVM_FFI_THROW(TypeError) << "Cannot order Map/Dict values";
      }
      frame.children.emplace_back(kv.second, (*it).second);
    }
    this->stack_.push_back(std::move(frame));
    return std::nullopt;
  }

  std::optional<int32_t> PushObjectFrame(const Object* lhs, const Object* rhs) {
    if (lhs->type_index() != rhs->type_index()) {
      auto pair = std::make_pair(lhs, rhs);
      on_stack_.erase(pair);
      if (eq_only_) return 1;
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
    refl::ForEachFieldInfo(type_info, [&](const TVMFFIFieldInfo* finfo) {
      if (finfo->flags & kTVMFFIFieldFlagBitMaskCompareOff) return;
      refl::FieldGetter getter(finfo);
      frame.children.emplace_back(getter(lhs), getter(rhs));
    });
    this->stack_.push_back(std::move(frame));
    return std::nullopt;
  }

  // ---------- Propagation ----------

  int32_t PropagateNonZero(int32_t result) {
    if (!eq_only_) {
      for (auto it = this->stack_.rbegin(); it != this->stack_.rend(); ++it) {
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
    for (auto& f : this->stack_) {
      if (f.lhs_obj != nullptr && f.rhs_obj != nullptr) {
        on_stack_.erase(std::make_pair(f.lhs_obj, f.rhs_obj));
      }
    }
    this->stack_.clear();
  }

  // ---------- Custom hook callbacks ----------

  Function CreateFnEq() {
    return Function::FromTyped([this](AnyView l, AnyView r) -> bool {
      Any lhs(l), rhs(r);
      int32_t cmp;
      if (TryCompareImmediate(lhs, rhs, &cmp)) return cmp == 0;
      std::vector<CompareFrame> saved;
      saved.swap(this->stack_);
      auto eager = PushPairFrame(lhs, rhs);
      if (eager.has_value()) {
        this->stack_.swap(saved);
        return *eager == 0;
      }
      cmp = this->RunLoop();
      this->stack_.swap(saved);
      return cmp == 0;
    });
  }

  Function CreateFnCompare() {
    return Function::FromTyped([this](AnyView l, AnyView r) -> int32_t {
      Any lhs(l), rhs(r);
      int32_t cmp;
      if (TryCompareImmediate(lhs, rhs, &cmp)) return cmp;
      std::vector<CompareFrame> saved;
      saved.swap(this->stack_);
      auto eager = PushPairFrame(lhs, rhs);
      if (eager.has_value()) {
        this->stack_.swap(saved);
        return *eager;
      }
      cmp = this->RunLoop();
      this->stack_.swap(saved);
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

  // ---------- String comparison ----------

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

  // ---------- Bytes comparison ----------

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

// ---------- Python-defined type support ----------

/*!
 * \brief Deleter for objects whose layout is defined from Python via Field descriptors.
 *
 * For the "strong" phase, iterates all reflected fields and destructs
 * Any/ObjectRef values in-place (to release references).  For the "weak"
 * phase, frees the underlying calloc'd memory.
 */
void PyClassDeleter(void* self_void, int flags) {
  TVMFFIObject* self = static_cast<TVMFFIObject*>(self_void);
  if (flags & kTVMFFIObjectDeleterFlagBitMaskStrong) {
    const TVMFFITypeInfo* type_info = TVMFFIGetTypeInfo(self->type_index);
    refl::ForEachFieldInfo(type_info, [&](const TVMFFIFieldInfo* finfo) {
      void* field_addr = reinterpret_cast<char*>(self) + finfo->offset;
      int32_t ti = finfo->field_static_type_index;
      if (ti == TypeIndex::kTVMFFIAny) {
        // Any field: call destructor to release owned references
        reinterpret_cast<Any*>(field_addr)->~Any();
      } else if (ti >= TypeIndex::kTVMFFIStaticObjectBegin) {
        // ObjectRef field: call destructor to DecRef
        reinterpret_cast<ObjectRef*>(field_addr)->~ObjectRef();
      }
      // POD types (int, float, bool, etc.): no cleanup needed
    });
  }
  if (flags & kTVMFFIObjectDeleterFlagBitMaskWeak) {
    std::free(self_void);
  }
}

/*!
 * \brief Generic field getter for Python-defined types.
 *
 * Reads a value of type T from the given field address and packs it into
 * a TVMFFIAny result.
 *
 * \tparam T The C++ type stored at the field address.
 */
template <typename T>
int PyClassFieldGetter(void* field, TVMFFIAny* result) {
  TVM_FFI_SAFE_CALL_BEGIN();
  *result = details::AnyUnsafe::MoveAnyToTVMFFIAny(Any(*reinterpret_cast<T*>(field)));
  TVM_FFI_SAFE_CALL_END();
}

/*!
 * \brief Return the TVMFFIFieldGetter function pointer for a given field type index.
 *
 * \param field_type_index The static type index of the field.
 * \return The function pointer as int64_t for FFI transport.
 */
int64_t GetFieldGetter(int32_t field_type_index) {
  TVMFFIFieldGetter getter = nullptr;
  switch (field_type_index) {
    case TypeIndex::kTVMFFIInt:
      getter = &PyClassFieldGetter<int64_t>;
      break;
    case TypeIndex::kTVMFFIFloat:
      getter = &PyClassFieldGetter<double>;
      break;
    case TypeIndex::kTVMFFIBool:
      getter = &PyClassFieldGetter<bool>;
      break;
    case TypeIndex::kTVMFFIOpaquePtr:
      getter = &PyClassFieldGetter<void*>;
      break;
    case TypeIndex::kTVMFFIDataType:
      getter = &PyClassFieldGetter<DLDataType>;
      break;
    case TypeIndex::kTVMFFIDevice:
      getter = &PyClassFieldGetter<DLDevice>;
      break;
    default:
      if (field_type_index == TypeIndex::kTVMFFIAny || field_type_index == TypeIndex::kTVMFFINone) {
        getter = &PyClassFieldGetter<Any>;
      } else if (field_type_index >= TypeIndex::kTVMFFIStaticObjectBegin) {
        getter = &PyClassFieldGetter<ObjectRef>;
      } else {
        TVM_FFI_THROW(ValueError) << "Unsupported field type index for getter: "
                                  << field_type_index;
      }
      break;
  }
  return reinterpret_cast<int64_t>(getter);
}

/*!
 * \brief Write a converted value to a field of the appropriate C++ type.
 *
 * Dispatches on field_type_index to reinterpret the destination address and
 * assign from the converted Any value.
 */
void WriteFieldValue(void* field_addr, int32_t field_type_index, Any value) {
  switch (field_type_index) {
    case TypeIndex::kTVMFFIInt:
      *reinterpret_cast<int64_t*>(field_addr) = value.cast<int64_t>();
      return;
    case TypeIndex::kTVMFFIFloat:
      *reinterpret_cast<double*>(field_addr) = value.cast<double>();
      return;
    case TypeIndex::kTVMFFIBool:
      *reinterpret_cast<bool*>(field_addr) = value.cast<bool>();
      return;
    case TypeIndex::kTVMFFIOpaquePtr:
      *reinterpret_cast<void**>(field_addr) = value.cast<void*>();
      return;
    case TypeIndex::kTVMFFIDataType:
      *reinterpret_cast<DLDataType*>(field_addr) = value.cast<DLDataType>();
      return;
    case TypeIndex::kTVMFFIDevice:
      *reinterpret_cast<DLDevice*>(field_addr) = value.cast<DLDevice>();
      return;
    default:
      break;
  }
  if (field_type_index == TypeIndex::kTVMFFIAny || field_type_index == TypeIndex::kTVMFFINone) {
    *reinterpret_cast<Any*>(field_addr) = std::move(value);
  } else if (field_type_index >= TypeIndex::kTVMFFIStaticObjectBegin) {
    *reinterpret_cast<ObjectRef*>(field_addr) = value.cast<ObjectRef>();
  } else {
    TVM_FFI_THROW(ValueError) << "Unsupported field type index for setter: " << field_type_index;
  }
}

/*!
 * \brief Create a FunctionObj setter for a Python-defined field.
 *
 * The returned Function accepts (OpaquePtr field_addr, AnyView value),
 * calls f_convert to coerce the value via the type_converter, and writes
 * the result to the field.
 *
 * \param field_type_index The static type index of the field.
 * \param type_converter_int Opaque pointer (as int64_t) to the Python _TypeConverter (borrowed).
 * \param f_convert_int C function pointer (as int64_t): int(void*, const TVMFFIAny*, TVMFFIAny*).
 *        Returns 0 on success, -1 on error (error stored in TLS).
 * \return A packed Function suitable for use as a FunctionObj setter.
 */
Function MakeFieldSetter(int32_t field_type_index, int64_t type_converter_int,
                         int64_t f_convert_int) {
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  void* type_converter = reinterpret_cast<void*>(type_converter_int);
  using FConvert = int (*)(void*, const TVMFFIAny*, TVMFFIAny*);
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  auto f_convert = reinterpret_cast<FConvert>(f_convert_int);

  return Function::FromPacked([field_type_index, type_converter, f_convert](
                                  const AnyView* args, int32_t num_args, Any* rv) {
    void* field_addr = args[0].cast<void*>();
    // Call the Cython-level type converter via C function pointer.
    TVMFFIAny converted;
    converted.type_index = TypeIndex::kTVMFFINone;
    converted.v_int64 = 0;
    int err = f_convert(type_converter, reinterpret_cast<const TVMFFIAny*>(&args[1]), &converted);
    if (err != 0) {
      throw details::MoveFromSafeCallRaised();
    }
    // Take ownership of the converted value and write to the field.
    Any owned = details::AnyUnsafe::MoveTVMFFIAnyToAny(&converted);
    WriteFieldValue(field_addr, field_type_index, std::move(owned));
  });
}

// ============================================================================
// Shared helpers for MakeInit / MakeInitInplace
// ============================================================================

/*!
 * \brief Return the lazily initialized KWARGS sentinel object.
 */
const ObjectRef& GetKwargsSentinel() {
  static ObjectRef kwargs_sentinel =
      Function::GetGlobalRequired("ffi.GetKwargsObject")().cast<ObjectRef>();
  return kwargs_sentinel;
}

/*! \brief Pre-computed field analysis for auto-generated init. */
struct AutoInitInfo {
  struct Entry {
    const TVMFFIFieldInfo* info;
    bool init;
    bool kw_only;
    bool has_default;
  };
  std::vector<Entry> all_fields;
  std::vector<size_t> init_indices;
  std::vector<size_t> pos_indices;
  std::unordered_map<std::string_view, size_t> name_to_index;
  std::string_view type_key;
};

/*!
 * \brief Build AutoInitInfo by analysing reflected fields for a type.
 */
std::shared_ptr<AutoInitInfo> BuildAutoInitInfo(int32_t type_index) {
  const TVMFFITypeInfo* type_info = TVMFFIGetTypeInfo(type_index);
  auto info = std::make_shared<AutoInitInfo>();
  info->type_key = std::string_view(type_info->type_key.data, type_info->type_key.size);

  refl::ForEachFieldInfo(type_info, [&](const TVMFFIFieldInfo* fi) {
    bool is_init = (fi->flags & kTVMFFIFieldFlagBitMaskInitOff) == 0;
    bool is_kw = (fi->flags & kTVMFFIFieldFlagBitMaskKwOnly) != 0;
    bool has_def = (fi->flags & kTVMFFIFieldFlagBitMaskHasDefault) != 0;
    info->all_fields.push_back({fi, is_init, is_kw, has_def});
    size_t idx = info->all_fields.size() - 1;
    if (is_init) {
      info->init_indices.push_back(idx);
      info->name_to_index[std::string_view(fi->name.data, fi->name.size)] = idx;
      if (!is_kw) {
        info->pos_indices.push_back(idx);
      }
    }
  });
  // Required fields come before optional ones in positional list.
  std::stable_partition(info->pos_indices.begin(), info->pos_indices.end(),
                        [&](size_t idx) { return !info->all_fields[idx].has_default; });
  return info;
}

/*!
 * \brief Bind packed arguments to fields on an existing object.
 *
 * Handles both positional-only and KWARGS calling conventions, sets fields
 * via ``CallFieldSetter``, fills defaults for unbound fields, and raises
 * ``TypeError`` for missing required arguments (with ``keyword-only``
 * distinction when applicable).
 */
void BindFieldArgs(Object* obj, const AutoInitInfo& info, const TVMFFIAny* raw_args, int num_args) {
  const ObjectRef& kwargs_sentinel = GetKwargsSentinel();
  std::vector<bool> field_set(info.all_fields.size(), false);

  auto set_field = [&](size_t fi, const TVMFFIAny* value) {
    void* addr = reinterpret_cast<char*>(obj) + info.all_fields[fi].info->offset;
    TVM_FFI_CHECK_SAFE_CALL(refl::CallFieldSetter(info.all_fields[fi].info, addr, value));
    field_set[fi] = true;
  };

  // ---- 1. Find KWARGS sentinel position ------------------------------------
  int kwargs_pos = -1;
  for (int i = 0; i < num_args; ++i) {
    auto opt = AnyView::CopyFromTVMFFIAny(raw_args[i]).as<ObjectRef>();
    if (opt.has_value() && opt.value().same_as(kwargs_sentinel)) {
      kwargs_pos = i;
      break;
    }
  }

  // ---- 2. Bind arguments to fields -----------------------------------------
  if (kwargs_pos >= 0) {
    // --- 2a. KWARGS mode ----------------------------------------------------
    int pos_arg = 0;
    for (size_t fi : info.pos_indices) {
      if (pos_arg < kwargs_pos) {
        set_field(fi, &raw_args[pos_arg]);
        ++pos_arg;
      }
    }
    if (pos_arg < kwargs_pos) {
      TVM_FFI_THROW(TypeError) << info.type_key << ".__ffi_init__() takes at most "
                               << info.pos_indices.size() << " positional argument(s), but "
                               << kwargs_pos << " were given";
    }
    int kv_count = num_args - kwargs_pos - 1;
    if (kv_count % 2 != 0) {
      TVM_FFI_THROW(TypeError)
          << info.type_key
          << ".__ffi_init__() KWARGS requires an even number of key-value arguments";
    }
    for (int i = kwargs_pos + 1; i < num_args; i += 2) {
      String key = AnyView::CopyFromTVMFFIAny(raw_args[i]).cast<String>();
      std::string_view key_sv(key.data(), key.size());
      auto it = info.name_to_index.find(key_sv);
      if (it == info.name_to_index.end()) {
        TVM_FFI_THROW(TypeError) << info.type_key
                                 << ".__ffi_init__() got an unexpected keyword argument '" << key
                                 << "'";
      }
      size_t idx = it->second;
      if (field_set[idx]) {
        TVM_FFI_THROW(TypeError) << info.type_key << ".__ffi_init__() got multiple values "
                                 << "for argument '" << key << "'";
      }
      set_field(idx, &raw_args[i + 1]);
    }
  } else {
    // --- 2b. Positional-only mode -------------------------------------------
    if (static_cast<size_t>(num_args) > info.pos_indices.size()) {
      TVM_FFI_THROW(TypeError) << info.type_key << ".__ffi_init__() takes at most "
                               << info.pos_indices.size() << " positional argument(s), but "
                               << num_args << " were given";
    }
    for (int i = 0; i < num_args; ++i) {
      set_field(info.pos_indices[i], &raw_args[i]);
    }
  }

  // ---- 3. Fill defaults and check required fields --------------------------
  for (size_t fi = 0; fi < info.all_fields.size(); ++fi) {
    if (field_set[fi]) continue;
    if (info.all_fields[fi].has_default) {
      void* addr = reinterpret_cast<char*>(obj) + info.all_fields[fi].info->offset;
      refl::SetFieldToDefault(info.all_fields[fi].info, addr);
    } else if (info.all_fields[fi].init) {
      auto fname = std::string_view(info.all_fields[fi].info->name.data,
                                    info.all_fields[fi].info->name.size);
      if (info.all_fields[fi].kw_only) {
        TVM_FFI_THROW(TypeError) << info.type_key
                                 << ".__ffi_init__() missing required keyword-only argument: '"
                                 << fname << "'";
      } else {
        TVM_FFI_THROW(TypeError) << info.type_key << ".__ffi_init__() missing required argument: '"
                                 << fname << "'";
      }
    }
    // init=False without default: leave at creator default.
  }
}

Function MakeInit(int32_t type_index) {
  auto info = BuildAutoInitInfo(type_index);
  const TVMFFITypeInfo* type_info = TVMFFIGetTypeInfo(type_index);
  return Function::FromPacked([info, type_info](PackedArgs args, Any* rv) {
    ObjectPtr<Object> obj_ptr = CreateEmptyObject(type_info);
    const auto raw_args = reinterpret_cast<const TVMFFIAny*>(args.data());
    BindFieldArgs(obj_ptr.get(), *info, raw_args, args.size());
    *rv = ObjectRef(obj_ptr);
  });
}

Function MakeInitInplace(int32_t type_index) {
  auto info = BuildAutoInitInfo(type_index);
  return Function::FromPacked([info](PackedArgs args, Any* rv) {
    TVM_FFI_ICHECK(args.size() >= 1)
        << "__ffi_init_inplace__ requires at least one argument (self)";
    ObjectRef self = args[0].cast<ObjectRef>();
    Object* obj = const_cast<Object*>(self.get());
    const auto raw_args = reinterpret_cast<const TVMFFIAny*>(args.data());
    BindFieldArgs(obj, *info, raw_args + 1, args.size() - 1);
  });
}

void RegisterFFIInit(int32_t type_index) {
  namespace refl = ::tvm::ffi::reflection;
  Function auto_init_fn = MakeInit(type_index);
  TVMFFIByteArray attr_name = refl::AsByteArray(refl::type_attr::kInit);
  TVMFFIAny attr_value = AnyView(auto_init_fn).CopyToTVMFFIAny();
  TVM_FFI_CHECK_SAFE_CALL(TVMFFITypeRegisterAttr(type_index, &attr_name, &attr_value));

  Function init_inplace_fn = MakeInitInplace(type_index);
  TVMFFIByteArray ip_attr_name = refl::AsByteArray(refl::type_attr::kInitInplace);
  TVMFFIAny ip_attr_value = AnyView(init_inplace_fn).CopyToTVMFFIAny();
  TVM_FFI_CHECK_SAFE_CALL(TVMFFITypeRegisterAttr(type_index, &ip_attr_name, &ip_attr_value));
}

/*!
 * \brief Combined registration for Python-defined types:
 * ``__ffi_init__``, ``__ffi_init_inplace__``, ``__ffi_new__``, ``__ffi_shallow_copy__``
 */
void PyClassRegisterTypeAttrColumns(int32_t type_index, int32_t total_size) {
  namespace refl = ::tvm::ffi::reflection;
  const TVMFFITypeInfo* type_info = TVMFFIGetTypeInfo(type_index);
  // Step 1. Register `__ffi_init__` and `__ffi_init_inplace__`
  RegisterFFIInit(type_index);
  // Step 2. Register `__ffi_new__`
  Function new_fn = Function::FromTyped([type_index, total_size]() -> ObjectRef {
    void* obj_ptr = std::calloc(1, static_cast<size_t>(total_size));
    if (!obj_ptr) {
      TVM_FFI_THROW(RuntimeError) << "Failed to allocate " << total_size << " bytes for type "
                                  << TypeIndexToTypeKey(type_index);
    }
    TVMFFIObject* ffi_obj = reinterpret_cast<TVMFFIObject*>(obj_ptr);
    ffi_obj->type_index = type_index;
    ffi_obj->combined_ref_count = details::kCombinedRefCountBothOne;
    ffi_obj->deleter = PyClassDeleter;
    // calloc zero-initializes all bytes.  For non-trivial field types:
    //   - Any: zero state is {type_index=kTVMFFINone, v_int64=0}, representing None.
    //   - ObjectRef: zero state is a null pointer.
    // Both are valid initial states whose destructors and assignment operators
    // handle correctly, so no placement construction is needed.
    Object* obj = reinterpret_cast<Object*>(obj_ptr);
    return ObjectRef(details::ObjectUnsafe::ObjectPtrFromOwned<Object>(obj));
  });
  TVMFFIByteArray attr_name = refl::AsByteArray(refl::type_attr::kNew);
  TVMFFIAny attr_value = AnyView(new_fn).CopyToTVMFFIAny();
  TVM_FFI_CHECK_SAFE_CALL(TVMFFITypeRegisterAttr(type_index, &attr_name, &attr_value));
  // Step 3. Register `__ffi_shallow_copy__`
  Function copy_fn =
      Function::FromTyped([type_index, total_size, type_info](const Object* src) -> ObjectRef {
        void* obj_ptr = std::calloc(1, static_cast<size_t>(total_size));
        if (!obj_ptr) {
          TVM_FFI_THROW(RuntimeError) << "Failed to allocate for shallow copy";
        }
        TVMFFIObject* ffi_obj = reinterpret_cast<TVMFFIObject*>(obj_ptr);
        ffi_obj->type_index = type_index;
        ffi_obj->combined_ref_count = details::kCombinedRefCountBothOne;
        ffi_obj->deleter = PyClassDeleter;
        refl::ForEachFieldInfo(type_info, [&](const TVMFFIFieldInfo* finfo) {
          void* dst = reinterpret_cast<char*>(obj_ptr) + finfo->offset;
          const void* field_src = reinterpret_cast<const char*>(src) + finfo->offset;
          int32_t ti = finfo->field_static_type_index;
          if (ti == TypeIndex::kTVMFFIAny) {
            new (dst) Any(*reinterpret_cast<const Any*>(field_src));
          } else if (ti >= TypeIndex::kTVMFFIStaticObjectBegin) {
            new (dst) ObjectRef(*reinterpret_cast<const ObjectRef*>(field_src));
          } else {
            // POD: memcpy
            std::memcpy(dst, field_src, static_cast<size_t>(finfo->size));
          }
        });
        Object* obj = reinterpret_cast<Object*>(obj_ptr);
        return ObjectRef(details::ObjectUnsafe::ObjectPtrFromOwned<Object>(obj));
      });
  TVMFFIByteArray copy_attr_name = refl::AsByteArray(refl::type_attr::kShallowCopy);
  TVMFFIAny copy_attr_value = AnyView(copy_fn).CopyToTVMFFIAny();
  TVM_FFI_CHECK_SAFE_CALL(TVMFFITypeRegisterAttr(type_index, &copy_attr_name, &copy_attr_value));
}

}  // namespace

// ============================================================================
// Deep Copy — public API
// ============================================================================

Any DeepCopy(const Any& value) {
  static refl::TypeAttrColumn column(refl::type_attr::kShallowCopy);
  ObjectDeepCopier copier(&column);
  return copier.Run(value);
}

// ============================================================================
// Repr Printing — public API
// ============================================================================

String ReprPrint(const Any& value) {
  ReprPrinter printer;
  return printer.Run(value);
}

// ============================================================================
// Recursive Hash / Compare — public API
// ============================================================================

int64_t RecursiveHash(const Any& value) {
  RecursiveHasher hasher;
  return static_cast<int64_t>(hasher.HashAny(value));
}

bool RecursiveEq(const Any& lhs, const Any& rhs) {
  RecursiveComparer cmp(/*eq_only=*/true);
  return cmp.CompareAny(lhs, rhs) == 0;
}

bool RecursiveLt(const Any& lhs, const Any& rhs) {
  RecursiveComparer cmp(/*eq_only=*/false);
  return cmp.CompareAny(lhs, rhs) < 0;
}

bool RecursiveLe(const Any& lhs, const Any& rhs) {
  RecursiveComparer cmp(/*eq_only=*/false);
  return cmp.CompareAny(lhs, rhs) <= 0;
}

bool RecursiveGt(const Any& lhs, const Any& rhs) {
  RecursiveComparer cmp(/*eq_only=*/false);
  return cmp.CompareAny(lhs, rhs) > 0;
}

bool RecursiveGe(const Any& lhs, const Any& rhs) {
  RecursiveComparer cmp(/*eq_only=*/false);
  return cmp.CompareAny(lhs, rhs) >= 0;
}

// ============================================================================
// Registration
// ============================================================================

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::EnsureTypeAttrColumn(refl::type_attr::kShallowCopy);
  refl::EnsureTypeAttrColumn(refl::type_attr::kRepr);
  refl::EnsureTypeAttrColumn(refl::type_attr::kHash);
  refl::EnsureTypeAttrColumn(refl::type_attr::kEq);
  refl::EnsureTypeAttrColumn(refl::type_attr::kCompare);
  refl::EnsureTypeAttrColumn(refl::type_attr::kNew);
  refl::EnsureTypeAttrColumn(refl::type_attr::kInit);
  refl::EnsureTypeAttrColumn(refl::type_attr::kInitInplace);
  refl::GlobalDef()
      .def("ffi._RegisterFFIInit", RegisterFFIInit)
      .def("ffi.MakeFieldSetter", MakeFieldSetter)
      .def("ffi._PyClassRegisterTypeAttrColumns", PyClassRegisterTypeAttrColumns)
      .def("ffi.DeepCopy", DeepCopy)
      .def("ffi.ReprPrint", ReprPrint)
      .def("ffi.RecursiveHash", RecursiveHash)
      .def("ffi.MakeFieldGetter", GetFieldGetter)
      .def("ffi.RecursiveEq", RecursiveEq)
      .def("ffi.RecursiveLt", RecursiveLt)
      .def("ffi.RecursiveLe", RecursiveLe)
      .def("ffi.RecursiveGt", RecursiveGt)
      .def("ffi.RecursiveGe", RecursiveGe);
}

}  // namespace ffi
}  // namespace tvm
