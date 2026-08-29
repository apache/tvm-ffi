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
 * \file src/ffi/reflection/structural_equal.cc
 *
 * \brief Structural equal implementation.
 */
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/dict.h>
#include <tvm/ffi/container/list.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/container/shape.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/structural_hash.h>
#include <tvm/ffi/reflection/accessor.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>

#include <cmath>
#include <limits>
#include <unordered_map>
#include <utility>

namespace tvm {
namespace ffi {
/**
 * \brief Internal Handler class for structural hash.
 */
class StructuralHashHandler {
 public:
  StructuralHashHandler() = default;

  uint64_t HashAny(ffi::Any src) {
    using ffi::details::AnyUnsafe;
    const TVMFFIAny* src_data = AnyUnsafe::TVMFFIAnyPtrFromAny(src);

    if (src_data->type_index < TypeIndex::kTVMFFIStaticObjectBegin) {
      // specially handle nan for float, as there can be multiple representations of nan
      // make sure they map to the same hash value
      if (src_data->type_index == TypeIndex::kTVMFFIFloat && std::isnan(src_data->v_float64)) {
        TVMFFIAny temp = *src_data;
        temp.v_float64 = std::numeric_limits<double>::quiet_NaN();
        return details::StableHashCombine(temp.type_index, temp.v_uint64);
      }
      if (src_data->type_index == TypeIndex::kTVMFFISmallStr) {
        // for small string, we use the same type key hash as normal string
        // so heap allocated string and on stack string will have the same hash
        return details::StableHashCombine(TypeIndex::kTVMFFIStr,
                                          details::StableHashSmallStrBytes(src_data));
      }
      // this is POD data, we can just hash the value
      return details::StableHashCombine(src_data->type_index, src_data->v_uint64);
    }

    switch (src_data->type_index) {
      case TypeIndex::kTVMFFIStr:
      case TypeIndex::kTVMFFIBytes: {
        // return same hash as AnyHash
        const details::BytesObjBase* src_str =
            AnyUnsafe::CopyFromAnyViewAfterCheck<const details::BytesObjBase*>(src);
        return details::StableHashCombine(src_data->type_index,
                                          details::StableHashBytes(src_str->data, src_str->size));
      }
      case TypeIndex::kTVMFFIArray: {
        return HashArray(AnyUnsafe::MoveFromAnyAfterCheck<Array<Any>>(std::move(src)));
      }
      case TypeIndex::kTVMFFIList: {
        return HashList(AnyUnsafe::MoveFromAnyAfterCheck<List<Any>>(std::move(src)));
      }
      case TypeIndex::kTVMFFIMap: {
        return HashMap(AnyUnsafe::MoveFromAnyAfterCheck<Map<Any, Any>>(std::move(src)));
      }
      case TypeIndex::kTVMFFIDict: {
        Dict<Any, Any> dict = AnyUnsafe::MoveFromAnyAfterCheck<Dict<Any, Any>>(std::move(src));
        return HashMapBase(static_cast<const MapBaseObj*>(dict.get()));
      }
      case TypeIndex::kTVMFFIShape: {
        return HashShape(AnyUnsafe::MoveFromAnyAfterCheck<Shape>(std::move(src)));
      }
      case TypeIndex::kTVMFFITensor: {
        return HashTensor(AnyUnsafe::MoveFromAnyAfterCheck<Tensor>(std::move(src)));
      }
      default: {
        return HashObject(AnyUnsafe::MoveFromAnyAfterCheck<ObjectRef>(std::move(src)));
      }
    }
  }

  uint64_t HashObject(const ObjectRef& obj) {
    // NOTE: invariant: lhs and rhs are already the same type
    const TVMFFITypeInfo* type_info = TVMFFIGetTypeInfo(obj->type_index());
    if (type_info->metadata == nullptr) {
      TVM_FFI_THROW(TypeError) << "Type metadata is not set for type `"
                               << String(type_info->type_key)
                               << "`, so StructuralHash is not supported for this type";
    }
    if (type_info->metadata->structural_eq_hash_kind == kTVMFFISEqHashKindUnsupported) {
      TVM_FFI_THROW(TypeError) << "_type_s_eq_hash_kind is not set for type `"
                               << String(type_info->type_key)
                               << "`, so StructuralHash is not supported for this type";
    }

    auto structural_eq_hash_kind = type_info->metadata->structural_eq_hash_kind;
    if (structural_eq_hash_kind == kTVMFFISEqHashKindUnsupported) {
      // Fallback to pointer hash
      return std::hash<const Object*>()(obj.get());
    }
    // return recored hash value if it is already computed
    auto it = hash_memo_.find(obj);
    if (it != hash_memo_.end()) {
      return it->second;
    }

    uint64_t hash_value;
    if (structural_eq_hash_kind != kTVMFFISEqHashKindFreeVar) {
      hash_value = HashFields(obj, type_info, obj->GetTypeKeyHash());
    } else {
      // FreeVar path. In a non-recursive def region the FreeVar's own
      // sub-fields are walked outside the def region (nested free vars
      // there hash by pointer, matching use semantics), so we clamp
      // ``def_region_kind_`` to ``kNone`` around the HashFields call and
      // restore before the FreeVar-level injection below.
      //
      // We always call HashFields, even in use mode where the returned
      // ``hash_value`` is discarded by the pointer-hash fallback. The
      // walk's side effect on ``free_var_counter_`` (incremented for
      // every nested FreeVar reached via SEqHashDef-tagged sub-fields)
      // is observable to FreeVars hashed later in the same traversal;
      // skipping the walk would silently change those subsequent hashes.
      TVMFFIDefRegionKind saved_def_region_kind = def_region_kind_;
      if (def_region_kind_ == kTVMFFIDefRegionKindNonRecursive) {
        def_region_kind_ = kTVMFFIDefRegionKindNone;
      }
      hash_value = HashFields(obj, type_info, obj->GetTypeKeyHash());
      def_region_kind_ = saved_def_region_kind;
      if (def_region_kind_ != kTVMFFIDefRegionKindNone) {
        // use lexical order of free var and its type
        hash_value = details::StableHashCombine(hash_value, free_var_counter_++);
      } else {
        // Fallback to pointer hash; we are not in a def region.
        hash_value = std::hash<const Object*>()(obj.get());
      }
    }

    // if it is a DAG node, also record the lexical order of graph counter
    // this helps to distinguish DAG from trees.
    if (structural_eq_hash_kind == kTVMFFISEqHashKindDAGNode) {
      hash_value = details::StableHashCombine(hash_value, graph_node_counter_++);
    }
    // record the hash value for this object
    hash_memo_[obj] = hash_value;
    return hash_value;
  }

  // Hash an object's fields (generic walk or via the custom __ffi_s_hash__
  // callback). Does not touch the FreeVar def-region clamp — that lives
  // inline in HashObject's FreeVar branch, which wraps this helper.
  uint64_t HashFields(const ObjectRef& obj, const TVMFFITypeInfo* type_info, uint64_t init_hash) {
    static reflection::TypeAttrColumn custom_s_hash =
        reflection::TypeAttrColumn(reflection::type_attr::kSHash);

    // Mirror structural equality: a custom hook serves subclasses unless one
    // registers a more specific override.
    AnyView custom = custom_s_hash.GetInherited(type_info->type_index);

    if (custom == nullptr) {
      // go over the content and hash the fields
      reflection::ForEachFieldInfo(type_info, [&](const TVMFFIFieldInfo* field_info) {
        // skip fields that are marked as structural eq hash ignore
        if (!(field_info->flags & kTVMFFIFieldFlagBitMaskSEqHashIgnore)) {
          reflection::FieldGetter getter(field_info);
          Any field_value = getter(obj);
          // Dispatch on the def-region flags (mirror of the equality side).
          constexpr int64_t kSEqHashDefAny = kTVMFFIFieldFlagBitMaskSEqHashDefRecursive |
                                             kTVMFFIFieldFlagBitMaskSEqHashDefNonRecursive;
          if (field_info->flags & kSEqHashDefAny) {
            TVMFFIDefRegionKind new_kind =
                (field_info->flags & kTVMFFIFieldFlagBitMaskSEqHashDefNonRecursive)
                    ? kTVMFFIDefRegionKindNonRecursive
                    : kTVMFFIDefRegionKindRecursive;
            std::swap(new_kind, def_region_kind_);
            init_hash = details::StableHashCombine(init_hash, HashAny(field_value));
            std::swap(new_kind, def_region_kind_);
          } else {
            init_hash = details::StableHashCombine(init_hash, HashAny(field_value));
          }
        }
      });
    } else {
      if (s_hash_callback_ == nullptr) {
        s_hash_callback_ =
            // Third parameter is a ``TVMFFIDefRegionKind`` (passed on the wire
            // as ``int`` to keep the FFI signature stable across language
            // boundaries).
            ffi::Function::FromTyped([this](AnyView val, uint64_t inner_init, int def_region_kind) {
              TVMFFIDefRegionKind new_kind =
                  (def_region_kind == kTVMFFIDefRegionKindNone)
                      ? def_region_kind_
                      : static_cast<TVMFFIDefRegionKind>(def_region_kind);
              std::swap(new_kind, def_region_kind_);
              uint64_t hv = HashAny(val);
              std::swap(new_kind, def_region_kind_);
              // we explicitly bitcast the result from `uint64_t` to `int64_t`.
              // The range of `uint64_t` is too large to fit as `int64_t`, so if we don't bitcast,
              // it will trigger an overflow error in `uint64_t` -> `Any` conversion.
              return static_cast<int64_t>(details::StableHashCombine(inner_init, hv));
            });
      }
      init_hash =
          custom.cast<ffi::Function>()(obj, static_cast<int64_t>(init_hash), s_hash_callback_)
              .cast<uint64_t>();
    }
    return init_hash;
  }

  // NOLINTNEXTLINE(performance-unnecessary-value-param)
  uint64_t HashArray(Array<Any> arr) { return HashSequence(std::move(arr)); }

  // NOLINTNEXTLINE(performance-unnecessary-value-param)
  uint64_t HashList(List<Any> list) { return HashSequence(std::move(list)); }

  template <typename SeqType>
  // NOLINTNEXTLINE(performance-unnecessary-value-param)
  uint64_t HashSequence(SeqType seq) {
    uint64_t hash_value = details::StableHashCombine(seq->GetTypeKeyHash(), seq.size());
    for (const auto& elem : seq) {
      hash_value = details::StableHashCombine(hash_value, HashAny(elem));
    }
    return hash_value;
  }

  // Find an order independent hash value for a given Any.
  // Order independent hash value means the hash value will remain stable independent
  // of the order we hash the content at the current context.
  // This property is needed to support stable hash for map.
  std::optional<uint64_t> FindOrderIndependentHash(const Any& src) {
    using ffi::details::AnyUnsafe;
    const TVMFFIAny* src_data = AnyUnsafe::TVMFFIAnyPtrFromAny(src);

    if (src_data->type_index < TypeIndex::kTVMFFIStaticObjectBegin) {
      if (src_data->type_index == TypeIndex::kTVMFFISmallStr) {
        // for small string, we use the same type key hash as normal string
        // so heap allocated string and on stack string will have the same hash
        return details::StableHashCombine(
            TypeIndex::kTVMFFIStr,
            details::StableHashBytes(src_data->v_bytes, src_data->small_str_len));
      }
      // this is POD data, we can just hash the value
      return details::StableHashCombine(src_data->type_index, src_data->v_uint64);
    } else {
      if (src_data->type_index == TypeIndex::kTVMFFIStr ||
          src_data->type_index == TypeIndex::kTVMFFIBytes) {
        const details::BytesObjBase* src_str =
            AnyUnsafe::CopyFromAnyViewAfterCheck<const details::BytesObjBase*>(src);
        // return same hash as AnyHash
        return details::StableHashCombine(src_data->type_index,
                                          details::StableHashBytes(src_str->data, src_str->size));
      } else {
        // if the hash of the object is already computed, return it
        auto it = hash_memo_.find(src.cast<ObjectRef>());
        if (it != hash_memo_.end()) {
          return it->second;
        }
        return std::nullopt;
      }
    }
  }

  // NOLINTNEXTLINE(performance-unnecessary-value-param)
  uint64_t HashMap(Map<Any, Any> map) {
    return HashMapBase(static_cast<const MapBaseObj*>(map.get()));
  }

  uint64_t HashMapBase(const MapBaseObj* map) {
    // Compute a deterministic hash value for the map.
    uint64_t hash_value = details::StableHashCombine(map->GetTypeKeyHash(), map->size());
    std::vector<std::pair<uint64_t, Any>> items;
    for (const auto& [key, value] : *map) {
      // if we cannot find order independent hash, we skip the key
      if (auto hash_key = FindOrderIndependentHash(key)) {
        items.emplace_back(*hash_key, value);
      }
    }
    // sort the items by the hash key, so the hash value is deterministic
    // and independent of the order of insertion
    std::sort(items.begin(), items.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    for (size_t i = 0; i < items.size();) {
      size_t k = i + 1;
      for (; k < items.size() && items[k].first == items[i].first; ++k) {
      }
      // detect ties, which are rare, but we need to skip value hash during ties
      // to make sure that the hash value is deterministic.
      if (k == i + 1) {
        // no ties, we just hash the key and value
        hash_value = details::StableHashCombine(hash_value, items[i].first);
        hash_value = details::StableHashCombine(hash_value, HashAny(items[i].second));
      } else {
        // ties occur, we skip the value hash to make sure that the hash value is deterministic.
        hash_value = details::StableHashCombine(hash_value, items[i].first);
      }
      i = k;
    }
    return hash_value;
  }

  // NOLINTNEXTLINE(performance-unnecessary-value-param)
  uint64_t HashShape(Shape shape) {
    uint64_t hash_value = details::StableHashCombine(shape->GetTypeKeyHash(), shape.size());
    for (int64_t i : shape) {
      hash_value = details::StableHashCombine(hash_value, i);
    }
    return hash_value;
  }

  // NOLINTNEXTLINE(performance-unnecessary-value-param)
  uint64_t HashTensor(Tensor tensor) {
    uint64_t hash_value = details::StableHashCombine(tensor->GetTypeKeyHash(), tensor.ndim());
    for (int i = 0; i < tensor.ndim(); ++i) {
      hash_value = details::StableHashCombine(hash_value, tensor.size(i));
    }
    TVMFFIAny temp;
    temp.v_uint64 = 0;
    temp.v_dtype = tensor.dtype();
    hash_value = details::StableHashCombine(hash_value, temp.v_int64);

    if (!skip_tensor_content_) {
      TVM_FFI_ICHECK_EQ(tensor.device().device_type, kDLCPU) << "can only hash CPU tensor";
      TVM_FFI_ICHECK(tensor.IsContiguous()) << "Can only hash contiguous tensor";
      size_t data_size = GetDataSize(tensor.numel(), tensor.dtype());
      uint64_t data_hash =
          details::StableHashBytes(static_cast<const char*>(tensor.data_ptr()), data_size);
      hash_value = details::StableHashCombine(hash_value, data_hash);
    }
    return hash_value;
  }

  // Current def-region kind. ``kNone`` means we are not in a def region; free
  // vars hash by pointer. ``kRecursive`` and ``kNonRecursive`` enable
  // ``free_var_counter_``-based hashing for the field-flag-driven walk and
  // for the custom-callback path respectively (see HashObject).
  TVMFFIDefRegionKind def_region_kind_{kTVMFFIDefRegionKindNone};
  bool skip_tensor_content_{false};
  // free var counter.
  uint32_t free_var_counter_{0};
  // graph node counter.
  uint32_t graph_node_counter_{0};
  // lazily initialize custom hash function
  ffi::Function s_hash_callback_ = nullptr;
  // map from lhs to rhs
  std::unordered_map<ObjectRef, uint64_t, ObjectPtrHash, ObjectPtrEqual> hash_memo_;
};

uint64_t StructuralHash::Hash(const Any& value, bool map_free_vars, bool skip_tensor_content) {
  StructuralHashHandler handler;
  handler.def_region_kind_ =
      map_free_vars ? kTVMFFIDefRegionKindRecursive : kTVMFFIDefRegionKindNone;
  handler.skip_tensor_content_ = skip_tensor_content;
  return handler.HashAny(value);
}

static int64_t FFIStructuralHash(const Any& value, bool map_free_vars, bool skip_tensor_content) {
  uint64_t result = StructuralHash::Hash(value, map_free_vars, skip_tensor_content);
  // we explicitly bitcast the result from `uint64_t` to `int64_t`.
  // The range of `uint64_t` is too large to fit as `int64_t`, so if we don't bitcast,
  // it will trigger an overflow error in `uint64_t` -> `Any` conversion.
  return static_cast<int64_t>(result);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("ffi.StructuralHash", FFIStructuralHash);
  refl::EnsureTypeAttrColumn(refl::type_attr::kSHash);
}

}  // namespace ffi
}  // namespace tvm
