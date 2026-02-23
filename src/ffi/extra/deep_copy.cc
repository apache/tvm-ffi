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
 * \file src/ffi/extra/deep_copy.cc
 *
 * \brief Reflection-based deep copy utilities.
 */
#include <tvm/ffi/any.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/dict.h>
#include <tvm/ffi/container/list.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/container/shape.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/deep_copy.h>
#include <tvm/ffi/reflection/accessor.h>
#include <tvm/ffi/reflection/registry.h>

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm {
namespace ffi {

/*!
 * \brief Deep copier with memoization.
 *
 * - Arrays / Maps: Resolve() recurses to rebuild with resolved children.
 * - Copyable objects: shallow-copied immediately into copy_map_ (so cyclic
 *   back-references resolve), then queued for field resolution.
 * - The queue is drained iteratively by Run(), bounding recursion depth
 *   to container nesting rather than object-graph depth.
 * - Shared references are preserved: the same original maps to the same copy.
 */
class ObjectDeepCopier {
 public:
  explicit ObjectDeepCopier(reflection::TypeAttrColumn* column) : column_(column) {}

  Any Run(const Any& value) {
    if (value.type_index() < TypeIndex::kTVMFFIStaticObjectBegin) return value;
    Any result = Resolve(value);
    // NOLINTNEXTLINE(modernize-loop-convert): queue grows during iteration
    for (size_t i = 0; i < resolve_queue_.size(); ++i) {
      ResolveFields(resolve_queue_[i]);
    }
    if (has_deferred_) {
      FixupDeferredReferences();
    }
    return result;
  }

 private:
  /*! \brief Resolve a value: pass through primitives, copy/rebuild objects. */
  Any Resolve(const Any& value) {
    if (value.type_index() < TypeIndex::kTVMFFIStaticObjectBegin) {
      return value;
    }
    const Object* obj = value.as<Object>();
    if (auto it = copy_map_.find(obj); it != copy_map_.end()) {
      return it->second;
    }
    // If this object is currently being built (in-progress immutable container),
    // return the original as a placeholder.  A fixup pass will replace these
    // stale references inside mutable containers (List/Dict) after all copies
    // are fully constructed.
    if (in_progress_.count(obj)) {
      has_deferred_ = true;
      return value;
    }
    int32_t ti = obj->type_index();
    // Strings, bytes, and shapes are immutable — return as-is.
    if (ti == TypeIndex::kTVMFFIStr || ti == TypeIndex::kTVMFFIBytes ||
        ti == TypeIndex::kTVMFFIShape) {
      return value;
    }
    if (ti == TypeIndex::kTVMFFIArray) {
      // Array is immutable (COW), so we cannot register early — COW would
      // create a new internal object on push_back when refcount > 1.
      // Instead, mark in-progress and fix up deferred back-references later.
      in_progress_.insert(obj);
      const ArrayObj* orig = value.as<ArrayObj>();
      Array<Any> new_arr;
      new_arr.reserve(static_cast<int64_t>(orig->size()));
      for (const Any& elem : *orig) {
        new_arr.push_back(Resolve(elem));
      }
      in_progress_.erase(obj);
      copy_map_[obj] = new_arr;
      return new_arr;
    }
    if (ti == TypeIndex::kTVMFFIList) {
      // List is mutable, so cyclic self-references are possible.
      // Register the empty copy in copy_map_ before resolving children
      // so that back-references resolve to the same new List.
      const ListObj* orig = value.as<ListObj>();
      List<Any> new_list;
      new_list.reserve(static_cast<int64_t>(orig->size()));
      copy_map_[obj] = new_list;
      for (const Any& elem : *orig) {
        new_list.push_back(Resolve(elem));
      }
      return new_list;
    }
    if (ti == TypeIndex::kTVMFFIMap) {
      // Map is immutable (COW), same treatment as Array above.
      in_progress_.insert(obj);
      const MapObj* orig = value.as<MapObj>();
      Map<Any, Any> new_map;
      for (const auto& [k, v] : *orig) {
        new_map.Set(Resolve(k), Resolve(v));
      }
      in_progress_.erase(obj);
      copy_map_[obj] = new_map;
      return new_map;
    }
    if (ti == TypeIndex::kTVMFFIDict) {
      // Dict is mutable, so cyclic self-references are possible.
      // Register the empty copy in copy_map_ before resolving children
      // so that back-references resolve to the same new Dict.
      const DictObj* orig = value.as<DictObj>();
      Dict<Any, Any> new_dict;
      copy_map_[obj] = new_dict;
      for (const auto& [k, v] : *orig) {
        new_dict.Set(Resolve(k), Resolve(v));
      }
      return new_dict;
    }
    // General object: shallow-copy, register, and queue for field resolution.
    const TVMFFITypeInfo* type_info = TVMFFIGetTypeInfo(ti);
    TVM_FFI_ICHECK((*column_)[ti] != nullptr)
        << "Cannot deep copy object of type \""
        << std::string_view(type_info->type_key.data, type_info->type_key.size)
        << "\" because it is not copy-constructible";
    Function copy_fn = (*column_)[ti].cast<Function>();
    Any copy = copy_fn(obj);
    copy_map_[obj] = copy;
    resolve_queue_.push_back(copy.as<Object>());
    return copy;
  }

  void ResolveFields(const Object* copy_obj) {
    const TVMFFITypeInfo* type_info = TVMFFIGetTypeInfo(copy_obj->type_index());
    reflection::ForEachFieldInfo(type_info, [&](const TVMFFIFieldInfo* fi) {
      reflection::FieldGetter getter(fi);
      Any fv = getter(copy_obj);
      if (fv.type_index() < TypeIndex::kTVMFFIStaticObjectBegin) return;
      Any resolved = Resolve(fv);
      if (!fv.same_as(resolved)) {
        reflection::FieldSetter setter(fi);
        setter(copy_obj, resolved);
      }
    });
  }

  /*!
   * \brief Replace stale original-object references inside mutable containers.
   *
   * When an immutable container (Array/Map) is in-progress and a mutable child
   * (List/Dict) references it back, the original is stored as a placeholder.
   * After all copies are built, this pass replaces those placeholders with
   * the actual copies from copy_map_.
   */
  void FixupDeferredReferences() {
    for (auto& [orig_ptr, copy_any] : copy_map_) {
      const Object* copy_obj = copy_any.as<Object>();
      if (!copy_obj) continue;
      int32_t ti = copy_obj->type_index();
      if (ti == TypeIndex::kTVMFFIList) {
        FixupList(copy_any);
      } else if (ti == TypeIndex::kTVMFFIDict) {
        FixupDict(copy_any);
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
    // Collect value fixups (safe to apply in-place since keys don't change).
    std::vector<std::pair<Any, Any>> fixups;
    for (const auto& [k, v] : *dict_obj) {
      if (v.type_index() < TypeIndex::kTVMFFIStaticObjectBegin) continue;
      const Object* v_obj = v.as<Object>();
      if (!v_obj) continue;
      auto it = copy_map_.find(v_obj);
      if (it != copy_map_.end()) {
        fixups.emplace_back(k, it->second);
      }
    }
    for (auto& [key, new_val] : fixups) {
      dict.Set(key, new_val);
    }
  }

  reflection::TypeAttrColumn* column_;
  std::unordered_map<const Object*, Any> copy_map_;
  std::vector<const Object*> resolve_queue_;
  std::unordered_set<const Object*> in_progress_;
  bool has_deferred_ = false;
};

Any DeepCopy(const Any& value) {
  static reflection::TypeAttrColumn column(reflection::type_attr::kShallowCopy);
  ObjectDeepCopier copier(&column);
  return copier.Run(value);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::EnsureTypeAttrColumn(refl::type_attr::kShallowCopy);
  refl::GlobalDef().def("ffi.DeepCopy", DeepCopy);
}

}  // namespace ffi
}  // namespace tvm
