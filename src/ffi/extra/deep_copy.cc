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
#include <tvm/ffi/container/list.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/container/shape.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/deep_copy.h>
#include <tvm/ffi/reflection/accessor.h>
#include <tvm/ffi/reflection/registry.h>

#include <unordered_map>
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
    int32_t ti = obj->type_index();
    // Strings, bytes, and shapes are immutable — return as-is.
    if (ti == TypeIndex::kTVMFFIStr || ti == TypeIndex::kTVMFFIBytes ||
        ti == TypeIndex::kTVMFFIShape) {
      return value;
    }
    if (ti == TypeIndex::kTVMFFIArray) {
      // NOTE: The new array is registered in copy_map_ only after all elements
      // are resolved.  This means a cyclic self-reference (array containing
      // itself) would not preserve pointer equality.  This is acceptable
      // because Array is immutable and such cycles cannot be constructed.
      const ArrayObj* orig = value.as<ArrayObj>();
      Array<Any> new_arr;
      new_arr.reserve(static_cast<int64_t>(orig->size()));
      for (const Any& elem : *orig) {
        new_arr.push_back(Resolve(elem));
      }
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
      // NOTE: Same as Array above — Map is immutable, so cyclic
      // self-references cannot occur and late registration is safe.
      const MapObj* orig = value.as<MapObj>();
      Map<Any, Any> new_map;
      for (const auto& [k, v] : *orig) {
        new_map.Set(Resolve(k), Resolve(v));
      }
      copy_map_[obj] = new_map;
      return new_map;
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

  reflection::TypeAttrColumn* column_;
  std::unordered_map<const Object*, Any> copy_map_;
  std::vector<const Object*> resolve_queue_;
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
