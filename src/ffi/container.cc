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
 * \file src/ffi/container.cc
 */
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/dict.h>
#include <tvm/ffi/container/list.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/device.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>

#include <type_traits>

#include "object_internal.h"

namespace tvm {
namespace ffi {

namespace {

template <typename MapObjType>
Any MutateMapValues(const AnyView& value, const MapObjType* source, bool allow_inplace,
                    const Function& mutator) {
  bool mutate_inplace = allow_inplace && source->unique();
  if (mutate_inplace) {
    MapObjType* target = const_cast<MapObjType*>(source);
    int64_t index = 0;
    for (auto it = target->begin(); it != target->end(); ++it, ++index) {
      const Any& old_value = it->second;
      Any mapped = mutator(old_value, index, true);
      if (!old_value.same_as(mapped)) {
        it->second = std::move(mapped);
      }
    }
    return Any(value);
  }

  if constexpr (std::is_same_v<MapObjType, MapObj>) {
    // Map is immutable, so its source iterator remains stable through
    // callbacks. Defer the shallow copy until the first actual change.
    ObjectPtr<Object> output = nullptr;
    MapBaseObj::iterator output_it;
    int64_t index = 0;
    for (auto source_it = source->begin(); source_it != source->end(); ++source_it, ++index) {
      const Any& old_value = source_it->second;
      Any mapped = mutator(old_value, index, false);
      bool changed = !old_value.same_as(mapped);
      if (output == nullptr) {
        if (!changed) {
          continue;
        }
        output = MapObj::ShallowCopy(source);
        output_it = static_cast<MapBaseObj*>(output.get())->begin();
        for (int64_t previous = 0; previous < index; ++previous) {
          ++output_it;
        }
      }
      if (changed) {
        output_it->second = std::move(mapped);
      }
      ++output_it;
    }
    if (output == nullptr) {
      return Any(value);
    }
    return Any(ObjectRef(std::move(output)));
  } else {
    // Dict is mutable. Copy before the first callback so re-entrant changes to
    // the source cannot invalidate iteration or enter the mapped snapshot.
    ObjectPtr<Object> output = DictObj::ShallowCopy(source);
    auto* target = static_cast<DictObj*>(output.get());
    bool changed = false;
    int64_t index = 0;
    for (auto it = target->begin(); it != target->end(); ++it, ++index) {
      const Any& old_value = it->second;
      Any mapped = mutator(old_value, index, false);
      if (!old_value.same_as(mapped)) {
        it->second = std::move(mapped);
        changed = true;
      }
    }
    if (!changed) {
      return Any(value);
    }
    return Any(ObjectRef(std::move(output)));
  }
}

template <typename MapObjType>
void MutateMapValuesPacked(PackedArgs args, Any* result) {
  TVM_FFI_ICHECK_EQ(args.size(), 3);
  AnyView value = args[0];
  bool allow_inplace = args[1].cast<bool>();
  Function mutator = args[2].cast<Function>();
  *result = MutateMapValues(value, value.cast<const MapObjType*>(), allow_inplace, mutator);
}

/*!
 * \brief Recursively scan an Any element for the first non-CPU tensor device.
 * \param elem The element to inspect.
 * \param out Output device; written only when a non-CPU tensor is found.
 * \return true if a non-CPU tensor was found.
 */
bool FindFirstNonCPUDevice(const Any& elem, DLDevice* out) {
  switch (elem.type_index()) {
    case TypeIndex::kTVMFFITensor: {
      const auto* tensor = elem.as<TensorObj>();
      if (tensor->device.device_type != kDLCPU) {
        *out = tensor->device;
        return true;
      }
      break;
    }
    case TypeIndex::kTVMFFIArray:
    case TypeIndex::kTVMFFIList: {
      const auto* seq = elem.as<SeqBaseObj>();
      for (const auto& it : *seq) {
        if (FindFirstNonCPUDevice(it, out)) return true;
      }
      break;
    }
    case TypeIndex::kTVMFFIMap:
    case TypeIndex::kTVMFFIDict: {
      const auto* map = elem.as<MapBaseObj>();
      for (const auto& it : *map) {
        if (FindFirstNonCPUDevice(it.second, out)) return true;
      }
      break;
    }
    default:
      break;
  }
  return false;
}
}  // namespace

// Favor struct outside function scope as MSVC may have bug for in fn scope struct.
class MapForwardIterFunctor {
 public:
  MapForwardIterFunctor(ffi::MapObj::iterator iter, ffi::MapObj::iterator end)
      : iter_(iter), end_(end) {}
  // 0 get current key
  // 1 get current value
  // 2 move to next: return true if success, false if end
  Any operator()(int command) const {
    if (command == 0) {
      return (*iter_).first;
    } else if (command == 1) {
      return (*iter_).second;
    } else {
      ++iter_;
      if (iter_ == end_) {
        return false;
      }
      return true;
    }
  }

 private:
  mutable ffi::MapObj::iterator iter_;
  ffi::MapObj::iterator end_;
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::EnsureTypeAttrColumn(refl::type_attr::kAnyHash);
  refl::EnsureTypeAttrColumn(refl::type_attr::kAnyEqual);
  refl::GlobalDef()
      .def_packed("ffi.Array",
                  [](ffi::PackedArgs args, Any* ret) {
                    *ret = Array<Any>(args.data(), args.data() + args.size());
                  })
      .def("ffi.ArrayGetItem", [](const ffi::ArrayObj* n, int64_t i) -> Any { return n->at(i); })
      .def("ffi.ArraySize",
           [](const ffi::ArrayObj* n) -> int64_t { return static_cast<int64_t>(n->size()); })
      .def("ffi.ArrayContains",
           [](const ffi::ArrayObj* n, const Any& value) -> bool {
             AnyEqual eq;
             return std::any_of(n->begin(), n->end(),
                                [&](const Any& elem) { return eq(elem, value); });
           })
      .def_packed("ffi.List",
                  [](ffi::PackedArgs args, Any* ret) {
                    *ret = List<Any>(args.data(), args.data() + args.size());
                  })
      .def("ffi.ListGetItem", [](const ffi::ListObj* n, int64_t i) -> Any { return n->at(i); })
      .def("ffi.ListSetItem",
           [](ffi::List<Any> n, int64_t i, Any value) -> void { n.Set(i, std::move(value)); })
      .def("ffi.ListSize",
           [](const ffi::ListObj* n) -> int64_t { return static_cast<int64_t>(n->size()); })
      .def("ffi.ListContains",
           [](const ffi::ListObj* n, const Any& value) -> bool {
             AnyEqual eq;
             return std::any_of(n->begin(), n->end(),
                                [&](const Any& elem) { return eq(elem, value); });
           })
      .def("ffi.ListAppend", [](ffi::List<Any> n, const Any& value) -> void { n.push_back(value); })
      .def("ffi.ListInsert",
           [](ffi::List<Any> n, int64_t i, const Any& value) -> void {
             n.insert(n.begin() + i, value);
           })
      .def("ffi.ListPop",
           [](const ffi::List<Any>& n, int64_t i) -> Any {
             ffi::ListObj* obj = n.GetListObj();
             Any value = obj->at(i);
             obj->erase(i);
             return value;
           })
      .def("ffi.ListErase",
           [](const ffi::List<Any>& n, int64_t i) -> void { n.GetListObj()->erase(i); })
      .def("ffi.ListEraseRange",
           [](const ffi::List<Any>& n, int64_t start, int64_t stop) -> void {
             n.GetListObj()->erase(start, stop);
           })
      .def("ffi.ListReplaceSlice",
           [](ffi::List<Any> n, int64_t start, int64_t stop,
              const ffi::List<Any>& replacement) -> void {
             // Snapshot replacement before erasing in case n and replacement alias the same object.
             ffi::List<Any> rep_copy = n.same_as(replacement)
                                           ? ffi::List<Any>(replacement.begin(), replacement.end())
                                           : replacement;
             n.GetListObj()->erase(start, stop);
             if (rep_copy.empty()) {
               return;
             }
             const ffi::ListObj* replacement_obj = rep_copy.GetListObj();
             TVM_FFI_ICHECK(replacement_obj != nullptr);
             n.insert(n.begin() + start, replacement_obj->begin(), replacement_obj->end());
           })
      .def("ffi.ListReverse",
           [](const ffi::List<Any>& n) -> void {
             ffi::ListObj* obj = n.GetListObj();
             if (obj != nullptr) {
               obj->SeqBaseObj::Reverse();
             }
           })
      .def("ffi.ListClear", [](ffi::List<Any> n) -> void { n.clear(); })
      .def_packed("ffi.Map",
                  [](ffi::PackedArgs args, Any* ret) {
                    TVM_FFI_ICHECK_EQ(args.size() % 2, 0);
                    Map<Any, Any> data;
                    for (int i = 0; i < args.size(); i += 2) {
                      data.Set(args[i], args[i + 1]);
                    }
                    *ret = data;
                  })
      .def("ffi.MapSize",
           [](const ffi::MapObj* n) -> int64_t { return static_cast<int64_t>(n->size()); })
      .def("ffi.MapGetItem", [](const ffi::MapObj* n, const Any& k) -> Any { return n->at(k); })
      .def("ffi.MapCount",
           [](const ffi::MapObj* n, const Any& k) -> int64_t {
             return static_cast<int64_t>(n->count(k));
           })
      .def("ffi.MapForwardIterFunctor",
           [](const ffi::MapObj* n) -> ffi::Function {
             return ffi::Function::FromTyped(MapForwardIterFunctor(n->begin(), n->end()));
           })
      .def_packed("ffi.MapMutateValues",
                  [](ffi::PackedArgs args, Any* ret) { MutateMapValuesPacked<MapObj>(args, ret); })
      .def("ffi.MapGetItemOrMissing",
           [](const ffi::MapObj* n, const Any& k) -> Any {
             auto it = n->find(k);
             if (it != n->end()) {
               return it->second;
             }
             return GetMissingObject();
           })
      .def_packed("ffi.Dict",
                  [](ffi::PackedArgs args, Any* ret) {
                    TVM_FFI_ICHECK_EQ(args.size() % 2, 0);
                    Dict<Any, Any> data;
                    for (int i = 0; i < args.size(); i += 2) {
                      data.Set(args[i], args[i + 1]);
                    }
                    *ret = data;
                  })
      .def("ffi.DictSize",
           [](const ffi::DictObj* n) -> int64_t { return static_cast<int64_t>(n->size()); })
      .def("ffi.DictGetItem", [](const ffi::DictObj* n, const Any& k) -> Any { return n->at(k); })
      .def("ffi.DictSetItem",
           [](ffi::Dict<Any, Any> d, const Any& k, const Any& v) -> void { d.Set(k, v); })
      .def("ffi.DictCount",
           [](const ffi::DictObj* n, const Any& k) -> int64_t {
             return static_cast<int64_t>(n->count(k));
           })
      .def("ffi.DictErase", [](ffi::Dict<Any, Any> d, const Any& k) -> void { d.erase(k); })
      .def("ffi.DictClear", [](ffi::Dict<Any, Any> d) -> void { d.clear(); })
      .def("ffi.DictForwardIterFunctor",
           [](const ffi::DictObj* n) -> ffi::Function {
             return ffi::Function::FromTyped(MapForwardIterFunctor(n->begin(), n->end()));
           })
      .def_packed("ffi.DictMutateValues",
                  [](ffi::PackedArgs args, Any* ret) { MutateMapValuesPacked<DictObj>(args, ret); })
      .def("ffi.DictGetItemOrMissing",
           [](const ffi::DictObj* n, const Any& k) -> Any {
             auto it = n->find(k);
             if (it != n->end()) {
               return it->second;
             }
             return GetMissingObject();
           })
      .def("ffi.ContainerFindFirstNonCPUDevice", [](const Any& container) -> DLDevice {
        DLDevice result{kDLCPU, 0};
        FindFirstNonCPUDevice(container, &result);
        return result;
      });
}
}  // namespace ffi
}  // namespace tvm
