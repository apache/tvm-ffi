
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
#include <tvm/ffi/container/list.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>

namespace tvm {
namespace ffi {

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

ObjectRef GetMissingObject() {
  static ObjectRef missing_obj(make_object<Object>());
  return missing_obj;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
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
      .def("ffi.MapGetMissingObject", GetMissingObject)
      .def("ffi.MapGetItemOrMissing", [](const ffi::MapObj* n, const Any& k) -> Any {
        try {
          return n->at(k);
        } catch (const tvm::ffi::Error& e) {
          return GetMissingObject();
        }
      });
}
}  // namespace ffi
}  // namespace tvm
