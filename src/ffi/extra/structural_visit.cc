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
 * \file src/ffi/extra/structural_visit.cc
 * \brief Structural visit implementation.
 */
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/dict.h>
#include <tvm/ffi/container/list.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/accessor.h>
#include <tvm/ffi/reflection/registry.h>

namespace tvm {
namespace ffi {

// ---------------------------------------------------------------------------
// Built-in container structural visit.
// ---------------------------------------------------------------------------

namespace details {

/*!
 * \brief Runtime structural walk for callback arrays.
 *
 * \param root The root value to visit.
 * \param callbacks Runtime callback entries of ``(type_index, ffi::Function)`` invoked as
 *                  ``callback(value)``.
 * \param callbacks_with_def_region_kind Runtime callback entries of ``(type_index, ffi::Function)``
 *                                       invoked as ``callback(value, def_region_kind)``.
 * \param order Integer value of \ref WalkOrder.
 * \return Expected interrupt state. An error means traversal failed.
 */
Expected<Optional<VisitInterrupt>> StructuralWalkExpected(
    AnyView root, const Array<Tuple<int32_t, Function>>& callbacks,
    const Array<Tuple<int32_t, Function>>& callbacks_with_def_region_kind, int order) noexcept {
  auto dispatch = [callbacks, callbacks_with_def_region_kind](
                      AnyView x, TVMFFIDefRegionKind kind) -> Expected<WalkResult> {
    for (const auto& entry : callbacks) {
      int32_t type_index = entry.template get<0>();
      if (!RuntimeTypeIndexMatch(x.type_index(), type_index)) {
        continue;
      }
      Function fn = entry.template get<1>();
      return fn.CallExpected<WalkResult>(x);
    }
    for (const auto& entry : callbacks_with_def_region_kind) {
      int32_t type_index = entry.template get<0>();
      if (!RuntimeTypeIndexMatch(x.type_index(), type_index)) {
        continue;
      }
      Function fn = entry.template get<1>();
      return fn.CallExpected<WalkResult>(x, kind);
    }
    return WalkResult::Advance();
  };

  if (order == static_cast<int>(WalkOrder::kPreOrder)) {
    using Visitor = StructuralWalkCallbackVisitorObj<WalkOrder::kPreOrder, decltype(dispatch)>;
    StructuralVisitor visitor(make_object<Visitor>(std::move(dispatch)));
    return visitor->VisitExpected(root);
  } else {
    using Visitor = StructuralWalkCallbackVisitorObj<WalkOrder::kPostOrder, decltype(dispatch)>;
    StructuralVisitor visitor(make_object<Visitor>(std::move(dispatch)));
    return visitor->VisitExpected(root);
  }
}

/*! \brief Visit entries in a sequence container. */
TVMFFIAny VisitSeqContainer(StructuralVisitorObj* visitor, const SeqBaseObj* seq) noexcept {
  for (const Any& item : *seq) {
    auto interrupt = visitor->VisitExpected(item);
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(interrupt);
  }
  return ExpectedUnsafe::MoveToTVMFFIAny(Expected<Optional<VisitInterrupt>>(std::nullopt));
}

/*! \brief Visit keys and values in a map container. */
TVMFFIAny VisitMapContainer(StructuralVisitorObj* visitor, const MapBaseObj* map) noexcept {
  for (const auto& kv : *map) {
    auto key_interrupt = visitor->VisitExpected(kv.first);
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(key_interrupt);
    auto val_interrupt = visitor->VisitExpected(kv.second);
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(val_interrupt);
  }
  return ExpectedUnsafe::MoveToTVMFFIAny(Expected<Optional<VisitInterrupt>>(std::nullopt));
}

/*! \brief Structural visit hook for ArrayObj. */
TVMFFIAny VisitArray(StructuralVisitorObj* visitor, AnyView value) noexcept {
  const auto* array = value.cast<const ArrayObj*>();
  return VisitSeqContainer(visitor, array);
}

/*! \brief Structural visit hook for ListObj. */
TVMFFIAny VisitList(StructuralVisitorObj* visitor, AnyView value) noexcept {
  const auto* list = value.cast<const ListObj*>();
  return VisitSeqContainer(visitor, list);
}

/*! \brief Structural visit hook for MapObj. */
TVMFFIAny VisitMap(StructuralVisitorObj* visitor, AnyView value) noexcept {
  const auto* map = value.cast<const MapObj*>();
  return VisitMapContainer(visitor, map);
}

/*! \brief Structural visit hook for DictObj. */
TVMFFIAny VisitDict(StructuralVisitorObj* visitor, AnyView value) noexcept {
  const auto* dict = value.cast<const DictObj*>();
  return VisitMapContainer(visitor, dict);
}

}  // namespace details

// ---------------------------------------------------------------------------
// Static registration.
// ---------------------------------------------------------------------------

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<VisitInterruptObj>()
      .def(refl::init<Any>(), "Constructor that creates a structural visit interrupt")
      .def_ro("value", &VisitInterruptObj::value);
  refl::ObjectDef<StructuralVisitorObj>().def(
      refl::init<>(), "Constructor that creates a default structural visitor");
  refl::GlobalDef()
      .def("ffi.VisitInterrupt", [](Any value) { return VisitInterrupt(std::move(value)); })
      .def("ffi.StructuralVisitor", []() { return StructuralVisitor(); })
      .def_method("ffi.StructuralVisitorVisit", &StructuralVisitorObj::Visit)
      .def_method("ffi.StructuralVisitorDefRegionKind", &StructuralVisitorObj::def_region_kind)
      .def_method(
          "ffi.StructuralVisitorWithDefRegionKind",
          [](const StructuralVisitor& visitor, TVMFFIDefRegionKind kind, const Function& callback) {
            return visitor->WithDefRegionKind(kind, callback);
          })
      .def("ffi.StructuralWalk",
           [](AnyView root, const Array<Tuple<int32_t, Function>>& callbacks,
              const Array<Tuple<int32_t, Function>>& callbacks_with_def_region_kind,
              int32_t order) -> Optional<VisitInterrupt> {
             return details::StructuralWalkExpected(root, callbacks, callbacks_with_def_region_kind,
                                                    order)
                 .value();
           });
  refl::EnsureTypeAttrColumn(refl::type_attr::kStructuralVisit);
  refl::TypeAttrDef<ArrayObj>().attr(
      refl::type_attr::kStructuralVisit,
      reinterpret_cast<void*>(static_cast<FStructuralVisit>(&details::VisitArray)));
  refl::TypeAttrDef<ListObj>().attr(
      refl::type_attr::kStructuralVisit,
      reinterpret_cast<void*>(static_cast<FStructuralVisit>(&details::VisitList)));
  refl::TypeAttrDef<MapObj>().attr(
      refl::type_attr::kStructuralVisit,
      reinterpret_cast<void*>(static_cast<FStructuralVisit>(&details::VisitMap)));
  refl::TypeAttrDef<DictObj>().attr(
      refl::type_attr::kStructuralVisit,
      reinterpret_cast<void*>(static_cast<FStructuralVisit>(&details::VisitDict)));
}

}  // namespace ffi
}  // namespace tvm
