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

namespace detail {

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
    using Visitor =
        StructuralWalkEngine<StructuralVisitorObj, WalkOrder::kPreOrder, decltype(dispatch)>;
    StructuralVisitor visitor(make_object<Visitor>(std::move(dispatch)));
    auto result = visitor->VisitExpected(root);
    if (TVM_FFI_PREDICT_FALSE(result.is_err())) {
      if (auto object_root = root.as<ObjectRef>()) {
        return Unexpected(
            VisitErrorContext::EnrichVisitError(std::move(result).error(), *object_root));
      }
      return Unexpected(std::move(result).error());
    }
    return result;
  } else {
    using Visitor =
        StructuralWalkEngine<StructuralVisitorObj, WalkOrder::kPostOrder, decltype(dispatch)>;
    StructuralVisitor visitor(make_object<Visitor>(std::move(dispatch)));
    auto result = visitor->VisitExpected(root);
    if (TVM_FFI_PREDICT_FALSE(result.is_err())) {
      if (auto object_root = root.as<ObjectRef>()) {
        return Unexpected(
            VisitErrorContext::EnrichVisitError(std::move(result).error(), *object_root));
      }
      return Unexpected(std::move(result).error());
    }
    return result;
  }
}

/*!
 * \brief Runtime callback-driven structural visit.
 * \param root The root value to visit.
 * \param callbacks Runtime callback entries of ``(type_index, ffi::Function)`` invoked as
 *                  ``callback(value, visitor)``.
 * \return Expected interrupt state. An error means traversal failed.
 */
Expected<Optional<VisitInterrupt>> StructuralVisitExpected(
    AnyView root, const Array<Tuple<int32_t, Function>>& callbacks) noexcept {
  auto dispatch = [callbacks](AnyView value,
                              StructuralVisitorObj* visitor) -> Expected<Optional<VisitInterrupt>> {
    for (const auto& entry : callbacks) {
      if (!RuntimeTypeIndexMatch(value.type_index(), entry.template get<0>())) continue;
      return entry.template get<1>().CallExpected<Optional<VisitInterrupt>>(
          value, GetRef<StructuralVisitor>(visitor));
    }
    return visitor->DefaultVisitExpected(value);
  };

  using Visitor = StructuralVisitEngine<StructuralVisitorObj, decltype(dispatch)>;
  StructuralVisitor visitor(make_object<Visitor>(std::move(dispatch)));
  auto result = visitor->VisitExpected(root);
  if (TVM_FFI_PREDICT_FALSE(result.is_err())) {
    if (auto object_root = root.as<ObjectRef>()) {
      return Unexpected(
          VisitErrorContext::EnrichVisitError(std::move(result).error(), *object_root));
    }
    return Unexpected(std::move(result).error());
  }
  return result;
}

/*! \brief Visit entries in a sequence container. */
template <typename SeqObj>
TVM_FFI_INLINE Expected<Optional<VisitInterrupt>> VisitSeqContainer(StructuralVisitorObj* visitor,
                                                                    AnyView value) noexcept {
  const auto* self = value.cast<const SeqObj*>();
  for (const Any& item : *self) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(item));
  }
  return Optional<VisitInterrupt>(std::nullopt);
}

/*! \brief Visit values in a map container while treating keys as structural anchors. */
template <typename MapObjType>
TVM_FFI_INLINE Expected<Optional<VisitInterrupt>> VisitMapContainer(StructuralVisitorObj* visitor,
                                                                    AnyView value) noexcept {
  const auto* self = value.cast<const MapObjType*>();
  for (const auto& kv : *self) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(kv.second));
  }
  return Optional<VisitInterrupt>(std::nullopt);
}

}  // namespace detail

// ---------------------------------------------------------------------------
// Static registration.
// ---------------------------------------------------------------------------

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<VisitInterruptObj>().def_ro("value", &VisitInterruptObj::value,
                                              refl::default_value(nullptr));
  refl::ObjectDef<StructuralVisitorObj>();  // NOLINT(bugprone-unused-raii)
  refl::GlobalDef()
      .def("ffi.VisitInterrupt", [](Any value) { return VisitInterrupt(std::move(value)); })
      .def_method("ffi.StructuralVisitorVisit", &StructuralVisitorObj::Visit)
      .def_method("ffi.StructuralVisitorDefaultVisit",
                  [](const StructuralVisitor& visitor, AnyView value) {
                    return visitor->DefaultVisitExpected(value).value();
                  })
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
             return detail::StructuralWalkExpected(root, callbacks, callbacks_with_def_region_kind,
                                                   order)
                 .value();
           })
      .def("ffi.StructuralVisit",
           [](AnyView root,
              const Array<Tuple<int32_t, Function>>& callbacks) -> Optional<VisitInterrupt> {
             return detail::StructuralVisitExpected(root, callbacks).value();
           });
  refl::EnsureTypeAttrColumn(type_attr::kStructuralVisit);
  refl::TypeAttrDef<ArrayObj>().native_fn_attr<&detail::VisitSeqContainer<ArrayObj>>(
      type_attr::kStructuralVisit);
  refl::TypeAttrDef<ListObj>().native_fn_attr<&detail::VisitSeqContainer<ListObj>>(
      type_attr::kStructuralVisit);
  refl::TypeAttrDef<MapObj>().native_fn_attr<&detail::VisitMapContainer<MapObj>>(
      type_attr::kStructuralVisit);
  refl::TypeAttrDef<DictObj>().native_fn_attr<&detail::VisitMapContainer<DictObj>>(
      type_attr::kStructuralVisit);
}

}  // namespace ffi
}  // namespace tvm
