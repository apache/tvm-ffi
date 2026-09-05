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
 * \file src/ffi/extra/structural_mutate.cc
 * \brief Structural mutator and structural map registration.
 */
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/dict.h>
#include <tvm/ffi/container/list.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/accessor.h>
#include <tvm/ffi/reflection/registry.h>

#include <utility>

namespace tvm {
namespace ffi {

namespace details {

/*!
 * \brief Runtime structural map for callback arrays.
 *
 * \param root The root value to map.
 * \param callbacks Runtime callback entries of ``(type_index, ffi::Function)`` invoked as
 *                  ``callback(value)``.
 * \param callbacks_with_def_region_kind Runtime callback entries of ``(type_index, ffi::Function)``
 *                                       invoked as ``callback(value, def_region_kind)``.
 * \param order Integer value of \ref WalkOrder.
 * \return The mapped owning value, or an Error.
 */
Expected<Any> StructuralMapExpected(
    AnyView root, const Array<Tuple<int32_t, Function>>& callbacks,
    const Array<Tuple<int32_t, Function>>& callbacks_with_def_region_kind, int order) noexcept {
  if (order == static_cast<int>(WalkOrder::kPreOrder)) {
    using Mutator = StructuralMapDynEngine<StructuralMapEngineBase, WalkOrder::kPreOrder>;
    StructuralMutator mutator(make_object<Mutator>(callbacks, callbacks_with_def_region_kind));
    return mutator->MaybeInplaceMutateIfUniqueExpected(root);
  } else {
    using Mutator = StructuralMapDynEngine<StructuralMapEngineBase, WalkOrder::kPostOrder>;
    StructuralMutator mutator(make_object<Mutator>(callbacks, callbacks_with_def_region_kind));
    return mutator->MaybeInplaceMutateIfUniqueExpected(root);
  }
}

// ---------------------------------------------------------------------------
// Built-in container structural mutation.
// ---------------------------------------------------------------------------

/*!
 * \brief Structurally mutate the elements of a sequence container.
 *
 * \tparam SeqObj The underlying sequence object type.
 * \param mutator The active structural mutator.
 * \param value The borrowed sequence container.
 * \param self The sequence object stored in \p value.
 * \return The mutated sequence, or an Error.
 */
template <typename SeqObj>
TVMFFIAny MutateSeqContainerRaw(StructuralMutatorObj* mutator, AnyView value,
                                const SeqObj* self) noexcept {
  int64_t size = static_cast<int64_t>(self->size());
  const Any* items = self->begin();
  ObjectPtr<SeqObj> output = nullptr;

  for (int64_t i = 0; i < size; ++i) {
    const Any& item = items[i];
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(Any, mapped_value, mutator->MutateExpected(item));

    if (output == nullptr) {
      if (item.same_as(mapped_value)) {
        continue;
      }
      output = SeqObj::CreateRepeated(size, Any());
      output->InitRange(0, items, items + i);
    }
    output->SetItemAfterCheck(i, std::move(mapped_value));
  }

  if (output == nullptr) {
    return AnyUnsafe::MoveAnyToTVMFFIAny(Any(value));
  }
  return AnyUnsafe::MoveAnyToTVMFFIAny(Any(std::move(output)));
}

/*!
 * \brief Structurally mutate the elements of a sequence container in place when safe.
 *
 * \tparam SeqObj The underlying sequence object type.
 * \param mutator The active structural mutator.
 * \param value The borrowed sequence container, which must be safe to mutate in place.
 * \param self The sequence object stored in \p value.
 * \return The mutated sequence, or an Error.
 */
template <typename SeqObj>
TVMFFIAny MaybeInplaceMutateSeqContainerRaw(StructuralMutatorObj* mutator, AnyView value,
                                            SeqObj* self) noexcept {
  for (int64_t i = 0; i < static_cast<int64_t>(self->size()); ++i) {
    const Any& item = self->begin()[i];
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(Any, mapped_value,
                                      mutator->MaybeInplaceMutateIfUniqueExpected(item));

    if (!item.same_as(mapped_value)) {
      self->SetItemAfterCheck(i, std::move(mapped_value));
    }
  }
  return AnyUnsafe::MoveAnyToTVMFFIAny(Any(value));
}

/*!
 * \brief Structurally mutate the values of a map container.
 *
 * \tparam MapObjType The underlying map object type.
 * \param mutator The active structural mutator.
 * \param value The borrowed map container.
 * \param self The map object stored in \p value.
 * \return The mutated map, or an Error.
 */
template <typename MapObjType>
TVMFFIAny MutateMapValuesRaw(StructuralMutatorObj* mutator, AnyView value,
                             const MapObjType* self) noexcept {
  ObjectPtr<Object> output = nullptr;
  MapBaseObj::iterator output_it;
  size_t index = 0;

  for (auto source_it = self->begin(); source_it != self->end(); ++source_it, ++index) {
    const Any& old_value = source_it->second;
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(Any, new_value, mutator->MutateExpected(old_value));
    bool changed = !old_value.same_as(new_value);
    if (output == nullptr) {
      if (!changed) {
        continue;
      }
      output = MapObjType::ShallowCopy(self);
      output_it = static_cast<MapBaseObj*>(output.get())->begin();
      for (size_t i = 0; i < index; ++i) {
        ++output_it;
      }
    }
    if (changed) {
      output_it->second = std::move(new_value);
    }
    ++output_it;
  }

  if (output == nullptr) {
    return AnyUnsafe::MoveAnyToTVMFFIAny(Any(value));
  }
  return AnyUnsafe::MoveAnyToTVMFFIAny(Any(std::move(output)));
}

/*!
 * \brief Structurally mutate the values of a map container in place when safe.
 *
 * \tparam MapObjType The underlying map object type.
 * \param mutator The active structural mutator.
 * \param value The borrowed map container, which must be safe to mutate in place.
 * \param self The map object stored in \p value.
 * \return The mutated map, or an Error.
 */
template <typename MapObjType>
TVMFFIAny MaybeInplaceMutateMapValuesRaw(StructuralMutatorObj* mutator, AnyView value,
                                         MapObjType* self) noexcept {
  for (auto it = self->begin(); it != self->end(); ++it) {
    const Any& old_value = it->second;
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(Any, new_value,
                                      mutator->MaybeInplaceMutateIfUniqueExpected(old_value));

    if (!old_value.same_as(new_value)) {
      it->second = std::move(new_value);
    }
  }
  return AnyUnsafe::MoveAnyToTVMFFIAny(Any(value));
}

/*! \brief Identity structural mutation hook for immutable String and Bytes leaves. */
TVMFFIAny MutateImmutableLeaf(StructuralMutatorObj*, AnyView value) noexcept {
  Expected<Any> result = Any(value);
  return ExpectedUnsafe::MoveToTVMFFIAny(std::move(result));
}

/*! \brief Structural mutation hook for ArrayObj. */
TVMFFIAny MutateArray(StructuralMutatorObj* mutator, AnyView value) noexcept {
  return MutateSeqContainerRaw(
      mutator, value, details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const ArrayObj>(value));
}

/*! \brief Maybe-in-place structural mutation hook for ArrayObj. */
TVMFFIAny MaybeInplaceMutateArray(StructuralMutatorObj* mutator, AnyView value) noexcept {
  return MaybeInplaceMutateSeqContainerRaw(
      mutator, value, details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<ArrayObj>(value));
}

/*! \brief Structural mutation hook for ListObj. */
TVMFFIAny MutateList(StructuralMutatorObj* mutator, AnyView value) noexcept {
  return MutateSeqContainerRaw(
      mutator, value, details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const ListObj>(value));
}

/*! \brief Maybe-in-place structural mutation hook for ListObj. */
TVMFFIAny MaybeInplaceMutateList(StructuralMutatorObj* mutator, AnyView value) noexcept {
  return MaybeInplaceMutateSeqContainerRaw(
      mutator, value, details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<ListObj>(value));
}

/*! \brief Structural mutation hook for MapObj. */
TVMFFIAny MutateMap(StructuralMutatorObj* mutator, AnyView value) noexcept {
  return MutateMapValuesRaw(
      mutator, value, details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const MapObj>(value));
}

/*! \brief Maybe-in-place structural mutation hook for MapObj. */
TVMFFIAny MaybeInplaceMutateMap(StructuralMutatorObj* mutator, AnyView value) noexcept {
  return MaybeInplaceMutateMapValuesRaw(
      mutator, value, details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<MapObj>(value));
}

/*! \brief Structural mutation hook for DictObj. */
TVMFFIAny MutateDict(StructuralMutatorObj* mutator, AnyView value) noexcept {
  return MutateMapValuesRaw(
      mutator, value, details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const DictObj>(value));
}

/*! \brief Maybe-in-place structural mutation hook for DictObj. */
TVMFFIAny MaybeInplaceMutateDict(StructuralMutatorObj* mutator, AnyView value) noexcept {
  return MaybeInplaceMutateMapValuesRaw(
      mutator, value, details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<DictObj>(value));
}
}  // namespace details

// ---------------------------------------------------------------------------
// Static registration.
// ---------------------------------------------------------------------------

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<StructuralMutatorObj>();  // NOLINT(bugprone-unused-raii)
  refl::GlobalDef()
      .def_method("ffi.StructuralMutatorMaybeInplaceMutate",
                  &StructuralMutatorObj::MaybeInplaceMutate)
      .def_method("ffi.StructuralMutatorMutate", &StructuralMutatorObj::Mutate)
      .def_method("ffi.StructuralMutatorVarRemapGet",
                  [](const StructuralMutator& mutator, AnyView var) {
                    return mutator->VarRemapGetExpected(var).value();
                  })
      .def_method("ffi.StructuralMutatorVarRemapSet",
                  [](const StructuralMutator& mutator, AnyView var, AnyView mapped_value) {
                    mutator->VarRemapSetExpected(var, mapped_value).value();
                  })
      .def_method("ffi.StructuralMutatorDefRegionKind", &StructuralMutatorObj::def_region_kind)
      .def_method(
          "ffi.StructuralMutatorWithDefRegionKind",
          [](const StructuralMutator& mutator, TVMFFIDefRegionKind kind, const Function& callback) {
            return mutator->WithDefRegionKind(kind, callback);
          })
      .def("ffi.StructuralMap",
           [](AnyView root, const Array<Tuple<int32_t, Function>>& callbacks,
              const Array<Tuple<int32_t, Function>>& callbacks_with_def_region_kind,
              int32_t order) -> Any {
             return details::StructuralMapExpected(root, callbacks, callbacks_with_def_region_kind,
                                                   order)
                 .value();
           });
  refl::EnsureTypeAttrColumn(refl::type_attr::kStructuralMutate);
  refl::EnsureTypeAttrColumn(refl::type_attr::kStructuralMaybeInplaceMutate);
  refl::TypeAttrDef<details::StringObj>()
      .attr(refl::type_attr::kStructuralMutate,
            reinterpret_cast<void*>(static_cast<FStructuralMutate>(&details::MutateImmutableLeaf)))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(static_cast<FStructuralMutate>(&details::MutateImmutableLeaf)));
  refl::TypeAttrDef<details::BytesObj>()
      .attr(refl::type_attr::kStructuralMutate,
            reinterpret_cast<void*>(static_cast<FStructuralMutate>(&details::MutateImmutableLeaf)))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(static_cast<FStructuralMutate>(&details::MutateImmutableLeaf)));
  refl::TypeAttrDef<ArrayObj>()
      .attr(refl::type_attr::kStructuralMutate,
            reinterpret_cast<void*>(static_cast<FStructuralMutate>(&details::MutateArray)))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(
                static_cast<FStructuralMutate>(&details::MaybeInplaceMutateArray)));
  refl::TypeAttrDef<ListObj>()
      .attr(refl::type_attr::kStructuralMutate,
            reinterpret_cast<void*>(static_cast<FStructuralMutate>(&details::MutateList)))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(
                static_cast<FStructuralMutate>(&details::MaybeInplaceMutateList)));
  refl::TypeAttrDef<MapObj>()
      .attr(refl::type_attr::kStructuralMutate,
            reinterpret_cast<void*>(static_cast<FStructuralMutate>(&details::MutateMap)))
      .attr(
          refl::type_attr::kStructuralMaybeInplaceMutate,
          reinterpret_cast<void*>(static_cast<FStructuralMutate>(&details::MaybeInplaceMutateMap)));
  refl::TypeAttrDef<DictObj>()
      .attr(refl::type_attr::kStructuralMutate,
            reinterpret_cast<void*>(static_cast<FStructuralMutate>(&details::MutateDict)))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(
                static_cast<FStructuralMutate>(&details::MaybeInplaceMutateDict)));
}

}  // namespace ffi
}  // namespace tvm
