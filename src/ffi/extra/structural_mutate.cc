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
  auto dispatch = [callbacks, callbacks_with_def_region_kind](
                      AnyView x, TVMFFIDefRegionKind kind) -> Expected<Any> {
    for (const auto& entry : callbacks) {
      int32_t type_index = entry.template get<0>();
      if (!RuntimeTypeIndexMatch(x.type_index(), type_index)) {
        continue;
      }
      Function fn = entry.template get<1>();
      return fn.CallExpected<Any>(x);
    }
    for (const auto& entry : callbacks_with_def_region_kind) {
      int32_t type_index = entry.template get<0>();
      if (!RuntimeTypeIndexMatch(x.type_index(), type_index)) {
        continue;
      }
      Function fn = entry.template get<1>();
      return fn.CallExpected<Any>(x, kind);
    }
    return Any(x);
  };

  if (order == static_cast<int>(WalkOrder::kPreOrder)) {
    using Mutator = StructuralMapMutatorObj<WalkOrder::kPreOrder, decltype(dispatch)>;
    StructuralMutator mutator(make_object<Mutator>(std::move(dispatch)));
    return mutator->MaybeInplaceMutateIfUniqueExpected(root);
  } else {
    using Mutator = StructuralMapMutatorObj<WalkOrder::kPostOrder, decltype(dispatch)>;
    StructuralMutator mutator(make_object<Mutator>(std::move(dispatch)));
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
 * \param source The sequence object stored in \p value.
 * \return The mutated sequence, or an Error.
 */
template <typename SeqObj>
Expected<Any> MutateSeqContainerExpected(StructuralMutatorObj* mutator, AnyView value,
                                         const SeqObj* source) noexcept {
  try {
    int64_t size = static_cast<int64_t>(source->size());
    ObjectPtr<SeqObj> output = nullptr;

    for (int64_t i = 0; i < size; ++i) {
      const Any& item = source->at(i);
      Expected<Any> mapped_item = mutator->MutateExpected(item);
      if (TVM_FFI_PREDICT_FALSE(mapped_item.is_err())) {
        return Unexpected(std::move(mapped_item).error());
      }
      const Any& mapped_value = details::ExpectedUnsafe::GetData(mapped_item);

      if (output == nullptr) {
        if (item.same_as(mapped_value)) {
          continue;
        }
        output = SeqObj::CreateRepeated(size, Any());
        for (int64_t j = 0; j < i; ++j) {
          output->SetItem(j, source->at(j));
        }
      }
      output->SetItem(i, mapped_value);
    }

    if (output == nullptr) {
      return Any(value);
    }
    return Any(ObjectRef(std::move(output)));
  } catch (const Error& err) {
    return Unexpected(err);
  }
}

/*!
 * \brief Structurally mutate the elements of a sequence container in place when safe.
 *
 * \tparam SeqObj The underlying sequence object type.
 * \param mutator The active structural mutator.
 * \param value The borrowed sequence container, which must be safe to mutate in place.
 * \param target The sequence object stored in \p value.
 * \return The mutated sequence, or an Error.
 */
template <typename SeqObj>
Expected<Any> MaybeInplaceMutateSeqContainerExpected(StructuralMutatorObj* mutator, AnyView value,
                                                     SeqObj* target) noexcept {
  try {
    for (int64_t i = 0; i < static_cast<int64_t>(target->size()); ++i) {
      const Any& item = target->at(i);
      Expected<Any> mapped_item = mutator->MaybeInplaceMutateIfUniqueExpected(item);
      if (TVM_FFI_PREDICT_FALSE(mapped_item.is_err())) {
        return Unexpected(std::move(mapped_item).error());
      }
      const Any& mapped_value = details::ExpectedUnsafe::GetData(mapped_item);

      if (!item.same_as(mapped_value)) {
        target->SetItem(i, mapped_value);
      }
    }
    return Any(value);
  } catch (const Error& err) {
    return Unexpected(err);
  }
}

/*!
 * \brief Structurally mutate the values of a map container.
 *
 * \tparam MapObjType The underlying map object type.
 * \param mutator The active structural mutator.
 * \param value The borrowed map container.
 * \param source The map object stored in \p value.
 * \return The mutated map, or an Error.
 */
template <typename MapObjType>
Expected<Any> MutateMapValuesExpected(StructuralMutatorObj* mutator, AnyView value,
                                      const MapObjType* source) noexcept {
  try {
    ObjectPtr<Object> output = nullptr;
    MapBaseObj::iterator output_it;
    size_t index = 0;

    for (auto source_it = source->begin(); source_it != source->end(); ++source_it, ++index) {
      const Any& old_value = source_it->second;
      Expected<Any> mapped_value = mutator->MutateExpected(old_value);
      if (TVM_FFI_PREDICT_FALSE(mapped_value.is_err())) {
        return Unexpected(std::move(mapped_value).error());
      }

      const Any& new_value = details::ExpectedUnsafe::GetData(mapped_value);
      bool changed = !old_value.same_as(new_value);
      if (output == nullptr) {
        if (!changed) {
          continue;
        }
        output = MapObjType::ShallowCopy(source);
        output_it = static_cast<MapBaseObj*>(output.get())->begin();
        for (size_t i = 0; i < index; ++i) {
          ++output_it;
        }
      }
      if (changed) {
        output_it->second = new_value;
      }
      ++output_it;
    }

    if (output == nullptr) {
      return Any(value);
    }
    return Any(ObjectRef(std::move(output)));
  } catch (const Error& err) {
    return Unexpected(err);
  }
}

/*!
 * \brief Structurally mutate the values of a map container in place when safe.
 *
 * \tparam MapObjType The underlying map object type.
 * \param mutator The active structural mutator.
 * \param value The borrowed map container, which must be safe to mutate in place.
 * \param target The map object stored in \p value.
 * \return The mutated map, or an Error.
 */
template <typename MapObjType>
Expected<Any> MaybeInplaceMutateMapValuesExpected(StructuralMutatorObj* mutator, AnyView value,
                                                  MapObjType* target) noexcept {
  try {
    for (auto it = target->begin(); it != target->end(); ++it) {
      const Any& old_value = it->second;
      Expected<Any> mapped_value = mutator->MaybeInplaceMutateIfUniqueExpected(old_value);
      if (TVM_FFI_PREDICT_FALSE(mapped_value.is_err())) {
        return Unexpected(std::move(mapped_value).error());
      }
      const Any& new_value = details::ExpectedUnsafe::GetData(mapped_value);

      if (!old_value.same_as(new_value)) {
        it->second = new_value;
      }
    }
    return Any(value);
  } catch (const Error& err) {
    return Unexpected(err);
  }
}

/*! \brief Identity structural mutation hook for immutable String and Bytes leaves. */
TVMFFIAny MutateImmutableLeaf(StructuralMutatorObj*, AnyView value) noexcept {
  Expected<Any> result = Any(value);
  return ExpectedUnsafe::MoveToTVMFFIAny(std::move(result));
}

/*! \brief Structural mutation hook for ArrayObj. */
TVMFFIAny MutateArray(StructuralMutatorObj* mutator, AnyView value) noexcept {
  Expected<Any> result = MutateSeqContainerExpected(mutator, value, value.cast<const ArrayObj*>());
  return ExpectedUnsafe::MoveToTVMFFIAny(std::move(result));
}

/*! \brief Maybe-in-place structural mutation hook for ArrayObj. */
TVMFFIAny MaybeInplaceMutateArray(StructuralMutatorObj* mutator, AnyView value) noexcept {
  Expected<Any> result = MaybeInplaceMutateSeqContainerExpected(
      mutator, value, const_cast<ArrayObj*>(value.cast<const ArrayObj*>()));
  return ExpectedUnsafe::MoveToTVMFFIAny(std::move(result));
}

/*! \brief Structural mutation hook for ListObj. */
TVMFFIAny MutateList(StructuralMutatorObj* mutator, AnyView value) noexcept {
  Expected<Any> result = MutateSeqContainerExpected(mutator, value, value.cast<const ListObj*>());
  return ExpectedUnsafe::MoveToTVMFFIAny(std::move(result));
}

/*! \brief Maybe-in-place structural mutation hook for ListObj. */
TVMFFIAny MaybeInplaceMutateList(StructuralMutatorObj* mutator, AnyView value) noexcept {
  Expected<Any> result = MaybeInplaceMutateSeqContainerExpected(
      mutator, value, const_cast<ListObj*>(value.cast<const ListObj*>()));
  return ExpectedUnsafe::MoveToTVMFFIAny(std::move(result));
}

/*! \brief Structural mutation hook for MapObj. */
TVMFFIAny MutateMap(StructuralMutatorObj* mutator, AnyView value) noexcept {
  Expected<Any> result = MutateMapValuesExpected(mutator, value, value.cast<const MapObj*>());
  return ExpectedUnsafe::MoveToTVMFFIAny(std::move(result));
}

/*! \brief Maybe-in-place structural mutation hook for MapObj. */
TVMFFIAny MaybeInplaceMutateMap(StructuralMutatorObj* mutator, AnyView value) noexcept {
  Expected<Any> result = MaybeInplaceMutateMapValuesExpected(
      mutator, value, const_cast<MapObj*>(value.cast<const MapObj*>()));
  return ExpectedUnsafe::MoveToTVMFFIAny(std::move(result));
}

/*! \brief Structural mutation hook for DictObj. */
TVMFFIAny MutateDict(StructuralMutatorObj* mutator, AnyView value) noexcept {
  Expected<Any> result = MutateMapValuesExpected(mutator, value, value.cast<const DictObj*>());
  return ExpectedUnsafe::MoveToTVMFFIAny(std::move(result));
}

/*! \brief Maybe-in-place structural mutation hook for DictObj. */
TVMFFIAny MaybeInplaceMutateDict(StructuralMutatorObj* mutator, AnyView value) noexcept {
  Expected<Any> result = MaybeInplaceMutateMapValuesExpected(
      mutator, value, const_cast<DictObj*>(value.cast<const DictObj*>()));
  return ExpectedUnsafe::MoveToTVMFFIAny(std::move(result));
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
