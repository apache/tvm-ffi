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
#include <vector>

namespace tvm {
namespace ffi {

namespace details {

/*!
 * \brief Runtime callback dispatcher for structural mapping.
 */
class StructuralMapRuntimeDispatch {
 public:
  /*!
   * \brief Construct a dispatcher from runtime callback arrays.
   * \param callbacks Callback entries invoked as ``callback(value)``.
   * \param callbacks_with_def_region_kind Callback entries invoked as
   *        ``callback(value, def_region_kind)``.
   */
  StructuralMapRuntimeDispatch(Array<Tuple<int32_t, Function>> callbacks,
                               Array<Tuple<int32_t, Function>> callbacks_with_def_region_kind)
      : callbacks_(std::move(callbacks)),
        callbacks_with_def_region_kind_(std::move(callbacks_with_def_region_kind)) {}

  /*!
   * \brief Return whether a runtime callback matches \p value.
   * \param value The value to match.
   * \return Whether a callback matches, or an Error.
   */
  Expected<bool> HasCallbackExpected(AnyView value) const noexcept {
    for (const auto& entry : callbacks_) {
      if (RuntimeTypeIndexMatch(value.type_index(), entry.template get<0>())) {
        return true;
      }
    }
    for (const auto& entry : callbacks_with_def_region_kind_) {
      if (RuntimeTypeIndexMatch(value.type_index(), entry.template get<0>())) {
        return true;
      }
    }
    return false;
  }

  /*!
   * \brief Invoke the first runtime callback matching \p value.
   * \param value The value passed to the callback.
   * \param kind The active def-region kind.
   * \return The callback result, the original value when no callback matches, or an Error.
   */
  Expected<Any> InvokeCallbackExpected(AnyView value, TVMFFIDefRegionKind kind) noexcept {
    for (const auto& entry : callbacks_) {
      if (RuntimeTypeIndexMatch(value.type_index(), entry.template get<0>())) {
        Function fn = entry.template get<1>();
        return fn.CallExpected<Any>(value);
      }
    }
    for (const auto& entry : callbacks_with_def_region_kind_) {
      if (RuntimeTypeIndexMatch(value.type_index(), entry.template get<0>())) {
        Function fn = entry.template get<1>();
        return fn.CallExpected<Any>(value, kind);
      }
    }
    return Any(value);
  }

 private:
  /*! \brief Runtime callbacks invoked with the mapped value. */
  Array<Tuple<int32_t, Function>> callbacks_;
  /*! \brief Runtime callbacks invoked with the mapped value and active def-region kind. */
  Array<Tuple<int32_t, Function>> callbacks_with_def_region_kind_;
};

/*!
 * \brief Runtime structural map for callback arrays.
 *
 * \param root The root value to transform.
 * \param callbacks Runtime callback entries invoked as ``callback(value)``.
 * \param callbacks_with_def_region_kind Runtime callback entries invoked as
 *        ``callback(value, def_region_kind)``.
 * \param order Integer value of \ref WalkOrder.
 * \return The transformed owning value, or an Error.
 */
Expected<Any> StructuralMapExpected(
    AnyView root, const Array<Tuple<int32_t, Function>>& callbacks,
    const Array<Tuple<int32_t, Function>>& callbacks_with_def_region_kind, int order) noexcept {
  StructuralMapRuntimeDispatch dispatch(callbacks, callbacks_with_def_region_kind);

  if (order == static_cast<int>(WalkOrder::kPreOrder)) {
    using Mutator = StructuralMapMutatorObj<WalkOrder::kPreOrder, StructuralMapRuntimeDispatch>;
    StructuralMutator mutator(make_object<Mutator>(std::move(dispatch)));
    return mutator->MaybeInplaceMutateExpected(root);
  } else {
    using Mutator = StructuralMapMutatorObj<WalkOrder::kPostOrder, StructuralMapRuntimeDispatch>;
    StructuralMutator mutator(make_object<Mutator>(std::move(dispatch)));
    return mutator->MaybeInplaceMutateExpected(root);
  }
}

// ---------------------------------------------------------------------------
// Built-in container structural mutation.
// ---------------------------------------------------------------------------

/*!
 * \brief Structurally mutate the elements of a sequence container.
 *
 * \tparam Container The owning container type to construct when copying is required.
 * \param mutator The active structural mutator.
 * \param value The borrowed sequence container.
 * \param seq The sequence object stored in \p value.
 * \param maybe_inplace Whether mutation may reuse a uniquely owned sequence.
 * \return The transformed sequence, or an Error.
 */
template <typename Container>
Expected<Any> MutateSeqContainerExpected(StructuralMutatorObj* mutator, AnyView value,
                                         const SeqBaseObj* seq, bool maybe_inplace) noexcept {
  try {
    bool mutate_inplace = maybe_inplace && seq->unique();
    std::vector<Any> mapped_items;
    if (!mutate_inplace) {
      mapped_items.reserve(seq->size());
    }

    bool changed = false;
    for (size_t i = 0; i < seq->size(); ++i) {
      const Any& item = seq->at(static_cast<int64_t>(i));
      Expected<Any> mapped_item = mutate_inplace ? mutator->MaybeInplaceMutateExpected(item)
                                                 : mutator->MutateExpected(item);
      if (TVM_FFI_PREDICT_FALSE(mapped_item.is_err())) {
        return Unexpected(std::move(mapped_item).error());
      }
      const Any& mapped_value = details::ExpectedUnsafe::GetData(mapped_item);
      if (!item.same_as(mapped_value)) {
        changed = true;
        if (mutate_inplace) {
          const_cast<SeqBaseObj*>(seq)->SetItem(static_cast<int64_t>(i), mapped_value);
        }
      }
      if (!mutate_inplace) {
        mapped_items.emplace_back(mapped_value);
      }
    }

    if (mutate_inplace || !changed) {
      return Any(value);
    }
    return Any(Container(mapped_items.begin(), mapped_items.end()));
  } catch (const Error& err) {
    return Unexpected(err);
  }
}

/*!
 * \brief Structurally mutate the keys and values of a map container.
 *
 * \tparam Container The owning container type to construct when an entry changes.
 * \param mutator The active structural mutator.
 * \param value The borrowed map container.
 * \param map The map object stored in \p value.
 * \param maybe_inplace Whether mutation may reuse uniquely owned values.
 * \return The transformed map, or an Error.
 */
template <typename Container>
Expected<Any> MutateMapContainerExpected(StructuralMutatorObj* mutator, AnyView value,
                                         const MapBaseObj* map, bool maybe_inplace) noexcept {
  try {
    bool can_mutate_values_inplace = maybe_inplace && map->unique();
    std::vector<std::pair<Any, Any>> mapped_entries;
    mapped_entries.reserve(map->size());

    bool keys_changed = false;
    for (const auto& entry : *map) {
      Expected<Any> mapped_key = mutator->MutateExpected(entry.first);
      if (TVM_FFI_PREDICT_FALSE(mapped_key.is_err())) {
        return Unexpected(std::move(mapped_key).error());
      }

      Expected<Any> mapped_value = can_mutate_values_inplace
                                       ? mutator->MaybeInplaceMutateExpected(entry.second)
                                       : mutator->MutateExpected(entry.second);
      if (TVM_FFI_PREDICT_FALSE(mapped_value.is_err())) {
        return Unexpected(std::move(mapped_value).error());
      }

      const Any& new_key = details::ExpectedUnsafe::GetData(mapped_key);
      const Any& new_value = details::ExpectedUnsafe::GetData(mapped_value);
      keys_changed = keys_changed || !entry.first.same_as(new_key);
      mapped_entries.emplace_back(new_key, new_value);
    }

    if (!can_mutate_values_inplace || keys_changed) {
      return Any(Container(mapped_entries.begin(), mapped_entries.end()));
    }

    auto mapped_entry = mapped_entries.begin();
    for (auto& entry : *const_cast<MapBaseObj*>(map)) {
      if (!entry.second.same_as(mapped_entry->second)) {
        entry.second = std::move(mapped_entry->second);
      }
      ++mapped_entry;
    }
    return Any(value);
  } catch (const Error& err) {
    return Unexpected(err);
  }
}

/*! \brief Structural mutation hook for ArrayObj. */
TVMFFIAny MutateArray(StructuralMutatorObj* mutator, AnyView value) noexcept {
  Expected<Any> result =
      MutateSeqContainerExpected<Array<Any>>(mutator, value, value.cast<const ArrayObj*>(), false);
  return ExpectedUnsafe::MoveToTVMFFIAny(std::move(result));
}

/*! \brief Maybe-in-place structural mutation hook for ArrayObj. */
TVMFFIAny MaybeInplaceMutateArray(StructuralMutatorObj* mutator, AnyView value) noexcept {
  Expected<Any> result =
      MutateSeqContainerExpected<Array<Any>>(mutator, value, value.cast<const ArrayObj*>(), true);
  return ExpectedUnsafe::MoveToTVMFFIAny(std::move(result));
}

/*! \brief Structural mutation hook for ListObj. */
TVMFFIAny MutateList(StructuralMutatorObj* mutator, AnyView value) noexcept {
  Expected<Any> result =
      MutateSeqContainerExpected<List<Any>>(mutator, value, value.cast<const ListObj*>(), false);
  return ExpectedUnsafe::MoveToTVMFFIAny(std::move(result));
}

/*! \brief Maybe-in-place structural mutation hook for ListObj. */
TVMFFIAny MaybeInplaceMutateList(StructuralMutatorObj* mutator, AnyView value) noexcept {
  Expected<Any> result =
      MutateSeqContainerExpected<List<Any>>(mutator, value, value.cast<const ListObj*>(), true);
  return ExpectedUnsafe::MoveToTVMFFIAny(std::move(result));
}

/*! \brief Structural mutation hook for MapObj. */
TVMFFIAny MutateMap(StructuralMutatorObj* mutator, AnyView value) noexcept {
  Expected<Any> result =
      MutateMapContainerExpected<Map<Any, Any>>(mutator, value, value.cast<const MapObj*>(), false);
  return ExpectedUnsafe::MoveToTVMFFIAny(std::move(result));
}

/*! \brief Maybe-in-place structural mutation hook for MapObj. */
TVMFFIAny MaybeInplaceMutateMap(StructuralMutatorObj* mutator, AnyView value) noexcept {
  Expected<Any> result =
      MutateMapContainerExpected<Map<Any, Any>>(mutator, value, value.cast<const MapObj*>(), true);
  return ExpectedUnsafe::MoveToTVMFFIAny(std::move(result));
}

/*! \brief Structural mutation hook for DictObj. */
TVMFFIAny MutateDict(StructuralMutatorObj* mutator, AnyView value) noexcept {
  Expected<Any> result = MutateMapContainerExpected<Dict<Any, Any>>(
      mutator, value, value.cast<const DictObj*>(), false);
  return ExpectedUnsafe::MoveToTVMFFIAny(std::move(result));
}

/*! \brief Maybe-in-place structural mutation hook for DictObj. */
TVMFFIAny MaybeInplaceMutateDict(StructuralMutatorObj* mutator, AnyView value) noexcept {
  Expected<Any> result = MutateMapContainerExpected<Dict<Any, Any>>(
      mutator, value, value.cast<const DictObj*>(), true);
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
