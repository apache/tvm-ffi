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
 * \file tvm/ffi/extra/structural_map.h
 * \brief Structural mapping and rewriting utilities.
 */
#ifndef TVM_FFI_EXTRA_STRUCTURAL_MAP_H_
#define TVM_FFI_EXTRA_STRUCTURAL_MAP_H_

#include <tvm/ffi/container/dict.h>
#include <tvm/ffi/extra/base.h>
#include <tvm/ffi/extra/structural_visit.h>

#include <type_traits>
#include <utility>

namespace tvm {
namespace ffi {

/*!
 * \brief Base class for structural mappers.
 *
 * ``Map`` is the normal borrowed-input entry point and returns an owned value.
 * Subclasses may override it to rewrite selected values and delegate to
 * ``StructuralMapper::Map`` for recursive mapping.
 */
class StructuralMapper {
 public:
  using FStructuralMap = Any (*)(StructuralMapper* mapper, AnyView value);
  using FStructuralInplaceMutator = Any (*)(StructuralMapper* mapper, ObjectRef&& obj);

  virtual ~StructuralMapper() = default;

  virtual Any Map(AnyView value);

  TVMFFIDefRegionKind def_region_kind() const { return def_region_mode_; }

  template <typename Callback>
  auto WithDefRegionKind(TVMFFIDefRegionKind kind, Callback&& callback)
      -> decltype(std::forward<Callback>(callback)()) {
    class DefRegionScope {
     public:
      DefRegionScope(StructuralMapper* mapper, TVMFFIDefRegionKind kind)
          : mapper_(mapper), old_kind_(mapper->def_region_mode_) {
        mapper_->def_region_mode_ = kind;
      }
      ~DefRegionScope() { mapper_->def_region_mode_ = old_kind_; }

      DefRegionScope(const DefRegionScope&) = delete;
      DefRegionScope& operator=(const DefRegionScope&) = delete;

     private:
      StructuralMapper* mapper_;
      TVMFFIDefRegionKind old_kind_;
    };

    DefRegionScope scope(this, kind);
    return std::forward<Callback>(callback)();
  }

  Optional<Any> LookupVarRemap(AnyView old_var) const;
  void SetVarRemap(Any old_var, Any new_var);

  Any MapOrInplaceMutator(ObjectRef&& obj);
  Any InplaceMutator(ObjectRef&& obj);

  Any CallStructuralMap(AnyView value);

 protected:
  Any MapReflectedFields(AnyView value);
  Any InplaceMutateReflectedFields(ObjectRef&& obj);
  bool HasInplaceMutator(int32_t type_index) const;

 private:
  TVMFFIDefRegionKind def_region_mode_ = kTVMFFIDefRegionKindNone;
  Dict<Any, Any> var_remap_;
};


/// called: map
// callback: objectref -> objectref
// most common: post order
template <typename... T, typename F>
Any StructuralRewrite(AnyView root, F&& callback, WalkOrder order = WalkOrder::kPostOrder) {
  static_assert(sizeof...(T) > 0, "StructuralRewrite requires at least one match type");

  using Callback = std::decay_t<F>;

  class Rewriter : public StructuralMapper {
   public:
    Rewriter(Callback callback, WalkOrder order)
        : callback_(std::move(callback)), order_(order) {}

    Any Map(AnyView value) override {
      if (order_ == WalkOrder::kPreOrder) {
        if (Optional<Any> rewritten = TryRewrite(value)) {
          return *std::move(rewritten);
        }
      }

      Any mapped = StructuralMapper::Map(value);

      if (order_ == WalkOrder::kPostOrder) {
        if (Optional<Any> rewritten = TryRewrite(mapped)) {
          return *std::move(rewritten);
        }
      }

      return mapped;
    }

   private:
    Optional<Any> TryRewrite(AnyView value) {
      if (auto matched = MatchCallbackArg<T...>(value)) {
        return callback_(*matched, this->def_region_kind());
      }
      return std::nullopt;
    }

    Callback callback_;
    WalkOrder order_;
  };

  Rewriter rewriter(std::forward<F>(callback), order);
  return rewriter.Map(root);
}

}  // namespace ffi
}  // namespace tvm

#endif  // TVM_FFI_EXTRA_STRUCTURAL_MAP_H_
