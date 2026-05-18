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
 * \file src/ffi/extra/structural_map.cc
 * \brief Structural map implementation.
 */
#include <tvm/ffi/extra/structural_map.h>
#include <tvm/ffi/reflection/accessor.h>

namespace tvm {
namespace ffi {

Any StructuralMapper::Map(AnyView value) {
  if (value.type_index() == TypeIndex::kTVMFFINone ||
      value.type_index() < TypeIndex::kTVMFFIStaticObjectBegin) {
    return Any(value);
  }
  return CallStructuralMap(value);
}

Optional<Any> StructuralMapper::LookupVarRemap(AnyView old_var) const {
  auto it = var_remap_.find(Any(old_var));
  if (it == var_remap_.end()) {
    return std::nullopt;
  }
  return (*it).second;
}

void StructuralMapper::SetVarRemap(Any old_var, Any new_var) {
  var_remap_.Set(std::move(old_var), std::move(new_var));
}

// objectref*, or object* (might be risky)
// return objectref
// 玩一玩 a version
Any StructuralMapper::MapOrInplaceMutator(ObjectRef&& obj) {
  TVM_FFI_ICHECK(obj.defined());

  if (obj.unique() && HasInplaceMutator(obj.type_index())) {
    return InplaceMutator(std::move(obj));
  }

  return Map(obj);
}

Any StructuralMapper::InplaceMutator(ObjectRef&& obj) {
  TVM_FFI_ICHECK(obj.defined());
  TVM_FFI_ICHECK(obj.unique());

  static reflection::TypeAttrColumn column(reflection::type_attr::kStructuralInplaceMutator);
  AnyView attr = column[obj.type_index()];
  if (attr.type_index() == TypeIndex::kTVMFFIOpaquePtr) {
    auto* fn = reinterpret_cast<FStructuralInplaceMutator>(attr.cast<void*>());
    TVM_FFI_ICHECK_NOTNULL(fn);
    return (*fn)(this, std::move(obj));
  }
  if (attr.type_index() != TypeIndex::kTVMFFINone) {
    TVM_FFI_THROW(TypeError) << reflection::type_attr::kStructuralInplaceMutator
                             << " must be an opaque function pointer";
  }
  return InplaceMutateReflectedFields(std::move(obj));
}

Any StructuralMapper::CallStructuralMap(AnyView value) {
  static reflection::TypeAttrColumn column(reflection::type_attr::kStructuralMap);
  AnyView attr = column[value.type_index()];
  if (attr.type_index() == TypeIndex::kTVMFFIOpaquePtr) {
    auto* fn = reinterpret_cast<FStructuralMap>(attr.cast<void*>());
    TVM_FFI_ICHECK_NOTNULL(fn);
    return (*fn)(this, value);
  }
  if (attr.type_index() != TypeIndex::kTVMFFINone) {
    TVM_FFI_THROW(TypeError) << reflection::type_attr::kStructuralMap
                             << " must be an opaque function pointer";
  }
  return MapReflectedFields(value);
}

Any StructuralMapper::MapReflectedFields(AnyView value) {
  return Any(value);
}

Any StructuralMapper::InplaceMutateReflectedFields(ObjectRef&& obj) {
  return Any(std::move(obj));
}

bool StructuralMapper::HasInplaceMutator(int32_t type_index) const {
  static reflection::TypeAttrColumn column(reflection::type_attr::kStructuralInplaceMutator);
  AnyView attr = column[type_index];
  if (attr.type_index() == TypeIndex::kTVMFFIOpaquePtr) {
    return true;
  }
  if (attr.type_index() != TypeIndex::kTVMFFINone) {
    TVM_FFI_THROW(TypeError) << reflection::type_attr::kStructuralInplaceMutator
                             << " must be an opaque function pointer";
  }
  return false;
}

}  // namespace ffi
}  // namespace tvm