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
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/accessor.h>

namespace tvm {
namespace ffi {
namespace {

// Walk reflected fields of `value` and recurse into each non-ignored field.
Optional<VisitInterrupt> VisitReflectedFields(StructuralVisitor* visitor, const ObjectRef& value) {
  const TVMFFITypeInfo* type_info = TVMFFIGetTypeInfo(value.type_index());

  Optional<VisitInterrupt> result = std::nullopt;
  reflection::ForEachFieldInfoWithEarlyStop(type_info, [&](const TVMFFIFieldInfo* field_info) {
    if (field_info->flags & kTVMFFIFieldFlagBitMaskSEqHashIgnore) {
      return false;
    }

    reflection::FieldGetter getter(field_info);
    Any field_value = getter(value);

    TVMFFIDefRegionKind kind = kTVMFFIDefRegionKindNone;
    if (field_info->flags & kTVMFFIFieldFlagBitMaskSEqHashDefNonRecursive) {
      kind = kTVMFFIDefRegionKindNonRecursive;
    } else if (field_info->flags & kTVMFFIFieldFlagBitMaskSEqHashDefRecursive) {
      kind = kTVMFFIDefRegionKindRecursive;
    }

    result = visitor->WithDefRegionKind(
        kind, [&]() { return visitor->Visit(field_value.cast<ObjectRef>()); });
    return result.has_value();
  });
  return result;
}

// Thunks installed by the default StructuralVisitor constructor.
//
// They are file-local so taking their address (in the constructor) is the only
// way to obtain a pointer to them, which keeps the C-ABI surface narrow.
Optional<VisitInterrupt> DefaultCppVisit(void* self, const ObjectRef& value) {
  return static_cast<StructuralVisitor*>(self)->DefaultVisit(value);
}

int DefaultSafeVisit(void* self, const ObjectRef& value, Optional<VisitInterrupt>* out) {
  TVM_FFI_SAFE_CALL_BEGIN();
  *out = DefaultCppVisit(self, value);
  TVM_FFI_SAFE_CALL_END();
}

}  // namespace

StructuralVisitor::StructuralVisitor() {
  this->self = this;
  this->cpp_visit = reinterpret_cast<void*>(&DefaultCppVisit);
  this->safe_visit = &DefaultSafeVisit;
  this->def_region_mode = kTVMFFIDefRegionKindNone;
}

Optional<VisitInterrupt> StructuralVisitor::DefaultVisit(const ObjectRef& value) {
  static reflection::TypeAttrColumn column(reflection::type_attr::kStructuralVisit);
  AnyView attr = column[value.type_index()];

  // Type-specific override registered as an opaque safe-call function pointer.
  if (attr.type_index() == TypeIndex::kTVMFFIOpaquePtr) {
    auto* fn = reinterpret_cast<FStructuralVisitSafe>(attr.cast<void*>());
    TVM_FFI_ICHECK_NOTNULL(fn);
    Optional<VisitInterrupt> out;
    TVM_FFI_CHECK_SAFE_CALL(fn(this, value, &out));
    return out;
  }

  // Type-specific override registered as an ffi::Function.
  if (attr.type_index() == TypeIndex::kTVMFFIFunction) {
    Function visit_child = Function::FromTyped(
        [this](const ObjectRef& child, int def_region_kind) -> Optional<VisitInterrupt> {
          return WithDefRegionKind(static_cast<TVMFFIDefRegionKind>(def_region_kind),
                                   [&]() { return Visit(child); });
        });
    return attr.cast<Function>()(value, visit_child).cast<Optional<VisitInterrupt>>();
  }

  if (attr.type_index() != TypeIndex::kTVMFFINone) {
    TVM_FFI_THROW(TypeError) << reflection::type_attr::kStructuralVisit
                             << " must be an opaque function pointer or ffi.Function";
  }

  return VisitReflectedFields(this, value);
}

}  // namespace ffi
}  // namespace tvm
