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
 * \file src/ffi/extra/reflection_extra.cc
 *
 * \brief Extra reflection registrations. *
 */
#include <tvm/ffi/extra/dataclass.h>
#include <tvm/ffi/extra/structural_key.h>
#include <tvm/ffi/reflection/access_path.h>
#include <tvm/ffi/reflection/accessor.h>
#include <tvm/ffi/reflection/native_function.h>
#include <tvm/ffi/reflection/registry.h>

#include <sstream>

namespace tvm {
namespace ffi {
namespace reflection {

void MakeObjectFromPackedArgs(ffi::PackedArgs args, Any* ret) {
  int32_t type_index;
  if (auto opt_type_index = args[0].try_cast<int32_t>()) {
    type_index = *opt_type_index;
  } else {
    String type_key = args[0].cast<String>();
    TVMFFIByteArray type_key_array = TVMFFIByteArray{type_key.data(), type_key.size()};
    TVM_FFI_CHECK_SAFE_CALL(TVMFFITypeKeyToIndex(&type_key_array, &type_index));
  }

  TVM_FFI_ICHECK(args.size() % 2 == 1);
  const TVMFFITypeInfo* type_info = TVMFFIGetTypeInfo(type_index);
  ObjectPtr<Object> ptr = CreateEmptyObject(type_info);

  std::vector<String> keys;
  std::vector<bool> keys_found;

  for (int i = 1; i < args.size(); i += 2) {
    keys.push_back(args[i].cast<String>());
  }
  keys_found.resize(keys.size(), false);

  auto search_field = [&](const TVMFFIByteArray& field_name) {
    for (size_t i = 0; i < keys.size(); ++i) {
      if (keys_found[i]) continue;
      if (keys[i].compare(field_name) == 0) {
        return i;
      }
    }
    return keys.size();
  };

  auto update_fields = [&](const TVMFFITypeInfo* tinfo) {
    for (int i = 0; i < tinfo->num_fields; ++i) {
      const TVMFFIFieldInfo* field_info = tinfo->fields + i;
      size_t arg_index = search_field(field_info->name);
      void* field_addr = reinterpret_cast<char*>(ptr.get()) + field_info->offset;
      if (arg_index < keys.size()) {
        AnyView field_value = args[static_cast<int>(arg_index * 2 + 2)];
        // The setter reports a type mismatch through the safe-call slot, not
        // by throwing; dropping the status would silently null the field.
        TVM_FFI_CHECK_SAFE_CALL(reflection::CallFieldSetter(
            field_info, field_addr, reinterpret_cast<const TVMFFIAny*>(&field_value)));
        keys_found[arg_index] = true;
      } else if (field_info->flags & kTVMFFIFieldFlagBitMaskHasDefault) {
        reflection::SetFieldToDefault(field_info, field_addr);
      } else {
        TVM_FFI_THROW(TypeError) << "Required field `"
                                 << String(field_info->name.data, field_info->name.size)
                                 << "` not set in type `" << TypeIndexToTypeKey(type_index) << "`";
      }
    }
  };

  // iterate through acenstors in parent to child order
  // skip the first one since it is always the root object
  for (int i = 1; i < type_info->type_depth; ++i) {
    update_fields(type_info->type_ancestors[i]);
  }
  update_fields(type_info);

  for (size_t i = 0; i < keys.size(); ++i) {
    if (!keys_found[i]) {
      TVM_FFI_THROW(TypeError) << "Type `" << TypeIndexToTypeKey(type_index)
                               << "` does not have field `" << keys[i] << "`";
    }
  }
  *ret = ObjectRef(ptr);
}

inline void AccessStepRegisterReflection() {
  // register access step reflection here since it is only needed for bindings
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<AccessStepObj>(refl::init(false))
      .def_ro("kind", &AccessStepObj::kind)
      .def_ro("key", &AccessStepObj::key);
  // Register __ffi_repr__ for AccessStep: format one step fragment.
  //   kAttr             -> ".name"
  //   kArrayItem        -> "[index]"
  //   kMapItem          -> "[<repr(key)>]"    (string keys are quoted via fn_repr)
  //   kAttrMissing      -> "[<missing:"name">]"
  //   kArrayItemMissing -> "[<missing:index>]"
  //   kMapItemMissing   -> "[<missing:<repr(key)>>]"
  refl::TypeAttrDef<AccessStepObj>().def(
      type_attr::kRepr.name, [](const Object* self, const ffi::Function& fn_repr) -> ffi::String {
        const auto* step = AnyView(self).as_or_throw<const AccessStepObj*>();
        std::ostringstream os;
        switch (step->kind) {
          case AccessKind::kAttr:
            os << "." << step->key.cast<ffi::String>();
            break;
          case AccessKind::kArrayItem:
            os << "[" << step->key.cast<int64_t>() << "]";
            break;
          case AccessKind::kMapItem:
            os << "[" << fn_repr(step->key).cast<ffi::String>() << "]";
            break;
          case AccessKind::kAttrMissing:
            os << "[<missing:" << fn_repr(step->key).cast<ffi::String>() << ">]";
            break;
          case AccessKind::kArrayItemMissing:
            os << "[<missing:" << step->key.cast<int64_t>() << ">]";
            break;
          case AccessKind::kMapItemMissing:
            os << "[<missing:" << fn_repr(step->key).cast<ffi::String>() << ">]";
            break;
          default:
            TVM_FFI_UNREACHABLE();
        }
        return os.str();
      });
}

inline void AccessPathRegisterReflection() {
  // register access path reflection here since it is only needed for bindings
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<AccessPathObj>(refl::init(false))
      .def_ro("parent", &AccessPathObj::parent)
      .def_ro("step", &AccessPathObj::step)
      .def_ro("depth", &AccessPathObj::depth)
      .def_static("_root", &AccessPath::Root)
      .def("_extend", &AccessPathObj::Extend)
      .def("_attr", &AccessPathObj::Attr)
      .def("_array_item", &AccessPathObj::ArrayItem)
      .def("_map_item", &AccessPathObj::MapItem)
      .def("_attr_missing", &AccessPathObj::AttrMissing)
      .def("_array_item_missing", &AccessPathObj::ArrayItemMissing)
      .def("_map_item_missing", &AccessPathObj::MapItemMissing)
      .def("_is_prefix_of", &AccessPathObj::IsPrefixOf)
      .def("_to_steps", &AccessPathObj::ToSteps)
      .def("_path_equal",
           [](const AccessPath& self, const AccessPath& other) { return self->PathEqual(other); });
  // Register __ffi_repr__ for AccessPath: flatten via ToSteps() and walk the resulting vector.
  // Root (depth == 0) emits "<root>"; non-root nodes concatenate each step's fragment.
  refl::TypeAttrDef<AccessPathObj>().def(
      type_attr::kRepr.name, [](const Object* self, const ffi::Function& fn_repr) -> ffi::String {
        const auto* path = AnyView(self).as_or_throw<const AccessPathObj*>();
        Array<AccessStep> steps = path->ToSteps();
        std::ostringstream os;
        os << "<root>";
        for (const AccessStep& step : steps) {
          os << fn_repr(step).cast<ffi::String>();
        }
        return os.str();
      });
}

int64_t StructuralKeyHash(const Any& key) {
  const auto* key_obj = key.as<StructuralKeyObj>();
  TVM_FFI_ICHECK_NOTNULL(key_obj);
  return key_obj->hash_i64;
}

bool StructuralKeyEqual(const Any& lhs, const Any& rhs) {
  if (lhs.same_as(rhs)) {
    return true;
  }
  const auto* lhs_obj = lhs.as<StructuralKeyObj>();
  const auto* rhs_obj = rhs.as<StructuralKeyObj>();
  TVM_FFI_ICHECK_NOTNULL(lhs_obj);
  TVM_FFI_ICHECK_NOTNULL(rhs_obj);
  if (lhs_obj->hash_i64 != rhs_obj->hash_i64) {
    return false;
  }
  return StructuralEqual::Equal(lhs_obj->key, rhs_obj->key);
}

inline void StructuralKeyRegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<StructuralKeyObj>()
      .def(refl::init<Any>(), "Constructor that takes any key and computes its structural hash")
      .def_ro("key", &StructuralKeyObj::key)
      .def_ro("hash_i64", &StructuralKeyObj::hash_i64);
  refl::TypeAttrDef<StructuralKeyObj>()
      .native_fn_attr<&StructuralKeyHash>(type_attr::kAnyHash)
      .native_fn_attr<&StructuralKeyEqual>(type_attr::kAnyEqual);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  AccessStepRegisterReflection();
  AccessPathRegisterReflection();
  StructuralKeyRegisterReflection();

  refl::GlobalDef()
      .def_packed("ffi.MakeObjectFromPackedArgs", MakeObjectFromPackedArgs)
      .def("ffi.StructuralKey", [](Any key) { return StructuralKey(std::move(key)); })
      .def("ffi.StructuralKeyEqual", StructuralKeyEqual);
}

}  // namespace reflection
}  // namespace ffi
}  // namespace tvm
