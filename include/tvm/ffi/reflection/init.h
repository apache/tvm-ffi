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
 * \file tvm/ffi/reflection/init.h
 * \brief Auto-generated __ffi_init__ constructor for reflected types.
 */
#ifndef TVM_FFI_REFLECTION_INIT_H_
#define TVM_FFI_REFLECTION_INIT_H_

#include <tvm/ffi/any.h>
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/function_details.h>
#include <tvm/ffi/object.h>
#include <tvm/ffi/reflection/accessor.h>
#include <tvm/ffi/string.h>

#include <algorithm>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace tvm {
namespace ffi {
namespace reflection {

/*!
 * \brief Create a packed ``__ffi_init__`` constructor for the given type.
 *
 * The returned Function creates an instance via the type's creator, binds
 * positional (and optionally keyword) arguments to ``init=True`` fields,
 * and fills defaults for unbound fields.
 *
 * \param type_index The runtime type index.
 * \return The init Function.
 */
inline Function MakeInit(int32_t type_index) {
  // Pre-computed field analysis for auto-generated init.
  struct AutoInitInfo {
    struct Entry {
      const TVMFFIFieldInfo* info;
      bool init;
      bool kw_only;
      bool has_default;
    };
    std::vector<Entry> all_fields;
    std::vector<size_t> init_indices;
    std::vector<size_t> pos_indices;
    std::unordered_map<std::string_view, size_t> name_to_index;
    std::string_view type_key;
  };
  // ---- Pre-compute field analysis (once per type) -------------------------
  const TVMFFITypeInfo* type_info = TVMFFIGetTypeInfo(type_index);
  TVM_FFI_ICHECK(HasCreator(type_info)) << "Type `" << TypeIndexToTypeKey(type_index)
                                        << "` has no creator or __ffi_new__ for __ffi_init__";

  auto info = std::make_shared<AutoInitInfo>();
  info->type_key = std::string_view(type_info->type_key.data, type_info->type_key.size);

  ForEachFieldInfo(type_info, [&](const TVMFFIFieldInfo* fi) {
    bool is_init = (fi->flags & kTVMFFIFieldFlagBitMaskInitOff) == 0;
    bool is_kw = (fi->flags & kTVMFFIFieldFlagBitMaskKwOnly) != 0;
    bool has_def = (fi->flags & kTVMFFIFieldFlagBitMaskHasDefault) != 0;
    info->all_fields.push_back({fi, is_init, is_kw, has_def});
    size_t idx = info->all_fields.size() - 1;
    if (is_init) {
      info->init_indices.push_back(idx);
      // name pointer is stable (static reflection data), safe for string_view key.
      info->name_to_index[std::string_view(fi->name.data, fi->name.size)] = idx;
      if (!is_kw) {
        info->pos_indices.push_back(idx);
      }
    }
  });

  // Reorder pos_indices so required fields come before optional ones,
  // matching the Python signature ordering produced by _make_init_signature.
  std::stable_partition(info->pos_indices.begin(), info->pos_indices.end(),
                        [&](size_t idx) { return !info->all_fields[idx].has_default; });

  // Eagerly resolve the KWARGS sentinel via global function registry.
  ObjectRef kwargs_sentinel =
      Function::GetGlobalRequired("ffi.GetKwargsObject")().cast<ObjectRef>();

  return Function::FromPacked(
      [info, kwargs_sentinel, type_info](PackedArgs args, Any* rv) {
        // ---- 1. Create object via CreateEmptyObject --------------------------
        ObjectPtr<Object> obj_ptr = CreateEmptyObject(type_info);

        // ---- 2. Find KWARGS sentinel position --------------------------------
        int kwargs_pos = -1;
        for (int i = 0; i < args.size(); ++i) {
          auto opt = args[i].as<ObjectRef>();
          if (opt.has_value() && opt.value().same_as(kwargs_sentinel)) {
            kwargs_pos = i;
            break;
          }
        }

        // ---- 3. Bind arguments to fields -------------------------------------
        const auto raw_args = reinterpret_cast<const TVMFFIAny*>(args.data());
        std::vector<bool> field_set(info->all_fields.size(), false);

        auto set_field = [&](size_t fi, const TVMFFIAny* value) {
          void* addr = reinterpret_cast<char*>(obj_ptr.get()) + info->all_fields[fi].info->offset;
          TVM_FFI_CHECK_SAFE_CALL(CallFieldSetter(info->all_fields[fi].info, addr, value));
          field_set[fi] = true;
        };

        if (kwargs_pos >= 0) {
          // --- 3a. KWARGS mode ------------------------------------------------
          int pos_arg = 0;
          for (size_t fi : info->pos_indices) {
            if (pos_arg < kwargs_pos) {
              set_field(fi, &raw_args[pos_arg]);
              ++pos_arg;
            }
          }
          if (pos_arg < kwargs_pos) {
            TVM_FFI_THROW(TypeError)
                << info->type_key << ".__ffi_init__() takes at most " << info->pos_indices.size()
                << " positional argument(s), but " << kwargs_pos << " were given";
          }
          // Key-value pairs after the sentinel.
          int kv_count = args.size() - kwargs_pos - 1;
          if (kv_count % 2 != 0) {
            TVM_FFI_THROW(TypeError)
                << info->type_key
                << ".__ffi_init__() KWARGS requires an even number of key-value arguments";
          }
          for (int i = kwargs_pos + 1; i < args.size(); i += 2) {
            String key = args[i].cast<String>();
            std::string_view key_sv(key.data(), key.size());
            auto it = info->name_to_index.find(key_sv);
            if (it == info->name_to_index.end()) {
              TVM_FFI_THROW(TypeError)
                  << info->type_key << ".__ffi_init__() got an unexpected keyword argument '" << key
                  << "'";
            }
            size_t idx = it->second;
            if (field_set[idx]) {
              TVM_FFI_THROW(TypeError) << info->type_key << ".__ffi_init__() got multiple values "
                                       << "for argument '" << key << "'";
            }
            set_field(idx, &raw_args[i + 1]);
          }
        } else {
          // --- 3b. Positional-only mode ---------------------------------------
          if (static_cast<size_t>(args.size()) > info->pos_indices.size()) {
            TVM_FFI_THROW(TypeError)
                << info->type_key << ".__ffi_init__() takes at most " << info->pos_indices.size()
                << " positional argument(s), but " << args.size() << " were given";
          }
          for (int i = 0; i < args.size(); ++i) {
            set_field(info->pos_indices[i], &raw_args[i]);
          }
        }

        // ---- 4. Fill defaults and check required fields ----------------------
        for (size_t fi = 0; fi < info->all_fields.size(); ++fi) {
          if (field_set[fi]) continue;
          if (info->all_fields[fi].has_default) {
            void* addr = reinterpret_cast<char*>(obj_ptr.get()) + info->all_fields[fi].info->offset;
            SetFieldToDefault(info->all_fields[fi].info, addr);
          } else if (info->all_fields[fi].init) {
            TVM_FFI_THROW(TypeError)
                << info->type_key << ".__ffi_init__() missing required argument: '"
                << std::string_view(info->all_fields[fi].info->name.data,
                                    info->all_fields[fi].info->name.size)
                << "'";
          }
          // init=False without default: leave at creator default.
        }

        // ---- 5. Return -------------------------------------------------------
        *rv = ObjectRef(obj_ptr);
      });
}

/*!
 * \brief Auto-generate, register a packed ``__ffi_init__`` method for the given type.
 *
 * Calls MakeInit and registers the result as the type's ``__ffi_init__``
 * method with appropriate metadata.
 *
 * \param type_index The runtime type index.
 */
inline void RegisterAutoInit(int32_t type_index) {
  Function auto_init_fn = MakeInit(type_index);
  TVMFFIMethodInfo info;
  static constexpr const char* kInitName = "__ffi_init__";
  info.name = TVMFFIByteArray{kInitName, std::char_traits<char>::length(kInitName)};
  info.doc = TVMFFIByteArray{nullptr, 0};
  info.flags = kTVMFFIFieldFlagBitMaskIsStaticMethod;
  info.method = AnyView(auto_init_fn).CopyToTVMFFIAny();
  static const std::string kMetadata =
      "{\"type_schema\":" + std::string(::tvm::ffi::details::TypeSchemaImpl<Function>::v()) +
      ",\"auto_init\":true}";
  info.metadata = TVMFFIByteArray{kMetadata.c_str(), kMetadata.size()};
  TVM_FFI_CHECK_SAFE_CALL(TVMFFITypeRegisterMethod(type_index, &info));
}

}  // namespace reflection
}  // namespace ffi
}  // namespace tvm
#endif  // TVM_FFI_REFLECTION_INIT_H_
