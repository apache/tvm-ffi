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
 * \file library_module_dynamic_lib.cc
 * \brief Create library module to load from dynamic shared library.
 */
#include <tvm/ffi/memory.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>

#include <condition_variable>
#include <mutex>
#include <unordered_map>

#include "module_internal.h"

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#if defined(__hexagon__)
extern "C" {
#include <HAP_farf.h>
}
#endif

namespace tvm {
namespace ffi {

struct DSOModuleCache {
  struct Entry {
    WeakObjectPtr<ModuleObj> root;
    WeakObjectPtr<ModuleObj> library;
  };

  std::recursive_mutex mutex;
  std::condition_variable_any released;
  std::unordered_map<void*, Entry> entries;

  static DSOModuleCache* Global() {
    static auto* cache = new DSOModuleCache();
    return cache;
  }
};

class DSOLibrary final : public Library {
 public:
  DSOLibrary(const String& name, DSOModuleCache* cache) : cache_(cache) { Load(name); }
  ~DSOLibrary() final {
    std::scoped_lock lock(cache_->mutex);
    if (cached_) cache_->entries.erase(lib_handle_);
    if (lib_handle_) Unload();
    if (cached_) cache_->released.notify_all();
  }

  void* GetSymbol(const String& name) final { return GetSymbol_(name.c_str()); }

  static Module LoadModule(const String& name) {
    auto* cache = DSOModuleCache::Global();
    std::unique_lock<std::recursive_mutex> lock(cache->mutex);
    while (true) {
      auto lib = make_object<DSOLibrary>(name, cache);
      auto it = cache->entries.find(lib->lib_handle_);
      Optional<Module> library_module;
      if (it != cache->entries.end()) {
        if (auto root = it->second.root.lock()) return Module(root);
        if (auto context = it->second.library.lock()) {
          library_module = Module(context);
        } else {
          // The last owner is being destroyed. Let it close the old image before reloading.
          lib.reset();
          cache->released.wait(lock);
          continue;
        }
      }
      Optional<Module> context;
      Module root = CreateLibraryModule(lib, library_module, &context);
      if (context.has_value()) {
        cache->entries[lib->lib_handle_] = {
            detail::ObjectUnsafe::ObjectPtrFromObjectRef<ModuleObj>(root),
            detail::ObjectUnsafe::ObjectPtrFromObjectRef<ModuleObj>(context.value())};
        // A reused context owns the original handle and its cache entry.
        lib->cached_ = !library_module.has_value();
      }
      return root;
    }
  }

 private:
  DSOModuleCache* cache_;
  bool cached_{false};
  // private system dependent implementation
  void* GetSymbol_(const char* name);
  void Load(const String& name);
  void Unload();

#if defined(_WIN32)
  //! \brief Windows library handle
  HMODULE lib_handle_{nullptr};
#else
  // \brief Linux library handle
  void* lib_handle_{nullptr};
#endif
};

#if defined(_WIN32)

void* DSOLibrary::GetSymbol_(const char* name) {
  return reinterpret_cast<void*>(GetProcAddress(lib_handle_, (LPCSTR)name));  // NOLINT(*)
}

void DSOLibrary::Load(const String& name) {
  // use wstring version that is needed by LLVM.
  std::wstring wname(name.data(), name.data() + name.size());
  lib_handle_ = LoadLibraryW(wname.c_str());
  TVM_FFI_ICHECK(lib_handle_ != nullptr) << "Failed to load dynamic shared library " << name;
}

void DSOLibrary::Unload() {
  FreeLibrary(lib_handle_);
  lib_handle_ = nullptr;
}

#else

void DSOLibrary::Load(const String& name) {
  lib_handle_ = dlopen(name.c_str(), RTLD_LAZY | RTLD_LOCAL);
  TVM_FFI_ICHECK(lib_handle_ != nullptr)
      << "Failed to load dynamic shared library " << name << " " << dlerror();
#if defined(__hexagon__)
  int p;
  int rc = dlinfo(lib_handle_, RTLD_DI_LOAD_ADDR, &p);
  if (rc)
    FARF(ERROR, "error getting model .so start address : %u", rc);
  else
    FARF(ALWAYS, "Model .so Start Address : %x", p);
#endif
}

void* DSOLibrary::GetSymbol_(const char* name) { return dlsym(lib_handle_, name); }

void DSOLibrary::Unload() {
  dlclose(lib_handle_);
  lib_handle_ = nullptr;
}
#endif

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("ffi.Module.load_from_file.so",
                        [](const String& library_path, const String&) {
                          return DSOLibrary::LoadModule(library_path);
                        });
}
}  // namespace ffi
}  // namespace tvm
