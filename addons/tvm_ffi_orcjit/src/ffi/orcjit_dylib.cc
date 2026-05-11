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
 * \file orcjit_dylib.cc
 * \brief LLVM ORC JIT DynamicLibrary implementation
 */

#include "orcjit_dylib.h"

#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/Object/ObjectFile.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/cast.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>

#include "orcjit_session.h"
#include "orcjit_utils.h"

namespace tvm {
namespace ffi {
namespace orcjit {

namespace {
// When JIT thunks start reading their ctx/handle arg, grow this wrapper with
// a DylibFnContext prefix whose first field is the thunk pointer, and flip
// `safe_call` at a slab-emitted redirect that dispatches via ctx[0]. Example:
//
//     struct DylibFnContext { TVMFFISafeCallType fn; /* + closure fields */ };
//     struct DylibFnContextWithModule {
//       DylibFnContext ctx;  // first — pointer-interconvertible with wrapper
//       Module module_ref;
//     };
struct DylibFnContextWithModule {
  Module module_ref;  // keeps the owning dylib (and its slab) alive
};

void DeleteDylibFnContextWithModule(void* p) { delete static_cast<DylibFnContextWithModule*>(p); }
}  // namespace

ORCJITDynamicLibraryObj::ORCJITDynamicLibraryObj(ORCJITExecutionSession session,
                                                 llvm::orc::JITDylib* dylib, llvm::orc::LLJIT* jit,
                                                 String name)
    : session_(std::move(session)), dylib_(dylib), jit_(jit), name_(std::move(name)) {
  TVM_FFI_CHECK(dylib_ != nullptr, ValueError) << "JITDylib cannot be null";
  TVM_FFI_CHECK(jit_ != nullptr, ValueError) << "LLJIT cannot be null";
  if (void** ctx_addr = reinterpret_cast<void**>(GetSymbol(ffi::symbol::tvm_ffi_library_ctx))) {
    *ctx_addr = this;
  }
  Module::VisitContextSymbols([this](const ffi::String& name, void* symbol) {
    if (void** ctx_addr = reinterpret_cast<void**>(GetSymbol(name))) {
      *ctx_addr = symbol;
    }
  });
}

ORCJITDynamicLibraryObj::~ORCJITDynamicLibraryObj() {
  // Step 1: run static destructors for code in this JITDylib via our
  // InitFiniPlugin (uniform across Linux / macOS / Windows — see the
  // plugin docstring in llvm_patches/init_fini_plugin.h).
  session_->RunPendingDeinitializers(GetJITDylib());
#ifdef __APPLE__
  // Step 1b (macOS only): drain per-dylib __cxa_atexit registrations in
  // LIFO.  Clang on Darwin lowers __attribute__((destructor)) and C++
  // global dtors as __cxa_atexit(fn, arg, &__dso_handle) registrations
  // during init; our shim (see orcjit_session.cc) captured them into
  // this vector via the TLS pointer published below.
  //
  // Pop-and-call handles re-entrant registrations from within a dtor —
  // the CxaAtexitRecordsScope keeps the TLS pointer live so any
  // ___cxa_atexit call from inside a dtor also lands here.
  {
    CxaAtexitRecordsScope scope(&cxa_atexit_records_);
    while (!cxa_atexit_records_.empty()) {
      auto [fn, arg] = cxa_atexit_records_.back();
      cxa_atexit_records_.pop_back();
      fn(arg);
    }
  }
#endif
  // Step 2: remove the JITDylib from the ExecutionSession. Triggers
  // JITDylib::clear(), which releases all tracked linker resources — in
  // particular, invokes the ObjectLinkingLayer's ResourceManager which calls
  // our memory manager's deallocate() for every FinalizedAlloc belonging to
  // this dylib.  Without this call JIT code pages accumulate in the arena
  // until the whole session is destroyed.  No Platform is registered in any
  // of our LLJIT configurations (see orcjit_session.cc), so there is no
  // Platform::teardownJITDylib callback to worry about.
  session_->RemoveDylib(dylib_);
  dylib_ = nullptr;
}

void ORCJITDynamicLibraryObj::AddObjectFile(const String& path) {
  // Read object file
  auto buffer_or_err = llvm::MemoryBuffer::getFile(path.c_str());
  if (!buffer_or_err) {
    TVM_FFI_THROW(IOError) << "Failed to read object file: " << path;
  }

  // Add object file to this JITDylib
  TVM_FFI_ORCJIT_LLVM_CALL(jit_->addObjectFile(*dylib_, std::move(*buffer_or_err)));
}

void ORCJITDynamicLibraryObj::SetLinkOrder(const std::vector<llvm::orc::JITDylib*>& dylibs) {
  // Rebuild the link order: user-specified libraries first, then the LLJIT
  // default link order (Main → Platform → ProcessSymbols).  Preserving the
  // default link order is essential — without ProcessSymbols, C++ objects
  // that need host-process symbols (runtime, libtvm_ffi) would fail to link.
  link_order_.clear();

  for (auto* lib : dylibs) {
    link_order_.emplace_back(lib, llvm::orc::JITDylibLookupFlags::MatchAllSymbols);
  }
  for (auto& kv : jit_->defaultLinkOrder()) {
    link_order_.emplace_back(kv.first, kv.second);
  }

  // Set the link order in the LLVM JITDylib
  dylib_->setLinkOrder(link_order_, false);
}

void* ORCJITDynamicLibraryObj::GetSymbol(const String& name) {
  // Build search order: this dylib first, then all linked dylibs
  llvm::orc::JITDylibSearchOrder search_order;
  search_order.emplace_back(dylib_, llvm::orc::JITDylibLookupFlags::MatchAllSymbols);
  // Append linked libraries
  search_order.insert(search_order.end(), link_order_.begin(), link_order_.end());

  // Look up symbol using the full search order
  auto symbol_or_err =
      jit_->getExecutionSession().lookup(search_order, jit_->mangleAndIntern(name.c_str()));

  // Run pending initializers via InitFiniPlugin.  RunPendingInitializers
  // drains and erases the map entry; subsequent calls are no-ops until new
  // object files add fresh entries — supports incremental loading.
  //
  // On macOS the scope publishes this dylib's __cxa_atexit records vector
  // so the ___cxa_atexit shim (see orcjit_session.cc) can route dtor
  // registrations here for our destructor to drain.  Static init on C
  // and C++ code typically happens here (first GetSymbol resolves the
  // `main` entry point, which materializes and fires __mod_init_func).
#ifdef __APPLE__
  CxaAtexitRecordsScope scope(&cxa_atexit_records_);
#endif
  session_->RunPendingInitializers(GetJITDylib());
  // Convert ExecutorAddr to pointer
  return symbol_or_err ? symbol_or_err->getAddress().toPtr<void*>() : nullptr;
}

llvm::orc::JITDylib& ORCJITDynamicLibraryObj::GetJITDylib() {
  TVM_FFI_CHECK(dylib_ != nullptr, InternalError) << "JITDylib is null";
  return *dylib_;
}

Optional<Function> ORCJITDynamicLibraryObj::GetFunction(const String& name) {
  if (name == "orcjit.add_object_file") {
    return Function::FromTyped([this](const String& path) { AddObjectFile(path); });
  }
  if (name == "orcjit.set_link_order") {
    return Function::FromTyped([this](const Array<ORCJITDynamicLibrary>& libraries) {
      std::vector<llvm::orc::JITDylib*> libs;
      libs.reserve(libraries.size());
      for (const ORCJITDynamicLibrary& lib : libraries) {
        libs.push_back(&lib->GetJITDylib());
      }
      SetLinkOrder(libs);
    });
  }

  // TVM-FFI exports have __tvm_ffi_ prefix
  std::string symbol_name = symbol::tvm_ffi_symbol_prefix + std::string(name);

  // Try to get the symbol - return NullOpt if not found
  if (void* symbol = GetSymbol(symbol_name)) {
    TVMFFISafeCallType c_func = reinterpret_cast<TVMFFISafeCallType>(symbol);
    auto* wrapper = new DylibFnContextWithModule{GetRef<Module>(this)};
    return Function::FromExternC(wrapper, c_func, DeleteDylibFnContextWithModule);
  }
  return std::nullopt;
}

//-------------------------------------
// Registration
//-------------------------------------

static void RegisterOrcJITFunctions() {
  static bool registered = false;
  if (registered) return;
  registered = true;

  namespace refl = tvm::ffi::reflection;

  refl::ObjectDef<ORCJITExecutionSessionObj>();

  refl::GlobalDef()
      .def("orcjit.ExecutionSession",
           [](const std::string& orc_rt_path, int64_t slab_size_bytes) {
             return ORCJITExecutionSession(orc_rt_path, slab_size_bytes);
           })
      .def("orcjit.ExecutionSessionCreateDynamicLibrary",
           [](const ORCJITExecutionSession& session, const String& name) -> Module {
             return session->CreateDynamicLibrary(name);
           })
      .def("orcjit.ExecutionSessionClearFreeSlabs",
           [](const ORCJITExecutionSession& session) -> int64_t {
             return session->ClearFreeSlabs();
           });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  // This block may not execute when loaded via dlopen on some platforms.
  // Call TVMFFIOrcJITInitialize() explicitly if functions are not registered.
  RegisterOrcJITFunctions();
}

}  // namespace orcjit
}  // namespace ffi
}  // namespace tvm

// C API for explicit initialization
extern "C" {

TVM_FFI_DLL_EXPORT void TVMFFIOrcJITInitialize() { tvm::ffi::orcjit::RegisterOrcJITFunctions(); }
}
