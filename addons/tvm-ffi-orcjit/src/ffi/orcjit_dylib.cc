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

ORCJITDynamicLibraryObj::ORCJITDynamicLibraryObj(ORCJITExecutionSession session,
                                                 llvm::orc::JITDylib* dylib, llvm::orc::LLJIT* jit,
                                                 String name)
    : session_(std::move(session)), dylib_(dylib), jit_(jit), name_(std::move(name)) {
  if (void** ctx_addr = reinterpret_cast<void**>(GetSymbol(ffi::symbol::tvm_ffi_library_ctx))) {
    *ctx_addr = this;
  }
  Module::VisitContextSymbols([this](const ffi::String& name, void* symbol) {
    if (void** ctx_addr = reinterpret_cast<void**>(GetSymbol(ffi::symbol::tvm_ffi_library_ctx))) {
      *ctx_addr = symbol;
    }
  });
  TVM_FFI_CHECK(dylib_ != nullptr, ValueError) << "JITDylib cannot be null";
  TVM_FFI_CHECK(jit_ != nullptr, ValueError) << "LLJIT cannot be null";
}

void ORCJITDynamicLibraryObj::AddObjectFile(const String& path) {
  // Read object file
  auto buffer_or_err = llvm::MemoryBuffer::getFile(path.c_str());
  if (!buffer_or_err) {
    TVM_FFI_THROW(IOError) << "Failed to read object file: " << path;
  }

  // Add object file to this JITDylib
  call_llvm(jit_->addObjectFile(*dylib_, std::move(*buffer_or_err)), "Failed to add object file");
  // call_llvm(jit_->initialize(*dylib_), "Failed to run initializers for JITDylib");
}

void ORCJITDynamicLibraryObj::SetLinkOrder(const std::vector<llvm::orc::JITDylib*>& dylibs) {
  // Clear and rebuild the link order
  link_order_.clear();

  for (auto* lib : dylibs) {
    link_order_.emplace_back(lib, llvm::orc::JITDylibLookupFlags::MatchAllSymbols);
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

// Run initializers after possible first successful lookup, to ensure materialization has occurred
#if defined(__linux__)
  session_->RunPendingInitializers(GetJITDylib());
#elif defined(__APPLE__)
  jit_->initialize(*dylib_);
#endif
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
        libs.push_back(&GetJITDylib());
      }
      SetLinkOrder(libs);
    });
  }

  // TVM-FFI exports have __tvm_ffi_ prefix
  std::string symbol_name = symbol::tvm_ffi_symbol_prefix + std::string(name);

  // Try to get the symbol - return NullOpt if not found
  if (void* symbol = GetSymbol(symbol_name)) {
    // Wrap C function pointer as tvm-ffi Function
    TVMFFISafeCallType c_func = reinterpret_cast<TVMFFISafeCallType>(symbol);

    return Function::FromPacked([c_func, name](PackedArgs args, Any* rv) {
      TVM_FFI_ICHECK_LT(rv->type_index(), ffi::TypeIndex::kTVMFFIStaticObjectBegin);
      TVM_FFI_CHECK_SAFE_CALL((*c_func)(nullptr, reinterpret_cast<const TVMFFIAny*>(args.data()),
                                        args.size(), reinterpret_cast<TVMFFIAny*>(rv)));
    });
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
           [](const std::string& orc_rt_path) { return ORCJITExecutionSession(orc_rt_path); })
      .def("orcjit.ExecutionSessionCreateDynamicLibrary",
           [](const ORCJITExecutionSession& session, const String& name) -> Module {
             return session->CreateDynamicLibrary(name);
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
