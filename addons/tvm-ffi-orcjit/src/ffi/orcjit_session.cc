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
 * \file orcjit_session.cc
 * \brief LLVM ORC JIT ExecutionSession implementation
 */

#include "orcjit_session.h"

#include <llvm/ExecutionEngine/Orc/AbsoluteSymbols.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h>
#include <llvm/ExecutionEngine/Orc/Shared/ExecutorSymbolDef.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/TargetSelect.h>
#include <tvm/ffi/cast.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/object.h>
#include <tvm/ffi/reflection/registry.h>

#include <cstddef>

#include "orcjit_dylib.h"
#include "orcjit_utils.h"

namespace tvm {
namespace ffi {
namespace orcjit {

// Global registry mapping dso_handle pointers to their owning session.
// The platform's __dso_handle for each JITDylib resolves to the JITDylib* address,
// so dso values passed to __cxa_atexit are JITDylib pointers.
static std::mutex g_dso_session_mutex;
static std::unordered_map<void*, ORCJITExecutionSessionObj*> g_dso_to_session;

extern "C" int orcjit_cxa_atexit(void (*f)(void*), void* arg, void* dso) {
  std::lock_guard<std::mutex> lock(g_dso_session_mutex);
  auto it = g_dso_to_session.find(dso);
  if (it != g_dso_to_session.end()) {
    it->second->RegisterCxaAtExit(f, arg, dso);
    return 0;
  }
  return 0;
}

// Initialize LLVM native target (only once)
struct LLVMInitializer {
  LLVMInitializer() {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();
  }
};

static LLVMInitializer llvm_initializer;

#define ITERATE_SECTION_PER_EDGE(SECTION, BLOCK, EDGE, TARGET, ...) \
  for (auto* BLOCK : SECTION.blocks()) {                            \
    for (auto& EDGE : BLOCK->edges()) {                             \
      auto& TARGET = EDGE.getTarget();                              \
      __VA_ARGS__                                                   \
    }                                                               \
  }

class InitFiniPlugin : public llvm::orc::ObjectLinkingLayer::Plugin {
  ORCJITExecutionSession session_;

 public:
  explicit InitFiniPlugin(ORCJITExecutionSession session) : session_(std::move(session)) {}

  void modifyPassConfig(llvm::orc::MaterializationResponsibility& MR, llvm::jitlink::LinkGraph& G,
                        llvm::jitlink::PassConfiguration& Config) override {
    auto& jit_dylib = MR.getTargetJITDylib();
    Config.PrePrunePasses.emplace_back([this, &jit_dylib](llvm::jitlink::LinkGraph& G) {
      // Mark .fini and .dtors symbols as live for deinitializers
      for (auto& Section : G.sections()) {
        auto section_name = Section.getName();
        if (section_name.starts_with(".init_array")) {
          int priority = 0;
          bool has_priority = section_name.consume_front(".init_array.") &&
                              !section_name.getAsInteger(10, priority);
          ITERATE_SECTION_PER_EDGE(Section, Block, Edge, Target, {
            if (Target.hasName()) {
              session_->AddPendingInitializer(
                  &jit_dylib,
                  {(*Target.getName()).str(), llvm::orc::ExecutorAddr(0),
                   has_priority
                       ? ORCJITExecutionSessionObj::InitFiniEntry::Section::kInitArrayWithPriority
                       : ORCJITExecutionSessionObj::InitFiniEntry::Section::kInitArray,
                   priority});
            }
          });
        } else if (section_name.starts_with(".init")) {
          ITERATE_SECTION_PER_EDGE(Section, Block, Edge, Target, {
            if (Target.hasName()) {
              session_->AddPendingInitializer(
                  &jit_dylib, {(*Target.getName()).str(), llvm::orc::ExecutorAddr(0),
                               ORCJITExecutionSessionObj::InitFiniEntry::Section::kInit, 0});
            }
          });
        }
        if (section_name.starts_with(".fini_array")) {
          int priority = 0;
          bool has_priority = section_name.consume_front(".fini_array.") &&
                              !section_name.getAsInteger(10, priority);
          ITERATE_SECTION_PER_EDGE(Section, Block, Edge, Target, {
            if (Target.hasName()) {
              session_->AddPendingDeinitializer(
                  &jit_dylib,
                  {(*Target.getName()).str(), llvm::orc::ExecutorAddr(0),
                   has_priority
                       ? ORCJITExecutionSessionObj::InitFiniEntry::Section::kFiniArrayWithPriority
                       : ORCJITExecutionSessionObj::InitFiniEntry::Section::kFiniArray,
                   -priority});
            }
          });
        } else if (section_name.starts_with(".fini")) {
          ITERATE_SECTION_PER_EDGE(Section, Block, Edge, Target, {
            if (Target.hasName()) {
              session_->AddPendingDeinitializer(
                  &jit_dylib, {(*Target.getName()).str(), llvm::orc::ExecutorAddr(0),
                               ORCJITExecutionSessionObj::InitFiniEntry::Section::kFini, 0});
            }
          });
        }

        if (section_name.starts_with(".dtors")) {
          for (auto* Block : Section.blocks()) {
            for (auto* Sym : G.defined_symbols()) {
              if (&Sym->getBlock() == Block) {
                Sym->setLive(true);
              }
            }
            for (auto& Edge : Block->edges()) {
              Edge.getTarget().setLive(true);
            }
          }
        }
      }
      return llvm::Error::success();
    });
    Config.PostFixupPasses.emplace_back([this, &jit_dylib](llvm::jitlink::LinkGraph& G) {
      for (auto& Sec : G.sections()) {
        auto section_name = Sec.getName();
        if (section_name.starts_with(".ctors")) {
          int priority = 0;
          bool has_priority =
              section_name.consume_front(".ctors.") && !section_name.getAsInteger(10, priority);
          for (auto* Block : Sec.blocks()) {
            // For .ctors, read function pointers directly from block content after fixup
            auto Content = Block->getContent();
            size_t PtrSize = G.getPointerSize();
            for (size_t Offset = 0; Offset + PtrSize <= Content.size(); Offset += PtrSize) {
              uint64_t FnAddr = 0;
              memcpy(&FnAddr, Content.data() + Offset, PtrSize);
              if (FnAddr != 0) {
                session_->AddPendingInitializer(
                    &jit_dylib,
                    {"", llvm::orc::ExecutorAddr(FnAddr),
                     has_priority
                         ? ORCJITExecutionSessionObj::InitFiniEntry::Section::kCtorsWithPriority
                         : ORCJITExecutionSessionObj::InitFiniEntry::Section::kCtors,
                     -priority});
              }
            }
          }
        }
        if (section_name.starts_with(".dtors")) {
          int priority = 0;
          bool has_priority =
              section_name.consume_front(".dtors.") && !section_name.getAsInteger(10, priority);
          for (auto* Block : Sec.blocks()) {
            // For .dtors, read function pointers directly from block content after fixup
            auto Content = Block->getContent();
            size_t PtrSize = G.getPointerSize();
            for (size_t Offset = 0; Offset + PtrSize <= Content.size(); Offset += PtrSize) {
              uint64_t FnAddr = 0;
              memcpy(&FnAddr, Content.data() + Offset, PtrSize);
              if (FnAddr != 0) {
                session_->AddPendingDeinitializer(
                    &jit_dylib,
                    {"", llvm::orc::ExecutorAddr(FnAddr),
                     has_priority
                         ? ORCJITExecutionSessionObj::InitFiniEntry::Section::kDtorsWithPriority
                         : ORCJITExecutionSessionObj::InitFiniEntry::Section::kDtors,
                     priority});
              }
            }
          }
        }
      }
      return llvm::Error::success();
    });
  }

  llvm::Error notifyFailed(llvm::orc::MaterializationResponsibility& MR) override {
    return llvm::Error::success();
  }

  llvm::Error notifyRemovingResources(llvm::orc::JITDylib& JD, llvm::orc::ResourceKey K) override {
    return llvm::Error::success();
  }

  void notifyTransferringResources(llvm::orc::JITDylib& JD, llvm::orc::ResourceKey DstKey,
                                   llvm::orc::ResourceKey SrcKey) override {}
};

ORCJITExecutionSessionObj::ORCJITExecutionSessionObj(const std::string& orc_rt_path)
    : jit_(nullptr) {
  if (!orc_rt_path.empty()) {
    jit_ = std::move(call_llvm(llvm::orc::LLJITBuilder()
                                   .setPlatformSetUp(llvm::orc::ExecutorNativePlatform(orc_rt_path))
                                   .create(),
                               "Failed to create LLJIT with ORC runtime"));
  } else {
    jit_ = std::move(call_llvm(llvm::orc::LLJITBuilder().create(), "Failed to create LLJIT"));
  }
  auto& objlayer = jit_->getObjLinkingLayer();
  static_cast<llvm::orc::ObjectLinkingLayer&>(objlayer).addPlugin(
      std::make_unique<InitFiniPlugin>(GetRef<ORCJITExecutionSession>(this)));
}

ORCJITExecutionSession::ORCJITExecutionSession(const std::string& orc_rt_path) {
  ObjectPtr<ORCJITExecutionSessionObj> obj = make_object<ORCJITExecutionSessionObj>(orc_rt_path);
  data_ = std::move(obj);
}

ORCJITDynamicLibrary ORCJITExecutionSessionObj::CreateDynamicLibrary(const String& name) {
  TVM_FFI_CHECK(jit_ != nullptr, InternalError) << "ExecutionSession not initialized";

  // Generate name if not provided
  String lib_name = name;
  if (lib_name.empty()) {
    std::ostringstream oss;
    oss << "dylib_" << dylib_counter_++;
    lib_name = oss.str();
  }

  llvm::orc::JITDylib& jit_dylib =
      call_llvm(jit_->getExecutionSession().createJITDylib(lib_name.c_str()));
  jit_dylib.addToLinkOrder(jit_->getMainJITDylib());
  if (auto* PlatformJD = jit_->getPlatformJITDylib().get()) {
    jit_dylib.addToLinkOrder(*PlatformJD);
  }

  // The platform defines __dso_handle = ExecutorAddr::fromPtr(&jit_dylib),
  // so the dso value passed to __cxa_atexit will be the JITDylib pointer.
  // Register it in our global map so orcjit_cxa_atexit can find the session.
  RegisterDsoHandle(reinterpret_cast<void*>(&jit_dylib));

  // Define __cxa_atexit as our custom implementation so static destructors are
  // tracked per-dylib and run at dylib teardown, not at process exit.
  llvm::orc::SymbolMap cxa_sym;
  auto& es = jit_->getExecutionSession();
  cxa_sym[es.intern("__cxa_atexit")] = llvm::orc::ExecutorSymbolDef(
      llvm::orc::ExecutorAddr::fromPtr(&orcjit_cxa_atexit),
      llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable);
  call_llvm(jit_dylib.define(llvm::orc::absoluteSymbols(std::move(cxa_sym))),
            "Failed to define __cxa_atexit absolute symbol");

  std::unique_ptr<llvm::orc::DynamicLibrarySearchGenerator> generator =
      call_llvm(llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
                    jit_->getDataLayout().getGlobalPrefix()),
                "Failed to create process symbol resolver");
  jit_dylib.addGenerator(std::move(generator));

  auto dylib = ORCJITDynamicLibrary(make_object<ORCJITDynamicLibraryObj>(
      GetRef<ORCJITExecutionSession>(this), &jit_dylib, jit_.get(), lib_name));

  return dylib;
}

llvm::orc::ExecutionSession& ORCJITExecutionSessionObj::GetLLVMExecutionSession() {
  TVM_FFI_CHECK(jit_ != nullptr, InternalError) << "ExecutionSession not initialized";
  return jit_->getExecutionSession();
}

llvm::orc::LLJIT& ORCJITExecutionSessionObj::GetLLJIT() {
  TVM_FFI_CHECK(jit_ != nullptr, InternalError) << "ExecutionSession not initialized";
  return *jit_;
}

using CtorDtor = void (*)();

void ORCJITExecutionSessionObj::RunPendingInitializers(llvm::orc::JITDylib& jit_dylib) {
  auto it = pending_initializers_.find(&jit_dylib);
  if (it != pending_initializers_.end()) {
    llvm::sort(it->second, [&](InitFiniEntry& a, InitFiniEntry& b) {
      return static_cast<int>(a.section) < static_cast<int>(b.section) ||
             (a.section == b.section && a.priority < b.priority);
    });
    llvm::orc::JITDylibSearchOrder search_order;
    search_order.emplace_back(&jit_dylib, llvm::orc::JITDylibLookupFlags::MatchAllSymbols);
    for (const auto& entry : it->second) {
      if (!entry.name.empty()) {
        // Look up symbol using the full search order
        auto symbol = call_llvm(
            jit_->getExecutionSession().lookup(search_order, jit_->mangleAndIntern(entry.name)));
        auto func = symbol.getAddress().toPtr<CtorDtor>();
        func();
      } else {
        auto func = entry.address.toPtr<CtorDtor>();
        func();
      }
    }
    pending_initializers_.erase(it);
  }
}

void ORCJITExecutionSessionObj::RunPendingDeinitializers(llvm::orc::JITDylib& jit_dylib) {
  auto it = pending_deinitializers_.find(&jit_dylib);
  if (it != pending_deinitializers_.end()) {
    llvm::sort(it->second, [&](InitFiniEntry& a, InitFiniEntry& b) {
      return static_cast<int>(a.section) < static_cast<int>(b.section) ||
             (a.section == b.section && a.priority < b.priority);
    });
    llvm::orc::JITDylibSearchOrder search_order;
    search_order.emplace_back(&jit_dylib, llvm::orc::JITDylibLookupFlags::MatchAllSymbols);
    for (const auto& entry : it->second) {
      if (!entry.name.empty()) {
        // Look up symbol using the full search order
        auto symbol = call_llvm(
            jit_->getExecutionSession().lookup(search_order, jit_->mangleAndIntern(entry.name)));
        auto func = symbol.getAddress().toPtr<CtorDtor>();
        func();
      } else {
        auto func = entry.address.toPtr<CtorDtor>();
        func();
      }
    }
    pending_deinitializers_.erase(it);
  }
}

void ORCJITExecutionSessionObj::AddPendingInitializer(llvm::orc::JITDylib* jit_dylib,
                                                      const InitFiniEntry& entry) {
  auto it = pending_initializers_.find(jit_dylib);
  if (it == pending_initializers_.end()) {
    pending_initializers_[jit_dylib] = std::vector<InitFiniEntry>(1, entry);
  } else {
    pending_initializers_[jit_dylib].push_back(entry);
  }
}

void ORCJITExecutionSessionObj::AddPendingDeinitializer(llvm::orc::JITDylib* jit_dylib,
                                                        const InitFiniEntry& entry) {
  auto it = pending_deinitializers_.find(jit_dylib);
  if (it == pending_deinitializers_.end()) {
    pending_deinitializers_[jit_dylib] = std::vector<InitFiniEntry>(1, entry);
  } else {
    pending_deinitializers_[jit_dylib].push_back(entry);
  }
}

void ORCJITExecutionSessionObj::RegisterCxaAtExit(void (*destructor)(void*), void* arg,
                                                   void* dso_handle) {
  std::lock_guard<std::mutex> lock(cxa_atexit_mutex_);
  cxa_atexit_handlers_[dso_handle].push_back({destructor, arg});
}

void ORCJITExecutionSessionObj::RunCxaFinalize(void* dso_handle) {
  std::vector<CxaAtExitEntry> handlers;
  {
    std::lock_guard<std::mutex> lock(cxa_atexit_mutex_);
    auto it = cxa_atexit_handlers_.find(dso_handle);
    if (it != cxa_atexit_handlers_.end()) {
      handlers = std::move(it->second);
      cxa_atexit_handlers_.erase(it);
    }
  }
  // Run in reverse (LIFO) order
  for (auto it = handlers.rbegin(); it != handlers.rend(); ++it) {
    it->destructor(it->arg);
  }
}

void ORCJITExecutionSessionObj::RegisterDsoHandle(void* dso_handle) {
  std::lock_guard<std::mutex> lock(g_dso_session_mutex);
  g_dso_to_session[dso_handle] = this;
}

void ORCJITExecutionSessionObj::UnregisterDsoHandle(void* dso_handle) {
  std::lock_guard<std::mutex> lock(g_dso_session_mutex);
  g_dso_to_session.erase(dso_handle);
}

}  // namespace orcjit
}  // namespace ffi
}  // namespace tvm
