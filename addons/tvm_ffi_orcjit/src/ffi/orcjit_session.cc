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

#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/Process.h>
#include <llvm/Support/TargetSelect.h>
#include <tvm/ffi/cast.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/object.h>
#include <tvm/ffi/reflection/registry.h>

#include <cstddef>
#include <iterator>
#include <utility>

#include "orcjit_dylib.h"
#include "orcjit_memory_manager.h"
#include "orcjit_utils.h"

#if defined(__linux__) && (defined(__x86_64__) || defined(_M_X64))
#include "llvm_patches/gotpcrelx_fix.h"
#endif
#include "llvm_patches/init_fini_plugin.h"
#ifdef __APPLE__
#include "llvm_patches/macho_cxa_atexit_shim.h"
#endif
#ifdef _WIN32
#include <llvm/ExecutionEngine/Orc/ObjectTransformLayer.h>

#include "llvm_patches/win_coff_pdata_strip.h"
#include "llvm_patches/win_dll_import_generator.h"
#endif

namespace tvm {
namespace ffi {
namespace orcjit {

// Initialize LLVM native target (only once)
struct LLVMInitializer {
  LLVMInitializer() {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();
  }
};

static LLVMInitializer llvm_initializer;

ORCJITExecutionSessionObj::ORCJITExecutionSessionObj(const std::string& orc_rt_path,
                                                     int64_t slab_size_bytes)
    : jit_(nullptr) {
  // Create slab-backed memory manager — pre-reserves a contiguous VA region
  // so all JIT allocations stay within PC-relative relocation range (±2 GB
  // x86_64, ±4 GB AArch64).  Eliminates scattered-mmap relocation overflow
  // (LLVM #173269).
  //
  // slab_size_bytes: 0 = arch default (1 GB x86_64 / AArch64, with fallback),
  //                  >0 = custom size, <0 = disable arena (LLJIT uses its
  //                  default allocator — scattered mmap, no PC-rel guarantee).
  // The parameter is Linux-only; on macOS/Windows the arena is compiled out
  // entirely (see #ifdef below) and the value is ignored.
  //
  // `slab_size_bytes` is the per-slab capacity for the growable pool.
  // Session memory grows in slab-sized increments; graphs that don't
  // fit a normal slab trigger a power-of-2 larger slab sized to fit
  // (see `Slab::capacityForFootprint`).
  //
  // The default (64 MB) is above typical ML JIT graph sizes while well
  // under the PC-relative relocation limit.  The initial-slab constructor
  // halves its capacity on mmap failure (RLIMIT_AS, containers) down to
  // 8 MB; subsequent slabs are reserved at the size returned by
  // `capacityForFootprint` (>= slab_size) and mmap errors propagate.
  //
  // LLJIT auto-configures ObjectLinkingLayer (JITLink) on x86_64 and aarch64
  // Linux (see LLJITBuilderState::prepareForConstruction).  We override
  // the layer creator to pass our memory manager.  macOS/Windows are gated
  // off pending testing.  (The historical "MachOPlatform teardown crashes
  // with the arena" concern is moot now that we skip MachOPlatform below,
  // but enabling the slab on macOS still needs a validation pass.)
#ifdef __linux__
  if (slab_size_bytes >= 0) {
    auto page_size = llvm::sys::Process::getPageSizeEstimate();
    size_t slab_size;
    if (slab_size_bytes > 0) {
      slab_size = static_cast<size_t>(slab_size_bytes);
    } else {
      slab_size = SlabPoolMemoryManager::kDefaultSlabSize;
    }
    memory_manager_ = std::make_unique<SlabPoolMemoryManager>(page_size, slab_size);
  }
#endif

  auto setup_builder = [this](llvm::orc::LLJITBuilder& builder) {
#ifdef __linux__
    if (memory_manager_) {
      builder.setObjectLinkingLayerCreator(
          [this](llvm::orc::ExecutionSession& ES)
              -> llvm::Expected<std::unique_ptr<llvm::orc::ObjectLayer>> {
            auto OLL = std::make_unique<llvm::orc::ObjectLinkingLayer>(ES, *memory_manager_);
#if defined(__x86_64__) || defined(_M_X64)
            OLL->addPlugin(std::make_unique<GOTPCRELXFixPlugin>());
#endif
            return OLL;
          });
    }  // if (memory_manager_)
#elif defined(__APPLE__) || defined(_WIN32)
    // Force ObjectLinkingLayer (JITLink) so we can attach InitFiniPlugin.
    // macOS: LLJIT already defaults to JITLink for Darwin, but the explicit
    // creator keeps the static_cast in the addPlugin site below type-safe.
    // Windows: LLJIT defaults to RTDyld; we need JITLink for InitFiniPlugin
    // and DLLImportDefinitionGenerator.
    builder.setObjectLinkingLayerCreator(
        [](llvm::orc::ExecutionSession& ES)
            -> llvm::Expected<std::unique_ptr<llvm::orc::ObjectLayer>> {
          return std::make_unique<llvm::orc::ObjectLinkingLayer>(ES);
        });
#endif
#ifdef _WIN32
    // Override ProcessSymbols setup to NOT add the default
    // EPCDynamicLibrarySearchGenerator. That generator resolves symbols to
    // absolute host-process addresses, which causes PCRel32 overflow when
    // JIT code calls into DLLs >2GB away. Our DLLImportDefinitionGenerator
    // (added after construction) wraps every resolved address in a
    // JIT-allocated PLT stub, keeping all fixups in range.
    builder.setProcessSymbolsJITDylibSetup(
        [](llvm::orc::LLJIT& J) -> llvm::Expected<llvm::orc::JITDylibSP> {
          return &J.getExecutionSession().createBareJITDylib("<Process Symbols>");
        });
#endif
    (void)builder;
  };

  auto builder = llvm::orc::LLJITBuilder();
#ifndef __APPLE__
  // macOS: always skip ExecutorNativePlatform / MachOPlatform to sidestep
  // the compact-unwind 32-bit-delta bug in JITLink's CompactUnwindSupport
  // (personality delta against a per-JITDylib header base wraps `uint64_t`
  // and fails `isUInt<32>` when a later user graph mmaps below the header;
  // see the repo-root fix-machoplatform-libunwind-dso-base.patch for the
  // full analysis).  InitFiniPlugin below handles __mod_init_func /
  // __mod_term_func instead.  Tradeoff: no C++ exception unwinding across
  // JIT frames on macOS.
  if (!orc_rt_path.empty()) {
    builder.setPlatformSetUp(llvm::orc::ExecutorNativePlatform(orc_rt_path));
  }
#else
  (void)orc_rt_path;
#endif
  setup_builder(builder);
  jit_ = TVM_FFI_ORCJIT_LLVM_CALL(builder.create());
#ifdef _WIN32
  // Strip .pdata/.xdata relocations from COFF objects before JITLink graph
  // building.  See llvm_patches/win_coff_pdata_strip.h for the rationale.
  jit_->getObjTransformLayer().setTransform(&StripCoffPdataXdata);
#endif
  // Use our custom InitFiniPlugin on every platform for init/fini section
  // collection and priority-ordered execution (ELF .init_array/.fini_array,
  // MachO __mod_init_func/__mod_term_func, COFF .CRT$XC*/.CRT$XT*).  See
  // llvm_patches/init_fini_plugin.h for per-platform removal criteria.
  auto& objlayer = jit_->getObjLinkingLayer();
  static_cast<llvm::orc::ObjectLinkingLayer&>(objlayer).addPlugin(
      std::make_unique<InitFiniPlugin>(this));
#ifdef _WIN32
  // On Windows, the default process-symbol generator only searches the main
  // exe module via GetProcAddress(GetModuleHandle(NULL), ...). Add a
  // comprehensive generator that searches all loaded DLLs (vcruntime140,
  // ucrtbase, tvm_ffi, etc.) and creates __imp_* pointer stubs.
  if (auto PSG = jit_->getProcessSymbolsJITDylib()) {
    auto& ObjLayer = static_cast<llvm::orc::ObjectLinkingLayer&>(jit_->getObjLinkingLayer());
    PSG->addGenerator(
        std::make_unique<DLLImportDefinitionGenerator>(jit_->getExecutionSession(), ObjLayer));
  }
#endif
}

ORCJITExecutionSession::ORCJITExecutionSession(const std::string& orc_rt_path,
                                               int64_t slab_size_bytes) {
  ObjectPtr<ORCJITExecutionSessionObj> obj =
      make_object<ORCJITExecutionSessionObj>(orc_rt_path, slab_size_bytes);
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
      TVM_FFI_ORCJIT_LLVM_CALL(jit_->getExecutionSession().createJITDylib(lib_name.c_str()));
  // Use the LLJIT's default link order (Main → Platform → ProcessSymbols).
  // This provides host process symbols via the ProcessSymbols JITDylib's generator,
  // while ensuring the platform's __cxa_atexit interposer (in PlatformJD) takes
  // precedence — so __cxa_atexit handlers are managed by the platform and can be
  // drained per-JITDylib via __lljit_run_atexits at teardown.
  for (auto& kv : jit_->defaultLinkOrder()) {
    jit_dylib.addToLinkOrder(*kv.first, kv.second);
  }

  auto dylib_obj = make_object<ORCJITDynamicLibraryObj>(GetRef<ORCJITExecutionSession>(this),
                                                        &jit_dylib, jit_.get(), lib_name);
  RegisterDylibOwner(&jit_dylib, dylib_obj.get());

#ifdef __APPLE__
  // Inject ___cxa_atexit on the user JITDylib so it wins over <Platform>'s
  // fallback (which resolves to libSystem's and would orphan dtors from
  // our drop-time drain).  See llvm_patches/macho_cxa_atexit_shim.h.
  InstallCxaAtexitShim(jit_->getExecutionSession(), jit_dylib);
#endif

  return ORCJITDynamicLibrary(std::move(dylib_obj));
}

llvm::orc::ExecutionSession& ORCJITExecutionSessionObj::GetLLVMExecutionSession() {
  TVM_FFI_CHECK(jit_ != nullptr, InternalError) << "ExecutionSession not initialized";
  return jit_->getExecutionSession();
}

llvm::orc::LLJIT& ORCJITExecutionSessionObj::GetLLJIT() {
  TVM_FFI_CHECK(jit_ != nullptr, InternalError) << "ExecutionSession not initialized";
  return *jit_;
}

llvm::Expected<llvm::orc::ExecutorSymbolDef> ORCJITExecutionSessionObj::Lookup(
    llvm::orc::JITDylib& root, const llvm::orc::JITDylibSearchOrder& search_order,
    llvm::orc::SymbolStringPtr symbol) {
  std::lock_guard<std::recursive_mutex> lock(lookup_mutex_);
  auto link_order = root.getReverseDFSLinkOrder();
  if (!link_order) return link_order.takeError();

  uint64_t lookup_id = 0;
  {
    std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
    lookup_id = ++next_lookup_id_;
    active_lookup_ids_.push_back(lookup_id);
  }

  auto result = jit_->getExecutionSession().lookup(search_order, std::move(symbol));
  if (!result) {
    FinishLookup(lookup_id, false, *link_order);
    return result.takeError();
  }
  FinishLookup(lookup_id, true, *link_order);
  RunPendingInitializers(*link_order);
  return std::move(*result);
}

using CtorDtor = void (*)();

bool ORCJITExecutionSessionObj::GetContextSymbols(llvm::orc::JITDylib& jit_dylib,
                                                  std::unordered_map<std::string, void*>* symbols) {
  ORCJITDynamicLibraryObj* owner = nullptr;
  {
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    auto it = dylib_owners_.find(&jit_dylib);
    if (it == dylib_owners_.end()) return false;
    owner = it->second;
  }

  symbols->clear();
  Module::VisitContextSymbols([this, symbols](const String& name, void* value) {
    (*symbols)[(*jit_->mangleAndIntern(name.c_str())).str()] = value;
  });
  (*symbols)[(*jit_->mangleAndIntern(symbol::tvm_ffi_library_ctx)).str()] =
      static_cast<ModuleObj*>(owner);
  return true;
}

llvm::Error ORCJITExecutionSessionObj::StageMaterialization(
    llvm::orc::MaterializationResponsibility& mr, llvm::orc::JITDylib& jit_dylib,
    std::vector<ContextEntry> contexts, std::vector<InitFiniEntry> initializers,
    std::vector<InitFiniEntry> deinitializers) {
  std::lock_guard<std::mutex> lock(lifecycle_mutex_);
  if (active_lookup_ids_.empty()) {
    return llvm::make_error<llvm::StringError>(
        "ORCJIT context materialization occurred outside a serialized lookup",
        llvm::inconvertibleErrorCode());
  }
  auto [it, inserted] = staged_materializations_.emplace(
      &mr, MaterializationRecord{&jit_dylib, active_lookup_ids_.back(), 0, std::move(contexts),
                                 std::move(initializers), std::move(deinitializers)});
  if (!inserted) {
    return llvm::make_error<llvm::StringError>("Duplicate ORCJIT materialization record",
                                               llvm::inconvertibleErrorCode());
  }
  return llvm::Error::success();
}

llvm::Error ORCJITExecutionSessionObj::RecordEmittedMaterialization(
    llvm::orc::MaterializationResponsibility& mr) {
  uint64_t readiness_id = 0;
  uint64_t lookup_id = 0;
  llvm::orc::JITDylib* jit_dylib = nullptr;
  llvm::orc::SymbolLookupSet symbols;
  for (const auto& [name, flags] : mr.getSymbols()) {
    symbols.add(name, flags.hasMaterializationSideEffectsOnly()
                          ? llvm::orc::SymbolLookupFlags::WeaklyReferencedSymbol
                          : llvm::orc::SymbolLookupFlags::RequiredSymbol);
  }
  {
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    auto staged_it = staged_materializations_.find(&mr);
    if (staged_it == staged_materializations_.end()) return llvm::Error::success();

    staged_it->second.readiness_id = ++next_readiness_id_;
    readiness_id = staged_it->second.readiness_id;
    lookup_id = staged_it->second.lookup_id;
    jit_dylib = staged_it->second.jit_dylib;
    emitted_materializations_.push_back(std::move(staged_it->second));
    staged_materializations_.erase(staged_it);
    ++pending_readiness_counts_[lookup_id];
  }

  if (symbols.empty()) {
    CompleteMaterializationReadiness(readiness_id, lookup_id, false);
    return llvm::Error::success();
  }

  llvm::orc::JITDylibSearchOrder search_order;
  search_order.emplace_back(jit_dylib, llvm::orc::JITDylibLookupFlags::MatchAllSymbols);
  jit_->getExecutionSession().lookup(
      llvm::orc::LookupKind::Static, search_order, std::move(symbols),
      llvm::orc::SymbolState::Ready,
      [this, readiness_id, lookup_id](llvm::Expected<llvm::orc::SymbolMap> ready_symbols) {
        bool ready = static_cast<bool>(ready_symbols);
        if (!ready_symbols) llvm::consumeError(ready_symbols.takeError());
        CompleteMaterializationReadiness(readiness_id, lookup_id, ready);
      },
      llvm::orc::NoDependenciesToRegister);
  return llvm::Error::success();
}

void ORCJITExecutionSessionObj::CompleteMaterializationReadiness(uint64_t readiness_id,
                                                                 uint64_t lookup_id, bool ready) {
  std::lock_guard<std::mutex> lock(lifecycle_mutex_);
  for (auto it = emitted_materializations_.begin(); it != emitted_materializations_.end(); ++it) {
    if (it->readiness_id != readiness_id) continue;
    if (ready) ready_materializations_.push_back(std::move(*it));
    emitted_materializations_.erase(it);
    break;
  }
  auto pending_it = pending_readiness_counts_.find(lookup_id);
  if (pending_it != pending_readiness_counts_.end() && pending_it->second != 0) {
    --pending_it->second;
  }
  lifecycle_cv_.notify_all();
}

void ORCJITExecutionSessionObj::FinishLookup(uint64_t lookup_id, bool lookup_succeeded,
                                             const std::vector<llvm::orc::JITDylibSP>& link_order) {
  // Plugin notifyEmitted runs before LLVM records the finalized allocation and
  // before MR.notifyEmitted makes its symbols Ready. Each plugin callback lodges
  // a full-MR Ready barrier; wait for those callbacks so late plugin/resource/MR
  // failures can never publish stale context or initializer addresses. LLJIT
  // intentionally uses its default InPlace dispatcher, so every materializing
  // graph reaches notifyEmitted/notifyFailed before the blocking lookup returns.
  std::unique_lock<std::mutex> lock(lifecycle_mutex_);
  lifecycle_cv_.wait(lock, [this, lookup_id] {
    auto it = pending_readiness_counts_.find(lookup_id);
    return it == pending_readiness_counts_.end() || it->second == 0;
  });
  pending_readiness_counts_.erase(lookup_id);
  for (auto it = staged_materializations_.begin(); it != staged_materializations_.end();) {
    if (it->second.lookup_id == lookup_id) {
      it = staged_materializations_.erase(it);
    } else {
      ++it;
    }
  }

  if (lookup_succeeded) {
    auto is_in_link_order = [&link_order](llvm::orc::JITDylib* jit_dylib) {
      for (const llvm::orc::JITDylibSP& linked : link_order) {
        if (linked.get() == jit_dylib) return true;
      }
      return false;
    };
    for (auto it = ready_materializations_.begin(); it != ready_materializations_.end();) {
      if (!is_in_link_order(it->jit_dylib)) {
        ++it;
        continue;
      }

      for (const ContextEntry& context : it->contexts) {
        void** slot = context.address.toPtr<void**>();
        if (*slot != context.value) *slot = context.value;
      }
      if (!it->initializers.empty()) {
        auto& pending = pending_initializers_[it->jit_dylib];
        pending.insert(pending.end(), std::make_move_iterator(it->initializers.begin()),
                       std::make_move_iterator(it->initializers.end()));
      }
      if (!it->deinitializers.empty()) {
        auto& pending = pending_deinitializers_[it->jit_dylib];
        pending.insert(pending.end(), std::make_move_iterator(it->deinitializers.begin()),
                       std::make_move_iterator(it->deinitializers.end()));
      }
      it = ready_materializations_.erase(it);
    }
  }

  for (auto it = active_lookup_ids_.begin(); it != active_lookup_ids_.end(); ++it) {
    if (*it == lookup_id) {
      active_lookup_ids_.erase(it);
      break;
    }
  }
}

void ORCJITExecutionSessionObj::DiscardMaterialization(
    llvm::orc::MaterializationResponsibility& mr) {
  std::lock_guard<std::mutex> lock(lifecycle_mutex_);
  staged_materializations_.erase(&mr);
}

void ORCJITExecutionSessionObj::RunPendingInitializers(
    const std::vector<llvm::orc::JITDylibSP>& link_order) {
  for (const llvm::orc::JITDylibSP& jit_dylib_ref : link_order) {
    llvm::orc::JITDylib* jit_dylib = jit_dylib_ref.get();
    std::vector<InitFiniEntry> entries;
    ORCJITDynamicLibraryObj* owner = nullptr;
    {
      std::lock_guard<std::mutex> lock(lifecycle_mutex_);
      auto pending_it = pending_initializers_.find(jit_dylib);
      if (pending_it == pending_initializers_.end()) continue;
      entries = std::move(pending_it->second);
      pending_initializers_.erase(pending_it);

      auto owner_it = dylib_owners_.find(jit_dylib);
      if (owner_it != dylib_owners_.end()) owner = owner_it->second;
    }

    llvm::sort(entries, [](const InitFiniEntry& a, const InitFiniEntry& b) {
      if (a.section != b.section) return static_cast<int>(a.section) < static_cast<int>(b.section);
      return a.priority < b.priority;
    });
    auto run_entries = [&entries]() {
      for (const auto& entry : entries) {
        entry.address.toPtr<CtorDtor>()();
      }
    };
#ifdef __APPLE__
    if (owner != nullptr) {
      CxaAtexitRecordsScope scope(&owner->cxa_atexit_records_);
      run_entries();
    } else {
      run_entries();
    }
#else
    (void)owner;
    run_entries();
#endif
  }
}

void ORCJITExecutionSessionObj::RunPendingDeinitializers(llvm::orc::JITDylib& jit_dylib) {
  std::vector<InitFiniEntry> entries;
  {
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    auto it = pending_deinitializers_.find(&jit_dylib);
    if (it == pending_deinitializers_.end()) return;
    entries = std::move(it->second);
    pending_deinitializers_.erase(it);
  }
  llvm::sort(entries, [](const InitFiniEntry& a, const InitFiniEntry& b) {
    if (a.section != b.section) return static_cast<int>(a.section) < static_cast<int>(b.section);
    return a.priority < b.priority;
  });
  for (const auto& entry : entries) {
    entry.address.toPtr<CtorDtor>()();
  }
}

void ORCJITExecutionSessionObj::RegisterDylibOwner(llvm::orc::JITDylib* jit_dylib,
                                                   ORCJITDynamicLibraryObj* owner) {
  std::lock_guard<std::mutex> lock(lifecycle_mutex_);
  dylib_owners_[jit_dylib] = owner;
}

int64_t ORCJITExecutionSessionObj::ClearFreeSlabs() {
#ifdef __linux__
  if (memory_manager_) {
    return static_cast<int64_t>(memory_manager_->clearFreeSlabs());
  }
#endif
  return 0;
}

void ORCJITExecutionSessionObj::RemoveDylib(llvm::orc::JITDylib* jit_dylib) {
  if (jit_dylib == nullptr) return;
  {
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    // Drop every record keyed by this JITDylib before its address can be
    // recycled for a freshly-created library.
    dylib_owners_.erase(jit_dylib);
    pending_initializers_.erase(jit_dylib);
    pending_deinitializers_.erase(jit_dylib);
    for (auto it = staged_materializations_.begin(); it != staged_materializations_.end();) {
      if (it->second.jit_dylib == jit_dylib) {
        it = staged_materializations_.erase(it);
      } else {
        ++it;
      }
    }
    for (auto it = emitted_materializations_.begin(); it != emitted_materializations_.end();) {
      if (it->jit_dylib == jit_dylib) {
        it = emitted_materializations_.erase(it);
      } else {
        ++it;
      }
    }
    for (auto it = ready_materializations_.begin(); it != ready_materializations_.end();) {
      if (it->jit_dylib == jit_dylib) {
        it = ready_materializations_.erase(it);
      } else {
        ++it;
      }
    }
  }

  if (jit_ == nullptr) return;
  // removeJITDylib is best-effort at destruction time: the session may already
  // be tearing down, the platform may report an error during clear(), etc.
  // Swallow errors rather than throwing from a destructor; the session
  // destructor will munmap everything when it runs.
  if (auto err = jit_->getExecutionSession().removeJITDylib(*jit_dylib)) {
    llvm::consumeError(std::move(err));
  }
}

}  // namespace orcjit
}  // namespace ffi
}  // namespace tvm
