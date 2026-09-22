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

#include <llvm/ADT/ScopeExit.h>
#include <llvm/ExecutionEngine/Orc/EPCDynamicLibrarySearchGenerator.h>
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Process.h>
#include <llvm/Support/TargetSelect.h>
#include <tvm/ffi/cast.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/object.h>
#include <tvm/ffi/reflection/registry.h>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <sstream>

#if defined(__linux__) && defined(__GLIBCXX__)
#include <bits/functexcept.h>
#endif

#if defined(__linux__) || defined(__APPLE__)
#include <dlfcn.h>
#endif

#include "orcjit_dylib.h"
#include "orcjit_memory_manager.h"
#include "orcjit_utils.h"

#if defined(__APPLE__) || defined(_WIN32)
#include "llvm_patches/init_fini_plugin.h"
#endif
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

#ifdef TVM_FFI_ORCJIT_EMBED_ORC_RT
// liborc_rt.a embedded in .rodata by orc_rt_embed.S.in. C linkage binds to the
// assembler symbols; the linkage spec must be at namespace scope, not in-fn.
extern "C" const char orc_rt_archive_start[];
extern "C" const char orc_rt_archive_end[];
#endif
namespace {
#ifdef TVM_FFI_ORCJIT_EMBED_ORC_RT
// Zero-copy view of the embedded archive; its bytes live for the image lifetime.
// Size via uintptr_t: the two symbols are distinct objects, so subtracting the
// pointers directly would be UB.
std::unique_ptr<llvm::MemoryBuffer> GetEmbeddedOrcRuntimeBuffer() {
  return llvm::MemoryBuffer::getMemBuffer(
      llvm::StringRef(orc_rt_archive_start,
                      reinterpret_cast<std::uintptr_t>(orc_rt_archive_end) -
                          reinterpret_cast<std::uintptr_t>(orc_rt_archive_start)),
      "liborc_rt.a", /*RequiresNullTerminator=*/false);
}
#endif

#if (defined(__linux__) && defined(__GLIBCXX__)) || defined(__APPLE__)
const char* GetAddonCxxRuntimeName() {
#if defined(__APPLE__)
  return "/usr/lib/libc++.1.dylib";
#else
  return "libstdc++.so.6";
#endif
}

#if defined(__linux__) && defined(__GLIBCXX__)
void* GetCxxRuntimeHandle(llvm::StringRef runtime_path) {
  // Keep one process-lifetime local handle per compiler-selected runtime. This
  // avoids incrementing the dlopen reference count on every load_module call.
  // Leak the cache deliberately: destroying C++ objects from this DSO during
  // process finalization can run after the C++ runtime has begun teardown.
  struct HandleCache {
    std::mutex mutex;
    std::unordered_map<std::string, void*> handles;
  };
  static auto* cache = new HandleCache();
  std::lock_guard<std::mutex> lock(cache->mutex);
  std::string path = runtime_path.str();
  auto it = cache->handles.find(path);
  if (it != cache->handles.end()) return it->second;
  void* handle = dlopen(path.c_str(), RTLD_LAZY | RTLD_LOCAL);
  if (handle) cache->handles.emplace(std::move(path), handle);
  return handle;
}

class LibStdCxxNonsharedGenerator final : public llvm::orc::DefinitionGenerator {
 public:
  LibStdCxxNonsharedGenerator(
      void* shared_handle,
      std::unique_ptr<llvm::orc::StaticLibraryDefinitionGenerator> archive_generator)
      : shared_handle_(shared_handle), archive_generator_(std::move(archive_generator)) {}

  llvm::Error tryToGenerate(llvm::orc::LookupState& lookup_state, llvm::orc::LookupKind kind,
                            llvm::orc::JITDylib& jit_dylib,
                            llvm::orc::JITDylibLookupFlags jit_dylib_lookup_flags,
                            const llvm::orc::SymbolLookupSet& symbols) override {
    llvm::orc::SymbolLookupSet missing_from_shared;
    for (const auto& [name, lookup_flags] : symbols) {
      std::string symbol_name = (*name).str();
      if (!dlsym(shared_handle_, symbol_name.c_str())) {
        missing_from_shared.add(name, lookup_flags);
      }
    }
    if (missing_from_shared.empty()) return llvm::Error::success();
    return archive_generator_->tryToGenerate(lookup_state, kind, jit_dylib, jit_dylib_lookup_flags,
                                             missing_from_shared);
  }

 private:
  void* shared_handle_;
  std::unique_ptr<llvm::orc::StaticLibraryDefinitionGenerator> archive_generator_;
};
#endif
#endif

// Install ExecutorNativePlatform per the `orc_rt` selector (see the ctor doc).
// A no-op off Linux/ELF: those targets retain LLJIT's generic platform support
// and use the local lifecycle adapter below.
bool SetUpOrcPlatform(llvm::orc::LLJITBuilder& builder,
                      const Optional<Variant<String, Bytes>>& orc_rt) {
#if defined(__APPLE__) || defined(_WIN32)
  (void)builder;
  (void)orc_rt;
  return false;
#else
  if (!orc_rt.has_value()) {
    // Explicitly honor `None`: LLJIT otherwise installs its generic platform.
    builder.setPlatformSetUp(llvm::orc::setUpInactivePlatform);
    return false;
  }
  const Variant<String, Bytes>& sel = orc_rt.value();
  if (auto opt_path = sel.as<String>()) {
    const String& path = *opt_path;
    if (path.empty()) {  // "auto" -> embedded
#ifdef TVM_FFI_ORCJIT_EMBED_ORC_RT
      builder.setPlatformSetUp(llvm::orc::ExecutorNativePlatform(GetEmbeddedOrcRuntimeBuffer()));
      return true;
#else
      return false;
#endif
    } else {
      builder.setPlatformSetUp(llvm::orc::ExecutorNativePlatform(path.operator std::string()));
      return true;
    }
  } else {  // Bytes: ExecutorNativePlatform takes ownership of the copy.
    const Bytes& bytes = sel.get<Bytes>();
    builder.setPlatformSetUp(llvm::orc::ExecutorNativePlatform(llvm::MemoryBuffer::getMemBufferCopy(
        llvm::StringRef(bytes.data(), bytes.size()), "liborc_rt.a")));
    return true;
  }
#endif
}
}  // namespace

ORCJITExecutionSessionObj::ORCJITExecutionSessionObj(const Optional<Variant<String, Bytes>>& orc_rt,
                                                     int64_t slab_size_bytes)
    : jit_(nullptr) {
  // Create slab-backed memory manager — pre-reserves a contiguous VA region
  // so all JIT allocations stay within PC-relative relocation range (±2 GB
  // x86_64, ±4 GB AArch64).  Eliminates scattered-mmap relocation overflow
  // (LLVM #173269).
  //
  // slab_size_bytes: 0 = default (64 MB, with fallback),
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
  // Linux (see LLJITBuilderState::prepareForConstruction). We replace its
  // memory manager with our slab pool. macOS/Windows are gated off pending
  // testing. (The historical "MachOPlatform teardown crashes
  // with the arena" concern is moot now that we skip MachOPlatform below,
  // but enabling the slab on macOS still needs a validation pass.)
#ifdef __linux__
  if (slab_size_bytes >= 0) {
    auto page_size = llvm::sys::Process::getPageSizeEstimate();
    size_t slab_size;
    if (slab_size_bytes > 0) {
      constexpr std::size_t kMinCustomSlabSize = 2 * Slab::kCommitGranularity;
      TVM_FFI_CHECK(static_cast<uint64_t>(slab_size_bytes) >= kMinCustomSlabSize, ValueError)
          << "slab_size must be 0, negative (disabled), or at least " << kMinCustomSlabSize
          << " bytes, but got " << slab_size_bytes;
      slab_size = static_cast<size_t>(slab_size_bytes);
    } else {
      slab_size = SlabPoolMemoryManager::kDefaultSlabSize;
    }
    pending_memory_manager_ = std::make_unique<SlabPoolMemoryManager>(page_size, slab_size);
    memory_manager_ = pending_memory_manager_.get();
  }
#endif

  auto setup_builder = [this](llvm::orc::LLJITBuilder& builder) {
#ifdef __linux__
    if (memory_manager_) {
      builder.setMemoryManagerCreator(
          [this](llvm::orc::ExecutionSession&)
              -> llvm::Expected<std::unique_ptr<llvm::jitlink::JITLinkMemoryManager>> {
            return std::move(pending_memory_manager_);
          });
      builder.setObjectLinkingLayerCreator(
          [](llvm::orc::ExecutionSession& ES, llvm::jitlink::JITLinkMemoryManager& memory_manager)
              -> llvm::Expected<std::unique_ptr<llvm::orc::ObjectLayer>> {
            return std::make_unique<llvm::orc::ObjectLinkingLayer>(ES, memory_manager);
          });
    }  // if (memory_manager_)
#elif defined(__APPLE__) || defined(_WIN32)
    // Force ObjectLinkingLayer (JITLink) so we can attach InitFiniPlugin.
    // macOS: LLJIT already defaults to JITLink for Darwin, but the explicit
    // creator keeps the static_cast in the addPlugin site below type-safe.
    // Windows: LLJIT defaults to RTDyld; we need JITLink for InitFiniPlugin
    // and DLLImportDefinitionGenerator.
    builder.setObjectLinkingLayerCreator(
        [](llvm::orc::ExecutionSession& ES, llvm::jitlink::JITLinkMemoryManager& memory_manager)
            -> llvm::Expected<std::unique_ptr<llvm::orc::ObjectLayer>> {
          return std::make_unique<llvm::orc::ObjectLinkingLayer>(ES, memory_manager);
        });
#endif
#if (defined(__linux__) && defined(__GLIBCXX__)) || defined(__APPLE__)
    builder.setPrePlatformSetup([](llvm::orc::LLJIT& J) -> llvm::Error {
      auto process_symbols = J.getProcessSymbolsJITDylib();
      if (!process_symbols) {
        return llvm::make_error<llvm::StringError>(
            "C++ runtime support requires a process symbols JITDylib",
            llvm::inconvertibleErrorCode());
      }

      // The addon itself is RTLD_LOCAL, so the default process generator
      // cannot see its C++ runtime dependency. Search that dependency through
      // a private handle instead of promoting the addon (and its statically
      // linked LLVM) into the process-global namespace.
      auto cxx_runtime_generator = llvm::orc::EPCDynamicLibrarySearchGenerator::Load(
          J.getExecutionSession(), J.getDylibMgr(), GetAddonCxxRuntimeName());
      if (!cxx_runtime_generator) return cxx_runtime_generator.takeError();
      process_symbols->addGenerator(std::move(*cxx_runtime_generator));

#if defined(__linux__) && defined(__GLIBCXX__)
      // GCC's libstdc++.so linker script may satisfy this helper from
      // libstdc++_nonshared.a. Keep that definition local to this DSO, but make
      // its address available to ORC before liborc_rt is bootstrapped. Adding
      // an archive generator here is unsafe: materializing an archive member
      // during bootstrap tries to register its sections before the ELF runtime
      // has initialized.
      llvm::orc::SymbolMap symbols;
      symbols[J.mangleAndIntern("_ZSt28__throw_bad_array_new_lengthv")] =
          llvm::orc::ExecutorSymbolDef::fromPtr(&std::__throw_bad_array_new_length,
                                                llvm::JITSymbolFlags::Exported);
      return process_symbols->define(llvm::orc::absoluteSymbols(std::move(symbols)));
#else
      return llvm::Error::success();
#endif
    });
#elif defined(_WIN32)
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
  // Configure the native ORC platform from `orc_rt` (a no-op off Linux/ELF;
  // see SetUpOrcPlatform). macOS must skip ExecutorNativePlatform /
  // MachOPlatform to sidestep the compact-unwind 32-bit-delta bug in JITLink's
  // CompactUnwindSupport (personality delta against a per-JITDylib header base
  // wraps `uint64_t` and fails `isUInt<32>` when a later user graph mmaps below
  // the header).
  // InitFiniPlugin below handles __mod_init_func / __mod_term_func; LLJIT's
  // generic platform still registers unwind information for JIT frames.
  has_orc_platform_ = SetUpOrcPlatform(builder, orc_rt);
  setup_builder(builder);
  jit_ = TVM_FFI_ORCJIT_LLVM_CALL(builder.create());
#ifdef _WIN32
  // Strip .pdata/.xdata relocations from COFF objects before JITLink graph
  // building.  See llvm_patches/win_coff_pdata_strip.h for the rationale.
  jit_->getObjTransformLayer().setTransform(&StripCoffPdataXdata);
#endif
#if defined(__APPLE__) || defined(_WIN32)
  // Mach-O and COFF still need the local lifecycle adapter. Linux uses LLVM
  // 23's ELFNixPlatform initialize/deinitialize path instead.
  auto& objlayer = jit_->getObjLinkingLayer();
  static_cast<llvm::orc::ObjectLinkingLayer&>(objlayer).addPlugin(
      std::make_unique<InitFiniPlugin>(this));
#endif
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

ORCJITExecutionSession::ORCJITExecutionSession(const Optional<Variant<String, Bytes>>& orc_rt,
                                               int64_t slab_size_bytes) {
  ObjectPtr<ORCJITExecutionSessionObj> obj =
      make_object<ORCJITExecutionSessionObj>(orc_rt, slab_size_bytes);
  data_ = std::move(obj);
}

ORCJITExecutionSession ORCJITExecutionSessionObj::GlobalDefault() {
  // Leaked, never-destroyed: the owned LLJIT and slab arena must not be torn
  // down during interpreter finalization, where teardown could call back into
  // the host language. Always "auto" (empty String) and not user-configurable
  // (see the header) — the embedded runtime's .rodata outlives it trivially.
  static ORCJITExecutionSession* inst =
      new ORCJITExecutionSession(Variant<String, Bytes>(String("")), 0);
  return *inst;
}

ORCJITDynamicLibrary ORCJITExecutionSessionObj::CreateDynamicLibrary(
    const String& name, const Optional<String>& cxx_runtime_path,
    const Optional<String>& libstdcxx_nonshared_path) {
  TVM_FFI_CHECK(jit_ != nullptr, InternalError) << "ExecutionSession not initialized";

#if defined(__linux__) && defined(__GLIBCXX__)
  void* cxx_runtime = nullptr;
  if (cxx_runtime_path.has_value()) {
    const String& runtime_path = cxx_runtime_path.value();
    cxx_runtime = GetCxxRuntimeHandle(llvm::StringRef(runtime_path.data(), runtime_path.size()));
    if (!cxx_runtime) {
      const char* error = dlerror();
      TVM_FFI_THROW(RuntimeError) << "Failed to open the JIT compiler's C++ runtime "
                                  << cxx_runtime_path.value() << ": "
                                  << (error ? error : "unknown dlopen error");
    }
  }
#else
  (void)cxx_runtime_path;
  (void)libstdcxx_nonshared_path;
#endif

  // Compound topology op — serialize against concurrent create / add / lookup /
  // teardown on this shared session (lock order: session lock first).
  std::lock_guard<std::mutex> lock(mutex_);

  // Generate name if not provided
  String lib_name = name;
  if (lib_name.empty()) {
    std::ostringstream oss;
    oss << "dylib_" << dylib_counter_++;
    lib_name = oss.str();
  }

  llvm::orc::JITDylib& jit_dylib =
      TVM_FFI_ORCJIT_LLVM_CALL(jit_->getExecutionSession().createJITDylib(lib_name.c_str()));
  // If any subsequent link-order/generator setup fails, remove the partially
  // configured dylib while the session mutex is still held.
  llvm::scope_exit cleanup_dylib([this, &jit_dylib]() { RemoveDylib(&jit_dylib); });
#if defined(__linux__) && defined(__GLIBCXX__)
  llvm::orc::JITDylib* cxx_runtime_dylib = nullptr;
  if (cxx_runtime) {
    std::string runtime_path = cxx_runtime_path.value();
    auto it = cxx_runtime_dylibs_.find(runtime_path);
    if (it == cxx_runtime_dylibs_.end()) {
      std::string runtime_name = "<C++ runtime " + std::to_string(cxx_runtime_dylibs_.size()) + ">";
      auto& runtime_dylib = jit_->getExecutionSession().createBareJITDylib(std::move(runtime_name));
      auto runtime_generator =
          TVM_FFI_ORCJIT_LLVM_CALL(llvm::orc::EPCDynamicLibrarySearchGenerator::Load(
              jit_->getExecutionSession(), jit_->getDylibMgr(), runtime_path.c_str()));
      runtime_dylib.addGenerator(std::move(runtime_generator));
      it = cxx_runtime_dylibs_.emplace(std::move(runtime_path), &runtime_dylib).first;
    }
    cxx_runtime_dylib = it->second;
  }
#endif

  // Start from LLJIT's default link order (Main → Platform →
  // ProcessSymbols). On Linux, insert the compiler-selected C++ runtime after
  // Platform and before ProcessSymbols. This keeps the platform's
  // __cxa_atexit interposer ahead of libstdc++ while resolving other C++
  // symbols from the same toolchain that produced the object.
#if defined(__linux__) && defined(__GLIBCXX__)
  auto process_symbols = jit_->getProcessSymbolsJITDylib();
  bool added_cxx_runtime = false;
#endif
  for (auto& kv : jit_->defaultLinkOrder()) {
#if defined(__linux__) && defined(__GLIBCXX__)
    if (cxx_runtime_dylib && process_symbols && kv.first == process_symbols.get()) {
      jit_dylib.addToLinkOrder(*cxx_runtime_dylib);
      added_cxx_runtime = true;
    }
#endif
    jit_dylib.addToLinkOrder(*kv.first, kv.second);
  }
#if defined(__linux__) && defined(__GLIBCXX__)
  if (cxx_runtime_dylib && !added_cxx_runtime) jit_dylib.addToLinkOrder(*cxx_runtime_dylib);

  // Model GCC's libstdc++.so linker script without making either library
  // process-global: only archive symbols absent from the shared library may
  // materialize, and their sections belong to this user JITDylib's lifetime.
  if (cxx_runtime && libstdcxx_nonshared_path.has_value()) {
    const String& archive_path = libstdcxx_nonshared_path.value();
    auto archive_generator =
        TVM_FFI_ORCJIT_LLVM_CALL(llvm::orc::StaticLibraryDefinitionGenerator::Load(
            jit_->getObjLinkingLayer(), archive_path.c_str()));
    jit_dylib.addGenerator(
        std::make_unique<LibStdCxxNonsharedGenerator>(cxx_runtime, std::move(archive_generator)));
  }
#endif

#ifdef __APPLE__
  // Inject ___cxa_atexit on the user JITDylib so it wins over the generic
  // platform's libSystem fallback, which would orphan dtors from our drop-time
  // drain. See llvm_patches/macho_cxa_atexit_shim.h.
  TVM_FFI_ORCJIT_LLVM_CALL(InstallCxaAtexitShim(jit_->getExecutionSession(), jit_dylib));
#endif

  auto dylib_obj = make_object<ORCJITDynamicLibraryObj>(GetRef<ORCJITExecutionSession>(this),
                                                        &jit_dylib, jit_.get(), lib_name);
  cleanup_dylib.release();
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

using CtorDtor = void (*)();

namespace {
// Remove a dylib's entries from `pending` and return them sorted into run order
// (by section, then priority). Caller holds the session lock.
std::vector<ORCJITExecutionSessionObj::InitFiniEntry> DrainSorted(
    std::unordered_map<llvm::orc::JITDylib*, std::vector<ORCJITExecutionSessionObj::InitFiniEntry>>&
        pending,
    llvm::orc::JITDylib& jit_dylib) {
  std::vector<ORCJITExecutionSessionObj::InitFiniEntry> entries;
  auto it = pending.find(&jit_dylib);
  if (it != pending.end()) {
    entries = std::move(it->second);
    pending.erase(it);
    llvm::sort(entries, [](const ORCJITExecutionSessionObj::InitFiniEntry& a,
                           const ORCJITExecutionSessionObj::InitFiniEntry& b) {
      return a.priority < b.priority;
    });
  }
  return entries;
}
}  // namespace

std::vector<ORCJITExecutionSessionObj::InitFiniEntry>
ORCJITExecutionSessionObj::DrainPendingInitializers(llvm::orc::JITDylib& jit_dylib) {
  return DrainSorted(pending_initializers_, jit_dylib);
}

std::vector<ORCJITExecutionSessionObj::InitFiniEntry>
ORCJITExecutionSessionObj::DrainPendingDeinitializers(llvm::orc::JITDylib& jit_dylib) {
  return DrainSorted(pending_deinitializers_, jit_dylib);
}

void ORCJITExecutionSessionObj::RunInitFiniEntries(const std::vector<InitFiniEntry>& entries) {
  for (const auto& entry : entries) {
    entry.address.toPtr<CtorDtor>()();
  }
}

void ORCJITExecutionSessionObj::AddPendingInitializer(llvm::orc::JITDylib* jit_dylib,
                                                      const InitFiniEntry& entry) {
  pending_initializers_[jit_dylib].push_back(entry);
}

void ORCJITExecutionSessionObj::AddPendingDeinitializer(llvm::orc::JITDylib* jit_dylib,
                                                        const InitFiniEntry& entry) {
  pending_deinitializers_[jit_dylib].push_back(entry);
}

int64_t ORCJITExecutionSessionObj::ClearFreeSlabs() {
#ifdef __linux__
  if (memory_manager_) {
    // Synchronize with lookup/materialization and dylib teardown. This makes
    // the public API safe even when another host thread is using the session.
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<int64_t>(memory_manager_->clearFreeSlabs());
  }
#endif
  return 0;
}

void ORCJITExecutionSessionObj::RemoveDylib(llvm::orc::JITDylib* jit_dylib) {
  if (jit_dylib == nullptr) return;
  // Drop any pending init/fini records keyed by this JITDylib*. After removal
  // the address may be recycled for a freshly-created JITDylib; leftover
  // entries would then be attributed to the wrong dylib.
  pending_initializers_.erase(jit_dylib);
  pending_deinitializers_.erase(jit_dylib);

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
