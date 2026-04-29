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

#include <llvm/ADT/DenseMap.h>
#include <llvm/ExecutionEngine/JITLink/JITLink.h>
#include <llvm/ExecutionEngine/JITLink/x86_64.h>
#include <llvm/ExecutionEngine/Orc/AbsoluteSymbols.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h>
#include <llvm/ExecutionEngine/Orc/ObjectTransformLayer.h>
#include <llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h>
#include <llvm/ExecutionEngine/Orc/Shared/ExecutorSymbolDef.h>
#include <llvm/Support/DynamicLibrary.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/Process.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/TargetParser/SubtargetFeature.h>
#include <tvm/ffi/cast.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/object.h>
#include <tvm/ffi/reflection/registry.h>

#include <cstddef>
#include <cstring>

#include "orcjit_memory_manager.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <psapi.h>
#include <windows.h>
#endif

#include "orcjit_dylib.h"
#include "orcjit_utils.h"

#if defined(__linux__) && (defined(__x86_64__) || defined(_M_X64))
#include "llvm_patches/gotpcrelx_fix.h"
#endif
#if defined(__linux__) || defined(_WIN32)
#include "llvm_patches/init_fini_plugin.h"
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

#ifdef _WIN32
/*!
 * \brief Custom definition generator for Windows DLL import symbols.
 *
 * On Windows with the MSVC ABI, COFF objects reference DLL-imported symbols
 * via __imp_XXX pointer stubs and direct calls. Without COFFPlatform (which
 * we skip due to MSVC CRT dependency issues), JITLink cannot resolve these.
 *
 * For each resolved symbol, this generator creates a JIT-allocated LinkGraph
 * containing:
 *   - __imp_XXX: a GOT-like pointer entry holding the real address
 *   - XXX: a PLT-like jump stub (`jmp [__imp_XXX]`) for direct calls
 *
 * By allocating stubs in JIT memory (rather than using absoluteSymbols at
 * host-process addresses), all PCRel32 fixups from JIT code reach safely.
 *
 * Symbol search order:
 *   1. Specific MSVC runtime DLLs (vcruntime140, ucrtbase, msvcp140)
 *   2. All loaded process modules (EnumProcessModules)
 *   3. LLVM's SearchForAddressOfSymbol
 */
class DLLImportDefinitionGenerator : public llvm::orc::DefinitionGenerator {
  llvm::orc::ExecutionSession& ES_;
  llvm::orc::ObjectLinkingLayer& L_;

  static void* FindInProcessModules(const std::string& Name) {
    // Try specific runtime DLLs first, then tvm_ffi.dll (loaded by Python),
    // then all process modules, then LLVM's search.
    static const char* kRuntimeDLLs[] = {
        "vcruntime140.dll",
        "vcruntime140_1.dll",
        "ucrtbase.dll",
        "msvcp140.dll",
    };
    // NOTE: We intentionally do not call FreeLibrary() here. These runtime DLLs
    // (vcruntime140, ucrtbase, etc.) are already loaded by the process and will
    // remain loaded for its lifetime. LoadLibraryA merely increments the refcount;
    // the extra refcount is harmless and avoids the overhead of balancing
    // Get/FreeLibrary for every symbol lookup.
    for (const char* dll : kRuntimeDLLs) {
      if (HMODULE hMod = LoadLibraryA(dll)) {
        if (auto addr = GetProcAddress(hMod, Name.c_str())) {
          return reinterpret_cast<void*>(addr);
        }
      }
    }
    // Also check tvm_ffi.dll (host process symbol provider)
    if (HMODULE hTvmFfi = GetModuleHandleA("tvm_ffi.dll")) {
      if (auto addr = GetProcAddress(hTvmFfi, Name.c_str())) {
        return reinterpret_cast<void*>(addr);
      }
    }
    HMODULE hMods[1024];
    DWORD cbNeeded;
    if (EnumProcessModules(GetCurrentProcess(), hMods, sizeof(hMods), &cbNeeded)) {
      DWORD count = cbNeeded / sizeof(HMODULE);
      if (count > 1024) count = 1024;
      for (DWORD i = 0; i < count; ++i) {
        if (auto addr = GetProcAddress(hMods[i], Name.c_str())) {
          return reinterpret_cast<void*>(addr);
        }
      }
    }
    if (void* addr = llvm::sys::DynamicLibrary::SearchForAddressOfSymbol(Name)) {
      return addr;
    }
    return nullptr;
  }

 public:
  DLLImportDefinitionGenerator(llvm::orc::ExecutionSession& ES, llvm::orc::ObjectLinkingLayer& L)
      : ES_(ES), L_(L) {}

  llvm::Error tryToGenerate(llvm::orc::LookupState& LS, llvm::orc::LookupKind K,
                            llvm::orc::JITDylib& JD, llvm::orc::JITDylibLookupFlags JDLookupFlags,
                            const llvm::orc::SymbolLookupSet& LookupSet) override {
    // Step 1: Collect unique base names (strip __imp_ prefix) and resolve addresses.
    llvm::DenseMap<llvm::orc::SymbolStringPtr, llvm::orc::ExecutorAddr> Resolved;
    for (auto& [Name, Flags] : LookupSet) {
      llvm::StringRef NameStr = *Name;
      std::string BaseName =
          NameStr.starts_with("__imp_") ? NameStr.drop_front(6).str() : NameStr.str();
      if (BaseName == "__ImageBase") continue;
      auto InternedBase = ES_.intern(BaseName);
      if (Resolved.count(InternedBase)) continue;
      void* Addr = FindInProcessModules(BaseName);
      if (Addr) {
        Resolved[InternedBase] = llvm::orc::ExecutorAddr::fromPtr(Addr);
      }
    }
    if (Resolved.empty()) return llvm::Error::success();

    // Step 2: Build a LinkGraph with __imp_ pointers and PLT jump stubs.
    auto G = std::make_unique<llvm::jitlink::LinkGraph>(
        "<DLL_IMPORT_STUBS>", ES_.getSymbolStringPool(), ES_.getTargetTriple(),
        llvm::SubtargetFeatures(), llvm::jitlink::getGenericEdgeKindName);
    auto Prot = static_cast<llvm::orc::MemProt>(static_cast<unsigned>(llvm::orc::MemProt::Read) |
                                                static_cast<unsigned>(llvm::orc::MemProt::Exec));
    auto& Sec = G->createSection("__dll_stubs", Prot);

    for (auto& [InternedName, Addr] : Resolved) {
      // Absolute symbol at the real address (local to this graph)
      auto& Target = G->addAbsoluteSymbol(G->intern(("__real_" + *InternedName).str()), Addr,
                                          G->getPointerSize(), llvm::jitlink::Linkage::Strong,
                                          llvm::jitlink::Scope::Local, false);
      // __imp_XXX pointer (GOT-like entry)
      auto& Ptr = llvm::jitlink::x86_64::createAnonymousPointer(*G, Sec, &Target);
      Ptr.setName(G->intern(("__imp_" + *InternedName).str()));
      Ptr.setLinkage(llvm::jitlink::Linkage::Strong);
      Ptr.setScope(llvm::jitlink::Scope::Default);
      // XXX jump stub (PLT-like entry) for direct calls
      auto& StubBlock = llvm::jitlink::x86_64::createPointerJumpStubBlock(*G, Sec, Ptr);
      G->addDefinedSymbol(StubBlock, 0, *InternedName, StubBlock.getSize(),
                          llvm::jitlink::Linkage::Strong, llvm::jitlink::Scope::Default, true,
                          false);
    }
    return L_.add(JD, std::move(G));
  }
};
#endif  // _WIN32

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
  // Stage A of the slab-pool refactor: there is exactly one Slab per session,
  // so `slab_size_bytes` == the whole arena capacity.  Stage B introduces a
  // growable pool where this parameter becomes the per-slab size and total
  // session memory grows in slab-sized increments.
  //
  // The default is sized to cover typical ML JIT workloads while staying
  // well under the PC-relative relocation limit; JITLink's own overflow
  // check fires first if it is exhausted, matching dlopen/ld.so semantics.
  // The constructor halves capacity on mmap failure (RLIMIT_AS, containers)
  // down to 256 MB.
  //
  // LLJIT auto-configures ObjectLinkingLayer (JITLink) on x86_64 and aarch64
  // Linux (see LLJITBuilderState::prepareForConstruction).  We override
  // the layer creator to pass our memory manager.  macOS/Windows are excluded:
  // macOS MachOPlatform teardown crashes with the arena; Windows needs
  // further testing.
#ifdef __linux__
  if (slab_size_bytes >= 0) {
    auto page_size = llvm::sys::Process::getPageSizeEstimate();
    size_t capacity;
    if (slab_size_bytes > 0) {
      capacity = static_cast<size_t>(slab_size_bytes);
    } else {
#if defined(__aarch64__)
      capacity = ArenaJITLinkMemoryManager::kDefaultSlabCapacity_AArch64;
#else
      capacity = ArenaJITLinkMemoryManager::kDefaultSlabCapacity_x86_64;
#endif
    }
    memory_manager_ = std::make_unique<ArenaJITLinkMemoryManager>(page_size, capacity);
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
    // macOS: MachOPlatform (via ExecutorNativePlatform) requires ObjectLinkingLayer.
    // Windows: need ObjectLinkingLayer for InitFiniPlugin and DLLImportDefinitionGenerator.
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

  if (!orc_rt_path.empty()) {
    auto builder = llvm::orc::LLJITBuilder();
    builder.setPlatformSetUp(llvm::orc::ExecutorNativePlatform(orc_rt_path));
    setup_builder(builder);
    jit_ = TVM_FFI_ORCJIT_LLVM_CALL(builder.create());
  } else {
    auto builder = llvm::orc::LLJITBuilder();
    setup_builder(builder);
    jit_ = TVM_FFI_ORCJIT_LLVM_CALL(builder.create());
  }
#ifdef _WIN32
  // Strip .pdata/.xdata relocations from COFF objects before JITLink graph building.
  // clang-cl puts static functions in COMDAT sections, and .pdata SEH unwind data
  // has relocations targeting COMDAT leader symbols. JITLink's COFFLinkGraphBuilder
  // doesn't register COMDAT leaders in its symbol table when the second COMDAT symbol
  // is CLASS_STATIC (not CLASS_EXTERNAL), causing "Could not find symbol" errors.
  // We already strip .pdata/.xdata edges in a PostAllocationPass; this moves the
  // stripping earlier to prevent the graph builder error.
  // Strip .pdata/.xdata relocations using raw COFF binary manipulation.
  // We avoid llvm/Object/COFF.h because windows.h (included transitively by
  // LLJIT.h) defines IMAGE_* macros that conflict with LLVM's COFF enums.
  jit_->getObjTransformLayer().setTransform(
      [](std::unique_ptr<llvm::MemoryBuffer> Buf)
          -> llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>> {
        const char* Data = Buf->getBufferStart();
        size_t Size = Buf->getBufferSize();
        if (Size < 20) return std::move(Buf);

        // Parse COFF header (regular or bigobj format)
        uint16_t w0, w1;
        std::memcpy(&w0, Data, 2);
        std::memcpy(&w1, Data + 2, 2);
        bool bigobj = (w0 == 0 && w1 == 0xFFFF);

        uint16_t machine;
        uint32_t num_sections, ptr_to_symtab, num_symbols;
        size_t sec_hdr_start, sym_entry_size;
        if (bigobj) {
          if (Size < 56) return std::move(Buf);
          std::memcpy(&machine, Data + 6, 2);
          std::memcpy(&num_sections, Data + 44, 4);
          std::memcpy(&ptr_to_symtab, Data + 48, 4);
          std::memcpy(&num_symbols, Data + 52, 4);
          sec_hdr_start = 56;
          sym_entry_size = 20;
        } else {
          machine = w0;
          uint16_t ns, opt_hdr_size;
          std::memcpy(&ns, Data + 2, 2);
          std::memcpy(&opt_hdr_size, Data + 16, 2);
          std::memcpy(&ptr_to_symtab, Data + 8, 4);
          std::memcpy(&num_symbols, Data + 12, 4);
          num_sections = ns;
          sec_hdr_start = 20 + opt_hdr_size;
          sym_entry_size = 18;
        }
        if (machine != 0x8664) return std::move(Buf);

        // String table follows the symbol table
        size_t strtab_start = ptr_to_symtab + static_cast<size_t>(num_symbols) * sym_entry_size;

        // Resolve a section name (inline 8-byte or "/offset" string table ref)
        constexpr size_t kSecHdrSize = 40;
        auto resolve_name = [&](size_t hdr_off) -> llvm::StringRef {
          const char* raw = Data + hdr_off;
          if (raw[0] == '/' && raw[1] >= '0' && raw[1] <= '9') {
            uint32_t offset = 0;
            for (int j = 1; j < 8 && raw[j] >= '0' && raw[j] <= '9'; ++j)
              offset = offset * 10 + (raw[j] - '0');
            size_t pos = strtab_start + offset;
            if (pos < Size) {
              size_t len = 0;
              while (pos + len < Size && Data[pos + len]) ++len;
              return {Data + pos, len};
            }
          }
          size_t len = 0;
          while (len < 8 && raw[len]) ++len;
          return {raw, len};
        };

        // Collect section header offsets needing relocation stripping
        llvm::SmallVector<size_t, 8> strip_offsets;
        for (uint32_t i = 0; i < num_sections; ++i) {
          size_t off = sec_hdr_start + i * kSecHdrSize;
          if (off + kSecHdrSize > Size) break;
          auto name = resolve_name(off);
          if (name.starts_with(".pdata") || name.starts_with(".xdata")) {
            uint16_t num_relocs;
            std::memcpy(&num_relocs, Data + off + 32, 2);
            if (num_relocs > 0) strip_offsets.push_back(off);
          }
        }
        if (strip_offsets.empty()) return std::move(Buf);

        // Create mutable copy, zero out PointerToRelocations and NumberOfRelocations
        llvm::SmallVector<char> MutableBuf(Data, Data + Size);
        for (auto off : strip_offsets) {
          std::memset(&MutableBuf[off + 24], 0, 4);  // PointerToRelocations
          std::memset(&MutableBuf[off + 32], 0, 2);  // NumberOfRelocations
        }
        return llvm::MemoryBuffer::getMemBufferCopy(
            llvm::StringRef(MutableBuf.data(), MutableBuf.size()), Buf->getBufferIdentifier());
      });
#endif
#if defined(__linux__) || defined(_WIN32)
  // Linux/Windows: use our custom InitFiniPlugin for init/fini section
  // collection and priority-ordered execution. See llvm_patches/init_fini_plugin.h
  // for the three-platform init/fini strategy and removal criteria.
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
    llvm::sort(it->second, [](const InitFiniEntry& a, const InitFiniEntry& b) {
      if (a.section != b.section) return static_cast<int>(a.section) < static_cast<int>(b.section);
      return a.priority < b.priority;
    });
    for (const auto& entry : it->second) {
      entry.address.toPtr<CtorDtor>()();
    }
    pending_initializers_.erase(it);
  }
}

void ORCJITExecutionSessionObj::RunPendingDeinitializers(llvm::orc::JITDylib& jit_dylib) {
  auto it = pending_deinitializers_.find(&jit_dylib);
  if (it != pending_deinitializers_.end()) {
    llvm::sort(it->second, [](const InitFiniEntry& a, const InitFiniEntry& b) {
      if (a.section != b.section) return static_cast<int>(a.section) < static_cast<int>(b.section);
      return a.priority < b.priority;
    });
    for (const auto& entry : it->second) {
      entry.address.toPtr<CtorDtor>()();
    }
    pending_deinitializers_.erase(it);
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
