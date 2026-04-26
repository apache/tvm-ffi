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

// Custom ObjectLinkingLayer plugin for init/fini section handling.
//
// Collects function pointers from init/fini sections (.init_array, .fini_array,
// .ctors, .dtors, .CRT$XC*, .CRT$XT*) and runs them in priority order.
//
// Three-platform init/fini strategy:
//
//   macOS:   MachOPlatform (via orc_rt) handles __mod_init_func/__mod_term_func
//            and __cxa_atexit natively. We delegate to jit_->initialize() and
//            deinitialize(). No InitFiniPlugin needed.
//
//   Windows: COFFPlatform is unusable -- it requires MSVC CRT symbols
//            (_CxxThrowException, RTTI vtables, etc.) that LLVM's COFF ORC
//            runtime cannot provide (stalled for 2+ years). Our plugin handles
//            .CRT$XC*/.CRT$XT* sections instead.
//
//   Linux:   ELFNixPlatform does not handle .init_array/.fini_array correctly
//            prior to llvm/llvm-project#175981. Once a new LLVM release includes
//            that patch, we can switch Linux to ELFNixPlatform and remove the
//            plugin for that platform.
//
// The plugin is compiled only on Linux/Windows and can be removed per-platform
// as LLVM's native platform support matures.
class InitFiniPlugin : public llvm::orc::ObjectLinkingLayer::Plugin {
  // Store a raw pointer to avoid a reference cycle:
  //   Session → LLJIT → ObjectLinkingLayer → Plugin → Session
  // The plugin's lifetime is bounded by the ObjectLinkingLayer which is
  // owned by LLJIT which is owned by the session, so the pointer is always valid.
  ORCJITExecutionSessionObj* session_;

 public:
  explicit InitFiniPlugin(ORCJITExecutionSessionObj* session) : session_(session) {}

  void modifyPassConfig(llvm::orc::MaterializationResponsibility& MR, llvm::jitlink::LinkGraph& G,
                        llvm::jitlink::PassConfiguration& Config) override {
    auto& jit_dylib = MR.getTargetJITDylib();
    // Mark all init/fini section blocks and their edge targets as live
    // so they survive dead-stripping.
    Config.PrePrunePasses.emplace_back([](llvm::jitlink::LinkGraph& G) {
      for (auto& Section : G.sections()) {
        auto section_name = Section.getName();
        // ELF: .init_array*, .fini_array*, .ctors*, .dtors*
        // Mach-O: __DATA,__mod_init_func, __DATA,__mod_term_func
        // COFF: .CRT$XC* (ctors), .CRT$XT* (dtors)
        if (section_name.starts_with(".init_array") || section_name.starts_with(".fini_array") ||
            section_name.starts_with(".ctors") || section_name.starts_with(".dtors") ||
            section_name == "__DATA,__mod_init_func" || section_name == "__DATA,__mod_term_func" ||
            section_name.starts_with(".CRT$XC") || section_name.starts_with(".CRT$XT")) {
          for (auto* Block : Section.blocks()) {
            bool has_live_sym = false;
            for (auto* Sym : G.defined_symbols()) {
              if (&Sym->getBlock() == Block) {
                Sym->setLive(true);
                has_live_sym = true;
              }
            }
            // MSVC may emit .CRT$XC* blocks with data but no symbol table
            // entries (static variables in __declspec(allocate) sections).
            // Add an anonymous symbol so the block survives dead-stripping.
            if (!has_live_sym) {
              G.addAnonymousSymbol(*Block, 0, Block->getSize(), false, true).setLive(true);
            }
            for (auto& Edge : Block->edges()) {
              Edge.getTarget().setLive(true);
            }
          }
        }
      }
      return llvm::Error::success();
    });
#ifdef _WIN32
    // Without COFFPlatform, __ImageBase (used by IMAGE_REL_AMD64_ADDR32NB /
    // Pointer32NB relocations) defaults to 0. This causes all Pointer32 fixups
    // to overflow since JIT addresses don't fit in 32 bits.
    //
    // Fix: set __ImageBase to the lowest block address in the graph after
    // allocation. This makes all intra-graph Pointer32NB offsets small.
    //
    // Also strip .pdata/.xdata edges: SEH unwind data references external
    // handlers (e.g., __CxxFrameHandler3) in DLLs that may be >4GB from
    // __ImageBase. Since we don't call RtlAddFunctionTable, SEH data is
    // unused anyway.
    Config.PostAllocationPasses.emplace_back([](llvm::jitlink::LinkGraph& G) {
      // Set __ImageBase to the lowest allocated block address.
      auto ImageBaseName = G.intern("__ImageBase");
      llvm::jitlink::Symbol* ImageBase = nullptr;
      for (auto* Sym : G.external_symbols()) {
        if (Sym->getName() == ImageBaseName) {
          ImageBase = Sym;
          break;
        }
      }
      if (ImageBase) {
        llvm::orc::ExecutorAddr BaseAddr;
        for (auto* B : G.blocks()) {
          if (!BaseAddr || B->getAddress() < BaseAddr) {
            BaseAddr = B->getAddress();
          }
        }
        ImageBase->getAddressable().setAddress(BaseAddr);
      }
      // Strip .pdata/.xdata edges: external handlers may be >4GB from
      // __ImageBase, and we don't register SEH data anyway.
      for (auto& Sec : G.sections()) {
        if (Sec.getName() == ".pdata" || Sec.getName().starts_with(".xdata")) {
          for (auto* B : Sec.blocks()) {
            while (!B->edges_empty()) {
              B->removeEdge(B->edges().begin());
            }
          }
        }
      }
      return llvm::Error::success();
    });
#endif
    // After fixups, read resolved function pointers from all init/fini data sections.
    // Handles ELF (.init_array, .ctors, .fini_array, .dtors),
    // Mach-O (__DATA,__mod_init_func, __DATA,__mod_term_func),
    // and COFF (.CRT$XC*, .CRT$XT*) section conventions.
    Config.PostFixupPasses.emplace_back([this, &jit_dylib](llvm::jitlink::LinkGraph& G) {
      using Entry = ORCJITExecutionSessionObj::InitFiniEntry;
      for (auto& Sec : G.sections()) {
        auto section_name = Sec.getName();

        // --- ELF sections ---
        bool is_init_array = section_name.starts_with(".init_array");
        bool is_ctors = section_name.starts_with(".ctors");
        bool is_fini_array = section_name.starts_with(".fini_array");
        bool is_dtors = section_name.starts_with(".dtors");
        // --- Mach-O sections ---
        bool is_mod_init = (section_name == "__DATA,__mod_init_func");
        bool is_mod_term = (section_name == "__DATA,__mod_term_func");
        // --- COFF sections ---
        bool is_crt_xc = section_name.starts_with(".CRT$XC");
        bool is_crt_xt = section_name.starts_with(".CRT$XT");

        if (!is_init_array && !is_ctors && !is_fini_array && !is_dtors && !is_mod_init &&
            !is_mod_term && !is_crt_xc && !is_crt_xt)
          continue;

        int priority = 0;
        Entry::Section sec;
        bool is_init;
        // ELF default priority for sections without a numeric suffix is 65535.
        // Lower priority numbers run first for .init_array; .fini_array and .ctors
        // negate so that higher-numbered entries run first (reverse order).
        if (is_init_array) {
          if (section_name.consume_front(".init_array.")) {
            section_name.getAsInteger(10, priority);
          } else {
            priority = 65535;
          }
          sec = Entry::Section::kInitArray;
          is_init = true;
        } else if (is_ctors) {
          if (section_name.consume_front(".ctors.") && !section_name.getAsInteger(10, priority)) {
            priority = -priority;
          }
          sec = Entry::Section::kCtors;
          is_init = true;
        } else if (is_fini_array) {
          if (section_name.consume_front(".fini_array.") &&
              !section_name.getAsInteger(10, priority)) {
            priority = -priority;
          } else {
            priority = -65535;
          }
          sec = Entry::Section::kFiniArray;
          is_init = false;
        } else if (is_dtors) {
          if (section_name.consume_front(".dtors.")) {
            section_name.getAsInteger(10, priority);
          }
          sec = Entry::Section::kDtors;
          is_init = false;
        } else if (is_mod_init) {
          // Mach-O __mod_init_func: no priority system, treated as init_array
          sec = Entry::Section::kInitArray;
          is_init = true;
        } else if (is_mod_term) {
          // Mach-O __mod_term_func: no priority system, treated as fini_array
          sec = Entry::Section::kFiniArray;
          is_init = false;
        } else if (is_crt_xc) {
          // COFF .CRT$XC[suffix]: C++ constructors, sorted alphabetically by suffix.
          // Convert suffix to integer priority that preserves alphabetical ordering.
          // E.g., .CRT$XCA → 'A'*100000=6500000, .CRT$XCU → 'U'*100000=8500000,
          //        .CRT$XCT00200 → 'T'*100000+200=8400200
          sec = Entry::Section::kInitArray;
          is_init = true;
          auto suffix = section_name.substr(7);  // after ".CRT$XC"
          if (!suffix.empty()) {
            priority = static_cast<int>(suffix[0]) * 100000;
            if (suffix.size() > 1) {
              int num = 0;
              suffix.substr(1).getAsInteger(10, num);
              priority += num;
            }
          }
        } else {
          // COFF .CRT$XT[suffix]: C++ destructors, same suffix-to-priority scheme.
          sec = Entry::Section::kFiniArray;
          is_init = false;
          auto suffix = section_name.substr(7);  // after ".CRT$XT"
          if (!suffix.empty()) {
            priority = static_cast<int>(suffix[0]) * 100000;
            if (suffix.size() > 1) {
              int num = 0;
              suffix.substr(1).getAsInteger(10, num);
              priority += num;
            }
          }
        }

        for (auto* Block : Sec.blocks()) {
          auto Content = Block->getContent();
          size_t PtrSize = G.getPointerSize();
          for (size_t Offset = 0; Offset + PtrSize <= Content.size(); Offset += PtrSize) {
            uint64_t FnAddr = 0;
            memcpy(&FnAddr, Content.data() + Offset, PtrSize);
            if (FnAddr != 0) {
              Entry entry{llvm::orc::ExecutorAddr(FnAddr), sec, priority};
              if (is_init) {
                session_->AddPendingInitializer(&jit_dylib, entry);
              } else {
                session_->AddPendingDeinitializer(&jit_dylib, entry);
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

#if defined(__linux__) && (defined(__x86_64__) || defined(_M_X64))
/*! \brief Fix LLVM JITLink GOTPCRELX relaxation bug (x86_64).
 *
 * optimizeGOTAndStubAccesses() relaxes `call *foo@GOTPCREL(%rip)` (ff 15)
 * to `addr32 call foo` (67 e8) and sets the edge to Pointer32 (absolute
 * 32-bit).  But `call rel32` is always PC-relative — the CPU computes
 * target = RIP + imm32, not target = imm32.  The Pointer32 fixup writes
 * the absolute address, producing a wrong displacement.
 *
 * This only manifests when external symbols resolve to low addresses
 * (< 4 GB, e.g. PLT entries in a non-PIE executable) while JIT code is
 * at high addresses (the arena at 0x7f...).  The optimization fires
 * because isUInt<32>(target) is true, but the resulting fixup is wrong.
 *
 * The PreFixupPass reverts broken relaxations back to indirect calls
 * through the GOT.  See orcjit_memory_manager.h for full context.
 */
class GOTPCRELXFixPlugin : public llvm::orc::ObjectLinkingLayer::Plugin {
 public:
  void modifyPassConfig(llvm::orc::MaterializationResponsibility& MR, llvm::jitlink::LinkGraph& G,
                        llvm::jitlink::PassConfiguration& Config) override {
    Config.PreFixupPasses.emplace_back(fixBrokenGOTPCRELXRelaxation);
  }
  llvm::Error notifyFailed(llvm::orc::MaterializationResponsibility& MR) override {
    return llvm::Error::success();
  }
  llvm::Error notifyRemovingResources(llvm::orc::JITDylib& JD, llvm::orc::ResourceKey K) override {
    return llvm::Error::success();
  }
  void notifyTransferringResources(llvm::orc::JITDylib& JD, llvm::orc::ResourceKey DstKey,
                                   llvm::orc::ResourceKey SrcKey) override {}

 private:
  /*! \brief Correct broken GOTPCRELX relaxations produced by
   *         optimizeGOTAndStubAccesses().
   *
   * Strategy:
   *  1. Build target-symbol → GOT-entry-symbol map (O(B+S) up front).
   *  2. For every Pointer32 edge whose preceding bytes are 67 e8
   *     (relaxed call) or e9 (relaxed jmp):
   *     - If the target is reachable via a signed 32-bit PC-relative
   *       displacement, change the edge to BranchPCRel32.
   *     - Otherwise revert the relaxation: restore the original
   *       indirect-call/jmp opcode bytes (ff 15 / ff 25), retarget
   *       the edge to the GOT entry, and use PCRel32 with addend 0
   *       (JITLink normalises GOTPCRELX addends to 0).
   */
  static llvm::Error fixBrokenGOTPCRELXRelaxation(llvm::jitlink::LinkGraph& G) {
    using namespace llvm::jitlink;
    // Build block → first symbol at offset 0 (for GOT entry symbol lookup).
    llvm::DenseMap<Block*, Symbol*> BlockToSym;
    for (auto* Sym : G.defined_symbols()) {
      if (Sym->getOffset() == 0 && !BlockToSym.count(&Sym->getBlock())) {
        BlockToSym[&Sym->getBlock()] = Sym;
      }
    }

    // Build target symbol → GOT entry symbol map.
    // GOT entries are pointer-sized blocks with exactly one Pointer64 edge.
    llvm::DenseMap<Symbol*, Symbol*> SymToGOTSym;
    for (auto* B : G.blocks()) {
      if (B->getSize() != G.getPointerSize()) continue;
      if (B->edges_size() != 1) continue;
      auto& E = *B->edges().begin();
      if (E.getKind() == x86_64::Pointer64) {
        auto It = BlockToSym.find(B);
        if (It != BlockToSym.end()) {
          SymToGOTSym[&E.getTarget()] = It->second;
        }
      }
    }

    for (auto* B : G.blocks()) {
      for (auto& E : B->edges()) {
        if (E.getKind() != x86_64::Pointer32) continue;
        if (E.getOffset() < 2) continue;

        auto MutableContent = B->getMutableContent(G);
        auto* FixupData = reinterpret_cast<uint8_t*>(MutableContent.data()) + E.getOffset();
        uint8_t Prev2 = FixupData[-2];
        uint8_t Prev1 = FixupData[-1];

        bool isRelaxedCall = (Prev2 == 0x67 && Prev1 == 0xe8);
        bool isRelaxedJmp = (Prev1 == 0xe9);
        if (!isRelaxedCall && !isRelaxedJmp) continue;

        // Check if PC-relative displacement would fit.
        auto TargetAddr = E.getTarget().getAddress();
        auto FixupAddr = B->getFixupAddress(E);
        int64_t Displacement = TargetAddr.getValue() - (FixupAddr.getValue() + 4) + E.getAddend();
        if (llvm::isInt<32>(Displacement)) {
          E.setKind(x86_64::BranchPCRel32);
          continue;
        }

        // Distance doesn't fit — revert to indirect call/jmp through GOT.
        auto It = SymToGOTSym.find(&E.getTarget());
        if (It == SymToGOTSym.end()) {
          return llvm::make_error<llvm::StringError>(
              "Cannot revert GOTPCRELX relaxation: no GOT entry for " +
                  (E.getTarget().hasName() ? std::string(*E.getTarget().getName())
                                           : std::string("<anon>")),
              llvm::inconvertibleErrorCode());
        }

        Symbol* GOTSym = It->second;
        if (isRelaxedCall) {
          // Restore: 67 e8 → ff 15 (call *[rip+disp32])
          FixupData[-2] = 0xff;
          FixupData[-1] = 0x15;
        } else {
          // Restore: e9 XX XX XX XX 90 → ff 25 XX XX XX XX
          FixupData[-1] = 0xff;
          FixupData[0] = 0x25;
          // For jmp, the optimization shifted offset by -1; shift back.
          E.setOffset(E.getOffset() + 1);
        }
        E.setKind(x86_64::PCRel32);
        E.setTarget(*GOTSym);
        E.setAddend(0);
      }
    }
    return llvm::Error::success();
  }
};
#endif  // __linux__ && __x86_64__

ORCJITExecutionSessionObj::ORCJITExecutionSessionObj(const std::string& orc_rt_path,
                                                     int64_t arena_size_bytes)
    : jit_(nullptr) {
  // Create arena memory manager — pre-reserves contiguous VA region so all
  // JIT allocations stay within PC-relative relocation range (±2GB x86_64,
  // ±4GB AArch64).  Eliminates scattered-mmap relocation overflow (LLVM #173269).
  //
  // arena_size_bytes: 0 = arch default (4GB x86_64, 8GB AArch64, with fallback),
  //                   >0 = custom size, <0 = disable arena.
  // The parameter is Linux-only; on macOS/Windows the arena is compiled out
  // entirely (see #ifdef below) and the value is ignored.
  //
  // The default is strictly larger than the relocation limit so the arena is
  // never the bottleneck — JITLink's own overflow check fires first, matching
  // dlopen/ld.so semantics.  The constructor halves capacity on mmap failure
  // (RLIMIT_AS, containers) down to 256 MB.
  //
  // LLJIT auto-configures ObjectLinkingLayer (JITLink) on x86_64 and aarch64
  // Linux (see LLJITBuilderState::prepareForConstruction).  We override
  // the layer creator to pass our arena MM.  macOS/Windows are excluded:
  // macOS MachOPlatform teardown crashes with the arena; Windows needs
  // further testing.
#ifdef __linux__
  if (arena_size_bytes >= 0) {
    auto page_size = llvm::sys::Process::getPageSizeEstimate();
    size_t capacity;
    if (arena_size_bytes > 0) {
      capacity = static_cast<size_t>(arena_size_bytes);
    } else {
#if defined(__aarch64__)
      capacity = ArenaJITLinkMemoryManager::kDefaultArenaCapacity_AArch64;
#else
      capacity = ArenaJITLinkMemoryManager::kDefaultArenaCapacity_x86_64;
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
  // collection and priority-ordered execution. See InitFiniPlugin class
  // documentation for the three-platform init/fini strategy.
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
                                               int64_t arena_size_bytes) {
  ObjectPtr<ORCJITExecutionSessionObj> obj =
      make_object<ORCJITExecutionSessionObj>(orc_rt_path, arena_size_bytes);
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

}  // namespace orcjit
}  // namespace ffi
}  // namespace tvm
