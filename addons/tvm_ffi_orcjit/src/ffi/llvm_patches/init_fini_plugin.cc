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
 * \file init_fini_plugin.cc
 * \brief Init/fini section handling for ELF and COFF JIT objects.
 *
 * See init_fini_plugin.h for the trigger, symptom, and removal
 * procedure for each platform.
 */
#include "init_fini_plugin.h"

#if defined(__linux__) || defined(_WIN32)

#include <llvm/ExecutionEngine/JITLink/JITLink.h>
#include <llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace tvm {
namespace ffi {
namespace orcjit {

void InitFiniPlugin::modifyPassConfig(llvm::orc::MaterializationResponsibility& MR,
                                      llvm::jitlink::LinkGraph& G,
                                      llvm::jitlink::PassConfiguration& Config) {
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

llvm::Error InitFiniPlugin::notifyFailed(llvm::orc::MaterializationResponsibility& MR) {
  return llvm::Error::success();
}

llvm::Error InitFiniPlugin::notifyRemovingResources(llvm::orc::JITDylib& JD,
                                                    llvm::orc::ResourceKey K) {
  return llvm::Error::success();
}

void InitFiniPlugin::notifyTransferringResources(llvm::orc::JITDylib& JD,
                                                 llvm::orc::ResourceKey DstKey,
                                                 llvm::orc::ResourceKey SrcKey) {}

}  // namespace orcjit
}  // namespace ffi
}  // namespace tvm

#endif  // __linux__ || _WIN32
