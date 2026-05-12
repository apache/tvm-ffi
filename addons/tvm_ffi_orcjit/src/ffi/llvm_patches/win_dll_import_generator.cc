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
 * \file win_dll_import_generator.cc
 * \brief Windows DLL-import symbol generator implementation.
 *
 * See win_dll_import_generator.h for the trigger, symptom, and removal
 * procedure.
 */

#include "win_dll_import_generator.h"

#ifdef _WIN32

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <llvm/ADT/DenseMap.h>
#include <llvm/ExecutionEngine/JITLink/JITLink.h>
#include <llvm/ExecutionEngine/JITLink/x86_64.h>
#include <llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h>
#include <llvm/Support/DynamicLibrary.h>
#include <llvm/Support/Error.h>
#include <llvm/TargetParser/SubtargetFeature.h>

// windows.h must precede psapi.h — psapi.h uses SIZE_T / DWORD typedefs
// defined in windows.h.  Left to its own devices clang-format sorts these
// alphabetically, landing psapi.h first and breaking the MSVC build.
// clang-format off
#include <windows.h>
#include <psapi.h>
// clang-format on

#include <string>

namespace tvm {
namespace ffi {
namespace orcjit {

void* DLLImportDefinitionGenerator::FindInProcessModules(const std::string& Name) {
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

llvm::Error DLLImportDefinitionGenerator::tryToGenerate(
    llvm::orc::LookupState& LS, llvm::orc::LookupKind K, llvm::orc::JITDylib& JD,
    llvm::orc::JITDylibLookupFlags JDLookupFlags, const llvm::orc::SymbolLookupSet& LookupSet) {
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
                        llvm::jitlink::Linkage::Strong, llvm::jitlink::Scope::Default, true, false);
  }
  return L_.add(JD, std::move(G));
}

}  // namespace orcjit
}  // namespace ffi
}  // namespace tvm

#endif  // _WIN32
