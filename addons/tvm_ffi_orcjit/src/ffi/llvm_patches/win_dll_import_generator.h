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
 * \file win_dll_import_generator.h
 * \brief Windows DLL-import symbol generator for JIT code (COFF/x86_64).
 *
 * On Windows with the MSVC ABI, COFF objects reference DLL-imported
 * symbols via `__imp_XXX` pointer stubs and direct calls.  Without
 * `COFFPlatform` (which we skip because its MSVC CRT dependencies
 * cannot be satisfied — see init_fini_plugin.h for the Windows
 * removal notes), JITLink has no way to resolve those references.
 *
 * `DLLImportDefinitionGenerator` is a `DefinitionGenerator` that, for
 * each requested symbol:
 *   1. Looks up the real address by walking the MSVC runtime DLLs,
 *      `tvm_ffi.dll`, all loaded process modules, and finally LLVM's
 *      `SearchForAddressOfSymbol`.
 *   2. Emits a fresh `LinkGraph` containing a GOT-like `__imp_XXX`
 *      pointer holding the real address, plus a PLT-like jump stub
 *      (`jmp [__imp_XXX]`) so direct calls stay in range.  All stubs
 *      land in JIT memory, keeping every PCRel32 fixup within ±2 GB
 *      of the JIT arena.
 *
 * Trigger: any Windows x86_64 JIT graph that references a DLL-imported
 *          symbol (which, for MSVC-compiled objects, is effectively
 *          any C runtime or libtvm_ffi entry point).
 * Symptom without the generator: unresolved `__imp_XXX` externals
 *          at JITLink time, or PCRel32 overflow for direct DLL calls.
 *
 * ## Removal
 *
 * Tied to enabling `COFFPlatform`.  When `COFFPlatform` becomes usable
 * end-to-end with clang-cl / MSVC objects (blocked upstream on MSVC
 * CRT symbol requirements), delete this file and the corresponding
 * `addGenerator` call in `orcjit_session.cc`.
 */
#ifndef TVM_FFI_ORCJIT_LLVM_PATCHES_WIN_DLL_IMPORT_GENERATOR_H_
#define TVM_FFI_ORCJIT_LLVM_PATCHES_WIN_DLL_IMPORT_GENERATOR_H_

#ifdef _WIN32

#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h>

namespace tvm {
namespace ffi {
namespace orcjit {

/*! \brief JIT-allocated `__imp_XXX` pointer stubs + PLT jumps for DLL imports.
 *
 * See the file-level docstring for the trigger, symptom, and removal
 * procedure.
 */
class DLLImportDefinitionGenerator : public llvm::orc::DefinitionGenerator {
 public:
  DLLImportDefinitionGenerator(llvm::orc::ExecutionSession& ES, llvm::orc::ObjectLinkingLayer& L)
      : ES_(ES), L_(L) {}

  llvm::Error tryToGenerate(llvm::orc::LookupState& LS, llvm::orc::LookupKind K,
                            llvm::orc::JITDylib& JD, llvm::orc::JITDylibLookupFlags JDLookupFlags,
                            const llvm::orc::SymbolLookupSet& LookupSet) override;

 private:
  static void* FindInProcessModules(const std::string& Name);

  llvm::orc::ExecutionSession& ES_;
  llvm::orc::ObjectLinkingLayer& L_;
};

}  // namespace orcjit
}  // namespace ffi
}  // namespace tvm

#endif  // _WIN32

#endif  // TVM_FFI_ORCJIT_LLVM_PATCHES_WIN_DLL_IMPORT_GENERATOR_H_
