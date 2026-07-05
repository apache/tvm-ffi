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
 * \file init_fini_plugin.h
 * \brief Init/fini section handling for ELF, MachO, and COFF JIT objects.
 *
 * Emulates the missing/broken/bypassed LLVM ORC platform support for
 * init/fini sections on all three host platforms.  Collects function
 * pointers from `.init_array` / `.fini_array` / `.ctors` / `.dtors`
 * (ELF), `__DATA,__mod_init_func` / `__DATA,__mod_term_func` (MachO),
 * and `.CRT$XC*` / `.CRT$XT*` (COFF); ties them to the containing
 * `JITDylib`; and runs them in priority order through
 * `ORCJITExecutionSessionObj::Run{Pending{Init,De}initializers}`.
 *
 * On Windows the plugin additionally patches `__ImageBase` (set to the
 * lowest block address so `IMAGE_REL_AMD64_ADDR32NB` fixups don't
 * overflow) and strips `.pdata` / `.xdata` edges (SEH handlers live in
 * DLLs > 4 GB from `__ImageBase` and are never registered with
 * `RtlAddFunctionTable` anyway).  Those pieces also disappear once
 * `COFFPlatform` becomes usable.
 *
 * Trigger: any JIT module on any platform containing constructors,
 *          destructors, or `__attribute__((constructor))` /
 *          MSVC `#pragma init_seg` equivalents.
 * Symptom without the patch:
 *   - ELF: constructors/destructors never run (`ELFNixPlatform`
 *     enumerates but does not invoke them before
 *     llvm/llvm-project#175981).
 *   - MachO: we skip `MachOPlatform` entirely to sidestep the
 *     compact-unwind 32-bit delta bug (see `orcjit_session.cc`), so no
 *     platform runs init/fini; this plugin is the only mechanism.
 *   - COFF: relocation overflow / unresolved-SEH crashes
 *     (`COFFPlatform` is not hooked up because its MSVC CRT symbol
 *     requirements cannot be satisfied).
 *
 * ## Removal — Linux
 *
 * LLVM issue: https://github.com/llvm/llvm-project/issues/175981
 * When the upstream fix lands and the project's minimum LLVM version
 * bumps past the first release containing it, replace this plugin's
 * Linux usage with `ELFNixPlatform` and delete the ELF handling path
 * from this file.  Concretely:
 *   - Remove the ELF-section branches (`.init_array`, `.ctors`,
 *     `.fini_array`, `.dtors`) from `InitFiniPlugin::modifyPassConfig`.
 *   - If no platform still needs this plugin, delete this file outright
 *     and follow the checklist in `llvm_patches/README.md`.
 *
 * ## Removal — macOS
 *
 * Tied to re-enabling `MachOPlatform`.  That requires the compact-unwind
 * per-graph `dso_base` fix (see `fix-machoplatform-libunwind-dso-base.patch`
 * in the repo root) to land in our LLVM.  Until then we skip MachOPlatform
 * and this plugin handles `__mod_init_func` / `__mod_term_func`.  When
 * MachOPlatform is re-enabled, delete the MachO-section branches from
 * `InitFiniPlugin::modifyPassConfig` and drop the macOS side of the
 * `addPlugin` call in `orcjit_session.cc`.
 *
 * ## Removal — Windows
 *
 * LLVM status: `COFFPlatform` has been stalled for 2+ years because it
 * requires MSVC CRT symbols (`_CxxThrowException`, RTTI vtables, ...)
 * that LLVM's COFF ORC runtime cannot provide.
 * When `COFFPlatform` becomes usable end-to-end with clang-cl / MSVC
 * objects, replace this plugin's Windows usage with it and delete the
 * COFF handling path (including the `__ImageBase` fixup and the
 * `.pdata` / `.xdata` edge stripping).
 */
#ifndef TVM_FFI_ORCJIT_LLVM_PATCHES_INIT_FINI_PLUGIN_H_
#define TVM_FFI_ORCJIT_LLVM_PATCHES_INIT_FINI_PLUGIN_H_

#include <llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h>

#include "../orcjit_session.h"

namespace tvm {
namespace ffi {
namespace orcjit {

/*! \brief Init/fini section collector and runner for ELF, MachO, and COFF.
 *
 * See the file-level docstring above for the three-platform strategy
 * and the removal procedure for each platform.
 */
class InitFiniPlugin : public llvm::orc::ObjectLinkingLayer::Plugin {
  // Store a raw pointer to avoid a reference cycle:
  //   Session → LLJIT → ObjectLinkingLayer → Plugin → Session
  // The plugin's lifetime is bounded by the ObjectLinkingLayer which is
  // owned by LLJIT which is owned by the session, so the pointer is always valid.
  ORCJITExecutionSessionObj* session_;

 public:
  explicit InitFiniPlugin(ORCJITExecutionSessionObj* session) : session_(session) {}

  void modifyPassConfig(llvm::orc::MaterializationResponsibility& MR, llvm::jitlink::LinkGraph& G,
                        llvm::jitlink::PassConfiguration& Config) override;
  llvm::Error notifyEmitted(llvm::orc::MaterializationResponsibility& MR) override;
  llvm::Error notifyFailed(llvm::orc::MaterializationResponsibility& MR) override;
  llvm::Error notifyRemovingResources(llvm::orc::JITDylib& JD, llvm::orc::ResourceKey K) override;
  void notifyTransferringResources(llvm::orc::JITDylib& JD, llvm::orc::ResourceKey DstKey,
                                   llvm::orc::ResourceKey SrcKey) override;
};

}  // namespace orcjit
}  // namespace ffi
}  // namespace tvm

#endif  // TVM_FFI_ORCJIT_LLVM_PATCHES_INIT_FINI_PLUGIN_H_
