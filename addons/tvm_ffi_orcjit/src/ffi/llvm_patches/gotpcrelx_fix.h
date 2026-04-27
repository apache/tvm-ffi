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
 * \file gotpcrelx_fix.h
 * \brief LLVM JITLink GOTPCRELX relaxation bug workaround (x86_64).
 *
 * LLVM issue: not yet filed, internal TODO.
 * Affected versions: observed on LLVM 20.x and 21.x; likely all versions
 *                    since `optimizeGOTAndStubAccesses` landed in
 *                    JITLink/x86_64.cpp.
 * Trigger: x86_64 JITLink with external symbols whose resolved addresses
 *          fit in uint32 (e.g. libc PLT entries in a non-PIE executable,
 *          or any low-VA process symbol) while JIT code is at high
 *          addresses (as produced by our arena memory manager).
 * Symptom: `optimizeGOTAndStubAccesses` relaxes
 *            `call *foo@GOTPCREL(%rip)` (ff 15)
 *          into
 *            `addr32 call foo` (67 e8)
 *          and sets the edge kind to `Pointer32` (absolute 32-bit).  But
 *          `call rel32` is always PC-relative, so the absolute fixup
 *          produces a garbage displacement.  Result: SIGSEGV at JIT
 *          execution or during ORC-runtime teardown.
 *
 * The `GOTPCRELXFixPlugin` registers a PreFixupPass that runs *after*
 * `optimizeGOTAndStubAccesses`, detects `Pointer32` edges preceded by
 * `67 e8` / `e9` bytes, and either converts them to `BranchPCRel32` (if
 * the PC-relative displacement fits in int32) or reverts the relaxation
 * to an indirect call/jmp through the GOT (`ff 15` / `ff 25`, edge kind
 * `PCRel32`, addend 0).
 *
 * ## Removal
 *
 * When the upstream fix lands and the project's minimum LLVM version
 * bumps past the first release containing it, delete this file and
 * remove:
 *   - the `#include "llvm_patches/gotpcrelx_fix.h"` in orcjit_session.cc
 *   - the `OLL->addPlugin(std::make_unique<GOTPCRELXFixPlugin>())` call
 *     inside the `setObjectLinkingLayerCreator` lambda in orcjit_session.cc.
 *   - the `llvm_patches/gotpcrelx_fix.cc` entry in
 *     addons/tvm_ffi_orcjit/CMakeLists.txt.
 */
#ifndef TVM_FFI_ORCJIT_LLVM_PATCHES_GOTPCRELX_FIX_H_
#define TVM_FFI_ORCJIT_LLVM_PATCHES_GOTPCRELX_FIX_H_

#if defined(__linux__) && (defined(__x86_64__) || defined(_M_X64))

#include <llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h>

namespace tvm {
namespace ffi {
namespace orcjit {

/*! \brief PreFixupPass plugin that corrects broken GOTPCRELX
 *         relaxations produced by JITLink's `optimizeGOTAndStubAccesses`.
 *
 * See the file-level docstring above for the trigger, symptom, and
 * removal procedure.
 */
class GOTPCRELXFixPlugin : public llvm::orc::ObjectLinkingLayer::Plugin {
 public:
  void modifyPassConfig(llvm::orc::MaterializationResponsibility& MR, llvm::jitlink::LinkGraph& G,
                        llvm::jitlink::PassConfiguration& Config) override;
  llvm::Error notifyFailed(llvm::orc::MaterializationResponsibility& MR) override;
  llvm::Error notifyRemovingResources(llvm::orc::JITDylib& JD, llvm::orc::ResourceKey K) override;
  void notifyTransferringResources(llvm::orc::JITDylib& JD, llvm::orc::ResourceKey DstKey,
                                   llvm::orc::ResourceKey SrcKey) override;
};

}  // namespace orcjit
}  // namespace ffi
}  // namespace tvm

#endif  // __linux__ && __x86_64__

#endif  // TVM_FFI_ORCJIT_LLVM_PATCHES_GOTPCRELX_FIX_H_
