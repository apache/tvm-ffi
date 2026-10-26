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
 * \file win_arm64_rtdyld.h
 * \brief RTDyld fallback for Windows ARM64 object files.
 *
 * LLVM 23.1.1 JITLink rejects COFF/AArch64 objects. RuntimeDyld supports
 * their relocations, long branches and DLL-import pointer slots, so use
 * RTDyldObjectLinkingLayer on this platform. A loaded-object callback records
 * .CRT$XC* / .CRT$XT* entries for the session's normal init/fini drain.
 *
 * Trigger: loading any native Windows ARM64 COFF object.
 * Symptom without the patch: "Unsupported target machine architecture in
 * COFF object ...: ARM64" from JITLink.
 *
 * Removal: when LLVM gains a usable COFF/AArch64 JITLink backend, switch
 * Windows ARM64 to ObjectLinkingLayer and InitFiniPlugin, add AArch64 stubs
 * to DLLImportDefinitionGenerator, and delete this adapter.
 */
#ifndef TVM_FFI_ORCJIT_LLVM_PATCHES_WIN_ARM64_RTDYLD_H_
#define TVM_FFI_ORCJIT_LLVM_PATCHES_WIN_ARM64_RTDYLD_H_

#include <llvm/ExecutionEngine/Orc/Layer.h>

#include <memory>

#include "../orcjit_session.h"

namespace tvm {
namespace ffi {
namespace orcjit {

std::unique_ptr<llvm::orc::ObjectLayer> CreateWindowsArm64ObjectLinkingLayer(
    llvm::orc::ExecutionSession& execution_session, ORCJITExecutionSessionObj* session);

}  // namespace orcjit
}  // namespace ffi
}  // namespace tvm

#endif  // TVM_FFI_ORCJIT_LLVM_PATCHES_WIN_ARM64_RTDYLD_H_
