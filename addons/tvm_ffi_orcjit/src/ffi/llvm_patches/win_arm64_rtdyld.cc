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
 * \file win_arm64_rtdyld.cc
 * \brief RTDyld fallback for Windows ARM64 object files.
 *
 * See win_arm64_rtdyld.h for the trigger, symptom and removal procedure.
 */
#include "win_arm64_rtdyld.h"

#include <llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h>
#include <llvm/ExecutionEngine/SectionMemoryManager.h>

#include <cstdint>
#include <memory>

namespace tvm {
namespace ffi {
namespace orcjit {

std::unique_ptr<llvm::orc::ObjectLayer> CreateWindowsArm64ObjectLinkingLayer(
    llvm::orc::ExecutionSession& execution_session, ORCJITExecutionSessionObj* session) {
  auto layer = std::make_unique<llvm::orc::RTDyldObjectLinkingLayer>(
      execution_session,
      [](const llvm::MemoryBuffer&) { return std::make_unique<llvm::SectionMemoryManager>(); });
  layer->setOverrideObjectFlagsWithResponsibilityFlags(true);
  layer->setAutoClaimResponsibilityForObjectSymbols(true);
  layer->setNotifyLoaded([session](llvm::orc::MaterializationResponsibility& responsibility,
                                   const llvm::object::ObjectFile& object,
                                   const llvm::RuntimeDyld::LoadedObjectInfo& loaded) {
    auto& dylib = responsibility.getTargetJITDylib();
    for (const auto& section : object.sections()) {
      auto name = section.getName();
      if (!name) {
        llvm::consumeError(name.takeError());
        continue;
      }
      bool is_init = name->starts_with(".CRT$XC");
      if (!is_init && !name->starts_with(".CRT$XT")) continue;
      auto address = loaded.getSectionLoadAddress(section);
      if (!address) continue;
      auto suffix = name->substr(7);
      int priority = 0;
      if (!suffix.empty()) {
        priority = static_cast<int>(suffix[0]) * 100000;
        if (suffix.size() > 1) {
          int number = 0;
          suffix.substr(1).getAsInteger(10, number);
          priority += number;
        }
      }
      // NotifyLoaded precedes relocation. Keep addresses of the pointer
      // slots; RunInitFiniEntries dereferences them after materialization.
      constexpr uint64_t kPointerSize = 8;
      for (uint64_t offset = 0; offset + kPointerSize <= section.getSize();
           offset += kPointerSize) {
        ORCJITExecutionSessionObj::InitFiniEntry entry{llvm::orc::ExecutorAddr(address + offset),
                                                       priority, true};
        if (is_init) {
          session->AddPendingInitializer(&dylib, entry);
        } else {
          session->AddPendingDeinitializer(&dylib, entry);
        }
      }
    }
  });
  return layer;
}

}  // namespace orcjit
}  // namespace ffi
}  // namespace tvm
