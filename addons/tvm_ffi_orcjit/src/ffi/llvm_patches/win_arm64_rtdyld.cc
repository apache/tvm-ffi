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
#include <mutex>
#include <unordered_map>
#include <vector>

namespace tvm {
namespace ffi {
namespace orcjit {
namespace {

struct PendingCrtEntries {
  std::vector<ORCJITExecutionSessionObj::InitFiniEntry> initializers;
  std::vector<ORCJITExecutionSessionObj::InitFiniEntry> deinitializers;
};

struct PendingCrtObjects {
  std::mutex mutex;
  std::unordered_map<const char*, std::shared_ptr<PendingCrtEntries>> objects;
};

// RuntimeDyld destroys the memory manager on failed materialization. Discard
// the object's staged callbacks then, before their slot memory is released.
class CrtSectionMemoryManager : public llvm::SectionMemoryManager {
 public:
  CrtSectionMemoryManager(std::shared_ptr<PendingCrtObjects> pending, const char* key)
      : pending_(std::move(pending)), key_(key), entries_(std::make_shared<PendingCrtEntries>()) {
    std::lock_guard<std::mutex> lock(pending_->mutex);
    pending_->objects[key_] = entries_;
  }

  ~CrtSectionMemoryManager() override {
    std::lock_guard<std::mutex> lock(pending_->mutex);
    auto it = pending_->objects.find(key_);
    // A successful object's buffer can be reused while its JIT memory lives.
    if (it != pending_->objects.end() && it->second == entries_) pending_->objects.erase(it);
  }

 private:
  std::shared_ptr<PendingCrtObjects> pending_;
  const char* key_;
  std::shared_ptr<PendingCrtEntries> entries_;
};

}  // namespace

std::unique_ptr<llvm::orc::ObjectLayer> CreateWindowsArm64ObjectLinkingLayer(
    llvm::orc::ExecutionSession& execution_session, ORCJITExecutionSessionObj* session) {
  auto pending = std::make_shared<PendingCrtObjects>();
  auto layer = std::make_unique<llvm::orc::RTDyldObjectLinkingLayer>(
      execution_session, [pending](const llvm::MemoryBuffer& buffer) {
        return std::make_unique<CrtSectionMemoryManager>(pending, buffer.getBufferStart());
      });
  layer->setOverrideObjectFlagsWithResponsibilityFlags(true);
  layer->setAutoClaimResponsibilityForObjectSymbols(true);
  layer->setNotifyLoaded([pending](llvm::orc::MaterializationResponsibility&,
                                   const llvm::object::ObjectFile& object,
                                   const llvm::RuntimeDyld::LoadedObjectInfo& loaded) {
    std::lock_guard<std::mutex> lock(pending->mutex);
    auto& entries = *pending->objects.at(object.getMemoryBufferRef().getBufferStart());
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
      // NotifyLoaded precedes relocation. Stage pointer slots here and publish
      // them only after NotifyEmitted confirms successful materialization.
      constexpr uint64_t kPointerSize = 8;
      for (uint64_t offset = 0; offset + kPointerSize <= section.getSize();
           offset += kPointerSize) {
        ORCJITExecutionSessionObj::InitFiniEntry entry{llvm::orc::ExecutorAddr(address + offset),
                                                       priority, true};
        if (is_init) {
          entries.initializers.push_back(entry);
        } else {
          entries.deinitializers.push_back(entry);
        }
      }
    }
  });
  layer->setNotifyEmitted(
      [pending, session](llvm::orc::MaterializationResponsibility& responsibility,
                         std::unique_ptr<llvm::MemoryBuffer> buffer) {
        PendingCrtEntries entries;
        {
          std::lock_guard<std::mutex> lock(pending->mutex);
          auto it = pending->objects.find(buffer->getBufferStart());
          entries = std::move(*it->second);
          pending->objects.erase(it);
        }
        auto* dylib = &responsibility.getTargetJITDylib();
        for (const auto& entry : entries.initializers) session->AddPendingInitializer(dylib, entry);
        for (const auto& entry : entries.deinitializers)
          session->AddPendingDeinitializer(dylib, entry);
      });
  return layer;
}

}  // namespace orcjit
}  // namespace ffi
}  // namespace tvm
