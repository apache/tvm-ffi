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
 * \file orcjit_memory_manager.cc
 * \brief Thin wrapper delegating JITLinkMemoryManager ops to a Slab.
 */
#include "orcjit_memory_manager.h"

#ifdef __linux__

#include <llvm/Support/Error.h>
#include <llvm/Support/FormatVariadic.h>

#include <algorithm>

namespace tvm {
namespace ffi {
namespace orcjit {

using llvm::Error;

ArenaJITLinkMemoryManager::ArenaJITLinkMemoryManager(std::size_t page_size,
                                                     std::size_t slab_capacity) {
  // Try requested capacity, halve on failure down to a minimum floor.
  // The floor is the smaller of kMinSlabCapacity and the requested size,
  // so explicit small slabs (e.g. 16 MB for tests) are honoured.
  // mmap(PROT_NONE | MAP_NORESERVE) can still fail under RLIMIT_AS or
  // extreme VA fragmentation.
  std::size_t floor = std::min(slab_capacity, kMinSlabCapacity);
  std::size_t cap = slab_capacity;
  while (cap >= floor) {
    auto slab = std::make_unique<Slab>(page_size, cap);
    if (slab->isValid()) {
      slab_ = std::move(slab);
      return;
    }
    cap /= 2;
  }
  llvm::report_fatal_error("ArenaJITLinkMemoryManager: failed to reserve at least " +
                           llvm::Twine(floor / (1024 * 1024)) + " MB of virtual address space");
}

void ArenaJITLinkMemoryManager::allocate(const llvm::jitlink::JITLinkDylib* /*JD*/,
                                         llvm::jitlink::LinkGraph& G,
                                         OnAllocatedFunction OnAllocated) {
  slab_->allocate(G, std::move(OnAllocated));
}

void ArenaJITLinkMemoryManager::deallocate(std::vector<FinalizedAlloc> Allocs,
                                           OnDeallocatedFunction OnDeallocated) {
  Error DeallocErr = Error::success();
  for (auto& Alloc : Allocs) {
    auto* FA = Alloc.release().toPtr<FinalizedAllocInfo*>();
    FA->owner->deallocateOne(FA, DeallocErr);
    delete FA;
  }
  OnDeallocated(std::move(DeallocErr));
}

}  // namespace orcjit
}  // namespace ffi
}  // namespace tvm

#endif  // __linux__
