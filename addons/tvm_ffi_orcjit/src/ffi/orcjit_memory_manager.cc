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
 * \brief Growable per-session pool of `Slab`s.
 */
#include "orcjit_memory_manager.h"

#ifdef __linux__

#include <llvm/Support/Error.h>
#include <llvm/Support/FormatVariadic.h>

#include <algorithm>
#include <optional>
#include <utility>

namespace tvm {
namespace ffi {
namespace orcjit {

using llvm::Error;
using llvm::Expected;

SlabPoolMemoryManager::SlabPoolMemoryManager(std::size_t page_size, std::size_t slab_size)
    : page_size_(page_size), slab_size_(slab_size) {
  // Reserve the initial slab.  Halving retry only applies here: if the
  // very first mmap fails (RLIMIT_AS, container limits), we halve the
  // requested size down to kMinSlabSize before giving up.  Subsequent
  // slabs added during allocate() use exactly slab_size_ and propagate
  // errors on mmap failure.
  std::size_t floor = std::min(slab_size_, kMinSlabSize);
  std::size_t cap = slab_size_;
  while (cap >= floor) {
    auto slab = std::make_unique<Slab>(page_size_, cap);
    if (slab->isValid()) {
      // Pin the actual initial-slab size to whatever we succeeded with.
      // If RLIMIT_AS forced us to 8 MB, we keep 8 MB as the working slab
      // size; growing later at 64 MB would just fail again.
      slab_size_ = cap;
      slabs_.push_back(std::move(slab));
      return;
    }
    cap /= 2;
  }
  llvm::report_fatal_error("SlabPoolMemoryManager: failed to reserve at least " +
                           llvm::Twine(floor / (1024 * 1024)) + " MB of virtual address space");
}

std::unique_ptr<Slab> SlabPoolMemoryManager::createSlab(std::size_t capacity) {
  auto slab = std::make_unique<Slab>(page_size_, capacity);
  if (!slab->isValid()) return nullptr;
  return slab;
}

void SlabPoolMemoryManager::allocate(const llvm::jitlink::JITLinkDylib* /*JD*/,
                                     llvm::jitlink::LinkGraph& G, OnAllocatedFunction OnAllocated) {
  using AllocResult = Expected<std::unique_ptr<InFlightAlloc>>;

  // Step 1: first-fit over existing slabs.  `pool_mu_` only protects
  // the slabs_ vector — never held across a Slab::allocate call or a
  // user callback, since the LLJIT linker issues nested lookups (and
  // thus re-entrant allocate() calls via materialization) from inside
  // OnAllocated and a coarse lock would deadlock.  Snapshot raw pointers
  // under the lock; slabs are guaranteed to outlive this call because
  // clearFreeSlabs() is only safe when the session is quiescent.
  //
  // Slab::allocate is synchronous (invokes its callback inline on every
  // code path), so a captured std::optional observes the result before
  // the call returns.
  std::vector<Slab*> snapshot;
  {
    std::lock_guard<std::mutex> lock(pool_mu_);
    snapshot.reserve(slabs_.size());
    for (auto& s : slabs_) snapshot.push_back(s.get());
  }
  for (Slab* slab : snapshot) {
    std::optional<AllocResult> observed;
    slab->allocate(G, [&](AllocResult R) { observed.emplace(std::move(R)); });
    AllocResult result = std::move(*observed);
    if (result) {
      OnAllocated(std::move(result));
      return;
    }
    Error E = result.takeError();
    if (E.isA<SlabPoolExhaustedError>()) {
      // Retriable: this graph didn't fit this slab's per-pool budget.
      llvm::consumeError(std::move(E));
      continue;
    }
    // Terminal (mmap, mprotect, JITLink, BasicLayout).
    OnAllocated(std::move(E));
    return;
  }

  // Step 2: grow.  Size the fresh slab to fit this graph's per-pool
  // footprint — normal graphs fall through at slab_size_, skewed or
  // oversize graphs double up until both pools can host them (see
  // Slab::capacityForFootprint).  A single growth branch replaces the
  // pre-gate "normal vs oversize" split; we trade one `N × failed
  // Slab::allocate` scan (step 1) for the extra pre-filter — negligible
  // since each failed call is a BasicLayout build with no mmap.
  auto fp = Slab::computeGraphFootprint(G, page_size_);
  std::size_t cap = Slab::capacityForFootprint(fp, slab_size_);
  auto slab = createSlab(cap);
  if (!slab) {
    OnAllocated(
        llvm::make_error<llvm::StringError>("SlabPoolMemoryManager: mmap failed for new slab of " +
                                                llvm::formatv("{0:x}", cap).str() + " bytes",
                                            llvm::inconvertibleErrorCode()));
    return;
  }
  Slab* raw = slab.get();
  {
    std::lock_guard<std::mutex> lock(pool_mu_);
    slabs_.push_back(std::move(slab));
  }
  raw->allocate(G, std::move(OnAllocated));
}

void SlabPoolMemoryManager::deallocate(std::vector<FinalizedAlloc> Allocs,
                                       OnDeallocatedFunction OnDeallocated) {
  Error DeallocErr = Error::success();
  for (auto& Alloc : Allocs) {
    auto* FA = Alloc.release().toPtr<FinalizedAllocInfo*>();
    FA->owner->deallocateOne(FA, DeallocErr);
    delete FA;
  }
  OnDeallocated(std::move(DeallocErr));
}

std::size_t SlabPoolMemoryManager::clearFreeSlabs() {
  // Partition under the lock, move discards to a local vector, drop the
  // lock, then let ~Slab (which calls munmap) run outside the lock.
  // Keeping munmap outside pool_mu_ matches the rest of allocate/deallocate,
  // which also never hold the lock across syscalls that might block.
  std::vector<std::unique_ptr<Slab>> discard;
  {
    std::lock_guard<std::mutex> lock(pool_mu_);
    auto keep_end = std::partition(slabs_.begin(), slabs_.end(),
                                   [](const auto& s) { return !s->isReclaimable(); });
    discard.reserve(static_cast<std::size_t>(slabs_.end() - keep_end));
    for (auto it = keep_end; it != slabs_.end(); ++it) {
      discard.push_back(std::move(*it));
    }
    slabs_.erase(keep_end, slabs_.end());
  }
  std::size_t reclaimed = discard.size();
  // discard goes out of scope — Slab destructors munmap each reservation.
  return reclaimed;
}

}  // namespace orcjit
}  // namespace ffi
}  // namespace tvm

#endif  // __linux__
