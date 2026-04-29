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

#include <llvm/Support/Alignment.h>
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
                                     llvm::jitlink::LinkGraph& G,
                                     OnAllocatedFunction OnAllocated) {
  // Step 1: pre-compute footprint to decide normal vs oversize.
  auto footprint = Slab::computeGraphFootprint(G, page_size_);
  std::size_t total = footprint.total();

  // Step 2: conservative usable-per-slab estimate.  The dual-pool
  // midpoint split means a graph cannot use the entire slab — one
  // pool's cursor is bounded at midpoint, the other at
  // exec_bump_limit.  2 MB of slack covers midpoint alignment.  A
  // false-positive oversize just costs one extra mmap sized to fit,
  // never a crash.
  std::size_t usable = slab_size_ > 2 * Slab::kCommitGranularity
                           ? slab_size_ - 2 * Slab::kCommitGranularity
                           : slab_size_ / 2;

  // `pool_mu_` only protects the slabs_ vector itself.  We never hold
  // it across a call into Slab::allocate or a user callback: the
  // LLJIT linker will frequently invoke nested lookups (which trigger
  // recursive allocate() calls via materialization) from inside
  // OnAllocated, and a coarse lock here would deadlock.  Slabs we've
  // seen in a snapshot are guaranteed to outlive this call because
  // Stage B never removes slabs from the pool.
  using AllocResult = Expected<std::unique_ptr<InFlightAlloc>>;

  // Step 3: oversize path — one graph per dedicated slab.
  if (total > usable) {
    std::size_t needed = total + 2 * Slab::kCommitGranularity;
    std::size_t cap = llvm::alignTo(needed, Slab::kCommitGranularity);
    if (cap < slab_size_) cap = slab_size_;
    auto slab = createSlab(cap);
    if (!slab) {
      OnAllocated(llvm::make_error<llvm::StringError>(
          "SlabPoolMemoryManager: mmap failed for oversize slab of " +
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
    return;
  }

  // Step 4: first-fit over existing slabs.  Take a snapshot of raw
  // pointers under the lock, then iterate without holding it.
  // Slab::allocate is synchronous (invokes its callback inline on
  // every code path), so we observe the result via a captured
  // std::optional that the callback fills before the call returns.
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
      // Retriable: this graph didn't fit in this slab.  Try next.
      llvm::consumeError(std::move(E));
      continue;
    }
    // Terminal error (mmap, mprotect, JITLink, BasicLayout).
    OnAllocated(std::move(E));
    return;
  }

  // Step 5: no existing slab fits.  Mmap a new normal-size slab.
  // Another thread may have added slabs meanwhile; we don't re-scan
  // (would duplicate work under contention).  Concurrent creates
  // would at worst make the pool grow faster than strictly necessary
  // — never incorrect.
  auto slab = createSlab(slab_size_);
  if (!slab) {
    OnAllocated(llvm::make_error<llvm::StringError>(
        "SlabPoolMemoryManager: mmap failed for new slab of " +
            llvm::formatv("{0:x}", slab_size_).str() + " bytes",
        llvm::inconvertibleErrorCode()));
    return;
  }
  Slab* raw = slab.get();
  {
    std::lock_guard<std::mutex> lock(pool_mu_);
    slabs_.push_back(std::move(slab));
  }
  // A fresh slab must fit any graph we've already decided is in-range
  // (step 2 + step 3 gate).  If somehow it doesn't, the error is
  // propagated through — not retried.
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

}  // namespace orcjit
}  // namespace ffi
}  // namespace tvm

#endif  // __linux__
