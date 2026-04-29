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
 * \file orcjit_slab.cc
 * \brief Slab implementation — bump allocator + commit bitmap + free list.
 */

#include "orcjit_slab.h"

#ifdef __linux__

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ExecutionEngine/JITLink/JITLink.h>
#include <llvm/ExecutionEngine/JITLink/aarch64.h>
#include <llvm/ExecutionEngine/JITLink/x86_64.h>
#include <llvm/Support/Alignment.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/Memory.h>
#include <sys/mman.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <limits>

namespace tvm {
namespace ffi {
namespace orcjit {

using namespace llvm;
using namespace llvm::jitlink;
using namespace llvm::orc;

// ── Overflow section edge classification ───────────────────────────
//
// Conservative whitelist: only known absolute relocation kinds return true.
// Unknown or future edge kinds default to PC-relative → sections stay in
// the slab (safe: never breaks relocations, just forgoes the overflow
// optimization for unknown kinds).

namespace {

bool isAbsoluteEdge(const Triple& TT, Edge::Kind K) {
  if (K < Edge::FirstRelocation) return true;  // KeepAlive, Invalid — not a relocation constraint
  if (TT.isAArch64()) {
    using namespace llvm::jitlink::aarch64;
    switch (K) {
      case Pointer64:
      case Pointer32:
      case Pointer64Authenticated:
      case MoveWide16:
        return true;
      default:
        return false;
    }
  }
  if (TT.isX86()) {
    using namespace llvm::jitlink::x86_64;
    switch (K) {
      case Pointer64:
      case Pointer32:
      case Pointer32Signed:
      case Pointer16:
      case Pointer8:
      case Size64:
      case Size32:
        return true;
      default:
        return false;
    }
  }
  return false;  // Unknown arch — treat as PC-relative (safe)
}

}  // namespace

// ── Platform abstraction ────────────────────────────────────────────

void* Slab::reserveVA(std::size_t size) {
  void* p = ::mmap(nullptr, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  if (p == MAP_FAILED) return nullptr;
  return p;
}

void Slab::releaseVA(void* addr, std::size_t size) {
  int rc = ::munmap(addr, size);
  assert(rc == 0 && "munmap failed in Slab destructor");
  (void)rc;
}

Error Slab::commitPages(void* addr, std::size_t size) {
  if (size == 0) return Error::success();
  // Commit at commit-chunk (2 MB) granularity for THP promotion.
  std::size_t offset = static_cast<char*>(addr) - arena_base_;
  std::size_t first_chunk = offset / kCommitGranularity;
  std::size_t last_chunk = (offset + size - 1) / kCommitGranularity;

  for (std::size_t i = first_chunk; i <= last_chunk; ++i) {
    if (committed_[i].load(std::memory_order_acquire) != 0) continue;
    std::size_t chunk_offset = i * kCommitGranularity;
    std::size_t chunk_len = std::min(kCommitGranularity, arena_capacity_ - chunk_offset);
    // mprotect is idempotent, so a concurrent racer calling it on the same chunk
    // is harmless.  Only flip the flag after success — otherwise a failed commit
    // followed by freeRegion() would leave committed_[i] == 1, causing a
    // later allocation to skip mprotect and write into PROT_NONE memory.
    if (::mprotect(arena_base_ + chunk_offset, chunk_len, PROT_READ | PROT_WRITE) != 0) {
      return make_error<StringError>(
          "Slab: mprotect(RW) failed for chunk at offset " + formatv("{0:x}", chunk_offset) + ": " +
              std::strerror(errno),
          inconvertibleErrorCode());
    }
    committed_[i].store(1, std::memory_order_release);
  }
  return Error::success();
}

void Slab::decommitPages(void* addr, std::size_t size) {
  // Intentionally a no-op for slab pages.  The ORC runtime may still reference
  // deallocated JIT memory during session teardown (e.g., ELFNixPlatform
  // deinitializers run after some allocations are freed).  Decommitting
  // (MADV_DONTNEED or mprotect PROT_NONE) would cause segfaults or illegal
  // instructions during shutdown.
  //
  // Physical pages stay committed but are returned to the free list for reuse.
  // The slab destructor releases all VA and physical memory via munmap.
  (void)addr;
  (void)size;
}

Error Slab::protectPages(void* addr, std::size_t size, MemProt Prot) {
  int prot = PROT_NONE;
  if ((Prot & MemProt::Read) != MemProt::None) prot |= PROT_READ;
  if ((Prot & MemProt::Write) != MemProt::None) prot |= PROT_WRITE;
  if ((Prot & MemProt::Exec) != MemProt::None) prot |= PROT_EXEC;
  if (::mprotect(addr, size, prot) != 0) {
    return make_error<StringError>("Slab: mprotect failed at " + formatv("{0:x}", addr) +
                                       " size " + formatv("{0:x}", size) + ": " +
                                       std::strerror(errno),
                                   inconvertibleErrorCode());
  }
  if ((Prot & MemProt::Exec) != MemProt::None) {
    sys::Memory::InvalidateInstructionCache(addr, size);
  }
  return Error::success();
}

// ── InFlightAlloc ───────────────────────────────────────────────────

class Slab::InFlightAlloc : public JITLinkMemoryManager::InFlightAlloc {
 public:
  // A contiguous region within one pool: [offset, offset + standard_size + finalize_size).
  // Standard-lifetime bytes come first; Finalize-lifetime bytes follow and are freed
  // at the end of finalize().  Any field may be 0 to indicate no allocation from
  // that pool on this call.
  struct PoolRegion {
    std::size_t offset;
    std::size_t standard_size;
    std::size_t finalize_size;
  };

  InFlightAlloc(Slab& S, LinkGraph& G, BasicLayout BL, PoolRegion non_exec, PoolRegion exec,
                std::vector<FinalizedAllocInfo::OverflowBlock> overflow_blocks)
      : S(S),
        G(&G),
        BL(std::move(BL)),
        non_exec_(non_exec),
        exec_(exec),
        overflow_blocks_(std::move(overflow_blocks)) {}

  ~InFlightAlloc() override {
    assert(!G && "Slab::InFlightAlloc destroyed without finalize or abandon");
  }

  void finalize(OnFinalizedFunction OnFinalized) override {
    // Apply target protections for each slab segment.
    if (auto Err = applyProtections()) {
      OnFinalized(std::move(Err));
      return;
    }

    // Apply target protections for overflow blocks.
    for (auto& ob : overflow_blocks_) {
      if (auto Err = S.protectPages(ob.addr, ob.size, ob.prot)) {
        OnFinalized(std::move(Err));
        return;
      }
    }

    // Run finalization actions (e.g., register EH frames).
    auto DeallocActions = shared::runFinalizeActions(BL.graphAllocActions());
    if (!DeallocActions) {
      OnFinalized(DeallocActions.takeError());
      return;
    }

    // Decommit finalize-lifetime pages in each pool — they're no longer needed.
    for (auto& R : {non_exec_, exec_}) {
      if (R.finalize_size > 0) {
        S.decommitPages(S.arena_base_ + R.offset + R.standard_size, R.finalize_size);
        S.freeRegion(R.offset + R.standard_size, R.finalize_size);
      }
    }

#ifndef NDEBUG
    G = nullptr;
#endif

    // Create finalized allocation handle.  LLVM's FinalizedAlloc stores an
    // opaque ExecutorAddr (integer), so we must use raw new here.  Ownership
    // transfers to deallocate(), which LLVM guarantees is called for every
    // finalized allocation.
    auto* FA = new FinalizedAllocInfo{&S,
                                      non_exec_.offset,
                                      non_exec_.standard_size,
                                      exec_.offset,
                                      exec_.standard_size,
                                      std::move(*DeallocActions),
                                      std::move(overflow_blocks_)};
    OnFinalized(JITLinkMemoryManager::FinalizedAlloc(ExecutorAddr::fromPtr(FA)));
  }

  void abandon(OnAbandonedFunction OnAbandoned) override {
    // Decommit and return each pool's full region to the appropriate free list.
    for (auto& R : {non_exec_, exec_}) {
      std::size_t total = R.standard_size + R.finalize_size;
      if (total > 0) {
        S.decommitPages(S.arena_base_ + R.offset, total);
        S.freeRegion(R.offset, total);
      }
    }

    // Release overflow blocks.
    for (auto& ob : overflow_blocks_) {
      ::munmap(ob.addr, ob.size);
    }

#ifndef NDEBUG
    G = nullptr;
#endif

    OnAbandoned(Error::success());
  }

 private:
  Error applyProtections() {
    for (auto& KV : BL.segments()) {
      const auto& AG = KV.first;
      auto& Seg = KV.second;

      auto SegSize = alignTo(Seg.ContentSize + Seg.ZeroFillSize, S.page_size_);
      if (auto Err = S.protectPages(Seg.WorkingMem, SegSize, AG.getMemProt())) return Err;
    }
    return Error::success();
  }

  Slab& S;
  LinkGraph* G;
  BasicLayout BL;
  PoolRegion non_exec_;
  PoolRegion exec_;
  std::vector<FinalizedAllocInfo::OverflowBlock> overflow_blocks_;
};

// ── Slab ────────────────────────────────────────────────────────────

Slab::Slab(std::size_t page_size, std::size_t capacity)
    : arena_base_(nullptr),
      arena_capacity_(capacity),
      page_size_(page_size),
      midpoint_(0),
      exec_bump_limit_(0),
      non_exec_bump_(0),
      exec_bump_(0) {
  arena_base_ = static_cast<char*>(reserveVA(capacity));
  if (!arena_base_) return;  // Caller inspects isValid() and retries.

  // Partition the slab into two pools at a 2 MB-aligned midpoint.  The
  // exec pool starts at midpoint_, which is therefore on a 2 MB
  // boundary — r-x segments pack into a minimum number of 2 MB pages.
  //
  // Constraint: cross-pool displacements (e.g. .text → .rodata via
  // ADRP+ADD on aarch64) must fit in ±kPCRelReach.  The farthest pair of
  // bytes is (end of exec, start of non-exec), separated by at most
  // `exec_bump_limit_`, so we cap the exec pool's upper bound at
  // kPCRelReach even when the VA reservation is larger.
  exec_bump_limit_ = std::min(capacity, kPCRelReach);
  std::size_t raw_midpoint = static_cast<std::size_t>(exec_bump_limit_ * kDefaultNonExecFraction);
  midpoint_ = (raw_midpoint / kCommitGranularity) * kCommitGranularity;
  if (midpoint_ == 0) midpoint_ = kCommitGranularity;
  if (midpoint_ >= exec_bump_limit_) midpoint_ = exec_bump_limit_ - kCommitGranularity;
  non_exec_bump_ = 0;
  exec_bump_ = midpoint_;
  // Initialize commit tracking.  make_unique<T[]>(n) value-initializes
  // the array to zero in C++17.
  num_commit_chunks_ = (capacity + kCommitGranularity - 1) / kCommitGranularity;
  committed_ = std::make_unique<std::atomic<std::uint8_t>[]>(num_commit_chunks_);
  // Hint THP promotion for the entire slab.  Intentionally unchecked —
  // MADV_HUGEPAGE is advisory and may fail if THP is disabled system-wide.
  (void)::madvise(arena_base_, capacity, MADV_HUGEPAGE);
}

Slab::~Slab() {
  if (arena_base_) {
    releaseVA(arena_base_, arena_capacity_);
  }
}

Expected<std::size_t> Slab::bumpAllocate(std::size_t size, bool is_exec) {
  std::lock_guard<std::mutex> Lock(mu_);

  auto& free_list = is_exec ? free_list_exec_ : free_list_non_exec_;
  auto& bump = is_exec ? exec_bump_ : non_exec_bump_;
  std::size_t limit = is_exec ? exec_bump_limit_ : midpoint_;

  // Try free list first (best-fit).  O(n) scan — acceptable for the expected
  // workload of tens of JIT allocations, not thousands.
  std::size_t best_idx = free_list.size();
  std::size_t best_waste = std::numeric_limits<std::size_t>::max();
  for (std::size_t i = 0; i < free_list.size(); ++i) {
    if (free_list[i].size >= size && free_list[i].size - size < best_waste) {
      best_idx = i;
      best_waste = free_list[i].size - size;
      if (best_waste == 0) break;
    }
  }

  if (best_idx < free_list.size()) {
    std::size_t offset = free_list[best_idx].offset;
    if (free_list[best_idx].size == size) {
      free_list.erase(free_list.begin() + best_idx);
    } else {
      free_list[best_idx].offset += size;
      free_list[best_idx].size -= size;
    }
    return offset;
  }

  // Bump allocate within the pool's limit.
  if (bump + size > limit) {
    return make_error<StringError>(
        std::string("Slab: ") + (is_exec ? "exec" : "non-exec") + " pool exhausted (used " +
            formatv("{0:x}", bump).str() + " + requested " + formatv("{0:x}", size).str() +
            " > limit " + formatv("{0:x}", limit).str() + ")",
        inconvertibleErrorCode());
  }

  std::size_t offset = bump;
  bump += size;
  return offset;
}

void Slab::freeRegion(std::size_t offset, std::size_t size) {
  if (size == 0) return;
  std::lock_guard<std::mutex> Lock(mu_);

  // Route to the correct pool's free list based on offset.
  auto& free_list = (offset >= midpoint_) ? free_list_exec_ : free_list_non_exec_;

  // Insert into free list in sorted order.
  auto it = std::lower_bound(free_list.begin(), free_list.end(), offset,
                             [](const FreeBlock& fb, std::size_t off) { return fb.offset < off; });
  it = free_list.insert(it, FreeBlock{offset, size});

  // Coalesce with next.
  auto next = it + 1;
  if (next != free_list.end() && it->offset + it->size == next->offset) {
    it->size += next->size;
    free_list.erase(next);
  }

  // Coalesce with previous.
  if (it != free_list.begin()) {
    auto prev = it - 1;
    if (prev->offset + prev->size == it->offset) {
      prev->size += it->size;
      free_list.erase(it);
    }
  }
}

void Slab::allocate(LinkGraph& G,
                    JITLinkMemoryManager::OnAllocatedFunction OnAllocated) {
  // ── Overflow section classification ──
  //
  // Sections matching known overflow names (e.g. .nv_fatbin — large GPU
  // device blobs referenced only by absolute relocations) are allocated
  // outside the slab via separate mmap(), keeping the slab compact for
  // code + small rodata.
  //
  // Two-phase check:
  //   Phase 1 — Name-based candidate selection (.nv_fatbin).
  //   Phase 2 — Edge validation: any PC-relative cross-section edge
  //             targeting a candidate section disqualifies it (the
  //             section stays in the slab).  This handles cases where
  //             the compiler generates ADRP/RIP-relative refs even for
  //             data sections.
  //
  // Validated candidates are temporarily set to NoAlloc so BasicLayout
  // skips them, then immediately restored before returning.  By the time
  // JITLink's fixUpBlocks runs, sections are back to Standard — avoiding
  // the debug assert that prohibits edges from allocated sections to
  // NoAlloc sections.
  DenseSet<Section*> overflow_candidates;
  for (auto& Sec : G.sections()) {
    if (Sec.getMemLifetime() == MemLifetime::NoAlloc) continue;
    StringRef Name = Sec.getName();
    if (Name.starts_with(".nv_fatbin")) {
      overflow_candidates.insert(&Sec);
    }
  }

  // Phase 2: edge validation — disqualify candidates with incoming PC-relative edges.
  if (!overflow_candidates.empty()) {
    const auto& TT = G.getTargetTriple();
    for (auto& Sec : G.sections()) {
      for (auto* B : Sec.blocks()) {
        for (auto& E : B->edges()) {
          if (!E.isRelocation()) continue;
          if (isAbsoluteEdge(TT, E.getKind())) continue;
          // PC-relative edge — if it targets a candidate, disqualify.
          if (!E.getTarget().isDefined()) continue;
          auto* TargetSec = &E.getTarget().getBlock().getSection();
          overflow_candidates.erase(TargetSec);
        }
      }
      if (overflow_candidates.empty()) break;
    }
  }

  // Apply: temporarily hide validated overflow sections from BasicLayout.
  SmallVector<std::pair<Section*, MemLifetime>, 4> overflow_sections;
  for (auto* Sec : overflow_candidates) {
    overflow_sections.push_back({Sec, Sec->getMemLifetime()});
    Sec->setMemLifetime(MemLifetime::NoAlloc);
  }

  BasicLayout BL(G);

  // Restore overflow sections to their original lifetime immediately.
  // BasicLayout has already captured its segment list; subsequent LLVM
  // passes (fixUpBlocks) will see the sections as normal Standard sections.
  for (auto& [Sec, OrigLifetime] : overflow_sections) {
    Sec->setMemLifetime(OrigLifetime);
  }

  // Compute total sizes grouped by lifetime.
  auto SegsSizes = BL.getContiguousPageBasedLayoutSizes(page_size_);
  if (!SegsSizes) {
    OnAllocated(SegsSizes.takeError());
    return;
  }

  if (SegsSizes->total() > std::numeric_limits<std::size_t>::max()) {
    OnAllocated(make_error<llvm::jitlink::JITLinkError>(
        "Total requested size " + formatv("{0:x}", SegsSizes->total()) + " for graph " +
        G.getName() + " exceeds address space"));
    return;
  }

  auto TotalSize = static_cast<std::size_t>(SegsSizes->total());
  if (TotalSize == 0 && overflow_sections.empty()) {
    // Empty graph — return a no-op allocation.
    OnAllocated(std::make_unique<InFlightAlloc>(
        *this, G, std::move(BL), InFlightAlloc::PoolRegion{0, 0, 0},
        InFlightAlloc::PoolRegion{midpoint_, 0, 0},
        std::vector<FinalizedAllocInfo::OverflowBlock>{}));
    return;
  }

  // ── Dual-pool split ──
  //
  // Partition each segment into one of four buckets based on (Prot, Lifetime):
  //   non-exec × Standard / Finalize   →  non-exec pool (below midpoint_)
  //   exec     × Standard / Finalize   →  exec pool     (at/above midpoint_)
  //
  // Within each pool, Standard segments come first and Finalize segments
  // second, so the Finalize tail of each pool can be freed after finalize().
  std::size_t ne_std_size = 0, ne_fin_size = 0;
  std::size_t e_std_size = 0, e_fin_size = 0;
  for (auto& KV : BL.segments()) {
    auto& AG = KV.first;
    auto& Seg = KV.second;
    auto SegSize = alignTo(Seg.ContentSize + Seg.ZeroFillSize, page_size_);
    bool is_exec = (AG.getMemProt() & MemProt::Exec) != MemProt::None;
    bool is_finalize = AG.getMemLifetime() == MemLifetime::Finalize;
    if (is_exec) {
      (is_finalize ? e_fin_size : e_std_size) += SegSize;
    } else {
      (is_finalize ? ne_fin_size : ne_std_size) += SegSize;
    }
  }
  std::size_t ne_total = ne_std_size + ne_fin_size;
  std::size_t e_total = e_std_size + e_fin_size;

  InFlightAlloc::PoolRegion ne_region{0, 0, 0};
  InFlightAlloc::PoolRegion e_region{midpoint_, 0, 0};

  auto allocPool = [&](std::size_t req, bool is_exec) -> Expected<std::size_t> {
    if (req == 0) return std::size_t{0};
    auto off = bumpAllocate(req, is_exec);
    if (!off) return off.takeError();
    if (auto Err = commitPages(arena_base_ + *off, req)) {
      freeRegion(*off, req);
      return std::move(Err);
    }
    // Recycled-region protection reset.  commitPages() only mprotects a 2 MB
    // commit-chunk the first time it's touched (guarded by committed_).
    // When the region was previously handed out, finalized to r-x/r--/rw-
    // and later returned to the free list by deallocateOne(), those finalize
    // protections are still in effect — the memset(0) below would fault on
    // read-only pages.  mprotect(RW) has no decommit effect, so the
    // physical pages stay resident; we're only restoring write access for
    // the upcoming zero-fill and subsequent JITLink content writes.
    if (auto Err = protectPages(arena_base_ + *off, req, MemProt::Read | MemProt::Write)) {
      freeRegion(*off, req);
      return std::move(Err);
    }
    std::memset(arena_base_ + *off, 0, req);
    return *off;
  };

  if (ne_total > 0) {
    auto off = allocPool(ne_total, /*is_exec=*/false);
    if (!off) {
      OnAllocated(off.takeError());
      return;
    }
    ne_region = {*off, ne_std_size, ne_fin_size};
  }
  if (e_total > 0) {
    auto off = allocPool(e_total, /*is_exec=*/true);
    if (!off) {
      // Unwind non-exec allocation on failure to keep the pools consistent.
      if (ne_total > 0) {
        decommitPages(arena_base_ + ne_region.offset, ne_total);
        freeRegion(ne_region.offset, ne_total);
      }
      OnAllocated(off.takeError());
      return;
    }
    e_region = {*off, e_std_size, e_fin_size};
  }

  // Assign addresses to segments from four cursors.  Standard comes first in
  // each pool, then Finalize.
  auto NeStdCursor = ExecutorAddr::fromPtr(arena_base_ + ne_region.offset);
  auto NeFinCursor = ExecutorAddr::fromPtr(arena_base_ + ne_region.offset + ne_std_size);
  auto EStdCursor = ExecutorAddr::fromPtr(arena_base_ + e_region.offset);
  auto EFinCursor = ExecutorAddr::fromPtr(arena_base_ + e_region.offset + e_std_size);

  for (auto& KV : BL.segments()) {
    auto& AG = KV.first;
    auto& Seg = KV.second;
    bool is_exec = (AG.getMemProt() & MemProt::Exec) != MemProt::None;
    bool is_finalize = AG.getMemLifetime() == MemLifetime::Finalize;
    auto& Cursor = is_exec ? (is_finalize ? EFinCursor : EStdCursor)
                           : (is_finalize ? NeFinCursor : NeStdCursor);
    Seg.WorkingMem = Cursor.toPtr<char*>();
    Seg.Addr = Cursor;
    auto SegSize = alignTo(Seg.ContentSize + Seg.ZeroFillSize, page_size_);
    Cursor += SegSize;
  }

  // Apply layout — copies content and assigns block addresses for slab segments.
  if (auto Err = BL.apply()) {
    // On error: decommit and free both pool regions.
    if (ne_total > 0) {
      decommitPages(arena_base_ + ne_region.offset, ne_total);
      freeRegion(ne_region.offset, ne_total);
    }
    if (e_total > 0) {
      decommitPages(arena_base_ + e_region.offset, e_total);
      freeRegion(e_region.offset, e_total);
    }
    OnAllocated(std::move(Err));
    return;
  }

  // ── Allocate overflow sections via mmap() outside the slab ──
  std::vector<FinalizedAllocInfo::OverflowBlock> overflow_allocs;

  for (auto& [Sec, _] : overflow_sections) {
    // Compute total size for this section's blocks.
    std::size_t total_sec_size = 0;
    for (auto* B : Sec->blocks()) {
      total_sec_size = alignTo(total_sec_size, B->getAlignment());
      total_sec_size += B->getSize();
    }
    if (total_sec_size == 0) continue;
    total_sec_size = alignTo(total_sec_size, page_size_);

    // mmap outside the slab.
    void* addr =
        ::mmap(nullptr, total_sec_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (addr == MAP_FAILED) {
      // Clean up prior overflow allocs, free both pool regions, report error.
      for (auto& ob : overflow_allocs) ::munmap(ob.addr, ob.size);
      if (ne_total > 0) {
        decommitPages(arena_base_ + ne_region.offset, ne_total);
        freeRegion(ne_region.offset, ne_total);
      }
      if (e_total > 0) {
        decommitPages(arena_base_ + e_region.offset, e_total);
        freeRegion(e_region.offset, e_total);
      }
      OnAllocated(make_error<StringError>("Slab: overflow mmap failed for section " +
                                              Sec->getName() + ": " + std::strerror(errno),
                                          inconvertibleErrorCode()));
      return;
    }

    // Layout blocks within the mmap'd region.
    char* ptr = static_cast<char*>(addr);
    for (auto* B : Sec->blocks()) {
      uint64_t align = B->getAlignment();
      ptr = reinterpret_cast<char*>(alignTo(reinterpret_cast<uintptr_t>(ptr), align));
      std::size_t bsize = B->getSize();
      // Copy content and redirect block's mutable content pointer.
      if (!B->isZeroFill()) {
        auto content = B->getContent();
        std::memcpy(ptr, content.data(), content.size());
        B->setMutableContent(MutableArrayRef<char>(ptr, bsize));
      }
      // Assign block address (working mem == executor addr for in-process JIT).
      B->setAddress(ExecutorAddr::fromPtr(ptr));
      ptr += bsize;
    }

    overflow_allocs.push_back({addr, total_sec_size, Sec->getMemProt()});
  }

  OnAllocated(std::make_unique<InFlightAlloc>(*this, G, std::move(BL), ne_region, e_region,
                                              std::move(overflow_allocs)));
}

void Slab::deallocateOne(FinalizedAllocInfo* FA, Error& err_out) {
  // Run deallocation actions in reverse order.
  while (!FA->DeallocActions.empty()) {
    if (auto Err = FA->DeallocActions.back().runWithSPSRetErrorMerged()) {
      err_out = joinErrors(std::move(err_out), std::move(Err));
    }
    FA->DeallocActions.pop_back();
  }

  // Decommit and free each pool's Standard region.
  //
  // We intentionally do *not* reset page protection here.  During session
  // teardown the ORC runtime may still execute (or read) deallocated JIT
  // pages from other dylibs while their DeallocActions unwind — same
  // rationale as decommitPages being a no-op.  Protection is reset lazily
  // in allocate() when the region is re-handed out (see the
  // "recycled-region protection reset" block in allocPool).
  if (FA->non_exec_standard_size > 0) {
    decommitPages(arena_base_ + FA->non_exec_offset, FA->non_exec_standard_size);
    freeRegion(FA->non_exec_offset, FA->non_exec_standard_size);
  }
  if (FA->exec_standard_size > 0) {
    decommitPages(arena_base_ + FA->exec_offset, FA->exec_standard_size);
    freeRegion(FA->exec_offset, FA->exec_standard_size);
  }

  // Release overflow blocks.
  for (auto& ob : FA->overflow_blocks) {
    ::munmap(ob.addr, ob.size);
  }
}

}  // namespace orcjit
}  // namespace ffi
}  // namespace tvm

#endif  // __linux__
