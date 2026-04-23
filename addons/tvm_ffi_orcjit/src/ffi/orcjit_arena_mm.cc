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
 * \file orcjit_arena_mm.cc
 * \brief Arena-based JITLinkMemoryManager implementation.
 *
 * Follows the InProcessMemoryManager pattern from LLVM but replaces
 * per-object mmap with bump allocation from a pre-reserved arena.
 * Pages are committed in 2 MB slabs to enable Transparent Huge Page
 * (THP) promotion — see the class docstring in orcjit_arena_mm.h.
 */
#include "orcjit_arena_mm.h"

#ifdef __linux__

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ExecutionEngine/JITLink/JITLink.h>
#include <llvm/ExecutionEngine/JITLink/aarch64.h>
#include <llvm/ExecutionEngine/JITLink/x86_64.h>
#include <llvm/ExecutionEngine/Orc/Shared/AllocationActions.h>
#include <llvm/Support/Alignment.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/Memory.h>
#include <sys/mman.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>

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
// the arena (safe: never breaks relocations, just forgoes the overflow
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

void* ArenaJITLinkMemoryManager::reserveVA(size_t size) {
  void* p = ::mmap(nullptr, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  if (p == MAP_FAILED) return nullptr;
  return p;
}

void ArenaJITLinkMemoryManager::releaseVA(void* addr, size_t size) {
  int rc = ::munmap(addr, size);
  assert(rc == 0 && "munmap failed in arena destructor");
  (void)rc;
}

Error ArenaJITLinkMemoryManager::commitPages(void* addr, size_t size) {
  if (size == 0) return Error::success();
  // Commit at slab (2 MB) granularity for THP promotion.
  size_t offset = static_cast<char*>(addr) - arena_base_;
  size_t first_slab = offset / kSlabSize;
  size_t last_slab = (offset + size - 1) / kSlabSize;

  for (size_t i = first_slab; i <= last_slab; ++i) {
    uint8_t expected = 0;
    if (slab_committed_[i].compare_exchange_strong(expected, 1, std::memory_order_acq_rel)) {
      size_t slab_offset = i * kSlabSize;
      size_t slab_len = std::min(kSlabSize, arena_capacity_ - slab_offset);
      if (::mprotect(arena_base_ + slab_offset, slab_len, PROT_READ | PROT_WRITE) != 0) {
        return make_error<StringError>(
            "ArenaJITLinkMemoryManager: mprotect(RW) failed for slab at offset " +
                formatv("{0:x}", slab_offset) + ": " + std::strerror(errno),
            inconvertibleErrorCode());
      }
    }
  }
  return Error::success();
}

void ArenaJITLinkMemoryManager::decommitPages(void* addr, size_t size) {
  // Intentionally a no-op for arena pages.  The ORC runtime may still reference
  // deallocated JIT memory during session teardown (e.g., ELFNixPlatform
  // deinitializers run after some allocations are freed).  Decommitting
  // (MADV_DONTNEED or mprotect PROT_NONE) would cause segfaults or illegal
  // instructions during shutdown.
  //
  // Physical pages stay committed but are returned to the free list for reuse.
  // The arena destructor releases all VA and physical memory via munmap.
  (void)addr;
  (void)size;
}

Error ArenaJITLinkMemoryManager::protectPages(void* addr, size_t size, MemProt Prot) {
  int prot = PROT_NONE;
  if ((Prot & MemProt::Read) != MemProt::None) prot |= PROT_READ;
  if ((Prot & MemProt::Write) != MemProt::None) prot |= PROT_WRITE;
  if ((Prot & MemProt::Exec) != MemProt::None) prot |= PROT_EXEC;
  if (::mprotect(addr, size, prot) != 0) {
    return make_error<StringError>("ArenaJITLinkMemoryManager: mprotect failed at " +
                                       formatv("{0:x}", addr) + " size " + formatv("{0:x}", size) +
                                       ": " + std::strerror(errno),
                                   inconvertibleErrorCode());
  }
  if ((Prot & MemProt::Exec) != MemProt::None) {
    sys::Memory::InvalidateInstructionCache(addr, size);
  }
  return Error::success();
}

// ── ArenaInFlightAlloc ──────────────────────────────────────────────

class ArenaJITLinkMemoryManager::ArenaInFlightAlloc : public JITLinkMemoryManager::InFlightAlloc {
 public:
  // A contiguous region within one pool: [offset, offset + standard_size + finalize_size).
  // Standard-lifetime bytes come first; Finalize-lifetime bytes follow and are freed
  // at the end of finalize().  Any field may be 0 to indicate no allocation from
  // that pool on this call.
  struct PoolRegion {
    size_t offset;
    size_t standard_size;
    size_t finalize_size;
  };

  ArenaInFlightAlloc(ArenaJITLinkMemoryManager& MM, LinkGraph& G, BasicLayout BL,
                     PoolRegion non_exec, PoolRegion exec,
                     std::vector<OverflowBlock> overflow_blocks)
      : MM(MM),
        G(&G),
        BL(std::move(BL)),
        non_exec_(non_exec),
        exec_(exec),
        overflow_blocks_(std::move(overflow_blocks)) {}

  ~ArenaInFlightAlloc() override {
    assert(!G && "ArenaInFlightAlloc destroyed without finalize or abandon");
  }

  void finalize(OnFinalizedFunction OnFinalized) override {
    // Apply target protections for each arena segment.
    if (auto Err = applyProtections()) {
      OnFinalized(std::move(Err));
      return;
    }

    // Apply target protections for overflow blocks.
    for (auto& ob : overflow_blocks_) {
      if (auto Err = MM.protectPages(ob.addr, ob.size, ob.prot)) {
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
        MM.decommitPages(MM.arena_base_ + R.offset + R.standard_size, R.finalize_size);
        MM.freeRegion(R.offset + R.standard_size, R.finalize_size);
      }
    }

#ifndef NDEBUG
    G = nullptr;
#endif

    // Create finalized allocation handle.  LLVM's FinalizedAlloc stores an
    // opaque ExecutorAddr (integer), so we must use raw new here.  Ownership
    // transfers to deallocate(), which LLVM guarantees is called for every
    // finalized allocation.
    auto* FA = new FinalizedAllocInfo{non_exec_.offset,      non_exec_.standard_size,
                                      exec_.offset,          exec_.standard_size,
                                      std::move(*DeallocActions), std::move(overflow_blocks_)};
    OnFinalized(FinalizedAlloc(ExecutorAddr::fromPtr(FA)));
  }

  void abandon(OnAbandonedFunction OnAbandoned) override {
    // Decommit and return each pool's full region to the appropriate free list.
    for (auto& R : {non_exec_, exec_}) {
      size_t total = R.standard_size + R.finalize_size;
      if (total > 0) {
        MM.decommitPages(MM.arena_base_ + R.offset, total);
        MM.freeRegion(R.offset, total);
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

      auto SegSize = alignTo(Seg.ContentSize + Seg.ZeroFillSize, MM.page_size_);
      if (auto Err = MM.protectPages(Seg.WorkingMem, SegSize, AG.getMemProt())) return Err;
    }
    return Error::success();
  }

  ArenaJITLinkMemoryManager& MM;
  LinkGraph* G;
  BasicLayout BL;
  PoolRegion non_exec_;
  PoolRegion exec_;
  std::vector<OverflowBlock> overflow_blocks_;
};

// ── ArenaJITLinkMemoryManager ───────────────────────────────────────

ArenaJITLinkMemoryManager::ArenaJITLinkMemoryManager(size_t page_size, size_t arena_capacity)
    : arena_base_(nullptr),
      arena_capacity_(arena_capacity),
      page_size_(page_size),
      midpoint_(0),
      exec_bump_limit_(0),
      non_exec_bump_(0),
      exec_bump_(0) {
  // Try requested capacity, halve on failure down to a minimum floor.
  // The floor is the smaller of kMinArenaCapacity and the requested size,
  // so explicit small arenas (e.g. 16 MB for tests) are honoured.
  // mmap(PROT_NONE | MAP_NORESERVE) can still fail under RLIMIT_AS or
  // extreme VA fragmentation.
  size_t floor = std::min(arena_capacity_, kMinArenaCapacity);
  size_t cap = arena_capacity_;
  while (cap >= floor) {
    arena_base_ = static_cast<char*>(reserveVA(cap));
    if (arena_base_) {
      arena_capacity_ = cap;
      // Partition the arena into two pools at a 2 MB-aligned midpoint.
      // The exec pool starts at midpoint_, which is therefore on a 2 MB
      // boundary — r-x segments pack into a minimum number of 2 MB pages.
      //
      // Constraint: cross-pool displacements (e.g. .text → .rodata via
      // ADRP+ADD on aarch64) must fit in ±kPCRelReach.  The farthest pair
      // of bytes is (end of exec, start of non-exec), separated by at most
      // `exec_bump_limit_`, so we cap the exec pool's upper bound at
      // kPCRelReach even when the VA reservation is larger.
      exec_bump_limit_ = std::min(cap, kPCRelReach);
      size_t raw_midpoint = static_cast<size_t>(exec_bump_limit_ * kDefaultNonExecFraction);
      midpoint_ = (raw_midpoint / kSlabSize) * kSlabSize;
      if (midpoint_ == 0) midpoint_ = kSlabSize;
      if (midpoint_ >= exec_bump_limit_) midpoint_ = exec_bump_limit_ - kSlabSize;
      non_exec_bump_ = 0;
      exec_bump_ = midpoint_;
      // Initialize slab commit tracking.  make_unique<T[]>(n) value-initializes
      // the array to zero in C++17.
      num_slabs_ = (cap + kSlabSize - 1) / kSlabSize;
      slab_committed_ = std::make_unique<std::atomic<uint8_t>[]>(num_slabs_);
      // Hint THP promotion for the entire arena.  Intentionally unchecked —
      // MADV_HUGEPAGE is advisory and may fail if THP is disabled system-wide.
      (void)::madvise(arena_base_, cap, MADV_HUGEPAGE);
      return;
    }
    cap /= 2;
  }
  llvm::report_fatal_error("ArenaJITLinkMemoryManager: failed to reserve at least " +
                           Twine(floor / (1024 * 1024)) + " MB of virtual address space");
}

ArenaJITLinkMemoryManager::~ArenaJITLinkMemoryManager() {
  if (arena_base_) {
    releaseVA(arena_base_, arena_capacity_);
  }
}

Expected<size_t> ArenaJITLinkMemoryManager::bumpAllocate(size_t size, bool is_exec) {
  std::lock_guard<std::mutex> Lock(mu_);

  auto& free_list = is_exec ? free_list_exec_ : free_list_non_exec_;
  auto& bump = is_exec ? exec_bump_ : non_exec_bump_;
  size_t limit = is_exec ? exec_bump_limit_ : midpoint_;

  // Try free list first (best-fit).  O(n) scan — acceptable for the expected
  // workload of tens of JIT allocations, not thousands.
  size_t best_idx = free_list.size();
  size_t best_waste = std::numeric_limits<size_t>::max();
  for (size_t i = 0; i < free_list.size(); ++i) {
    if (free_list[i].size >= size && free_list[i].size - size < best_waste) {
      best_idx = i;
      best_waste = free_list[i].size - size;
      if (best_waste == 0) break;
    }
  }

  if (best_idx < free_list.size()) {
    size_t offset = free_list[best_idx].offset;
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
        std::string("ArenaJITLinkMemoryManager: ") + (is_exec ? "exec" : "non-exec") +
            " pool exhausted (used " + formatv("{0:x}", bump).str() + " + requested " +
            formatv("{0:x}", size).str() + " > limit " + formatv("{0:x}", limit).str() + ")",
        inconvertibleErrorCode());
  }

  size_t offset = bump;
  bump += size;
  return offset;
}

void ArenaJITLinkMemoryManager::freeRegion(size_t offset, size_t size) {
  if (size == 0) return;
  std::lock_guard<std::mutex> Lock(mu_);

  // Route to the correct pool's free list based on offset.
  auto& free_list = (offset >= midpoint_) ? free_list_exec_ : free_list_non_exec_;

  // Insert into free list in sorted order.
  auto it = std::lower_bound(free_list.begin(), free_list.end(), offset,
                             [](const FreeBlock& fb, size_t off) { return fb.offset < off; });
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

void ArenaJITLinkMemoryManager::allocate(const JITLinkDylib* JD, LinkGraph& G,
                                         OnAllocatedFunction OnAllocated) {
  // ── Overflow section classification ──
  //
  // Sections matching known overflow names (e.g. .nv_fatbin — large GPU
  // device blobs referenced only by absolute relocations) are allocated
  // outside the arena via separate mmap(), keeping the arena compact for
  // code + small rodata.
  //
  // Two-phase check:
  //   Phase 1 — Name-based candidate selection (.nv_fatbin).
  //   Phase 2 — Edge validation: any PC-relative cross-section edge
  //             targeting a candidate section disqualifies it (the
  //             section stays in the arena).  This handles cases where
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

  if (SegsSizes->total() > std::numeric_limits<size_t>::max()) {
    OnAllocated(make_error<llvm::jitlink::JITLinkError>(
        "Total requested size " + formatv("{0:x}", SegsSizes->total()) + " for graph " +
        G.getName() + " exceeds address space"));
    return;
  }

  auto TotalSize = static_cast<size_t>(SegsSizes->total());
  if (TotalSize == 0 && overflow_sections.empty()) {
    // Empty graph — return a no-op allocation.
    OnAllocated(std::make_unique<ArenaInFlightAlloc>(
        *this, G, std::move(BL), ArenaInFlightAlloc::PoolRegion{0, 0, 0},
        ArenaInFlightAlloc::PoolRegion{midpoint_, 0, 0}, std::vector<OverflowBlock>{}));
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
  size_t ne_std_size = 0, ne_fin_size = 0;
  size_t e_std_size = 0, e_fin_size = 0;
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
  size_t ne_total = ne_std_size + ne_fin_size;
  size_t e_total = e_std_size + e_fin_size;

  ArenaInFlightAlloc::PoolRegion ne_region{0, 0, 0};
  ArenaInFlightAlloc::PoolRegion e_region{midpoint_, 0, 0};


  auto allocPool = [&](size_t req, bool is_exec) -> Expected<size_t> {
    if (req == 0) return size_t{0};
    auto off = bumpAllocate(req, is_exec);
    if (!off) return off.takeError();
    if (auto Err = commitPages(arena_base_ + *off, req)) {
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

  // Apply layout — copies content and assigns block addresses for arena segments.
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

  // ── Allocate overflow sections via mmap() outside the arena ──
  std::vector<OverflowBlock> overflow_allocs;

  for (auto& [Sec, _] : overflow_sections) {
    // Compute total size for this section's blocks.
    size_t total_sec_size = 0;
    for (auto* B : Sec->blocks()) {
      total_sec_size = alignTo(total_sec_size, B->getAlignment());
      total_sec_size += B->getSize();
    }
    if (total_sec_size == 0) continue;
    total_sec_size = alignTo(total_sec_size, page_size_);

    // mmap outside the arena.
    void* addr = ::mmap(nullptr, total_sec_size, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
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
      OnAllocated(make_error<StringError>(
          "ArenaJITLinkMemoryManager: overflow mmap failed for section " +
              Sec->getName() + ": " + std::strerror(errno),
          inconvertibleErrorCode()));
      return;
    }

    // Layout blocks within the mmap'd region.
    char* ptr = static_cast<char*>(addr);
    for (auto* B : Sec->blocks()) {
      uint64_t align = B->getAlignment();
      ptr = reinterpret_cast<char*>(alignTo(reinterpret_cast<uintptr_t>(ptr), align));
      size_t bsize = B->getSize();
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

  OnAllocated(std::make_unique<ArenaInFlightAlloc>(*this, G, std::move(BL), ne_region, e_region,
                                                   std::move(overflow_allocs)));
}

void ArenaJITLinkMemoryManager::deallocate(std::vector<FinalizedAlloc> Allocs,
                                           OnDeallocatedFunction OnDeallocated) {
  Error DeallocErr = Error::success();

  for (auto& Alloc : Allocs) {
    // Reclaim ownership of the FinalizedAllocInfo created in finalize().
    auto* FA = Alloc.release().toPtr<FinalizedAllocInfo*>();

    // Run deallocation actions in reverse order.
    while (!FA->DeallocActions.empty()) {
      if (auto Err = FA->DeallocActions.back().runWithSPSRetErrorMerged())
        DeallocErr = joinErrors(std::move(DeallocErr), std::move(Err));
      FA->DeallocActions.pop_back();
    }

    // Decommit and free each pool's Standard region.
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

    delete FA;
  }

  OnDeallocated(std::move(DeallocErr));
}

}  // namespace orcjit
}  // namespace ffi
}  // namespace tvm

#endif  // __linux__
