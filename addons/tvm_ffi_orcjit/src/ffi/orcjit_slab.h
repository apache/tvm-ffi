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
 * \file orcjit_slab.h
 * \brief Single contiguous-VA region + dual-pool bump allocator.
 *
 * A `Slab` owns one `mmap(PROT_NONE)` reservation of fixed capacity and
 * bump-allocates from it, keeping all JIT allocations within range of
 * PC-relative relocations (±2 GB on x86_64, ±4 GB on AArch64).
 *
 * The `Slab` is the unit-of-VA-reservation for the OrcJIT memory manager.
 * Today it is used as a single-slab arena owned by
 * `ArenaJITLinkMemoryManager`. Stage B of the refactor will introduce a
 * `SlabPoolMemoryManager` that holds multiple Slabs and grows by mmap-ing
 * new ones on demand.
 *
 * ## Page commit + Transparent Huge Page (THP) support
 *
 * Pages are committed in 2 MB chunks (`kCommitGranularity`) — the 2 MB
 * size matches the Linux huge-page granule on both x86_64 and AArch64,
 * enabling THP promotion via `madvise(MADV_HUGEPAGE)` on the full
 * reservation. Each 2 MB commit-chunk is `mprotect`-ed to RW exactly once
 * via an atomic bitmap flag (`committed_`), avoiding lock contention with
 * the per-pool allocator mutex.
 *
 * ## Dual-pool exec / non-exec split
 *
 * The slab is partitioned at a 2 MB-aligned `midpoint_` into two bump
 * pools:
 *
 *   non-exec pool  = [base,           base + midpoint_                     )
 *   exec pool      = [base + midpoint_, base + exec_bump_limit_           )
 *
 * Both pools grow upward; cross-pool displacements (.text → .rodata etc.)
 * must fit in ±`kPCRelReach` — we cap `exec_bump_limit_` at that reach
 * even when the VA reservation is larger.
 */
#ifndef TVM_FFI_ORCJIT_ORCJIT_SLAB_H_
#define TVM_FFI_ORCJIT_ORCJIT_SLAB_H_

#include <llvm/ExecutionEngine/JITLink/JITLinkMemoryManager.h>
#include <llvm/ExecutionEngine/Orc/Shared/AllocationActions.h>
#include <llvm/ExecutionEngine/Orc/Shared/MemoryFlags.h>
#include <llvm/Support/Error.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace tvm {
namespace ffi {
namespace orcjit {

class Slab;  // forward-declared for FinalizedAllocInfo.

/*!
 * \brief Metadata for a finalized allocation, stored via FinalizedAlloc
 *        handle.
 *
 * Each allocate() call may consume a region from either or both pools.
 * Standard-lifetime pages remain committed after finalize();
 * Finalize-lifetime pages are decommitted at the end of finalize().
 * Zero-sized sub-regions indicate no allocation from that pool.
 *
 * \p owner points to the Slab that handed out this allocation.  With one
 * slab per session today this is redundant, but stamping it now makes
 * Stage B's pool-manager routing O(1) without address comparison.
 */
struct FinalizedAllocInfo {
  Slab* owner;                    ///< Slab that owns these offsets.
  std::size_t non_exec_offset;    ///< offset of non-exec Standard region (or 0 if unused).
  std::size_t non_exec_standard_size;
  std::size_t exec_offset;        ///< offset of exec Standard region (or midpoint_ if unused).
  std::size_t exec_standard_size;
  std::vector<llvm::orc::shared::WrapperFunctionCall> DeallocActions;
  struct OverflowBlock {
    void* addr;                   ///< separately-mmap'd base (outside the slab).
    std::size_t size;             ///< mapping size (page-aligned).
    llvm::orc::MemProt prot;      ///< target protection for finalize.
  };
  std::vector<OverflowBlock> overflow_blocks;
};

/*!
 * \brief One contiguous JIT memory reservation.
 *
 * Exposes the per-graph `allocate` entry point (matching
 * `JITLinkMemoryManager::allocate`'s callback signature) and the
 * per-FinalizedAlloc `deallocateOne` used by the outer memory manager to
 * route deallocation.
 */
class Slab {
 public:
  // Commit / THP granularity.  Every 2 MB chunk is mprotect'd RW exactly
  // once via `committed_`; `madvise(MADV_HUGEPAGE)` can then promote a
  // fully-faulted chunk into a single huge page.
  static constexpr std::size_t kCommitGranularity = std::size_t{2} << 20;  // 2 MB

  // PC-relative relocation reach (tightest binding fixup).  Cross-pool
  // references must fit in a signed 32-bit displacement.  The binding
  // constraint on both x86_64 and aarch64 is the signed 32-bit Delta32
  // used in .eh_frame unwind records (±2 GB), not the wider ADRP+ADD /
  // RIP-rel reach.  `exec_bump_limit_` is capped at this reach so
  // cross-pool Delta32 fixups always resolve.
  static constexpr std::size_t kPCRelReach =
      (std::size_t{1} << 31) - kCommitGranularity;  // ~2 GB

  // Fraction of the slab reserved for non-exec segments (r--, rw-).  The
  // remainder holds exec (r-x).  Typical CUDA binding objects: ~2 parts
  // rodata+data to 1 part text.
  static constexpr double kDefaultNonExecFraction = 2.0 / 3.0;

  /*! \brief Construct a Slab and reserve \p capacity bytes of VA.
   *
   *  On reservation failure, returns with \c base() == nullptr — the
   *  caller is expected to retry at a smaller capacity or
   *  \c report_fatal_error.
   */
  Slab(std::size_t page_size, std::size_t capacity);

  ~Slab();

  Slab(const Slab&) = delete;
  Slab& operator=(const Slab&) = delete;
  Slab(Slab&&) = delete;
  Slab& operator=(Slab&&) = delete;

  /*! \brief True iff the reservation succeeded. */
  bool isValid() const noexcept { return arena_base_ != nullptr; }

  /*! \brief Single-graph JIT allocation entry point. */
  void allocate(llvm::jitlink::LinkGraph& G,
                llvm::jitlink::JITLinkMemoryManager::OnAllocatedFunction OnAllocated);

  /*! \brief Per-FA teardown: run DeallocActions, free pool regions,
   *         release overflow blocks. Caller deletes the FA afterwards.
   *
   *  Errors are joined into \p err_out; never throws.
   */
  void deallocateOne(FinalizedAllocInfo* FA, llvm::Error& err_out);

  /*! \brief Address-range ownership check. */
  bool contains(const void* addr) const noexcept {
    auto* p = static_cast<const char*>(addr);
    return p >= arena_base_ && p < arena_base_ + arena_capacity_;
  }

  char* base() const noexcept { return arena_base_; }
  std::size_t capacity() const noexcept { return arena_capacity_; }
  std::size_t page_size() const noexcept { return page_size_; }

 private:
  class InFlightAlloc;  // defined in orcjit_slab.cc

  /*! \brief Bump-allocate from the selected pool.  Returns offset within
   *         the slab's VA reservation. */
  llvm::Expected<std::size_t> bumpAllocate(std::size_t size, bool is_exec);

  /*! \brief Return a region to the appropriate free list.  Pool is
   *         identified by comparing offset against midpoint_. */
  void freeRegion(std::size_t offset, std::size_t size);

  // ── Platform abstraction (all implemented in orcjit_slab.cc) ──
  static void* reserveVA(std::size_t size);
  static void releaseVA(void* addr, std::size_t size);
  llvm::Error commitPages(void* addr, std::size_t size);
  static void decommitPages(void* addr, std::size_t size);
  static llvm::Error protectPages(void* addr, std::size_t size, llvm::orc::MemProt Prot);

  char* arena_base_;
  std::size_t arena_capacity_;
  std::size_t page_size_;

  // Dual-pool split.  See class docstring.
  std::size_t midpoint_;
  std::size_t exec_bump_limit_;

  std::mutex mu_;
  std::size_t non_exec_bump_;  // next free offset in non-exec pool ∈ [0, midpoint_]
  std::size_t exec_bump_;      // next free offset in exec pool     ∈ [midpoint_, exec_bump_limit_]

  struct FreeBlock {
    std::size_t offset;
    std::size_t size;
  };
  std::vector<FreeBlock> free_list_non_exec_;
  std::vector<FreeBlock> free_list_exec_;

  /*! \brief Per-commit-chunk flags (0 = uncommitted, 1 = committed).
   *         Lock-free: each chunk is mprotect'd exactly once via
   *         compare_exchange. */
  std::unique_ptr<std::atomic<std::uint8_t>[]> committed_;
  std::size_t num_commit_chunks_ = 0;
};

}  // namespace orcjit
}  // namespace ffi
}  // namespace tvm

#endif  // TVM_FFI_ORCJIT_ORCJIT_SLAB_H_
