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
 * \file orcjit_arena_mm.h
 * \brief Arena-based JITLinkMemoryManager for LLVM ORC JIT.
 *
 * Pre-reserves a contiguous virtual address region and bump-allocates
 * from it, keeping all JIT allocations within range of PC-relative
 * relocations (±2GB on x86_64, ±4GB on AArch64).
 *
 * This eliminates relocation overflow caused by scattered mmap
 * allocations under ASLR (LLVM issue #173269).
 *
 * ## GOTPCRELX relaxation workaround
 *
 * The arena triggers a latent bug in LLVM JITLink's
 * `optimizeGOTAndStubAccesses()` (x86_64.cpp).  That pass relaxes
 * `call *foo@GOTPCREL(%rip)` (ff 15) → `addr32 call foo` (67 e8) and
 * sets the edge kind to `Pointer32` (absolute 32-bit address).  However
 * the `call rel32` instruction is always **PC-relative** — the `67`
 * prefix is just padding — so the fixup should be PC-relative too
 * (matching the static linker's `R_X86_64_PC32`).
 *
 * The bug is latent because the relaxation only fires when the target
 * address fits in 32 bits (`isUInt<32>`).  On PIE executables every
 * resolved symbol is at a high address, so the guard is never true and
 * the relaxation never runs.  On **non-PIE** executables the PLT
 * entries for libc functions (malloc, free, …) live near 0x400000, the
 * guard passes, and the wrong fixup produces a garbage displacement →
 * SIGSEGV during ORC-runtime teardown.
 *
 * `GOTPCRELXFixPlugin` in orcjit_session.cc works around this: a
 * PreFixupPass that runs *after* `optimizeGOTAndStubAccesses` detects
 * `Pointer32` edges on `67 e8` / `e9` instructions and either
 *   (a) converts to `BranchPCRel32` when the PC-relative displacement
 *       fits in int32, or
 *   (b) reverts the relaxation entirely — restores the `ff 15` /
 *       `ff 25` opcode bytes and retargets the edge to the GOT entry
 *       with `PCRel32` + addend 0.
 */
#ifndef TVM_FFI_ORCJIT_ORCJIT_ARENA_MM_H_
#define TVM_FFI_ORCJIT_ORCJIT_ARENA_MM_H_

#include <llvm/ExecutionEngine/JITLink/JITLinkMemoryManager.h>
#include <llvm/ExecutionEngine/Orc/Shared/MemoryFlags.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

namespace tvm {
namespace ffi {
namespace orcjit {

/*! \brief Arena-based memory manager for JITLink.
 *
 * Reserves a large contiguous VA region at construction time using
 * PROT_NONE (zero physical memory cost).  Each allocate() call
 * bump-allocates from this region, commits pages as RW, and assigns
 * addresses.  On finalization, pages receive their target protections.
 * On deallocation, pages are decommitted and returned to a free list.
 *
 * The default arena size is strictly larger than the architecture's
 * PC-relative relocation limit (4 GB on x86_64, 8 GB on AArch64) so
 * the arena is never the bottleneck — JITLink's own relocation overflow
 * checker fires first, matching dlopen/ld.so failure semantics.  If the
 * initial reservation fails (RLIMIT_AS, container limits), the
 * constructor halves the capacity down to kMinArenaCapacity (256 MB).
 *
 * ## Slab-based commit with Transparent Huge Page (THP) support
 *
 * Arena pages are committed in 2 MB slabs (kSlabSize) rather than
 * per-allocation.  Each slab is committed exactly once via an atomic
 * flag (lock-free, no contention with the allocator mutex).
 *
 * Benefits:
 *   - Batches up to 512 page faults into a single sequential mprotect
 *     per slab, reducing kernel trap overhead.
 *   - 2 MB matches the Linux huge page size on both x86_64 and AArch64.
 *     Combined with madvise(MADV_HUGEPAGE) applied at construction, the
 *     kernel can promote each fully-faulted slab into a single TLB
 *     entry (replacing 512 x 4 KB entries), reducing TLB misses during
 *     JIT code execution.
 *   - Worst-case waste is <2 MB in the last partially-used slab —
 *     negligible for typical ML workloads.
 */
class ArenaJITLinkMemoryManager : public llvm::jitlink::JITLinkMemoryManager {
 public:
  // Default arena: strictly larger than the relocation limit so the arena
  // is never the bottleneck.  JITLink's own overflow check fires first,
  // matching dlopen/ld.so failure semantics.
  //
  // x86_64 PC32: ±2GB  →  4GB default (2× headroom)
  // AArch64 ADRP: ±4GB →  8GB default (2× headroom)
  static constexpr size_t kDefaultArenaCapacity_x86_64 = size_t{4} << 30;   // 4 GB
  static constexpr size_t kDefaultArenaCapacity_AArch64 = size_t{8} << 30;  // 8 GB
  static constexpr size_t kMinArenaCapacity = size_t{256} << 20;            // 256 MB floor
  // Slab commit granularity.  Matches Linux huge page size (2 MB) on both
  // x86_64 and AArch64, enabling THP promotion via madvise(MADV_HUGEPAGE).
  static constexpr size_t kSlabSize = size_t{2} << 20;  // 2 MB

  explicit ArenaJITLinkMemoryManager(size_t page_size, size_t arena_capacity);
  ~ArenaJITLinkMemoryManager() override;

  ArenaJITLinkMemoryManager(const ArenaJITLinkMemoryManager&) = delete;
  ArenaJITLinkMemoryManager& operator=(const ArenaJITLinkMemoryManager&) = delete;
  ArenaJITLinkMemoryManager(ArenaJITLinkMemoryManager&&) = delete;
  ArenaJITLinkMemoryManager& operator=(ArenaJITLinkMemoryManager&&) = delete;

  void allocate(const llvm::jitlink::JITLinkDylib* JD, llvm::jitlink::LinkGraph& G,
                OnAllocatedFunction OnAllocated) override;

  void deallocate(std::vector<FinalizedAlloc> Allocs, OnDeallocatedFunction OnDeallocated) override;

 private:
  class ArenaInFlightAlloc;

  /*! \brief Metadata for a finalized allocation, stored via FinalizedAlloc handle. */
  struct FinalizedAllocInfo {
    size_t arena_offset;
    size_t arena_size;
    std::vector<llvm::orc::shared::WrapperFunctionCall> DeallocActions;
  };

  /*! \brief Bump-allocate from arena. Returns offset within arena. */
  llvm::Expected<size_t> bumpAllocate(size_t size);

  /*! \brief Return a region to the free list (coalesces adjacent blocks). */
  void freeRegion(size_t offset, size_t size);

  // ── Platform abstraction ──
  static void* reserveVA(size_t size);
  static void releaseVA(void* addr, size_t size);
  llvm::Error commitPages(void* addr, size_t size);
  static void decommitPages(void* addr, size_t size);
  static llvm::Error protectPages(void* addr, size_t size, llvm::orc::MemProt Prot);

  char* arena_base_;
  size_t arena_capacity_;
  size_t page_size_;

  std::mutex mu_;
  size_t bump_offset_;

  struct FreeBlock {
    size_t offset;
    size_t size;
  };
  std::vector<FreeBlock> free_list_;

  /*! \brief Per-slab commit flags (0 = uncommitted, 1 = committed).
   *  Lock-free: each slab is committed exactly once via compare_exchange. */
  std::unique_ptr<std::atomic<uint8_t>[]> slab_committed_;
  size_t num_slabs_ = 0;
};

}  // namespace orcjit
}  // namespace ffi
}  // namespace tvm

#endif  // TVM_FFI_ORCJIT_ORCJIT_ARENA_MM_H_
