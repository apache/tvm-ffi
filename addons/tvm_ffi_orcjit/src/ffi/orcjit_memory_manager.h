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
 * \file orcjit_memory_manager.h
 * \brief JITLinkMemoryManager backed by one fixed-size Slab (Stage A).
 *
 * Holds a single `Slab` (see orcjit_slab.h) and delegates all
 * `JITLinkMemoryManager` operations to it.  The capacity-negotiation
 * retry loop (halve-on-mmap-failure) lives here because it is a
 * caller/kernel negotiation, not something a Slab itself should do.
 *
 * In later stages this class is superseded by a `SlabPoolMemoryManager`
 * that owns multiple Slabs per session and grows on demand.  The public
 * API surface (one memory manager per LLJIT, constructed with a
 * page-size and a slab-capacity) stays the same.
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
 * `GOTPCRELXFixPlugin` in llvm_patches/gotpcrelx_fix.cc works around
 * this: a PreFixupPass that runs *after* `optimizeGOTAndStubAccesses`
 * detects `Pointer32` edges on `67 e8` / `e9` instructions and either
 *   (a) converts to `BranchPCRel32` when the PC-relative displacement
 *       fits in int32, or
 *   (b) reverts the relaxation entirely — restores the `ff 15` /
 *       `ff 25` opcode bytes and retargets the edge to the GOT entry
 *       with `PCRel32` + addend 0.
 */
#ifndef TVM_FFI_ORCJIT_ORCJIT_MEMORY_MANAGER_H_
#define TVM_FFI_ORCJIT_ORCJIT_MEMORY_MANAGER_H_

#include <llvm/ExecutionEngine/JITLink/JITLinkMemoryManager.h>

#include <cstddef>
#include <memory>

#include "orcjit_slab.h"

namespace tvm {
namespace ffi {
namespace orcjit {

/*! \brief JITLink memory manager backed by a single `Slab`.
 *
 *  Reserves `slab_capacity` bytes of contiguous VA at construction time,
 *  halving the request down to `kMinSlabCapacity` if the initial `mmap`
 *  fails (RLIMIT_AS, container limits).
 *
 *  `allocate` and `deallocate` forward to the underlying `Slab`.
 */
class ArenaJITLinkMemoryManager : public llvm::jitlink::JITLinkMemoryManager {
 public:
  // Default slab capacity: 1 GB on both architectures.  Well within the
  // PC-relative relocation limit (x86_64 ±2 GB, AArch64 ±4 GB) so
  // cross-section fixups always fit; large enough to cover typical ML
  // JIT workloads without oversubscribing virtual address space on
  // memory-constrained hosts (containers, CI runners).
  static constexpr std::size_t kDefaultSlabCapacity_x86_64 = std::size_t{1} << 30;   // 1 GB
  static constexpr std::size_t kDefaultSlabCapacity_AArch64 = std::size_t{1} << 30;  // 1 GB
  static constexpr std::size_t kMinSlabCapacity = std::size_t{256} << 20;            // 256 MB floor

  explicit ArenaJITLinkMemoryManager(std::size_t page_size, std::size_t slab_capacity);
  ~ArenaJITLinkMemoryManager() override = default;

  ArenaJITLinkMemoryManager(const ArenaJITLinkMemoryManager&) = delete;
  ArenaJITLinkMemoryManager& operator=(const ArenaJITLinkMemoryManager&) = delete;
  ArenaJITLinkMemoryManager(ArenaJITLinkMemoryManager&&) = delete;
  ArenaJITLinkMemoryManager& operator=(ArenaJITLinkMemoryManager&&) = delete;

  void allocate(const llvm::jitlink::JITLinkDylib* JD, llvm::jitlink::LinkGraph& G,
                OnAllocatedFunction OnAllocated) override;

  void deallocate(std::vector<FinalizedAlloc> Allocs,
                  OnDeallocatedFunction OnDeallocated) override;

 private:
  std::unique_ptr<Slab> slab_;
};

}  // namespace orcjit
}  // namespace ffi
}  // namespace tvm

#endif  // TVM_FFI_ORCJIT_ORCJIT_MEMORY_MANAGER_H_
