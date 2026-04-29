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
 * \brief Per-session growable slab pool (Stage B).
 *
 * `SlabPoolMemoryManager` implements `JITLinkMemoryManager` on top of a
 * per-session `std::vector<std::unique_ptr<Slab>>`.  On each `allocate`
 * it picks the first `Slab` that can fit the graph; if none do, it
 * `mmap`s a new fixed-size (`slab_size`) slab and appends it.  Graphs
 * larger than a single normal slab go to the oversize path — a
 * dedicated `Slab` sized to fit that one graph.
 *
 * ## Lifecycle (Stage B)
 *
 * Once a slab is added to the pool it stays mapped until the pool
 * (and its enclosing session) is destroyed.  Individual graphs are
 * deallocated via `FA->owner->deallocateOne(...)`, returning bytes to
 * the slab's free list, but the slab's VA reservation is not reclaimed.
 * Stage C will add warm-slab eviction that munmaps drained slabs.
 *
 * ## GOTPCRELX relaxation workaround
 *
 * Unchanged from Stage A — see `llvm_patches/gotpcrelx_fix.cc`.  The
 * plugin is added per-session to the `ObjectLinkingLayer` alongside
 * this memory manager and is orthogonal to pool growth.
 */
#ifndef TVM_FFI_ORCJIT_ORCJIT_MEMORY_MANAGER_H_
#define TVM_FFI_ORCJIT_ORCJIT_MEMORY_MANAGER_H_

#include <llvm/ExecutionEngine/JITLink/JITLinkMemoryManager.h>

#include <cstddef>
#include <memory>
#include <mutex>
#include <vector>

#include "orcjit_slab.h"

namespace tvm {
namespace ffi {
namespace orcjit {

/*!
 * \brief `JITLinkMemoryManager` backed by a growable pool of `Slab`s.
 *
 * The constructor reserves one initial slab (halving its capacity down
 * to `kMinSlabSize` if `mmap` fails under RLIMIT_AS).  Subsequent
 * slabs are added on demand by `allocate()` and reserved at exactly
 * `slab_size_` bytes each — no retry, no halving, errors propagate.
 */
class SlabPoolMemoryManager : public llvm::jitlink::JITLinkMemoryManager {
 public:
  // Default per-slab capacity.  64 MB is above the p99 size of typical
  // ML JIT graphs (single-kernel bindings, fused kernels), below the
  // PC-relative relocation limit, and a multiple of the 2 MB THP
  // granule.  Small enough that a pinned slab only wastes 64 MB of RSS.
  static constexpr std::size_t kDefaultSlabSize = std::size_t{64} << 20;  // 64 MB

  // Lower bound on initial-slab reservation.  If the first `mmap`
  // fails and halving drops below this, the constructor aborts.
  // 8 MB is enough for a minimal JITDylib setup under very tight
  // RLIMIT_AS.
  static constexpr std::size_t kMinSlabSize = std::size_t{8} << 20;  // 8 MB

  explicit SlabPoolMemoryManager(std::size_t page_size, std::size_t slab_size);
  ~SlabPoolMemoryManager() override = default;

  SlabPoolMemoryManager(const SlabPoolMemoryManager&) = delete;
  SlabPoolMemoryManager& operator=(const SlabPoolMemoryManager&) = delete;
  SlabPoolMemoryManager(SlabPoolMemoryManager&&) = delete;
  SlabPoolMemoryManager& operator=(SlabPoolMemoryManager&&) = delete;

  void allocate(const llvm::jitlink::JITLinkDylib* JD, llvm::jitlink::LinkGraph& G,
                OnAllocatedFunction OnAllocated) override;

  void deallocate(std::vector<FinalizedAlloc> Allocs,
                  OnDeallocatedFunction OnDeallocated) override;

  /*! \brief Number of slabs currently held (test introspection). */
  std::size_t numSlabs() const {
    std::lock_guard<std::mutex> lock(pool_mu_);
    return slabs_.size();
  }

 private:
  /*! \brief Reserve a fresh slab at exactly \p capacity bytes.  Returns
   *         nullptr on mmap failure (caller reports the error). */
  std::unique_ptr<Slab> createSlab(std::size_t capacity);

  std::size_t page_size_;
  std::size_t slab_size_;

  mutable std::mutex pool_mu_;
  std::vector<std::unique_ptr<Slab>> slabs_;
};

}  // namespace orcjit
}  // namespace ffi
}  // namespace tvm

#endif  // TVM_FFI_ORCJIT_ORCJIT_MEMORY_MANAGER_H_
