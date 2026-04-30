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
 * \brief Per-session growable slab pool.
 *
 * `SlabPoolMemoryManager` implements `JITLinkMemoryManager` on top of a
 * per-session `std::vector<std::unique_ptr<Slab>>`.  On each `allocate`
 * it picks the first `Slab` that can fit the graph; if none do, it
 * `mmap`s a fresh slab sized to fit (`Slab::capacityForFootprint`) and
 * appends it.  Normal-size graphs land on a `slab_size`-sized slab;
 * skewed or oversize graphs land on a power-of-2 larger slab whose
 * per-pool budgets cover the graph.
 *
 * ## Lifecycle
 *
 * Once a slab is added to the pool it stays mapped until it is
 * reclaimed or until the pool (and its enclosing session) is
 * destroyed. Individual graphs are deallocated via
 * `FA->owner->deallocateOne(...)`, returning bytes to the slab's free
 * list. Drained slabs can be returned to the OS via
 * `clearFreeSlabs()`.
 *
 * ## GOTPCRELX relaxation workaround
 *
 * See `llvm_patches/gotpcrelx_fix.cc`. The plugin is added per-session
 * to the `ObjectLinkingLayer` alongside this memory manager and is
 * orthogonal to pool growth.
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
 * slabs are added on demand by `allocate()` at a capacity chosen by
 * `Slab::capacityForFootprint` — `slab_size_` for normal graphs, the
 * next power of two up for skewed / oversize graphs.  No retry, no
 * halving on growth; mmap errors propagate.
 */
class SlabPoolMemoryManager : public llvm::jitlink::JITLinkMemoryManager {
 public:
  // Default per-slab capacity.  64 MB is above the p99 size of typical
  // ML JIT graphs (single-kernel bindings, fused kernels), below the
  // PC-relative relocation limit, and a multiple of the 2 MB THP
  // granule. Small enough that a pinned slab only wastes 64 MB of RSS.
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

  void deallocate(std::vector<FinalizedAlloc> Allocs, OnDeallocatedFunction OnDeallocated) override;

  /*! \brief Number of slabs currently held (test introspection). */
  std::size_t numSlabs() const {
    std::lock_guard<std::mutex> lock(pool_mu_);
    return slabs_.size();
  }

  /*!
   * \brief Release drained slabs — zero live allocations and at least one
   *        prior allocation — back to the OS via `munmap`.
   *
   *  Returns the number of slabs reclaimed.  Safe to call any time the
   *  session is quiescent (no concurrent JIT work in flight).  A typical
   *  pattern is to call this after dropping a batch of libraries:
   *
   *      for lib in libs: del lib
   *      session.clear_free_slabs()   # Python API
   *
   *  Fresh slabs that have never been allocated on are preserved — the
   *  session remains ready to accept new JIT work.
   */
  std::size_t clearFreeSlabs();

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
