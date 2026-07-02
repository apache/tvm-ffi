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
/*
 * \file tvm_ffi_python_object.h
 * \brief PyObject-tying state machine: binds one Python wrapper to one C++ FFI object
 *        ("chandle") for the object's lifetime so identity is stable (``a.x is a.x``,
 *        stable ``id()`` across drop+refetch, ``f(x) is x`` for FFI returns).
 *
 * Split out of tvm_ffi_python_helpers.h. The design overview is the banner comment below.
 */
#ifndef TVM_FFI_PYTHON_OBJECT_H_
#define TVM_FFI_PYTHON_OBJECT_H_

#include <Python.h>
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/memory.h>

// Define here to avoid dependencies on non-c headers for now
#ifndef TVM_FFI_INLINE
#if defined(_MSC_VER)
#define TVM_FFI_INLINE [[msvc::forceinline]] inline
#else
#define TVM_FFI_INLINE [[gnu::always_inline]] inline
#endif
#endif

// Managed-dict (`__slots__ = ("__dict__",)` without an explicit dictoffset)
// is a CPython 3.11+ feature. On 3.9/3.10 such types instead use a regular
// ``tp_dictoffset != 0``, which the inactive-eligibility check catches anyway,
// so defining the flag as 0 here yields the correct (no-op) behavior.
#ifndef Py_TPFLAGS_MANAGED_DICT
#define Py_TPFLAGS_MANAGED_DICT 0
#endif

#include <atomic>
#include <cassert>
#include <cstring>
#include <utility>

// ``_Interlocked*`` intrinsics for the MSVC arm of the spin-lock leaves below. <intrin.h>, not
// <windows.h>, to keep min/max etc. macros out of the Cython TU.
#if defined(_MSC_VER) && defined(Py_GIL_DISABLED)
#include <intrin.h>
#endif

//================================================================================
// PyObject-tying state machine.
//
// Ties one Python wrapper to one C++ chandle so that
//   - ``a.x is a.x`` while the wrapper is live;
//   - ``id(a.x)`` is stable across drop+refetch (when other C++ holders keep
//     the chandle alive);
//   - ``f(x) is x`` whenever an FFI function returns a chandle that already
//     has a canonical wrapper.
//
// Layout
// ------
// Every Object allocated through the registered Python allocator
// (`TVMFFIPyAllocate`) is preceded by a 16-byte ``PyCustomAllocHeader``:
//
//   malloc start
//   +-------------------+--------------------------+--------+
//   |   tagged_pyobj    | TVMFFIObjectAllocHeader  |   T    |
//   |   (offset 0..8)   |   delete_space (8..16)   |        |
//   +-------------------+--------------------------+--------+
//                                                  ^ ptr = malloc + 16
//
// ``tagged_pyobj`` is a tagged pointer to the canonical Python wrapper. The
// wrapper is >= 16-aligned, so the low 4 bits are free; two encode the state
// (see below) without growing the header past its fixed 16 bytes.
//
// States
// ------
// Bit 0 (Inactive) and bit 1 (InTransit) tag ``tagged_pyobj`` into four states:
//   Detached:  ``tagged_pyobj == NULL`` -- no wrapper bound to this chandle.
//   Active:    ``ptr, bits == 00`` -- the live canonical wrapper.
//   Inactive:  ``ptr | Inactive`` -- dead, untracked allocation cached for
//              address-stable revival (settled).
//   InTransit: ``ptr | Inactive | InTransit`` -- a transition on this binding is in
//              flight (Inactive stays set, so ``TVMFFIPyTagIsInactive`` matches too).
//
// Invariants
// ----------
//   I1. When a PyObject goes out of scope (no Python var refers to it), its
//       +1 on chandle is always released (in ``__dealloc__`` ->
//       ``TVMFFIPyTpDealloc``).
//   I2. When a chandle is destroyed, its cached allocation (if any) is
//       reclaimed.
//   I3'. ``wrapper.chandle`` is only ever a real C++ object pointer or NULL,
//       never a sentinel. A non-NULL chandle owns +1, except inside the
//       wrapper's own dealloc window (where it is kept only as a header locator).
//   I4. Every ``PyObject*`` the Cython side passes to a helper here is a live
//       wrapper (tag bits 0); only this header sets or clears the tag bits.
//   I5. InTransit is never a state of its own: it overlays a non-live binding
//       mid-transition (Inactive(W) or Detached(NULL)), never the live Active
//       wrapper, and a reader that sees it waits the transition out.
//
// The dealloc handshake
// ---------------------
// One allocation can be torn down from two directions, and the handshake stops them
// from racing into a double free or a leak:
//   * from Python -- the wrapper's refcount hits 0, so ``tp_dealloc`` -> ``tp_free`` run;
//   * from C++    -- the chandle's weak count hits 0, so its Weak deleter fires
//                    ``TVMFFIPyDeleteSpace``.
// ``tp_dealloc`` cannot read the chandle refcount to decide which side will be last (an
// FFI ``DecRef`` may race it from another thread), so instead of deciding up front it
// pre-tags ``Inactive | InTransit`` and ``DecRef``s unconditionally; the InTransit bit is
// a baton that whichever side settles last clears, and that side performs the single free.
//
//   Flow 1 -- wrapper dies, chandle outlives it (cache the allocation):
//     tp_dealloc   : Active -> ``Inactive | InTransit``, then DecRef (chandle still has
//                    refs, so no deleter fires).
//     tp_free      : InTransit still set => clear it, keep ``self`` cached Inactive.
//     delete_space : later, when the chandle dies => settled Inactive => reclaim the
//                    cached wrapper and free the block.
//
//   Flow 2 -- wrapper held the last ref (free the allocation now):
//     tp_dealloc   : Active -> ``Inactive | InTransit``, then DecRef drops the last ref.
//     delete_space : fires (same thread, reentrant) => InTransit set => defer the free
//                    back to tp_free and clear it.
//     tp_free      : InTransit cleared => free the C++ block here.
//
// Where transitions happen
// ------------------------
// ``TVMFFIPyMakeRetObject`` (this header), behind ``make_ret_object``
// (object.pxi) -- owns the whole return-object transition in one frame:
//     Detached/Active/Inactive -> Active : fresh / cached / revived-in-place.
//
// ``TVMFFIPyTpDealloc`` (CObject.__dealloc__) -- runs when the wrapper's
// refcount hits 0, before the free:
//     Active   -> Inactive : eligible; tag Inactive | InTransit, DecRef (the
//                            handshake; ``tp_free`` / ``TVMFFIPyDeleteSpace``
//                            settle it).
//     Active   -> Detached : type not eligible; detach first, then DecRef.
//
// ``TVMFFIPyArgSetterObjectRValueRef_`` (function.pxi),
// ``__move_handle_from__`` (object.pxi):
//     Active   -> Detached : detach the binding before a move nulls the
//                            source chandle.
//
// ``TVMFFIPyDeleteSpace`` (Weak deleter) -- the chandle's weak count hit 0:
//     Inactive|InTransit   : in-flight dealloc; defer both frees to ``tp_free``.
//     Inactive (settled)   : reclaim the cached wrapper and free the C++ block.
//
// Slot install
// ------------
// ``tp_alloc`` / ``tp_free`` are NOT inherited by dynamic subtypes (CPython
// resets them per dynamic subtype), so each registered type needs its own
// install. ``_update_registry`` (object.pxi) -- the choke point every
// registered FFI type funnels through -- calls ``TVMFFIPyInstallTypeSlots``
// there, once per type.
//
// Shutdown guard
// --------------
// ``TVMFFIPyMarkPythonFinalizing`` is wired to atexit from Cython module
// init. After it fires, inactive cached allocations on still-live chandles are
// intentionally leaked (process exiting; OS reclaims) rather than reaching
// for ``PyGILState_Ensure`` on a teardown interpreter.
//
// Free-threaded builds (``Py_GIL_DISABLED``)
// ------------------------------------------
// Without the GIL the bare ``tagged_pyobj`` reads/writes above race -- the Active-hit
// read is a use-after-free (``make_ret`` reads the wrapper, a concurrent dealloc frees
// it before the IncRef). The tie stays enabled; three FT-only mechanisms close the gap,
// all behind ``#ifdef Py_GIL_DISABLED`` so the GIL build is byte-for-byte unchanged:
//   * The word is its own spin-lock (a Locked tag bit, CAS-acquired via the portable
//     pointer-atomic leaves -- ``__atomic_*`` on GCC/Clang, ``_Interlocked*`` on MSVC),
//     so every transition serializes its word edits. Details in the word-access leaves.
//   * The Active hit uses ``PyUnstable_TryIncRef`` (inc-if-nonzero), not ``Py_INCREF``,
//     so it fails on a wrapper a concurrent dealloc is collecting -- closing the UAF.
//================================================================================

/*!
 * \brief Python-side derived header. ``base.delete_space`` sits at
 *        ``ptr - sizeof(TVMFFIObjectAllocHeader)`` so the generic C++
 *        deleter (which knows nothing about Python) can find it.
 */
struct PyCustomAllocHeader {
  PyObject* tagged_pyobj;
  TVMFFIObjectAllocHeader base;
};

static_assert(sizeof(PyCustomAllocHeader) == 16,
              "header must be 16 bytes so T at ptr = malloc + 16 is naturally "
              "aligned for alignof(T) up to alignof(max_align_t)");
static_assert(offsetof(PyCustomAllocHeader, base) ==
                  sizeof(PyCustomAllocHeader) - sizeof(TVMFFIObjectAllocHeader),
              "base must sit at ptr - sizeof(TVMFFIObjectAllocHeader) for the "
              "C++ deleter to find it");

TVM_FFI_INLINE PyCustomAllocHeader* TVMFFIPyHeader(void* ptr) {
  return reinterpret_cast<PyCustomAllocHeader*>(static_cast<char*>(ptr) -
                                                sizeof(PyCustomAllocHeader));
}

// Low-bit tags on ``tagged_pyobj`` (wrappers are >= 16-aligned -> low 4 bits free); semantics
// are the States/Invariants above. bit 0 Inactive, bit 1 InTransit, bit 2 Locked (free-threaded
// spin-lock only; the GIL build never sets it). ``TVMFFIPyRemoveTag`` masks every defined bit.
constexpr uintptr_t kPyCachedInactiveTagBit = 1;
constexpr uintptr_t kPyInTransitTagBit = 2;
#ifdef Py_GIL_DISABLED
constexpr uintptr_t kPyLockedTagBit = 4;
constexpr uintptr_t kPyTagBitMask = kPyCachedInactiveTagBit | kPyInTransitTagBit | kPyLockedTagBit;
#else
constexpr uintptr_t kPyTagBitMask = kPyCachedInactiveTagBit | kPyInTransitTagBit;
#endif

TVM_FFI_INLINE bool TVMFFIPyTagIsInactive(PyObject* tagged) {
  return (reinterpret_cast<uintptr_t>(tagged) & kPyCachedInactiveTagBit) != 0;
}
TVM_FFI_INLINE bool TVMFFIPyTagInTransit(PyObject* tagged) {
  return (reinterpret_cast<uintptr_t>(tagged) & kPyInTransitTagBit) != 0;
}
TVM_FFI_INLINE PyObject* TVMFFIPyRemoveTag(PyObject* tagged) {
  return reinterpret_cast<PyObject*>(reinterpret_cast<uintptr_t>(tagged) & ~kPyTagBitMask);
}
// Clear ONLY the InTransit bit (Inactive|InTransit -> Inactive, settled).
TVM_FFI_INLINE PyObject* TVMFFIPyTagClearInTransit(PyObject* tagged) {
  return reinterpret_cast<PyObject*>(reinterpret_cast<uintptr_t>(tagged) & ~kPyInTransitTagBit);
}

//---------------------------------------------------------------
// Word-access leaves: the ONE place the GIL / free-threaded divergence lives. Every
// transition body below (make_ret, the dealloc family, Rebind) is written once against
// this small vocabulary, so the logic reads identically on both builds and the build
// difference is confined here:
//   * lock:   ``TVMFFIPyLockWord`` (acquire, return prior state) / ``TVMFFIPyUnlockWord``
//             (release, publish new state) / ``TVMFFIPyUnlockKeep`` (release unchanged).
//   * other:  ``TVMFFIPyAcquireLoad`` (read without acquiring) / ``TVMFFIPyEnableTryIncRef``
//             (arm a wrapper for a racing reader's TryIncRef before publish) /
//             ``TVMFFIPyLockYield`` (GC-safe back-off, free-threaded only).
//
// Free-threaded build: the word is a spin-lock encoded in ``tagged_pyobj`` (the Locked
// bit), CAS-acquired and release-stored via the portable pointer-atomic helpers
// (``__atomic_*`` on GCC/Clang, ``_Interlocked*`` on MSVC). Held only across short,
// *park-free* sections (no alloc / DecRef / GC op / blocking call), and every wait goes
// through ``TVMFFIPyLockYield`` (detaches the thread state), so the cyclic GC's
// stop-the-world can never freeze a holder nor starve on a waiter.
//
// GIL build: the GIL already serializes every transition, so there is no lock -- each leaf
// collapses to the plain field access the pre-merge code performed (load / store / no-op),
// and the merged bodies emit byte-for-byte unchanged.
//---------------------------------------------------------------

#ifdef Py_GIL_DISABLED
TVM_FFI_INLINE bool TVMFFIPyTagIsLocked(PyObject* tagged) {
  return (reinterpret_cast<uintptr_t>(tagged) & kPyLockedTagBit) != 0;
}

/*! \brief GC-safe back-off for any wait on the word. Must run with an attached thread state
 *         and WITHOUT the word lock held. */
TVM_FFI_INLINE void TVMFFIPyLockYield() {
  PyThreadState* tstate = PyEval_SaveThread();
  PyEval_RestoreThread(tstate);
}

// Portable pointer atomics on ``tagged_pyobj``: ``_Interlocked*`` on MSVC (no ``__atomic_*``
// there), ``__atomic_*`` elsewhere -- same dual-coding the C++ core uses for its refcount word
// (``object.h`` ``TryPromoteWeakPtr`` / ``init_once.cc``). Confining the split to these four
// leaves keeps every transition body below build-agnostic. The relaxed load is only the CAS seed
// (ordering comes from the CAS); the MSVC CAS is strong, the builtin weak -- the spin loop copes.
TVM_FFI_INLINE PyObject* TVMFFIPyWordLoadRelaxed(PyCustomAllocHeader* h) {
#if defined(_MSC_VER)
  return reinterpret_cast<PyObject* const volatile*>(&h->tagged_pyobj)[0];  // NOLINT(*)
#else
  return __atomic_load_n(&h->tagged_pyobj, __ATOMIC_RELAXED);
#endif
}

TVM_FFI_INLINE PyObject* TVMFFIPyWordLoadAcquire(PyCustomAllocHeader* h) {
#if defined(_MSC_VER)
  // CAS NULL/NULL: an acquire-ordered read that never writes (stores only if it already was NULL).
  return reinterpret_cast<PyObject*>(_InterlockedCompareExchangePointer(
      reinterpret_cast<void* volatile*>(&h->tagged_pyobj), nullptr, nullptr));
#else
  return __atomic_load_n(&h->tagged_pyobj, __ATOMIC_ACQUIRE);
#endif
}

TVM_FFI_INLINE void TVMFFIPyWordStoreRelease(PyCustomAllocHeader* h, PyObject* v) {
#if defined(_MSC_VER)
  _InterlockedExchangePointer(reinterpret_cast<void* volatile*>(&h->tagged_pyobj), v);
#else
  __atomic_store_n(&h->tagged_pyobj, v, __ATOMIC_RELEASE);
#endif
}

// Acquire CAS: on match set ``desired`` and return true; else reload ``*expect`` and return false.
TVM_FFI_INLINE bool TVMFFIPyWordCASAcquire(PyCustomAllocHeader* h, PyObject** expect,
                                           PyObject* desired) {
#if defined(_MSC_VER)
  PyObject* prev = reinterpret_cast<PyObject*>(_InterlockedCompareExchangePointer(
      reinterpret_cast<void* volatile*>(&h->tagged_pyobj), desired, *expect));
  if (prev == *expect) return true;
  *expect = prev;
  return false;
#else
  return __atomic_compare_exchange_n(&h->tagged_pyobj, expect, desired, /*weak=*/true,
                                     __ATOMIC_ACQUIRE, __ATOMIC_RELAXED);
#endif
}

/*! \brief Acquire the per-word spin-lock (CAS on the Locked bit). Returns the prior binding
 *         (Locked bit cleared); release it via ``TVMFFIPyUnlockWord`` / ``TVMFFIPyUnlockKeep``. */
TVM_FFI_INLINE PyObject* TVMFFIPyLockWord(PyCustomAllocHeader* h) {
  for (;;) {
    PyObject* cur = TVMFFIPyWordLoadRelaxed(h);
    if (!TVMFFIPyTagIsLocked(cur)) {
      PyObject* locked =
          reinterpret_cast<PyObject*>(reinterpret_cast<uintptr_t>(cur) | kPyLockedTagBit);
      // Acquire on success so the locked section happens-after the matching release.
      if (TVMFFIPyWordCASAcquire(h, &cur, locked)) {
        return cur;
      }
      // CAS failed (lost the race or spurious); ``cur`` reloaded -- retry without
      // yielding, the word was not Locked so contention is brief.
      continue;
    }
    TVMFFIPyLockYield();
  }
}

/*! \brief Release the lock, transitioning the binding to ``new_state``. */
TVM_FFI_INLINE void TVMFFIPyUnlockWord(PyCustomAllocHeader* h, PyObject* new_state) {
  TVMFFIPyWordStoreRelease(h, new_state);
}

/*! \brief Release the lock leaving the binding unchanged. On free-threaded builds this is just
 *         ``TVMFFIPyUnlockWord`` republishing ``cur``; the GIL arm makes it a no-op. */
TVM_FFI_INLINE void TVMFFIPyUnlockKeep(PyCustomAllocHeader* h, PyObject* cur) {
  TVMFFIPyUnlockWord(h, cur);
}

/*! \brief Read the binding without acquiring the lock (a lock-free peek). */
TVM_FFI_INLINE PyObject* TVMFFIPyAcquireLoad(PyCustomAllocHeader* h) {
  return TVMFFIPyWordLoadAcquire(h);
}

/*! \brief Arm ``obj`` for a concurrent reader's ``TryIncRef``; sequence before publishing it
 *         Active. No-op for NULL. ``PyUnstable_EnableTryIncRef`` is part of the free-threading
 *         API introduced in Python 3.14 -- ``PyUnstable_``-prefixed, hence not ABI-stable; only
 *         ever compiled on the free-threaded arm, which is 3.14+ by construction. */
TVM_FFI_INLINE void TVMFFIPyEnableTryIncRef(PyObject* obj) {
  if (obj != nullptr) PyUnstable_EnableTryIncRef(obj);
}

/*! \brief Settle the binding and acquire the right to transition it. Returns with the lock
 *         HELD in both outcomes:
 *           (true,  cur)        Active   -- ``cur`` is the live wrapper, already inc-ref'd.
 *           (false, W|Inactive) Inactive -- ``cur`` is a revivable cached allocation.
 *           (false, NULL)       Detached -- no wrapper bound.
 *           (true,  NULL)       cannot occur.
 *         Waits out any in-flight transition (a dealloc handshake or a peer make_ret's claim,
 *         both marked InTransit) before settling. */
TVM_FFI_INLINE bool TVMFFIPyLockClassifyActive(PyCustomAllocHeader* h, PyObject** out_pyobj) {
  for (;;) {
    PyObject* cur = TVMFFIPyLockWord(h);
    if (TVMFFIPyTagInTransit(cur)) {  // (1) a dealloc or peer make_ret is mid-transition
      TVMFFIPyUnlockKeep(h, cur);
      TVMFFIPyLockYield();
      continue;
    }
    if (cur != nullptr && !TVMFFIPyTagIsInactive(cur)) {  // (2) Active candidate
      // Branch (1) ruled out InTransit and this guard rules out Inactive, so no tag bits are
      // set: ``cur`` is a bare, valid PyObject* -- safe to hand to TryIncRef.
      if (PyUnstable_TryIncRef(cur)) {
        *out_pyobj = cur;
        return true;  // Active hit -- lock HELD, cur inc-ref'd
      }
      TVMFFIPyUnlockKeep(h, cur);  // dying: let its dealloc settle the word, then retry
      TVMFFIPyLockYield();
      continue;
    }
    *out_pyobj = cur;  // (3) Inactive(W) clean, or (4) Detached(NULL)
    return false;      // lock HELD
  }
}

/*! \brief Release the lock publishing the InTransit claim (``cur`` | InTransit) so peers yield
 *         while make_ret allocates/revives lock-free; a no-op on the GIL arm (no peers). */
TVM_FFI_INLINE void TVMFFIPyUnlockPublishInTransit(PyCustomAllocHeader* h, PyObject* cur) {
  // NULL folds to bare InTransit (0 | bit); non-NULL to W|Inactive|InTransit.
  uintptr_t claim = reinterpret_cast<uintptr_t>(cur) | kPyInTransitTagBit;
  TVMFFIPyUnlockWord(h, reinterpret_cast<PyObject*>(claim));
}

#else
// GIL build: the word is a plain field; the GIL is the lock. Each leaf is the exact
// field access the pre-merge code performed (or a no-op where it did nothing).
TVM_FFI_INLINE PyObject* TVMFFIPyLockWord(PyCustomAllocHeader* h) { return h->tagged_pyobj; }
TVM_FFI_INLINE void TVMFFIPyUnlockWord(PyCustomAllocHeader* h, PyObject* new_state) {
  h->tagged_pyobj = new_state;
}
TVM_FFI_INLINE void TVMFFIPyUnlockKeep(PyCustomAllocHeader*, PyObject*) {}  // unchanged: no store
TVM_FFI_INLINE PyObject* TVMFFIPyAcquireLoad(PyCustomAllocHeader* h) { return h->tagged_pyobj; }
TVM_FFI_INLINE void TVMFFIPyEnableTryIncRef(PyObject*) {}  // no TryIncRef synchronizer on the GIL

// make_ret classify (GIL): a plain field read + Py_INCREF on an Active hit. The FT
// InTransit-yield loop has no GIL analog (single-threaded; a reentrant claimer cannot yield
// to a holder up its own stack), so this leaf stays split.
TVM_FFI_INLINE bool TVMFFIPyLockClassifyActive(PyCustomAllocHeader* h, PyObject** out_pyobj) {
  PyObject* cur = h->tagged_pyobj;
  if (cur != nullptr && !TVMFFIPyTagIsInactive(cur)) {  // Active: live canonical wrapper
    Py_INCREF(cur);
    *out_pyobj = cur;
    return true;
  }
  *out_pyobj = cur;  // Inactive(W) or Detached(NULL)
  return false;
}
// No claim under the GIL (no peers to make yield); the word stays at ``cur`` across the alloc.
TVM_FFI_INLINE void TVMFFIPyUnlockPublishInTransit(PyCustomAllocHeader*, PyObject*) {}
#endif  // Py_GIL_DISABLED

/*!
 * \brief Per-thread vehicle carrying the inactive cached allocation address from
 *        ``make_ret_object`` (which knows the chandle) down into
 *        ``TVMFFIPyTpAlloc`` (which is handed only ``type`` and an item count).
 *
 * The slot's sole access primitive is a swap: store ``next``, return the prior
 * value. Taking the block is thus ``TVMFFIPyTLSReviveSlot(nullptr)`` --
 * read-and-clear in one step, with no separate clear to forget. Per-thread ->
 * free-threading safe.
 */
inline PyObject* TVMFFIPyTLSReviveSlot(PyObject* next) {
  static thread_local PyObject* slot = nullptr;
  std::swap(slot, next);
  return next;
}

/*! \brief Arm the cached allocation to be reused by the next ``tp_alloc`` on
 *         this thread. Called by ``make_ret_object`` immediately before
 *         ``cls.__new__``. */
TVM_FFI_INLINE void TVMFFIPySetReviveBlock(PyObject* cached_alloc) {
  TVMFFIPyTLSReviveSlot(cached_alloc);
}

// Forward decl; defined below.
//
// NOTE: deliberately *not* TVM_FFI_INLINE. TVM_FFI_INLINE expands to
// [[gnu::always_inline]] which forbids taking the function's address as
// a stable, callable pointer — and we hand the address to the C++ side
// (stored in PyCustomAllocHeader::base.delete_space at allocate time).
inline void TVMFFIPyDeleteSpace(void* ptr);

// Atexit-driven shutdown guard. ``TVMFFIPyMarkPythonFinalizing`` flips
// the flag to false from an atexit hook registered in Cython module init;
// ``TVMFFIPyDeleteSpace`` reads it via ``TVMFFIPyIsPythonAlive`` before
// ``PyGILState_Ensure`` to avoid touching a teardown interpreter.
inline std::atomic<bool>& TVMFFIPyAliveFlagStorage() {
  static std::atomic<bool> flag{true};
  return flag;
}

inline bool TVMFFIPyIsPythonAlive() noexcept {
  return TVMFFIPyAliveFlagStorage().load(std::memory_order_acquire);
}

inline void TVMFFIPyMarkPythonFinalizing() noexcept {
  TVMFFIPyAliveFlagStorage().store(false, std::memory_order_release);
}

/*!
 * \brief True iff ``chandle`` was allocated through the Python custom
 *        allocator (full ``PyCustomAllocHeader`` ahead of it). False for
 *        allocations that came through libtvm_ffi's builtin default
 *        (only the base ``TVMFFIObjectAllocHeader``).
 *
 * Detection is by comparing ``base.delete_space`` against
 * ``TVMFFIPyDeleteSpace``: each frontend recognizes its own deleter
 * pointer, so multiple frontends can coexist without a flag bit on
 * ``TVMFFIObject``.
 */
TVM_FFI_INLINE bool TVMFFIPyIsCanonical(void* chandle) {
  if (chandle == nullptr) return false;
  TVMFFIObjectAllocHeader* base = reinterpret_cast<TVMFFIObjectAllocHeader*>(
      static_cast<char*>(chandle) - sizeof(TVMFFIObjectAllocHeader));
  return base->delete_space == &TVMFFIPyDeleteSpace;
}

//---------------------------------------------------------------
// Forward declarations shared by SECTION A (make_ret) and the lifecycle sections.
//---------------------------------------------------------------

// Address of a CObject wrapper's ``chandle`` field, defined in object.pxi.
__PYX_EXTERN_C void** TVMFFICyObjectGetCHandlePtr(PyObject* ptr);

inline void TVMFFIPyTpFree(void* self);

//---------------------------------------------------------------
// SECTION A -- alloc / revival / make_ret (HOT: per construction / per FFI return).
//---------------------------------------------------------------

/*!
 * \brief Allocator entry registered with TVMFFISetCustomAllocator at
 *        Cython module init. Allocates ``sizeof(PyCustomAllocHeader) + size`` bytes
 *        with ``alignment``, zero-inits the header to the Detached
 *        state, wires ``base.delete_space = &TVMFFIPyDeleteSpace``, and
 *        returns the T location.
 *
 * Handler::New static_asserts ``alignof(T) <= alignof(max_align_t)``, so
 * the runtime ``alignment`` is bounded and ``base + sizeof(PyCustomAllocHeader)``
 * (= ``base + 16``) lands T naturally aligned for any T we allocate.
 */
inline void* TVMFFIPyAllocate(size_t size, size_t alignment, int32_t /*type_index*/,
                              void* /*context*/) {
  void* base_alloc =
      ::tvm::ffi::details::AlignedAlloc(sizeof(PyCustomAllocHeader) + size, alignment);
  auto* h = static_cast<PyCustomAllocHeader*>(base_alloc);
  h->tagged_pyobj = nullptr;  // Detached
  h->base.delete_space = &TVMFFIPyDeleteSpace;
  return static_cast<char*>(base_alloc) + sizeof(PyCustomAllocHeader);
}

/*! \brief Allocate (fresh) or revive (in place, at ``revive``'s address) a wrapper of
 *         type ``tp`` via ``tp_new``. Returns a new reference (refcount 1) or NULL with
 *         a Python error set. Build-agnostic: touches no word state, only the per-thread
 *         revive slot + ``tp_new``; shared by both builds' ``TVMFFIPyMakeRetObject``. */
inline PyObject* TVMFFIPyNewWrapper(PyTypeObject* tp, PyObject* revive) {
  if (revive != nullptr) TVMFFIPySetReviveBlock(revive);
  PyObject* args = PyTuple_New(0);
  // Near-dead (() is an immortal singleton) but required: tp_new does PyTuple_GET_SIZE(args)
  // unchecked, so a NULL here would segfault rather than report.
  if (args == nullptr) {
    TVMFFIPySetReviveBlock(nullptr);  // disarm: tp_new will not run
    return nullptr;
  }
  PyObject* obj = tp->tp_new(tp, args, nullptr);
  Py_DECREF(args);
  TVMFFIPySetReviveBlock(nullptr);  // defensive: clear if tp_new bypassed tp_alloc
  return obj;
}

/*!
 * \brief Atomically rebind ``chandle``'s canonical PyObject to ``new_object`` iff the current
 *        binding is exactly ``expect`` (Active(expect)) or Detached; otherwise leave the
 *        word untouched. ``new_object == NULL`` clears the binding (Active -> Detached).
 *
 * One compare-and-rebind critical section covers every (re)binding the tie needs:
 *   - move:     ``CompareAndRebind(chandle, other, self)`` -- transfer canonical status.
 *   - construct:``CompareAndRebind(chandle, NULL, self)``  -- attach self iff Detached.
 *   - detach:   ``CompareAndRebind(chandle, obj,  NULL)``  -- clear iff we are the binding.
 * No-op for non-canonical chandles, or when the word is otherwise bound/busy (``new_object``
 * simply does not become canonical -- an identity-only outcome, never a safety issue).
 */
TVM_FFI_INLINE void TVMFFIPyCompareAndRebindPyObject(void* chandle, PyObject* expect,
                                                     PyObject* new_object) {
  if (!TVMFFIPyIsCanonical(chandle)) return;
  PyCustomAllocHeader* h = TVMFFIPyHeader(chandle);
  // Arm before publishing Active, so a racing Active-hit make_ret can safely TryIncRef
  // ``new_object`` (no-op for new_object == NULL and on the GIL build).
  TVMFFIPyEnableTryIncRef(new_object);
  PyObject* cur = TVMFFIPyLockWord(h);
  if (cur == expect || cur == nullptr) {
    TVMFFIPyUnlockWord(h, new_object);  // publish new_object (Active, or Detached when NULL)
  } else {
    TVMFFIPyUnlockKeep(h, cur);  // not ours / busy: release unchanged (GIL: no store)
  }
}

/*!
 * \brief Wrap a returned ``chandle`` into its canonical Python wrapper -- the
 *        whole Detached / Active / Inactive transition in one frame, behind
 *        Cython's ``make_ret_object``.
 *
 * The caller owns +1 (strong) on ``chandle``; ownership transfers to the
 * returned wrapper. Returns a new owned reference, or NULL with a Python error
 * set (the Cython side declares this ``object``, so NULL propagates as an
 * exception).
 *
 * \param chandle The returned object handle (caller owns +1 strong).
 * \param cls_type The wrapper class to instantiate (a ``PyTypeObject*``).
 * \return New owned wrapper reference, or NULL with a Python error set.
 */
// make_ret: one shared body over the four word states, written against the make_ret leaves.
//   Non-canonical  -> fresh wrapper, no tie (FT cannot even locate a header here).
//   Active         -> return the live canonical wrapper (classify inc-ref'd it).
//   Inactive(W)    -> revive W in place at the same address (stable id()).
//   Detached(NULL) -> fresh wrapper, bound canonical.
inline PyObject* TVMFFIPyMakeRetObject(void* chandle, PyObject* cls_type) {
  PyTypeObject* tp = reinterpret_cast<PyTypeObject*>(cls_type);
  // Non-canonical chandle (no Python alloc header, e.g. a C++-static registry object):
  // never tied -- wrap fresh, transferring the caller's +1.
  if (!TVMFFIPyIsCanonical(chandle)) {
    PyObject* obj = TVMFFIPyNewWrapper(tp, nullptr);
    if (obj == nullptr) {  // live OOM/tp_new failure: release the caller's +1 before propagating
      TVMFFIObjectDecRef(chandle);
      return nullptr;
    }
    *TVMFFICyObjectGetCHandlePtr(obj) = chandle;
    return obj;
  }
  PyCustomAllocHeader* h = TVMFFIPyHeader(chandle);
  PyObject* cur;
  // Active hit: return the live canonical wrapper (classify inc-ref'd it); drop caller's +1.
  if (TVMFFIPyLockClassifyActive(h, &cur)) {
    TVMFFIPyUnlockKeep(h, cur);
    TVMFFIObjectDecRef(chandle);
    return cur;
  }
  // Inactive(W) -> revive, Detached(NULL) -> fresh. The InTransit claim opens here and is
  // cleared on every exit below: the word carries the InTransit bit (FT only) across the
  // lock-free allocation so peers yield, and exactly one of the two un-claim sites removes it.
  PyObject* reused_pyobj_space = TVMFFIPyRemoveTag(cur);
  TVMFFIPyUnlockPublishInTransit(h, cur);  // === claim: mark InTransit, drop lock for the alloc ===
  PyObject* obj = TVMFFIPyNewWrapper(tp, reused_pyobj_space);
  if (obj == nullptr) {  // live OOM/tp_new failure: undo the claim before propagating
    TVMFFIPyLockWord(h);
    TVMFFIPyUnlockWord(h, cur);  // un-claim: remove InTransit, restore Inactive(W) / Detached
    TVMFFIObjectDecRef(chandle);
    return nullptr;
  }
  *TVMFFICyObjectGetCHandlePtr(obj) = chandle;  // caller's +1 transfers to obj
  TVMFFIPyEnableTryIncRef(obj);
  TVMFFIPyLockWord(h);
  TVMFFIPyUnlockWord(h, obj);  // un-claim: remove InTransit, publish Active(obj)
  return obj;
}

/*!
 * \brief True iff a wrapper of ``wrapper``'s type may be cached & revived.
 *
 * Requirements (all must hold, else we genuinely free and lose only
 * stable-id-across-drop):
 *  - GC type: revival re-tracks and genuine free / reclaim use GC_Del.
 *  - our custom ``tp_free`` is installed: otherwise the generic free would
 *    reclaim the block while ``tagged_pyobj`` still points at it (UAF).
 *  - no instance ``__dict__`` (plain or managed): reusing a cached allocation whose dict
 *    region was cleared would need dict re-init we do not perform. Lean
 *    wrappers (the common, tested case) have ``tp_dictoffset == 0``.
 *  - no finalizer (``tp_finalize``, i.e. a Python ``__del__``): CPython sets
 *    a permanent GC-finalized bit the first time ``tp_finalize`` runs and
 *    never runs it again on that block. Reusing an inactive cached allocation would silently
 *    suppress ``__del__`` for every revived generation (the cached allocation's bit is
 *    already set and there is no public API to clear it). Excluding these
 *    types makes ``__del__`` fire correctly once per drop (genuine free each
 *    time); the only cost is no stable-id-across-drop.
 */
TVM_FFI_INLINE bool TVMFFIPyIsInactiveEligible(PyObject* wrapper) {
  PyTypeObject* tp = Py_TYPE(wrapper);
  if (!PyType_IS_GC(tp)) return false;
  if (tp->tp_free != &TVMFFIPyTpFree) return false;
  if (tp->tp_dictoffset != 0) return false;
  if ((tp->tp_flags & Py_TPFLAGS_MANAGED_DICT) != 0) return false;
  if (tp->tp_finalize != nullptr) return false;
  return true;
}

/*!
 * \brief Custom ``tp_alloc``. On the revival path (an inactive cached allocation was handed
 *        to this thread via ``TVMFFIPySetReviveBlock``) it revives the cached allocation
 *        in place — same address, so ``id()`` is stable — re-initializing it
 *        to match ``PyType_GenericAlloc``'s contract. Otherwise (miss) it
 *        forwards to ``PyType_GenericAlloc`` for a fresh, tracked object.
 *
 * Revive-path contract (must match what ``tp_new`` expects from
 * ``PyType_GenericAlloc``):
 *  1. zero the body ``[sizeof(PyObject), tp_basicsize)`` so ``__cinit__``
 *     sees clean fields;
 *  2. ``PyObject_Init`` -> ob_refcnt = 1, ob_type, INCREF(type);
 *  3. ``PyObject_GC_Track`` -> GenericAlloc returns a *tracked* object and
 *     ``tp_new`` does not track again, so the revive path must track.
 * (No stale GC-finalized bit to clear: the design uses no tp_finalize.)
 *
 * Fixed-size only: ``PyObject_Init`` resets ``ob_refcnt``/``ob_type`` but not
 * ``ob_size``. Every registered FFI wrapper is a fixed-size cdef class
 * (``tp_itemsize == 0``), asserted below; a future variable-sized type would
 * need ``PyObject_InitVar(.., nitems)`` here and a matching basicsize check.
 */
inline PyObject* TVMFFIPyTpAlloc(PyTypeObject* type, Py_ssize_t nitems) {
  // Take the revive block and leave the slot NULL in one step (per-thread).
  PyObject* blk = TVMFFIPyTLSReviveSlot(nullptr);
  if (blk != nullptr) {
    // REVIVAL: revive the inactive cached allocation at the same address. The body memset
    // below assumes ``type->tp_basicsize`` equals the cached allocation's original
    // basicsize. This holds because a chandle's ``type_index`` maps to one
    // stable wrapper class for the life of the process, and ``make_ret_object``
    // derives both the cached allocation (from the chandle) and ``type`` (= cls for that
    // same type_index) from the very same chandle on the revival path.
    assert(type->tp_itemsize == 0 &&
           "cache-&-revive supports only fixed-size wrappers; a variable-sized "
           "type needs PyObject_InitVar and a per-instance basicsize check");
    std::memset(reinterpret_cast<char*>(blk) + sizeof(PyObject), 0,
                static_cast<size_t>(type->tp_basicsize) - sizeof(PyObject));
    PyObject_Init(blk, type);
    PyObject_GC_Track(blk);
    return blk;
  }
  return PyType_GenericAlloc(type, nitems);  // MISS: fresh (already tracked)
}

//---------------------------------------------------------------
// SECTION B -- dealloc (HOT: per wrapper death).
//---------------------------------------------------------------

// The dealloc binding transition (shared): release the wrapper's +1 on chandle and open
// the cache-vs-free handshake for an eligible canonical wrapper. Three transitions:
//   - Eligible canonical Active binding: Active -> Inactive | InTransit, keep the cached
//     allocation, then DecRef (``tp_free`` settles the InTransit handshake).
//   - Ineligible wrapper: Active -> Detached BEFORE the DecRef, so a deleter firing inside
//     it sees no stale Active back-pointer.
//   - Non-canonical / not-our binding: leave the word, just null ``chandle`` and DecRef.
// The DecRef MUST run outside the lock -- it can fire the chandle deleter (TVMFFIPyDeleteSpace),
// which re-locks the same non-reentrant word. Called from ``__dealloc__`` on both builds.
TVM_FFI_INLINE void TVMFFIPyTpDeallocImpl(void** ptr_to_chandle, PyObject* wrapper) {
  void* chandle = *ptr_to_chandle;
  // Already released by an eager move (detached to NULL); nothing to do. NULL is
  // the only released state here -- the transit marker lives in ``tagged_pyobj``.
  if (chandle == nullptr) return;
  if (TVMFFIPyIsCanonical(chandle)) {
    PyCustomAllocHeader* h = TVMFFIPyHeader(chandle);
    PyObject* cur = TVMFFIPyLockWord(h);
    if (cur == wrapper) {
      if (TVMFFIPyIsInactiveEligible(wrapper)) {
        // Active -> Inactive | InTransit, then DecRef outside the lock.
        TVMFFIPyUnlockWord(
            h, reinterpret_cast<PyObject*>(reinterpret_cast<uintptr_t>(wrapper) |
                                           kPyCachedInactiveTagBit | kPyInTransitTagBit));
        TVMFFIObjectDecRef(chandle);
        return;
      }
      // Active -> Detached: not eligible. Publish NULL before the DecRef so a
      // deleter firing inside it sees no stale Active binding.
      TVMFFIPyUnlockWord(h, nullptr);
    } else {
      TVMFFIPyUnlockKeep(h, cur);  // not our binding -- release unchanged (GIL: no store)
    }
  }
  // Tail (not eligible / not ours / non-canonical): genuine free. Null wrapper.chandle
  // BEFORE DecRef so the deleter chain observes a consistent (chandle == NULL, no +1
  // owed) state.
  *ptr_to_chandle = nullptr;
  TVMFFIObjectDecRef(chandle);
}

/*!
 * \brief Custom ``tp_free``, the second step of the dealloc handshake. The InTransit bit
 *        (read here) says whether the chandle's deleter fired during the
 *        ``TVMFFIPyTpDealloc`` DecRef: still set => the chandle outlived us, settle to
 *        Inactive and keep ``self`` cached; cleared => free the C++ block too. The
 *        ``chandle == NULL`` / non-canonical path is a plain genuine free, dispatching on
 *        GC-ness like CPython's default.
 */
inline void TVMFFIPyTpFree(void* self) {
  void** chandle_ptr = TVMFFICyObjectGetCHandlePtr(static_cast<PyObject*>(self));
  void* chandle = *chandle_ptr;
  if (chandle != nullptr && TVMFFIPyIsCanonical(chandle)) {
    PyCustomAllocHeader* h = TVMFFIPyHeader(chandle);  // header read BEFORE any free below
    PyObject* cur = TVMFFIPyLockWord(h);
    if (TVMFFIPyTagInTransit(cur)) {
      // Case 0: chandle outlived us -- settle to stable Inactive, keep ``self`` cached.
      // ``*chandle_ptr = nullptr`` MUST precede the publish (still under the lock): else a
      // make_ret revive could grab the Inactive word and re-set the chandle, only for this
      // stale NULL to clobber it (Active wrapper with chandle == NULL -> crash).
      *chandle_ptr = nullptr;
      TVMFFIPyUnlockWord(h, TVMFFIPyTagClearInTransit(cur));
      return;
    }
    // Case 1: deleter fired and deferred the block free to us (its sole free).
    *chandle_ptr = nullptr;
    TVMFFIPyUnlockKeep(h, cur);
    ::tvm::ffi::details::AlignedFree(static_cast<char*>(chandle) - sizeof(PyCustomAllocHeader));
  }
  PyObject* op = static_cast<PyObject*>(self);
  if (PyObject_IS_GC(op)) {
    PyObject_GC_Del(op);
  } else {
    PyObject_Free(op);
  }
}

/*!
 * \brief delete_space callback (installed by TVMFFIPyAllocate), invoked from the C++ Weak
 *        deleter when the chandle's block's weak count hits 0. Detached => free the block
 *        (lock-free fast path, the common C++-only-object case). Inactive => read InTransit:
 *        set => an in-flight ``tp_free`` will free the block, so defer; clear => reclaim the
 *        cached wrapper and free the block here.
 *
 * At weak->0 there are no live refs, so the binding is only ever Detached or
 * Inactive(|InTransit) -- never Active/Locked, and never a make_ret claim (a claim holds a
 * strong chandle ref => weak > 0), so any InTransit seen is unambiguously the dealloc
 * handshake's.
 */
inline void TVMFFIPyDeleteSpace(void* ptr) {
  void* base_alloc = static_cast<char*>(ptr) - sizeof(PyCustomAllocHeader);
  auto* h = static_cast<PyCustomAllocHeader*>(base_alloc);
  PyObject* cur0 = TVMFFIPyAcquireLoad(h);
  if (!TVMFFIPyTagIsInactive(cur0)) {  // Detached: lock-free free
    ::tvm::ffi::details::AlignedFree(base_alloc);
    return;
  }
  if (TVMFFIPyIsPythonAlive()) {
    PyGILState_STATE gstate = PyGILState_Ensure();
    if (TVMFFIPyIsPythonAlive()) {
      PyObject* cur = TVMFFIPyLockWord(h);
      if (TVMFFIPyTagInTransit(cur)) {
        // In-flight: defer both frees to tp_free; keep the block.
        TVMFFIPyUnlockWord(h, TVMFFIPyTagClearInTransit(cur));
        PyGILState_Release(gstate);
        return;
      }
      // Settled: detach, reclaim the cached wrapper (outside the word lock), then
      // free the block below.
      PyObject* wrapper = TVMFFIPyRemoveTag(cur);
      TVMFFIPyUnlockWord(h, nullptr);
      PyObject_GC_Del(wrapper);
      PyGILState_Release(gstate);
    } else {
      PyGILState_Release(gstate);
    }
  } else if (TVMFFIPyTagInTransit(cur0)) {
    // Teardown, same-thread in-flight: defer the block free. No lock is held here (Python is
    // finalizing single-threaded, no thread state to lock under) -- this is the one word
    // store in the file with no matching TVMFFIPyLockWord; the release-store is still correct.
    TVMFFIPyUnlockWord(h, TVMFFIPyTagClearInTransit(cur0));
    return;
  }
  ::tvm::ffi::details::AlignedFree(base_alloc);
}

/*!
 * \brief ``__dealloc__`` hook (Cython's ``CObject.__dealloc__`` calls it). Build-agnostic: runs the
 *        binding transition (``TVMFFIPyTpDeallocImpl``) on both the GIL and free-threaded builds.
 */
TVM_FFI_INLINE void TVMFFIPyTpDealloc(void** ptr_to_chandle, PyObject* wrapper) {
  TVMFFIPyTpDeallocImpl(ptr_to_chandle, wrapper);
}

//---------------------------------------------------------------
// SECTION C -- installation (COLD: once per registered type / once per process).
//---------------------------------------------------------------

TVM_FFI_INLINE void TVMFFIPyInstallTypeSlots(PyObject* type_obj) {
  if (type_obj == nullptr || !PyType_Check(type_obj)) return;
  PyTypeObject* tp = reinterpret_cast<PyTypeObject*>(type_obj);
  tp->tp_alloc = &TVMFFIPyTpAlloc;
  tp->tp_free = &TVMFFIPyTpFree;
}

/*!
 * \brief Install ``TVMFFIPyAllocate`` as the process-wide custom allocator.
 *        Storage for the registered entry is a function-static so the
 *        address is process-stable.
 */
TVM_FFI_INLINE int TVMFFIPyRegisterDefaultAllocator() {
  // Installed on both the GIL and free-threaded builds; on free-threaded builds the
  // header state machine is lock-synchronized (see "Free-threaded builds" above).
  static TVMFFICustomAllocator allocator{&TVMFFIPyAllocate, /*context=*/nullptr};
  return TVMFFISetCustomAllocator(&allocator);
}

#endif  // TVM_FFI_PYTHON_OBJECT_H_
