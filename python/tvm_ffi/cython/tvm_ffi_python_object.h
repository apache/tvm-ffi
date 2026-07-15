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
// PyObject-tying state machine
//
// Ties one Python wrapper to one C++ FFI object ("chandle") for the chandle's
// lifetime, so Python identity is stable:
//   - ``a.x is a.x`` while the wrapper is live;
//   - ``id(a.x)`` is stable across drop+refetch (while another C++ holder keeps
//     the chandle alive);
//   - ``f(x) is x`` whenever an FFI call returns a chandle that already has a
//     canonical wrapper.
// Works on both the GIL and free-threaded (``Py_GIL_DISABLED``) builds.
//
// Wrapper vs native object
// ------------------------
// A "wrapper" is the Python object that represents a native TVM-FFI object; it is
// not the native object itself. For ``x = outer.x``:
//
//     Python wrapper W                         Native TVM-FFI object H
//
//     +----------------------+                 +----------------------+
//     | PyObject header      |                 | refcounts            |
//     |   ob_refcnt          |                 | type_index           |
//     |   ob_type            |                 | actual object fields |
//     | CObject.chandle  --------------------> +----------------------+
//     +----------------------+                            |
//                ^                                        |
//                +--------- tagged_pyobj in header <------+
//
// ``W`` is a Cython ``CObject`` subclass (a container, reflected dataclass,
// ``Function``, or other registered class). It provides Python identity, methods,
// and Python reference counting; its ``chandle`` points to the native object
// ``H``, which holds the real FFI data and TVM-FFI's own strong/weak refcounts.
//
// Canonical wrapper
// -----------------
// Without tying, one chandle could back several wrappers -- ``a = outer.x;
// b = outer.x`` gives ``a.same_as(b)`` but ``a is not b``. Tying designates ONE
// wrapper as canonical and records its address in the native allocation's
// ``tagged_pyobj`` field, so ``a is b``. That field is a RAW pointer, not a
// Python reference: storing ``W`` there does not bump ``W.ob_refcnt`` and does
// not keep ``W`` alive.
//
// Memory layout
// -------------
// Every Object allocated through the registered Python allocator
// (``TVMFFIPyAllocate``) is preceded by a fixed 16-byte ``PyCustomAllocHeader``:
//
//     malloc start
//     +-------------------+--------------------------+--------+
//     |   tagged_pyobj    | TVMFFIObjectAllocHeader  |   T    |
//     |   (offset 0..8)   |   delete_space (8..16)   |        |
//     +-------------------+--------------------------+--------+
//                                                    ^ ptr = malloc + 16
//
// The body ``T`` starts at ``malloc + 16``, and ``base.delete_space`` sits at
// ``ptr - sizeof(TVMFFIObjectAllocHeader)`` so the generic C++ deleter (which
// knows nothing about Python) can find it. Objects NOT allocated through this
// allocator (e.g. some C++-static objects) have no header and never tie.
//
// Pointer tagging
// ---------------
// A wrapper allocation is >= 16-byte aligned, so ``tagged_pyobj``'s low bits are
// free to encode lifecycle state without growing the header. If ``W = 0x1000``:
//
//     Active:     0x1000       # W
//     Inactive:   0x1001       # W | 1       (bit 0)
//     InTransit:  0x1003       # W | 1 | 2   (bits 0 and 1)
//     Locked:     0x1004       # W | 4       (bit 2, free-threaded only)
//
// Tagged values are not dereferenceable ``PyObject*``; ``TVMFFIPyRemoveTag`` masks
// every tag bit to recover ``W``.
//
// The four states
// ---------------
// ``tagged_pyobj`` encodes four lifecycle states (Locked, bit 2, is a separate
// free-threaded overlay described below, not a fifth state):
//
//   Detached (``NULL``)
//     No canonical wrapper for this chandle: freshly created and not yet returned
//     to Python, moved-from, an ineligible wrapper that died and was genuinely
//     freed, or allocated-but-not-yet-wrapped. The next FFI return makes a fresh
//     wrapper: Detached -> Active.
//
//   Active (``W``)
//     ``W`` is the live canonical wrapper: ``W.chandle == H`` owns one strong
//     native ref, while the header's back-pointer owns no Python ref. When an FFI
//     call returns ``H``: find ``W``, ``Py_INCREF`` it, drop the redundant native
//     ref from the return, return ``W`` (hence ``f(x) is x``). Exits: -> Inactive
//     (last Python ref gone, another native owner keeps ``H`` alive); -> Detached
//     (move, ineligible-wrapper death, or rebind); -> freed (wrapper and ``H``
//     both end, via the InTransit handshake).
//
//   Inactive (``W | 1``)
//     ``W``'s Python lifetime ended (refcount 0, dealloc ran, untracked from GC,
//     no strong native ref, not yet ``PyObject_GC_Del``'d) but its raw allocation
//     is retained because ``H`` is still alive. A later fetch revives it in place
//     -- zero fields -> ``PyObject_Init`` -> ``PyObject_GC_Track`` -> restore
//     chandle -- giving Inactive -> Active at the SAME address (stable ``id()``).
//     This is a new object lifetime at that address, not resurrection of the old
//     one. If ``H`` dies first, its deleter frees both the cached storage and
//     ``H``.
//
//   InTransit (``W | 1 | 2``)
//     A short-lived teardown baton (see "The dealloc handshake"). It overlays a
//     non-live binding while the Python-side and C++-side teardown settle
//     ownership; it never overlays a live Active wrapper.
//
// State flow:
//
//                            move / ineligible death
//                       +------------------------------+
//                       |                              v
//     Detached ------> Active ------> InTransit ------> Detached/freed
//       ^                |                |
//       |                |                | native object survives
//       |                |                v
//       |                +----------> Inactive
//       |                                 |
//       +---------------------------------+
//                  native object returned again
//
// Invariants
// ----------
//   I1. When a wrapper goes out of scope, its +1 on the chandle is always released
//       (in ``__dealloc__`` -> ``TVMFFIPyTpDealloc``).
//   I2. When a chandle is destroyed, its cached allocation (if any) is reclaimed.
//   I3. ``wrapper.chandle`` is only ever a real C++ object pointer or NULL, never a
//       sentinel. A non-NULL chandle owns +1, except inside the wrapper's own
//       dealloc window (where it is kept only as a header locator).
//   I4. Every ``PyObject*`` the Cython side passes to a helper here is a live
//       wrapper (tag bits 0); only this header sets or clears the tag bits.
//   I5. InTransit is the dealloc handshake's baton and nothing else: it overlays a
//       non-live binding (Inactive(W) or Detached(NULL)) while the two teardown
//       sides settle, never the live Active wrapper. Any reader that sees it -- a
//       peer settler, or make_ret's classify -- waits the transition out.
//
// The dealloc handshake
// ---------------------
// An allocation can be torn down from two directions, and the handshake stops them
// from racing into a double free or a leak:
//   * from Python -- the wrapper's refcount hits 0, so ``tp_dealloc`` ->
//     ``tp_free`` run;
//   * from C++    -- the chandle's weak count hits 0, so its Weak deleter fires
//     ``TVMFFIPyDeleteSpace``.
// ``tp_dealloc`` cannot know which side is last (an FFI ``DecRef`` may race it from
// another thread), so it pre-tags ``Inactive | InTransit`` and ``DecRef``s
// unconditionally. The InTransit bit is a baton: the FIRST settler clears it and
// defers, so the SECOND finds it clear and performs the free.
//
//   Flow 1 -- wrapper dies, chandle outlives it (cache the allocation):
//     tp_dealloc   : Active -> Inactive, InTransit 0 -> 1,
//                    DecRef (chandle still has refs, so no deleter fires)
//     tp_free      : InTransit 1 -> 0, keep ``self`` cached Inactive
//     ... later, the chandle dies:
//     delete_space : InTransit == 0, reclaim the cached wrapper and free the block
//
//   Flow 2 -- wrapper held the last ref (free the allocation now):
//     tp_dealloc   : Active -> Inactive, InTransit 0 -> 1,
//                    DecRef (last ref dropped, so reentrantly fires delete_space)
//     delete_space : InTransit 1 -> 0, defer the free back to tp_free
//     tp_free      : InTransit == 0, free the C++ block here
//
// Where transitions happen
// ------------------------
//   ``TVMFFIPyMakeRetObject`` (behind ``make_ret_object``, object.pxi) owns the
//     whole return-object transition in one frame:
//       Detached / Active / Inactive -> Active (fresh / revived-in-place / cached).
//   ``TVMFFIPyTpDealloc`` (CObject.__dealloc__), when the wrapper refcount hits 0:
//       Active -> Inactive (eligible; tag Inactive | InTransit, then DecRef);
//       Active -> Detached (ineligible; detach first, then DecRef).
//   ``TVMFFIPyArgSetterObjectRValueRef_`` (function.pxi) and
//     ``__move_handle_from__`` (object.pxi):
//       Active -> Detached (detach before a move nulls the source chandle).
//   ``TVMFFIPyDeleteSpace`` (Weak deleter), when the chandle weak count hits 0:
//       Inactive | InTransit -> defer both frees to ``tp_free``;
//       Inactive (settled)   -> reclaim the cached wrapper and free the block.
//
// Slot install
// ------------
// Two slot families, each unmissable over a different scope:
//   * ``tp_dealloc`` (correctness, I1): installed once on ``CObject`` and inherited
//     by every subtype -- nothing to install per type.
//   * ``tp_alloc`` / ``tp_free`` (the cache-&-revive optimization): installed per
//     registered type at the sole registration choke point ``_update_registry``
//     (object.pxi), because these slots are not inherited by dynamic subtypes. An
//     unregistered subtype simply fails eligibility and genuine-frees (losing
//     stable-id-across-drop, not correctness).
// ``tp_dealloc`` works with either pairing: with the custom ``tp_alloc`` /
// ``tp_free`` it caches (eligible), and with the generic ones it detaches +
// genuine-frees -- the eligibility gate keys on the same ``tp_free``, so the two
// can never disagree.
//
// Shutdown guard
// --------------
// ``TVMFFIPyMarkPythonFinalizing`` is wired to atexit from Cython module init.
// After it fires, inactive cached allocations on still-live chandles are
// intentionally leaked (the process is exiting; the OS reclaims) rather than
// reaching for ``PyGILState_Ensure`` on a teardown interpreter.
//
// Free-threaded builds (``Py_GIL_DISABLED``)
// ------------------------------------------
// Without the GIL the bare ``tagged_pyobj`` reads/writes race -- an Active-hit read
// is a use-after-free (``make_ret`` reads the wrapper; a concurrent dealloc frees
// it before the IncRef). The tie stays enabled; two FT-only mechanisms close the
// gap, both behind ``#ifdef Py_GIL_DISABLED`` so the GIL build is byte-for-byte
// unchanged:
//   * The word is its own spin-lock: the Locked bit (bit 2) is CAS-acquired via the
//     portable pointer-atomic leaves (``__atomic_*`` on GCC/Clang, ``_Interlocked*``
//     on MSVC), so every transition serializes its word edits. A holder sets bit 2,
//     reasons about the underlying state with the bit masked off, then publishes the
//     new state without it. The GIL build never sets it.
//   * The Active hit uses ``PyUnstable_TryIncRef`` (inc-if-nonzero), not
//     ``Py_INCREF``, so it fails on a wrapper a concurrent dealloc is collecting.
//     The Active lookup then: lock the word; ``TryIncRef``; if alive, return the
//     wrapper; if its refcount already hit 0, wait for the dealloc to transition the
//     state and retry as Inactive or Detached.
//================================================================================

/*!
 * \brief Python-side derived allocation header, sitting immediately before the
 *        object body (see "Memory layout" in the banner above).
 *
 * \c tagged_pyobj is the tagged pointer to the canonical wrapper (Detached /
 * Active / Inactive / InTransit, plus the free-threaded Locked overlay -- see
 * "Pointer tagging" and "The four states" in the banner). \c base is the generic
 * ``TVMFFIObjectAllocHeader``; its \c delete_space sits at
 * ``ptr - sizeof(TVMFFIObjectAllocHeader)`` so the C++ deleter (which knows
 * nothing about Python) can find it. The ``TVMFFIPyTag*`` helpers below only
 * inspect or transform the encoded word -- they never touch refcounts, allocate,
 * or free.
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

/*! \brief Recover the ``PyCustomAllocHeader`` sitting immediately before an object body.
 *  \param ptr The object (``T``) pointer returned by the allocator.
 *  \return The header at ``ptr - sizeof(PyCustomAllocHeader)``. */
TVM_FFI_INLINE PyCustomAllocHeader* TVMFFIPyHeader(void* ptr) {
  return reinterpret_cast<PyCustomAllocHeader*>(static_cast<char*>(ptr) -
                                                sizeof(PyCustomAllocHeader));
}

// The tag bits on ``tagged_pyobj`` (see "Pointer tagging" in the banner): bit 0 Inactive, bit 1
// InTransit, bit 2 Locked (free-threaded spin-lock only; the GIL build never sets it).
// ``TVMFFIPyRemoveTag`` masks every defined bit.
constexpr uintptr_t kPyCachedInactiveTagBit = 1;
constexpr uintptr_t kPyInTransitTagBit = 2;
#ifdef Py_GIL_DISABLED
constexpr uintptr_t kPyLockedTagBit = 4;
constexpr uintptr_t kPyTagBitMask = kPyCachedInactiveTagBit | kPyInTransitTagBit | kPyLockedTagBit;
#else
constexpr uintptr_t kPyTagBitMask = kPyCachedInactiveTagBit | kPyInTransitTagBit;
#endif

/*! \brief True iff the Inactive bit is set on ``tagged``.
 *  \param tagged The raw ``tagged_pyobj`` word.
 *  \return Whether bit 0 (Inactive) is set. */
TVM_FFI_INLINE bool TVMFFIPyTagIsInactive(PyObject* tagged) {
  return (reinterpret_cast<uintptr_t>(tagged) & kPyCachedInactiveTagBit) != 0;
}
/*! \brief True iff the InTransit bit is set on ``tagged``.
 *  \param tagged The raw ``tagged_pyobj`` word.
 *  \return Whether bit 1 (InTransit) is set. */
TVM_FFI_INLINE bool TVMFFIPyTagInTransit(PyObject* tagged) {
  return (reinterpret_cast<uintptr_t>(tagged) & kPyInTransitTagBit) != 0;
}
/*! \brief Strip every tag bit, yielding the bare wrapper pointer.
 *  \param tagged The raw ``tagged_pyobj`` word.
 *  \return ``tagged`` with all defined tag bits cleared. */
TVM_FFI_INLINE PyObject* TVMFFIPyRemoveTag(PyObject* tagged) {
  return reinterpret_cast<PyObject*>(reinterpret_cast<uintptr_t>(tagged) & ~kPyTagBitMask);
}
/*! \brief Clear ONLY the InTransit bit (Inactive|InTransit -> Inactive, settled).
 *  \param tagged The raw ``tagged_pyobj`` word.
 *  \return ``tagged`` with bit 1 cleared, others kept. */
TVM_FFI_INLINE PyObject* TVMFFIPyTagClearInTransit(PyObject* tagged) {
  return reinterpret_cast<PyObject*>(reinterpret_cast<uintptr_t>(tagged) & ~kPyInTransitTagBit);
}

//---------------------------------------------------------------
// Word-access leaves: the ONE place the GIL / free-threaded divergence lives. Every transition
// body below (make_ret, the dealloc family, Rebind) is written once against the small vocabulary
// below, so the logic reads identically on both builds and the build difference is confined here.
//
// Vocabulary -- two layers, two naming rules, never mixed:
//   * Public ``TVMFFIPy*`` leaves name LOCK SEMANTICS -- what you hold on return:
//       ``Lock...``   -> returns HOLDING the lock + the prior binding (``LockWord``,
//                        ``LockClassifyActive``);
//       ``Unlock...`` -> returns having RELEASED it, publishing the named new state
//                        (``UnlockWord`` = arbitrary, ``UnlockKeep`` = unchanged);
//       ``Peek...``   -> never touches the lock (``PeekWord``, a lock-free read).
//     So "Acquire" as a lock verb never appears up here -- ``PeekWord`` is the lock-free read even
//     though its body uses an acquire-ordered load. Two non-lock helpers complete the set:
//     ``EnableTryIncRef`` (arm a wrapper for a racing reader's TryIncRef before publish) and
//     ``SpinYield`` (GC-safe back-off on a wait; free-threaded only).
//   * Detail ``pyobj_detail::Word*`` leaves name the MEMORY-ORDER MECHANISM --
//     ``Load{Relaxed,Acquire}`` / ``StoreRelease`` / ``CASAcquire``. "Acquire"/"Release" mean the
//     C++ memory order here, and live ONLY in this layer.
//
// Consumers -- three operations. Each ``lock #N`` ... ``unlock #N`` pair brackets one critical
// section; indented branches are the mutually exclusive paths, each "condition: actions".
//
//   Return -- make_ret:
//     LockClassifyActive                                  <- lock #1 (returns HELD)
//     ├─ Active hit : UnlockKeep                          <- unlock #1; drop +1; return live
//     └─ miss       : NewWrapper, then
//                     UnlockWord(obj on ok / cur on OOM)  <- unlock #1 (publish Active / undo)
//
//   Rebind -- CompareAndRebindPyObject (move / construct / detach):
//     LockWord                                            <- lock #2
//     ├─ cur == expect or Detached : UnlockWord(new)      <- unlock #2 (swap the binding)
//     └─ else (not ours / busy)    : UnlockKeep           <- unlock #2 (release unchanged)
//
//   Teardown -- the two-direction handshake (Flow 1 / Flow 2 above); three participants, each its
//   own bracket. tp_dealloc opens (sets InTransit); tp_free / delete_space settle it.
//     tp_dealloc:  LockWord                               <- lock #3
//        ├─ ours + eligible : UnlockWord(Inactive|InTransit)  <- unlock #3 (then DecRef)
//        ├─ ours, ineligible: UnlockWord(NULL)            <- unlock #3  (detach; then DecRef)
//        └─ not ours        : UnlockKeep                  <- unlock #3  (then DecRef)
//     tp_free:     LockWord                               <- lock #4
//        ├─ InTransit == 1 : UnlockWord(clear it)         <- unlock #4  (keep self cached)
//        └─ InTransit == 0 : UnlockKeep ; AlignedFree     <- unlock #4  (we free the block)
//     delete_space: PeekWord                              <- lock-free peek
//        ├─ Detached  : AlignedFree                       <- no bracket (fast path)
//        └─ Inactive  : LockWord                          <- lock #5
//           ├─ InTransit == 1 : UnlockWord(clear it)      <- unlock #5 (defer to tp_free)
//           └─ InTransit == 0 : UnlockWord(NULL) ; reclaim wrapper + free  <- unlock #5
//
// On the GIL build the GIL already serializes every transition, so there is no spin-lock: each
// leaf collapses to a plain field access (load / store / no-op), a lock-free simplification of the
// free-threaded structure that emits byte-for-byte unchanged.
//---------------------------------------------------------------

#ifdef Py_GIL_DISABLED
/*! \brief True iff the Locked spin-lock bit is set on ``tagged`` (free-threaded only).
 *  \param tagged The raw ``tagged_pyobj`` word.
 *  \return Whether bit 2 (Locked) is set. */
TVM_FFI_INLINE bool TVMFFIPyTagIsLocked(PyObject* tagged) {
  return (reinterpret_cast<uintptr_t>(tagged) & kPyLockedTagBit) != 0;
}

/*! \brief GC-safe back-off for any wait on the word (the spin-loop's yield -- not a lock op
 *         itself). Must run with an attached thread state and WITHOUT the word lock held. */
TVM_FFI_INLINE void TVMFFIPySpinYield() {
  PyThreadState* tstate = PyEval_SaveThread();
  PyEval_RestoreThread(tstate);
}

// Raw pointer-atomics on the word -- the only bare loads/stores/CAS (transitions go through
// Lock/Unlock/Peek). Dual-coded ``_Interlocked*`` on MSVC / ``__atomic_*`` elsewhere.
namespace tvm {
namespace ffi {
namespace pyobj_detail {

/*! \brief Relaxed load of the word -- only ever the CAS seed, so a stale value just retries.
 *  \param h The allocation header.
 *  \return The current ``tagged_pyobj`` (relaxed). */
TVM_FFI_INLINE PyObject* WordLoadRelaxed(PyCustomAllocHeader* h) {
#if defined(_MSC_VER)
  return reinterpret_cast<PyObject* const volatile*>(&h->tagged_pyobj)[0];  // NOLINT(*)
#else
  return __atomic_load_n(&h->tagged_pyobj, __ATOMIC_RELAXED);
#endif
}

/*! \brief Acquire load of the word (standalone sync edge). On MSVC a CAS NULL/NULL: an
 *         acquire-ordered read that never writes.
 *  \param h The allocation header.
 *  \return The current ``tagged_pyobj`` (acquire). */
TVM_FFI_INLINE PyObject* WordLoadAcquire(PyCustomAllocHeader* h) {
#if defined(_MSC_VER)
  return reinterpret_cast<PyObject*>(_InterlockedCompareExchangePointer(
      reinterpret_cast<void* volatile*>(&h->tagged_pyobj), nullptr, nullptr));
#else
  return __atomic_load_n(&h->tagged_pyobj, __ATOMIC_ACQUIRE);
#endif
}

/*! \brief Release store of the word, publishing ``v`` to a later acquire.
 *  \param h The allocation header.
 *  \param v The new ``tagged_pyobj`` value to publish. */
TVM_FFI_INLINE void WordStoreRelease(PyCustomAllocHeader* h, PyObject* v) {
#if defined(_MSC_VER)
  _InterlockedExchangePointer(reinterpret_cast<void* volatile*>(&h->tagged_pyobj), v);
#else
  __atomic_store_n(&h->tagged_pyobj, v, __ATOMIC_RELEASE);
#endif
}

/*! \brief CAS, acquire-on-success (not acq_rel -- nothing synchronizes on the Locked-bit store it
 *         publishes).
 *  \param h The allocation header.
 *  \param expect In: value to match; on failure, reloaded to the current word.
 *  \param desired The value to store on a match.
 *  \return True if ``*expect`` matched and ``desired`` stored; else false, ``*expect`` reloaded. */
TVM_FFI_INLINE bool WordCASAcquire(PyCustomAllocHeader* h, PyObject** expect, PyObject* desired) {
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

}  // namespace pyobj_detail
}  // namespace ffi
}  // namespace tvm

/*! \brief Acquire the per-word spin-lock (CAS on the Locked bit); release it via
 *         ``TVMFFIPyUnlockWord`` / ``TVMFFIPyUnlockKeep``.
 *  \param h The allocation header whose word to lock.
 *  \return The prior binding with the Locked bit cleared (the state to reason about while held). */
TVM_FFI_INLINE PyObject* TVMFFIPyLockWord(PyCustomAllocHeader* h) {
  for (;;) {
    PyObject* cur = ::tvm::ffi::pyobj_detail::WordLoadRelaxed(h);
    if (!TVMFFIPyTagIsLocked(cur)) {
      PyObject* locked =
          reinterpret_cast<PyObject*>(reinterpret_cast<uintptr_t>(cur) | kPyLockedTagBit);
      // Acquire on success so the locked section happens-after the matching release.
      if (::tvm::ffi::pyobj_detail::WordCASAcquire(h, &cur, locked)) {
        return cur;
      }
      // CAS failed (lost the race or spurious); ``cur`` reloaded -- retry without
      // yielding, the word was not Locked so contention is brief.
      continue;
    }
    TVMFFIPySpinYield();
  }
}

/*! \brief Release the lock, transitioning the binding to ``new_state``.
 *  \param h The allocation header whose lock to release.
 *  \param new_state The binding to publish (Active wrapper, Inactive|InTransit, or NULL). */
TVM_FFI_INLINE void TVMFFIPyUnlockWord(PyCustomAllocHeader* h, PyObject* new_state) {
  ::tvm::ffi::pyobj_detail::WordStoreRelease(h, new_state);
}

/*! \brief Release the lock leaving the binding unchanged (a no-op on the GIL arm).
 *  \param h The allocation header whose lock to release.
 *  \param cur The unchanged binding to republish (as returned by ``TVMFFIPyLockWord``). */
TVM_FFI_INLINE void TVMFFIPyUnlockKeep(PyCustomAllocHeader* h, PyObject* cur) {
  TVMFFIPyUnlockWord(h, cur);
}

/*! \brief Read the binding WITHOUT taking the lock (a lock-free peek). Acquire-ordered: this
 *         standalone read is the sync edge, so an observed state happens-after its publishing
 *         unlock. (The "acquire" is the memory order, not the lock; hence ``Peek``.)
 *  \param h The allocation header.
 *  \return The current binding (tag bits intact). */
TVM_FFI_INLINE PyObject* TVMFFIPyPeekWord(PyCustomAllocHeader* h) {
  return ::tvm::ffi::pyobj_detail::WordLoadAcquire(h);
}

/*! \brief Arm ``obj`` for a concurrent reader's ``TryIncRef``; sequence before publishing it Active
 *         (no-op for NULL). ``PyUnstable_EnableTryIncRef`` is 3.14+ free-threading API, compiled
 *         only on this arm.
 *  \param obj The wrapper about to be published Active, or NULL. */
TVM_FFI_INLINE void TVMFFIPyEnableTryIncRef(PyObject* obj) {
  if (obj != nullptr) PyUnstable_EnableTryIncRef(obj);
}

/*! \brief Settle the binding and acquire the right to transition it. Returns with the lock
 *         HELD in both outcomes:
 *           (true,  cur)        Active   -- ``cur`` is the live wrapper, already inc-ref'd.
 *           (false, W|Inactive) Inactive -- ``cur`` is a revivable cached allocation.
 *           (false, NULL)       Detached -- no wrapper bound.
 *           (true,  NULL)       cannot occur.
 *         Waits out any in-flight dealloc handshake (marked InTransit) before settling.
 *  \param h The allocation header.
 *  \param out_pyobj Out: the settled binding (inc-ref'd live wrapper / cached alloc / NULL).
 *  \return True on an Active hit (``*out_pyobj`` inc-ref'd); false for Inactive or Detached. */
TVM_FFI_INLINE bool TVMFFIPyLockClassifyActive(PyCustomAllocHeader* h, PyObject** out_pyobj) {
  for (;;) {
    PyObject* cur = TVMFFIPyLockWord(h);
    if (TVMFFIPyTagInTransit(cur)) {  // (1) a dealloc handshake is mid-transition
      TVMFFIPyUnlockKeep(h, cur);
      TVMFFIPySpinYield();
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
      TVMFFIPySpinYield();
      continue;
    }
    *out_pyobj = cur;  // (3) Inactive(W) clean, or (4) Detached(NULL)
    return false;      // lock HELD
  }
}

#else
// GIL build: the word is a plain field; the GIL is the lock. Each leaf is the exact field access
// the pre-merge code performed (or a no-op where it did nothing); see the free-threaded arm above
// for the full per-function contracts these mirror.
/*! \brief GIL arm of ``TVMFFIPyLockWord``: a plain field read (the GIL is the lock). */
TVM_FFI_INLINE PyObject* TVMFFIPyLockWord(PyCustomAllocHeader* h) { return h->tagged_pyobj; }
/*! \brief GIL arm of ``TVMFFIPyUnlockWord``: a plain field store. */
TVM_FFI_INLINE void TVMFFIPyUnlockWord(PyCustomAllocHeader* h, PyObject* new_state) {
  h->tagged_pyobj = new_state;
}
/*! \brief GIL arm of ``TVMFFIPyUnlockKeep``: unchanged binding, so a no-op (no store). */
TVM_FFI_INLINE void TVMFFIPyUnlockKeep(PyCustomAllocHeader*, PyObject*) {}
/*! \brief GIL arm of ``TVMFFIPyPeekWord``: a plain field read. */
TVM_FFI_INLINE PyObject* TVMFFIPyPeekWord(PyCustomAllocHeader* h) { return h->tagged_pyobj; }
/*! \brief GIL arm of ``TVMFFIPyEnableTryIncRef``: no TryIncRef synchronizer on the GIL, a no-op. */
TVM_FFI_INLINE void TVMFFIPyEnableTryIncRef(PyObject*) {}

/*! \brief GIL arm of ``TVMFFIPyLockClassifyActive``: a plain field read + Py_INCREF on an Active
 *         hit. The FT arm's TryIncRef + InTransit-wait loop has no GIL analog (the GIL serializes
 *         everything), so this leaf stays split.
 *  \param h The allocation header.
 *  \param out_pyobj Out: the binding (INCREF'd live wrapper / cached alloc / NULL).
 *  \return True on an Active hit; false for Inactive or Detached. */
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
 *
 * \param next The value to store into the slot (a cached alloc to arm, or NULL to clear).
 * \return The prior slot value (the armed block on a take, else NULL).
 */
inline PyObject* TVMFFIPyTLSReviveSlot(PyObject* next) {
  static thread_local PyObject* slot = nullptr;
  std::swap(slot, next);
  return next;
}

/*! \brief Arm the cached allocation to be reused by the next ``tp_alloc`` on
 *         this thread. Called by ``make_ret_object`` immediately before ``cls.__new__``.
 *  \param cached_alloc The inactive cached allocation to revive on the next ``tp_alloc``. */
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
/*! \brief The process-wide "Python still alive" flag storage (function-static).
 *  \return Reference to the atomic flag (true until finalization begins). */
inline std::atomic<bool>& TVMFFIPyAliveFlagStorage() {
  static std::atomic<bool> flag{true};
  return flag;
}

/*! \brief Whether Python is still alive (finalization has not begun).
 *  \return True until ``TVMFFIPyMarkPythonFinalizing`` has fired. */
inline bool TVMFFIPyIsPythonAlive() noexcept {
  return TVMFFIPyAliveFlagStorage().load(std::memory_order_acquire);
}

/*! \brief Mark Python as finalizing (atexit hook); after this, ``delete_space`` skips the GIL. */
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
 *
 * \param chandle The FFI object handle to test (NULL yields false).
 * \return True iff ``chandle`` carries a ``PyCustomAllocHeader`` (our deleter).
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

/*! \brief Address of a CObject wrapper's ``chandle`` field (defined in object.pxi).
 *  \param ptr The wrapper object.
 *  \return Pointer to its ``chandle`` field. */
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
 *
 * \param size The object body (``T``) size requested by the core allocator.
 * \param alignment The required alignment of ``T``.
 * \param type_index (unused) The FFI type index of the allocation.
 * \param context (unused) The allocator context registered alongside this entry.
 * \return Pointer to the ``T`` body (header prepended, Detached).
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
 *         type ``tp`` via ``tp_new``. Build-agnostic: touches no word state, only the per-thread
 *         revive slot + ``tp_new``; shared by both builds' ``TVMFFIPyMakeRetObject``.
 *
 * On return ``*out_revive_consumed`` reports whether ``revive`` was taken out of the slot by a
 * ``tp_alloc`` (revived in place). The caller needs this on the failure path: a consumed block no
 * longer exists (the failed instance's ``tp_free`` freed it), so the word must NOT be republished
 * as ``Inactive(revive)`` -- that would dangle. May be NULL when ``revive`` is NULL.
 *
 *  \param tp The wrapper type to instantiate.
 *  \param revive An inactive cached allocation to revive in place, or NULL for a fresh alloc.
 *  \param out_revive_consumed Out: set true iff ``revive`` (non-NULL) was consumed by a tp_alloc.
 *  \return A new reference (refcount 1), or NULL with a Python error set. */
inline PyObject* TVMFFIPyNewWrapper(PyTypeObject* tp, PyObject* revive, bool* out_revive_consumed) {
  if (out_revive_consumed != nullptr) *out_revive_consumed = false;
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
  // Clear the slot and observe whether tp_alloc consumed the armed block: a cleared slot
  // (leftover != revive) means the block was revived in place -- and on a tp_new failure its bytes
  // were then freed by tp_free -- so the caller must not republish it as Inactive(revive).
  PyObject* leftover = TVMFFIPyTLSReviveSlot(nullptr);
  if (out_revive_consumed != nullptr) {
    *out_revive_consumed = (revive != nullptr && leftover != revive);
  }
  return obj;
}

/*!
 * \brief Set ``chandle``'s cached canonical PyObject to ``new_object``, but only if the word is
 *        still what the caller expected (``cur == expect``, or Detached); otherwise leave it
 *        untouched. ``new_object == NULL`` clears the binding. No-op for non-canonical chandles.
 *
 * All three callers are the same conditional set -- "make ``new_object`` canonical iff the word is
 * still ``expect``" -- differing only in the arguments:
 *   - construct (object.pxi, ``expect=NULL, new=self``): a fresh wrapper claims a just-constructed
 *     chandle; Detached is expected because nothing else holds this brand-new chandle yet.
 *   - move (object.pxi ``__move_handle_from__``, ``expect=other, new=self``): hand canonical status
 *     from ``other`` to ``self``.
 *   - detach (function.pxi rvalue-ref setter, ``expect=src, new=NULL``): clear the binding before a
 *     move nulls the source chandle.
 *
 * If the word is NOT ``expect`` (a concurrent make_ret or move rebound it first), the set simply
 * no-ops -- ``new_object`` stays a valid wrapper that owns the chandle but is not the canonical
 * one. The only cost is a missed identity share; its ``tp_dealloc`` sees ``cur != wrapper``, takes
 * the not-ours branch, and genuine-frees without touching the cache.
 *
 * \param chandle    FFI object handle whose cached-wrapper word is (re)bound; non-canonical: no-op.
 * \param expect     Rebind only if the current binding equals this (Detached also matches).
 * \param new_object Wrapper to publish as canonical, or NULL to detach.
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
    PyObject* obj = TVMFFIPyNewWrapper(tp, nullptr, nullptr);
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
  // Inactive(W) -> revive at W's address, Detached(NULL) -> fresh. ``classify`` returned the lock
  // HELD and we KEEP it across the alloc, so a peer make_ret just spins on ``LockWord`` until we
  // publish -- one critical section, no InTransit needed here (like the GIL build).
  PyObject* reused_pyobj_space = TVMFFIPyRemoveTag(cur);
  // NOTE: alloc runs UNDER the lock, so a tied ``__cinit__`` must not re-enter the tie on this same
  // chandle (would self-deadlock). Holds today: tied ``__cinit__`` only nulls the chandle field.
  bool revive_consumed = false;
  PyObject* obj = TVMFFIPyNewWrapper(tp, reused_pyobj_space,
                                     &revive_consumed);  // alloc / revive WITH LOCK HELD
  if (obj == nullptr) {
    // OOM/tp_new failure. If a tp_alloc already revived (and thus, on failure, freed) the cached
    // block, ``reused_pyobj_space`` no longer exists: publish Detached, NOT Inactive(W), or the
    // word would dangle at freed bytes (double free when the chandle later dies). If it was
    // untouched (failure before tp_alloc), restore the original word (Inactive(W) / Detached).
    TVMFFIPyUnlockWord(h, revive_consumed ? nullptr : cur);
    TVMFFIObjectDecRef(chandle);
    return nullptr;
  }
  *TVMFFICyObjectGetCHandlePtr(obj) = chandle;  // caller's +1 transfers to obj
  TVMFFIPyEnableTryIncRef(obj);
  TVMFFIPyUnlockWord(h, obj);  // release, publish Active(obj)
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
 *
 * \param wrapper A live wrapper whose type is examined.
 * \return True iff its type may be cached and revived (all requirements above hold).
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
 *
 * \param type The wrapper type being allocated.
 * \param nitems Item count for variable-sized types (0 for our fixed-size wrappers);
 *        forwarded to ``PyType_GenericAlloc`` on the miss path.
 * \return The revived cached allocation (same address) on a hit, else a fresh tracked object.
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

/*!
 * \brief Custom ``tp_free``, the second step of the dealloc handshake. The InTransit bit
 *        (read here) says whether the chandle's deleter fired during the
 *        ``TVMFFIPyTpDealloc`` DecRef: still set => the chandle outlived us, settle to
 *        Inactive and keep ``self`` cached; cleared => free the C++ block too. The
 *        ``chandle == NULL`` / non-canonical path is a plain genuine free, dispatching on
 *        GC-ness like CPython's default.
 *
 * \param self The wrapper being freed (CPython's ``tp_free`` argument).
 */
inline void TVMFFIPyTpFree(void* self) {
  void** chandle_ptr = TVMFFICyObjectGetCHandlePtr(static_cast<PyObject*>(self));
  void* chandle = *chandle_ptr;
  if (chandle != nullptr && TVMFFIPyIsCanonical(chandle)) {
    PyCustomAllocHeader* h = TVMFFIPyHeader(chandle);  // header read BEFORE any free below
    PyObject* cur = TVMFFIPyLockWord(h);
    if (TVMFFIPyTagInTransit(cur)) {
      // Case 0 (Flow 1): chandle outlived us -- settle to stable Inactive, keep ``self`` cached.
      // continuation: delete_space reclaims this block when the chandle later dies.
      // ``*chandle_ptr = nullptr`` MUST precede the publish (still under the lock): else a
      // make_ret revive could grab the Inactive word and re-set the chandle, only for this
      // stale NULL to clobber it (Active wrapper with chandle == NULL -> crash).
      *chandle_ptr = nullptr;
      TVMFFIPyUnlockWord(h, TVMFFIPyTagClearInTransit(cur));
      return;
    }
    // Case 1 (Flow 2): deleter already fired and deferred the block free to us.
    // continuation: none -- we are the last settler and free the block here.
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
 * Inactive(|InTransit) -- never Active/Locked.
 *
 * \param ptr The object (``T``) pointer whose C++ block reached weak-count 0.
 */
inline void TVMFFIPyDeleteSpace(void* ptr) {
  void* base_alloc = static_cast<char*>(ptr) - sizeof(PyCustomAllocHeader);
  auto* h = static_cast<PyCustomAllocHeader*>(base_alloc);
  PyObject* cur0 = TVMFFIPyPeekWord(h);
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
 * \brief ``__dealloc__`` hook (Cython's ``CObject.__dealloc__`` calls it). Build-agnostic: the same
 *        binding transition runs on both the GIL and free-threaded builds.
 *
 * Releases the wrapper's +1 on the chandle and opens the cache-vs-free handshake for an eligible
 * canonical wrapper. Four cases, keyed on the wrapper's relationship to the chandle's binding word:
 *   - chandle already NULL: an eager move (``__move_handle_from__`` / rvalue-ref setter) already
 *     detached and released this wrapper's ref -- nothing owed, return.
 *   - eligible canonical Active binding (``cur == wrapper``): Active -> Inactive | InTransit, keep
 *     the cached allocation for in-place revival, then DecRef (``tp_free`` settles the handshake).
 *   - ineligible canonical Active binding (``cur == wrapper``): Active -> Detached before the
 *     DecRef, then fall through to the genuine-free tail.
 *   - not our binding (``cur != wrapper``) or non-canonical chandle: some other wrapper is (or no
 *     wrapper is) canonical for this chandle -- leave the word untouched and fall through to the
 *     genuine-free tail.
 * The DecRef MUST run outside the lock: it can fire the chandle deleter (``TVMFFIPyDeleteSpace``),
 * which re-locks the same non-reentrant word.
 *
 * \param ptr_to_chandle Address of the wrapper's ``chandle`` field (nulled here before the DecRef).
 * \param wrapper The dying wrapper (used to check it is still the canonical binding).
 */
TVM_FFI_INLINE void TVMFFIPyTpDealloc(void** ptr_to_chandle, PyObject* wrapper) {
  void* chandle = *ptr_to_chandle;
  // Case: chandle already NULL. Released by an eager move (detached to NULL); nothing to do. NULL
  // is the only released state here -- the transit marker lives in ``tagged_pyobj``, not here.
  if (chandle == nullptr) return;
  if (TVMFFIPyIsCanonical(chandle)) {
    PyCustomAllocHeader* h = TVMFFIPyHeader(chandle);
    PyObject* cur = TVMFFIPyLockWord(h);
    if (cur == wrapper) {
      // We ARE the canonical wrapper for this chandle.
      if (TVMFFIPyIsInactiveEligible(wrapper)) {
        // Case: eligible canonical binding. Active -> Inactive | InTransit, then DecRef outside the
        // lock. continuation: the DecRef either leaves the chandle alive (-> tp_free keeps it
        // cached, delete_space frees later: Flow 1) or drops its last ref (-> delete_space fires
        // reentrantly and defers, tp_free frees: Flow 2). See the handshake flows up top.
        TVMFFIPyUnlockWord(
            h, reinterpret_cast<PyObject*>(reinterpret_cast<uintptr_t>(wrapper) |
                                           kPyCachedInactiveTagBit | kPyInTransitTagBit));
        TVMFFIObjectDecRef(chandle);
        return;
      }
      // Case: ineligible canonical binding. Active -> Detached. Publish NULL before the DecRef so a
      // deleter firing inside it sees no stale Active binding; then fall to the genuine-free tail.
      TVMFFIPyUnlockWord(h, nullptr);
    } else {
      // Case: not our binding (``cur != wrapper``). A concurrent make_ret / move made a DIFFERENT
      // wrapper canonical for this chandle (or it is Detached/Inactive). Leave the word as-is --
      // this wrapper only owns its +1 -- and fall to the genuine-free tail. (GIL: no store.)
      TVMFFIPyUnlockKeep(h, cur);
    }
  }
  // Tail (ineligible / not-ours / non-canonical): no handshake -- genuine free. Null
  // ``wrapper.chandle`` BEFORE the DecRef so the deleter chain observes a consistent
  // (chandle == NULL, no +1 owed) state. tp_free then genuine-frees the wrapper storage.
  *ptr_to_chandle = nullptr;
  TVMFFIObjectDecRef(chandle);
}

//---------------------------------------------------------------
// SECTION C -- installation (COLD: once per registered type / once per process).
//---------------------------------------------------------------

/*! \brief Install the custom ``tp_alloc`` / ``tp_free`` (cache-&-revive) slots on ``type_obj``.
 *         Called once per registered FFI type from ``_update_registry`` (object.pxi).
 *  \param type_obj The registered wrapper type (must be a type object; ignored otherwise). */
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
 * \return The status code from ``TVMFFISetCustomAllocator`` (0 on success).
 */
TVM_FFI_INLINE int TVMFFIPyRegisterDefaultAllocator() {
  // Installed on both the GIL and free-threaded builds; on free-threaded builds the
  // header state machine is lock-synchronized (see "Free-threaded builds" above).
  static TVMFFICustomAllocator allocator{&TVMFFIPyAllocate, /*context=*/nullptr};
  return TVMFFISetCustomAllocator(&allocator);
}

#endif  // TVM_FFI_PYTHON_OBJECT_H_
