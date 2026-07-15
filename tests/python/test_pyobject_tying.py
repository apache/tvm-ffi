# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Tests for the PyObject-tying feature (cache & revive).

A 16-byte ``PyCustomAllocHeader`` is prepended to every Object that goes
through the Python custom allocator. Its ``tagged_pyobj`` field is a tagged
pointer to the canonical Python wrapper, encoding a three-state machine
driven by the custom ``tp_alloc`` / ``tp_free`` slots and the
``make_ret_object`` dispatcher in ``tvm_ffi_python_object.h``:

  Detached  no Python wrapper bound to the chandle (raw == NULL)
  Active    canonical wrapper alive; ``wrapper.chandle == chandle`` (tag 0)
  Inactive  wrapper dropped but its memory is cached as a dead,
            untracked allocation (tag 1); revived in place at the same address
            on the next ``make_ret_object``

Active gives ``a.x is a.x``. Inactive gives stable ``id()`` across a
drop-then-rewrap cycle when the C++ object outlives the wrapper — without a
leak: the cached allocation is revived (``tp_alloc``), not resurrected.
"""

from __future__ import annotations

import gc
import itertools
import pickle
import sys
import threading
import weakref
from typing import Any

import pytest
import tvm_ffi
import tvm_ffi.testing
from tvm_ffi import dataclasses as dc

# ---------------------------------------------------------------------------
# Test type registration
# ---------------------------------------------------------------------------
_counter = itertools.count()


def _unique_key(base: str) -> str:
    return f"testing.pyobject_tying.{base}_{next(_counter)}"


def _is_free_threaded_python() -> bool:
    return hasattr(sys, "_is_gil_enabled") and not sys._is_gil_enabled()


# PyObject-tying (wrapper identity stickiness: ``a.x is a.x``, stable ``id()`` across
# drop+revive, ``f(x) is x``) relies on a custom allocator + tp_alloc / tp_free /
# tp_dealloc slots whose header state machine is GIL-serialized on the classic build
# and lock-synchronized on free-threaded builds (see the "Free-threaded builds" note
# in ``tvm_ffi_python_object.h``). The feature is enabled on both, so the identity
# tests below run unconditionally on every build.


# Module-level fixtures so the registered types persist across all tests
# in a file (re-registering with the same key is an error).


@dc.py_class(_unique_key("Inner"))
class Inner(dc.Object):
    val: int


@dc.py_class(_unique_key("Outer"))
class Outer(dc.Object):
    x: Inner


@dc.py_class(_unique_key("MutableOuter"), frozen=False)
class MutableOuter(dc.Object):
    x: Inner


# ---------------------------------------------------------------------------
# Active: identity stable while wrapper is alive
# ---------------------------------------------------------------------------
class TestActiveIdentity:
    """``a.x is a.x`` and ``id(a.x)`` stable while ``a`` lives."""

    def test_attr_access_is_same_object(self) -> None:
        """Repeated (and chained) attribute access returns the same wrapper object."""
        a = Outer(Inner(42))
        assert a.x is a.x
        # id() is stable across many accesses, and chained access reuses the
        # intermediate wrapper rather than minting transient garbage.
        assert len({id(a.x) for _ in range(100)}) == 1
        assert a.x.val == 42

    def test_two_outers_distinct_inners(self) -> None:
        """Distinct C++ objects produce distinct Python wrappers."""
        a = Outer(Inner(1))
        b = Outer(Inner(2))
        assert a.x is not b.x
        assert id(a.x) != id(b.x)


# ---------------------------------------------------------------------------
# Universal cache-on: function returns alias the canonical wrapper
# ---------------------------------------------------------------------------
class TestFunctionReturns:
    """Every FFI return funnels through ``make_ret_object``: the wrapper
    for a chandle that already has a canonical Python wrapper *is* that
    wrapper.

    Note: ``get_global_func("name")`` does NOT participate in this cache
    (most registry entries are allocated at C++ static init, before the
    Python custom allocator is registered, so their chandle has no
    PyCustomAllocHeader). Stable ``id()`` for ``get_global_func`` is
    deferred — see TODO in ``function.pxi::_get_global_func``.
    """

    def test_function_return_aliases_arg(self) -> None:
        """An FFI return aliases the canonical wrapper of its argument."""
        identity = tvm_ffi.convert(lambda x: x)
        x = tvm_ffi.convert([1, 2])
        assert identity(x) is x


# ---------------------------------------------------------------------------
# Active -> Inactive -> Active: identity preserved across wrapper drop
# ---------------------------------------------------------------------------
class TestRevive:
    """When the C++ object outlives the wrapper, re-fetching it via the
    cached field-getter path reuses the preserved memory (same address,
    same chandle, correct value).
    """

    def test_id_chandle_and_value_preserved_across_revive(self) -> None:
        """id(), chandle, and field value all survive a drop-and-revive cycle."""
        outer = Outer(Inner(123))
        first = outer.x  # Active: canonical wrapper installed
        first_id = id(first)
        first_chandle = first.__chandle__()
        del first
        gc.collect()  # transitions to Inactive

        revived = outer.x  # revive at the same address
        assert id(revived) == first_id
        assert revived.__chandle__() == first_chandle
        assert revived.val == 123

    def test_no_leak_churn_2k_cycles(self) -> None:
        """Revive must reuse exactly one address across many cycles."""
        outer = Outer(Inner(1))
        addresses: set[int] = set()
        for i in range(2000):
            ref = outer.x
            addresses.add(id(ref))
            del ref
            if i % 500 == 499:
                gc.collect()
        gc.collect()
        assert len(addresses) == 1, (
            f"Inactive -> Active revive should yield exactly one wrapper address; "
            f"saw {len(addresses)}"
        )

    def test_chandle_dies_while_inactive(self) -> None:
        """A chandle dying while Inactive reclaims the cached allocation safely."""
        # When the chandle finally dies while its wrapper is Inactive,
        # ``TVMFFIPyDeleteSpace`` must reclaim the cached (untracked)
        # allocation under the GIL via ``PyObject_GC_Del`` and then free the
        # malloc block. A regression here typically manifests as a
        # segfault — getting past ``gc.collect()`` and the follow-up
        # alloc means the path stayed safe.
        #
        # A UAF/corruption here is deterministic -- it trips within the first
        # few iterations (the alloc-after-free that surfaces heap damage needs
        # only a handful of repeats), so a small count suffices. Each iter runs
        # two gc.collect()s (~13ms each in this env), so keep the count modest.
        for _ in range(8):
            outer = Outer(Inner(7))
            _ = outer.x  # Detached -> Active
            gc.collect()  # wrapper drops; Active -> Inactive
            del outer  # drops the last C++ ref; deleter fires on the inactive cached allocation
            gc.collect()
        # If we got here, the cached-allocation cleanup path is safe.
        fresh = Outer(Inner(11))
        assert fresh.x.val == 11


# ---------------------------------------------------------------------------
# No unbounded wrapper leak across many distinct chandles
# ---------------------------------------------------------------------------
class TestNoUnboundedLeak:
    """Many distinct chandles, each pinned alive by a parent's field, must
    not accumulate one live Python wrapper per chandle.

    ``TestRevive.test_no_leak_churn_2k_cycles`` cycles a *single* chandle and
    asserts address reuse; this asserts the complementary property -- across
    *many distinct* chandles the count of live wrappers stays flat, because
    cache & revive reuses the cached allocation instead of pinning it (rather
    than keeping one live wrapper alive per chandle for as long as the chandle
    survives).
    """

    def test_distinct_chandles_no_wrapper_leak(self) -> None:
        """Many distinct chandles do not pin one live wrapper each."""

        # Locally-registered types: the gc.get_objects() scan below counts
        # only instances of *this* InnerLeak, so leftover instances from other
        # tests (which use the module-level Inner) cannot pollute the count.
        @dc.py_class(_unique_key("InnerLeak"))
        class InnerLeak(dc.Object):
            val: int

        @dc.py_class(_unique_key("OuterLeak"))
        class OuterLeak(dc.Object):
            x: InnerLeak

        def live_inner_count() -> int:
            return sum(1 for ob in gc.get_objects() if type(ob) is InnerLeak)

        # n is deliberately modest: a leak here would be strictly linear (one
        # pinned wrapper per distinct chandle), so leaked == 0 trips at any n
        # under a regression -- a larger n buys no detection power, only time.
        n = 1000
        parents = []
        gc.collect()
        before = live_inner_count()
        for i in range(n):
            outer = OuterLeak(InnerLeak(i))
            parents.append(outer)  # keep the chandle legitimately alive
            w = outer.x  # Active wrapper for this distinct chandle
            del w  # -> Inactive (revivable cached allocation, must not pin the wrapper)
        gc.collect()
        leaked = live_inner_count() - before
        assert leaked == 0, (
            f"PyObject-tying leak: {leaked} wrappers pinned alive (expected 0; "
            f"one pinned wrapper per distinct chandle == unbounded leak)"
        )


# ---------------------------------------------------------------------------
# Finalizer (__del__) types are excluded from inactivation
# ---------------------------------------------------------------------------
class TestFinalizerNotInactivated:
    """A subclass that defines ``__del__`` gets a ``tp_finalize`` slot, and
    CPython sets a *permanent* GC-finalized bit the first time it runs —
    never running it again on that block. Reviving an inactive cached allocation would
    therefore silently suppress ``__del__`` for every revived generation.
    ``TVMFFIPyIsInactiveEligible`` excludes such types, so they genuinely free on
    every drop and ``__del__`` fires once per generation. The cost is no
    stable-``id``-across-drop for finalizer types (acceptable, documented).
    """

    def test_del_fires_every_generation(self) -> None:
        """__del__ fires on every drop of a finalizer type, not once-ever."""
        calls: list[int] = []

        @dc.py_class(_unique_key("InnerDel"))
        class InnerDel(dc.Object):
            val: int

            def __del__(self) -> None:
                # ``self`` is being finalized; record that __del__ ran.
                calls.append(1)

        @dc.py_class(_unique_key("OuterDel"))
        class OuterDel(dc.Object):
            x: InnerDel

        outer = OuterDel(InnerDel(5))
        for _ in range(5):
            w = outer.x  # fresh wrapper (finalizer type is never inactivated)
            assert w.val == 5
            del w
            gc.collect()  # genuine free -> __del__ runs
        # __del__ fired once per drop (5), not once-ever (the cached-allocation bug).
        assert len(calls) >= 5, (
            f"__del__ must fire on every drop of a finalizer type; saw {len(calls)}"
        )

    def test_finalizer_type_value_correct_across_refetch(self) -> None:
        """A finalizer type's value and live identity hold across refetch."""

        # Even without stable id(), the value and live identity must hold.
        @dc.py_class(_unique_key("InnerDel"))
        class InnerDel2(dc.Object):
            val: int

            def __del__(self) -> None:
                pass

        @dc.py_class(_unique_key("OuterDel"))
        class OuterDel2(dc.Object):
            x: InnerDel2

        outer = OuterDel2(InnerDel2(7))
        assert outer.x is outer.x  # live identity still holds
        assert outer.x.val == 7
        gc.collect()
        assert outer.x.val == 7  # refetch after drop still correct


# ---------------------------------------------------------------------------
# Inactivation only happens when other C++ holders exist
# ---------------------------------------------------------------------------
class TestLastRefCleanFree:
    """When the wrapper is the last C++ ref, ``__dealloc__`` must NOT
    inactivate — it detaches and genuinely frees the allocation (``tp_free`` runs
    the real free). Regression guard for use-after-free in the deleter.
    """

    def test_drop_last_ref_no_crash(self) -> None:
        """The bare convert-and-drop pattern must not segfault."""
        for _ in range(100):
            x = tvm_ffi.convert([1, 2, 3])
            del x
        gc.collect()

    def test_drop_then_realloc_value_correct(self) -> None:
        """Dropping the last ref frees the memory; a fresh alloc is correct."""
        # Inner has no other holder -> __dealloc__ with strong_count == 1 ->
        # genuine free (no inactivation), memory reclaimed. Next Inner is fresh.
        a = Inner(1)
        del a
        gc.collect()
        b = Inner(2)
        assert b.val == 2


# ---------------------------------------------------------------------------
# RValue-ref move path (regression for the test_rvalue_ref segfault)
# ---------------------------------------------------------------------------
class TestRValueRef:
    """The ``_move()`` path nulls the source's ``chandle`` while the
    Python wrapper stays alive. The RValueRef arg setter detaches the
    canonical-wrapper binding eagerly so a downstream cache lookup
    doesn't see a stale back-pointer to a wrapper whose chandle is NULL.
    """

    def test_move_into_callback_no_crash(self) -> None:
        """Moving an arg into a callback that re-moves it does not crash."""
        # Universal cache-on: callback arg aliases caller's ``x`` (one
        # wrapper, one chandle ref). The original blocker was a UAF in
        # the deleter path triggered by the eager-detach + chandle-null
        # interplay during a callback that returns ``x._move()`` — so we
        # exercise that path here, asserting only that we don't crash
        # and the post-call chandle is null.
        use_count = tvm_ffi.get_global_func("testing.object_use_count")

        def callback(x: Any, expected: int) -> Any:
            gc.collect()
            assert expected == use_count(x)
            return x._move()

        f = tvm_ffi.convert(callback)
        x = tvm_ffi.convert([1, 2])
        # Caller-side _move: the rvalue setter detaches and the
        # callback receives a fresh canonical wrapper for the chandle.
        f(x._move(), 1)
        assert x.__ctypes_handle__().value is None

    def test_ffi_move_then_re_fetch_works(self) -> None:
        """Re-fetching a field after an FFI move yields a valid wrapper."""
        # The rvalue setter only runs when an FFI call consumes the
        # ObjectRValueRef. ``discard`` provides that consumption — the
        # setter detaches the canonical-wrapper binding (clearing
        # ``tagged_pyobj`` to NULL) and the C++ side nulls the
        # source wrapper's ``chandle``. ``outer``'s field still owns
        # its strong ref, so re-fetching produces a valid wrapper.
        outer = Outer(Inner(5))
        discard = tvm_ffi.convert(lambda _: None)
        discard(outer.x._move())
        assert outer.x.val == 5

    def test_active_caching_after_ffi_move(self) -> None:
        """Active caching resumes on a fresh wrapper after an FFI move."""
        # Eager-detach during the move clears the header binding, so
        # the chandle returns to Detached. The next access installs a fresh
        # canonical wrapper, which then participates in Active caching
        # like any other freshly-wrapped chandle.
        outer = Outer(Inner(5))
        discard = tvm_ffi.convert(lambda _: None)
        discard(outer.x._move())
        assert outer.x is outer.x
        assert outer.x.val == 5


# ---------------------------------------------------------------------------
# Pickle round-trip: must not corrupt the binding
# ---------------------------------------------------------------------------
class TestPickle:
    """Pickle uses ``__init_handle_by_constructor__`` which calls
    ``_install_chandle_binding`` — the binding must end in a valid
    Active configuration.
    """

    def test_pickle_roundtrip_basic(self) -> None:
        """A basic pickle round-trip preserves the value."""
        a = tvm_ffi.convert([1, 2, 3])
        s = pickle.dumps(a)
        b = pickle.loads(s)
        assert list(b) == [1, 2, 3]

    def test_pickle_roundtrip_preserves_attr_identity(self) -> None:
        """A restored wrapper has stable attribute identity."""
        outer = Outer(Inner(77))
        s = pickle.dumps(outer)
        outer2 = pickle.loads(s)
        # Restored wrapper is fully functional and identity is stable
        # within the new instance.
        assert outer2.x is outer2.x
        assert outer2.x.val == 77


# ---------------------------------------------------------------------------
# PyNativeObject (String / Bytes) — exempt from the header binding
# ---------------------------------------------------------------------------
class TestPyNativeExempt:
    """PyNativeObject types (``String``, ``Bytes``) are value-typed: the
    transient ``Object`` wrapper is discarded after construction.
    They MUST NOT install a header binding.
    """

    def test_string_construction_and_compare(self) -> None:
        """A converted String behaves as a native str."""
        s = tvm_ffi.convert("hello")
        assert isinstance(s, str)
        assert s == "hello"

    def test_bytes_construction_and_compare(self) -> None:
        """A converted Bytes behaves as native bytes."""
        b = tvm_ffi.convert(b"world")
        assert isinstance(b, bytes)
        assert b == b"world"

    def test_string_repeated_construction(self) -> None:
        """Repeated String construction does not corrupt adjacent headers."""
        # Repeatedly constructing strings shouldn't break the header on
        # adjacent objects.
        for _ in range(100):
            s = tvm_ffi.convert("x" * 16)
            assert s == "x" * 16


# ---------------------------------------------------------------------------
# Type-mismatch fallthrough on revive
# ---------------------------------------------------------------------------
class TestTypeMismatchOnRevive:
    """Registering many unrelated types between attribute accesses must
    not corrupt existing wrappers' bindings, and the revive path must
    still hit Inactive for the original wrapper.
    """

    def test_field_access_works_after_many_unrelated_registrations(self) -> None:
        """Unrelated type registrations don't corrupt an existing binding."""
        outer = Outer(Inner(1))
        first_id = id(outer.x)
        # Register a bunch of unrelated classes (no shared chandle).
        for _ in range(20):

            @dc.py_class(_unique_key("Tmp"))
            class Tmp(dc.Object):
                v: int

            _ = Tmp(0)
        gc.collect()
        # Original outer's revive cycle still works.
        revived = outer.x
        assert id(revived) == first_id
        assert revived.val == 1


# ---------------------------------------------------------------------------
# Field setter after revive
# ---------------------------------------------------------------------------
class TestFieldSetterAfterRevive:
    """Replacing a field after a revive cycle should bind the new value
    cleanly without disturbing the prior preserved wrapper for the
    *previous* chandle (which may still be alive elsewhere).
    """

    def test_assign_then_access(self) -> None:
        """Replacing a field after a revive cycle binds the new value cleanly."""
        outer = MutableOuter(Inner(1))
        first_inner = outer.x
        first_chandle = first_inner.__chandle__()
        del first_inner

        # Replace x with a fresh Inner.
        new_inner = Inner(2)
        outer.x = new_inner
        gc.collect()

        # The new outer.x has a different chandle (= new_inner's), so it
        # must produce a wrapper that is NOT the preserved address of
        # the original (which had a different chandle).
        revised = outer.x
        assert revised.val == 2
        assert revised.__chandle__() != first_chandle
        # The new wrapper is canonical for its own chandle.
        assert outer.x is revised


# ---------------------------------------------------------------------------
# Pickle stress: repeated round-trips
# ---------------------------------------------------------------------------
class TestPickleStress:
    """Pickle round-trips go through ``__init_handle_by_constructor__``
    which calls ``_install_chandle_binding``. Repeated round-trips must
    not leak wrappers or corrupt the binding state.
    """

    def test_many_roundtrips(self) -> None:
        """Repeated pickle round-trips keep the binding state valid."""
        for _ in range(200):
            outer = Outer(Inner(7))
            restored = pickle.loads(pickle.dumps(outer))
            assert restored.x.val == 7
            assert restored.x is restored.x  # Active on restored binding

    def test_roundtrip_then_revive(self) -> None:
        """A pickle-installed binding survives a finalize-and-revive cycle."""
        # Round-trip, then exercise Inactive -> Active revive on the restored
        # wrapper to confirm the binding installed by pickle survives
        # finalize+rewrap.
        outer = pickle.loads(pickle.dumps(Outer(Inner(99))))
        first_id = id(outer.x)
        gc.collect()
        assert id(outer.x) == first_id
        assert outer.x.val == 99


# ---------------------------------------------------------------------------
# Threading stress: concurrent attribute access
# ---------------------------------------------------------------------------
class TestThreadingStress:
    """Multi-threaded smoke test for the cache-hit / Inactive revive
    paths. Under classic CPython the GIL serializes Python bytecode so
    most races are confined to the FFI call's nogil block; this test
    exercises enough iterations to surface obvious crashes from any
    GIL-released DecRef on the C++ side.
    """

    def test_concurrent_shared_churn_no_crash(self) -> None:
        """Many threads hammering ``outer.x`` on ONE shared object must not crash.

        This is the high-pressure free-threaded reproducer: a single ``Outer`` is
        shared across all threads, so every thread's ``outer.x`` drop+refetch races
        the others' through the same ``tagged_pyobj`` word. The cached ``Inner``
        wrapper churns Active <-> Inactive while ``make_ret`` Active-hits read it and
        ``tp_dealloc`` mutates it. On free-threaded builds with no synchronization
        this segfaults within ~1s; it is the exact race the lock-synchronized state
        machine plus the pre-bump ``tp_dealloc`` binding-clear close. Also asserts the
        live wrapper identity (``outer.x is outer.x``) holds under contention.
        """
        outer = Outer(Inner(123))
        # More threads than cores maximizes interleaving; iteration count is sized to
        # surface the crash reliably on FT while keeping the GIL build well under a
        # second. Drop the count a bit on the GIL build (the race cannot occur there).
        nthreads = 16
        iters = 20_000 if _is_free_threaded_python() else 2_000
        barrier = threading.Barrier(nthreads)
        errors: list[BaseException] = []

        def worker() -> None:
            try:
                barrier.wait()
                for _ in range(iters):
                    a = outer.x  # Detached/Inactive -> Active, or Active hit
                    b = outer.x  # Active hit on the just-installed wrapper
                    assert a is b, "live wrapper identity must hold under contention"
                    assert a.val == 123
                    del a, b
            except BaseException as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(nthreads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, f"worker(s) raised: {errors!r}"


# ---------------------------------------------------------------------------
# Carrier family: tying covers all CObject subtypes, not just dataclasses
# ---------------------------------------------------------------------------
class TestCarrierTypes:
    """``tp_dealloc`` is defined once on ``CObject`` and inherited by every
    subtype (``Function``, ``Error``, heap subclasses, ``OpaquePyObject``), so
    tying and clean teardown must hold for the whole family, not only the
    dataclass ``Inner`` path. Each carrier is pinned alive and re-fetched; the
    Active-hit aliases the canonical wrapper.
    """

    def test_function_and_error_carrier_alias(self) -> None:
        """Function and Error carriers re-fetched from a container alias one wrapper."""
        fn_box = tvm_ffi.convert([tvm_ffi.convert(lambda z: z)])
        assert fn_box[0] is fn_box[0]
        err_box = tvm_ffi.convert([tvm_ffi.convert(ValueError("boom"))])
        assert err_box[0] is err_box[0]

    def test_multilevel_heap_subtype_alias(self) -> None:
        """A 2-level heap dataclass subtype ties like any other CObject."""

        # ``Derived``'s tp_dealloc is CPython's subtype_dealloc, which walks to
        # the inherited CObject slot -- so tying still applies.
        @dc.py_class(_unique_key("CarrierBase"))
        class Base(dc.Object):
            a: int

        @dc.py_class(_unique_key("CarrierDerived"))
        class Derived(Base):
            pass

        @dc.py_class(_unique_key("CarrierHolder"))
        class Holder(dc.Object):
            x: Base

        holder = Holder(Derived(9))
        assert holder.x is holder.x
        assert holder.x.a == 9


# ---------------------------------------------------------------------------
# OpaquePyObject carrier
# ---------------------------------------------------------------------------
class TestOpaquePyObjectCarrier:
    """``OpaquePyObject`` carries an arbitrary (non-FFI) Python value across the
    FFI boundary. Its round-trip must preserve identity and its teardown must
    release the held value (the inherited ``CObject`` dealloc runs the chandle
    DecRef), so payloads are reclaimed rather than leaked.
    """

    def test_opaque_roundtrip_preserves_identity(self) -> None:
        """A Python value round-tripped through an FFI echo is the same object."""
        echo = tvm_ffi.convert(lambda x: x)

        class Payload:
            __slots__ = ("v",)

            def __init__(self, v: int) -> None:
                self.v = v

        p = Payload(7)
        r = echo(p)
        assert r is p
        assert r.v == 7

    def test_opaque_roundtrip_does_not_leak(self) -> None:
        """Echoed payloads are reclaimed after drop — the carrier DecRef runs."""
        echo = tvm_ffi.convert(lambda x: x)

        class Payload:
            __slots__ = ("__weakref__", "v")

            def __init__(self, v: int) -> None:
                self.v = v

        refs: list[weakref.ref[Any]] = []
        for i in range(50):
            p = Payload(i)
            wr = weakref.ref(p)
            r = echo(p)  # p crosses the FFI as an OpaquePyObject
            assert r is p
            del r, p
            refs.append(wr)
        gc.collect()
        leaked = sum(1 for wr in refs if wr() is not None)
        assert leaked == 0, f"OpaquePyObject carrier leaked {leaked}/50 payloads"


# ---------------------------------------------------------------------------
# GC integration with Inactive
# ---------------------------------------------------------------------------
class TestReviveWithCyclicGC:
    """Inactive cached allocations are dead, untracked memory (``subtype_dealloc``
    untracks before inactivation, and ``tp_alloc`` re-tracks only on revival).
    ``gc.collect()`` between a drop and a refetch must not crash: it must
    neither traverse the untracked cached allocation nor trip over the revived object.
    """

    def test_gc_collect_inside_revive_loop(self) -> None:
        """gc.collect() between drop and refetch does not crash."""
        outer = Outer(Inner(5))
        # A bad traversal of the untracked cached allocation crashes deterministically on
        # the first collect, so a modest count suffices; the cost here is the
        # gc.collect() itself (~13ms each in this env), not the FFI work.
        for _ in range(10):
            ref = outer.x
            del ref
            gc.collect()  # cached allocation is inactive (untracked) between drops
        assert outer.x.val == 5

    def test_gc_collect_with_holder_keeping_chandle(self) -> None:
        """gc.collect() around inactive memory keeps id() stable."""
        # The Inner chandle is held by ``outer``'s field for the lifetime
        # of the test, so wrapper drops always inactivate the allocation. gc.collect
        # between drops exercises GC stability around the inactive memory.
        outer = Outer(Inner(11))
        first_id = id(outer.x)
        # GC stability around the inactive cached allocation is deterministic per cycle;
        # a modest count surfaces it. Dominated by gc.collect() (~13ms each).
        for _ in range(10):
            _ = outer.x
            gc.collect()
        assert id(outer.x) == first_id


# ---------------------------------------------------------------------------
# Isolation: many distinct chandles each Inactive at once
# ---------------------------------------------------------------------------
class TestMultipleChandlesIsolation:
    """A revive on one chandle must not corrupt the cached binding on
    another. Each chandle's header is independent — verify by cycling
    many in parallel.
    """

    def test_distinct_chandles_revive_independently(self) -> None:
        """Each chandle revives to its own address and value independently."""
        outers = [Outer(Inner(i * 10)) for i in range(50)]
        # Capture Active addresses, then drop to Inactive.
        b_ids = [id(o.x) for o in outers]
        gc.collect()
        # Revive each independently; each must return its own preserved
        # address with its own value.
        for i, o in enumerate(outers):
            revived = o.x
            assert id(revived) == b_ids[i]
            assert revived.val == i * 10

    def test_interleaved_revives_do_not_cross_contaminate(self) -> None:
        """Reviving in a different order keeps ids matched per-chandle."""
        # Build N outers, capture ids, drop all, then revive in a
        # different order. Ids must still match per-chandle.
        outers = [Outer(Inner(i)) for i in range(30)]
        ids = [id(o.x) for o in outers]
        gc.collect()
        for i in reversed(range(30)):
            assert id(outers[i].x) == ids[i]


# ---------------------------------------------------------------------------
# Weakref limitation (sentinel)
# ---------------------------------------------------------------------------
class TestWeakrefNotSupported:
    """``CObject`` is a Cython cdef class without a ``__weakref__`` slot,
    so weakrefs are not supported. This test documents that limitation:
    if it ever starts passing, the Inactive-state interaction needs
    design review (a weakref to an Active wrapper transitioning to Inactive
    must behave correctly — the wrapper genuinely dies at refcount 0, so
    old weakrefs should observe death and a revived wrapper should bind a
    fresh weakref).
    """

    def test_weakref_raises_typeerror(self) -> None:
        """Taking a weakref to a wrapper raises TypeError (documented limit)."""
        outer = Outer(Inner(1))
        with pytest.raises(TypeError):
            weakref.ref(outer.x)
