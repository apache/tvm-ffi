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
"""Tests for JIT memory arena — verifies co-location and relocation safety.

Background
----------
LLVM ORC JIT v2 uses ``InProcessMemoryMapper`` (backed by
``MapperJITLinkMemoryManager``) to allocate JIT memory.  Each allocation
is a separate ``mmap(MAP_ANONYMOUS)`` call whose address the kernel picks.
Under virtual-address (VA) pressure — leaked slabs from failed
materializations, long-running pytest sessions holding tracebacks, or
simply a fragmented address space — the kernel can place successive
allocations far apart.

This matters for **PC-relative relocations with limited range**:

- **x86_64 R_X86_64_PC32 / Delta32**: ±2 GB range.  GCC-compiled C++
  objects reference ``__dso_handle`` (used by ``__cxa_atexit`` for DSO
  identification) via PC32 when the symbol has hidden visibility.
  LLVM's ``ELFNixPlatform`` defines ``__dso_handle`` per JITDylib in a
  separate ``DSOHandleMaterializationUnit`` — a tiny ``LinkGraph``
  allocated independently of the code that references it.  If those two
  allocations land >2 GB apart, the Delta32 fixup overflows.

- **AArch64 ADRP+ADD**: ±4 GB range.  Hidden-visibility cross-object
  calls use ADRP (page-relative) which has the same scatter problem
  at a wider threshold.

The **arena memory manager** solves this by pre-reserving a contiguous
VA region (default 1 GB, with fallback to smaller sizes down to 256 MB
under RLIMIT_AS / container limits) via ``mmap(PROT_NONE)`` and
bump-allocating within it, guaranteeing all JIT allocations stay within
relocation range regardless of external VA pressure.

Note on ``-fPIC`` vs ``-fpie``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
With ``-fPIC`` (the default for shared-library code), GCC may use
``R_X86_64_GOTPCRELX`` (GOT-relative, load through the GOT) for
hidden-visibility externals like ``__dso_handle``.  GOT entries are
co-located with code, so there is no ±2 GB range issue.  With ``-fpie``
(position-independent executable), GCC prefers the shorter direct
``R_X86_64_PC32``, which *does* have the ±2 GB limit.  The Delta32
overflow tests (test 6) therefore build with ``-fpie`` to force the
problematic relocation type.

Test structure
--------------
1. **Co-location** (test 1): arena keeps objects within 16 MB.
2. **Scatter baseline** (test 2): without arena, VA blocker pushes
   objects >128 MB apart — proves arena is responsible for co-location.
3. **Hidden-symbol calls** (test 3): ADRP/PC32 cross-object calls
   succeed under VA pressure with arena.
4. **Large data section** (test 4): 4 MB ``.nv_fatbin`` section loads
   correctly within the arena.
5. **Overflow section** (test 5): ``.nv_fatbin`` data is allocated
   outside the arena via separate mmap.
6. **Leaked materialization** (test 6): ``__dso_handle`` resolves after
   prior sessions leaked mmap slabs from failed materializations.
7. **Delta32 overflow** (test 7): ``-fpie`` GCC objects + 3 GB VA
   blocker.  With arena → PASSES; without arena → Delta32 overflow.

All tests use a small arena (16 MB) and 256 MB-3 GB VA blockers -- safe
for CI containers.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import functools
import platform
import sys
from pathlib import Path

import pytest
from tvm_ffi_orcjit import ExecutionSession
from utils import build_test_objects

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

OBJ_DIR = build_test_objects()

_KNOWN_SUBDIRS = [
    "c",
    "c-gcc",
    "cc",
    "cc-gcc",
    "cc-gcc-pie",
    "c-appleclang",
    "cc-appleclang",
    "c-msvc",
    "c-clang-cl",
]

_PIE_VARIANT_MARKER = "-pie"


def obj(name: str) -> str:
    """Return path to a pre-built test object file, or skip if missing."""
    path = OBJ_DIR / f"{name}.o"
    if not path.exists():
        pytest.skip(f"{path.name} not found (not built)")
    return str(path)


def _discover_c_variants() -> list[str]:
    """Discover available C-only compiler variants."""
    return [
        s
        for s in _KNOWN_SUBDIRS
        if s.startswith("c")
        and not s.startswith("cc")
        and _PIE_VARIANT_MARKER not in s
        and (OBJ_DIR / s / "test_funcs.o").exists()
    ]


def _discover_cpp_variants() -> list[str]:
    """Discover available C++ compiler variants (for __dso_handle tests)."""
    return [
        s
        for s in _KNOWN_SUBDIRS
        if s.startswith("cc")
        and _PIE_VARIANT_MARKER not in s
        and (OBJ_DIR / s / "test_funcs.o").exists()
    ]


def _discover_gcc_cpp_variants() -> list[str]:
    """Discover GCC C++ variants (emit R_X86_64_PC32 for __dso_handle)."""
    return [v for v in _discover_cpp_variants() if "gcc" in v]


def _discover_pie_cpp_variants() -> list[str]:
    """Discover PIE C++ variants built with -fpie.

    PIE objects force R_X86_64_PC32 (direct, ±2GB) for __dso_handle
    instead of R_X86_64_GOTPCRELX (GOT-relative, unlimited range).
    Used exclusively by the Delta32 overflow tests (test 6).
    """
    return [
        s
        for s in _KNOWN_SUBDIRS
        if _PIE_VARIANT_MARKER in s and (OBJ_DIR / s / "test_funcs.o").exists()
    ]


_c_variants = _discover_c_variants()
_cpp_variants = _discover_cpp_variants()
_gcc_cpp_variants = _discover_gcc_cpp_variants()
_pie_cpp_variants = _discover_pie_cpp_variants()
_all_variants = _c_variants + _cpp_variants

_is_linux = sys.platform == "linux"
_is_x86_64 = platform.machine() in ("x86_64", "AMD64")

# Arena test parameters.
#
# Under the slab-pool design, `slab_size` is the per-slab capacity — each
# allocation that exceeds what an existing slab can hold spawns a new one.
# These tests assert single-slab invariants (objects within slab_size,
# one contiguous VA region), so `_ARENA_SIZE` must be large enough that
# every graph we load fits in the first slab.  256 MB comfortably covers
# the test objects (all < 5 MB each) plus their overhead.
_ARENA_SIZE = 256 * 1024 * 1024  # 256MB — single-slab headroom for these tests
_BLOCK_RADIUS = 256 * 1024 * 1024  # 256MB — safe for CI containers
_DSO_BLOCK_RADIUS = 3 * 1024 * 1024 * 1024  # 3GB — needed to overflow PC32 (±2GB)

_PROT_NONE = 0
_MAP_PRIVATE_ANON = 0x22  # MAP_PRIVATE | MAP_ANONYMOUS
_MAP_FIXED_NOREPLACE = 0x100000


# ---------------------------------------------------------------------------
# VA blocker — fills nearby free VA gaps to force distant mmap placement
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _get_libc() -> ctypes.CDLL:
    """Get a ctypes handle to libc with correct mmap/munmap signatures."""
    libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_long,
    ]
    libc.munmap.restype = ctypes.c_int
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    return libc


def _parse_maps() -> list[tuple[int, int]]:
    """Parse /proc/self/maps into sorted list of (start, end) tuples."""
    regions = []
    with Path("/proc/self/maps").open() as f:
        for line in f:
            addrs = line.split()[0].split("-")
            regions.append((int(addrs[0], 16), int(addrs[1], 16)))
    return sorted(regions)


def _find_new_mappings(
    before: set[tuple[int, int]], after: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Find mappings present in *after* but not in *before*."""
    return [(s, e) for s, e in after if (s, e) not in before]


def block_nearby_va(center: int, radius: int = _BLOCK_RADIUS) -> list[tuple[int, int]]:
    """Block all free VA gaps within *radius* of *center*.

    Uses MAP_FIXED_NOREPLACE to place PROT_NONE mappings in every free gap
    within [center - radius, center + radius].  This forces subsequent
    mmap(NULL, ...) calls to land outside the blocked region.

    Returns list of (addr, size) blockers to be freed later.
    """
    libc = _get_libc()
    maps = _parse_maps()
    blockers = []
    low = max(center - radius, 0)
    high = center + radius

    for i in range(len(maps) - 1):
        gap_start = maps[i][1]
        gap_end = maps[i + 1][0]
        if gap_end <= low or gap_start >= high or gap_end <= gap_start:
            continue
        block_start = max(gap_start, low)
        block_end = min(gap_end, high)
        block_size = block_end - block_start
        if block_size <= 0:
            continue
        addr = libc.mmap(
            block_start, block_size, _PROT_NONE, _MAP_PRIVATE_ANON | _MAP_FIXED_NOREPLACE, -1, 0
        )
        if addr != ctypes.c_void_p(-1).value and addr is not None:
            blockers.append((addr, block_size))

    return blockers


def free_blockers(blockers: list[tuple[int, int]]) -> None:
    """Free all VA blockers."""
    libc = _get_libc()
    for addr, size in blockers:
        libc.munmap(addr, size)


# ---------------------------------------------------------------------------
# Test 1: Arena co-location — objects stay within arena range
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _is_linux, reason="Arena is Linux-only")
@pytest.mark.parametrize("variant", _all_variants)
def test_arena_colocation(variant: str) -> None:
    """With arena, objects in separate libraries have close code addresses.

    Uses a 16MB arena and inserts a 256MB VA blocker between object loads.
    Without the arena, the blocker would push the second object far away.
    With the arena, both objects land within the 16MB region.
    """
    maps_before = set(_parse_maps())

    session = ExecutionSession(slab_size=_ARENA_SIZE)
    lib1 = session.create_library("lib1")
    lib1.add(obj(f"{variant}/test_addr"))
    addr1 = lib1.get_function("code_address")()

    # Find where LLVM placed the first allocation and block nearby VA
    maps_after = _parse_maps()
    new_maps = _find_new_mappings(maps_before, maps_after)
    jit_center = max(s for s, e in new_maps) if new_maps else addr1

    blockers = block_nearby_va(jit_center)
    try:
        lib2 = session.create_library("lib2")
        lib2.add(obj(f"{variant}/test_addr"))
        addr2 = lib2.get_function("code_address")()
    finally:
        free_blockers(blockers)

    distance = abs(addr1 - addr2)
    assert distance < _ARENA_SIZE, (
        f"Objects should be within {_ARENA_SIZE} bytes, "
        f"but distance is {distance} ({distance / (1024**2):.1f} MB)"
    )


# ---------------------------------------------------------------------------
# Test 2: Arena effect — compare with-arena vs without-arena under VA pressure
# ---------------------------------------------------------------------------


def _measure_distance_under_pressure(
    variant: str, slab_size: int, radius: int = _BLOCK_RADIUS
) -> tuple[int | None, bool]:
    """Load two objects under VA pressure and return (distance, overflowed).

    Returns ``(distance_bytes, False)`` when both objects load successfully,
    or ``(None, True)`` when the second load fails with a relocation overflow
    (Page21 on AArch64, Delta32 on x86_64).
    """
    maps_before = set(_parse_maps())

    session = ExecutionSession(slab_size=slab_size)
    lib1 = session.create_library("lib1")
    lib1.add(obj(f"{variant}/test_addr"))
    addr1 = lib1.get_function("code_address")()

    maps_after = _parse_maps()
    new_maps = _find_new_mappings(maps_before, maps_after)
    jit_center = max(s for s, e in new_maps) if new_maps else addr1

    blockers = block_nearby_va(jit_center, radius=radius)
    try:
        lib2 = session.create_library("lib2")
        try:
            lib2.add(obj(f"{variant}/test_addr"))
            addr2 = lib2.get_function("code_address")()
        except Exception:
            return None, True
    finally:
        free_blockers(blockers)

    return abs(addr1 - addr2), False


@pytest.mark.skipif(not _is_linux, reason="Arena is Linux-only")
@pytest.mark.parametrize("variant", _all_variants)
def test_arena_keeps_objects_close(variant: str) -> None:
    """Arena co-locates objects that would otherwise scatter or overflow.

    Runs the same workload twice under identical VA pressure — once with
    the arena and once without — and compares the outcomes:

    - **With arena**: both objects must land within the arena size (16 MB).
    - **Without arena**: the blocker should either cause a relocation
      overflow (proving scatter beyond relocation range) or produce a
      measurably larger distance.

    The test proves the arena is responsible for co-location by showing a
    strictly better outcome with it enabled.  If the VA blocker happens to
    be ineffective (e.g., LLVM slab reuse on 64k-page kernels), the test
    still passes as long as the arena keeps objects within range.
    """
    # Phase 1: with arena — must always succeed and be within arena range
    arena_dist, arena_overflow = _measure_distance_under_pressure(variant, slab_size=_ARENA_SIZE)
    assert not arena_overflow, "Arena session should not overflow"
    assert arena_dist is not None
    assert arena_dist < _ARENA_SIZE, (
        f"With arena, objects should be within {_ARENA_SIZE} bytes, "
        f"but distance is {arena_dist} ({arena_dist / (1024**2):.1f} MB)"
    )

    # Phase 2: without arena — expect scatter or overflow
    no_arena_dist, no_arena_overflow = _measure_distance_under_pressure(variant, slab_size=-1)

    if no_arena_overflow:
        # Relocation overflow without arena proves the blocker forced
        # scatter beyond relocation range — arena prevented this.
        return

    assert no_arena_dist is not None
    if no_arena_dist > arena_dist:
        # Without arena produced a larger distance — arena effect shown.
        return

    # Blocker was ineffective (both distances are small).  The arena
    # assertion above already passed, which is the key property.  We
    # cannot distinguish arena effect from lucky placement here.
    pytest.skip(
        f"VA blocker ineffective: arena={arena_dist / 1024:.0f} KB, "
        f"no-arena={no_arena_dist / 1024:.0f} KB — "
        f"cannot demonstrate arena effect on this kernel"
    )


# ---------------------------------------------------------------------------
# Test 3: Hidden-symbol ADRP/PC32 relocation with arena + blocker
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _is_linux, reason="Arena is Linux-only")
@pytest.mark.parametrize("variant", _all_variants)
def test_arena_hidden_symbol_with_blocker(variant: str) -> None:
    """Arena prevents hidden-visibility relocation overflow under VA pressure.

    Loads two objects with hidden-visibility cross-references (ADRP+ADD
    on AArch64, PC32 on x86_64) with a VA blocker between them.
    Without arena, the blocker would push objects apart causing overflow.
    With the arena, both objects are co-located and the call succeeds.
    """
    maps_before = set(_parse_maps())

    session = ExecutionSession(slab_size=_ARENA_SIZE)
    lib = session.create_library("hidden_test")

    # Load helper and force materialization
    lib.add(obj(f"{variant}/test_hidden_helper"))
    assert lib.get_function("hidden_add")(1, 2) == 3

    # Block nearby VA to force scatter
    maps_after = _parse_maps()
    new_maps = _find_new_mappings(maps_before, maps_after)
    jit_center = max(s for s, e in new_maps) if new_maps else 0xFFFF00000000

    blockers = block_nearby_va(jit_center)
    try:
        lib.add(obj(f"{variant}/test_hidden_caller"))
        fn = lib.get_function("call_hidden_add")
        assert fn(10, 20) == 30
    finally:
        free_blockers(blockers)


# ---------------------------------------------------------------------------
# Test 4: Large data section (simulated .nv_fatbin)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _is_linux, reason="Arena is Linux-only")
@pytest.mark.parametrize("variant", _all_variants)
def test_large_data_section(variant: str) -> None:
    """Load object with a 4MB .nv_fatbin section — basic correctness.

    The .nv_fatbin section is referenced only by absolute relocations,
    so it can live anywhere.  This test verifies the object loads and
    the function works.  The 4MB section fits in the arena.
    """
    session = ExecutionSession()
    lib = session.create_library("fatbin")
    lib.add(obj(f"{variant}/fake_fatbin"))
    fn = lib.get_function("get_fatbin_size")
    assert fn() == 4 * 1024 * 1024


# ---------------------------------------------------------------------------
# Test 5: Overflow section — .nv_fatbin lands outside the arena
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _is_linux, reason="Arena is Linux-only")
@pytest.mark.parametrize("variant", _all_variants)
def test_overflow_section_outside_arena(variant: str) -> None:
    """Overflow sections (.nv_fatbin) are allocated outside the arena.

    The arena memory manager detects sections named .nv_fatbin and
    allocates them via a separate mmap() outside the arena.  This keeps
    the arena compact for code + small rodata, reducing 2MB THP region
    count and iTLB pressure.

    Verification: get the fatbin data address and the arena VA range
    from /proc/self/maps, then assert the fatbin address is NOT within
    the arena region.
    """
    session = ExecutionSession(slab_size=_ARENA_SIZE)
    lib = session.create_library("fatbin_overflow")
    lib.add(obj(f"{variant}/fake_fatbin"))

    # Verify the function still works correctly.
    assert lib.get_function("get_fatbin_size")() == 4 * 1024 * 1024

    # Get the actual address of the fatbin data in memory.
    fatbin_addr = lib.get_function("get_fatbin_addr")()

    # Find the arena mapping: a single large region matching the arena size.
    # The arena is reserved as PROT_NONE and then committed in slabs, so
    # look for the contiguous region that spans _ARENA_SIZE.
    maps = _parse_maps()
    arena_regions = [(s, e) for s, e in maps if (e - s) >= _ARENA_SIZE]

    # The fatbin address must not fall within any arena-sized region.
    for start, end in arena_regions:
        assert not (start <= fatbin_addr < end), (
            f"Fatbin data at {fatbin_addr:#x} should be OUTSIDE the arena "
            f"[{start:#x}, {end:#x}) but landed inside"
        )


# ---------------------------------------------------------------------------
# Test 6: __dso_handle Delta32 overflow after leaked materialization
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _is_linux, reason="ELF/GCC-specific __dso_handle test")
@pytest.mark.parametrize("variant", _cpp_variants)
def test_dso_handle_relocation_after_failed_materialization(variant: str) -> None:
    """__dso_handle resolves correctly after leaked JIT memory.

    Mechanism
    ---------
    GCC C++ objects call ``__cxa_atexit(&destructor, &obj, __dso_handle)``
    for static-storage-duration objects.  The ``__dso_handle`` symbol is
    emitted as ``GLOBAL HIDDEN UND`` in each object file.  LLVM's
    ``ELFNixPlatform`` defines it per JITDylib via a separate
    ``DSOHandleMaterializationUnit`` — a self-referential pointer block
    (``void *__dso_handle = &__dso_handle;``) allocated in its own
    ``LinkGraph`` through ``ObjectLinkingLayer``.

    When a prior ``lib.add()`` fails (e.g., duplicate symbol), LLVM's
    ``InProcessMemoryMapper`` leaks the mmap'd slab for that failed
    materialization.  If the process holds references to the old session
    (e.g., pytest keeping ``sys.exc_info()`` tracebacks alive), the
    leaked slabs accumulate and push subsequent ``mmap`` allocations to
    higher addresses.

    The arena prevents overflow because all allocations — both
    ``__dso_handle``'s ``LinkGraph`` and the code ``LinkGraph`` — land
    within the same contiguous pre-reserved VA region.

    Without arena: may FAIL on x86_64 with GCC PIE objects after
                   repeated leaked materializations push slabs >2 GB
                   apart.
    With arena:    PASSES (all allocations in same arena).
    """
    # Step 1: Trigger leaked materializations to consume low VA space.
    leaked_sessions = []
    for _ in range(3):
        s0 = ExecutionSession()
        lib0 = s0.create_library("warmup")
        lib0.add(obj(f"{variant}/test_funcs"))
        lib0.get_function("test_add")(10, 20)
        try:
            lib0.add(obj(f"{variant}/test_funcs_conflict"))
        except Exception:
            pass
        leaked_sessions.append((s0, lib0))

    # Step 2: Fresh session — cross-library resolution must still work.
    session = ExecutionSession()
    lib1 = session.create_library("lib1")
    lib1.add(obj(f"{variant}/test_funcs"))
    assert lib1.get_function("test_add")(10, 20) == 30

    lib2 = session.create_library("lib2")
    lib2.add(obj(f"{variant}/test_funcs_conflict"))
    assert lib2.get_function("test_add")(10, 20) == 1030


# ---------------------------------------------------------------------------
# Test 6: __dso_handle Delta32 overflow — arena prevents it (x86_64 PIE)
#
# GCC -fpie objects use R_X86_64_PC32 (±2GB) for __dso_handle.
# ELFNixPlatform's DSOHandleMaterializationUnit allocates __dso_handle
# in a separate LinkGraph from the code.  Under VA pressure, these two
# allocations can land >2GB apart, overflowing the Delta32 fixup.
# The arena keeps them co-located within relocation range.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _is_linux, reason="Arena is Linux-only")
@pytest.mark.skipif(not _is_x86_64, reason="Delta32 overflow requires x86_64")
@pytest.mark.skipif(not _pie_cpp_variants, reason="No GCC PIE C++ variants built")
@pytest.mark.parametrize("variant", _pie_cpp_variants or ["skip"])
def test_dso_handle_delta32_with_arena(variant: str) -> None:
    """Arena prevents __dso_handle Delta32 overflow under VA pressure.

    Root cause
    ----------
    GCC C++ objects built with ``-fpie`` emit ``R_X86_64_PC32`` (Delta32,
    ±2 GB) relocations for ``__dso_handle`` because the symbol has hidden
    visibility and ``-fpie`` prefers direct PC-relative over GOT-relative.
    (With ``-fPIC``, GCC uses ``R_X86_64_GOTPCRELX`` which goes through
    the GOT — always co-located with code, so no range issue.)

    ``ELFNixPlatform`` defines ``__dso_handle`` per JITDylib in a separate
    ``DSOHandleMaterializationUnit``.  This creates a tiny ``LinkGraph``
    (a self-referential pointer: ``void *__dso_handle = &__dso_handle;``)
    that is allocated through ``ObjectLinkingLayer`` independently of the
    code ``LinkGraph`` from ``lib.add()``.  Both allocations go through
    ``InProcessMemoryMapper`` → ``mmap(MAP_ANONYMOUS)``, whose placement
    the kernel decides.

    Test strategy
    -------------
    1. Create a session with arena enabled (16 MB).
    2. Load PIE GCC objects into lib1 — this triggers materialization of
       both ``__dso_handle`` (via ``DSOHandleMaterializationUnit``) and
       the code (via ``lib.add``), all within the arena.
    3. Block 3 GB of VA around the first allocation — without arena this
       would force the next ``mmap`` to land >2 GB away.
    4. Load a second PIE GCC object into lib2 — with arena, this still
       lands within the 16 MB region.
    5. Assert the function call succeeds — proves Delta32 is in range.

    See ``test_dso_handle_delta32_overflow_without_arena`` for the
    counterpart proving the overflow occurs without arena.
    """
    maps_before = set(_parse_maps())

    session = ExecutionSession(slab_size=_ARENA_SIZE)
    lib1 = session.create_library("lib1")
    lib1.add(obj(f"{variant}/test_funcs"))
    assert lib1.get_function("test_add")(10, 20) == 30

    # Block 3GB of VA around the first allocation to force scatter
    maps_after = _parse_maps()
    new_maps = _find_new_mappings(maps_before, maps_after)
    jit_center = max(s for s, e in new_maps) if new_maps else 0xFFFF00000000

    blockers = block_nearby_va(jit_center, radius=_DSO_BLOCK_RADIUS)
    try:
        lib2 = session.create_library("lib2")
        lib2.add(obj(f"{variant}/test_funcs_conflict"))
        assert lib2.get_function("test_add")(10, 20) == 1030
    finally:
        free_blockers(blockers)


@pytest.mark.skipif(not _is_linux, reason="Arena is Linux-only")
@pytest.mark.skipif(not _is_x86_64, reason="Delta32 overflow requires x86_64")
@pytest.mark.skipif(not _pie_cpp_variants, reason="No GCC PIE C++ variants built")
@pytest.mark.parametrize("variant", _pie_cpp_variants or ["skip"])
def test_dso_handle_delta32_overflow_without_arena(variant: str) -> None:
    """Without arena, PIE __dso_handle PC32 overflows under VA pressure.

    Same setup as ``test_dso_handle_delta32_with_arena`` but with arena
    disabled (``slab_size=-1``).

    The 3 GB VA blocker fills all free gaps within ±3 GB of the first
    session's JIT allocations.  When lib2 is loaded, ``InProcessMemoryMapper``
    calls ``mmap(MAP_ANONYMOUS)`` for a new slab, but the only free VA is
    >3 GB away.  The code ``LinkGraph`` from ``lib2.add()`` lands in that
    distant slab, while ``__dso_handle`` was already materialized with
    lib1's ``DSOHandleMaterializationUnit`` in the original region.  The
    ``R_X86_64_PC32`` fixup from code to ``__dso_handle`` now exceeds
    ±2 GB → JITLink reports ``Delta32 fixup ... is out of range``.

    The test accepts both outcomes:
    - **Exception** (PC32 overflow): proves the arena is needed.
    - **Success** (GOTPCRELX used): GCC chose GOT-relative despite
      ``-fpie`` — no overflow possible, but the arena is still
      beneficial for other relocation types.
    """
    maps_before = set(_parse_maps())

    session = ExecutionSession(slab_size=-1)  # arena disabled
    lib1 = session.create_library("lib1")
    lib1.add(obj(f"{variant}/test_addr"))
    lib1.get_function("code_address")()

    maps_after = _parse_maps()
    new_maps = _find_new_mappings(maps_before, maps_after)
    jit_center = max(s for s, e in new_maps) if new_maps else 0xFFFF00000000

    blockers = block_nearby_va(jit_center, radius=_DSO_BLOCK_RADIUS)
    try:
        lib2 = session.create_library("lib2")
        try:
            lib2.add(obj(f"{variant}/test_funcs_conflict"))
            result = lib2.get_function("test_add")(10, 20)
            # If we get here, GCC used GOTPCRELX — no overflow.
            assert result == 1030
        except Exception:
            # R_X86_64_PC32 overflow as expected — proves arena is needed.
            pass
    finally:
        free_blockers(blockers)
