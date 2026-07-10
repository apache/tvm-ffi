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
"""Basic tests for tvm_ffi_orcjit functionality.

Everything loads through the high-level ``ExecutionSession.load_module`` API
(there is no low-level incremental library API). Each ``load_module`` links its
objects into one fresh JITDylib and returns a plain ``tvm_ffi.Module``; dropping
the module tears the dylib down.
"""

from __future__ import annotations

import gc
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import tvm_ffi
from tvm_ffi import Module
from tvm_ffi_orcjit import ExecutionSession
from utils import build_test_objects

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OBJ_DIR = build_test_objects()

_KNOWN_SUBDIRS = [
    "cc",
    "c",  # default (LLVM clang)
    "cc-gcc",
    "c-gcc",  # GCC (Linux)
    "cc-appleclang",
    "c-appleclang",  # Apple Clang (macOS)
    "c-msvc",  # MSVC (Windows, C only)
    "c-clang-cl",  # clang-cl (Windows, C only)
]


def obj(name: str) -> str:
    """Return path to a pre-built test object file, or skip if missing."""
    path = OBJ_DIR / f"{name}.o"
    if not path.exists():
        pytest.skip(f"{path.name} not found (not built)")
    return str(path)


def load(
    *obj_names: str,
    session: ExecutionSession | None = None,
    name: str = "",
    keep_module_alive: bool = False,
) -> Module:
    """Load one or more object files into a fresh module on ``session``."""
    if session is None:
        session = ExecutionSession()
    return session.load_module(
        [obj(o) for o in obj_names], name, keep_module_alive=keep_module_alive
    )


# ---------------------------------------------------------------------------
# Variants: C and C++ sources live in separate subdirectories (c/, cc/).
# Function names are identical; only the object file path differs.
# Tests are parametrized over both variants where the logic is identical.
# ---------------------------------------------------------------------------


class Variant:
    """Describes a C or C++ test variant (object file path mapping)."""

    def __init__(self, subdir: str) -> None:
        self.subdir = subdir  # "cc" for C++, "c" for C

    def funcs_obj(self) -> str:
        """Return path prefix for test_funcs object."""
        return f"{self.subdir}/test_funcs"

    def funcs2_obj(self) -> str:
        """Return path prefix for test_funcs2 object."""
        return f"{self.subdir}/test_funcs2"

    def conflict_obj(self) -> str:
        """Return path prefix for test_funcs_conflict object."""
        return f"{self.subdir}/test_funcs_conflict"

    def call_global_obj(self) -> str:
        """Return path prefix for test_call_global object."""
        return f"{self.subdir}/test_call_global"

    def types_obj(self) -> str:
        """Return path prefix for test_types object."""
        return f"{self.subdir}/test_types"

    def error_obj(self) -> str:
        """Return path prefix for test_error object."""
        return f"{self.subdir}/test_error"

    def ctor_dtor_obj(self) -> str:
        """Return path prefix for test_ctor_dtor object."""
        return f"{self.subdir}/test_ctor_dtor"

    def containers_obj(self) -> str:
        """Return path prefix for test_containers object.

        C++-only source; C variants return an empty prefix so ``obj()`` sees
        a missing path and skips the parametrized case for that variant.
        """
        if not self.subdir.startswith("cc"):
            return ""
        return f"{self.subdir}/test_containers"

    def fn(self, base_name: str) -> str:
        """Return the function name for a given base name."""
        return base_name

    def __repr__(self) -> str:
        parts = self.subdir.split("-", 1)
        lang = "C++" if parts[0] == "cc" else "C"
        return f"{lang}-{parts[1]}" if len(parts) > 1 else lang


def _discover_variants() -> list[Variant]:
    return [Variant(s) for s in _KNOWN_SUBDIRS if (OBJ_DIR / s / "test_funcs.o").exists()]


_all_variants = _discover_variants()


def _variant_id(v: Variant) -> str:
    return repr(v)


# ---------------------------------------------------------------------------
# Session creation
# ---------------------------------------------------------------------------


def test_create_session() -> None:
    """Create an ExecutionSession successfully."""
    session = ExecutionSession()
    assert session is not None


# ---------------------------------------------------------------------------
# Load & execute — parametrized over C / C++
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("v", _all_variants, ids=_variant_id)
def test_load_and_execute(v: Variant) -> None:
    """Load test_funcs and verify add/multiply."""
    mod = load(v.funcs_obj())
    assert mod.get_function(v.fn("test_add"))(10, 20) == 30
    assert mod.get_function(v.fn("test_multiply"))(7, 6) == 42


@pytest.mark.parametrize("v", _all_variants, ids=_variant_id)
def test_load_and_execute_second_set(v: Variant) -> None:
    """Load test_funcs2 and verify subtract/divide."""
    mod = load(v.funcs2_obj())
    assert mod.get_function(v.fn("test_subtract"))(10, 3) == 7
    assert mod.get_function(v.fn("test_divide"))(20, 4) == 5


@pytest.mark.parametrize("v", _all_variants, ids=_variant_id)
def test_multiple_objects_in_one_module(v: Variant) -> None:
    """Multiple objects linked into one module expose all their functions."""
    mod = load(v.funcs_obj(), v.funcs2_obj())
    assert mod.get_function(v.fn("test_add"))(5, 3) == 8
    assert mod.get_function(v.fn("test_multiply"))(4, 5) == 20
    assert mod.get_function(v.fn("test_subtract"))(10, 3) == 7
    assert mod.get_function(v.fn("test_divide"))(20, 4) == 5


@pytest.mark.parametrize("v", _all_variants, ids=_variant_id)
def test_function_not_found(v: Variant) -> None:
    """Raise AttributeError for a missing function name."""
    mod = load(v.funcs_obj())
    with pytest.raises(AttributeError, match="Module has no function"):
        mod.get_function("nonexistent_function")


# ---------------------------------------------------------------------------
# Multi-module — parametrized over C / C++
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("v", _all_variants, ids=_variant_id)
def test_two_separate_modules(v: Variant) -> None:
    """Separate modules on one session expose only their own functions."""
    session = ExecutionSession()
    mod1 = load(v.funcs_obj(), session=session, name="lib1")
    mod2 = load(v.funcs2_obj(), session=session, name="lib2")

    assert mod1.get_function(v.fn("test_add"))(5, 3) == 8
    assert mod1.get_function(v.fn("test_multiply"))(4, 5) == 20
    assert mod2.get_function(v.fn("test_subtract"))(10, 3) == 7
    assert mod2.get_function(v.fn("test_divide"))(20, 4) == 5

    with pytest.raises(AttributeError, match="Module has no function"):
        mod1.get_function(v.fn("test_subtract"))
    with pytest.raises(AttributeError, match="Module has no function"):
        mod2.get_function(v.fn("test_add"))


# ---------------------------------------------------------------------------
# Symbol conflicts — parametrized over C / C++
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("v", _all_variants, ids=_variant_id)
def test_symbol_conflict_same_module(v: Variant) -> None:
    """Duplicate symbol across objects in one module raises an error."""
    with pytest.raises(Exception):
        load(v.funcs_obj(), v.conflict_obj())


@pytest.mark.parametrize("v", _all_variants, ids=_variant_id)
def test_symbol_conflict_different_modules(v: Variant) -> None:
    """Same symbol in different modules resolves independently."""
    session = ExecutionSession()
    mod1 = load(v.funcs_obj(), session=session, name="lib1")
    mod2 = load(v.conflict_obj(), session=session, name="lib2")

    assert mod1.get_function(v.fn("test_add"))(10, 20) == 30
    assert mod2.get_function(v.fn("test_add"))(10, 20) == 1030
    assert mod1.get_function(v.fn("test_multiply"))(5, 6) == 30
    assert mod2.get_function(v.fn("test_multiply"))(5, 6) == 60


# ---------------------------------------------------------------------------
# Global function callbacks — parametrized over C / C++
# ---------------------------------------------------------------------------


@pytest.fixture()
def _register_host_functions() -> None:
    """Register host add/multiply functions for JIT code to call."""

    @tvm_ffi.register_global_func("test_host_add", override=True)
    def _host_add(a: int, b: int) -> int:
        return a + b

    @tvm_ffi.register_global_func("test_host_multiply", override=True)
    def _host_mul(a: int, b: int) -> int:
        return a * b


@pytest.mark.usefixtures("_register_host_functions")
@pytest.mark.parametrize("v", _all_variants, ids=_variant_id)
def test_call_global(v: Variant) -> None:
    """JIT code calls back into Python-registered global functions."""
    mod = load(v.call_global_obj())
    add_func = mod.get_function(v.fn("test_call_global_add"))
    assert add_func(10, 20) == 30
    assert add_func(100, 200) == 300

    mul_func = mod.get_function(v.fn("test_call_global_mul"))
    assert mul_func(7, 6) == 42
    assert mul_func(11, 11) == 121


# ---------------------------------------------------------------------------
# Library-context injection
# ---------------------------------------------------------------------------


def test_context_injected_at_load() -> None:
    """Context is injected eagerly at load, so concurrent lookups all observe it."""
    mod = load("c/test_context")

    num_workers = 8
    barrier = threading.Barrier(num_workers)

    def lookup(_worker: int) -> int:
        barrier.wait()
        return mod.get_function("context_is_set")()

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        assert list(pool.map(lookup, range(num_workers))) == [1] * num_workers


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_missing_function_on_empty_module() -> None:
    """get_function for a missing name raises AttributeError."""
    mod = load(_all_variants[0].funcs_obj())
    with pytest.raises(AttributeError, match="Module has no function"):
        mod.get_function("nonexistent")


def test_invalid_object_file_path() -> None:
    """Loading a nonexistent object file raises an error."""
    session = ExecutionSession()
    with pytest.raises(Exception):
        session.load_module("/nonexistent_path/does_not_exist.o")


def test_invalid_object_file_content() -> None:
    """Loading a file with garbage content raises an error."""
    with tempfile.NamedTemporaryFile(suffix=".o", delete=False) as f:
        f.write(b"this is not a valid object file")
        f.flush()
        session = ExecutionSession()
        with pytest.raises(Exception):
            session.load_module(f.name)


def test_multiple_independent_sessions() -> None:
    """Two independent sessions don't interfere with each other."""
    session1 = ExecutionSession()
    session2 = ExecutionSession()
    mod1 = load(_all_variants[0].funcs_obj(), session=session1)
    mod2 = load(_all_variants[0].funcs2_obj(), session=session2)

    assert mod1.get_function("test_add")(5, 3) == 8
    assert mod2.get_function("test_subtract")(10, 3) == 7

    # Each session's module doesn't see the other's functions
    with pytest.raises(AttributeError, match="Module has no function"):
        mod1.get_function("test_subtract")
    with pytest.raises(AttributeError, match="Module has no function"):
        mod2.get_function("test_add")


# ---------------------------------------------------------------------------
# Type variety — parametrized over C / C++
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("v", _all_variants, ids=_variant_id)
def test_zero_arg_function(v: Variant) -> None:
    """Zero-arg function returns constant 42."""
    mod = load(v.types_obj())
    assert mod.get_function(v.fn("test_zero_arg"))() == 42


@pytest.mark.parametrize("v", _all_variants, ids=_variant_id)
def test_four_arg_function(v: Variant) -> None:
    """Four integer arguments summed."""
    mod = load(v.types_obj())
    assert mod.get_function(v.fn("test_four_args"))(1, 2, 3, 4) == 10
    assert mod.get_function(v.fn("test_four_args"))(100, 200, 300, 400) == 1000


@pytest.mark.parametrize("v", _all_variants, ids=_variant_id)
def test_float_function(v: Variant) -> None:
    """Float multiply returns approximate result."""
    mod = load(v.types_obj())
    result = mod.get_function(v.fn("test_float_multiply"))(3.14, 2.0)
    assert result == pytest.approx(6.28)


@pytest.mark.parametrize("v", _all_variants, ids=_variant_id)
def test_void_function(v: Variant) -> None:
    """Void function returns None."""
    mod = load(v.types_obj())
    result = mod.get_function(v.fn("test_void_function"))()
    assert result is None


# ---------------------------------------------------------------------------
# Error propagation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("v", _all_variants, ids=_variant_id)
def test_error_propagation(v: Variant) -> None:
    """JIT function that signals an error raises a Python exception."""
    mod = load(v.error_obj())
    with pytest.raises(Exception, match="test error"):
        mod.get_function(v.fn("test_error"))()


# ---------------------------------------------------------------------------
# CUDA (optional)
# ---------------------------------------------------------------------------


def test_load_and_execute_cuda_function() -> None:
    """Load and execute CUDA-compiled test objects."""
    mod = load("cuda/test_funcs")
    assert mod.get_function("test_add")(10, 20) == 30
    assert mod.get_function("test_multiply")(7, 6) == 42


# ---------------------------------------------------------------------------
# Constructor / destructor (parametrized over C / C++)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("v", _all_variants, ids=_variant_id)
def test_ctor_dtor(v: Variant) -> None:
    """Verify constructor/destructor ordering across platforms."""
    log = ""

    @tvm_ffi.register_global_func("append_log", override=True)
    def _append_ctor_log(x: str) -> None:
        nonlocal log
        log += x

    mod = load(v.ctor_dtor_obj())
    mod.get_function(v.fn("main"))()
    del mod

    main_idx = log.index("<main>")
    pre = log[:main_idx]
    post = log[main_idx:]

    if sys.platform == "win32":
        # Windows (all compilers): COFF .CRT$XC* constructors + .CRT$XT* terminators.
        # All Windows compilers (MSVC, clang-cl, and LLVM Clang targeting MSVC ABI)
        # define _MSC_VER, so the source uses #pragma section / __declspec(allocate).
        assert "<crt.XCA>" in pre, f"CRT initializers not found in log: {log!r}"
        assert pre.index("<crt.XCA>") < pre.index("<crt.XCB>")
        assert pre.index("<crt.XCB>") < pre.index("<crt.XCC>")
        assert pre.index("<crt.XCC>") < pre.index("<crt.XCU>")
        assert "<crt.XTA>" in post, f"CRT terminators not found in log: {log!r}"
        assert post.index("<crt.XTA>") < post.index("<crt.XTZ>")
    elif sys.platform == "linux":
        # ELF: init_array (priority order) + .ctors (reversed priority)
        assert pre.index("<init_array.101>") < pre.index("<init_array.102>")
        assert pre.index("<init_array.102>") < pre.index("<init_array.103>")
        assert pre.index("<init_array.103>") < pre.index("<init_array>")
        assert pre.index("<ctors.103>") < pre.index("<ctors.102>")
        assert pre.index("<ctors.102>") < pre.index("<ctors.101>")
        assert pre.index("<ctors.101>") < pre.index("<ctors>")
        assert "<dtors>" in post
    elif sys.platform == "darwin":
        # Mach-O: init_array (priority order) + explicit __mod_init_func
        assert pre.index("<init_array.101>") < pre.index("<init_array.102>")
        assert pre.index("<init_array.102>") < pre.index("<init_array.103>")
        assert pre.index("<init_array.103>") < pre.index("<init_array>")
        assert "<mod_init_func>" in pre
        assert post.index("<fini_array>") < post.index("<fini_array.103>")
        assert post.index("<fini_array.103>") < post.index("<fini_array.102>")
        assert post.index("<fini_array.102>") < post.index("<fini_array.101>")
        assert "<ctors>" not in log
        assert "<dtors>" not in log


# ---------------------------------------------------------------------------
# Module drop — dropping a loaded Module while its session is still alive.
#
# Dropping the last reference to a load_module result removes the JITDylib
# from the ExecutionSession (releasing its JIT memory) in addition to running
# any pending static destructors. These tests pin that contract and guard
# against regressions in the slab-pool memory-manager refactor, which assumes
# per-dylib deallocation actually happens at drop time.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("v", _all_variants, ids=_variant_id)
def test_drop_loaded_module_then_reload(v: Variant) -> None:
    """Drop a module with live JIT code, then load and use a fresh one."""
    session = ExecutionSession()

    mod1 = load(v.funcs_obj(), session=session, name="lib1")
    assert mod1.get_function(v.fn("test_add"))(3, 4) == 7
    del mod1

    mod2 = load(v.funcs_obj(), session=session, name="lib2")
    assert mod2.get_function(v.fn("test_add"))(100, 1) == 101


@pytest.mark.parametrize("v", _all_variants, ids=_variant_id)
def test_repeated_load_drop_many_iterations(v: Variant) -> None:
    """Long load / call / drop cycle must not degrade or crash.

    Exercises the memory manager's recycled-region path: each iteration
    frees JIT memory that a later iteration may be handed back by the arena.
    """
    session = ExecutionSession()
    for i in range(32):
        mod = load(v.funcs_obj(), session=session, name=f"iter_{i}")
        assert mod.get_function(v.fn("test_multiply"))(i + 1, 2) == (i + 1) * 2
        del mod


@pytest.mark.parametrize("v", _all_variants, ids=_variant_id)
def test_drop_one_module_does_not_affect_another(v: Variant) -> None:
    """Dropping one module must leave unrelated sibling modules working."""
    session = ExecutionSession()

    keep = load(v.funcs_obj(), session=session, name="keep")

    drop = load(v.funcs2_obj(), session=session, name="drop")
    assert drop.get_function(v.fn("test_subtract"))(10, 3) == 7

    del drop
    gc.collect()

    # Untouched sibling still works; room now exists for a fresh module.
    assert keep.get_function(v.fn("test_add"))(5, 5) == 10
    fresh = load(v.funcs2_obj(), session=session, name="fresh")
    assert fresh.get_function(v.fn("test_divide"))(20, 4) == 5


@pytest.mark.parametrize("v", _all_variants, ids=_variant_id)
def test_captured_function_keeps_module_alive(v: Variant) -> None:
    """A captured Function keeps the module alive after the handle is dropped."""
    session = ExecutionSession()
    mod = load(v.funcs_obj(), session=session, name="hold")

    captured = mod.get_function(v.fn("test_add"))
    del mod
    gc.collect()

    assert captured(11, 31) == 42


@pytest.mark.parametrize("v", _all_variants, ids=_variant_id)
def test_drop_runs_static_destructors(v: Variant) -> None:
    """Drop runs static destructors immediately rather than at session teardown."""
    log: list[str] = []

    @tvm_ffi.register_global_func("append_log", override=True)
    def _append(x: str) -> None:
        log.append(x)

    session = ExecutionSession()
    mod = load(v.ctor_dtor_obj(), session=session, name="ctor_dtor")
    mod.get_function(v.fn("main"))()

    ctor_len = len(log)
    assert "<main>" in log, f"main not observed: {log}"

    del mod
    gc.collect()

    post = log[ctor_len:]
    # Platform-specific fini markers: ELF (.fini_array / .dtors), Mach-O
    # (__mod_term_func surfaces as fini_array), COFF (.CRT$XT*).
    if sys.platform == "win32":
        markers = ("crt.XT",)
    else:
        markers = ("dtors", "fini_array")
    assert any(m in entry for entry in post for m in markers), (
        f"no static destructors on drop (platform={sys.platform}): pre={log[:ctor_len]} post={post}"
    )


@pytest.mark.parametrize("v", _all_variants, ids=_variant_id)
def test_drop_all_modules_then_session(v: Variant) -> None:
    """Dropping every module before the session still leaves clean teardown."""
    session = ExecutionSession()
    mods = [load(v.funcs_obj(), session=session, name=f"lib_{i}") for i in range(4)]
    for mod in reversed(mods):
        del mod
    mods.clear()
    gc.collect()
    # Session destructor runs at end-of-test; a regression in removal would
    # surface as a crash under pytest.


# ---------------------------------------------------------------------------
# FFI-container returns under keep_module_alive.
#
# JIT-emitted code allocates each returned Object through inline templates
# (make_object / make_inplace_array_object), so the object's header.deleter
# points into JIT-mapped memory. If the owning Module were torn down before
# the escaped Object, the deleter would dangle. keep_module_alive=True pins
# the module in the runtime's process-global registry, keeping the JITDylib
# mapped for the process lifetime. Each test loads with the flag, drops the
# local mod reference, forces GC, and then reads the returned container —
# proving the deleter is still reachable after the local mod is gone.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("v", _all_variants, ids=_variant_id)
def test_containers_string(v: Variant) -> None:
    """String return survives del mod when pinned."""
    mod = load(v.containers_obj(), keep_module_alive=True)
    result = mod.test_string_replace("hello world", "l", "L")
    del mod
    gc.collect()
    assert str(result) == "heLLo worLd"


@pytest.mark.parametrize("v", _all_variants, ids=_variant_id)
def test_containers_array(v: Variant) -> None:
    """Array<int64_t> return survives del mod when pinned."""
    mod = load(v.containers_obj(), keep_module_alive=True)
    result = mod.test_array_concat([1, 2, 3], [4, 5, 6])
    del mod
    gc.collect()
    assert list(result) == [1, 2, 3, 4, 5, 6]


@pytest.mark.parametrize("v", _all_variants, ids=_variant_id)
def test_containers_map(v: Variant) -> None:
    """Map<String, int64_t> return survives del mod when pinned."""
    mod = load(v.containers_obj(), keep_module_alive=True)
    result = mod.test_map_build()
    del mod
    gc.collect()
    assert dict(result) == {"a": 1, "b": 2}


@pytest.mark.parametrize("v", _all_variants, ids=_variant_id)
def test_containers_tuple(v: Variant) -> None:
    """Tuple<...> return survives del mod when pinned."""
    mod = load(v.containers_obj(), keep_module_alive=True)
    result = mod.test_tuple_reverse((1, "a", 2, "b"))
    del mod
    gc.collect()
    assert tuple(result) == ("b", 2, "a", 1)


# ---------------------------------------------------------------------------
# Slab-pool growth (Stage B).
#
# A session holds a growable pool of Slabs, each `slab_size` bytes. When a
# JITLink graph won't fit in any existing slab, the pool mmap's a new one;
# graphs larger than a single slab go to a dedicated oversize slab. These tests
# pin the behavioral contract: small slab_size sessions must still link many
# modules (by growing) and must reuse drained bytes within a slab (free list).
# ---------------------------------------------------------------------------


# 8 MB is the practical floor — Slab::kCommitGranularity is 2 MB and the
# dual-pool midpoint needs at least two commit chunks of headroom above it,
# so smaller capacities break the pool layout. See SlabPoolMemoryManager
# kMinSlabSize.
_SMALL_SLAB = 8 * 1024 * 1024


@pytest.mark.skipif(sys.platform != "linux", reason="slab pool is Linux-only")
@pytest.mark.parametrize("v", _all_variants, ids=_variant_id)
def test_pool_grows_under_small_slab(v: Variant) -> None:
    """Tight slab_size forces the pool to grow as more modules load."""
    session = ExecutionSession(slab_size=_SMALL_SLAB)

    mods = []
    for i in range(16):
        mod = load(v.funcs_obj(), session=session, name=f"grow_{i}")
        assert mod.get_function(v.fn("test_add"))(i, i + 1) == 2 * i + 1
        mods.append(mod)

    # All modules remain callable after growth. Catches bugs where the pool
    # routes deallocate to the wrong Slab via FA->owner.
    for i, mod in enumerate(mods):
        assert mod.get_function(v.fn("test_multiply"))(i, 2) == i * 2


@pytest.mark.skipif(sys.platform != "linux", reason="slab pool is Linux-only")
@pytest.mark.parametrize("v", _all_variants, ids=_variant_id)
def test_small_slab_recycles_after_drop(v: Variant) -> None:
    """Drop returns bytes to the slab's free list for subsequent reuse.

    If the free list weren't working, 32 iterations of load/drop under an
    8 MB slab would either exhaust VA or force many new slabs; this test
    passing on CI containers is the recycling evidence.
    """
    session = ExecutionSession(slab_size=_SMALL_SLAB)
    for i in range(32):
        mod = load(v.funcs_obj(), session=session, name=f"recycle_{i}")
        assert mod.get_function(v.fn("test_add"))(i, 1) == i + 1
        del mod
        gc.collect()


@pytest.mark.skipif(sys.platform != "linux", reason="slab pool is Linux-only")
@pytest.mark.parametrize("v", _all_variants, ids=_variant_id)
def test_pool_survives_mixed_load_drop(v: Variant) -> None:
    """Interleaved load / drop exercises growth + free-list together."""
    session = ExecutionSession(slab_size=_SMALL_SLAB)

    # Ramp up to 4 concurrent modules.
    live: list[Module | None] = [
        load(v.funcs_obj(), session=session, name=f"base_{i}") for i in range(4)
    ]

    # Drop two, load three, verify remaining two still work, verify new three
    # work. Mixing drop/load in one session exercises slab growth alongside
    # free-list reuse from the dropped bytes.
    live[0] = None
    live[2] = None
    gc.collect()

    new_mods = [load(v.funcs2_obj(), session=session, name=f"after_drop_{i}") for i in range(3)]

    assert live[1].get_function(v.fn("test_add"))(10, 5) == 15
    assert live[3].get_function(v.fn("test_multiply"))(7, 6) == 42
    for i, mod in enumerate(new_mods):
        assert mod.get_function(v.fn("test_subtract"))(i + 10, i) == 10


# ---------------------------------------------------------------------------
# Manual slab reclamation via session.clear_free_slabs() (Stage C).
#
# After dropping a batch of modules, the user can call clear_free_slabs() to
# release any drained Slabs (zero live JIT allocations) back to the OS.
# ---------------------------------------------------------------------------


def _build_big_object(tmp_path: Path, byte_size: int, name: str = "big_object") -> str:
    """Compile a .c file with a large `.rodata` blob to force the oversize path.

    Returns the path to the generated .o file.  Skips the test if the build
    toolchain is unavailable.  ``name`` lets callers produce multiple
    distinctly-sized objects within the same ``tmp_path``.
    """
    try:
        import tvm_ffi.cpp  # noqa: PLC0415
    except ImportError:
        pytest.skip("tvm_ffi.cpp not available for on-the-fly object build")

    src = tmp_path / f"{name}.c"
    # Global BSS array — volatile + a runtime read in big_probe() prevents
    # the compiler from folding sizeof() out and eliding the array.  The
    # array becomes a ZeroFill segment of `byte_size` bytes in the
    # LinkGraph, which counts toward the slab footprint
    # (see Slab::computeGraphFootprint).
    src.write_text(
        f"""
        #include <tvm/ffi/c_api.h>
        #include <stdint.h>
        volatile unsigned char big_data[{byte_size}];
        TVM_FFI_DLL_EXPORT int __tvm_ffi_big_probe(void* self,
                                                   const TVMFFIAny* args,
                                                   int32_t num_args,
                                                   TVMFFIAny* result) {{
          (void)self; (void)args; (void)num_args;
          /* Reads big_data[0] (always zero for BSS) so the compiler must
             emit the array; returns the array's size for verification. */
          result->type_index = kTVMFFIInt;
          result->zero_padding = 0;
          result->v_int64 = (int64_t)sizeof(big_data) + (int64_t)big_data[0];
          return 0;
        }}
        """
    )
    try:
        return tvm_ffi.cpp.build(
            name=name,
            sources=[str(src)],
            output=f"{name}.o",
            extra_cflags=["-O2"],
            build_directory=str(tmp_path / f".build_{name}"),
        )
    except Exception as exc:
        pytest.skip(f"could not build big object for reclaim test: {exc}")


@pytest.mark.skipif(sys.platform != "linux", reason="slab pool is Linux-only")
def test_clear_free_slabs_no_drained() -> None:
    """Calling clear_free_slabs with nothing to reclaim returns 0."""
    session = ExecutionSession()
    # Fresh session — the initial slab was never used, so not reclaimable.
    assert session.clear_free_slabs() == 0
    # Load + keep a module: live allocations pin the slab.
    mod = load(_all_variants[0].funcs_obj(), session=session)
    mod.get_function("test_add")(1, 2)
    assert session.clear_free_slabs() == 0


@pytest.mark.skipif(sys.platform != "linux", reason="slab pool is Linux-only")
def test_clear_free_slabs_reclaims_oversize(tmp_path: Path) -> None:
    """Dropping an oversize-path module lets clear_free_slabs reclaim its slab.

    Uses a 3 MB rodata blob that exceeds the 4 MB slab's usable-per-slab
    threshold (~2 MB after midpoint slack), forcing the oversize path. The
    dedicated oversize slab then hosts exactly one graph — dropping it
    returns the slab to fully-drained state, so it is reclaimable.
    """
    big_obj = _build_big_object(tmp_path, 3 * 1024 * 1024)
    # 4 MB slab → usable = slab_size / 2 = 2 MB → 3 MB blob forces oversize.
    session = ExecutionSession(slab_size=4 * 1024 * 1024)
    mod = session.load_module(big_obj)
    assert mod.get_function("big_probe")() == 3 * 1024 * 1024
    del mod
    gc.collect()
    # The oversize slab holds only the now-dropped module's graph, so
    # clearFreeSlabs reclaims it. The initial (never-used) slab stays.
    reclaimed = session.clear_free_slabs()
    assert reclaimed >= 1, f"expected to reclaim the drained oversize slab, got {reclaimed}"


@pytest.mark.skipif(sys.platform != "linux", reason="slab pool is Linux-only")
def test_clear_free_slabs_idempotent(tmp_path: Path) -> None:
    """A second call after everything has been reclaimed returns 0."""
    big_obj = _build_big_object(tmp_path, 3 * 1024 * 1024)
    session = ExecutionSession(slab_size=4 * 1024 * 1024)
    mod = session.load_module(big_obj)
    mod.get_function("big_probe")()
    del mod
    gc.collect()

    assert session.clear_free_slabs() >= 1
    assert session.clear_free_slabs() == 0


@pytest.mark.skipif(sys.platform != "linux", reason="slab pool is Linux-only")
def test_clear_free_slabs_preserves_live_pool(tmp_path: Path) -> None:
    """Reclaim runs only on drained slabs; live modules keep working.

    Uses two different oversize blob sizes so the two modules land on
    separately-sized slabs: the 3 MB blob forces the pool to grow to
    8 MB (next power of two covering a 4 MB pool's non-exec budget),
    while the 5 MB blob forces 16 MB.  Dropping the 3 MB module then
    leaves its 8 MB slab drained while the 16 MB slab stays live.
    """
    small_obj = _build_big_object(tmp_path, 3 * 1024 * 1024, name="small_obj")
    large_obj = _build_big_object(tmp_path, 5 * 1024 * 1024, name="large_obj")
    session = ExecutionSession(slab_size=4 * 1024 * 1024)

    # Drop module — lands on the 8 MB grown slab.
    drop_mod = session.load_module(small_obj)
    drop_mod.get_function("big_probe")()
    del drop_mod

    # Keep module — doesn't fit the 8 MB slab (5 MB > its ~4 MB non-exec
    # budget), so the pool grows to a 16 MB slab for this one.
    keep_mod = session.load_module(large_obj)
    assert keep_mod.get_function("big_probe")() == 5 * 1024 * 1024

    gc.collect()
    reclaimed = session.clear_free_slabs()
    assert reclaimed >= 1, f"expected drained slab to be reclaimed, got {reclaimed}"

    # The kept module's slab was untouched; it still executes correctly.
    assert keep_mod.get_function("big_probe")() == 5 * 1024 * 1024


@pytest.mark.skipif(sys.platform != "linux", reason="slab pool is Linux-only")
def test_clear_free_slabs_disabled_pool() -> None:
    """When the slab pool is disabled, clear_free_slabs is a no-op (returns 0)."""
    session = ExecutionSession(slab_size=-1)
    assert session.clear_free_slabs() == 0
