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
"""Basic tests for tvm_ffi_orcjit functionality."""

from __future__ import annotations

import gc
import sys
import tempfile

import pytest
import tvm_ffi
from tvm_ffi_orcjit import ExecutionSession
from tvm_ffi_orcjit.dylib import DynamicLibrary
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


def make_lib(
    *obj_names: str, session: ExecutionSession | None = None, name: str = ""
) -> tuple[ExecutionSession, DynamicLibrary]:
    """Create a library and load one or more object files into it."""
    if session is None:
        session = ExecutionSession()
    lib = session.create_library(name)
    for o in obj_names:
        lib.add(obj(o))
    return session, lib


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

    def link_order_base_obj(self) -> str:
        """Return path prefix for test_link_order_base object."""
        return f"{self.subdir}/test_link_order_base"

    def link_order_caller_obj(self) -> str:
        """Return path prefix for test_link_order_caller object."""
        return f"{self.subdir}/test_link_order_caller"

    def error_obj(self) -> str:
        """Return path prefix for test_error object."""
        return f"{self.subdir}/test_error"

    def ctor_dtor_obj(self) -> str:
        """Return path prefix for test_ctor_dtor object."""
        return f"{self.subdir}/test_ctor_dtor"

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
_cpp_only = [v for v in _all_variants if v.subdir.startswith("cc")]


def _variant_id(v: Variant) -> str:
    return repr(v)


# ---------------------------------------------------------------------------
# Session / library creation
# ---------------------------------------------------------------------------


def test_create_session() -> None:
    """Create an ExecutionSession successfully."""
    session = ExecutionSession()
    assert session is not None


def test_create_library() -> None:
    """Create a DynamicLibrary from an ExecutionSession."""
    session = ExecutionSession()
    lib = session.create_library()
    assert lib is not None


def test_multiple_libraries() -> None:
    """Create multiple named libraries in one session."""
    session = ExecutionSession()
    lib1 = session.create_library("lib1")
    lib2 = session.create_library("lib2")
    assert lib1 is not None
    assert lib2 is not None


# ---------------------------------------------------------------------------
# Load & execute — parametrized over C / C++
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("v", _all_variants, ids=_variant_id)
def test_load_and_execute(v: Variant) -> None:
    """Load test_funcs and verify add/multiply."""
    _, lib = make_lib(v.funcs_obj())
    assert lib.get_function(v.fn("test_add"))(10, 20) == 30
    assert lib.get_function(v.fn("test_multiply"))(7, 6) == 42


@pytest.mark.parametrize("v", _all_variants, ids=_variant_id)
def test_load_and_execute_second_set(v: Variant) -> None:
    """Load test_funcs2 and verify subtract/divide."""
    _, lib = make_lib(v.funcs2_obj())
    assert lib.get_function(v.fn("test_subtract"))(10, 3) == 7
    assert lib.get_function(v.fn("test_divide"))(20, 4) == 5


@pytest.mark.parametrize("v", _all_variants, ids=_variant_id)
def test_function_not_found(v: Variant) -> None:
    """Raise AttributeError for a missing function name."""
    _, lib = make_lib(v.funcs_obj())
    with pytest.raises(AttributeError, match="Module has no function"):
        lib.get_function("nonexistent_function")


# ---------------------------------------------------------------------------
# Multi-object / multi-library — parametrized over C / C++
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("v", _all_variants, ids=_variant_id)
def test_gradually_add_objects(v: Variant) -> None:
    """Add multiple objects incrementally and verify all functions work."""
    _session, lib = make_lib(v.funcs_obj())

    add_func = lib.get_function(v.fn("test_add"))
    mul_func = lib.get_function(v.fn("test_multiply"))
    assert add_func(5, 3) == 8
    assert mul_func(4, 5) == 20

    lib.add(obj(v.funcs2_obj()))
    assert lib.get_function(v.fn("test_subtract"))(10, 3) == 7
    assert lib.get_function(v.fn("test_divide"))(20, 4) == 5

    # First object's functions still work
    assert add_func(10, 20) == 30
    assert mul_func(7, 6) == 42


@pytest.mark.parametrize("v", _all_variants, ids=_variant_id)
def test_two_separate_libraries(v: Variant) -> None:
    """Separate libraries expose only their own functions."""
    session = ExecutionSession()
    _, lib1 = make_lib(v.funcs_obj(), session=session, name="lib1")
    _, lib2 = make_lib(v.funcs2_obj(), session=session, name="lib2")

    assert lib1.get_function(v.fn("test_add"))(5, 3) == 8
    assert lib1.get_function(v.fn("test_multiply"))(4, 5) == 20
    assert lib2.get_function(v.fn("test_subtract"))(10, 3) == 7
    assert lib2.get_function(v.fn("test_divide"))(20, 4) == 5

    with pytest.raises(AttributeError, match="Module has no function"):
        lib1.get_function(v.fn("test_subtract"))
    with pytest.raises(AttributeError, match="Module has no function"):
        lib2.get_function(v.fn("test_add"))


# ---------------------------------------------------------------------------
# Symbol conflicts — parametrized over C / C++
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("v", _all_variants, ids=_variant_id)
def test_symbol_conflict_same_library(v: Variant) -> None:
    """Duplicate symbol in the same library raises an error."""
    _, lib = make_lib(v.funcs_obj())
    assert lib.get_function(v.fn("test_add"))(10, 20) == 30
    with pytest.raises(Exception):
        lib.add(obj(v.conflict_obj()))


@pytest.mark.parametrize("v", _all_variants, ids=_variant_id)
def test_symbol_conflict_different_libraries(v: Variant) -> None:
    """Same symbol in different libraries resolves independently."""
    session = ExecutionSession()
    _, lib1 = make_lib(v.funcs_obj(), session=session, name="lib1")
    _, lib2 = make_lib(v.conflict_obj(), session=session, name="lib2")

    assert lib1.get_function(v.fn("test_add"))(10, 20) == 30
    assert lib2.get_function(v.fn("test_add"))(10, 20) == 1030
    assert lib1.get_function(v.fn("test_multiply"))(5, 6) == 30
    assert lib2.get_function(v.fn("test_multiply"))(5, 6) == 60


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
    _, lib = make_lib(v.call_global_obj())
    add_func = lib.get_function(v.fn("test_call_global_add"))
    assert add_func(10, 20) == 30
    assert add_func(100, 200) == 300

    mul_func = lib.get_function(v.fn("test_call_global_mul"))
    assert mul_func(7, 6) == 42
    assert mul_func(11, 11) == 121


# ---------------------------------------------------------------------------
# Error handling — pure Python (Group 1)
# ---------------------------------------------------------------------------


def test_empty_library() -> None:
    """get_function on an empty library raises AttributeError."""
    session = ExecutionSession()
    lib = session.create_library("empty")
    with pytest.raises(AttributeError, match="Module has no function"):
        lib.get_function("nonexistent")


def test_invalid_object_file_path() -> None:
    """Adding a nonexistent object file raises an error."""
    session = ExecutionSession()
    lib = session.create_library("bad_path")
    with pytest.raises(Exception):
        lib.add("/nonexistent_path/does_not_exist.o")


def test_invalid_object_file_content() -> None:
    """Adding a file with garbage content raises an error."""
    with tempfile.NamedTemporaryFile(suffix=".o", delete=False) as f:
        f.write(b"this is not a valid object file")
        f.flush()
        session = ExecutionSession()
        lib = session.create_library("bad_content")
        with pytest.raises(Exception):
            lib.add(f.name)


def test_multiple_independent_sessions() -> None:
    """Two independent sessions don't interfere with each other."""
    session1 = ExecutionSession()
    session2 = ExecutionSession()
    _, lib1 = make_lib(_all_variants[0].funcs_obj(), session=session1)
    _, lib2 = make_lib(_all_variants[0].funcs2_obj(), session=session2)

    assert lib1.get_function("test_add")(5, 3) == 8
    assert lib2.get_function("test_subtract")(10, 3) == 7

    # Each session's library doesn't see the other's functions
    with pytest.raises(AttributeError, match="Module has no function"):
        lib1.get_function("test_subtract")
    with pytest.raises(AttributeError, match="Module has no function"):
        lib2.get_function("test_add")


# ---------------------------------------------------------------------------
# Type variety — parametrized over C / C++ (Group 2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("v", _all_variants, ids=_variant_id)
def test_zero_arg_function(v: Variant) -> None:
    """Zero-arg function returns constant 42."""
    _, lib = make_lib(v.types_obj())
    assert lib.get_function(v.fn("test_zero_arg"))() == 42


@pytest.mark.parametrize("v", _all_variants, ids=_variant_id)
def test_four_arg_function(v: Variant) -> None:
    """Four integer arguments summed."""
    _, lib = make_lib(v.types_obj())
    assert lib.get_function(v.fn("test_four_args"))(1, 2, 3, 4) == 10
    assert lib.get_function(v.fn("test_four_args"))(100, 200, 300, 400) == 1000


@pytest.mark.parametrize("v", _all_variants, ids=_variant_id)
def test_float_function(v: Variant) -> None:
    """Float multiply returns approximate result."""
    _, lib = make_lib(v.types_obj())
    result = lib.get_function(v.fn("test_float_multiply"))(3.14, 2.0)
    assert result == pytest.approx(6.28)


@pytest.mark.parametrize("v", _all_variants, ids=_variant_id)
def test_void_function(v: Variant) -> None:
    """Void function returns None."""
    _, lib = make_lib(v.types_obj())
    result = lib.get_function(v.fn("test_void_function"))()
    assert result is None


# ---------------------------------------------------------------------------
# Advanced — cross-library linking and error propagation (Group 3)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("v", _all_variants, ids=_variant_id)
def test_set_link_order(v: Variant) -> None:
    """Cross-library symbol resolution via set_link_order."""
    session = ExecutionSession()
    # Base library exports helper_add
    lib_base = session.create_library("base")
    lib_base.add(obj(v.link_order_base_obj()))
    # Caller library references helper_add from base
    lib_caller = session.create_library("caller")
    lib_caller.set_link_order(lib_base)
    lib_caller.add(obj(v.link_order_caller_obj()))

    cross_add = lib_caller.get_function(v.fn("cross_lib_add"))
    assert cross_add(10, 20) == 30
    assert cross_add(100, 200) == 300


@pytest.mark.parametrize("v", _all_variants, ids=_variant_id)
def test_error_propagation(v: Variant) -> None:
    """JIT function that signals an error raises a Python exception."""
    _, lib = make_lib(v.error_obj())
    with pytest.raises(Exception, match="test error"):
        lib.get_function(v.fn("test_error"))()


# ---------------------------------------------------------------------------
# CUDA (optional)
# ---------------------------------------------------------------------------


def test_load_and_execute_cuda_function() -> None:
    """Load and execute CUDA-compiled test objects."""
    _, lib = make_lib("cuda/test_funcs")
    assert lib.get_function("test_add")(10, 20) == 30
    assert lib.get_function("test_multiply")(7, 6) == 42


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

    _, lib = make_lib(v.ctor_dtor_obj())
    lib.get_function(v.fn("main"))()
    del lib

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
# Dylib removal — dropping a DynamicLibrary while its session is still alive.
#
# Dropping the last Python reference to a DynamicLibrary removes the JITDylib
# from the ExecutionSession (releasing its JIT memory) in addition to running
# any pending static destructors. These tests pin that contract and guard
# against regressions in the slab-pool memory-manager refactor, which assumes
# per-dylib deallocation actually happens at drop time.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("v", _all_variants, ids=_variant_id)
def test_drop_empty_library(v: Variant) -> None:
    """Drop an empty library, then reuse the session."""
    session = ExecutionSession()
    lib = session.create_library("empty")
    del lib
    # Session still works.
    lib2 = session.create_library("after")
    lib2.add(obj(v.funcs_obj()))
    assert lib2.get_function(v.fn("test_add"))(1, 2) == 3


@pytest.mark.parametrize("v", _all_variants, ids=_variant_id)
def test_drop_loaded_library_then_recreate(v: Variant) -> None:
    """Drop a library with live JIT code, then create and use a fresh one."""
    session = ExecutionSession()

    lib1 = session.create_library("lib1")
    lib1.add(obj(v.funcs_obj()))
    assert lib1.get_function(v.fn("test_add"))(3, 4) == 7
    del lib1

    lib2 = session.create_library("lib2")
    lib2.add(obj(v.funcs_obj()))
    assert lib2.get_function(v.fn("test_add"))(100, 1) == 101


@pytest.mark.parametrize("v", _all_variants, ids=_variant_id)
def test_repeated_create_drop_many_iterations(v: Variant) -> None:
    """Long create / load / call / drop cycle must not degrade or crash.

    Exercises the memory manager's recycled-region path: each iteration
    frees JIT memory that a later iteration may be handed back by the arena.
    """
    session = ExecutionSession()
    for i in range(32):
        lib = session.create_library(f"iter_{i}")
        lib.add(obj(v.funcs_obj()))
        assert lib.get_function(v.fn("test_multiply"))(i + 1, 2) == (i + 1) * 2
        del lib


@pytest.mark.parametrize("v", _all_variants, ids=_variant_id)
def test_drop_one_library_does_not_affect_another(v: Variant) -> None:
    """Dropping one library must leave unrelated sibling libraries working."""
    session = ExecutionSession()

    keep = session.create_library("keep")
    keep.add(obj(v.funcs_obj()))

    drop = session.create_library("drop")
    drop.add(obj(v.funcs2_obj()))
    assert drop.get_function(v.fn("test_subtract"))(10, 3) == 7

    del drop
    gc.collect()

    # Untouched sibling still works; room now exists for a fresh library.
    assert keep.get_function(v.fn("test_add"))(5, 5) == 10
    fresh = session.create_library("fresh")
    fresh.add(obj(v.funcs2_obj()))
    assert fresh.get_function(v.fn("test_divide"))(20, 4) == 5


@pytest.mark.parametrize("v", _all_variants, ids=_variant_id)
def test_captured_function_keeps_library_alive(v: Variant) -> None:
    """A captured Function keeps the library alive after the lib handle is dropped."""
    session = ExecutionSession()
    lib = session.create_library("hold")
    lib.add(obj(v.funcs_obj()))

    captured = lib.get_function(v.fn("test_add"))
    del lib
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
    lib = session.create_library("ctor_dtor")
    lib.add(obj(v.ctor_dtor_obj()))
    lib.get_function(v.fn("main"))()

    ctor_len = len(log)
    assert "<main>" in log, f"main not observed: {log}"

    del lib
    gc.collect()

    post = log[ctor_len:]
    # Platform-specific fini markers: ELF (.fini_array / .dtors), Mach-O
    # (__mod_term_func surfaces as fini_array), COFF (.CRT$XT*).
    if sys.platform == "win32":
        markers = ("crt.XT",)
    else:
        markers = ("dtors", "fini_array")
    assert any(m in entry for entry in post for m in markers), (
        f"no static destructors on drop (platform={sys.platform}): "
        f"pre={log[:ctor_len]} post={post}"
    )


@pytest.mark.parametrize("v", _all_variants, ids=_variant_id)
def test_drop_caller_leaves_base_usable(v: Variant) -> None:
    """Dropping a set_link_order caller leaves its base library usable."""
    session = ExecutionSession()

    base = session.create_library("base")
    base.add(obj(v.link_order_base_obj()))

    caller = session.create_library("caller")
    caller.set_link_order(base)
    caller.add(obj(v.link_order_caller_obj()))
    assert caller.get_function(v.fn("cross_lib_add"))(1, 2) == 3

    del caller
    gc.collect()

    caller2 = session.create_library("caller2")
    caller2.set_link_order(base)
    caller2.add(obj(v.link_order_caller_obj()))
    assert caller2.get_function(v.fn("cross_lib_add"))(10, 20) == 30


@pytest.mark.parametrize("v", _all_variants, ids=_variant_id)
def test_drop_all_libraries_then_session(v: Variant) -> None:
    """Dropping every library before the session still leaves clean teardown."""
    session = ExecutionSession()
    libs = []
    for i in range(4):
        lib = session.create_library(f"lib_{i}")
        lib.add(obj(v.funcs_obj()))
        libs.append(lib)
    for lib in reversed(libs):
        del lib
    libs.clear()
    gc.collect()
    # Session destructor runs at end-of-test; a regression in removal would
    # surface as a crash under pytest.
