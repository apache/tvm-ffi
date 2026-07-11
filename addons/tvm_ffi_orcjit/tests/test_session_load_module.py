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
"""Tests for the high-level session ``load_module`` API and shared session.

These cover the refactor's new surface:

- ``default_session()`` — the process-wide shared execution session.
- ``ExecutionSession.load_module`` — unified path/bytes input, single item or
  list, returning a plain ``tvm_ffi.Module``.
- Concurrent create / load / call / drop on the shared session.

They deliberately use pure-C objects: the C ABI path is uniform across
platforms, whereas some C++ object layouts hit environment-specific JITLink
limitations unrelated to this API.
"""

from __future__ import annotations

import gc
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import tvm_ffi
from tvm_ffi_orcjit import ExecutionSession, default_session
from utils import build_test_objects

OBJ_DIR = build_test_objects()


def obj(name: str) -> str:
    """Return path to a pre-built test object file, or skip if missing."""
    path = OBJ_DIR / f"{name}.o"
    if not path.exists():
        pytest.skip(f"{path.name} not found (not built)")
    return str(path)


def _find_orc_rt_archive() -> str:
    """Locate a liborc_rt archive under LLVM_PREFIX, or skip.

    Used only by the custom-``orc_rt`` override tests. There is no bundled copy
    next to the extension anymore (the runtime is embedded), so these tests fall
    back to the archive shipped with the build's LLVM.
    """
    import os  # noqa: PLC0415

    prefix = os.environ.get("LLVM_PREFIX")
    if not prefix:
        pytest.skip("LLVM_PREFIX not set; cannot locate a liborc_rt archive")
    matches = sorted(Path(prefix).glob("lib/clang/*/lib/*/liborc_rt*.a"))
    if not matches:
        pytest.skip("no liborc_rt archive found under LLVM_PREFIX")
    return str(matches[0])


# ---------------------------------------------------------------------------
# default_session()
# ---------------------------------------------------------------------------


def test_default_session_is_cached() -> None:
    """default_session() returns the same shared instance every time."""
    assert default_session() is default_session()


def test_default_session_is_execution_session() -> None:
    """default_session() returns an ExecutionSession."""
    assert isinstance(default_session(), ExecutionSession)


def test_default_session_load_and_call() -> None:
    """A module loaded on the shared session is callable."""
    mod = default_session().load_module(obj("c/test_funcs"))
    assert mod.test_add(10, 20) == 30
    assert mod.test_multiply(7, 6) == 42


def test_default_session_concurrent_first_call() -> None:
    """Many threads racing the first default_session() all get one shared instance.

    Guards the double-checked-locking init: the underlying FFI call releases the
    GIL, so without the lock concurrent first callers could build separate
    sessions.
    """
    num_workers = 16
    barrier = threading.Barrier(num_workers)

    def get(_worker: int) -> ExecutionSession:
        barrier.wait()
        return default_session()

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        sessions = list(pool.map(get, range(num_workers)))
    assert all(s is sessions[0] for s in sessions)


# ---------------------------------------------------------------------------
# orc_rt override — custom runtime selectors on a user-created session.
# (The "auto" default is covered via ExecutionSession() elsewhere.)
# ---------------------------------------------------------------------------

# A custom ORC runtime is only installed on Linux/ELF; elsewhere it is ignored.
elf_only = pytest.mark.skipif(
    sys.platform != "linux", reason="ORC platform (custom liborc_rt) is Linux/ELF only"
)


@elf_only
def test_orc_rt_explicit_path() -> None:
    """A custom on-disk liborc_rt archive is accepted (str and Path)."""
    archive = _find_orc_rt_archive()
    for spec in (archive, Path(archive)):
        mod = ExecutionSession(orc_rt=spec).load_module(obj("c/test_funcs"))
        assert mod.test_add(4, 5) == 9


@elf_only
def test_orc_rt_in_memory_bytes() -> None:
    """A liborc_rt archive supplied as in-memory bytes is accepted."""
    data = Path(_find_orc_rt_archive()).read_bytes()
    mod = ExecutionSession(orc_rt=data).load_module(obj("c/test_funcs"))
    assert mod.test_add(6, 7) == 13


def test_orc_rt_none_no_platform() -> None:
    """orc_rt=None runs with no ORC platform; C-ABI objects still load."""
    mod = ExecutionSession(orc_rt=None).load_module(obj("c/test_funcs"))
    assert mod.test_add(8, 9) == 17


def test_orc_rt_rejects_bad_type() -> None:
    """A non-str, non-bytes, non-None selector raises a clear TypeError."""
    with pytest.raises(TypeError, match=r"orc_rt must be"):
        ExecutionSession(orc_rt=12345)


# ---------------------------------------------------------------------------
# load_module — input shapes
# ---------------------------------------------------------------------------


def test_load_module_single_path() -> None:
    """A single path (not wrapped in a list) loads correctly."""
    mod = default_session().load_module(obj("c/test_funcs"))
    assert isinstance(mod, tvm_ffi.Module)
    assert mod.test_add(1, 2) == 3


def test_load_module_single_bytes() -> None:
    """A single in-memory object image loads correctly."""
    blob = Path(obj("c/test_funcs")).read_bytes()
    mod = default_session().load_module(blob)
    assert mod.test_add(4, 5) == 9


def test_load_module_list_of_paths() -> None:
    """A list of object paths is linked into one module."""
    mod = default_session().load_module([obj("c/test_funcs"), obj("c/test_funcs2")])
    assert mod.test_add(10, 20) == 30
    assert mod.test_subtract(10, 3) == 7
    assert mod.test_divide(20, 4) == 5


def test_load_module_mixed_path_and_bytes() -> None:
    """A list mixing a path and in-memory bytes links into one module."""
    blob = Path(obj("c/test_funcs2")).read_bytes()
    mod = default_session().load_module([obj("c/test_funcs"), blob])
    assert mod.test_multiply(7, 6) == 42
    assert mod.test_subtract(9, 4) == 5


def test_load_module_returns_plain_module_kind() -> None:
    """The returned module reports the orcjit kind."""
    mod = default_session().load_module(obj("c/test_funcs"))
    assert mod.kind == "orcjit"


def test_load_module_rejects_bad_element_type() -> None:
    """A non-path, non-bytes element raises a clear TypeError."""
    with pytest.raises(TypeError, match=r"path .* or object-file bytes"):
        default_session().load_module([12345])


def test_load_module_empty_list() -> None:
    """An empty object list yields a valid, empty module (no functions)."""
    mod = default_session().load_module([])
    assert isinstance(mod, tvm_ffi.Module)
    with pytest.raises(AttributeError, match="Module has no function"):
        mod.get_function("anything")


def test_load_module_named() -> None:
    """An explicit name is accepted."""
    mod = default_session().load_module(obj("c/test_funcs"), name="my_named_lib")
    assert mod.test_add(2, 2) == 4


def test_load_module_context_symbol_injected_eagerly() -> None:
    """Eager load wires the library-context pointer before any get_function."""
    mod = default_session().load_module(obj("c/test_context"))
    # The context slot is populated at load time, so the first (and only)
    # lookup already observes a non-null __tvm_ffi__library_ctx.
    assert mod.context_is_set() == 1


def test_load_module_drop_frees() -> None:
    """Dropping every reference to a loaded module tears down its dylib."""
    session = ExecutionSession()  # isolated session so teardown is observable
    mod = session.load_module(obj("c/test_funcs"))
    assert mod.test_add(1, 1) == 2
    del mod
    gc.collect()
    # Session still usable after the drop.
    mod2 = session.load_module(obj("c/test_funcs2"))
    assert mod2.test_subtract(5, 2) == 3


# ---------------------------------------------------------------------------
# Embedded library binary — the __tvm_ffi__library_bin path (§3.3).
#
# When a loaded object embeds a serialized import tree, load_module must
# deserialize each embedded module by kind (via ffi.Module.load_from_bytes.*),
# wire the imports onto the "_lib" placeholder (the JIT dylib itself), and
# return the root module — so cross-module lookups resolve through imports.
# ---------------------------------------------------------------------------


def _u64(value: int) -> bytes:
    return int(value).to_bytes(8, "little")


def _blob_str(data: bytes) -> bytes:
    return _u64(len(data)) + data


def _u64_vec(values: list[int]) -> bytes:
    return _u64(len(values)) + b"".join(_u64(v) for v in values)


def _make_library_bin(
    modules: list[tuple[str, bytes]], indptr: list[int], children: list[int]
) -> bytes:
    """Serialize a library-bin blob matching the core ProcessLibraryBin format."""
    stream = _u64_vec(indptr) + _u64_vec(children)
    for kind, payload in modules:
        stream += _blob_str(kind.encode())
        if kind != "_lib":
            stream += _blob_str(payload)
    return _u64(len(stream)) + stream


def _raw_library_bin(stream: bytes) -> bytes:
    """Wrap a hand-crafted (possibly malformed) inner stream with its length header."""
    return _u64(len(stream)) + stream


def _build_library_bin_object(tmp_path: Path, blob: bytes, extra_export: str) -> str:
    """Compile a C object exposing __tvm_ffi__library_bin plus one function."""
    try:
        import tvm_ffi.cpp  # noqa: PLC0415
    except ImportError:
        pytest.skip("tvm_ffi.cpp not available for on-the-fly object build")

    array_body = ", ".join(str(b) for b in blob)
    src = tmp_path / "libbin.c"
    src.write_text(
        f"""
        #include <tvm/ffi/c_api.h>
        #include <stdint.h>
        TVM_FFI_DLL_EXPORT unsigned char __tvm_ffi__library_bin[] = {{ {array_body} }};
        TVM_FFI_DLL_EXPORT int {extra_export}(void* self, const TVMFFIAny* args,
                                              int32_t num_args, TVMFFIAny* result) {{
          (void)self; (void)args; (void)num_args;
          result->type_index = kTVMFFIInt;
          result->zero_padding = 0;
          result->v_int64 = 777;
          return 0;
        }}
        """
    )
    try:
        return tvm_ffi.cpp.build(
            name="libbin",
            sources=[str(src)],
            output="libbin.o",
            extra_cflags=["-O2"],
            build_directory=str(tmp_path / ".build_libbin"),
        )
    except Exception as exc:
        pytest.skip(f"could not build library-bin object: {exc}")


def test_load_module_expands_embedded_library_bin(tmp_path: Path) -> None:
    """An embedded library binary is deserialized and its imports are wired."""

    # A custom module kind whose loader returns a real orcjit module. This also
    # forces the loader to re-enter load_module while the outer call holds the
    # session lock — exercising the recursive session lock.
    @tvm_ffi.register_global_func("ffi.Module.load_from_bytes.orcjit_test_probe", override=True)
    def _load_probe(_data: bytes) -> tvm_ffi.Module:
        return default_session().load_module(obj("c/test_funcs2"))

    # 2-module import tree: module 0 = "_lib" (the JIT dylib), module 1 = the
    # custom probe module. Module 0 imports module 1.
    #   indptr[i]..indptr[i+1] indexes children[]: module 0 -> child_indices[0:1] = [1]
    blob = _make_library_bin(
        modules=[("_lib", b""), ("orcjit_test_probe", b"")],
        indptr=[0, 1, 1],
        children=[1],
    )
    libbin_obj = _build_library_bin_object(tmp_path, blob, extra_export="__tvm_ffi_probe_marker")

    # Load the JIT dylib's own function object alongside the library-bin object.
    root = default_session().load_module([obj("c/test_funcs"), libbin_obj])

    # The dylib's own exported function resolves directly.
    assert root.get_function("test_add")(10, 20) == 30
    assert root.get_function("probe_marker")() == 777

    # The embedded module's function resolves through the wired import tree.
    assert root.get_function("test_subtract", query_imports=True)(10, 3) == 7
    assert len(root.imports) == 1


# ---------------------------------------------------------------------------
# Concurrency — the shared session driven by many threads at once.
#
# Exercises the recursive session lock: overlapping create / add / lookup /
# drop on one shared ExecutionSession must not corrupt linker state.
# ---------------------------------------------------------------------------


def test_shared_session_concurrent_load_call_drop() -> None:
    """Many threads load + call + drop on the shared session simultaneously."""
    session = default_session()
    num_workers = 8
    barrier = threading.Barrier(num_workers)

    def worker(i: int) -> int:
        barrier.wait()
        total = 0
        for _ in range(4):
            mod = session.load_module(obj("c/test_funcs"), name=f"conc_{i}")
            total += mod.test_add(i, i)
            total += mod.test_multiply(i, 2)
            del mod
        return total

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        results = list(pool.map(worker, range(num_workers)))

    # Each worker: 4 * (i+i + i*2) = 4 * 4i = 16i
    assert results == [16 * i for i in range(num_workers)]


def test_shared_session_concurrent_distinct_modules() -> None:
    """Concurrent loads of different objects resolve independently."""
    session = default_session()
    num_workers = 6
    barrier = threading.Barrier(num_workers)

    def worker(i: int) -> tuple[int, int]:
        barrier.wait()
        m_funcs = session.load_module(obj("c/test_funcs"))
        m_funcs2 = session.load_module(obj("c/test_funcs2"))
        return m_funcs.test_add(i, 1), m_funcs2.test_subtract(i + 5, i)

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        results = list(pool.map(worker, range(num_workers)))

    assert results == [(i + 1, 5) for i in range(num_workers)]


# ---------------------------------------------------------------------------
# Malformed embedded library binary — the parser must reject corrupt blobs
# with a clean RuntimeError, never crash. Each case targets one guard in
# ProcessEmbeddedLibraryBin / BlobReader.
# ---------------------------------------------------------------------------


def _load_bad_blob(tmp_path: Path, blob: bytes) -> tvm_ffi.Module:
    """Build an object carrying `blob` as __tvm_ffi__library_bin and load it."""
    libbin_obj = _build_library_bin_object(tmp_path, blob, extra_export="__tvm_ffi_probe_marker")
    return default_session().load_module(libbin_obj)


def test_malformed_indptr_single_entry(tmp_path: Path) -> None:
    """Indptr with one entry (num_modules == 0) must be rejected, not return modules[0]."""
    blob = _make_library_bin(modules=[], indptr=[0], children=[])
    with pytest.raises(Exception, match="at least 2 entries"):
        _load_bad_blob(tmp_path, blob)


def test_malformed_indptr_empty(tmp_path: Path) -> None:
    """An empty indptr vector is rejected."""
    blob = _make_library_bin(modules=[], indptr=[], children=[])
    with pytest.raises(Exception, match="at least 2 entries"):
        _load_bad_blob(tmp_path, blob)


def test_malformed_indptr_nonzero_start(tmp_path: Path) -> None:
    """Indptr must start at 0."""
    blob = _make_library_bin(modules=[("_lib", b"")], indptr=[1, 1], children=[])
    with pytest.raises(Exception, match="must start at 0"):
        _load_bad_blob(tmp_path, blob)


def test_malformed_indptr_non_monotonic(tmp_path: Path) -> None:
    """Indptr must be non-decreasing."""
    # 3 entries -> 2 modules; indptr goes 0 -> 2 -> 1 (decreasing).
    blob = _make_library_bin(
        modules=[("_lib", b""), ("_lib", b"")], indptr=[0, 2, 1], children=[0, 0]
    )
    with pytest.raises(Exception, match="non-decreasing"):
        _load_bad_blob(tmp_path, blob)


def test_malformed_indptr_tail_exceeds_children(tmp_path: Path) -> None:
    """indptr.back() must not exceed the child index count."""
    blob = _make_library_bin(modules=[("_lib", b"")], indptr=[0, 5], children=[])
    with pytest.raises(Exception, match="exceeds child index count"):
        _load_bad_blob(tmp_path, blob)


def test_malformed_child_index_out_of_range(tmp_path: Path) -> None:
    """A child index >= module count must be rejected."""
    blob = _make_library_bin(modules=[("_lib", b"")], indptr=[0, 1], children=[9])
    with pytest.raises(Exception, match="child index out of range"):
        _load_bad_blob(tmp_path, blob)


def test_malformed_truncated_stream(tmp_path: Path) -> None:
    """A length header larger than the actual bytes triggers a bounds error."""
    # Inner stream claims a u64 vector of 4 entries but supplies no bytes for them.
    inner = _u64(4)  # count=4, then nothing
    blob = _raw_library_bin(inner)
    with pytest.raises(Exception, match=r"end of blob|exceeds remaining|exceeds blob"):
        _load_bad_blob(tmp_path, blob)


def test_malformed_oversize_vector_count(tmp_path: Path) -> None:
    """A huge vector count is rejected before reserve(), not OOM."""
    inner = _u64(2**60)  # absurd count with no backing bytes
    blob = _raw_library_bin(inner)
    with pytest.raises(Exception, match="exceeds remaining blob"):
        _load_bad_blob(tmp_path, blob)


def test_malformed_unregistered_kind(tmp_path: Path) -> None:
    """An embedded module of unknown kind reports the missing loader."""
    blob = _make_library_bin(
        modules=[("_lib", b""), ("no_such_kind", b"payload")],
        indptr=[0, 1, 1],
        children=[1],
    )
    with pytest.raises(Exception, match=r"load_from_bytes\.no_such_kind is not registered"):
        _load_bad_blob(tmp_path, blob)


def test_malformed_loader_returns_non_module(tmp_path: Path) -> None:
    """A loader returning a non-Module (e.g. None) is rejected by the cast."""

    @tvm_ffi.register_global_func("ffi.Module.load_from_bytes.orcjit_null_probe", override=True)
    def _load_null(_data: bytes):  # noqa: ANN202
        return None

    blob = _make_library_bin(
        modules=[("_lib", b""), ("orcjit_null_probe", b"x")],
        indptr=[0, 1, 1],
        children=[1],
    )
    with pytest.raises(Exception, match=r"Cannot convert.*ffi\.Module"):
        _load_bad_blob(tmp_path, blob)
