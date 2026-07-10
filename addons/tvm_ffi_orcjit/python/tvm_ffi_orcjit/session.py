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
"""ORC JIT Execution Session."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING

import tvm_ffi
from tvm_ffi import Module, Object, register_object

from . import _ffi_api

if TYPE_CHECKING:
    from collections.abc import Sequence


@register_object("tvm_ffi_orcjit.ExecutionSession")
class ExecutionSession(Object):
    """ORC JIT Execution Session.

    Manages the LLVM ORC JIT execution environment and creates dynamic libraries (JITDylibs).
    This is the top-level context for JIT compilation and symbol management.

    Prefer :func:`default_session` for a process-wide shared session; construct
    an ``ExecutionSession`` directly only for an isolated session or a tuned arena.

    Examples
    --------
    >>> session = ExecutionSession()
    >>> mod = session.load_module("add.o")
    >>> add_func = mod.get_function("add")

    """

    def __init__(
        self,
        orc_rt: str | Path | bytes | bytearray | None = "auto",
        slab_size: int = 0,
    ) -> None:
        """Initialize ExecutionSession.

        Parameters
        ----------
        orc_rt : str or Path or bytes or None
            Which ORC runtime to install. Linux/ELF only — ignored on macOS and
            Windows, which never configure an ORC platform.

            - ``"auto"`` (default): the runtime embedded in this extension.
            - a path (``str`` or ``Path``): a custom liborc_rt archive on disk.
            - ``bytes``: a custom liborc_rt archive held in memory.
            - ``None``: no ORC platform at all.

            A custom runtime (path or bytes) must match the LLVM/compiler-rt this
            extension was built against; ``"auto"`` is almost always what you want.
        slab_size : int
            Per-slab capacity in bytes for the JIT memory manager. Linux only —
            ignored on macOS and Windows, where the slab allocator is compiled
            out. 0 = arch default (64 MB; initial slab halves on mmap failure
            down to 8 MB under RLIMIT_AS / container limits), >0 = custom size,
            <0 = disable slab allocator (LLJIT uses its default scattered-mmap
            allocator).

            The session holds a growable pool of slabs: a fresh slab is mmap'd
            on demand when no existing one can fit a graph. Graphs that don't
            fit a normal slab trigger a power-of-2 larger slab (slab_size,
            2*slab_size, ...) sized to fit. Drained slabs stay mapped until the
            session is destroyed or ``clear_free_slabs()`` is called.

        """
        # Normalize to what the C++ ctor (Optional<Variant<String, Bytes>>) reads:
        # None -> no platform; "" -> auto/embedded; str -> path; bytes -> in-memory.
        rt: str | bytes | None
        if orc_rt is None:
            rt = None
        elif orc_rt == "auto":
            rt = ""
        elif isinstance(orc_rt, (bytes, bytearray)):
            rt = bytes(orc_rt)
        elif isinstance(orc_rt, (str, Path)):
            rt = str(orc_rt)
        else:
            raise TypeError(
                "orc_rt must be 'auto', a path (str or Path), liborc_rt bytes, or None, "
                f"but got {type(orc_rt).__name__}"
            )
        self.__init_handle_by_constructor__(_ffi_api.ExecutionSession, rt, slab_size)  # type: ignore

    def load_module(
        self,
        objects: str | Path | bytes | bytearray | Sequence[str | Path | bytes | bytearray],
        name: str = "",
        keep_module_alive: bool = False,
    ) -> Module:
        """Load one or more object files into a fresh module.

        All objects are linked into one dynamic library (JITDylib), context
        symbols are wired up, and any embedded library binary is expanded — so
        the result behaves like a module loaded by :func:`tvm_ffi.load_module`.

        Parameters
        ----------
        objects : str or Path or bytes or bytearray, or a sequence of them
            Object files to load, given as paths and/or in-memory object-file
            images. A single item is accepted as shorthand for a one-element list.
        name : str
            Optional name for the underlying library. Auto-generated if empty.
        keep_module_alive : bool
            If True, pin the module in the runtime's process-global registry
            (see Notes). Defaults to False.

        Returns
        -------
        Module
            A :class:`tvm_ffi.Module` whose imports and library context are fully
            wired.

        Notes
        -----
        ``keep_module_alive`` mirrors :func:`tvm_ffi.load_module`'s option of
        the same name. When True, the module is inserted into the runtime's
        process-global module registry, so its JITDylib — and every function
        pointer, deleter, and static allocation it owns — stays mapped until
        the interpreter unloads ``libtvm_ffi``. Use this when Objects produced
        by the module may outlive the local ``mod`` reference (e.g., a
        JIT-allocated ``String`` or ``Array`` returned to Python and held past
        ``del mod``). When False (default), the caller owns the module's
        lifetime and its JIT memory is reclaimed on drop; callers must ensure
        no JIT-produced Object outlives the returned module.

        Pinning is transitive: the module holds a strong reference to its
        :class:`ExecutionSession`, so pinning one module also keeps that
        session (and its slab-pool memory manager) alive for the process
        lifetime. Reclaim is slab-granular — a slab shared between a pinned
        and an unpinned module stays mapped until the pinned module is gone.

        Examples
        --------
        >>> session = default_session()
        >>> mod = session.load_module(["a.o", "b.o", Path("c.o").read_bytes()])
        >>> mod.my_function(1, 2)

        """
        if isinstance(objects, (str, Path, bytes, bytearray)):
            objects = [objects]
        normalized: list[str | bytes] = []
        for obj in objects:
            if isinstance(obj, (bytes, bytearray)):
                normalized.append(bytes(obj))
            elif isinstance(obj, (str, Path)):
                normalized.append(str(obj))
            else:
                raise TypeError(
                    "load_module objects must be a path (str or Path) or object-file "
                    f"bytes, but got {type(obj).__name__}"
                )
        mod = _ffi_api.SessionLoadModule(self, normalized, name)  # type: ignore
        if keep_module_alive:
            tvm_ffi._ffi_api.ModuleGlobalsAdd(mod)  # type: ignore
        return mod

    def clear_free_slabs(self) -> int:
        """Release drained slabs (no live JIT allocations) back to the OS.

        Call this after dropping a batch of libraries to reclaim RSS.
        Fresh slabs that have never been allocated on are preserved, so
        the session remains ready to accept new work.

        Safety: call when no JIT work is in flight on another thread. From
        single-threaded Python this is always safe; once ``del lib`` has
        returned, the C++ destructor has finished and the slab's live count
        reflects the drop.

        Returns
        -------
        int
            Number of slabs actually munmap'd. Returns 0 on macOS/Windows
            (slab pool compiled out) or when the pool is disabled via
            ``slab_size=-1``.

        """
        return int(_ffi_api.SessionClearFreeSlabs(self))  # type: ignore


_default_session: ExecutionSession | None = None
_default_session_lock = threading.Lock()


def default_session() -> ExecutionSession:
    """Return the process-wide shared execution session.

    A single leaked, never-destroyed session shared by all callers in the
    process, so they share one LLVM ``ExecutionSession`` — hence process
    symbols, the slab arena, and cross-library linking. Created on first call
    and cached for the lifetime of the process.

    The session uses the ORC runtime embedded in the extension (no on-disk path
    lookup). For an isolated session or a tuned arena, construct an
    :class:`ExecutionSession` directly.

    Returns
    -------
    ExecutionSession
        The shared execution session.

    Examples
    --------
    >>> import tvm_ffi_orcjit as oj
    >>> session = oj.default_session()
    >>> mod = session.load_module("dylib0.o")

    """
    global _default_session  # noqa: PLW0603 — module-level singleton cache
    # Double-checked locking: the FFI call releases the GIL, so guard the
    # first-call caching to avoid two threads racing to fetch it.
    if _default_session is None:
        with _default_session_lock:
            if _default_session is None:
                _default_session = _ffi_api.GlobalDefaultSession()
    return _default_session
