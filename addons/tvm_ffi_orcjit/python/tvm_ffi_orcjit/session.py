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

import sys

from tvm_ffi import Object, register_object

from . import _ffi_api, _lib_dir
from .dylib import DynamicLibrary


def _find_orc_rt_library() -> str | None:
    """Find the bundled liborc_rt library in the same directory as the .so/.dll."""
    # Windows: skip ORC runtime entirely. LLVM's COFFPlatform (loaded via
    # ExecutorNativePlatform with liborc_rt) depends on MSVC C++ runtime symbols
    # that are not available in the JIT environment. On Windows, ORC JIT uses a
    # C-only strategy: JIT objects are compiled as pure C (TVMFFISafeCallType ABI),
    # avoiding all C++ runtime dependencies (magic statics, RTTI, sized delete,
    # SEH, COMDAT). Our custom InitFiniPlugin handles .CRT$XC*/.CRT$XT* init/fini
    # sections, and DLLImportDefinitionGenerator resolves __imp_ DLL import stubs.
    if sys.platform == "win32":
        return None
    patterns = ["liborc_rt*.a"]
    for pattern in patterns:
        for lib_path in _lib_dir.glob(pattern):
            return str(lib_path)
    return None


@register_object("orcjit.ExecutionSession")
class ExecutionSession(Object):
    """ORC JIT Execution Session.

    Manages the LLVM ORC JIT execution environment and creates dynamic libraries (JITDylibs).
    This is the top-level context for JIT compilation and symbol management.

    Examples
    --------
    >>> session = ExecutionSession()
    >>> lib = session.create_library(name="main")
    >>> lib.add("add.o")
    >>> add_func = lib.get_function("add")

    """

    def __init__(self, orc_rt_path: str | None = None, slab_size: int = 0) -> None:
        """Initialize ExecutionSession.

        Args:
            orc_rt_path: Optional path to the liborc_rt library. If not provided,
                        it will be automatically discovered using clang.
            slab_size: Per-slab capacity in bytes for the JIT memory manager.
                       Linux only — ignored on macOS and Windows, where the
                       slab allocator is compiled out.
                       0 = arch default (64 MB; initial slab halves on mmap
                       failure down to 8 MB under RLIMIT_AS / container
                       limits), >0 = custom size, <0 = disable slab allocator
                       (LLJIT uses its default scattered-mmap allocator).

                       The session holds a growable pool of slabs: a fresh
                       slab is mmap'd on demand when no existing one can fit
                       a graph, and graphs larger than slab_size go to a
                       dedicated oversize slab sized to fit. Drained slabs
                       stay mapped until the session is destroyed.

        """
        if orc_rt_path is None:
            orc_rt_path = _find_orc_rt_library()
            if orc_rt_path is None:
                orc_rt_path = ""
        self.__init_handle_by_constructor__(_ffi_api.ExecutionSession, orc_rt_path, slab_size)  # type: ignore

    def create_library(self, name: str = "") -> DynamicLibrary:
        """Create a new dynamic library associated with this execution session.

        Args:
            name: Optional name for the library. If empty, a unique name will be generated.

        Returns:
            A new DynamicLibrary instance.

        """
        handle = _ffi_api.ExecutionSessionCreateDynamicLibrary(self, name)  # type: ignore
        lib = DynamicLibrary.__new__(DynamicLibrary)
        lib.__move_handle_from__(handle)
        return lib

    def clear_free_slabs(self) -> int:
        """Release drained slabs (no live JIT allocations) back to the OS.

        Call this after dropping a batch of libraries to reclaim RSS.
        Fresh slabs that have never been allocated on are preserved, so
        the session remains ready to accept new work.

        Safety: call when no JIT work is in flight on another thread. From
        single-threaded Python this is always safe; once ``del lib`` has
        returned, the C++ destructor has finished and the slab's live count
        reflects the drop.

        Returns:
            Number of slabs actually munmap'd. Returns 0 on macOS/Windows
            (slab pool compiled out) or when the pool is disabled via
            ``slab_size=-1``.

        """
        return int(_ffi_api.ExecutionSessionClearFreeSlabs(self))  # type: ignore
