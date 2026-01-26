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

import platform
import subprocess
from pathlib import Path

from tvm_ffi import Object, register_object

from . import _ffi_api
from .dylib import DynamicLibrary


def _find_orc_rt_library(clang_path: str = "clang") -> str | None:
    """Find the liborc_rt library using clang -print-runtime-dir.

    If the path returned by clang -print-runtime-dir does not exist,
    search recursively in the parent directory as a fallback.
    """
    arch = platform.machine()
    lib_pattern = f"liborc_rt-{arch}.a"
    print(f"lib_pattern: {lib_pattern}")

    try:
        result = subprocess.run(
            [clang_path, "-print-runtime-dir"],
            capture_output=True,
            text=True,
            check=True,
        )
        runtime_dir = Path(result.stdout.strip())

        if runtime_dir.exists():
            lib_path = runtime_dir / lib_pattern
            if lib_path.exists():
                return str(lib_path)
            for lib_path in runtime_dir.glob(f"**/liborc_rt*{arch}*"):
                return str(lib_path)
        else:
            # Fallback: search recursively in parent directory
            search_dir = runtime_dir.parent
            if search_dir.exists():
                for lib_path in search_dir.glob(f"**/{lib_pattern}"):
                    return str(lib_path)
                for lib_path in search_dir.glob(f"**/liborc_rt*{arch}*"):
                    return str(lib_path)

        return None
    except (subprocess.CalledProcessError, FileNotFoundError):
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

    def __init__(self, orc_rt_path: str | None = None) -> None:
        """Initialize ExecutionSession.

        Args:
            orc_rt_path: Optional path to the liborc_rt library. If not provided,
                        it will be automatically discovered using clang.

        """
        if orc_rt_path is None:
            orc_rt_path = _find_orc_rt_library()
            if orc_rt_path is None:
                raise RuntimeError(
                    "Could not find liborc_rt library. "
                    "Please ensure clang is installed and accessible, "
                    "or provide the path explicitly."
                )
        self.__init_handle_by_constructor__(_ffi_api.ExecutionSession, orc_rt_path)  # type: ignore

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
