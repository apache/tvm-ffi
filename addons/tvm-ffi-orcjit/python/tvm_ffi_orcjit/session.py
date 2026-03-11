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

from tvm_ffi import Object, register_object

from . import _ffi_api, _lib_dir
from .dylib import DynamicLibrary


def _find_orc_rt_library() -> str | None:
    """Find the bundled liborc_rt library in the same directory as the .so."""
    for lib_path in _lib_dir.glob("liborc_rt*.a"):
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
                    "Could not find bundled liborc_rt library. "
                    "Please reinstall the package or provide the path explicitly."
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
