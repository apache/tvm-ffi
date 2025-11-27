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
"""ORC JIT Dynamic Library."""

from __future__ import annotations

from pathlib import Path

from tvm_ffi import Module


class DynamicLibrary(Module):
    """ORC JIT Dynamic Library (JITDylib).

    Represents a collection of symbols that can be loaded from object files and linked
    against other dynamic libraries. Supports JIT compilation and symbol resolution.

    Examples
    --------
    >>> session = create_session()
    >>> lib = session.create_library()
    >>> lib.add("add.o")
    >>> lib.add("multiply.o")
    >>> add_func = lib.get_function("add")
    >>> result = add_func(1, 2)

    """

    def add(self, object_file: str | Path) -> None:
        """Add an object file to this dynamic library.

        Parameters
        ----------
        object_file : str or Path
            Path to the object file to load.

        Examples
        --------
        >>> lib.add("add.o")
        >>> lib.add(Path("multiply.o"))

        """
        if isinstance(object_file, Path):
            object_file = str(object_file)
        self.get_function("orcjit.add_object_file")(object_file)

    def set_link_order(self, *libraries: DynamicLibrary) -> None:
        """Set the link order for symbol resolution.

        When resolving symbols, this library will search in the specified libraries
        in the order provided. This replaces any previous link order.

        Parameters
        ----------
        *libraries : DynamicLibrary
            One or more dynamic libraries to search for symbols (in order).

        Examples
        --------
        >>> session = create_session()
        >>> lib_utils = session.create_library()
        >>> lib_utils.add("utils.o")
        >>> lib_core = session.create_library()
        >>> lib_core.add("core.o")
        >>> lib_main = session.create_library()
        >>> lib_main.add("main.o")
        >>> # main can call symbols from utils and core (utils searched first)
        >>> lib_main.set_link_order(lib_utils, lib_core)

        """
        self.get_function("orcjit.set_link_order")(libraries)
