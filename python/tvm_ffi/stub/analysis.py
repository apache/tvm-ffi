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
"""Analysis utilities."""

from __future__ import annotations

from tvm_ffi.registry import list_global_func_names

from . import consts as C
from .file_utils import FileInfo
from .utils import Options


def collect_global_funcs() -> dict[str, list[str]]:
    """Collect global functions from TVM FFI's global registry."""
    # Build global function table only if we are going to process blocks.
    global_funcs: dict[str, list[str]] = {}
    for name in list_global_func_names():
        try:
            prefix, suffix = name.rsplit(".", 1)
        except ValueError:
            print(f"{C.TERM_YELLOW}[Skipped] Invalid name in global function: {name}{C.TERM_RESET}")
        else:
            global_funcs.setdefault(prefix, []).append(suffix)
    # Ensure stable ordering for deterministic output.
    for k in list(global_funcs.keys()):
        global_funcs[k].sort()
    return global_funcs


def collect_ty_maps(files: list[FileInfo], opt: Options) -> dict[str, str]:
    """Collect type maps from the given files."""
    ty_map: dict[str, str] = C.TY_MAP_DEFAULTS.copy()
    for file in files:
        for code in file.code_blocks:
            if code.kind == "ty-map":
                try:
                    lhs, rhs = code.param.split("->")
                except ValueError as e:
                    raise ValueError(
                        f"Invalid ty_map format at line {code.lineno_start}. Example: `A.B -> C.D`"
                    ) from e
                ty_map[lhs.strip()] = rhs.strip()
    if opt.verbose:
        for lhs in sorted(ty_map):
            rhs = ty_map[lhs]
            print(f"{C.TERM_CYAN}[TY-MAP] {lhs} -> {rhs}{C.TERM_RESET}")
    return ty_map
