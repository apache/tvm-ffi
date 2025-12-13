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

from tvm_ffi._ffi_api import GetRegisteredTypeKeys
from tvm_ffi.registry import list_global_func_names

from . import consts as C
from .utils import FuncInfo


def collect_global_funcs() -> dict[str, list[FuncInfo]]:
    """Collect global functions from TVM FFI's global registry."""
    # Build global function table only if we are going to process blocks.
    global_funcs: dict[str, list[FuncInfo]] = {}
    for name in list_global_func_names():
        try:
            prefix, _ = name.rsplit(".", 1)
        except ValueError:
            print(f"{C.TERM_YELLOW}[Skipped] Invalid name in global function: {name}{C.TERM_RESET}")
        else:
            try:
                global_funcs.setdefault(prefix, []).append(FuncInfo.from_global_name(name))
            except Exception:
                print(f"{C.TERM_YELLOW}[Skipped] Function has no type schema: {name}{C.TERM_RESET}")
    # Ensure stable ordering for deterministic output.
    for k in list(global_funcs.keys()):
        global_funcs[k].sort(key=lambda x: x.schema.name)
    return global_funcs


def collect_type_keys() -> dict[str, list[str]]:
    """Collect registered object type keys from TVM FFI's global registry."""
    global_objects: dict[str, list[str]] = {}
    for type_key in GetRegisteredTypeKeys():
        try:
            prefix, _ = type_key.rsplit(".", 1)
        except ValueError:
            pass
        else:
            global_objects.setdefault(prefix, []).append(type_key)
    # Ensure stable ordering for deterministic output.
    for k in list(global_objects.keys()):
        global_objects[k].sort()
    return global_objects
