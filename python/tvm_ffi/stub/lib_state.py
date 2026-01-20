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
"""Stateful helpers for querying TVM FFI runtime metadata."""

from __future__ import annotations

import functools
import heapq
from collections import defaultdict

from tvm_ffi._ffi_api import GetRegisteredTypeKeys
from tvm_ffi.core import TypeSchema, _lookup_or_register_type_info_from_type_key
from tvm_ffi.registry import get_global_func_metadata, list_global_func_names

from . import consts as C
from .utils import FuncInfo, NamedTypeSchema, ObjectInfo


@functools.lru_cache(maxsize=None)
def object_info_from_type_key(type_key: str) -> ObjectInfo:
    """Construct an `ObjectInfo` from an object type key."""
    type_info = _lookup_or_register_type_info_from_type_key(str(type_key))
    assert type_info.type_key == type_key
    return ObjectInfo.from_type_info(type_info)


def collect_global_funcs() -> dict[str, list[FuncInfo]]:
    """Collect global functions from TVM FFI's global registry."""
    global_funcs: dict[str, list[FuncInfo]] = {}
    for name in list_global_func_names():
        try:
            prefix, _ = name.rsplit(".", 1)
        except ValueError:
            print(f"{C.TERM_YELLOW}[Skipped] Invalid name in global function: {name}{C.TERM_RESET}")
        else:
            try:
                global_funcs.setdefault(prefix, []).append(_func_info_from_global_name(name))
            except Exception:
                print(f"{C.TERM_YELLOW}[Skipped] Function has no type schema: {name}{C.TERM_RESET}")
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
    for k in list(global_objects.keys()):
        global_objects[k].sort()
    return global_objects


def toposort_objects(type_keys: list[str]) -> list[ObjectInfo]:
    """Collect ObjectInfo objects for type keys, topologically sorted by inheritance."""
    # Remove duplicates while preserving order.
    unique_type_keys = list(dict.fromkeys(type_keys))
    infos: dict[str, ObjectInfo] = {
        type_key: object_info_from_type_key(type_key) for type_key in unique_type_keys
    }

    child_types: dict[str, list[str]] = defaultdict(list)
    in_degree: dict[str, int] = defaultdict(int)
    for type_key, info in infos.items():
        parent_type_key = info.parent_type_key
        if parent_type_key in infos:
            child_types[parent_type_key].append(type_key)
            in_degree[type_key] += 1
            in_degree[parent_type_key] += 0
        else:
            in_degree[type_key] += 0

    for children in child_types.values():
        children.sort()

    queue: list[str] = [ty for ty, deg in in_degree.items() if deg == 0]
    heapq.heapify(queue)
    sorted_keys: list[str] = []
    while queue:
        type_key = heapq.heappop(queue)
        sorted_keys.append(type_key)
        for child_type_key in child_types[type_key]:
            in_degree[child_type_key] -= 1
            if in_degree[child_type_key] == 0:
                heapq.heappush(queue, child_type_key)

    assert len(sorted_keys) == len(infos)
    return [infos[type_key] for type_key in sorted_keys]


@functools.lru_cache(maxsize=None)
def _func_info_from_global_name(name: str) -> FuncInfo:
    """Construct a `FuncInfo` from a global function name."""
    return FuncInfo(
        schema=NamedTypeSchema(
            name=name,
            schema=TypeSchema.from_json_str(get_global_func_metadata(name)["type_schema"]),
        ),
        is_member=False,
    )
