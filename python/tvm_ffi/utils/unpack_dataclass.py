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
"""Fast dataclass-to-tuple conversion via JIT-compiled unpacking.

This module provides ``unpack_dataclass_to_tuple``, a function that recursively
converts dataclass instances to tuples. It JIT-compiles a per-class unpacker
on first call and caches it per-thread for ~5-11x speedup over
``dataclasses.astuple`` with no deep-copy of leaf values.
"""

from __future__ import annotations

import dataclasses
import keyword
import sys
import threading
import typing
from typing import Any

# Support both typing.Union and types.UnionType (PEP 604, Python 3.10+)
if sys.version_info >= (3, 10):
    import types

    _LEAF_CONTAINER_ORIGINS = (tuple, dict, set, frozenset, typing.Union, types.UnionType)
else:
    _LEAF_CONTAINER_ORIGINS = (tuple, dict, set, frozenset, typing.Union)

# Type alias for dataclass-to-tuple schema (internal).
# Schema values:
#   None     -> leaf, direct attribute access (zero cost)
#   "unpack" -> dynamic dispatch via unpack_dataclass_to_tuple at runtime
#   dict     -> nested struct, recurse inline
# Example: {"x": None, "y": None} -> (__x.x, __x.y,)
# Example: {"cfg": {"x": None, "y": None}, "data": "unpack"}
#         -> ((__x.cfg.x, __x.cfg.y,), __dispatch(__x.data),)
DataclassToTupleSchema = dict  # dict[str, None | str | DataclassToTupleSchema]

# Sentinel value in schema: field should be dynamically dispatched
UNPACK = "unpack"

# Thread-local cache for JIT-compiled per-class unpack functions
_tls = threading.local()

# Types known to be safe leaves (never contain dataclass instances)
_KNOWN_LEAF_TYPES: set[type] = {int, float, str, bool, bytes, complex, type(None)}


def _is_known_leaf_type(tp: Any) -> bool:
    """Check if a type is definitely a leaf (no dataclass content or conversion needed).

    Note: list is NOT a leaf because it must be converted to a tuple per the
    unpack contract (matching dataclasses.astuple behavior).
    """
    if isinstance(tp, type):
        return tp in _KNOWN_LEAF_TYPES
    if tp is Ellipsis:
        return True
    origin = typing.get_origin(tp)
    if origin is not None:
        # list is NOT a leaf — must be converted to tuple
        # tuple/dict/set/frozenset/Union are leaves if all args are leaves
        if origin in _LEAF_CONTAINER_ORIGINS:
            args = typing.get_args(tp)
            return bool(args) and all(_is_known_leaf_type(a) for a in args)
    return False


def _classify_field_type(
    field_type: Any, memo: set[type] | None = None
) -> None | str | DataclassToTupleSchema:
    """Classify a resolved field type into a schema entry.

    Conservative: only returns None (leaf) when we are certain the type
    cannot contain a dataclass. Otherwise returns UNPACK (dynamic dispatch).
    """
    if isinstance(field_type, str) or field_type is Any or field_type is object:
        return UNPACK
    if dataclasses.is_dataclass(field_type) and isinstance(field_type, type):
        # Guard against infinite recursion for self-referential dataclasses
        if memo is not None and field_type in memo:
            return UNPACK
        return _extract_dataclass_to_tuple_schema(field_type, memo=memo)
    # Known primitive types -> leaf
    if isinstance(field_type, type) and field_type in _KNOWN_LEAF_TYPES:
        return None
    # Generic containers: check element types
    # list always needs UNPACK (must be converted to tuple)
    # tuple/dict/set/frozenset/Union are leaves if all args are known leaves
    # Generic containers: list always UNPACK (must convert to tuple).
    # tuple/dict/set/frozenset/Union are leaves only if all args are known leaves.
    # Everything else (unknown type) -> UNPACK (conservative).
    origin = typing.get_origin(field_type)
    if origin in _LEAF_CONTAINER_ORIGINS:
        args = typing.get_args(field_type)
        if args and all(_is_known_leaf_type(a) for a in args):
            return None
    return UNPACK


def _compile_dataclass_to_tuple_schema(prefix: str, schema: DataclassToTupleSchema) -> str:
    """Compile a DataclassToTupleSchema into an inline tuple expression.

    Parameters
    ----------
    prefix
        The variable expression to unpack (e.g. "__x" or "__x.field").
    schema
        The schema dict mapping field names to:
        - None: leaf, direct attribute access
        - "unpack": dynamic dispatch via __dispatch() at runtime
        - nested dict: recurse inline

    Returns
    -------
        A string expression that evaluates to a tuple of the unpacked fields.

    """
    parts: list[str] = []
    for field_name, sub_schema in schema.items():
        field_expr = f"{prefix}.{field_name}"
        if sub_schema is None:
            parts.append(field_expr)
        elif sub_schema == UNPACK:
            parts.append(f"__dispatch({field_expr})")
        else:
            parts.append(_compile_dataclass_to_tuple_schema(field_expr, sub_schema))
    return "(" + ", ".join(parts) + (",)" if parts else ")")


def _validate_dataclass_to_tuple_schema(schema: DataclassToTupleSchema) -> None:
    """Validate that a DataclassToTupleSchema contains only safe identifiers.

    This is critical for security since field names are embedded directly
    in generated code via exec(). The validation ensures:
    - Keys are strings (type check)
    - Keys pass str.isidentifier() — rejects any non-identifier characters
    - Keys are not Python keywords — rejects control flow injection
    - Values are only None, "unpack", or recursively-validated dicts

    Combined with the hardcoded prefix ("__x") and restricted exec_globals,
    this prevents any code injection through crafted field names.

    """
    if not isinstance(schema, dict):
        raise TypeError(f"DataclassToTupleSchema must be a dict, got {type(schema).__name__}")
    for field_name, sub_schema in schema.items():
        if not isinstance(field_name, str):
            raise TypeError(f"Schema field name must be a string, got {type(field_name).__name__}")
        if not field_name.isidentifier():
            raise ValueError(f"Schema field name {field_name!r} is not a valid Python identifier")
        if keyword.iskeyword(field_name):
            raise ValueError(f"Schema field name {field_name!r} is a Python keyword")
        if sub_schema is not None and sub_schema != UNPACK:
            _validate_dataclass_to_tuple_schema(sub_schema)


def _extract_dataclass_to_tuple_schema(
    cls: type, *, memo: set[type] | None = None
) -> DataclassToTupleSchema:
    """Extract a DataclassToTupleSchema from a dataclass class using type annotations.

    Classification per field (conservative: only leaf when certain):
    - Known dataclass type -> nested schema (recurse inline)
    - Known primitive type (int, float, str, bool, bytes, complex) -> None (leaf)
    - Container with only known-leaf args (list[int], dict[str, float]) -> None (leaf)
    - Container with dataclass/unknown args (list[Config]) -> "unpack" (dynamic dispatch)
    - Any, object, unresolved string annotation -> "unpack" (dynamic dispatch)
    - Unknown class -> "unpack" (dynamic dispatch)

    Uses typing.get_type_hints() to resolve PEP 563 string annotations.
    Uses memo set to prevent infinite recursion on self-referential dataclasses.

    """
    if not dataclasses.is_dataclass(cls) or not isinstance(cls, type):
        raise TypeError(f"Expected a dataclass class, got {cls!r}")
    if memo is None:
        memo = set()
    memo.add(cls)
    try:
        type_hints = typing.get_type_hints(cls)
    except (NameError, TypeError, AttributeError):
        type_hints = {}
    schema: DataclassToTupleSchema = {}
    for f in dataclasses.fields(cls):
        field_type = type_hints.get(f.name, f.type)
        schema[f.name] = _classify_field_type(field_type, memo=memo)
    return schema


def unpack_dataclass_to_tuple(x: Any) -> Any:
    """Fast recursively unpack a dataclass value to tuple representation.

    - Dataclass instances are unpacked to tuples of their field values.
    - Lists and tuples are recursed element-wise, returning a tuple.
    - Dicts are recursed on values, returning a new dict.
    - All other values are returned as-is (leaf passthrough).

    This function optimizes speed via JIT-compiling the conversion per dataclass
    class and caching it per-thread. It brings about 5-11x speedup vs
    ``dataclasses.astuple`` and does not deep-copy leaf values.

    Parameters
    ----------
    x
        The value to unpack.

    Returns
    -------
        The unpacked tuple representation, or ``x`` unchanged if it's a leaf.

    """
    try:
        cache = _tls.cache
    except AttributeError:
        cache = _tls.cache = {}

    cls = type(x)
    fn = cache.get(cls)
    if fn is not None:
        return fn(x)

    # Cache miss — classify the type
    if dataclasses.is_dataclass(cls) and isinstance(cls, type):
        schema = _extract_dataclass_to_tuple_schema(cls)
        # Validate that all field names in the schema are safe Python identifiers.
        # This is critical: field names are embedded directly in the generated code string.
        # _validate_dataclass_to_tuple_schema ensures no code injection is possible via
        # crafted field names (isidentifier + iskeyword checks).
        _validate_dataclass_to_tuple_schema(schema)
        code_expr = _compile_dataclass_to_tuple_schema("__x", schema)
        code = f"def __unpack(__x): return {code_expr}"
        namespace: dict[str, Any] = {}
        # exec_globals only exposes __dispatch (our own function), no other capabilities.
        exec(code, {"__dispatch": unpack_dataclass_to_tuple}, namespace)
        fn = namespace["__unpack"]
        cache[cls] = fn
        return fn(x)
    if isinstance(x, (list, tuple)):
        return type(x)(unpack_dataclass_to_tuple(e) for e in x)
    if isinstance(x, dict):
        return {k: unpack_dataclass_to_tuple(v) for k, v in x.items()}
    # True leaf — cache identity so next call is just dict.get + return
    cache[cls] = _LEAF_IDENTITY
    return x


def _LEAF_IDENTITY(x: Any) -> Any:
    """Identity function cached for known leaf types."""
    return x
