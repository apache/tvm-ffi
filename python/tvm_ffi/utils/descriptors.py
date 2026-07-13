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
"""Descriptor utilities for the tvm_ffi Python package."""

from __future__ import annotations

from typing import Any, Callable, Generic, TypeVar, overload

_T = TypeVar("_T")


class init_property(Generic[_T]):
    """Auto-registered C++ field with eager computation at ``__init__`` time.

    ``@py_class`` detects ``init_property`` descriptors and registers each one
    as ``field(init=False, structural_eq="ignore")``, so the computed value
    lives in C++ object storage and is accessible cross-language.  The value
    is computed once — immediately after ``__ffi_init__`` — and stored via the
    field's C++ slot.  Subsequent reads go directly to C++ memory.

    The return annotation of the decorated function is injected into the class
    ``__annotations__`` during class body execution so the field resolution
    machinery picks it up automatically.  If no return annotation is present,
    ``typing.Any`` is used.
    """

    def __init__(self, func: Callable[[Any], _T]) -> None:
        self.func = func
        self.name: str | None = None
        self._return_annotation: Any = func.__annotations__.get("return")

    @overload
    def __get__(self, obj: None, objtype: type) -> init_property[_T]: ...
    @overload
    def __get__(self, obj: object, objtype: type) -> _T: ...
    def __get__(self, obj: object | None, objtype: type) -> _T | init_property[_T]:
        return self

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

        ann = self._return_annotation if self._return_annotation is not None else Any
        # Inject into the owner's own __annotations__ so on_fields_resolved
        # processes this name as a typed field.
        if "__annotations__" not in owner.__dict__:
            owner.__annotations__ = {}
        owner.__annotations__[name] = ann
