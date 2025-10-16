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
"""Utilities for constructing Python proxies of FFI types."""

from __future__ import annotations

import functools
from dataclasses import MISSING
from typing import Any, Callable, Type, TypeVar, cast

from ..core import (
    Object,
    TypeField,
    TypeInfo,
)

_InputClsType = TypeVar("_InputClsType")


def type_info_to_cls(
    type_info: TypeInfo,
    cls: Type[_InputClsType],  # noqa: UP006
    methods: dict[str, Callable[..., Any] | None],
) -> Type[_InputClsType]:  # noqa: UP006
    assert type_info.type_cls is None, "Type class is already created"
    # Step 1. Determine the base classes
    cls_bases = cls.__bases__
    if cls_bases == (object,):
        # If the class inherits from `object`, we need to set the base class to `Object`
        cls_bases = (Object,)

    # Step 2. Define the new class attributes
    attrs = dict(cls.__dict__)
    attrs.pop("__dict__", None)
    attrs.pop("__weakref__", None)
    attrs["__slots__"] = ()
    attrs["__tvm_ffi_type_info__"] = type_info

    # Step 2. Add fields
    for field in type_info.fields:
        attrs[field.name] = field.as_property(cls)

    # Step 3. Add methods
    def _add_method(name: str, func: Callable[..., Any]) -> None:
        if name == "__ffi_init__":
            name = "__c_ffi_init__"
        if name in attrs:  # already defined
            return
        func.__module__ = cls.__module__
        func.__name__ = name
        func.__qualname__ = f"{cls.__qualname__}.{name}"
        func.__doc__ = f"Method `{name}` of class `{cls.__qualname__}`"
        attrs[name] = func
        setattr(cls, name, func)

    for name, method_impl in methods.items():
        if method_impl is not None:
            _add_method(name, method_impl)
    for method in type_info.methods:
        _add_method(method.name, method.func)

    # Step 4. Create the new class
    new_cls = type(cls.__name__, cls_bases, attrs)
    new_cls.__module__ = cls.__module__
    new_cls = functools.wraps(cls, updated=())(new_cls)  # type: ignore
    return cast(Type[_InputClsType], new_cls)


def fill_dataclass_field(type_cls: type, type_field: TypeField) -> None:
    from .field import Field, field  # noqa: PLC0415

    field_name = type_field.name
    rhs: Any = getattr(type_cls, field_name, MISSING)
    if rhs is MISSING:
        rhs = field()
    elif isinstance(rhs, Field):
        pass
    elif isinstance(rhs, (int, float, str, bool, type(None))):
        rhs = field(default=rhs)
    else:
        raise ValueError(f"Cannot recognize field: {type_field.name}: {rhs}")
    assert isinstance(rhs, Field)
    rhs.name = type_field.name
    type_field.dataclass_field = rhs


def method_init(type_cls: type, type_info: TypeInfo) -> Callable[..., None]:
    """Generate an ``__init__`` that forwards to the FFI constructor.

    The generated initializer has a proper Python signature built from the
    reflected field list, supporting default values and ``__post_init__``.
    """
    # Step 0. Collect all fields from the type hierarchy
    fields: list[TypeField] = []
    cur_type_info: TypeInfo | None = type_info
    while cur_type_info is not None:
        fields.extend(reversed(cur_type_info.fields))
        cur_type_info = cur_type_info.parent_type_info
    fields.reverse()
    # sanity check
    for type_method in type_info.methods:
        if type_method.name == "__ffi_init__":
            break
    else:
        raise ValueError(f"Cannot find constructor method: `{type_info.type_key}.__ffi_init__`")
    # Step 1. Split args into sections and register default factories
    args_no_defaults: list[str] = []
    args_with_defaults: list[str] = []
    fields_with_defaults: list[tuple[str, bool]] = []
    ffi_arg_order: list[str] = []
    exec_globals = {"MISSING": MISSING}
    for field in fields:
        assert field.name is not None
        assert field.dataclass_field is not None
        dataclass_field = field.dataclass_field
        has_default_factory = (default_factory := dataclass_field.default_factory) is not MISSING
        if dataclass_field.init:
            ffi_arg_order.append(field.name)
            if has_default_factory:
                args_with_defaults.append(field.name)
                fields_with_defaults.append((field.name, True))
                exec_globals[f"_default_factory_{field.name}"] = default_factory
            else:
                args_no_defaults.append(field.name)
        elif has_default_factory:
            ffi_arg_order.append(field.name)
            fields_with_defaults.append((field.name, False))
            exec_globals[f"_default_factory_{field.name}"] = default_factory

    args: list[str] = ["self"]
    args.extend(args_no_defaults)
    args.extend(f"{name}=MISSING" for name in args_with_defaults)
    body_lines: list[str] = []
    for field_name, is_init in fields_with_defaults:
        if is_init:
            body_lines.append(
                f"if {field_name} is MISSING: {field_name} = _default_factory_{field_name}()"
            )
        else:
            body_lines.append(f"{field_name} = _default_factory_{field_name}()")
    body_lines.append(f"self.__ffi_init__({', '.join(ffi_arg_order)})")
    body_lines.extend(
        [
            "try:",
            "    fn_post_init = self.__post_init__",
            "except AttributeError:",
            "    pass",
            "else:",
            "    fn_post_init()",
        ]
    )
    source_lines = [f"def __init__({', '.join(args)}):"]
    source_lines.extend(f"    {line}" for line in body_lines)
    source_lines.append("    ...")
    source = "\n".join(source_lines)
    # Note: Code generation in this case is guaranteed to be safe,
    # because the generated code does not contain any untrusted input.
    # This is also a common practice used by `dataclasses` and `pydantic`.
    namespace: dict[str, Any] = {}
    exec(source, exec_globals, namespace)
    __init__ = namespace["__init__"]
    return __init__
