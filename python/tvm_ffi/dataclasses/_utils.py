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

import copy as copy_module
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
        # Allow overriding methods (including from base classes like Object.__repr__)
        # by always adding to attrs, which will be used when creating the new class
        func.__module__ = cls.__module__
        func.__name__ = name
        func.__qualname__ = f"{cls.__qualname__}.{name}"
        func.__doc__ = f"Method `{name}` of class `{cls.__qualname__}`"
        attrs[name] = func

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


def fill_dataclass_field(
    type_cls: type,
    type_field: TypeField,
    *,
    class_kw_only: bool = False,
    kw_only_from_sentinel: bool = False,
) -> None:
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

    # Resolve kw_only: field-level > KW_ONLY sentinel > class-level
    if rhs.kw_only is MISSING:
        if kw_only_from_sentinel:
            rhs.kw_only = True
        else:
            rhs.kw_only = class_kw_only

    type_field.dataclass_field = rhs


def _get_all_fields(type_info: TypeInfo) -> list[TypeField]:
    """Collect all fields from the type hierarchy, from parents to children."""
    fields: list[TypeField] = []
    cur_type_info: TypeInfo | None = type_info
    while cur_type_info is not None:
        fields.extend(reversed(cur_type_info.fields))
        cur_type_info = cur_type_info.parent_type_info
    fields.reverse()
    return fields


def _classify_fields_for_copy(
    type_info: TypeInfo,
) -> tuple[list[str], list[str], list[str]]:
    """Classify fields for copy/replace operations.

    Returns:
        Tuple of (ffi_arg_order, init_fields, non_init_fields):
        - ffi_arg_order: Fields passed to FFI constructor
        - init_fields: Fields with init=True (replaceable)
        - non_init_fields: Fields with init=False

    """
    fields = _get_all_fields(type_info)
    ffi_arg_order: list[str] = []
    init_fields: list[str] = []
    non_init_fields: list[str] = []

    for field in fields:
        assert field.name is not None
        assert field.dataclass_field is not None
        dataclass_field = field.dataclass_field

        if dataclass_field.init:
            init_fields.append(field.name)
            ffi_arg_order.append(field.name)
        elif dataclass_field.default_factory is not MISSING:
            ffi_arg_order.append(field.name)
        else:
            non_init_fields.append(field.name)

    return ffi_arg_order, init_fields, non_init_fields


def method_repr(type_cls: type, type_info: TypeInfo) -> Callable[..., str]:
    """Generate a ``__repr__`` method for the dataclass.

    The generated representation includes all fields with ``repr=True`` in
    the format ``ClassName(field1=value1, field2=value2, ...)``.
    """
    # Step 0. Collect all fields from the type hierarchy
    fields = _get_all_fields(type_info)

    # Step 1. Filter fields that should appear in repr
    repr_fields: list[str] = []
    for field in fields:
        assert field.name is not None
        assert field.dataclass_field is not None
        if field.dataclass_field.repr:
            repr_fields.append(field.name)

    # Step 2. Generate the repr method
    if not repr_fields:
        # No fields to show, return a simple class name representation
        body_lines = [f"return f'{type_cls.__name__}()'"]
    else:
        # Build field representations
        fields_str = ", ".join(
            f"{field_name}={{self.{field_name}!r}}" for field_name in repr_fields
        )
        body_lines = [f"return f'{type_cls.__name__}({fields_str})'"]

    source_lines = ["def __repr__(self) -> str:"]
    source_lines.extend(f"    {line}" for line in body_lines)
    source = "\n".join(source_lines)

    # Note: Code generation in this case is guaranteed to be safe,
    # because the generated code does not contain any untrusted input.
    namespace: dict[str, Any] = {}
    exec(source, {}, namespace)
    __repr__ = namespace["__repr__"]
    return __repr__


def method_init(_type_cls: type, type_info: TypeInfo) -> Callable[..., None]:
    """Generate an ``__init__`` that forwards to the FFI constructor.

    The generated initializer has a proper Python signature built from the
    reflected field list, supporting default values, keyword-only args, and ``__post_init__``.
    """
    # Step 0. Collect all fields from the type hierarchy
    fields = _get_all_fields(type_info)
    # sanity check
    if not any(m.name == "__ffi_init__" for m in type_info.methods):
        raise ValueError(f"Cannot find constructor method: `{type_info.type_key}.__ffi_init__`")
    # Step 1. Split args into sections and register default factories
    pos_no_defaults: list[str] = []
    pos_with_defaults: list[str] = []
    kw_no_defaults: list[str] = []
    kw_with_defaults: list[str] = []
    fields_with_defaults: list[tuple[str, bool]] = []
    ffi_arg_order: list[str] = []
    exec_globals: dict[str, Any] = {"MISSING": MISSING}

    for field in fields:
        assert field.name is not None
        assert field.dataclass_field is not None
        dataclass_field = field.dataclass_field
        has_default = (default_factory := dataclass_field.default_factory) is not MISSING
        is_kw_only = dataclass_field.kw_only is True

        if dataclass_field.init:
            ffi_arg_order.append(field.name)
            if has_default:
                (kw_with_defaults if is_kw_only else pos_with_defaults).append(field.name)
                fields_with_defaults.append((field.name, True))
                exec_globals[f"_default_factory_{field.name}"] = default_factory
            else:
                (kw_no_defaults if is_kw_only else pos_no_defaults).append(field.name)
        elif has_default:
            ffi_arg_order.append(field.name)
            fields_with_defaults.append((field.name, False))
            exec_globals[f"_default_factory_{field.name}"] = default_factory

    # Step 2. Build signature
    args: list[str] = ["self"]
    args.extend(pos_no_defaults)
    args.extend(f"{name}=MISSING" for name in pos_with_defaults)
    if kw_no_defaults or kw_with_defaults:
        args.append("*")
        args.extend(kw_no_defaults)
        args.extend(f"{name}=MISSING" for name in kw_with_defaults)

    # Step 3. Build body
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


def method_copy(_type_cls: type, type_info: TypeInfo) -> Callable[..., Any]:
    """Generate a ``__copy__`` method for the dataclass (shallow copy).

    The generated method creates a shallow copy by calling the FFI constructor
    directly with the current field values (bypassing custom Python __init__).
    """
    ffi_arg_order, _, non_init_fields = _classify_fields_for_copy(type_info)

    body_lines: list[str] = []
    if ffi_arg_order:
        ffi_args = ", ".join(f"self.{name}" for name in ffi_arg_order)
        body_lines.append(f"new_obj = type(self).__c_ffi_init__({ffi_args})")
    else:
        body_lines.append("new_obj = type(self).__c_ffi_init__()")
    for name in non_init_fields:
        body_lines.append(f"new_obj.{name} = self.{name}")
    body_lines.append("return new_obj")

    source_lines = ["def __copy__(self):"]
    source_lines.extend(f"    {line}" for line in body_lines)
    source = "\n".join(source_lines)

    namespace: dict[str, Any] = {}
    exec(source, {}, namespace)
    return namespace["__copy__"]


def method_deepcopy(_type_cls: type, type_info: TypeInfo) -> Callable[..., Any]:
    """Generate a ``__deepcopy__`` method for the dataclass.

    The generated method creates a deep copy using copy.deepcopy for field values,
    handling circular references via the memo dictionary.
    """
    ffi_arg_order, _, non_init_fields = _classify_fields_for_copy(type_info)

    body_lines: list[str] = []
    if ffi_arg_order:
        ffi_args = ", ".join(f"_copy_deepcopy(self.{name}, memo)" for name in ffi_arg_order)
        body_lines.append(f"new_obj = type(self).__c_ffi_init__({ffi_args})")
    else:
        body_lines.append("new_obj = type(self).__c_ffi_init__()")
    body_lines.append("memo[id(self)] = new_obj")
    for name in non_init_fields:
        body_lines.append(f"new_obj.{name} = _copy_deepcopy(self.{name}, memo)")
    body_lines.append("return new_obj")

    source_lines = ["def __deepcopy__(self, memo):"]
    source_lines.extend(f"    {line}" for line in body_lines)
    source = "\n".join(source_lines)

    exec_globals: dict[str, Any] = {"_copy_deepcopy": copy_module.deepcopy}
    namespace: dict[str, Any] = {}
    exec(source, exec_globals, namespace)
    return namespace["__deepcopy__"]


def method_replace(_type_cls: type, type_info: TypeInfo) -> Callable[..., Any]:
    """Generate a ``__replace__`` method for the dataclass.

    The generated method returns a new instance with specified fields replaced.
    Only fields with init=True can be changed. Fields with init=False are copied unchanged.
    """
    ffi_arg_order, init_fields, non_init_fields = _classify_fields_for_copy(type_info)

    body_lines: list[str] = []
    body_lines.append("for key in changes:")
    body_lines.append("    if key not in _valid_fields:")
    body_lines.append(
        "        raise TypeError(f\"__replace__() got an unexpected keyword argument '{key}'\")"
    )
    if ffi_arg_order:
        ffi_args = ", ".join(f"changes.get('{name}', self.{name})" for name in ffi_arg_order)
        body_lines.append(f"new_obj = type(self).__c_ffi_init__({ffi_args})")
    else:
        body_lines.append("new_obj = type(self).__c_ffi_init__()")
    for name in non_init_fields:
        body_lines.append(f"new_obj.{name} = self.{name}")
    body_lines.append("return new_obj")

    source_lines = ["def __replace__(self, **changes):"]
    source_lines.extend(f"    {line}" for line in body_lines)
    source = "\n".join(source_lines)

    exec_globals: dict[str, Any] = {"_valid_fields": frozenset(init_fields)}
    namespace: dict[str, Any] = {}
    exec(source, exec_globals, namespace)
    return namespace["__replace__"]
