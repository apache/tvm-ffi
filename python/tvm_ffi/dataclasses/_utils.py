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
import inspect
from dataclasses import MISSING
from typing import Any, Callable, NamedTuple, TypeVar, cast

from ..core import (
    Object,
    TypeField,
    TypeInfo,
    _lookup_type_info_from_type_key,
)

_InputClsType = TypeVar("_InputClsType")


def get_parent_type_info(type_cls: type) -> TypeInfo:
    """Find the nearest ancestor with registered ``__tvm_ffi_type_info__``.

    If none are found, return the base ``ffi.Object`` type info.
    """
    for base in type_cls.__bases__:
        if (info := getattr(base, "__tvm_ffi_type_info__", None)) is not None:
            return info
    return _lookup_type_info_from_type_key("ffi.Object")


def type_info_to_cls(
    type_info: TypeInfo,
    cls: type[_InputClsType],
    methods: dict[str, Callable[..., Any] | None],
) -> type[_InputClsType]:
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
    return cast(type[_InputClsType], new_cls)


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


def method_init(type_cls: type, type_info: TypeInfo) -> Callable[..., None]:  # noqa: PLR0915
    """Generate an ``__init__`` that forwards to the FFI constructor.

    The generated initializer has a proper Python signature built from the
    reflected field list, supporting default values and ``__post_init__``.
    """

    class DefaultFactory(NamedTuple):
        """Wrapper that marks a parameter as having a default factory."""

        fn: Callable[[], Any]

    fields: list[TypeField] = []
    cur_type_info: TypeInfo | None = type_info
    while cur_type_info is not None:
        fields.extend(reversed(cur_type_info.fields))
        cur_type_info = cur_type_info.parent_type_info
    fields.reverse()

    annotations: dict[str, Any] = {"return": None}
    # Step 1. Split the parameters into two groups to ensure that
    # those without defaults appear first in the signature.
    params_without_defaults: list[inspect.Parameter] = []
    params_with_defaults: list[inspect.Parameter] = []
    ordering = [0] * len(fields)
    for i, field in enumerate(fields):
        assert field.name is not None
        name: str = field.name
        annotations[name] = Any  # NOTE: We might be able to handle annotations better
        assert field.dataclass_field is not None
        default_factory = field.dataclass_field.default_factory
        if default_factory is MISSING:
            ordering[i] = len(params_without_defaults)
            params_without_defaults.append(
                inspect.Parameter(name=name, kind=inspect.Parameter.POSITIONAL_OR_KEYWORD)
            )
        else:
            ordering[i] = -len(params_with_defaults) - 1
            params_with_defaults.append(
                inspect.Parameter(
                    name=name,
                    kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=DefaultFactory(fn=default_factory),
                )
            )
    for i, order in enumerate(ordering):
        if order < 0:
            ordering[i] = len(params_without_defaults) - order - 1
    # Step 2. Create the signature object
    sig = inspect.Signature(parameters=[*params_without_defaults, *params_with_defaults])
    signature_str = (
        f"{type_cls.__module__}.{type_cls.__qualname__}.__init__("
        + ", ".join(p.name for p in sig.parameters.values())
        + ")"
    )

    # Step 3. Create the `binding` method that reorders parameters
    def touch_arg(x: Any) -> Any:
        return x.fn() if isinstance(x, DefaultFactory) else x

    def bind_args(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        args = bound.args
        args = tuple(touch_arg(args[i]) for i in ordering)
        return args

    for type_method in type_info.methods:
        if type_method.name == "__ffi_init__":
            break
    else:
        raise ValueError(f"Cannot find constructor method: `{type_info.type_key}.__ffi_init__`")

    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        e = None
        try:
            args = bind_args(*args, **kwargs)
            del kwargs
            self.__ffi_init__(*args)
        except Exception as _e:
            e = TypeError(f"Error in `{signature_str}`: {_e}").with_traceback(_e.__traceback__)
        if e is not None:
            raise e
        try:
            fn_post_init = self.__post_init__  # type: ignore[attr-defined]
        except AttributeError:
            pass
        else:
            fn_post_init()

    __init__.__signature__ = sig  # type: ignore[attr-defined]
    __init__.__annotations__ = annotations
    return __init__
