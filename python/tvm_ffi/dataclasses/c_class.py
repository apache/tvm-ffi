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
"""Helpers for mirroring registered C++ FFI types with Python dataclass syntax.

The :func:`c_class` decorator is the primary entry point.  It inspects the
reflection metadata that the C++ runtime exposes via the TVM FFI registry and
turns it into Python ``dataclass``-style descriptors: annotated attributes become
properties that forward to the underlying C++ object, while an ``__init__``
method is synthesized to call the FFI constructor when requested.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import InitVar
from typing import ClassVar, Type, TypeVar, get_origin, get_type_hints

from typing_extensions import dataclass_transform

from ..core import TypeField, TypeInfo, _lookup_or_register_type_info_from_type_key, _set_type_cls
from . import _utils
from .field import field

_InputClsType = TypeVar("_InputClsType")


@dataclass_transform(field_specifiers=(field,))
def c_class(
    type_key: str, init: bool = True
) -> Callable[[Type[_InputClsType]], Type[_InputClsType]]:  # noqa: UP006
    """(Experimental) Create a dataclass-like proxy for a C++ class registered with TVM FFI.

    The decorator reads the reflection metadata that was registered on the C++
    side using ``tvm::ffi::reflection::ObjectDef`` and binds it to the annotated
    attributes in the decorated Python class. Each field defined in C++ becomes
    a property on the Python class, and optional default values can be provided
    with :func:`tvm_ffi.dataclasses.field` in the same way as Python's native
    ``dataclasses.field``.

    The intent is to offer a familiar dataclass authoring experience while still
    exposing the underlying C++ object.  The ``type_key`` of the C++ class must
    match the string passed to :func:`c_class`, and inheritance relationships are
    preserved—subclasses registered in C++ can subclass the Python proxy defined
    for their parent.

    Parameters
    ----------
    type_key
        The reflection key that identifies the C++ type in the FFI registry,
        e.g. ``"testing.MyClass"`` as registered in
        ``src/ffi/extra/testing.cc``.

    init
        If ``True`` and the Python class does not define ``__init__``, an
        initializer is auto-generated that mirrors the reflected constructor
        signature.  The generated initializer calls the C++ ``__init__``
        function registered with ``ObjectDef`` and invokes ``__post_init__`` if
        it exists on the Python class.

    Returns
    -------
    Callable[[type], type]
        A class decorator that materializes the final proxy class.

    Examples
    --------
    Register the C++ type and its fields with TVM FFI:

    .. code-block:: c++

        TVM_FFI_STATIC_INIT_BLOCK() {
          namespace refl = tvm::ffi::reflection;
          refl::ObjectDef<MyClass>()
              .def_static("__init__", [](int64_t v_i64, int32_t v_i32,
                                         double v_f64, float v_f32) -> Any {
                   return ObjectRef(ffi::make_object<MyClass>(
                       v_i64, v_i32, v_f64, v_f32));
               })
              .def_rw("v_i64", &MyClass::v_i64)
              .def_rw("v_i32", &MyClass::v_i32)
              .def_rw("v_f64", &MyClass::v_f64)
              .def_rw("v_f32", &MyClass::v_f32);
        }

    Mirror the same structure in Python using dataclass-style annotations:

    .. code-block:: python

        from tvm_ffi.dataclasses import c_class, field

        @c_class("example.MyClass")
        class MyClass:
            v_i64: int
            v_i32: int
            v_f64: float = field(default=0.0)
            v_f32: float = field(default_factory=lambda: 1.0)

        obj = MyClass(v_i64=4, v_i32=8)
        obj.v_f64 = 3.14  # transparently forwards to the underlying C++ object

    """

    def decorator(super_type_cls: Type[_InputClsType]) -> Type[_InputClsType]:  # noqa: UP006
        nonlocal init
        init = init and "__init__" not in super_type_cls.__dict__
        # Step 1. Retrieve `type_info` from registry
        type_info: TypeInfo = _lookup_or_register_type_info_from_type_key(type_key)
        assert type_info.parent_type_info is not None
        # Step 2. Reflect all the fields of the type
        type_info.fields = _inspect_c_class_fields(super_type_cls, type_info)
        for type_field in type_info.fields:
            _utils.fill_dataclass_field(super_type_cls, type_field)
        # Step 3. Create the proxy class with the fields as properties
        fn_init = _utils.method_init(super_type_cls, type_info) if init else None
        type_cls: Type[_InputClsType] = _utils.type_info_to_cls(  # noqa: UP006
            type_info=type_info,
            cls=super_type_cls,
            methods={"__init__": fn_init},
        )
        _set_type_cls(type_info, type_cls)
        return type_cls

    return decorator


def _inspect_c_class_fields(type_cls: type, type_info: TypeInfo) -> list[TypeField]:
    if sys.version_info >= (3, 9):
        type_hints_resolved = get_type_hints(type_cls, include_extras=True)
    else:
        type_hints_resolved = get_type_hints(type_cls)
    type_hints_py = {
        name: type_hints_resolved[name]
        for name in getattr(type_cls, "__annotations__", {}).keys()
        if get_origin(type_hints_resolved[name])
        not in [  # ignore non-field annotations
            ClassVar,
            InitVar,
        ]
    }
    del type_hints_resolved

    type_fields_cxx: dict[str, TypeField] = {f.name: f for f in type_info.fields}
    type_fields: list[TypeField] = []
    for field_name, _field_ty_py in type_hints_py.items():
        if field_name.startswith("__tvm_ffi"):  # TVM's private fields - skip
            continue
        type_field = type_fields_cxx.pop(field_name, None)
        if type_field is None:
            raise ValueError(
                f"Extraneous field `{type_cls}.{field_name}`. Defined in Python but not in C++"
            )
        type_fields.append(type_field)
    if type_fields_cxx:
        extra_fields = ", ".join(f"`{f.name}`" for f in type_fields_cxx.values())
        raise ValueError(
            f"Missing fields in `{type_cls}`: {extra_fields}. Defined in C++ but not in Python"
        )
    return type_fields
