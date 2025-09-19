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
import dataclasses


cdef class FieldGetter:
    cdef dict __dict__
    cdef TVMFFIFieldGetter getter
    cdef int64_t offset

    def __call__(self, Object obj):
        cdef TVMFFIAny result
        cdef int c_api_ret_code
        cdef void* field_ptr = (<char*>(<Object>obj).chandle) + self.offset
        result.type_index = kTVMFFINone
        result.v_int64 = 0
        c_api_ret_code = self.getter(field_ptr, &result)
        CHECK_CALL(c_api_ret_code)
        return make_ret(result)


cdef class FieldSetter:
    cdef dict __dict__
    cdef TVMFFIFieldSetter setter
    cdef int64_t offset

    def __call__(self, Object obj, value):
        cdef int c_api_ret_code
        cdef void* field_ptr = (<char*>(<Object>obj).chandle) + self.offset
        TVMFFIPyCallFieldSetter(
            TVMFFIPyArgSetterFactory_,
            self.setter,
            field_ptr,
            <PyObject*>value,
            &c_api_ret_code
        )
        # NOTE: logic is same as check_call
        # directly inline here to simplify traceback
        if c_api_ret_code == 0:
            return
        elif c_api_ret_code == -2:
            raise_existing_error()
        raise move_from_last_error().py_error()


@dataclasses.dataclass(eq=False)
class TypeField:
    """Description of a single reflected field on an FFI-backed type."""

    name: str
    doc: str | None
    size: int
    offset: int
    frozen: bool
    getter: FieldGetter
    setter: FieldSetter

    def __post_init__(self):
        assert self.setter is not None
        assert self.getter is not None

    def as_property(self, cls: type) -> property:
        """Create a Python ``property`` object for this field on ``cls``."""
        name = self.name
        fget = self.getter
        fset = self.setter
        fget.__name__ = fset.__name__ = name
        fget.__module__ = fset.__module__ = cls.__module__
        fget.__qualname__ = fset.__qualname__ = f"{cls.__qualname__}.{name}"  # type: ignore[attr-defined]
        fget.__doc__ = fset.__doc__ = f"Property `{name}` of class `{cls.__qualname__}`"  # type: ignore[attr-defined]

        return property(
            fget=fget if self.getter is not None else None,
            fset=fset if (not self.frozen) and self.setter is not None else None,
            doc=f"{cls.__module__}.{cls.__qualname__}.{name}",
        )


@dataclasses.dataclass(eq=False)
class TypeMethod:
    """Description of a single reflected method on an FFI-backed type."""

    name: str
    doc: str | None
    func: object
    is_static: bool


@dataclasses.dataclass(eq=False)
class TypeInfo:
    """Aggregated type information required to build a proxy class."""

    type_cls: type | None
    type_index: int
    type_key: str
    fields: list[TypeField]
    methods: list[TypeMethod]
    parent_type_info: TypeInfo | None
