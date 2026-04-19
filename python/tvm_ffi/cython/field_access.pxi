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
"""Field-access optimization: specialized getter/setter classes and property helpers.

This module consolidates the Cython-level infrastructure for fast field access:

- **FieldGetter / FieldSetter**: Generic field accessor classes that call through
  the C++ field getter/setter function pointers.
- **FieldGetterInt64, FieldGetterFloat64, ...**: Type-specialized (fused) getter
  classes that bypass the generic C++ getter path, reading field values directly
  from object memory via inline C helpers (``TVMFFIPyFieldGet*``).  These avoid
  try/catch, ``Any`` construction, and ``make_ret`` dispatch overhead.
- **_make_specialized_getter**: Factory function that selects the best getter
  class for a given field type and size.
- **_make_field_property**: Creates a Python ``property`` wrapping a specialized
  getter/setter.  Using exactly ``property`` (not a subclass) triggers CPython
  3.12+ ``LOAD_ATTR_PROPERTY`` inline-cache specialization, which avoids MRO
  walk on every read (~15-25ns vs ~50-100ns for the generic descriptor path).
- **_install_field_helpers**: Installs ``_force_set_field`` (frozen escape hatch)
  and ``__replace__`` (copy-and-modify) on classes with reflected fields.
"""

# ---------------------------------------------------------------------------
# Generic field getter/setter
# ---------------------------------------------------------------------------

cdef class FieldGetter:
    cdef dict __dict__
    cdef TVMFFIFieldGetter getter
    cdef int64_t offset

    def __call__(self, CObject obj):
        cdef TVMFFIAny result
        cdef int c_api_ret_code
        cdef void* field_ptr = (<char*>(<CObject>obj).chandle) + self.offset
        result.type_index = kTVMFFINone
        result.v_int64 = 0
        c_api_ret_code = self.getter(field_ptr, &result)
        CHECK_CALL(c_api_ret_code)
        return make_ret(result)


cdef class FieldSetter:
    cdef dict __dict__
    cdef void* setter
    cdef int64_t offset
    cdef int64_t flags

    def __call__(self, CObject obj, value):
        cdef int c_api_ret_code
        cdef void* field_ptr = (<char*>(<CObject>obj).chandle) + self.offset
        TVMFFIPyCallFieldSetter(
            TVMFFIPyArgSetterFactory_,
            self.setter,
            self.flags,
            field_ptr,
            <PyObject*>value,
            &c_api_ret_code
        )
        # NOTE: logic is same as check_call
        # directly inline here to simplify backtrace
        if c_api_ret_code == 0:
            return
        # backward compact with error already set case
        # TODO(tqchen): remove after we move beyond a few versions.
        if c_api_ret_code == -2:
            raise raise_existing_error()
        # epecial handle env error already set
        error = move_from_last_error()
        if error.kind == "EnvErrorAlreadySet":
            raise raise_existing_error()
        raise error.py_error()


# ---------------------------------------------------------------------------
# Type-specialized field getters
#
# Each class reads a specific C type directly from object memory via the
# inline C helpers in tvm_ffi_python_helpers.h, bypassing the generic
# getter path (no try/catch, no Any construction, no make_ret dispatch).
# ---------------------------------------------------------------------------

cdef class FieldGetterInt64:
    cdef dict __dict__
    cdef int64_t offset

    def __call__(self, CObject obj):
        return TVMFFIPyFieldGetInt64((<CObject>obj).chandle, self.offset)


cdef class FieldGetterInt32:
    cdef dict __dict__
    cdef int64_t offset

    def __call__(self, CObject obj):
        return TVMFFIPyFieldGetInt32((<CObject>obj).chandle, self.offset)


cdef class FieldGetterInt16:
    cdef dict __dict__
    cdef int64_t offset

    def __call__(self, CObject obj):
        return TVMFFIPyFieldGetInt16((<CObject>obj).chandle, self.offset)


cdef class FieldGetterInt8:
    cdef dict __dict__
    cdef int64_t offset

    def __call__(self, CObject obj):
        return TVMFFIPyFieldGetInt8((<CObject>obj).chandle, self.offset)


cdef class FieldGetterFloat64:
    cdef dict __dict__
    cdef int64_t offset

    def __call__(self, CObject obj):
        return TVMFFIPyFieldGetFloat64((<CObject>obj).chandle, self.offset)


cdef class FieldGetterFloat32:
    cdef dict __dict__
    cdef int64_t offset

    def __call__(self, CObject obj):
        return TVMFFIPyFieldGetFloat32((<CObject>obj).chandle, self.offset)


cdef class FieldGetterBool:
    cdef dict __dict__
    cdef int64_t offset

    def __call__(self, CObject obj):
        return bool(TVMFFIPyFieldGetBool((<CObject>obj).chandle, self.offset))


cdef class FieldGetterObjectRef:
    cdef dict __dict__
    cdef int64_t offset

    def __call__(self, CObject obj):
        cdef TVMFFIAny result
        TVMFFIPyFieldGetObjectRef((<CObject>obj).chandle, self.offset, &result)
        if result.type_index == kTVMFFINone:
            return None
        return make_ret_object(result)


cdef class FieldGetterAny:
    cdef dict __dict__
    cdef int64_t offset

    def __call__(self, CObject obj):
        cdef TVMFFIAny result
        TVMFFIPyFieldGetAny((<CObject>obj).chandle, self.offset, &result)
        return make_ret(result)


# ---------------------------------------------------------------------------
# Specialized getter factory
# ---------------------------------------------------------------------------

cdef object _make_specialized_getter(
    int32_t field_type_index,
    int64_t field_size,
    int64_t field_offset,
    FieldGetter fallback_getter,
):
    cdef FieldGetterInt64 int64_getter
    cdef FieldGetterInt32 int32_getter
    cdef FieldGetterInt16 int16_getter
    cdef FieldGetterInt8 int8_getter
    cdef FieldGetterFloat64 float64_getter
    cdef FieldGetterFloat32 float32_getter
    cdef FieldGetterBool bool_getter
    cdef FieldGetterObjectRef obj_getter
    cdef FieldGetterAny any_getter

    if field_type_index == kTVMFFIInt:
        if field_size == sizeof(int64_t):
            int64_getter = FieldGetterInt64.__new__(FieldGetterInt64)
            int64_getter.offset = field_offset
            return int64_getter
        if field_size == sizeof(int32_t):
            int32_getter = FieldGetterInt32.__new__(FieldGetterInt32)
            int32_getter.offset = field_offset
            return int32_getter
        if field_size == sizeof(int16_t):
            int16_getter = FieldGetterInt16.__new__(FieldGetterInt16)
            int16_getter.offset = field_offset
            return int16_getter
        if field_size == sizeof(uint8_t):
            int8_getter = FieldGetterInt8.__new__(FieldGetterInt8)
            int8_getter.offset = field_offset
            return int8_getter
        return fallback_getter

    if field_type_index == kTVMFFIFloat:
        if field_size == sizeof(double):
            float64_getter = FieldGetterFloat64.__new__(FieldGetterFloat64)
            float64_getter.offset = field_offset
            return float64_getter
        if field_size == sizeof(float):
            float32_getter = FieldGetterFloat32.__new__(FieldGetterFloat32)
            float32_getter.offset = field_offset
            return float32_getter
        return fallback_getter

    if field_type_index == kTVMFFIBool and field_size == sizeof(uint8_t):
        bool_getter = FieldGetterBool.__new__(FieldGetterBool)
        bool_getter.offset = field_offset
        return bool_getter

    if field_type_index == kTVMFFIAny and field_size == sizeof(TVMFFIAny):
        any_getter = FieldGetterAny.__new__(FieldGetterAny)
        any_getter.offset = field_offset
        return any_getter

    if field_type_index >= kTVMFFIStaticObjectBegin and field_size == sizeof(void*):
        obj_getter = FieldGetterObjectRef.__new__(FieldGetterObjectRef)
        obj_getter.offset = field_offset
        return obj_getter

    return fallback_getter


# ---------------------------------------------------------------------------
# Python property factory
#
# Creates a ``property`` wrapping a specialized getter/setter.  This used to
# live in _dunder.py (pure Python) so closures were PyFunctionObject for
# CPython 3.12+ LOAD_ATTR_PROPERTY specialization.  Now that tp_getattro
# handles reads directly in C, LOAD_ATTR_PROPERTY is no longer on the hot
# path and the indirection is unnecessary.
# ---------------------------------------------------------------------------

def _make_field_property(type_field):
    """Create a Python ``property`` for a reflected field."""
    getter = type_field.getter
    setter = type_field.setter
    name = type_field.name
    frozen = type_field.frozen

    def fget(self):
        return getter(self)

    if frozen:
        def fset(self, value):
            raise AttributeError(f"cannot assign to field {name!r}")
    else:
        def fset(self, value):
            setter(self, value)

    def fdel(self):
        raise AttributeError(f"cannot delete field {name!r}")

    fget.__name__ = fget.__qualname__ = name
    return property(fget=fget, fset=fset, fdel=fdel, doc=type_field.doc)


# ---------------------------------------------------------------------------
# Field helpers: _force_set_field + __replace__
#
# Installed on classes with reflected fields so that frozen fields can
# be mutated via an explicit escape hatch, and copy-and-modify works.
# ---------------------------------------------------------------------------

import copy as _copy_module


def _install_field_helpers(cls, type_info):
    """Install ``_force_set_field`` and ``__replace__`` on *cls*.

    Called after field properties are installed.  Unlike the old
    shadow-slot approach, no ``__setattr__``/``__getattr__`` wrappers
    or populate hooks are needed — the ``property`` descriptors handle
    reads and writes directly via C++ memory.
    """
    if cls.__dict__.get("_tvm_ffi_field_helpers_installed", False):
        return

    # Collect setters for all reflected fields (including inherited).
    all_setters = {}
    info = type_info
    while info is not None:
        for tf in info.fields or ():
            if tf.name not in all_setters:
                all_setters[tf.name] = tf.setter
        info = info.parent_type_info

    if not all_setters:
        return

    copy_copy = _copy_module.copy

    def _force_set_field(obj, name, value):
        """Write to a reflected field, bypassing the frozen guard."""
        setter = all_setters.get(name)
        if setter is not None:
            setter(obj, value)
            return
        raise AttributeError(f"{type(obj).__name__!r} has no reflected field {name!r}")

    cls._force_set_field = staticmethod(_force_set_field)

    # Install __replace__ unless the class body already defines one.
    replace_fn = cls.__dict__.get("__replace__")
    if replace_fn is None or getattr(replace_fn, "__module__", None) == "tvm_ffi._dunder":

        def __replace__(self, **kwargs):
            obj = copy_copy(self)
            force_set = type(obj)._force_set_field
            for key, value in kwargs.items():
                force_set(obj, key, value)
            return obj

        __replace__.__qualname__ = f"{cls.__qualname__}.__replace__"
        __replace__.__module__ = cls.__module__
        cls.__replace__ = __replace__

    cls._tvm_ffi_field_helpers_installed = True


# ---------------------------------------------------------------------------
# Fast tp_getattro: C-level field table for reflected fields
#
# After properties are installed (for __setattr__/writes), we also build a
# compact C field table and override tp_getattro on the class.  Reads go
# through the C table (pointer comparison + inline read), bypassing the
# property descriptor mechanism entirely.  Non-field attributes fall through
# to PyObject_GenericGetAttr.
# ---------------------------------------------------------------------------

import sys as _sys_module
from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cpython.ref cimport Py_INCREF

cdef extern from "Python.h":
    PyObject* PyObject_GenericGetAttr(PyObject*, PyObject*)

# Module-level: byte-offset of CObject.chandle in the Python struct.
# Computed once at module init.
cdef Py_ssize_t _CHANDLE_PYOFFSET = 0

# Interned dict key for the field table capsule.
cdef object _FT_KEY_STR = None


# Byte-offset of tp_getattro inside PyTypeObject.
# Discovered at init time by finding PyObject_GenericGetAttr in the `object` type.
cdef Py_ssize_t _TP_GETATTRO_OFFSET = 0


cdef void _init_field_getattro():
    """Compute CObject.chandle offset, tp_getattro offset, and register callbacks."""
    global _CHANDLE_PYOFFSET, _FT_KEY_STR, _TP_GETATTRO_OFFSET
    cdef CObject dummy = CObject.__new__(CObject)
    _CHANDLE_PYOFFSET = <Py_ssize_t>(<char*>&dummy.chandle - <char*><void*>dummy)
    _FT_KEY_STR = _sys_module.intern("_ffi_field_table_ptr")
    # Discover tp_getattro offset by scanning `object` for PyObject_GenericGetAttr.
    cdef char* base = <char*><void*>object
    cdef void* target = <void*>PyObject_GenericGetAttr
    cdef Py_ssize_t off
    for off in range(0, 2048, sizeof(void*)):
        if (<void**>(base + off))[0] == target:
            _TP_GETATTRO_OFFSET = off
            break
    TVMFFIPyFieldTableSetup(
        <void*>_FT_KEY_STR,
        _make_ret_object_cb,
        _make_ret_cb,
    )

_init_field_getattro()


cdef PyObject* _make_ret_object_cb(const TVMFFIAny* result) noexcept:
    """Callback: convert TVMFFIAny (ObjectRef) to Python object. New reference."""
    cdef object ret = make_ret_object((<TVMFFIAny*>result)[0])
    Py_INCREF(ret)
    return <PyObject*>ret


cdef PyObject* _make_ret_cb(const TVMFFIAny* result) noexcept:
    """Callback: convert TVMFFIAny (Any) to Python object. New reference."""
    cdef object ret = make_ret((<TVMFFIAny*>result)[0])
    Py_INCREF(ret)
    return <PyObject*>ret


DEF _FIELD_TYPE_UNSUPPORTED = -999

cdef int32_t _field_type_index_from_getter(object getter):
    """Infer the FFI type index from a specialized getter instance."""
    if isinstance(getter, (FieldGetterInt64, FieldGetterInt32, FieldGetterInt16, FieldGetterInt8)):
        return kTVMFFIInt
    if isinstance(getter, FieldGetterFloat64) or isinstance(getter, FieldGetterFloat32):
        return kTVMFFIFloat
    if isinstance(getter, FieldGetterBool):
        return kTVMFFIBool
    if isinstance(getter, FieldGetterObjectRef):
        return kTVMFFIStaticObjectBegin
    if isinstance(getter, FieldGetterAny):
        return kTVMFFIAny
    return _FIELD_TYPE_UNSUPPORTED


def _install_field_getattro(cls, type_info):
    """Build a C field table and install custom tp_getattro on *cls*.

    Collects all reflected fields (own + inherited) into a compact C struct
    stored as a PyCapsule in ``cls._ffi_field_table_ptr``.  Then replaces
    ``tp_getattro`` with ``TVMFFIPyFieldGetAttro`` for fast C-level lookup.
    """
    # Collect ALL fields (own + inherited), child-first dedup.
    all_fields = {}
    info = type_info
    while info is not None:
        for tf in info.fields or ():
            if tf.name not in all_fields:
                all_fields[tf.name] = tf
        info = info.parent_type_info

    if not all_fields:
        return

    # First pass: count how many fields have a supported getter type.
    # Unsupported fields are skipped in the C table and served by the
    # fallback property-descriptor path instead.
    cdef int32_t supported_count = 0
    cdef int32_t tidx
    for name, tf in all_fields.items():
        if _field_type_index_from_getter(tf.getter) != _FIELD_TYPE_UNSUPPORTED:
            supported_count += 1
    if supported_count == 0:
        return

    cdef TVMFFIPyFieldTable* table
    cdef char* cls_ptr
    if supported_count > 64:
        return  # Too many fields — fall back to property path

    # Allocate and fill the C field table.
    table = <TVMFFIPyFieldTable*>malloc(sizeof(TVMFFIPyFieldTable))
    if table == NULL:
        return
    memset(table, 0, sizeof(TVMFFIPyFieldTable))
    table.count = supported_count
    table.chandle_pyoffset = _CHANDLE_PYOFFSET

    cdef int32_t i = 0
    for name, tf in all_fields.items():
        tidx = _field_type_index_from_getter(tf.getter)
        if tidx == _FIELD_TYPE_UNSUPPORTED:
            continue  # This field falls back to the property path
        interned_name = _sys_module.intern(name)
        Py_INCREF(interned_name)
        table.entries[i].name = <PyObject*>interned_name
        table.entries[i].offset = tf.offset
        table.entries[i].type_index = tidx
        table.entries[i].field_size = tf.size
        i += 1

    # Save the original tp_getattro (may be slot_tp_getattr_hook for __getattr__).
    if _TP_GETATTRO_OFFSET > 0:
        cls_ptr = <char*><void*>cls
        table.fallback = <TVMFFIPyGetAttroFunc>(<void**>(cls_ptr + _TP_GETATTRO_OFFSET))[0]
    else:
        table.fallback = NULL

    # Register in the global field-table map (avoids tp_dict access).
    if not TVMFFIPyFieldTableInsert(<void*>cls, table):
        # Probe limit exceeded — free the table and fall back to property path.
        free(table)
        return

    # Override tp_getattro using the offset discovered at init.
    if _TP_GETATTRO_OFFSET > 0:
        (<void**>(cls_ptr + _TP_GETATTRO_OFFSET))[0] = <void*>TVMFFIPyFieldGetAttro
        TVMFFIPyTypeModified(<void*>cls)
