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

"""Type converter implementation for TypeSchema.

Provides the ``_type_convert_impl`` function used by
``TypeSchema.convert`` and ``TypeSchema.check_value``.

Each ``_TypeConverter`` dispatches directly to a Cython cdef function that
returns a fully materialized :class:`CAny`. Container converters recurse into
their child schemas, rebuild the target FFI container shape, and then wrap the
final result in ``CAny``.
"""

import ctypes
import os
from numbers import Integral, Real
from collections.abc import Mapping


cdef object _INT64_MIN = -(1 << 63)
cdef object _INT64_MAX = (1 << 63) - 1
cdef int _VALUE_PROTOCOL_MAX_DEPTH = 64
cdef str _TYPE_ATTR_FFI_CONVERT = "__ffi_convert__"
cdef str _TYPE_ATTR_ENUM_VALUE_ENTRIES = "__ffi_enum_value_entries__"
cdef class _TypeConverter
ctypedef CAny (*_dispatch_fn_t)(_TypeConverter, object, bint*) except *


# ---------------------------------------------------------------------------
# cdef class _TypeConverter — holds dispatch state as C-level struct fields
# ---------------------------------------------------------------------------
cdef class _TypeConverter:
    """Pre-built converter holding a C function pointer and sub-converters."""

    cdef _dispatch_fn_t dispatch
    cdef int32_t type_index
    cdef tuple subs
    cdef str err_hint
    cdef Function _fn_convert
    cdef bint _fn_convert_ready

    @property
    def fn_convert(self):
        cdef object attr
        assert self.type_index >= kTVMFFIStaticObjectBegin
        if not self._fn_convert_ready:
            attr = _lookup_type_attr(self.type_index, _TYPE_ATTR_FFI_CONVERT)
            if attr is not None:
                assert isinstance(attr, Function)
                self._fn_convert = <Function>attr
            else:
                self._fn_convert = None
            self._fn_convert_ready = True
        return self._fn_convert


class _ConvertError(Exception):
    """Internal exception used to signal conversion failure."""

    __slots__ = ()

    def __init__(self, message):
        super().__init__(message)

    @property
    def message(self):
        return self.args[0]


# ---------------------------------------------------------------------------
# Converters (1/N): Simple value converters
# ---------------------------------------------------------------------------

cdef CAny _tc_convert_any(_TypeConverter _conv, object value, bint* changed) except *:
    """Any: accept any marshalable FFI value."""
    cdef CAny packed
    assert _CLASS_DEVICE is not None
    assert _CLASS_DTYPE is not None
    packed = CAnyChecked(value, "Any", value)
    if not isinstance(
        value,
        (
            type(None),
            bool,
            int,
            float,
            ctypes.c_void_p,
            String,
            Bytes,
            Tensor,
            DataType,
            CObject,
            _CLASS_DEVICE,
            _CLASS_DTYPE,
        ),
    ):
        changed[0] = True
    return packed


cdef CAny _tc_convert_none(_TypeConverter _conv, object value, bint* changed) except *:
    """None accepts: None only."""
    if value is None:
        return CAny(None)
    raise _ConvertError(f"expected None, got {_tc_describe_value_type(value)}")


cdef CAny _tc_convert_int(_TypeConverter _conv, object value, bint* changed) except *:
    """int accepts: int, bool, Integral, __tvm_ffi_int__ protocol."""
    cdef object ivalue
    if isinstance(value, bool):
        changed[0] = True
        return CAny(int(value))
    if isinstance(value, int):
        if not (_INT64_MIN <= value <= _INT64_MAX):
            raise _ConvertError(
                f"integer {value} out of int64 range [{_INT64_MIN}, {_INT64_MAX}]"
            )
        return CAny(value)
    if isinstance(value, Integral):
        try:
            ivalue = int(value)
        except Exception as err:
            raise _ConvertError(f"int() failed for {type(value).__qualname__}: {err}") from None
        if not (_INT64_MIN <= ivalue <= _INT64_MAX):
            raise _ConvertError(
                f"integer {ivalue} out of int64 range [{_INT64_MIN}, {_INT64_MAX}]"
            )
        changed[0] = True
        return CAny(ivalue)
    if hasattr(type(value), "__tvm_ffi_int__"):
        changed[0] = True
        return CAnyChecked(value, "int", value)
    raise _ConvertError(f"expected int, got {_tc_describe_value_type(value)}")


cdef CAny _tc_convert_float(_TypeConverter _conv, object value, bint* changed) except *:
    """float accepts: float, int, bool, Real, __tvm_ffi_float__ protocol."""
    cdef object fvalue
    if isinstance(value, float):
        return CAny(value)
    if isinstance(value, (int, bool)):
        changed[0] = True
        return CAny(float(value))
    if isinstance(value, (Integral, Real)):
        try:
            fvalue = float(value)
        except Exception as err:
            raise _ConvertError(f"float() failed for {type(value).__qualname__}: {err}") from None
        changed[0] = True
        return CAny(fvalue)
    if hasattr(type(value), "__tvm_ffi_float__"):
        changed[0] = True
        return CAnyChecked(value, "float", value)
    raise _ConvertError(f"expected float, got {_tc_describe_value_type(value)}")


cdef CAny _tc_convert_bool(_TypeConverter _conv, object value, bint* changed) except *:
    """bool accepts: bool, int, Integral."""
    cdef object bvalue
    if isinstance(value, bool):
        return CAny(value)
    if isinstance(value, Integral):  # TODO: do we coerce Integral to bool?
        try:
            bvalue = bool(value)
        except Exception as err:
            raise _ConvertError(f"bool() failed for {type(value).__qualname__}: {err}") from None
        changed[0] = True
        return CAny(bvalue)
    raise _ConvertError(f"expected bool, got {_tc_describe_value_type(value)}")


cdef CAny _tc_convert_str(_TypeConverter _conv, object value, bint* changed) except *:
    """str accepts: str only.  Short strings use SmallStr (inline in Any)."""
    if isinstance(value, String):
        return CAny(value)
    if isinstance(value, str):
        changed[0] = True
        return CAny(value)
    raise _ConvertError(f"expected str, got {_tc_describe_value_type(value)}")


cdef CAny _tc_convert_bytes(_TypeConverter _conv, object value, bint* changed) except *:
    """bytes accepts: bytes, bytearray.  Short bytes use SmallBytes (inline in Any)."""
    if isinstance(value, Bytes):
        return CAny(value)
    if isinstance(value, bytes):
        changed[0] = True
        return CAny(value)
    if isinstance(value, bytearray):
        changed[0] = True
        return CAny(bytes(value))
    raise _ConvertError(f"expected bytes, got {_tc_describe_value_type(value)}")


cdef CAny _tc_convert_device(_TypeConverter _conv, object value, bint* changed) except *:
    """Device accepts: Device and __dlpack_device__ without __dlpack__."""
    cdef object vtype = type(value)
    assert _CLASS_DEVICE is not None
    if isinstance(value, _CLASS_DEVICE):
        return CAny(value)
    if hasattr(vtype, "__dlpack_device__") and not hasattr(vtype, "__dlpack__"):
        changed[0] = True
        return CAnyChecked(value, "Device", value)
    raise _ConvertError(f"expected Device, got {_tc_describe_value_type(value)}")


cdef CAny _tc_convert_dtype(_TypeConverter _conv, object value, bint* changed) except *:
    """dtype accepts: DataType, dtype wrapper, str and dtype protocols."""
    cdef object dtype_value
    assert _CLASS_DTYPE is not None
    if isinstance(value, (DataType, _CLASS_DTYPE)):
        return CAny(value)
    if isinstance(value, str):
        try:
            dtype_value = DataType(value)
        except Exception:
            raise _ConvertError(f"expected dtype, got invalid dtype string {value!r}") from None
        changed[0] = True
        return CAny(dtype_value)
    if (
        (torch is not None and isinstance(value, torch.dtype))
        or (numpy is not None and isinstance(value, numpy.dtype))
        or hasattr(value, "__dlpack_data_type__")
    ):
        changed[0] = True
        return CAnyChecked(value, "dtype", value)
    raise _ConvertError(f"expected dtype, got {_tc_describe_value_type(value)}")


cdef CAny _tc_convert_opaque_ptr(_TypeConverter _conv, object value, bint* changed) except *:
    """ctypes.c_void_p accepts ctypes.c_void_p, None and opaque pointer protocols."""
    cdef object vtype = type(value)
    if isinstance(value, ctypes.c_void_p):
        return CAny(value)
    # TODO: noticed that `OpaquePtr(nullptr) != None` - need to double check if this is correct
    if value is None:
        changed[0] = True
        return CAny(ctypes.c_void_p(None))
    if hasattr(vtype, "__tvm_ffi_opaque_ptr__") or hasattr(vtype, "__cuda_stream__"):
        changed[0] = True
        return CAnyChecked(value, "ctypes.c_void_p", value)
    raise _ConvertError(f"expected ctypes.c_void_p, got {_tc_describe_value_type(value)}")


cdef CAny _tc_convert_tensor(_TypeConverter _conv, object value, bint* changed) except *:
    """Tensor accepts Tensor, Tensor subtypes and DLPack exporters."""
    cdef object vtype = type(value)
    if isinstance(value, Tensor):
        return CAny(value)
    if hasattr(vtype, "__dlpack__"):
        changed[0] = True
        return CAnyChecked(value, "Tensor", value)
    if os.environ.get("TVM_FFI_SKIP_DLPACK_C_EXCHANGE_API", "0") != "1":
        if hasattr(vtype, "__dlpack_c_exchange_api__"):
            changed[0] = True
            return CAnyChecked(value, "Tensor", value)
    raise _ConvertError(f"expected Tensor, got {_tc_describe_value_type(value)}")


cdef CAny _tc_convert_callable(_TypeConverter _conv, object value, bint* changed) except *:
    """Callable accepts Function and any Python callable."""
    cdef Function func
    if isinstance(value, Function):
        return CAny(value)
    if callable(value):
        if isinstance(value, CObject):
            raise _ConvertError(f"expected Callable, got {_tc_describe_value_type(value)}")
        changed[0] = True
        return CAnyChecked(value, "Callable", value)
    raise _ConvertError(f"expected Callable, got {_tc_describe_value_type(value)}")


# ---------------------------------------------------------------------------
# Converters (2/N): Sequence/Mapping Containers
# ---------------------------------------------------------------------------


cdef CAny _tc_convert_array(_TypeConverter conv, object value, bint* changed) except *:
    """Dispatch for Array[T]. Accepts Array or List CObjects (cross-type)."""
    from tvm_ffi.container import Array

    return _tc_convert_seq(conv, value, changed, Array)


cdef CAny _tc_convert_list(_TypeConverter conv, object value, bint* changed) except *:
    """Dispatch for List[T]. Accepts List or Array CObjects (cross-type)."""
    from tvm_ffi.container import List

    return _tc_convert_seq(conv, value, changed, List)


cdef CAny _tc_convert_map(_TypeConverter conv, object value, bint* changed) except *:
    """Dispatch for Map[K, V]. Accepts Map or Dict CObjects (cross-type)."""
    from tvm_ffi import _ffi_api
    from tvm_ffi.container import Map

    return _tc_convert_mapping(conv, value, changed, Map, _ffi_api.Map)


cdef CAny _tc_convert_dict(_TypeConverter conv, object value, bint* changed) except *:
    """Dispatch for Dict[K, V]. Accepts Dict or Map CObjects (cross-type)."""
    from tvm_ffi import _ffi_api
    from tvm_ffi.container import Dict

    return _tc_convert_mapping(conv, value, changed, Dict, _ffi_api.Dict)


cdef CAny _tc_convert_seq(_TypeConverter conv, object value, bint* changed, object seq_type) except *:
    from tvm_ffi.container import Array, List

    cdef _TypeConverter elem_conv = conv.subs[0] if conv.subs else None

    if not isinstance(value, (Array, List, list, tuple)):
        raise _ConvertError(f"expected {seq_type.__name__}, got {_tc_describe_value_type(value)}")

    if elem_conv is None and isinstance(value, seq_type):
        return CAny(value)

    cdef list converted = []
    cdef bint item_changed
    cdef object raw_item
    cdef int i = 0
    cdef CAny item
    for raw_item in value:
        if elem_conv is not None:
            item_changed = False
            try:
                item = _type_convert_dispatch_with_fallback(elem_conv, raw_item, &item_changed)
            except _ConvertError as err:
                raise _ConvertError(f"element [{i}]: {err.message}") from None
            if item_changed:
                changed[0] = True
                converted.append(_to_py_class_value(item))
            else:
                converted.append(raw_item)
        else:
            converted.append(raw_item)
        i += 1
    if isinstance(value, seq_type) and not changed[0]:
        return CAny(value)
    changed[0] = True
    return CAnyChecked(seq_type(converted), seq_type.__name__, value)


cdef CAny _tc_convert_tuple(_TypeConverter conv, object value, bint* changed) except *:
    """Dispatch for tuple[T1, T2, ...] or bare tuple."""
    cdef int i
    cdef int n
    cdef list converted = []
    cdef CAny item
    cdef bint item_changed
    cdef object raw_item

    from tvm_ffi.container import Array, List

    if not isinstance(value, (Array, List, list, tuple)):
        raise _ConvertError(f"expected tuple, got {_tc_describe_value_type(value)}")

    if conv.subs is None:
        if isinstance(value, Array):
            return CAny(value)
        changed[0] = True
        return CAnyChecked(Array(value), "tuple", value)

    n = len(conv.subs)
    if len(value) != n:
        raise _ConvertError(
            f"expected tuple of length {n}, got {type(value).__name__} of length {len(value)}"
        )

    for i in range(n):
        raw_item = value[i]
        item_changed = False
        try:
            item = _type_convert_dispatch_with_fallback(
                <_TypeConverter>(conv.subs[i]),
                raw_item,
                &item_changed,
            )
        except _ConvertError as err:
            raise _ConvertError(f"element [{i}]: {err.message}") from None
        if item_changed:
            changed[0] = True
            converted.append(_to_py_class_value(item))
        else:
            converted.append(raw_item)
    if isinstance(value, Array) and not changed[0]:
        return CAny(value)
    changed[0] = True
    return CAnyChecked(Array(converted), "tuple", value)


cdef CAny _tc_convert_mapping(
    _TypeConverter conv,
    object value,
    bint* changed,
    object mapping_type,
    object constructor,
) except *:
    cdef _TypeConverter key_conv = conv.subs[0] if conv.subs else None
    cdef _TypeConverter val_conv = conv.subs[1] if conv.subs else None
    cdef list list_kvs = []
    cdef CAny item
    cdef bint item_changed
    cdef object raw_key
    cdef object raw_val
    cdef object new_key
    cdef object new_val
    cdef object mapping
    cdef str expected = mapping_type.__name__

    if not isinstance(value, Mapping):
        raise _ConvertError(f"expected {expected}, got {_tc_describe_value_type(value)}")

    if key_conv is None and val_conv is None and isinstance(value, mapping_type):
        return CAny(value)

    for raw_key, raw_val in value.items():
        new_key = raw_key
        if key_conv is not None:
            item_changed = False
            try:
                item = _type_convert_dispatch_with_fallback(key_conv, raw_key, &item_changed)
            except _ConvertError as err:
                raise _ConvertError(f"key {raw_key!r}: {err.message}") from None
            if item_changed:
                changed[0] = True
                new_key = _to_py_class_value(item)
        new_val = raw_val
        if val_conv is not None:
            item_changed = False
            try:
                item = _type_convert_dispatch_with_fallback(val_conv, raw_val, &item_changed)
            except _ConvertError as err:
                raise _ConvertError(f"value for key {raw_key!r}: {err.message}") from None
            if item_changed:
                changed[0] = True
                new_val = _to_py_class_value(item)
        list_kvs.append(new_key)
        list_kvs.append(new_val)
    if isinstance(value, mapping_type) and not changed[0]:
        return CAny(value)
    changed[0] = True
    try:
        mapping = constructor(*list_kvs)
    except _ConvertError:
        raise
    except Exception as err:
        raise _ConvertError(
            f"expected {expected}, got {_tc_describe_value_type(value)}: {err}"
        ) from None
    return CAnyChecked(mapping, expected, value)


# ---------------------------------------------------------------------------
# Converters (3/N): Optional and Union
# ---------------------------------------------------------------------------

cdef CAny _tc_convert_optional(_TypeConverter conv, object value, bint* changed) except *:
    """Dispatch for Optional[T]: None passthrough or inner dispatch."""
    if value is None:
        return CAny(None)
    return _type_convert_dispatch_with_fallback(<_TypeConverter>(conv.subs[0]), value, changed)


cdef CAny _tc_convert_union(_TypeConverter conv, object value, bint* changed) except *:
    """Dispatch for Union[T1, T2, ...]."""
    cdef _TypeConverter alt
    cdef bint alt_changed
    cdef CAny result
    for alt_obj in conv.subs:
        alt = <_TypeConverter>alt_obj
        try:
            alt_changed = False
            result = alt.dispatch(alt, value, &alt_changed)
            changed[0] = alt_changed
            return result
        except _ConvertError:
            pass
    raise _ConvertError(f"expected {conv.err_hint}, got {_tc_describe_value_type(value)}")


# ---------------------------------------------------------------------------
# Converters (4/N): Object Types
# ---------------------------------------------------------------------------

cdef inline object _tc_get_registered_cls(int32_t type_index):
    if 0 <= type_index < len(TYPE_INDEX_TO_CLS):
        return TYPE_INDEX_TO_CLS[type_index]
    return None


cdef inline bint _tc_registered_cls_has_base(
    int32_t type_index,
    str module_name,
    str base_name,
) except *:
    cdef object cls = _tc_get_registered_cls(type_index)
    cdef object base
    if cls is None:
        return False
    for base in cls.__mro__:
        if (
            getattr(base, "__module__", None) == module_name
            and getattr(base, "__name__", None) == base_name
        ):
            return True
    return False


cdef object _tc_find_payload_enum_variant(
    int32_t type_index, object enum_cls, object payload
) except *:
    """Resolve *payload* to its singleton variant (``None`` if no match).

    Primary path: O(1) lookup in the cross-language value-entries column
    (``__ffi_enum_value_entries__``).  Falls back to an O(n) linear scan
    over ``enum_cls.all_entries()`` when the column has no entry for
    *payload* — needed so correctness is preserved for variants whose
    creators haven't populated the column.
    """
    cdef object value_entries
    cdef object variant
    value_entries = _lookup_type_attr(type_index, _TYPE_ATTR_ENUM_VALUE_ENTRIES)
    if value_entries is not None:
        variant = value_entries.get(payload)
        if variant is not None:
            return variant
    for variant in enum_cls.all_entries():
        if variant.value == payload:
            return variant
    return None


cdef object _tc_normalize_int_enum_payload(object value, bint* matched) except *:
    cdef object ivalue
    matched[0] = False
    if isinstance(value, bool):
        matched[0] = True
        return int(value)
    if isinstance(value, int):
        if not (_INT64_MIN <= value <= _INT64_MAX):
            raise _ConvertError(
                f"integer {value} out of int64 range [{_INT64_MIN}, {_INT64_MAX}]"
            )
        matched[0] = True
        return value
    if isinstance(value, Integral):
        try:
            ivalue = int(value)
        except Exception as err:
            raise _ConvertError(f"int() failed for {type(value).__qualname__}: {err}") from None
        if not (_INT64_MIN <= ivalue <= _INT64_MAX):
            raise _ConvertError(
                f"integer {ivalue} out of int64 range [{_INT64_MIN}, {_INT64_MAX}]"
            )
        matched[0] = True
        return ivalue
    return None


cdef CAny _tc_convert_object_marshaled(_TypeConverter conv, object value) except *:
    cdef int32_t actual_type_index = kTVMFFINone
    cdef CAny packed
    cdef CAny converted
    cdef Function fn_convert
    cdef object err = None

    packed = CAnyChecked(value, conv.err_hint, value)
    fn_convert = conv.fn_convert
    try:
        if fn_convert is not None:
            converted = CAny.__new__(CAny)
            CHECK_CALL(TVMFFIFunctionCall(fn_convert.chandle, &packed.cdata, 1, &converted.cdata))
        else:
            converted = packed
    except Exception as err_:
        err = err_
    else:
        actual_type_index = converted.type_index
        if actual_type_index >= kTVMFFIStaticObjectBegin:
            if _tc_type_index_is_instance(actual_type_index, conv.type_index):
                return converted
    raise _ConvertError(f"expected {conv.err_hint}, got {_tc_describe_value_type(value)}") from err


cdef CAny _tc_convert_object(_TypeConverter conv, object value, bint* changed) except *:
    """Convert *value* to an object compatible with ``conv.type_index``."""
    # TODO: SmallStr and SmallBytes => ObjectRef conversion is not supported yet
    cdef int32_t actual_type_index = kTVMFFINone

    # Step 1: existing FFI objects that already satisfy the target schema are passthrough.
    assert conv.type_index >= kTVMFFIStaticObjectBegin
    if isinstance(value, CObject):
        actual_type_index = TVMFFIObjectGetTypeIndex((<CObject>value).chandle)
        if _tc_type_index_is_instance(actual_type_index, conv.type_index):
            return CAny(value)
    changed[0] = True

    # Step 2: pack, and convert to the target type.
    return _tc_convert_object_marshaled(conv, value)


cdef CAny _tc_convert_int_enum(_TypeConverter conv, object value, bint* changed) except *:
    """Convert *value* to an IntEnum-compatible object."""
    cdef int32_t actual_type_index = kTVMFFINone
    cdef object target_cls
    cdef object ivalue
    cdef object variant
    cdef bint is_int_like = False

    assert conv.type_index >= kTVMFFIStaticObjectBegin
    if isinstance(value, CObject):
        actual_type_index = TVMFFIObjectGetTypeIndex((<CObject>value).chandle)
        if _tc_type_index_is_instance(actual_type_index, conv.type_index):
            return CAny(value)

    target_cls = _tc_get_registered_cls(conv.type_index)
    if target_cls is not None:
        ivalue = _tc_normalize_int_enum_payload(value, &is_int_like)
        if is_int_like:
            changed[0] = True
            variant = _tc_find_payload_enum_variant(conv.type_index, target_cls, ivalue)
            if variant is not None:
                return CAny(variant)

    changed[0] = True
    return _tc_convert_object_marshaled(conv, value)


cdef CAny _tc_convert_str_enum(_TypeConverter conv, object value, bint* changed) except *:
    """Convert *value* to a StrEnum-compatible object."""
    cdef int32_t actual_type_index = kTVMFFINone
    cdef object target_cls
    cdef object variant

    assert conv.type_index >= kTVMFFIStaticObjectBegin
    if isinstance(value, CObject):
        actual_type_index = TVMFFIObjectGetTypeIndex((<CObject>value).chandle)
        if _tc_type_index_is_instance(actual_type_index, conv.type_index):
            return CAny(value)

    target_cls = _tc_get_registered_cls(conv.type_index)
    if target_cls is not None and isinstance(value, str):
        changed[0] = True
        variant = _tc_find_payload_enum_variant(conv.type_index, target_cls, value)
        if variant is not None:
            return CAny(variant)

    changed[0] = True
    return _tc_convert_object_marshaled(conv, value)


cdef inline bint _tc_type_index_is_instance(int32_t actual_tindex, int32_t target_tindex) noexcept:
    """Check if *actual_tindex* is *target_tindex* or a subclass thereof."""
    # TODO: this can be optimized by looking up `TYPE_INDEX_TO_INFO`
    if actual_tindex == target_tindex:
        return True
    cdef const TVMFFITypeInfo* actual_info = TVMFFIGetTypeInfo(actual_tindex)
    if actual_info == NULL:
        return False
    cdef const TVMFFITypeInfo* target_info = TVMFFIGetTypeInfo(target_tindex)
    if target_info == NULL:
        return False
    cdef int32_t target_depth = target_info.type_depth
    if actual_info.type_depth <= target_depth:
        return False
    return actual_info.type_ancestors[target_depth].type_index == target_tindex


# ---------------------------------------------------------------------------
# Helper: describe the Python type of a value for error messages
# ---------------------------------------------------------------------------
cdef str _tc_describe_value_type(object value):
    """Return a human-readable type description for *value*."""
    cdef object type_info
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, (bytes, bytearray)):
        return "bytes"
    if isinstance(value, CObject):
        type_info = getattr(type(value), "__tvm_ffi_type_info__", None)
        if type_info is not None:
            return type_info.type_key
        return _type_index_to_key(TVMFFIObjectGetTypeIndex((<CObject>value).chandle))
    return type(value).__qualname__


cdef CAny CAnyChecked(object value, str expected, object original_value) except *:
    """Pack *value* into CAny and normalize packing failures to _ConvertError."""
    try:
        return CAny(value)
    except _ConvertError:
        raise
    except Exception as err:
        raise _ConvertError(
            f"expected {expected}, got {_tc_describe_value_type(original_value)}: {err}"
        ) from None


# ---------------------------------------------------------------------------
# Builder (runs once per TypeSchema at construction time)
# ---------------------------------------------------------------------------

def _build_converter(schema):
    """Build a ``_TypeConverter`` for *schema*."""
    cdef _TypeConverter conv = _TypeConverter.__new__(_TypeConverter)
    cdef _TypeConverter sub_conv
    cdef _TypeConverter key_conv
    cdef _TypeConverter val_conv
    origin = schema.origin
    args = schema.args
    origin_tindex = schema.origin_type_index
    conv.err_hint = origin

    def _to_type_converter_or_none(object schema_arg):
        if schema_arg.origin == "Any":
            return None
        return <_TypeConverter>(schema_arg._converter)

    if origin_tindex == kTVMFFIAny or origin == "Any":
        conv.dispatch = _tc_convert_any
        conv.err_hint = "Any"
        return conv

    if origin == "Optional":
        conv.dispatch = _tc_convert_optional
        conv.subs = (<_TypeConverter>(args[0]._converter),)
        return conv
    if origin == "Union":
        conv.dispatch = _tc_convert_union
        conv.subs = tuple(<_TypeConverter>(a._converter) for a in args)
        conv.err_hint = " | ".join(repr(a) for a in args)
        return conv

    if origin == "int":
        conv.dispatch = _tc_convert_int
        return conv
    if origin == "float":
        conv.dispatch = _tc_convert_float
        return conv
    if origin == "bool":
        conv.dispatch = _tc_convert_bool
        return conv
    if origin == "None":
        conv.dispatch = _tc_convert_none
        return conv
    if origin == "str":
        conv.dispatch = _tc_convert_str
        return conv
    if origin == "bytes":
        conv.dispatch = _tc_convert_bytes
        return conv
    if origin == "Device":
        conv.dispatch = _tc_convert_device
        return conv
    if origin == "dtype":
        conv.dispatch = _tc_convert_dtype
        return conv
    if origin == "ctypes.c_void_p":
        conv.dispatch = _tc_convert_opaque_ptr
        return conv
    if origin == "Tensor":
        conv.dispatch = _tc_convert_tensor
        return conv
    if origin == "Callable":
        conv.dispatch = _tc_convert_callable
        return conv

    if origin == "Array":
        conv.dispatch = _tc_convert_array
        if len(args) > 0:
            sub_conv = _to_type_converter_or_none(args[0])
            if sub_conv is not None:
                conv.subs = (sub_conv,)
        return conv
    if origin in ("List", "list"):
        conv.dispatch = _tc_convert_list
        conv.err_hint = "List"
        if len(args) > 0:
            sub_conv = _to_type_converter_or_none(args[0])
            if sub_conv is not None:
                conv.subs = (sub_conv,)
        return conv
    if origin == "Map":
        conv.dispatch = _tc_convert_map
        if len(args) == 2:
            key_conv = _to_type_converter_or_none(args[0])
            val_conv = _to_type_converter_or_none(args[1])
            if key_conv is not None or val_conv is not None:
                conv.subs = (key_conv, val_conv)
        return conv
    if origin in ("Dict", "dict"):
        conv.dispatch = _tc_convert_dict
        conv.err_hint = "Dict"
        if len(args) == 2:
            key_conv = _to_type_converter_or_none(args[0])
            val_conv = _to_type_converter_or_none(args[1])
            if key_conv is not None or val_conv is not None:
                conv.subs = (key_conv, val_conv)
        return conv
    if origin == "tuple":
        conv.dispatch = _tc_convert_tuple
        if args is not None:
            conv.subs = tuple(<_TypeConverter>(a._converter) for a in args)
        return conv

    if origin == "Object":
        conv.dispatch = _tc_convert_object
        conv.type_index = kTVMFFIObject
        conv.err_hint = "Object"
        return conv
    if origin_tindex >= kTVMFFIStaticObjectBegin:
        if _tc_registered_cls_has_base(origin_tindex, "tvm_ffi.dataclasses.enum", "IntEnum"):
            conv.dispatch = _tc_convert_int_enum
        elif _tc_registered_cls_has_base(origin_tindex, "tvm_ffi.dataclasses.enum", "StrEnum"):
            conv.dispatch = _tc_convert_str_enum
        else:
            conv.dispatch = _tc_convert_object
        conv.type_index = origin_tindex
        conv.err_hint = origin
        return conv

    tindex = _object_type_key_to_index(origin)
    if tindex is not None:
        if _tc_registered_cls_has_base(tindex, "tvm_ffi.dataclasses.enum", "IntEnum"):
            conv.dispatch = _tc_convert_int_enum
        elif _tc_registered_cls_has_base(tindex, "tvm_ffi.dataclasses.enum", "StrEnum"):
            conv.dispatch = _tc_convert_str_enum
        else:
            conv.dispatch = _tc_convert_object
        conv.type_index = tindex
        conv.err_hint = origin
        return conv

    raise TypeError(f"unknown TypeSchema origin: {origin!r}")


# ---------------------------------------------------------------------------
# Eager protocol normalization and dispatch
# ---------------------------------------------------------------------------


cdef void _tc_raise_eager_value_protocol_error(_TypeConverter conv, object value) except *:
    if conv.dispatch == _tc_convert_optional:
        _tc_raise_eager_value_protocol_error(<_TypeConverter>(conv.subs[0]), value)
    if conv.err_hint == "Any":
        raise _ConvertError(f"failed to convert Any from {_tc_describe_value_type(value)}")
    raise _ConvertError(f"expected {conv.err_hint}, got {_tc_describe_value_type(value)}")


cdef object _tc_eager_protocol_step(object value, bint* stalled_value_protocol) except *:
    cdef object vtype
    cdef object inner
    if isinstance(value, (Tensor, CObject, ObjectRValueRef, PyNativeObject)):
        return value
    vtype = type(value)
    if hasattr(vtype, "__tvm_ffi_object__"):
        try:
            return value.__tvm_ffi_object__()
        except Exception:
            raise _ConvertError(
                f"__tvm_ffi_object__() failed for {_tc_describe_value_type(value)}"
            ) from None
    if hasattr(vtype, "__tvm_ffi_value__"):
        try:
            inner = value.__tvm_ffi_value__()
        except Exception:
            # Report the schema mismatch instead of leaking the raw
            # __tvm_ffi_value__ implementation error.
            stalled_value_protocol[0] = True
            return value
        if inner is value:
            stalled_value_protocol[0] = True
        return inner
    if isinstance(value, ObjectConvertible):
        # Normalize ObjectConvertible eagerly so nested Union/container dispatch
        # sees the inner FFI object instead of the Python wrapper.
        try:
            inner = value.asobject()
        except Exception:
            raise _ConvertError(f"asobject() failed for {_tc_describe_value_type(value)}") from None
        if not isinstance(inner, CObject):
            raise _ConvertError(
                f"asobject() returned {_tc_describe_value_type(inner)} "
                f"for {_tc_describe_value_type(value)}"
            )
        return inner
    return value


cdef CAny _type_convert_dispatch_with_fallback(_TypeConverter conv, object value, bint* changed) except *:
    """Dispatch after eager protocol normalization with cycle protection."""
    cdef int depth = 0
    cdef object inner
    cdef bint stalled_value_protocol
    cdef bint used_value_protocol = False
    cdef CAny result
    while True:
        stalled_value_protocol = False
        inner = _tc_eager_protocol_step(value, &stalled_value_protocol)
        if stalled_value_protocol:
            _tc_raise_eager_value_protocol_error(conv, value)
        if inner is value:
            break
        depth += 1
        if depth > _VALUE_PROTOCOL_MAX_DEPTH:
            raise _ConvertError("infinite __tvm_ffi_value__ cycle detected") from None
        used_value_protocol = True
        value = inner
    changed[0] = False
    result = conv.dispatch(conv, value, changed)
    if used_value_protocol:
        changed[0] = True
    return result


# ---------------------------------------------------------------------------
# Main dispatcher (thin entry point from Python-level TypeSchema methods)
# ---------------------------------------------------------------------------
cdef CAny _type_convert_impl(_TypeConverter converter, object value) except *:
    """Dispatch to the C-level converter."""
    cdef bint changed
    return _type_convert_dispatch_with_fallback(converter, value, &changed)
