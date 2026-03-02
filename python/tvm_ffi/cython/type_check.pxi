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
``TypeSchema.convert`` and ``TypeSchema.try_convert``.
The conversion logic follows the Python FFI marshal path
(``TVMFFIPyArgSetterFactory_`` in ``function.pxi`` and
``tvm_ffi.convert`` in ``_convert.py``), accepting all
value types and protocols that the runtime would accept.

Key design: ``_type_convert_impl`` returns the **converted value** on success,
or a ``_ConvertError`` instance on failure. This cleanly distinguishes
``None``-as-a-valid-result (e.g. Optional[int] converting None) from
conversion failure.

Performance: Each ``TypeSchema`` stores a ``_TypeConverter`` cdef class instance
built at construction time.  Dispatch uses a C function pointer stored in
the converter — each ``convert()``/``check_value()`` call is a single
indirect C function call with zero Python overhead and no branch cascade.
"""
import ctypes
from numbers import Integral, Real


# ---------------------------------------------------------------------------
# Object type hierarchy check (pure Cython, uses C API declared in base.pxi)
# ---------------------------------------------------------------------------
cdef inline bint _is_object_instance(int32_t obj_tindex, int32_t target_tindex) noexcept:
    """Check if *obj_tindex* is *target_tindex* or a subclass thereof.

    Walks the TVMFFITypeInfo.type_ancestors chain.  All data comes from
    the C runtime type registry; no Python objects are touched.
    """
    if obj_tindex == target_tindex:
        return True
    cdef const TVMFFITypeInfo* obj_info = TVMFFIGetTypeInfo(obj_tindex)
    if obj_info == NULL:
        return False
    cdef const TVMFFITypeInfo* target_info = TVMFFIGetTypeInfo(target_tindex)
    if target_info == NULL:
        return False
    cdef int32_t target_depth = target_info.type_depth
    if obj_info.type_depth <= target_depth:
        return False
    return obj_info.type_ancestors[target_depth].type_index == target_tindex


class _ConvertError:
    """Sentinel returned by ``_type_convert_impl`` on conversion failure.

    Instances are never valid conversion results, so the caller can
    reliably use ``isinstance(result, _ConvertError)`` to distinguish
    failure from a successful conversion whose result happens to be ``None``.
    """

    __slots__ = ("message",)

    def __init__(self, message):
        self.message = message


# ---------------------------------------------------------------------------
# Function pointer type for single-dispatch converters
# ---------------------------------------------------------------------------
ctypedef object (*_dispatch_fn_t)(object, object)


# ---------------------------------------------------------------------------
# cdef class _TypeConverter — holds dispatch state as C-level struct fields
# ---------------------------------------------------------------------------
cdef class _TypeConverter:
    """Pre-built converter holding a C function pointer and sub-converters.

    The ``dispatch`` field is a C function pointer set once at build time.
    Calling ``conv.dispatch(conv, value)`` compiles to a single indirect
    C function call — no Python attribute lookup, no branch cascade.

    Fields
    ------
    dispatch : C function pointer ``(object, object) -> object``
    type_index : int32_t — target type index (Object types only)
    subs : tuple of _TypeConverter — sub-converters for composite types
    err_hint : str — pre-built string for error messages
    """
    cdef _dispatch_fn_t dispatch
    cdef int32_t type_index
    cdef tuple subs
    cdef str err_hint


# ---------------------------------------------------------------------------
# Helper: describe the Python type of a value for error messages
# ---------------------------------------------------------------------------
cdef str _tc_describe_value_type(object value):
    """Return a human-readable type description for *value*."""
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
        return _type_index_to_key(
            TVMFFIObjectGetTypeIndex((<CObject>value).chandle)
        )
    return type(value).__qualname__


# ---------------------------------------------------------------------------
# Leaf type converters (cdef for C-level performance)
# ---------------------------------------------------------------------------
_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1


cdef object _tc_convert_any(object _conv, object value):
    """Any: accept everything."""
    return value


cdef object _tc_convert_int(object _conv, object value):
    """int accepts: int, bool, Integral, __tvm_ffi_int__ protocol.

    Converts bool/Integral to int. Rejects values outside int64 range
    since the FFI marshals Python int to C++ ``int64_t``.
    Values with ``__tvm_ffi_int__`` are accepted as-is (marshal handles them).
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        if not (_INT64_MIN <= value <= _INT64_MAX):
            return _ConvertError(
                f"integer {value} out of int64 range "
                f"[{_INT64_MIN}, {_INT64_MAX}]"
            )
        return value
    if isinstance(value, Integral):
        try:
            value = int(value)
        except Exception as e:
            return _ConvertError(f"int() failed for {type(value).__qualname__}: {e}")
        if not (_INT64_MIN <= value <= _INT64_MAX):
            return _ConvertError(
                f"integer {value} out of int64 range "
                f"[{_INT64_MIN}, {_INT64_MAX}]"
            )
        return value
    if hasattr(type(value), "__tvm_ffi_int__"):
        return value
    return _ConvertError(f"expected int, got {_tc_describe_value_type(value)}")


cdef object _tc_convert_float(object _conv, object value):
    """float accepts: float, int, bool, Real, __tvm_ffi_float__ protocol.

    Converts int/bool/Integral/Real to float.
    Values with ``__tvm_ffi_float__`` are accepted as-is (marshal handles them).
    """
    if isinstance(value, float):
        return value
    if isinstance(value, (int, bool)):
        return float(value)
    if isinstance(value, (Integral, Real)):
        try:
            return float(value)
        except Exception as e:
            return _ConvertError(f"float() failed for {type(value).__qualname__}: {e}")
    if hasattr(type(value), "__tvm_ffi_float__"):
        return value
    return _ConvertError(f"expected float, got {_tc_describe_value_type(value)}")


cdef object _tc_convert_bool(object _conv, object value):
    """bool accepts: bool, int, Integral (mirrors C++ bool <- bool, int).

    Converts int/Integral to bool.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    if isinstance(value, Integral):
        return bool(value)
    return _ConvertError(f"expected bool, got {_tc_describe_value_type(value)}")


cdef object _tc_convert_str(object _conv, object value):
    """str accepts: str only."""
    if isinstance(value, str):
        return value
    return _ConvertError(f"expected str, got {_tc_describe_value_type(value)}")


cdef object _tc_convert_bytes(object _conv, object value):
    """bytes accepts: bytes, bytearray. Converts bytearray to bytes."""
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    return _ConvertError(f"expected bytes, got {_tc_describe_value_type(value)}")


cdef object _tc_convert_none(object _conv, object value):
    """None accepts: None only."""
    if value is None:
        return None
    return _ConvertError(f"expected None, got {_tc_describe_value_type(value)}")


cdef object _tc_convert_device(object _conv, object value):
    """Device accepts: Device, __dlpack_device__ protocol (without __dlpack__).

    Matches runtime precedence: if a class has both ``__dlpack__`` and
    ``__dlpack_device__``, the runtime routes it as a Tensor, not Device.
    """
    if isinstance(value, _CLASS_DEVICE):
        return value
    cdef object vtype = type(value)
    if hasattr(vtype, "__dlpack_device__") and not hasattr(vtype, "__dlpack__"):
        return value
    return _ConvertError(f"expected Device, got {_tc_describe_value_type(value)}")


cdef object _tc_convert_dtype(object _conv, object value):
    """dtype accepts: DataType, dtype wrapper, str, torch.dtype, numpy.dtype,
    __dlpack_data_type__ protocol.

    Converts str to DataType via parsing. Invalid dtype strings return
    _ConvertError instead of raising, so try_convert/try_check_value
    never throw unexpectedly. torch.dtype, numpy.dtype, and
    __dlpack_data_type__ values are accepted as-is (marshal handles them).
    """
    if isinstance(value, DataType):
        return value
    if _CLASS_DTYPE is not None and isinstance(value, _CLASS_DTYPE):
        return value
    if isinstance(value, str):
        try:
            return DataType(value)
        except Exception:
            return _ConvertError(f"expected dtype, got invalid dtype string {value!r}")
    if torch is not None and isinstance(value, torch.dtype):
        return value
    if numpy is not None and isinstance(value, numpy.dtype):
        return value
    if hasattr(type(value), "__dlpack_data_type__"):
        return value
    return _ConvertError(f"expected dtype, got {_tc_describe_value_type(value)}")


cdef object _tc_convert_opaque_ptr(object _conv, object value):
    """ctypes.c_void_p accepts: ctypes.c_void_p, None, __tvm_ffi_opaque_ptr__,
    __cuda_stream__ protocol.

    ``__cuda_stream__`` produces ``kTVMFFIOpaquePtr`` in the marshal path
    (function.pxi ``TVMFFIPyArgSetterCUDAStreamProtocol_``).
    """
    if value is None:
        return value
    if isinstance(value, ctypes.c_void_p):
        return value
    cdef object vtype = type(value)
    if hasattr(vtype, "__tvm_ffi_opaque_ptr__"):
        return value
    if hasattr(vtype, "__cuda_stream__"):
        return value
    return _ConvertError(f"expected ctypes.c_void_p, got {_tc_describe_value_type(value)}")


cdef object _tc_convert_tensor(object _conv, object value):
    """Tensor accepts: Tensor, __dlpack__, __dlpack_c_exchange_api__.

    ``__dlpack_c_exchange_api__`` is gated by the
    ``TVM_FFI_SKIP_DLPACK_C_EXCHANGE_API`` environment variable,
    matching the runtime check in ``TVMFFIPyArgSetterFactory_``.
    """
    cdef int32_t obj_tindex
    if isinstance(value, Tensor):
        return value
    if isinstance(value, CObject):
        obj_tindex = TVMFFIObjectGetTypeIndex((<CObject>value).chandle)
        if _is_object_instance(obj_tindex, kTVMFFITensor):
            return value
    cdef object vtype = type(value)
    if hasattr(vtype, "__dlpack__"):
        return value
    if os.environ.get("TVM_FFI_SKIP_DLPACK_C_EXCHANGE_API", "0") != "1":
        if hasattr(vtype, "__dlpack_c_exchange_api__"):
            return value
    return _ConvertError(f"expected Tensor, got {_tc_describe_value_type(value)}")


cdef object _tc_convert_callable(object _conv, object value):
    """Callable accepts: Function, any callable."""
    if callable(value):
        return value
    return _ConvertError(f"expected Callable, got {_tc_describe_value_type(value)}")


# ---------------------------------------------------------------------------
# Object type converter (cdef)
# ---------------------------------------------------------------------------
cdef int32_t _get_marshal_object_type(object value):
    """Determine the Object type_index the marshal path would produce.

    Returns the type_index if the marshal path would produce an Object
    subtype for *value*, or ``-1`` if the value would be marshaled as a
    non-Object (POD) type.  This mirrors the dispatch order of
    ``TVMFFIPyArgSetterFactory_`` and ``tvm_ffi.convert``.
    """
    # Types that marshal to Object subtypes
    if isinstance(value, str):
        return kTVMFFIStr
    if isinstance(value, (bytes, bytearray)):
        return kTVMFFIBytes
    if isinstance(value, (tuple, list)):
        return kTVMFFIArray
    if isinstance(value, dict):
        return kTVMFFIMap
    if isinstance(value, Exception):
        return kTVMFFIError
    cdef object vtype = type(value)
    if hasattr(vtype, "__dlpack__"):
        return kTVMFFITensor
    if callable(value):
        return kTVMFFIFunction
    # POD types that don't marshal to Objects
    if value is None:
        return -1
    if isinstance(value, (bool, int, float)):
        return -1
    if isinstance(value, (Integral, Real)):
        return -1
    if isinstance(value, ctypes.c_void_p):
        return -1
    if isinstance(value, _CLASS_DEVICE):
        return -1
    if isinstance(value, DataType):
        return -1
    if _CLASS_DTYPE is not None and isinstance(value, _CLASS_DTYPE):
        return -1
    if hasattr(vtype, "__tvm_ffi_opaque_ptr__"):
        return -1
    if hasattr(vtype, "__cuda_stream__"):
        return -1
    if hasattr(vtype, "__tvm_ffi_int__"):
        return -1
    if hasattr(vtype, "__tvm_ffi_float__"):
        return -1
    if hasattr(vtype, "__dlpack_data_type__"):
        return -1
    if hasattr(vtype, "__dlpack_device__"):
        # __dlpack__ already checked above; reaching here means no __dlpack__
        return -1
    if torch is not None and isinstance(value, torch.dtype):
        return -1
    if numpy is not None and isinstance(value, numpy.dtype):
        return -1
    # Everything else → OpaquePyObject (marshal fallback)
    return kTVMFFIOpaquePyObject


cdef object _tc_convert_object(object conv_obj, object value):
    """Convert *value* to an object of *conv.type_index*.

    Accepts CObject directly, values implementing ``__tvm_ffi_object__``
    or ``ObjectConvertible.asobject()`` protocols, and values that the
    marshal path would implicitly convert to compatible Object subtypes
    (e.g. ``Exception`` → ``ffi.Error``, arbitrary → ``OpaquePyObject``).
    """
    cdef _TypeConverter conv = <_TypeConverter>conv_obj
    cdef int32_t target_type_index = conv.type_index
    cdef int32_t obj_tindex
    cdef object obj
    cdef str target_key
    if isinstance(value, CObject):
        obj_tindex = TVMFFIObjectGetTypeIndex((<CObject>value).chandle)
        if _is_object_instance(obj_tindex, target_type_index):
            return value
        return _ConvertError(
            f"expected {_type_index_to_key(target_type_index)}, "
            f"got {_type_index_to_key(obj_tindex)}"
        )
    # __tvm_ffi_object__ protocol
    if hasattr(type(value), "__tvm_ffi_object__"):
        try:
            obj = value.__tvm_ffi_object__()
        except Exception:
            target_key = _type_index_to_key(target_type_index)
            return _ConvertError(
                f"expected {target_key}, "
                f"__tvm_ffi_object__() failed for {_tc_describe_value_type(value)}"
            )
        if isinstance(obj, CObject):
            obj_tindex = TVMFFIObjectGetTypeIndex((<CObject>obj).chandle)
            if _is_object_instance(obj_tindex, target_type_index):
                return obj
        target_key = _type_index_to_key(target_type_index)
        return _ConvertError(
            f"expected {target_key}, "
            f"got {_tc_describe_value_type(obj)} from __tvm_ffi_object__()"
        )
    # Precedence gate: if the value has eligible __tvm_ffi_value__, defer
    # to the fallback wrapper.  In the marshal path, __tvm_ffi_value__
    # (line 843) precedes Exception (846) and ObjectConvertible (849),
    # so these lower-precedence checks must not run first.
    if (hasattr(type(value), "__tvm_ffi_value__") and
            _would_marshal_use_value_protocol(value)):
        return _ConvertError(
            f"expected {_type_index_to_key(target_type_index)}, "
            f"got {_tc_describe_value_type(value)}"
        )
    # ObjectConvertible protocol (lower-precedence than __tvm_ffi_value__)
    if isinstance(value, ObjectConvertible):
        try:
            obj = value.asobject()
        except Exception:
            target_key = _type_index_to_key(target_type_index)
            return _ConvertError(
                f"expected {target_key}, "
                f"asobject() failed for {_tc_describe_value_type(value)}"
            )
        if isinstance(obj, CObject):
            obj_tindex = TVMFFIObjectGetTypeIndex((<CObject>obj).chandle)
            if _is_object_instance(obj_tindex, target_type_index):
                return obj
        target_key = _type_index_to_key(target_type_index)
        return _ConvertError(
            f"expected {target_key}, "
            f"got {_tc_describe_value_type(obj)} from asobject()"
        )
    # Marshal path implicit conversions: values that the marshal path
    # would convert to Object subtypes (e.g. str→String, list→Array,
    # Exception→Error, arbitrary→OpaquePyObject).
    cdef int32_t marshal_tindex = _get_marshal_object_type(value)
    if marshal_tindex >= 0:
        if _is_object_instance(marshal_tindex, target_type_index):
            return value
        target_key = _type_index_to_key(target_type_index)
        return _ConvertError(
            f"expected {target_key}, "
            f"got {_tc_describe_value_type(value)} "
            f"(marshals as {_type_index_to_key(marshal_tindex)})"
        )
    return _ConvertError(
        f"expected {_type_index_to_key(target_type_index)}, "
        f"got {_tc_describe_value_type(value)}"
    )


# ---------------------------------------------------------------------------
# Container dispatch helpers (all cdef — pure C recursion)
# ---------------------------------------------------------------------------
cdef object _dispatch_convert_elems(object value, _TypeConverter elem_conv):
    """Convert all elements via elem_conv. Returns list, original, or _ConvertError.

    True copy-on-first-change: for indexable inputs (list/tuple), no
    output container is allocated until the first changed element.
    On exact match the fast path is allocation-free.
    """
    cdef int i
    cdef int n
    cdef object result_item, item
    cdef list out

    # Fast path for list/tuple — index-based, allocation-free on exact match
    if isinstance(value, (list, tuple)):
        n = len(value)
        out = None
        for i in range(n):
            item = value[i]
            result_item = _type_convert_dispatch_with_fallback(elem_conv, item)
            if isinstance(result_item, _ConvertError):
                return _ConvertError(f"element [{i}]: {result_item.message}")
            if out is not None:
                out.append(result_item)
            elif result_item is not item:
                # First change — retroactively copy prior unchanged items
                out = list(value[:i])
                out.append(result_item)
        return out if out is not None else value

    # Fallback for CObject iterables (Array, List) — must accumulate
    i = 0
    out = []
    cdef bint changed = False
    for item in value:
        result_item = _type_convert_dispatch_with_fallback(elem_conv, item)
        if isinstance(result_item, _ConvertError):
            return _ConvertError(f"element [{i}]: {result_item.message}")
        if result_item is not item:
            changed = True
        out.append(result_item)
        i += 1
    if changed:
        return out
    return value


cdef object _dispatch_convert_mapping(
    object value, _TypeConverter key_conv, _TypeConverter val_conv
):
    """Convert mapping entries. Returns dict, original, or _ConvertError.

    Copy-on-first-change: no output dict is allocated until the first
    changed entry.  On exact match the fast path avoids dict allocation.
    """
    cdef dict out = None
    cdef object ck, cv
    cdef list prior_pairs = None
    items = value.items() if hasattr(value, "items") else ()
    for k, v in items:
        if key_conv is not None:
            ck = _type_convert_dispatch_with_fallback(key_conv, k)
            if isinstance(ck, _ConvertError):
                return _ConvertError(f"key {k!r}: {ck.message}")
        else:
            ck = k
        if val_conv is not None:
            cv = _type_convert_dispatch_with_fallback(val_conv, v)
            if isinstance(cv, _ConvertError):
                return _ConvertError(f"value for key {k!r}: {cv.message}")
        else:
            cv = v
        if out is not None:
            out[ck] = cv
        elif ck is not k or cv is not v:
            # First change — retroactively build dict from tracked pairs
            out = {}
            if prior_pairs is not None:
                for pk, pv in prior_pairs:
                    out[pk] = pv
            out[ck] = cv
        else:
            # Track unchanged pairs for potential retroactive copy
            if prior_pairs is None:
                prior_pairs = []
            prior_pairs.append((k, v))
    return out if out is not None else value


cdef object _dispatch_optional(object conv_obj, object value):
    """Dispatch for Optional[T]: None passthrough or inner dispatch."""
    if value is None:
        return None
    cdef _TypeConverter conv = <_TypeConverter>conv_obj
    return _type_convert_dispatch_with_fallback(<_TypeConverter>conv.subs[0], value)


cdef object _dispatch_array(object conv_obj, object value):
    """Dispatch for Array[T]. Accepts Array or List CObjects (cross-type)."""
    cdef _TypeConverter conv = <_TypeConverter>conv_obj
    cdef int32_t obj_tindex
    if isinstance(value, CObject):
        obj_tindex = TVMFFIObjectGetTypeIndex((<CObject>value).chandle)
        if not (_is_object_instance(obj_tindex, kTVMFFIArray) or
                _is_object_instance(obj_tindex, kTVMFFIList)):
            return _ConvertError(f"expected Array, got {_type_index_to_key(obj_tindex)}")
    elif not isinstance(value, (list, tuple)):
        return _ConvertError(f"expected Array, got {_tc_describe_value_type(value)}")
    if conv.subs is not None:
        return _dispatch_convert_elems(value, <_TypeConverter>conv.subs[0])
    return value


cdef object _dispatch_list(object conv_obj, object value):
    """Dispatch for List[T]. Accepts List or Array CObjects (cross-type)."""
    cdef _TypeConverter conv = <_TypeConverter>conv_obj
    cdef int32_t obj_tindex
    if isinstance(value, CObject):
        obj_tindex = TVMFFIObjectGetTypeIndex((<CObject>value).chandle)
        if not (_is_object_instance(obj_tindex, kTVMFFIList) or
                _is_object_instance(obj_tindex, kTVMFFIArray)):
            return _ConvertError(f"expected List, got {_type_index_to_key(obj_tindex)}")
    elif not isinstance(value, (list, tuple)):
        return _ConvertError(f"expected List, got {_tc_describe_value_type(value)}")
    if conv.subs is not None:
        return _dispatch_convert_elems(value, <_TypeConverter>conv.subs[0])
    return value


cdef object _dispatch_map(object conv_obj, object value):
    """Dispatch for Map[K, V]. Accepts Map or Dict CObjects (cross-type)."""
    cdef _TypeConverter conv = <_TypeConverter>conv_obj
    cdef int32_t obj_tindex
    if isinstance(value, CObject):
        obj_tindex = TVMFFIObjectGetTypeIndex((<CObject>value).chandle)
        if not (_is_object_instance(obj_tindex, kTVMFFIMap) or
                _is_object_instance(obj_tindex, kTVMFFIDict)):
            return _ConvertError(f"expected Map, got {_type_index_to_key(obj_tindex)}")
    elif not isinstance(value, dict):
        return _ConvertError(f"expected Map, got {_tc_describe_value_type(value)}")
    if conv.subs is not None:
        return _dispatch_convert_mapping(
            value,
            <_TypeConverter>conv.subs[0],
            <_TypeConverter>conv.subs[1],
        )
    return value


cdef object _dispatch_dict(object conv_obj, object value):
    """Dispatch for Dict[K, V]. Accepts Dict or Map CObjects (cross-type)."""
    cdef _TypeConverter conv = <_TypeConverter>conv_obj
    cdef int32_t obj_tindex
    if isinstance(value, CObject):
        obj_tindex = TVMFFIObjectGetTypeIndex((<CObject>value).chandle)
        if not (_is_object_instance(obj_tindex, kTVMFFIDict) or
                _is_object_instance(obj_tindex, kTVMFFIMap)):
            return _ConvertError(f"expected Dict, got {_type_index_to_key(obj_tindex)}")
    elif not isinstance(value, dict):
        return _ConvertError(f"expected Dict, got {_tc_describe_value_type(value)}")
    if conv.subs is not None:
        return _dispatch_convert_mapping(
            value,
            <_TypeConverter>conv.subs[0],
            <_TypeConverter>conv.subs[1],
        )
    return value


cdef object _dispatch_union(object conv_obj, object value):
    """Dispatch for Union[T1, T2, ...].

    Tries each alternative with direct dispatch first.  If all fail and
    the value is eligible for ``__tvm_ffi_value__`` fallback, unwraps
    the value **once** and retries all alternatives.  This avoids calling
    ``__tvm_ffi_value__`` N times (once per alternative).
    """
    cdef _TypeConverter conv = <_TypeConverter>conv_obj
    cdef _TypeConverter alt
    # First pass: direct dispatch (no fallback)
    for alt_obj in conv.subs:
        alt = <_TypeConverter>alt_obj
        result = _type_convert_dispatch(alt, value)
        if not isinstance(result, _ConvertError):
            return result
    # Second pass: try __tvm_ffi_value__ fallback once, then retry
    if (hasattr(type(value), "__tvm_ffi_value__") and
            _would_marshal_use_value_protocol(value)):
        try:
            inner = value.__tvm_ffi_value__()
        except Exception:
            inner = value  # keep original on failure
        if inner is not value:
            for alt_obj in conv.subs:
                alt = <_TypeConverter>alt_obj
                result = _type_convert_dispatch_with_fallback(alt, inner)
                if not isinstance(result, _ConvertError):
                    return result
    return _ConvertError(
        f"expected {conv.err_hint}, got {_tc_describe_value_type(value)}"
    )


cdef object _dispatch_tuple(object conv_obj, object value):
    """Dispatch for tuple[T1, T2, ...].

    Accepts Python tuple, list, and CObject Array (since Python
    tuple/list both become ArrayObj in the FFI call path, and C++
    Tuple<Types...>::TryCastFromAnyView accepts kTVMFFIArray).
    """
    cdef _TypeConverter conv = <_TypeConverter>conv_obj
    cdef int32_t obj_tindex
    cdef int n
    cdef int i
    cdef _TypeConverter elem_conv
    cdef object result_item
    cdef list out

    if isinstance(value, CObject):
        obj_tindex = TVMFFIObjectGetTypeIndex((<CObject>value).chandle)
        if not _is_object_instance(obj_tindex, kTVMFFIArray):
            return _ConvertError(f"expected tuple, got {_type_index_to_key(obj_tindex)}")
    elif not isinstance(value, (tuple, list)):
        return _ConvertError(f"expected tuple, got {_tc_describe_value_type(value)}")
    if conv.subs is None:
        return value
    n = len(conv.subs)
    if len(value) != n:
        return _ConvertError(
            f"expected tuple of length {n}, "
            f"got {type(value).__name__} of length {len(value)}"
        )
    # Copy-on-first-change: no list allocated until first changed element.
    out = None
    for i in range(n):
        elem_conv = <_TypeConverter>conv.subs[i]
        result_item = _type_convert_dispatch_with_fallback(elem_conv, value[i])
        if isinstance(result_item, _ConvertError):
            return _ConvertError(f"element [{i}]: {result_item.message}")
        if out is not None:
            out.append(result_item)
        elif result_item is not value[i]:
            out = list(value[:i])
            out.append(result_item)
    if out is not None or not isinstance(value, tuple):
        return tuple(out) if out is not None else tuple(value)
    return value


# ---------------------------------------------------------------------------
# Core dispatch function (single indirect call through function pointer)
# ---------------------------------------------------------------------------
cdef object _type_convert_dispatch(_TypeConverter conv, object value):
    """Dispatch to the converter via C function pointer.

    Single indirect call through ``conv.dispatch`` — no tag comparison
    or branch cascade.
    """
    return conv.dispatch(conv, value)


# ---------------------------------------------------------------------------
# Builder (runs once per TypeSchema at construction time)
# ---------------------------------------------------------------------------
def _build_converter(schema):
    """Build a ``_TypeConverter`` for *schema*.

    Creates a cdef class instance with the appropriate dispatch function
    pointer and pre-resolved sub-converter references. This runs once at
    TypeSchema construction; every subsequent ``convert()``/``check_value()``
    call goes through the C function pointer ``conv.dispatch``.
    """
    cdef _TypeConverter conv = _TypeConverter.__new__(_TypeConverter)
    origin = schema.origin
    args = schema.args
    origin_tindex = schema.origin_type_index

    # Any
    if origin_tindex == kTVMFFIAny or origin == "Any":
        conv.dispatch = _tc_convert_any
        return conv

    # Structural composite types
    if origin == "Optional":
        conv.dispatch = _dispatch_optional
        conv.subs = (<_TypeConverter>(args[0]._converter),)
        return conv
    if origin == "Union":
        conv.dispatch = _dispatch_union
        conv.subs = tuple(<_TypeConverter>(a._converter) for a in args)
        conv.err_hint = " | ".join(repr(a) for a in args)
        return conv

    # Leaf POD types
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

    # Container types ("list" is Python-native origin, mapped to Array)
    if origin in ("Array", "List", "list"):
        conv.dispatch = _dispatch_list if origin == "List" else _dispatch_array
        if len(args) > 0 and args[0].origin != "Any":
            conv.subs = (<_TypeConverter>(args[0]._converter),)
        return conv
    if origin in ("Map", "Dict", "dict"):
        conv.dispatch = _dispatch_dict if origin == "Dict" else _dispatch_map
        if len(args) == 2:
            key_conv = <_TypeConverter>(args[0]._converter) if args[0].origin != "Any" else None
            val_conv = <_TypeConverter>(args[1]._converter) if args[1].origin != "Any" else None
            if key_conv is not None or val_conv is not None:
                conv.subs = (key_conv, val_conv)
        return conv
    if origin == "tuple":
        conv.dispatch = _dispatch_tuple
        if len(args) > 0:
            conv.subs = tuple(<_TypeConverter>(a._converter) for a in args)
        return conv

    # Object types
    if origin == "Object":
        conv.dispatch = _tc_convert_object
        conv.type_index = kTVMFFIObject
        return conv
    if origin_tindex >= kTVMFFIStaticObjectBegin:
        conv.dispatch = _tc_convert_object
        conv.type_index = origin_tindex
        return conv

    # Late resolution: origin may be an object type key registered after
    # TypeSchema construction (cached_property defers build to first use).
    tindex = _object_type_key_to_index(origin)
    if tindex is not None:
        conv.dispatch = _tc_convert_object
        conv.type_index = tindex
        return conv

    raise TypeError(f"unknown TypeSchema origin: {origin!r}")


# ---------------------------------------------------------------------------
# __tvm_ffi_value__ precedence gate
# ---------------------------------------------------------------------------
cdef bint _would_marshal_use_value_protocol(object value):
    """Return True if the marshal path would dispatch *value* to
    ``__tvm_ffi_value__`` (i.e., no higher-precedence check matches).

    This mirrors lines 724-842 of ``TVMFFIPyArgSetterFactory_`` in
    ``function.pxi``.  The ``__tvm_ffi_value__`` setter is at line 843,
    so any check above it takes precedence.

    NOTE: this function must be kept in sync with the marshal dispatch
    order.  If ``TVMFFIPyArgSetterFactory_`` gains new checks above
    ``__tvm_ffi_value__``, add them here too.
    """
    if value is None:
        return False
    # CObject covers most FFI objects; ObjectRValueRef and PyNativeObject
    # are plain Python classes (not CObject subclasses) that have their
    # own marshal entries before __tvm_ffi_value__ (lines 730, 733, 787).
    if isinstance(value, (Tensor, CObject, ObjectRValueRef, PyNativeObject)):
        return False
    cdef object vtype = type(value)
    if hasattr(vtype, "__tvm_ffi_object__"):
        return False
    if os.environ.get("TVM_FFI_SKIP_DLPACK_C_EXCHANGE_API", "0") != "1":
        if hasattr(vtype, "__dlpack_c_exchange_api__"):
            return False
    if hasattr(vtype, "__cuda_stream__"):
        return False
    if hasattr(vtype, "__dlpack__"):
        return False
    if isinstance(value, (bool, Integral)):
        return False
    if isinstance(value, Real):
        return False
    if _CLASS_DTYPE is not None and isinstance(value, _CLASS_DTYPE):
        return False
    if isinstance(value, DataType):
        return False
    if isinstance(value, _CLASS_DEVICE):
        return False
    if isinstance(value, str):
        return False
    if isinstance(value, (bytes, bytearray)):
        return False
    if isinstance(value, (tuple, list)):
        return False
    if isinstance(value, dict):
        return False
    if isinstance(value, ctypes.c_void_p):
        return False
    if hasattr(vtype, "__tvm_ffi_opaque_ptr__"):
        return False
    if callable(value):
        return False
    if torch is not None and isinstance(value, torch.dtype):
        return False
    if numpy is not None and isinstance(value, numpy.dtype):
        return False
    if hasattr(vtype, "__dlpack_data_type__"):
        return False
    if hasattr(vtype, "__dlpack_device__"):
        return False
    if hasattr(vtype, "__tvm_ffi_int__"):
        return False
    if hasattr(vtype, "__tvm_ffi_float__"):
        return False
    return True


# ---------------------------------------------------------------------------
# __tvm_ffi_value__ fallback wrapper
# ---------------------------------------------------------------------------
cdef int _VALUE_PROTOCOL_MAX_DEPTH = 64


cdef object _type_convert_dispatch_with_fallback(
    _TypeConverter conv, object value
):
    """Dispatch with ``__tvm_ffi_value__`` fallback.

    If the primary converter rejects the value and the value implements
    the ``__tvm_ffi_value__`` protocol **and** no higher-precedence
    marshal protocol would intercept the value, call
    ``__tvm_ffi_value__()`` and re-dispatch the result.

    The precedence gate (``_would_marshal_use_value_protocol``) prevents
    false positives: a class with both ``__tvm_ffi_int__`` and
    ``__tvm_ffi_value__`` is dispatched as int by the runtime, so the
    checker must not silently fall through to ``__tvm_ffi_value__``.

    Cycle protection: self-cycles (``__tvm_ffi_value__()`` returns the
    same object) are caught immediately.  Mutual cycles are bounded by
    an explicit depth limit, since Cython ``cdef`` calls do not count
    toward Python's recursion limit and would otherwise overflow the
    C stack (segfault).
    """
    cdef int depth = 0
    while True:
        result = _type_convert_dispatch(conv, value)
        if not (isinstance(result, _ConvertError) and
                hasattr(type(value), "__tvm_ffi_value__") and
                _would_marshal_use_value_protocol(value)):
            return result
        depth += 1
        if depth > _VALUE_PROTOCOL_MAX_DEPTH:
            return _ConvertError(
                f"infinite __tvm_ffi_value__ cycle detected"
            )
        try:
            inner = value.__tvm_ffi_value__()
        except Exception:
            return result
        if inner is value:
            return result  # self-cycle protection
        value = inner


# ---------------------------------------------------------------------------
# Main dispatcher (thin entry point from Python-level TypeSchema methods)
# ---------------------------------------------------------------------------
cdef object _type_convert_impl(object schema, object value):
    """Dispatch to the C-level converter on *schema*."""
    return _type_convert_dispatch_with_fallback(
        <_TypeConverter>schema._converter, value
    )


# ---------------------------------------------------------------------------
# Public API helpers (called from TypeSchema methods in type_info.pxi)
# ---------------------------------------------------------------------------
def _type_schema_check_value(schema, value):
    """Raise TypeError if *value* is not compatible with *schema*."""
    try:
        result = _type_convert_impl(schema, value)
    except RecursionError:
        raise TypeError(
            f"type check failed for {schema!r}: "
            f"infinite __tvm_ffi_value__ cycle detected"
        ) from None
    if isinstance(result, _ConvertError):
        raise TypeError(f"type check failed for {schema!r}: {result.message}")


def _type_schema_try_check_value(schema, value):
    """Return None on success or an error message string on failure.

    Never raises — all exceptions (including lazy converter build
    failures and custom ``__int__``/``__float__`` errors) are caught
    and returned as error message strings.
    """
    try:
        result = _type_convert_impl(schema, value)
    except Exception as e:
        return str(e)
    if isinstance(result, _ConvertError):
        return result.message
    return None


def _type_schema_convert(schema, value):
    """Convert *value* or raise TypeError."""
    try:
        result = _type_convert_impl(schema, value)
    except RecursionError:
        raise TypeError(
            f"type conversion failed for {schema!r}: "
            f"infinite __tvm_ffi_value__ cycle detected"
        ) from None
    if isinstance(result, _ConvertError):
        raise TypeError(f"type conversion failed for {schema!r}: {result.message}")
    return result


def _type_schema_try_convert(schema, value):
    """Return (True, converted) or (False, error_message).

    Never raises — all exceptions (including lazy converter build
    failures and custom ``__int__``/``__float__`` errors) are caught
    and returned as ``(False, error_message)``.
    """
    try:
        result = _type_convert_impl(schema, value)
    except Exception as e:
        return (False, str(e))
    if isinstance(result, _ConvertError):
        return (False, result.message)
    return (True, result)
