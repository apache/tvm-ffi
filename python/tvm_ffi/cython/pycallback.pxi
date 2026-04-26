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

# ----------------------------------------------------------------------------
#  Cython-side implementation of the C++ -> Python callback path.
#
#  The C++ side (tvm_ffi_python_helpers.h + TVMFFIPyCallManager::PyCallback)
#  calls into the functions defined here via function pointers registered in
#  callback_arg_dispatch_table_. Each per-type setter takes a borrowed
#  AnyView and produces a new-reference PyObject*.
#
# The caller will grab the thread state before caling into each individual setter.
#
#  This file also hosts `_convert_to_ffi_func`, the Cython entry point that
#  wraps a Python callable as a FFI Function backed by a TVMFFIPyCallback
#  closure (see TVMFFIPyConvertPyCallback in the header).
# ----------------------------------------------------------------------------


cdef int TVMFFIPyCallbackArgSetterTensor_(
    TVMFFIPyCallbackArgSetter* handle,
    const DLPackExchangeAPI* api,
    const TVMFFIAny* arg,
    PyObject** out
) except -1:
    """Callback arg setter for kTVMFFITensor -> ffi.Tensor or torch.Tensor (via DLPack).

    The DLPack branch is inlined rather than delegated to
    ``make_tensor_from_chandle`` so we can pass a borrowed chandle:
    ``TensorObj::ToDLPackVersioned`` incs internally, so the inc/dec pair
    that ``make_tensor_from_chandle`` requires on an owned chandle is pure
    waste here. The non-DLPack branch upgrades to owned and reuses
    ``make_tensor_from_chandle`` for consistency with the RValueRef path.
    """
    cdef TVMFFIObjectHandle chandle = arg.v_ptr
    cdef DLManagedTensorVersioned* dlpack
    cdef void* py_obj

    if api != NULL and api.managed_tensor_to_py_object_no_sync != NULL:
        if TVMFFITensorToDLPackVersioned(chandle, &dlpack) == 0:
            try:
                api.managed_tensor_to_py_object_no_sync(dlpack, &py_obj)
            except Exception:
                dlpack.deleter(dlpack)
                raise
            # py_obj already holds +1 from the DLPack import; transfer to caller.
            out[0] = <PyObject*>py_obj
            return 0
    # Non-DLPack path: upgrade borrowed -> owned, wrap via make_tensor_from_chandle.
    TVMFFIObjectIncRef(chandle)
    obj = make_tensor_from_chandle(chandle, NULL)
    Py_INCREF(obj)
    out[0] = <PyObject*>obj
    return 0


cdef int TVMFFIPyCallbackArgSetterObject_(
    TVMFFIPyCallbackArgSetter* handle,
    const DLPackExchangeAPI* api,
    const TVMFFIAny* arg,
    PyObject** out
) except -1:
    """Callback arg setter for generic static object types (type_index >= kTVMFFIStaticObjectBegin)."""
    cdef TVMFFIObjectHandle chandle = arg.v_ptr
    TVMFFIObjectIncRef(chandle)
    try:
        obj = make_ret_object(arg[0])
        if api != NULL and isinstance(obj, CContainerBase):
            (<CContainerBase>obj)._dlpack_exchange_api = api
    except BaseException:
        TVMFFIObjectDecRef(chandle)
        raise
    Py_INCREF(obj)
    out[0] = <PyObject*>obj
    return 0


cdef int TVMFFIPyCallbackArgSetterOpaquePyObject_(
    TVMFFIPyCallbackArgSetter* handle,
    const DLPackExchangeAPI* api,
    const TVMFFIAny* arg,
    PyObject** out
) except -1:
    """Callback arg setter for kTVMFFIOpaquePyObject -> underlying Python object.

    Inlined equivalent of `make_ret_opaque_object`: reads the cell's Python
    handle directly, skipping the throwaway OpaquePyObject wrapper that
    would otherwise be created just to extract the handle. `arg` is
    borrowed, but the cell stays alive for the callback's duration, so the
    handle is safe to read without inc'ing the chandle.
    """
    cdef PyObject* py_handle = <PyObject*>TVMFFIOpaqueObjectGetCellPtr(arg.v_ptr).handle
    cdef object obj = <object>py_handle
    Py_INCREF(obj)
    out[0] = <PyObject*>obj
    return 0


cdef int TVMFFIPyCallbackArgSetterOpaquePtr_(
    TVMFFIPyCallbackArgSetter* handle,
    const DLPackExchangeAPI* api,
    const TVMFFIAny* arg,
    PyObject** out
) except -1:
    """Callback arg setter for kTVMFFIOpaquePtr -> ctypes.c_void_p."""
    obj = ctypes_handle(arg.v_ptr)
    Py_INCREF(obj)
    out[0] = <PyObject*>obj
    return 0


cdef int TVMFFIPyCallbackArgSetterDataType_(
    TVMFFIPyCallbackArgSetter* handle,
    const DLPackExchangeAPI* api,
    const TVMFFIAny* arg,
    PyObject** out
) except -1:
    """Callback arg setter for kTVMFFIDataType -> DataType."""
    obj = make_ret_dtype(arg[0])
    Py_INCREF(obj)
    out[0] = <PyObject*>obj
    return 0


cdef int TVMFFIPyCallbackArgSetterDevice_(
    TVMFFIPyCallbackArgSetter* handle,
    const DLPackExchangeAPI* api,
    const TVMFFIAny* arg,
    PyObject** out
) except -1:
    """Callback arg setter for kTVMFFIDevice -> Device."""
    obj = make_ret_device(arg[0])
    Py_INCREF(obj)
    out[0] = <PyObject*>obj
    return 0


cdef int TVMFFIPyCallbackArgSetterDLTensorPtr_(
    TVMFFIPyCallbackArgSetter* handle,
    const DLPackExchangeAPI* api,
    const TVMFFIAny* arg,
    PyObject** out
) except -1:
    """Callback arg setter for kTVMFFIDLTensorPtr -> ffi.Tensor (via DLTensor pointer)."""
    obj = make_ret_dltensor(arg[0])
    Py_INCREF(obj)
    out[0] = <PyObject*>obj
    return 0


cdef int TVMFFIPyCallbackArgSetterRawStr_(
    TVMFFIPyCallbackArgSetter* handle,
    const DLPackExchangeAPI* api,
    const TVMFFIAny* arg,
    PyObject** out
) except -1:
    """Callback arg setter for kTVMFFIRawStr -> Python str (UTF-8 decode).

    ``arg.v_c_str`` is a non-owning ``const char*`` pointer into C-side storage
    that remains valid for the duration of the callback invocation.  We copy the
    contents into a Python ``str`` immediately, so there is no dangling-pointer
    concern after the setter returns.
    """
    obj = arg.v_c_str.decode("utf-8")
    Py_INCREF(obj)
    out[0] = <PyObject*>obj
    return 0


cdef int TVMFFIPyCallbackArgSetterByteArrayPtr_(
    TVMFFIPyCallbackArgSetter* handle,
    const DLPackExchangeAPI* api,
    const TVMFFIAny* arg,
    PyObject** out
) except -1:
    """Callback arg setter for kTVMFFIByteArrayPtr -> Python bytes.

    ``arg.v_ptr`` is a non-owning ``TVMFFIByteArray*`` pointer into C-side
    storage valid for the callback's lifetime.  ``bytearray_to_bytes`` copies
    the raw bytes into a new Python ``bytes`` object immediately.
    """
    obj = bytearray_to_bytes(<TVMFFIByteArray*>arg.v_ptr)
    Py_INCREF(obj)
    out[0] = <PyObject*>obj
    return 0


cdef int TVMFFIPyCallbackArgSetterRValueRef_(
    TVMFFIPyCallbackArgSetter* handle,
    const DLPackExchangeAPI* api,
    const TVMFFIAny* arg,
    PyObject** out
) except -1:
    """Callback arg setter for kTVMFFIObjectRValueRef.

    For RValueRef, ``arg.v_ptr`` is an ``Object**`` (address of the caller's
    mutable slot holding the moved chandle), not the chandle itself. We read
    the slot, null it out to prevent a double-move, and wrap WITHOUT inc'ing
    (the move already gave us the +1).
    """
    cdef TVMFFIObjectHandle* slot_ptr = <TVMFFIObjectHandle*>arg.v_ptr
    cdef TVMFFIObjectHandle chandle = slot_ptr[0]
    if chandle == NULL:
        raise ValueError("RValueRef already moved")
    # mark as moved before constructing wrappers (so error paths don't double-move)
    slot_ptr[0] = NULL
    cdef int32_t actual_type_index = TVMFFIObjectGetTypeIndex(chandle)
    cdef TVMFFIAny synthesized
    synthesized.type_index = actual_type_index
    synthesized.zero_padding = 0
    synthesized.v_int64 = 0
    synthesized.v_ptr = chandle
    try:
        if actual_type_index == kTVMFFITensor:
            obj = make_tensor_from_chandle(chandle, api)
        else:
            obj = make_ret_object(synthesized)
            if api != NULL and isinstance(obj, CContainerBase):
                (<CContainerBase>obj)._dlpack_exchange_api = api
    except BaseException:
        # Caller's moved +1 needs to be released on error.
        TVMFFIObjectDecRef(chandle)
        raise
    Py_INCREF(obj)
    out[0] = <PyObject*>obj
    return 0


cdef public int TVMFFICyCallbackArgSetterFactory(int32_t type_index,
                                                 TVMFFIPyCallbackArgSetter* out) except -1:
    """Factory that creates callback arg setters for a given type index.

    POD setters live in tvm_ffi_python_helpers.h (header-inline);
    object-bearing setters are the Cython functions above.
    """
    if type_index >= kTVMFFIStaticObjectBegin:
        if type_index == kTVMFFITensor:
            out.func = TVMFFIPyCallbackArgSetterTensor_
        elif type_index == kTVMFFIOpaquePyObject:
            out.func = TVMFFIPyCallbackArgSetterOpaquePyObject_
        else:
            out.func = TVMFFIPyCallbackArgSetterObject_
        return 0
    if type_index == kTVMFFINone:
        out.func = TVMFFIPyCallbackArgSetterNone_
    elif type_index == kTVMFFIBool:
        out.func = TVMFFIPyCallbackArgSetterBool_
    elif type_index == kTVMFFIInt:
        out.func = TVMFFIPyCallbackArgSetterInt_
    elif type_index == kTVMFFIFloat:
        out.func = TVMFFIPyCallbackArgSetterFloat_
    elif type_index == kTVMFFISmallStr:
        out.func = TVMFFIPyCallbackArgSetterSmallStr_
    elif type_index == kTVMFFISmallBytes:
        out.func = TVMFFIPyCallbackArgSetterSmallBytes_
    elif type_index == kTVMFFIOpaquePtr:
        out.func = TVMFFIPyCallbackArgSetterOpaquePtr_
    elif type_index == kTVMFFIDataType:
        out.func = TVMFFIPyCallbackArgSetterDataType_
    elif type_index == kTVMFFIDevice:
        out.func = TVMFFIPyCallbackArgSetterDevice_
    elif type_index == kTVMFFIDLTensorPtr:
        out.func = TVMFFIPyCallbackArgSetterDLTensorPtr_
    elif type_index == kTVMFFIObjectRValueRef:
        out.func = TVMFFIPyCallbackArgSetterRValueRef_
    elif type_index == kTVMFFIByteArrayPtr:
        out.func = TVMFFIPyCallbackArgSetterByteArrayPtr_
    elif type_index == kTVMFFIRawStr:
        out.func = TVMFFIPyCallbackArgSetterRawStr_
    else:
        raise ValueError("Unhandled type index %d" % type_index)
    return 0


def _convert_to_ffi_func(
    object pyfunc: Callable[..., Any],
    object tensor_cls: object = None,
) -> Function:
    """Convert a python function to a TVM FFI Function.

    Parameters
    ----------
    pyfunc : Callable
        The Python callable to wrap. Incref'd into a TVMFFIPyCallbackClosure.
    tensor_cls : type, optional
        If given, its ``__dlpack_c_exchange_api__`` capsule is threaded into the
        closure and used when constructing tensor return values inside the
        callback.

    Returns
    -------
    Function
        The wrapped FFI function.
    """
    cdef TVMFFIObjectHandle chandle
    cdef const DLPackExchangeAPI* dlpack_api = NULL
    if tensor_cls is not None:
        if not hasattr(tensor_cls, "__dlpack_c_exchange_api__"):
            raise TypeError(
                f"tensor_cls {tensor_cls!r} must expose __dlpack_c_exchange_api__"
            )
        _get_dlpack_exchange_api(
            tensor_cls.__dlpack_c_exchange_api__, &dlpack_api
        )
    CHECK_CALL(TVMFFIPyConvertPyCallback(<PyObject*>pyfunc, dlpack_api, &chandle))
    ret = Function.__new__(Function)
    (<CObject>ret).chandle = chandle
    return ret
