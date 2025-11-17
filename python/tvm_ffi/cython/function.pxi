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
import ctypes
import threading
import os
from numbers import Integral, Real
from typing import Any, Callable


if os.environ.get("TVM_FFI_BUILD_DOCS", "0") == "0":
    try:
        # optionally import torch and setup torch related utils
        import torch
    except ImportError:
        torch = None

    try:
        # optionally import numpy
        import numpy
    except ImportError:
        numpy = None

    try:
        from cuda.bindings import driver as cuda_driver
    except ImportError:
        cuda_driver = None
else:
    torch = None
    numpy = None
    cuda_driver = None


cdef int _RELEASE_GIL_BY_DEFAULT = int(
  os.environ.get("TVM_FFI_RELEASE_GIL_BY_DEFAULT", "1")
)

cdef inline object make_ret_small_str(TVMFFIAny result):
    """convert small string to return value."""
    cdef TVMFFIByteArray bytes
    bytes = TVMFFISmallBytesGetContentByteArray(&result)
    return bytearray_to_str(&bytes)


cdef inline object make_ret_small_bytes(TVMFFIAny result):
    """convert small bytes to return value."""
    cdef TVMFFIByteArray bytes
    bytes = TVMFFISmallBytesGetContentByteArray(&result)
    return bytearray_to_bytes(&bytes)


cdef inline object make_ret(TVMFFIAny result, const DLPackExchangeAPI* c_ctx_dlpack_api = NULL):
    """convert result to return value."""
    cdef int32_t type_index
    type_index = result.type_index
    if type_index == kTVMFFITensor:
        # specially handle Tensor as it needs a special dltensor field
        return make_tensor_from_any(result, c_ctx_dlpack_api)
    elif type_index == kTVMFFIOpaquePyObject:
        return make_ret_opaque_object(result)
    elif type_index >= kTVMFFIStaticObjectBegin:
        return make_ret_object(result)
    # the following code should be optimized to switch case
    if type_index == kTVMFFINone:
        return None
    elif type_index == kTVMFFIBool:
        return bool(result.v_int64)
    elif type_index == kTVMFFIInt:
        return result.v_int64
    elif type_index == kTVMFFIFloat:
        return result.v_float64
    elif type_index == kTVMFFISmallStr:
        return make_ret_small_str(result)
    elif type_index == kTVMFFISmallBytes:
        return make_ret_small_bytes(result)
    elif type_index == kTVMFFIOpaquePtr:
        return ctypes_handle(result.v_ptr)
    elif type_index == kTVMFFIDataType:
        return make_ret_dtype(result)
    elif type_index == kTVMFFIDevice:
        return make_ret_device(result)
    elif type_index == kTVMFFIDLTensorPtr:
        return make_ret_dltensor(result)
    elif type_index == kTVMFFIObjectRValueRef:
        raise ValueError("Return value cannot be ObjectRValueRef")
    elif type_index == kTVMFFIByteArrayPtr:
        raise ValueError("Return value cannot be ByteArrayPtr")
    elif type_index == kTVMFFIRawStr:
        raise ValueError("Return value cannot be RawStr")
    raise ValueError("Unhandled type index %d" % type_index)


# ----------------------------------------------------------------------------
#  Helper to simplify calling constructor
# ----------------------------------------------------------------------------
cdef inline int ConstructorCall(void* constructor_handle,
                                PyObject* py_arg_tuple,
                                void** handle,
                                TVMFFIPyCallContext* parent_ctx) except -1:
    """Call contructor of a handle function"""
    cdef TVMFFIAny result
    cdef int c_api_ret_code
    # IMPORTANT: caller need to initialize result->type_index to kTVMFFINone
    result.type_index = kTVMFFINone
    result.v_int64 = 0
    TVMFFIPyConstructorCall(
        TVMFFIPyArgSetterFactory_, constructor_handle, py_arg_tuple, &result, &c_api_ret_code,
        parent_ctx
    )
    CHECK_CALL(c_api_ret_code)
    handle[0] = result.v_ptr
    return 0

# ----------------------------------------------------------------------------
#  Implementation of setters using same naming style as TVMFFIPyArgSetterXXX_
# ----------------------------------------------------------------------------
cdef int TVMFFIPyArgSetterTensor_(
    TVMFFIPyArgSetter* handle, TVMFFIPyCallContext* ctx,
    PyObject* arg, TVMFFIAny* out
) except -1:
    if (<Object>arg).chandle != NULL:
        out.type_index = kTVMFFITensor
        out.v_ptr = (<Tensor>arg).chandle
    else:
        out.type_index = kTVMFFIDLTensorPtr
        out.v_ptr = (<Tensor>arg).cdltensor
    return 0


cdef int TVMFFIPyArgSetterObject_(
    TVMFFIPyArgSetter* handle, TVMFFIPyCallContext* ctx,
    PyObject* arg, TVMFFIAny* out
) except -1:
    out.type_index = TVMFFIObjectGetTypeIndex((<Object>arg).chandle)
    out.v_ptr = (<Object>arg).chandle
    return 0


cdef int TVMFFIPyArgSetterDLPackExchangeAPI_(
    TVMFFIPyArgSetter* this, TVMFFIPyCallContext* ctx,
    PyObject* arg, TVMFFIAny* out
) except -1:
    cdef DLManagedTensorVersioned* temp_managed_tensor
    cdef TVMFFIObjectHandle temp_chandle
    cdef void* current_stream = NULL
    cdef const DLPackExchangeAPI* exchange_api = this.c_dlpack_exchange_api

    # Set the exchange API in context
    ctx.c_dlpack_exchange_api = exchange_api

    # Convert PyObject to DLPack using the struct's function pointer
    if exchange_api.managed_tensor_from_py_object_no_sync(arg, &temp_managed_tensor) != 0:
        return -1

    # Query current stream from producer if device is not CPU
    if temp_managed_tensor.dl_tensor.device.device_type != kDLCPU:
        if ctx.device_type == -1 and exchange_api.current_work_stream != NULL:
            # First time seeing a device, query the stream
            if exchange_api.current_work_stream(
                temp_managed_tensor.dl_tensor.device.device_type,
                temp_managed_tensor.dl_tensor.device.device_id,
                &current_stream
            ) == 0:
                ctx.stream = <TVMFFIStreamHandle>current_stream
                ctx.device_type = temp_managed_tensor.dl_tensor.device.device_type
                ctx.device_id = temp_managed_tensor.dl_tensor.device.device_id

    # Convert to TVM Tensor
    if TVMFFITensorFromDLPackVersioned(temp_managed_tensor, 0, 0, &temp_chandle) != 0:
        # recycle the managed tensor to avoid leak
        if temp_managed_tensor.deleter != NULL:
            temp_managed_tensor.deleter(temp_managed_tensor)
        raise BufferError("Failed to convert DLManagedTensorVersioned to ffi.Tensor")

    out.type_index = kTVMFFITensor
    out.v_ptr = temp_chandle
    TVMFFIPyPushTempFFIObject(ctx, temp_chandle)
    return 0


cdef int TorchManagedTensorToPyObjectNoSyncFallback_(
    DLManagedTensorVersioned* dltensor, void** py_obj_out
) except -1:
    # a bit convoluted but ok as a fallback
    cdef TVMFFIObjectHandle temp_chandle
    if TVMFFITensorFromDLPackVersioned(dltensor, 0, 0, &temp_chandle) != 0:
        return -1

    tensor = make_tensor_from_chandle(temp_chandle)
    torch_tensor = torch.from_dlpack(tensor)
    Py_INCREF(torch_tensor)
    py_obj_out[0] = <void*>(<PyObject*>torch_tensor)
    return 0


cdef inline const DLPackExchangeAPI* GetTorchFallbackExchangeAPI() noexcept:
    global _torch_fallback_exchange_api

    _torch_fallback_exchange_api.header.version.major = DLPACK_MAJOR_VERSION
    _torch_fallback_exchange_api.header.version.minor = DLPACK_MINOR_VERSION
    _torch_fallback_exchange_api.header.prev_api = NULL
    _torch_fallback_exchange_api.managed_tensor_allocator = NULL
    _torch_fallback_exchange_api.managed_tensor_from_py_object_no_sync = NULL
    _torch_fallback_exchange_api.managed_tensor_to_py_object_no_sync = (
        TorchManagedTensorToPyObjectNoSyncFallback_
    )
    _torch_fallback_exchange_api.dltensor_from_py_object_no_sync = NULL
    _torch_fallback_exchange_api.current_work_stream = NULL

    return &_torch_fallback_exchange_api

# Static storage for the fallback exchange API
cdef DLPackExchangeAPI _torch_fallback_exchange_api


cdef int TVMFFIPyArgSetterTorchFallback_(
    TVMFFIPyArgSetter* handle, TVMFFIPyCallContext* ctx,
    PyObject* py_arg, TVMFFIAny* out
) except -1:
    """Current setter for torch.Tensor, go through python and not as fast as c exporter"""
    # TODO(tqchen): remove this once torch always support fast DLPack importer
    cdef object arg = <object>py_arg
    cdef long long temp_ptr
    is_cuda = arg.is_cuda
    arg = from_dlpack(torch.utils.dlpack.to_dlpack(arg))
    out.type_index = kTVMFFITensor
    out.v_ptr = (<Tensor>arg).chandle
    temp_dltensor = TVMFFITensorGetDLTensorPtr((<Tensor>arg).chandle)
    ctx.c_dlpack_exchange_api = GetTorchFallbackExchangeAPI()
    # record the stream and device for torch context
    if is_cuda and ctx.device_type == -1:
        ctx.device_type = temp_dltensor.device.device_type
        ctx.device_id = temp_dltensor.device.device_id
        # This is an API that dynamo and other uses to get the raw stream from torch
        temp_ptr = torch._C._cuda_getCurrentRawStream(temp_dltensor.device.device_id)
        ctx.stream = <TVMFFIStreamHandle>temp_ptr
    # push to temp and clear the handle
    TVMFFIPyPushTempPyObject(ctx, <PyObject*>arg)
    return 0


cdef int TVMFFIPyArgSetterDLPack_(
    TVMFFIPyArgSetter* handle, TVMFFIPyCallContext* ctx,
    PyObject* py_arg, TVMFFIAny* out
) except -1:
    """Setter for __dlpack__ mechanism through python, not as fast as c exporter"""
    cdef TVMFFIObjectHandle temp_chandle
    cdef object arg = <object>py_arg
    _from_dlpack_universal(arg, 0, 0, &temp_chandle)
    out.type_index = kTVMFFITensor
    out.v_ptr = temp_chandle
    # record the stream from the source framework context when possible
    temp_dltensor = TVMFFITensorGetDLTensorPtr(temp_chandle)
    if (temp_dltensor.device.device_type != kDLCPU and ctx.device_type != -1):
        # __tvm_ffi_env_stream__ returns the expected stream that should be set
        # through TVMFFIEnvSetStream when calling a TVM FFI function
        if hasattr(arg, "__tvm_ffi_env_stream__"):
            # Ideally projects should directly setup their stream context API
            # write through by also calling TVMFFIEnvSetStream
            # so we do not need this protocol to do exchange
            ctx.device_type = temp_dltensor.device.device_type
            ctx.device_id = temp_dltensor.device.device_id
            temp_ptr= arg.__tvm_ffi_env_stream__()
            ctx.stream = <TVMFFIStreamHandle>temp_ptr
    TVMFFIPyPushTempFFIObject(ctx, temp_chandle)
    return 0


cdef int TVMFFIPyArgSetterIntegral_(
    TVMFFIPyArgSetter* handle, TVMFFIPyCallContext* ctx,
    PyObject* py_arg, TVMFFIAny* out
) except -1:
    """Setter for Integral"""
    cdef object arg = <object>py_arg
    out.type_index = kTVMFFIInt
    # keep it in cython so it will also check for fallback cases
    # where the arg is not exactly the int class
    out.v_int64 = <long long>arg
    return 0


cdef int TVMFFIPyArgSetterReal_(
    TVMFFIPyArgSetter* handle, TVMFFIPyCallContext* ctx,
    PyObject* py_arg, TVMFFIAny* out
) except -1:
    """Setter for Real"""
    cdef object arg = <object>py_arg
    out.type_index = kTVMFFIFloat
    # keep it in cython so it will also check for fallback cases
    # where the arg is not exactly the float class
    out.v_float64 = <double>arg
    return 0


cdef int TVMFFIPyArgSetterFFIObjectProtocol_(
    TVMFFIPyArgSetter* handle, TVMFFIPyCallContext* ctx,
    PyObject* py_arg, TVMFFIAny* out
) except -1:
    """Setter for objects that implement the `__tvm_ffi_object__` protocol."""
    cdef object arg = <object>py_arg
    cdef TVMFFIObjectHandle temp_chandle
    cdef Object obj = arg.__tvm_ffi_object__()
    cdef long ref_count = Py_REFCNT(obj)
    temp_chandle = obj.chandle
    out.type_index = TVMFFIObjectGetTypeIndex(temp_chandle)
    out.v_ptr = temp_chandle
    if ref_count == 1:
        # keep alive the tensor, since the tensor is temporary
        # and will be freed after we exit here
        TVMFFIObjectIncRef(temp_chandle)
        TVMFFIPyPushTempFFIObject(ctx, temp_chandle)
    return 0


cdef int TVMFFIPyArgSetterCUDAStreamProtocol_(
    TVMFFIPyArgSetter* handle, TVMFFIPyCallContext* ctx,
    PyObject* py_arg, TVMFFIAny* out
) except -1:
    """Setter for cuda stream protocol"""
    cdef object arg = <object>py_arg
    # cuda stream is a subclass of str, so this check occur before str
    cdef tuple cu_stream_tuple = arg.__cuda_stream__()
    cdef long long long_ptr = <long long>cu_stream_tuple[1]
    out.type_index = kTVMFFIOpaquePtr
    out.v_ptr = <void*>long_ptr
    return 0


cdef int TVMFFIPyArgSetterCUDADriverStreamFallback_(
    TVMFFIPyArgSetter* handle, TVMFFIPyCallContext* ctx,
    PyObject* py_arg, TVMFFIAny* out
) except -1:
    """Setter for cuda.bindings.driver.CUstream as a fallback without __cuda_stream__ protocol"""
    cdef object arg = <object>py_arg
    # call driver stream
    cdef long long long_ptr = int(arg)
    out.type_index = kTVMFFIOpaquePtr
    out.v_ptr = <void*>long_ptr
    return 0


cdef int TVMFFIPyArgSetterDType_(
    TVMFFIPyArgSetter* handle, TVMFFIPyCallContext* ctx,
    PyObject* py_arg, TVMFFIAny* out
) except -1:
    """Setter for dtype"""
    cdef object arg = <object>py_arg
    # dtype is a subclass of str, so this check occur before str
    arg = arg._tvm_ffi_dtype
    out.type_index = kTVMFFIDataType
    out.v_dtype = (<DataType>arg).cdtype
    return 0


cdef int TVMFFIPyArgSetterDevice_(
    TVMFFIPyArgSetter* handle, TVMFFIPyCallContext* ctx,
    PyObject* py_arg, TVMFFIAny* out
) except -1:
    """Setter for device"""
    cdef object arg = <object>py_arg
    out.type_index = kTVMFFIDevice
    out.v_device = (<Device>arg).cdevice
    return 0

cdef int TVMFFIPyArgSetterDLPackDeviceProtocol_(
    TVMFFIPyArgSetter* handle, TVMFFIPyCallContext* ctx,
    PyObject* py_arg, TVMFFIAny* out
) except -1:
    """Setter for dlpack device protocol"""
    cdef object arg = <object>py_arg
    cdef tuple dlpack_device = arg.__dlpack_device__()
    out.type_index = kTVMFFIDevice
    out.v_device = TVMFFIDLDeviceFromIntPair(
        <int32_t>dlpack_device[0],
        <int32_t>dlpack_device[1]
    )
    return 0

cdef int TVMFFIPyArgSetterStr_(
    TVMFFIPyArgSetter* handle, TVMFFIPyCallContext* ctx,
    PyObject* py_arg, TVMFFIAny* out
) except -1:
    """Setter for str"""
    cdef object arg = <object>py_arg
    cdef bytes tstr = arg.encode("utf-8")
    cdef char* data
    cdef Py_ssize_t size
    cdef TVMFFIByteArray cdata

    PyBytes_AsStringAndSize(tstr, &data, &size)
    cdata.data = data
    cdata.size = size
    CHECK_CALL(TVMFFIStringFromByteArray(&cdata, out))
    if out.type_index >= kTVMFFIStaticObjectBegin:
        TVMFFIPyPushTempFFIObject(ctx, out.v_ptr)
    return 0


cdef int TVMFFIPyArgSetterPyNativeObjectStr_(
    TVMFFIPyArgSetter* handle, TVMFFIPyCallContext* ctx,
    PyObject* py_arg, TVMFFIAny* out
) except -1:
    """Specially handle String as its _tvm_ffi_cached_object may be empty"""
    cdef object arg = <object>py_arg
    # need to check if the arg is a large string returned from ffi
    if arg._tvm_ffi_cached_object is not None:
        arg = arg._tvm_ffi_cached_object
        out.type_index = TVMFFIObjectGetTypeIndex((<Object>arg).chandle)
        out.v_ptr = (<Object>arg).chandle
        return 0
    return TVMFFIPyArgSetterStr_(handle, ctx, py_arg, out)


cdef int TVMFFIPyArgSetterBytes_(
    TVMFFIPyArgSetter* handle, TVMFFIPyCallContext* ctx,
    PyObject* py_arg, TVMFFIAny* out
) except -1:
    """Setter for bytes"""
    cdef object arg = <object>py_arg

    if isinstance(arg, bytearray):
        arg = bytes(arg)

    cdef char* data
    cdef Py_ssize_t size
    cdef TVMFFIByteArray cdata

    PyBytes_AsStringAndSize(arg, &data, &size)
    cdata.data = data
    cdata.size = size
    CHECK_CALL(TVMFFIBytesFromByteArray(&cdata, out))

    if out.type_index >= kTVMFFIStaticObjectBegin:
        TVMFFIPyPushTempFFIObject(ctx, out.v_ptr)
    return 0


cdef int TVMFFIPyArgSetterPyNativeObjectBytes_(
    TVMFFIPyArgSetter* handle, TVMFFIPyCallContext* ctx,
    PyObject* py_arg, TVMFFIAny* out
) except -1:
    """Specially handle Bytes as its _tvm_ffi_cached_object may be empty"""
    cdef object arg = <object>py_arg
    # need to check if the arg is a large bytes returned from ffi
    if arg._tvm_ffi_cached_object is not None:
        arg = arg._tvm_ffi_cached_object
        out.type_index = TVMFFIObjectGetTypeIndex((<Object>arg).chandle)
        out.v_ptr = (<Object>arg).chandle
        return 0
    return TVMFFIPyArgSetterBytes_(handle, ctx, py_arg, out)


cdef int TVMFFIPyArgSetterPyNativeObjectGeneral_(
    TVMFFIPyArgSetter* handle, TVMFFIPyCallContext* ctx,
    PyObject* py_arg, TVMFFIAny* out
) except -1:
    """Specially handle Object as its _tvm_ffi_cached_object may be empty"""
    cdef object arg = <object>py_arg
    if arg._tvm_ffi_cached_object is None:
        raise ValueError(f"_tvm_ffi_cached_object is None for {type(arg)}")
    assert arg._tvm_ffi_cached_object is not None
    arg = arg._tvm_ffi_cached_object
    out.type_index = TVMFFIObjectGetTypeIndex((<Object>arg).chandle)
    out.v_ptr = (<Object>arg).chandle
    return 0


cdef int TVMFFIPyArgSetterCtypesVoidPtr_(
    TVMFFIPyArgSetter* handle, TVMFFIPyCallContext* ctx,
    PyObject* py_arg, TVMFFIAny* out
) except -1:
    """Setter for ctypes.c_void_p"""
    out.type_index = kTVMFFIOpaquePtr
    out.v_ptr = c_handle(<object>py_arg)
    return 0


cdef int TVMFFIPyArgSetterFFIOpaquePtrCompatible_(
    TVMFFIPyArgSetter* handle, TVMFFIPyCallContext* ctx,
    PyObject* py_arg, TVMFFIAny* out
) except -1:
    """Setter for objects that implement the `__tvm_ffi_opaque_ptr__` protocol."""
    cdef object arg = <object>py_arg
    cdef long long long_ptr = <long long>arg.__tvm_ffi_opaque_ptr__()
    out.type_index = kTVMFFIOpaquePtr
    out.v_ptr = <void*>long_ptr
    return 0


cdef int TVMFFIPyArgSetterObjectRValueRef_(
    TVMFFIPyArgSetter* handle, TVMFFIPyCallContext* ctx,
    PyObject* py_arg, TVMFFIAny* out
) except -1:
    """Setter for ObjectRValueRef"""
    cdef object arg = <object>py_arg
    out.type_index = kTVMFFIObjectRValueRef
    out.v_ptr = &((<Object>(arg.obj)).chandle)
    return 0


cdef int TVMFFIPyArgSetterCallable_(
    TVMFFIPyArgSetter* handle, TVMFFIPyCallContext* ctx,
    PyObject* py_arg, TVMFFIAny* out
) except -1:
    """Setter for Callable"""
    cdef object arg = <object>py_arg
    cdef TVMFFIObjectHandle chandle
    _convert_to_ffi_func_handle(arg, &chandle)
    out.type_index = TVMFFIObjectGetTypeIndex(chandle)
    out.v_ptr = chandle
    TVMFFIPyPushTempFFIObject(ctx, chandle)
    return 0


cdef int TVMFFIPyArgSetterException_(
    TVMFFIPyArgSetter* handle, TVMFFIPyCallContext* ctx,
    PyObject* py_arg, TVMFFIAny* out
) except -1:
    """Setter for Exception"""
    cdef object arg = <object>py_arg
    arg = _convert_to_ffi_error(arg)
    out.type_index = TVMFFIObjectGetTypeIndex((<Object>arg).chandle)
    out.v_ptr = (<Object>arg).chandle
    TVMFFIPyPushTempPyObject(ctx, <PyObject*>arg)
    return 0


cdef int TVMFFIPyArgSetterTuple_(
    TVMFFIPyArgSetter* handle, TVMFFIPyCallContext* ctx,
    PyObject* py_arg, TVMFFIAny* out
) except -1:
    """Setter for Tuple"""
    # recursively construct a new tuple
    cdef TVMFFIObjectHandle chandle
    ConstructorCall(_CONSTRUCTOR_ARRAY.chandle, py_arg, &chandle, ctx)
    out.type_index = TVMFFIObjectGetTypeIndex(chandle)
    out.v_ptr = chandle
    TVMFFIPyPushTempFFIObject(ctx, chandle)
    return 0


cdef int TVMFFIPyArgSetterTupleLike_(
    TVMFFIPyArgSetter* handle, TVMFFIPyCallContext* ctx,
    PyObject* py_arg, TVMFFIAny* out
) except -1:
    """Setter for TupleLike"""
    # recursively construct a new tuple
    cdef tuple tuple_arg = tuple(<object>py_arg)
    cdef TVMFFIObjectHandle chandle
    ConstructorCall(_CONSTRUCTOR_ARRAY.chandle, <PyObject*>tuple_arg, &chandle, ctx)
    out.type_index = TVMFFIObjectGetTypeIndex(chandle)
    out.v_ptr = chandle
    TVMFFIPyPushTempFFIObject(ctx, chandle)
    return 0


cdef int TVMFFIPyArgSetterMap_(
    TVMFFIPyArgSetter* handle, TVMFFIPyCallContext* ctx,
    PyObject* py_arg, TVMFFIAny* out
) except -1:
    """Setter for Map"""
    # recursively construct a new map
    cdef dict dict_arg = <dict>py_arg
    cdef list list_kvs = []
    for k, v in dict_arg.items():
        list_kvs.append(k)
        list_kvs.append(v)
    cdef tuple_arg_kvs = tuple(list_kvs)
    cdef TVMFFIObjectHandle chandle
    ConstructorCall(_CONSTRUCTOR_MAP.chandle, <PyObject*>tuple_arg_kvs, &chandle, ctx)
    out.type_index = TVMFFIObjectGetTypeIndex(chandle)
    out.v_ptr = chandle
    TVMFFIPyPushTempFFIObject(ctx, chandle)
    return 0


cdef int TVMFFIPyArgSetterObjectConvertible_(
    TVMFFIPyArgSetter* handle, TVMFFIPyCallContext* ctx,
    PyObject* py_arg, TVMFFIAny* out
) except -1:
    """Setter for ObjectConvertible"""
    # recursively construct a new map
    cdef object arg = <object>py_arg
    arg = arg.asobject()
    out.type_index = TVMFFIObjectGetTypeIndex((<Object>arg).chandle)
    out.v_ptr = (<Object>arg).chandle
    TVMFFIPyPushTempPyObject(ctx, <PyObject*>arg)


cdef int TVMFFIPyArgSetterFallback_(
    TVMFFIPyArgSetter* handle, TVMFFIPyCallContext* ctx,
    PyObject* py_arg, TVMFFIAny* out
) except -1:
    """Fallback setter for all other types"""
    cdef object arg = <object>py_arg
    cdef TVMFFIObjectHandle chandle
    _convert_to_opaque_object_handle(arg, &chandle)
    out.type_index = kTVMFFIOpaquePyObject
    out.v_ptr = chandle
    TVMFFIPyPushTempFFIObject(ctx, chandle)


cdef int TVMFFIPyArgSetterDTypeFromTorch_(
    TVMFFIPyArgSetter* handle, TVMFFIPyCallContext* ctx,
    PyObject* py_arg, TVMFFIAny* out
) except -1:
    """Setter for torch dtype"""
    cdef py_obj = <object>py_arg
    if py_obj not in TORCH_DTYPE_TO_DL_DATA_TYPE:
        raise ValueError("Unsupported torch dtype: ", py_obj)
    out.type_index = kTVMFFIDataType
    out.v_dtype = TORCH_DTYPE_TO_DL_DATA_TYPE[py_obj]
    return 0

cdef int TVMFFIPyArgSetterDTypeFromNumpy_(
    TVMFFIPyArgSetter* handle, TVMFFIPyCallContext* ctx,
    PyObject* py_arg, TVMFFIAny* out
) except -1:
    """Setter for torch dtype"""
    cdef py_obj = <object>py_arg
    if py_obj not in NUMPY_DTYPE_TO_DL_DATA_TYPE:
        raise ValueError("Unsupported numpy or ml_dtypes dtype: ", py_obj)
    out.type_index = kTVMFFIDataType
    out.v_dtype = NUMPY_DTYPE_TO_DL_DATA_TYPE[py_obj]
    return 0


cdef int TVMFFIPyArgSetterDLPackDataTypeProtocol_(
    TVMFFIPyArgSetter* handle, TVMFFIPyCallContext* ctx,
    PyObject* py_arg, TVMFFIAny* out
) except -1:
    """Setter for dtype protocol"""
    cdef object arg = <object>py_arg
    cdef tuple dltype_data_type = arg.__dlpack_data_type__()
    out.type_index = kTVMFFIDataType
    out.v_dtype.code = <long long>dltype_data_type[0]
    out.v_dtype.bits = <long long>dltype_data_type[1]
    out.v_dtype.lanes = <long long>dltype_data_type[2]
    return 0


cdef int TVMFFIPyArgSetterIntProtocol_(
    TVMFFIPyArgSetter* handle, TVMFFIPyCallContext* ctx,
    PyObject* py_arg, TVMFFIAny* out
) except -1:
    """Setter for class with __tvm_ffi_int__() method"""
    cdef object arg = <object>py_arg
    out.type_index = kTVMFFIInt
    out.v_int64 = <long long>(arg.__tvm_ffi_int__())
    return 0


cdef int TVMFFIPyArgSetterFloatProtocol_(
    TVMFFIPyArgSetter* handle, TVMFFIPyCallContext* ctx,
    PyObject* py_arg, TVMFFIAny* out
) except -1:
    """Setter for class with __tvm_ffi_float__() method"""
    cdef object arg = <object>py_arg
    out.type_index = kTVMFFIFloat
    out.v_float64 = <double>(arg.__tvm_ffi_float__())
    return 0


cdef _DISPATCH_TYPE_KEEP_ALIVE = set()
cdef _DISPATCH_TYPE_KEEP_ALIVE_LOCK = threading.Lock()

cdef int TVMFFIPyArgSetterFactory_(PyObject* value, TVMFFIPyArgSetter* out) except -1:
    """
    Factory function that creates an argument setter for a given Python argument type.
    """
    # NOTE: the order of checks matter here
    # becase each argument may satisfy multiple checks
    # priortize native types over external types
    cdef object arg = <object>value
    cdef long long temp_ptr

    # The C++ dispatcher dispatches the argument passing by TYPE(obj) pointer which
    # is non-owning. This means that there is the following edge case:
    # - type A is registered through dispatcher
    # - type A gets garbage collected (because it is a local type)
    # - type B is created and uses the same memory address as type A
    #
    # Then when we pass in type B, it will mistakenly use the dispatch function for type A
    #
    # To prevent this, we keep alive the types that are registered through dispatcher
    # by adding them to _DISPATCH_TYPE_KEEP_ALIVE
    #
    # NOTE that the total number of types that are registered through dispatcher is expected
    # to be limited in practice so we can afford to keep them alive
    # Lock is used to ensure thread-safety for future thread-free python case
    with _DISPATCH_TYPE_KEEP_ALIVE_LOCK:
        _DISPATCH_TYPE_KEEP_ALIVE.add(type(arg))

    if arg is None:
        out.func = TVMFFIPyArgSetterNone_
        return 0
    if isinstance(arg, Tensor):
        out.func = TVMFFIPyArgSetterTensor_
        return 0
    if isinstance(arg, Object):
        out.func = TVMFFIPyArgSetterObject_
        return 0
    if isinstance(arg, ObjectRValueRef):
        out.func = TVMFFIPyArgSetterObjectRValueRef_
        return 0
    arg_class = type(arg)
    if hasattr(arg_class, "__tvm_ffi_object__"):
        # can directly map to tvm ffi object
        # usually used for solutions that takes subclass of ffi.Object
        # as a member variable
        out.func = TVMFFIPyArgSetterFFIObjectProtocol_
        return 0
    if os.environ.get("TVM_FFI_SKIP_C_DLPACK_EXCHANGE_API", "0") != "1":
        # Check for DLPackExchangeAPI struct (new approach)
        # This is checked on the CLASS, not the instance
        if hasattr(arg_class, "__c_dlpack_exchange_api__"):
            out.func = TVMFFIPyArgSetterDLPackExchangeAPI_
            temp_ptr = arg_class.__c_dlpack_exchange_api__
            out.c_dlpack_exchange_api = <const DLPackExchangeAPI*>(<long long>temp_ptr)
            return 0
    if hasattr(arg_class, "__cuda_stream__"):
        # cuda stream protocol
        out.func = TVMFFIPyArgSetterCUDAStreamProtocol_
        return 0
    if cuda_driver is not None and isinstance(arg, cuda_driver.CUstream):
        # TODO(tqchen): remove this once cuda-python supports __cuda_stream__ protocol
        out.func = TVMFFIPyArgSetterCUDADriverStreamFallback_
        return 0
    if torch is not None and isinstance(arg, torch.Tensor):
        out.func = TVMFFIPyArgSetterTorchFallback_
        return 0
    if hasattr(arg_class, "__dlpack__"):
        out.func = TVMFFIPyArgSetterDLPack_
        return 0
    if isinstance(arg, bool):
        # A python `bool` is a subclass of `int`, so this check
        # must occur before `Integral`.
        out.func = TVMFFIPyArgSetterBool_
        return 0
    if isinstance(arg, Integral):
        # must occur before Real check
        # cannot simply use TVMFFIPyArgSetterInt
        # because Integral may not be exactly the int class
        out.func = TVMFFIPyArgSetterIntegral_
        return 0
    if isinstance(arg, Real):
        # cannot simply use TVMFFIPyArgSetterFloat
        # because Real may not be exactly the float class
        out.func = TVMFFIPyArgSetterReal_
        return 0
    # dtype is a subclass of str, so this check must occur before str
    if isinstance(arg, _CLASS_DTYPE):
        out.func = TVMFFIPyArgSetterDType_
        return 0
    if isinstance(arg, _CLASS_DEVICE):
        out.func = TVMFFIPyArgSetterDevice_
        return 0
    if isinstance(arg, PyNativeObject):
        # check for PyNativeObject
        # this check must happen before str/bytes/tuple
        if isinstance(arg, str):
            out.func = TVMFFIPyArgSetterPyNativeObjectStr_
            return 0
        if isinstance(arg, bytes):
            out.func = TVMFFIPyArgSetterPyNativeObjectBytes_
            return 0
        out.func = TVMFFIPyArgSetterPyNativeObjectGeneral_
        return 0
    if isinstance(arg, str):
        out.func = TVMFFIPyArgSetterStr_
        return 0
    if isinstance(arg, (bytes, bytearray)):
        out.func = TVMFFIPyArgSetterBytes_
        return 0
    if isinstance(arg, tuple):
        out.func = TVMFFIPyArgSetterTuple_
        return 0
    if isinstance(arg, list):
        out.func = TVMFFIPyArgSetterTupleLike_
        return 0
    if isinstance(arg, dict):
        out.func = TVMFFIPyArgSetterMap_
        return 0
    if isinstance(arg, ctypes.c_void_p):
        out.func = TVMFFIPyArgSetterCtypesVoidPtr_
        return 0
    if hasattr(arg_class, "__tvm_ffi_opaque_ptr__"):
        out.func = TVMFFIPyArgSetterFFIOpaquePtrCompatible_
        return 0
    if callable(arg):
        out.func = TVMFFIPyArgSetterCallable_
        return 0
    if torch is not None and isinstance(arg, torch.dtype):
        out.func = TVMFFIPyArgSetterDTypeFromTorch_
        return 0
    if numpy is not None and isinstance(arg, numpy.dtype):
        out.func = TVMFFIPyArgSetterDTypeFromNumpy_
        return 0
    if hasattr(arg_class, "__dlpack_data_type__"):
        # prefer dlpack as it covers all DLDataType struct
        out.func = TVMFFIPyArgSetterDLPackDataTypeProtocol_
        return 0
    if hasattr(arg_class, "__dlpack_device__") and not hasattr(arg_class, "__dlpack__"):
        # if a class have __dlpack_device__ but not __dlpack__
        # then it is a DLPack device protocol
        out.func = TVMFFIPyArgSetterDLPackDeviceProtocol_
        return 0
    if hasattr(arg_class, "__tvm_ffi_int__"):
        out.func = TVMFFIPyArgSetterIntProtocol_
        return 0
    if hasattr(arg_class, "__tvm_ffi_float__"):
        out.func = TVMFFIPyArgSetterFloatProtocol_
        return 0
    if isinstance(arg, Exception):
        out.func = TVMFFIPyArgSetterException_
        return 0
    if isinstance(arg, ObjectConvertible):
        out.func = TVMFFIPyArgSetterObjectConvertible_
        return 0
    # default to opaque object
    out.func = TVMFFIPyArgSetterFallback_
    return 0


# ---------------------------------------------------------------------------------------------
# Implementation of function calling
# ---------------------------------------------------------------------------------------------
cdef class Function(Object):
    """Callable wrapper around a TVM FFI function.

    Instances are obtained by converting Python callables with
    :func:`tvm_ffi.convert`, or by looking up globally-registered FFI
    functions using :func:`tvm_ffi.get_global_func`.

    Examples
    --------
    .. code-block:: python

        @tvm_ffi.register_global_func("my.add")
        def add(a, b):
            return a + b

        f = tvm_ffi.get_global_func("my.add")
        assert isinstance(f, tvm_ffi.Function)
        assert f(1, 2) == 3

    See Also
    --------
    :py:func:`tvm_ffi.register_global_func`
        Register a Python callable as a global FFI function.
    :py:func:`tvm_ffi.get_global_func`
        Look up a previously registered global FFI function by name.
    """
    cdef int c_release_gil
    cdef dict __dict__

    def __cinit__(self) -> None:
        self.c_release_gil = _RELEASE_GIL_BY_DEFAULT

    property release_gil:
        """Whether calls release the Python GIL while executing."""

        def __get__(self) -> bool:
            return self.c_release_gil != 0

        def __set__(self, value: bool) -> None:
            self.c_release_gil = value

    def __call__(self, *args: Any) -> Any:
        """Invoke the wrapped FFI function with ``args``."""
        cdef TVMFFIAny result
        cdef int c_api_ret_code
        cdef const DLPackExchangeAPI* c_ctx_dlpack_api = NULL
        # IMPORTANT: caller need to initialize result->type_index to kTVMFFINone
        result.type_index = kTVMFFINone
        result.v_int64 = 0
        TVMFFIPyFuncCall(
            TVMFFIPyArgSetterFactory_,
            (<Object>self).chandle, <PyObject*>args,
            &result,
            &c_api_ret_code,
            self.release_gil,
            &c_ctx_dlpack_api
        )
        # NOTE: logic is same as check_call
        # directly inline here to simplify the resulting trace
        if c_api_ret_code == 0:
            return make_ret(result, c_ctx_dlpack_api)
        elif c_api_ret_code == -2:
            raise_existing_error()
        raise move_from_last_error().py_error()

    @staticmethod
    def __from_extern_c__(
        c_symbol: int,
        *,
        keep_alive_object: object | None = None
    ) -> Function:
        """Convert a function from extern C address.

        Parameters
        ----------
        c_symbol : int
            Function pointer to the safe call function.
            The function pointer must ignore the first argument,
            which is the function handle.

        keep_alive_object : object
            Optional object to be captured and kept alive.
            Usually this can be the execution engine that JIT-compiled the function
            to ensure we keep the execution environment alive
            as long as the function is alive.

        Returns
        -------
        Function
            The function object.
        """
        cdef TVMFFIObjectHandle chandle
        # must first convert to int64_t
        cdef int64_t c_symbol_as_long_long = c_symbol
        cdef void* safe_call_addr_ptr = <void*>c_symbol_as_long_long
        cdef PyObject* closure_py_obj = <PyObject*>keep_alive_object
        cdef int ret_code
        if keep_alive_object is None:
            ret_code = TVMFFIFunctionCreate(
                NULL, <TVMFFISafeCallType>safe_call_addr_ptr, NULL,
                &chandle
            )
        else:
            # otherwise, we use Python object
            Py_INCREF(keep_alive_object)
            ret_code = TVMFFIFunctionCreate(
                closure_py_obj, <TVMFFISafeCallType>safe_call_addr_ptr, TVMFFIPyObjectDeleter,
                &chandle
            )
            if ret_code != 0:
                # cleanup during error handling
                Py_DECREF(keep_alive_object)

        CHECK_CALL(ret_code)
        func = Function.__new__(Function)
        (<Object>func).chandle = chandle
        return func

    @staticmethod
    def __from_mlir_packed_safe_call__(
        mlir_packed_symbol: int,
        *,
        keep_alive_object: object | None = None
    ) -> Function:
        """Convert a function from MLIR packed safe call function pointer.

        Parameters
        ----------
        mlir_packed_symbol : int
            Function pointer to the MLIR packed call function
            that represents a safe call function.

        keep_alive_object : object
            Optional object to be captured and kept alive.
            Usually this can be the execution engine that JIT-compiled the function
            to ensure we keep the execution environment alive
            as long as the function is alive.

        Returns
        -------
        Function
            The function object.
        """
        cdef TVMFFIObjectHandle chandle
        # must first convert to int64_t
        cdef int64_t c_symbol_as_long_long = mlir_packed_symbol
        cdef void* packed_call_addr_ptr = <void*>c_symbol_as_long_long
        cdef PyObject* keepalive_py_obj
        if keep_alive_object is None:
            keepalive_py_obj = NULL
        else:
            keepalive_py_obj = <PyObject*>keep_alive_object

        cdef void* mlir_packed_safe_call = TVMFFIPyMLIRPackedSafeCallCreate(
            <void (*)(void**) noexcept>packed_call_addr_ptr,
            keepalive_py_obj
        )
        cdef int ret_code
        ret_code = TVMFFIFunctionCreate(
            mlir_packed_safe_call,
            TVMFFIPyMLIRPackedSafeCallInvoke,
            TVMFFIPyMLIRPackedSafeCallDeleter,
            &chandle
        )
        if ret_code != 0:
            # cleanup during error handling
            TVMFFIPyMLIRPackedSafeCallDeleter(mlir_packed_safe_call)
        CHECK_CALL(ret_code)
        func = Function.__new__(Function)
        (<Object>func).chandle = chandle
        return func


def _register_global_func(name: str, pyfunc: Callable[..., Any] | Function, override: bool) -> Function:
    cdef TVMFFIObjectHandle chandle
    cdef int c_api_ret_code
    cdef int ioverride = override
    cdef ByteArrayArg name_arg = ByteArrayArg(c_str(name))

    if not isinstance(pyfunc, Function):
        pyfunc = _convert_to_ffi_func(pyfunc)

    CHECK_CALL(TVMFFIFunctionSetGlobal(name_arg.cptr(), (<Object>pyfunc).chandle, ioverride))
    return pyfunc


def _get_global_func(name: str, allow_missing: bool):
    cdef TVMFFIObjectHandle chandle
    cdef ByteArrayArg name_arg = ByteArrayArg(c_str(name))

    CHECK_CALL(TVMFFIFunctionGetGlobal(name_arg.cptr(), &chandle))
    if chandle != NULL:
        ret = Function.__new__(Function)
        (<Object>ret).chandle = chandle
        return ret

    if allow_missing:
        return None

    raise ValueError("Cannot find global function %s" % name)


cdef int tvm_ffi_callback(void* context,
                          const TVMFFIAny* packed_args,
                          int32_t num_args,
                          TVMFFIAny* result) noexcept with gil:
    cdef list pyargs
    cdef TVMFFIAny temp_result
    cdef int c_api_ret_code
    cdef object local_pyfunc = <object>(context)
    pyargs = []

    for i in range(num_args):
        CHECK_CALL(TVMFFIAnyViewToOwnedAny(&packed_args[i], &temp_result))
        pyargs.append(make_ret(temp_result))

    try:
        rv = local_pyfunc(*pyargs)
        TVMFFIPyPyObjectToFFIAny(
            TVMFFIPyArgSetterFactory_,
            <PyObject*>rv,
            result,
            &c_api_ret_code
        )
        return c_api_ret_code
    except Exception as err:
        set_last_ffi_error(err)
        return -1


cdef inline int _convert_to_ffi_func_handle(
    object pyfunc, TVMFFIObjectHandle* out_handle
) except -1:
    """Convert a python function to TVM FFI function handle"""
    Py_INCREF(pyfunc)
    CHECK_CALL(TVMFFIFunctionCreate(
        <void*>(pyfunc),
        tvm_ffi_callback,
        TVMFFIPyObjectDeleter,
        out_handle))
    return 0


def _convert_to_ffi_func(object pyfunc: Callable[..., Any]) -> Function:
    """Convert a python function to TVM FFI function"""
    cdef TVMFFIObjectHandle chandle
    _convert_to_ffi_func_handle(pyfunc, &chandle)
    ret = Function.__new__(Function)
    (<Object>ret).chandle = chandle
    return ret


cdef inline int _convert_to_opaque_object_handle(
    object pyobject, TVMFFIObjectHandle* out_handle
) except -1:
    """Convert a python object to TVM FFI opaque object handle"""
    Py_INCREF(pyobject)
    CHECK_CALL(TVMFFIObjectCreateOpaque(
        <void*>(pyobject),
        kTVMFFIOpaquePyObject,
        TVMFFIPyObjectDeleter,
        out_handle))
    return 0


def _convert_to_opaque_object(object pyobject: Any) -> OpaquePyObject:
    """Convert a python object to TVM FFI opaque object"""
    cdef TVMFFIObjectHandle chandle
    _convert_to_opaque_object_handle(pyobject, &chandle)
    ret = OpaquePyObject.__new__(OpaquePyObject)
    (<Object>ret).chandle = chandle
    return ret


def _print_debug_info() -> None:
    """Get the size of the dispatch map"""
    cdef size_t size = TVMFFIPyGetDispatchMapSize()
    print(f"TVMFFIPyGetDispatchMapSize: {size}")


cdef Function _OBJECT_FROM_JSON_GRAPH_STR = _get_global_func("ffi.FromJSONGraphString", True)
cdef Function _OBJECT_TO_JSON_GRAPH_STR = _get_global_func("ffi.ToJSONGraphString", True)
cdef Function _CONSTRUCTOR_ARRAY = _get_global_func("ffi.Array", True)
cdef Function _CONSTRUCTOR_MAP = _get_global_func("ffi.Map", True)
