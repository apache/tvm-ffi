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

__dlpack_version__ = (DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION)
_CLASS_TENSOR = None


def _set_class_tensor(cls):
    global _CLASS_TENSOR
    _CLASS_TENSOR = cls


cdef const char* _c_str_dltensor = "dltensor"
cdef const char* _c_str_used_dltensor = "used_dltensor"
cdef const char* _c_str_dltensor_versioned = "dltensor_versioned"
cdef const char* _c_str_used_dltensor_versioned = "used_dltensor_versioned"

cdef void _c_dlpack_deleter(object pycaps):
    cdef DLManagedTensor* dltensor
    if pycapsule.PyCapsule_IsValid(pycaps, _c_str_dltensor):
        dltensor = <DLManagedTensor*>pycapsule.PyCapsule_GetPointer(pycaps, _c_str_dltensor)
        dltensor.deleter(dltensor)

cdef void _c_dlpack_versioned_deleter(object pycaps):
    cdef DLManagedTensorVersioned* dltensor
    if pycapsule.PyCapsule_IsValid(pycaps, _c_str_dltensor_versioned):
        dltensor = <DLManagedTensorVersioned*>pycapsule.PyCapsule_GetPointer(
            pycaps, _c_str_dltensor_versioned)
        dltensor.deleter(dltensor)


cdef inline object _from_dlpack_intptr(
    void* dlpack
):
    cdef TVMFFIObjectHandle chandle
    cdef DLManagedTensor* ptr = <DLManagedTensor*>dlpack
    cdef int c_api_ret_code
    cdef int c_req_alignment = 0
    cdef int c_req_contiguous = 0
    c_api_ret_code = TVMFFITensorFromDLPack(
        ptr, c_req_alignment, c_req_contiguous, &chandle)
    CHECK_CALL(c_api_ret_code)
    return make_tensor_from_chandle(chandle)


cdef inline int _from_dlpack(
    object dltensor, int require_alignment,
    int require_contiguous, TVMFFIObjectHandle* out
) except -1:
    cdef DLManagedTensor* ptr
    cdef int c_api_ret_code
    cdef int c_req_alignment = require_alignment
    cdef int c_req_contiguous = require_contiguous
    if pycapsule.PyCapsule_IsValid(dltensor, _c_str_dltensor):
        ptr = <DLManagedTensor*>pycapsule.PyCapsule_GetPointer(dltensor, _c_str_dltensor)
        c_api_ret_code = TVMFFITensorFromDLPack(
            ptr, c_req_alignment, c_req_contiguous, out)
        CHECK_CALL(c_api_ret_code)
        # set name and destructor to be empty
        pycapsule.PyCapsule_SetDestructor(dltensor, NULL)
        pycapsule.PyCapsule_SetName(dltensor, _c_str_used_dltensor)
        return 0
    raise ValueError("Expect a dltensor field, PyCapsule can only be consumed once")


cdef inline int _from_dlpack_versioned(
    object dltensor, int require_alignment,
    int require_contiguous, TVMFFIObjectHandle* out
) except -1:
    cdef DLManagedTensorVersioned* ptr
    cdef int c_api_ret_code
    cdef int c_req_alignment = require_alignment
    cdef int c_req_contiguous = require_contiguous
    if pycapsule.PyCapsule_IsValid(dltensor, _c_str_dltensor_versioned):
        ptr = <DLManagedTensorVersioned*>pycapsule.PyCapsule_GetPointer(
            dltensor, _c_str_dltensor_versioned)
        c_api_ret_code = TVMFFITensorFromDLPackVersioned(
            ptr, c_req_alignment, c_req_contiguous, out)
        CHECK_CALL(c_api_ret_code)
        # set name and destructor to be empty
        pycapsule.PyCapsule_SetDestructor(dltensor, NULL)
        pycapsule.PyCapsule_SetName(dltensor, _c_str_used_dltensor_versioned)
        return 0
    raise ValueError("Expect a dltensor_versioned field, PyCapsule can only be consumed once")


cdef inline int _from_dlpack_universal(
    object ext_tensor, int require_alignment,
    int require_contiguous, TVMFFIObjectHandle* out
) except -1:
    # as of most frameworks do not yet support v1.1
    # move to false as most frameworks get upgraded.
    cdef int favor_legacy_dlpack = True

    if hasattr(ext_tensor, "__dlpack__"):
        if favor_legacy_dlpack:
            _from_dlpack(
                ext_tensor.__dlpack__(),
                require_alignment,
                require_contiguous,
                out
            )
        else:
            try:
                _from_dlpack_versioned(
                    ext_tensor.__dlpack__(max_version=__dlpack_version__),
                    require_alignment,
                    require_contiguous,
                    out
                )
            except TypeError:
                _from_dlpack(
                    ext_tensor.__dlpack__(),
                    require_alignment,
                    require_contiguous,
                    out
                )
    else:
        if pycapsule.PyCapsule_IsValid(ext_tensor, _c_str_dltensor_versioned):
            _from_dlpack_versioned(
                ext_tensor,
                require_alignment,
                require_contiguous,
                out
            )
        elif pycapsule.PyCapsule_IsValid(ext_tensor, _c_str_dltensor):
            _from_dlpack(
                ext_tensor,
                require_alignment,
                require_contiguous,
                out
            )
        else:
            raise TypeError("Expect from_dlpack to take either a compatible tensor or PyCapsule")


def from_dlpack(ext_tensor, *, require_alignment=0, require_contiguous=False):
    """
    Convert an external tensor to an Tensor.

    Parameters
    ----------
    ext_tensor : object
        The external tensor to convert.

    require_alignment : int
        The minimum required alignment to check for the tensor.

    require_contiguous : bool
        Whether to check for contiguous memory.

    Returns
    -------
    tensor : :py:class:`tvm_ffi.Tensor`
        The converted tensor.
    """
    cdef TVMFFIObjectHandle chandle
    _from_dlpack_universal(ext_tensor, require_alignment, require_contiguous, &chandle)
    return make_tensor_from_chandle(chandle)


# helper class for shape handling
def _shape_obj_get_py_tuple(obj):
    cdef TVMFFIShapeCell* shape = TVMFFIShapeGetCellPtr((<Object>obj).chandle)
    return tuple(shape.data[i] for i in range(shape.size))


def _make_strides_from_shape(tuple shape):
    cdef int64_t expected_stride = 1
    cdef list strides = []
    cdef int64_t ndim = len(shape)
    cdef int64_t reverse_index
    for i in range(ndim):
        reverse_index = ndim - i - 1
        strides.append(expected_stride)
        expected_stride *= shape[reverse_index]
    return tuple(reversed(strides))


cdef class Tensor(Object):
    """Tensor object that represents a managed n-dimensional array.
    """
    cdef DLTensor* cdltensor

    @property
    def shape(self):
        """Shape of this array"""
        return tuple(self.cdltensor.shape[i] for i in range(self.cdltensor.ndim))

    @property
    def strides(self):
        """Strides of this array"""
        if self.cdltensor.strides == NULL:
            return _make_strides_from_shape(self.shape)
        return tuple(self.cdltensor.strides[i] for i in range(self.cdltensor.ndim))

    @property
    def dtype(self):
        """Data type of this array"""
        cdef TVMFFIAny dtype_any
        dtype_any.v_dtype = self.cdltensor.dtype
        return make_ret_dtype(dtype_any)

    @property
    def device(self):
        """Device of this Tensor"""
        cdef TVMFFIAny device_any
        device_any.v_device = self.cdltensor.device
        return make_ret_device(device_any)

    def _to_dlpack(self):
        cdef DLManagedTensor* dltensor
        cdef int c_api_ret_code
        c_api_ret_code = TVMFFITensorToDLPack(self.chandle, &dltensor)
        CHECK_CALL(c_api_ret_code)
        return pycapsule.PyCapsule_New(dltensor, _c_str_dltensor, <PyCapsule_Destructor>_c_dlpack_deleter)

    def _to_dlpack_versioned(self):
        cdef DLManagedTensorVersioned* dltensor
        cdef int c_api_ret_code
        c_api_ret_code = TVMFFITensorToDLPackVersioned(self.chandle, &dltensor)
        CHECK_CALL(c_api_ret_code)
        return pycapsule.PyCapsule_New(
            dltensor, _c_str_dltensor_versioned, <PyCapsule_Destructor>_c_dlpack_versioned_deleter)

    def __dlpack_device__(self):
        cdef int device_type = self.cdltensor.device.device_type
        cdef int device_id = self.cdltensor.device.device_id
        return (device_type, device_id)

    def __dlpack__(self, *, stream=None, max_version=None, dl_device=None, copy=None):
        """Produce a DLPack tensor from this array

        Parameters
        ----------
        stream : Optional[int]
            The stream to use for the DLPack tensor

        max_version : int, optional
            The maximum version of the DLPack tensor to produce

        dl_device : Optional[Tuple[int, int]]
            The device to use for the DLPack tensor

        copy : Optional[bool]
            Whether to copy the data to the new device

        Returns
        -------
        dlpack : DLPack tensor

        Raises
        ------
        BufferError
            Export failed
        """
        if max_version is None:
            # Keep and use the DLPack 0.X implementation
            # Note: from March 2025 onwards (but ideally as late as
            # possible), it's okay to raise BufferError here
            return self._to_dlpack()
        else:
            # We get to produce `DLManagedTensorVersioned` now. Note that
            # our_own_dlpack_version is the max version that the *producer*
            # supports and fills in the `DLManagedTensorVersioned::version`
            # field
            if max_version[0] >= __dlpack_version__[0]:
                if dl_device is not None and dl_device != self.__dlpack_device__():
                    raise BufferError("dl_device of different type not supported")
                if copy is not None and copy:
                    raise BufferError("copy not yet supported")
                return self._to_dlpack_versioned()
            elif max_version[0] < 1:
                return self.__ctypes_handle__to_dlpack()
            else:
                raise BufferError(f"Unsupported max_version {max_version}")


_set_class_tensor(Tensor)
_register_object_by_index(kTVMFFITensor, Tensor)


cdef int _dltensor_test_wrapper_from_pyobject(
    void* obj, DLManagedTensorVersioned** out
) except -1:
    """DLPackExchangeAPI: managed_tensor_from_py_object_no_sync"""
    cdef PyObject* py_obj = <PyObject*>obj
    cdef DLTensorTestWrapper wrapper = <DLTensorTestWrapper>py_obj
    return TVMFFITensorToDLPackVersioned(wrapper.tensor.chandle, out)


cdef int _dltensor_test_wrapper_to_pyobject(
    DLManagedTensorVersioned* tensor, void** out_py_object
) except -1:
    """DLPackExchangeAPI: managed_tensor_to_py_object_no_sync"""
    cdef TVMFFIObjectHandle temp_chandle
    if TVMFFITensorFromDLPackVersioned(tensor, 0, 0, &temp_chandle) != 0:
        return -1
    py_tensor = make_tensor_from_chandle(temp_chandle)
    Py_INCREF(py_tensor)
    out_py_object[0] = <void*>(<PyObject*>py_tensor)
    return 0


cdef int _dltensor_test_wrapper_current_work_stream(
    int device_type, int32_t device_id, void** out_stream
) except -1:
    """DLPackExchangeAPI: current_work_stream"""
    if device_type != kDLCPU:
        out_stream[0] = <void*>TVMFFIEnvGetStream(device_type, device_id)
    return 0


# Module-level static DLPackExchangeAPI for DLTensorTestWrapper
cdef DLPackExchangeAPI _dltensor_test_wrapper_static_api

cdef const DLPackExchangeAPI* _dltensor_test_wrapper_get_exchange_api() noexcept:
    """Get the static DLPackExchangeAPI instance for DLTensorTestWrapper."""
    global _dltensor_test_wrapper_static_api

    # Initialize header using macros from dlpack.h
    _dltensor_test_wrapper_static_api.header.version.major = DLPACK_MAJOR_VERSION
    _dltensor_test_wrapper_static_api.header.version.minor = DLPACK_MINOR_VERSION
    _dltensor_test_wrapper_static_api.header.prev_api = NULL

    # Initialize function pointers
    _dltensor_test_wrapper_static_api.managed_tensor_allocator = NULL
    _dltensor_test_wrapper_static_api.managed_tensor_from_py_object_no_sync = (
        <DLPackManagedTensorFromPyObjectNoSync>_dltensor_test_wrapper_from_pyobject
    )
    _dltensor_test_wrapper_static_api.managed_tensor_to_py_object_no_sync = (
        <DLPackManagedTensorToPyObjectNoSync>_dltensor_test_wrapper_to_pyobject
    )
    _dltensor_test_wrapper_static_api.dltensor_from_py_object_no_sync = NULL
    _dltensor_test_wrapper_static_api.current_work_stream = (
        <DLPackCurrentWorkStream>_dltensor_test_wrapper_current_work_stream
    )

    return &_dltensor_test_wrapper_static_api


def _dltensor_test_wrapper_exchange_api_ptr():
    """Return the pointer to the DLPackExchangeAPI struct as an integer."""
    return <long long>_dltensor_test_wrapper_get_exchange_api()


cdef class DLTensorTestWrapper:
    """Wrapper of a Tensor that exposes DLPack protocol, only for testing purpose.
    """
    __c_dlpack_exchange_api__ = _dltensor_test_wrapper_exchange_api_ptr()

    cdef Tensor tensor
    cdef dict __dict__

    def __init__(self, tensor):
        self.tensor = tensor

    def __tvm_ffi_env_stream__(self):
        cdef TVMFFIStreamHandle stream
        cdef long long stream_as_int
        cdef int c_api_ret_code
        stream = TVMFFIEnvGetStream(
            self.tensor.cdltensor.device.device_type, self.tensor.cdltensor.device.device_id)
        stream_as_int = <long long>stream
        return stream_as_int

    def __dlpack_device__(self):
        return self.tensor.__dlpack_device__()

    def __dlpack__(self, *, **kwargs):
        return self.tensor.__dlpack__(**kwargs)


cdef inline object make_ret_dltensor(TVMFFIAny result):
    cdef DLTensor* dltensor
    dltensor = <DLTensor*>result.v_ptr
    tensor = _CLASS_TENSOR.__new__(_CLASS_TENSOR)
    (<Object>tensor).chandle = NULL
    (<Tensor>tensor).cdltensor = dltensor
    return tensor


cdef inline object make_tensor_from_chandle(
    TVMFFIObjectHandle chandle, const DLPackExchangeAPI* c_ctx_dlpack_api = NULL
):
    cdef object tensor
    cdef void* py_obj
    cdef DLManagedTensorVersioned* dlpack

    if c_ctx_dlpack_api != NULL and c_ctx_dlpack_api.managed_tensor_to_py_object_no_sync != NULL:
        # try convert and import into the environment array if possible
        if TVMFFITensorToDLPackVersioned(chandle, &dlpack) == 0:
            try:
                # note that py_obj already holds an extra reference to the tensor
                # so we need to decref it after the conversion
                c_ctx_dlpack_api.managed_tensor_to_py_object_no_sync(dlpack, &py_obj)
                tensor = <object>(<PyObject*>py_obj)
                Py_DECREF(tensor)
                # decref original handle to prevent leak.
                # note that DLManagedTensor also hold a reference to the tensor
                # so we need to decref the original handle if the conversion is successful
                TVMFFIObjectDecRef(chandle)
                return tensor
            except Exception:
                # call the deleter to free the memory since we will continue to use the chandle
                dlpack.deleter(dlpack)
                pass
    # default return the tensor
    tensor = _CLASS_TENSOR.__new__(_CLASS_TENSOR)
    (<Object>tensor).chandle = chandle
    (<Tensor>tensor).cdltensor = TVMFFITensorGetDLTensorPtr(chandle)
    return tensor


cdef inline object make_tensor_from_any(TVMFFIAny any, const DLPackExchangeAPI* c_ctx_dlpack_api):
    return make_tensor_from_chandle(any.v_ptr, c_ctx_dlpack_api)
