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
# error handling for FFI

import types
import re

ERROR_NAME_TO_TYPE = {}
ERROR_TYPE_TO_NAME = {}

_WITH_APPEND_BACKTRACE = None
_TRACEBACK_TO_BACKTRACE_STR = None


cdef class Error(Object):
    """Base class for all FFI errors, usually they are attached to errors

    Note
    ----
    Do not directly raise this object, instead use the `py_error` method
    to convert it to a python error then raise it.
    """

    def __init__(self, kind, message, backtrace):
        cdef ByteArrayArg kind_arg = ByteArrayArg(c_str(kind))
        cdef ByteArrayArg message_arg = ByteArrayArg(c_str(message))
        cdef ByteArrayArg backtrace_arg = ByteArrayArg(c_str(backtrace))
        cdef TVMFFIObjectHandle out
        cdef int ret = TVMFFIErrorCreate(
            kind_arg.cptr(), message_arg.cptr(), backtrace_arg.cptr(), &out
        )
        if ret != 0:
            raise MemoryError("Failed to create error object")
        (<Object>self).chandle = out

    def update_backtrace(self, backtrace):
        """Update the backtrace of the error

        Parameters
        ----------
        backtrace : str
            The backtrace to update.

        Note
        ----
        The backtrace is stored in the reverse order of python traceback.
        Such storage pattern makes it easier to append the backtrace to
        the end when error is propagated. When we translate the backtrace to python traceback,
        we will reverse the order of the lines to make it align with python traceback.
        """
        cdef ByteArrayArg backtrace_arg = ByteArrayArg(c_str(backtrace))
        TVMFFIErrorGetCellPtr(self.chandle).update_backtrace(
            self.chandle, backtrace_arg.cptr(), kTVMFFIBacktraceUpdateModeReplace
        )

    def py_error(self):
        """
        Convert the FFI error to the python error
        """
        error_cls = ERROR_NAME_TO_TYPE.get(self.kind, RuntimeError)
        py_error = error_cls(self.message)
        py_error = _WITH_APPEND_BACKTRACE(py_error, self.backtrace)
        py_error.__tvm_ffi_error__ = self
        return py_error

    @property
    def kind(self):
        return bytearray_to_str(&(TVMFFIErrorGetCellPtr(self.chandle).kind))

    @property
    def message(self):
        return bytearray_to_str(&(TVMFFIErrorGetCellPtr(self.chandle).message))

    @property
    def backtrace(self):
        return bytearray_to_str(&(TVMFFIErrorGetCellPtr(self.chandle).backtrace))


_register_object_by_index(kTVMFFIError, Error)


cdef inline Error move_from_last_error():
    # raise last error
    error = Error.__new__(Error)
    TVMFFIErrorMoveFromRaised(&(<Object>error).chandle)
    return error


cdef inline int raise_existing_error() except -2:
    return -2


cdef inline int set_last_ffi_error(error) except -1:
    """Set the last FFI error"""
    cdef Error ffi_error

    kind = ERROR_TYPE_TO_NAME.get(type(error), "RuntimeError")
    message = error.__str__()
    # NOTE: backtrace storage convention is reverse of python traceback
    py_backtrace = _TRACEBACK_TO_BACKTRACE_STR(error.__traceback__)
    c_backtrace = bytearray_to_str(TVMFFIBacktrace(NULL, 0, NULL, 0))

    # error comes from an exception thrown from C++ side
    if hasattr(error, "__tvm_ffi_error__"):
        # already have stack trace
        ffi_error = error.__tvm_ffi_error__
        # attach the python backtrace together with the C++ backtrace to get full trace
        ffi_error.update_backtrace(py_backtrace + c_backtrace)
        TVMFFIErrorSetRaised(ffi_error.chandle)
    else:
        ffi_error = Error(kind, message, py_backtrace + c_backtrace)
        TVMFFIErrorSetRaised(ffi_error.chandle)


def _convert_to_ffi_error(error):
    """Convert the python error to the FFI error"""
    py_backtrace = _TRACEBACK_TO_BACKTRACE_STR(error.__traceback__)
    if hasattr(error, "__tvm_ffi_error__"):
        error.__tvm_ffi_error__.update_backtrace(py_backtrace)
        return error.__tvm_ffi_error__
    else:
        kind = ERROR_TYPE_TO_NAME.get(type(error), "RuntimeError")
        message = error.__str__()
        return Error(kind, message, py_backtrace)


cdef inline int CHECK_CALL(int ret) except -2:
    """Check the return code of the C API function call"""
    if ret == 0:
        return 0
    # -2 brings exception
    if ret == -2:
        raise raise_existing_error()
    raise move_from_last_error().py_error()
