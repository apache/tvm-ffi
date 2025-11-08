
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

# helper class for string/bytes handling

cdef inline str _string_obj_get_py_str(obj):
    cdef TVMFFIByteArray* bytes = TVMFFIBytesGetByteArrayPtr((<Object>obj).chandle)
    return bytearray_to_str(bytes)


cdef inline bytes _bytes_obj_get_py_bytes(obj):
    cdef TVMFFIByteArray* bytes = TVMFFIBytesGetByteArrayPtr((<Object>obj).chandle)
    return bytearray_to_bytes(bytes)


from typing import Any


class String(str, PyNativeObject):
    __slots__ = ["_tvm_ffi_cached_object"]
    """UTF-8 string that interoperates with FFI while behaving like ``str``.

    ``String`` is a :class:`str` subclass that can travel across the
    FFI boundary without copying for large payloads. For most Python
    APIs, using a plain ``str`` works seamlessly; the runtime converts
    to and from ``String`` as needed.

    Examples
    --------
    .. code-block:: python

        fecho = tvm_ffi.get_global_func("testing.echo")
        s = tvm_ffi.core.String("hello")
        assert fecho(s) == "hello"
        assert fecho("world") == "world"

    """
    _tvm_ffi_cached_object: Object | None

    def __new__(cls, value: str) -> "String":
        val = str.__new__(cls, value)
        val._tvm_ffi_cached_object = None
        return val

    # pylint: disable=no-self-argument
    def __from_tvm_ffi_object__(cls, obj: Any) -> "String":
        """Construct a ``String`` from an FFI object (internal)."""
        content = _string_obj_get_py_str(obj)
        val = str.__new__(cls, content)
        val._tvm_ffi_cached_object = obj
        return val


class Bytes(bytes, PyNativeObject):
    """Byte buffer that interoperates with FFI while behaving like ``bytes``.

    Like :class:`String`, this class enables zero-copy exchange for
    large data. Most Python code can use ``bytes`` directly; the FFI
    layer constructs :class:`Bytes` as needed.
    """
    _tvm_ffi_cached_object: Object | None

    def __new__(cls, value: bytes) -> "Bytes":
        val = bytes.__new__(cls, value)
        val._tvm_ffi_cached_object = None
        return val

    # pylint: disable=no-self-argument
    def __from_tvm_ffi_object__(cls, obj: Any) -> "Bytes":
        """Construct ``Bytes`` from an FFI object (internal)."""
        content = _bytes_obj_get_py_bytes(obj)
        val = bytes.__new__(cls, content)
        val._tvm_ffi_cached_object = obj
        return val
