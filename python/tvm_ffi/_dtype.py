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
"""Lightweight ``dtype`` wrapper for TVM FFI."""

# pylint: disable=invalid-name
from __future__ import annotations

from enum import IntEnum
from typing import Any, ClassVar

from . import core


class DataTypeCode(IntEnum):
    """DLDataTypeCode code in DLTensor."""

    INT = 0
    UINT = 1
    FLOAT = 2
    HANDLE = 3
    BFLOAT = 4
    Float8E3M4 = 7
    Float8E4M3 = 8
    Float8E4M3B11FNUZ = 9
    Float8E4M3FN = 10
    Float8E4M3FNUZ = 11
    Float8E5M2 = 12
    Float8E5M2FNUZ = 13
    Float8E8M0FNU = 14
    Float6E2M3FN = 15
    Float6E3M2FN = 16
    Float4E2M1FN = 17


class dtype(str):
    """Lightweight ``dtype`` in TVM FFI.

    ``dtype`` behaves like a Python ``str`` but also carries an internal FFI
    representation. You can construct it from strings, NumPy/ML dtypes, or
    via :py:meth:`from_dlpack_data_type`.

    Parameters
    ----------
    dtype_str
        The string representation of the dtype.

    Examples
    --------
    .. code-block:: python

        import tvm_ffi

        # Create from string
        f32 = tvm_ffi.dtype("float32")
        assert f32.bits == 32
        assert f32.itemsize == 4

        # Adjust lanes to create vector types
        v4f32 = f32.with_lanes(4)
        assert v4f32 == "float32x4"

        # Round-trip from a DLPack (code, bits, lanes) triple
        f16 = tvm_ffi.dtype.from_dlpack_data_type((2, 16, 1))
        assert f16 == "float16"

    Note
    ----
    This class subclasses str so it can be directly passed
    into other array api's dtype arguments.

    """

    __slots__ = ["_tvm_ffi_dtype"]
    _tvm_ffi_dtype: core.DataType

    _NUMPY_DTYPE_TO_STR: ClassVar[dict[Any, str]] = {}

    def __new__(cls, content: Any) -> dtype:
        content = str(content)
        val = str.__new__(cls, content)
        val._tvm_ffi_dtype = core.DataType(content)
        return val

    @staticmethod
    def from_dlpack_data_type(dltype_data_type: tuple[int, int, int]) -> dtype:
        """Create a dtype from a DLPack data type tuple.

        Parameters
        ----------
        dltype_data_type
            The DLPack data type tuple ``(type_code, bits, lanes)``.

        Returns
        -------
        dtype
            The created dtype.

        Examples
        --------
        .. code-block:: python

            import tvm_ffi

            # Create float16 and int8 directly from DLPack triples
            f16 = tvm_ffi.dtype.from_dlpack_data_type((2, 16, 1))
            i8 = tvm_ffi.dtype.from_dlpack_data_type((0, 8, 1))
            assert f16 == "float16"
            assert i8 == "int8"

        See Also
        --------
        :py:class:`tvm_ffi.dtype`
            User-facing dtype wrapper.
        :py:meth:`tvm_ffi.dtype.with_lanes`
            Create vector dtypes from a scalar base.

        """
        cdtype = core._create_cdtype_from_tuple(
            core.DataType,
            dltype_data_type[0],
            dltype_data_type[1],
            dltype_data_type[2],
        )
        val = str.__new__(dtype, str(cdtype))
        val._tvm_ffi_dtype = cdtype
        return val

    def __repr__(self) -> str:
        return f"dtype('{self}')"

    def with_lanes(self, lanes: int) -> dtype:
        """Create a new dtype with the given number of lanes.

        Parameters
        ----------
        lanes
            The number of lanes for the resulting vector type.

        Returns
        -------
        dtype
            The new dtype with the given number of lanes.

        Examples
        --------
        .. code-block:: python

            import tvm_ffi

            f32 = tvm_ffi.dtype("float32")
            v4f32 = f32.with_lanes(4)
            assert v4f32 == "float32x4"
            assert v4f32.bits == f32.bits and v4f32.lanes == 4

        See Also
        --------
        :py:meth:`tvm_ffi.dtype.from_dlpack_data_type`
            Construct from a DLPack ``(code, bits, lanes)`` triple.

        """
        cdtype = core._create_cdtype_from_tuple(
            core.DataType,
            self._tvm_ffi_dtype.type_code,
            self._tvm_ffi_dtype.bits,
            lanes,
        )
        val = str.__new__(dtype, str(cdtype))
        val._tvm_ffi_dtype = cdtype
        return val

    @property
    def itemsize(self) -> int:
        """Size of one element in bytes.

        The size is computed as ``bits * lanes // 8``. When the number of
        lanes is greater than 1, the ``itemsize`` represents the byte size
        of the vector element.

        Examples
        --------
        .. code-block:: python

            import tvm_ffi

            assert tvm_ffi.dtype("float32").itemsize == 4
            assert tvm_ffi.dtype("float32").with_lanes(4).itemsize == 16

        See Also
        --------
        :py:attr:`tvm_ffi.dtype.bits`
            Bit width of the scalar base type.
        :py:attr:`tvm_ffi.dtype.lanes`
            Number of lanes for vector types.
        :py:meth:`tvm_ffi.dtype.with_lanes`
            Create a vector dtype from a scalar base.

        """
        return self._tvm_ffi_dtype.itemsize

    @property
    def type_code(self) -> int:
        """Integer DLDataTypeCode of the scalar base type.

        Examples
        --------
        .. code-block:: python

            import tvm_ffi

            f32 = tvm_ffi.dtype("float32")
            # The type code is an integer following DLPack conventions
            assert isinstance(f32.type_code, int)
            # Consistent with constructing from an explicit (code, bits, lanes)
            assert f32.type_code == tvm_ffi.dtype.from_dlpack_data_type((2, 32, 1)).type_code

        See Also
        --------
        :py:meth:`tvm_ffi.dtype.from_dlpack_data_type`
            Construct a dtype from a DLPack ``(code, bits, lanes)`` triple.

        """
        return self._tvm_ffi_dtype.type_code

    @property
    def bits(self) -> int:
        """Number of bits of the scalar base type.

        Examples
        --------
        .. code-block:: python

            import tvm_ffi

            assert tvm_ffi.dtype("int8").bits == 8
            v4f32 = tvm_ffi.dtype("float32").with_lanes(4)
            assert v4f32.bits == 32  # per-lane bit width

        See Also
        --------
        :py:attr:`tvm_ffi.dtype.itemsize`
            Byte size accounting for lanes.
        :py:attr:`tvm_ffi.dtype.lanes`
            Number of lanes for vector types.

        """
        return self._tvm_ffi_dtype.bits

    @property
    def lanes(self) -> int:
        """Number of lanes (for vector types).

        Returns ``1`` for scalar dtypes and the lane count for vector dtypes
        created via :py:meth:`tvm_ffi.dtype.with_lanes`.

        Examples
        --------
        .. code-block:: python

            import tvm_ffi

            assert tvm_ffi.dtype("float32").lanes == 1
            assert tvm_ffi.dtype("float32").with_lanes(4).lanes == 4

        See Also
        --------
        :py:meth:`tvm_ffi.dtype.with_lanes`
            Create a vector dtype from a scalar base.
        :py:attr:`tvm_ffi.dtype.itemsize`
            Byte size accounting for lanes.

        """
        return self._tvm_ffi_dtype.lanes


try:
    # this helps to make numpy as optional
    # although almost in all cases we want numpy
    import numpy as np

    dtype._NUMPY_DTYPE_TO_STR[np.dtype(np.bool_)] = "bool"
    dtype._NUMPY_DTYPE_TO_STR[np.dtype(np.int8)] = "int8"
    dtype._NUMPY_DTYPE_TO_STR[np.dtype(np.int16)] = "int16"
    dtype._NUMPY_DTYPE_TO_STR[np.dtype(np.int32)] = "int32"
    dtype._NUMPY_DTYPE_TO_STR[np.dtype(np.int64)] = "int64"
    dtype._NUMPY_DTYPE_TO_STR[np.dtype(np.uint8)] = "uint8"
    dtype._NUMPY_DTYPE_TO_STR[np.dtype(np.uint16)] = "uint16"
    dtype._NUMPY_DTYPE_TO_STR[np.dtype(np.uint32)] = "uint32"
    dtype._NUMPY_DTYPE_TO_STR[np.dtype(np.uint64)] = "uint64"
    dtype._NUMPY_DTYPE_TO_STR[np.dtype(np.float16)] = "float16"
    dtype._NUMPY_DTYPE_TO_STR[np.dtype(np.float32)] = "float32"
    dtype._NUMPY_DTYPE_TO_STR[np.dtype(np.float64)] = "float64"
    if hasattr(np, "float_"):
        dtype._NUMPY_DTYPE_TO_STR[np.dtype(np.float64)] = "float64"
except ImportError:
    pass

try:
    import ml_dtypes

    dtype._NUMPY_DTYPE_TO_STR[np.dtype(ml_dtypes.bfloat16)] = "bfloat16"
    dtype._NUMPY_DTYPE_TO_STR[np.dtype(ml_dtypes.float8_e4m3fn)] = "float8_e4m3fn"
    dtype._NUMPY_DTYPE_TO_STR[np.dtype(ml_dtypes.float8_e5m2)] = "float8_e5m2"
    if hasattr(ml_dtypes, "float4_e2m1fn"):  # ml_dtypes >= 0.5.0
        dtype._NUMPY_DTYPE_TO_STR[np.dtype(ml_dtypes.float4_e2m1fn)] = "float4_e2m1fn"
except ImportError:
    pass

core._set_class_dtype(dtype)
