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
"""Typestubs for Cython."""

from __future__ import annotations

import types
from ctypes import c_void_p
from enum import IntEnum
from typing import Any, Callable

# Public module-level variables referenced by Python code
ERROR_NAME_TO_TYPE: dict[str, type]
ERROR_TYPE_TO_NAME: dict[type, str]

_WITH_APPEND_BACKTRACE: Callable[[BaseException, str], BaseException] | None
_TRACEBACK_TO_BACKTRACE_STR: Callable[[types.TracebackType | None], str] | None
# DLPack protocol version (defined in tensor.pxi)
__dlpack_version__: tuple[int, int]

class Object:
    """Base class of all TVM FFI objects.

    This is the root Python type for objects backed by the TVM FFI
    runtime. Each instance references a handle to a C++ runtime
    object. Python subclasses typically correspond to C++ runtime
    types and are registered via ``tvm_ffi.register_object``.

    Notes
    -----
    - Equality of two ``Object`` instances uses underlying handle
      identity unless an overridden implementation is provided on the
      concrete type. Use :py:meth:`same_as` to check whether two
      references point to the same underlying object.
    - Most users interact with subclasses (e.g. :class:`Tensor`,
      :class:`Function`) rather than ``Object`` directly.

    Examples
    --------
    Constructing objects is typically performed by Python wrappers that
    call into registered constructors on the FFI side.

    .. code-block:: python

        # Acquire a testing object constructed through FFI
        obj = tvm_ffi.testing.create_object("testing.TestObjectBase", v_i64=12)
        assert isinstance(obj, tvm_ffi.Object)
        assert obj.same_as(obj)

    """

    def __ctypes_handle__(self) -> Any: ...
    def __chandle__(self) -> int: ...
    def __reduce__(self) -> Any: ...
    def __getstate__(self) -> dict[str, Any]: ...
    def __setstate__(self, state: dict[str, Any]) -> None: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: Any) -> bool: ...
    def __ne__(self, other: Any) -> bool: ...
    def __hash__(self) -> int: ...
    def __init_handle_by_constructor__(self, fconstructor: Any, *args: Any) -> None: ...
    def __ffi_init__(self, *args: Any) -> None:
        """Initialize the instance using the ` __ffi_init__` method registered on C++ side.

        Parameters
        ----------
        args: list of objects
            The arguments to the constructor

        """
    def same_as(self, other: Any) -> bool:
        """Return ``True`` if both references point to the same object.

        This checks identity of the underlying FFI handle rather than
        performing a structural, value-based comparison.

        Parameters
        ----------
        other : Any
            The object to compare against.

        Returns
        -------
        bool

        Examples
        --------
        .. code-block:: python

            x = tvm_ffi.testing.create_object("testing.TestObjectBase")
            y = x
            z = tvm_ffi.testing.create_object("testing.TestObjectBase")
            assert x.same_as(y)
            assert not x.same_as(z)

        """

    def _move(self) -> ObjectRValueRef:
        """Create an rvalue reference that transfers ownership.

        The returned :class:`ObjectRValueRef` indicates move semantics
        to the FFI layer, and is intended for performance-sensitive
        paths that wish to avoid an additional retain/release pair.

        Notes
        -----
        After a successful move, the original object should be treated
        as invalid on the FFI side. Do not rely on the handle after
        transferring.

        Examples
        --------
        .. code-block:: python

            use_count = tvm_ffi.get_global_func("testing.object_use_count")
            x = tvm_ffi.convert([1, 2])
            _ = tvm_ffi.convert(lambda o: o._move())(x)
            # After move, ``x`` no longer owns the FFI handle
            assert x.__ctypes_handle__().value is None

        """

    def __move_handle_from__(self, other: Object) -> None:
        """Steal the FFI handle from ``other``.

        Internal helper used by the runtime to implement move
        semantics. Users should prefer :py:meth:`_move`.
        """

class ObjectConvertible:
    """Base class for Python classes convertible to :class:`Object`.

    Subclasses implement :py:meth:`asobject` to produce an
    :class:`Object` instance used by the FFI runtime.
    """

    def asobject(self) -> Object:
        """Return an :class:`Object` view of this value.

        This method is used by the conversion helpers (e.g.
        :func:`tvm_ffi.convert`) when a Python value needs to be passed
        into FFI calls.

        Returns
        -------
        tvm_ffi.core.Object

        """

class ObjectRValueRef:
    """Rvalue reference wrapper used to express move semantics.

    Instances are created from :py:meth:`Object._move` and signal to
    the FFI layer that ownership of the underlying handle can be
    transferred.
    """

    obj: Object
    def __init__(self, obj: Object) -> None:
        """Construct from an existing :class:`Object`.

        Parameters
        ----------
        obj : Object
            The source object from which to move the underlying handle.

        """

class OpaquePyObject(Object):
    """Wrapper that carries an arbitrary Python object across the FFI.

    The contained object is held with correct reference counting, and
    can be recovered on the Python side using :py:meth:`pyobject`.

    Notes
    -----
    ``OpaquePyObject`` is useful when a Python value must traverse the
    FFI boundary without conversion into a native FFI type.

    """

    def pyobject(self) -> Any:
        """Return the original Python object held by this wrapper."""

class PyNativeObject:
    """Base class for TVM objects that also inherit Python builtins.

    This mixin is used by Python-native proxy types such as
    :class:`String` and :class:`Bytes`, which subclass :class:`str` and
    :class:`bytes` respectively while also carrying an attached FFI
    object for zero-copy exchange with the runtime when beneficial.
    """

    __slots__: list[str]
    def __init_tvm_ffi_object_by_constructor__(self, fconstructor: Any, *args: Any) -> None: ...

def _set_class_object(cls: type) -> None: ...
def _register_object_by_index(type_index: int, type_cls: type) -> TypeInfo: ...
def _object_type_key_to_index(type_key: str) -> int | None: ...
def _set_type_cls(type_info: TypeInfo, type_cls: type) -> None: ...
def _lookup_or_register_type_info_from_type_key(type_key: str) -> TypeInfo: ...

class Error(Object):
    """Base class for FFI errors.

    An :class:`Error` is a lightweight wrapper around a concrete Python
    exception raised by FFI calls.  It stores the error ``kind`` (e.g.
    ``"ValueError"``), the message, and a serialized FFI backtrace that
    can be re-attached to produce a Python traceback.

    Users normally interact with specific error subclasses that are
    registered via :func:`tvm_ffi.error.register_error`.
    """

    def __init__(self, kind: str, message: str, backtrace: str) -> None:
        """Construct an error wrapper.

        Parameters
        ----------
        kind : str
            Name of the Python exception type (e.g. ``"ValueError"``).
        message : str
            The error message from the FFI side.
        backtrace : str
            Serialized backtrace encoded by the runtime.

        """

    def update_backtrace(self, backtrace: str) -> None:
        """Replace the stored backtrace string with ``backtrace``."""

    def py_error(self) -> BaseException:
        """Return a Python :class:`BaseException` instance for this error."""
    @property
    def kind(self) -> str:
        """The name of the Python exception class (e.g. ``"ValueError"``)."""
    @property
    def message(self) -> str:
        """The error message."""
    @property
    def backtrace(self) -> str:
        """The serialized FFI backtrace string."""

def _convert_to_ffi_error(error: BaseException) -> Error: ...
def _env_set_current_stream(
    device_type: int, device_id: int, stream: int | c_void_p
) -> int | c_void_p: ...

class DataType:
    """Internal wrapper around ``DLDataType``.

    This is a low-level representation used by the FFI layer. It is
    not intended as a user-facing API. For user code, prefer
    :class:`tvm_ffi.dtype`, which behaves like a Python ``str`` and
    integrates with array libraries.

    Examples
    --------
    .. code-block:: python

        # Prefer the user-facing helper
        d = tvm_ffi.dtype("int32")
        assert d.bits == 32
        assert str(d) == "int32"

    """

    def __init__(self, dtype_str: str) -> None: ...
    def __reduce__(self) -> Any: ...
    def __eq__(self, other: Any) -> bool: ...
    def __ne__(self, other: Any) -> bool: ...
    @property
    def type_code(self) -> int:
        """Integer DLDataTypeCode of the scalar base type."""
    @property
    def bits(self) -> int:
        """Number of bits of the scalar base type."""
    @property
    def lanes(self) -> int:
        """Number of lanes (for vector types)."""
    @property
    def itemsize(self) -> int:
        """Size of one element in bytes (``bits * lanes // 8``)."""
    def __str__(self) -> str: ...

def _set_class_dtype(cls: type) -> None: ...
def _convert_torch_dtype_to_ffi_dtype(torch_dtype: Any) -> DataType: ...
def _convert_numpy_dtype_to_ffi_dtype(numpy_dtype: Any) -> DataType: ...
def _create_dtype_from_tuple(cls: type[DataType], code: int, bits: int, lanes: int) -> DataType: ...

class DLDeviceType(IntEnum):
    """Enumeration mirroring DLPack's ``DLDeviceType``.

    Values can be compared against :py:meth:`Device.dlpack_device_type`.

    Examples
    --------
    .. code-block:: python

        dev = tvm_ffi.device("cuda", 0)
        assert dev.dlpack_device_type() == tvm_ffi.DLDeviceType.kDLCUDA

    """

    kDLCPU = 1
    kDLCUDA = 2
    kDLCUDAHost = 3
    kDLOpenCL = 4
    kDLVulkan = 7
    kDLMetal = 8
    kDLVPI = 9
    kDLROCM = 10
    kDLROCMHost = 11
    kDLExtDev = 12
    kDLCUDAManaged = 13
    kDLOneAPI = 14
    kDLWebGPU = 15
    kDLHexagon = 16
    kDLTrn = 17

class Device:
    """A device descriptor used by TVM FFI and DLPack.

    A :class:`Device` identifies a placement (e.g. CPU, CUDA GPU) and
    a device index within that placement. Most users construct devices
    using :func:`tvm_ffi.device`.

    Examples
    --------
    .. code-block:: python

        dev = tvm_ffi.device("cuda:0")
        assert dev.type == "cuda"
        assert dev.index == 0
        assert str(dev) == "cuda:0"

    """

    def __init__(self, device_type: str | int, index: int | None = None) -> None:
        """Construct a device from a type and optional index.

        Parameters
        ----------
        device_type : str or int
            A device type name (e.g. ``"cpu"``, ``"cuda"``) or a
            DLPack device type code.
        index : int, optional
            Zero-based device index (defaults to ``0`` when omitted).

        """
    def __reduce__(self) -> Any: ...
    def __eq__(self, other: Any) -> bool: ...
    def __ne__(self, other: Any) -> bool: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def __hash__(self) -> int: ...
    def __device_type_name__(self) -> str:
        """Return the canonical device type name (e.g. ``"cuda"``)."""
    @property
    def type(self) -> str:
        """Device type name such as ``"cpu"`` or ``"cuda"``."""
    @property
    def index(self) -> int:
        """Zero-based device index."""
    def dlpack_device_type(self) -> int:
        """Return the corresponding :class:`DLDeviceType` enum value."""

def _set_class_device(cls: type) -> None: ...

_CLASS_DEVICE: type[Device]

def _shape_obj_get_py_tuple(obj: Any) -> tuple[int, ...]: ...

class Tensor(Object):
    """Managed n-dimensional array compatible with DLPack.

    ``Tensor`` provides zero-copy interoperability with array libraries
    through the DLPack protocol. Instances are typically created with
    :func:`from_dlpack` or returned from FFI functions.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        x = tvm_ffi.from_dlpack(np.arange(6, dtype="int32"))
        assert x.shape == (6,)
        assert x.dtype == tvm_ffi.dtype("int32")
        # Round-trip through NumPy using DLPack
        np.testing.assert_equal(np.from_dlpack(x), np.arange(6, dtype="int32"))

    """

    @property
    def shape(self) -> tuple[int, ...]:
        """Tensor shape as a tuple of integers."""
    @property
    def dtype(self) -> Any:
        """Data type as :class:`tvm_ffi.dtype` (``str`` subclass)."""
    @property
    def device(self) -> Device:
        """The :class:`Device` on which the tensor is placed."""
    def _to_dlpack(self) -> Any:
        """Return a DLPack capsule representing this tensor (internal)."""
    def _to_dlpack_versioned(self) -> Any:
        """Return a versioned DLPack capsule (internal)."""
    def __dlpack_device__(self) -> tuple[int, int]:
        """Implement the standard ``__dlpack_device__`` protocol."""
    def __dlpack__(
        self,
        *,
        stream: Any | None = None,
        max_version: tuple[int, int] | None = None,
        dl_device: tuple[int, int] | None = None,
        copy: bool | None = None,
    ) -> Any:
        """Implement the standard ``__dlpack__`` protocol.

        Parameters
        ----------
        stream : Any, optional
            Framework-specific stream/context object.
        max_version : Tuple[int, int], optional
            Upper bound on the supported DLPack version of the
            consumer. When ``None``, use the built-in protocol version.
        dl_device : Tuple[int, int], optional
            Override the device reported by :py:meth:`__dlpack_device__`.
        copy : bool, optional
            If ``True``, produce a copy rather than exporting in-place.

        """

_CLASS_TENSOR: type[Tensor] = Tensor

def _set_class_tensor(cls: type[Tensor]) -> None: ...
def from_dlpack(
    ext_tensor: Any, *, require_alignment: int = ..., require_contiguous: bool = ...
) -> Tensor:
    """Import a foreign array that implements the DLPack producer protocol.

    Parameters
    ----------
    ext_tensor : Any
        An object supporting ``__dlpack__`` and ``__dlpack_device__``.
    require_alignment : int, optional
        If greater than zero, require the underlying data pointer to be
        aligned to this many bytes. Misaligned inputs raise
        :class:`ValueError`.
    require_contiguous : bool, optional
        When ``True``, require the layout to be contiguous. Non-contiguous
        inputs raise :class:`ValueError`.

    Returns
    -------
    Tensor
        A TVM FFI :class:`Tensor` that references the same memory.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        x_np = np.arange(8, dtype="int32")
        x = tvm_ffi.from_dlpack(x_np)
        y_np = np.from_dlpack(x)
        assert np.shares_memory(x_np, y_np)

    """

class DLTensorTestWrapper:
    """Wrapper of a Tensor that exposes DLPack protocol, only for testing purpose."""

    __c_dlpack_from_pyobject__: int
    def __init__(self, tensor: Tensor) -> None: ...
    def __tvm_ffi_env_stream__(self) -> int: ...
    def __dlpack_device__(self) -> tuple[int, int]: ...
    def __dlpack__(self, **kwargs: Any) -> Any: ...

def _dltensor_test_wrapper_c_dlpack_from_pyobject_as_intptr() -> int: ...

class Function(Object):
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

    """

    @property
    def release_gil(self) -> bool:
        """Whether calls release the Python GIL while executing."""
    @release_gil.setter
    def release_gil(self, value: bool) -> None:
        """Configure GIL release behavior for this function."""
    def __call__(self, *args: Any) -> Any:
        """Invoke the wrapped FFI function with ``args``.

        Arguments are automatically converted between Python values and
        FFI-compatible forms. The return value is converted back to a
        Python object.
        """

def _register_global_func(
    name: str, pyfunc: Callable[..., Any] | Function, override: bool
) -> Function: ...
def _get_global_func(name: str, allow_missing: bool) -> Function | None: ...
def _convert_to_ffi_func(pyfunc: Callable[..., Any]) -> Function: ...
def _convert_to_opaque_object(pyobject: Any) -> OpaquePyObject: ...
def _print_debug_info() -> None: ...

class String(str, PyNativeObject):
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

    __slots__ = ["__tvm_ffi_object__"]
    __tvm_ffi_object__: Object | None

    def __new__(cls, value: str) -> String:
        """Create a new ``String`` from a Python ``str``."""

    # pylint: disable=no-self-argument
    def __from_tvm_ffi_object__(cls, obj: Any) -> String:
        """Construct a ``String`` from an FFI object (internal)."""

class Bytes(bytes, PyNativeObject):
    """Byte buffer that interoperates with FFI while behaving like ``bytes``.

    Like :class:`String`, this class enables zero-copy exchange for
    large data. Most Python code can use ``bytes`` directly; the FFI
    layer constructs :class:`Bytes` as needed.
    """

    __slots__ = ["__tvm_ffi_object__"]
    __tvm_ffi_object__: Object | None

    def __new__(cls, value: bytes) -> Bytes:
        """Create a new ``Bytes`` from a Python ``bytes`` value."""

    # pylint: disable=no-self-argument
    def __from_tvm_ffi_object__(cls, obj: Any) -> Bytes:
        """Construct ``Bytes`` from an FFI object (internal)."""

# ---------------------------------------------------------------------------
# Type reflection metadata (from cython/type_info.pxi)
# ---------------------------------------------------------------------------

class TypeSchema:
    """Type schema that describes a TVM FFI type.

    The schema is expressed using a compact JSON-compatible structure
    and can be rendered as a Python typing string with
    :py:meth:`repr`.
    """

    origin: str
    args: tuple[TypeSchema, ...] = ()

    @staticmethod
    def from_json_obj(obj: dict[str, Any]) -> TypeSchema:
        """Construct a :class:`TypeSchema` from a parsed JSON object."""
    @staticmethod
    def from_json_str(s: str) -> TypeSchema:
        """Construct a :class:`TypeSchema` from a JSON string."""
    def repr(self, ty_map: Callable[[str], str] | None = None) -> str:
        """Render a human-readable representation of this schema.

        Parameters
        ----------
        ty_map : Callable[[str], str], optional
            A mapping function applied to the schema origin name before
            rendering (e.g. map ``"list" -> "Sequence"`` and
            ``"dict" -> "Mapping"``). If ``None``, the raw origin is used.

        Returns
        -------
        str
            A readable string using Python typing syntax. Formats include:
            - Unions as ``"T1 | T2"``
            - Optional as ``"T | None"``
            - Callables as ``"Callable[[arg1, ...], ret]"``
            - Containers as ``"origin[arg1, ...]"``

        Examples
        --------
        .. code-block:: python

            # From JSON emitted by the runtime
            s = TypeSchema.from_json_str('{"type":"Optional","args":[{"type":"int"}]}')
            assert s.repr() == "int | None"

            # Callable where the first arg is return type, remaining are parameters
            s = TypeSchema("Callable", (TypeSchema("int"), TypeSchema("str")))
            assert s.repr() == "Callable[[str], int]"

            # Custom mapping to stdlib typing collections
            def _map(t: str) -> str:
                return {"list": "Sequence", "dict": "Mapping"}.get(t, t)

            s = TypeSchema.from_json_str('{"type":"dict","args":[{"type":"str"},{"type":"int"}]}')
            assert s.repr(_map) == "Mapping[str, int]"

        """

class TypeField:
    """Description of a single reflected field on an FFI-backed type.

    Instances are used to synthesize Python properties on generated
    proxy classes.
    """

    name: str
    doc: str | None
    size: int
    offset: int
    frozen: bool
    metadata: dict[str, Any]
    getter: Any
    setter: Any
    dataclass_field: Any | None

    def as_property(self, cls: type) -> property:
        """Produce a Python :class:`property` for the given class ``cls``."""

class TypeMethod:
    """Description of a single reflected method on an FFI-backed type.

    Instances are used to synthesize bound callables on generated proxy
    classes.
    """

    name: str
    doc: str | None
    func: Any
    is_static: bool
    metadata: dict[str, Any]

    def as_callable(self, cls: type) -> Callable[..., Any]:
        """Produce a bound Python callable for the given class ``cls``."""

class TypeInfo:
    """Aggregated type information required to build a proxy class.

    This structure contains the reflected fields and methods for an FFI
    type, along with hierarchy information used during Python class
    synthesis.
    """

    type_cls: type | None
    type_index: int
    type_key: str
    fields: list[TypeField]
    methods: list[TypeMethod]
    parent_type_info: TypeInfo | None

    def prototype_py(self) -> str:
        """Render a Python prototype string for debugging and testing."""
