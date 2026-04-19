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
import json
import typing
from abc import ABCMeta
from typing import Any


_CLASS_OBJECT = None


def _set_class_object(cls):
    global _CLASS_OBJECT
    _CLASS_OBJECT = cls


def object_repr(obj: "CObject") -> str:
    """Return a human-readable repr of *obj* via ``ffi.ReprPrint``.

    Falls back to ``TypeName(handle)`` if ``ReprPrint`` is unavailable.
    """
    if (<CObject>obj).chandle == NULL:
        return type(obj).__name__ + "(chandle=None)"
    try:
        from tvm_ffi._ffi_api import ReprPrint
        return str(ReprPrint(obj))
    except Exception:  # noqa: BLE001
        # Silently fall back: repr must never raise.
        return type(obj).__name__ + "(" + str(obj.__chandle__()) + ")"


def _new_object(cls):
    """Helper function for pickle"""
    return cls.__new__(cls)


class ObjectConvertible:
    """Base class for Python classes convertible to :class:`Object`.

    Subclasses implement :py:meth:`asobject` to produce an
    :class:`Object` instance used by the FFI runtime.
    """

    def asobject(self) -> "Object":
        """Return an :class:`Object` view of this value.

        This method is used by the conversion helpers (e.g.
        :func:`tvm_ffi.convert`) when a Python value needs to be passed
        into FFI calls.

        Returns
        -------
        tvm_ffi.core.Object

        """
        raise NotImplementedError()


class ObjectRValueRef:
    """Rvalue reference wrapper used to express move semantics.

    Instances are created from :py:meth:`Object._move` and signal to
    the FFI layer that ownership of the underlying handle can be
    transferred.

    Parameters
    ----------
    obj : tvm_ffi.core.Object
        The source object from which to move the underlying handle.
    """

    __slots__ = ["obj"]

    def __init__(self, obj):
        self.obj = obj


cdef class CObject:
    """Cython base class for TVM FFI objects.

    This extension type owns the low-level handle. Prefer subclassing
    :class:`Object` in Python to enforce slots policy.
    """
    __slots__ = ()
    cdef void* chandle

    def __cinit__(self):
        # initialize chandle to NULL to avoid leak in
        # case of error before chandle is set
        self.chandle = NULL

    def __dealloc__(self):
        if self.chandle != NULL:
            CHECK_CALL(TVMFFIObjectDecRef(self.chandle))
            self.chandle = NULL

    def __ctypes_handle__(self) -> object:
        return ctypes_handle(self.chandle)

    def __chandle__(self) -> int:
        cdef uint64_t chandle = <uint64_t>self.chandle
        return chandle

    @property
    def id_(self) -> int:
        """The integer address of the underlying FFI handle.

        Alias for :py:meth:`__chandle__`.  Returns ``0`` when the
        handle is NULL.
        """
        cdef uint64_t chandle = <uint64_t>self.chandle
        return chandle

    def __reduce__(self):
        cls = type(self)
        return (_new_object, (cls,), self.__getstate__())

    def __getstate__(self) -> dict[str, Any]:
        if _OBJECT_TO_JSON_GRAPH_STR is None:
            raise RuntimeError("ffi.ToJSONGraphString is not registered, make sure build project with extra API")
        if not self.__chandle__() == 0:
            # need to explicit convert to str in case String
            # returned and triggered another infinite recursion in get state
            return {"handle": str(_OBJECT_TO_JSON_GRAPH_STR(self, None))}
        return {"handle": None}

    def __setstate__(self, state: dict[str, Any]) -> None:
        # pylint: disable=assigning-non-slot, assignment-from-no-return
        if _OBJECT_FROM_JSON_GRAPH_STR is None:
            raise RuntimeError("ffi.FromJSONGraphString is not registered, make sure build project with extra API")
        handle = state["handle"]
        if handle is not None:
            self.__init_handle_by_constructor__(_OBJECT_FROM_JSON_GRAPH_STR, handle)
        else:
            self.chandle = NULL

    def __repr__(self) -> str:
        return object_repr(self)

    def __eq__(self, other: object) -> bool:
        return self.same_as(other)

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        cdef uint64_t hash_value = <uint64_t>self.chandle
        return hash_value

    def same_as(self, other: object) -> bool:
        return isinstance(other, CObject) and self.chandle == (<CObject>other).chandle

    def is_(self, other: object) -> bool:
        """Return ``True`` if both references point to the same FFI handle.

        Alias for :py:meth:`same_as`.  Checks identity of the
        underlying handle rather than performing a structural,
        value-based comparison.
        """
        return isinstance(other, CObject) and self.chandle == (<CObject>other).chandle

    def __move_handle_from__(self, other: CObject) -> None:
        self.chandle = (<CObject>other).chandle
        (<CObject>other).chandle = NULL

    def __init_handle_by_constructor__(self, fconstructor: Any, *args: Any) -> None:
        # avoid error raised during construction.
        self.chandle = NULL
        cdef void* chandle
        ConstructorCall(
            (<CObject>fconstructor).chandle, <PyObject*>args, &chandle, NULL)
        self.chandle = chandle


cdef class CContainerBase(CObject):
    """Cython base for container types that support lazy DLPack conversion.

    Stores a ``DLPackExchangeAPI*`` tag so that element access on a
    returned container can automatically convert ``ffi.Tensor`` to
    the framework tensor type (e.g. ``torch.Tensor``).
    """
    # Raw pointer to the DLPack exchange API struct.  Not ref-counted.
    #
    # Lifetime safety: the two sources of this pointer are both
    # effectively process-lifetime:
    #
    # 1. __dlpack_c_exchange_api__ (e.g. torch.Tensor) — points to a
    #    static struct in the framework's C++ runtime.  The source
    #    type is kept alive by _DISPATCH_TYPE_KEEP_ALIVE (set in
    #    TVMFFIPyArgSetterFactory_), which prevents module unloading.
    #
    # 2. GetTorchFallbackExchangeAPI() — returns the address of a
    #    module-level Cython static; lives for the entire process.
    #
    # The DLPack spec also mandates that DLPackExchangeAPI* must stay
    # alive throughout the lifetime of the process (dlpack.h line 600).
    cdef const DLPackExchangeAPI* _dlpack_exchange_api

    def __cinit__(self):
        self._dlpack_exchange_api = NULL


# Lazy-resolved KW_ONLY sentinels (tvm_ffi.dataclasses may not be
# importable when this .pxi is first included into core.pyx).
_PY_KW_ONLY = None
_FFI_KW_ONLY = None
_KW_ONLY_RESOLVED = False


def _resolve_kw_only_sentinels():
    global _PY_KW_ONLY, _FFI_KW_ONLY, _KW_ONLY_RESOLVED
    if _KW_ONLY_RESOLVED:
        return
    try:
        from dataclasses import KW_ONLY as _py  # noqa: PLC0415
        _PY_KW_ONLY = _py
    except ImportError:
        pass
    try:
        from tvm_ffi.dataclasses import KW_ONLY as _ffi  # noqa: PLC0415
        _FFI_KW_ONLY = _ffi
    except Exception:  # noqa: BLE001
        pass
    _KW_ONLY_RESOLVED = True


class _ObjectSlotsMeta(ABCMeta):
    @staticmethod
    def _annotation_is_classvar(annotation: Any) -> bool:
        if isinstance(annotation, str):
            stripped = annotation.removeprefix("typing.")
            return stripped == "ClassVar" or stripped.startswith("ClassVar[")
        return annotation is typing.ClassVar or typing.get_origin(annotation) is typing.ClassVar

    @staticmethod
    def _annotation_is_kw_only(annotation: Any) -> bool:
        if isinstance(annotation, str):
            return annotation == "KW_ONLY" or annotation.endswith(".KW_ONLY")
        _resolve_kw_only_sentinels()
        return annotation is _PY_KW_ONLY or annotation is _FFI_KW_ONLY

    @staticmethod
    def _extract_slots_from_annotations(ns: dict[str, Any]) -> tuple[tuple[str, ...], dict[str, Any]]:
        annotations = ns.get("__annotations__", {})
        if not isinstance(annotations, dict):
            return (), {}
        slot_names: list[str] = []
        saved: dict[str, Any] = {}
        for name, annotation in annotations.items():
            if (
                _ObjectSlotsMeta._annotation_is_classvar(annotation)
                or _ObjectSlotsMeta._annotation_is_kw_only(annotation)
            ):
                continue
            slot_names.append(name)
            if name in ns:
                saved[name] = ns.pop(name)
        return tuple(slot_names), saved

    def __new__(mcls, name: str, bases: tuple[type, ...], ns: dict[str, Any], **kwargs: Any):
        if "__slots__" not in ns:
            slots, saved = mcls._extract_slots_from_annotations(ns)
            ns["__slots__"] = slots
            if saved:
                ns["_tvm_ffi_saved_defaults"] = saved
        return super().__new__(mcls, name, bases, ns, **kwargs)

    def __init__(cls, name: str, bases: tuple[type, ...], ns: dict[str, Any], **kwargs: Any):
        super().__init__(name, bases, ns, **kwargs)


class Object(CObject, metaclass=_ObjectSlotsMeta):
    """Base class of all TVM FFI objects.

    This is the root Python type for objects backed by the TVM FFI
    runtime. Each instance references a handle to a C++ runtime
    object. Python subclasses typically correspond to C++ runtime
    types and are registered via :py:meth:`tvm_ffi.register_object`.

    Notes
    -----
    - Equality of two :py:class:`Object` instances uses underlying handle
      identity unless an overridden implementation is provided on the
      concrete type. Use :py:meth:`same_as` to check whether two
      references point to the same underlying object.
    - Subclasses that omit ``__slots__`` get slots derived from their
      annotations, or ``__slots__ = ()`` when they have no annotated
      fields. To allow a per-instance ``__dict__``, declare
      ``__slots__ = ("__dict__",)`` explicitly in the class body.
    - Most users interact with subclasses (e.g. :class:`Tensor`,
      :class:`Function`) rather than :py:class:`Object` directly.

    Examples
    --------
    Constructing objects is typically performed by Python wrappers that
    call into registered constructors on the FFI side.

    .. code-block:: python

        import tvm_ffi.testing

        # Acquire a testing object constructed through FFI
        obj = tvm_ffi.testing.create_object("testing.TestObjectBase", v_i64=12)
        assert isinstance(obj, tvm_ffi.Object)
        assert obj.same_as(obj)

    Subclasses can declare explicit slots when needed.

    .. code-block:: python

        @tvm_ffi.register_object("my.MyObject")
        class MyObject(tvm_ffi.Object):
            __slots__ = ()

    Subclasses that need a per-instance ``__dict__`` (e.g. for attribute
    caching) can opt in explicitly.

    .. code-block:: python

        @tvm_ffi.register_object("my.MyDynObject")
        class MyDynObject(tvm_ffi.Object):
            __slots__ = ("__dict__",)

    """
    __slots__ = ()

    def same_as(self, other: object) -> bool:
        """Return ``True`` if both references point to the same object.

        This checks identity of the underlying FFI handle rather than
        performing a structural, value-based comparison.

        Parameters
        ----------
        other : object
            The object to compare against.

        Returns
        -------
        bool

        Examples
        --------
        .. code-block:: python

            import tvm_ffi.testing

            x = tvm_ffi.testing.create_object("testing.TestObjectBase")
            y = x
            z = tvm_ffi.testing.create_object("testing.TestObjectBase")
            assert x.same_as(y)
            assert not x.same_as(z)

        """
        return CObject.same_as(self, other)

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

        Returns
        -------
        ObjectRValueRef
            The rvalue reference wrapper.
        """
        return ObjectRValueRef(self)

    def __move_handle_from__(self, other: CObject) -> None:
        """Steal the FFI handle from ``other``.

        Internal helper used by the runtime to implement move
        semantics. Users should prefer :py:meth:`_move`.
        """
        CObject.__move_handle_from__(self, other)

    def __init_handle_by_constructor__(self, fconstructor: Any, *args: Any) -> None:
        """Initialize the handle by calling constructor function.

        Parameters
        ----------
        fconstructor : Function
            Constructor function.

        args: list of objects
            The arguments to the constructor

        Notes
        -----
        We have a special calling convention to call constructor functions.
        So the return handle is directly set into the Node object
        instead of creating a new Node.
        """
        CObject.__init_handle_by_constructor__(self, fconstructor, *args)


cdef class OpaquePyObject(CObject):
    """Wrapper that carries an arbitrary Python object across the FFI.

    The contained object is held with correct reference counting, and
    can be recovered on the Python side using :py:meth:`pyobject`.

    Notes
    -----
    ``OpaquePyObject`` is useful when a Python value must traverse the
    FFI boundary without conversion into a native FFI type.
    """
    __slots__ = ()

    def pyobject(self) -> object:
        """Return the original Python object held by this wrapper."""
        cdef object obj
        cdef PyObject* py_handle
        py_handle = <PyObject*>(TVMFFIOpaqueObjectGetCellPtr(self.chandle).handle)
        obj = <object>py_handle
        return obj


class PyNativeObject:
    """Base class for TVM objects that also inherit Python builtins.

    This mixin is used by Python-native proxy types such as
    :class:`String` and :class:`Bytes`, which subclass :class:`str` and
    :class:`bytes` respectively while also carrying an attached FFI
    object for zero-copy exchange with the runtime when beneficial.
    """
    __slots__ = []

    def __init_cached_object_by_constructor__(self, fconstructor: Any, *args: Any) -> None:
        """Initialize the internal _tvm_ffi_cached_object by calling constructor function.

        Parameters
        ----------
        fconstructor : Function
            Constructor function.

        args: list of objects
            The arguments to the constructor

        Note
        ----
        We have a special calling convention to call constructor functions.
        So the return object is directly set into the object
        """
        obj = _CLASS_OBJECT.__new__(_CLASS_OBJECT)
        obj.__init_handle_by_constructor__(fconstructor, *args)
        self._tvm_ffi_cached_object = obj


def _object_type_key_to_index(str type_key):
    """get the type index of object class"""
    cdef int32_t tidx
    type_key_arg = ByteArrayArg(c_str(type_key))
    if TVMFFITypeKeyToIndex(type_key_arg.cptr(), &tidx) == 0:
        return tidx
    return None


cdef inline str _type_index_to_key(int32_t tindex):
    """get the type key of object class"""
    cdef const TVMFFITypeInfo* info = TVMFFIGetTypeInfo(tindex)
    cdef const TVMFFIByteArray* type_key
    if info == NULL:
        return "<unknown>"
    type_key = &(info.type_key)
    return bytearray_to_str(type_key)


cdef inline object make_ret_opaque_object(TVMFFIAny result):
    obj = OpaquePyObject.__new__(OpaquePyObject)
    (<CObject>obj).chandle = result.v_obj
    return obj.pyobject()

cdef inline object make_fallback_cls_for_type_index(int32_t type_index):
    cdef str type_key = _type_index_to_key(type_index)
    cdef object type_info = _lookup_or_register_type_info_from_type_key(type_key)
    cdef object parent_type_info = type_info.parent_type_info
    assert type_info.type_cls is None

    # Ensure parent classes are created first
    assert parent_type_info is not None
    if parent_type_info.type_cls is None:   # recursively create parent class first
        make_fallback_cls_for_type_index(parent_type_info.type_index)
    assert parent_type_info.type_cls is not None

    # Create `type_info.type_cls` now
    class cls(parent_type_info.type_cls):
        __slots__ = ()

    cls.__tvm_ffi_type_info__ = type_info
    cls.__name__ = type_key.split(".")[-1]
    cls.__qualname__ = type_key
    cls.__module__ = ".".join(type_key.split(".")[:-1])
    cls.__doc__ = (
        f"Auto-generated fallback class for {type_key}.\n"
        "This class is generated because the class is not registered.\n"
        "Please do not use this class directly, instead register the class\n"
        "using `register_object` decorator."
    )
    for field in type_info.fields:
        setattr(cls, field.name, field.as_property(cls))
    for method in type_info.methods:
        setattr(cls, method.name, method.as_callable(cls))
    # Install fast C-level field access (same as registered classes).
    if type_info.fields:
        _install_field_getattro(cls, type_info)
    # Update the registry
    type_info.type_cls = cls
    _update_registry(type_index, type_key, type_info, cls)
    return cls


cdef inline object make_ret_object(TVMFFIAny result):
    cdef int32_t type_index
    cdef object cls, obj
    type_index = result.type_index

    if type_index < len(TYPE_INDEX_TO_CLS) and (cls := TYPE_INDEX_TO_CLS[type_index]) is not None:
        if issubclass(cls, PyNativeObject):
            obj = Object.__new__(Object)
            (<CObject>obj).chandle = result.v_obj
            return cls.__from_tvm_ffi_object__(cls, obj)
    else:
        # Slow path: object is not found in registered entry
        # In this case create a dummy stub class for future usage.
        # For every unregistered class, this slow path will be triggered only once.
        cls = make_fallback_cls_for_type_index(type_index)
    obj = cls.__new__(cls)
    (<CObject>obj).chandle = result.v_obj
    return obj


cdef _get_method_from_method_info(const TVMFFIMethodInfo* method):
    cdef TVMFFIAny result
    CHECK_CALL(TVMFFIAnyViewToOwnedAny(&(method.method), &result))
    return make_ret(result)


cdef _type_info_create_from_type_key(object type_cls, str type_key):
    cdef const TVMFFIFieldInfo* field
    cdef const TVMFFIMethodInfo* method
    cdef const TVMFFITypeInfo* info
    cdef int32_t type_index
    cdef list ancestors = []
    cdef int ancestor
    cdef dict metadata_obj
    cdef object fields = []
    cdef object methods = []
    cdef str type_schema_json
    cdef FieldGetter generic_getter
    cdef object getter
    cdef FieldSetter setter
    cdef bint has_default
    cdef bint default_from_factory
    cdef TVMFFIAny owned_default
    cdef object c_default
    cdef object c_default_factory
    cdef object c_structural_eq
    cdef ByteArrayArg type_key_arg = ByteArrayArg(c_str(type_key))

    # NOTE: `type_key_arg` must be kept alive until after the call to `TVMFFITypeKeyToIndex`,
    # because Cython doesn't defer the destruction of `type_key_arg` until after the call.
    if TVMFFITypeKeyToIndex(type_key_arg.cptr(), &type_index) != 0:
        raise ValueError(f"Cannot find type key: {type_key}")
    info = TVMFFIGetTypeInfo(type_index)
    for i in range(info.num_fields):
        field = &(info.fields[i])
        generic_getter = FieldGetter.__new__(FieldGetter)
        generic_getter.getter = field.getter
        generic_getter.offset = field.offset
        getter = _make_specialized_getter(
            field.field_static_type_index,
            field.size,
            field.offset,
            generic_getter,
        )
        setter = FieldSetter.__new__(FieldSetter)
        (<FieldSetter>setter).setter = field.setter
        (<FieldSetter>setter).offset = field.offset
        (<FieldSetter>setter).flags = field.flags
        metadata_obj = json.loads(bytearray_to_str(&field.metadata)) if field.metadata.size != 0 else {}
        # Decode the static default value or factory (if any) registered by C++.
        has_default = (field.flags & kTVMFFIFieldFlagBitMaskHasDefault) != 0
        default_from_factory = (field.flags & kTVMFFIFieldFlagBitMaskDefaultFromFactory) != 0
        c_default = MISSING
        c_default_factory = MISSING
        if has_default:
            CHECK_CALL(TVMFFIAnyViewToOwnedAny(&field.default_value_or_factory, &owned_default))
            if default_from_factory:
                c_default_factory = make_ret(owned_default)
            else:
                c_default = make_ret(owned_default)
        # Decode SEqHashIgnore / SEqHashDef into the Field.structural_eq vocabulary.
        if (field.flags & kTVMFFIFieldFlagBitMaskSEqHashIgnore) != 0:
            c_structural_eq = "ignore"
        elif (field.flags & kTVMFFIFieldFlagBitMaskSEqHashDef) != 0:
            c_structural_eq = "def"
        else:
            c_structural_eq = None
        fields.append(
            TypeField(
                name=bytearray_to_str(&field.name),
                doc=bytearray_to_str(&field.doc) if field.doc.size != 0 else None,
                size=field.size,
                offset=field.offset,
                frozen=(field.flags & kTVMFFIFieldFlagBitMaskWritable) == 0,
                metadata=metadata_obj,
                getter=getter,
                setter=setter,
                c_init=(field.flags & kTVMFFIFieldFlagBitMaskInitOff) == 0,
                c_kw_only=(field.flags & kTVMFFIFieldFlagBitMaskKwOnly) != 0,
                c_has_default=has_default,
                c_default=c_default,
                c_default_factory=c_default_factory,
                c_repr=(field.flags & kTVMFFIFieldFlagBitMaskReprOff) == 0,
                c_compare=(field.flags & kTVMFFIFieldFlagBitMaskCompareOff) == 0,
                c_hash=(field.flags & kTVMFFIFieldFlagBitMaskHashOff) == 0,
                c_structural_eq=c_structural_eq,
            )
        )

    for i in range(info.num_methods):
        method = &(info.methods[i])
        metadata_obj = json.loads(bytearray_to_str(&method.metadata)) if method.metadata.size != 0 else {}
        methods.append(
            TypeMethod(
                name=bytearray_to_str(&method.name),
                doc=bytearray_to_str(&method.doc) if method.doc.size != 0 else None,
                func=_get_method_from_method_info(method),
                is_static=(method.flags & kTVMFFIFieldFlagBitMaskIsStaticMethod) != 0,
                metadata=metadata_obj,
            )
        )

    for i in range(info.type_depth):
        ancestor = info.type_ancestors[i].type_index
        ancestors.append(ancestor)

    return TypeInfo(
        type_cls=type_cls,
        type_index=type_index,
        type_key=bytearray_to_str(&info.type_key),
        type_ancestors=ancestors,
        fields=fields,
        methods=methods,
        parent_type_info=None,
    )


cdef _update_registry(int type_index, object type_key, object type_info, object type_cls):
    cdef int extra = type_index + 1 - len(TYPE_INDEX_TO_INFO)
    assert len(TYPE_INDEX_TO_INFO) == len(TYPE_INDEX_TO_CLS)
    if extra > 0:
        TYPE_INDEX_TO_INFO.extend([None] * extra)
        TYPE_INDEX_TO_CLS.extend([None] * extra)
    TYPE_INDEX_TO_CLS[type_index] = type_cls
    TYPE_INDEX_TO_INFO[type_index] = type_info
    TYPE_KEY_TO_INFO[type_key] = type_info
    if type_cls is not None:
        TYPE_CLS_TO_INFO[type_cls] = type_info


def _register_object_by_index(int type_index, object type_cls):
    global TYPE_INDEX_TO_INFO, TYPE_KEY_TO_INFO, TYPE_INDEX_TO_CLS
    cdef str type_key = _type_index_to_key(type_index)
    cdef object info = _type_info_create_from_type_key(type_cls, type_key)
    _update_registry(type_index, type_key, info, type_cls)
    return info


def _set_type_cls(object type_info, object type_cls):
    global TYPE_INDEX_TO_INFO, TYPE_INDEX_TO_CLS, TYPE_CLS_TO_INFO
    assert type_info.type_cls is None, f"Type already registered for {type_info.type_key}"
    assert TYPE_INDEX_TO_INFO[type_info.type_index] is type_info
    assert TYPE_KEY_TO_INFO[type_info.type_key] is type_info
    type_info.type_cls = type_cls
    TYPE_INDEX_TO_CLS[type_info.type_index] = type_cls
    TYPE_CLS_TO_INFO[type_cls] = type_info


def _lookup_or_register_type_info_from_type_key(type_key: str) -> TypeInfo:
    if info := TYPE_KEY_TO_INFO.get(type_key, None):
        return info
    info = _type_info_create_from_type_key(None, type_key)
    _update_registry(info.type_index, type_key, info, None)
    return info


def _register_py_class(parent_type_info, str type_key, object type_cls):
    """Register a new Python-defined TVM-FFI type.

    Allocates a dynamic type index for *type_key* as a child of
    *parent_type_info* and registers it in the global type tables.

    Parameters
    ----------
    parent_type_info : TypeInfo
        The parent type's TypeInfo (e.g., Object's TypeInfo).
    type_key : str
        The unique type key string for the new type.
    type_cls : type
        The Python class to associate with this type.

    Returns
    -------
    TypeInfo
        The newly created TypeInfo with ``fields=None`` (pending registration).

    Raises
    ------
    ValueError
        If *type_key* is already registered.
    """
    # Reject duplicate type keys
    if type_key in TYPE_KEY_TO_INFO:
        raise ValueError(
            f"Type key '{type_key}' is already registered"
        )

    cdef int32_t parent_type_index = parent_type_info.type_index
    cdef int32_t parent_type_depth = len(parent_type_info.type_ancestors)
    cdef int32_t type_depth = parent_type_depth + 1
    cdef ByteArrayArg type_key_arg = ByteArrayArg(c_str(type_key))
    cdef int32_t type_index

    # Allocate a new type index
    # static_type_index=-1 means dynamic allocation
    # num_child_slots=0, child_slots_can_overflow=1
    type_index = TVMFFITypeGetOrAllocIndex(
        type_key_arg.cptr(),
        -1,           # static_type_index (dynamic)
        type_depth,
        0,            # num_child_slots
        1,            # child_slots_can_overflow
        parent_type_index,
    )

    # Build ancestors list
    cdef list ancestors = list(parent_type_info.type_ancestors)
    ancestors.append(parent_type_index)

    # Create TypeInfo with fields=None (pending _register_fields call)
    cdef object info = TypeInfo(
        type_cls=type_cls,
        type_index=type_index,
        type_key=type_key,
        type_ancestors=ancestors,
        fields=None,
        methods=[],
        parent_type_info=parent_type_info,
    )

    _update_registry(type_index, type_key, info, type_cls)
    return info


def _rollback_py_class(object type_info):
    """Roll back a ``_register_py_class`` call from the Python-level registry.

    Called by ``@py_class`` when phase-2 (field validation) fails, so
    the type key can be reused after the user fixes the error.  The
    C-level type index is permanently consumed (cannot be reclaimed),
    but the Python dicts are cleaned up so that a retry does not hit
    "already registered".
    """
    cdef int32_t idx = type_info.type_index
    cdef str key = type_info.type_key
    cdef object cls = type_info.type_cls
    TYPE_KEY_TO_INFO.pop(key, None)
    if cls is not None:
        TYPE_CLS_TO_INFO.pop(cls, None)
    if 0 <= idx < len(TYPE_INDEX_TO_INFO):
        TYPE_INDEX_TO_INFO[idx] = None
        TYPE_INDEX_TO_CLS[idx] = None


def _lookup_type_attr(type_index: int32_t, attr_key: str) -> Any:
    cdef ByteArrayArg attr_key_bytes = ByteArrayArg(c_str(attr_key))
    cdef const TVMFFITypeAttrColumn* column = TVMFFIGetTypeAttrColumn(&attr_key_bytes.cdata)
    cdef TVMFFIAny data
    cdef int32_t offset
    if column == NULL:
        return None
    offset = type_index - column.begin_index
    if offset < 0 or offset >= column.size:
        return None
    CHECK_CALL(TVMFFIAnyViewToOwnedAny(&(column.data[offset]), &data))
    return make_ret(data)


def _register_type_attr(type_index: int32_t, attr_key: str, value: object) -> None:
    """Register a value for the ``(type_index, attr_key)`` slot.

    Wraps :c:func:`TVMFFITypeRegisterAttr`, which raises :class:`RuntimeError`
    if a value is already registered for the slot.  To update the stored
    value, register a mutable container (e.g. ``Dict``/``List``) once and
    mutate it in place on subsequent calls.

    ``TVMFFIPyPyObjectToFFIAny`` produces a non-owning :c:type:`TVMFFIAny`
    view of *value*; ``TVMFFITypeRegisterAttr`` incref's the underlying
    object when it stores the slot, so no explicit refcount management is
    needed here.
    """
    cdef ByteArrayArg attr_key_bytes = ByteArrayArg(c_str(attr_key))
    cdef TVMFFIAny temp
    cdef int c_api_ret_code
    temp.type_index = kTVMFFINone
    temp.v_int64 = 0
    TVMFFIPyPyObjectToFFIAny(
        TVMFFIPyArgSetterFactory_,
        <PyObject*>value,
        &temp,
        &c_api_ret_code,
    )
    CHECK_CALL(c_api_ret_code)
    CHECK_CALL(TVMFFITypeRegisterAttr(type_index, &attr_key_bytes.cdata, &temp))


def _type_cls_to_type_info(type_cls: type) -> TypeInfo | None:
    return TYPE_CLS_TO_INFO.get(type_cls, None)


cdef list TYPE_INDEX_TO_CLS = []
cdef list TYPE_INDEX_TO_INFO = []
cdef dict TYPE_CLS_TO_INFO = {}
cdef dict TYPE_KEY_TO_INFO = {}

_set_class_object(Object)


# ---------------------------------------------------------------------------
# CAny: Owned TVMFFIAny value container
# ---------------------------------------------------------------------------
cdef class CAny:
    """Owned :c:type:`TVMFFIAny` value container.

    Holds sole ownership of the underlying value.  For object types
    (``type_index >= kTVMFFIStaticObjectBegin``), the reference is
    properly ref-counted and released in ``__dealloc__``.

    Use :meth:`to_py` to recover the Python object.
    """

    cdef TVMFFIAny cdata

    def __cinit__(self):
        """Initialize the contained value to ``None``."""
        self.cdata.type_index = kTVMFFINone
        self.cdata.v_int64 = 0

    def __init__(self, value=None):
        """Pack a Python value into an owned :c:type:`TVMFFIAny`.

        Uses ``TVMFFIPyPyObjectToFFIAny`` to produce a non-owning AnyView,
        then ``TVMFFIAnyViewToOwnedAny`` to convert to an owned Any.

        Parameters
        ----------
        value : object, optional
            The Python value to pack.  When ``None`` (the default), the
            container stays in the ``kTVMFFINone`` state set by ``__cinit__``.
        """
        if value is None:
            return
        cdef TVMFFIAny temp
        cdef int c_api_ret_code
        temp.type_index = kTVMFFINone
        temp.v_int64 = 0
        TVMFFIPyPyObjectToFFIAny(
            TVMFFIPyArgSetterFactory_,
            <PyObject*>value,
            &temp,
            &c_api_ret_code
        )
        CHECK_CALL(c_api_ret_code)
        CHECK_CALL(TVMFFIAnyViewToOwnedAny(&temp, &self.cdata))

    def __dealloc__(self):
        """Release owned object reference, if any."""
        if self.cdata.type_index >= kTVMFFIStaticObjectBegin:
            if self.cdata.v_obj != NULL:
                CHECK_CALL(TVMFFIObjectDecRef(<TVMFFIObjectHandle>self.cdata.v_obj))
                self.cdata.v_obj = NULL

    @property
    def type_index(self) -> int:
        """The TVM FFI type index of the contained value."""
        return self.cdata.type_index

    def __repr__(self) -> str:
        """Return a developer-friendly representation."""
        cdef int32_t ti = self.cdata.type_index
        if ti == kTVMFFINone:
            return "CAny(None)"
        elif ti == kTVMFFIInt:
            return f"CAny(int={self.cdata.v_int64})"
        elif ti == kTVMFFIFloat:
            return f"CAny(float={self.cdata.v_float64})"
        elif ti == kTVMFFIBool:
            return f"CAny(bool={bool(self.cdata.v_int64)})"
        elif ti >= kTVMFFIStaticObjectBegin:
            return f"CAny(object, type_index={ti})"
        else:
            return f"CAny(type_index={ti})"


cpdef object _to_py_class_value(CAny self):
    """Convert a CAny to a Python object (module-level cdef for direct C dispatch)."""
    cdef TVMFFIAny copy = self.cdata
    if copy.type_index >= kTVMFFIStaticObjectBegin:
        if copy.v_obj != NULL:
            TVMFFIObjectIncRef(<TVMFFIObjectHandle>copy.v_obj)
    cdef object result = make_ret(copy)
    # Promote inline SmallStr/SmallBytes to their FFI wrapper types
    # so that convert().to_py() always yields tvm_ffi.String / tvm_ffi.Bytes.
    if copy.type_index == kTVMFFISmallStr:
        return String(result)
    if copy.type_index == kTVMFFISmallBytes:
        return Bytes(result)
    return result
