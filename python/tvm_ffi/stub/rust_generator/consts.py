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
"""Rust-specific constants for the ``tvm-ffi-stubgen`` Rust backend."""

from __future__ import annotations

#: One-line directives the Rust backend consumes.
RUST_DIRECTIVE_KINDS = frozenset(
    {
        "import-object",
        "field",
        "nullable",
        "visibility",
        "enum",
        "opaque",
        "upcast",
        "custom-new",
        "missing-docs",
    }
)

#: Default FFI-origin -> Rust-type map; ``::`` paths get a ``use``, bare names do not.
RUST_TY_MAP_DEFAULTS = {
    "int": "i64",
    "float": "f64",
    "bool": "bool",
    "None": "()",
    "str": "tvm_ffi::String",
    "bytes": "tvm_ffi::Bytes",
    "Any": "tvm_ffi::Any",
    "Callable": "tvm_ffi::Function",
    "Array": "tvm_ffi::Array",  # the crate's own Array<T>, NOT Vec
    "List": "tvm_ffi::List",  # the crate's own List<T>, NOT Vec
    "Map": "tvm_ffi::Map",  # the crate's own Map<K, V>, NOT HashMap
    # An object value is the `ObjectRef` handle; `Object` only appears as an embedded `base_`.
    "Object": "tvm_ffi::object::ObjectRef",
    "Tensor": "tvm_ffi::Tensor",
    "Shape": "tvm_ffi::Shape",
    "Device": "tvm_ffi::DLDevice",
    "dtype": "tvm_ffi::DLDataType",
    "DataType": "tvm_ffi::DLDataType",
    # --- builtin object type keys (ffi.*) ---
    "ffi.String": "tvm_ffi::String",
    "ffi.Bytes": "tvm_ffi::Bytes",
    "ffi.BigInt": "tvm_ffi::BigInt",
    "ffi.Module": "tvm_ffi::Module",
    "ffi.Error": "tvm_ffi::Error",
    "ffi.Object": "tvm_ffi::object::ObjectRef",
    "ffi.Tensor": "tvm_ffi::Tensor",
    "ffi.Shape": "tvm_ffi::Shape",
    "ffi.Function": "tvm_ffi::Function",
    "ffi.reflection.AccessPath": "tvm_ffi::reflection::AccessPath",
    "ffi.reflection.AccessStep": "tvm_ffi::reflection::AccessStep",
}

#: Origins without a crate mirror; such a field is read as ``tvm_ffi::Any``.
RUST_UNSUPPORTED_ORIGINS = frozenset({"Dict", "Union", "tuple"})

#: Mirror scalar by ``(origin, reflected size)``. The schema erases integer widths and
#: signedness, so only an 8-byte integer, which the field getter also reads as ``i64``, is
#: mirrored; a narrower one (``uint32_t``, ``int32_t``, ...) needs a ``field`` or ``enum``
#: directive, else its type stays opaque.
RUST_SCALAR_BY_SIZE = {
    ("int", 8): "i64",
    ("float", 4): "f32",
    ("float", 8): "f64",
}

#: ``(size, alignment)`` of the mirror of a field by schema origin; any other mirror is
#: one object pointer (:data:`RUST_POINTER_LAYOUT`). A field whose reflected bytes differ
#: has no mirror.
RUST_MIRROR_LAYOUTS = {
    "int": (8, 8),
    "bool": (1, 1),
    "str": (16, 8),
    "bytes": (16, 8),
    "Any": (16, 8),
    "Device": (8, 4),
    "dtype": (4, 2),
    "DataType": (4, 2),
}
RUST_POINTER_LAYOUT = (8, 8)

#: Byte width of the scalars a ``field`` / ``enum`` directive may name; checked against the field.
RUST_SCALAR_WIDTHS = {
    "i8": 1,
    "u8": 1,
    "bool": 1,
    "i16": 2,
    "u16": 2,
    "i32": 4,
    "u32": 4,
    "f32": 4,
    "i64": 8,
    "u64": 8,
    "f64": 8,
}

#: Parameters of the wrapper's allocator taken as their own type, as ``#[rs_object]``'s
#: ``new`` takes them: a literal infers only from its exact type. An object reference is
#: ``impl IntoArg<T>`` and any other parameter ``impl Into<T>``; ``<Leaf>Obj::new``
#: takes every one as its own type.
RUST_EXACT_PARAMS = frozenset(
    {"bool", "i8", "i16", "i32", "i64", "isize", "u8", "u16", "u32", "u64", "usize", "f32", "f64"}
)

#: Builtin origins whose Rust type is an object reference, taken as ``impl IntoArg<T>``.
#: A type key outside ``ffi.*`` is one too: a generated or ``ty-map``'d binding. ``Shape``,
#: like ``Array`` / ``List`` / ``Map``, is a container taken as ``impl Into<T>``, which also
#: accepts what it is built from (``From<impl AsRef<[i64]>>``).
RUST_OBJECT_REF_ORIGINS = frozenset(
    {
        "Object",
        "Tensor",
        "Callable",
        "ffi.Object",
        "ffi.Tensor",
        "ffi.Function",
        "ffi.Module",
        "ffi.Error",
        "ffi.reflection.AccessPath",
        "ffi.reflection.AccessStep",
    }
)

#: Size of an object reference field; ``nullable`` may only wrap those.
RUST_POINTER_SIZE = 8

#: C++ ``Optional<T>`` is a 16-byte ``TVMFFIAny`` cell, or a nullable pointer for object payloads.
RUST_OPTIONAL_PATH = "tvm_ffi::Optional"
RUST_OPTIONAL_FIELD_SIZE = 16
RUST_OBJECT_OPTIONAL_FIELD_SIZE = 8

#: ``Optional`` payloads kept as the 16-byte cell (a nested ``Optional`` too); the size is checked.
RUST_ANY_BACKED_OPTIONAL_PAYLOADS = frozenset(
    {"int", "float", "bool", "Device", "dtype", "DataType", "str", "bytes"}
)

#: ``use``-path rewrites: builtin ``ffi.*`` type keys live at the crate root.
RUST_MOD_MAP = {
    "ffi": "tvm_ffi",
}

#: Root of the object hierarchy.
RUST_ROOT_TYPE_KEY = "ffi.Object"

#: Builtin types whose object struct the crate defines; a subtype embeds it as ``base_``.
RUST_CRATE_OBJECTS = {"ffi.Enum": "tvm_ffi::enum_ref::EnumObj"}

#: Root of the enum types: their registry allocates the values, and their references
#: implement ``tvm_ffi::EnumRef``.
RUST_ENUM_TYPE_KEY = "ffi.Enum"

#: Keywords a field name may collide with: spelled ``r#name``, or ``name_`` for the four
#: that cannot be raw identifiers.
RUST_KEYWORDS = frozenset(
    "as async await break const continue crate dyn else enum extern false fn for if impl in "
    "let loop match mod move mut pub ref return self Self static struct super trait true type "
    "unsafe use where while abstract become box do final gen macro override priv try typeof "
    "unsized virtual yield".split()
)
RUST_NOT_RAW_IDENTIFIERS = frozenset({"self", "Self", "super", "crate"})

#: The members the generated structs use themselves: ``base_``, the parent's object in every
#: object struct, and ``data_``, the ``ObjectArc`` of every reference. The trailing ``_`` keeps
#: them apart from reflected names such as ``base`` and ``data``.
RUST_GENERATED_MEMBERS = frozenset({"base_", "data_"})


#: ``rustfmt``'s default ``max_width``; a wider allocator signature wraps one parameter per line.
RUST_MAX_WIDTH = 100
#: ``rustfmt``'s default ``fn_call_width``; wider call arguments take one line each.
RUST_FN_CALL_WIDTH = 60
