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

#: Default FFI-origin -> Rust-type map. Values are fully qualified paths so
#: ``RustUse``/``RustImports`` can derive both the leaf name and the ``use``
#: import; values without ``::`` (primitives) need no import.
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
    "Map": "tvm_ffi::Map",  # the crate's own Map<K, V>, NOT HashMap
    "Object": "tvm_ffi::Object",
    "Tensor": "tvm_ffi::Tensor",
    "Shape": "tvm_ffi::Shape",
    "Device": "tvm_ffi::DLDevice",
    "dtype": "tvm_ffi::DLDataType",
    "DataType": "tvm_ffi::DLDataType",
    # --- builtin object type keys (ffi.*) ---
    "ffi.String": "tvm_ffi::String",
    "ffi.Bytes": "tvm_ffi::Bytes",
    "ffi.Module": "tvm_ffi::Module",
    "ffi.Error": "tvm_ffi::Error",
    "ffi.Object": "tvm_ffi::Object",
    "ffi.Tensor": "tvm_ffi::Tensor",
    "ffi.Shape": "tvm_ffi::Shape",
    "ffi.Function": "tvm_ffi::Function",
}

#: Width-correct scalar for a ``#[repr(C)]`` struct field, keyed by
#: ``(ffi origin, sizeof(T))``: the type schema erases scalar widths, but the
#: generated structs read fields at their real offsets, so the width must be
#: recovered from the reflected field size. Signedness is not recorded;
#: unsigned C++ fields render as the same-width signed type.
RUST_SCALAR_BY_SIZE = {
    ("int", 1): "i8",
    ("int", 2): "i16",
    ("int", 4): "i32",
    ("int", 8): "i64",
    ("float", 4): "f32",
    ("float", 8): "f64",
}

#: Origins the crate has no FFI type for (do NOT map to ``HashMap``/``Vec``;
#: Rust tuples don't match ``ffi::Tuple``'s layout). ``render_rust_type``
#: raises wherever one appears and the enclosing object is skipped.
RUST_UNSUPPORTED_ORIGINS = frozenset({"Dict", "List", "Union", "tuple"})

#: Origins whose rendering cannot satisfy the crate's ``AnyCompatible``
#: element/payload bounds (``ctypes.c_void_p`` -- ``void*`` -- has no Rust
#: rendering at all): raise and skip the object.
RUST_NON_ELEMENT_ORIGINS = frozenset({"Any", "Object", "ffi.Object", "ctypes.c_void_p"})

#: In-place mirror of an ``Optional<T>`` FIELD: C++ ``ffi::Optional<T>`` is
#: uniformly a single 16-byte ``TVMFFIAny`` for every ``T`` with Any storage.
RUST_OPTIONAL_PATH = "tvm_ffi::Optional"
#: Any other reflected field size is the ``std::optional`` fallback of
#: storage-disabled types, which has no mirror.
RUST_OPTIONAL_FIELD_SIZE = 16

#: Module-prefix rewrites for ``use`` paths: builtin ``ffi.*`` type keys live at
#: the crate root.
RUST_MOD_MAP = {
    "ffi": "tvm_ffi",
}
