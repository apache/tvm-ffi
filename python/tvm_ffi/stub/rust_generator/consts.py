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

#: One-line directives the Rust backend consumes. ``import-object`` carries a
#: ``use`` path; the other three are parsed by :mod:`.directives`.
RUST_DIRECTIVE_KINDS = frozenset({"import-object", "field", "nullable", "enum"})

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
    # A generic object VALUE is the single-pointer `ObjectRef` handle, not the
    # `Object` header struct (which only ever appears as an embedded `base`).
    "Object": "tvm_ffi::object::ObjectRef",
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
    "ffi.Object": "tvm_ffi::object::ObjectRef",
    "ffi.Tensor": "tvm_ffi::Tensor",
    "ffi.Shape": "tvm_ffi::Shape",
    "ffi.Function": "tvm_ffi::Function",
}

#: Origins the crate has no type for (Rust tuples do not match ``ffi::Tuple``,
#: and ``Dict``/``List``/``Union`` have no FFI mirror). A field of such a type
#: is read as ``tvm_ffi::Any``.
RUST_UNSUPPORTED_ORIGINS = frozenset({"Dict", "List", "Union", "tuple"})

#: Module-prefix rewrites for ``use`` paths: builtin ``ffi.*`` type keys live at
#: the crate root.
RUST_MOD_MAP = {
    "ffi": "tvm_ffi",
}

#: The root of the object hierarchy; every ``ObjectInfo.ancestors`` starts with it.
RUST_ROOT_TYPE_KEY = "ffi.Object"

#: Keywords a reflected field name may collide with; they are spelled as raw
#: identifiers (``r#type``), except the four that Rust forbids as raw
#: identifiers, which get a trailing underscore instead.
RUST_KEYWORDS = frozenset(
    "as async await break const continue crate dyn else enum extern false fn for if impl in "
    "let loop match mod move mut pub ref return self Self static struct super trait true type "
    "unsafe use where while abstract become box do final gen macro override priv try typeof "
    "unsized virtual yield".split()
)
RUST_NOT_RAW_IDENTIFIERS = frozenset({"self", "Self", "super", "crate"})
