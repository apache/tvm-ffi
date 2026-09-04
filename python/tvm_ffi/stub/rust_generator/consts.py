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
RUST_DIRECTIVE_KINDS = frozenset({"import-object", "field", "nullable", "enum"})

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
    "Map": "tvm_ffi::Map",  # the crate's own Map<K, V>, NOT HashMap
    # An object value is the `ObjectRef` handle; `Object` only appears as an embedded `base`.
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

#: Origins without a crate mirror; such a field is read as ``tvm_ffi::Any``.
RUST_UNSUPPORTED_ORIGINS = frozenset({"Dict", "List", "Union", "tuple"})

#: ``use``-path rewrites: builtin ``ffi.*`` type keys live at the crate root.
RUST_MOD_MAP = {
    "ffi": "tvm_ffi",
}

#: Root of the object hierarchy.
RUST_ROOT_TYPE_KEY = "ffi.Object"
