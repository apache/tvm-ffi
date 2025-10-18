..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

Rust API
========

This page contains the API reference for the Rust API. The tvm-ffi project provides
Rust bindings that allow you to use the TVM FFI from Rust programs.

Crates
------

The Rust API is organized into three crates:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Crate
     - Description
   * - 📦 `tvm-ffi <generated/tvm_ffi/index.html>`_
     - High-level Rust bindings with safe abstractions for FFI objects, functions, tensors, and containers.
   * - 📦 `tvm-ffi-sys <generated/tvm_ffi_sys/index.html>`_
     - Low-level unsafe bindings to the C API.
   * - 📦 `tvm-ffi-macros <generated/tvm_ffi_macros/index.html>`_
     - Procedural macros for deriving Object and ObjectRef traits.
