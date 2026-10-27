/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

//! A library whose static init block registers a global function when the library loads, as a
//! C++ library does in a `TVM_FFI_STATIC_INIT_BLOCK`. Cargo builds it as a `cdylib`, which
//! `tests/python/test_rust_static_init.py` loads with `tvm_ffi.load_module`.

use tvm_ffi::{Function, Result};

fn add_one(x: i64) -> Result<i64> {
    Ok(x + 1)
}

tvm_ffi::static_init_block! {
    Function::register_global_typed("static_init_example.add_one", add_one, "Add one to an integer.")?;
    Ok(())
}

// The errors of the blocks that failed when the library loaded, which Python raises as
// `module.run_static_init_blocks()`.
tvm_ffi::tvm_ffi_dll_export_typed_func!(run_static_init_blocks, tvm_ffi::run_static_init_blocks);
