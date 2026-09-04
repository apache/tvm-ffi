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
//! Use the stubgen-generated `IntPair` binding (see ../../README.md).

mod generated;

use generated::rust_stubgen::{IntPair, IntRange, PairKind};
use tvm_ffi::{Module, Result};

/// Path of the C++ shared library built by CMake into `../build`.
fn lib_path() -> String {
    let name = if cfg!(target_os = "windows") {
        "rust_stubgen.dll"
    } else if cfg!(target_os = "macos") {
        "librust_stubgen.dylib"
    } else {
        "librust_stubgen.so"
    };
    format!("{}/../build/{}", env!("CARGO_MANIFEST_DIR"), name)
}

fn main() -> Result<()> {
    // Load the C++ library so `IntPair` is registered with the FFI type registry.
    // Keep it alive for as long as the bindings are used.
    let _lib = Module::load_from_file(lib_path())?;

    // The object is opaque to Rust: it is constructed by the registered C++
    // function and its fields are read through the reflection getters.
    let pair: IntPair = tvm_ffi::cached_global_func!("rust_stubgen.IntPair")
        .call_tuple((1i64, 2i64, i64::from(PairKind::Ordered.as_raw())))?
        .try_into()?;
    println!("a={} b={} kind={:?}", pair.a()?, pair.b()?, pair.kind()?);
    assert_eq!(pair.kind()?, PairKind::Ordered);

    let sum: i64 = tvm_ffi::cached_global_func!("rust_stubgen.IntPairSum")
        .call_tuple((pair.clone(),))?
        .try_into()?;
    println!("sum={sum}");

    // `IntRange` has a reproducible layout: its fields are plain struct members.
    let range: IntRange = tvm_ffi::cached_global_func!("rust_stubgen.IntRange")
        .call_tuple((10i64, 5i64))?
        .try_into()?;
    println!("begin={} extent={}", range.begin, range.extent);
    assert_eq!(range.begin + i64::from(range.extent), 15);
    Ok(())
}
