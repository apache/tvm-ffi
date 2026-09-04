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
//! Use the stubgen-generated `IntPair` and `IntRange` bindings (see ../../README.md).

mod generated;

use generated::rust_stubgen::{IntPair, IntRange, IntRangeObj, PairKind};
use tvm_ffi::{Error, Module, ObjectArc, ObjectRefCore, Result, VALUE_ERROR};

/// The hand-written constructor of `IntRange`: the `custom-new` directive in
/// the generated file keeps the generator from emitting one.
impl IntRange {
    pub fn new(begin: i64, extent: i32) -> Result<Self> {
        if extent < 0 {
            return Err(Error::new(VALUE_ERROR, "IntRange extent must not be negative", ""));
        }
        let data = ObjectArc::new(IntRangeObj::new(begin, extent));
        // SAFETY: `data` holds a freshly allocated `IntRangeObj`, the container type of `IntRange`.
        Ok(unsafe { Self::from_data(data) })
    }
}

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
    // Load the C++ library so the objects are registered with the FFI type
    // registry. Keep it alive for as long as the bindings are used.
    let _lib = Module::load_from_file(lib_path())?;

    // Both objects have a reproducible layout: they are allocated in Rust and
    // their fields are plain struct members, on both sides of the ABI.
    let pair = IntPair::new(1, 2, PairKind::Ordered);
    println!("a={} b={} kind={:?}", pair.a, pair.b, pair.kind);
    assert_eq!(pair.kind, PairKind::Ordered);

    let sum: i64 = tvm_ffi::cached_global_func!("rust_stubgen.IntPairSum")
        .call_tuple((pair.clone(),))?
        .try_into()?;
    println!("sum={sum}");
    assert_eq!(sum, 3);

    // `IntRange::new` is hand-written above (`custom-new`), so it can validate.
    let range = IntRange::new(10, 5)?;
    println!("begin={} extent={}", range.begin, range.extent);
    let end: i64 = tvm_ffi::cached_global_func!("rust_stubgen.IntRangeEnd")
        .call_tuple((range.clone(),))?
        .try_into()?;
    println!("end={end}");
    assert_eq!(end, 15);
    assert!(IntRange::new(0, -1).is_err());
    Ok(())
}
