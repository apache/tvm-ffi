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
//! `tvm-ffi-sys`'s build script links libtvm_ffi, but the loader-path
//! environment it emits only applies to its own package; re-emit it here
//! so plain `cargo run` finds libtvm_ffi at startup.

use std::env;
use std::process::Command;

fn main() {
    let output = Command::new("tvm-ffi-config")
        .arg("--libdir")
        .output()
        .expect("failed to run tvm-ffi-config; install tvm-ffi and activate the virtualenv");
    assert!(output.status.success(), "tvm-ffi-config --libdir failed");
    let lib_dir = String::from_utf8(output.stdout).unwrap().trim().to_string();

    let loader_var = match env::var("CARGO_CFG_TARGET_OS").as_deref() {
        Ok("windows") => "PATH",
        Ok("macos") => "DYLD_LIBRARY_PATH",
        _ => "LD_LIBRARY_PATH",
    };
    let sep = if loader_var == "PATH" { ";" } else { ":" };
    let prev = env::var(loader_var).unwrap_or_default();
    let val = if prev.is_empty() {
        lib_dir
    } else {
        format!("{prev}{sep}{lib_dir}")
    };
    println!("cargo:rustc-env={loader_var}={val}");
}
