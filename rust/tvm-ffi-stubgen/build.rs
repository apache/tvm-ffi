// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::env;
use std::process::Command;

fn main() {
    let lib_dir = tvm_ffi_libdir();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    if target_os == "linux" || target_os == "macos" {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir);
    }

    let ld_var = match target_os.as_str() {
        "windows" => "PATH",
        "macos" => "DYLD_LIBRARY_PATH",
        "linux" => "LD_LIBRARY_PATH",
        _ => "",
    };
    if !ld_var.is_empty() {
        let current = env::var(ld_var).unwrap_or_default();
        let separator = if ld_var == "PATH" { ";" } else { ":" };
        let value = if current.is_empty() {
            lib_dir.clone()
        } else {
            format!("{}{}{}", lib_dir, separator, current)
        };
        println!("cargo:rustc-env={}={}", ld_var, value);
    }
}

fn tvm_ffi_libdir() -> String {
    let output = Command::new("tvm-ffi-config")
        .arg("--libdir")
        .output()
        .expect("tvm-ffi-config --libdir");
    if !output.status.success() {
        panic!("tvm-ffi-config --libdir failed");
    }
    let lib_dir = String::from_utf8(output.stdout)
        .expect("tvm-ffi-config output")
        .trim()
        .to_string();
    if lib_dir.is_empty() {
        panic!("tvm-ffi-config returned empty libdir");
    }
    lib_dir
}
