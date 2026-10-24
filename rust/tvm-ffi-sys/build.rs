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
use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=DOCS_RS");
    if env::var_os("DOCS_RS").is_some() {
        return;
    }

    let source = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap()).join("../..");
    for path in [
        "3rdparty",
        "CMakeLists.txt",
        "cmake/Utils",
        "include",
        "src/ffi",
    ] {
        println!("cargo:rerun-if-changed={}", source.join(path).display());
    }

    let testing = env::var_os("CARGO_FEATURE_TESTING").is_some();
    let mut config = cmake::Config::new(&source);
    let profile = config.get_profile().to_owned();
    let dst: PathBuf = config
        .define("TVM_FFI_BUILD_PYTHON_MODULE", "OFF")
        .define("TVM_FFI_BUILD_TESTS", "OFF")
        .build_target(if testing {
            "tvm_ffi_testing"
        } else {
            "tvm_ffi_shared"
        })
        .build();
    let lib_dir = dst.join("build/lib");
    // Multi-config generators append the selected configuration to the output directory.
    let configured_dir = lib_dir.join(profile);
    let lib_dir = if configured_dir.is_dir() {
        configured_dir
    } else {
        lib_dir
    };

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=tvm_ffi");
    if testing {
        println!("cargo:rustc-link-lib=dylib=tvm_ffi_testing");
    }
}
