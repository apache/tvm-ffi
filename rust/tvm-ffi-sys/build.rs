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
use std::process::Command;

/// Update the LD_LIBRARY_PATH environment variable
/// so cargo run/test can directly pick it up
/// note that it won't always work later consumption of the
/// library, and we still need to figure out linking by setting LD_LIBRARY_PATH
fn update_ld_library_path(lib_dir: &str) {
    let os_env_var = match env::var("CARGO_CFG_TARGET_OS").as_deref() {
        Ok("windows") => "PATH",
        Ok("macos") => "DYLD_LIBRARY_PATH",
        Ok("linux") => "LD_LIBRARY_PATH",
        _ => "",
    };
    if os_env_var.is_empty() {
        return;
    }
    // Get the current value of the environment variable at build time (if any)
    let current_val = env::var(os_env_var).unwrap_or_else(|_| String::new());
    // Use platform-specific separator
    let separator = if os_env_var == "PATH" { ";" } else { ":" };
    let new_ld_path = if current_val.is_empty() {
        lib_dir.to_string()
    } else {
        format!("{}{}{}", current_val, separator, lib_dir)
    };
    // this env is only used for cargo run/test
    println!("cargo:rustc-env={}={}", os_env_var, new_ld_path);
}

fn main() {
    // When building documentation, we may not need actual linking
    // Check if this is a rustdoc build
    let is_rustdoc = env::var("RUSTDOC").is_ok() || env::var("CARGO_CFG_DOC").is_ok();

    // Run `mylib-config --libdir` to get the library path
    let config_result = Command::new("tvm-ffi-config").arg("--libdir").output();

    let lib_dir = match config_result {
        Ok(output) if output.status.success() => String::from_utf8(output.stdout)
            .unwrap_or_default()
            .trim()
            .to_string(),
        _ if is_rustdoc => {
            // For rustdoc builds, we can proceed without the library
            eprintln!("Warning: tvm-ffi-config not available, skipping for documentation build");
            return;
        }
        _ => {
            panic!("Failed to run tvm-ffi-config. Make sure tvm-ffi is installed.");
        }
    };

    if lib_dir.is_empty() {
        if is_rustdoc {
            eprintln!(
                "Warning: Empty lib_dir from tvm-ffi-config, skipping for documentation build"
            );
            return;
        }
        panic!("tvm-ffi-config returned empty library path");
    }

    // add the library directory to the linker search path
    println!("cargo:rustc-link-search=native={}", lib_dir);
    // link the library
    println!("cargo:rustc-link-lib=dylib=tvm_ffi");
    println!("cargo:rustc-link-lib=dylib=tvm_ffi_testing");
    // update the LD_LIBRARY_PATH environment variable
    update_ld_library_path(&lib_dir);
}
