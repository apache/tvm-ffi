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

fn generate_example_lib() {
    let is_example_build = env::var("CARGO_FEATURE_EXAMPLE").is_ok();
    if !is_example_build {
        return;
    }
    println!("Running optional build step to generate example library");
    let output_dir = env::var("OUT_DIR").unwrap();
    let _ = Command::new("python")
        .arg("scripts/generate_example_lib.py")
        .arg(output_dir)
        .output()
        .expect("Failed to generate example library");
    println!("cargo:rerun-if-changed=scripts/generate_example_lib.py");
}

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
    // Run `mylib-config --libdir` to get the library path
    let config_output = Command::new("tvm-ffi-config")
        .arg("--libdir")
        .output()
        .expect("Failed to run tvm-ffi-config");
    let lib_dir = String::from_utf8(config_output.stdout)
        .expect("Invalid UTF-8 output from tvm-ffi-config")
        .trim()
        .to_string();
    // update the LD_LIBRARY_PATH environment variable
    // note that we will also need to update ld_library_path for
    // the cases here besides the tvm-ffi-sys crate so cargo test works out of the box
    update_ld_library_path(&lib_dir);
    // generate the example library
    generate_example_lib();
}
