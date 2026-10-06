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
    println!("cargo:rerun-if-env-changed={}", os_env_var);
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
    // docs.rs builds the documentation without tvm-ffi installed, and so does
    // docs/conf.py, which sets DOCS_RS as docs.rs does. Cargo sets RUSTDOC for
    // every build script, so it cannot tell a documentation build apart.
    let docs_only = env::var_os("DOCS_RS").is_some();
    println!("cargo:rerun-if-env-changed=DOCS_RS");
    // The library directory comes from the tvm-ffi-config found on PATH.
    println!("cargo:rerun-if-env-changed=PATH");
    println!("cargo:rerun-if-changed=build.rs");

    // Run `tvm-ffi-config --libdir` to get the library path
    let found = match Command::new("tvm-ffi-config").arg("--libdir").output() {
        Ok(output) if output.status.success() => {
            let lib_dir = String::from_utf8(output.stdout)
                .unwrap_or_default()
                .trim()
                .to_string();
            if lib_dir.is_empty() {
                Err("`tvm-ffi-config --libdir` printed no library directory".to_string())
            } else {
                Ok(lib_dir)
            }
        }
        Ok(output) => Err(format!(
            "`tvm-ffi-config --libdir` failed ({}): {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        )),
        Err(err) => Err(format!("could not run `tvm-ffi-config`: {err}")),
    };
    let lib_dir = match found {
        Ok(lib_dir) => lib_dir,
        Err(reason) if docs_only => {
            println!(
                "cargo:warning={reason}; not setting the loader path for a documentation build"
            );
            return;
        }
        Err(reason) => panic!(
            "{reason}. tvm-ffi sets the dynamic loader path for `cargo run` and `cargo test` \
             from the directory that `tvm-ffi-config --libdir` prints: install tvm-ffi (e.g. \
             `pip install apache-tvm-ffi`) and put tvm-ffi-config on PATH. To build \
             documentation only, set DOCS_RS=1."
        ),
    };

    // Let `cargo run` and `cargo test` in this workspace find libtvm_ffi. Cargo only
    // forwards a package's own build-script environment to the binaries it runs, so
    // the tvm-ffi-sys build script cannot do this for us, and downstream crates set
    // the loader path themselves (see ../README.md).
    update_ld_library_path(&lib_dir);
    // generate the example library
    generate_example_lib();
}
