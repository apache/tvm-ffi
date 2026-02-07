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
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use tvm_ffi_stubgen::{run, Args};

#[test]
fn stubgen_tvm_ffi_testing() {
    let lib_dir = tvm_ffi_libdir().expect("tvm-ffi-config --libdir");
    let dlls = resolve_testing_dlls(&lib_dir).expect("unable to locate tvm_ffi testing libraries");
    let testing_lib = dlls
        .iter()
        .find(|path| {
            path.file_name()
                .map(|name| name.to_string_lossy().contains("tvm_ffi_testing"))
                .unwrap_or(false)
        })
        .cloned()
        .expect("tvm_ffi_testing library");
    let out_dir = unique_temp_dir("tvm_ffi_stubgen_test");
    let args = Args {
        out_dir: out_dir.clone(),
        dlls: vec![testing_lib.clone()],
        init_prefix: "testing".to_string(),
        init_crate: "tvm_ffi_testing_stub".to_string(),
        tvm_ffi_path: None,
        overwrite: true,
    };

    run(args).expect("stubgen run");

    let cargo_toml = out_dir.join("Cargo.toml");
    let functions_rs = out_dir.join("src").join("functions.rs");
    assert!(cargo_toml.exists(), "Cargo.toml not generated");
    assert!(functions_rs.exists(), "functions.rs not generated");

    let functions_body = fs::read_to_string(functions_rs).expect("read functions.rs");
    assert!(functions_body.contains("add_one"), "missing add_one stub");

    write_integration_test(&out_dir, &testing_lib).expect("write integration test");
    run_generated_tests(&out_dir, &lib_dir).expect("run generated tests");
}

fn resolve_testing_dlls(lib_dir: &Path) -> Result<Vec<PathBuf>, String> {
    if let Some(dlls) = dlls_from_dir(lib_dir) {
        return Ok(dlls);
    }
    Err("tvm-ffi-config --libdir did not contain tvm_ffi libraries".to_string())
}

fn dlls_from_dir(dir: &Path) -> Option<Vec<PathBuf>> {
    let tvm_ffi = dir.join(lib_filename("tvm_ffi"));
    let tvm_ffi_testing = dir.join(lib_filename("tvm_ffi_testing"));
    if tvm_ffi.exists() && tvm_ffi_testing.exists() {
        Some(vec![tvm_ffi, tvm_ffi_testing])
    } else {
        None
    }
}

fn lib_filename(name: &str) -> String {
    if cfg!(target_os = "windows") {
        format!("{}.dll", name)
    } else if cfg!(target_os = "macos") {
        format!("lib{}.dylib", name)
    } else {
        format!("lib{}.so", name)
    }
}

fn unique_temp_dir(prefix: &str) -> PathBuf {
    let base = env::temp_dir();
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    base.join(format!("{}_{}_{}", prefix, pid, nanos))
}

fn tvm_ffi_libdir() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let output = Command::new("tvm-ffi-config").arg("--libdir").output()?;
    if !output.status.success() {
        return Err("tvm-ffi-config --libdir failed".into());
    }
    let lib_dir = String::from_utf8(output.stdout)?.trim().to_string();
    if lib_dir.is_empty() {
        return Err("tvm-ffi-config returned empty libdir".into());
    }
    Ok(PathBuf::from(lib_dir))
}

fn write_integration_test(
    out_dir: &Path,
    testing_lib: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let tests_dir = out_dir.join("tests");
    fs::create_dir_all(&tests_dir)?;
    let test_body = format!(
        "use tvm_ffi_testing_stub::add_one;\n\n#[test]\nfn add_one_roundtrip() {{\n    let lib_path = \"{}\";\n    tvm_ffi::Module::load_from_file(lib_path).expect(\"load tvm_ffi_testing\");\n    let value = add_one(1).expect(\"call add_one\");\n    assert_eq!(value, 2);\n}}\n",
        testing_lib.display()
    );
    fs::write(tests_dir.join("integration.rs"), test_body)?;
    Ok(())
}

fn run_generated_tests(out_dir: &Path, lib_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = Command::new("cargo");
    cmd.arg("test")
        .arg("--manifest-path")
        .arg(out_dir.join("Cargo.toml"))
        .current_dir(out_dir);

    let ld_var = if cfg!(target_os = "windows") {
        "PATH"
    } else if cfg!(target_os = "macos") {
        "DYLD_LIBRARY_PATH"
    } else {
        "LD_LIBRARY_PATH"
    };

    let current_ld = env::var(ld_var).unwrap_or_default();
    let separator = if ld_var == "PATH" { ";" } else { ":" };
    let lib_dir_str = lib_dir.to_string_lossy();
    let new_ld = if current_ld.is_empty() {
        lib_dir_str.to_string()
    } else {
        format!("{}{}{}", lib_dir_str, separator, current_ld)
    };
    cmd.env(ld_var, new_ld);

    let path_value = env::var("PATH").unwrap_or_default();
    cmd.env("PATH", path_value);

    let status = cmd.status()?;
    if !status.success() {
        return Err("generated crate tests failed".into());
    }
    Ok(())
}
