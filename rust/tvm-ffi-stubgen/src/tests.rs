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

use super::{run, Args};
use crate::utils;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

#[test]
fn stubgen_tvm_ffi_testing() {
    let dlls = resolve_testing_dlls().expect("unable to locate tvm_ffi testing libraries");
    let out_dir = unique_temp_dir("tvm_ffi_stubgen_test");
    let args = Args {
        out_dir: out_dir.clone(),
        dlls: dlls.clone(),
        init_prefix: "testing".to_string(),
        init_crate: "tvm_ffi_testing_stub".to_string(),
        tvm_ffi_path: Some(utils::default_tvm_ffi_path().expect("tvm-ffi path")),
        overwrite: true,
    };

    run(args).expect("stubgen run");

    let cargo_toml = out_dir.join("Cargo.toml");
    let functions_rs = out_dir.join("src").join("functions.rs");
    assert!(cargo_toml.exists(), "Cargo.toml not generated");
    assert!(functions_rs.exists(), "functions.rs not generated");

    let functions_body = fs::read_to_string(functions_rs).expect("read functions.rs");
    assert!(functions_body.contains("add_one"), "missing add_one stub");

    write_integration_test(&out_dir).expect("write integration test");
    run_generated_tests(&out_dir, &dlls).expect("run generated tests");
}

fn resolve_testing_dlls() -> Result<Vec<PathBuf>, String> {
    if let Ok(value) = env::var("TVM_FFI_TESTING_DLLS") {
        let dlls = split_paths(&value);
        if !dlls.is_empty() {
            return Ok(dlls);
        }
    }

    if let Ok(dir) = env::var("TVM_FFI_TESTING_LIB_DIR") {
        let dir = PathBuf::from(dir);
        if let Some(dlls) = dlls_from_dir(&dir) {
            return Ok(dlls);
        }
    }

    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let tvm_ffi_root = manifest.join("../tvm-ffi");
    let mut candidates = vec![tvm_ffi_root.join("build/lib")];

    if let Ok(venv) = env::var("VIRTUAL_ENV") {
        if let Some(path) = find_venv_lib_dir(Path::new(&venv)) {
            candidates.push(path);
        }
    }

    for dir in candidates {
        if let Some(dlls) = dlls_from_dir(&dir) {
            return Ok(dlls);
        }
    }

    Err("set TVM_FFI_TESTING_DLLS or TVM_FFI_TESTING_LIB_DIR to run tests".to_string())
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

fn split_paths(value: &str) -> Vec<PathBuf> {
    let normalized = value.replace(';', ":");
    normalized
        .split(':')
        .filter(|item| !item.trim().is_empty())
        .map(PathBuf::from)
        .collect()
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

fn find_venv_lib_dir(venv: &Path) -> Option<PathBuf> {
    for lib_root in ["lib", "lib64"] {
        let base = venv.join(lib_root);
        let entries = fs::read_dir(&base).ok()?;
        for entry in entries.flatten() {
            let path = entry.path();
            let name = path.file_name()?.to_string_lossy();
            if !name.starts_with("python") {
                continue;
            }
            let candidate = path.join("site-packages").join("tvm_ffi").join("lib");
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }
    None
}

fn write_integration_test(out_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let tests_dir = out_dir.join("tests");
    fs::create_dir_all(&tests_dir)?;
    let test_body = r#"use tvm_ffi_testing_stub::add_one;

#[test]
fn add_one_roundtrip() {
    let lib_dir = std::env::var("TVM_FFI_TESTING_LIB_DIR").expect("lib dir");
    let lib_path = format!("{}/{}", lib_dir, lib_filename("tvm_ffi_testing"));
    tvm_ffi::Module::load_from_file(&lib_path).expect("load tvm_ffi_testing");
    let value = add_one(1).expect("call add_one");
    assert_eq!(value, 2);
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
"#;
    fs::write(tests_dir.join("integration.rs"), test_body)?;
    Ok(())
}

fn run_generated_tests(out_dir: &Path, dlls: &[PathBuf]) -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = Command::new("cargo");
    cmd.arg("test")
        .arg("--manifest-path")
        .arg(out_dir.join("Cargo.toml"))
        .current_dir(out_dir);

    let lib_dir = dlls
        .get(0)
        .and_then(|path| path.parent())
        .map(|path| path.to_path_buf())
        .ok_or("missing library directory")?;

    let ld_var = if cfg!(target_os = "windows") {
        "PATH"
    } else if cfg!(target_os = "macos") {
        "DYLD_LIBRARY_PATH"
    } else {
        "LD_LIBRARY_PATH"
    };

    let current_ld = env::var(ld_var).unwrap_or_default();
    let separator = if ld_var == "PATH" { ";" } else { ":" };
    let new_ld = if current_ld.is_empty() {
        lib_dir.to_string_lossy().to_string()
    } else {
        format!("{}{}{}", lib_dir.to_string_lossy(), separator, current_ld)
    };
    cmd.env(ld_var, new_ld);
    cmd.env("TVM_FFI_TESTING_LIB_DIR", lib_dir);

    let mut path_value = env::var("PATH").unwrap_or_default();
    if let Ok(venv) = env::var("VIRTUAL_ENV") {
        let venv_bin = Path::new(&venv).join("bin");
        let venv_str = venv_bin.to_string_lossy();
        if !path_value.split(':').any(|item| item == venv_str) {
            if !path_value.is_empty() {
                path_value = format!("{}:{}", venv_str, path_value);
            } else {
                path_value = venv_str.to_string();
            }
        }
    }
    cmd.env("PATH", path_value);

    let status = cmd.status()?;
    if !status.success() {
        return Err("generated crate tests failed".into());
    }
    Ok(())
}
