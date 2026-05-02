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

use std::fs;
use std::path::{Path, PathBuf};

pub(crate) fn normalize_prefix(prefix: &str) -> String {
    if prefix.is_empty() {
        return String::new();
    }
    if prefix.ends_with('.') {
        prefix.to_string()
    } else {
        format!("{}.", prefix)
    }
}

pub(crate) fn ensure_out_dir(
    out_dir: &Path,
    overwrite: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if out_dir.exists() {
        let mut has_entries = false;
        for entry in fs::read_dir(out_dir)? {
            let entry = entry?;
            if entry.file_name() != "." && entry.file_name() != ".." {
                has_entries = true;
                break;
            }
        }
        if has_entries && !overwrite {
            return Err("output directory is not empty (use --overwrite to proceed)".into());
        }
    } else {
        fs::create_dir_all(out_dir)?;
    }
    Ok(())
}

pub(crate) fn default_tvm_ffi_path() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let current = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidate = current.join("../tvm-ffi");
    if candidate.exists() {
        return Ok(candidate);
    }
    Err("unable to locate tvm-ffi path (use --tvm-ffi-path)".into())
}
