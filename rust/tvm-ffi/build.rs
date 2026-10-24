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

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=DOCS_RS");
    if env::var_os("DOCS_RS").is_none() {
        generate_example_lib();
    }
}
