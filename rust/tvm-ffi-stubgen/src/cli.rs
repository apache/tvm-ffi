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

use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "tvm-ffi-stubgen",
    about = "Generate Rust stubs from tvm-ffi metadata"
)]
pub struct Args {
    #[arg(value_name = "OUT_DIR")]
    pub out_dir: PathBuf,
    #[arg(long = "dlls", value_delimiter = ';', num_args = 1..)]
    pub dlls: Vec<PathBuf>,
    #[arg(long = "init-prefix")]
    pub init_prefix: String,
    #[arg(long = "init-crate")]
    pub init_crate: String,
    #[arg(long = "tvm-ffi-path")]
    pub tvm_ffi_path: Option<PathBuf>,
    #[arg(long = "overwrite")]
    pub overwrite: bool,
}
