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

mod cli;
mod ffi;
mod generate;
mod model;
mod schema;
mod utils;

pub use cli::Args;

pub fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    let prefix = utils::normalize_prefix(&args.init_prefix);
    if args.dlls.is_empty() {
        return Err("--dlls is required".into());
    }
    utils::ensure_out_dir(&args.out_dir, args.overwrite)?;

    let _loaded_libs = ffi::load_dlls(&args.dlls)?;

    let global_funcs = ffi::list_global_function_names()?;
    let filtered_funcs: Vec<String> = global_funcs
        .into_iter()
        .filter(|name| name.starts_with(&prefix))
        .collect();

    let type_keys = ffi::list_registered_type_keys()?;
    let filtered_types: Vec<String> = type_keys
        .into_iter()
        .filter(|name| name.starts_with(&prefix))
        .collect();

    let type_map = generate::build_type_map(&filtered_types, &prefix);
    let functions = generate::build_function_entries(&filtered_funcs, &type_map, &prefix)?;
    let types = generate::build_type_entries(&filtered_types, &type_map, &prefix)?;

    let functions_root = generate::build_function_modules(functions, &prefix);
    let types_root = generate::build_type_modules(types, &prefix);

    let cargo_toml = generate::render_cargo_toml(&args, &type_map)?;
    let lib_rs = generate::render_lib_rs();
    let functions_rs = generate::render_functions_rs(&functions_root);
    let types_rs = generate::render_types_rs(&types_root);

    let src_dir = args.out_dir.join("src");
    std::fs::create_dir_all(&src_dir)?;
    std::fs::write(args.out_dir.join("Cargo.toml"), cargo_toml)?;
    std::fs::write(src_dir.join("lib.rs"), lib_rs)?;
    std::fs::write(src_dir.join("functions.rs"), functions_rs)?;
    std::fs::write(src_dir.join("types.rs"), types_rs)?;

    Ok(())
}
