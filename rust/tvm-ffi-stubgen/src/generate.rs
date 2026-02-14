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

use crate::cli::Args;
use crate::ffi;
use crate::model::{
    FieldGen, FunctionGen, FunctionSig, GetterSpec, MethodGen, ModuleNode, RustType, TypeGen,
};
use crate::repr_c;
use crate::schema::{extract_type_schema, parse_type_schema, TypeSchema};
use crate::utils;
use std::collections::BTreeMap;
use std::fmt::Write as _;
use toml::value::Table;

const METHOD_FLAG_STATIC: i64 = 1 << 2;

pub(crate) fn build_type_map(type_keys: &[String], prefix: &str) -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();
    for key in type_keys {
        let (mods, name) = split_name(key, prefix);
        let rust_name = sanitize_ident(&name, IdentStyle::Type);
        let module_path = module_path(&mods);
        let path = if module_path.is_empty() {
            format!("crate::{}", rust_name)
        } else {
            format!("crate::{}::{}", module_path, rust_name)
        };
        map.insert(key.clone(), path);
    }
    map
}

pub(crate) fn build_function_entries(
    func_names: &[String],
    type_map: &BTreeMap<String, String>,
    prefix: &str,
) -> tvm_ffi::Result<Vec<(Vec<String>, FunctionGen)>> {
    let mut out = Vec::new();
    for full_name in func_names {
        let metadata = ffi::get_global_func_metadata(full_name)?;
        let schema = metadata
            .and_then(|meta| extract_type_schema(&meta))
            .and_then(|schema| parse_type_schema(&schema));
        let sig = build_function_sig(schema.as_ref(), type_map, None);
        let (mods, name) = split_name(full_name, prefix);
        let rust_name = sanitize_ident(&name, IdentStyle::Function);
        out.push((
            mods,
            FunctionGen {
                full_name: full_name.clone(),
                rust_name,
                sig,
            },
        ));
    }
    Ok(out)
}

pub(crate) fn build_type_entries(
    type_keys: &[String],
    type_map: &BTreeMap<String, String>,
    prefix: &str,
) -> tvm_ffi::Result<Vec<(Vec<String>, TypeGen)>> {
    let mut out = Vec::new();
    for key in type_keys {
        let (mods, name) = split_name(key, prefix);
        let rust_name = sanitize_ident(&name, IdentStyle::Type);
        let mut methods = Vec::new();
        let mut fields = Vec::new();
        let mut type_depth = 0i32;
        let repr_c_info = repr_c::check_repr_c(key, type_map);
        if let Some(info) = ffi::get_type_info(key) {
            type_depth = info.type_depth;
            if info.num_methods > 0 && !info.methods.is_null() {
                let method_slice =
                    unsafe { std::slice::from_raw_parts(info.methods, info.num_methods as usize) };
                for method in method_slice {
                    let method_name = match ffi::byte_array_to_string_opt(&method.name) {
                        Some(name) => name,
                        None => continue,
                    };
                    let rust_method_name = map_method_name(&method_name);
                    let is_static = (method.flags & METHOD_FLAG_STATIC) != 0;
                    let meta = ffi::byte_array_to_string_opt(&method.metadata);
                    let schema = meta
                        .as_deref()
                        .and_then(extract_type_schema)
                        .and_then(|s| parse_type_schema(&s));
                    let sig =
                        build_method_sig(schema.as_ref(), type_map, Some(key.as_str()), is_static);
                    let full_name = format!("{}.{}", key, method_name);
                    methods.push(MethodGen {
                        full_name,
                        rust_name: rust_method_name,
                        sig,
                        is_static,
                    });
                }
            }
            if info.num_fields > 0 && !info.fields.is_null() {
                let field_slice =
                    unsafe { std::slice::from_raw_parts(info.fields, info.num_fields as usize) };
                for field in field_slice {
                    let field_name = match ffi::byte_array_to_string_opt(&field.name) {
                        Some(name) => name,
                        None => continue,
                    };
                    let rust_field_name = sanitize_ident(&field_name, IdentStyle::Function);
                    let meta = ffi::byte_array_to_string_opt(&field.metadata);
                    let schema = meta
                        .as_deref()
                        .and_then(extract_type_schema)
                        .and_then(|s| parse_type_schema(&s));
                    let ty = match schema.as_ref() {
                        Some(schema) => rust_type_for_schema(schema, type_map, Some(key.as_str())),
                        None => RustType::unsupported("tvm_ffi::Any"),
                    };
                    fields.push(FieldGen {
                        name: field_name,
                        rust_name: rust_field_name,
                        ty,
                    });
                }
            }
        }
        out.push((
            mods,
            TypeGen {
                type_key: key.clone(),
                rust_name: rust_name.clone(),
                methods,
                fields,
                type_depth,
                repr_c_info: repr_c_info.clone(),
                getter_specs: Vec::new(),
                ancestor_chain: Vec::new(),
            },
        ));
    }
    // Second pass: fill getter_specs and ancestor_chain for repr_c types in dependency order (base before derived).
    let mut type_key_to_idx: BTreeMap<String, usize> = BTreeMap::new();
    for (idx, (_, ty)) in out.iter().enumerate() {
        type_key_to_idx.insert(ty.type_key.clone(), idx);
    }
    let mut order: Vec<usize> = (0..out.len()).collect();
    order.sort_by_key(|&i| out[i].1.type_depth);
    for &idx in &order {
        let (_, ref ty) = out[idx];
        let repr_c_info = match &ty.repr_c_info {
            Some(r) => r,
            None => continue,
        };
        let parent_specs: Vec<GetterSpec> =
            if let Some(ref parent_key) = repr_c_info.parent_type_key {
                let parent_idx = *type_key_to_idx.get(parent_key).unwrap_or(&idx);
                out[parent_idx].1.getter_specs.clone()
            } else {
                Vec::new()
            };
        let getter_specs = build_getter_specs(&ty.type_key, &ty.repr_c_info, &parent_specs);

        // Build ancestor chain: [DirectParent, Grandparent, ..., ObjectRef]
        let ancestor_chain = if let Some(ref parent_key) = repr_c_info.parent_type_key {
            if parent_key == "ffi.Object" {
                vec!["tvm_ffi::object::ObjectRef".to_string()]
            } else if let Some(parent_rust) = type_map.get(parent_key) {
                let parent_idx = *type_key_to_idx.get(parent_key).unwrap_or(&idx);
                let mut chain = vec![parent_rust.clone()];
                // Inherit parent's ancestors
                chain.extend(out[parent_idx].1.ancestor_chain.clone());
                chain
            } else {
                vec!["tvm_ffi::object::ObjectRef".to_string()]
            }
        } else {
            vec!["tvm_ffi::object::ObjectRef".to_string()]
        };

        out[idx].1.getter_specs = getter_specs;
        out[idx].1.ancestor_chain = ancestor_chain;
    }
    Ok(out)
}

fn build_getter_specs(
    _type_key: &str,
    repr_c_info: &Option<repr_c::ReprCInfo>,
    parent_specs: &[GetterSpec],
) -> Vec<GetterSpec> {
    let info = match repr_c_info {
        Some(i) => i,
        None => return Vec::new(),
    };
    let mut specs = Vec::new();
    for parent in parent_specs {
        let access_expr = if parent.access_expr.starts_with("self.data.") {
            format!(
                "self.data.parent.{}",
                &parent.access_expr["self.data.".len()..]
            )
        } else {
            parent.access_expr.clone()
        };
        specs.push(GetterSpec {
            method_name: parent.method_name.clone(),
            access_expr,
            ret_type: parent.ret_type.clone(),
        });
    }
    for f in &info.direct_fields {
        let method_name = format!("get_{}", f.rust_name);
        let access_expr = if f.is_pod {
            format!("self.data.{}", f.rust_name)
        } else {
            format!("self.data.{}.clone()", f.rust_name)
        };
        specs.push(GetterSpec {
            method_name,
            access_expr,
            ret_type: f.rust_type.clone(),
        });
    }
    specs
}

pub(crate) fn build_function_modules(
    funcs: Vec<(Vec<String>, FunctionGen)>,
    _prefix: &str,
) -> ModuleNode {
    let mut root = ModuleNode::default();
    for (mods, func) in funcs {
        insert_function(&mut root, &mods, func);
    }
    root
}

pub(crate) fn build_type_modules(types: Vec<(Vec<String>, TypeGen)>, _prefix: &str) -> ModuleNode {
    let mut root = ModuleNode::default();
    for (mods, ty) in types {
        insert_type(&mut root, &mods, ty);
    }
    root
}

fn build_function_sig(
    schema: Option<&TypeSchema>,
    type_map: &BTreeMap<String, String>,
    self_type_key: Option<&str>,
) -> FunctionSig {
    match schema {
        None => FunctionSig::packed(),
        Some(schema) if schema.origin != "ffi.Function" => FunctionSig::packed(),
        Some(schema) if schema.args.is_empty() => FunctionSig::packed(),
        Some(schema) => {
            let ret = rust_type_for_schema(&schema.args[0], type_map, self_type_key);
            let args: Vec<RustType> = schema.args[1..]
                .iter()
                .map(|arg| rust_type_for_schema(arg, type_map, self_type_key))
                .collect();
            FunctionSig::from_types(args, ret)
        }
    }
}

fn build_method_sig(
    schema: Option<&TypeSchema>,
    type_map: &BTreeMap<String, String>,
    self_type_key: Option<&str>,
    is_static: bool,
) -> FunctionSig {
    if !is_static {
        return FunctionSig::packed();
    }
    build_function_sig(schema, type_map, self_type_key)
}

fn rust_type_for_schema(
    schema: &TypeSchema,
    type_map: &BTreeMap<String, String>,
    _self_type_key: Option<&str>,
) -> RustType {
    match schema.origin.as_str() {
        "None" => RustType::supported("()"),
        "bool" => RustType::supported("bool"),
        "int" => RustType::supported("i64"),
        "float" => RustType::supported("f64"),
        "Device" => RustType::supported("tvm_ffi::DLDevice"),
        "DataType" => RustType::supported("tvm_ffi::DLDataType"),
        "ffi.String" | "std::string" | "const char*" | "ffi.SmallStr" => {
            RustType::supported("tvm_ffi::String")
        }
        "ffi.Bytes" | "TVMFFIByteArray*" | "ffi.SmallBytes" => {
            RustType::supported("tvm_ffi::Bytes")
        }
        "ffi.Function" => RustType::supported("tvm_ffi::Function"),
        "ffi.Object" => RustType::supported("tvm_ffi::object::ObjectRef"),
        "ffi.Tensor" | "DLTensor*" => RustType::supported("tvm_ffi::Tensor"),
        "ffi.Shape" => RustType::supported("tvm_ffi::Shape"),
        "ffi.Module" => RustType::supported("tvm_ffi::Module"),
        "Optional" => match schema.args.as_slice() {
            [inner] => {
                let inner_ty = rust_type_for_schema(inner, type_map, _self_type_key);
                if inner_ty.supported {
                    RustType::supported(&format!("Option<{}>", inner_ty.name))
                } else {
                    RustType::unsupported("tvm_ffi::Any")
                }
            }
            _ => RustType::unsupported("tvm_ffi::Any"),
        },
        "ffi.Array" => match schema.args.as_slice() {
            [inner] => {
                let inner_ty = rust_type_for_schema(inner, type_map, _self_type_key);
                if inner_ty.supported {
                    RustType::supported(&format!("tvm_ffi::Array<{}>", inner_ty.name))
                } else {
                    RustType::unsupported("tvm_ffi::Any")
                }
            }
            _ => RustType::unsupported("tvm_ffi::Any"),
        },
        "ffi.Map" => match schema.args.as_slice() {
            [key, value] => {
                let key_ty = rust_type_for_schema(key, type_map, _self_type_key);
                let value_ty = rust_type_for_schema(value, type_map, _self_type_key);
                if key_ty.supported && value_ty.supported {
                    RustType::supported(&format!(
                        "tvm_ffi::Map<{}, {}>",
                        key_ty.name, value_ty.name
                    ))
                } else {
                    RustType::unsupported("tvm_ffi::Any")
                }
            }
            _ => RustType::unsupported("tvm_ffi::Any"),
        },
        "Any" | "ffi.Any" => RustType::supported("tvm_ffi::AnyValue"),
        "Union" | "Variant" | "tuple" | "list" | "dict" => RustType::unsupported("tvm_ffi::Any"),
        other => match type_map.get(other) {
            Some(path) => RustType::object_wrapper(path),
            None => RustType::unsupported("tvm_ffi::Any"),
        },
    }
}

fn insert_function(root: &mut ModuleNode, mods: &[String], func: FunctionGen) {
    let mut node = root;
    for module in mods {
        node = node
            .children
            .entry(module.clone())
            .or_insert_with(|| ModuleNode {
                name: module.clone(),
                ..ModuleNode::default()
            });
    }
    node.functions.push(func);
}

fn insert_type(root: &mut ModuleNode, mods: &[String], ty: TypeGen) {
    let mut node = root;
    for module in mods {
        node = node
            .children
            .entry(module.clone())
            .or_insert_with(|| ModuleNode {
                name: module.clone(),
                ..ModuleNode::default()
            });
    }
    node.types.push(ty);
}

fn split_name(full_name: &str, prefix: &str) -> (Vec<String>, String) {
    let remainder = if prefix.is_empty() {
        full_name
    } else {
        full_name.strip_prefix(prefix).unwrap_or(full_name)
    };
    let parts: Vec<&str> = remainder.split('.').filter(|p| !p.is_empty()).collect();
    if parts.is_empty() {
        return (Vec::new(), "ffi".to_string());
    }
    if parts.len() == 1 {
        return (Vec::new(), parts[0].to_string());
    }
    let mut mods = Vec::new();
    for part in &parts[..parts.len() - 1] {
        mods.push(sanitize_ident(part, IdentStyle::Module));
    }
    (mods, parts[parts.len() - 1].to_string())
}

fn module_path(mods: &[String]) -> String {
    if mods.is_empty() {
        return String::new();
    }
    mods.join("::")
}

pub(crate) fn render_cargo_toml(
    args: &Args,
    _type_map: &BTreeMap<String, String>,
) -> Result<String, Box<dyn std::error::Error>> {
    let tvm_ffi_path = match &args.tvm_ffi_path {
        Some(path) => path.clone(),
        None => utils::default_tvm_ffi_path()?,
    };
    let tvm_ffi_path = tvm_ffi_path.canonicalize().unwrap_or_else(|_| tvm_ffi_path);
    let tvm_ffi_path_str = tvm_ffi_path.to_string_lossy().to_string();

    let mut package = Table::new();
    package.insert(
        "name".to_string(),
        toml::Value::String(args.init_crate.clone()),
    );
    package.insert(
        "version".to_string(),
        toml::Value::String("0.1.0".to_string()),
    );
    package.insert(
        "edition".to_string(),
        toml::Value::String("2021".to_string()),
    );

    let mut tvm_ffi = Table::new();
    tvm_ffi.insert("path".to_string(), toml::Value::String(tvm_ffi_path_str));

    let mut dependencies = Table::new();
    dependencies.insert("tvm-ffi".to_string(), toml::Value::Table(tvm_ffi));

    let mut doc = Table::new();
    doc.insert("package".to_string(), toml::Value::Table(package));
    doc.insert("dependencies".to_string(), toml::Value::Table(dependencies));

    Ok(toml::to_string(&toml::Value::Table(doc))?)
}

pub(crate) fn render_lib_rs(functions_root: &ModuleNode, types_root: &ModuleNode) -> String {
    let mut out = String::new();
    out.push_str(
        r#"pub mod _tvm_ffi_stubgen_detail {
    pub mod functions;
    pub mod types;
}

"#,
    );
    render_facade_module(
        &mut out,
        Some(functions_root),
        Some(types_root),
        &[],
        0,
        true,
    );
    out.push_str(
        r#"
pub fn load_library(path: &str) -> tvm_ffi::Result<tvm_ffi::Module> {
    tvm_ffi::Module::load_from_file(path)
}
"#,
    );
    out
}

pub(crate) fn render_build_rs() -> String {
    let mut out = String::new();
    out.push_str(
        r#"use std::env;
use std::process::Command;

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
    let current_val = env::var(os_env_var).unwrap_or_else(|_| String::new());
    let separator = if os_env_var == "PATH" { ";" } else { ":" };
    let new_ld_path = if current_val.is_empty() {
        lib_dir.to_string()
    } else {
        format!("{}{}{}", current_val, separator, lib_dir)
    };
    println!("cargo:rustc-env={}={}", os_env_var, new_ld_path);
}

fn main() {
    let output = Command::new("tvm-ffi-config")
        .arg("--libdir")
        .output()
        .expect("Failed to run tvm-ffi-config");
    if !output.status.success() {
        panic!("tvm-ffi-config --libdir failed");
    }
    let lib_dir = String::from_utf8(output.stdout)
        .expect("Invalid UTF-8 output from tvm-ffi-config")
        .trim()
        .to_string();
    if lib_dir.is_empty() {
        panic!("tvm-ffi-config returned empty library path");
    }
    println!("cargo:rustc-link-search=native={}", lib_dir);
    println!("cargo:rustc-link-lib=dylib=tvm_ffi");
    update_ld_library_path(&lib_dir);
}
"#,
    );
    out
}

pub(crate) fn render_functions_rs(root: &ModuleNode) -> String {
    let mut out = String::new();
    out.push_str(
        r#"#![allow(unused_imports)]
#![allow(non_snake_case, nonstandard_style)]

use std::sync::LazyLock;
use tvm_ffi::{Any, AnyView, Function, Result};

"#,
    );
    render_function_module(&mut out, root, 0);
    out
}

pub(crate) fn render_types_rs(root: &ModuleNode, type_map: &BTreeMap<String, String>) -> String {
    let mut out = String::new();
    out.push_str(
        r#"#![allow(unused_imports)]
#![allow(non_snake_case, nonstandard_style)]

use std::sync::LazyLock;
use tvm_ffi::{Any, AnyView, ObjectArc, Result};

"#,
    );
    render_type_module(&mut out, root, 0, type_map);
    out
}

fn render_facade_module(
    out: &mut String,
    functions: Option<&ModuleNode>,
    types: Option<&ModuleNode>,
    path: &[String],
    indent: usize,
    is_root: bool,
) {
    // Check if this module has any actual content
    let has_functions = functions.map_or(false, |node| !node.functions.is_empty());
    let has_types = types.map_or(false, |node| {
        node.types.iter().any(|ty| !is_builtin_type(&ty.type_key))
    });

    let mut child_names = std::collections::BTreeSet::new();
    if let Some(node) = functions {
        child_names.extend(node.children.keys().cloned());
    }
    if let Some(node) = types {
        child_names.extend(node.children.keys().cloned());
    }

    // Skip rendering if the module is empty and has no children
    if !is_root && !has_functions && !has_types && child_names.is_empty() {
        return;
    }

    let indent_str = " ".repeat(indent);
    if !is_root {
        let name = path.last().expect("module path missing");
        writeln!(out, "{}pub mod {} {{", indent_str, name).ok();
    }

    let current_indent = if is_root {
        indent_str.clone()
    } else {
        " ".repeat(indent + 4)
    };
    let module_path = if path.is_empty() {
        String::new()
    } else {
        format!("::{}", path.join("::"))
    };

    if let Some(node) = functions {
        for func in &node.functions {
            writeln!(
                out,
                "{}pub use crate::_tvm_ffi_stubgen_detail::functions{}::{};",
                current_indent, module_path, func.rust_name
            )
            .ok();
        }
    }
    if let Some(node) = types {
        for ty in &node.types {
            // Skip built-in types that are not generated
            if is_builtin_type(&ty.type_key) {
                continue;
            }
            writeln!(
                out,
                "{}pub use crate::_tvm_ffi_stubgen_detail::types{}::{};",
                current_indent, module_path, ty.rust_name
            )
            .ok();
        }
    }

    for child in child_names {
        let mut child_path = path.to_vec();
        child_path.push(child.clone());
        let func_child = functions.and_then(|node| node.children.get(&child));
        let type_child = types.and_then(|node| node.children.get(&child));
        render_facade_module(out, func_child, type_child, &child_path, indent + 4, false);
    }

    if !is_root {
        writeln!(out, "{}}}", indent_str).ok();
    }
}

fn render_function_module(out: &mut String, node: &ModuleNode, indent: usize) {
    let indent_str = " ".repeat(indent);
    if indent > 0 {
        writeln!(out, "{}use std::sync::LazyLock;", indent_str).ok();
        writeln!(
            out,
            "{}use tvm_ffi::{{Any, AnyView, Function, Result}};",
            indent_str
        )
        .ok();
        writeln!(out).ok();
    }
    for func in &node.functions {
        render_function(out, func, indent);
    }
    for child in node.children.values() {
        writeln!(out, "{}pub mod {} {{", indent_str, child.name).ok();
        render_function_module(out, child, indent + 4);
        writeln!(out, "{}}}", indent_str).ok();
    }
}

fn render_type_module(
    out: &mut String,
    node: &ModuleNode,
    indent: usize,
    type_map: &BTreeMap<String, String>,
) {
    let indent_str = " ".repeat(indent);
    if indent > 0 {
        writeln!(out, "{}use std::sync::LazyLock;", indent_str).ok();
        writeln!(out, "{}use tvm_ffi::{{Any, AnyView, Result}};", indent_str).ok();
        writeln!(out).ok();
    }
    for ty in &node.types {
        render_type(out, ty, indent, type_map);
    }
    for child in node.children.values() {
        writeln!(out, "{}pub mod {} {{", indent_str, child.name).ok();
        render_type_module(out, child, indent + 4, type_map);
        writeln!(out, "{}}}", indent_str).ok();
    }
}

fn render_function(out: &mut String, func: &FunctionGen, indent: usize) {
    let indent_str = " ".repeat(indent);
    let static_name = static_ident("FUNC", &func.full_name);
    writeln!(
        out,
        "{}static {}: LazyLock<Function> = LazyLock::new(|| Function::get_global(\"{}\").expect(\"missing global function\"));",
        indent_str, static_name, func.full_name
    )
    .ok();
    if func.sig.packed {
        writeln!(
            out,
            "{}pub fn {}(args: &[Any]) -> Result<Any> {{",
            indent_str, func.rust_name
        )
        .ok();
        writeln!(out, "{}    let func = &*{};", indent_str, static_name).ok();
        writeln!(
            out,
            "{}    let views: Vec<AnyView<'_>> = args.iter().map(AnyView::from).collect();",
            indent_str
        )
        .ok();
        writeln!(out, "{}    func.call_packed(&views)", indent_str).ok();
        writeln!(out, "{}}}", indent_str).ok();
        writeln!(out).ok();
        return;
    }
    let args = render_args(&func.sig.args);
    writeln!(
        out,
        "{}pub fn {}({}) -> Result<{}> {{",
        indent_str, func.rust_name, args, func.sig.ret.name
    )
    .ok();
    writeln!(out, "{}    let func = &*{};", indent_str, static_name).ok();
    writeln!(
        out,
        "{}    let typed = tvm_ffi::into_typed_fn!(func.clone(), Fn({}) -> Result<{}>);",
        indent_str,
        render_type_list(&func.sig.args),
        func.sig.ret.typed_ret_name()
    )
    .ok();
    let call_expr = format!("typed({})", render_call_args_typed(&func.sig.args));
    writeln!(
        out,
        "{}    {}",
        indent_str,
        func.sig
            .ret
            .wrap_typed_return(&call_expr, func.sig.ret.typed_ret_name())
    )
    .ok();
    writeln!(out, "{}}}", indent_str).ok();
    writeln!(out).ok();
}

fn type_key_to_short_rust_name(type_map: &BTreeMap<String, String>, type_key: &str) -> String {
    type_map
        .get(type_key)
        .and_then(|path| path.split("::").last().map(String::from))
        .unwrap_or_else(|| type_key.to_string())
}

fn render_type(out: &mut String, ty: &TypeGen, indent: usize, type_map: &BTreeMap<String, String>) {
    // Filter out built-in types that are already provided by tvm-ffi
    if is_builtin_type(&ty.type_key) {
        return;
    }

    let _indent_str = " ".repeat(indent);
    if let Some(ref info) = ty.repr_c_info {
        render_repr_c_type(out, ty, info, indent, type_map);
        return;
    }
    render_fallback_type(out, ty, indent);
}

fn is_builtin_type(type_key: &str) -> bool {
    // Filter ffi.* primitive types and aliases that are provided by tvm-ffi
    matches!(
        type_key,
        "ffi.Object"
            | "ffi.String"
            | "ffi.Function"
            | "ffi.Module"
            | "ffi.Tensor"
            | "ffi.Shape"
            | "ffi.Array"
            | "ffi.Map"
            | "ffi.Bytes"
            | "ffi.SmallStr"
            | "ffi.SmallBytes"
            | "DLTensor*"
            | "DataType"
            | "Device"
            | "bool"
            | "int"
            | "float"
            | "None"
    )
}

fn render_repr_c_type(
    out: &mut String,
    ty: &TypeGen,
    info: &repr_c::ReprCInfo,
    indent: usize,
    _type_map: &BTreeMap<String, String>,
) {
    let indent_str = " ".repeat(indent);
    let obj_name = format!("{}Obj", ty.rust_name);

    // Determine parent type for *Obj struct
    let parent_ty = match &info.parent_type_key {
        None => "tvm_ffi::object::Object".to_string(),
        Some(parent_key) if parent_key == "ffi.Object" => "tvm_ffi::object::Object".to_string(),
        Some(parent_key) => {
            // Use the type from type_map to get the full Rust path
            let parent_rust = _type_map
                .get(parent_key)
                .map(|s| s.clone())
                .unwrap_or_else(|| format!("{}Obj", sanitize_ident(parent_key, IdentStyle::Type)));
            // Extract just the type name and append "Obj"
            if let Some(last) = parent_rust.split("::").last() {
                format!("{}Obj", last)
            } else {
                format!("{}Obj", sanitize_ident(parent_key, IdentStyle::Type))
            }
        }
    };

    // Generate *Obj struct with #[repr(C)]
    writeln!(out, "{}#[repr(C)]", indent_str).ok();
    writeln!(out, "{}#[derive(tvm_ffi::derive::Object)]", indent_str).ok();
    writeln!(out, "{}#[type_key = \"{}\"]", indent_str, ty.type_key).ok();
    writeln!(out, "{}pub struct {} {{", indent_str, obj_name).ok();
    writeln!(out, "{}    parent: {},", indent_str, parent_ty).ok();
    for f in &info.direct_fields {
        writeln!(out, "{}    {}: {},", indent_str, f.rust_name, f.rust_type).ok();
    }
    writeln!(out, "{}}}\n", indent_str).ok();

    // Generate *Ref wrapper with #[repr(C)]
    writeln!(out, "{}#[repr(C)]", indent_str).ok();
    writeln!(
        out,
        "{}#[derive(tvm_ffi::derive::ObjectRef, Clone)]",
        indent_str
    )
    .ok();
    writeln!(out, "{}pub struct {} {{", indent_str, ty.rust_name).ok();
    writeln!(
        out,
        "{}    data: tvm_ffi::object::ObjectArc<{}>,",
        indent_str, obj_name
    )
    .ok();
    writeln!(out, "{}}}\n", indent_str).ok();

    // Generate impl_object_hierarchy! macro call
    if !ty.ancestor_chain.is_empty() {
        write!(
            out,
            "{}tvm_ffi::impl_object_hierarchy!({}:",
            indent_str, ty.rust_name
        )
        .ok();
        for (i, ancestor) in ty.ancestor_chain.iter().enumerate() {
            if i == 0 {
                write!(out, " {}", ancestor).ok();
            } else {
                write!(out, ", {}", ancestor).ok();
            }
        }
        writeln!(out, ");").ok();
        writeln!(out).ok();
    }

    // Generate getter methods for direct fields only
    writeln!(out, "{}impl {} {{", indent_str, ty.rust_name).ok();
    for f in &info.direct_fields {
        let method_name = format!("get_{}", f.rust_name);
        let access_expr = if f.is_pod {
            format!("self.data.{}", f.rust_name)
        } else {
            format!("self.data.{}.clone()", f.rust_name)
        };
        writeln!(
            out,
            "{}    pub fn {}(&self) -> {} {{",
            indent_str, method_name, f.rust_type
        )
        .ok();
        writeln!(out, "{}        {}", indent_str, access_expr).ok();
        writeln!(out, "{}    }}", indent_str).ok();
    }
    writeln!(out, "{}}}\n", indent_str).ok();

    // Generate method statics and impls
    for method in &ty.methods {
        render_method_static(out, ty, method, indent);
    }
    writeln!(out, "{}impl {} {{", indent_str, ty.rust_name).ok();
    for method in &ty.methods {
        render_method(out, ty, method, indent + 4);
    }
    writeln!(out, "{}}}\n", indent_str).ok();
}

fn render_fallback_type(out: &mut String, ty: &TypeGen, indent: usize) {
    let indent_str = " ".repeat(indent);
    writeln!(
        out,
        "{}tvm_ffi::define_object_wrapper!({}, \"{}\");\n",
        indent_str, ty.rust_name, ty.type_key
    )
    .ok();

    for field in &ty.fields {
        render_field_static(out, ty, field, indent);
    }
    for method in &ty.methods {
        render_method_static(out, ty, method, indent);
    }

    writeln!(out, "{}impl {} {{", indent_str, ty.rust_name).ok();
    for field in &ty.fields {
        render_field(out, ty, field, indent + 4);
    }
    for method in &ty.methods {
        render_method(out, ty, method, indent + 4);
    }
    writeln!(out, "{}}}\n", indent_str).ok();
}

fn render_field_static(out: &mut String, ty: &TypeGen, field: &FieldGen, indent: usize) {
    let indent_str = " ".repeat(indent);
    let static_name = static_ident("FIELD", &format!("{}::{}", ty.type_key, field.name));
    writeln!(
        out,
        "{}static {}: LazyLock<tvm_ffi::object_wrapper::FieldGetter<{}>> = LazyLock::new(|| tvm_ffi::object_wrapper::FieldGetter::new(\"{}\", \"{}\").expect(\"missing field\"));",
        indent_str, static_name, field.ty.name, ty.type_key, field.name
    )
    .ok();
}

fn render_field(out: &mut String, ty: &TypeGen, field: &FieldGen, indent: usize) {
    let indent_str = " ".repeat(indent);
    let static_name = static_ident("FIELD", &format!("{}::{}", ty.type_key, field.name));
    writeln!(
        out,
        "{}pub fn {}(&self) -> Result<{}> {{",
        indent_str, field.rust_name, field.ty.name
    )
    .ok();
    if field.ty.name == "tvm_ffi::Any" {
        writeln!(
            out,
            "{}    {}.get_any(self.as_object_ref())",
            indent_str, static_name
        )
        .ok();
    } else {
        writeln!(
            out,
            "{}    {}.get(self.as_object_ref())",
            indent_str, static_name
        )
        .ok();
    }
    writeln!(out, "{}}}", indent_str).ok();
}

fn render_method_static(out: &mut String, ty: &TypeGen, method: &MethodGen, indent: usize) {
    let indent_str = " ".repeat(indent);
    let static_name = static_ident("METHOD", &format!("{}::{}", ty.type_key, method.rust_name));
    writeln!(
        out,
        "{}static {}: LazyLock<tvm_ffi::Function> = LazyLock::new(|| tvm_ffi::Function::get_global(\"{}\").expect(\"missing method\"));",
        indent_str, static_name, method.full_name
    )
    .ok();
}

fn render_method(out: &mut String, ty: &TypeGen, method: &MethodGen, indent: usize) {
    let indent_str = " ".repeat(indent);
    let static_name = static_ident("METHOD", &format!("{}::{}", ty.type_key, method.rust_name));
    let self_prefix = if method.is_static { "" } else { "&self" };
    if method.sig.packed {
        if method.is_static {
            writeln!(
                out,
                "{}pub fn {}(args: &[Any]) -> Result<Any> {{",
                indent_str, method.rust_name
            )
            .ok();
            writeln!(out, "{}    let func = &*{};", indent_str, static_name).ok();
            writeln!(
                out,
                "{}    let views: Vec<AnyView<'_>> = args.iter().map(AnyView::from).collect();",
                indent_str
            )
            .ok();
            writeln!(out, "{}    func.call_packed(&views)", indent_str).ok();
            writeln!(out, "{}}}", indent_str).ok();
            return;
        }
        writeln!(
            out,
            "{}pub fn {}(&self, args: &[Any]) -> Result<Any> {{",
            indent_str, method.rust_name
        )
        .ok();
        writeln!(out, "{}    let func = &*{};", indent_str, static_name).ok();
        writeln!(
            out,
            "{}    let mut views: Vec<AnyView<'_>> = Vec::with_capacity(args.len() + 1);",
            indent_str
        )
        .ok();
        // For repr(C) types, use deref coercion to upcast to ObjectRef
        // For ObjectWrapper types, use the as_object_ref() method
        if ty.repr_c_info.is_some() {
            writeln!(
                out,
                "{}    views.push(AnyView::from(self as &tvm_ffi::object::ObjectRef));",
                indent_str
            )
            .ok();
        } else {
            writeln!(
                out,
                "{}    views.push(AnyView::from(self.as_object_ref()));",
                indent_str
            )
            .ok();
        }
        writeln!(
            out,
            "{}    views.extend(args.iter().map(AnyView::from));",
            indent_str
        )
        .ok();
        writeln!(out, "{}    func.call_packed(&views)", indent_str).ok();
        writeln!(out, "{}}}", indent_str).ok();
        return;
    }

    let args = render_args(&method.sig.args);
    let signature = if method.is_static {
        format!("{}({})", method.rust_name, args)
    } else if args.is_empty() {
        format!("{}({})", method.rust_name, self_prefix)
    } else {
        format!("{}({}, {})", method.rust_name, self_prefix, args)
    };
    writeln!(
        out,
        "{}pub fn {} -> Result<{}> {{",
        indent_str, signature, method.sig.ret.name
    )
    .ok();
    writeln!(out, "{}    let func = &*{};", indent_str, static_name).ok();
    let type_list = if method.is_static {
        render_type_list(&method.sig.args)
    } else {
        let mut types = vec!["tvm_ffi::object::ObjectRef".to_string()];
        types.extend(
            method
                .sig
                .args
                .iter()
                .map(|arg| arg.typed_arg_name().to_string()),
        );
        types.join(", ")
    };
    writeln!(
        out,
        "{}    let typed = tvm_ffi::into_typed_fn!(func.clone(), Fn({}) -> Result<{}>);",
        indent_str,
        type_list,
        method.sig.ret.typed_ret_name()
    )
    .ok();
    let call_expr = format!("typed({})", render_method_call_args(method));
    writeln!(
        out,
        "{}    {}",
        indent_str,
        method
            .sig
            .ret
            .wrap_typed_return(&call_expr, method.sig.ret.typed_ret_name())
    )
    .ok();
    writeln!(out, "{}}}", indent_str).ok();
}

fn render_args(args: &[RustType]) -> String {
    let mut out = Vec::new();
    for (i, arg) in args.iter().enumerate() {
        out.push(format!("_{}: {}", i, arg.name));
    }
    out.join(", ")
}

fn render_type_list(args: &[RustType]) -> String {
    args.iter()
        .map(|arg| arg.typed_arg_name().to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

fn render_call_args_typed(args: &[RustType]) -> String {
    let mut out = Vec::new();
    for (i, arg) in args.iter().enumerate() {
        out.push(arg.call_expr(&format!("_{}", i)));
    }
    out.join(", ")
}

fn render_method_call_args(method: &MethodGen) -> String {
    if method.is_static {
        return render_call_args_typed(&method.sig.args);
    }
    let mut out = Vec::new();
    let self_type = RustType::object_wrapper("Self");
    out.push(self_type.call_expr("self"));
    for (i, arg) in method.sig.args.iter().enumerate() {
        out.push(arg.call_expr(&format!("_{}", i)));
    }
    out.join(", ")
}

fn map_method_name(name: &str) -> String {
    if name == "__ffi_init__" {
        return "c_ffi_init".to_string();
    }
    sanitize_ident(name, IdentStyle::Function)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum IdentStyle {
    Function,
    Module,
    Type,
}

fn sanitize_ident(name: &str, style: IdentStyle) -> String {
    let mut out = String::new();
    let mut prev_underscore = false;
    for (i, ch) in name.chars().enumerate() {
        let mut c = ch;
        if style == IdentStyle::Module && ch.is_ascii_uppercase() {
            if i > 0 && !prev_underscore {
                out.push('_');
            }
            c = ch.to_ascii_lowercase();
        }
        if c.is_ascii_alphanumeric() || c == '_' {
            out.push(c);
            prev_underscore = c == '_';
        } else {
            out.push('_');
            prev_underscore = true;
        }
    }
    if out.is_empty() {
        out.push('_');
    }
    if out.chars().next().unwrap().is_ascii_digit() {
        out.insert(0, '_');
    }
    const KEYWORDS: &[&str] = &[
        "as", "break", "const", "continue", "crate", "else", "enum", "extern", "false", "fn",
        "for", "if", "impl", "in", "let", "loop", "match", "mod", "move", "mut", "pub", "ref",
        "return", "self", "Self", "static", "struct", "super", "trait", "true", "type", "unsafe",
        "use", "where", "while", "async", "await", "dyn",
    ];
    if KEYWORDS.contains(&out.as_str()) {
        out.push('_');
    }
    match style {
        IdentStyle::Type => to_pascal_case(&out),
        _ => out,
    }
}

fn to_pascal_case(name: &str) -> String {
    let mut out = String::new();
    let mut uppercase = true;
    for ch in name.chars() {
        if ch == '_' {
            uppercase = true;
            continue;
        }
        if uppercase {
            out.extend(ch.to_uppercase());
            uppercase = false;
        } else {
            out.push(ch);
        }
    }
    if out.is_empty() {
        "Type".to_string()
    } else {
        out
    }
}

fn static_ident(prefix: &str, full_name: &str) -> String {
    let mut out = String::new();
    out.push_str(prefix);
    out.push('_');
    for ch in full_name.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_uppercase());
        } else {
            out.push('_');
        }
    }
    if out.chars().next().unwrap().is_ascii_digit() {
        out.insert(0, '_');
    }
    out
}
