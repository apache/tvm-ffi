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
use crate::model::{FunctionGen, FunctionSig, MethodGen, ModuleNode, RustType, TypeGen};
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
            format!("crate::types::{}", rust_name)
        } else {
            format!("crate::types::{}::{}", module_path, rust_name)
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
        if let Some(info) = ffi::get_type_info(key) {
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
        }
        out.push((
            mods,
            TypeGen {
                type_key: key.clone(),
                rust_name,
                methods,
            },
        ));
    }
    Ok(out)
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

pub(crate) fn render_lib_rs() -> String {
    let mut out = String::new();
    out.push_str("pub mod functions;\n");
    out.push_str("pub mod types;\n\n");
    out.push_str("pub use functions::*;\n");
    out.push_str("pub use types::*;\n\n");
    out.push_str("pub fn load_library(path: &str) -> tvm_ffi::Result<tvm_ffi::Module> {\n");
    out.push_str("    tvm_ffi::Module::load_from_file(path)\n");
    out.push_str("}\n");
    out
}

pub(crate) fn render_functions_rs(root: &ModuleNode) -> String {
    let mut out = String::new();
    out.push_str("use std::sync::LazyLock;\n");
    out.push_str("use tvm_ffi::{Any, AnyView, Function, Result};\n\n");
    render_function_module(&mut out, root, 0);
    out
}

pub(crate) fn render_types_rs(root: &ModuleNode) -> String {
    let mut out = String::new();
    out.push_str("use std::sync::LazyLock;\n");
    out.push_str("use tvm_ffi::object::ObjectRef;\n");
    out.push_str("use tvm_ffi::{Any, AnyView, Result};\n\n");
    render_type_module(&mut out, root, 0);
    out
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

fn render_type_module(out: &mut String, node: &ModuleNode, indent: usize) {
    let indent_str = " ".repeat(indent);
    if indent > 0 {
        writeln!(out, "{}use std::sync::LazyLock;", indent_str).ok();
        writeln!(out, "{}use tvm_ffi::object::ObjectRef;", indent_str).ok();
        writeln!(out, "{}use tvm_ffi::{{Any, AnyView, Result}};", indent_str).ok();
        writeln!(out).ok();
    }
    for ty in &node.types {
        render_type(out, ty, indent);
    }
    for child in node.children.values() {
        writeln!(out, "{}pub mod {} {{", indent_str, child.name).ok();
        render_type_module(out, child, indent + 4);
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
        writeln!(out, "{}#[allow(non_snake_case)]", indent_str).ok();
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
    writeln!(out, "{}#[allow(non_snake_case)]", indent_str).ok();
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
        func.sig.ret.typed_name()
    )
    .ok();
    let call_expr = format!("typed({})", render_call_args_typed(&func.sig.args));
    writeln!(
        out,
        "{}    {}",
        indent_str,
        func.sig.ret.wrap_return(&call_expr)
    )
    .ok();
    writeln!(out, "{}}}", indent_str).ok();
    writeln!(out).ok();
}

fn render_type(out: &mut String, ty: &TypeGen, indent: usize) {
    let indent_str = " ".repeat(indent);
    writeln!(out, "{}#[derive(Clone)]", indent_str).ok();
    writeln!(out, "{}pub struct {} {{", indent_str, ty.rust_name).ok();
    writeln!(out, "{}    inner: ObjectRef,", indent_str).ok();
    writeln!(out, "{}}}\n", indent_str).ok();

    writeln!(out, "{}impl {} {{", indent_str, ty.rust_name).ok();
    writeln!(
        out,
        "{}    pub fn from_object(inner: ObjectRef) -> Self {{",
        indent_str
    )
    .ok();
    writeln!(out, "{}        Self {{ inner }}", indent_str).ok();
    writeln!(out, "{}    }}", indent_str).ok();
    writeln!(
        out,
        "{}    pub fn as_object_ref(&self) -> &ObjectRef {{",
        indent_str
    )
    .ok();
    writeln!(out, "{}        &self.inner", indent_str).ok();
    writeln!(out, "{}    }}", indent_str).ok();
    writeln!(out, "{}}}\n", indent_str).ok();

    writeln!(
        out,
        "{}impl From<ObjectRef> for {} {{",
        indent_str, ty.rust_name
    )
    .ok();
    writeln!(
        out,
        "{}    fn from(inner: ObjectRef) -> Self {{",
        indent_str
    )
    .ok();
    writeln!(out, "{}        Self {{ inner }}", indent_str).ok();
    writeln!(out, "{}    }}", indent_str).ok();
    writeln!(out, "{}}}\n", indent_str).ok();

    for method in &ty.methods {
        render_method_static(out, ty, method, indent);
    }

    writeln!(out, "{}impl {} {{", indent_str, ty.rust_name).ok();
    for method in &ty.methods {
        render_method(out, ty, method, indent + 4);
    }
    writeln!(out, "{}}}\n", indent_str).ok();
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
        writeln!(
            out,
            "{}    views.push(AnyView::from(self.as_object_ref()));",
            indent_str
        )
        .ok();
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
                .map(|arg| arg.typed_name().to_string()),
        );
        types.join(", ")
    };
    writeln!(
        out,
        "{}    let typed = tvm_ffi::into_typed_fn!(func.clone(), Fn({}) -> Result<{}>);",
        indent_str,
        type_list,
        method.sig.ret.typed_name()
    )
    .ok();
    let call_expr = format!("typed({})", render_method_call_args(method));
    writeln!(
        out,
        "{}    {}",
        indent_str,
        method.sig.ret.wrap_return(&call_expr)
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
        .map(|arg| arg.typed_name().to_string())
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
