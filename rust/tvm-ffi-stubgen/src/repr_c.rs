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

//! Validation that a type has a compact C-compatible layout (check_repr_c)
//! and extraction of field layout for repr(C) code generation.

use crate::ffi;
use crate::schema::{extract_type_schema, parse_type_schema, TypeSchema};
use std::collections::BTreeMap;

/// Result of check_repr_c: type passes and we have full layout for codegen.
#[derive(Debug, Clone)]
pub(crate) struct ReprCInfo {
    /// Type key of the immediate parent (Object or a subclass). None for root types.
    pub(crate) parent_type_key: Option<String>,
    /// Total size of the struct in bytes.
    pub(crate) total_size: i32,
    /// Direct fields of this type only (not inherited), sorted by offset.
    /// For codegen: first field of *Obj is parent (or Object), then these.
    pub(crate) direct_fields: Vec<ReprCField>,
}

#[derive(Debug, Clone)]
pub(crate) struct ReprCField {
    pub(crate) name: String,
    pub(crate) rust_name: String,
    pub(crate) offset: i64,
    pub(crate) size: i64,
    pub(crate) alignment: i64,
    /// Rust type name for the field (e.g. "i64", "Shape").
    pub(crate) rust_type: String,
    /// True if Copy type (getter returns value); false if Ref (getter returns Ref and clone).
    pub(crate) is_pod: bool,
}

/// Returns ReprCInfo if the type passes check_repr_c; None otherwise.
pub(crate) fn check_repr_c(
    type_key: &str,
    type_map: &BTreeMap<String, String>,
) -> Option<ReprCInfo> {
    let info = ffi::get_type_info(type_key)?;
    let total_size = total_size_from_info(info)?;
    if total_size <= 0 {
        return None;
    }

    let parent_type_key = if info.type_depth > 0 && !info.type_acenstors.is_null() {
        // Direct parent is ancestor[type_depth - 1]; ancestor[0] is the root.
        let ancestor_ptr = unsafe { *info.type_acenstors.add((info.type_depth - 1) as usize) };
        if ancestor_ptr.is_null() {
            return None;
        }
        let parent_info = unsafe { &*ancestor_ptr };
        let key = ffi::byte_array_to_string_opt(&parent_info.type_key)?;
        if !check_repr_c(&key, type_map).is_some() {
            return None;
        }
        Some(key)
    } else {
        None
    };

    let parent_total_size: i64 = if let Some(ref parent_key) = parent_type_key {
        let parent_info = ffi::get_type_info(parent_key)?;
        total_size_from_info(parent_info)? as i64
    } else {
        // Root type: first field starts after Object header. Use Object's registered size.
        let obj_info = ffi::get_type_info("ffi.Object")?;
        total_size_from_info(obj_info)? as i64
    };

    let mut direct_fields: Vec<ReprCField> = Vec::new();
    if info.num_fields > 0 && !info.fields.is_null() {
        let field_slice =
            unsafe { std::slice::from_raw_parts(info.fields, info.num_fields as usize) };
        for field in field_slice {
            let name = ffi::byte_array_to_string_opt(&field.name)?;
            if field.offset < 0 || field.size < 0 || field.alignment <= 0 {
                return None;
            }
            let meta = ffi::byte_array_to_string_opt(&field.metadata);
            let schema = meta
                .as_deref()
                .and_then(extract_type_schema)
                .and_then(|s| parse_type_schema(&s));
            let (rust_type, is_pod) =
                repr_c_field_type(schema.as_ref(), type_map, type_key, field.size)?;
            direct_fields.push(ReprCField {
                rust_name: sanitize_ident(&name, IdentStyle::Function),
                name,
                offset: field.offset,
                size: field.size,
                alignment: field.alignment,
                rust_type,
                is_pod,
            });
        }
    }

    direct_fields.sort_by_key(|f| f.offset);

    let first_offset = direct_fields
        .first()
        .map(|f| f.offset)
        .unwrap_or(parent_total_size);
    if first_offset != parent_total_size {
        return None;
    }

    let mut pos = parent_total_size;
    for f in &direct_fields {
        // Only allow alignment padding (field must start at aligned position)
        let aligned_pos = align_up(pos, f.alignment);
        if f.offset != aligned_pos {
            return None;
        }
        pos = f.offset + f.size;
    }
    if pos != total_size as i64 {
        return None;
    }

    Some(ReprCInfo {
        parent_type_key,
        total_size,
        direct_fields,
    })
}

fn total_size_from_info(info: &tvm_ffi::tvm_ffi_sys::TVMFFITypeInfo) -> Option<i32> {
    if info.metadata.is_null() {
        return None;
    }
    let meta = unsafe { &*info.metadata };
    if meta.total_size <= 0 {
        return None;
    }
    Some(meta.total_size)
}

/// Align `value` up to the next multiple of `alignment`.
fn align_up(value: i64, alignment: i64) -> i64 {
    if alignment <= 0 {
        return value;
    }
    (value + alignment - 1) / alignment * alignment
}

/// Map schema to (rust_type_name, is_pod). None if not repr_c compatible.
fn repr_c_field_type(
    schema: Option<&TypeSchema>,
    type_map: &BTreeMap<String, String>,
    _self_type_key: &str,
    field_size: i64,
) -> Option<(String, bool)> {
    let schema = schema?;
    match schema.origin.as_str() {
        "Any" | "ffi.Any" => Some(("tvm_ffi::AnyValue".to_string(), false)),
        "bool" => Some(("bool".to_string(), true)),
        "int" => match field_size {
            1 => Some(("i8".to_string(), true)),
            2 => Some(("i16".to_string(), true)),
            4 => Some(("i32".to_string(), true)),
            8 => Some(("i64".to_string(), true)),
            _ => None, // Unsupported int size
        },
        "float" => match field_size {
            4 => Some(("f32".to_string(), true)),
            8 => Some(("f64".to_string(), true)),
            _ => None, // Unsupported float size
        },
        "Device" => Some(("tvm_ffi::DLDevice".to_string(), true)),
        "DataType" => Some(("tvm_ffi::DLDataType".to_string(), true)),
        "ffi.String" | "std::string" | "const char*" | "ffi.SmallStr" => {
            Some(("tvm_ffi::String".to_string(), false))
        }
        "ffi.Bytes" | "ffi.SmallBytes" => Some(("tvm_ffi::Bytes".to_string(), false)),
        "ffi.Function" => Some(("tvm_ffi::Function".to_string(), false)),
        "ffi.Object" => Some(("tvm_ffi::object::ObjectRef".to_string(), false)),
        "ffi.Shape" => Some(("tvm_ffi::Shape".to_string(), false)),
        "ffi.Module" => Some(("tvm_ffi::Module".to_string(), false)),
        "ffi.Tensor" | "DLTensor*" => Some(("tvm_ffi::Tensor".to_string(), false)),
        "Optional" => match schema.args.as_slice() {
            [inner] => repr_c_field_type(Some(inner), type_map, _self_type_key, field_size)
                .map(|(inner_ty, pod)| (format!("Option<{}>", inner_ty), pod)),
            _ => None,
        },
        "ffi.Array" => match schema.args.as_slice() {
            [inner] => {
                let (inner_ty, _) =
                    repr_c_field_type(Some(inner), type_map, _self_type_key, field_size)?;
                Some((format!("tvm_ffi::Array<{}>", inner_ty), false))
            }
            _ => None,
        },
        "ffi.Map" => match schema.args.as_slice() {
            [k, v] => {
                let (k_ty, _) = repr_c_field_type(Some(k), type_map, _self_type_key, field_size)?;
                let (v_ty, _) = repr_c_field_type(Some(v), type_map, _self_type_key, field_size)?;
                Some((format!("tvm_ffi::Map<{}, {}>", k_ty, v_ty), false))
            }
            _ => None,
        },
        other => type_map.get(other).map(|path| (path.clone(), false)),
    }
}

#[derive(Clone, Copy)]
enum IdentStyle {
    Function,
}

fn sanitize_ident(name: &str, style: IdentStyle) -> String {
    let mut out = String::new();
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            out.push(ch);
        } else {
            out.push('_');
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
        "for", "if", "in", "let", "loop", "match", "move", "mut", "pub", "ref", "return", "self",
        "Self", "static", "struct", "super", "trait", "true", "type", "unsafe", "use", "where",
        "while", "async", "await", "dyn",
    ];
    if KEYWORDS.contains(&out.as_str()) {
        out.push('_');
    }
    match style {
        IdentStyle::Function => out,
    }
}
