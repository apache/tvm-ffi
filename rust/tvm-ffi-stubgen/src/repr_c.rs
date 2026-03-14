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
//!
//! The strategy is gap-filling: given the parent struct size and the registered
//! field offsets/sizes, any byte range not covered by a known field is emitted
//! as a `[u8; N]` padding member.  This handles C++ tail padding, vtable
//! pointers, and unregistered fields uniformly without requiring alignment
//! inference.

use crate::ffi;
use crate::schema::{TypeSchema, extract_type_schema, parse_type_schema};
use log::{debug, trace};
use std::collections::BTreeMap;

/// Result of check_repr_c: type passes and we have full layout for codegen.
#[derive(Debug, Clone)]
pub(crate) struct ReprCInfo {
    /// Type key of the immediate parent (Object or a subclass). None for root types.
    pub(crate) parent_type_key: Option<String>,
    /// Ordered layout entries (fields and gaps) covering [parent_total_size .. total_size).
    pub(crate) layout: Vec<LayoutEntry>,
    /// Fields registered in this type's ObjectDef that are NOT part of the repr(C) struct
    /// layout.  Two causes: (1) offset < parent_total_size — the field occupies a slot
    /// within the parent's address range; (2) schema not mappable — the field's type
    /// cannot be expressed as a repr(C) Rust type.  All such fields can still be read at
    /// runtime via FieldGetter.
    pub(crate) non_layout_fields: Vec<NonLayoutField>,
}

/// A registered field that does not appear in the repr(C) struct layout.
#[derive(Debug, Clone)]
pub(crate) struct NonLayoutField {
    /// Original C++ field name (used as the FieldGetter key).
    pub(crate) name: String,
    /// Sanitized Rust identifier (used as the getter method name suffix).
    pub(crate) rust_name: String,
    /// Mapped Rust type string, or None if the schema could not be mapped.
    /// When None, the generated getter returns `tvm_ffi::Any` via get_any().
    pub(crate) rust_type: Option<String>,
}

/// A single entry in the repr(C) struct body after the parent.
#[derive(Debug, Clone)]
pub(crate) enum LayoutEntry {
    /// A known, typed field.
    Field(ReprCField),
    /// An opaque gap (padding, vtable pointer, or unregistered field).
    Gap { name: String, size: i64 },
}

#[derive(Debug, Clone)]
pub(crate) struct ReprCField {
    pub(crate) rust_name: String,
    pub(crate) offset: i64,
    pub(crate) size: i64,
    /// Rust type name for the field (e.g. "i64", "Shape").
    pub(crate) rust_type: String,
    /// True if Copy type (getter returns value); false if Ref (getter returns clone).
    pub(crate) is_pod: bool,
}

impl ReprCInfo {
    /// Iterate only the typed fields (skipping gaps).
    pub(crate) fn fields(&self) -> impl Iterator<Item = &ReprCField> {
        self.layout.iter().filter_map(|e| match e {
            LayoutEntry::Field(f) => Some(f),
            LayoutEntry::Gap { .. } => None,
        })
    }
}

/// Returns ReprCInfo if the type can be laid out as repr(C); None otherwise.
///
/// Failure reasons (all logged at DEBUG level):
/// - No type info registered at all
/// - Metadata missing or total_size unknown
/// - Parent type not in type_map or parent itself fails
/// - A field's type schema cannot be mapped to a Rust type
pub(crate) fn check_repr_c(
    type_key: &str,
    type_map: &BTreeMap<String, String>,
) -> Option<ReprCInfo> {
    let info = match ffi::get_type_info(type_key) {
        Some(i) => i,
        None => {
            debug!("{}: no type info registered", type_key);
            return None;
        }
    };
    let total_size = match total_size_from_info(info) {
        Some(s) if s > 0 => s as i64,
        _ => {
            debug!("{}: metadata missing or total_size <= 0", type_key);
            return None;
        }
    };
    trace!(
        "{}: total_size={}, type_depth={}, num_fields={}, num_methods={}",
        type_key, total_size, info.type_depth, info.num_fields, info.num_methods
    );

    // Resolve parent.
    // If the direct parent is in type_map and passes check_repr_c, we use it as
    // the typed parent field.  Otherwise we fall back to ffi.Object as the parent
    // and let gap-filling cover the bytes between Object and our first field.
    let obj_size = {
        let oi = ffi::get_type_info("ffi.Object")?;
        total_size_from_info(oi)? as i64
    };
    let (parent_type_key, parent_total_size) =
        if info.type_depth > 0 && !info.type_acenstors.is_null() {
            let ancestor_ptr = unsafe { *info.type_acenstors.add((info.type_depth - 1) as usize) };
            let direct_parent_key = if !ancestor_ptr.is_null() {
                let pi = unsafe { &*ancestor_ptr };
                ffi::byte_array_to_string_opt(&pi.type_key)
            } else {
                None
            };
            match direct_parent_key {
                Some(ref key) if key == "ffi.Object" => (None, obj_size),
                Some(ref key)
                    if type_map.contains_key(key) && check_repr_c(key, type_map).is_some() =>
                {
                    let pi = ffi::get_type_info(key)?;
                    let ps = total_size_from_info(pi)? as i64;
                    trace!("{}: parent='{}' (typed, size={})", type_key, key, ps);
                    (Some(key.clone()), ps)
                }
                Some(ref key) => {
                    // Parent exists but not mappable — use Object as parent, gap covers the rest.
                    trace!(
                        "{}: parent='{}' not mappable, falling back to Object",
                        type_key, key
                    );
                    (None, obj_size)
                }
                None => (None, obj_size),
            }
        } else {
            (None, obj_size)
        };
    trace!(
        "{}: parent={:?}, parent_total_size={}",
        type_key, parent_type_key, parent_total_size
    );

    // Collect and sort fields that belong to this type (offset >= parent_total_size).
    // Any registered field that cannot become a direct struct member is tracked in
    // non_layout_fields so it can be exposed via a FieldGetter accessor.
    let mut typed_fields: Vec<ReprCField> = Vec::new();
    let mut non_layout_fields: Vec<NonLayoutField> = Vec::new();
    if info.num_fields > 0 && !info.fields.is_null() {
        let field_slice =
            unsafe { std::slice::from_raw_parts(info.fields, info.num_fields as usize) };
        for field in field_slice {
            let name = match ffi::byte_array_to_string_opt(&field.name) {
                Some(n) => n,
                None => {
                    debug!("{}: a field name is unreadable", type_key);
                    return None;
                }
            };
            // Fields whose offset falls inside the parent type's address range cannot be
            // part of the repr(C) struct layout (they occupy a slot the parent owns).
            if field.offset < parent_total_size {
                let rust_type = if field.offset >= 0 && field.size >= 0 {
                    let meta = ffi::byte_array_to_string_opt(&field.metadata);
                    let schema = meta
                        .as_deref()
                        .and_then(extract_type_schema)
                        .and_then(|s| parse_type_schema(&s));
                    repr_c_field_type(schema.as_ref(), type_map, type_key, field.size)
                        .map(|(ty, _)| ty)
                } else {
                    None
                };
                trace!(
                    "{}:   field '{}' at offset={} is in parent range → non-layout (rust_type={:?})",
                    type_key, name, field.offset, rust_type
                );
                non_layout_fields.push(NonLayoutField {
                    name: name.clone(),
                    rust_name: sanitize_ident(&name),
                    rust_type,
                });
                continue;
            }
            trace!(
                "{}:   field '{}': offset={}, size={}",
                type_key, name, field.offset, field.size
            );
            if field.offset < 0 || field.size < 0 {
                debug!("{}: field '{}' has invalid offset/size", type_key, name);
                return None;
            }
            let meta = ffi::byte_array_to_string_opt(&field.metadata);
            let schema = meta
                .as_deref()
                .and_then(extract_type_schema)
                .and_then(|s| parse_type_schema(&s));
            trace!(
                "{}:   field '{}' schema origin={:?}",
                type_key,
                name,
                schema.as_ref().map(|s| &s.origin)
            );
            let mapped = repr_c_field_type(schema.as_ref(), type_map, type_key, field.size);
            let (rust_type, is_pod) = match mapped {
                Some(v) => v,
                None => {
                    // Schema not mappable: cannot be a struct field, but still accessible
                    // at runtime via FieldGetter with an untyped (Any) return.
                    debug!(
                        "{}: field '{}' type not mappable, covered by gap + non-layout FieldGetter (schema_origin={:?})",
                        type_key,
                        name,
                        schema.as_ref().map(|s| &s.origin)
                    );
                    non_layout_fields.push(NonLayoutField {
                        name: name.clone(),
                        rust_name: sanitize_ident(&name),
                        rust_type: None,
                    });
                    continue;
                }
            };
            trace!(
                "{}:   field '{}' -> rust_type='{}', is_pod={}",
                type_key, name, rust_type, is_pod
            );
            typed_fields.push(ReprCField {
                rust_name: sanitize_ident(&name),
                offset: field.offset,
                size: field.size,
                rust_type,
                is_pod,
            });
        }
    }
    typed_fields.sort_by_key(|f| f.offset);

    // Build layout by walking [parent_total_size .. total_size) and inserting
    // gaps wherever there is no registered field.
    let mut layout = Vec::new();
    let mut pos = parent_total_size;
    let mut gap_idx = 0usize;
    for f in &typed_fields {
        if f.offset > pos {
            let gap_size = f.offset - pos;
            trace!(
                "{}:   gap at {}..{} ({} bytes)",
                type_key, pos, f.offset, gap_size
            );
            layout.push(LayoutEntry::Gap {
                name: format!("_gap{}", gap_idx),
                size: gap_size,
            });
            gap_idx += 1;
            pos = f.offset;
        }
        if f.offset < pos {
            // Overlapping fields — shouldn't happen, bail out.
            debug!(
                "{}: field '{}' at offset={} overlaps pos={}",
                type_key, f.rust_name, f.offset, pos
            );
            return None;
        }
        layout.push(LayoutEntry::Field(f.clone()));
        pos = f.offset + f.size;
    }
    // Trailing gap (tail padding, or fields after last registered one)
    if pos < total_size {
        let gap_size = total_size - pos;
        trace!(
            "{}:   trailing gap at {}..{} ({} bytes)",
            type_key, pos, total_size, gap_size
        );
        layout.push(LayoutEntry::Gap {
            name: format!("_gap{}", gap_idx),
            size: gap_size,
        });
    } else if pos > total_size {
        debug!(
            "{}: fields exceed total_size (pos={} > total_size={})",
            type_key, pos, total_size
        );
        return None;
    }

    debug!(
        "{}: repr_c OK ({} fields, {} gaps, {} layout entries)",
        type_key,
        typed_fields.len(),
        layout
            .iter()
            .filter(|e| matches!(e, LayoutEntry::Gap { .. }))
            .count(),
        layout.len()
    );
    Some(ReprCInfo {
        parent_type_key,
        layout,
        non_layout_fields,
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
            _ => None,
        },
        "float" => match field_size {
            4 => Some(("f32".to_string(), true)),
            8 => Some(("f64".to_string(), true)),
            _ => None,
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
            [] => Some(("Option<tvm_ffi::object::ObjectRef>".to_string(), false)),
            _ => None,
        },
        "ffi.Array" => match schema.args.as_slice() {
            [inner] => {
                let (inner_ty, _) =
                    repr_c_field_type(Some(inner), type_map, _self_type_key, field_size)?;
                Some((format!("tvm_ffi::Array<{}>", inner_ty), false))
            }
            [] => Some((
                "tvm_ffi::Array<tvm_ffi::object::ObjectRef>".to_string(),
                false,
            )),
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

fn sanitize_ident(name: &str) -> String {
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
        "for", "if", "impl", "in", "let", "loop", "match", "mod", "move", "mut", "pub", "ref",
        "return", "self", "Self", "static", "struct", "super", "trait", "true", "type", "unsafe",
        "use", "where", "while", "async", "await", "dyn",
    ];
    if KEYWORDS.contains(&out.as_str()) {
        out.push('_');
    }
    out
}
