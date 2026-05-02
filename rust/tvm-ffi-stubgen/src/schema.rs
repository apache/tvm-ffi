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

use serde::Deserialize;
use std::collections::{BTreeSet, HashSet};

#[derive(Debug, Clone)]
pub(crate) struct TypeSchema {
    pub(crate) origin: String,
    pub(crate) args: Vec<TypeSchema>,
}

#[derive(Deserialize)]
struct TypeSchemaJson {
    #[serde(rename = "type")]
    ty: String,
    #[serde(default)]
    args: Vec<TypeSchemaJson>,
}

pub(crate) fn extract_type_schema(metadata: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(metadata).ok()?;
    value
        .get("type_schema")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

pub(crate) fn parse_type_schema(schema: &str) -> Option<TypeSchema> {
    let json: TypeSchemaJson = serde_json::from_str(schema).ok()?;
    Some(parse_type_schema_json(&json))
}

pub(crate) fn collect_type_keys(
    schema: &TypeSchema,
    known: &HashSet<String>,
    out: &mut BTreeSet<String>,
) {
    if known.contains(&schema.origin) {
        out.insert(schema.origin.clone());
    }
    for arg in &schema.args {
        collect_type_keys(arg, known, out);
    }
}

fn parse_type_schema_json(json: &TypeSchemaJson) -> TypeSchema {
    TypeSchema {
        origin: json.ty.clone(),
        args: json.args.iter().map(parse_type_schema_json).collect(),
    }
}
