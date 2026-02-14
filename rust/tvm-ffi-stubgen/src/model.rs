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

use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub(crate) struct RustType {
    pub(crate) name: String,
    pub(crate) supported: bool,
    pub(crate) kind: RustTypeKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RustTypeKind {
    Plain,
    ObjectWrapper,
}

#[derive(Debug, Clone)]
pub(crate) struct FunctionSig {
    pub(crate) args: Vec<RustType>,
    pub(crate) ret: RustType,
    pub(crate) packed: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct FunctionGen {
    pub(crate) full_name: String,
    pub(crate) rust_name: String,
    pub(crate) sig: FunctionSig,
}

#[derive(Debug, Clone)]
pub(crate) struct MethodGen {
    pub(crate) full_name: String,
    pub(crate) rust_name: String,
    pub(crate) sig: FunctionSig,
    pub(crate) is_static: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct FieldGen {
    pub(crate) name: String,
    pub(crate) rust_name: String,
    pub(crate) ty: RustType,
}

/// Spec for a single get_* method on a repr(C) Ref type.
#[derive(Debug, Clone)]
pub(crate) struct GetterSpec {
    /// Method name, e.g. "get_first"
    pub(crate) method_name: String,
    /// Expression to produce the value, e.g. "self.data.first.clone()"
    pub(crate) access_expr: String,
    /// Return type, e.g. "Shape" or "i64"
    pub(crate) ret_type: String,
}

#[derive(Debug, Clone)]
pub(crate) struct TypeGen {
    pub(crate) type_key: String,
    pub(crate) rust_name: String,
    pub(crate) methods: Vec<MethodGen>,
    pub(crate) fields: Vec<FieldGen>,
    /// Depth in inheritance hierarchy (0 = Object, 1 = direct subclass of Object, ...).
    pub(crate) type_depth: i32,
    /// If Some, type passes check_repr_c and we generate repr(C) *Obj + *Ref.
    pub(crate) repr_c_info: Option<crate::repr_c::ReprCInfo>,
    /// Getter specs for repr_c types (get_* methods). Empty for non-repr_c.
    pub(crate) getter_specs: Vec<GetterSpec>,
    /// Ancestor chain for repr_c types: [DirectParent, Grandparent, ..., ObjectRef].
    /// Empty for non-repr_c or types without proper hierarchy info.
    pub(crate) ancestor_chain: Vec<String>,
}

#[derive(Debug, Default)]
pub(crate) struct ModuleNode {
    pub(crate) name: String,
    pub(crate) functions: Vec<FunctionGen>,
    pub(crate) types: Vec<TypeGen>,
    pub(crate) children: BTreeMap<String, ModuleNode>,
}

impl FunctionSig {
    pub(crate) fn packed() -> Self {
        Self {
            args: Vec::new(),
            ret: RustType::unsupported("tvm_ffi::Any"),
            packed: true,
        }
    }

    pub(crate) fn from_types(args: Vec<RustType>, ret: RustType) -> Self {
        let typed = args.len() <= 12 && args.iter().all(|arg| arg.supported) && ret.supported;
        Self {
            args,
            ret,
            packed: !typed,
        }
    }
}

impl RustType {
    pub(crate) fn supported(name: &str) -> Self {
        Self {
            name: name.to_string(),
            supported: true,
            kind: RustTypeKind::Plain,
        }
    }

    pub(crate) fn unsupported(name: &str) -> Self {
        Self {
            name: name.to_string(),
            supported: false,
            kind: RustTypeKind::Plain,
        }
    }

    pub(crate) fn object_wrapper(name: &str) -> Self {
        Self {
            name: name.to_string(),
            supported: true,
            kind: RustTypeKind::ObjectWrapper,
        }
    }

    pub(crate) fn typed_arg_name(&self) -> &str {
        match self.kind {
            RustTypeKind::Plain => &self.name,
            RustTypeKind::ObjectWrapper => "tvm_ffi::object::ObjectRef",
        }
    }

    pub(crate) fn typed_ret_name(&self) -> &str {
        match self.kind {
            RustTypeKind::Plain => &self.name,
            RustTypeKind::ObjectWrapper => &self.name,
        }
    }

    pub(crate) fn call_expr(&self, arg_name: &str) -> String {
        match self.kind {
            RustTypeKind::Plain => arg_name.to_string(),
            RustTypeKind::ObjectWrapper => {
                // Use Into<ObjectRef> trait for upcast
                format!("{}.into()", arg_name)
            }
        }
    }

    pub(crate) fn wrap_typed_return(&self, expr: &str, typed_ret_name: &str) -> String {
        match self.kind {
            RustTypeKind::Plain => expr.to_string(),
            RustTypeKind::ObjectWrapper => {
                if typed_ret_name == "tvm_ffi::object::ObjectRef" {
                    if self.name == "Self" {
                        format!("{}.map(Self::from)", expr)
                    } else {
                        format!("{}.map({}::from)", expr, self.name)
                    }
                } else {
                    expr.to_string()
                }
            }
        }
    }
}
