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

//! Rust structural hooks: per-type descent functions for the Rust engines.
//!
//! The structural engines descend into a value of a registered object type
//! through, in order: the built-in containers natively, a Rust hook from
//! this registry, the type's registered C++ `__s_visit__` or `__s_mutate__`
//! hook, and finally its reflected fields. A C++ hook costs a round trip
//! across the ABI vtable for the node and for every child it visits; a Rust
//! hook runs in the engine's own frame and reaches each child with one
//! indirect call, so hooks for the hot node kinds of an IR belong here.
//!
//! A hook applies to the exact type index it is registered for, like the C++
//! attribute columns: hooks do not inherit. Register them once at startup;
//! registering again replaces the previous hooks of that type.

use std::ptr::null_mut;

use crate::any::Any;
use crate::error::{Error, Result, TYPE_ERROR};
use crate::object::type_index_table::TypeIndexTable;
use crate::object::ObjectCore;
use crate::tvm_ffi_sys::TVMFFITypeIndex;

use super::structural_common::StructuralView;
use super::structural_containers as containers;
use super::structural_mutate::MutateChildren;
use super::structural_visit::{VisitChildren, VisitInterrupt};
use super::unchanged::UnchangedOr;

/// Visit the children of `value` through `children`; the Rust form of a C++
/// `__s_visit__` hook.
pub type StructuralVisitHook = fn(
    value: &StructuralView,
    children: &mut VisitChildren<'_>,
) -> std::result::Result<(), VisitHalt>;

/// Mutate the children of `value` through `children` and return the rebuilt
/// node, or [`Unchanged`](super::unchanged::Unchanged) when nothing changed
/// or the node was updated in place; the Rust form of the C++ `__s_mutate__`
/// and `__s_maybe_inplace_mutate__` hooks.
pub type StructuralMutateHook =
    fn(value: &StructuralView, children: &mut MutateChildren<'_>) -> Result<UnchangedOr<Any>>;

/// Why a visit hook stopped early: a child interrupted the traversal, or an
/// error occurred. Both propagate with `?` from
/// [`VisitChildren::visit`].
pub enum VisitHalt {
    /// A visit callback interrupted the traversal.
    Interrupt(VisitInterrupt),
    /// An error occurred.
    Error(Error),
}

impl From<Error> for VisitHalt {
    #[inline]
    fn from(error: Error) -> Self {
        Self::Error(error)
    }
}

impl From<VisitInterrupt> for VisitHalt {
    #[inline]
    fn from(interrupt: VisitInterrupt) -> Self {
        Self::Interrupt(interrupt)
    }
}

impl std::fmt::Debug for VisitHalt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Interrupt(_) => f.write_str("VisitHalt::Interrupt"),
            Self::Error(error) => write!(f, "VisitHalt::Error({error})"),
        }
    }
}

/// The Rust structural hooks of one object type. A `None` entry leaves that
/// engine on the type's C++ hook or reflected fields.
#[derive(Clone, Copy, Default)]
pub struct StructuralHooks {
    /// Descent for the visit and walk engines.
    pub visit: Option<StructuralVisitHook>,
    /// Descent for the map and mutate engines.
    pub mutate: Option<StructuralMutateHook>,
}

static VISIT_HOOKS: TypeIndexTable<()> = TypeIndexTable::new();
static MUTATE_HOOKS: TypeIndexTable<()> = TypeIndexTable::new();

/// Register the Rust structural hooks of object type `T`, replacing any
/// registered before.
///
/// The hooks are consulted before `T`'s C++ hooks and reflected fields, for
/// values whose runtime type is exactly `T`. Register them before the first
/// traversal that should use them; registration is process-wide.
pub fn register_structural_hooks<T: ObjectCore>(hooks: StructuralHooks) -> Result<()> {
    register_structural_hooks_for_type_index(T::type_index(), hooks)
}

/// [`register_structural_hooks`] for a type known only by its runtime index,
/// such as a C++ type this crate has no mirror struct for.
pub fn register_structural_hooks_for_type_index(
    type_index: i32,
    hooks: StructuralHooks,
) -> Result<()> {
    if type_index < TVMFFITypeIndex::kTVMFFIStaticObjectBegin as i32 {
        return Err(Error::new(
            TYPE_ERROR,
            &format!("structural hooks apply to object types; type index {type_index} is not one"),
            "",
        ));
    }
    if containers::native_container(type_index).is_some()
        || containers::is_immutable_leaf(type_index)
    {
        return Err(Error::new(
            TYPE_ERROR,
            "the built-in containers, ffi.Str and ffi.Bytes always descend natively; \
             they cannot take structural hooks",
            "",
        ));
    }
    // The runtime aborts on a type-info query for an unknown index, so the
    // index is not validated further: callers obtain it from `T::type_index()`
    // or `TVMFFITypeKeyToIndex`.
    if !TypeIndexTable::<()>::in_range(type_index) {
        return Err(Error::new(
            TYPE_ERROR,
            &format!("type index {type_index} is beyond the structural hook registry"),
            "",
        ));
    }
    VISIT_HOOKS.set(
        type_index,
        hooks.visit.map_or(null_mut(), |hook| hook as *mut ()),
    );
    MUTATE_HOOKS.set(
        type_index,
        hooks.mutate.map_or(null_mut(), |hook| hook as *mut ()),
    );
    Ok(())
}

/// The registered visit hook of `type_index`, if any.
#[inline]
pub(crate) fn visit_hook(type_index: i32) -> Option<StructuralVisitHook> {
    let pointer = VISIT_HOOKS.get(type_index);
    if pointer.is_null() {
        None
    } else {
        // SAFETY: only `register_structural_hooks_for_type_index` stores into
        // this table, and it stores exactly a `StructuralVisitHook`.
        Some(unsafe { std::mem::transmute::<*mut (), StructuralVisitHook>(pointer) })
    }
}

/// The registered mutate hook of `type_index`, if any.
#[inline]
pub(crate) fn mutate_hook(type_index: i32) -> Option<StructuralMutateHook> {
    let pointer = MUTATE_HOOKS.get(type_index);
    if pointer.is_null() {
        None
    } else {
        // SAFETY: only `register_structural_hooks_for_type_index` stores into
        // this table, and it stores exactly a `StructuralMutateHook`.
        Some(unsafe { std::mem::transmute::<*mut (), StructuralMutateHook>(pointer) })
    }
}
