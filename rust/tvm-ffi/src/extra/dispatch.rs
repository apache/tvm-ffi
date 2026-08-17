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

//! Typed visitor dispatch for [`super::structural_visit::structural_walk`]:
//! the [`VisitDispatch`] trait targeted by `#[dispatch(visit)]`, and the
//! walker adapter that runs such a visitor at the phase selected by the walk
//! order. The traversal engine, closure walkers, and tuple chains live in
//! [`super::structural_visit`], which re-exports these items to keep its
//! public paths stable.

use crate::error::Result;

use super::structural_visit::{
    DefRegionKind, IntoWalker, NativeVisit, VisitResult, VisitValue, WalkResult,
};

/// Typed dispatch implemented by a walk-layer observer.
///
/// [`crate::dispatch`] tests the implementation's `visit_*` methods in source
/// order. Borrowed node arguments use refcount-free subtype checks, owned
/// FFI-compatible arguments use exact value casts, and `&VisitValue` is a
/// catch-all. `None` reports that no handler matched: a standalone walk then
/// advances normally, while a tuple chain hands the value to the next link —
/// so a spliced visitor that "handled" a value must not return `None`.
///
/// This is the observer layer, mirroring C++ `StructuralWalk` callbacks: the
/// walker owns recursion, and a handler steers it only through the returned
/// [`WalkResult`]. A traversal that must visit children itself — selected
/// children, custom orders, explicit definition-region overrides — belongs in
/// a [`super::structural_visit::StructuralVisitor`] instead.
///
/// The definition-region state active at the dispatched value arrives as the
/// `def_region_kind` argument. A `#[dispatch(visit)]` handler opts into it by
/// declaring a trailing `DefRegionKind` parameter — the analog of a C++
/// `StructuralWalk` callback accepting `(value, def_region_kind)` instead of
/// `(value)`.
pub trait VisitDispatch: Sized {
    fn dispatch_visit(
        &mut self,
        value: &VisitValue,
        def_region_kind: DefRegionKind,
    ) -> Option<VisitResult>;
}

impl<V: VisitDispatch> VisitDispatch for &mut V {
    #[inline]
    fn dispatch_visit(
        &mut self,
        value: &VisitValue,
        def_region_kind: DefRegionKind,
    ) -> Option<VisitResult> {
        (**self).dispatch_visit(value, def_region_kind)
    }
}

#[doc(hidden)]
pub enum ByDispatch {}

impl<'a, V: VisitDispatch> IntoWalker<ByDispatch> for &'a mut V {
    type Walker = DispatchVisitor<&'a mut V>;
    fn into_walker(self) -> Self::Walker {
        DispatchVisitor { visitor: self }
    }
}

/// Owns its walker so a closure's state stays inline and a `&mut` visitor
/// keeps a single level of indirection. Public only as an
/// [`IntoWalker::Walker`] projection.
#[doc(hidden)]
pub struct DispatchVisitor<V> {
    visitor: V,
}

impl<V: VisitDispatch> NativeVisit for DispatchVisitor<V> {
    fn visit(&mut self, value: &VisitValue, def_region_kind: DefRegionKind) -> Result<WalkResult> {
        self.visitor
            .dispatch_visit(value, def_region_kind)
            .unwrap_or(Ok(WalkResult::Advance))
    }
}
