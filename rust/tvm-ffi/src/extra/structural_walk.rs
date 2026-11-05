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

//! Observer callbacks and adapters for structural walking.
//!
//! Callbacks report failures with [`Result`] and must not panic. A panic reaching
//! the non-unwinding native callback boundary aborts the process.

use super::structural_visit::{
    DefRegionKind, StructuralVisitEngine, StructuralVisitorView, VisitInterrupt,
    visit_result_from_any,
};
use crate::{Any, AnyView, Dispatcher, IntoDispatcher, Result};
use std::borrow::BorrowMut;

// -----------------------------------------------------------------------------
// Key data structures and callback results.
// -----------------------------------------------------------------------------

/// What a callback asks the Rust walker to do with the current value.
pub enum WalkResult {
    /// Continue and visit this value's children.
    Advance,
    /// Continue without visiting this value's children or firing its exit hook.
    Skip,
    /// Halt the entire traversal.
    Interrupt,
    /// Halt the entire traversal and return a payload to the caller.
    InterruptWith(Any),
}

/// Convert either an infallible or fallible typed handler result.
///
/// This keeps simple handlers terse while allowing a handler to return
/// `tvm_ffi::Result<WalkResult>` and use `?`.
pub trait IntoWalkResult {
    fn into_walk_result(self) -> Result<WalkResult>;
}

impl IntoWalkResult for WalkResult {
    #[inline(always)]
    fn into_walk_result(self) -> Result<WalkResult> {
        Ok(self)
    }
}

impl IntoWalkResult for Result<WalkResult> {
    #[inline(always)]
    fn into_walk_result(self) -> Result<WalkResult> {
        self
    }
}

/// Callback order for [`structural_walk`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WalkOrder {
    /// Run the typed handler before the current value's children.
    #[default]
    PreOrder,
    /// Run the typed handler after the current value's children.
    PostOrder,
}

// -----------------------------------------------------------------------------
// Walk adapter to Dispatcher.
// -----------------------------------------------------------------------------

impl<O: IntoWalkResult> super::dispatch::CallbackResult<Result<WalkResult>> for O {
    #[inline(always)]
    fn callback_result(self) -> Result<WalkResult> {
        self.into_walk_result()
    }
}

// -----------------------------------------------------------------------------
// Implementation of walk.
// -----------------------------------------------------------------------------

// A walk is an ordinary visitor handler: the visitor core owns all child
// recursion, native hooks and region transitions. This adapter only chooses
// when the observer runs and translates its control result.
struct WalkVisitor<'a, D, const PRE_ORDER: bool> {
    observer: &'a mut D,
}

impl<'s, 'v, 'c, 'r, 'a, D, const PRE_ORDER: bool>
    Dispatcher<
        (
            AnyView<'v>,
            &'c mut StructuralVisitorView<'r, WalkVisitor<'a, D, PRE_ORDER>>,
        ),
        Option<Result<Option<VisitInterrupt>>>,
    > for &'s mut WalkVisitor<'a, D, PRE_ORDER>
where
    for<'d, 'b> &'d mut D: Dispatcher<(AnyView<'b>, DefRegionKind), Option<Result<WalkResult>>>,
{
    #[inline(always)]
    fn dispatch_tuple(
        self,
        (value, visitor): (
            AnyView<'v>,
            &'c mut StructuralVisitorView<'r, WalkVisitor<'a, D, PRE_ORDER>>,
        ),
    ) -> Option<Result<Option<VisitInterrupt>>> {
        Some(self.visit(value, visitor))
    }
}

impl<'a, D, const PRE_ORDER: bool> WalkVisitor<'a, D, PRE_ORDER>
where
    for<'s, 'v> &'s mut D: Dispatcher<(AnyView<'v>, DefRegionKind), Option<Result<WalkResult>>>,
{
    #[inline(always)]
    fn visit(
        &mut self,
        value: AnyView<'_>,
        visitor: &mut StructuralVisitorView<'_, Self>,
    ) -> Result<Option<VisitInterrupt>> {
        let kind = visitor.def_region_kind();
        if !PRE_ORDER {
            if let Some(interrupt) = visitor.default_visit(self, value)? {
                core::hint::cold_path();
                return Ok(Some(interrupt));
            }
        }
        match (&mut *self.observer)
            .dispatch_tuple((value, kind))
            .unwrap_or_else(|| Ok(WalkResult::Advance))?
        {
            WalkResult::Advance if PRE_ORDER => visitor.default_visit(self, value),
            WalkResult::Advance => Ok(None),
            WalkResult::Skip => {
                core::hint::cold_path();
                Ok(None)
            }
            WalkResult::Interrupt => {
                core::hint::cold_path();
                Ok(Some(VisitInterrupt::with(Any::new())))
            }
            WalkResult::InterruptWith(payload) => {
                core::hint::cold_path();
                Ok(Some(VisitInterrupt::with(payload)))
            }
        }
    }
}

#[inline(always)]
fn run_walker<D>(
    root: AnyView<'_>,
    observer: &mut D,
    order: WalkOrder,
) -> Result<Option<VisitInterrupt>>
where
    for<'s, 'v> &'s mut D: Dispatcher<(AnyView<'v>, DefRegionKind), Option<Result<WalkResult>>>,
{
    let root = *root.as_raw_ffi_any();
    // Own the result while dropping the observer: user capture destructors can
    // unwind after the native callback has returned.
    let result = match order {
        WalkOrder::PreOrder => {
            let mut visitor = WalkVisitor::<D, true> { observer };
            let raw = StructuralVisitEngine::run_visitor_raw::<WalkVisitor<'_, D, true>>(
                root,
                &mut visitor,
            );
            let result = unsafe { Any::from_raw_ffi_any(raw) };
            drop(visitor);
            result
        }
        WalkOrder::PostOrder => {
            let mut visitor = WalkVisitor::<D, false> { observer };
            let raw = StructuralVisitEngine::run_visitor_raw::<WalkVisitor<'_, D, false>>(
                root,
                &mut visitor,
            );
            let result = unsafe { Any::from_raw_ffi_any(raw) };
            drop(visitor);
            result
        }
    };
    visit_result_from_any(result)
}

/// Walk `root` with an observer, the Rust analog of C++
/// `StructuralWalk<order>(root, callbacks...)`.
///
/// `walker` is anything implementing [`IntoDispatcher`]: a `&mut` reference to a
/// stateful [`Dispatcher`] walker (declared with `#[dispatch]`), a bare closure
/// in any [`crate::IntoDispatcher`] callback shape (catch-all `AnyView<'_>`, typed, or node,
/// with an optional trailing [`DefRegionKind`]), or a tuple of such
/// callbacks tried in order — the C++ callback overloads and variadic
/// chain.
/// Callback views cannot escape their invocation; retain an [`Any`] explicitly
/// when a value must outlive the callback:
///
/// ```compile_fail,E0521
/// use tvm_ffi::{structural_walk, AnyView, WalkOrder, WalkResult};
/// let mut retained: Option<AnyView<'static>> = None;
/// structural_walk(&1_i64, |value: AnyView<'_>| {
///     retained = Some(value);
///     WalkResult::Advance
/// }, WalkOrder::PreOrder).unwrap();
/// ```
///
/// The walker owns recursion: the handler runs once per value,
/// before or after the value's children according to `order`, and steers
/// traversal through the returned [`WalkResult`]. Walk accepts its existing
/// dispatcher borrows, closure observers and callback tuples; each adapter runs
/// through the same borrowed traversal engine.
///
/// Callbacks must report failures through [`Result`] and must not panic: a panic
/// reaching the non-unwinding native callback boundary aborts the process.
///
/// # Arguments
///
/// * `root` - Value borrowed for traversal, convertible into an [`AnyView`].
/// * `walker` - Mutable dispatcher, closure, or first-match callback tuple
///   implementing [`IntoDispatcher`]. Each callback observes a value and returns a
///   result convertible through [`IntoWalkResult`] to control traversal.
/// * `order` - [`WalkOrder::PreOrder`] observes each value before its children;
///   [`WalkOrder::PostOrder`] observes it after its children. Skipping children
///   with [`WalkResult::Skip`] only affects pre-order traversal.
///
/// # Examples
///
/// Collect integer leaves with a typed closure:
///
/// ```
/// use tvm_ffi::{structural_walk, Array, WalkOrder, WalkResult};
///
/// let root = Array::new(vec![1_i64, 2, 3]);
/// let mut integers = Vec::new();
/// structural_walk(&root, |value: i64| {
///     integers.push(value);
///     WalkResult::Advance
/// }, WalkOrder::PreOrder)?;
/// assert_eq!(integers, vec![1, 2, 3]);
/// # Ok::<(), tvm_ffi::Error>(())
/// ```
pub fn structural_walk<'input, R, M, H, D, O>(
    root: R,
    walker: H,
    order: WalkOrder,
) -> Result<Option<VisitInterrupt>>
where
    for<'v> H: IntoDispatcher<
            (AnyView<'v>, DefRegionKind),
            Option<Result<WalkResult>>,
            M,
            Target = D,
            Output = O,
        >,
    for<'s, 'v> &'s mut D: Dispatcher<(AnyView<'v>, DefRegionKind), Option<Result<WalkResult>>>,
    R: Into<AnyView<'input>>,
    O: BorrowMut<D>,
{
    let mut storage = walker.into_dispatcher();
    run_walker(root.into(), storage.borrow_mut(), order)
}
