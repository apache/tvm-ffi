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
pub mod cold_init_lazy_lock;
pub mod dispatch;
pub mod module;
mod structural_common;
pub mod structural_map;
pub mod structural_mutate;
pub mod structural_visit;
pub mod structural_walk;
pub mod unchanged;
mod visit_error_context;

/// Borrow guarantees of the structural views, checked as `compile_fail` doctests.
///
/// `StructuralVisitorView::base` cannot outlive its view:
/// ```compile_fail
/// use tvm_ffi::{StructuralVisitorObj, StructuralVisitorView};
/// fn retain<D>(view: &StructuralVisitorView<'_, D>) -> &'static StructuralVisitorObj {
///     view.base()
/// }
/// ```
///
/// The visitor prefix cannot be converted into an owning FFI value:
///
/// ```compile_fail
/// use tvm_ffi::{Any, StructuralVisitorView};
/// fn retain<D>(view: &StructuralVisitorView<'_, D>) -> Any {
///     Any::from(view.base())
/// }
/// ```
///
/// A node borrowed through a copied `MutationView` cannot survive consumption of the view:
/// ```compile_fail,E0505
/// use tvm_ffi::{StructuralMutatorView, MutationView};
/// fn invalid<D>(value: MutationView<'_>, dispatch: &mut D, mutator: &mut StructuralMutatorView<'_, D>) {
///     let view = value.as_any_view();
///     let copy = view;
///     let node = copy.downcast_ref::<tvm_ffi::collections::array::ArrayObj>().unwrap();
///     mutator.default_mutate(dispatch, value).unwrap();
///     println!("{}", node.size);
/// }
/// ```
///
/// Moving the handle into a nested callback cannot bypass that borrow:
/// ```compile_fail,E0505
/// use tvm_ffi::{dispatch, structural_mutate, Any, MutationView, StructuralMutatorView, Result, UnchangedOr};
/// struct Inner<'a>(Option<MutationView<'a>>);
/// #[dispatch(mutate, (MutationView<'_>, &mut StructuralMutatorView<'_, Self>), Result<UnchangedOr<Any>>)]
/// impl<'a> Inner<'a> {
///     fn mutate_bool(&mut self, _: bool, mutator: &mut StructuralMutatorView<'_, Self>) -> Result<UnchangedOr<Any>> {
///         let pending = self.0.take().unwrap();
///         mutator.default_mutate(self, pending)
///     }
/// }
/// fn invalid(value: MutationView<'_>) {
///     let view = value.as_any_view();
///     let node = view.downcast_ref::<tvm_ffi::collections::array::ArrayObj>().unwrap();
///     structural_mutate(true, Inner(Some(value))).unwrap();
///     println!("{}", node.size);
/// }
/// ```
///
/// A copied inspection view also cannot survive low-level default mutation:
/// ```compile_fail,E0505
/// use tvm_ffi::{dispatch, Any, MutationView, StructuralMutatorView, Result, Unchanged, UnchangedOr};
/// struct Pass;
/// #[dispatch(mutate, (MutationView<'_>, &mut StructuralMutatorView<'_, Self>), Result<UnchangedOr<Any>>)]
/// impl Pass {
///     fn mutate_any(&mut self, value: MutationView<'_>, mutator: &mut StructuralMutatorView<'_, Self>) -> Result<UnchangedOr<Any>> {
///         let view = value.as_any_view();
///         let copy = view;
///         mutator.default_mutate(self, value).unwrap();
///         println!("{}", copy.type_index());
///         Ok(Unchanged.into())
///     }
/// }
/// ```
#[cfg(doctest)]
pub struct StructuralViewBorrowChecks;
