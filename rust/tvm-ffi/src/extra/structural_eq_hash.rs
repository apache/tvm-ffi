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
//! Structural equality and hashing of values.

use crate::reflection::AccessPath;
use crate::{AnyView, Array, Result};

/// Options of structural equality and hashing.
///
/// See also [`structural_equal_with`], [`structural_hash_with`] and
/// [`get_first_structural_mismatch_with`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StructuralOptions {
    /// Whether free variables (variables without a definition site) can be
    /// mapped to each other.
    pub map_free_vars: bool,
    /// Whether to skip the data content of tensors.
    pub skip_tensor_content: bool,
}

/// Whether two values are structurally equal: they have the same structure
/// and content, compared recursively through containers and reflected
/// fields, not by identity.
///
/// Objects compare as their [`SEqHashKind`](crate::reflection::SEqHashKind) declares.
///
/// # Arguments
///
/// * `lhs` - The left value.
/// * `rhs` - The right value.
///
/// # Returns
///
/// Whether the values are structurally equal.
///
/// # Errors
///
/// Returns an error if an object in either value does not support structural
/// equality.
///
/// # Examples
///
/// ```standalone_crate
/// use tvm_ffi::derive::ObjectRef;
/// use tvm_ffi::{rs_object, structural_equal, Array, Object, ObjectArc};
///
/// #[rs_object(type_key = "my_ext.MyPair", final, s_eq_hash_kind = "TreeNode")]
/// pub struct MyPairObj {
///     base_: Object,
///     pub first: i64,
///     pub second: Array<i64>,
/// }
///
/// #[derive(Clone, ObjectRef)]
/// pub struct MyPair {
///     data: ObjectArc<MyPairObj>,
/// }
///
/// let pair = |second: Vec<i64>| {
///     MyPair::from_object(MyPairObj::new(Object::new(), 1, Array::from_vec(second)))
/// };
/// assert!(structural_equal(&pair(vec![2, 3]), &pair(vec![2, 3]))?);
/// assert!(!structural_equal(&pair(vec![2, 3]), &pair(vec![2, 4]))?);
/// # Ok::<(), tvm_ffi::Error>(())
/// ```
pub fn structural_equal<'l, 'r>(
    lhs: impl Into<AnyView<'l>>,
    rhs: impl Into<AnyView<'r>>,
) -> Result<bool> {
    structural_equal_with(lhs, rhs, StructuralOptions::default())
}

/// As [`structural_equal`], with [`StructuralOptions`].
pub fn structural_equal_with<'l, 'r>(
    lhs: impl Into<AnyView<'l>>,
    rhs: impl Into<AnyView<'r>>,
    options: StructuralOptions,
) -> Result<bool> {
    let args = [
        lhs.into(),
        rhs.into(),
        AnyView::from(&options.map_free_vars),
        AnyView::from(&options.skip_tensor_content),
    ];
    crate::cached_global_func!("ffi.StructuralEqual")
        .call_packed(&args)?
        .try_into()
}

/// The structural hash of a value, consistent with [`structural_equal`]:
/// structurally equal values have the same hash.
///
/// # Arguments
///
/// * `value` - The value to hash.
///
/// # Returns
///
/// The hash.
///
/// # Errors
///
/// Returns an error if an object in the value does not support structural
/// hashing.
///
/// See also [`structural_equal`].
pub fn structural_hash<'a>(value: impl Into<AnyView<'a>>) -> Result<u64> {
    structural_hash_with(value, StructuralOptions::default())
}

/// As [`structural_hash`], with [`StructuralOptions`].
pub fn structural_hash_with<'a>(
    value: impl Into<AnyView<'a>>,
    options: StructuralOptions,
) -> Result<u64> {
    let args = [
        value.into(),
        AnyView::from(&options.map_free_vars),
        AnyView::from(&options.skip_tensor_content),
    ];
    let hash: i64 = crate::cached_global_func!("ffi.StructuralHash")
        .call_packed(&args)?
        .try_into()?;
    // The hash crosses the ABI as an `i64` bit pattern.
    Ok(hash as u64)
}

/// The first place where two values are not structurally equal, as the
/// [`AccessPath`] of that place in each value.
///
/// # Arguments
///
/// * `lhs` - The left value.
/// * `rhs` - The right value.
///
/// # Returns
///
/// `None` if the values are structurally equal, else the paths of the first
/// mismatch in `lhs` and in `rhs`.
///
/// # Errors
///
/// Returns an error if an object in either value does not support structural
/// equality.
///
/// See also [`structural_equal`].
pub fn get_first_structural_mismatch<'l, 'r>(
    lhs: impl Into<AnyView<'l>>,
    rhs: impl Into<AnyView<'r>>,
) -> Result<Option<(AccessPath, AccessPath)>> {
    get_first_structural_mismatch_with(lhs, rhs, StructuralOptions::default())
}

/// As [`get_first_structural_mismatch`], with [`StructuralOptions`].
pub fn get_first_structural_mismatch_with<'l, 'r>(
    lhs: impl Into<AnyView<'l>>,
    rhs: impl Into<AnyView<'r>>,
    options: StructuralOptions,
) -> Result<Option<(AccessPath, AccessPath)>> {
    let args = [
        lhs.into(),
        rhs.into(),
        AnyView::from(&options.map_free_vars),
        AnyView::from(&options.skip_tensor_content),
    ];
    // The pair crosses the ABI as a two-item array.
    let pair: Option<Array<AccessPath>> =
        crate::cached_global_func!("ffi.GetFirstStructuralMismatch")
            .call_packed(&args)?
            .try_into()?;
    Ok(pair.map(|pair| (pair[0].clone(), pair[1].clone())))
}
