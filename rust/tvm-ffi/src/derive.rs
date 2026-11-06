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
//! The derive macros of object types and object references.

pub use tvm_ffi_macros::Object;

/// Declare an object reference: the managed handle to an object type, as a
/// C++ `ObjectRef` subclass is.
///
/// The struct holds one field, `ObjectArc<FooObj>`. The derive implements
/// [`ObjectRefCore`](crate::ObjectRefCore), the default
/// [`ObjectRefAnyCompatible`](crate::ObjectRefAnyCompatible) (conversions to
/// and from [`Any`](crate::Any), in-place downcasts, typed function
/// arguments), the upcast to [`ObjectRef`](crate::object::ObjectRef),
/// `Deref<Target = FooObj>` for field reads, `Debug`, and a private
/// `fn from_object(object: FooObj) -> Self` for the type's constructors.
///
/// # Options
///
/// `#[object_ref(...)]` takes:
///
/// * `parent = Ref` - The reference to the parent's object; `Self` converts
///   into it by [`unsafe_impl_object_ref_upcast!`](crate::unsafe_impl_object_ref_upcast),
///   whose requirement it carries. A further ancestor takes its own
///   `unsafe_impl_object_ref_upcast!`.
/// * `skip_deref` - No `Deref`, for a reference whose object is internal.
/// * `skip_debug` - `Debug` is written by hand.
/// * `skip_any_compatible` - `ObjectRefAnyCompatible` is written by hand,
///   for a typed reference with a stricter check or fallback conversions.
/// * `crate = path` - Where the generated code finds tvm-ffi, as for
///   [`rs_object`](macro@crate::rs_object).
///
/// # Examples
///
/// ```
/// use tvm_ffi::derive::ObjectRef;
/// use tvm_ffi::{rs_object, Object, ObjectArc};
///
/// #[rs_object("my_ext.Point")]
/// pub struct PointObj {
///     base: Object,
///     pub x: i64,
/// }
///
/// #[derive(Clone, ObjectRef)]
/// pub struct Point {
///     data: ObjectArc<PointObj>,
/// }
///
/// #[rs_object("my_ext.Point3", final)]
/// pub struct Point3Obj {
///     base: PointObj,
///     pub z: i64,
/// }
///
/// #[derive(Clone, ObjectRef)]
/// #[object_ref(parent = Point)]
/// pub struct Point3 {
///     data: ObjectArc<Point3Obj>,
/// }
///
/// impl Point3 {
///     pub fn new(x: i64, z: i64) -> Self {
///         Self::from_object(Point3Obj::new(PointObj::new(Object::new(), x), z))
///     }
/// }
///
/// let p = Point3::new(1, 2);
/// assert_eq!(p.x + p.z, 3);
/// let _parent: Point = p.into();
/// ```
pub use tvm_ffi_macros::ObjectRef;
