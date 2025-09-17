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
use crate::derive::{Object, ObjectRef};
use crate::object::{Object, ObjectArc, ObjectCoreWithExtraItems};
use std::cmp::{Eq, Ord, Ordering, PartialEq, PartialOrd};
use std::ops::Deref;
use tvm_ffi_sys::TVMFFIShapeCell;
use tvm_ffi_sys::TVMFFITypeIndex as TypeIndex;

//-----------------------------------------------------
// Shape
//-----------------------------------------------------
// ShapeObj for heap-allocated shape
#[repr(C)]
#[derive(Object)]
#[type_key = "ffi.Shape"]
#[type_index(TypeIndex::kTVMFFIShape)]
pub struct ShapeObj {
    object: Object,
    data: TVMFFIShapeCell,
}

/// ABI stable owned Shape for ffi
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct Shape {
    data: ObjectArc<ShapeObj>,
}

impl Shape {
    /// Create a new empty Shape
    pub fn new() -> Self {
        let shape_obj = ShapeObj {
            object: Object::new(),
            data: TVMFFIShapeCell {
                data: std::ptr::null(),
                size: 0,
            },
        };
        Self {
            data: ObjectArc::new(shape_obj),
        }
    }

    /// Get the shape as a slice
    pub fn as_slice(&self) -> &[i64] {
        unsafe { std::slice::from_raw_parts(self.data.data.data, self.data.data.size) }
    }

    /// Fill the strides from the shape
    pub fn fill_strides_from_shape<T>(shape: T, strides: &mut [i64])
    where
        T: AsRef<[i64]>,
    {
        let shape = shape.as_ref();
        let mut stride = 1;
        for i in (0..shape.len()).rev() {
            strides[i] = stride;
            stride *= shape[i];
        }
    }
}

unsafe impl ObjectCoreWithExtraItems for ShapeObj {
    type ExtraItem = i64;
    fn extra_items_count(this: &Self) -> usize {
        this.data.size
    }
}

impl<T> From<T> for Shape
where
    T: AsRef<[i64]>,
{
    fn from(value: T) -> Self {
        unsafe {
            let value_slice: &[i64] = value.as_ref();
            let mut obj_arc = ObjectArc::new_with_extra_items(ShapeObj {
                object: Object::new(),
                data: TVMFFIShapeCell {
                    data: std::ptr::null(),
                    size: value_slice.len(),
                },
            });
            // reset the data ptr correctly after Arc is created
            obj_arc.data.data = ShapeObj::extra_items(&obj_arc).as_ptr();
            let extra_items = ShapeObj::extra_items_mut(&mut obj_arc);
            extra_items.copy_from_slice(value_slice);
            Self { data: obj_arc }
        }
    }
}

impl Deref for Shape {
    type Target = [i64];
    #[inline]
    fn deref(&self) -> &[i64] {
        self.as_slice()
    }
}

impl PartialEq for Shape {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl Eq for Shape {}

impl PartialOrd for Shape {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.as_slice().partial_cmp(other.as_slice())
    }
}

impl Ord for Shape {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}
