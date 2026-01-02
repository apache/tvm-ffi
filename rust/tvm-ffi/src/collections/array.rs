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
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::Deref;

use crate::any::TryFromTemp;
use crate::derive::Object;
use crate::object::{Object, ObjectArc};
use crate::{Any, AnyCompatible, AnyView, ObjectCoreWithExtraItems, ObjectRefCore};
use tvm_ffi_sys::TVMFFITypeIndex as TypeIndex;
use tvm_ffi_sys::{TVMFFIAny, TVMFFIObject};

#[repr(C)]
#[derive(Object)]
#[type_key = "ffi.Array"]
#[type_index(TypeIndex::kTVMFFIArray)]
pub struct ArrayObj {
    pub object: Object,
    /// Pointer to the start of the element buffer (AddressOf(0)).
    pub data: *mut core::ffi::c_void,
    pub size: i64,
    pub capacity: i64,
    /// Optional custom deleter for the data pointer.
    pub data_deleter: Option<unsafe extern "C" fn(*mut core::ffi::c_void)>,
}

unsafe impl ObjectCoreWithExtraItems for ArrayObj {
    type ExtraItem = TVMFFIAny;
    fn extra_items_count(this: &Self) -> usize {
        this.size as usize
    }
}

#[repr(C)]
#[derive(Clone)]
pub struct Array<T: AnyCompatible + Clone> {
    data: ObjectArc<ArrayObj>,
    _marker: PhantomData<T>,
}

impl<T: AnyCompatible + Clone> Debug for Array<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let full_name = std::any::type_name::<T>();
        let short_name = full_name.split("::").last().unwrap_or(full_name);
        write!(f, "Array<{}>[{}]", short_name, self.len())
    }
}

impl<T: AnyCompatible + Clone> Default for Array<T> {
    fn default() -> Self {
        Self::new(vec![])
    }
}

unsafe impl<T: AnyCompatible + Clone> ObjectRefCore for Array<T> {
    type ContainerType = ArrayObj;

    fn data(this: &Self) -> &ObjectArc<Self::ContainerType> {
        &this.data
    }

    fn into_data(this: Self) -> ObjectArc<Self::ContainerType> {
        this.data
    }

    fn from_data(data: ObjectArc<Self::ContainerType>) -> Self {
        Self {
            data,
            _marker: PhantomData,
        }
    }
}

impl<T: AnyCompatible + Clone> Array<T> {
    /// Creates a new Array from a vector of items.
    pub fn new(items: Vec<T>) -> Self {
        let capacity = items.len();
        Self::new_with_capacity(items, capacity)
    }

    /// Internal helper to allocate an ArrayObj with specific headroom.
    fn new_with_capacity(items: Vec<T>, capacity: usize) -> Self {
        let size = items.len();

        // Allocate with capacity
        let arc = ObjectArc::<ArrayObj>::new_with_extra_items(ArrayObj {
            object: Object::new(),
            data: core::ptr::null_mut(),
            size: size as i64,
            capacity: capacity as i64,
            data_deleter: None,
        });

        unsafe {
            let raw_ptr = ObjectArc::as_raw(&arc) as *mut ArrayObj;
            let container = &mut *raw_ptr;

            let base_ptr = ArrayObj::extra_items_mut(container).as_ptr() as *mut TVMFFIAny;
            container.data = base_ptr as *mut _;

            for (i, item) in items.into_iter().enumerate() {
                let any: Any = Any::from(item);
                let raw = Any::into_raw_ffi_any(any);
                core::ptr::write(base_ptr.add(i), raw);
            }
        }
        Self::from_data(arc)
    }

    pub fn len(&self) -> usize {
        self.data.size as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Retrieves an item at the given index.
    pub fn get(&self, index: usize) -> Result<T, crate::Error> {
        if index >= self.len() {
            crate::bail!(crate::error::INDEX_ERROR, "Array get index out of bound");
        }
        unsafe {
            let container = self.data.deref();
            let base_ptr = container.data as *const TVMFFIAny;
            let raw_any_ref = &*base_ptr.add(index);

            match T::try_cast_from_any_view(raw_any_ref) {
                Ok(val) => Ok(val),
                Err(_) => crate::bail!(
                    crate::error::TYPE_ERROR,
                    "Failed to cast element at {} to {}",
                    index,
                    T::type_str()
                ),
            }
        }
    }

    pub fn iter(&'_ self) -> ArrayIterator<'_, T> {
        ArrayIterator {
            array: self,
            index: 0,
            len: self.len(),
        }
    }

    #[inline]
    fn as_container(&self) -> &ArrayObj {
        unsafe {
            let ptr = ObjectArc::as_raw(&self.data) as *const ArrayObj;
            &*ptr
        }
    }
}

// --- Index Implementation ---

impl<T: AnyCompatible + Clone> std::ops::Index<usize> for Array<T> {
    type Output = AnyView<'static>;

    fn index(&self, index: usize) -> &Self::Output {
        let container = self.as_container();
        let len = container.size as usize;
        if index >= len {
            panic!(
                "Index out of bounds: the len is {} but the index is {}",
                len, index
            );
        }
        unsafe {
            let ptr = (container.data as *const AnyView<'static>).add(index);
            &*ptr
        }
    }
}

// --- Iterator Implementations ---

pub struct ArrayIterator<'a, T: AnyCompatible + Clone> {
    array: &'a Array<T>,
    index: usize,
    len: usize,
}

impl<'a, T: AnyCompatible + Clone> Iterator for ArrayIterator<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.len {
            let item = self.array.get(self.index).ok();
            self.index += 1;
            item
        } else {
            None
        }
    }
}

impl<'a, T: AnyCompatible + Clone> IntoIterator for &'a Array<T> {
    type Item = T;
    type IntoIter = ArrayIterator<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T: AnyCompatible + Clone> FromIterator<T> for Array<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let items: Vec<T> = iter.into_iter().collect();
        Self::new(items)
    }
}

// --- Any Type System Conversions ---

unsafe impl<T> AnyCompatible for Array<T>
where
    T: AnyCompatible + Clone + 'static,
{
    fn type_str() -> String {
        format!("Array<{}>", T::type_str())
    }

    unsafe fn check_any_strict(data: &TVMFFIAny) -> bool {
        if data.type_index != TypeIndex::kTVMFFIArray as i32 {
            return false;
        }

        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<Any>() {
            return true;
        }

        let container = &*(data.data_union.v_obj as *const ArrayObj);
        let base_ptr = container.data as *const TVMFFIAny;
        for i in 0..container.size {
            let elem_any = &*base_ptr.add(i as usize);
            if !T::check_any_strict(elem_any) {
                return false;
            }
        }
        true
    }

    unsafe fn copy_to_any_view(src: &Self, data: &mut TVMFFIAny) {
        data.type_index = TypeIndex::kTVMFFIArray as i32;
        data.data_union.v_obj = ObjectArc::as_raw(Self::data(src)) as *mut TVMFFIObject;
        data.small_str_len = 0;
    }

    unsafe fn move_to_any(src: Self, data: &mut TVMFFIAny) {
        data.type_index = TypeIndex::kTVMFFIArray as i32;
        data.data_union.v_obj = ObjectArc::into_raw(Self::into_data(src)) as *mut TVMFFIObject;
        data.small_str_len = 0;
    }

    unsafe fn copy_from_any_view_after_check(data: &TVMFFIAny) -> Self {
        let ptr = data.data_union.v_obj as *const ArrayObj;
        crate::object::unsafe_::inc_ref(ptr as *mut TVMFFIObject);
        Self::from_data(ObjectArc::from_raw(ptr))
    }

    unsafe fn move_from_any_after_check(data: &mut TVMFFIAny) -> Self {
        let ptr = data.data_union.v_obj as *const ArrayObj;
        let obj = Self::from_data(ObjectArc::from_raw(ptr));

        data.type_index = TypeIndex::kTVMFFINone as i32;
        data.data_union.v_int64 = 0;

        obj
    }

    unsafe fn try_cast_from_any_view(data: &TVMFFIAny) -> Result<Self, ()> {
        if data.type_index != TypeIndex::kTVMFFIArray as i32 {
            return Err(());
        }

        // Fast path: if types match exactly, we can just copy the reference.
        if Self::check_any_strict(data) {
            return Ok(Self::copy_from_any_view_after_check(data));
        }

        // Slow path: try to convert element by element.
        let container = &*(data.data_union.v_obj as *const ArrayObj);
        let base_ptr = container.data as *const TVMFFIAny;
        let mut items = Vec::with_capacity(container.size as usize);

        for i in 0..container.size {
            let any_v = &*base_ptr.add(i as usize);
            if let Ok(item) = T::try_cast_from_any_view(any_v) {
                items.push(item);
            } else {
                return Err(());
            }
        }

        Ok(Array::new(items))
    }
}

impl<T> TryFrom<Any> for Array<T>
where
    T: AnyCompatible + Clone + 'static,
{
    type Error = crate::error::Error;

    fn try_from(value: Any) -> Result<Self, Self::Error> {
        let temp: TryFromTemp<Self> = TryFromTemp::try_from(value)?;
        Ok(TryFromTemp::into_value(temp))
    }
}

impl<'a, T> TryFrom<AnyView<'a>> for Array<T>
where
    T: AnyCompatible + Clone + 'static,
{
    type Error = crate::error::Error;

    fn try_from(value: AnyView<'a>) -> Result<Self, Self::Error> {
        let temp: TryFromTemp<Self> = TryFromTemp::try_from(value)?;
        Ok(TryFromTemp::into_value(temp))
    }
}
