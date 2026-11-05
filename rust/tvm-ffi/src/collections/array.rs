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
use crate::collections::list::seq_cells;
use crate::derive::Object;
use crate::object::{Object, ObjectArc};
use crate::type_traits::{ContainerElement, RefFromAny};
use crate::{Any, AnyCompatible, AnyView, ObjectCoreWithExtraItems, ObjectRefCore};
use tvm_ffi_sys::TVMFFITypeIndex as TypeIndex;
use tvm_ffi_sys::{TVMFFIAny, TVMFFIObject};

#[repr(C)]
#[derive(Object)]
#[type_key = "ffi.Array"]
#[type_index(TypeIndex::kTVMFFIArray)]
pub struct ArrayObj {
    pub(crate) object: Object,
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

/// An immutable, reference-counted sequence of `T`, shared with C++ and
/// Python as `ffi.Array`.
///
/// Cloning an `Array` is cheap: it shares the same elements. Use [`List`]
/// for a sequence that changes in place.
///
/// [`List`]: crate::List
///
/// # Examples
///
/// ```
/// use tvm_ffi::Array;
///
/// let array = Array::from_vec(vec![1_i64, 2, 3]);
/// assert_eq!(array.len(), 3);
/// assert_eq!(array.get(1)?, 2);
/// assert!(array.get(3).is_err());
///
/// let doubled: Array<i64> = array.iter().map(|x| x * 2).collect();
/// assert_eq!(doubled.iter().collect::<Vec<_>>(), vec![2, 4, 6]);
/// # Ok::<(), tvm_ffi::Error>(())
/// ```
#[repr(C)]
#[derive(Clone)]
pub struct Array<T: ContainerElement + Clone> {
    data: ObjectArc<ArrayObj>,
    _marker: PhantomData<T>,
}

impl<T: ContainerElement + Clone> Debug for Array<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let full_name = std::any::type_name::<T>();
        let short_name = full_name.split("::").last().unwrap_or(full_name);
        write!(f, "Array<{}>[{}]", short_name, self.len())
    }
}

impl<T: ContainerElement + Clone> Default for Array<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: ContainerElement + Clone> From<Vec<T>> for Array<T> {
    fn from(items: Vec<T>) -> Self {
        Self::from_vec(items)
    }
}

unsafe impl<T: ContainerElement + Clone> ObjectRefCore for Array<T> {
    type ContainerType = ArrayObj;

    fn data(this: &Self) -> &ObjectArc<Self::ContainerType> {
        &this.data
    }

    fn into_data(this: Self) -> ObjectArc<Self::ContainerType> {
        this.data
    }

    unsafe fn from_data(data: ObjectArc<Self::ContainerType>) -> Self {
        Self {
            data,
            _marker: PhantomData,
        }
    }
}

impl<T: ContainerElement + Clone> Array<T> {
    /// Create an empty array.
    ///
    /// # Returns
    ///
    /// The empty array.
    ///
    /// See also [`Array::from_vec`].
    pub fn new() -> Self {
        Self::from_vec(Vec::new())
    }

    /// Create an array holding `items`, like `Array::from(items)`.
    ///
    /// # Arguments
    ///
    /// * `items` - The elements.
    ///
    /// # Returns
    ///
    /// The array.
    ///
    /// # Examples
    ///
    /// ```
    /// use tvm_ffi::Array;
    ///
    /// let array = Array::from_vec(vec![1_i64, 2, 3]);
    /// assert_eq!(array[1], 2);
    /// ```
    pub fn from_vec(items: Vec<T>) -> Self {
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
                let mut raw = TVMFFIAny::new();
                T::container_move_to_any(item, &mut raw);
                core::ptr::write(base_ptr.add(i), raw);
            }
        }
        // SAFETY: `arc` was allocated and initialized above as an `Array<T>`.
        unsafe { Self::from_data(arc) }
    }

    /// Return the number of elements.
    ///
    /// # Returns
    ///
    /// The number of elements.
    pub fn len(&self) -> usize {
        self.data.size as usize
    }

    /// Return `true` if the array has no elements.
    ///
    /// # Returns
    ///
    /// `true` if the array has no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return the element at `index`.
    ///
    /// # Arguments
    ///
    /// * `index` - The position of the element.
    ///
    /// # Returns
    ///
    /// The element at `index`.
    ///
    /// # Errors
    ///
    /// Returns an `IndexError` if `index` is out of bounds, or a `TypeError`
    /// if the element is not a `T`.
    pub fn get(&self, index: usize) -> Result<T, crate::Error> {
        if index >= self.len() {
            crate::bail!(crate::error::INDEX_ERROR, "Array get index out of bound");
        }
        unsafe {
            let container = self.data.deref();
            let base_ptr = container.data as *const TVMFFIAny;
            let raw_any_ref = &*base_ptr.add(index);

            match T::container_try_cast_from_any_view(raw_any_ref) {
                Ok(val) => Ok(val),
                Err(_) => crate::bail!(
                    crate::error::TYPE_ERROR,
                    "Failed to cast element at {} to {}",
                    index,
                    T::container_type_str()
                ),
            }
        }
    }

    /// Return an iterator over the elements.
    ///
    /// # Returns
    ///
    /// An iterator over the elements.
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

/// Lends `&T` from the array's cell in place. Every cell of an `Array<T>`
/// holds a `T` (checked when the array is built), so only a debug check runs.
impl<T: ContainerElement + Clone + RefFromAny> std::ops::Index<usize> for Array<T> {
    type Output = T;

    fn index(&self, index: usize) -> &T {
        let container = self.as_container();
        let len = container.size as usize;
        if index >= len {
            panic!(
                "Index out of bounds: the len is {} but the index is {}",
                len, index
            );
        }
        unsafe {
            let cell = &*(container.data as *const TVMFFIAny).add(index);
            debug_assert!(T::container_check_any_strict(cell));
            T::ref_from_any(cell)
        }
    }
}

// --- Iterator Implementations ---

pub struct ArrayIterator<'a, T: ContainerElement + Clone> {
    array: &'a Array<T>,
    index: usize,
    len: usize,
}

impl<'a, T: ContainerElement + Clone> Iterator for ArrayIterator<'a, T> {
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

impl<'a, T: ContainerElement + Clone> IntoIterator for &'a Array<T> {
    type Item = T;
    type IntoIter = ArrayIterator<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T: ContainerElement + Clone> FromIterator<T> for Array<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let items: Vec<T> = iter.into_iter().collect();
        Self::from_vec(items)
    }
}

// --- Any Type System Conversions ---

unsafe impl<T> AnyCompatible for Array<T>
where
    T: ContainerElement + Clone,
{
    #[inline]
    unsafe fn is_undefined(field: *const Self) -> bool {
        ObjectArc::is_null(std::ptr::addr_of!((*field).data))
    }

    fn type_str() -> String {
        format!("Array<{}>", T::container_type_str())
    }

    fn type_schema() -> String {
        crate::type_traits::type_schema(
            <ArrayObj as crate::ObjectCore>::TYPE_KEY,
            &[T::container_type_schema()],
        )
    }

    unsafe fn check_any_strict(data: &TVMFFIAny) -> bool {
        if data.type_index != TypeIndex::kTVMFFIArray as i32 {
            return false;
        }

        let container = &*(data.data_union.v_obj as *const ArrayObj);
        let base_ptr = container.data as *const TVMFFIAny;
        for i in 0..container.size {
            let elem_any = &*base_ptr.add(i as usize);
            if !T::container_check_any_strict(elem_any) {
                return false;
            }
        }
        true
    }

    unsafe fn match_constraints(data: &TVMFFIAny) -> bool {
        let container = &*(data.data_union.v_obj as *const ArrayObj);
        let base_ptr = container.data as *const TVMFFIAny;
        for i in 0..container.size {
            let elem_any = &*base_ptr.add(i as usize);
            if !T::container_match_constraints(elem_any) {
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
        // Mirrors C++ `SeqTypeTraitsBase::TryCastFromAnyView`: `ffi.List` is
        // the accepted other sequence type, and is always copied.
        if data.type_index != TypeIndex::kTVMFFIArray as i32
            && data.type_index != TypeIndex::kTVMFFIList as i32
        {
            return Err(());
        }

        // Fast path: an array whose elements all match shares the reference.
        if <Self as AnyCompatible>::check_any_strict(data) {
            return Ok(<Self as AnyCompatible>::copy_from_any_view_after_check(
                data,
            ));
        }

        // Slow path: convert element by element into a new array.
        let cells = seq_cells(data);
        let mut items = Vec::with_capacity(cells.len());
        for cell in cells {
            items.push(T::container_try_cast_from_any_view(cell)?);
        }
        Ok(Array::from_vec(items))
    }
}

// The cell's payload word is the single object pointer of the handle.
unsafe impl<T> crate::type_traits::RefFromAny for Array<T>
where
    T: ContainerElement + Clone,
{
    #[inline(always)]
    unsafe fn ref_from_any(cell: &TVMFFIAny) -> &Self {
        &*(core::ptr::addr_of!(cell.data_union) as *const Self)
    }
}

impl<T> TryFrom<Any> for Array<T>
where
    T: ContainerElement + Clone,
{
    type Error = crate::error::Error;

    fn try_from(value: Any) -> Result<Self, Self::Error> {
        let temp: TryFromTemp<Self> = TryFromTemp::try_from(value)?;
        Ok(TryFromTemp::into_value(temp))
    }
}

impl<'a, T> TryFrom<AnyView<'a>> for Array<T>
where
    T: ContainerElement + Clone,
{
    type Error = crate::error::Error;

    fn try_from(value: AnyView<'a>) -> Result<Self, Self::Error> {
        let temp: TryFromTemp<Self> = TryFromTemp::try_from(value)?;
        Ok(TryFromTemp::into_value(temp))
    }
}
