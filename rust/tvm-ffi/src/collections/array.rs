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
use crate::type_traits::{ContainerElement, RefFromAnyUnchecked};
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

// A Rust-allocated array holds `capacity` cells inline, of which the first `size`
// are initialized.
unsafe impl ObjectCoreWithExtraItems for ArrayObj {
    type ExtraItem = TVMFFIAny;
    fn extra_items_count(this: &Self) -> usize {
        this.capacity as usize
    }
}

/// Move `value` into the uninitialized `cell`.
unsafe fn write_cell<T: ContainerElement>(cell: *mut TVMFFIAny, value: T) {
    let mut raw = TVMFFIAny::new();
    T::container_move_to_any(value, &mut raw);
    std::ptr::write(cell, raw);
}

// Runs only for a Rust-allocated array; one allocated in C++ is released by its
// own deleter.
impl Drop for ArrayObj {
    fn drop(&mut self) {
        let cells = self.data as *const TVMFFIAny;
        for i in 0..self.size as usize {
            // SAFETY: the first `size` cells are initialized and owned by the array.
            drop(unsafe { Any::from_raw_ffi_any(*cells.add(i)) });
        }
    }
}

/// A reference-counted sequence of `T` with value semantics, shared with C++
/// and Python as `ffi.Array`.
///
/// Cloning an `Array` is cheap: it shares the same elements. The mutators
/// ([`push`], [`insert`], [`pop`], [`truncate`], [`set`]) are copy-on-write:
/// they copy the elements when the array is shared, and write in place when
/// this handle is the only reference, so other handles never see the change.
/// Use [`List`] for a sequence whose changes all handles see.
///
/// [`List`]: crate::List
/// [`push`]: Array::push
/// [`insert`]: Array::insert
/// [`pop`]: Array::pop
/// [`truncate`]: Array::truncate
/// [`set`]: Array::set
///
/// # Examples
///
/// ```
/// use tvm_ffi::Array;
///
/// let mut array = Array::from_vec(vec![1_i64, 2, 3]);
/// assert_eq!(array.len(), 3);
/// assert_eq!(array.get(1)?, 2);
/// assert!(array.get(3).is_err());
///
/// let snapshot = array.clone();
/// array.push(4);
/// assert_eq!(array.len(), 4);
/// assert_eq!(snapshot.len(), 3);
///
/// let doubled: Array<i64> = array.iter().map(|x| x * 2).collect();
/// assert_eq!(doubled.iter().copied().collect::<Vec<_>>(), vec![2, 4, 6, 8]);
/// # Ok::<(), tvm_ffi::Error>(())
/// ```
#[repr(C)]
#[derive(Clone)]
pub struct Array<T> {
    data: ObjectArc<ArrayObj>,
    _marker: PhantomData<T>,
}

// SAFETY: a shared `Array` is never mutated (mutators copy unless this handle
// is unique), so sharing it is as safe as sharing its elements.
unsafe impl<T: Send + Sync> Send for Array<T> {}
unsafe impl<T: Send + Sync> Sync for Array<T> {}

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

/// An array upcasts to the root [`ObjectRef`](crate::object::ObjectRef).
impl<T: ContainerElement + Clone> From<Array<T>> for crate::object::ObjectRef {
    #[inline]
    fn from(value: Array<T>) -> Self {
        crate::object::into_root_object_ref(value)
    }
}

/// The strict downcast from the root [`ObjectRef`](crate::object::ObjectRef).
impl<T: ContainerElement + Clone> TryFrom<crate::object::ObjectRef> for Array<T> {
    type Error = crate::error::Error;
    #[inline]
    fn try_from(value: crate::object::ObjectRef) -> crate::error::Result<Self> {
        crate::object::root_try_into(value)
    }
}

impl<T: ContainerElement + Clone> TryFrom<&crate::object::ObjectRef> for Array<T> {
    type Error = crate::error::Error;
    #[inline]
    fn try_from(value: &crate::object::ObjectRef) -> crate::error::Result<Self> {
        crate::object::root_try_into_ref(value)
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
            let raw_ptr = ObjectArc::as_ptr(&arc) as *mut ArrayObj;
            let container = &mut *raw_ptr;

            let base_ptr = ArrayObj::extra_items_mut(container).as_mut_ptr();
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

    /// Append `value` to the end.
    ///
    /// See also [`List::push`](crate::List::push).
    pub fn push(&mut self, value: T) {
        let size = self.len();
        let container = self.make_mut(size + 1);
        // SAFETY: `make_mut` left room for one more cell past `size`.
        unsafe { write_cell((container.data as *mut TVMFFIAny).add(size), value) };
        container.size += 1;
    }

    /// Insert `value` at `index`, shifting the elements after it.
    ///
    /// # Errors
    ///
    /// Returns an `IndexError` if `index > len`.
    ///
    /// See also [`List::insert`](crate::List::insert).
    pub fn insert(&mut self, index: usize, value: T) -> Result<(), crate::Error> {
        let size = self.len();
        if index > size {
            crate::bail!(crate::error::INDEX_ERROR, "Array insert index out of bound");
        }
        let container = self.make_mut(size + 1);
        // SAFETY: `make_mut` left room for one more cell; the tail moves up by one.
        unsafe {
            let cell = (container.data as *mut TVMFFIAny).add(index);
            std::ptr::copy(cell, cell.add(1), size - index);
            write_cell(cell, value);
        }
        container.size += 1;
        Ok(())
    }

    /// Remove and return the last element.
    ///
    /// # Errors
    ///
    /// Returns an `IndexError` if the array is empty.
    ///
    /// See also [`List::pop`](crate::List::pop).
    pub fn pop(&mut self) -> Result<T, crate::Error> {
        let size = self.len();
        if size == 0 {
            crate::bail!(crate::error::INDEX_ERROR, "Array pop on an empty array");
        }
        let container = self.make_mut(size);
        container.size -= 1;
        // SAFETY: the last cell is initialized, holds a `T`, and is no longer
        // counted by `size`, so it is moved out exactly once.
        unsafe {
            let mut cell = std::ptr::read((container.data as *const TVMFFIAny).add(size - 1));
            debug_assert!(T::container_check_any_strict(&cell));
            Ok(T::container_move_from_any_after_check(&mut cell))
        }
    }

    /// Shorten the array to `len` elements, dropping the rest; a no-op when it
    /// is already that short.
    ///
    /// See also [`List::truncate`](crate::List::truncate).
    pub fn truncate(&mut self, len: usize) {
        let size = self.len();
        if len >= size {
            return;
        }
        let container = self.make_mut(size);
        // Shrink `size` first, so a panicking element drop leaks rather than
        // double-frees the remaining cells.
        container.size = len as i64;
        let cells = container.data as *const TVMFFIAny;
        for i in len..size {
            // SAFETY: cells `len..size` are initialized and no longer counted.
            drop(unsafe { Any::from_raw_ffi_any(*cells.add(i)) });
        }
    }

    /// Replace the element at `index`.
    ///
    /// # Errors
    ///
    /// Returns an `IndexError` if `index >= len`.
    ///
    /// See also [`List::set`](crate::List::set).
    pub fn set(&mut self, index: usize, value: T) -> Result<(), crate::Error> {
        let size = self.len();
        if index >= size {
            crate::bail!(crate::error::INDEX_ERROR, "Array set index out of bound");
        }
        let container = self.make_mut(size);
        // SAFETY: the cell at `index` is initialized; its old value is dropped
        // after the new one is written.
        unsafe {
            let cell = (container.data as *mut TVMFFIAny).add(index);
            let old = Any::from_raw_ffi_any(std::ptr::read(cell));
            write_cell(cell, value);
            drop(old);
        }
        Ok(())
    }

    /// Make this handle the only reference to its cells, with room for at least
    /// `capacity` of them, and return the container to edit in place.
    fn make_mut(&mut self, capacity: usize) -> &mut ArrayObj {
        // As C++ `Array::CopyOnWrite(reserve)`.
        let unique = ObjectArc::is_unique(&self.data);
        let current = self.data.capacity as usize;
        if !unique || capacity > current {
            let capacity = if capacity > current {
                (current * 2).max(capacity)
            } else {
                current
            };
            self.copy_to_new_buffer(capacity, unique);
        }
        // SAFETY: this handle is now the only reference, so no one else observes
        // the container while it is mutably borrowed.
        unsafe { &mut *ObjectArc::as_mut_ptr(&mut self.data) }
    }

    /// Replace `self` with a unique array of `capacity` cells holding the same
    /// elements: moved out of a `unique` source, otherwise each re-referenced.
    fn copy_to_new_buffer(&mut self, capacity: usize, unique: bool) {
        let size = self.len();
        let mut grown = Self::new_with_capacity(Vec::new(), capacity);
        let source = self.as_container().data as *const TVMFFIAny;
        // SAFETY: `grown` is fresh with room for `size` cells. A unique source
        // forgets the cells it moved; a shared one keeps its own references.
        unsafe {
            let target = ObjectArc::as_mut_ptr(&mut grown.data);
            let cells = (*target).data as *mut TVMFFIAny;
            if unique {
                std::ptr::copy_nonoverlapping(source, cells, size);
                (*ObjectArc::as_mut_ptr(&mut self.data)).size = 0;
            } else {
                for i in 0..size {
                    let copy = Any::from(AnyView::from_raw_ffi_any(*source.add(i)));
                    std::ptr::write(cells.add(i), Any::into_raw_ffi_any(copy));
                }
            }
            (*target).size = size as i64;
        }
        *self = grown;
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

    /// The cells of the array, one per element.
    #[inline]
    fn cells(&self) -> &[TVMFFIAny] {
        let container = self.as_container();
        if container.size == 0 {
            return &[];
        }
        // SAFETY: a non-empty array owns `size` initialized cells at `data`,
        // and an `Array` is immutable while it is borrowed.
        unsafe {
            std::slice::from_raw_parts(container.data as *const TVMFFIAny, container.size as usize)
        }
    }

    #[inline]
    fn as_container(&self) -> &ArrayObj {
        // SAFETY: `self.data` keeps the array object alive while it is borrowed.
        unsafe { &*ObjectArc::as_ptr(&self.data) }
    }
}

// --- Index Implementation ---

/// Lends `&T` from the array's cell in place. Every cell of an `Array<T>`
/// holds a `T` (checked when the array is built), so only a debug check runs.
impl<T: ContainerElement + Clone + RefFromAnyUnchecked> std::ops::Index<usize> for Array<T> {
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
            T::ref_from_any_unchecked(cell)
        }
    }
}

// --- Iterator Implementations ---

impl<T: ContainerElement + Clone + RefFromAnyUnchecked> Array<T> {
    /// Return an iterator that borrows each element in place, without a
    /// reference-count bump, as `array[i]` does.
    ///
    /// Element types stored in another representation, such as `bool` or
    /// `i32` (held as `i64`), iterate by value instead: `for x in array`.
    ///
    /// ```
    /// use tvm_ffi::Array;
    /// let array = Array::from_vec(vec![1_i64, 2, 3]);
    /// assert_eq!(array.iter().sum::<i64>(), 6);
    /// ```
    pub fn iter(&self) -> Iter<'_, T> {
        Iter {
            cells: self.cells().iter(),
            _marker: std::marker::PhantomData,
        }
    }
}

/// An iterator that borrows the elements of an [`Array`] in place.
pub struct Iter<'a, T> {
    cells: std::slice::Iter<'a, TVMFFIAny>,
    _marker: std::marker::PhantomData<&'a T>,
}

impl<'a, T: ContainerElement + RefFromAnyUnchecked> Iterator for Iter<'a, T> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<&'a T> {
        self.cells.next().map(|cell| unsafe {
            // SAFETY: every cell of an `Array<T>` holds a `T`, checked when the
            // array is built.
            debug_assert!(T::container_check_any_strict(cell));
            T::ref_from_any_unchecked(cell)
        })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.cells.size_hint()
    }
}

impl<T: ContainerElement + RefFromAnyUnchecked> DoubleEndedIterator for Iter<'_, T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.cells.next_back().map(|cell| unsafe {
            debug_assert!(T::container_check_any_strict(cell));
            T::ref_from_any_unchecked(cell)
        })
    }
}

impl<T: ContainerElement + RefFromAnyUnchecked> ExactSizeIterator for Iter<'_, T> {}

impl<'a, T: ContainerElement + Clone + RefFromAnyUnchecked> IntoIterator for &'a Array<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// An iterator that yields the elements of an [`Array`] by value.
pub struct IntoIter<T> {
    array: Array<T>,
    index: usize,
}

impl<T: ContainerElement + Clone> Iterator for IntoIter<T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        let cell = self.array.cells().get(self.index)?;
        self.index += 1;
        // SAFETY: every cell of an `Array<T>` holds a `T`, checked when the
        // array is built.
        Some(unsafe {
            debug_assert!(T::container_check_any_strict(cell));
            T::container_copy_from_any_view_after_check(cell)
        })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.array.len() - self.index;
        (remaining, Some(remaining))
    }
}

impl<T: ContainerElement + Clone> ExactSizeIterator for IntoIter<T> {}

/// Iterate by value, for every element type. Cloning an `Array` first costs one
/// reference-count bump for the whole array.
impl<T: ContainerElement + Clone> IntoIterator for Array<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            array: self,
            index: 0,
        }
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
        data.data_union.v_obj = ObjectArc::as_ptr(Self::data(src)) as *mut TVMFFIObject;
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
unsafe impl<T> crate::type_traits::RefFromAnyUnchecked for Array<T>
where
    T: ContainerElement + Clone,
{
    #[inline(always)]
    unsafe fn ref_from_any_unchecked(cell: &TVMFFIAny) -> &Self {
        &*(core::ptr::addr_of!(cell.data_union) as *const Self)
    }
}

unsafe impl<T> crate::type_traits::DowncastRefFromAny for Array<T>
where
    T: ContainerElement + Clone,
{
    #[inline(always)]
    unsafe fn downcast_ref_from_any(cell: &TVMFFIAny) -> Option<&Self> {
        crate::type_traits::downcast_ref_from_any_in_place::<Self>(cell)
    }
}

unsafe impl<T> crate::type_traits::DowncastRefFromObject for Array<T>
where
    T: ContainerElement + Clone,
{
    #[inline(always)]
    unsafe fn downcast_ref_from_object(place: &*mut TVMFFIObject) -> Option<&Self> {
        crate::type_traits::downcast_ref_from_object_in_place::<Self>(place)
    }
}

impl<'b, T> TryFrom<&'b Any> for Array<T>
where
    T: ContainerElement + Clone,
{
    type Error = crate::error::Error;

    fn try_from(value: &'b Any) -> Result<Self, Self::Error> {
        let temp: TryFromTemp<Self> = TryFromTemp::try_from(value)?;
        Ok(TryFromTemp::into_value(temp))
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
