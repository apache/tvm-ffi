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
//! Mutable [`List`] container sharing the `ffi.List` object layout with C++.
//!
//! `ListObj` mirrors C++ `ListObj` (`Object` followed by `TVMFFISeqCell`), so
//! Rust reads and edits the element cells in place: [`List::get`],
//! [`List::set`], [`List::insert`], [`List::remove`], [`List::clear`] and
//! iteration touch the buffer directly, with no FFI call. The object and its
//! element buffer are allocated by the C++ runtime (`ffi.List`), because the
//! buffer is a separate allocation that `data_deleter` has to free with the
//! allocator that produced it. For the same reason, growing a full buffer is
//! left to C++: [`List::push`] and [`List::insert`] call `ffi.ListAppend` or
//! `ffi.ListInsert` only when there is no spare capacity.
//!
//! A `List` is a shared handle, as in C++ and Python: cloning it bumps the
//! refcount, and a mutation through one handle is seen through all of them.
//! That is why the mutating methods take `&self`, and why elements are
//! returned by value rather than by reference. Like its C++ counterpart it is
//! not synchronized; the raw buffer pointer keeps it `!Send` and `!Sync`.
//!
//! `Any` conversions follow C++ `SeqTypeTraitsBase` (seq_base.h): an `ffi.List`
//! whose elements all match `T` is shared; an `ffi.Array`, or a list whose
//! elements need converting, is copied into a new list.
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ptr;

use crate::any::TryFromTemp;
use crate::collections::map::element_view;
use crate::derive::Object;
use crate::object::{Object, ObjectArc};
use crate::type_traits::ContainerElement;
use crate::{Any, AnyCompatible, AnyView, Error, ObjectRefCore, Result};
use tvm_ffi_sys::TVMFFITypeIndex as TypeIndex;
use tvm_ffi_sys::{TVMFFIAny, TVMFFIObject};

/// Container object for [`List`], laid out as C++ `ListObj`. Unlike
/// `ffi.Array`, the element buffer `data` is a separate allocation that the
/// C++ runtime replaces when the list grows.
#[repr(C)]
#[derive(Object)]
#[type_key = "ffi.List"]
#[type_index(TypeIndex::kTVMFFIList)]
pub struct ListObj {
    pub object: Object,
    /// Pointer to the start of the element buffer.
    pub data: *mut core::ffi::c_void,
    pub size: i64,
    pub capacity: i64,
    /// Deleter for the element buffer.
    pub data_deleter: Option<unsafe extern "C" fn(*mut core::ffi::c_void)>,
}

impl ListObj {
    /// Cell `index` of the buffer; the caller keeps `index` below `capacity`.
    #[inline]
    unsafe fn cell(this: *mut ListObj, index: usize) -> *mut TVMFFIAny {
        ((*this).data as *mut TVMFFIAny).add(index)
    }

    /// Drops the elements from the back down to `len`, like C++ `ShrinkBy`.
    unsafe fn truncate(this: *mut ListObj, len: usize) {
        while (*this).size as usize > len {
            (*this).size -= 1;
            drop(take_cell(Self::cell(this, (*this).size as usize)));
        }
    }
}

/// Moves the element out of `cell`; the caller overwrites the cell or drops
/// it from the list's accounting.
#[inline]
unsafe fn take_cell(cell: *mut TVMFFIAny) -> Any {
    Any::from_raw_ffi_any(ptr::read(cell))
}

#[inline]
unsafe fn write_cell<T: ContainerElement>(cell: *mut TVMFFIAny, value: T) {
    let mut raw = TVMFFIAny::new();
    T::container_move_to_any(value, &mut raw);
    ptr::write(cell, raw);
}

// The error paths stay out of line so the hot paths are small enough to inline.
#[cold]
#[inline(never)]
fn index_error<T>(what: &str) -> Result<T> {
    crate::bail!(
        crate::error::INDEX_ERROR,
        "List {} index out of bound",
        what
    )
}

#[cold]
#[inline(never)]
fn type_error<T: ContainerElement>(index: usize) -> Result<T> {
    crate::bail!(
        crate::error::TYPE_ERROR,
        "Failed to cast element at {} to {}",
        index,
        T::container_type_str()
    )
}

/// Mutable, reference-counted sequence of `T`, sharing its underlying
/// `ListObj` with C++. Cloning is cheap (it bumps the refcount) and yields a
/// handle to the same list.
#[repr(C)]
pub struct List<T> {
    data: ObjectArc<ListObj>,
    _marker: PhantomData<T>,
}

impl<T> Clone for List<T> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            _marker: PhantomData,
        }
    }
}

unsafe impl<T> ObjectRefCore for List<T> {
    type ContainerType = ListObj;

    fn data(this: &Self) -> &ObjectArc<ListObj> {
        &this.data
    }

    fn into_data(this: Self) -> ObjectArc<ListObj> {
        this.data
    }

    unsafe fn from_data(data: ObjectArc<ListObj>) -> Self {
        Self {
            data,
            _marker: PhantomData,
        }
    }
}

/// The elements of the `ffi.List` or `ffi.Array` object `data` holds; both
/// share the `TVMFFISeqCell` layout, so this is the one accessor for either.
///
/// # Safety
/// `data` must hold one of those two objects, and the list must not be
/// mutated while the slice is alive.
pub(crate) unsafe fn seq_cells(data: &TVMFFIAny) -> &[TVMFFIAny] {
    let seq = &*(data.data_union.v_obj as *const ListObj);
    if seq.size == 0 {
        return &[];
    }
    std::slice::from_raw_parts(seq.data as *const TVMFFIAny, seq.size as usize)
}

impl<T: ContainerElement> List<T> {
    /// Creates a new, empty list (via an `ffi.List()` call to the C++ runtime).
    pub fn new() -> Self {
        Self::from_items(&[]).expect("ffi.List() failed to construct an empty list")
    }

    /// Builds a list by calling the C++ `ffi.List` constructor.
    fn from_items(items: &[T]) -> Result<Self> {
        let args: Vec<AnyView<'_>> = items.iter().map(element_view).collect();
        let result = crate::cached_global_func!("ffi.List").call_packed(&args)?;
        Self::try_from(result)
    }

    #[inline]
    fn obj(&self) -> *mut ListObj {
        // SAFETY: `self.data` keeps the object alive for as long as the pointer is used.
        unsafe { ObjectArc::as_raw(&self.data) as *mut ListObj }
    }

    /// Returns the number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.size as usize
    }

    /// Returns `true` if the list contains no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the number of elements the buffer holds before it is reallocated.
    pub fn capacity(&self) -> usize {
        self.data.capacity as usize
    }

    /// Returns the element at `index`.
    #[inline]
    pub fn get(&self, index: usize) -> Result<T> {
        let obj = self.obj();
        unsafe {
            if index >= (*obj).size as usize {
                return index_error("get");
            }
            match T::container_try_cast_from_any_view(&*ListObj::cell(obj, index)) {
                Ok(value) => Ok(value),
                Err(()) => type_error(index),
            }
        }
    }

    /// Replaces the element at `index`.
    #[inline]
    pub fn set(&self, index: usize, value: T) -> Result<()> {
        let obj = self.obj();
        unsafe {
            if index >= (*obj).size as usize {
                return index_error("set");
            }
            let cell = ListObj::cell(obj, index);
            let old = take_cell(cell);
            write_cell(cell, value);
            drop(old);
        }
        Ok(())
    }

    /// Appends `value` to the end of the list.
    #[inline]
    pub fn push(&self, value: T) -> Result<()> {
        let obj = self.obj();
        unsafe {
            if (*obj).size < (*obj).capacity {
                write_cell(ListObj::cell(obj, (*obj).size as usize), value);
                (*obj).size += 1;
                return Ok(());
            }
        }
        // Full: the C++ runtime grows the buffer with the allocator that owns it.
        crate::cached_global_func!("ffi.ListAppend")
            .call_packed(&[AnyView::from(self), element_view(&value)])?;
        Ok(())
    }

    /// Inserts `value` at `index`, shifting the elements after it.
    pub fn insert(&self, index: usize, value: T) -> Result<()> {
        let size = self.len();
        if index > size {
            return index_error("insert");
        }
        let obj = self.obj();
        unsafe {
            if (*obj).size < (*obj).capacity {
                let cell = ListObj::cell(obj, index);
                ptr::copy(cell, cell.add(1), size - index);
                write_cell(cell, value);
                (*obj).size += 1;
                return Ok(());
            }
        }
        // Full: the C++ runtime grows the buffer with the allocator that owns it.
        crate::cached_global_func!("ffi.ListInsert").call_packed(&[
            AnyView::from(self),
            AnyView::from(&(index as i64)),
            element_view(&value),
        ])?;
        Ok(())
    }

    /// Removes and returns the element at `index`, shifting the elements after
    /// it. If that element is not a `T`, the list is left untouched.
    #[inline]
    pub fn remove(&self, index: usize) -> Result<T> {
        let obj = self.obj();
        unsafe {
            let size = (*obj).size as usize;
            if index >= size {
                return index_error("remove");
            }
            let cell = ListObj::cell(obj, index);
            // Convert first, so that a mismatch leaves the list untouched.
            let value = if T::container_check_any_strict(&*cell) {
                let mut raw = ptr::read(cell);
                T::container_move_from_any_after_check(&mut raw)
            } else {
                match T::container_try_cast_from_any_view(&*cell) {
                    Ok(value) => {
                        drop(take_cell(cell));
                        value
                    }
                    Err(()) => return type_error(index),
                }
            };
            if index + 1 < size {
                ptr::copy(cell.add(1), cell, size - index - 1);
            }
            (*obj).size -= 1;
            Ok(value)
        }
    }

    /// Removes all elements, keeping the buffer.
    #[inline]
    pub fn clear(&self) {
        unsafe { ListObj::truncate(self.obj(), 0) }
    }

    /// Returns an iterator over the elements, yielding `Result<T>`; see
    /// [`ListIterator`].
    pub fn iter(&self) -> ListIterator<'_, T> {
        ListIterator {
            list: self,
            index: 0,
        }
    }
}

impl<T: ContainerElement> Default for List<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Debug for List<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let full_name = std::any::type_name::<T>();
        let short_name = full_name.split("::").last().unwrap_or(full_name);
        write!(f, "List<{}>[{}]", short_name, self.data.size)
    }
}

impl<T: ContainerElement> FromIterator<T> for List<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let items: Vec<T> = iter.into_iter().collect();
        Self::from_items(&items).expect("ffi.List() failed to construct a list")
    }
}

// --- Iterator ---

/// Iterator over a [`List`], created by [`List::iter`]. Each step yields
/// `Ok(T)`, or the `TYPE_ERROR` [`List::get`] would give when the element at
/// that position is not a `T` (another handle to the same list may have put
/// it there), and then moves on to the next element. It re-reads the length
/// on every step, so it stays in bounds if the list is mutated meanwhile.
pub struct ListIterator<'a, T> {
    list: &'a List<T>,
    index: usize,
}

impl<T: ContainerElement> Iterator for ListIterator<'_, T> {
    type Item = Result<T>;

    #[inline]
    fn next(&mut self) -> Option<Result<T>> {
        let obj = self.list.obj();
        unsafe {
            if self.index >= (*obj).size as usize {
                return None;
            }
            let index = self.index;
            self.index += 1;
            Some(
                match T::container_try_cast_from_any_view(&*ListObj::cell(obj, index)) {
                    Ok(value) => Ok(value),
                    Err(()) => type_error(index),
                },
            )
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.list.len().saturating_sub(self.index)))
    }
}

impl<'a, T: ContainerElement> IntoIterator for &'a List<T> {
    type Item = Result<T>;
    type IntoIter = ListIterator<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

// --- Any Type System Conversions ---

unsafe impl<T: ContainerElement> AnyCompatible for List<T> {
    fn type_str() -> String {
        format!("List<{}>", T::container_type_str())
    }

    unsafe fn check_any_strict(data: &TVMFFIAny) -> bool {
        data.type_index == TypeIndex::kTVMFFIList as i32
            && seq_cells(data)
                .iter()
                .all(|cell| T::container_check_any_strict(cell))
    }

    unsafe fn copy_to_any_view(src: &Self, data: &mut TVMFFIAny) {
        data.type_index = TypeIndex::kTVMFFIList as i32;
        data.data_union.v_obj = ObjectArc::as_raw(Self::data(src)) as *mut TVMFFIObject;
        data.small_str_len = 0;
    }

    unsafe fn move_to_any(src: Self, data: &mut TVMFFIAny) {
        data.type_index = TypeIndex::kTVMFFIList as i32;
        data.data_union.v_obj = ObjectArc::into_raw(Self::into_data(src)) as *mut TVMFFIObject;
        data.small_str_len = 0;
    }

    unsafe fn copy_from_any_view_after_check(data: &TVMFFIAny) -> Self {
        let ptr = data.data_union.v_obj as *const ListObj;
        crate::object::unsafe_::inc_ref(ptr as *mut TVMFFIObject);
        Self::from_data(ObjectArc::from_raw(ptr))
    }

    unsafe fn move_from_any_after_check(data: &mut TVMFFIAny) -> Self {
        let ptr = data.data_union.v_obj as *const ListObj;
        let obj = Self::from_data(ObjectArc::from_raw(ptr));
        data.type_index = TypeIndex::kTVMFFINone as i32;
        data.data_union.v_int64 = 0;
        obj
    }

    unsafe fn try_cast_from_any_view(data: &TVMFFIAny) -> Result<Self, ()> {
        // Mirrors C++ `SeqTypeTraitsBase::TryCastFromAnyView`: `ffi.Array` is
        // the accepted other sequence type.
        if data.type_index != TypeIndex::kTVMFFIList as i32
            && data.type_index != TypeIndex::kTVMFFIArray as i32
        {
            return Err(());
        }

        // Fast path: a list whose elements all match shares the reference.
        if <Self as AnyCompatible>::check_any_strict(data) {
            return Ok(<Self as AnyCompatible>::copy_from_any_view_after_check(
                data,
            ));
        }

        // Slow path: convert element by element into a new list.
        let mut items = Vec::new();
        for cell in seq_cells(data) {
            items.push(T::container_try_cast_from_any_view(cell)?);
        }
        Self::from_items(&items).map_err(|_| ())
    }
}

impl<T: ContainerElement> TryFrom<Any> for List<T> {
    type Error = Error;

    fn try_from(value: Any) -> Result<Self> {
        let temp: TryFromTemp<Self> = TryFromTemp::try_from(value)?;
        Ok(TryFromTemp::into_value(temp))
    }
}

impl<'a, T: ContainerElement> TryFrom<AnyView<'a>> for List<T> {
    type Error = Error;

    fn try_from(value: AnyView<'a>) -> Result<Self> {
        let temp: TryFromTemp<Self> = TryFromTemp::try_from(value)?;
        Ok(TryFromTemp::into_value(temp))
    }
}
