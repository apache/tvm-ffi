use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

use crate::any::TryFromTemp;
use crate::derive::Object;
use crate::object::{Object, ObjectArc};
use crate::{Any, AnyCompatible, AnyView, ObjectCore, ObjectCoreWithExtraItems, ObjectRefCore};
use tvm_ffi_sys::TVMFFITypeIndex as TypeIndex;
use tvm_ffi_sys::{TVMFFIAny, TVMFFIAnyDataUnion, TVMFFIObject};

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

impl Drop for ArrayObj {
    fn drop(&mut self) {
        if !self.data.is_null() {
            unsafe {
                let p = self.data as *mut TVMFFIAny;
                for i in 0..self.size {
                    let any = &mut *p.add(i as usize);
                    if any.type_index >= TypeIndex::kTVMFFIStaticObjectBegin as i32 {
                        crate::object::unsafe_::dec_ref(any.data_union.v_obj);
                    }
                }
            }
        }
    }
}

#[repr(C)]
#[derive(Clone)]
pub struct Array<T: ObjectRefCore> {
    data: ObjectArc<ArrayObj>,
    _marker: PhantomData<T>,
}

impl<T: ObjectRefCore> Debug for Array<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let full_name = std::any::type_name::<T>();
        let short_name = full_name.split("::").last().unwrap_or(full_name);
        write!(f, "Array<{}>[{}]", short_name, self.len())
    }
}

impl<T: ObjectRefCore> Default for Array<T> {
    fn default() -> Self {
        Self::new(vec![])
    }
}

unsafe impl<T: ObjectRefCore> ObjectRefCore for Array<T> {
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

impl<T: ObjectRefCore> Array<T> {
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
            size: capacity as i64, // Set to capacity for correct allocation size
            capacity: capacity as i64,
            data_deleter: None,
        });

        unsafe {
            let container = &mut *(ObjectArc::as_raw(&arc) as *mut ArrayObj);
            // Calculate base pointer (memory immediately following the struct)
            let base_ptr = (container as *mut ArrayObj).add(1) as *mut TVMFFIAny;

            container.data = base_ptr as *mut _;
            container.size = size as i64;

            for (i, item) in items.into_iter().enumerate() {
                *base_ptr.add(i) = TVMFFIAny {
                    type_index: T::ContainerType::type_index(),
                    data_union: TVMFFIAnyDataUnion {
                        v_obj: ObjectArc::into_raw(T::into_data(item)) as *mut _,
                    },
                    ..TVMFFIAny::new()
                };
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

    /// Retrieves an item at the given index. Increments the object's reference count.
    pub fn get(&self, index: usize) -> Result<T, crate::Error> {
        if index >= self.len() {
            crate::bail!(crate::error::INDEX_ERROR, "Array get index out of bound");
        }
        unsafe {
            let container = self.data.deref();
            let base_ptr = container.data as *const TVMFFIAny;
            let any_ref = &*base_ptr.add(index);

            if any_ref.type_index >= TypeIndex::kTVMFFIStaticObjectBegin as i32 {
                let obj_handle = any_ref.data_union.v_obj as *mut TVMFFIObject;
                crate::object::unsafe_::inc_ref(obj_handle);
                Ok(T::from_data(ObjectArc::from_raw(
                    obj_handle as *const T::ContainerType,
                )))
            } else {
                crate::bail!(
                    crate::error::TYPE_ERROR,
                    "Expected static object, found type_index {}",
                    any_ref.type_index
                );
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

    /// Ensures this Array has unique ownership of the underlying data.
    /// If shared, a new copy of the ArrayObj is created (Copy-on-Write).
    fn ensure_unique(&mut self) {
        let ref_cnt = unsafe {
            crate::object::unsafe_::strong_count(ObjectArc::as_raw(&self.data) as *mut _)
        };
        // Only clone if there are other references to this ObjectArc
        if ref_cnt > 1 {
            // Create a new Array with the same items and same capacity
            let items: Vec<T> = self.iter().collect();
            let capacity = self.data.capacity as usize;
            *self = Self::new_with_capacity(items, capacity);
        }
    }

    /// Appends an item. Triggers reallocation if the capacity is exceeded.
    pub fn push(&mut self, item: T) {
        self.ensure_unique();

        let current_size = self.data.size as usize;
        let current_cap = self.data.capacity as usize;

        if current_size < current_cap {
            unsafe {
                // Scenario 1: Just write to the existing extra items
                let container = self.data.deref_mut();
                let base_ptr = (container as *mut ArrayObj).add(1) as *mut TVMFFIAny;
                *base_ptr.add(current_size) = TVMFFIAny {
                    type_index: T::ContainerType::type_index(),
                    data_union: TVMFFIAnyDataUnion {
                        v_obj: ObjectArc::into_raw(T::into_data(item)) as *mut _,
                    },
                    ..TVMFFIAny::new()
                };

                container.size += 1;
            }
        } else {
            // Scenario 2: Reallocate (Grow)
            let new_cap = if current_cap == 0 { 4 } else { current_cap * 2 };
            let mut new_items: Vec<T> = self.iter().collect();
            new_items.push(item);
            *self = Self::new_with_capacity(new_items, new_cap as usize);
        }
    }

    /// Inserts an item at a specific index, shifting existing elements.
    pub fn insert(&mut self, index: usize, item: T) -> Result<(), crate::Error> {
        self.ensure_unique();

        let current_size = self.data.size as usize;
        let current_cap = self.data.capacity as usize;
        if index > current_size {
            crate::bail!(crate::error::INDEX_ERROR, "Array insert index out of bound");
        }

        if current_size < current_cap {
            unsafe {
                let container = self.data.deref_mut();
                let base_ptr = (container as *mut ArrayObj).add(1) as *mut TVMFFIAny;

                // Shift elements to the right to make a hole
                if index < current_size {
                    core::ptr::copy(
                        base_ptr.add(index),
                        base_ptr.add(index + 1),
                        current_size - index,
                    );
                }

                // Move ownership of the new item into the hole
                *base_ptr.add(index) = TVMFFIAny {
                    type_index: T::ContainerType::type_index(),
                    data_union: TVMFFIAnyDataUnion {
                        v_obj: ObjectArc::into_raw(T::into_data(item)) as *mut _,
                    },
                    ..TVMFFIAny::new()
                };

                container.size += 1;
            }
        } else {
            // Reallocate to grow
            let mut items: Vec<T> = self.iter().collect();
            items.insert(index, item);
            let new_cap = if current_cap == 0 { 4 } else { current_cap * 2 };
            *self = Self::new_with_capacity(items, new_cap);
        }

        Ok(())
    }

    /// Pops the last item. Returns None if empty.
    pub fn pop(&mut self) -> Option<T> {
        self.ensure_unique();

        if self.is_empty() {
            return None;
        }
        unsafe {
            let last_index = (self.data.size - 1) as usize;

            // 1. Get the item (this increments ref count)
            let item = self.get(last_index).ok();

            // 2. Clear the slot in the array (this decrements ref count)
            // Effectively, we "moved" the reference from the array to the return value
            let container = self.data.deref_mut();
            let extra_items = ArrayObj::extra_items_mut(container);
            let any = &mut extra_items[last_index];
            if any.type_index >= TypeIndex::kTVMFFIStaticObjectBegin as i32 {
                crate::object::unsafe_::dec_ref(any.data_union.v_obj);
            }
            *any = TVMFFIAny::new();
            container.size -= 1;

            item
        }
    }

    /// Removes an item at index, shifting subsequent elements left.
    pub fn remove(&mut self, index: usize) -> Result<T, crate::Error> {
        self.ensure_unique();

        let current_size = self.data.size as usize;
        if index >= current_size {
            crate::bail!(crate::error::INDEX_ERROR, "Array remove index out of bound");
        }

        unsafe {
            // 1. Get the item to return (increments ref count)
            let item = self.get(index).expect("Failed to get item for removal");

            let container = self.data.deref_mut();
            let base_ptr = (container as *mut ArrayObj).add(1) as *mut TVMFFIAny;

            // 2. Decrement ref count of the physical copy remaining in the array
            // because the array no longer owns this specific reference.
            let any_to_remove = &*base_ptr.add(index);
            if any_to_remove.type_index >= TypeIndex::kTVMFFIStaticObjectBegin as i32 {
                crate::object::unsafe_::dec_ref(any_to_remove.data_union.v_obj);
            }

            // 3. Shift elements to the left to close the gap
            if index < current_size - 1 {
                core::ptr::copy(
                    base_ptr.add(index + 1),
                    base_ptr.add(index),
                    current_size - index - 1,
                );
            }

            // 4. Zero out the now-unused last slot to prevent accidental double-frees
            *base_ptr.add(current_size - 1) = TVMFFIAny::new();

            container.size -= 1;

            Ok(item)
        }
    }

    /// Clears the array and decrements ref-counts of all stored objects.
    pub fn clear(&mut self) {
        self.ensure_unique();

        unsafe {
            let size = self.data.size as usize;
            let container = self.data.deref_mut();
            let extra_items = ArrayObj::extra_items_mut(container);
            for i in 0..size {
                // This drops the TVMFFIAny, but we need to dec_ref if it's an object
                let any = &mut extra_items[i];
                if any.type_index >= TypeIndex::kTVMFFIStaticObjectBegin as i32 {
                    crate::object::unsafe_::dec_ref(any.data_union.v_obj);
                }
                *any = TVMFFIAny::new(); // Reset to None
            }
            container.size = 0;
        }
    }
}

// --- Iterator Implementations ---

impl<T: ObjectRefCore> Extend<T> for Array<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for item in iter {
            self.push(item);
        }
    }
}

pub struct ArrayIterator<'a, T: ObjectRefCore> {
    array: &'a Array<T>,
    index: usize,
    len: usize,
}

impl<'a, T: ObjectRefCore> Iterator for ArrayIterator<'a, T> {
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

impl<'a, T: ObjectRefCore> IntoIterator for &'a Array<T> {
    type Item = T;
    type IntoIter = ArrayIterator<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T: ObjectRefCore> FromIterator<T> for Array<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let items: Vec<T> = iter.into_iter().collect();
        Self::new(items)
    }
}

// --- Any Type System Conversions ---

unsafe impl<T> AnyCompatible for Array<T>
where
    T: ObjectRefCore + AnyCompatible + 'static,
{
    fn type_str() -> String {
        "Array".into()
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
    }

    unsafe fn move_to_any(src: Self, data: &mut TVMFFIAny) {
        data.type_index = TypeIndex::kTVMFFIArray as i32;
        let ptr = ObjectArc::into_raw(Self::into_data(src));
        data.data_union.v_obj = ptr as *mut TVMFFIObject;
    }

    unsafe fn copy_from_any_view_after_check(data: &TVMFFIAny) -> Self {
        let ptr = data.data_union.v_obj as *const ArrayObj;
        crate::object::unsafe_::inc_ref(ptr as *mut TVMFFIObject);
        Self::from_data(ObjectArc::from_raw(ptr))
    }

    unsafe fn move_from_any_after_check(data: &mut TVMFFIAny) -> Self {
        let ptr = data.data_union.v_obj as *const ArrayObj;
        Self::from_data(ObjectArc::from_raw(ptr))
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
        let mut new_items = Vec::with_capacity(container.size as usize);

        for i in 0..container.size {
            let elem_any = &*base_ptr.add(i as usize);
            match T::try_cast_from_any_view(elem_any) {
                Ok(converted) => new_items.push(converted),
                Err(_) => return Err(()),
            }
        }

        Ok(Array::new(new_items))
    }
}

impl<T> TryFrom<Any> for Array<T>
where
    T: ObjectRefCore + AnyCompatible + 'static,
{
    type Error = crate::error::Error;

    fn try_from(value: Any) -> Result<Self, Self::Error> {
        let temp: TryFromTemp<Self> = TryFromTemp::try_from(value)?;
        Ok(TryFromTemp::into_value(temp))
    }
}

impl<'a, T> TryFrom<AnyView<'a>> for Array<T>
where
    T: ObjectRefCore + AnyCompatible + 'static,
{
    type Error = crate::error::Error;

    fn try_from(value: AnyView<'a>) -> Result<Self, Self::Error> {
        let temp: TryFromTemp<Self> = TryFromTemp::try_from(value)?;
        Ok(TryFromTemp::into_value(temp))
    }
}
