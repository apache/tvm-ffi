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
use std::ops::Deref;
use std::ptr::NonNull;
use std::sync::atomic::AtomicU64;

use crate::derive::ObjectRef;
use crate::type_traits::AnyCompatible;
pub use tvm_ffi_sys::TVMFFITypeIndex as TypeIndex;
/// Object related ABI handling
use tvm_ffi_sys::{TVMFFIAny, TVMFFIGetTypeInfo, TVMFFIObject, COMBINED_REF_COUNT_BOTH_ONE};

/// Held while `#[derive(Object)]` with `#[type_register]` registers a type, so
/// that registrations from Rust do not race each other: the runtime does not
/// lock its type table and leaves that to callers.
#[doc(hidden)]
pub static TYPE_REGISTRATION: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Object type is by default the TVMFFIObject
#[repr(C)]
pub struct Object {
    /// example implementation of the object
    header: TVMFFIObject,
}

/// Arc-like wrapper for Object that allows shared ownership
///
/// \tparam T The type of the object to be wrapped
#[repr(C)]
pub struct ObjectArc<T: ObjectCore> {
    ptr: std::ptr::NonNull<T>,
    _phantom: std::marker::PhantomData<T>,
}

// Not `Send`/`Sync` by default: an object's thread safety is that of its (often
// C++) implementation, which the handle cannot see. An object type opts in with
// `ImmutableObject`; tvm-ffi's own immutable handles opt in explicitly.

/// An object type whose shared instances are never mutated, so handles to it may
/// cross threads.
///
/// A handle `ObjectArc<T>`, and every reference built on it, is `Send + Sync`
/// when `T: ImmutableObject` and `T`'s fields are themselves `Send + Sync`, so
/// a field such as `Any`, which may hold a mutable object, keeps the handle
/// single-threaded. The stub generator implements it for each class whose C++
/// declaration is not `_type_mutable`. An erased `ObjectRef` may point at any
/// object, so it never crosses threads.
///
/// # Safety
///
/// No code may mutate an instance while it is shared: changes go through
/// copy-on-write, in place only on a uniquely held instance, as for C++ classes
/// that are not `_type_mutable`.
pub unsafe trait ImmutableObject: ObjectCore {}

// SAFETY: `T` is never mutated while shared, its fields are thread-safe, and the
// reference count is atomic.
unsafe impl<T: ImmutableObject + Send + Sync> Send for ObjectArc<T> {}
// SAFETY: as for `Send`.
unsafe impl<T: ImmutableObject + Send + Sync> Sync for ObjectArc<T> {}

/// The native representation of an object-system type.
///
/// # Safety
/// `Self` must have a compatible C layout with a valid `TVMFFIObject` prefix
/// at offset zero. Its registered type, inheritance and payload must describe
/// that layout correctly, including for foreign declarations.
///
/// A live managed object and its header must remain at their original allocation
/// address until destruction. Safe APIs must not move, replace or expose mutable
/// access to that header while managed; payload mutation must preserve its type
/// invariants. Preallocation builders and values moved out during destruction
/// are not live managed objects and cannot be borrowed as `AnyView`.
pub unsafe trait ObjectCore: Sized + 'static {
    /// the type key of the object
    const TYPE_KEY: &'static str;
    /// Depth of this type in the object inheritance tree.
    ///
    /// The root [`Object`] has depth zero, and every registered subtype has
    /// depth one greater than its parent. This value must be non-negative and
    /// agree with the runtime type table entry for `Self`.
    const TYPE_DEPTH: i32;
    /// Whether every instance of this type has exactly `Self::type_index()`.
    ///
    /// A final type has no separately registered object-system subtype.
    #[doc(hidden)]
    const TYPE_FINAL: bool = false;
    // return the type index of the object
    fn type_index() -> i32;
    /// Return the object header
    /// This function is implemented as a static function so
    ///
    /// # Arguments
    /// * `this` - The object to get the header
    ///
    /// # Returns
    /// * `&mut TVMFFIObject` - The object header
    /// \return The object header
    ///
    /// # Safety
    /// The caller must have exclusive access to the header and preserve the
    /// managed allocation's address, reference counts, deleter and dynamic type.
    unsafe fn object_header_mut(this: &mut Self) -> &mut TVMFFIObject;
}

/// Traits for objects with extra items that follows the object
///
/// This extra trait can be helpful to implement array types and string types
pub unsafe trait ObjectCoreWithExtraItems: ObjectCore {
    /// type of extra items storage that follows the object
    type ExtraItem;
    /// Return the number of extra items
    fn extra_items_count(this: &Self) -> usize;
    /// Return the extra items data pointer
    unsafe fn extra_items(this: &Self) -> &[Self::ExtraItem] {
        let extra_items_ptr = (this as *const Self as *const u8).add(std::mem::size_of::<Self>());
        std::slice::from_raw_parts(
            extra_items_ptr as *const Self::ExtraItem,
            Self::extra_items_count(this),
        )
    }
    /// Return the extra items data pointer
    unsafe fn extra_items_mut(this: &mut Self) -> &mut [Self::ExtraItem] {
        let extra_items_ptr = (this as *mut Self as *mut u8).add(std::mem::size_of::<Self>());
        std::slice::from_raw_parts_mut(
            extra_items_ptr as *mut Self::ExtraItem,
            Self::extra_items_count(this),
        )
    }
}

/// A managed object reference: the category of types such as [`ObjectRef`](struct@ObjectRef),
/// `Array<T>` and every `#[derive(ObjectRef)]` type, with the operations every
/// reference supports: identity ([`same_as`](Self::same_as)) and strict casts
/// between reference types, which check the object's runtime type and never
/// convert. [`downcast_ref`](Self::downcast_ref) borrows in place, and
/// [`downcast`](Self::downcast) takes the reference, as std's
/// `dyn Any::downcast_ref` and `Box<dyn Any>::downcast` do. Upcasts are
/// `From`/`Into`, which `#[derive(ObjectRef)]` and
/// [`unsafe_impl_object_upcast!`](crate::unsafe_impl_object_upcast) generate.
///
/// The derive implements it. `data`, `into_data` and `from_data` are the
/// handle plumbing the FFI uses, called as associated functions; the trait is
/// `unsafe` to implement because they move the reference's internals.
///
/// # Safety
///
/// `data`, `into_data`, and `from_data` must preserve the same object
/// allocation and form an ownership-preserving round trip. That allocation must
/// start with a valid `TVMFFIObject` header whose registered object-range
/// runtime type index correctly describes its layout and inheritance.
///
/// When `Self` also implements [`AnyCompatible`], `copy_to_any_view` must
/// produce a non-owning view, while `move_to_any` must transfer ownership of
/// the same object pointer and dynamic type index. `move_from_any_after_check`
/// must be able to reclaim that owned representation exactly once, and a true
/// `check_any_strict` result must guarantee that both after-check constructors
/// are valid for it.
pub unsafe trait ObjectRefCore: Sized + Clone + AnyCompatible {
    type ContainerType: ObjectCore;
    fn data(this: &Self) -> &ObjectArc<Self::ContainerType>;
    fn into_data(this: Self) -> ObjectArc<Self::ContainerType>;

    /// Construct a reference view from an owning container handle.
    ///
    /// # Safety
    ///
    /// In addition to containing a valid `ContainerType` allocation, `data`
    /// must satisfy every semantic invariant imposed by `Self`. This matters
    /// for zero-state views that share a container type but accept only a
    /// subset of its values, such as a typed expression view.
    unsafe fn from_data(data: ObjectArc<Self::ContainerType>) -> Self;

    /// Return whether two object references point to the same allocation.
    #[inline]
    fn same_as<Other: ObjectRefCore>(&self, other: &Other) -> bool {
        ObjectArc::as_ptr(Self::data(self)).cast::<()>()
            == ObjectArc::as_ptr(Other::data(other)).cast::<()>()
    }

    /// Borrow the underlying object as `T` when it is one.
    ///
    /// `T` is an object type such as `FooObj`, or a managed reference such as
    /// `Foo`: `&Expr` borrows as `&Add` when the expression is an `Add`.
    /// Unrelated types return `None`. Unlike [`downcast`](Self::downcast), this
    /// method neither consumes the reference nor changes the object's reference
    /// count. The returned borrow cannot outlive `self`.
    /// See [`crate::Any::downcast_ref`] for an example.
    #[inline(always)]
    fn downcast_ref<T: crate::type_traits::DowncastRefFromObject>(&self) -> Option<&T> {
        // SAFETY: `ObjectArc` is exactly one non-null object pointer, and
        // ObjectRefCore guarantees a live object allocation.
        unsafe {
            let place = &*(Self::data(self) as *const ObjectArc<Self::ContainerType>
                as *const *mut TVMFFIObject);
            T::downcast_ref_from_object(place)
        }
    }

    /// Take `self` as the reference type `B` when its object is one, without
    /// copying the object or changing its reference count.
    ///
    /// `B` is a subtype, or a parameterized container such as `Array<i64>`,
    /// whose elements are checked too. On a mismatch the reference is returned
    /// unchanged, so another type can be tried. For `?`, use the `TryFrom` that
    /// each upcast provides, which returns a `TypeError` instead.
    ///
    /// ```
    /// use tvm_ffi::{object::ObjectRef, Array, ObjectRefCore};
    /// let object: ObjectRef = Array::from_vec(vec![1_i64]).into();
    /// let object = object.downcast::<Array<f64>>().unwrap_err();
    /// let array: Array<i64> = object.try_into()?;
    /// assert_eq!(array.len(), 1);
    /// # Ok::<(), tvm_ffi::Error>(())
    /// ```
    #[inline(always)]
    fn downcast<B>(self) -> std::result::Result<B, Self>
    where
        B: ObjectRefCore,
    {
        let mut any_data = TVMFFIAny::new();
        unsafe {
            // Keep ownership in `self` while the target check runs, so a
            // mismatch can hand it back and a panic unwinds normally.
            Self::copy_to_any_view(&self, &mut any_data);
            debug_assert!(any_data.type_index >= TypeIndex::kTVMFFIStaticObjectBegin as i32);
            // SAFETY: ObjectRefCore's contract requires its AnyCompatible
            // representation to contain a valid object-range type index.
            std::hint::assert_unchecked(
                any_data.type_index >= TypeIndex::kTVMFFIStaticObjectBegin as i32,
            );
            if B::check_any_strict(&any_data) {
                // Transfer ownership only after the borrowed representation has
                // passed the target's complete hierarchy/container check.
                Self::move_to_any(self, &mut any_data);
                Ok(B::move_from_any_after_check(&mut any_data))
            } else {
                Err(self)
            }
        }
    }
}

/// A value that converts into the object reference `T`, owned or borrowed.
///
/// An owned reference converts into an owned `T`, and a borrowed one borrows
/// as `&T` without changing a reference count, as [`IntoIterator`] consumes a
/// `Vec` but borrows from a `&Vec`. [`unsafe_impl_object_upcast!`](crate::unsafe_impl_object_upcast)
/// implements it for each derived reference and its reference into the base,
/// and `#[derive(ObjectRef)]` for the type itself. A function that only reads
/// the value borrows the output with `.borrow()`; one that keeps it takes
/// ownership with [`into_owned_object_ref`](Self::into_owned_object_ref), which clones only a
/// borrowed reference.
///
/// # Examples
///
/// ```
/// use std::borrow::Borrow;
/// use tvm_ffi::object::{IntoObjectRef, ObjectRef, ObjectRefCore};
/// use tvm_ffi::{def_object_ref, rs_object, Object};
///
/// #[rs_object("my_ext.MyPair", final)]
/// pub struct MyPairObj {
///     base: Object,
///     pub first: i64,
///     pub second: i64,
/// }
/// def_object_ref!(pub MyPair, ObjectRef, MyPairObj);
///
/// fn same(a: impl IntoObjectRef<ObjectRef>, b: &ObjectRef) -> bool {
///     a.into_object_ref().borrow().same_as(b)
/// }
///
/// let pair = MyPair::from(MyPairObj::new(Object::new(), 1, 2));
/// let base: ObjectRef = pair.clone().into();
/// assert!(same(&pair, &base)); // borrowed as `&ObjectRef`, no new reference
/// assert!(same(pair.clone(), &base)); // moved into an `ObjectRef`
///
/// // A function that keeps the value takes ownership.
/// fn keep(value: impl IntoObjectRef<ObjectRef>) -> ObjectRef {
///     value.into_owned_object_ref()
/// }
/// assert!(keep(&pair).same_as(&base));
/// ```
pub trait IntoObjectRef<T> {
    /// `T` for an owned value, or `&T` for a borrowed one.
    type Output: std::borrow::Borrow<T> + Into<T>;

    /// Convert into the reference `T`.
    ///
    /// # Returns
    ///
    /// The owned or borrowed reference.
    fn into_object_ref(self) -> Self::Output;

    /// Convert into an owned `T`, as [`Cow::into_owned`](std::borrow::Cow::into_owned)
    /// does: an owned value moves, and only a borrowed reference is cloned.
    ///
    /// # Returns
    ///
    /// The owned reference.
    #[inline(always)]
    fn into_owned_object_ref(self) -> T
    where
        Self: Sized,
    {
        self.into_object_ref().into()
    }
}

/// An owning, hashable identity key for an FFI object.
///
/// The retained strong reference prevents the allocation address from being
/// reused while the key is alive. This makes it suitable for identity-based
/// maps without exposing raw object pointers to downstream code.
#[derive(Clone)]
pub struct ObjectIdentity {
    data: ObjectArc<Object>,
}

impl ObjectIdentity {
    /// Retain the allocation referenced by `value` as an identity key.
    pub fn of<T: ObjectRefCore>(value: &T) -> Self {
        unsafe {
            let ptr = ObjectArc::as_ptr(T::data(value)) as *mut TVMFFIObject;
            unsafe_::inc_ref(ptr);
            Self {
                data: ObjectArc::from_raw(ptr.cast::<Object>()),
            }
        }
    }

    #[inline]
    fn as_ptr(&self) -> *const TVMFFIObject {
        ObjectArc::as_ptr(&self.data).cast::<TVMFFIObject>()
    }
}

impl PartialEq for ObjectIdentity {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_ptr() == other.as_ptr()
    }
}

impl Eq for ObjectIdentity {}

impl std::hash::Hash for ObjectIdentity {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.as_ptr().hash(state);
    }
}

impl std::fmt::Debug for ObjectIdentity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("ObjectIdentity")
            .field(&self.as_ptr())
            .finish()
    }
}

/// Check whether a runtime type index refers to `Target` or one of its
/// subtypes.
///
/// The subtype relation lives in the process-wide type table maintained by the
/// tvm-ffi library: every registered type records its depth in the single
/// inheritance tree together with the chain of its ancestors. The check is
/// O(1) — if `target` really is an ancestor, it must appear in the candidate's
/// ancestor array exactly at `target`'s depth.
///
/// This is a hidden support function for derive-generated object checks. Object
/// indices in the registered range must refer to entries in the runtime type
/// table.
#[doc(hidden)]
#[inline(always)]
pub fn is_instance_of<Target: ObjectCore>(object_type_index: i32) -> bool {
    let target_type_index = Target::type_index();
    if object_type_index == target_type_index {
        return true;
    }
    // A final type cannot have a separately registered subtype. Keep common
    // borrowed checks, such as `IntImmObj`, to one integer comparison.
    if Target::TYPE_FINAL {
        return false;
    }
    let object_begin = TypeIndex::kTVMFFIStaticObjectBegin as i32;
    // Only object types participate in the type hierarchy.
    if object_type_index < object_begin || target_type_index < object_begin {
        return false;
    }
    // Only the root Object type has depth zero.
    if const { Target::TYPE_DEPTH == 0 } {
        return true;
    }
    // Parent indices are always smaller than their descendants.
    if object_type_index < target_type_index {
        return false;
    }
    unsafe {
        let object_info = TVMFFIGetTypeInfo(object_type_index);
        if object_info.is_null() {
            return false;
        }
        let target_depth = Target::TYPE_DEPTH;
        if (*object_info).type_depth <= target_depth {
            return false;
        }
        let ancestor = *(*object_info).type_acenstors.add(target_depth as usize);
        !ancestor.is_null() && (*ancestor).type_index == target_type_index
    }
}

/// The strict downcast from the root [`ObjectRef`] to `B`, for the containers'
/// `TryFrom<ObjectRef>`, which are hand-written as their upcasts are.
#[inline(always)]
pub(crate) fn root_try_into<B>(value: ObjectRef) -> crate::error::Result<B>
where
    B: ObjectRefCore,
{
    value
        .downcast::<B>()
        .map_err(crate::error::Error::downcast_failed::<B>)
}

/// [`root_try_into`] from a borrowed reference, cloning only on success.
#[inline(always)]
pub(crate) fn root_try_into_ref<B>(value: &ObjectRef) -> crate::error::Result<B>
where
    B: ObjectRefCore + crate::type_traits::DowncastRefFromObject,
{
    match value.downcast_ref::<B>() {
        Some(found) => Ok(found.clone()),
        None => Err(crate::error::Error::downcast_failed::<B>(value.clone())),
    }
}

/// Upcast a managed reference to the root [`ObjectRef`], keeping its allocation.
#[inline(always)]
pub(crate) fn into_root_object_ref<S: ObjectRefCore>(value: S) -> ObjectRef {
    let data = S::into_data(value);
    // SAFETY: every object starts with the root `Object` header, and an
    // `ObjectRef` imposes no further invariant.
    unsafe {
        ObjectRef::from_data(ObjectArc::from_raw(
            ObjectArc::into_raw(data).cast::<Object>(),
        ))
    }
}

/// Format a managed reference as its runtime type key and object address, as
/// the `Debug` implementation of `#[derive(ObjectRef)]` does.
///
/// Call it from `Debug` for a reference type implemented by hand, such as a
/// typed view over another reference, so that `downcast(..).unwrap()` and
/// `assert_eq!` can print it.
///
/// ```
/// use tvm_ffi::object::{fmt_object_ref, ObjectRef};
/// use tvm_ffi::Array;
/// struct Shown(ObjectRef);
/// impl std::fmt::Debug for Shown {
///     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
///         fmt_object_ref(&self.0, f)
///     }
/// }
/// let shown = Shown(Array::from_vec(vec![1_i64]).into());
/// assert!(format!("{shown:?}").starts_with("ffi.Array@"));
/// ```
pub fn fmt_object_ref<T: ObjectRefCore>(
    this: &T,
    f: &mut std::fmt::Formatter<'_>,
) -> std::fmt::Result {
    let object = ObjectArc::as_ptr(T::data(this)).cast::<TVMFFIObject>();
    // SAFETY: `this` keeps the object, and so its header, alive.
    let type_index = unsafe { (*object).type_index };
    write!(f, "{}@{:p}", type_key_of_index(type_index), object)
}

/// The registered type key of `type_index`, or `<type_index N>` when it has none.
pub(crate) fn type_key_of_index(type_index: i32) -> String {
    unsafe {
        let info = TVMFFIGetTypeInfo(type_index);
        if info.is_null() {
            format!("<type_index {type_index}>")
        } else {
            (*info).type_key.as_str().to_string()
        }
    }
}

/// A generic managed reference to a TVM-FFI object.
///
/// Holds an `ObjectArc<Object>` that manages the underlying object's lifetime
/// through reference counting.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct ObjectRef {
    data: ObjectArc<Object>,
}

/// Unsafe operations on object
#[doc(hidden)]
pub mod unsafe_ {
    use tvm_ffi_sys::{
        COMBINED_REF_COUNT_BOTH_ONE, COMBINED_REF_COUNT_MASK_U32, COMBINED_REF_COUNT_STRONG_ONE,
        COMBINED_REF_COUNT_WEAK_ONE,
    };

    use std::ffi::c_void;
    use std::sync::atomic::{fence, Ordering};
    use tvm_ffi_sys::TVMFFIObject;
    use tvm_ffi_sys::TVMFFIObjectDeleterFlagBitMask::{
        kTVMFFIObjectDeleterFlagBitMaskBoth, kTVMFFIObjectDeleterFlagBitMaskStrong,
        kTVMFFIObjectDeleterFlagBitMaskWeak,
    };

    /// Increase the strong reference count of the object
    ///
    /// This function is same as TVMFFIObjectIncRef but implemented natively in Rust
    ///
    /// # Arguments
    /// * `obj` - The object to increase the reference count
    #[inline]
    pub unsafe fn inc_ref(handle: *mut TVMFFIObject) {
        let obj = &*handle;
        obj.combined_ref_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Decrease the strong reference count of the object
    ///
    /// This function is same as TVMFFIObjectDecRef but implemented natively in Rust
    ///
    /// # Arguments
    /// * `obj` - The object to decrease the reference count
    #[inline]
    pub(crate) unsafe fn dec_ref(handle: *mut TVMFFIObject) {
        let old_combined_count = (*handle)
            .combined_ref_count
            .fetch_sub(COMBINED_REF_COUNT_STRONG_ONE, Ordering::Relaxed);
        if old_combined_count == COMBINED_REF_COUNT_BOTH_ONE {
            if let Some(deleter) = (*handle).deleter {
                fence(Ordering::Acquire);
                deleter(
                    handle.cast::<c_void>(),
                    kTVMFFIObjectDeleterFlagBitMaskBoth as i32,
                );
            }
        } else if (old_combined_count & COMBINED_REF_COUNT_MASK_U32)
            == COMBINED_REF_COUNT_STRONG_ONE
        {
            // slow path, there is still a weak reference left
            // need to run two phase decrement
            fence(Ordering::Acquire);
            if let Some(deleter) = (*handle).deleter {
                deleter(
                    handle.cast::<c_void>(),
                    kTVMFFIObjectDeleterFlagBitMaskStrong as i32,
                );
            }
            let old_weak_count = (*handle)
                .combined_ref_count
                .fetch_sub(COMBINED_REF_COUNT_WEAK_ONE, Ordering::Release);
            if old_weak_count == COMBINED_REF_COUNT_WEAK_ONE {
                fence(Ordering::Acquire);
                if let Some(deleter) = (*handle).deleter {
                    deleter(
                        handle.cast::<c_void>(),
                        kTVMFFIObjectDeleterFlagBitMaskWeak as i32,
                    );
                }
            }
        }
    }

    #[inline]
    pub(crate) unsafe fn strong_count(handle: *mut TVMFFIObject) -> usize {
        let obj = &*handle;
        (obj.combined_ref_count.load(Ordering::Relaxed) & COMBINED_REF_COUNT_MASK_U32) as usize
    }

    #[inline]
    pub(crate) unsafe fn weak_count(handle: *mut TVMFFIObject) -> usize {
        let obj = &*handle;
        (obj.combined_ref_count.load(Ordering::Relaxed) >> 32) as usize
    }

    /// Generic object deleter for objects allocated through Rust's global allocator.
    pub(crate) unsafe extern "C" fn object_deleter_for_new<T>(ptr: *mut c_void, flags: i32) {
        let obj = ptr as *mut T;
        if flags & kTVMFFIObjectDeleterFlagBitMaskStrong as i32 != 0 {
            std::ptr::drop_in_place(obj);
        }
        if flags & kTVMFFIObjectDeleterFlagBitMaskWeak as i32 != 0 {
            std::alloc::dealloc(ptr as *mut u8, std::alloc::Layout::new::<T>());
        }
    }

    /// Layout of a `T` followed by `count` extra items, kept at least one word past the
    /// header: once a weak reference outlives the data, the deleter stores `count` there.
    pub(crate) fn extra_items_layout<T, U>(count: usize) -> std::alloc::Layout {
        let size = std::mem::size_of::<T>() + count * std::mem::size_of::<U>();
        let min_size = std::mem::size_of::<TVMFFIObject>() + std::mem::size_of::<u64>();
        std::alloc::Layout::from_size_align(size.max(min_size), std::mem::align_of::<T>()).unwrap()
    }

    pub(crate) unsafe extern "C" fn object_deleter_for_new_with_extra_items<T, U>(
        ptr: *mut c_void,
        flags: i32,
    ) where
        T: super::ObjectCoreWithExtraItems<ExtraItem = U>,
    {
        let obj = ptr as *mut T;
        if flags == kTVMFFIObjectDeleterFlagBitMaskBoth as i32 {
            let extra_items_count = T::extra_items_count(&(*obj));
            std::ptr::drop_in_place(obj);
            std::alloc::dealloc(
                ptr as *mut u8,
                extra_items_layout::<T, U>(extra_items_count),
            );
        } else {
            // Weak references keep using the header, so the weak phase finds the item
            // count in the first word after it.
            let count_slot = (ptr as *mut u8).add(std::mem::size_of::<TVMFFIObject>()) as *mut u64;
            if flags & kTVMFFIObjectDeleterFlagBitMaskStrong as i32 != 0 {
                let extra_items_count = T::extra_items_count(&(*obj));
                std::ptr::drop_in_place(obj);
                std::ptr::write(count_slot, extra_items_count as u64);
            }
            if flags & kTVMFFIObjectDeleterFlagBitMaskWeak as i32 != 0 {
                let extra_items_count = std::ptr::read(count_slot) as usize;
                std::alloc::dealloc(
                    ptr as *mut u8,
                    extra_items_layout::<T, U>(extra_items_count),
                );
            }
        }
    }
}

//---------------------
// Object
//---------------------

impl Object {
    pub fn new() -> Self {
        Self {
            header: TVMFFIObject::new(),
        }
    }
}

unsafe impl ObjectCore for Object {
    const TYPE_KEY: &'static str = "ffi.Object";
    const TYPE_DEPTH: i32 = 0;
    #[inline]
    fn type_index() -> i32 {
        TypeIndex::kTVMFFIStaticObjectBegin as i32
    }
    #[inline]
    unsafe fn object_header_mut(this: &mut Self) -> &mut TVMFFIObject {
        &mut this.header
    }
}

//---------------------
// ObjectArc
//---------------------

// Allocate and initialize the normal FFI header while preserving the concrete
// payload's alignment and drop/layout deleter. The payload starts with an Object.
unsafe fn allocate_object<T>(data: T, type_index: i32) -> NonNull<T> {
    let layout = std::alloc::Layout::new::<T>();
    let allocation = std::alloc::alloc(layout);
    if allocation.is_null() {
        std::alloc::handle_alloc_error(layout);
    }
    let ptr = allocation.cast::<T>();
    std::ptr::write(ptr, data);
    std::ptr::write(
        ptr.cast::<TVMFFIObject>(),
        TVMFFIObject {
            combined_ref_count: AtomicU64::new(COMBINED_REF_COUNT_BOTH_ONE),
            type_index,
            __padding: 0,
            deleter: Some(unsafe_::object_deleter_for_new::<T>),
        },
    );
    NonNull::new_unchecked(ptr)
}

impl ObjectArc<Object> {
    /// Allocate a synchronous runtime payload that may contain borrowed Rust state.
    ///
    /// # Safety
    /// `T` begins with a valid Object header and is ABI-compatible with the
    /// registered `type_index`. The caller must keep this owner scoped to the
    /// payload's borrows and ensure every temporary strong and weak reference
    /// is released before those borrows end. Neither native handles nor this
    /// erased owner may escape that synchronous scope. Destruction uses T's
    /// exact layout and drop function, not the erased Object layout.
    /// The erased owner and handles must remain on this thread unless the
    /// concrete payload permits sharing or transfer; erasure grants neither.
    pub(crate) unsafe fn new_opaque_unchecked<T>(data: T, type_index: i32) -> Self {
        Self {
            ptr: allocate_object(data, type_index).cast(),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: ObjectCore> ObjectArc<T> {
    /// Allocate an initialized object at its stable managed address.
    pub fn new(data: T) -> Self {
        // Resolve metadata before moving the payload into its allocation.
        let type_index = T::type_index();
        unsafe {
            Self {
                ptr: allocate_object(data, type_index),
                _phantom: std::marker::PhantomData,
            }
        }
    }

    /// Allocate a stable object followed by its type's extra item storage.
    pub fn new_with_extra_items<U>(data: T) -> Self
    where
        T: ObjectCoreWithExtraItems<ExtraItem = U>,
    {
        unsafe {
            // ensure strict alignment requirements
            // so we can have { T, U*extra_items } layout
            assert_eq!(std::mem::align_of::<T>() % std::mem::align_of::<U>(), 0);
            assert_eq!(std::mem::size_of::<T>() % std::mem::align_of::<U>(), 0);
            let extra_items_count = T::extra_items_count(&data);
            let layout = unsafe_::extra_items_layout::<T, U>(extra_items_count);
            let raw_data_ptr = std::alloc::alloc(layout);
            if raw_data_ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            let ptr = raw_data_ptr as *mut T;
            std::ptr::write(ptr, data);
            // now override the header directly
            std::ptr::write(
                ptr as *mut TVMFFIObject,
                TVMFFIObject {
                    combined_ref_count: AtomicU64::new(COMBINED_REF_COUNT_BOTH_ONE),
                    type_index: T::type_index(),
                    __padding: 0,
                    deleter: Some(unsafe_::object_deleter_for_new_with_extra_items::<T, U>),
                },
            );
            // move into the object arc ptr
            Self {
                ptr: std::ptr::NonNull::new_unchecked(ptr as *mut T),
                _phantom: std::marker::PhantomData,
            }
        }
    }

    /// Whether `this` holds a null pointer, as all-zero bytes do: no
    /// `ObjectArc`, but the field of an object that a creator left undefined.
    ///
    /// # Safety
    ///
    /// `this` points to an `ObjectArc<T>` or to all-zero bytes.
    #[doc(hidden)]
    #[inline]
    pub unsafe fn is_null(this: *const Self) -> bool {
        (*std::ptr::addr_of!((*this).ptr).cast::<*const T>()).is_null()
    }

    /// Move a previously allocated object into the ObjectArc
    ///
    /// # Arguments
    /// * `ptr` - The raw pointer to move into the ObjectArc
    ///
    /// # Returns
    /// * `ObjectArc<T>` - The ObjectArc
    /// \return The ObjectArc
    ///
    /// # Safety
    /// `ptr` must own one strong reference to a live, stable managed allocation
    /// whose dynamic type is compatible with `T` and whose payload is valid.
    /// Ownership of that reference is transferred into the returned handle.
    #[inline(always)]
    pub unsafe fn from_raw(ptr: *const T) -> Self {
        Self {
            ptr: std::ptr::NonNull::new_unchecked(ptr as *mut T),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Move the ObjectArc into a raw pointer
    ///
    /// # Arguments
    /// * `this` - The ObjectArc to move into a raw pointer
    ///
    /// # Returns
    /// * `*const T` - The raw pointer, which owns the strong reference; pass it
    ///   back to [`ObjectArc::from_raw`] to release it, as std `Arc::into_raw`.
    #[inline(always)]
    pub fn into_raw(this: Self) -> *const T {
        let droped_this = std::mem::ManuallyDrop::new(this);
        droped_this.ptr.as_ptr() as *const T
    }

    /// Get the raw pointer from the ObjectArc
    ///
    /// Caller should view this as a non-owning reference
    ///
    /// # Arguments
    /// * `this` - The ObjectArc to get the raw pointer
    ///
    /// # Returns
    /// * `*const T` - The raw pointer, as std `Arc::as_ptr`. Dereferencing it
    ///   must respect existing borrows, and it cannot outlive an owning handle.
    #[inline]
    pub fn as_ptr(this: &Self) -> *const T {
        this.ptr.as_ptr() as *const T
    }

    /// Get the raw mutable pointer from the ObjectArc
    ///
    /// Caller should view this as a non-owning reference
    ///
    /// # Arguments
    /// * `this` - The ObjectArc to get the raw pointer
    ///
    /// # Returns
    /// * `*mut T` - The raw pointer. The handle need not be unique: writing
    ///   through it needs exclusive access to the object, and must not move or
    ///   replace the object or its header.
    #[inline]
    pub fn as_mut_ptr(this: &mut Self) -> *mut T {
        this.ptr.as_ptr()
    }

    /// Get the strong reference count of the ObjectArc
    ///
    /// # Arguments
    /// * `this` - The ObjectArc to get the strong reference count
    ///
    /// # Returns
    /// * `usize` - The strong reference count
    #[inline]
    pub fn strong_count(this: &Self) -> usize {
        unsafe { unsafe_::strong_count(this.ptr.as_ptr().cast::<TVMFFIObject>()) }
    }

    /// Get the weak reference count of the ObjectArc
    ///
    /// # Arguments
    /// * `this` - The ObjectArc to get the weak reference count
    ///
    /// # Returns
    /// * `usize` - The weak reference count
    #[inline]
    pub fn weak_count(this: &Self) -> usize {
        unsafe { unsafe_::weak_count(this.ptr.as_ptr().cast::<TVMFFIObject>()) }
    }
}

// implement Deref for ObjectArc
impl<T: ObjectCore> Deref for ObjectArc<T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { self.ptr.as_ref() }
    }
}

// implement Drop for ObjectArc
impl<T: ObjectCore> Drop for ObjectArc<T> {
    #[inline(always)]
    fn drop(&mut self) {
        unsafe { unsafe_::dec_ref(self.ptr.as_ptr().cast::<TVMFFIObject>()) }
    }
}

// implement Clone for ObjectArc
impl<T: ObjectCore> Clone for ObjectArc<T> {
    #[inline]
    fn clone(&self) -> Self {
        unsafe { unsafe_::inc_ref(self.ptr.as_ptr().cast::<TVMFFIObject>()) }
        Self {
            ptr: self.ptr,
            _phantom: std::marker::PhantomData,
        }
    }
}
