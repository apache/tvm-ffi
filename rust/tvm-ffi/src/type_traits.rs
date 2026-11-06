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
use crate::any::{Any, AnyView};
use tvm_ffi_sys::TVMFFITypeIndex as TypeIndex;
use tvm_ffi_sys::{TVMFFIAny, TVMFFIGetTypeInfo, TVMFFIObject};

//-----------------------------------------------------
// AnyCompatible
//-----------------------------------------------------
/// Trait to enable a value to be compatible with Any
/// Enables TryFrom/Into AnyView/Any
pub unsafe trait AnyCompatible: Sized {
    /// Whether the `AnyView -> Self` conversion matches one exact leaf type.
    ///
    /// If this is `true`, [`Self::match_any_exact_type_index`] must always
    /// return the same index for the process lifetime, and conversion from an
    /// object-backed `AnyView` must succeed if and only if its runtime type
    /// index equals that index. Matching must not depend on any other part of
    /// the value. `match_any!` may use this contract to skip conversions for
    /// all other arms.
    #[doc(hidden)]
    const MATCH_ANY_EXACT: bool = false;

    /// Whether strict matching accepts exactly `match_any_exact_type_index()`.
    /// Unlike `MATCH_ANY_EXACT`, this covers every tag, including scalars:
    /// `check_any_strict` and `match_constraints` together must succeed if and
    /// only if the index matches.
    /// Container-element checks must have the same acceptance set.
    #[doc(hidden)]
    const MATCH_ANY_STRICT_EXACT: bool = false;

    /// Return the sole runtime object type index accepted by this conversion.
    #[doc(hidden)]
    fn match_any_exact_type_index() -> i32 {
        unreachable!("MATCH_ANY_EXACT is false")
    }

    /// Whether `field`, which holds a value of this type or all-zero bytes,
    /// holds all-zero bytes that are no value of this type, as a null object
    /// reference, a string of type index None or a zero device type: a field
    /// that a creator left undefined for reflection to set, as C++ `UnsafeInit`
    /// leaves a reference undefined (null). Unlike a C++ undefined reference,
    /// it is no value; a nullable reference is `Option<T>`, whose `None` is a
    /// defined value.
    ///
    /// The field setter writes an undefined field without dropping it, the
    /// field getter reads it as None, and the object's deleter skips it. The default,
    /// false, fits a type whose all-zero bytes are a value whose drop does
    /// nothing, as a scalar's, `Any`'s or `Optional`'s.
    ///
    /// # Safety
    ///
    /// `field` points to a value of this type or to all-zero bytes. An
    /// implementation reads only bytes that each value initializes and, when
    /// `Option<Self>` is niche-optimized, that its `None` initializes too.
    #[inline]
    unsafe fn is_undefined(field: *const Self) -> bool {
        let _ = field;
        false
    }

    /// the value to copy to TVMFFIAny
    unsafe fn copy_to_any_view(src: &Self, data: &mut TVMFFIAny);
    /// consume the value to move to Any
    unsafe fn move_to_any(src: Self, data: &mut TVMFFIAny);
    // check if the value is compatible with the type
    unsafe fn check_any_strict(data: &TVMFFIAny) -> bool;

    /// Additional value-preservation checks used only by `match_any!`, after
    /// `check_any_strict` succeeds. Ordinary conversion APIs are unchanged.
    ///
    /// # Safety
    /// `data` must have passed `Self::check_any_strict`.
    #[doc(hidden)]
    unsafe fn match_constraints(_data: &TVMFFIAny) -> bool {
        true
    }

    /// Copy value from TVMFFIAny after checking
    /// caller must ensure that the value is compatible with the type
    unsafe fn copy_from_any_view_after_check(data: &TVMFFIAny) -> Self;
    /// the value to move from TVMFFIAny
    /// NOTE: pay very careful attention to avoid memory leak!
    /// - When calling from managed Any, remember to use std::mem::ManuallyDrop
    unsafe fn move_from_any_after_check(data: &mut TVMFFIAny) -> Self;
    /// try to cast the value from AnyView
    unsafe fn try_cast_from_any_view(data: &TVMFFIAny) -> Result<Self, ()>;
    /// Get the type key of a type when TryCastFromAnyView fails.
    fn get_mismatch_type_info(data: &TVMFFIAny) -> String {
        unsafe {
            let info = TVMFFIGetTypeInfo(data.type_index);
            assert!(!info.is_null(), "TVMFFIGetTypeInfo returned null");
            (*info).type_key.as_str().to_string()
        }
    }
    /// the type string of the type
    fn type_str() -> String;

    /// The type schema of the type: the JSON object C++
    /// `TypeTraits<T>::TypeSchema` writes, which the type schemas of functions
    /// and fields embed.
    ///
    /// The default, `{"type":"<type_str>"}`, fits a type whose type string is
    /// its type key, as for object references.
    fn type_schema() -> String {
        type_schema(&Self::type_str(), &[])
    }

    /// Borrow `self` into an owned [`Any`], increfing object-backed values.
    ///
    /// The by-reference counterpart of `Any::from(value)`, for fields reached
    /// through a `Deref` into shared object storage. `impl From<&T> for Any`
    /// cannot be added instead: `&` is fundamental, so a downstream crate may
    /// implement `AnyCompatible` for its own `&T` and the two impls overlap
    /// (E0119).
    #[inline]
    fn to_any(&self) -> Any {
        AnyView::from(self).into()
    }
}

/// Writes a type schema, `{"type":"<origin>"}` or, with type arguments,
/// `{"type":"<origin>","args":[...]}`, as C++ `TypeTraits<T>::TypeSchema` does.
pub(crate) fn type_schema(origin: &str, args: &[String]) -> String {
    if args.is_empty() {
        format!(r#"{{"type":"{origin}"}}"#)
    } else {
        format!(r#"{{"type":"{origin}","args":[{}]}}"#, args.join(","))
    }
}

/// The metadata C++ records for a function or field of type schema
/// `type_schema`: `{"type_schema":<type_schema as a JSON string>}`.
pub(crate) fn type_schema_metadata(type_schema: &str) -> String {
    format!(r#"{{"type_schema":{}}}"#, escape_string_json(type_schema))
}

/// `text` as a JSON string literal, escaped as C++ `EscapeStringJSON`
/// (`string.h`) escapes it, so that metadata matches the C++ one byte for byte.
fn escape_string_json(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '/' => out.push_str("\\/"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if (ch as u32) < 0x20 || ch == '\u{7f}' => {
                out.push_str(&format!("\\u{:04x}", ch as u32))
            }
            ch => out.push(ch),
        }
    }
    out.push('"');
    out
}

/// The type schema of a function parameter or return type.
///
/// This is [`AnyCompatible::type_schema`] for a type that converts through
/// [`Any`], and `{"type":"Any"}` for `Any` and `AnyView` themselves, as C++
/// `details::TypeSchema<T>` writes them.
pub(crate) trait TypeSchema {
    fn type_schema() -> String;
}

impl<T: AnyCompatible> TypeSchema for T {
    fn type_schema() -> String {
        <T as AnyCompatible>::type_schema()
    }
}

impl TypeSchema for Any {
    fn type_schema() -> String {
        type_schema("Any", &[])
    }
}

impl TypeSchema for AnyView<'_> {
    fn type_schema() -> String {
        type_schema("Any", &[])
    }
}

/// Marker for a value that can be stored in an FFI container.
///
/// This is an implementation detail of [`crate::Array`] and [`crate::Map`].
/// Users should implement [`AnyCompatible`]; the blanket implementation below
/// then makes that type a container element automatically. [`Any`] is handled
/// separately so heterogeneous containers such as `Array<Any>` also work.
///
/// The private supertrait seals this marker and owns all conversion operations,
/// keeping them out of the user-facing trait API.
#[doc(hidden)]
pub trait ContainerElement: container_element_ops::Ops {}

impl<T: AnyCompatible> ContainerElement for T {}
impl ContainerElement for Any {}

/// Types whose value an FFI `Any` cell stores in place, so `&Self` can be
/// borrowed from the cell, as [`crate::Array`] indexing does.
///
/// # Safety
///
/// For a cell that passes `container_check_any_strict` for `Self`,
/// `ref_from_any_unchecked` must return a reference into that cell that is a valid
/// `Self` for as long as the cell is borrowed.
#[doc(hidden)]
pub unsafe trait RefFromAnyUnchecked: ContainerElement {
    /// Borrow the value stored in `cell`.
    ///
    /// # Safety
    ///
    /// `cell` must hold a `Self`.
    unsafe fn ref_from_any_unchecked(cell: &TVMFFIAny) -> &Self;
}

// The whole cell is the value.
unsafe impl RefFromAnyUnchecked for Any {
    #[inline(always)]
    unsafe fn ref_from_any_unchecked(cell: &TVMFFIAny) -> &Self {
        &*(cell as *const TVMFFIAny as *const Any)
    }
}

// A `BigInt` is one cell, holding an inline `i64` or a `BigInt` object.
unsafe impl RefFromAnyUnchecked for crate::big_int::BigInt {
    #[inline(always)]
    unsafe fn ref_from_any_unchecked(cell: &TVMFFIAny) -> &Self {
        const {
            assert!(
                std::mem::size_of::<crate::big_int::BigInt>() == std::mem::size_of::<TVMFFIAny>()
            );
            assert!(
                std::mem::align_of::<crate::big_int::BigInt>() == std::mem::align_of::<TVMFFIAny>()
            );
        };
        &*(cell as *const TVMFFIAny as *const crate::big_int::BigInt)
    }
}

unsafe impl RefFromAnyUnchecked for crate::String {
    #[inline(always)]
    unsafe fn ref_from_any_unchecked(cell: &TVMFFIAny) -> &Self {
        &*(cell as *const TVMFFIAny as *const crate::String)
    }
}

unsafe impl RefFromAnyUnchecked for crate::string::Bytes {
    #[inline(always)]
    unsafe fn ref_from_any_unchecked(cell: &TVMFFIAny) -> &Self {
        &*(cell as *const TVMFFIAny as *const crate::string::Bytes)
    }
}

// The payload word holds the value in its own representation.
unsafe impl RefFromAnyUnchecked for i64 {
    #[inline(always)]
    unsafe fn ref_from_any_unchecked(cell: &TVMFFIAny) -> &Self {
        &cell.data_union.v_int64
    }
}

unsafe impl RefFromAnyUnchecked for f64 {
    #[inline(always)]
    unsafe fn ref_from_any_unchecked(cell: &TVMFFIAny) -> &Self {
        &cell.data_union.v_float64
    }
}

unsafe impl RefFromAnyUnchecked for tvm_ffi_sys::dlpack::DLDataType {
    #[inline(always)]
    unsafe fn ref_from_any_unchecked(cell: &TVMFFIAny) -> &Self {
        &cell.data_union.v_dtype
    }
}

unsafe impl RefFromAnyUnchecked for tvm_ffi_sys::dlpack::DLDevice {
    #[inline(always)]
    unsafe fn ref_from_any_unchecked(cell: &TVMFFIAny) -> &Self {
        &cell.data_union.v_device
    }
}

// An optional object reference is one nullable object pointer (`None` is null,
// as `OptionalCompatible` relies on), which is the payload word of a `None` or
// object cell, so `&Option<T>` borrows that word.
unsafe impl<T: crate::object::ObjectRefCore> RefFromAnyUnchecked for Option<T> {
    #[inline(always)]
    unsafe fn ref_from_any_unchecked(cell: &TVMFFIAny) -> &Self {
        const {
            assert!(std::mem::size_of::<Option<T>>() == std::mem::size_of::<*mut TVMFFIObject>());
            assert!(std::mem::align_of::<Option<T>>() == std::mem::align_of::<*mut TVMFFIObject>());
        };
        if cell.type_index == TypeIndex::kTVMFFINone as i32 {
            // A `None` cell's payload is zero by convention (C++ `Any()`), but
            // do not rely on a producer for it.
            return &None;
        }
        &*(&cell.data_union.v_obj as *const *mut TVMFFIObject as *const Option<T>)
    }
}

unsafe impl<T: crate::object::ObjectRefCore> DowncastRefFromAny for Option<T> {
    #[inline(always)]
    unsafe fn downcast_ref_from_any(cell: &TVMFFIAny) -> Option<&Self> {
        downcast_ref_from_any_in_place::<Self>(cell)
    }
}

/// Types that [`crate::Any::downcast_ref`] and [`crate::AnyView::downcast_ref`]
/// can borrow from an FFI cell in place: object types (`ArrayObj`), and the
/// [`RefFromAnyUnchecked`] types (managed references such as `Array<T>`, `String`,
/// `i64`).
///
/// # Safety
///
/// `downcast_ref_from_any` must return `Some` only when `cell` holds a `Self`, with
/// a reference that stays valid while `cell` is borrowed.
#[doc(hidden)]
pub unsafe trait DowncastRefFromAny {
    /// Borrow `cell` as `Self` when it holds one.
    ///
    /// # Safety
    ///
    /// `cell` must be a valid FFI cell.
    unsafe fn downcast_ref_from_any(cell: &TVMFFIAny) -> Option<&Self>;
}

// An object type borrows the object the cell points to.
unsafe impl<T: crate::object::ObjectCore> DowncastRefFromAny for T {
    #[inline(always)]
    unsafe fn downcast_ref_from_any(cell: &TVMFFIAny) -> Option<&Self> {
        if cell.type_index < TypeIndex::kTVMFFIStaticObjectBegin as i32
            || !crate::object::is_instance_of::<T>(cell.type_index)
        {
            return None;
        }
        // SAFETY: the type check establishes T's layout, and the cell keeps the
        // object alive while it is borrowed.
        Some(&*cell.data_union.v_obj.cast::<T>())
    }
}

/// Borrow `cell` as a [`RefFromAnyUnchecked`] type when it strictly holds one.
///
/// # Safety
///
/// `cell` must be a valid FFI cell.
#[doc(hidden)]
#[inline(always)]
pub unsafe fn downcast_ref_from_any_in_place<T: RefFromAnyUnchecked>(
    cell: &TVMFFIAny,
) -> Option<&T> {
    if T::container_check_any_strict(cell) {
        Some(T::ref_from_any_unchecked(cell))
    } else {
        None
    }
}

/// Types that an object reference's [`ObjectRefCore::downcast_ref`] can borrow
/// from the reference's object pointer: object types (`ArrayObj`), and managed
/// references that are exactly one object pointer (`Array<T>`, every
/// `#[derive(ObjectRef)]` type).
///
/// [`ObjectRefCore::downcast_ref`]: crate::object::ObjectRefCore::downcast_ref
///
/// # Safety
///
/// `downcast_ref_from_object` must return `Some` only when the object at
/// `*place` is a `Self`, with a reference that stays valid while `place` is
/// borrowed.
#[doc(hidden)]
pub unsafe trait DowncastRefFromObject {
    /// Borrow the object at `*place` as `Self` when it is one.
    ///
    /// # Safety
    ///
    /// `place` must hold a pointer to a live object.
    unsafe fn downcast_ref_from_object(place: &*mut TVMFFIObject) -> Option<&Self>;
}

// An object type borrows the object itself.
unsafe impl<T: crate::object::ObjectCore> DowncastRefFromObject for T {
    #[inline(always)]
    unsafe fn downcast_ref_from_object(place: &*mut TVMFFIObject) -> Option<&Self> {
        let object = *place;
        if !crate::object::is_instance_of::<T>((*object).type_index) {
            return None;
        }
        // SAFETY: the type check establishes T's layout.
        Some(&*object.cast::<T>())
    }
}

/// Borrow the object pointer at `place` as the one-pointer reference type `T`
/// when the object strictly is a `T`.
///
/// # Safety
///
/// `place` must hold a pointer to a live object, and `T` must be exactly one
/// object pointer.
#[doc(hidden)]
#[inline(always)]
pub unsafe fn downcast_ref_from_object_in_place<T: RefFromAnyUnchecked>(
    place: &*mut TVMFFIObject,
) -> Option<&T> {
    const {
        assert!(
            std::mem::size_of::<T>() == std::mem::size_of::<*mut TVMFFIObject>(),
            "a reference downcast target must be exactly one object pointer"
        )
    };
    let object = *place;
    let mut cell = TVMFFIAny::new();
    cell.type_index = (*object).type_index;
    cell.data_union.v_obj = object;
    if T::container_check_any_strict(&cell) {
        // SAFETY: `T` is exactly one object pointer, and the check shows the
        // object satisfies `T`.
        Some(&*(place as *const *mut TVMFFIObject as *const T))
    } else {
        None
    }
}

macro_rules! impl_downcast_ref_from_any {
    ($($t:ty),* $(,)?) => {
        $(unsafe impl DowncastRefFromAny for $t {
            #[inline(always)]
            unsafe fn downcast_ref_from_any(cell: &TVMFFIAny) -> Option<&Self> {
                downcast_ref_from_any_in_place::<Self>(cell)
            }
        })*
    };
}

// An `&Any` must own what it points at, as an `Any` does: a view's cell may
// instead borrow a C string, byte array, `DLTensor` or rvalue slot, whose
// `&Any` would clone into an owner of a borrowed resource. Those are refused;
// an `Any` or container cell never holds them.
unsafe impl DowncastRefFromAny for Any {
    #[inline(always)]
    unsafe fn downcast_ref_from_any(cell: &TVMFFIAny) -> Option<&Self> {
        let index = cell.type_index;
        let borrowed = index == TypeIndex::kTVMFFIRawStr as i32
            || index == TypeIndex::kTVMFFIByteArrayPtr as i32
            || index == TypeIndex::kTVMFFIDLTensorPtr as i32
            || index == TypeIndex::kTVMFFIObjectRValueRef as i32;
        if borrowed {
            None
        } else {
            Some(Any::ref_from_any_unchecked(cell))
        }
    }
}

impl_downcast_ref_from_any!(
    crate::String,
    crate::string::Bytes,
    i64,
    f64,
    crate::big_int::BigInt,
    tvm_ffi_sys::dlpack::DLDataType,
    tvm_ffi_sys::dlpack::DLDevice,
);

mod container_element_ops {
    use super::{Any, AnyCompatible, AnyView, TVMFFIAny};

    pub trait Ops: Sized {
        const CONTAINER_IS_ANY: bool = false;

        unsafe fn container_copy_to_any_view(src: &Self, data: &mut TVMFFIAny);
        unsafe fn container_move_to_any(src: Self, data: &mut TVMFFIAny);
        unsafe fn container_check_any_strict(data: &TVMFFIAny) -> bool;
        unsafe fn container_match_constraints(data: &TVMFFIAny) -> bool;
        unsafe fn container_copy_from_any_view_after_check(data: &TVMFFIAny) -> Self;
        unsafe fn container_move_from_any_after_check(data: &mut TVMFFIAny) -> Self;
        unsafe fn container_try_cast_from_any_view(data: &TVMFFIAny) -> Result<Self, ()>;
        fn container_get_mismatch_type_info(data: &TVMFFIAny) -> String;
        fn container_type_str() -> String;
        fn container_type_schema() -> String;
    }

    impl<T: AnyCompatible> Ops for T {
        #[inline]
        unsafe fn container_copy_to_any_view(src: &Self, data: &mut TVMFFIAny) {
            <T as AnyCompatible>::copy_to_any_view(src, data)
        }

        #[inline]
        unsafe fn container_move_to_any(src: Self, data: &mut TVMFFIAny) {
            <T as AnyCompatible>::move_to_any(src, data)
        }

        #[inline]
        unsafe fn container_check_any_strict(data: &TVMFFIAny) -> bool {
            <T as AnyCompatible>::check_any_strict(data)
        }
        #[inline]
        unsafe fn container_match_constraints(data: &TVMFFIAny) -> bool {
            <T as AnyCompatible>::match_constraints(data)
        }

        #[inline]
        unsafe fn container_copy_from_any_view_after_check(data: &TVMFFIAny) -> Self {
            <T as AnyCompatible>::copy_from_any_view_after_check(data)
        }

        #[inline]
        unsafe fn container_move_from_any_after_check(data: &mut TVMFFIAny) -> Self {
            <T as AnyCompatible>::move_from_any_after_check(data)
        }

        #[inline]
        unsafe fn container_try_cast_from_any_view(data: &TVMFFIAny) -> Result<Self, ()> {
            <T as AnyCompatible>::try_cast_from_any_view(data)
        }

        #[inline]
        fn container_get_mismatch_type_info(data: &TVMFFIAny) -> String {
            <T as AnyCompatible>::get_mismatch_type_info(data)
        }

        #[inline]
        fn container_type_str() -> String {
            <T as AnyCompatible>::type_str()
        }

        #[inline]
        fn container_type_schema() -> String {
            <T as AnyCompatible>::type_schema()
        }
    }

    impl Ops for Any {
        const CONTAINER_IS_ANY: bool = true;

        #[inline]
        unsafe fn container_copy_to_any_view(src: &Self, data: &mut TVMFFIAny) {
            *data = *src.as_raw_ffi_any();
        }

        #[inline]
        unsafe fn container_move_to_any(src: Self, data: &mut TVMFFIAny) {
            *data = Any::into_raw_ffi_any(src);
        }

        #[inline]
        unsafe fn container_check_any_strict(_data: &TVMFFIAny) -> bool {
            true
        }
        #[inline]
        unsafe fn container_match_constraints(_data: &TVMFFIAny) -> bool {
            true
        }

        #[inline]
        unsafe fn container_copy_from_any_view_after_check(data: &TVMFFIAny) -> Self {
            Any::from(AnyView::from_raw_ffi_any(*data))
        }

        #[inline]
        unsafe fn container_move_from_any_after_check(data: &mut TVMFFIAny) -> Self {
            Any::from_raw_ffi_any(std::mem::replace(data, TVMFFIAny::new()))
        }

        #[inline]
        unsafe fn container_try_cast_from_any_view(data: &TVMFFIAny) -> Result<Self, ()> {
            Ok(Any::from(AnyView::from_raw_ffi_any(*data)))
        }

        #[inline]
        fn container_get_mismatch_type_info(_data: &TVMFFIAny) -> String {
            "Any".to_string()
        }

        #[inline]
        fn container_type_str() -> String {
            "Any".to_string()
        }

        #[inline]
        fn container_type_schema() -> String {
            <Any as super::TypeSchema>::type_schema()
        }
    }
}

/// AnyCompatible for bool
unsafe impl AnyCompatible for bool {
    unsafe fn copy_to_any_view(src: &Self, data: &mut TVMFFIAny) {
        data.type_index = TypeIndex::kTVMFFIBool as i32;
        data.small_str_len = 0;
        data.data_union.v_int64 = *src as i64;
    }
    unsafe fn check_any_strict(data: &TVMFFIAny) -> bool {
        data.type_index == TypeIndex::kTVMFFIBool as i32
    }

    unsafe fn copy_from_any_view_after_check(data: &TVMFFIAny) -> Self {
        data.data_union.v_int64 != 0
    }

    unsafe fn move_to_any(src: Self, data: &mut TVMFFIAny) {
        data.type_index = TypeIndex::kTVMFFIBool as i32;
        data.small_str_len = 0;
        data.data_union.v_int64 = src as i64;
    }

    unsafe fn move_from_any_after_check(data: &mut TVMFFIAny) -> Self {
        data.data_union.v_int64 != 0
    }

    unsafe fn try_cast_from_any_view(data: &TVMFFIAny) -> Result<Self, ()> {
        if data.type_index == TypeIndex::kTVMFFIBool as i32
            || data.type_index == TypeIndex::kTVMFFIInt as i32
        {
            unsafe { Ok(data.data_union.v_int64 != 0) }
        } else {
            Err(())
        }
    }

    fn type_str() -> String {
        "bool".to_string()
    }
}

/// Macro to implement AnyCompatible for integer types
macro_rules! impl_any_compatible_for_int {
    ($($int_type:ty),* $(,)?) => {
        $(
            unsafe impl AnyCompatible for $int_type {
                fn type_str() -> String {
                    "int".to_string()
                }

                unsafe fn copy_to_any_view(src: &Self, data: &mut TVMFFIAny) {
                    data.type_index = TypeIndex::kTVMFFIInt as i32;
                    data.small_str_len = 0;
                    data.data_union.v_int64 = *src as i64;
                }

                unsafe fn check_any_strict(data: &TVMFFIAny) -> bool {
                    data.type_index == TypeIndex::kTVMFFIInt as i32
                }

                unsafe fn match_constraints(data: &TVMFFIAny) -> bool {
                    <$int_type>::try_from(data.data_union.v_int64).is_ok()
                }

                unsafe fn copy_from_any_view_after_check(data: &TVMFFIAny) -> Self {
                    data.data_union.v_int64 as $int_type
                }

                unsafe fn move_to_any(src: Self, data: &mut TVMFFIAny) {
                    data.type_index = TypeIndex::kTVMFFIInt as i32;
                    data.small_str_len = 0;
                    data.data_union.v_int64 = src as i64;
                }

                unsafe fn move_from_any_after_check(data: &mut TVMFFIAny) -> Self {
                    data.data_union.v_int64 as $int_type
                }

                unsafe fn try_cast_from_any_view(data: &TVMFFIAny) -> Result<Self, ()> {
                    if data.type_index == TypeIndex::kTVMFFIInt as i32 ||
                       data.type_index == TypeIndex::kTVMFFIBool as i32 {
                        Ok(data.data_union.v_int64 as $int_type)
                    } else {
                        Err(())
                    }
                }
            }
        )*
    };
}

// Implement AnyCompatible for all integer types
impl_any_compatible_for_int!(i8, i16, i32, i64, isize, u8, u16, u32, u64, usize);

// Implement AnyCompatible for `Option<T>`
unsafe impl<T: AnyCompatible> AnyCompatible for Option<T> {
    fn type_str() -> String {
        // make it consistent with c++ representation
        "Optional<".to_string() + T::type_str().as_str() + ">"
    }

    fn type_schema() -> String {
        type_schema("Optional", &[<T as AnyCompatible>::type_schema()])
    }

    unsafe fn is_undefined(field: *const Self) -> bool {
        // With a tag, all-zero bytes are `None`. A niche-optimized `Option<T>`
        // keeps `T`'s bytes, where all-zero bytes are `None` if `None` looks
        // undefined to `T` (a reference's null pointer, a device's zero type),
        // and undefined if `None` sits elsewhere in the niche.
        if std::mem::size_of::<Self>() != std::mem::size_of::<T>() {
            return false;
        }
        let none = std::mem::MaybeUninit::new(None::<T>);
        T::is_undefined(field.cast::<T>()) && !T::is_undefined(none.as_ptr().cast::<T>())
    }

    unsafe fn copy_to_any_view(src: &Self, data: &mut TVMFFIAny) {
        if let Some(ref value) = src {
            T::copy_to_any_view(value, data);
        } else {
            data.type_index = TypeIndex::kTVMFFINone as i32;
            data.small_str_len = 0;
            data.data_union.v_int64 = 0;
        }
    }

    unsafe fn move_to_any(src: Self, data: &mut TVMFFIAny) {
        if let Some(value) = src {
            T::move_to_any(value, data);
        } else {
            data.type_index = TypeIndex::kTVMFFINone as i32;
            data.small_str_len = 0;
            data.data_union.v_int64 = 0;
        }
    }

    unsafe fn check_any_strict(data: &TVMFFIAny) -> bool {
        return T::check_any_strict(data) || data.type_index == TypeIndex::kTVMFFINone as i32;
    }

    unsafe fn match_constraints(data: &TVMFFIAny) -> bool {
        data.type_index == TypeIndex::kTVMFFINone as i32 || T::match_constraints(data)
    }

    unsafe fn copy_from_any_view_after_check(data: &TVMFFIAny) -> Self {
        if data.type_index == TypeIndex::kTVMFFINone as i32 {
            None
        } else {
            Some(T::copy_from_any_view_after_check(data))
        }
    }

    unsafe fn move_from_any_after_check(data: &mut TVMFFIAny) -> Self {
        if data.type_index == TypeIndex::kTVMFFINone as i32 {
            None
        } else {
            Some(T::move_from_any_after_check(data))
        }
    }

    unsafe fn try_cast_from_any_view(data: &TVMFFIAny) -> Result<Self, ()> {
        if data.type_index == TypeIndex::kTVMFFINone as i32 {
            Ok(None)
        } else {
            T::try_cast_from_any_view(data).map(Some)
        }
    }
}

/// AnyCompatible for void*
unsafe impl AnyCompatible for *mut core::ffi::c_void {
    unsafe fn copy_to_any_view(src: &Self, data: &mut TVMFFIAny) {
        data.type_index = TypeIndex::kTVMFFIOpaquePtr as i32;
        data.small_str_len = 0;
        data.data_union.v_ptr = *src;
    }
    unsafe fn check_any_strict(data: &TVMFFIAny) -> bool {
        data.type_index == TypeIndex::kTVMFFIOpaquePtr as i32
    }

    unsafe fn copy_from_any_view_after_check(data: &TVMFFIAny) -> Self {
        data.data_union.v_ptr
    }

    unsafe fn move_to_any(src: Self, data: &mut TVMFFIAny) {
        data.type_index = TypeIndex::kTVMFFIOpaquePtr as i32;
        data.small_str_len = 0;
        data.data_union.v_int64 = src as i64;
    }

    unsafe fn move_from_any_after_check(data: &mut TVMFFIAny) -> Self {
        data.data_union.v_ptr
    }

    unsafe fn try_cast_from_any_view(data: &TVMFFIAny) -> Result<Self, ()> {
        if data.type_index == TypeIndex::kTVMFFINone as i32 {
            Ok(std::ptr::null_mut())
        } else if data.type_index == TypeIndex::kTVMFFIOpaquePtr as i32 {
            unsafe { Ok(data.data_union.v_ptr) }
        } else {
            Err(())
        }
    }

    fn type_str() -> String {
        "void*".to_string()
    }
}

/// Macro to implement AnyCompatible for integer types
macro_rules! impl_any_compatible_for_float {
    ($($float_type:ty),* $(,)?) => {
        $(
            unsafe impl AnyCompatible for $float_type {
                fn type_str() -> String {
                    "float".to_string()
                }

                unsafe fn copy_to_any_view(src: &Self, data: &mut TVMFFIAny) {
                    data.type_index = TypeIndex::kTVMFFIFloat as i32;
                    data.small_str_len = 0;
                    data.data_union.v_float64 = *src as f64;
                }

                unsafe fn check_any_strict(data: &TVMFFIAny) -> bool {
                    data.type_index == TypeIndex::kTVMFFIFloat as i32
                }

                unsafe fn match_constraints(data: &TVMFFIAny) -> bool {
                    let value = data.data_union.v_float64;
                    ((value as $float_type) as f64).to_bits() == value.to_bits()
                }

                unsafe fn copy_from_any_view_after_check(data: &TVMFFIAny) -> Self {
                    data.data_union.v_float64 as $float_type
                }

                unsafe fn move_to_any(src: Self, data: &mut TVMFFIAny) {
                    data.type_index = TypeIndex::kTVMFFIFloat as i32;
                    data.small_str_len = 0;
                    data.data_union.v_float64 = src as f64;
                }

                unsafe fn move_from_any_after_check(data: &mut TVMFFIAny) -> Self {
                    data.data_union.v_float64 as $float_type
                }

                unsafe fn try_cast_from_any_view(data: &TVMFFIAny) -> Result<Self, ()> {
                    if data.type_index == TypeIndex::kTVMFFIFloat as i32 {
                        Ok(data.data_union.v_float64 as $float_type)
                    } else if data.type_index == TypeIndex::kTVMFFIInt as i32 ||
                       data.type_index == TypeIndex::kTVMFFIBool as i32 {
                        Ok(data.data_union.v_int64 as $float_type)
                    } else {
                        Err(())
                    }
                }
            }
        )*
    };
}

// Implement AnyCompatible for all float types
impl_any_compatible_for_float!(f32, f64);

// Special rule: we convert () to None
// This allows us to effectively pass () around as null value
// note that this is indeed a bit relaxation of the type
// but it is necessary for us to enable void/none interoperability
unsafe impl AnyCompatible for () {
    fn type_str() -> String {
        // make it consistent with c++ representation
        "None".to_string()
    }

    unsafe fn copy_to_any_view(_src: &Self, data: &mut TVMFFIAny) {
        data.type_index = TypeIndex::kTVMFFINone as i32;
        data.small_str_len = 0;
        data.data_union.v_int64 = 0;
    }

    unsafe fn move_to_any(_src: Self, data: &mut TVMFFIAny) {
        data.type_index = TypeIndex::kTVMFFINone as i32;
        data.small_str_len = 0;
        data.data_union.v_int64 = 0;
    }

    unsafe fn check_any_strict(data: &TVMFFIAny) -> bool {
        return data.type_index == TypeIndex::kTVMFFINone as i32;
    }

    unsafe fn copy_from_any_view_after_check(_data: &TVMFFIAny) -> Self {
        ()
    }

    unsafe fn move_from_any_after_check(_data: &mut TVMFFIAny) -> Self {
        ()
    }

    unsafe fn try_cast_from_any_view(data: &TVMFFIAny) -> Result<Self, ()> {
        if data.type_index == TypeIndex::kTVMFFINone as i32 {
            Ok(())
        } else {
            Err(())
        }
    }
}
