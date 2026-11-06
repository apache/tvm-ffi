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
// Re-export paste for the export macros, so downstream crates need not depend on it.
#[doc(hidden)]
pub use paste;

// ----------------------------------------------------------------------------
// Macros for error handling
// ----------------------------------------------------------------------------

/// The path of the enclosing function, as `function_name!()`.
#[doc(hidden)]
#[macro_export]
macro_rules! function_name {
    () => {{
        // dummy function to get the name of the function
        fn f() {}
        fn type_name_of<T>(_: T) -> &'static str {
            std::any::type_name::<T>()
        }
        let name = type_name_of(f);
        // remove the f() from the name
        &name[..name.len() - 3]
    }};
}

/// Look up a registered global [`Function`](crate::Function) by name once, and
/// keep it for the process: a `&'static Function`.
///
/// Each call site has its own cache, which later calls read without a lock.
/// Use it for a function that must be registered, such as a built-in one; it
/// panics, naming the function, if the function is not. See also
/// [`Function::get_global`](crate::Function::get_global), which returns an
/// error instead.
///
/// # Examples
///
/// ```
/// use tvm_ffi::{cached_global_func, Array};
///
/// let array = cached_global_func!("ffi.Array").call_tuple((1_i64, 2_i64))?;
/// assert_eq!(Array::<i64>::try_from(array)?.len(), 2);
/// # Ok::<(), tvm_ffi::Error>(())
/// ```
#[macro_export]
macro_rules! cached_global_func {
    ($name:literal) => {{
        static FUNC: $crate::extra::cold_init_lazy_lock::ColdInitLazyLock<
            $crate::function::Function,
        > = $crate::extra::cold_init_lazy_lock::ColdInitLazyLock::new(|| {
            $crate::function::Function::get_global($name).unwrap_or_else(|_| {
                panic!(concat!("global function `", $name, "` is not registered"))
            })
        });
        &*FUNC
    }};
}

/// Turn the return code of a safe call into a `Result`, taking the raised
/// error when it is nonzero: `check_safe_call!(function(args))?`.
#[doc(hidden)]
#[macro_export]
macro_rules! check_safe_call {
    ($expr:expr) => {{
        let ret_code = $expr;
        if ret_code == 0 {
            Ok(())
        } else {
            let error = $crate::error::Error::from_raised();
            Err(error)
        }
    }};
}

/// Return an [`Error`](crate::Error) of a kind with a formatted message from
/// the current function.
///
/// `bail!(kind, fmt, args...)` formats the message as `format!` does and
/// records the source location in the backtrace.
///
/// # Arguments
///
/// * `kind` - The kind of the error, such as [`VALUE_ERROR`](crate::VALUE_ERROR).
/// * `fmt` - The format string of the message.
/// * `args` - The format arguments.
///
/// # Returns
///
/// Returns `Err` of the error from the enclosing function.
///
/// # Examples
///
/// ```
/// use tvm_ffi::{bail, Result, INDEX_ERROR};
///
/// fn get(items: &[i64], index: usize) -> Result<i64> {
///     if index >= items.len() {
///         bail!(INDEX_ERROR, "index {} out of range for length {}", index, items.len());
///     }
///     Ok(items[index])
/// }
///
/// assert_eq!(get(&[1, 2], 1)?, 2);
/// assert_eq!(get(&[1, 2], 5).unwrap_err().message(), "index 5 out of range for length 2");
/// # Ok::<(), tvm_ffi::Error>(())
/// ```
#[macro_export]
macro_rules! bail {
    ($error_kind:expr, $fmt:expr $(, $args:expr)* $(,)?) => {{
        let context = format!(
            "  File \"{}\", line {}, in {}\n",
            file!(),
            line!(),
            $crate::function_name!()
        );
        return Err($crate::error::Error::new($error_kind, &format!($fmt $(, $args)*), &context));
    }};
}

/// Return an [`Error`](crate::Error) from the current function unless a
/// condition holds.
///
/// `ensure!(cond, kind, fmt, args...)` is [`bail!`](crate::bail)`(kind, fmt,
/// args...)` when `cond` is false.
///
/// # Arguments
///
/// * `cond` - The condition that must hold.
/// * `kind` - The kind of the error.
/// * `fmt` - The format string of the message.
/// * `args` - The format arguments.
///
/// # Returns
///
/// Returns `Err` of the error from the enclosing function when `cond` is
/// false, and does nothing otherwise.
///
/// # Examples
///
/// ```
/// use tvm_ffi::{ensure, Result, VALUE_ERROR};
///
/// fn sqrt(x: f64) -> Result<f64> {
///     ensure!(x >= 0.0, VALUE_ERROR, "sqrt of negative number {}", x);
///     Ok(x.sqrt())
/// }
///
/// assert_eq!(sqrt(4.0)?, 2.0);
/// assert!(sqrt(-1.0).is_err());
/// # Ok::<(), tvm_ffi::Error>(())
/// ```
#[macro_export]
macro_rules! ensure {
    ($cond:expr, $error_kind:expr, $fmt:expr $(, $args:expr)* $(,)?) => {{
        if !$cond {
            $crate::bail!($error_kind, $fmt $(, $args)*);
        }
    }};
}

/// Append the current source location to the backtrace of an `Err` result,
/// and pass an `Ok` through.
///
/// # Arguments
///
/// * `result` - The result to attach the context to.
///
/// # Returns
///
/// The result, with the location appended to an error's backtrace.
///
/// # Examples
///
/// ```
/// use tvm_ffi::{attach_context, bail, Result, RUNTIME_ERROR};
///
/// fn inner() -> Result<()> {
///     bail!(RUNTIME_ERROR, "failed");
/// }
///
/// fn outer() -> Result<()> {
///     attach_context!(inner())
/// }
///
/// assert!(outer().unwrap_err().backtrace().contains("outer"));
/// ```
#[macro_export]
macro_rules! attach_context {
    ($error:expr) => {{
        match $error {
            Ok(value) => Ok(value),
            Err(error) => {
                let context = format!(
                    "  File \"{}\", line {}, in {}\n",
                    file!(),
                    line!(),
                    $crate::function_name!()
                );
                Err($crate::error::Error::with_appended_backtrace(
                    error, &context,
                ))
            }
        }
    }};
}

// ----------------------------------------------------------------------------
// Macros for object references
// ----------------------------------------------------------------------------

/// Implement zero-copy conversions from a derived object reference to one of
/// its base object-reference types: `From<Source>` and `From<&Source>` into
/// `Target`, `AsRef<Target>`, which borrows a `&Source` as `&Target`
/// without touching the reference count, and
/// [`IntoObjectRef<Target>`](crate::object::IntoObjectRef) for `Source` and `&Source`.
/// It also implements the strict downcast back, `TryFrom<Target>` and
/// `TryFrom<&Target>` for `Source`, which return a `TypeError` naming both types.
///
/// # Safety
///
/// The declaration is trusted, as an `unsafe impl` is, hence the `unsafe_`
/// name: `target` must be a registered base of `source`, and every `source`
/// must satisfy the target reference invariants. An incorrect declaration makes
/// the generated safe conversions unsound. The Rust stub generator writes these
/// from the registered type hierarchy.
#[macro_export]
macro_rules! unsafe_impl_object_ref_upcast {
    ($($source:ty => $target:ty),+ $(,)?) => {
        $(
            impl ::std::convert::From<$source> for $target {
                #[inline]
                fn from(value: $source) -> Self {
                    let data = <$source as $crate::object::ObjectRefCore>::into_data(value);
                    // SAFETY: The macro declaration promises that `target` is
                    // a registered base of `source`. Both references retain
                    // the same allocation and only change its static view.
                    let data = unsafe {
                        $crate::object::ObjectArc::from_raw(
                            $crate::object::ObjectArc::into_raw(data).cast::
                                <<$target as $crate::object::ObjectRefCore>::ContainerType>(),
                        )
                    };
                    // SAFETY: The macro declaration promises both the
                    // container inheritance relation and every additional
                    // invariant imposed by the target reference view.
                    unsafe {
                        <$target as $crate::object::ObjectRefCore>::from_data(data)
                    }
                }
            }

            impl ::std::convert::From<&$source> for $target {
                #[inline]
                fn from(value: &$source) -> Self {
                    value.clone().into()
                }
            }

            impl $crate::object::IntoObjectRef<$target> for $source {
                type Output = $target;
                #[inline(always)]
                fn into_object_ref(self) -> $target {
                    ::core::convert::Into::into(self)
                }
            }

            impl<'a> $crate::object::IntoObjectRef<$target> for &'a $source {
                type Output = &'a $target;
                #[inline(always)]
                fn into_object_ref(self) -> &'a $target {
                    ::core::convert::AsRef::<$target>::as_ref(self)
                }
            }

            impl ::core::convert::AsRef<$target> for $source {
                #[inline(always)]
                fn as_ref(&self) -> &$target {
                    const _: () = assert!(
                        ::core::mem::size_of::<$source>() == ::core::mem::size_of::<$target>()
                            && ::core::mem::align_of::<$source>()
                                == ::core::mem::align_of::<$target>(),
                        "an object reference upcast must keep the reference's layout"
                    );
                    // SAFETY: both references are exactly one object pointer, and the
                    // macro declaration promises that every `source` object is a valid
                    // `target`, so a `&source` is a valid `&target`.
                    unsafe { &*(self as *const $source as *const $target) }
                }
            }

            // The strict downcast back, as a `TryFrom` for `?`: a reference
            // never converts, so this only checks the object's runtime type.
            impl ::core::convert::TryFrom<$target> for $source {
                type Error = $crate::error::Error;
                #[inline]
                fn try_from(value: $target) -> ::core::result::Result<Self, Self::Error> {
                    $crate::object::ObjectRefCore::downcast::<$source>(value)
                        .map_err($crate::error::Error::downcast_failed::<$source>)
                }
            }

            impl ::core::convert::TryFrom<&$target> for $source {
                type Error = $crate::error::Error;
                #[inline]
                fn try_from(value: &$target) -> ::core::result::Result<Self, Self::Error> {
                    match $crate::object::ObjectRefCore::downcast_ref::<$source>(value) {
                        ::core::option::Option::Some(found) => {
                            ::core::result::Result::Ok(::core::clone::Clone::clone(found))
                        }
                        ::core::option::Option::None => ::core::result::Result::Err(
                            $crate::error::Error::downcast_failed::<$source>(
                                ::core::clone::Clone::clone(value),
                            ),
                        ),
                    }
                }
            }
        )+
    };
}

/// Implement the in-place `downcast_ref` borrows (`array[i]`, `Any::downcast_ref`,
/// [`ObjectRefCore::downcast_ref`](crate::ObjectRefCore::downcast_ref)) for each
/// listed object reference type. [`#[derive(ObjectRef)]`](crate::derive::ObjectRef)
/// emits it; a reference that implements `ObjectRefCore` by hand calls it.
///
/// # Safety
///
/// Each type is `repr(transparent)` over one object pointer, and its
/// [`AnyCompatible::check_any_strict`](crate::AnyCompatible::check_any_strict)
/// accepts exactly the objects that are a valid value of it.
///
/// # Examples
///
/// ```
/// use tvm_ffi::derive::ObjectRef;
/// use tvm_ffi::{rs_object, unsafe_impl_object_ref_downcast_ref, Any};
/// use tvm_ffi::{Object, ObjectArc, ObjectRefAnyCompatible, ObjectRefCore};
///
/// #[rs_object("my_ext.MyPair", final)]
/// pub struct MyPairObj {
///     base: Object,
///     pub first: i64,
///     pub second: i64,
/// }
///
/// #[derive(Clone, ObjectRef)]
/// pub struct MyPair {
///     data: ObjectArc<MyPairObj>,
/// }
///
/// /// A `MyPair` whose `first` is positive.
/// #[derive(Clone)]
/// #[repr(transparent)]
/// pub struct PositivePair(ObjectArc<MyPairObj>);
///
/// unsafe impl ObjectRefCore for PositivePair {
///     type ContainerType = MyPairObj;
///     fn data(this: &Self) -> &ObjectArc<MyPairObj> {
///         &this.0
///     }
///     fn into_data(this: Self) -> ObjectArc<MyPairObj> {
///         this.0
///     }
///     unsafe fn from_data(data: ObjectArc<MyPairObj>) -> Self {
///         Self(data)
///     }
/// }
///
/// unsafe impl ObjectRefAnyCompatible for PositivePair {
///     fn extra_check(pair: &MyPairObj) -> bool {
///         pair.first > 0
///     }
/// }
/// unsafe_impl_object_ref_downcast_ref!(PositivePair);
///
/// let pair = Any::from(MyPair::from_object(MyPairObj::new(Object::new(), 1, 2)));
/// assert!(pair.downcast_ref::<PositivePair>().is_some());
/// ```
#[macro_export]
macro_rules! unsafe_impl_object_ref_downcast_ref {
    ($($name:ty),+ $(,)?) => {
        $(
            // SAFETY: the macro declaration promises that `$name` is one object
            // pointer whose `check_any_strict` is its type check.
            unsafe impl $crate::type_traits::RefFromAnyUnchecked for $name {
                #[inline(always)]
                unsafe fn ref_from_any_unchecked(
                    cell: &$crate::tvm_ffi_sys::TVMFFIAny,
                ) -> &Self {
                    const {
                        assert!(
                            ::core::mem::size_of::<$name>()
                                == ::core::mem::size_of::<*mut $crate::tvm_ffi_sys::TVMFFIObject>()
                                && ::core::mem::align_of::<$name>()
                                    == ::core::mem::align_of::<*mut $crate::tvm_ffi_sys::TVMFFIObject>(),
                            "a reference downcast target must be exactly one object pointer"
                        )
                    };
                    unsafe { &*(::core::ptr::addr_of!(cell.data_union) as *const Self) }
                }
            }

            unsafe impl $crate::type_traits::DowncastRefFromAny for $name {
                #[inline(always)]
                unsafe fn downcast_ref_from_any(
                    cell: &$crate::tvm_ffi_sys::TVMFFIAny,
                ) -> ::core::option::Option<&Self> {
                    unsafe { $crate::type_traits::downcast_ref_from_any_in_place::<Self>(cell) }
                }
            }

            unsafe impl $crate::type_traits::DowncastRefFromObject for $name {
                #[inline(always)]
                unsafe fn downcast_ref_from_object(
                    place: &*mut $crate::tvm_ffi_sys::TVMFFIObject,
                ) -> ::core::option::Option<&Self> {
                    unsafe { $crate::type_traits::downcast_ref_from_object_in_place::<Self>(place) }
                }
            }
        )+
    };
}

// ----------------------------------------------------------------------------
// Macros for any definitions
// ----------------------------------------------------------------------------

/// Implement the conversions between each listed [`AnyCompatible`](crate::AnyCompatible)
/// type and FFI values: `TryFrom<AnyView>`, `TryFrom<Any>` and `TryFrom<&Any>`,
/// and use as an argument of [`Function::call_tuple`](crate::Function::call_tuple)
/// and of functions from [`into_typed_fn!`](crate::into_typed_fn).
///
/// `#[derive(ObjectRef)]` calls it; a type that implements `AnyCompatible` by hand
/// calls it after that impl. See also
/// [`unsafe_impl_object_ref_downcast_ref!`](crate::unsafe_impl_object_ref_downcast_ref)
/// for an object reference implemented by hand.
#[macro_export]
macro_rules! impl_any_compatible_conversions {
    ($($t:ty),* $(,)?) => {
        $crate::impl_try_from_any!($($t),*);
        $crate::impl_arg_into_ref!($($t),*);
        $crate::impl_into_arg_holder_default!($($t),*);
    };
}

/// `TryFrom` from FFI values; part of [`impl_any_compatible_conversions!`](crate::impl_any_compatible_conversions).
#[doc(hidden)]
#[macro_export]
macro_rules! impl_try_from_any {
    ($($t:ty),* $(,)?) => {
        $(
            impl<'a> TryFrom<$crate::any::AnyView<'a>> for $t {
                type Error = $crate::error::Error;
                #[inline(always)]
                fn try_from(
                    value: $crate::any::AnyView<'a>
                ) -> Result<Self, Self::Error> {
                    type TryFromTemp = $crate::any::TryFromTemp<$t>;
                    return TryFromTemp::try_from(value).map(TryFromTemp::into_value);
                }
            }

            impl TryFrom<$crate::any::Any> for $t {
                type Error = $crate::error::Error;
                #[inline(always)]
                fn try_from(
                    value: $crate::any::Any
                ) -> Result<Self, Self::Error> {
                    type TryFromTemp = $crate::any::TryFromTemp<$t>;
                    return TryFromTemp::try_from(value).map(TryFromTemp::into_value);
                }
            }

            impl<'a> TryFrom<&'a $crate::any::Any> for $t {
                type Error = $crate::error::Error;
                #[inline(always)]
                fn try_from(
                    value: &'a $crate::any::Any
                ) -> Result<Self, Self::Error> {
                    type TryFromTemp = $crate::any::TryFromTemp<$t>;
                    return TryFromTemp::try_from(value).map(TryFromTemp::into_value);
                }
            }
        )*
    };
}

/// Macro to implement `TryFrom<AnyView>` and `TryFrom<Any>` for generic types like `Option<T>`
#[doc(hidden)]
#[macro_export]
macro_rules! impl_try_from_any_for_parametric {
    ($generic_type:ident<$param:ident>) => {
        impl<'a, $param: AnyCompatible> TryFrom<$crate::any::AnyView<'a>>
            for $generic_type<$param>
        {
            type Error = $crate::error::Error;
            #[inline(always)]
            fn try_from(value: $crate::any::AnyView<'a>) -> Result<Self, Self::Error> {
                type TryFromTemp<T> = $crate::any::TryFromTemp<$generic_type<$param>>;
                return TryFromTemp::<T>::try_from(value).map(TryFromTemp::<T>::into_value);
            }
        }

        impl<$param: AnyCompatible> TryFrom<$crate::any::Any> for $generic_type<$param> {
            type Error = $crate::error::Error;
            #[inline(always)]
            fn try_from(value: $crate::any::Any) -> Result<Self, Self::Error> {
                type TryFromTemp<T> = $crate::any::TryFromTemp<$generic_type<$param>>;
                return TryFromTemp::<T>::try_from(value).map(TryFromTemp::<T>::into_value);
            }
        }

        impl<'a, $param: AnyCompatible> TryFrom<&'a $crate::any::Any> for $generic_type<$param> {
            type Error = $crate::error::Error;
            #[inline(always)]
            fn try_from(value: &'a $crate::any::Any) -> Result<Self, Self::Error> {
                type TryFromTemp<T> = $crate::any::TryFromTemp<$generic_type<$param>>;
                return TryFromTemp::<T>::try_from(value).map(TryFromTemp::<T>::into_value);
            }
        }
    };
}

/// Typed-function arguments; part of [`impl_any_compatible_conversions!`](crate::impl_any_compatible_conversions).
#[doc(hidden)]
#[macro_export]
macro_rules! impl_into_arg_holder_default {
    ($($t:ty),*) => {
        $(
            impl $crate::function_internal::IntoArgHolder for $t {
                type Target = $t;
                fn into_arg_holder(self) -> Self::Target {
                    self
                }
            }
            impl<'a> $crate::function_internal::IntoArgHolder for &'a $t {
                type Target = &'a $t;
                fn into_arg_holder(self) -> Self::Target {
                    self
                }
            }
        )*
    };
}

/// `call_tuple` arguments; part of [`impl_any_compatible_conversions!`](crate::impl_any_compatible_conversions).
#[doc(hidden)]
#[macro_export]
macro_rules! impl_arg_into_ref {
    ($($t:ty),*) => {
        $(
            impl $crate::function_internal::ArgIntoRef for $t {
                type Target = $t;
                fn to_ref(&self) -> &Self::Target {
                    &self
                }
            }
            impl<'a> $crate::function_internal::ArgIntoRef for &'a $t {
                type Target = $t;
                fn to_ref(&self) -> &Self::Target {
                    &self
                }
            }
        )*
    }
}

// ----------------------------------------------------------------------------
// Macros for function definitions
// ----------------------------------------------------------------------------

/// Macro to export a typed function as a C symbol that follows the tvm-ffi ABI
///
/// An error the function returns is raised to the caller. Report errors by
/// returning them; a panic is also caught and raised, as an `InternalError`,
/// since unwinding into the caller would abort the process, but panicking
/// is discouraged.
///
/// With the `export-metadata` feature, the counterpart of C++
/// `TVM_FFI_DLL_EXPORT_INCLUDE_METADATA`, the macro also exports the
/// function's metadata, `{"type_schema":...}`, as `__tvm_ffi__metadata_<name>`,
/// which [`Module::get_function_metadata`](crate::Module::get_function_metadata)
/// and the stub generator read. A
/// function's schema lists its parameter and return types; a callable whose
/// signature is not known, such as a custom [`AsPackedCallable`], has the
/// schema of an untyped function. The feature is off by default, so exports
/// are minimal.
///
/// [`AsPackedCallable`]: crate::function_internal::AsPackedCallable
///
/// # Arguments
/// * `$name` - The name of the function
/// * `$func` - The function to export
///
/// # Example
/// ```rust
/// use tvm_ffi::*;
///
/// fn add_one(x: i32) -> Result<i32> { Ok(x + 1) }
///
/// tvm_ffi_dll_export_typed_func!(add_one, add_one);
/// ```
#[macro_export]
macro_rules! tvm_ffi_dll_export_typed_func {
    ($name:ident, $func:expr) => {
        $crate::macros::paste::paste! {
            // `no_mangle` keeps the symbol in a `cdylib` under the
            // `__tvm_ffi_<name>` name that `Module` looks up.
            //
            // The path-qualified `$crate::tvm_ffi_sys::…` reference (rather
            // than a bare `tvm_ffi_sys::…`) lets downstream crates use the
            // macro without having to add `tvm-ffi-sys` to their own
            // `[dependencies]`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<__tvm_ffi_ $name>](
                _handle: *mut std::ffi::c_void,
                args: *const $crate::tvm_ffi_sys::TVMFFIAny,
                num_args: i32,
                result: *mut $crate::tvm_ffi_sys::TVMFFIAny,
            ) -> i32 {
                let packed_args =
                    std::slice::from_raw_parts(args as *const $crate::any::AnyView, num_args as usize);
                let ret_value = match ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(|| {
                    $crate::function_internal::call_packed_callable($func, packed_args)
                })) {
                    Ok(ret_value) => ret_value,
                    Err(payload) => Err($crate::function_internal::panic_to_error(payload)),
                };
                match ret_value {
                    Ok(value) => {
                        *result = $crate::any::Any::into_raw_ffi_any(value);
                        0
                    }
                    Err(error) => {
                        $crate::error::Error::set_raised(&error);
                        -1
                    }
                }
            }
        }
        $crate::__tvm_ffi_dll_export_metadata!($name, $func);
    };
}

/// Macro to export the doc string of a function exported with
/// [`tvm_ffi_dll_export_typed_func!`], as C++ `TVM_FFI_DLL_EXPORT_TYPED_FUNC_DOC`
/// does
///
/// With the `export-metadata` feature, the doc string is exported as
/// `__tvm_ffi__doc_<name>`, which
/// [`Module::get_function_doc`](crate::Module::get_function_doc) and the stub
/// generator read. Without it, the macro exports nothing, but
/// still checks that the doc string is a string constant. The macro never
/// exports the function itself.
///
/// # Arguments
/// * `$name` - The name the function is exported as
/// * `$doc` - The doc string, a string literal
///
/// # Example
/// ```rust
/// use tvm_ffi::*;
///
/// fn add(a: i64, b: i64) -> Result<i64> { Ok(a + b) }
///
/// tvm_ffi_dll_export_typed_func!(add, add);
/// tvm_ffi_dll_export_typed_func_doc!(add, "Add two integers and return the sum.");
/// ```
#[macro_export]
macro_rules! tvm_ffi_dll_export_typed_func_doc {
    ($name:ident, $doc:expr) => {
        $crate::__tvm_ffi_dll_export_doc!($name, $doc);
    };
}

// The metadata and doc getters the export macros emit with the
// `export-metadata` feature. A `#[cfg]` in a `macro_rules!` expansion would
// test the features of the crate that invokes the macro, so the feature
// selects which definition of these helpers this crate compiles instead.

#[cfg(feature = "export-metadata")]
#[doc(hidden)]
#[macro_export]
macro_rules! __tvm_ffi_dll_export_metadata {
    ($name:ident, $func:expr) => {
        $crate::macros::paste::paste! {
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<__tvm_ffi__metadata_ $name>](
                _handle: *mut std::ffi::c_void,
                _args: *const $crate::tvm_ffi_sys::TVMFFIAny,
                _num_args: i32,
                result: *mut $crate::tvm_ffi_sys::TVMFFIAny,
            ) -> i32 {
                let metadata = match ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(|| {
                    $crate::function_internal::exported_metadata(&$func)
                })) {
                    Ok(metadata) => metadata,
                    Err(payload) => Err($crate::function_internal::panic_to_error(payload)),
                };
                $crate::function_internal::write_exported_str(metadata, result)
            }
        }
    };
}

#[cfg(not(feature = "export-metadata"))]
#[doc(hidden)]
#[macro_export]
macro_rules! __tvm_ffi_dll_export_metadata {
    ($name:ident, $func:expr) => {};
}

#[cfg(feature = "export-metadata")]
#[doc(hidden)]
#[macro_export]
macro_rules! __tvm_ffi_dll_export_doc {
    ($name:ident, $doc:expr) => {
        $crate::macros::paste::paste! {
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<__tvm_ffi__doc_ $name>](
                _handle: *mut std::ffi::c_void,
                _args: *const $crate::tvm_ffi_sys::TVMFFIAny,
                _num_args: i32,
                result: *mut $crate::tvm_ffi_sys::TVMFFIAny,
            ) -> i32 {
                const DOC: &str = $doc;
                $crate::function_internal::write_exported_str(Ok(DOC.to_string()), result)
            }
        }
    };
}

#[cfg(not(feature = "export-metadata"))]
#[doc(hidden)]
#[macro_export]
macro_rules! __tvm_ffi_dll_export_doc {
    ($name:ident, $doc:expr) => {
        const _: &str = $doc;
    };
}

///-----------------------------------------------------------
/// into_typed_fn
///
/// Wrap a [`Function`](crate::Function) as a typed closure with the given
/// signature: `into_typed_fn!(func, Fn(A, B) -> Result<R>)`.
///
/// The closure converts its arguments and the result, and checks the argument
/// count at compile time; it takes 0 to 8 arguments. An argument may be taken
/// by value or by reference; prefer a reference for an object reference such as
/// a `Tensor`, which the caller then keeps.
///
/// # Examples
///
/// ```
/// use tvm_ffi::*;
///
/// let func = Function::from_typed(|x: i32, y: i32| -> Result<i32> { Ok(x + y) });
/// let add = into_typed_fn!(func, Fn(i32, &i32) -> Result<i32>);
/// assert_eq!(add(10, &20)?, 30);
/// # Ok::<(), tvm_ffi::Error>(())
/// ```
#[macro_export]
macro_rules! into_typed_fn {
    // Case for 0 arguments
    ($f:expr, $trait:ident() -> $ret_ty:ty) => {{
        let _f = $f;
        move || -> $ret_ty {
            _f.call_tuple_with_len::<0, _>(())?
                .try_into()
                .map_err(Into::into)
        }
    }};
    // Case for 1 argument
    ($f:expr, $trait:ident($t0:ty) -> $ret_ty:ty) => {{
        let _f = $f;
        move |a0: $t0| -> $ret_ty {
            use $crate::function_internal::IntoArgHolderTuple;
            let tuple_args = (a0,).into_arg_holder_tuple();
            _f.call_tuple_with_len::<1, _>(tuple_args)?
                .try_into()
                .map_err(Into::into)
        }
    }};
    // Case for 2 arguments
    ($f:expr, $trait:ident($t0:ty, $t1:ty) -> $ret_ty:ty) => {{
        let _f = $f;
        move |a0: $t0, a1: $t1| -> $ret_ty {
            use $crate::function_internal::IntoArgHolderTuple;
            let tuple_args = (a0, a1).into_arg_holder_tuple();
            _f.call_tuple_with_len::<2, _>(tuple_args)?
                .try_into()
                .map_err(Into::into)
        }
    }};
    // Case for 3 arguments
    ($f:expr, $trait:ident($t0:ty, $t1:ty, $t2:ty) -> $ret_ty:ty) => {{
        let _f = $f;
        move |a0: $t0, a1: $t1, a2: $t2| -> $ret_ty {
            use $crate::function_internal::IntoArgHolderTuple;
            let tuple_args = (a0, a1, a2).into_arg_holder_tuple();
            _f.call_tuple_with_len::<3, _>(tuple_args)?
                .try_into()
                .map_err(Into::into)
        }
    }};
    // Case for 4 arguments
    ($f:expr, $trait:ident($t0:ty, $t1:ty, $t2:ty, $t3:ty) -> $ret_ty:ty) => {{
        let _f = $f;
        move |a0: $t0, a1: $t1, a2: $t2, a3: $t3| -> $ret_ty {
            use $crate::function_internal::IntoArgHolderTuple;
            let tuple_args = (a0, a1, a2, a3).into_arg_holder_tuple();
            _f.call_tuple_with_len::<4, _>(tuple_args)?
                .try_into()
                .map_err(Into::into)
        }
    }};
    // Case for 5 arguments
    ($f:expr, $trait:ident($t0:ty, $t1:ty, $t2:ty, $t3:ty, $t4:ty) -> $ret_ty:ty) => {{
        let _f = $f;
        move |a0: $t0, a1: $t1, a2: $t2, a3: $t3, a4: $t4| -> $ret_ty {
            use $crate::function_internal::IntoArgHolderTuple;
            let tuple_args = (a0, a1, a2, a3, a4).into_arg_holder_tuple();
            _f.call_tuple_with_len::<5, _>(tuple_args)?
                .try_into()
                .map_err(Into::into)
        }
    }};
    // Case for 6 arguments
    ($f:expr, $trait:ident($t0:ty, $t1:ty, $t2:ty, $t3:ty, $t4:ty, $t5:ty) -> $ret_ty:ty) => {{
        let _f = $f;
        move |a0: $t0, a1: $t1, a2: $t2, a3: $t3, a4: $t4, a5: $t5| -> $ret_ty {
            use $crate::function_internal::IntoArgHolderTuple;
            let tuple_args = (a0, a1, a2, a3, a4, a5).into_arg_holder_tuple();
            _f.call_tuple_with_len::<6, _>(tuple_args)?
                .try_into()
                .map_err(Into::into)
        }
    }};
    // Case for 7 arguments
    ($f:expr, $trait:ident($t0:ty, $t1:ty, $t2:ty, $t3:ty, $t4:ty, $t5:ty, $t6:ty)
        -> $ret_ty:ty) => {{
        let _f = $f;
        move |a0: $t0, a1: $t1, a2: $t2, a3: $t3, a4: $t4, a5: $t5, a6: $t6| -> $ret_ty {
            use $crate::function_internal::IntoArgHolderTuple;
            let tuple_args = (a0, a1, a2, a3, a4, a5, a6).into_arg_holder_tuple();
            _f.call_tuple_with_len::<7, _>(tuple_args)?
                .try_into()
                .map_err(Into::into)
        }
    }};
    // Case for 8 arguments
    ($f:expr, $trait:ident($t0:ty, $t1:ty, $t2:ty, $t3:ty, $t4:ty, $t5:ty, $t6:ty, $t7:ty)
        -> $ret_ty:ty) => {{
        let _f = $f;
        move |a0: $t0, a1: $t1, a2: $t2, a3: $t3, a4: $t4, a5: $t5, a6: $t6, a7: $t7| -> $ret_ty {
            use $crate::function_internal::IntoArgHolderTuple;
            let tuple_args = (a0, a1, a2, a3, a4, a5, a6, a7).into_arg_holder_tuple();
            _f.call_tuple_with_len::<8, _>(tuple_args)?
                .try_into()
                .map_err(Into::into)
        }
    }};
}
