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
//! Typed hooks that registries hold: native functions with a packed fallback.

use std::ffi::c_void;
use std::marker::PhantomData;

use crate::object::type_key_of_index;
use crate::tvm_ffi_sys::{TVMFFIAny, TVMFFITypeIndex};
use crate::type_traits::DowncastRefFromObject;
use crate::{Any, AnyView, Error, Function, Result, TYPE_ERROR};

/// A typed hook that a type attribute, an enum attribute or another registry
/// slot holds: a native function, called directly through the C ABI, or a
/// packed [`Function`] registered in its place.
///
/// `Args` is the tuple of argument types and `R` the result type, as in
/// `Fn(Args) -> R`: a hook that checks a pair is a
/// `NativeFunction<(&MyPairObj,)>`, and one that measures it a
/// `NativeFunction<(&MyPairObj,), i64>`. An argument is a borrowed object
/// (`&MyPairObj`), a borrowed managed reference (`&Array<T>`) or `&Any`, and a
/// native function receives its address, which C++ declares as
/// `const MyPairObj*`, `const Array<T>&` or `const Any&`.
///
/// A native function carries no signature at run time, so the slot declares
/// it. Reading a hook with [`from_any`](Self::from_any) is `unsafe`, the
/// caller vouching for the signature, and then calling it is safe. Building a
/// hook from a Rust function with `into()` is safe; store it, through
/// `Any::from`, only in a slot whose callers use exactly this signature.
///
/// # Examples
///
/// ```
/// use tvm_ffi::derive::Object;
/// use tvm_ffi::reflection::{NativeFunction, TypeAttrColumn, TypeAttrDef};
/// use tvm_ffi::{Object, ObjectArc, ObjectCore, Result};
///
/// #[repr(C)]
/// #[derive(Object)]
/// #[type_key = "my_ext.MyPair"]
/// #[type_register]
/// pub struct MyPairObj {
///     base_: Object,
///     first: i64,
///     second: i64,
/// }
///
/// fn sum(pair: &MyPairObj) -> Result<i64> {
///     Ok(pair.first + pair.second)
/// }
///
/// let hook: NativeFunction<(&MyPairObj,), i64> = sum.into();
/// TypeAttrDef::<MyPairObj>::new().attr("my_ext.sum", hook)?;
///
/// let stored = TypeAttrColumn::new("my_ext.sum").unwrap().get(MyPairObj::type_index()).unwrap();
/// // SAFETY: `my_ext.sum` holds `i64(const MyPairObj*)` hooks.
/// let sum: NativeFunction<(&MyPairObj,), i64> = unsafe { NativeFunction::from_any(stored)? };
/// let pair = ObjectArc::new(MyPairObj { base_: Object::new(), first: 1, second: 2 });
/// assert_eq!(sum.call((&*pair,))?, 3);
/// # Ok::<(), tvm_ffi::Error>(())
/// ```
pub struct NativeFunction<Args, R = ()> {
    // A non-null opaque pointer to the native function, or a `Function`.
    data: Any,
    // Covariant in `Args`: a hook that takes `&'a T` takes any shorter borrow.
    signature: PhantomData<fn() -> (Args, R)>,
}

impl<Args: NativeArgs, R: TryFrom<Any>> NativeFunction<Args, R> {
    /// Call the hook with `args`.
    ///
    /// # Arguments
    ///
    /// * `args` - The arguments, as a tuple such as `(&array,)`.
    ///
    /// # Returns
    ///
    /// The hook's result.
    ///
    /// # Errors
    ///
    /// Returns the error the hook raises, or a `TypeError` when a packed
    /// function returns a value that does not convert to `R`.
    ///
    /// # Panics
    ///
    /// Panics if an object argument is not managed, such as an object built
    /// on the stack: the hook may retain it.
    ///
    /// See also [`NativeFunction`].
    pub fn call(&self, args: Args) -> Result<R> {
        let raw = self.data.as_raw_ffi_any();
        let result = if raw.type_index == TVMFFITypeIndex::kTVMFFIOpaquePtr as i32 {
            // SAFETY: `from_any`'s caller vouches that the native function
            // takes `Args` and returns `R`.
            unsafe { args.call_native(raw.data_union.v_ptr) }
        } else {
            let function = self.data.downcast_ref::<Function>();
            args.call_packed(function.expect("a hook is a native or packed function"))?
        };
        // A native function returns its error as its result.
        if result.type_index() == TVMFFITypeIndex::kTVMFFIError as i32 {
            return Err(result.try_into()?);
        }
        R::try_from(result)
            .map_err(|_| Error::new(TYPE_ERROR, "Packed hook result type mismatch", ""))
    }
}

impl<Args, R> Clone for NativeFunction<Args, R> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            signature: PhantomData,
        }
    }
}

impl<Args, R> NativeFunction<Args, R> {
    /// Read the hook that `value` holds: a non-null opaque pointer to a native
    /// function, or a packed [`Function`].
    ///
    /// # Arguments
    ///
    /// * `value` - The hook, as its registry slot holds it.
    ///
    /// # Returns
    ///
    /// The hook, typed with `Args` and `R`.
    ///
    /// # Errors
    ///
    /// Returns a `TypeError` if `value` is neither a non-null opaque pointer
    /// nor a [`Function`].
    ///
    /// # Safety
    ///
    /// A native function in `value` must take `Args` and return `R`, as the
    /// code that registers it in the slot declares. A packed [`Function`] is
    /// called through checked conversions and has no such requirement.
    ///
    /// See also [`NativeFunction`].
    pub unsafe fn from_any(value: Any) -> Result<Self> {
        let raw = value.as_raw_ffi_any();
        // SAFETY: `v_ptr` is the payload of an opaque pointer.
        let native = raw.type_index == TVMFFITypeIndex::kTVMFFIOpaquePtr as i32
            && !unsafe { raw.data_union.v_ptr }.is_null();
        if !native && raw.type_index != TVMFFITypeIndex::kTVMFFIFunction as i32 {
            let message = format!(
                "Cannot convert from type `{}` to `NativeFunction`",
                type_key_of_index(raw.type_index)
            );
            return Err(Error::new(TYPE_ERROR, &message, ""));
        }
        Ok(Self {
            data: value,
            signature: PhantomData,
        })
    }
}

/// Stores the hook in a registry slot, such as with
/// [`TypeAttrDef::attr`](super::TypeAttrDef::attr): the opaque pointer of a
/// native function, or the packed [`Function`]. Store it only in a slot whose
/// callers call it with exactly `Args` and `R`, which native callers do not
/// check.
impl<Args, R> From<NativeFunction<Args, R>> for Any {
    fn from(hook: NativeFunction<Args, R>) -> Self {
        hook.data
    }
}

/// The argument tuples a [`NativeFunction`] takes: up to eight borrowed
/// objects, managed references or [`Any`] values.
///
/// This trait is sealed.
pub trait NativeArgs: sealed::NativeArgs {}

impl<T: sealed::NativeArgs> NativeArgs for T {}

mod sealed {
    use super::*;

    pub trait NativeArgs {
        /// Call the native function `function` with these arguments.
        ///
        /// # Safety
        ///
        /// `function` is an `extern "C" fn(Args...) -> TVMFFIAny`.
        unsafe fn call_native(self, function: *mut c_void) -> Any;

        fn call_packed(self, function: &Function) -> Result<Any>;
    }

    /// An argument passed to a native function as itself: a pointer.
    pub trait NativeArg: Copy {
        /// The pointee, which a native function receives the address of.
        type Target: 'static;

        /// The argument as a packed function receives it.
        fn to_any_view(&self) -> AnyView<'_>;
    }

    impl<T: DowncastRefFromObject + 'static> NativeArg for &T {
        type Target = T;

        #[inline(always)]
        fn to_any_view(&self) -> AnyView<'_> {
            // Panics on an unmanaged object.
            AnyView::from_object(self.as_object())
        }
    }

    impl NativeArg for &Any {
        type Target = Any;

        #[inline(always)]
        fn to_any_view(&self) -> AnyView<'_> {
            AnyView::from(*self)
        }
    }

    /// The only value of the zero-sized `F`, for the adapter, which cannot
    /// receive it.
    ///
    /// # Safety
    ///
    /// A value of `F` exists. `F` is `Copy`, so it has no drop and may be
    /// duplicated: a zero-sized value that must stay unique, such as a token,
    /// is not `Copy`.
    #[inline(always)]
    unsafe fn conjure<F: Copy>() -> F {
        const {
            assert!(
                size_of::<F>() == 0,
                "a native hook is built from a function or a closure that captures nothing"
            )
        };
        // SAFETY: a zero-sized type has no bytes, so any value is initialized,
        // and duplicating it is a copy (see above).
        unsafe { std::mem::zeroed() }
    }

    /// The result of a native function as its caller owns it: the value, or
    /// the error.
    #[inline(always)]
    fn into_raw<R: Into<Any>>(result: Result<R>) -> TVMFFIAny {
        Any::into_raw_ffi_any(match result {
            Ok(value) => value.into(),
            Err(error) => Any::from(error),
        })
    }

    macro_rules! impl_native_args {
        ($($arg:ident $index:tt),*) => {
            impl<$($arg: NativeArg),*> NativeArgs for ($($arg,)*) {
                #[inline(always)]
                unsafe fn call_native(self, function: *mut c_void) -> Any {
                    // The function may retain an object argument, which must be managed.
                    $(let _ = self.$index.to_any_view();)*
                    // SAFETY: the caller passes a function of this signature,
                    // which returns an owned value.
                    unsafe {
                        let function = std::mem::transmute::<
                            *mut c_void,
                            unsafe extern "C" fn($($arg),*) -> TVMFFIAny,
                        >(function);
                        Any::from_raw_ffi_any(function($(self.$index),*))
                    }
                }

                #[inline(always)]
                fn call_packed(self, function: &Function) -> Result<Any> {
                    function.call_packed(&[$(self.$index.to_any_view()),*])
                }
            }
        };
    }

    macro_rules! impl_native_fn {
        ($($arg:ident $raw:ident),*) => {
            /// Build a native hook from a function item or a closure that
            /// captures nothing, with the C ABI adapter that native callers
            /// call.
            ///
            /// The function takes the arguments as `call` passes them and
            /// returns `Result<R>`; its error is the error the hook raises. A
            /// hook reports failure through its `Result` and must not panic: a
            /// panic aborts the process. A closure that captures a value does
            /// not compile.
            ///
            /// See also [`NativeFunction`].
            // The function takes borrows of any lifetime, so it cannot keep an
            // argument past the call.
            impl<F, $($arg: NativeArg,)* R: Into<Any>> From<F> for NativeFunction<($($arg,)*), R>
            where
                F: for<'a> Fn($(&'a $arg::Target),*) -> Result<R> + Copy + 'static,
            {
                fn from(_function: F) -> Self {
                    extern "C" fn adapter<F, $($arg: NativeArg,)* R: Into<Any>>(
                        $($raw: *const $arg::Target),*
                    ) -> TVMFFIAny
                    where
                        F: for<'a> Fn($(&'a $arg::Target),*) -> Result<R> + Copy + 'static,
                    {
                        // SAFETY: `from` received a value of `F`.
                        let function = unsafe { conjure::<F>() };
                        // SAFETY: a native caller passes the address of a live
                        // argument for the duration of the call, or null.
                        $(let Some($raw) = (unsafe { $raw.as_ref() }) else {
                            let message = "A native hook argument is null";
                            return into_raw::<R>(Err(Error::new(TYPE_ERROR, message, "")));
                        };)*
                        // A panic aborts here rather than unwind into the caller.
                        into_raw(function($($raw),*))
                    }
                    let adapter = adapter::<F, $($arg,)* R> as *mut c_void;
                    Self { data: Any::from(adapter), signature: PhantomData }
                }
            }
        };
    }

    impl_native_fn!();
    impl_native_fn!(A0 a0);
    impl_native_fn!(A0 a0, A1 a1);
    impl_native_fn!(A0 a0, A1 a1, A2 a2);
    impl_native_fn!(A0 a0, A1 a1, A2 a2, A3 a3);
    impl_native_fn!(A0 a0, A1 a1, A2 a2, A3 a3, A4 a4);
    impl_native_fn!(A0 a0, A1 a1, A2 a2, A3 a3, A4 a4, A5 a5);
    impl_native_fn!(A0 a0, A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6);
    impl_native_fn!(A0 a0, A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6, A7 a7);

    impl_native_args!();
    impl_native_args!(A0 0);
    impl_native_args!(A0 0, A1 1);
    impl_native_args!(A0 0, A1 1, A2 2);
    impl_native_args!(A0 0, A1 1, A2 2, A3 3);
    impl_native_args!(A0 0, A1 1, A2 2, A3 3, A4 4);
    impl_native_args!(A0 0, A1 1, A2 2, A3 3, A4 4, A5 5);
    impl_native_args!(A0 0, A1 1, A2 2, A3 3, A4 4, A5 5, A6 6);
    impl_native_args!(A0 0, A1 1, A2 2, A3 3, A4 4, A5 5, A6 6, A7 7);
}
