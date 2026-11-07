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
/// it, and reading a hook with [`from_any`](Self::from_any) is `unsafe`: the
/// caller vouches for the signature. Calling a hook is safe.
///
/// # Examples
///
/// A frontend registers a packed function where a native hook is expected:
///
/// ```
/// use tvm_ffi::reflection::NativeFunction;
/// use tvm_ffi::{Any, Array, Function, Result};
///
/// let len = Function::from_typed(|xs: Array<i64>| -> Result<i64> { Ok(xs.len() as i64) });
/// // SAFETY: a packed function is called through checked conversions.
/// let hook: NativeFunction<(&Array<i64>,), i64> = unsafe { NativeFunction::from_any(len.into())? };
///
/// let xs = Array::from_vec(vec![1_i64, 2, 3]);
/// assert_eq!(hook.call((&xs,))?, 3);
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
        /// The argument as a packed function receives it.
        fn to_any_view(&self) -> AnyView<'_>;
    }

    impl<T: DowncastRefFromObject> NativeArg for &T {
        #[inline(always)]
        fn to_any_view(&self) -> AnyView<'_> {
            // Panics on an unmanaged object.
            AnyView::from_object(self.as_object())
        }
    }

    impl NativeArg for &Any {
        #[inline(always)]
        fn to_any_view(&self) -> AnyView<'_> {
            AnyView::from(*self)
        }
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
