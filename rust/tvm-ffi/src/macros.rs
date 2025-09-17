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
// rexport paste under macro namespace so downstream do not need to specify dep
pub use paste;
// ----------------------------------------------------------------------------
// Macros for error handling
// ----------------------------------------------------------------------------

/// Macro gto get the name of the function
///
/// # Usage
/// Usage: function_name!()
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

/// Check the return code of the safe call
///
/// # Arguments
/// * `ret_code` - The return code of the safe call
///
/// # Returns
/// * `Result<(), Error>` - The result of the safe call
/// Macro to check safe calls and automatically update traceback with file/line info
///
/// Usage: check_safe_call!(function(args))?;
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

/// Create a new error with file/line info attached
///
/// This macro automatically appends file/line info to the traceback
///
/// # Arguments
/// * `error_kind` - The kind of the error
/// * `msg` - The message of the error
/// * `args` - The posisble format arguments
///
/// # Returns
/// * `Result<(), Error>` - The result of the safe call
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

/// Create a new error with file/line info attached
///
/// This macro automatically appends file/line info to the traceback
///
/// # Arguments
/// * `kind` - The kind of the error
/// * `msg` - The message of the error
/// * `args` - The posisble format arguments
///
/// # Returns
/// * `Result<(), Error>` - The result of the safe call
#[macro_export]
macro_rules! ensure {
    ($cond:expr, $error_kind:expr, $fmt:expr $(, $args:expr)* $(,)?) => {{
        if !$cond {
            crate::bail!($error_kind, $fmt $(, $args)*);
        }
    }};
}

/// Attach a context to a result if it is error
///
/// This macro automatically appends file/line info to the traceback
///
/// # Arguments
/// * `error` - The error to attach the context to
/// * `msg` - The message of the error
///
/// # Returns
/// * `Result<(), Error>` - The result of the safe call
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
                Err(Error::with_appended_backtrace(error, &context))
            }
        }
    }};
}

// ----------------------------------------------------------------------------
// Macros for any definitions
// ----------------------------------------------------------------------------

// implements try from any for all integer types
/// Macro to implement `TryFrom<AnyView>` and `TryFrom<Any>` for a list of types
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
        )*
    };
}

/// Macro to implement `TryFrom<AnyView>` and `TryFrom<Any>` for generic types like `Option<T>`
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
    };
}

/// Macro to implement IntoArgHolder for a list of types
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

/// Macro to implement ArgIntoRef for a list of types
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
            pub unsafe extern "C" fn [<__tvm_ffi_ $name>](
                _handle: *mut std::ffi::c_void,
                args: *const tvm_ffi_sys::TVMFFIAny,
                num_args: i32,
                result: *mut tvm_ffi_sys::TVMFFIAny,
            ) -> i32 {
                let packed_args =
                    std::slice::from_raw_parts(args as *const $crate::any::AnyView, num_args as usize);
                let ret_value = $crate::function_internal::call_packed_callable($func, packed_args);
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
    };
}

///-----------------------------------------------------------
/// into_typed_fn
///
/// Converts a generic `Function` into a typed function with compile-time
/// argument count and type checking. This macro provides a convenient way
/// to create type-safe wrappers around TVM functions.
///
/// # Arguments
/// * `$f` - The function identifier to convert
/// * `$trait` - The trait type (typically `Fn`)
/// * `($t0, $t1, ...)` - The argument types
/// * `$ret_ty` - The return type
///
/// # Example
/// ```rust
/// use tvm_ffi::*;
///
/// let func = Function::from_typed(|x: i32, y: i32| -> Result<i32> { Ok(x + y) });
/// let typed_func = into_typed_fn!(func, Fn(i32, &i32) -> Result<i32>);
/// let result = typed_func(10, &20).unwrap(); // Returns 30
/// assert_eq!(result, 30);
/// ```
/// Note that the `into_typed_fn!` macro can specify arguments to be passed either
/// by reference or by value in the argument list.
/// We recommend passing by reference for ObjectRef types such as Tensor.
/// Since the ffi mechanism requires us to pass arguments by reference.
///
/// # Supported Argument Counts
/// This macro supports functions with 0 to 8 arguments.
///-----------------------------------------------------------
#[macro_export]
macro_rules! into_typed_fn {
    // Case for 0 arguments
    ($f:expr, $trait:ident() -> $ret_ty:ty) => {{
        let _f = $f;
        move || -> $ret_ty { Ok(_f.call_tuple_with_len::<0, _>(())?.try_into()?) }
    }};
    // Case for 1 argument
    ($f:expr, $trait:ident($t0:ty) -> $ret_ty:ty) => {{
        let _f = $f;
        move |a0: $t0| -> $ret_ty {
            use $crate::function_internal::IntoArgHolderTuple;
            let tuple_args = (a0,).into_arg_holder_tuple();
            Ok(_f.call_tuple_with_len::<1, _>(tuple_args)?.try_into()?)
        }
    }};
    // Case for 2 arguments
    ($f:expr, $trait:ident($t0:ty, $t1:ty) -> $ret_ty:ty) => {{
        let _f = $f;
        move |a0: $t0, a1: $t1| -> $ret_ty {
            use $crate::function_internal::IntoArgHolderTuple;
            let tuple_args = (a0, a1).into_arg_holder_tuple();
            Ok(_f.call_tuple_with_len::<2, _>(tuple_args)?.try_into()?)
        }
    }};
    // Case for 3 arguments
    ($f:expr, $trait:ident($t0:ty, $t1:ty, $t2:ty) -> $ret_ty:ty) => {{
        let _f = $f;
        move |a0: $t0, a1: $t1, a2: $t2| -> $ret_ty {
            use $crate::function_internal::IntoArgHolderTuple;
            let tuple_args = (a0, a1, a2).into_arg_holder_tuple();
            Ok(_f.call_tuple_with_len::<3, _>(tuple_args)?.try_into()?)
        }
    }};
    // Case for 4 arguments
    ($f:expr, $trait:ident($t0:ty, $t1:ty, $t2:ty, $t3:ty) -> $ret_ty:ty) => {{
        let _f = $f;
        move |a0: $t0, a1: $t1, a2: $t2, a3: $t3| -> $ret_ty {
            use $crate::function_internal::IntoArgHolderTuple;
            let tuple_args = (a0, a1, a2, a3).into_arg_holder_tuple();
            Ok(_f.call_tuple_with_len::<4, _>(tuple_args)?.try_into()?)
        }
    }};
    // Case for 5 arguments
    ($f:expr, $trait:ident($t0:ty, $t1:ty, $t2:ty, $t3:ty, $t4:ty) -> $ret_ty:ty) => {{
        let _f = $f;
        move |a0: $t0, a1: $t1, a2: $t2, a3: $t3, a4: $t4| -> $ret_ty {
            use $crate::function_internal::IntoArgHolderTuple;
            let tuple_args = (a0, a1, a2, a3, a4).into_arg_holder_tuple();
            Ok(_f.call_tuple_with_len::<5, _>(tuple_args)?.try_into()?)
        }
    }};
    // Case for 6 arguments
    ($f:expr, $trait:ident($t0:ty, $t1:ty, $t2:ty, $t3:ty, $t4:ty, $t5:ty) -> $ret_ty:ty) => {{
        let _f = $f;
        move |a0: $t0, a1: $t1, a2: $t2, a3: $t3, a4: $t4, a5: $t5| -> $ret_ty {
            use $crate::function_internal::IntoArgHolderTuple;
            let tuple_args = (a0, a1, a2, a3, a4, a5).into_arg_holder_tuple();
            Ok(_f.call_tuple_with_len::<6, _>(tuple_args)?.try_into()?)
        }
    }};
    // Case for 7 arguments
    ($f:expr, $trait:ident($t0:ty, $t1:ty, $t2:ty, $t3:ty, $t4:ty, $t5:ty, $t6:ty)
        -> $ret_ty:ty) => {{
        let _f = $f;
        move |a0: $t0, a1: $t1, a2: $t2, a3: $t3, a4: $t4, a5: $t5, a6: $t6| -> $ret_ty {
            use $crate::function_internal::IntoArgHolderTuple;
            let tuple_args = (a0, a1, a2, a3, a4, a5, a6).into_arg_holder_tuple();
            Ok(_f.call_tuple_with_len::<7, _>(tuple_args)?.try_into()?)
        }
    }};
    // Case for 8 arguments
    ($f:expr, $trait:ident($t0:ty, $t1:ty, $t2:ty, $t3:ty, $t4:ty, $t5:ty, $t6:ty, $t7:ty)
        -> $ret_ty:ty) => {{
        let _f = $f;
        move |a0: $t0, a1: $t1, a2: $t2, a3: $t3, a4: $t4, a5: $t5, a6: $t6, a7: $t7| -> $ret_ty {
            use $crate::function_internal::IntoArgHolderTuple;
            let tuple_args = (a0, a1, a2, a3, a4, a5, a6, a7).into_arg_holder_tuple();
            Ok(_f.call_tuple_with_len::<8, _>(tuple_args)?.try_into()?)
        }
    }};
}
