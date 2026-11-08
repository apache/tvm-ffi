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
use crate::any::{Any, AnyView, ArgTryFromAnyView};
use crate::error::{Error, Result};
use crate::function::FunctionObj;
use crate::object::{ObjectCore, ObjectRefCore};
use crate::rvalue_ref::RValueRef;
use crate::string::{Bytes, String};
use crate::type_traits::{AnyCompatible, ContainerElement, TypeSchema};
use tvm_ffi_sys::{TVMFFIAny, TVMFFIByteArray, TVMFFIStringFromByteArray};

//------------------------------------------------------------------------
// PackedCallable
//------------------------------------------------------------------------
/// The error to raise for a panic caught at an FFI boundary, where unwinding
/// into the caller would abort the process.
#[doc(hidden)]
pub fn panic_to_error(payload: Box<dyn std::any::Any + Send>) -> Error {
    Error::from_panic(payload)
}

/// The arguments of a packed call from C.
///
/// # Safety
/// If `num_args` is positive, `args` must point to `num_args` valid
/// arguments that outlive `'a`.
#[doc(hidden)]
#[inline(always)]
pub unsafe fn packed_args<'a>(args: *const TVMFFIAny, num_args: i32) -> &'a [AnyView<'a>] {
    if num_args <= 0 {
        // A predicted branch: a select would make the argument loads depend
        // on `num_args`.
        core::hint::cold_path();
        return &[];
    }
    std::slice::from_raw_parts(args as *const AnyView, num_args as usize)
}

pub trait AsPackedCallable<I, O> {
    // Call the function in packed convention
    fn call_packed(&self, packed_args: &[AnyView]) -> Result<Any>;

    /// The type schema of the function, a JSON description of its signature.
    ///
    /// A typed function lists its return and parameter types. The default,
    /// for a callable whose signature is not known, is the schema of an
    /// untyped function.
    fn type_schema() -> std::string::String
    where
        Self: Sized,
    {
        crate::type_traits::type_schema(FunctionObj::TYPE_KEY, &[])
    }
}

/// The type schema of `func`, a value of a callable type.
#[inline]
pub fn type_schema_of<Fun, I, O>(_func: &Fun) -> std::string::String
where
    Fun: AsPackedCallable<I, O>,
{
    Fun::type_schema()
}

/// Writes `text` to `result` as a string allocated by the runtime, so that it
/// outlives the library that wrote it, as the getters that
/// `TVM_FFI_DLL_EXPORT_TYPED_FUNC` and `TVM_FFI_DLL_EXPORT_TYPED_FUNC_DOC`
/// export do. Returns the safe call return code.
///
/// # Safety
///
/// `result` must be valid for writes of one `TVMFFIAny`.
#[doc(hidden)]
pub unsafe fn write_exported_str(text: Result<std::string::String>, result: *mut TVMFFIAny) -> i32 {
    match text {
        Ok(text) => {
            let text = TVMFFIByteArray::from_str(&text);
            TVMFFIStringFromByteArray(&text, result)
        }
        Err(error) => {
            Error::set_raised(&error);
            -1
        }
    }
}

/// The metadata a library exports for a function `func` as
/// `__tvm_ffi__metadata_<name>`, as `TVM_FFI_DLL_EXPORT_TYPED_FUNC` does.
#[doc(hidden)]
pub fn exported_metadata<Fun, I, O>(func: &Fun) -> Result<std::string::String>
where
    Fun: AsPackedCallable<I, O>,
{
    let type_schema = type_schema_of(func);
    Ok(crate::type_traits::type_schema_metadata(&type_schema))
}

#[inline]
pub fn call_packed_callable<Fun, I, O>(func: Fun, packed_args: &[AnyView]) -> Result<Any>
where
    Fun: AsPackedCallable<I, O>,
{
    func.call_packed(packed_args)
}

macro_rules! impl_as_packed_callable {
    ($len:literal; $($t:ident),*) => {
        impl<Fun, $($t,)* Out> AsPackedCallable<($($t,)*), Out> for Fun
        where
            Fun: Fn($($t,)*) -> Result<Out> + 'static,
            Any: From<Out>,
            Out: TypeSchema,
            $($t: ArgTryFromAnyView),*
        {
            fn type_schema() -> std::string::String {
                let params: &[std::string::String] = &[$(<$t as TypeSchema>::type_schema()),*];
                format!(
                    r#"{{"type":"{}","named_args":{{"return":[{}],"params":[{}]}}}}"#,
                    FunctionObj::TYPE_KEY,
                    Out::type_schema(),
                    params.join(","),
                )
            }

            fn call_packed(&self, packed_args: &[AnyView]) -> Result<Any>
            {
                crate::ensure!(
                    packed_args.len() == $len, crate::error::TYPE_ERROR,
                    "Expected {} arguments, got {}", $len, packed_args.len()
                );
                // Expand the function call, consuming the iterator.
                let mut _arg_iter = packed_args.iter().enumerate();
                let ret_value = self(
                    $({
                        // unwrap is safe due to the length check above
                        let (i, view) = _arg_iter.next().unwrap();
                        $t::try_from_any_view(view, i)?
                    }),*
                )?;
                Ok(Any::from(ret_value))
            }
        }
    }
}

impl_as_packed_callable!(0;);
impl_as_packed_callable!(1; T0);
impl_as_packed_callable!(2; T0, T1);
impl_as_packed_callable!(3; T0, T1, T2);
impl_as_packed_callable!(4; T0, T1, T2, T3);
impl_as_packed_callable!(5; T0, T1, T2, T3, T4);
impl_as_packed_callable!(6; T0, T1, T2, T3, T4, T5);
impl_as_packed_callable!(7; T0, T1, T2, T3, T4, T5, T6);
impl_as_packed_callable!(8; T0, T1, T2, T3, T4, T5, T6, T7);

/// One argument of a packed call: held for the call, then viewed as an FFI
/// value. `&str` and `&[u8]` are held as `String` and `Bytes`. Implemented by
/// `#[derive(ObjectRef)]` and `impl_any_compatible_conversions!`.
pub trait PackedCallArg {
    /// What keeps the argument alive during the call.
    type Holder;
    fn into_packed_arg(self) -> Self::Holder;
    fn as_packed_arg(holder: &Self::Holder) -> AnyView<'_>;
}

// `Any` itself is an argument; every other value type is listed in `any.rs`.
crate::impl_into_packed_arg!(Any);

impl PackedCallArg for &str {
    type Holder = String;
    #[inline]
    fn into_packed_arg(self) -> String {
        String::from(self)
    }
    #[inline(always)]
    fn as_packed_arg(holder: &String) -> AnyView<'_> {
        AnyView::from(holder)
    }
}

impl PackedCallArg for &[u8] {
    type Holder = Bytes;
    #[inline]
    fn into_packed_arg(self) -> Bytes {
        Bytes::from(self)
    }
    #[inline(always)]
    fn as_packed_arg(holder: &Bytes) -> AnyView<'_> {
        AnyView::from(holder)
    }
}

impl<T: ObjectRefCore> PackedCallArg for RValueRef<T> {
    type Holder = Self;
    #[inline(always)]
    fn into_packed_arg(self) -> Self {
        self
    }
    #[inline(always)]
    fn as_packed_arg(holder: &Self::Holder) -> AnyView<'_> {
        AnyView::from(holder)
    }
}

// Generic types, which the per-type lists cannot name.
crate::impl_into_packed_arg!(impl[T: AnyCompatible] Option<T>);
crate::impl_into_packed_arg!(impl[T: ContainerElement + Clone] crate::Array<T>);
crate::impl_into_packed_arg!(impl[K: ContainerElement, V: ContainerElement] crate::Map<K, V>);
crate::impl_into_packed_arg!(impl[T: ContainerElement] crate::List<T>);

/// The arguments of a packed call: a tuple of 0 to 8 [`PackedCallArg`] values.
pub trait PackedCallArgs {
    const LEN: usize;
    /// Call `f` with the arguments as views, which live until `f` returns.
    fn with_packed_args<R>(self, f: impl FnOnce(&[AnyView<'_>]) -> R) -> R;
}

macro_rules! impl_into_packed_args {
    ($len:literal; $($T:ident),*; $($idx:tt),*) => {
        impl<$($T: PackedCallArg),*> PackedCallArgs for ($($T,)*) {
            const LEN: usize = $len;
            #[inline(always)]
            fn with_packed_args<R>(self, f: impl FnOnce(&[AnyView<'_>]) -> R) -> R {
                let _holders = ($(self.$idx.into_packed_arg(),)*);
                let views: [AnyView<'_>; $len] = [$($T::as_packed_arg(&_holders.$idx)),*];
                f(&views)
            }
        }
    };
}

impl_into_packed_args!(0;;);
impl_into_packed_args!(1; T0; 0);
impl_into_packed_args!(2; T0, T1; 0, 1);
impl_into_packed_args!(3; T0, T1, T2; 0, 1, 2);
impl_into_packed_args!(4; T0, T1, T2, T3; 0, 1, 2, 3);
impl_into_packed_args!(5; T0, T1, T2, T3, T4; 0, 1, 2, 3, 4);
impl_into_packed_args!(6; T0, T1, T2, T3, T4, T5; 0, 1, 2, 3, 4, 5);
impl_into_packed_args!(7; T0, T1, T2, T3, T4, T5, T6; 0, 1, 2, 3, 4, 5, 6);
impl_into_packed_args!(8; T0, T1, T2, T3, T4, T5, T6, T7; 0, 1, 2, 3, 4, 5, 6, 7);
