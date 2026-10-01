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
use crate::error::{Error, Result, INTERNAL_ERROR};
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
    let message = payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| {
            payload
                .downcast_ref::<std::string::String>()
                .map(|s| s.as_str())
        })
        .unwrap_or("unknown payload");
    Error::new(INTERNAL_ERROR, &format!("panicked: {message}"), "")
}

pub trait AsPackedCallable<I, O> {
    // Call the function in packed convention
    fn call_packed(&self, packed_args: &[AnyView]) -> Result<Any>;

    /// The type schema of the function, as C++ `FunctionInfo::TypeSchema`
    /// writes it.
    ///
    /// A typed function lists its return and parameter types. The default,
    /// for a callable whose signature is not known, is the schema of an
    /// untyped function, as C++ `refl::GlobalDef().def_packed` records it.
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

/// The metadata C++ records for a function or field of type schema
/// `type_schema`: `{"type_schema":<type_schema as a JSON string>}`.
///
/// The string is escaped by the runtime's JSON writer, which escapes as C++
/// `EscapeStringJSON` does, so the metadata matches the C++ one byte for byte.
pub(crate) fn type_schema_metadata(type_schema: &str) -> Result<std::string::String> {
    let escaped: String = crate::cached_global_func!("ffi.json.Stringify")
        .call_tuple_with_len::<2, _>((String::from(type_schema), ()))?
        .try_into()?;
    Ok(format!(r#"{{"type_schema":{}}}"#, escaped.as_str()))
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
    type_schema_metadata(&type_schema_of(func))
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
                    packed_args.len() == $len, crate::error::VALUE_ERROR,
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

//--------------------------------------------------------------
// IntoArgHolder, helper to convert to canonical holding type
//
// This is needed sometimes for reference types that may need to
// be converted to value types.
//--------------------------------------------------------------
pub trait IntoArgHolder {
    type Target;
    fn into_arg_holder(self) -> Self::Target;
}

crate::impl_into_arg_holder_default!(
    (),
    bool,
    i8,
    i16,
    i32,
    i64,
    isize,
    u8,
    u16,
    u32,
    u64,
    usize,
    f32,
    f64,
    String,
    Bytes,
    crate::big_int::BigInt,
    Any,
    crate::DLDataType,
    crate::DLDevice
);

// string will be converted to String for argument passing
impl IntoArgHolder for &str {
    type Target = String;
    fn into_arg_holder(self) -> Self::Target {
        String::from(self)
    }
}

// string will be converted to String for argument passing
impl IntoArgHolder for &[u8] {
    type Target = Bytes;
    fn into_arg_holder(self) -> Self::Target {
        Bytes::from(self)
    }
}

// helper trait to implement IntoArgHolderTuple to apply into_arg_holder to each element
pub trait IntoArgHolderTuple {
    type Target;
    fn into_arg_holder_tuple(self) -> Self::Target;
}

macro_rules! impl_into_arg_holder_tuple {
    ( $($T:ident),* ; $($idx:tt),* ) => {
        impl<$($T),*> $crate::function_internal::IntoArgHolderTuple for ($($T,)*)
        where
            $($T: IntoArgHolder),* {
            type Target = ($($T::Target,)*);

            fn into_arg_holder_tuple(self) -> Self::Target {
                ($(self.$idx.into_arg_holder(),)*)
            }
        }
    };
}

impl_into_arg_holder_tuple!(;);
impl_into_arg_holder_tuple!(T0; 0);
impl_into_arg_holder_tuple!(T0, T1; 0, 1);
impl_into_arg_holder_tuple!(T0, T1, T2; 0, 1, 2);
impl_into_arg_holder_tuple!(T0, T1, T2, T3; 0, 1, 2, 3);
impl_into_arg_holder_tuple!(T0, T1, T2, T3, T4; 0, 1, 2, 3, 4);
impl_into_arg_holder_tuple!(T0, T1, T2, T3, T4, T5; 0, 1, 2, 3, 4, 5);
impl_into_arg_holder_tuple!(T0, T1, T2, T3, T4, T5, T6; 0, 1, 2, 3, 4, 5, 6);
impl_into_arg_holder_tuple!(T0, T1, T2, T3, T4, T5, T6, T7; 0, 1, 2, 3, 4, 5, 6, 7);

//------------------------------------------------------------
// ArgIntoRef
//
// Helper to turn argument type to reference type
// This is effectively AsRef<T> but removes the need of T
//-----------------------------------------------------------
pub trait ArgIntoRef {
    type Target;
    fn to_ref(&self) -> &Self::Target;
}

/// Convert a canonical argument holder into its packed ABI view.
#[doc(hidden)]
pub trait PackedArg {
    fn as_packed_arg(&self) -> AnyView<'_>;
}

impl<T: AnyCompatible> PackedArg for T {
    #[inline]
    fn as_packed_arg(&self) -> AnyView<'_> {
        AnyView::from(self)
    }
}

impl PackedArg for Any {
    #[inline]
    fn as_packed_arg(&self) -> AnyView<'_> {
        AnyView::from(self)
    }
}

impl<T> PackedArg for RValueRef<T>
where
    T: ObjectRefCore + AnyCompatible,
{
    #[inline]
    fn as_packed_arg(&self) -> AnyView<'_> {
        AnyView::from(self)
    }
}

crate::impl_arg_into_ref!(
    (),
    bool,
    i8,
    i16,
    i32,
    i64,
    isize,
    u8,
    u16,
    u32,
    u64,
    usize,
    f32,
    f64,
    String,
    Bytes,
    crate::big_int::BigInt,
    Any,
    crate::DLDataType,
    crate::DLDevice
);

// Generic holders require explicit implementations rather than scalar macro entries.
impl<T: AnyCompatible> IntoArgHolder for Option<T> {
    type Target = Self;
    fn into_arg_holder(self) -> Self::Target {
        self
    }
}

impl<'a, T: AnyCompatible> IntoArgHolder for &'a Option<T> {
    type Target = &'a Option<T>;
    fn into_arg_holder(self) -> Self::Target {
        self
    }
}

impl<T: AnyCompatible> ArgIntoRef for Option<T> {
    type Target = Self;
    fn to_ref(&self) -> &Self::Target {
        self
    }
}

impl<T: AnyCompatible> ArgIntoRef for &Option<T> {
    type Target = Option<T>;
    fn to_ref(&self) -> &Self::Target {
        self
    }
}

impl<T: ContainerElement + Clone> IntoArgHolder for crate::Array<T> {
    type Target = crate::Array<T>;
    fn into_arg_holder(self) -> Self::Target {
        self
    }
}
impl<'a, T: ContainerElement + Clone> IntoArgHolder for &'a crate::Array<T> {
    type Target = &'a crate::Array<T>;
    fn into_arg_holder(self) -> Self::Target {
        self
    }
}
impl<T: ContainerElement + Clone> ArgIntoRef for crate::Array<T> {
    type Target = crate::Array<T>;
    fn to_ref(&self) -> &Self::Target {
        self
    }
}
impl<T: ContainerElement + Clone> ArgIntoRef for &crate::Array<T> {
    type Target = crate::Array<T>;
    fn to_ref(&self) -> &Self::Target {
        self
    }
}

impl<T> IntoArgHolder for RValueRef<T>
where
    T: ObjectRefCore + AnyCompatible,
{
    type Target = Self;
    fn into_arg_holder(self) -> Self::Target {
        self
    }
}

impl<T> ArgIntoRef for RValueRef<T>
where
    T: ObjectRefCore + AnyCompatible,
{
    type Target = Self;
    fn to_ref(&self) -> &Self::Target {
        self
    }
}

impl<K: ContainerElement, V: ContainerElement> IntoArgHolder for crate::Map<K, V> {
    type Target = crate::Map<K, V>;
    fn into_arg_holder(self) -> Self::Target {
        self
    }
}
impl<'a, K: ContainerElement, V: ContainerElement> IntoArgHolder for &'a crate::Map<K, V> {
    type Target = &'a crate::Map<K, V>;
    fn into_arg_holder(self) -> Self::Target {
        self
    }
}
impl<K: ContainerElement, V: ContainerElement> ArgIntoRef for crate::Map<K, V> {
    type Target = crate::Map<K, V>;
    fn to_ref(&self) -> &Self::Target {
        self
    }
}
impl<K: ContainerElement, V: ContainerElement> ArgIntoRef for &crate::Map<K, V> {
    type Target = crate::Map<K, V>;
    fn to_ref(&self) -> &Self::Target {
        self
    }
}

//-----------------------------------------------------------
// TupleAsPackedArgs
//
// Helper to turn tuple type to packed arguments
//-----------------------------------------------------------
pub trait TupleAsPackedArgs {
    const LEN: usize;
    fn fill_any_view<'a>(&'a self, any_view: &mut [AnyView<'a>]);
}

macro_rules! impl_tuple_as_packed_args {
    ( $len:expr; $($T:ident),* ; $($idx:tt),* ) => {
        impl<$($T),*> TupleAsPackedArgs for ($($T,)*)
        where
            $(
                $T: ArgIntoRef,
                $T::Target: PackedArg,
            )*
        {
            const LEN: usize = $len;

            fn fill_any_view<'a>(&'a self, _any_view: &mut [AnyView<'a>]) {
                $(
                    _any_view[$idx] = self.$idx.to_ref().as_packed_arg();
                )*
            }
        }
    };
}

impl_tuple_as_packed_args!(0;;);
impl_tuple_as_packed_args!(1; T0; 0);
impl_tuple_as_packed_args!(2; T0, T1; 0, 1);
impl_tuple_as_packed_args!(3; T0, T1, T2; 0, 1, 2);
impl_tuple_as_packed_args!(4; T0, T1, T2, T3; 0, 1, 2, 3);
impl_tuple_as_packed_args!(5; T0, T1, T2, T3, T4; 0, 1, 2, 3, 4);
impl_tuple_as_packed_args!(6; T0, T1, T2, T3, T4, T5; 0, 1, 2, 3, 4, 5);
impl_tuple_as_packed_args!(7; T0, T1, T2, T3, T4, T5, T6; 0, 1, 2, 3, 4, 5, 6);
impl_tuple_as_packed_args!(8; T0, T1, T2, T3, T4, T5, T6, T7; 0, 1, 2, 3, 4, 5, 6, 7);
