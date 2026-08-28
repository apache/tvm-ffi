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
use crate::error::Result;
use crate::object::ObjectRefCore;
use crate::rvalue_ref::RValueRef;
use crate::string::{Bytes, String};
use crate::type_traits::{AnyCompatible, ContainerElement};

//------------------------------------------------------------------------
// PackedCallable
//------------------------------------------------------------------------
pub trait AsPackedCallable<I, O> {
    // Call the function in packed convention
    fn call_packed(&self, packed_args: &[AnyView]) -> Result<Any>;
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
            $($t: ArgTryFromAnyView),*
        {
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
    Any,
    crate::DLDataType,
    crate::DLDevice
);

// `Option<T>` uses the same packed representation as `T`, with `None` encoded
// as kTVMFFINone. It needs generic implementations rather than macro entries.
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

// Parametric containers pass by value/reference like the scalars above, but
// their type parameters keep them out of the `impl_*!` macros.
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
