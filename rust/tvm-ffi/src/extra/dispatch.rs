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

//! Runtime type dispatch indexed by its tuple of arguments.

/// A handler that the structural traversals call with a tuple of arguments.
///
/// See also [`#[dispatch]`](crate::dispatch).
#[diagnostic::on_unimplemented(
    message = "this receiver does not implement Dispatcher for the requested argument tuple",
    note = "use #[dispatch(prefix, (Carrier, ...), Ret)] with compatible typed handlers"
)]
pub trait Dispatcher<Args, Ret> {
    /// Dispatch on `args.0` with the complete argument tuple `args`.
    ///
    /// # Arguments
    ///
    /// * `args` - The argument tuple, whose first element is the value to dispatch on.
    ///
    /// # Returns
    ///
    /// `Some` of the matched handler's result, or `None` when no handler matches.
    fn dispatch_tuple(self, args: Args) -> Ret;
}

use crate::AnyView;

/// Compile-time source metadata for the supported dispatch carriers.
#[doc(hidden)]
#[diagnostic::on_unimplemented(
    message = "the first dispatch argument must be AnyView or MutationView"
)]
pub trait DispatchInput {
    type Source;
}
impl<'a> DispatchInput for AnyView<'a> {
    type Source = crate::match_any_internal::ViewSource<'a>;
}
impl<'a, T> DispatchInput for crate::MutationView<'a, T> {
    type Source = crate::match_any_internal::MutationSource<'a>;
}

// Generated alias bounds must not make recovery depend on a temporary borrow.
#[doc(hidden)]
pub trait DispatchRecovery<S> {
    type Recovery;
}
impl<S, T: crate::type_traits::ContainerElement> DispatchRecovery<S> for T {
    type Recovery = ();
}
impl<'source, 'value> DispatchRecovery<crate::match_any_internal::ViewSource<'source>>
    for AnyView<'value>
{
    type Recovery = ();
}

use std::{borrow::BorrowMut, marker::PhantomData};

/// Inferred selectors for dispatcher normalization.
///
/// Entry points infer one marker from the input. Callback markers select argument
/// matching and a leading argument count; tuple markers contain each link's marker.
/// Normalization stores the input once, before traversal, without cloning it.
#[doc(hidden)]
pub mod marker {
    use std::marker::PhantomData;
    /// Store an owned dispatcher with the exact target argument and result types.
    pub enum FromOwned {}
    /// Store a mutable borrow of an exact-signature dispatcher.
    pub enum FromBorrowed {}
    /// Match a callback's first argument using `Shape`, then pass its first `N`
    /// arguments. `N` ranges from one through four, including the matched value.
    pub struct FromCallback<const N: usize, Shape>(PhantomData<fn() -> Shape>);
    /// Try tuple links in order. `Markers` is their marker tuple, including nested
    /// tuples; `Args` selects how arguments are reborrowed between links.
    pub struct FromTuple<Markers, Args = CopyArgs>(PhantomData<fn() -> (Markers, Args)>);
    /// Replay copyable observer arguments; this does not copy dispatcher storage.
    pub enum CopyArgs {}
    /// Sequentially reborrow a mutable second argument for each tuple link.
    pub enum MutArg {}
    /// Match and convert the first `AnyView` to the FFI value type `T`.
    pub struct Value<T>(PhantomData<fn() -> T>);
    /// Match a borrowed object node `&T` without retaining it, unlike the
    /// by-value conversion selected by [`Value<T>`].
    pub struct BorrowedObject<T>(PhantomData<fn() -> T>);
    /// Pass the first `AnyView` without a runtime type filter.
    pub enum AnyView {}
    /// Match a mutation view for `T`, keeping its in-place mode.
    pub struct Mutation<T = crate::Any>(PhantomData<fn() -> T>);
}

/// Convert a dispatcher, a callback, or a tuple of them, into a [`Dispatcher`]
/// for the argument tuple `Args` and result `Ret`; the traversals take any of
/// these inputs.
///
/// * An owned `D` or a borrowed `&mut D` keeps `D` as the
///   [`Target`](Self::Target), and must have exactly that signature.
/// * A callback takes a leading prefix of one to four of the arguments, and
///   its first parameter selects what it matches, as for a `#[dispatch]`
///   handler: an FFI value, a borrowed object node or `AnyView`, or a typed
///   `MutationView` for mutation. It is passed by value, `&` or `&mut`.
/// * A tuple, possibly nested, tries its links in order. Only an outer `None`
///   moves on to the next link; an error or an inner `None` stops.
///
/// `M` marks the input form and is inferred; when calling `into_dispatcher`
/// directly, give `Args` and `Ret`. See also [`#[dispatch]`](crate::dispatch).
///
/// ```
/// use tvm_ffi::{Any, AnyView, Dispatcher, IntoDispatcher, Result};
/// let callbacks = (|value: i64| Any::from(value + 1), |_value: AnyView<'_>| Any::new());
/// let mut dispatcher = IntoDispatcher::<(AnyView<'_>,), Option<Result<Any>>, _>::into_dispatcher(callbacks);
/// let result: Option<Result<Any>> = dispatcher.dispatch_tuple((AnyView::from(&41_i64),));
/// let result = result.unwrap()?;
/// assert_eq!(i64::try_from(result)?, 42);
/// # Ok::<(), tvm_ffi::Error>(())
/// ```
pub trait IntoDispatcher<Args, Ret, M> {
    /// Actual dispatcher type, independent of its owned or borrowed storage.
    type Target;
    /// Storage normalized once at entry, retaining the target's identity.
    type Output: BorrowMut<Self::Target>;
    /// Convert the input, once, before the traversal starts.
    fn into_dispatcher(self) -> Self::Output;
}

/// Normalized callback or tuple storage; direct dispatchers retain their own type.
#[doc(hidden)]
pub struct DispatchAdapter<F, M> {
    input: F,
    marker: PhantomData<fn() -> M>,
}

impl<D, Args, Ret> IntoDispatcher<Args, Ret, marker::FromOwned> for D
where
    for<'s> &'s mut D: Dispatcher<Args, Ret>,
{
    type Target = D;
    type Output = D;
    #[inline(always)]
    fn into_dispatcher(self) -> Self::Output {
        self
    }
}

impl<'d, D, Args, Ret> IntoDispatcher<Args, Ret, marker::FromBorrowed> for &'d mut D
where
    for<'s> &'s mut D: Dispatcher<Args, Ret>,
{
    type Target = D;
    type Output = &'d mut D;
    #[inline(always)]
    fn into_dispatcher(self) -> Self::Output {
        self
    }
}

impl<F, Args, Ret, const N: usize, Shape> IntoDispatcher<Args, Ret, marker::FromCallback<N, Shape>>
    for F
where
    F: DispatchLink<Args, Ret, marker::FromCallback<N, Shape>>,
{
    type Target = DispatchAdapter<F, marker::FromCallback<N, Shape>>;
    type Output = Self::Target;
    #[inline(always)]
    fn into_dispatcher(self) -> Self::Output {
        DispatchAdapter {
            input: self,
            marker: PhantomData,
        }
    }
}

impl<F, Args, Ret, Markers, Reborrow>
    IntoDispatcher<Args, Ret, marker::FromTuple<Markers, Reborrow>> for F
where
    F: DispatchLink<Args, Ret, marker::FromTuple<Markers, Reborrow>>,
{
    type Target = DispatchAdapter<F, marker::FromTuple<Markers, Reborrow>>;
    type Output = Self::Target;
    #[inline(always)]
    fn into_dispatcher(self) -> Self::Output {
        DispatchAdapter {
            input: self,
            marker: PhantomData,
        }
    }
}

/// Implementation boundary shared by callback and tuple adapters.
#[doc(hidden)]
pub trait DispatchLink<Args, Ret, M> {
    fn dispatch_link(&mut self, args: Args) -> Ret;
}

impl<F, Args, Ret, M> Dispatcher<Args, Ret> for &mut DispatchAdapter<F, M>
where
    F: DispatchLink<Args, Ret, M>,
{
    #[inline(always)]
    fn dispatch_tuple(self, args: Args) -> Ret {
        self.input.dispatch_link(args)
    }
}

impl<D, Args, Ret> DispatchLink<Args, Ret, marker::FromOwned> for D
where
    for<'s> &'s mut D: Dispatcher<Args, Ret>,
{
    #[inline(always)]
    fn dispatch_link(&mut self, args: Args) -> Ret {
        self.dispatch_tuple(args)
    }
}
impl<D, Args, Ret> DispatchLink<Args, Ret, marker::FromBorrowed> for &mut D
where
    for<'s> &'s mut D: Dispatcher<Args, Ret>,
{
    #[inline(always)]
    fn dispatch_link(&mut self, args: Args) -> Ret {
        (&mut **self).dispatch_tuple(args)
    }
}

/// Operation-specific callback result conversion; matching and chaining are shared.
#[doc(hidden)]
pub trait CallbackResult<R> {
    fn callback_result(self) -> R;
}

macro_rules! callback_prefix {
    ($n:literal; ($($A:ident:$a:ident),*); ($($B:ident:$b:ident),*)) => {
        impl<'v, F, T, O, R, $($A,)*>
            DispatchLink<(AnyView<'v>, $($A,)*), Option<R>, marker::FromCallback<$n, marker::Value<T>>> for F
        where F: FnMut(T, $($B,)*) -> O, T: crate::type_traits::AnyCompatible,
              O: CallbackResult<R>,
        {
            #[inline(always)]
            fn dispatch_link(&mut self, (value, $($a,)*): (AnyView<'v>, $($A,)*)) -> Option<R> {
                $(let _ = &$a;)*
                // Strict: a handler matches only its exact type (a `bool` handler
                // never catches an int), so this does not convert.
                let cell = value.as_raw_ffi_any();
                // SAFETY: the view's cell is valid for the call, and the copy runs
                // only after the strict check.
                let typed = unsafe {
                    T::check_any_strict(cell).then(|| T::copy_from_any_view_after_check(cell))
                };
                typed.map(|typed| self(typed, $($b,)*).callback_result())
            }
        }
        impl<'v, F, T, O, R, $($A,)*>
            DispatchLink<(AnyView<'v>, $($A,)*), Option<R>, marker::FromCallback<$n, marker::BorrowedObject<T>>> for F
        where F: for<'node> FnMut(&'node T, $($B,)*) -> O,
              T: crate::type_traits::DowncastRefFromAny + 'static,
              O: CallbackResult<R>,
        {
            #[inline(always)]
            fn dispatch_link(&mut self, (value, $($a,)*): (AnyView<'v>, $($A,)*)) -> Option<R> {
                $(let _ = &$a;)*
                value.downcast_ref::<T>().map(|node| self(node, $($b,)*).callback_result())
            }
        }
        impl<'v, F, O, R, $($A,)*>
            DispatchLink<(AnyView<'v>, $($A,)*), Option<R>, marker::FromCallback<$n, marker::AnyView>> for F
        where F: FnMut(AnyView<'v>, $($B,)*) -> O, O: CallbackResult<R>,
        {
            #[inline(always)]
            fn dispatch_link(&mut self, (value, $($a,)*): (AnyView<'v>, $($A,)*)) -> Option<R> {
                $(let _ = &$a;)*
                Some(self(value, $($b,)*).callback_result())
            }
        }
    };
}
callback_prefix!(1; (); ());
callback_prefix!(1; (A1:a1); ());
callback_prefix!(2; (A1:a1); (A1:a1));
callback_prefix!(1; (A1:a1, A2:a2); ());
callback_prefix!(2; (A1:a1, A2:a2); (A1:a1));
callback_prefix!(3; (A1:a1, A2:a2); (A1:a1, A2:a2));
callback_prefix!(1; (A1:a1, A2:a2, A3:a3); ());
callback_prefix!(2; (A1:a1, A2:a2, A3:a3); (A1:a1));
callback_prefix!(3; (A1:a1, A2:a2, A3:a3); (A1:a1, A2:a2));
callback_prefix!(4; (A1:a1, A2:a2, A3:a3); (A1:a1, A2:a2, A3:a3));

macro_rules! mutation_callback_prefix {
    ($n:literal; ($($A:ident:$a:ident),*); ($($B:ident:$b:ident),*)) => {
        impl<'v, F, T, O, R, $($A,)*>
            DispatchLink<(crate::MutationView<'v>, $($A,)*), Option<R>, marker::FromCallback<$n, marker::Mutation<T>>> for F
        where F: FnMut(crate::MutationView<'v, T>, $($B,)*) -> O,
              T: crate::type_traits::ContainerElement, O: CallbackResult<R>,
        {
            #[inline(always)]
            fn dispatch_link(&mut self, (value, $($a,)*): (crate::MutationView<'v>, $($A,)*)) -> Option<R> {
                $(let _ = &$a;)*
                value.downcast::<T>().ok().map(|typed| self(typed, $($b,)*).callback_result())
            }
        }
    };
}
mutation_callback_prefix!(1; (); ());
mutation_callback_prefix!(1; (A1:a1); ());
mutation_callback_prefix!(2; (A1:a1); (A1:a1));
mutation_callback_prefix!(1; (A1:a1, A2:a2); ());
mutation_callback_prefix!(2; (A1:a1, A2:a2); (A1:a1));
mutation_callback_prefix!(3; (A1:a1, A2:a2); (A1:a1, A2:a2));
mutation_callback_prefix!(1; (A1:a1, A2:a2, A3:a3); ());
mutation_callback_prefix!(2; (A1:a1, A2:a2, A3:a3); (A1:a1));
mutation_callback_prefix!(3; (A1:a1, A2:a2, A3:a3); (A1:a1, A2:a2));
mutation_callback_prefix!(4; (A1:a1, A2:a2, A3:a3); (A1:a1, A2:a2, A3:a3));

// Concrete tuple storage bounds keep trait inference from recursively inventing
// unknown adapter receiver layers. Tuple arity is independent of callback arity.
macro_rules! dispatch_tuple_links {
    ($(($F:ident, $M:ident, $idx:tt)),+) => {
        impl<Args: Copy, R, $($F, $M,)+>
            DispatchLink<Args, Option<R>, marker::FromTuple<($($M,)+)>> for ($($F,)+)
        where $($F: DispatchLink<Args, Option<R>, $M>,)+
        {
            #[inline(always)]
            fn dispatch_link(&mut self, args: Args) -> Option<R> {
                $(if let Some(result) = self.$idx.dispatch_link(args) { return Some(result); })+
                None
            }
        }
        impl<'v, 'c, C, R, $($F, $M,)+>
            DispatchLink<(AnyView<'v>, &'c mut C), Option<R>, marker::FromTuple<($($M,)+), marker::MutArg>> for ($($F,)+)
        where $(for<'borrow> $F: DispatchLink<(AnyView<'v>, &'borrow mut C), Option<R>, $M>,)+
        {
            #[inline(always)]
            fn dispatch_link(&mut self, (value, context): (AnyView<'v>, &'c mut C)) -> Option<R> {
                $(if let Some(result) = self.$idx.dispatch_link((value, &mut *context)) { return Some(result); })+
                None
            }
        }
        impl<'v, 'c, C, R, $($F, $M,)+>
            DispatchLink<(crate::MutationView<'v>, &'c mut C), Option<R>, marker::FromTuple<($($M,)+), marker::MutArg>> for ($($F,)+)
        where $(for<'value, 'borrow> $F: DispatchLink<(crate::MutationView<'value>, &'borrow mut C), Option<R>, $M>,)+
        {
            #[inline(always)]
            fn dispatch_link(&mut self, (mut value, context): (crate::MutationView<'v>, &'c mut C)) -> Option<R> {
                $(if let Some(result) = self.$idx.dispatch_link((value.reborrow(), &mut *context)) { return Some(result); })+
                None
            }
        }
    };
}
// Generate the tuple arities supported by the standard library (1 through 12).
macro_rules! impl_callback_chain_tuple_arities {
    ($impl_chain:ident) => {
        impl_callback_chain_tuple_arities!(
            @prefixes $impl_chain;
            [];
            (F0, M0, 0),
            (F1, M1, 1),
            (F2, M2, 2),
            (F3, M3, 3),
            (F4, M4, 4),
            (F5, M5, 5),
            (F6, M6, 6),
            (F7, M7, 7),
            (F8, M8, 8),
            (F9, M9, 9),
            (F10, M10, 10),
            (F11, M11, 11)
        );
    };
    (@prefixes $impl_chain:ident; [$($prefix:tt)*];) => {};
    (
        @prefixes $impl_chain:ident;
        [$($prefix:tt)*];
        $next:tt $(, $rest:tt)*
    ) => {
        $impl_chain!($($prefix)* $next);
        impl_callback_chain_tuple_arities!(
            @prefixes $impl_chain;
            [$($prefix)* $next,];
            $($rest),*
        );
    };
}

impl_callback_chain_tuple_arities!(dispatch_tuple_links);
