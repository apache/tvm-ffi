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

// The three ways a downstream crate makes an object reference `AnyCompatible`.

use tvm_ffi::derive::ObjectRef;
use tvm_ffi::object::ObjectArc;
use tvm_ffi::tvm_ffi_sys::TVMFFIAny;
use tvm_ffi::{impl_any_compatible_conversions, match_any, rs_object};
use tvm_ffi::{Any, AnyCompatible, AnyView, Object, ObjectRefCore};

// A derived reference.
#[rs_object("testing.object_ref_any.Pair", final)]
pub struct PairObj {
    base_: Object,
    pub first: i64,
    pub second: i64,
}
#[derive(Clone, ObjectRef)]
pub struct Pair {
    data: ObjectArc<PairObj>,
}

fn pair(first: i64, second: i64) -> Pair {
    Pair::from_object(PairObj::new(Object::new(), first, second))
}

// A typed reference: a `Pair` whose `first` is positive; a positive integer
// `n` converts to `(n, n)`.
#[derive(Clone, ObjectRef)]
#[object_ref(skip_any_compatible)]
pub struct PositivePair {
    data: ObjectArc<PairObj>,
}

unsafe impl tvm_ffi::ObjectRefAnyCompatible for PositivePair {
    fn extra_check(pair: &PairObj) -> bool {
        pair.first > 0
    }

    fn convert_fallback(value: AnyView<'_>) -> Option<Self> {
        let n = i64::try_from(value).ok().filter(|n| *n > 0)?;
        Some(Self::from_object(PairObj::new(Object::new(), n, n)))
    }

    fn custom_type_str() -> String {
        "PositivePair".into()
    }

    fn custom_mismatch_type_info(value: AnyView<'_>) -> String {
        match value.downcast_ref::<Pair>() {
            Some(pair) => format!("Pair(first={})", pair.first),
            None => "non-pair".into(),
        }
    }
}

// A reference with a hand-written `AnyCompatible`, as the containers have.
#[derive(Clone)]
#[repr(transparent)]
pub struct RawPair(Pair);

unsafe impl ObjectRefCore for RawPair {
    type ContainerType = PairObj;
    fn data(this: &Self) -> &ObjectArc<PairObj> {
        Pair::data(&this.0)
    }
    fn into_data(this: Self) -> ObjectArc<PairObj> {
        Pair::into_data(this.0)
    }
    unsafe fn from_data(data: ObjectArc<PairObj>) -> Self {
        Self(Pair::from_data(data))
    }
}

unsafe impl AnyCompatible for RawPair {
    fn type_str() -> String {
        "RawPair".into()
    }
    unsafe fn copy_to_any_view(src: &Self, data: &mut TVMFFIAny) {
        Pair::copy_to_any_view(&src.0, data)
    }
    unsafe fn move_to_any(src: Self, data: &mut TVMFFIAny) {
        Pair::move_to_any(src.0, data)
    }
    unsafe fn check_any_strict(data: &TVMFFIAny) -> bool {
        Pair::check_any_strict(data)
    }
    unsafe fn copy_from_any_view_after_check(data: &TVMFFIAny) -> Self {
        Self(Pair::copy_from_any_view_after_check(data))
    }
    unsafe fn move_from_any_after_check(data: &mut TVMFFIAny) -> Self {
        Self(Pair::move_from_any_after_check(data))
    }
    unsafe fn try_cast_from_any_view(data: &TVMFFIAny) -> Result<Self, ()> {
        Pair::try_cast_from_any_view(data).map(Self)
    }
}

impl_any_compatible_conversions!(RawPair);

// The trait's doc example covers the strict check and the fallback; here, what
// it does not: borrows never fall back, and the mismatch message is custom.
#[test]
fn typed_reference_borrow_never_falls_back_and_names_the_mismatch() {
    assert!(Any::from(3_i64).downcast_ref::<PositivePair>().is_none());
    let error = PositivePair::try_from(Any::from(pair(-1, 2)))
        .err()
        .unwrap();
    assert_eq!(
        error.message(),
        "Cannot convert from type `Pair(first=-1)` to `PositivePair`"
    );
}

// With 20 or more arms, `match_any!` routes final object types by type index,
// which must not skip the typed arm's check.
macro_rules! classify {
    ($value:expr, $($pair:ident)*) => {
        match_any! {
            $value {
                PositivePair(_) => "positive",
                $(Pair($pair) => "pair",)*
                _ => "other",
            }
        }
    };
}

#[test]
fn match_any_checks_a_typed_arm_of_a_final_type() {
    let classify =
        |value: Any| classify!(value, _a _b _c _d _e _f _g _h _i _j _k _l _m _n _o _p _q _r _s);
    assert_eq!(classify(Any::from(pair(1, 2))), "positive");
    assert_eq!(classify(Any::from(pair(-1, 2))), "pair");
    assert_eq!(classify(Any::from(3_i64)), "other");
}

#[test]
fn hand_written_any_compatible_round_trips() {
    let raw = RawPair::try_from(Any::from(pair(1, 2))).unwrap();
    assert_eq!(Any::from(raw).downcast_ref::<Pair>().unwrap().second, 2);
}
