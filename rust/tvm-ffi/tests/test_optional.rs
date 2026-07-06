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
use tvm_ffi::*;

/// The 16-byte `TVMFFIAny` cell backing an `Optional<T>` (type_index@0,
/// small_str_len@4, union@8); no padding.
fn cell_image<T: AnyCompatible>(opt: &Optional<T>) -> [u8; 16] {
    let p = opt as *const Optional<T> as *const u8;
    let mut b = [0u8; 16];
    // `Optional<T>` is a fully-initialized 16-byte cell.
    unsafe { std::ptr::copy_nonoverlapping(p, b.as_mut_ptr(), 16) };
    b
}

#[test]
fn layout_is_uniform_16_bytes() {
    // Independent of `T`: every `Optional<T>` is the 16-byte `TVMFFIAny` cell.
    assert_eq!(std::mem::size_of::<Optional<i32>>(), 16);
    assert_eq!(std::mem::size_of::<Optional<i64>>(), 16);
    assert_eq!(std::mem::size_of::<Optional<bool>>(), 16);
    assert_eq!(std::mem::size_of::<Optional<f64>>(), 16);
    assert_eq!(std::mem::size_of::<Optional<String>>(), 16);
    assert_eq!(std::mem::size_of::<Optional<Array<i64>>>(), 16);
    assert_eq!(
        std::mem::align_of::<Optional<i32>>(),
        std::mem::align_of::<Optional<Array<i64>>>()
    );
}

#[test]
fn none_is_all_zero_cell() {
    // `kTVMFFINone == 0` and the union is zeroed, so `nullopt` is 16 zero bytes.
    assert_eq!(cell_image(&Optional::<i32>::none()), [0u8; 16]);
    assert_eq!(cell_image(&Optional::<String>::none()), [0u8; 16]);
    assert_eq!(cell_image(&Optional::<Array<i64>>::none()), [0u8; 16]);
}

#[test]
#[cfg(target_endian = "little")]
fn byte_image_some_int_matches_ffi_any() {
    // some(0x12345678) => type_index = kTVMFFIInt @0, v_int64 = value @8.
    let o = Optional::<i32>::some(0x1234_5678);
    let b = cell_image(&o);
    assert_eq!(
        &b[0..4],
        &(TypeIndex::kTVMFFIInt as i32).to_le_bytes(),
        "type_index must be kTVMFFIInt"
    );
    assert_eq!(&b[4..8], &[0u8; 4], "zero padding");
    assert_eq!(
        &b[8..16],
        &0x1234_5678_i64.to_le_bytes(),
        "payload sits in v_int64 @8"
    );
}

#[test]
fn pod_roundtrip_all_supported_types() {
    fn check<T: AnyCompatible + PartialEq + Copy + std::fmt::Debug>(val: T) {
        let ty = std::any::type_name::<T>();
        let some = Optional::<T>::some(val);
        assert!(some.has_value(), "engaged has_value for {ty}");
        assert_eq!(some.get(), Some(val), "engaged roundtrip for {ty}");

        let none = Optional::<T>::none();
        assert!(none.is_none(), "disengaged is_none for {ty}");
        assert!(none.get().is_none(), "disengaged roundtrip for {ty}");
    }
    check::<bool>(true);
    check::<i8>(0x12);
    check::<i16>(0x1234);
    check::<i32>(0x1234_5678);
    check::<i64>(0x1122_3344_5566_7788);
    check::<u8>(0xAB);
    check::<u16>(0xABCD);
    check::<u32>(0xABCD_EF01);
    check::<u64>(0xABCD_EF01_2345_6789);
    check::<f32>(1.5);
    check::<f64>(2.5);
}

#[test]
fn roundtrip_get_set() {
    let mut o = Optional::<i32>::some(42);
    assert_eq!(o.get(), Some(42));
    assert!(o.has_value());

    o.set(None);
    assert_eq!(o.get(), None);
    assert!(o.is_none());

    o.set(Some(-7));
    assert_eq!(o.get(), Some(-7));
}

#[test]
fn conversions_and_default() {
    assert_eq!(Optional::<f64>::from(Some(2.5)).get(), Some(2.5));
    assert_eq!(Optional::<f64>::from(None).get(), None);
    let back: Option<i16> = Optional::<i16>::some(9).into();
    assert_eq!(back, Some(9));
    assert_eq!(Optional::<u8>::default().get(), None);
    assert_eq!(Optional::<bool>::some(true).get(), Some(true));
}

#[test]
fn clone_preserves_state_pod() {
    let some = Optional::<i64>::some(123);
    assert_eq!(some.clone().get(), Some(123));
    let none = Optional::<i64>::none();
    assert_eq!(none.clone().get(), None);
}

#[test]
fn equality_pod_and_string() {
    assert_eq!(Optional::<i32>::some(1), Optional::<i32>::some(1));
    assert_ne!(Optional::<i32>::some(1), Optional::<i32>::some(2));
    assert_ne!(Optional::<i32>::some(1), Optional::<i32>::none());
    assert_eq!(Optional::<i32>::none(), Optional::<i32>::none());
    // `set(None)` resets to a plain `nullopt` cell; equality must still hold.
    let mut cleared = Optional::<i32>::some(7);
    cleared.set(None);
    assert_eq!(cleared, Optional::<i32>::none());

    let a = || Optional::<String>::some(String::from("a"));
    assert_eq!(a(), a());
    assert_ne!(a(), Optional::<String>::some(String::from("b")));
    assert_ne!(a(), Optional::<String>::none());
    assert_eq!(Optional::<String>::none(), Optional::<String>::none());
}

#[test]
#[cfg(target_endian = "little")]
fn string_byte_image_matches_ffi_any() {
    // ffi::Optional<String> none => 16 zero bytes;
    // some("hi") => kTVMFFISmallStr @0 | small_str_len=2 @4 | "hi" inline @8.
    assert_eq!(cell_image(&Optional::<String>::none()), [0u8; 16]);
    let some = Optional::<String>::some(String::from("hi"));
    let b = cell_image(&some);
    assert_eq!(
        &b[0..4],
        &(TypeIndex::kTVMFFISmallStr as i32).to_le_bytes(),
        "type_index = kTVMFFISmallStr"
    );
    assert_eq!(&b[4..8], &[0x02, 0, 0, 0], "small_str_len = 2");
    assert_eq!(&b[8..10], b"hi", "inline payload");
}

#[test]
fn string_roundtrip_and_as_str() {
    // small (inline) string
    let s = Optional::<String>::some(String::from("hi"));
    assert!(s.has_value());
    assert_eq!(s.as_str(), Some("hi"));
    assert_eq!(s.get().as_deref(), Some("hi"));

    // nullopt
    let n = Optional::<String>::none();
    assert!(n.is_none());
    assert_eq!(n.as_str(), None);
    assert_eq!(n.get(), None);

    // conversions + default
    let from_some: Optional<String> = Some(String::from("x")).into();
    assert_eq!(from_some.as_str(), Some("x"));
    let back: Option<String> = Optional::<String>::none().into();
    assert!(back.is_none());
    assert!(Optional::<String>::default().is_none());
}

#[test]
fn string_heap_clone_no_double_free() {
    // long (heap) string exercises refcounted Clone/Drop through the wrapper.
    let long = String::from("a-very-long-heap-allocated-string-value");
    let a = Optional::<String>::some(long);
    let b = a.clone();
    assert_eq!(a.as_str(), Some("a-very-long-heap-allocated-string-value"));
    assert_eq!(b.as_str(), Some("a-very-long-heap-allocated-string-value"));
    // both drop here: two dec_refs balancing the clone's inc_ref, no leak/UAF.
}

#[test]
fn string_set_in_place() {
    // Replace an engaged heap string: `set` must drop (dec_ref) the old heap
    // string before storing the new one.
    let mut o = Optional::<String>::some(String::from("first-long-heap-allocated-value"));
    assert_eq!(o.as_str(), Some("first-long-heap-allocated-value"));

    o.set(Some(String::from("second-long-heap-allocated-value")));
    assert_eq!(o.as_str(), Some("second-long-heap-allocated-value"));

    o.set(None);
    assert!(o.is_none());
    assert_eq!(o.as_str(), None);

    o.set(Some(String::from("x")));
    assert_eq!(o.as_str(), Some("x"));
}

#[test]
fn object_ref_payload_roundtrip_and_clone() {
    // An ObjectRef payload is now the same 16-byte cell (was an 8-byte pointer).
    let arr = Array::new(vec![1i64, 2, 3]);
    let opt = Optional::<Array<i64>>::some(arr);
    assert!(opt.has_value());
    let got = opt.get().expect("engaged");
    assert_eq!(got.len(), 3);

    // clone shares the underlying object; both drop without double-free.
    let cloned = opt.clone();
    assert!(cloned.has_value());
    assert_eq!(cloned.get().expect("engaged").len(), 3);

    // move the payload out
    let moved: Option<Array<i64>> = opt.into_option();
    assert_eq!(moved.expect("moved").len(), 3);

    let none = Optional::<Array<i64>>::none();
    assert!(none.is_none());
    assert!(none.get().is_none());
}
