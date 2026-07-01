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

/// `(size, align)` pairs verified against real `ffi::Optional<T>` instances
/// (libstdc++ and libc++): `size = round_up(sizeof(T)+1, alignof(T))`.
#[test]
fn layout_matches_cpp_optional() {
    fn sa<T: OptionalPod>() -> (usize, usize) {
        (
            std::mem::size_of::<Optional<T>>(),
            std::mem::align_of::<Optional<T>>(),
        )
    }
    assert_eq!(sa::<i8>(), (2, 1));
    assert_eq!(sa::<bool>(), (2, 1));
    assert_eq!(sa::<i16>(), (4, 2));
    assert_eq!(sa::<i32>(), (8, 4));
    assert_eq!(sa::<f32>(), (8, 4));
    assert_eq!(sa::<i64>(), (16, 8));
    assert_eq!(sa::<f64>(), (16, 8));
}

/// Payload+flag bytes `[0, size_of::<T>()]`; padding is excluded (not ABI, not
/// guaranteed initialized).
fn image<T: OptionalPod>(opt: &Optional<T>) -> Vec<u8> {
    let p = opt as *const Optional<T> as *const u8;
    let n = std::mem::size_of::<T>() + 1; // payload + flag, no padding
    // Payload and flag are always initialized; padding is not read.
    unsafe { std::slice::from_raw_parts(p, n).to_vec() }
}

/// The scalar's own bytes, to assert the optional's payload matches it verbatim.
fn raw_bytes<T: OptionalPod>(v: &T) -> Vec<u8> {
    let p = v as *const T as *const u8;
    // size_of::<T>() bytes of an initialized `Copy` scalar.
    unsafe { std::slice::from_raw_parts(p, std::mem::size_of::<T>()).to_vec() }
}

#[test]
fn byte_image_some_i32_matches_cpp() {
    // C++ probe (both STLs): some(0x12345678) => 78 56 34 12 | 01 ..
    let o = Optional::<i32>::some(0x1234_5678);
    let b = image(&o);
    assert_eq!(&b[0..4], &0x1234_5678_i32.to_le_bytes());
    assert_eq!(b[4], 1, "engaged flag must sit at offset size_of::<i32>()");
}

#[test]
fn byte_image_some_i64_matches_cpp() {
    let o = Optional::<i64>::some(0x1122_3344_5566_7788);
    let b = image(&o);
    assert_eq!(&b[0..8], &0x1122_3344_5566_7788_i64.to_le_bytes());
    assert_eq!(b[8], 1, "engaged flag must sit at offset size_of::<i64>()");
}

#[test]
fn byte_image_none_clears_flag() {
    let o = Optional::<i32>::none();
    let b = image(&o);
    assert_eq!(b[4], 0, "engaged flag must be clear for nullopt");
}

/// Payload@0 and flag@`size_of::<T>()` for every supported type, not just i32/i64.
#[test]
fn flag_offset_all_supported_types() {
    fn check<T: OptionalPod + PartialEq + std::fmt::Debug>(val: T) {
        let ty = std::any::type_name::<T>();
        let sz = std::mem::size_of::<T>();
        let some = Optional::<T>::some(val);
        let b = image(&some);
        assert_eq!(&b[0..sz], &raw_bytes(&val)[..], "payload@0 for {ty}");
        assert_eq!(b[sz], 1, "engaged flag must sit at offset size_of for {ty}");
        assert_eq!(some.get(), Some(val), "engaged roundtrip for {ty}");
        let none = Optional::<T>::none();
        assert_eq!(image(&none)[sz], 0, "flag must be clear for none for {ty}");
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
    let o = Optional::<i32>::some(42);
    assert_eq!(o.get(), Some(42));
    assert!(o.has_value());

    // in-place mutation through a shared reference
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
fn clone_preserves_state() {
    let some = Optional::<i64>::some(123);
    assert_eq!(some.clone().get(), Some(123));
    let none = Optional::<i64>::none();
    assert_eq!(none.clone().get(), None);
}

// 16-byte cell (type_index@0, small_str_len@4, union@8); no padding.
fn str_image(o: &OptionalStr) -> [u8; 16] {
    let p = o as *const OptionalStr as *const u8;
    let mut b = [0u8; 16];
    // OptionalStr is a fully-initialized 16-byte TVMFFIAny cell.
    unsafe { std::ptr::copy_nonoverlapping(p, b.as_mut_ptr(), 16) };
    b
}

#[test]
fn optional_str_byte_image_matches_cpp() {
    // C++ probe: Optional<String> none => 16 zero bytes;
    //            some("hi")           => 0b 00 00 00 | 02 00 00 00 | 68 69 00 ...
    assert_eq!(str_image(&OptionalStr::none()), [0u8; 16]);
    let some = OptionalStr::some(String::from("hi"));
    let b = str_image(&some);
    assert_eq!(&b[0..4], &[0x0b, 0, 0, 0], "type_index = kTVMFFISmallStr");
    assert_eq!(&b[4..8], &[0x02, 0, 0, 0], "small_str_len = 2");
    assert_eq!(&b[8..10], b"hi", "inline payload");
}

#[test]
fn optional_str_roundtrip_and_conversions() {
    // small (inline) string
    let s = OptionalStr::some(String::from("hi"));
    assert!(s.has_value());
    assert_eq!(s.as_str(), Some("hi"));
    assert_eq!(s.get().as_deref(), Some("hi"));

    // nullopt
    let n = OptionalStr::none();
    assert!(n.is_none());
    assert_eq!(n.as_str(), None);
    assert_eq!(n.get(), None);

    // conversions + default
    let from_some: OptionalStr = Some(String::from("x")).into();
    assert_eq!(from_some.as_str(), Some("x"));
    let back: Option<String> = OptionalStr::none().into();
    assert!(back.is_none());
    assert!(OptionalStr::default().is_none());
}

#[test]
fn optional_str_heap_clone_no_double_free() {
    // long (heap) string exercises refcounted Clone/Drop through the wrapper.
    let long = String::from("a-very-long-heap-allocated-string-value");
    let a = OptionalStr::some(long);
    let b = a.clone();
    assert_eq!(a.as_str(), Some("a-very-long-heap-allocated-string-value"));
    assert_eq!(b.as_str(), Some("a-very-long-heap-allocated-string-value"));
    // both drop here: two dec_refs balancing the clone's inc_ref, no leak/UAF.
}

#[test]
fn optional_str_set_in_place() {
    // Start engaged with a heap string, then replace it: `set` must drop
    // (dec_ref) the old heap string before moving the new one in.
    let mut o = OptionalStr::some(String::from("first-long-heap-allocated-value"));
    assert_eq!(o.as_str(), Some("first-long-heap-allocated-value"));

    o.set(Some(String::from("second-long-heap-allocated-value")));
    assert_eq!(o.as_str(), Some("second-long-heap-allocated-value"));

    // disengage (drops the heap string), then re-engage
    o.set(None);
    assert!(o.is_none());
    assert_eq!(o.as_str(), None);

    o.set(Some(String::from("x")));
    assert_eq!(o.as_str(), Some("x"));
}
