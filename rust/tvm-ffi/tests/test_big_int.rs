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
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use tvm_ffi::*;

// (1 << 255) | (0x0123456789ABCDEF << 128) | 0xFEDCBA9876543210: distinct words, an
// interior zero, and a top word whose sign bit needs the zero guard word.
const WIDE_WORDS: [i64; 5] = [
    0xFEDCBA9876543210_u64 as i64,
    0,
    0x0123456789ABCDEF,
    i64::MIN,
    0,
];
const WIDE_DECIMAL: &str =
    "57896044618658097711813390734279005840776448950576999782953508207182870753808";

fn wide() -> BigInt {
    BigInt::from_words(&WIDE_WORDS)
}

/// The dummy call keeps `libtvm_ffi_testing`, which registers `testing.echo`, linked.
fn echo() -> Function {
    assert_eq!(unsafe { tvm_ffi_sys::TVMFFITestingDummyTarget() }, 0);
    Function::get_global("testing.echo").unwrap()
}

fn type_index_of(value: &BigInt) -> i32 {
    AnyView::from(value).type_index()
}

fn hash_of(value: &BigInt) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

#[test]
fn test_big_int_inline_representation() {
    for value in [BigInt::new(), BigInt::default(), BigInt::from(0i64)] {
        assert_eq!(type_index_of(&value), TypeIndex::kTVMFFIInt as i32);
        assert_eq!(value.words(), &[0]);
        assert_eq!(value.to_i64(), Some(0));
    }
    let inline = [
        BigInt::from(-1i8),
        BigInt::from(i16::MIN),
        BigInt::from(u32::MAX),
        BigInt::from(i64::MAX),
        BigInt::from(i64::MIN),
        BigInt::from(usize::MAX >> 1),
        BigInt::from(i128::from(i64::MAX)),
        BigInt::from(u128::from(u64::MAX >> 1)),
    ];
    let expected = [
        -1,
        i64::from(i16::MIN),
        i64::from(u32::MAX),
        i64::MAX,
        i64::MIN,
        (usize::MAX >> 1) as i64,
        i64::MAX,
        i64::MAX,
    ];
    for (value, expected) in inline.iter().zip(expected) {
        assert_eq!(type_index_of(value), TypeIndex::kTVMFFIInt as i32);
        assert_eq!(value.to_i64(), Some(expected));
        assert_eq!(value.words(), &[expected]);
        assert_eq!(value.is_negative(), expected < 0);
    }
}

#[test]
fn test_big_int_heap_representation() {
    let heap = [
        (BigInt::from(u64::MAX), vec![-1, 0]),
        (BigInt::from(usize::MAX), vec![-1, 0]),
        (BigInt::from(1u128 << 63), vec![i64::MIN, 0]),
        (BigInt::from(i128::from(i64::MIN) - 1), vec![i64::MAX, -1]),
        (BigInt::from(i128::MIN), vec![0, i64::MIN]),
        (BigInt::from(u128::MAX), vec![-1, -1, 0]),
        (wide(), WIDE_WORDS.to_vec()),
    ];
    for (value, words) in heap {
        assert_eq!(type_index_of(&value), TypeIndex::kTVMFFIBigInt as i32);
        assert_eq!(value.words(), words.as_slice());
        assert_eq!(value.to_i64(), None);
        assert_eq!(value.is_negative(), words[words.len() - 1] < 0);
        assert_eq!(AnyView::from(&value).debug_strong_count(), Some(1));
        let copy = value.clone();
        assert_eq!(AnyView::from(&value).debug_strong_count(), Some(2));
        assert_eq!(copy.words().as_ptr(), value.words().as_ptr());
        drop(copy);
        assert_eq!(AnyView::from(&value).debug_strong_count(), Some(1));
    }
}

#[test]
fn test_big_int_from_words_normalizes() {
    let zero = BigInt::from_words(&[]);
    assert_eq!(type_index_of(&zero), TypeIndex::kTVMFFIInt as i32);
    assert_eq!(zero, BigInt::new());
    for (words, expected) in [
        (vec![5, 0, 0], 5),
        (vec![-5, -1, -1], -5),
        (vec![i64::MIN, -1], i64::MIN),
        (vec![i64::MAX, 0], i64::MAX),
    ] {
        let value = BigInt::from_words(&words);
        assert_eq!(type_index_of(&value), TypeIndex::kTVMFFIInt as i32);
        assert_eq!(value.to_i64(), Some(expected));
    }
    let padded = BigInt::from_words(&[-1, 0, 0, 0]);
    assert_eq!(type_index_of(&padded), TypeIndex::kTVMFFIBigInt as i32);
    assert_eq!(padded.words(), &[-1, 0]);
    assert_eq!(padded, BigInt::from(u64::MAX));
    let mut padded_wide = WIDE_WORDS.to_vec();
    padded_wide.extend([0, 0]);
    assert_eq!(BigInt::from_words(&padded_wide).words(), &WIDE_WORDS);
    let mut padded_negative = WIDE_WORDS.map(|word| !word).to_vec();
    padded_negative.extend([-1, -1]);
    assert_eq!(
        BigInt::from_words(&padded_negative).words(),
        &WIDE_WORDS.map(|word| !word)
    );
}

#[test]
fn test_big_int_native_conversions() {
    let cases: [(BigInt, Option<i64>, Option<u64>, Option<i128>, Option<u128>); 9] = [
        (BigInt::from(0i64), Some(0), Some(0), Some(0), Some(0)),
        (BigInt::from(-1i64), Some(-1), None, Some(-1), None),
        (
            BigInt::from(i64::MAX),
            Some(i64::MAX),
            Some(i64::MAX as u64),
            Some(i128::from(i64::MAX)),
            Some(u128::from(i64::MAX as u64)),
        ),
        (
            BigInt::from(i64::MIN),
            Some(i64::MIN),
            None,
            Some(i128::from(i64::MIN)),
            None,
        ),
        (
            BigInt::from(u64::MAX),
            None,
            Some(u64::MAX),
            Some(i128::from(u64::MAX)),
            Some(u128::from(u64::MAX)),
        ),
        (
            BigInt::from(i128::from(i64::MIN) - 1),
            None,
            None,
            Some(i128::from(i64::MIN) - 1),
            None,
        ),
        (
            BigInt::from(i128::MAX),
            None,
            None,
            Some(i128::MAX),
            Some(i128::MAX as u128),
        ),
        (BigInt::from(i128::MIN), None, None, Some(i128::MIN), None),
        (BigInt::from(u128::MAX), None, None, None, Some(u128::MAX)),
    ];
    for (value, i64_value, u64_value, i128_value, u128_value) in cases {
        assert_eq!(value.to_i64(), i64_value, "{value}");
        assert_eq!(value.to_u64(), u64_value, "{value}");
        assert_eq!(value.to_i128(), i128_value, "{value}");
        assert_eq!(value.to_u128(), u128_value, "{value}");
    }
    assert_eq!(wide().to_u128(), None);
    assert_eq!(wide().to_i128(), None);
}

#[test]
fn test_big_int_ordering_and_hash() {
    let negative_wide = BigInt::from_words(&WIDE_WORDS.map(|word| !word));
    let ascending = [
        negative_wide.clone(),
        BigInt::from(i128::MIN),
        BigInt::from(-(1i128 << 64)),
        BigInt::from(i128::from(i64::MIN) - 1),
        BigInt::from(i64::MIN),
        BigInt::from(-1i64),
        BigInt::from(0i64),
        BigInt::from(1i64),
        BigInt::from(i64::MAX),
        BigInt::from(1u128 << 63),
        BigInt::from(u64::MAX),
        BigInt::from(1u128 << 64),
        BigInt::from(i128::MAX),
        BigInt::from(u128::MAX),
        wide(),
    ];
    for (i, lhs) in ascending.iter().enumerate() {
        for (j, rhs) in ascending.iter().enumerate() {
            assert_eq!(lhs.cmp(rhs), i.cmp(&j), "{lhs} vs {rhs}");
            assert_eq!(lhs == rhs, i == j, "{lhs} vs {rhs}");
        }
    }
    let same = [
        (BigInt::from(u64::MAX), BigInt::from_words(&[-1, 0, 0])),
        (BigInt::from(7u8), BigInt::from_words(&[7, 0])),
        (wide(), BigInt::from_words(&WIDE_WORDS)),
        (
            negative_wide,
            BigInt::from_words(&WIDE_WORDS.map(|word| !word)),
        ),
    ];
    for (lhs, rhs) in same {
        assert_eq!(lhs, rhs);
        assert_eq!(hash_of(&lhs), hash_of(&rhs));
    }
}

#[test]
fn test_big_int_display() {
    let cases = [
        (BigInt::from(0i64), "0"),
        (BigInt::from(-1i64), "-1"),
        (BigInt::from(i64::MAX), "9223372036854775807"),
        (BigInt::from(i64::MIN), "-9223372036854775808"),
        (BigInt::from(1u128 << 63), "9223372036854775808"),
        (
            BigInt::from(i128::from(i64::MIN) - 1),
            "-9223372036854775809",
        ),
        (BigInt::from(u64::MAX), "18446744073709551615"),
        (BigInt::from(1u128 << 64), "18446744073709551616"),
        (
            BigInt::from(1u128 << 100),
            "1267650600228229401496703205376",
        ),
        (
            BigInt::from(-(1i128 << 100)),
            "-1267650600228229401496703205376",
        ),
        (
            BigInt::from(i128::MIN),
            "-170141183460469231731687303715884105728",
        ),
        (
            BigInt::from(u128::MAX),
            "340282366920938463463374607431768211455",
        ),
        (
            BigInt::from_words(&[0, 0, 0, 1]),
            "6277101735386680763835789423207666416102355444464034512896",
        ),
        (wide(), WIDE_DECIMAL),
    ];
    for (value, expected) in cases {
        assert_eq!(value.to_string(), expected);
        assert_eq!(
            format!("{value:?}"),
            format!("ffi.BigInt {{ data: {expected} }}")
        );
    }
    assert_eq!(
        format!("{:>22}", BigInt::from(u64::MAX)),
        "  18446744073709551615"
    );
    assert_eq!(
        format!("{:+}", BigInt::from(1u128 << 64)),
        "+18446744073709551616"
    );
}

#[test]
fn test_big_int_any_round_trip() {
    let value = wide();
    let any = Any::from(value.clone());
    assert_eq!(any.type_index(), TypeIndex::kTVMFFIBigInt as i32);
    assert_eq!(any.debug_strong_count(), Some(2));
    assert_eq!(any.try_as::<BigInt>(), Some(value.clone()));
    let view = AnyView::from(&value);
    assert_eq!(BigInt::try_from(view).unwrap(), value);
    let restored = BigInt::try_from(any).unwrap();
    assert_eq!(restored.words().as_ptr(), value.words().as_ptr());
    drop(restored);
    assert_eq!(AnyView::from(&value).debug_strong_count(), Some(1));

    // An inline value is a plain int cell in both directions.
    let small = BigInt::from(-42i64);
    let any = Any::from(small.clone());
    assert_eq!(any.type_index(), TypeIndex::kTVMFFIInt as i32);
    assert_eq!(i64::try_from(any).unwrap(), -42);
    assert_eq!(BigInt::try_from(Any::from(-42i64)).unwrap(), small);
    assert_eq!(Any::from(7i32).try_as::<BigInt>(), Some(BigInt::from(7i64)));
    assert_eq!(
        BigInt::try_from(Any::from(true)).unwrap(),
        BigInt::from(1i64)
    );
    assert_eq!(Any::from(true).try_as::<BigInt>(), None);

    let error = BigInt::try_from(Any::from(1.5f64)).unwrap_err();
    assert_eq!(error.kind(), TYPE_ERROR);
    assert!(error.message().contains("BigInt"));
    let error = i64::try_from(Any::from(wide())).unwrap_err();
    assert_eq!(error.kind(), TYPE_ERROR);
    assert!(error.message().contains("ffi.BigInt"));
}

#[test]
fn test_big_int_function_round_trip() {
    let echo = echo();
    for value in [wide(), BigInt::from(i128::MIN), BigInt::from(i64::MIN)] {
        let result = echo.call_tuple((value.clone(),)).unwrap();
        assert_eq!(result.type_index(), type_index_of(&value));
        let echoed = BigInt::try_from(result).unwrap();
        assert_eq!(echoed, value);
        // A heap value crosses the call by reference; an inline one is copied.
        if value.to_i64().is_none() {
            assert_eq!(echoed.words().as_ptr(), value.words().as_ptr());
        }
    }

    let identity = Function::from_typed(|value: BigInt| -> Result<BigInt> { Ok(value) });
    let result = identity.call_tuple((wide(),)).unwrap();
    assert_eq!(BigInt::try_from(result).unwrap(), wide());
    let result = identity.call_tuple((5i64,)).unwrap();
    assert_eq!(result.type_index(), TypeIndex::kTVMFFIInt as i32);
    assert_eq!(BigInt::try_from(result).unwrap(), BigInt::from(5i64));
    let error = identity.call_tuple((String::from("text"),)).err().unwrap();
    assert_eq!(error.kind(), TYPE_ERROR);
    assert!(error.message().contains("BigInt"));

    let narrow = Function::from_typed(|value: i64| -> Result<i64> { Ok(value) });
    let error = narrow.call_tuple((&wide(),)).err().unwrap();
    assert_eq!(error.kind(), TYPE_ERROR);
    assert!(error.message().contains("ffi.BigInt"));
}

#[test]
fn test_big_int_containers() {
    let values = vec![wide(), BigInt::from(-1i64), BigInt::from(u64::MAX)];
    let array = Array::new(values.clone());
    assert_eq!(array.len(), 3);
    for (i, value) in values.iter().enumerate() {
        assert_eq!(array.get(i).unwrap(), *value);
    }
    let echoed = echo()
        .call_tuple((array.clone(),))
        .and_then(Array::<BigInt>::try_from)
        .unwrap();
    assert_eq!(echoed.iter().collect::<Vec<_>>(), values);

    let map: Map<BigInt, i64> = values
        .iter()
        .enumerate()
        .map(|(i, value)| (value.clone(), i as i64))
        .collect();
    assert_eq!(map.len(), 3);
    for (i, value) in values.iter().enumerate() {
        assert!(map.contains_key(value));
        assert_eq!(map.get(value).unwrap(), Some(i as i64));
    }
    assert_eq!(map.get(&BigInt::from(1u128 << 64)).unwrap(), None);
}

#[test]
fn test_big_int_optional() {
    assert_eq!(std::mem::size_of::<Optional<BigInt>>(), 16);
    let some = Optional::some(wide());
    assert!(some.has_value());
    assert_eq!(some.get(), Some(wide()));
    let none = Optional::<BigInt>::none();
    assert!(none.is_none());
    assert_eq!(none.get(), None);
    assert_eq!(some.into_option(), Some(wide()));

    let any = Any::from(Some(wide()));
    assert_eq!(any.type_index(), TypeIndex::kTVMFFIBigInt as i32);
    assert_eq!(Option::<BigInt>::try_from(any).unwrap(), Some(wide()));
    assert_eq!(Option::<BigInt>::try_from(Any::new()).unwrap(), None);
    let echoed = echo().call_tuple((Some(BigInt::from(u64::MAX)),)).unwrap();
    assert_eq!(
        Option::<BigInt>::try_from(echoed).unwrap(),
        Some(BigInt::from(u64::MAX))
    );
}

#[test]
fn test_big_int_structural_traversal() {
    let root = Array::new(vec![Any::from(1i64), Any::from(wide())]);
    let wide_pointer = root
        .get(1)
        .unwrap()
        .try_as::<BigInt>()
        .unwrap()
        .words()
        .as_ptr();

    // A BigInt callback also accepts inline integers, mirroring the C++ cast rule.
    let seen = std::cell::RefCell::new(Vec::new());
    assert!(structural_visit(
        &root,
        |value: BigInt, _visitor: &mut VisitContext<'_, ()>| {
            seen.borrow_mut().push(value);
        }
    )
    .unwrap()
    .is_none());
    assert_eq!(seen.into_inner(), vec![BigInt::from(1i64), wide()]);

    // The heap integer is a leaf: mapping the inline integer leaves it untouched.
    let mapped = structural_map(
        root.clone(),
        |integer: i64| Any::from(integer + 1),
        WalkOrder::PostOrder,
    )
    .and_then(Array::<Any>::try_from)
    .unwrap();
    assert_eq!(mapped.get(0).unwrap().try_as::<i64>(), Some(2));
    let leaf = mapped.get(1).unwrap().try_as::<BigInt>().unwrap();
    assert_eq!(leaf, wide());
    assert_eq!(leaf.words().as_ptr(), wide_pointer);
}
