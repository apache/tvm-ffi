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
fn test_big_int_from_bool() {
    for (input, expected) in [(false, 0i64), (true, 1)] {
        let value = BigInt::from(input);
        assert_eq!(type_index_of(&value), TypeIndex::kTVMFFIInt as i32);
        assert_eq!(value.to_i64(), Some(expected));
    }
}

/// Rebuild a double from canonical words; exact when the magnitude has at most 53 significant bits.
fn words_as_f64(value: &BigInt) -> f64 {
    let negative = value.is_negative();
    let mut words: Vec<u64> = value
        .words()
        .iter()
        .map(|&w| if negative { !(w as u64) } else { w as u64 })
        .collect();
    if negative {
        for word in words.iter_mut() {
            let (sum, overflow) = word.overflowing_add(1);
            *word = sum;
            if !overflow {
                break;
            }
        }
    }
    let magnitude: f64 = words
        .iter()
        .enumerate()
        .filter(|(_, &word)| word != 0)
        .map(|(i, &word)| word as f64 * 2f64.powi(64 * i as i32))
        .sum();
    if negative {
        -magnitude
    } else {
        magnitude
    }
}

#[test]
fn test_big_int_from_f64() {
    let cases = [
        (0.0, BigInt::from(0i64)),
        (-0.0, BigInt::from(0i64)),
        (1.9, BigInt::from(1i64)),
        (-1.9, BigInt::from(-1i64)),
        (-0.999, BigInt::from(0i64)),
        (123456789.987, BigInt::from(123456789i64)),
        (9007199254740994.0, BigInt::from(9007199254740994i64)),
        (i64::MIN as f64, BigInt::from(i64::MIN)),
        // i64::MAX rounds up to 2^63 as a double, which no longer fits inline.
        (i64::MAX as f64, BigInt::from(1u128 << 63)),
        (-(i64::MAX as f64), BigInt::from(i64::MIN)),
        (18446744073709551616.0, BigInt::from(1u128 << 64)),
        (1e30, BigInt::from(1000000000000000019884624838656i128)),
        (
            -1.5 * 2f64.powi(100),
            BigInt::from(-1901475900342344102245054808064i128),
        ),
    ];
    for (input, expected) in cases {
        assert_eq!(BigInt::try_from(input).unwrap(), expected, "{input}");
    }
    // Wide magnitudes: the significand straddles words at every alignment.
    for input in [
        1e300,
        -1e300,
        f64::MAX,
        -f64::MAX,
        1.75 * 2f64.powi(1000),
        -(2f64.powi(1023)),
        3.0 * 2f64.powi(191),
    ] {
        let value = BigInt::try_from(input).unwrap();
        assert_eq!(type_index_of(&value), TypeIndex::kTVMFFIBigInt as i32);
        assert_eq!(value.is_negative(), input < 0.0);
        assert_eq!(words_as_f64(&value), input, "{input}");
    }
    assert_eq!(
        BigInt::try_from(3.0 * 2f64.powi(191)).unwrap().words(),
        &[0, 0, i64::MIN, 1]
    );
    let error = BigInt::try_from(f64::NAN).unwrap_err();
    assert_eq!(error.kind(), VALUE_ERROR);
    assert!(error.message().contains("NaN"));
    for input in [f64::INFINITY, f64::NEG_INFINITY] {
        let error = BigInt::try_from(input).unwrap_err();
        assert_eq!(error.kind(), OVERFLOW_ERROR);
        assert!(error.message().contains("infinity"));
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
fn test_big_int_words_mirror_runtime_content() {
    // The native word slice must be the runtime's own view of the same cell.
    for value in [
        wide(),
        BigInt::from(u64::MAX),
        BigInt::from(i128::MIN),
        BigInt::from(-5i64),
    ] {
        let mut any = Any::from(value.clone());
        let content = unsafe { tvm_ffi_sys::TVMFFIBigIntGetContentByteArray(any.as_data_ptr()) };
        let runtime = unsafe {
            std::slice::from_raw_parts(
                content.data as *const i64,
                content.size / std::mem::size_of::<i64>(),
            )
        };
        assert_eq!(value.words(), runtime);
        // A heap value is shared, so both views borrow the same object words.
        if value.to_i64().is_none() {
            assert_eq!(value.words().as_ptr(), runtime.as_ptr());
        }
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

// ============================================================================
// Arithmetic: ported from tests/cpp/test_big_int.cc, then checked against the
// C++ operators through the testing oracles.
// ============================================================================

fn big(value: i128) -> BigInt {
    BigInt::from(value)
}

fn pow2(bits: i64) -> BigInt {
    &big(1) << bits
}

fn assert_inline(value: &BigInt, expected: i64) {
    assert_eq!(
        type_index_of(value),
        TypeIndex::kTVMFFIInt as i32,
        "{value}"
    );
    assert_eq!(value.to_i64(), Some(expected));
}

#[test]
fn test_big_int_promotion_demotion_and_mixed_operands() {
    let high = &big(i64::MAX as i128) + 1i64;
    assert_eq!(type_index_of(&high), TypeIndex::kTVMFFIBigInt as i32);
    assert_eq!(high.to_string(), "9223372036854775808");
    assert_inline(&(&high - 1i64), i64::MAX);
    assert_inline(&-&high, i64::MIN);
    assert_inline(&(&high + i64::MIN), 0);
    assert_eq!(&big(i64::MIN as i128) - 1i64, -(&high + 1i64));
    assert_eq!(-big(i64::MIN as i128), high);
    assert_eq!(&big(i64::MIN as i128) / -1i64, high);
    assert_inline(&(&big(i64::MIN as i128) % -1i64), 0);
    assert_eq!(BigInt::from(u64::MAX).to_string(), "18446744073709551615");
    assert_eq!(1i64 + BigInt::from(u64::MAX), pow2(64));
    assert_eq!(
        BigInt::from(u64::MAX) - big(1),
        &BigInt::from(u64::MAX) - 1i64
    );
    for (a, b) in [(i64::MAX, 17i64), (-123, 17)] {
        let (x, y) = (BigInt::from(a), BigInt::from(b));
        assert_eq!(&x + &y, &x + b);
        assert_eq!(&x + &y, a + &y);
        assert_eq!(&x - &y, &x - b);
        assert_eq!(&x - &y, a - &y);
        assert_eq!(&x * &y, &x * b);
        assert_eq!(&x * &y, a * &y);
        assert_eq!((&x * &y) / &y, x);
        assert_eq!(&x / &y, a / &y);
        assert_eq!(&x / &y, &x / b);
        assert_eq!(&x % &y, a % &y);
        assert_eq!(&x % &y, &x % b);
        assert_eq!(&x & &y, a & &y);
        assert_eq!(&x | &y, &x | b);
        assert_eq!(&x ^ &y, a ^ &y);
        assert_eq!(std::cmp::min(&x, &y), &BigInt::from(a.min(b)));
        assert_eq!(std::cmp::max(&x, &y), &BigInt::from(a.max(b)));
    }
    let heap = pow2(255);
    assert_eq!(std::cmp::min(heap.clone(), -&heap), -&heap);
    assert_eq!(std::cmp::max(-&heap, heap.clone()), heap);
    let mut counter = big(i64::MAX as i128);
    counter += 1i64;
    assert_eq!(counter, high);
    counter -= 1i64;
    assert_inline(&counter, i64::MAX);
    counter -= &big(1);
    assert_inline(&counter, i64::MAX - 1);
}

#[test]
fn test_big_int_normalized_equality() {
    let value = pow2(255) + 13i64;
    let equal = (&big(2) << 254i64) + 13i64;
    assert_eq!(value, equal);
    assert_ne!(value, &value + 1i64);
    assert_ne!(value, &value + pow2(128));
    assert_ne!(pow2(192), pow2(193));
    assert_ne!(value, pow2(128));
    assert_ne!(value, -&value);
    let above = big(i64::MAX as i128) + 1i64;
    let below = big(i64::MIN as i128) - 1i64;
    assert_ne!(above, big(i64::MAX as i128));
    assert_ne!(below, big(i64::MIN as i128));
}

#[test]
fn test_big_int_signed_division() {
    // (a, b, trunc q, trunc r, floor q, floor r)
    for (a, b, q, r, floor_q, floor_r) in [
        (7i64, 3i64, 2i64, 1i64, 2i64, 1i64),
        (-7, 3, -2, -1, -3, 2),
        (7, -3, -2, 1, -3, -2),
        (-7, -3, 2, -1, 2, -1),
        (-6, 3, -2, 0, -2, 0),
        (0, 3, 0, 0, 0, 0),
    ] {
        let (x, y) = (BigInt::from(a), BigInt::from(b));
        assert_inline(&(&x / &y), q);
        assert_inline(&(&x % b), r);
        assert_inline(&x.try_div(&y).unwrap(), q);
        assert_inline(&x.try_rem(&y).unwrap(), r);
        assert_inline(&x.floor_div(&y).unwrap(), floor_q);
        assert_inline(&x.floor_mod(&y).unwrap(), floor_r);
        let (dq, dr) = x.div_rem(&y).unwrap();
        assert_inline(&dq, q);
        assert_inline(&dr, r);
        // Wide operands with the same values take the multiword paths.
        let shift = 200i64;
        let (wx, wy) = (&x << shift, &y << shift);
        assert_inline(&(&wx / &wy), q);
        assert_eq!(&wx % &wy, &BigInt::from(r) << shift);
        assert_inline(&wx.floor_div(&wy).unwrap(), floor_q);
        assert_eq!(wx.floor_mod(&wy).unwrap(), &BigInt::from(floor_r) << shift);
    }
    let wide = pow2(255) + 13i64;
    for divisor in [big(0), pow2(0) - 1i64] {
        for result in [
            wide.try_div(&divisor),
            wide.try_rem(&divisor),
            wide.floor_div(&divisor),
            wide.floor_mod(&divisor),
            big(0).floor_mod(&divisor),
            big(5).try_div(&divisor),
            wide.div_rem(&divisor).map(|pair| pair.0),
        ] {
            let error = result.unwrap_err();
            assert_eq!(error.kind(), ZERO_DIVISION_ERROR);
            assert_eq!(error.message(), "Division by zero");
        }
    }
}

#[test]
#[should_panic(expected = "Division by zero")]
fn test_big_int_division_by_zero_panics() {
    let _ = &pow2(100) / 0i64;
}

#[test]
#[should_panic(expected = "Negative BigInt shift count")]
fn test_big_int_negative_shift_panics() {
    let _ = &big(1) << -1i64;
}

#[test]
fn test_big_int_wide_arithmetic_and_bitwise() {
    for shift in [0i64, 63, 64, 255, 256] {
        let x = pow2(shift);
        assert_eq!(&x >> shift, big(1));
        assert_eq!(-&x >> shift, big(-1));
        assert_inline(&(&x - &x), 0);
        assert_inline(&((&x + 3i64) - &x), 3);
        assert_eq!(&x & -1i64, x);
        assert_eq!(-1i64 & &x, x);
        assert_eq!(&x | -1i64, big(-1));
        assert_eq!(!&x, -&x - 1i64);
        assert_eq!((&x - 1i64) ^ &x, 2i64 * &x - 1i64);
        assert_eq!((-&x - 1i64) >> shift, big(-2));
        assert_eq!(&x << big(2), &x * 4i64);
        assert_eq!(1i64 << BigInt::from(shift), x);
        assert_eq!(x.try_shl(2).unwrap(), &x * 4i64);
        assert_eq!(x.try_shr(shift).unwrap(), big(1));
    }
    let huge = pow2(255);
    assert_inline(&(&huge >> &huge), 0);
    assert_inline(&(-&huge >> &huge), -1);
    assert_inline(&(&big(0) << &huge), 0);
    let error = huge.try_shl(huge.clone()).unwrap_err();
    assert_eq!(error.kind(), OVERFLOW_ERROR);
    assert_eq!(error.message(), "BigInt shift count is too large");
    assert_inline(&huge.try_shr(huge.clone()).unwrap(), 0);
    let error = huge.try_shl(-1).unwrap_err();
    assert_eq!(error.kind(), VALUE_ERROR);
    let error = big(0).try_shr(-1).unwrap_err();
    assert_eq!(error.kind(), VALUE_ERROR);
    assert_eq!(pow2(100).to_string(), "1267650600228229401496703205376");
    // Mixed inline/heap bitwise operands and negative heap values.
    let negative = -pow2(200) - 5i64;
    assert_eq!(&negative & 0xffi64, big((-5i128) & 0xff));
    assert_eq!(&negative | 0i64, negative);
    assert_eq!(&negative ^ &negative, big(0));
    assert_eq!(!&negative, pow2(200) + 4i64);
}

#[test]
fn test_big_int_independent_wide_fixture() {
    // Decimal expectations were computed with Python integers, independently of these operators.
    let a: BigInt =
        "-57896044618658097711813390734279005840776448950576999782953508207182870753808"
            .parse()
            .unwrap();
    let b: BigInt = "1361129467683753853854892183719458155744".parse().unwrap();
    assert_eq!(
        (&a + &b).to_string(),
        "-57896044618658097711813390734279005839415319482893245929098616023463412598064"
    );
    assert_eq!(
        (&a * &b).to_string(),
        "-78804012392788958424676746146478616550595249492184343272504214848687229926200124701108634385188332729146218745073152"
    );
    assert_eq!(
        a.floor_div(&b).unwrap().to_string(),
        "-42535295865117307932898767499013107221"
    );
    assert_eq!(
        a.floor_mod(&b).unwrap().to_string(),
        "340250229142126476510862697120718273616"
    );
    assert_eq!((&a & &b).to_string(), "72903423425023200");
    assert_eq!(
        (&a ^ &b).to_string(),
        "-57896044618658097711813390734279005839415319482893245929098761830310262644464"
    );
    assert_eq!(
        (&a >> 97i64).to_string(),
        "-365375409332725729551097270762435786773001928705"
    );
}

#[test]
fn test_big_int_to_f64() {
    for x in [-7.9, -0.0, 0.0, 0.5, 7.9, f64::MIN_POSITIVE] {
        assert_inline(&BigInt::try_from(x).unwrap(), x as i64);
    }
    assert_eq!(BigInt::try_from(2f64.powi(100)).unwrap(), pow2(100));
    // shift=64 writes one significand word between a low zero word and a high sign guard.
    assert_eq!(BigInt::try_from(2f64.powi(116)).unwrap(), pow2(116));
    assert_eq!(BigInt::try_from(-(2f64.powi(100))).unwrap(), -pow2(100));
    let x = pow2(100);
    assert_eq!(x.to_f64().unwrap(), 2f64.powi(100));
    assert_eq!((&x + pow2(47)).to_f64().unwrap(), 2f64.powi(100));
    let above = f64::from_bits(2f64.powi(100).to_bits() + 1);
    assert_eq!((&x + pow2(47) + 1i64).to_f64().unwrap(), above);
    assert_eq!(big(i64::MIN as i128).to_f64().unwrap(), i64::MIN as f64);
    assert_eq!((-pow2(100) - 1i64).to_f64().unwrap(), -(2f64.powi(100)));
    let largest = BigInt::try_from(f64::MAX).unwrap();
    assert_eq!(largest.to_f64().unwrap(), f64::MAX);
    assert_eq!((&largest + pow2(970) - 1i64).to_f64().unwrap(), f64::MAX);
    for value in [&largest + pow2(970), pow2(1024), -pow2(1024), pow2(1100)] {
        let error = value.to_f64().unwrap_err();
        assert_eq!(error.kind(), OVERFLOW_ERROR);
        assert!(error.message().contains("finite double"));
    }
}

#[test]
fn test_big_int_from_str() {
    for value in [
        big(0),
        big(-1),
        big(i64::MIN as i128),
        big(i64::MAX as i128),
        pow2(63),
        -pow2(63) - 1i64,
        pow2(255),
        -pow2(255),
        wide(),
        -wide(),
        BigInt::from(u128::MAX),
    ] {
        let text = value.to_string();
        assert_eq!(text.parse::<BigInt>().unwrap(), value, "{text}");
        if !value.is_negative() {
            assert_eq!(format!("+{text}").parse::<BigInt>().unwrap(), value);
        }
    }
    assert_inline(&"+0007".parse::<BigInt>().unwrap(), 7);
    assert_inline(&"-0".parse::<BigInt>().unwrap(), 0);
    assert_eq!(
        "1267650600228229401496703205376".parse::<BigInt>().unwrap(),
        pow2(100)
    );
    assert_eq!(
        "000000000000000000001267650600228229401496703205376"
            .parse::<BigInt>()
            .unwrap(),
        pow2(100)
    );
    for text in ["", "+", "-", "x", "12x", " 1", "1 ", "1.0", "--1", "0x10"] {
        let error = text.parse::<BigInt>().unwrap_err();
        assert_eq!(error.kind(), VALUE_ERROR, "{text:?}");
    }
}

#[test]
fn test_big_int_word_multiply() {
    // Half-product carry, complete-word carry, and unequal multiword lengths.
    for (m, n) in [(32i64, 32i64), (64, 64), (255, 256)] {
        let (a, b) = (pow2(m) - 1i64, pow2(n) - 1i64);
        let expected = pow2(m + n) - pow2(m) - pow2(n) + 1i64;
        assert_eq!(&a * &b, expected);
    }
    let (a, b) = (pow2(255) - 1i64, pow2(256) - 1i64);
    let product = pow2(511) - pow2(255) - pow2(256) + 1i64;
    assert_eq!(-&a * &b, -&product);
    assert_eq!(&a * -&b, -&product);
    assert_eq!(-&a * -&b, product);
    assert_inline(&(&a * 0i64), 0);
    assert_inline(&(0i64 * &a), 0);
    assert_inline(&(-&a * 0i64), 0);
}

#[test]
fn test_big_int_signed_div_rem() {
    let divisor = pow2(130) + 5i64;
    let quotient = pow2(120) + 3i64;
    let exact = &quotient * &divisor;
    let dividend = &exact + 7i64;
    // General zero, all nonexact sign quadrants, and an exact low-zero/minimum-word magnitude.
    for (a, b, q, r) in [
        (big(0), divisor.clone(), big(0), big(0)),
        (big(7), divisor.clone(), big(0), big(7)),
        (dividend.clone(), divisor.clone(), quotient.clone(), big(7)),
        (-&dividend, divisor.clone(), -&quotient, big(-7)),
        (dividend.clone(), -&divisor, -&quotient, big(7)),
        (-&dividend, -&divisor, quotient.clone(), big(-7)),
        (
            &big(i64::MIN as i128) << 128i64,
            pow2(64),
            -pow2(127),
            big(0),
        ),
    ] {
        assert_eq!(a.div_rem(&b).unwrap(), (q, r), "{a} / {b}");
    }
    assert_eq!((-&exact).floor_div(&divisor).unwrap(), -&quotient);
    assert_eq!((-&exact).floor_mod(&divisor).unwrap(), big(0));
    assert_eq!((-&dividend).floor_div(&divisor).unwrap(), -&quotient - 1i64);
    assert_eq!((-&dividend).floor_mod(&divisor).unwrap(), &divisor - 7i64);
}

#[test]
fn test_big_int_multiword_division_estimates() {
    let base = pow2(32);
    let divisor = pow2(63) + &base - 1i64;
    let clamp_divisor = pow2(63) + 1i64;
    let addback_divisor = pow2(95) + 1i64;
    let shifted_divisor = pow2(64) + 1i64;
    let wide_quotient = pow2(64) + 7i64;
    let wide_remainder = pow2(63) + &base + 5i64;
    // Clamp, one/two estimate corrections, addback, and maximal normalization shift.
    for (a, b, q, r) in [
        (
            &clamp_divisor * &base - 1i64,
            clamp_divisor.clone(),
            &base - 1i64,
            pow2(63),
        ),
        (
            &divisor * 2i64 - 1i64,
            divisor.clone(),
            big(1),
            &divisor - 1i64,
        ),
        (
            &divisor * (&base - 2i64) - 1i64,
            divisor.clone(),
            &base - 3i64,
            &divisor - 1i64,
        ),
        (
            &addback_divisor * (&base + 1i64) - 1i64,
            addback_divisor.clone(),
            base.clone(),
            pow2(95),
        ),
        (
            &shifted_divisor * &wide_quotient + &wide_remainder,
            shifted_divisor.clone(),
            wide_quotient.clone(),
            wide_remainder.clone(),
        ),
        (&divisor - 1i64, divisor.clone(), big(0), &divisor - 1i64),
        (divisor.clone(), divisor.clone(), big(1), big(0)),
    ] {
        assert_eq!(a.div_rem(&b).unwrap(), (q, r), "{a} / {b}");
    }
}

#[test]
fn test_big_int_scalar_div_rem_boundaries() {
    let quotient = pow2(192) + 3i64;
    let dividend = &quotient * 3i64 + 2i64;
    let maximum_remainder = &quotient * pow2(63) + i64::MAX;
    let half_max = i64::from(u32::MAX);
    let half_boundary = (&BigInt::from(half_max) << 64i64) - 1i64;
    let word_quotient = BigInt::from(u64::MAX);
    let correction_divisor = 0x400000007fffffffi64;
    // Zero, exact/nonexact, signed scalar remainder, and the full |INT64_MIN| bound.
    for (a, b, q, r) in [
        (big(0), 3i64, big(0), 0i64),
        (big(2), 3, big(0), 2),
        (quotient.clone(), 1, quotient.clone(), 0),
        (-&quotient * 3i64, 3, -&quotient, 0),
        (&quotient * 3i64, 3, quotient.clone(), 0),
        (dividend.clone(), 3, quotient.clone(), 2),
        (-&dividend, 3, -&quotient, -2),
        (maximum_remainder.clone(), i64::MIN, -&quotient, i64::MAX),
        (-&maximum_remainder, i64::MIN, quotient.clone(), -i64::MAX),
        (big(i64::MIN as i128), -1, pow2(63), 0),
        (
            half_boundary.clone(),
            half_max,
            word_quotient.clone(),
            half_max - 1,
        ),
        (
            -&half_boundary,
            -half_max,
            word_quotient.clone(),
            1 - half_max,
        ),
        (
            (&BigInt::from(half_max + 1) << 64i64) - 1i64,
            half_max + 1,
            word_quotient.clone(),
            half_max,
        ),
        // One quotient correction, then two corrections with both overflow guards.
        (pow2(64), half_max + 2, BigInt::from(half_max), 1),
        (
            (&BigInt::from(correction_divisor) << 64i64) - 1i64,
            correction_divisor,
            word_quotient.clone(),
            correction_divisor - 1,
        ),
    ] {
        let (dq, dr) = a.div_rem(&BigInt::from(b)).unwrap();
        assert_eq!(dq, q, "{a} / {b}");
        assert_inline(&dr, r);
    }
    assert_eq!(&dividend / 3i64, quotient);
    assert_inline(&(-&dividend % 3i64), -2);
    assert_eq!((-&dividend).floor_div(&big(3)).unwrap(), -&quotient - 1i64);
    assert_inline(&(-&dividend).floor_mod(&big(3)).unwrap(), 1);
}

#[test]
fn test_big_int_small_divisor_remainders() {
    let value =
        pow2(255) | (&BigInt::from(0x12345678i64) << 128i64) | BigInt::from(0xfedcba9876543211u64);
    // Independent signed int32 boundary answers, plus an exact negative-divisor remainder.
    for (a, b, trunc, floor) in [
        (value.clone(), -2147483648i64, 1985229329i64, -162254319i64),
        (-&value, 2147483647, -391319368, 1756164279),
        (-&value, -2147483648, -1985229329, -1985229329),
        (value.clone(), 2147483647, 391319368, 391319368),
        (pow2(255), -2147483648, 0, 0),
    ] {
        assert_inline(&(&a % b), trunc);
        assert_inline(&a.floor_mod(&BigInt::from(b)).unwrap(), floor);
    }
    assert_eq!(
        value.try_rem(&big(0)).unwrap_err().kind(),
        ZERO_DIVISION_ERROR
    );
    assert_eq!(
        value.floor_mod(&big(0)).unwrap_err().kind(),
        ZERO_DIVISION_ERROR
    );
}

/// Deterministic xorshift64 generator for the differential test.
struct XorShift(u64);

impl XorShift {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// One to six words mixing carry-sensitive patterns with random bits.
    fn big_int(&mut self) -> BigInt {
        let len = (self.next() % 6 + 1) as usize;
        let words: Vec<i64> = (0..len)
            .map(|_| match self.next() % 8 {
                0 => 0,
                1 => -1,
                2 => i64::MIN,
                3 => i64::MAX,
                4 => 1,
                _ => self.next() as i64,
            })
            .collect();
        BigInt::from_words(&words)
    }
}

fn abs(value: &BigInt) -> BigInt {
    if value.is_negative() {
        -value
    } else {
        value.clone()
    }
}

#[test]
fn test_big_int_random_algebraic_properties() {
    let pool = [
        big(0),
        big(1),
        big(-1),
        big(i64::MIN as i128),
        big(i64::MAX as i128),
        pow2(63),
        -pow2(63),
        pow2(64),
        -pow2(64) - 1i64,
        BigInt::from(u64::MAX),
        pow2(128),
        big(i128::MIN),
        big(i128::MAX),
        BigInt::from(u128::MAX),
        wide(),
        -wide(),
        BigInt::from_words(&[0, 0, i64::MIN, 1]),
        pow2(255) + 13i64,
    ];
    let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);
    let pick = |rng: &mut XorShift| {
        if rng.next() % 4 == 0 {
            pool[(rng.next() % pool.len() as u64) as usize].clone()
        } else {
            rng.big_int()
        }
    };
    let largest_double = BigInt::try_from(f64::MAX).unwrap();
    for _ in 0..1500 {
        let a = pick(&mut rng);
        let b = pick(&mut rng);
        let context = format!("a = {a}, b = {b}");

        // Ring identities; every result is checked for canonical form by `==`.
        assert_eq!(&a + &b, &b + &a, "{context}");
        assert_eq!((&a + &b) - &b, a, "{context}");
        assert_eq!(&a - &b, &a + -&b, "{context}");
        assert_eq!(&a * &b, &b * &a, "{context}");
        assert_eq!(-(&a * &b), -&a * &b, "{context}");
        assert_eq!(!&a, -&a - 1i64, "{context}");
        assert_eq!(&a ^ &b, (&a | &b) - (&a & &b), "{context}");
        assert_eq!((&a & &b) + (&a | &b), &a + &b, "{context}");
        assert_eq!(a.cmp(&b), (&a - &b).cmp(&big(0)), "{context}");

        // Native integers agree wherever they can represent the operands and result.
        if let (Some(x), Some(y)) = (a.to_i128(), b.to_i128()) {
            for (expected, actual) in [
                (x.checked_add(y), &a + &b),
                (x.checked_sub(y), &a - &b),
                (x.checked_mul(y), &a * &b),
                (Some(x & y), &a & &b),
                (Some(x | y), &a | &b),
                (Some(x ^ y), &a ^ &b),
            ] {
                if let Some(expected) = expected {
                    assert_eq!(actual, BigInt::from(expected), "{context}");
                }
            }
            if y != 0 && !(x == i128::MIN && y == -1) {
                assert_eq!(a.try_div(&b).unwrap(), BigInt::from(x / y), "{context}");
                assert_eq!(a.try_rem(&b).unwrap(), BigInt::from(x % y), "{context}");
                assert_eq!(
                    a.floor_div(&b).unwrap(),
                    BigInt::from(x.div_euclid(y) - i128::from((y < 0) & (x.rem_euclid(y) != 0))),
                    "{context}"
                );
            }
        }

        // Division identities: a = q*b + r with the truncating and flooring sign rules.
        if b.is_zero() {
            for result in [
                a.try_div(&b),
                a.try_rem(&b),
                a.floor_div(&b),
                a.floor_mod(&b),
            ] {
                assert_eq!(result.unwrap_err().kind(), ZERO_DIVISION_ERROR);
            }
        } else {
            let (q, r) = a.div_rem(&b).unwrap();
            assert_eq!(q, a.try_div(&b).unwrap(), "{context}");
            assert_eq!(r, a.try_rem(&b).unwrap(), "{context}");
            assert_eq!(&q * &b + &r, a, "{context}");
            assert!(abs(&r) < abs(&b), "{context}");
            assert!(
                r.is_zero() || r.is_negative() == a.is_negative(),
                "{context}"
            );
            let (fq, fr) = (a.floor_div(&b).unwrap(), a.floor_mod(&b).unwrap());
            assert_eq!(&fq * &b + &fr, a, "{context}");
            assert!(abs(&fr) < abs(&b), "{context}");
            assert!(
                fr.is_zero() || fr.is_negative() == b.is_negative(),
                "{context}"
            );
            let rounds_down = !r.is_zero() && a.is_negative() != b.is_negative();
            assert_eq!(
                fq,
                if rounds_down { &q - 1i64 } else { q.clone() },
                "{context}"
            );
            assert_eq!((&a * &b).try_div(&b).unwrap(), a, "{context}");
            assert!((&a * &b).try_rem(&b).unwrap().is_zero(), "{context}");
        }

        // Shifts are multiplication and floor division by powers of two.
        let count = (rng.next() % 300) as i64;
        let shifted = a.try_shl(count).unwrap();
        assert_eq!(shifted, &a * pow2(count), "{context}, count = {count}");
        assert_eq!(
            shifted.try_shr(count).unwrap(),
            a,
            "{context}, count = {count}"
        );
        assert_eq!(
            a.try_shr(count).unwrap(),
            a.floor_div(&pow2(count)).unwrap(),
            "{context}, count = {count}"
        );
        assert_eq!(a.try_shl(-1).unwrap_err().kind(), VALUE_ERROR);
        assert_eq!(a.try_shr(-1).unwrap_err().kind(), VALUE_ERROR);

        // Text round-trips exactly; the double is the nearest representable value.
        assert_eq!(a.to_string().parse::<BigInt>().unwrap(), a, "{context}");
        match a.to_f64() {
            Ok(value) => {
                assert!(value.is_finite());
                let nearest = BigInt::try_from(value).unwrap();
                let difference = abs(&(&a - &nearest));
                let exponent = ((value.abs().to_bits() >> 52) & 0x7ff) as i64 - 1075;
                let ulp = if exponent > 0 { pow2(exponent) } else { big(1) };
                assert!(
                    &difference * 2i64 <= ulp,
                    "{context}: {value} is not the nearest double"
                );
                if abs(&a) < pow2(53) {
                    assert_eq!(nearest, a, "{context}");
                }
            }
            Err(error) => {
                assert_eq!(error.kind(), OVERFLOW_ERROR);
                assert!(abs(&a) > largest_double, "{context}");
            }
        }
    }
}

#[test]
fn test_big_int_native_objects_cross_runtime() {
    let echo = echo();
    // A Rust-allocated object: shared with a C++ container and released by C++ last.
    let product = &wide() * &wide();
    // wide < 2^256, so the square needs 512 bits: eight words with no sign guard.
    assert_eq!(product.words().len(), 8);
    assert_eq!(AnyView::from(&product).debug_strong_count(), Some(1));
    let array = Array::<BigInt>::try_from(
        Function::get_global("ffi.Array")
            .unwrap()
            .call_tuple((&product,))
            .unwrap(),
    )
    .unwrap();
    assert_eq!(AnyView::from(&product).debug_strong_count(), Some(2));
    let echoed = BigInt::try_from(echo.call_tuple((&product,)).unwrap()).unwrap();
    assert_eq!(echoed.words().as_ptr(), product.words().as_ptr());
    let pointer = product.words().as_ptr();
    drop(product);
    drop(echoed);
    let held = array.get(0).unwrap();
    assert_eq!(held.words().as_ptr(), pointer);
    assert_eq!(AnyView::from(&held).debug_strong_count(), Some(2));
    drop(array);
    assert_eq!(AnyView::from(&held).debug_strong_count(), Some(1));

    // The runtime allocates an equal twin from the Rust-built words.
    let from_runtime = {
        let words = held.words();
        let input = tvm_ffi_sys::TVMFFIByteArray::new(
            words.as_ptr() as *const u8,
            std::mem::size_of_val(words),
        );
        let mut raw = tvm_ffi_sys::TVMFFIAny::new();
        assert_eq!(
            unsafe { tvm_ffi_sys::TVMFFIBigIntFromByteArray(&input, &mut raw) },
            0
        );
        BigInt::try_from(unsafe { Any::from_raw_ffi_any(raw) }).unwrap()
    };
    assert_ne!(from_runtime.words().as_ptr(), held.words().as_ptr());
    assert_eq!(from_runtime, held);
    // C++ hashes and compares the Rust-built key against the runtime-built probe.
    let map: Map<BigInt, i64> = [(held.clone(), 1i64)].into_iter().collect();
    assert_eq!(map.get(&from_runtime).unwrap(), Some(1));
    // Rust arithmetic on the runtime-allocated operand; equal values demote to inline zero.
    assert_inline(&(&held - &from_runtime), 0);
    assert_eq!(&held + &from_runtime, &held << 1i64);
    drop(map);
    assert_eq!(AnyView::from(&held).debug_strong_count(), Some(1));
}
