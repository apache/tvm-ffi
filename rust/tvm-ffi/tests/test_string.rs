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

// ============================================================================
// Bytes Tests
// ============================================================================

#[test]
fn test_bytes_new() {
    let bytes = Bytes::new();
    assert_eq!(bytes.len(), 0);
    assert_eq!(bytes.as_slice(), &[]);
}

#[test]
fn test_bytes_default() {
    let bytes = Bytes::default();
    assert_eq!(bytes.len(), 0);
    assert_eq!(bytes.as_slice(), &[]);
}

#[test]
fn test_bytes_from_small() {
    let data = b"hello";
    let bytes = Bytes::from(data.as_slice());
    assert_eq!(bytes.len(), 5);
    assert_eq!(bytes.as_slice(), data);
}

#[test]
fn test_bytes_from_large() {
    let data = b"this is a very long string that exceeds the small bytes limit";
    let bytes = Bytes::from(data.as_slice());
    assert_eq!(
        AnyView::from(&bytes).type_index(),
        TypeIndex::kTVMFFIBytes as i32
    );
    assert_eq!(bytes.len(), data.len());
    assert_eq!(bytes.as_slice(), data);
}

#[test]
fn test_bytes_deref() {
    let data = b"test";
    let bytes = Bytes::from(data.as_slice());
    assert_eq!(&*bytes, data);
}

#[test]
fn test_bytes_clone_basic() {
    let data = b"clone test";
    let bytes1 = Bytes::from(data.as_slice());
    let bytes2 = bytes1.clone();
    assert_eq!(bytes1, bytes2);
    assert_eq!(bytes1.as_slice(), bytes2.as_slice());
}

#[test]
fn test_bytes_eq() {
    let data = b"equal";
    let bytes1 = Bytes::from(data.as_slice());
    let bytes2 = Bytes::from(data.as_slice());
    assert_eq!(bytes1, bytes2);

    let bytes3 = Bytes::from(b"different");
    assert_ne!(bytes1, bytes3);
}

#[test]
fn test_bytes_ord() {
    let bytes1 = Bytes::from(b"abc");
    let bytes2 = Bytes::from(b"def");
    let bytes3 = Bytes::from(b"abc");

    assert!(bytes1 < bytes2);
    assert!(bytes1 <= bytes2);
    assert!(bytes2 > bytes1);
    assert!(bytes2 >= bytes1);
    assert_eq!(bytes1.cmp(&bytes3), std::cmp::Ordering::Equal);
}

#[test]
fn test_bytes_hash() {
    let data = b"hash test";
    let bytes1 = Bytes::from(data.as_slice());
    let bytes2 = Bytes::from(data.as_slice());

    let mut hasher1 = DefaultHasher::new();
    let mut hasher2 = DefaultHasher::new();
    bytes1.hash(&mut hasher1);
    bytes2.hash(&mut hasher2);

    assert_eq!(hasher1.finish(), hasher2.finish());
}

// ============================================================================
// String Tests
// ============================================================================

#[test]
fn test_string_new() {
    let s = String::new();
    assert_eq!(s.len(), 0);
    assert_eq!(s.as_str(), "");
    assert_eq!(s.as_bytes(), &[]);
}

#[test]
fn test_string_default() {
    let s = String::default();
    assert_eq!(s.len(), 0);
    assert_eq!(s.as_str(), "");
}

#[test]
fn test_string_from_small() {
    let input = "hello";
    let s = String::from(input);
    assert_eq!(s.len(), 5);
    assert_eq!(s.as_str(), input);
    assert_eq!(s.as_bytes(), input.as_bytes());
}

#[test]
fn test_string_from_large() {
    let input = "this is a very long string that exceeds the small string limit";
    let s = String::from(input);
    assert_eq!(s.len(), input.len());
    assert_eq!(s.as_str(), input);
    assert_eq!(s.as_bytes(), input.as_bytes());
}

#[test]
fn test_string_deref() {
    let input = "deref test";
    let s = String::from(input);
    assert_eq!(&*s, input);
}

#[test]
fn test_string_clone_basic() {
    let input = "clone test";
    let s1 = String::from(input);
    let s2 = s1.clone();
    assert_eq!(s1, s2);
    assert_eq!(s1.as_str(), s2.as_str());
}

#[test]
fn test_string_eq() {
    let input = "equal";
    let s1 = String::from(input);
    let s2 = String::from(input);
    assert_eq!(s1, s2);

    let s3 = String::from("different");
    assert_ne!(s1, s3);
}

#[test]
fn test_string_ord() {
    let s1 = String::from("abc");
    let s2 = String::from("def");
    let s3 = String::from("abc");

    assert!(s1 < s2);
    assert!(s1 <= s2);
    assert!(s2 > s1);
    assert!(s2 >= s1);
    assert_eq!(s1.cmp(&s3), std::cmp::Ordering::Equal);
}

#[test]
fn test_string_hash() {
    let input = "hash test";
    let s1 = String::from(input);
    let s2 = String::from(input);

    let mut hasher1 = DefaultHasher::new();
    let mut hasher2 = DefaultHasher::new();
    s1.hash(&mut hasher1);
    s2.hash(&mut hasher2);

    assert_eq!(hasher1.finish(), hasher2.finish());
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_empty_strings() {
    let empty_str = String::from("");
    assert_eq!(empty_str.len(), 0);
    assert_eq!(empty_str.as_str(), "");

    let empty_bytes = Bytes::from(&[]);
    assert_eq!(empty_bytes.len(), 0);
    assert_eq!(empty_bytes.as_slice(), &[]);
}

#[test]
fn test_unicode_strings() {
    let unicode = "\u{1F680} Hello \u{4E16}\u{754C} \u{1F30D}";
    let s = String::from(unicode);
    assert_eq!(s.as_str(), unicode);
    assert_eq!(s.len(), unicode.len());
}

#[test]
fn test_small_vs_large_boundary() {
    // Test exactly at the boundary (7 bytes)
    let boundary = "1234567"; // 7 characters = 7 bytes
    let s = String::from(boundary);
    assert_eq!(s.as_str(), boundary);

    // Test just over the boundary (8 bytes)
    let over_boundary = "12345678"; // 8 characters = 8 bytes
    let s2 = String::from(over_boundary);
    assert_eq!(s2.as_str(), over_boundary);
}

#[test]
fn test_bytes_small_vs_large_boundary() {
    // Test exactly at the boundary (7 bytes)
    let boundary = b"1234567"; // 7 bytes
    let bytes = Bytes::from(boundary.as_slice());
    assert_eq!(bytes.as_slice(), boundary);

    // Test just over the boundary (8 bytes)
    let over_boundary = b"12345678"; // 8 bytes
    let bytes2 = Bytes::from(over_boundary.as_slice());
    assert_eq!(bytes2.as_slice(), over_boundary);
}

#[test]
fn test_mixed_comparisons() {
    let s1 = String::from("test");
    let s2 = String::from("test");
    let s3 = String::from("different");

    // Test all comparison operators
    assert!(s1 == s2);
    assert!(s1 != s3);
    assert!(s1 <= s2);
    assert!(s1 >= s2);
    assert!(s1 > s3);
    assert!(s3 < s1);
}

#[test]
fn test_bytes_mixed_comparisons() {
    let b1 = Bytes::from(b"test");
    let b2 = Bytes::from(b"test");
    let b3 = Bytes::from(b"different");

    // Test all comparison operators
    assert!(b1 == b2);
    assert!(b1 != b3);
    assert!(b1 <= b2);
    assert!(b1 >= b2);
    assert!(b1 > b3);
    assert!(b3 < b1);
}

#[test]
fn test_bytes_debug() {
    let data = b"hello";
    let bytes = Bytes::from(data.as_slice());
    let debug_str = format!("{:?}", bytes);
    assert!(debug_str.contains("Bytes"));
}

#[test]
fn test_string_debug() {
    let input = "world";
    let s = String::from(input);
    let debug_str = format!("{:?}", s);
    assert!(debug_str.contains("String"));
    assert!(debug_str.contains("world"));
}

#[test]
fn test_bytes_clone_small() {
    let data = b"small"; // 5 bytes - small case
    let bytes1 = Bytes::from(data.as_slice());
    let bytes2 = bytes1.clone();

    assert_eq!(bytes1, bytes2);
    assert_eq!(bytes1.as_slice(), bytes2.as_slice());
    assert_eq!(bytes1.len(), bytes2.len());

    // For small bytes, the underlying data should be different (copied)
    // We can verify this by checking that the slices have different pointers
    let slice1 = bytes1.as_slice();
    let slice2 = bytes2.as_slice();
    assert_ne!(slice1.as_ptr(), slice2.as_ptr());
}

#[test]
fn test_bytes_clone_large() {
    let data = b"this is a very long string that exceeds the small bytes limit"; // large case
    let bytes1 = Bytes::from(data.as_slice());
    let bytes2 = bytes1.clone();

    assert_eq!(bytes1, bytes2);
    assert_eq!(bytes1.as_slice(), bytes2.as_slice());
    assert_eq!(bytes1.len(), bytes2.len());

    // For large bytes, the underlying data should be the same (shared)
    // We can verify this by checking that the slices have the same pointer
    let slice1 = bytes1.as_slice();
    let slice2 = bytes2.as_slice();
    assert_eq!(slice1.as_ptr(), slice2.as_ptr());
}

#[test]
fn test_string_clone_small() {
    let input = "small"; // 5 bytes - small case
    let s1 = String::from(input);
    assert!(AnyView::from(&s1).debug_strong_count().is_none());
    let s2 = s1.clone();

    assert_eq!(s1, s2);
    assert_eq!(s1.as_str(), s2.as_str());
    assert_eq!(s1.len(), s2.len());

    // For small strings, the underlying data should be different (copied)
    // We can verify this by checking that the byte slices have different pointers
    let bytes1 = s1.as_bytes();
    let bytes2 = s2.as_bytes();
    assert_ne!(bytes1.as_ptr(), bytes2.as_ptr());
}

#[test]
fn test_string_clone_large() {
    // large case
    let input = "this is a very long string that exceeds the small string limit";
    let s1 = String::from(input);
    assert_eq!(AnyView::from(&s1).debug_strong_count().unwrap(), 1);
    let s2 = s1.clone();
    assert_eq!(AnyView::from(&s1).debug_strong_count().unwrap(), 2);
    assert_eq!(AnyView::from(&s2).debug_strong_count().unwrap(), 2);

    assert_eq!(s1, s2);
    assert_eq!(s1.as_str(), s2.as_str());
    assert_eq!(s1.len(), s2.len());

    // For large strings, the underlying data should be the same (shared)
    // We can verify this by checking that the byte slices have the same pointer
    let bytes1 = s1.as_bytes();
    let bytes2 = s2.as_bytes();
    assert_eq!(bytes1.as_ptr(), bytes2.as_ptr());
}
