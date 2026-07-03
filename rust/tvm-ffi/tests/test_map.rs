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

/// Helper to create a CPU `f32` tensor whose first element is `val`.
fn create_tensor(val: f32, shape: &[i64]) -> Tensor {
    let dtype = DLDataType::new(DLDataTypeCode::kDLFloat, 32, 1);
    let device = DLDevice::new(DLDeviceType::kDLCPU, 0);
    let tensor = Tensor::from_nd_alloc(CPUNDAlloc {}, shape, dtype, device);
    if let Ok(slice) = tensor.data_as_slice_mut::<f32>() {
        slice[0] = val;
    }
    tensor
}

/// Helper to read the first `f32` of a tensor.
fn get_val(tensor: &Tensor) -> f32 {
    tensor
        .data_as_slice::<f32>()
        .expect("Type mismatch or null")[0]
}

#[test]
fn test_map_basic_lookup() {
    let map: Map<i64, i64> = [(1i64, 10i64), (2, 20), (3, 30)].into_iter().collect();

    assert_eq!(map.len(), 3);
    assert!(!map.is_empty());

    assert_eq!(map.get(&1).unwrap(), Some(10));
    assert_eq!(map.get(&2).unwrap(), Some(20));
    assert_eq!(map.get(&3).unwrap(), Some(30));
    assert_eq!(map.get(&99).unwrap(), None);

    assert!(map.contains_key(&1));
    assert!(!map.contains_key(&99));
}

#[test]
fn test_map_empty() {
    let map: Map<i64, i64> = Map::new();
    assert_eq!(map.len(), 0);
    assert!(map.is_empty());
    assert_eq!(map.get(&1).unwrap(), None);
    // All three iterators must handle the empty (remaining == 0) case.
    assert_eq!(map.iter().count(), 0);
    assert_eq!(map.keys().count(), 0);
    assert_eq!(map.values().count(), 0);
}

/// `FromIterator` follows C++ `ffi.Map` last-wins semantics for duplicate keys,
/// so the resulting map is smaller than the input iterator.
#[test]
fn test_map_from_iter_duplicate_keys_last_wins() {
    let map: Map<i64, i64> = [(1i64, 10i64), (1, 20), (2, 30)].into_iter().collect();
    assert_eq!(map.len(), 2);
    assert_eq!(map.get(&1).unwrap(), Some(20));
    assert_eq!(map.get(&2).unwrap(), Some(30));
}

#[test]
fn test_map_iteration() {
    let map: Map<i64, i64> = [(1i64, 10i64), (2, 20), (3, 30)].into_iter().collect();

    let mut items: Vec<(i64, i64)> = map.iter().collect();
    items.sort();
    assert_eq!(items, vec![(1, 10), (2, 20), (3, 30)]);

    let mut keys: Vec<i64> = map.keys().collect();
    keys.sort();
    assert_eq!(keys, vec![1, 2, 3]);

    let mut values: Vec<i64> = map.values().collect();
    values.sort();
    assert_eq!(values, vec![10, 20, 30]);

    // `&map` IntoIterator yields the same pairs.
    let mut via_ref: Vec<(i64, i64)> = (&map).into_iter().collect();
    via_ref.sort();
    assert_eq!(via_ref, vec![(1, 10), (2, 20), (3, 30)]);
}

#[test]
fn test_map_string_keys() {
    let map: Map<String, i64> = [
        (String::from("a"), 1i64),
        (String::from("b"), 2),
        (String::from("c"), 3),
    ]
    .into_iter()
    .collect();

    assert_eq!(map.len(), 3);
    assert_eq!(map.get(&String::from("a")).unwrap(), Some(1));
    assert_eq!(map.get(&String::from("c")).unwrap(), Some(3));
    assert_eq!(map.get(&String::from("z")).unwrap(), None);
    assert!(map.contains_key(&String::from("b")));
}

#[test]
fn test_map_any_roundtrip() {
    let map: Map<i64, i64> = [(1i64, 10i64), (2, 20)].into_iter().collect();

    let any = Any::from(map.clone());
    assert_eq!(any.type_index(), TypeIndex::kTVMFFIMap as i32);

    let back: Map<i64, i64> = Map::try_from(any).expect("Any -> Map failed");
    assert_eq!(back.len(), 2);
    assert_eq!(back.get(&1).unwrap(), Some(10));
    assert_eq!(back.get(&2).unwrap(), Some(20));
}

#[test]
fn test_map_shares_underlying_object() {
    let map: Map<i64, i64> = [(1i64, 10i64)].into_iter().collect();
    // Cloning shares the same underlying MapObj rather than copying entries.
    let clone = map.clone();
    assert_eq!(clone.len(), 1);
    assert_eq!(clone.get(&1).unwrap(), Some(10));
}

#[test]
fn test_map_object_values_and_refcount() {
    let t = create_tensor(7.0, &[2, 3]);
    let base = AnyView::from(&t)
        .debug_strong_count()
        .expect("tensor is reference counted");
    assert_eq!(base, 1);

    // Building the map stores one internal reference to the tensor object.
    let map: Map<String, Tensor> = [(String::from("x"), t.clone())].into_iter().collect();
    assert_eq!(AnyView::from(&t).debug_strong_count().unwrap(), base + 1);

    // The value round-trips correctly and `get` hands back a fresh handle.
    let got = map
        .get(&String::from("x"))
        .unwrap()
        .expect("key should be present");
    assert_eq!(get_val(&got), 7.0);
    assert_eq!(got.ndim(), 2);
    assert_eq!(AnyView::from(&t).debug_strong_count().unwrap(), base + 2);

    // Iteration yields object handles too.
    let collected: Vec<(String, Tensor)> = map.iter().collect();
    assert_eq!(collected.len(), 1);
    assert_eq!(get_val(&collected[0].1), 7.0);
    // Peak: `t` + the map's internal ref + `got` + the iterated handle.
    assert_eq!(AnyView::from(&t).debug_strong_count().unwrap(), base + 3);

    // Dropping the borrowed handles restores the count to just `t` + the map.
    drop(collected);
    drop(got);
    assert_eq!(AnyView::from(&t).debug_strong_count().unwrap(), base + 1);

    // Dropping the map releases its internal reference; only `t` remains.
    drop(map);
    assert_eq!(AnyView::from(&t).debug_strong_count().unwrap(), base);
}

#[test]
fn test_map_as_function_argument() {
    let map: Map<i64, i64> = [(1i64, 10i64), (2, 20), (3, 30)].into_iter().collect();

    // Exercises the `ArgIntoRef`/`call_tuple` path, by value and by reference.
    let map_size = Function::get_global("ffi.MapSize").unwrap();
    let n_by_value: i64 = map_size
        .call_tuple((map.clone(),))
        .unwrap()
        .try_into()
        .unwrap();
    assert_eq!(n_by_value, 3);
    let n_by_ref: i64 = map_size.call_tuple((&map,)).unwrap().try_into().unwrap();
    assert_eq!(n_by_ref, 3);

    // Two-argument call mixing `&Map` with a key value.
    let get_item = Function::get_global("ffi.MapGetItem").unwrap();
    let v: i64 = get_item
        .call_tuple((&map, 2i64))
        .unwrap()
        .try_into()
        .unwrap();
    assert_eq!(v, 20);
}

/// `get` distinguishes a present object value from an absent key even when `V`
/// is object-typed (an absent lookup must not be miscast into a `V`).
#[test]
fn test_map_get_missing_with_object_values() {
    let map: Map<String, Tensor> = [(String::from("x"), create_tensor(7.0, &[2]))]
        .into_iter()
        .collect();

    let present = map.get(&String::from("x")).unwrap();
    assert!(present.is_some());
    assert_eq!(get_val(&present.unwrap()), 7.0);

    // Absent key returns `None`, never an object miscast as a tensor.
    assert!(map.get(&String::from("missing")).unwrap().is_none());
}

/// Views a map behind an `Any` as `Map<K, V>` while bypassing the strict
/// element check (`try_from` now walks all entries, C++-style, so a mistyped
/// view can only be produced this way), to exercise the access-time guards.
fn view_map_unchecked<K, V>(mut any: Any) -> Map<K, V>
where
    K: AnyCompatible,
    V: AnyCompatible,
{
    assert_eq!(any.type_index(), TypeIndex::kTVMFFIMap as i32);
    unsafe { Map::copy_from_any_view_after_check(&*any.as_data_ptr()) }
}

/// Builds an `i64`-valued map but views it as `Map<_, String>`, bypassing the
/// strict check, so the mismatch only surfaces on access.
fn mistyped_value_map() -> Map<i64, String> {
    let real: Map<i64, i64> = [(1i64, 10i64)].into_iter().collect();
    view_map_unchecked(Any::from(real))
}

/// `get` reports a value type mismatch as `Err`, not a panic.
#[test]
fn test_map_get_type_mismatch_is_err() {
    let map = mistyped_value_map();
    assert!(map.get(&1).is_err());
}

/// Iterators panic on a value type mismatch rather than silently truncating
/// (which would violate their `ExactSizeIterator` contract).
#[test]
#[should_panic(expected = "value does not match")]
fn test_map_iter_type_mismatch_panics() {
    let map = mistyped_value_map();
    let _ = map.values().next();
}

/// A key whose type does not match the map's key type reads as absent (keys are
/// hashed, not retrieved, so a pure lookup can't distinguish it from a real
/// miss). In debug builds the misuse is caught by an assertion on the miss path.
#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "does not match the map's stored key type")]
fn test_map_get_mistyped_key_debug_asserts() {
    // Real `Map<i64, i64>` viewed (unchecked) as `Map<String, i64>`: the i64
    // keys do not match `K = String`.
    let real: Map<i64, i64> = [(1i64, 10i64)].into_iter().collect();
    let map: Map<String, i64> = view_map_unchecked(Any::from(real));
    let _ = map.get(&String::from("anything"));
}

/// `try_from` rejects a map whose values can neither strictly match nor cast
/// to `V` (aligned with C++ `CheckAnyStrict`/`TryCastFromAnyView`).
#[test]
fn test_map_any_cast_value_mismatch_is_err() {
    let real: Map<i64, i64> = [(1i64, 10i64)].into_iter().collect();
    assert!(Map::<i64, String>::try_from(Any::from(real)).is_err());
}

/// When entries do not strictly match but are castable (i64 -> f64),
/// `try_from` builds a NEW map with converted entries instead of sharing the
/// object (C++ slow path).
#[test]
fn test_map_any_cast_slow_path_converts() {
    let real: Map<i64, i64> = [(1i64, 10i64), (2, 20)].into_iter().collect();
    let converted: Map<f64, f64> =
        Map::try_from(Any::from(real.clone())).expect("i64 entries cast to f64");
    assert_eq!(converted.len(), 2);
    assert_eq!(converted.get(&1.0).unwrap(), Some(10.0));
    assert_eq!(converted.get(&2.0).unwrap(), Some(20.0));
    // Each map owns its own object: nothing is shared between them.
    assert_eq!(AnyView::from(&real).debug_strong_count(), Some(1));
    assert_eq!(AnyView::from(&converted).debug_strong_count(), Some(1));
}
