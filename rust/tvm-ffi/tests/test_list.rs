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

/// Address of the element buffer and of its deleter.
fn buffer_identity<T>(list: &List<T>) -> (usize, usize) {
    let obj = <List<T> as ObjectRefCore>::data(list);
    (
        obj.data as usize,
        obj.data_deleter.map(|f| f as usize).unwrap_or(0),
    )
}

#[test]
fn test_list_basic() {
    let list: List<i64> = [1i64, 2, 3].into_iter().collect();
    assert_eq!(list.len(), 3);
    assert!(!list.is_empty());
    assert_eq!(list.get(0).unwrap(), 1);
    assert_eq!(list.get(2).unwrap(), 3);
    assert!(list.get(3).is_err());
    assert_eq!(
        list.iter().collect::<Result<Vec<_>>>().unwrap(),
        vec![1, 2, 3]
    );
    assert_eq!(format!("{list:?}"), "List<i64>[3]");
}

#[test]
fn test_list_empty() {
    let list: List<i64> = List::new();
    assert_eq!(list.len(), 0);
    assert!(list.is_empty());
    assert!(list.get(0).is_err());
    assert_eq!(list.iter().count(), 0);
    assert!(List::<String>::default().is_empty());
}

#[test]
fn test_list_mutation() {
    let list: List<i64> = List::new();
    // Grows past any initial capacity, so the element buffer is reallocated.
    for i in 0..100 {
        list.push(i).unwrap();
    }
    assert_eq!(list.len(), 100);
    assert!(list.capacity() >= 100);
    assert_eq!(list.get(99).unwrap(), 99);

    list.set(0, -1).unwrap();
    assert_eq!(list.get(0).unwrap(), -1);
    assert!(list.set(100, 0).is_err());

    list.insert(1, 7).unwrap();
    assert_eq!(list.get(1).unwrap(), 7);
    assert_eq!(list.get(2).unwrap(), 1);
    list.insert(list.len(), 8).unwrap();
    assert!(list.insert(list.len() + 1, 0).is_err());

    assert_eq!(list.remove(1).unwrap(), 7);
    assert_eq!(list.remove(list.len() - 1).unwrap(), 8);
    assert_eq!(list.len(), 100);
    assert!(list.remove(100).is_err());

    list.clear();
    assert!(list.is_empty());
    assert!(list.capacity() >= 100);
}

#[test]
fn test_list_growth_stays_with_runtime() {
    let list: List<i64> = List::new();
    let (_, deleter) = buffer_identity(&list);
    assert_ne!(deleter, 0);

    // Within capacity the cells are written in place: same buffer.
    while list.len() < list.capacity() {
        list.push(list.len() as i64).unwrap();
    }
    let (buffer, _) = buffer_identity(&list);
    assert_eq!(buffer_identity(&list), (buffer, deleter));

    // Full: C++ reallocates (new buffer) and keeps its own deleter.
    list.push(list.len() as i64).unwrap();
    let (grown, deleter_after) = buffer_identity(&list);
    assert_ne!(grown, buffer);
    assert_eq!(deleter_after, deleter);

    // Same for a full insert, and for C++ growing it further.
    while list.len() < list.capacity() {
        list.push(list.len() as i64).unwrap();
    }
    list.insert(0, -1).unwrap();
    assert_eq!(buffer_identity(&list).1, deleter);
    let append = Function::get_global("ffi.ListAppend").unwrap();
    for i in 0..50 {
        append.call_tuple((list.clone(), 1000 + i)).unwrap();
    }
    assert_eq!(buffer_identity(&list).1, deleter);

    let mut expected: Vec<i64> = vec![-1];
    expected.extend(0..(list.len() as i64 - 51));
    expected.extend(1000..1050);
    assert_eq!(list.iter().collect::<Result<Vec<_>>>().unwrap(), expected);
}

#[test]
fn test_list_releases_elements() {
    let array = Array::new(vec![1i64]);
    let probe = Any::from(array.clone());
    let base = probe.debug_strong_count().unwrap();

    let list: List<Array<i64>> = List::new();
    list.push(array.clone()).unwrap();
    list.push(array.clone()).unwrap();
    list.insert(0, array.clone()).unwrap();
    assert_eq!(probe.debug_strong_count().unwrap(), base + 3);

    list.set(0, Array::new(vec![2i64])).unwrap();
    assert_eq!(probe.debug_strong_count().unwrap(), base + 2);
    drop(list.remove(1).unwrap());
    assert_eq!(probe.debug_strong_count().unwrap(), base + 1);
    list.clear();
    assert_eq!(probe.debug_strong_count().unwrap(), base);

    // Bulk edits release elements too.
    list.extend([array.clone(), array.clone(), array.clone()])
        .unwrap();
    assert_eq!(probe.debug_strong_count().unwrap(), base + 3);
    list.remove_range(0, 2).unwrap();
    assert_eq!(probe.debug_strong_count().unwrap(), base + 1);
    list.reverse();
    list.truncate(0);
    assert_eq!(probe.debug_strong_count().unwrap(), base);

    // Elements pushed through the C++ growth path are released by the C++
    // destructor; elements written in place by Rust the same way.
    for _ in 0..10 {
        list.push(array.clone()).unwrap();
    }
    assert_eq!(probe.debug_strong_count().unwrap(), base + 10);
    drop(list);
    assert_eq!(probe.debug_strong_count().unwrap(), base);
}

#[test]
fn test_list_handles_share_one_object() {
    let list: List<String> = List::new();
    let alias = list.clone();
    alias.push(String::from("a")).unwrap();
    assert_eq!(list.len(), 1);

    // Through `Any` and back: still the same list.
    let back: List<String> = Any::from(list.clone()).try_into().unwrap();
    back.push(String::from("b")).unwrap();
    assert_eq!(list.len(), 2);
    assert_eq!(list.get(1).unwrap().as_str(), "b");

    // The C++ runtime mutates the same object.
    let append = Function::get_global("ffi.ListAppend").unwrap();
    append
        .call_tuple((list.clone(), String::from("c")))
        .unwrap();
    assert_eq!(list.len(), 3);
    assert_eq!(list.get(2).unwrap().as_str(), "c");
}

#[test]
fn test_list_reserve() {
    let list: List<i64> = (1..=3).collect();
    let (buffer, deleter) = buffer_identity(&list);

    // Within capacity: nothing happens.
    list.reserve(0).unwrap();
    assert_eq!(buffer_identity(&list).0, buffer);

    // Beyond: C++ reallocates once, keeps its deleter, and the pushes that
    // follow land in place.
    list.reserve(100).unwrap();
    assert!(list.capacity() >= 103);
    let (grown, deleter_after) = buffer_identity(&list);
    assert_ne!(grown, buffer);
    assert_eq!(deleter_after, deleter);
    assert_eq!(
        list.iter().collect::<Result<Vec<_>>>().unwrap(),
        vec![1, 2, 3]
    );
    for i in 4..=103 {
        list.push(i).unwrap();
    }
    assert_eq!(buffer_identity(&list).0, grown);
    assert_eq!(list.len(), 103);
}

#[test]
fn test_list_ends() {
    let list: List<i64> = List::new();
    assert!(list.first().is_err());
    assert!(list.last().is_err());
    assert!(list.pop().is_err());
    list.extend([1i64, 2, 3]).unwrap();
    assert_eq!(list.first().unwrap(), 1);
    assert_eq!(list.last().unwrap(), 3);
    assert_eq!(list.pop().unwrap(), 3);
    assert_eq!(list.pop().unwrap(), 2);
    assert_eq!(list.len(), 1);
}

#[test]
fn test_list_bulk_edits() {
    // Grow once so the buffer has headroom, then make room again.
    let list: List<i64> = List::new();
    list.extend(1..=5).unwrap();
    list.truncate(3);
    assert!(list.capacity() > list.len());

    // In place: the buffer stays.
    let buffer = buffer_identity(&list).0;
    list.insert_many(1, [10i64]).unwrap();
    assert_eq!(buffer_identity(&list).0, buffer);
    assert_eq!(
        list.iter().collect::<Result<Vec<_>>>().unwrap(),
        vec![1, 10, 2, 3]
    );

    // Full: C++ reallocates through `ffi.ListReserve`, then the batch lands in place.
    while list.len() < list.capacity() {
        list.push(0).unwrap();
    }
    let len = list.len();
    list.insert_many(2, [20i64, 21]).unwrap();
    assert_eq!(list.len(), len + 2);
    assert_eq!(list.get(1).unwrap(), 10);
    assert_eq!(list.get(2).unwrap(), 20);
    assert_eq!(list.get(3).unwrap(), 21);
    assert_eq!(list.get(4).unwrap(), 2);
    list.extend(std::iter::empty()).unwrap();
    assert!(list.insert_many(list.len() + 1, [0i64]).is_err());

    list.remove_range(2, 4).unwrap();
    assert_eq!(list.len(), len);
    assert_eq!(list.get(2).unwrap(), 2);
    list.remove_range(1, 1).unwrap();
    assert!(list.remove_range(3, 2).is_err());
    assert!(list.remove_range(0, list.len() + 1).is_err());

    list.truncate(3);
    assert_eq!(
        list.iter().collect::<Result<Vec<_>>>().unwrap(),
        vec![1, 10, 2]
    );
    list.truncate(10);
    assert_eq!(list.len(), 3);
    list.resize(1, 0).unwrap();
    list.resize(4, 9).unwrap();
    assert_eq!(
        list.iter().collect::<Result<Vec<_>>>().unwrap(),
        vec![1, 9, 9, 9]
    );
    list.reverse();
    assert_eq!(
        list.iter().collect::<Result<Vec<_>>>().unwrap(),
        vec![9, 9, 9, 1]
    );
    list.clear();
    list.reverse();
    assert!(list.is_empty());
}

#[test]
fn test_list_insert_many_with_self_mutating_iterator() {
    // The iterator pushes into the same list while `insert_many` drains it;
    // the length must be read afterwards or the shift and size go stale.
    let list: List<i64> = List::new();
    list.push(0).unwrap();
    list.insert_many(
        0,
        (1..=3).map(|i| {
            list.push(100 + i).unwrap();
            i
        }),
    )
    .unwrap();
    assert_eq!(list.len(), 7);
    assert!(list.len() <= list.capacity());
    assert_eq!(
        list.iter().collect::<Result<Vec<_>>>().unwrap(),
        vec![1, 2, 3, 0, 101, 102, 103]
    );

    // `extend` appends after whatever the iterator added.
    list.extend((1..=2).map(|i| {
        list.push(200 + i).unwrap();
        i
    }))
    .unwrap();
    assert_eq!(
        list.iter().collect::<Result<Vec<_>>>().unwrap(),
        vec![1, 2, 3, 0, 101, 102, 103, 201, 202, 1, 2]
    );
}

#[test]
fn test_list_iter_rev() {
    let list: List<i64> = (1..=5).collect();
    assert_eq!(
        list.iter().rev().collect::<Result<Vec<_>>>().unwrap(),
        vec![5, 4, 3, 2, 1]
    );
    let mut it = list.iter();
    assert_eq!(it.next().unwrap().unwrap(), 1);
    assert_eq!(it.next_back().unwrap().unwrap(), 5);
    assert_eq!(it.next_back().unwrap().unwrap(), 4);
    assert_eq!(it.collect::<Result<Vec<_>>>().unwrap(), vec![2, 3]);

    // Shrinking the list while iterating from the back stays in bounds.
    let mut it = list.iter().rev();
    assert_eq!(it.next().unwrap().unwrap(), 5);
    list.truncate(2);
    assert_eq!(it.next().unwrap().unwrap(), 2);
    assert_eq!(it.next().unwrap().unwrap(), 1);
    assert!(it.next().is_none());
}

#[test]
fn test_list_iter_during_mutation() {
    let list: List<i64> = [1i64, 2, 3].into_iter().collect();
    let mut seen = Vec::new();
    for item in &list {
        let item = item.unwrap();
        seen.push(item);
        if item == 1 {
            list.remove(list.len() - 1).unwrap();
        }
    }
    assert_eq!(seen, vec![1, 2]);
}

#[test]
fn test_list_of_any_and_objects() {
    let list: List<Any> = List::new();
    list.push(Any::from(1i64)).unwrap();
    list.push(Any::from(String::from("two"))).unwrap();
    list.push(Any::from(Array::new(vec![3i64]))).unwrap();
    assert_eq!(list.len(), 3);
    assert_eq!(i64::try_from(list.get(0).unwrap()).unwrap(), 1);
    let nested: Array<i64> = list.get(2).unwrap().try_into().unwrap();
    assert_eq!(nested.get(0).unwrap(), 3);

    // Typed views of the same object: shared when the elements match ...
    let ints: List<i64> = [1i64, 2].into_iter().collect();
    let as_any: List<Any> = Any::from(ints.clone()).try_into().unwrap();
    as_any.push(Any::from(3i64)).unwrap();
    assert_eq!(ints.len(), 3);
    // ... and rejected when one does not.
    assert!(List::<i64>::try_from(Any::from(list)).is_err());
}

#[test]
fn test_list_mismatched_element_is_reported() {
    // Another handle to the same object puts a `String` where an `i64` is expected.
    let ints: List<i64> = [1i64, 2, 3].into_iter().collect();
    let as_any: List<Any> = Any::from(ints.clone()).try_into().unwrap();
    as_any.set(1, Any::from(String::from("x"))).unwrap();

    // `get` and the iterator report that element instead of truncating.
    assert!(ints.get(1).is_err());
    let items: Vec<Result<i64>> = ints.iter().collect();
    assert_eq!(items.len(), 3);
    assert_eq!(*items[0].as_ref().unwrap(), 1);
    assert!(items[1].is_err());
    assert_eq!(*items[2].as_ref().unwrap(), 3);
    assert!(ints.iter().collect::<Result<Vec<_>>>().is_err());
    assert!(ints.iter().sum::<Result<i64>>().is_err());

    // `remove` of that element fails and leaves the list untouched.
    assert!(ints.remove(1).is_err());
    assert_eq!(ints.len(), 3);
    let kept: String = as_any.get(1).unwrap().try_into().unwrap();
    assert_eq!(kept.as_str(), "x");
    assert_eq!(ints.remove(2).unwrap(), 3);
    assert_eq!(ints.remove(0).unwrap(), 1);
    assert_eq!(ints.len(), 1);

    // A convertible element (an int cell read as `f64`) is removed and converted.
    let floats: List<f64> = [1.5f64].into_iter().collect();
    let alias: List<Any> = Any::from(floats.clone()).try_into().unwrap();
    alias.push(Any::from(2i64)).unwrap();
    assert_eq!(floats.remove(1).unwrap(), 2.0);
    assert_eq!(floats.len(), 1);
}

#[test]
fn test_list_conversion_copies() {
    // An `ffi.Array` converts into a new list.
    let array = Array::new(vec![1i64, 2]);
    let list: List<i64> = Any::from(array.clone()).try_into().unwrap();
    list.push(3).unwrap();
    assert_eq!(list.len(), 3);
    assert_eq!(array.len(), 2);

    // Elements that need converting (int -> float) also yield a new list.
    let ints: List<i64> = [1i64, 2].into_iter().collect();
    let floats: List<f64> = Any::from(ints.clone()).try_into().unwrap();
    floats.push(0.5).unwrap();
    assert_eq!(floats.get(0).unwrap(), 1.0);
    assert_eq!(ints.len(), 2);

    // A non-sequence is not a list.
    assert!(List::<i64>::try_from(Any::from(1i64)).is_err());
}

#[test]
fn test_list_converts_to_array() {
    // The other direction of C++ `SeqTypeTraitsBase`: an `ffi.List` becomes a
    // new `Array`, never a shared one.
    let list: List<i64> = [1i64, 2, 3].into_iter().collect();
    let array: Array<i64> = Any::from(list.clone()).try_into().unwrap();
    assert_eq!(array.len(), 3);
    assert_eq!(array.get(2).unwrap(), 3);
    list.push(4).unwrap();
    assert_eq!(array.len(), 3);

    // Elements convert on the way (int -> float), mismatches are rejected.
    let floats: Array<f64> = Any::from(list.clone()).try_into().unwrap();
    assert_eq!(floats.get(3).unwrap(), 4.0);
    let mixed: List<Any> = List::new();
    mixed.push(Any::from(1i64)).unwrap();
    mixed.push(Any::from(String::from("x"))).unwrap();
    assert!(Array::<i64>::try_from(Any::from(mixed.clone())).is_err());
    let as_any: Array<Any> = Any::from(mixed).try_into().unwrap();
    assert_eq!(as_any.len(), 2);

    // The common case: a list handed to a function typed with `Array<T>`.
    let sum = Function::from_typed(|items: Array<i64>| -> Result<i64> { Ok(items.iter().sum()) });
    let total: i64 = sum.call_tuple((list.clone(),)).unwrap().try_into().unwrap();
    assert_eq!(total, 10);
}

#[test]
fn test_list_as_function_argument() {
    let sum = Function::from_typed(|list: List<i64>| -> Result<i64> {
        list.push(10)?;
        list.iter().sum()
    });
    let list: List<i64> = [1i64, 2].into_iter().collect();
    let total: i64 = sum.call_tuple((list.clone(),)).unwrap().try_into().unwrap();
    assert_eq!(total, 13);
    assert_eq!(list.len(), 3);
}
