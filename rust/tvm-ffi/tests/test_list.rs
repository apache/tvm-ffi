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

#[test]
fn test_list_basic() {
    let list: List<i64> = [1i64, 2, 3].into_iter().collect();
    assert_eq!(list.len(), 3);
    assert!(!list.is_empty());
    assert_eq!(list.get(0).unwrap(), 1);
    assert_eq!(list.get(2).unwrap(), 3);
    assert!(list.get(3).is_err());
    assert_eq!(list.iter().collect::<Vec<_>>(), vec![1, 2, 3]);
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

    list.clear().unwrap();
    assert!(list.is_empty());
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
fn test_list_iter_during_mutation() {
    let list: List<i64> = [1i64, 2, 3].into_iter().collect();
    let mut seen = Vec::new();
    for item in &list {
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

    // A list is not an `Array`, and a non-sequence is not a list.
    assert!(List::<i64>::try_from(Any::from(1i64)).is_err());
}

#[test]
fn test_list_as_function_argument() {
    let sum = Function::from_typed(|list: List<i64>| -> Result<i64> {
        list.push(10)?;
        Ok(list.iter().sum())
    });
    let list: List<i64> = [1i64, 2].into_iter().collect();
    let total: i64 = sum.call_tuple((list.clone(),)).unwrap().try_into().unwrap();
    assert_eq!(total, 13);
    assert_eq!(list.len(), 3);
}
