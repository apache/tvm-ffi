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
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use tvm_ffi::*;

// must have repr(C) for the object header stays in the same position
#[repr(C)]
struct TestIntObj {
    object: Object,
    pub value: i64,
    // counter for recording the number of times the object is deleted
    delete_counter: Arc<AtomicU32>,
    pub extra_item_count: u64,
}

impl TestIntObj {
    pub fn new(value: i64, delete_counter: Arc<AtomicU32>, extra_item_count: u64) -> Self {
        Self {
            object: Object::new(),
            value,
            delete_counter,
            extra_item_count,
        }
    }
}

impl Drop for TestIntObj {
    fn drop(&mut self) {
        self.delete_counter.fetch_add(1, Ordering::Relaxed);
    }
}

unsafe impl ObjectCore for TestIntObj {
    const TYPE_KEY: &'static str = Object::TYPE_KEY;
    #[inline]
    fn type_index() -> i32 {
        Object::type_index()
    }
    #[inline]
    unsafe fn object_header_mut(this: &mut Self) -> &mut TVMFFIObject {
        Object::object_header_mut(&mut this.object)
    }
}

unsafe impl ObjectCoreWithExtraItems for TestIntObj {
    type ExtraItem = u64;
    #[inline]
    fn extra_items_count(this: &Self) -> usize {
        this.extra_item_count as usize
    }
}

#[test]
fn test_object_arc() {
    let delete_counter = Arc::new(AtomicU32::new(0));
    let obj_arc = ObjectArc::new(TestIntObj::new(11, delete_counter.clone(), 0));
    assert_eq!(obj_arc.value, 11);
    assert_eq!(ObjectArc::strong_count(&obj_arc), 1);
    assert_eq!(ObjectArc::weak_count(&obj_arc), 1);

    let ref1 = obj_arc.clone();
    assert_eq!(ObjectArc::strong_count(&obj_arc), 2);
    assert_eq!(ObjectArc::weak_count(&obj_arc), 1);

    let ref2 = obj_arc.clone();
    assert_eq!(ObjectArc::strong_count(&obj_arc), 3);
    assert_eq!(ObjectArc::weak_count(&obj_arc), 1);
    assert_eq!(ref1.value, 11);
    // drop obj_arc
    drop(obj_arc);
    assert_eq!(ObjectArc::strong_count(&ref1), 2);
    assert_eq!(ObjectArc::weak_count(&ref1), 1);
    assert_eq!(delete_counter.load(Ordering::Relaxed), 0);
    // drop ref1
    drop(ref1);
    assert_eq!(ObjectArc::strong_count(&ref2), 1);
    assert_eq!(ObjectArc::weak_count(&ref2), 1);
    assert_eq!(delete_counter.load(Ordering::Relaxed), 0);
    // drop ref2
    drop(ref2);
    assert_eq!(delete_counter.load(Ordering::Relaxed), 1);
}

#[test]
fn test_object_arc_with_extra_items() {
    let delete_counter = Arc::new(AtomicU32::new(0));
    let mut obj_arc =
        ObjectArc::new_with_extra_items(TestIntObj::new(12, delete_counter.clone(), 10));
    assert_eq!(obj_arc.value, 12);
    assert_eq!(ObjectArc::strong_count(&obj_arc), 1);
    assert_eq!(ObjectArc::weak_count(&obj_arc), 1);
    assert_eq!(delete_counter.load(Ordering::Relaxed), 0);
    unsafe {
        // layout check of extra items
        assert_eq!(TestIntObj::extra_items_count(&obj_arc), 10);
        assert_eq!(TestIntObj::extra_items(&obj_arc).len(), 10);
        assert_eq!(TestIntObj::extra_items_mut(&mut obj_arc).len(), 10);
        assert_eq!(
            TestIntObj::extra_items_mut(&mut obj_arc).as_ptr() as *mut u8,
            (ObjectArc::as_raw_mut(&mut obj_arc) as *mut u8).add(std::mem::size_of::<TestIntObj>())
        );
    }
    drop(obj_arc);
    assert_eq!(delete_counter.load(Ordering::Relaxed), 1);
}

#[test]
fn test_object_arc_from_raw() {
    unsafe {
        let delete_counter = Arc::new(AtomicU32::new(0));
        let obj_arc = ObjectArc::new(TestIntObj::new(11, delete_counter.clone(), 0));
        let raw_ptr = ObjectArc::into_raw(obj_arc);
        let obj_arc2 = ObjectArc::from_raw(raw_ptr);
        assert_eq!(obj_arc2.value, 11);
        assert_eq!(ObjectArc::strong_count(&obj_arc2), 1);
        assert_eq!(ObjectArc::weak_count(&obj_arc2), 1);
        assert_eq!(delete_counter.load(Ordering::Relaxed), 0);
        // drop obj_arc2
        drop(obj_arc2);
        assert_eq!(delete_counter.load(Ordering::Relaxed), 1);
    }
}

#[test]
fn test_object_arc_option_size() {
    assert_eq!(
        std::mem::size_of::<Option<ObjectArc<TestIntObj>>>(),
        std::mem::size_of::<ObjectArc<TestIntObj>>()
    );
}
