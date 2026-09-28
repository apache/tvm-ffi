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
use std::{collections::HashMap, hash::Hash};
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
    const TYPE_DEPTH: i32 = Object::TYPE_DEPTH;
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

// Records the size of this thread's latest deallocation, to check the layout an
// object is released with.
struct RecordingAllocator;

thread_local! {
    static LAST_DEALLOC_SIZE: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

unsafe impl std::alloc::GlobalAlloc for RecordingAllocator {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        std::alloc::System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        let _ = LAST_DEALLOC_SIZE.try_with(|size| size.set(layout.size()));
        std::alloc::System.dealloc(ptr, layout)
    }
}

#[global_allocator]
static ALLOCATOR: RecordingAllocator = RecordingAllocator;

/// Drops `obj_arc` while a weak reference outlives it, held the way C++ `WeakObjectPtr`
/// does (`Object::IncWeakRef`, then `Object::DecWeakRef`). `expired` runs while only the
/// weak reference is left. Returns the size the allocation is freed with.
fn release_outlived_by_weak_ref<T: ObjectCore>(
    mut obj_arc: ObjectArc<T>,
    expired: impl FnOnce(),
) -> usize {
    use std::sync::atomic::fence;
    use tvm_ffi_sys::COMBINED_REF_COUNT_WEAK_ONE as WEAK_ONE;
    unsafe {
        let header = ObjectArc::as_raw_mut(&mut obj_arc) as *mut TVMFFIObject;
        (*header)
            .combined_ref_count
            .fetch_add(WEAK_ONE, Ordering::Relaxed);
        drop(obj_arc);
        // The data is dropped; the weak reference still sees an intact, expired header.
        assert_eq!(
            (*header).combined_ref_count.load(Ordering::Relaxed),
            WEAK_ONE
        );
        expired();
        // The last weak reference frees the full allocation.
        let old = (*header)
            .combined_ref_count
            .fetch_sub(WEAK_ONE, Ordering::Release);
        assert_eq!(old, WEAK_ONE);
        fence(Ordering::Acquire);
        let weak = tvm_ffi_sys::TVMFFIObjectDeleterFlagBitMask::kTVMFFIObjectDeleterFlagBitMaskWeak;
        ((*header).deleter.unwrap())(header.cast(), weak as i32);
    }
    LAST_DEALLOC_SIZE.with(|size| size.get())
}

#[test]
fn test_object_arc_with_extra_items_outlived_by_weak_ref() {
    let delete_counter = Arc::new(AtomicU32::new(0));
    let obj_arc = ObjectArc::new_with_extra_items(TestIntObj::new(13, delete_counter.clone(), 10));
    let freed = release_outlived_by_weak_ref(obj_arc, || {
        assert_eq!(delete_counter.load(Ordering::Relaxed), 1);
    });
    assert_eq!(
        freed,
        std::mem::size_of::<TestIntObj>() + 10 * std::mem::size_of::<u64>()
    );
}

// Only the header, with a fixed number of items after it.
#[repr(C)]
struct HeaderOnlyObj<const N: usize> {
    object: Object,
}

unsafe impl<const N: usize> ObjectCore for HeaderOnlyObj<N> {
    const TYPE_KEY: &'static str = Object::TYPE_KEY;
    const TYPE_DEPTH: i32 = Object::TYPE_DEPTH;
    #[inline]
    fn type_index() -> i32 {
        Object::type_index()
    }
    #[inline]
    unsafe fn object_header_mut(this: &mut Self) -> &mut TVMFFIObject {
        Object::object_header_mut(&mut this.object)
    }
}

unsafe impl<const N: usize> ObjectCoreWithExtraItems for HeaderOnlyObj<N> {
    type ExtraItem = u64;
    #[inline]
    fn extra_items_count(_: &Self) -> usize {
        N
    }
}

#[test]
fn test_header_only_object_with_extra_items_outlived_by_weak_ref() {
    // The item count is kept in the word after the header: the first item for N = 2,
    // padding the allocation reserves for N = 0.
    let header_size = std::mem::size_of::<TVMFFIObject>();
    let obj_arc = ObjectArc::new_with_extra_items(HeaderOnlyObj::<2> {
        object: Object::new(),
    });
    assert_eq!(
        release_outlived_by_weak_ref(obj_arc, || {}),
        header_size + 16
    );
    let obj_arc = ObjectArc::new_with_extra_items(HeaderOnlyObj::<0> {
        object: Object::new(),
    });
    assert_eq!(
        release_outlived_by_weak_ref(obj_arc, || {}),
        header_size + 8
    );
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

#[test]
fn test_object_reference_identity() {
    fn assert_hash<T: Hash>(_value: &T) {}

    let first = Array::new(vec![1i64]);
    let alias = first.clone();
    let second = Array::new(vec![1i64]);

    assert!(first.same_as(&alias));
    assert!(!first.same_as(&second));

    let first_id = ObjectIdentity::of(&first);
    let alias_id = ObjectIdentity::of(&alias);
    let second_id = ObjectIdentity::of(&second);
    assert_hash(&first_id);
    assert_eq!(first_id, alias_id);
    assert_ne!(first_id, second_id);

    let mut identities = HashMap::new();
    identities.insert(first_id, "first");
    assert_eq!(identities.get(&alias_id), Some(&"first"));
    assert_eq!(identities.get(&second_id), None);
}
