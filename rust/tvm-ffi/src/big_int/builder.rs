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
//! Native allocation of `BigIntObj` storage.
//!
//! Like the C++ runtime, normalization prunes redundant sign words but keeps
//! the allocation, so an object can end up shorter than it was allocated. The
//! Rust allocator must be handed the original layout on release, so the
//! capacity is recorded in a prefix just before the object and a dedicated
//! deleter reads it back. The prefix is invisible to the ABI: `v_obj` points at
//! the `BigIntObj` header, and other languages only ever call the deleter.
use super::{int_ops, BigInt, BigIntObj};
use crate::object::{unsafe_, Object};
use std::alloc::{alloc_zeroed, dealloc, handle_alloc_error, Layout};
use std::ffi::c_void;
use std::mem::{align_of, size_of, ManuallyDrop};
use std::sync::atomic::AtomicU64;
use tvm_ffi_sys::TVMFFIObjectDeleterFlagBitMask::kTVMFFIObjectDeleterFlagBitMaskWeak;
use tvm_ffi_sys::TVMFFITypeIndex as TypeIndex;
use tvm_ffi_sys::{TVMFFIObject, COMBINED_REF_COUNT_BOTH_ONE};

/// Bytes reserved before the object for its capacity; keeps the object aligned.
const PREFIX: usize = if size_of::<usize>() > align_of::<BigIntObj>() {
    size_of::<usize>()
} else {
    align_of::<BigIntObj>()
};

fn layout(capacity: usize) -> Layout {
    capacity
        .checked_mul(size_of::<i64>())
        .and_then(|words| words.checked_add(PREFIX + size_of::<BigIntObj>()))
        .and_then(|size| Layout::from_size_align(size, align_of::<BigIntObj>()).ok())
        .expect("BigInt allocation is too large")
}

/// A zero-initialized construction buffer of `capacity` words with a logical size.
///
/// Word algorithms write their result into [`words_mut`](Self::words_mut) and
/// call [`finish`](Self::finish); dropping the builder instead releases the
/// allocation.
pub(super) struct WordsBuilder {
    obj: *mut BigIntObj,
    capacity: usize,
}

impl WordsBuilder {
    /// Allocate `size` zeroed words.
    pub(super) fn new(size: usize) -> Self {
        Self::with_capacity(size, size)
    }

    /// Allocate `capacity` zeroed words, the first `size` of them logical.
    pub(super) fn with_capacity(size: usize, capacity: usize) -> Self {
        debug_assert!(size <= capacity);
        let layout = layout(capacity);
        unsafe {
            let base = alloc_zeroed(layout);
            if base.is_null() {
                handle_alloc_error(layout);
            }
            base.cast::<usize>().write(capacity);
            let obj = base.add(PREFIX).cast::<BigIntObj>();
            obj.write(BigIntObj {
                object: Object::new(),
                size,
            });
            obj.cast::<TVMFFIObject>().write(TVMFFIObject {
                combined_ref_count: AtomicU64::new(COMBINED_REF_COUNT_BOTH_ONE),
                type_index: TypeIndex::kTVMFFIBigInt as i32,
                __padding: 0,
                deleter: Some(big_int_deleter),
            });
            Self { obj, capacity }
        }
    }

    /// The whole capacity, logical words first.
    #[inline]
    pub(super) fn words_mut(&mut self) -> &mut [i64] {
        unsafe { std::slice::from_raw_parts_mut(self.obj.add(1).cast::<i64>(), self.capacity) }
    }

    /// Prune redundant sign words and produce the canonical integer.
    ///
    /// A value that fits `i64` is demoted to the inline representation and the
    /// allocation is released; otherwise the allocation is kept at its pruned
    /// logical length.
    pub(super) fn finish(self) -> BigInt {
        let ptr = ManuallyDrop::new(self).obj;
        unsafe {
            let obj = &mut *ptr;
            let words = std::slice::from_raw_parts(ptr.add(1).cast::<i64>(), obj.size);
            let size = int_ops::normalized_len(words);
            if size <= 1 {
                let value = words.first().copied().unwrap_or(0);
                unsafe_::dec_ref(ptr.cast::<TVMFFIObject>());
                return BigInt::from_i64(value);
            }
            obj.size = size;
            BigInt::from_obj(ptr)
        }
    }
}

impl Drop for WordsBuilder {
    fn drop(&mut self) {
        unsafe { unsafe_::dec_ref(self.obj.cast::<TVMFFIObject>()) }
    }
}

/// Deleter for objects allocated by [`WordsBuilder`].
///
/// `BigIntObj` has no fields with drop glue, so the strong phase has nothing to
/// run; the weak phase releases the allocation with the layout it was created with.
unsafe extern "C" fn big_int_deleter(ptr: *mut c_void, flags: i32) {
    if flags & kTVMFFIObjectDeleterFlagBitMaskWeak as i32 != 0 {
        let base = ptr.cast::<u8>().sub(PREFIX);
        let capacity = base.cast::<usize>().read();
        dealloc(base, layout(capacity));
    }
}
