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

//! Native descent into the built-in containers.
//!
//! `ffi.Array`, `ffi.List`, `ffi.Map` and `ffi.Dict` register C++ structural
//! hooks. Reaching them through the type-attribute column would turn every
//! container child into a Rust → C++ → Rust round trip across the ABI
//! vtable, so the Rust engines descend into these four types here instead.
//! Traversal order and the copy / in-place rules mirror the C++ hooks in
//! `src/ffi/extra/structural_visit.cc` and `structural_mutate.cc`: a
//! sequence visits every element in order; a map visits its values only,
//! keys being structural anchors; a copy is made at the first changed child,
//! and a uniquely owned container is updated in place.
//!
//! The sequence layout comes from the collections module. The map layout
//! mirrors C++ `MapBaseObj` and its small and dense specializations, like
//! `MapObj` in the collections module assuming `TVM_FFI_DEBUG_WITH_ABI_CHANGE`
//! is off. It is verified once per process against the runtime's own forward
//! iteration before any native map descent runs; on a mismatch the engines
//! keep using the registered C++ map hooks.

use std::ffi::c_void;
use std::ptr::addr_of_mut;
use std::sync::OnceLock;

use crate::any::{Any, AnyView};
use crate::collections::list::ListObj;
use crate::collections::map::Map;
use crate::object::Object;
use crate::tvm_ffi_sys::{TVMFFIAny, TVMFFITypeIndex};

/// Container families with native descent.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum NativeContainer {
    /// `ffi.Array` or `ffi.List`: elements in order.
    Seq,
    /// `ffi.Map` or `ffi.Dict`: values in iteration order.
    Map,
}

const ARRAY: i32 = TVMFFITypeIndex::kTVMFFIArray as i32;
const LIST: i32 = TVMFFITypeIndex::kTVMFFIList as i32;
const MAP: i32 = TVMFFITypeIndex::kTVMFFIMap as i32;
const DICT: i32 = TVMFFITypeIndex::kTVMFFIDict as i32;

/// The native container family of `type_index`, if any.
#[inline]
pub(crate) fn native_container(type_index: i32) -> Option<NativeContainer> {
    match type_index {
        ARRAY | LIST => Some(NativeContainer::Seq),
        MAP | DICT => Some(NativeContainer::Map),
        _ => None,
    }
}

/// Whether `type_index` is a built-in immutable leaf object (`ffi.Str` or
/// `ffi.Bytes`), whose registered C++ mutation hook is the identity.
#[inline]
pub(crate) fn is_immutable_leaf(type_index: i32) -> bool {
    type_index == TVMFFITypeIndex::kTVMFFIStr as i32
        || type_index == TVMFFITypeIndex::kTVMFFIBytes as i32
}

/// The element buffer and length of a sequence container.
///
/// Read it again before each child: a list that user code resizes between two
/// children must stay within bounds, exactly like the index loop of the C++
/// in-place hook.
///
/// # Safety
///
/// `raw` must hold a live `ffi.Array` or `ffi.List`.
#[inline]
pub(crate) unsafe fn seq_cells(raw: &TVMFFIAny) -> (*mut TVMFFIAny, usize) {
    // `ArrayObj` and `ListObj` share the C++ `SeqBaseObj` prefix.
    let seq = raw.data_union.v_obj as *const ListObj;
    (
        (*seq).data as *mut TVMFFIAny,
        usize::try_from((*seq).size).unwrap_or(0),
    )
}

/// Copy the owned value in `cell` into a new owner.
///
/// # Safety
///
/// `cell` must point to a live container cell.
#[inline]
pub(crate) unsafe fn owned_cell(cell: *const TVMFFIAny) -> Any {
    super::StructuralView::from_raw(*cell).to_owned()
}

/// Release the value in `cell` and store `value` there.
///
/// # Safety
///
/// `cell` must point to a live container cell that nothing else borrows,
/// inside a container the caller is allowed to modify in place.
#[inline]
pub(crate) unsafe fn replace_cell(cell: *mut TVMFFIAny, value: Any) {
    let previous = std::ptr::replace(cell, Any::into_raw_ffi_any(value));
    drop(Any::from_raw_ffi_any(previous));
}

// ---------------------------------------------------------------------------
// Map layout
// ---------------------------------------------------------------------------

/// C++ `MapBaseObj`.
#[repr(C)]
struct MapBaseLayout {
    object: Object,
    data: *mut c_void,
    size: u64,
    slots: u64,
    data_deleter: *mut c_void,
}

/// C++ `DenseMapBaseObj`.
#[repr(C)]
struct DenseMapLayout {
    base: MapBaseLayout,
    fib_shift: u32,
    iter_list_head: u64,
    iter_list_tail: u64,
}

/// C++ `std::pair<Any, Any>`, the entry of a small map.
#[repr(C)]
struct SmallMapEntry {
    key: TVMFFIAny,
    value: TVMFFIAny,
}

/// C++ `DenseMapBaseObj::ItemType`.
#[repr(C)]
struct DenseMapItem {
    key: TVMFFIAny,
    value: TVMFFIAny,
    prev: u64,
    next: u64,
}

/// `MapBaseObj::kSmallTagMask`: the most significant bit of `slots` marks
/// the small layout.
const SMALL_TAG_MASK: u64 = 1 << 63;
/// `DenseMapBaseObj::kInvalidIndex`, the end of the iteration list.
const INVALID_INDEX: u64 = u64::MAX;
/// `DenseMapBaseObj::kBlockCap`.
const DENSE_BLOCK_CAP: usize = 16;
/// `DenseMapBaseObj::Block`: one metadata byte per item, then the items.
const DENSE_BLOCK_SIZE: usize =
    DENSE_BLOCK_CAP + DENSE_BLOCK_CAP * std::mem::size_of::<DenseMapItem>();

const _: () = {
    assert!(std::mem::size_of::<SmallMapEntry>() == 32);
    assert!(std::mem::size_of::<DenseMapItem>() == 48);
    assert!(std::mem::offset_of!(MapBaseLayout, data) == std::mem::size_of::<Object>());
    assert!(
        std::mem::offset_of!(DenseMapLayout, iter_list_head)
            == std::mem::size_of::<MapBaseLayout>() + 8
    );
};

/// Cursor over a map's entries in the runtime's iteration order.
pub(crate) struct MapEntries(Cursor);

enum Cursor {
    Small {
        next: *mut SmallMapEntry,
        remaining: usize,
    },
    Dense {
        data: *mut u8,
        slots: u64,
        index: u64,
    },
}

impl MapEntries {
    /// Start at the first entry of `raw`.
    ///
    /// # Safety
    ///
    /// `raw` must hold a live `ffi.Map` or `ffi.Dict`.
    #[inline]
    pub(crate) unsafe fn of(raw: &TVMFFIAny) -> Self {
        let base = raw.data_union.v_obj as *const MapBaseLayout;
        if (*base).slots & SMALL_TAG_MASK != 0 {
            MapEntries(Cursor::Small {
                next: (*base).data as *mut SmallMapEntry,
                remaining: usize::try_from((*base).size).unwrap_or(0),
            })
        } else {
            let dense = base as *const DenseMapLayout;
            MapEntries(Cursor::Dense {
                data: (*base).data as *mut u8,
                slots: (*base).slots,
                index: (*dense).iter_list_head,
            })
        }
    }

    /// The next `(key, value)` cell pair, or `None` at the end.
    ///
    /// # Safety
    ///
    /// The map must stay alive and must not be restructured (inserted into,
    /// erased from, or rehashed) while the returned pointers are in use.
    #[inline]
    pub(crate) unsafe fn next_entry(&mut self) -> Option<(*mut TVMFFIAny, *mut TVMFFIAny)> {
        match &mut self.0 {
            Cursor::Small { next, remaining } => {
                if *remaining == 0 {
                    return None;
                }
                let entry = *next;
                *next = next.add(1);
                *remaining -= 1;
                Some((addr_of_mut!((*entry).key), addr_of_mut!((*entry).value)))
            }
            Cursor::Dense { data, slots, index } => {
                // The runtime never links an index at or past the slot count;
                // treat one as the end instead of reading past the blocks.
                if *index == INVALID_INDEX || *index >= *slots {
                    return None;
                }
                let i = *index as usize;
                let offset = (i / DENSE_BLOCK_CAP) * DENSE_BLOCK_SIZE
                    + DENSE_BLOCK_CAP
                    + (i % DENSE_BLOCK_CAP) * std::mem::size_of::<DenseMapItem>();
                let item = data.add(offset) as *mut DenseMapItem;
                *index = (*item).next;
                Some((addr_of_mut!((*item).key), addr_of_mut!((*item).value)))
            }
        }
    }
}

/// Whether the map layout above matches this runtime.
///
/// Checked once by comparing native iteration with the runtime's forward
/// iterator on a small and on a dense map; the engines keep using the
/// registered C++ map hooks when it does not.
pub(crate) fn native_map_descent_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| [3, 37].into_iter().all(map_layout_matches))
}

fn map_layout_matches(entries: i64) -> bool {
    let map: Map<i64, i64> = (0..entries).map(|i| (i * 7 + 1, i * 11 + 2)).collect();
    let expected: Vec<(i64, i64)> = map.iter().collect();
    let raw = *AnyView::from(&map).as_raw_ffi_any();
    let int = TVMFFITypeIndex::kTVMFFIInt as i32;
    let mut seen = Vec::with_capacity(expected.len());
    // SAFETY: `map` keeps the object alive and nothing modifies it here. The
    // cursor reads stay inside the object and its block allocation even if
    // the layout were wrong, because every index is bounded by `slots`.
    unsafe {
        let mut cursor = MapEntries::of(&raw);
        while let Some((key, value)) = cursor.next_entry() {
            if seen.len() == expected.len()
                || (*key).type_index != int
                || (*value).type_index != int
            {
                return false;
            }
            seen.push(((*key).data_union.v_int64, (*value).data_union.v_int64));
        }
    }
    seen == expected
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::function::Function;

    #[test]
    fn map_layout_check_passes_on_this_runtime() {
        assert!(native_map_descent_enabled());
    }

    #[test]
    fn native_map_iteration_matches_runtime_for_every_layout() {
        for entries in [0i64, 1, 2, 3, 4, 5, 6, 8, 15, 16, 17, 33, 64, 100, 257] {
            assert!(map_layout_matches(entries), "{entries} entries");
        }
    }

    #[test]
    fn native_dict_iteration_survives_erasures() {
        let call = |name: &str, args: &[Any]| {
            let views: Vec<AnyView<'_>> = args.iter().map(AnyView::from).collect();
            Function::get_global(name)
                .unwrap()
                .call_packed(&views)
                .unwrap()
        };
        let mut args = Vec::new();
        for i in 0..50i64 {
            args.push(Any::from(i));
            args.push(Any::from(i * 3));
        }
        let dict = call("ffi.Dict", &args);
        for key in [0i64, 7, 8, 9, 31, 49] {
            call("ffi.DictErase", &[dict.clone(), Any::from(key)]);
        }
        let size = i64::try_from(call("ffi.DictSize", std::slice::from_ref(&dict))).unwrap();
        let raw = *dict.as_raw_ffi_any();
        let mut seen = 0;
        unsafe {
            let mut cursor = MapEntries::of(&raw);
            while let Some((key, value)) = cursor.next_entry() {
                let key = (*key).data_union.v_int64;
                let value = (*value).data_union.v_int64;
                let expected =
                    i64::try_from(call("ffi.DictGetItem", &[dict.clone(), Any::from(key)]))
                        .unwrap();
                assert_eq!(value, expected);
                seen += 1;
            }
        }
        assert_eq!(seen, size);
        assert_eq!(seen, 44);
    }
}
