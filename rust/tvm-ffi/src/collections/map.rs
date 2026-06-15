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
//! Immutable [`Map`] container backed by the C++ `ffi.Map` object.
//!
//! The `MapObj` header mirrors C++ `MapBaseObj`, so [`Map::len`] reads `size`
//! directly (no FFI). The hash-table storage itself is an implementation detail,
//! so lookups and iteration are delegated to the global functions the C++ runtime
//! registers (`ffi.MapGetItem`, `ffi.MapForwardIterFunctor`, ...): there is no
//! Map-specific C ABI, and a Rust re-implementation would have to replicate the
//! hashing (`AnyHash`/`AnyEqual`) and dense/small probing, just as the Python
//! bindings also delegate. Element types are erased to `Any`, so `K`/`V` are
//! validated lazily on access (see [`Map::get`] / [`Map::iter`]) rather than by
//! [`AnyCompatible::check_any_strict`].
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::Deref;

use crate::any::TryFromTemp;
use crate::derive::Object;
use crate::function::Function;
use crate::object::{Object, ObjectArc};
use crate::{Any, AnyCompatible, AnyView, Error, ObjectRefCore, Result};
use tvm_ffi_sys::TVMFFITypeIndex as TypeIndex;
use tvm_ffi_sys::{TVMFFIAny, TVMFFIObject};

/// Container object for [`Map`]. The header fields mirror C++ `MapBaseObj`
/// (`include/tvm/ffi/container/map_base.h`) so [`Map::len`] can read `size`
/// without an FFI call; the storage `data` points to stays opaque.
///
/// This is a partial mirror by design: `MapBaseObj` has one more trailing field
/// after `slots_` (a `data_deleter_` function pointer), omitted here because this
/// binding is read-only — it only reads fields up to `size`, which precede it,
/// and never allocates a `MapObj` itself (the C++ runtime allocates the larger
/// `Small`/`DenseMapBaseObj` subclass), so the trailing field is never touched.
/// The layout also assumes `TVM_FFI_DEBUG_WITH_ABI_CHANGE` is off (the default):
/// with that debug flag set, `MapBaseObj` gains a *leading* `state_marker` field
/// that would shift every offset below.
#[repr(C)]
#[derive(Object)]
#[type_key = "ffi.Map"]
#[type_index(TypeIndex::kTVMFFIMap)]
pub struct MapObj {
    pub object: Object,
    /// Pointer to the (opaque) key/value storage region (`MapBaseObj::data_`).
    pub data: *mut core::ffi::c_void,
    /// Number of entries (`MapBaseObj::size_`).
    pub size: u64,
    /// Number of hash slots; the MSB is a small-map tag (`MapBaseObj::slots_`).
    pub slots: u64,
}

/// Immutable, reference-counted map from `K` to `V`, sharing its underlying
/// `MapObj` with C++. Cloning is cheap (it bumps the refcount).
#[repr(C)]
pub struct Map<K, V> {
    data: ObjectArc<MapObj>,
    _marker: PhantomData<(K, V)>,
}

// Manual `Clone`: the data is the shared `ObjectArc`, so `Map<K, V>` is `Clone`
// regardless of whether `K`/`V` are (they are only phantom markers).
impl<K, V> Clone for Map<K, V> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            _marker: PhantomData,
        }
    }
}

unsafe impl<K, V> ObjectRefCore for Map<K, V> {
    type ContainerType = MapObj;

    fn data(this: &Self) -> &ObjectArc<MapObj> {
        &this.data
    }

    fn into_data(this: Self) -> ObjectArc<MapObj> {
        this.data
    }

    fn from_data(data: ObjectArc<MapObj>) -> Self {
        Self {
            data,
            _marker: PhantomData,
        }
    }
}

// A `Map<K, V>` is a counted handle to its `MapObj`, so it derefs to it (like
// `ObjectArc` does): methods can read header fields as `self.size` instead of
// `self.data.size`.
impl<K, V> Deref for Map<K, V> {
    type Target = MapObj;
    #[inline]
    fn deref(&self) -> &MapObj {
        &self.data
    }
}

impl<K, V> Map<K, V>
where
    K: AnyCompatible,
    V: AnyCompatible,
{
    /// Creates a new, empty map (via an `ffi.Map()` call to the C++ runtime).
    pub fn new() -> Self {
        Self::from_pairs(&[]).expect("ffi.Map() failed to construct an empty map")
    }

    /// Builds a map by calling the C++ `ffi.Map` constructor with a flattened
    /// `[k0, v0, k1, v1, ...]` argument list.
    fn from_pairs(pairs: &[(K, V)]) -> Result<Self> {
        let mut args: Vec<AnyView<'_>> = Vec::with_capacity(pairs.len() * 2);
        for (k, v) in pairs {
            args.push(AnyView::from(k));
            args.push(AnyView::from(v));
        }
        let result = crate::cached_global_func!("ffi.Map").call_packed(&args)?;
        Self::try_from(result)
    }

    /// Returns the number of entries in the map by reading the `MapObj` header
    /// directly (no FFI call), like [`Array::len`](crate::Array).
    pub fn len(&self) -> usize {
        self.size as usize
    }

    /// Returns `true` if the map contains no entries.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns whether `key` is present, propagating any FFI error (e.g. a key
    /// the C++ runtime cannot hash) rather than panicking. Backs both
    /// [`Map::contains_key`] and [`Map::get`].
    fn try_contains_key(&self, key: &K) -> Result<bool> {
        let result = crate::cached_global_func!("ffi.MapCount")
            .call_packed(&[AnyView::from(self), AnyView::from(key)])?;
        Ok(i64::try_from(result)? != 0)
    }

    /// In debug builds, panics if `K` does not match the map's actual key type,
    /// checked against one stored key. Compiles to nothing in release, so
    /// [`Map::get`] / [`Map::contains_key`] keep their single-FFI-call fast path.
    ///
    /// A pure key lookup *hashes* the key rather than retrieving one, so a wrong
    /// `K` simply fails to match and reads as "absent" — unlike [`Map::iter`],
    /// which retrieves keys and surfaces the mismatch. This assertion catches
    /// that misuse in dev/tests without taxing release lookups. Only meaningful
    /// on a miss (a hit already proves `K` matched a stored key).
    #[inline]
    fn debug_assert_key_type(&self) {
        #[cfg(debug_assertions)]
        {
            if !self.is_empty() {
                let functor = self.iter_functor();
                let first_key = functor
                    .call_packed(&[AnyView::from(&0i64)])
                    .expect("map iterator: reading current key failed");
                assert!(
                    first_key.try_as::<K>().is_some(),
                    "Map lookup: key type `{}` does not match the map's stored key type",
                    std::any::type_name::<K>(),
                );
            }
        }
    }

    /// Returns `true` if `key` is present in the map.
    ///
    /// Panics if the lookup fails in the C++ runtime (e.g. `key` is not
    /// hashable); use [`Map::get`] when such failures must be recovered from.
    /// Passing a `key` whose *type* does not match the map's key type is
    /// **undefined** (currently reads as `false`; see [`Map::get`]); debug builds
    /// assert against it.
    pub fn contains_key(&self, key: &K) -> bool {
        let present = self.try_contains_key(key).expect("ffi.MapCount call failed");
        if !present {
            self.debug_assert_key_type();
        }
        present
    }

    /// Looks up `key`, returning `Ok(None)` if absent, or `Err` if the lookup
    /// fails in the C++ runtime (e.g. `key` is not hashable) or the stored value
    /// cannot be converted to `V`. The map is immutable, so the lookup
    /// (existence check then `ffi.MapGetItem`) cannot race with itself.
    ///
    /// Passing a `key` whose *type* does not match the map's key type is
    /// **undefined**: the result is unspecified — it currently reads as absent
    /// (`Ok(None)`) because a pure lookup hashes the key rather than retrieving
    /// one, so the mismatch cannot be surfaced as an `Err` the way a value
    /// mismatch (or [`Map::iter`], which retrieves keys) is. Debug builds assert
    /// against this misuse.
    pub fn get(&self, key: &K) -> Result<Option<V>> {
        if !self.try_contains_key(key)? {
            self.debug_assert_key_type();
            return Ok(None);
        }
        let result = crate::cached_global_func!("ffi.MapGetItem")
            .call_packed(&[AnyView::from(self), AnyView::from(key)])?;
        let value = TryFromTemp::<V>::try_from(result).map(TryFromTemp::into_value)?;
        Ok(Some(value))
    }

    /// Returns an iterator over the `(key, value)` pairs of the map.
    pub fn iter(&self) -> MapItems<K, V> {
        self.make_iter(|f| (iter_read::<K>(f, 0, "key"), iter_read::<V>(f, 1, "value")))
    }

    /// Returns an iterator over the keys of the map.
    pub fn keys(&self) -> MapKeys<K> {
        self.make_iter(|f| iter_read::<K>(f, 0, "key"))
    }

    /// Returns an iterator over the values of the map.
    pub fn values(&self) -> MapValues<V> {
        self.make_iter(|f| iter_read::<V>(f, 1, "value"))
    }

    /// Builds a [`MapIter`] whose `read` extracts each entry as `T`. The
    /// forward-iteration functor is requested only for a non-empty map, so
    /// iterating an empty map makes no FFI call and allocates no functor.
    fn make_iter<T>(&self, read: fn(&Function) -> T) -> MapIter<T> {
        let remaining = self.len();
        MapIter {
            functor: (remaining != 0).then(|| self.iter_functor()),
            remaining,
            _keepalive: self.data.clone(),
            read,
        }
    }

    /// Obtains a fresh stateful forward-iteration functor from the C++ runtime.
    fn iter_functor(&self) -> Function {
        let result = crate::cached_global_func!("ffi.MapForwardIterFunctor")
            .call_packed(&[AnyView::from(self)])
            .expect("ffi.MapForwardIterFunctor call failed");
        Function::try_from(result).expect("ffi.MapForwardIterFunctor returned a non-function")
    }
}

impl<K, V> Default for Map<K, V>
where
    K: AnyCompatible,
    V: AnyCompatible,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V> Debug for Map<K, V>
where
    K: AnyCompatible,
    V: AnyCompatible,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fn short(name: &str) -> &str {
            name.split("::").last().unwrap_or(name)
        }
        write!(
            f,
            "Map<{}, {}>[{}]",
            short(std::any::type_name::<K>()),
            short(std::any::type_name::<V>()),
            self.len()
        )
    }
}

impl<K, V> FromIterator<(K, V)> for Map<K, V>
where
    K: AnyCompatible,
    V: AnyCompatible,
{
    /// Duplicate keys follow C++ `ffi.Map` semantics: a later pair overwrites an
    /// earlier one, so the resulting map may be smaller than the iterator.
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        let pairs: Vec<(K, V)> = iter.into_iter().collect();
        Self::from_pairs(&pairs).expect("ffi.Map() failed to construct a map")
    }
}

// --- Iterators ---
//
// The functor from `ffi.MapForwardIterFunctor` does not keep the map alive, so
// each iterator carries an `ObjectArc<MapObj>` keepalive. Functor command codes:
// `0` = read key, `1` = read value, `2` = advance.
//
// These are `ExactSizeIterator`s, so a key/value not matching the declared
// `K`/`V` panics rather than ending iteration early (which would drop entries).
// Use [`Map::get`], which returns `Err` on a type mismatch, when the element
// types are uncertain.

/// Reads the functor's current key (`command` 0) or value (`command` 1) as `T`,
/// panicking on a type mismatch (see the note above on `ExactSizeIterator`).
fn iter_read<T: AnyCompatible>(functor: &Function, command: i64, kind: &str) -> T {
    let any = functor
        .call_packed(&[AnyView::from(&command)])
        .expect("map iterator: reading current element failed");
    TryFromTemp::<T>::try_from(any)
        .map(TryFromTemp::into_value)
        .unwrap_or_else(|_| panic!("map iterator: {kind} does not match the map's {kind} type"))
}

/// Consumes one entry: decrements `remaining` and advances the functor unless
/// the map is now exhausted. Callers must guard `remaining > 0` (every `next`
/// returns early when it hits 0), so the decrement can never underflow.
fn iter_advance(functor: &Function, remaining: &mut usize) {
    debug_assert!(*remaining > 0, "iter_advance called with no remaining entries");
    *remaining -= 1;
    if *remaining > 0 {
        functor
            .call_packed(&[AnyView::from(&2i64)])
            .expect("map iterator: advancing failed");
    }
}

/// Forward iterator over a [`Map`], yielding `T` per entry. The `read` fn pulls
/// the current entry (key, value, or both) from the C++ functor, so each variant
/// touches only the command(s) it needs — `keys()` never reads values, and vice
/// versa. Created via the [`MapItems`] / [`MapKeys`] / [`MapValues`] aliases.
pub struct MapIter<T> {
    /// `None` for an empty map (no functor is requested); `Some` while at least
    /// one entry remains to be yielded.
    functor: Option<Function>,
    remaining: usize,
    _keepalive: ObjectArc<MapObj>,
    read: fn(&Function) -> T,
}

impl<T> Iterator for MapIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        if self.remaining == 0 {
            return None;
        }
        // `remaining > 0` implies `functor` is `Some` (see `Map::make_iter`).
        let functor = self
            .functor
            .as_ref()
            .expect("non-empty map iterator has a functor");
        let item = (self.read)(functor);
        iter_advance(functor, &mut self.remaining);
        Some(item)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<T> ExactSizeIterator for MapIter<T> {}

/// Iterator over `(key, value)` pairs, created by [`Map::iter`].
pub type MapItems<K, V> = MapIter<(K, V)>;
/// Iterator over keys, created by [`Map::keys`].
pub type MapKeys<K> = MapIter<K>;
/// Iterator over values, created by [`Map::values`].
pub type MapValues<V> = MapIter<V>;

impl<K, V> IntoIterator for &Map<K, V>
where
    K: AnyCompatible,
    V: AnyCompatible,
{
    type Item = (K, V);
    type IntoIter = MapItems<K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

// --- Any Type System Conversions ---

unsafe impl<K, V> AnyCompatible for Map<K, V>
where
    K: AnyCompatible,
    V: AnyCompatible,
{
    fn type_str() -> String {
        format!("Map<{}, {}>", K::type_str(), V::type_str())
    }

    unsafe fn check_any_strict(data: &TVMFFIAny) -> bool {
        // Container-level check only; element types are validated lazily on
        // access (see the module docs).
        data.type_index == TypeIndex::kTVMFFIMap as i32
    }

    unsafe fn copy_to_any_view(src: &Self, data: &mut TVMFFIAny) {
        data.type_index = TypeIndex::kTVMFFIMap as i32;
        data.data_union.v_obj = ObjectArc::as_raw(Self::data(src)) as *mut TVMFFIObject;
        data.small_str_len = 0;
    }

    unsafe fn move_to_any(src: Self, data: &mut TVMFFIAny) {
        data.type_index = TypeIndex::kTVMFFIMap as i32;
        data.data_union.v_obj = ObjectArc::into_raw(Self::into_data(src)) as *mut TVMFFIObject;
        data.small_str_len = 0;
    }

    unsafe fn copy_from_any_view_after_check(data: &TVMFFIAny) -> Self {
        let ptr = data.data_union.v_obj as *const MapObj;
        crate::object::unsafe_::inc_ref(ptr as *mut TVMFFIObject);
        Self::from_data(ObjectArc::from_raw(ptr))
    }

    unsafe fn move_from_any_after_check(data: &mut TVMFFIAny) -> Self {
        let ptr = data.data_union.v_obj as *const MapObj;
        let obj = Self::from_data(ObjectArc::from_raw(ptr));
        data.type_index = TypeIndex::kTVMFFINone as i32;
        data.data_union.v_int64 = 0;
        obj
    }

    unsafe fn try_cast_from_any_view(data: &TVMFFIAny) -> Result<Self, ()> {
        if data.type_index != TypeIndex::kTVMFFIMap as i32 {
            return Err(());
        }
        Ok(Self::copy_from_any_view_after_check(data))
    }
}

impl<K, V> TryFrom<Any> for Map<K, V>
where
    K: AnyCompatible,
    V: AnyCompatible,
{
    type Error = Error;

    fn try_from(value: Any) -> Result<Self> {
        let temp: TryFromTemp<Self> = TryFromTemp::try_from(value)?;
        Ok(TryFromTemp::into_value(temp))
    }
}

impl<'a, K, V> TryFrom<AnyView<'a>> for Map<K, V>
where
    K: AnyCompatible,
    V: AnyCompatible,
{
    type Error = Error;

    fn try_from(value: AnyView<'a>) -> Result<Self> {
        let temp: TryFromTemp<Self> = TryFromTemp::try_from(value)?;
        Ok(TryFromTemp::into_value(temp))
    }
}
