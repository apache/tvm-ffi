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
use std::marker::PhantomData;
use std::sync::LazyLock;

use crate::any::TryFromTemp;
use crate::derive::Object;
use crate::error::Result;
use crate::function::Function;
use crate::object::{Object, ObjectArc, ObjectRefCore};
use crate::type_traits::AnyCompatible;
use crate::{Any, AnyView};
use tvm_ffi_sys::TVMFFITypeIndex as TypeIndex;
use tvm_ffi_sys::{TVMFFIAny, TVMFFIObject};

#[repr(C)]
#[derive(Object)]
#[type_key = "ffi.Map"]
#[type_index(TypeIndex::kTVMFFIMap)]
pub struct MapObj {
    pub object: Object,
}

impl<K, V> Map<K, V>
where
    K: AnyCompatible + Clone + 'static,
    V: AnyCompatible + Clone + 'static,
{
    /// Create a new Map from key/value pairs.
    pub fn new<I: IntoIterator<Item = (K, V)>>(items: I) -> Result<Self> {
        static API_FUNC: LazyLock<Function> =
            LazyLock::new(|| Function::get_global("ffi.Map").unwrap());
        let items: Vec<(K, V)> = items.into_iter().collect();
        let mut args: Vec<AnyView> = Vec::with_capacity(items.len() * 2);
        for (key, value) in items.iter() {
            args.push(AnyView::from(key));
            args.push(AnyView::from(value));
        }
        (*API_FUNC).call_packed(&args)?.try_into()
    }

    /// Return the number of entries in the map.
    pub fn len(&self) -> Result<usize> {
        static API_FUNC: LazyLock<Function> =
            LazyLock::new(|| Function::get_global("ffi.MapSize").unwrap());
        let args = [AnyView::from(self)];
        let size_any = (*API_FUNC).call_packed(&args)?;
        let temp: TryFromTemp<i64> = TryFromTemp::try_from(size_any)?;
        let size = TryFromTemp::into_value(temp);
        Ok(size as usize)
    }

    /// Return true if the map is empty.
    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

    /// Return true if the map contains the key.
    pub fn contains_key(&self, key: &K) -> Result<bool> {
        static API_FUNC: LazyLock<Function> =
            LazyLock::new(|| Function::get_global("ffi.MapCount").unwrap());
        let args = [AnyView::from(self), AnyView::from(key)];
        let count_any = (*API_FUNC).call_packed(&args)?;
        let temp: TryFromTemp<i64> = TryFromTemp::try_from(count_any)?;
        let count = TryFromTemp::into_value(temp);
        Ok(count != 0)
    }

    /// Return the value for key or raise a KeyError.
    pub fn get(&self, key: &K) -> Result<V> {
        static API_FUNC: LazyLock<Function> =
            LazyLock::new(|| Function::get_global("ffi.MapGetItem").unwrap());
        let args = [AnyView::from(self), AnyView::from(key)];
        let value_any = (*API_FUNC).call_packed(&args)?;
        let temp: TryFromTemp<V> = TryFromTemp::try_from(value_any)?;
        Ok(TryFromTemp::into_value(temp))
    }

    /// Return the value for key or None if missing.
    pub fn get_optional(&self, key: &K) -> Result<Option<V>> {
        if !self.contains_key(key)? {
            return Ok(None);
        }
        self.get(key).map(Some)
    }

    /// Return the value for key or a default value if missing.
    pub fn get_or(&self, key: &K, default: V) -> Result<V> {
        match self.get_optional(key)? {
            Some(value) => Ok(value),
            None => Ok(default),
        }
    }

    /// Iterate over key/value pairs.
    pub fn iter(&self) -> Result<MapIterator<K, V>> {
        static API_FUNC: LazyLock<Function> =
            LazyLock::new(|| Function::get_global("ffi.MapForwardIterFunctor").unwrap());
        let args = [AnyView::from(self)];
        let functor: Function = (*API_FUNC).call_packed(&args)?.try_into()?;
        Ok(MapIterator {
            functor,
            remaining: self.len()?,
            _marker: PhantomData,
        })
    }
}

pub struct MapIterator<K: AnyCompatible + Clone, V: AnyCompatible + Clone> {
    functor: Function,
    remaining: usize,
    _marker: PhantomData<(K, V)>,
}

impl<K, V> Iterator for MapIterator<K, V>
where
    K: AnyCompatible + Clone + 'static,
    V: AnyCompatible + Clone + 'static,
{
    type Item = (K, V);

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let key_any = self
            .functor
            .call_tuple_with_len::<1, _>((0i64,))
            .ok()?;
        let key_temp: TryFromTemp<K> = TryFromTemp::try_from(key_any).ok()?;
        let key = TryFromTemp::into_value(key_temp);

        let value_any = self
            .functor
            .call_tuple_with_len::<1, _>((1i64,))
            .ok()?;
        let value_temp: TryFromTemp<V> = TryFromTemp::try_from(value_any).ok()?;
        let value = TryFromTemp::into_value(value_temp);
        let _ = self.functor.call_tuple_with_len::<1, _>((2i64,));
        self.remaining -= 1;
        Some((key, value))
    }
}
#[repr(C)]
#[derive(Clone)]
pub struct Map<K: AnyCompatible + Clone, V: AnyCompatible + Clone> {
    data: ObjectArc<MapObj>,
    _marker: PhantomData<(K, V)>,
}

unsafe impl<K: AnyCompatible + Clone, V: AnyCompatible + Clone> ObjectRefCore for Map<K, V> {
    type ContainerType = MapObj;

    fn data(this: &Self) -> &ObjectArc<Self::ContainerType> {
        &this.data
    }

    fn into_data(this: Self) -> ObjectArc<Self::ContainerType> {
        this.data
    }

    fn from_data(data: ObjectArc<Self::ContainerType>) -> Self {
        Self {
            data,
            _marker: PhantomData,
        }
    }
}

// --- Any Type System Conversions ---

unsafe impl<K, V> AnyCompatible for Map<K, V>
where
    K: AnyCompatible + Clone + 'static,
    V: AnyCompatible + Clone + 'static,
{
    fn type_str() -> String {
        format!("Map<{}, {}>", K::type_str(), V::type_str())
    }

    unsafe fn check_any_strict(data: &TVMFFIAny) -> bool {
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
    K: AnyCompatible + Clone + 'static,
    V: AnyCompatible + Clone + 'static,
{
    type Error = crate::error::Error;

    fn try_from(value: Any) -> Result<Self, Self::Error> {
        let temp: TryFromTemp<Self> = TryFromTemp::try_from(value)?;
        Ok(TryFromTemp::into_value(temp))
    }
}

impl<'a, K, V> TryFrom<AnyView<'a>> for Map<K, V>
where
    K: AnyCompatible + Clone + 'static,
    V: AnyCompatible + Clone + 'static,
{
    type Error = crate::error::Error;

    fn try_from(value: AnyView<'a>) -> Result<Self, Self::Error> {
        let temp: TryFromTemp<Self> = TryFromTemp::try_from(value)?;
        Ok(TryFromTemp::into_value(temp))
    }
}
