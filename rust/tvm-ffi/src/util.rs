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

//! Crate-internal helpers.

/// [`std::slice::from_raw_parts`] that allows a null `data` when `len` is 0.
///
/// # Safety
/// As for [`std::slice::from_raw_parts`] when `len` is not 0.
#[inline(always)]
pub(crate) unsafe fn slice_from_nullable_raw_parts<'a, T>(data: *const T, len: usize) -> &'a [T] {
    if len == 0 {
        return &[];
    }
    unsafe { std::slice::from_raw_parts(data, len) }
}

/// [`std::slice::from_raw_parts_mut`] that allows a null `data` when `len` is 0.
///
/// # Safety
/// As for [`std::slice::from_raw_parts_mut`] when `len` is not 0.
#[inline(always)]
pub(crate) unsafe fn slice_from_nullable_raw_parts_mut<'a, T>(
    data: *mut T,
    len: usize,
) -> &'a mut [T] {
    if len == 0 {
        return &mut [];
    }
    unsafe { std::slice::from_raw_parts_mut(data, len) }
}
