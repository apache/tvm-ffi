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
//! Lazy initialization with an outlined first-access path.

use std::ops::Deref;
use std::sync::LazyLock;

/// A lazy value that keeps initialization in a separate cold function.
///
/// Construct it with an initializer and dereference it to access the value.
/// Initialization runs once; concurrent callers wait, and an initializer panic
/// poisons the value, following [`LazyLock`].
///
/// # Example
///
/// ```
/// use tvm_ffi::extra::cold_init_lazy_lock::ColdInitLazyLock;
///
/// static NAME: ColdInitLazyLock<String> = ColdInitLazyLock::new(|| "ready".into());
/// assert_eq!(NAME.as_str(), "ready");
/// ```
///
/// # Why avoid inlining initialization?
///
/// Inlining initialization can enlarge the caller's stack frame even when the
/// value is already initialized. We keep initialization in a separate cold
/// function and inline only the initialized check.
pub struct ColdInitLazyLock<T, F = fn() -> T> {
    inner: LazyLock<T, F>,
}

impl<T, F: FnOnce() -> T> ColdInitLazyLock<T, F> {
    /// Store `initializer` for execution on the first dereference.
    ///
    /// Construction does not execute the initializer or allocate wrapper storage.
    pub const fn new(initializer: F) -> Self {
        Self {
            inner: LazyLock::new(initializer),
        }
    }
}

impl<T, F: FnOnce() -> T> Deref for ColdInitLazyLock<T, F> {
    type Target = T;

    #[inline(always)]
    fn deref(&self) -> &T {
        if let Some(value) = LazyLock::get(&self.inner) {
            value
        } else {
            force_outlined(&self.inner)
        }
    }
}

// Keep the lock's first-access machinery outside initialized callers' frames.
#[cold]
#[inline(never)]
fn force_outlined<T, F: FnOnce() -> T>(lock: &LazyLock<T, F>) -> &T {
    LazyLock::force(lock)
}
