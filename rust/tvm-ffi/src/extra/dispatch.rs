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

//! Typed callback dispatch for [`super::structural_visit::structural_walk`].

use crate::error::Result;

use super::structural_visit::{
    DefRegionKind, IntoWalker, NativeVisit, VisitValue, WalkCallbackResult, WalkResult,
};

/// Dispatch for typed `structural_walk` observer callbacks.
///
/// `None` means that no handler matched. `#[dispatch(walk)]` generates this
/// trait from source-ordered `walk_*` methods.
pub trait WalkDispatch: Sized {
    fn dispatch_walk(
        &mut self,
        value: &VisitValue,
        def_region_kind: DefRegionKind,
    ) -> Option<WalkCallbackResult>;
}

impl<V: WalkDispatch> WalkDispatch for &mut V {
    #[inline]
    fn dispatch_walk(
        &mut self,
        value: &VisitValue,
        def_region_kind: DefRegionKind,
    ) -> Option<WalkCallbackResult> {
        (**self).dispatch_walk(value, def_region_kind)
    }
}

#[doc(hidden)]
pub enum ByWalkDispatch {}

impl<'a, V: WalkDispatch> IntoWalker<ByWalkDispatch> for &'a mut V {
    type Walker = DispatchWalker<&'a mut V>;
    fn into_walker(self) -> Self::Walker {
        DispatchWalker { walker: self }
    }
}

/// Adapter from [`WalkDispatch`] to the traversal's native callback.
#[doc(hidden)]
pub struct DispatchWalker<V> {
    walker: V,
}

impl<V: WalkDispatch> NativeVisit for DispatchWalker<V> {
    fn visit(&mut self, value: &VisitValue, def_region_kind: DefRegionKind) -> Result<WalkResult> {
        self.walker
            .dispatch_walk(value, def_region_kind)
            .unwrap_or(Ok(WalkResult::Advance))
    }
}
