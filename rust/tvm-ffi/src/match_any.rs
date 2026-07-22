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

//! Runtime traits used by [`crate::match_any!`].

use crate::{Any, AnyCompatible, AnyView};

/// A value that can expose its contents as an erased [`AnyView`].
///
/// [`Any`], [`AnyView`], and every [`AnyCompatible`] value implement this
/// trait.
pub trait AsAnyView {
    /// Borrow this value as an [`AnyView`].
    fn as_any_view(&self) -> AnyView<'_>;
}

impl<T: AnyCompatible> AsAnyView for T {
    #[inline]
    fn as_any_view(&self) -> AnyView<'_> {
        AnyView::from(self)
    }
}

impl AsAnyView for Any {
    #[inline]
    fn as_any_view(&self) -> AnyView<'_> {
        AnyView::from(self)
    }
}

impl AsAnyView for AnyView<'_> {
    #[inline]
    fn as_any_view(&self) -> AnyView<'_> {
        *self
    }
}

/// A typed pattern accepted by [`crate::match_any!`].
pub trait AnyPattern {
    /// The binding produced by a successful match.
    type Bound<'a>;

    /// Try to match an erased value.
    fn try_match<'a>(value: AnyView<'a>) -> Option<Self::Bound<'a>>;
}
