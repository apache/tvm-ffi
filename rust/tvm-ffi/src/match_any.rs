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

//! Runtime trait used by [`crate::match_any!`].

use crate::AnyView;

/// An ObjectRef-style typed pattern accepted by [`crate::match_any!`].
///
/// The macro only calls this trait for values with an object type index.
pub trait ObjectPattern {
    /// The binding produced by a successful match.
    type Bound;

    /// Try to match an erased object value.
    fn try_match(value: AnyView<'_>) -> Option<Self::Bound>;
}
