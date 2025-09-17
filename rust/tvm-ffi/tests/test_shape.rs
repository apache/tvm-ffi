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
use tvm_ffi::*;

// ============================================================================
// Shape Tests
// ============================================================================

#[test]
fn test_any_shape() {
    let shape = Shape::from(vec![1, 2, 3, 4]);
    let any = Any::from(shape.clone());
    let any_view = AnyView::from(&shape);

    assert_eq!(any.type_index(), TypeIndex::kTVMFFIShape as i32);
    let converted = Shape::try_from(any).unwrap();
    assert_eq!(converted.as_slice(), &[1, 2, 3, 4]);

    assert_eq!(any_view.type_index(), TypeIndex::kTVMFFIShape as i32);
    let converted_view = Shape::try_from(any_view).unwrap();
    assert_eq!(converted_view.as_slice(), &[1, 2, 3, 4]);
}
