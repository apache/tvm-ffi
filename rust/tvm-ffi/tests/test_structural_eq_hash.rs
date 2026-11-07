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
use tvm_ffi::reflection::AccessKind;
use tvm_ffi::{
    get_first_structural_mismatch, structural_equal, structural_equal_with, structural_hash, Array,
    StructuralOptions,
};

#[test]
fn structural_functions_agree_and_locate_the_mismatch() -> tvm_ffi::Result<()> {
    let a = Array::from_vec(vec![1_i64, 2, 3]);
    let same = Array::from_vec(vec![1_i64, 2, 3]);
    let other = Array::from_vec(vec![1_i64, 2, 4]);

    assert!(structural_equal(&a, &same)?);
    assert!(!structural_equal(&a, &other)?);
    let options = StructuralOptions {
        map_free_vars: true,
        ..Default::default()
    };
    assert!(structural_equal_with(&a, &same, options)?);
    assert_eq!(structural_hash(&a)?, structural_hash(&same)?);
    assert_ne!(structural_hash(&a)?, structural_hash(&other)?);

    assert!(get_first_structural_mismatch(&a, &same)?.is_none());
    let (lhs, rhs) = get_first_structural_mismatch(&a, &other)?.unwrap();
    for path in [lhs, rhs] {
        let step = path.step.clone().unwrap();
        assert_eq!(
            (step.kind, i64::try_from(step.key.clone())?),
            (AccessKind::ArrayItem, 2)
        );
    }
    Ok(())
}
