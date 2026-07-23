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

use tvm_ffi::{
    match_any, Any, AnyCompatible, AnyView, Array, ObjectPattern, ObjectRefCore, Shape, Tensor,
};

struct ObjectMatcher<T>(T);

impl<T> ObjectPattern for ObjectMatcher<T>
where
    T: AnyCompatible + ObjectRefCore,
{
    type Bound = T;

    fn try_match(value: AnyView<'_>) -> Option<Self::Bound> {
        value.try_as::<T>()
    }
}

#[derive(Debug, PartialEq, Eq)]
enum Lowered {
    Matrix,
    Tensor(usize),
    Shape(usize),
    Unsupported,
}

type FuncContext = Vec<&'static str>;

#[test]
fn matches_concrete_object_containers_in_source_order() {
    fn lower_matrix(tensor: Tensor, func_context: &mut FuncContext) -> Lowered {
        func_context.push("matrix");
        debug_assert_eq!(tensor.shape().len(), 2);
        Lowered::Matrix
    }

    fn lower_tensor(tensor: Tensor, func_context: &mut FuncContext) -> Lowered {
        func_context.push("tensor");
        Lowered::Tensor(tensor.shape().len())
    }

    fn lower_shape(shape: Shape, func_context: &mut FuncContext) -> Lowered {
        func_context.push("shape");
        Lowered::Shape(shape.len())
    }

    fn report_unsupported(func_context: &mut FuncContext) -> Lowered {
        func_context.push("unsupported");
        Lowered::Unsupported
    }

    fn lower(expr: Any, func_context: &mut FuncContext) -> Lowered {
        match_any! {
            expr {
                ObjectMatcher::<Tensor>(tensor)
                    if tensor.shape().len() == 2 => lower_matrix(tensor, func_context),
                ObjectMatcher::<Tensor>(tensor) => lower_tensor(tensor, func_context),
                ObjectMatcher::<Shape>(shape) => lower_shape(shape, func_context),
                _ => report_unsupported(func_context),
            }
        }
    }

    let matrix = Tensor::from_slice(&[0_f32; 6], &[2, 3]).unwrap();
    let volume = Tensor::from_slice(&[0_f32; 24], &[2, 3, 4]).unwrap();
    let shape = Shape::from([2_i64, 3, 4, 5]);
    let mut func_context = FuncContext::new();

    assert_eq!(lower(Any::from(matrix), &mut func_context), Lowered::Matrix);
    assert_eq!(
        lower(Any::from(volume), &mut func_context),
        Lowered::Tensor(3)
    );
    assert_eq!(
        lower(Any::from(shape), &mut func_context),
        Lowered::Shape(4)
    );
    assert_eq!(
        lower(Any::from(Array::<i64>::default()), &mut func_context),
        Lowered::Unsupported
    );
    assert_eq!(func_context, ["matrix", "tensor", "shape", "unsupported"]);
}
