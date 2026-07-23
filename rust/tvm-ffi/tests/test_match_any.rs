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

use tvm_ffi::{match_any, Any, AnyView, Array, ObjectPattern, Shape, Tensor};

// Stand-ins for matcher types supplied by a downstream AST crate.
struct TensorType;
struct ShapedType;

struct TypedExpr<T>(PhantomData<T>);

enum ShapedExpr {
    Tensor(Tensor),
    Shape(Shape),
}

impl ShapedExpr {
    fn rank(&self) -> usize {
        match self {
            Self::Tensor(tensor) => tensor.shape().len(),
            Self::Shape(shape) => shape.len(),
        }
    }
}

impl ObjectPattern for TypedExpr<TensorType> {
    type Bound = Tensor;

    fn try_match(value: AnyView<'_>) -> Option<Self::Bound> {
        value.try_as::<Tensor>()
    }
}

impl ObjectPattern for TypedExpr<ShapedType> {
    type Bound = ShapedExpr;

    fn try_match(value: AnyView<'_>) -> Option<Self::Bound> {
        value
            .try_as::<Tensor>()
            .map(ShapedExpr::Tensor)
            .or_else(|| value.try_as::<Shape>().map(ShapedExpr::Shape))
    }
}

#[derive(Debug, PartialEq, Eq)]
enum Lowered {
    Matrix,
    Shaped(usize),
    Unsupported,
}

#[derive(Default)]
struct FuncContext {
    calls: Vec<(&'static str, usize)>,
}

#[test]
fn lowers_with_ordered_patterns_and_a_guard() {
    fn lower_tensor(tensor: Tensor, func_context: &mut FuncContext) -> Lowered {
        func_context.calls.push(("tensor", tensor.shape().len()));
        Lowered::Matrix
    }

    fn lower_shaped(shaped: ShapedExpr, func_context: &mut FuncContext) -> Lowered {
        let rank = shaped.rank();
        func_context.calls.push(("shaped", rank));
        Lowered::Shaped(rank)
    }

    fn report_unsupported(func_context: &mut FuncContext) -> Lowered {
        func_context.calls.push(("unsupported", 0));
        Lowered::Unsupported
    }

    fn lower(expr: Any, func_context: &mut FuncContext) -> Lowered {
        match_any! {
            expr {
                TypedExpr::<TensorType>(tensor)
                    if tensor.shape().len() == 2 => lower_tensor(tensor, func_context),
                TypedExpr::<ShapedType>(shaped) => lower_shaped(shaped, func_context),
                _ => report_unsupported(func_context),
            }
        }
    }

    let matrix = Tensor::from_slice(&[0_f32; 6], &[2, 3]).unwrap();
    let volume = Tensor::from_slice(&[0_f32; 24], &[2, 3, 4]).unwrap();
    let shape = Shape::from([2_i64, 3, 4, 5]);
    let mut func_context = FuncContext::default();

    assert_eq!(lower(Any::from(matrix), &mut func_context), Lowered::Matrix);
    assert_eq!(
        lower(Any::from(volume), &mut func_context),
        Lowered::Shaped(3)
    );
    assert_eq!(
        lower(Any::from(shape), &mut func_context),
        Lowered::Shaped(4)
    );
    assert_eq!(
        lower(Any::from(Array::<i64>::default()), &mut func_context),
        Lowered::Unsupported
    );
    assert_eq!(
        func_context.calls,
        [
            ("tensor", 2),
            ("shaped", 3),
            ("shaped", 4),
            ("unsupported", 0),
        ]
    );
}
