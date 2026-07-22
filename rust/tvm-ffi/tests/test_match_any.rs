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

struct TensorType;
struct ShapedType;
struct I64Pattern;

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

impl ObjectPattern for I64Pattern {
    type Bound = i64;

    fn try_match(value: AnyView<'_>) -> Option<Self::Bound> {
        value.try_as::<i64>()
    }
}

#[derive(Debug, PartialEq, Eq)]
enum Lowered {
    Matrix,
    Shaped(usize),
    Unsupported,
}

fn lower(expr: Any) -> Lowered {
    match_any! {
        expr {
            TypedExpr::<TensorType>(tensor)
                if tensor.shape().len() == 2 => Lowered::Matrix,
            TypedExpr::<ShapedType>(shaped) => Lowered::Shaped(shaped.rank()),
            _ => Lowered::Unsupported,
        }
    }
}

#[test]
fn matches_object_patterns_in_source_order() {
    let matrix = Tensor::from_slice(&[0_f32; 6], &[2, 3]).unwrap();
    let volume = Tensor::from_slice(&[0_f32; 24], &[2, 3, 4]).unwrap();
    let shape = Shape::from([2_i64, 3, 4, 5]);

    assert_eq!(lower(Any::from(matrix)), Lowered::Matrix);
    assert_eq!(lower(Any::from(volume)), Lowered::Shaped(3));
    assert_eq!(lower(Any::from(shape)), Lowered::Shaped(4));
    assert_eq!(
        lower(Any::from(Array::<i64>::default())),
        Lowered::Unsupported
    );
}

#[test]
fn scrutinee_is_evaluated_once() {
    let mut evaluations = 0;
    let mut make_value = || {
        evaluations += 1;
        Shape::from([2_i64, 3, 4])
    };
    let selected = match_any! {
        make_value() {
            TypedExpr::<ShapedType>(shaped) => shaped.rank(),
            _ => 0,
        }
    };
    assert_eq!(selected, 3);
    assert_eq!(evaluations, 1);
}

#[test]
fn non_object_values_use_fallback() {
    let selected = match_any! {
        8_i64 {
            I64Pattern(value) => value,
            _ => 0,
        }
    };
    assert_eq!(selected, 0);
}

#[test]
fn accepts_any_view() {
    let shape = Shape::from([2_i64, 3, 4, 5]);
    let view = AnyView::from(&shape);
    let selected = match_any! {
        view {
            TypedExpr::<ShapedType>(shaped) => shaped.rank(),
            _ => 0,
        }
    };
    assert_eq!(selected, 4);
}
