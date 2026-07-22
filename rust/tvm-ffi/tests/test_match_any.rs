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

use tvm_ffi::{match_any, Any, AnyPattern, AnyView, Shape, Tensor};

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

impl AnyPattern for TypedExpr<TensorType> {
    type Bound<'a> = Tensor;

    fn try_match<'a>(value: AnyView<'a>) -> Option<Self::Bound<'a>> {
        value.try_as::<Tensor>()
    }
}

impl AnyPattern for TypedExpr<ShapedType> {
    type Bound<'a> = ShapedExpr;

    fn try_match<'a>(value: AnyView<'a>) -> Option<Self::Bound<'a>> {
        value
            .try_as::<Tensor>()
            .map(ShapedExpr::Tensor)
            .or_else(|| value.try_as::<Shape>().map(ShapedExpr::Shape))
    }
}

#[derive(Debug, PartialEq, Eq)]
enum Lowered {
    Matrix,
    Tensor(usize),
    Shaped(usize),
    Unsupported,
}

#[derive(Default)]
struct FuncContext {
    lowered: Vec<Lowered>,
}

fn lower_matrix(_tensor: Tensor, func_context: &mut FuncContext) {
    func_context.lowered.push(Lowered::Matrix);
}

fn lower_tensor(tensor: Tensor, func_context: &mut FuncContext) {
    func_context
        .lowered
        .push(Lowered::Tensor(tensor.shape().len()));
}

fn lower_shaped(shaped: ShapedExpr, func_context: &mut FuncContext) {
    func_context.lowered.push(Lowered::Shaped(shaped.rank()));
}

fn report_unsupported(func_context: &mut FuncContext) {
    func_context.lowered.push(Lowered::Unsupported);
}

fn lower_expr(expr: Any, func_context: &mut FuncContext) {
    match_any! {
        expr {
            TypedExpr::<TensorType>(tensor)
                if tensor.shape().len() == 2 => lower_matrix(tensor, func_context),
            TypedExpr::<TensorType>(tensor) => lower_tensor(tensor, func_context),
            TypedExpr::<ShapedType>(shaped) => lower_shaped(shaped, func_context),
            _ => report_unsupported(func_context),
        }
    }
}

#[test]
fn lowers_real_object_types_in_source_order() {
    let matrix = Tensor::from_slice(&[0_f32; 6], &[2, 3]).unwrap();
    let volume = Tensor::from_slice(&[0_f32; 24], &[2, 3, 4]).unwrap();
    let shape = Shape::from([2_i64, 3, 4, 5]);
    let mut func_context = FuncContext::default();

    lower_expr(Any::from(matrix), &mut func_context);
    lower_expr(Any::from(volume), &mut func_context);
    lower_expr(Any::from(shape), &mut func_context);
    lower_expr(Any::from(true), &mut func_context);

    assert_eq!(
        func_context.lowered,
        [
            Lowered::Matrix,
            Lowered::Tensor(3),
            Lowered::Shaped(4),
            Lowered::Unsupported,
        ]
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
