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
use std::sync::atomic::{AtomicUsize, Ordering};

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

struct I64Pattern;

impl AnyPattern for I64Pattern {
    type Bound<'a> = i64;

    fn try_match<'a>(value: AnyView<'a>) -> Option<Self::Bound<'a>> {
        value.try_as::<i64>()
    }
}

static LIVE_BINDINGS: AtomicUsize = AtomicUsize::new(0);

struct CountedPattern;

struct CountedBinding;

impl Drop for CountedBinding {
    fn drop(&mut self) {
        LIVE_BINDINGS.fetch_sub(1, Ordering::SeqCst);
    }
}

impl AnyPattern for CountedPattern {
    type Bound<'a> = CountedBinding;

    fn try_match<'a>(value: AnyView<'a>) -> Option<Self::Bound<'a>> {
        value.try_as::<i64>().map(|_| {
            LIVE_BINDINGS.fetch_add(1, Ordering::SeqCst);
            CountedBinding
        })
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
fn first_successful_pattern_wins() {
    let selected = match_any! {
        4_i64 {
            I64Pattern(value) => value + 10,
            I64Pattern(value) => value + 20,
            _ => 0,
        }
    };
    assert_eq!(selected, 14);
}

#[test]
fn false_guard_continues_in_source_order() {
    let value = Any::from(3_i64);
    let selected = match_any! {
        value {
            I64Pattern(value) if value == 2 => 1,
            I64Pattern(value) => value,
            _ => 0,
        }
    };
    assert_eq!(selected, 3);
}

#[test]
fn false_guard_drops_its_binding_before_the_next_arm() {
    LIVE_BINDINGS.store(0, Ordering::SeqCst);
    match_any! {
        Any::from(3_i64) {
            CountedPattern(_binding) if false => unreachable!(),
            CountedPattern(_binding) => assert_eq!(LIVE_BINDINGS.load(Ordering::SeqCst), 1),
            _ => unreachable!(),
        }
    }
    assert_eq!(LIVE_BINDINGS.load(Ordering::SeqCst), 0);
}

#[test]
fn failed_pattern_skips_its_guard_and_uses_fallback() {
    let value = Any::from(true);
    let mut guard_evaluated = false;
    let selected = match_any! {
        value {
            I64Pattern(_value) if {
                guard_evaluated = true;
                true
            } => 1,
            _ => 2,
        }
    };
    assert!(!guard_evaluated);
    assert_eq!(selected, 2);
}

#[test]
fn scrutinee_is_evaluated_once() {
    let mut evaluations = 0;
    let mut make_value = || {
        evaluations += 1;
        Any::from(7_i64)
    };
    let selected = match_any! {
        make_value() {
            I64Pattern(value) => value,
            _ => 0,
        }
    };
    assert_eq!(selected, 7);
    assert_eq!(evaluations, 1);
}

#[test]
fn arm_body_preserves_caller_control_flow() {
    fn select(value: Any) -> i64 {
        match_any! {
            value {
                I64Pattern(value) => return value,
                _ => (),
            }
        }
        0
    }

    assert_eq!(select(Any::from(8_i64)), 8);
}

#[test]
fn arm_body_can_break_an_outer_loop() {
    let value = Any::from(6_i64);
    let mut iterations = 0;
    loop {
        iterations += 1;
        match_any! {
            value {
                I64Pattern(_value) => break,
                _ => (),
            }
        }
    }
    assert_eq!(iterations, 1);
}

#[test]
fn accepts_any_view() {
    let value = 10_i64;
    let view = AnyView::from(&value);
    let selected = match_any! {
        view {
            I64Pattern(value) => value,
            _ => 0,
        }
    };
    assert_eq!(selected, 10);
}
