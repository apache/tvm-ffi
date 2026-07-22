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

use tvm_ffi::{match_any, Any, AnyPattern, AnyView, AsAnyView};

struct TensorType;
struct ShapedType;

struct TypedExpr<T> {
    value: i64,
    _marker: PhantomData<T>,
}

impl<T> TypedExpr<T> {
    fn new(value: i64) -> Self {
        Self {
            value,
            _marker: PhantomData,
        }
    }
}

impl<T> AnyPattern for TypedExpr<T> {
    type Bound<'a> = Self;

    fn try_match<'a>(value: AnyView<'a>) -> Option<Self::Bound<'a>> {
        value.try_as::<i64>().map(Self::new)
    }
}

struct CustomSource(Any);

impl AsAnyView for CustomSource {
    fn as_any_view(&self) -> AnyView<'_> {
        AnyView::from(&self.0)
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

#[test]
fn first_successful_pattern_wins() {
    let selected = match_any! {
        4_i64 {
            TypedExpr::<TensorType>(tensor) => tensor.value + 10,
            TypedExpr::<ShapedType>(shaped) => shaped.value + 20,
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
            TypedExpr::<TensorType>(tensor) if tensor.value == 2 => 1,
            TypedExpr::<ShapedType>(shaped) => shaped.value,
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
            TypedExpr::<TensorType>(_tensor) if {
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
            TypedExpr::<TensorType>(tensor) => tensor.value,
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
                TypedExpr::<TensorType>(tensor) => return tensor.value,
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
                TypedExpr::<TensorType>(_tensor) => break,
                _ => (),
            }
        }
    }
    assert_eq!(iterations, 1);
}

#[test]
fn accepts_any_view_and_custom_sources() {
    let value = 10_i64;
    let view = AnyView::from(&value);
    let selected = match_any! {
        view {
            TypedExpr::<TensorType>(tensor) => tensor.value,
            _ => 0,
        }
    };
    assert_eq!(selected, 10);

    let custom = CustomSource(Any::from(11_i64));
    let selected = match_any! {
        custom {
            TypedExpr::<TensorType>(tensor) => tensor.value,
            _ => 0,
        }
    };
    assert_eq!(selected, 11);
}
