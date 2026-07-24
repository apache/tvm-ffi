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

use tvm_ffi::{match_any, Any, AnyView, Array, Map, Shape, Tensor};

#[test]
fn matches_concrete_object_containers_in_source_order() {
    fn classify(expr: Any) -> (&'static str, usize) {
        match_any! {
            expr {
                Tensor(tensor)
                    if tensor.shape().len() == 2 => ("matrix", tensor.shape().len()),
                Tensor(tensor) => ("tensor", tensor.shape().len()),
                Shape(shape) => ("shape", shape.len()),
                Array::<i64>(array) => ("array", array.len()),
                _ => ("unsupported", 0),
            }
        }
    }

    let matrix = Tensor::from_slice(&[0_f32; 6], &[2, 3]).unwrap();
    let volume = Tensor::from_slice(&[0_f32; 24], &[2, 3, 4]).unwrap();
    let shape = Shape::from([2_i64, 3, 4, 5]);
    let array = [1_i64, 2, 3].into_iter().collect::<Array<i64>>();

    assert_eq!(classify(Any::from(matrix)), ("matrix", 2));
    assert_eq!(classify(Any::from(volume)), ("tensor", 3));
    assert_eq!(classify(Any::from(shape)), ("shape", 4));
    assert_eq!(classify(Any::from(array)), ("array", 3));
    assert_eq!(
        classify(Any::from(Map::<i64, i64>::default())),
        ("unsupported", 0)
    );
    assert_eq!(classify(Any::from(1_i64)), ("unsupported", 0));

    let tensor = Tensor::from_slice(&[0_f32; 6], &[2, 3]).unwrap();
    let view = AnyView::from(&tensor);
    let matched_view = match_any! {
        view {
            Tensor(tensor) => ("tensor", tensor.shape().len()),
            _ => ("unsupported", 0),
        }
    };
    assert_eq!(matched_view, ("tensor", 2));
}

#[test]
fn parameterized_containers_keep_ordered_conversion() {
    let array = [1.5_f64, 2.5].into_iter().collect::<Array<f64>>();
    let selected = match_any! {
        Any::from(array) {
            Array::<i64>(_) => "integer array",
            Tensor(_) => "tensor",
            Shape(_) => "shape",
            Array::<f64>(_) => "float array",
            _ => "unsupported",
        }
    };

    assert_eq!(selected, "float array");
}
