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

/// Helper to create a Tensor with a specific float value and shape
fn create_tensor(val: f32, shape: &[i64]) -> Tensor {
    let dtype = DLDataType::new(DLDataTypeCode::kDLFloat, 32, 1);
    let device = DLDevice::new(DLDeviceType::kDLCPU, 0);
    let tensor = Tensor::from_nd_alloc(CPUNDAlloc {}, shape, dtype, device);
    if let Ok(slice) = tensor.data_as_slice_mut::<f32>() {
        slice[0] = val;
    }
    tensor
}

/// Helper to extract the first float value from a Tensor
fn get_val(tensor: &Tensor) -> f32 {
    tensor
        .data_as_slice::<f32>()
        .expect("Type mismatch or null")[0]
}

#[test]
fn test_array_core_and_iteration() {
    let t1 = create_tensor(10.0, &[1, 2]);
    let t2 = create_tensor(20.0, &[3, 4, 5]);

    let array = Array::new(vec![t1.clone(), t2.clone()]);

    // Core Accessors
    assert_eq!(array.len(), 2);
    assert!(!array.is_empty());
    assert!(array.get(2).is_err()); // Out of bounds

    // Value Integrity
    assert_eq!(get_val(&array.get(0).unwrap()), 10.0);
    assert_eq!(array.get(0).unwrap().ndim(), 2);
    assert_eq!(array.get(1).unwrap().ndim(), 3);

    // Iteration
    let vals: Vec<f32> = array.iter().map(|t| get_val(&t)).collect();
    assert_eq!(vals, vec![10.0, 20.0]);
}

#[test]
fn test_array_mutation_and_growth() {
    // Start with 1 item
    let mut array = Array::new(vec![create_tensor(1.0, &[1])]);

    // 1. Test Push & Growth (triggers realloc if internal capacity reached)
    array.push(create_tensor(2.0, &[1]));
    array.push(create_tensor(3.0, &[1]));
    assert_eq!(array.len(), 3);

    // 2. Test Insert (checks pointer shifting and Any alignment)
    // Layout: [1.0, 3.0] -> [1.0, 2.5, 3.0]
    array.remove(1).unwrap(); // Remove 2.0
    array.insert(1, create_tensor(2.5, &[1])).unwrap();
    assert_eq!(get_val(&array.get(1).unwrap()), 2.5);

    // 3. Test Pop
    let last = array.pop().unwrap();
    assert_eq!(get_val(&last), 3.0);
    assert_eq!(array.len(), 2);

    // 4. Test Remove from start (checks shifting for all subsequent elements)
    let first = array.remove(0).unwrap();
    assert_eq!(get_val(&first), 1.0);
    assert_eq!(array.len(), 1);

    // 5. Test Clear
    array.clear();
    assert!(array.is_empty());
}

#[test]
fn test_array_extend_and_conversions() {
    let mut array = Array::new(vec![create_tensor(0.0, &[1])]);

    // Test Extend (Bulk growth)
    array.extend(vec![create_tensor(1.0, &[1]), create_tensor(2.0, &[1])]);
    assert_eq!(array.len(), 3);

    // Test Any/AnyView Roundtrip (Verifies AnyCompatible and Trait Bounds)
    let any = Any::from(array.clone());
    assert_eq!(any.type_index(), TypeIndex::kTVMFFIArray as i32);

    let back: Array<Tensor> = Array::try_from(any).expect("Any -> Array failed");
    assert_eq!(back.len(), 3);
    assert_eq!(get_val(&back.get(2).unwrap()), 2.0);

    let view = AnyView::from(&back);
    let back_from_view: Array<Tensor> = Array::try_from(view).expect("AnyView -> Array failed");
    assert_eq!(back_from_view.len(), 3);
}

#[test]
fn test_array_recursive_type_checking() {
    // 1. Create an Array of Shapes
    let shape_array = Array::new(vec![Shape::from(vec![1, 2]), Shape::from(vec![3])]);

    // 2. Wrap it in Any
    let any_val = Any::from(shape_array);

    // 3. Try to convert Any (containing Shapes) into Array<Tensor>
    // This should FAIL because T::check_any_strict (Tensor) will fail on Shape elements
    let tensor_cast = Array::<Tensor>::try_from(any_val.clone());
    assert!(
        tensor_cast.is_err(),
        "Should not be able to cast Array<Shape> to Array<Tensor>"
    );

    // 4. Verify valid cast works
    let shape_cast = Array::<Shape>::try_from(any_val);
    assert!(
        shape_cast.is_ok(),
        "Should be able to cast back to correct type"
    );
}

#[test]
fn test_array_parametric_heterogeneity() {
    // Verify Array works with different ObjectRefCore types
    let array = Array::new(vec![Shape::from(vec![1, 2, 3]), Shape::from(vec![10])]);
    assert_eq!(array.get(0).unwrap().as_slice(), &[1, 2, 3]);
    assert_eq!(array.get(1).unwrap().as_slice(), &[10]);

    let array = Array::new(vec![
        Function::get_global("testing.echo").unwrap(),
        Function::get_global("testing.apply").unwrap(),
    ]);
    assert_eq!(
        into_typed_fn!(
            array.get(0).unwrap(),
            Fn(&str) -> Result<String>
        )("hello")
        .unwrap(),
        "hello"
    );
    assert_eq!(
        into_typed_fn!(
            array.get(1).unwrap(),
            Fn(Function, i32) -> Result<i32>
        )(
            // add_one
            Function::from_typed(|x: i32| -> Result<i32> { Ok(x + 1) }),
            3
        )
        .unwrap(),
        4
    );
}

#[test]
fn test_array_copy_on_write() {
    let t1 = create_tensor(1.0, &[1]);
    let a = Array::new(vec![t1]);
    let mut b = a.clone();

    // Mutating B should NOT affect A
    b.push(create_tensor(2.0, &[1]));

    assert_eq!(a.len(), 1);
    assert_eq!(b.len(), 2);
    assert_eq!(get_val(&a.get(0).unwrap()), 1.0);
}
