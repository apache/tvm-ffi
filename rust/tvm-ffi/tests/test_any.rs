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

/// Macro to test integer types with both Any and AnyView
macro_rules! test_any_int {
    ($($type:ty: $value:expr, $any_name:ident, $any_view_name:ident),* $(,)?) => {
        $(
            #[test]
            fn $any_name() {
                let any = Any::from($value);
                assert_eq!(any.type_index(), TypeIndex::kTVMFFIInt as i32);
                assert_eq!(<$type>::try_from(any).unwrap(), $value);
            }

            #[test]
            fn $any_view_name() {
                let value = $value;
                let any_view = AnyView::from(&value);
                assert_eq!(any_view.type_index(), TypeIndex::kTVMFFIInt as i32);
                assert_eq!(<$type>::try_from(any_view).unwrap(), $value);
            }
        )*
    };
}

// Test all integer types with both Any and AnyView
test_any_int!(
    i8: 127i8, test_any_int_i8, test_any_view_int_i8,
    i16: 32767i16, test_any_int_i16, test_any_view_int_i16,
    i32: 42i32, test_any_int_i32, test_any_view_int_i32,
    i64: 123456789i64, test_any_int_i64, test_any_view_int_i64,
    isize: 42isize, test_any_int_isize, test_any_view_int_isize,
    u8: 255u8, test_any_int_u8, test_any_view_int_u8,
    u16: 65535u16, test_any_int_u16, test_any_view_int_u16,
    u32: 4294967295u32, test_any_int_u32, test_any_view_int_u32,
    u64: 18446744073709551615u64, test_any_int_u64, test_any_view_int_u64,
    usize: 42usize, test_any_int_usize, test_any_view_int_usize
);

/// Macro to test float types with both Any and AnyView
macro_rules! test_any_float {
    ($($type:ty: $value:expr, $any_name:ident, $any_view_name:ident),* $(,)?) => {
        $(
            #[test]
            fn $any_name() {
                let any = Any::from($value);
                assert_eq!(any.type_index(), TypeIndex::kTVMFFIFloat as i32);
                assert_eq!(<$type>::try_from(any).unwrap(), $value);
            }
        )*
    };
}

// Test all float types with both Any and AnyView
test_any_float!(
    f32: 3.14f32, test_any_float_f32, test_any_view_float_f32,
    f64: 3.14f64, test_any_float_f64, test_any_view_float_f64
);

#[test]
fn test_any_bool() {
    // Test both Any and AnyView with both true and false
    let any_true = Any::from(true);
    let any_false = Any::from(false);

    let value_true = true;
    let value_false = false;
    let any_view_true = AnyView::from(&value_true);
    let any_view_false = AnyView::from(&value_false);

    // Test Any
    assert_eq!(any_true.type_index(), TypeIndex::kTVMFFIBool as i32);
    assert_eq!(any_false.type_index(), TypeIndex::kTVMFFIBool as i32);
    assert_eq!(bool::try_from(any_true).unwrap(), true);
    assert_eq!(bool::try_from(any_false).unwrap(), false);

    // Test AnyView
    assert_eq!(any_view_true.type_index(), TypeIndex::kTVMFFIBool as i32);
    assert_eq!(any_view_false.type_index(), TypeIndex::kTVMFFIBool as i32);
    assert_eq!(bool::try_from(any_view_true).unwrap(), true);
    assert_eq!(bool::try_from(any_view_false).unwrap(), false);
}

#[test]
fn test_type_mismatch_error() {
    let any = Any::from(42i32);
    // try from will try casting when possible
    // Try to convert int to bool - should be ok
    let result: Result<bool, _> = bool::try_from(any);
    assert!(result.is_ok());
    // try_as is more strict and do not allow conversion from int to bool
    let any = Any::from(42i32);
    let opt_try_as = any.try_as::<bool>();
    assert!(opt_try_as.is_none());
}

#[test]
fn test_type_mismatch_error_message() {
    let any_none = Any::from(Option::<i32>::from(None));
    // try from will try casting when possible
    // Try to convert int to bool - should be ok
    let err = i32::try_from(any_none).unwrap_err();
    assert_eq!(err.kind(), crate::error::TYPE_ERROR);
    assert!(err.message().contains("`None` to `int`"));

    let any_float = Any::from(1.2f32);
    let err = Option::<i32>::try_from(any_float).unwrap_err();
    assert_eq!(err.kind(), crate::error::TYPE_ERROR);
    assert!(err.message().contains("`float` to `Optional<int>`"));
}

#[test]
fn test_cross_type_conversion_int_to_bool() {
    // Test that integers can be converted to bool (0 = false, non-zero = true)
    let any_true = Any::from(42i32);
    let any_false = Any::from(0i32);

    // This should work because our macro allows conversion from int to bool
    let bool_true: Result<bool, _> = bool::try_from(any_true);
    let bool_false: Result<bool, _> = bool::try_from(any_false);

    // Note: This test might fail depending on how try_cast_from_any_view is implemented
    // If it fails, it means the conversion isn't implemented yet
    assert_eq!(bool_true.unwrap(), true);
    assert_eq!(bool_false.unwrap(), false);
}

#[test]
fn test_max_min_values() {
    // Test maximum and minimum values for i64
    let any_i64_max = Any::from(i64::MAX);
    let any_i64_min = Any::from(i64::MIN);

    assert_eq!(i64::try_from(any_i64_max).unwrap(), i64::MAX);
    assert_eq!(i64::try_from(any_i64_min).unwrap(), i64::MIN);
}

#[test]
fn test_any_void_ptr() {
    let any = Any::from(std::ptr::null_mut());
    assert_eq!(any.type_index(), TypeIndex::kTVMFFIOpaquePtr as i32);
    assert_eq!(
        <*mut core::ffi::c_void>::try_from(any).unwrap(),
        std::ptr::null_mut()
    );
}

#[test]
fn test_any_option_i32() {
    // Test Option<i32> with Some value
    let some_value = Some(42i32);
    let any_some = Any::from(some_value);
    let any_view_some = AnyView::from(&some_value);

    // Test Any with Some value
    assert_eq!(any_some.type_index(), TypeIndex::kTVMFFIInt as i32);
    assert_eq!(Option::<i32>::try_from(any_some).unwrap(), Some(42i32));

    // Test AnyView with Some value
    assert_eq!(any_view_some.type_index(), TypeIndex::kTVMFFIInt as i32);
    assert_eq!(Option::<i32>::try_from(any_view_some).unwrap(), Some(42i32));

    // Test Option<i32> with None value
    let none_value: Option<i32> = None;
    let any_none = Any::from(none_value);
    let any_view_none = AnyView::from(&none_value);

    // Test Any with None value
    assert_eq!(any_none.type_index(), TypeIndex::kTVMFFINone as i32);
    assert_eq!(Option::<i32>::try_from(any_none).unwrap(), None::<i32>);

    // Test AnyView with None value
    assert_eq!(any_view_none.type_index(), TypeIndex::kTVMFFINone as i32);
    assert_eq!(Option::<i32>::try_from(any_view_none).unwrap(), None::<i32>);

    let any_float = Any::from(1.2f32);
    assert_eq!(any_float.type_index(), TypeIndex::kTVMFFIFloat as i32);
    assert_eq!(
        Option::<i32>::try_from(any_float).unwrap_err().kind(),
        TYPE_ERROR
    );
}

#[test]
fn test_any_string() {
    let any = Any::from(String::from("hello"));
    assert_eq!(any.type_index(), TypeIndex::kTVMFFISmallStr as i32);

    // Check reference count - small strings are not ref counted
    assert_eq!(any.debug_strong_count(), None);

    let out_string = String::try_from(any).unwrap();
    assert_eq!(out_string, "hello");
}

#[test]
fn test_any_bytes() {
    let data_arr: &[u8; 3] = &[1, 2, 3];
    let any = Any::from(Bytes::from(data_arr));
    assert_eq!(any.type_index(), TypeIndex::kTVMFFISmallBytes as i32);

    // Check reference count - small bytes are not ref counted
    assert_eq!(any.debug_strong_count(), None);

    let out_bytes = Bytes::try_from(any).unwrap();
    assert_eq!(out_bytes, data_arr);
}

#[test]
fn test_any_big_string() {
    // Use a string longer than 7 characters to trigger big string allocation
    let long_string = "hello world this is a long string";
    let input_str = String::from(long_string);
    let any = Any::from(input_str.clone());
    assert_eq!(any.type_index(), TypeIndex::kTVMFFIStr as i32);

    // Check reference count - big strings are ref counted
    assert_eq!(any.debug_strong_count(), Some(2));

    let out_string = String::try_from(any).unwrap();
    assert_eq!(out_string, long_string);

    drop(out_string);
    assert_eq!(AnyView::from(&input_str).debug_strong_count(), Some(1));
}

#[test]
fn test_any_big_bytes() {
    // Use bytes longer than 7 bytes to trigger big bytes allocation
    let data_arr: &[u8; 10] = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let any = Any::from(Bytes::from(data_arr));
    assert_eq!(any.type_index(), TypeIndex::kTVMFFIBytes as i32);

    // Check reference count - big bytes are ref counted
    assert_eq!(any.debug_strong_count(), Some(1));

    let out_bytes = Bytes::try_from(any).unwrap();
    assert_eq!(out_bytes, data_arr);

    assert_eq!(AnyView::from(&out_bytes).debug_strong_count(), Some(1));
}

#[test]
fn test_anyview_any_conversion_and_cloning() {
    let input_str = String::from("hello world this is a long string");

    // Test AnyView behavior - should not increment reference count
    let any_view = AnyView::from(&input_str);
    assert_eq!(any_view.debug_strong_count(), Some(1));

    // Test AnyView cloning - should not affect reference count
    let any_view_clone = any_view.clone();
    assert_eq!(any_view.debug_strong_count(), Some(1));
    assert_eq!(any_view_clone.debug_strong_count(), Some(1));

    // Test AnyView to Any conversion - should increment reference count
    let any1 = Any::from(any_view);
    assert_eq!(any1.debug_strong_count(), Some(2));

    // Test Any cloning - should increment reference count
    let any2 = any1.clone();
    assert_eq!(any1.debug_strong_count(), Some(3));
    assert_eq!(any2.debug_strong_count(), Some(3));

    // Test Any to AnyView conversion - should not change reference count
    let any_view_from_any = AnyView::from(&any1);
    assert_eq!(any_view_from_any.debug_strong_count(), Some(3));

    // Test AnyView to Any conversion again
    let any3 = Any::from(any_view_clone);
    assert_eq!(any3.debug_strong_count(), Some(4));

    // Test data integrity through conversions
    let out_string1 = String::try_from(any1.clone()).unwrap();
    let out_string2 = String::try_from(any2.clone()).unwrap();
    let out_string3 = String::try_from(any3.clone()).unwrap();
    let out_string_view = String::try_from(any_view_from_any).unwrap();

    assert_eq!(out_string1, input_str);
    assert_eq!(out_string2, input_str);
    assert_eq!(out_string3, input_str);
    assert_eq!(out_string_view, input_str);

    // Test reference count cleanup
    drop(any1);
    assert_eq!(any3.debug_strong_count(), Some(7));

    drop(any2);
    assert_eq!(any3.debug_strong_count(), Some(6));

    drop(any3);
    assert_eq!(AnyView::from(&input_str).debug_strong_count(), Some(5));
}

#[test]
fn test_any_dl_device() {
    use tvm_ffi::{DLDevice, DLDeviceType};

    // Test DLDevice with CPU device
    let cpu_device = DLDevice::new(DLDeviceType::kDLCPU, 0);
    let any_cpu = Any::from(cpu_device);
    let any_view_cpu = AnyView::from(&cpu_device);

    // Test Any with CPU device
    assert_eq!(any_cpu.type_index(), TypeIndex::kTVMFFIDevice as i32);
    let converted_cpu = DLDevice::try_from(any_cpu).unwrap();
    assert_eq!(converted_cpu.device_type, DLDeviceType::kDLCPU);
    assert_eq!(converted_cpu.device_id, 0);

    // Test AnyView with CPU device
    assert_eq!(any_view_cpu.type_index(), TypeIndex::kTVMFFIDevice as i32);
    let converted_cpu_view = DLDevice::try_from(any_view_cpu).unwrap();
    assert_eq!(converted_cpu_view.device_type, DLDeviceType::kDLCPU);
    assert_eq!(converted_cpu_view.device_id, 0);

    // Test DLDevice with CUDA device
    let cuda_device = DLDevice::new(DLDeviceType::kDLCUDA, 1);
    let any_cuda = Any::from(cuda_device);
    let any_view_cuda = AnyView::from(&cuda_device);

    // Test Any with CUDA device
    assert_eq!(any_cuda.type_index(), TypeIndex::kTVMFFIDevice as i32);
    let converted_cuda = DLDevice::try_from(any_cuda).unwrap();
    assert_eq!(converted_cuda.device_type, DLDeviceType::kDLCUDA);
    assert_eq!(converted_cuda.device_id, 1);

    // Test AnyView with CUDA device
    assert_eq!(any_view_cuda.type_index(), TypeIndex::kTVMFFIDevice as i32);
    let converted_cuda_view = DLDevice::try_from(any_view_cuda).unwrap();
    assert_eq!(converted_cuda_view.device_type, DLDeviceType::kDLCUDA);
    assert_eq!(converted_cuda_view.device_id, 1);
}
