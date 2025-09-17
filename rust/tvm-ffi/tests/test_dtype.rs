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

#[test]
fn test_dl_datatype_from_string() {
    // Test valid dtype strings
    let test_cases = vec![
        (
            "int32",
            DLDataType {
                code: DLDataTypeCode::kDLInt as u8,
                bits: 32,
                lanes: 1,
            },
        ),
        (
            "float32",
            DLDataType {
                code: DLDataTypeCode::kDLFloat as u8,
                bits: 32,
                lanes: 1,
            },
        ),
        (
            "int8",
            DLDataType {
                code: DLDataTypeCode::kDLInt as u8,
                bits: 8,
                lanes: 1,
            },
        ),
        (
            "float64",
            DLDataType {
                code: DLDataTypeCode::kDLFloat as u8,
                bits: 64,
                lanes: 1,
            },
        ),
        (
            "uint32",
            DLDataType {
                code: DLDataTypeCode::kDLUInt as u8,
                bits: 32,
                lanes: 1,
            },
        ),
    ];

    for item in test_cases {
        let dtype = DLDataType::try_from_str(item.0).unwrap();
        assert_eq!(dtype, item.1);
    }
}

#[test]
fn test_dl_datatype_any_conversion() {
    let original_dtype = DLDataType {
        code: DLDataTypeCode::kDLInt as u8,
        bits: 32,
        lanes: 1,
    };

    // Test conversion to Any
    let any_dtype: Any = original_dtype.into();

    // Test conversion back from Any
    let converted_dtype: DLDataType = any_dtype.try_into().unwrap();
    assert_eq!(converted_dtype.code, original_dtype.code);
    assert_eq!(converted_dtype.bits, original_dtype.bits);
    assert_eq!(converted_dtype.lanes, original_dtype.lanes);
}

#[test]
fn test_dl_datatype_any_with_string_conversion() {
    // Test that DLDataType can be converted from string via Any
    let dtype_str = "int32";

    let dtype = DLDataType::try_from_str(dtype_str).unwrap();

    // Convert to Any
    let any_dtype: Any = dtype.into();

    // Convert back from Any
    let converted_dtype: DLDataType = any_dtype.try_into().unwrap();
    assert_eq!(converted_dtype.code, dtype.code);
    assert_eq!(converted_dtype.bits, dtype.bits);
    assert_eq!(converted_dtype.lanes, dtype.lanes);
}
