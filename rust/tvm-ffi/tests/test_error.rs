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
fn test_error_new() {
    let error = Error::new(RUNTIME_ERROR, "test error", "test traceback");
    assert_eq!(error.kind(), RUNTIME_ERROR);
    assert_eq!(error.message(), "test error");
    assert_eq!(error.backtrace(), "test traceback");
}

#[test]
fn test_error_from_raised() {
    let error0 = Error::new(RUNTIME_ERROR, "test error", "test traceback");
    Error::set_raised(&error0);
    let error1 = Error::from_raised();
    assert_eq!(error1.kind(), RUNTIME_ERROR);
    assert_eq!(error1.message(), "test error");
    assert_eq!(error1.backtrace(), "test traceback");
    // we have two references because one in error, and another one in the function call
    assert_eq!(ObjectArc::strong_count(Error::data(&error1)), 2);
}

fn error_fn0(flag: bool) -> Result<i32> {
    // if flag is true, throw an error
    ensure!(!flag, RUNTIME_ERROR, "test error {flag}");
    Ok(1)
}

fn error_fn1(flag: bool) -> Result<i32> {
    attach_context!(error_fn0(flag))
}

#[test]
fn test_error_with_context() {
    let error0 = attach_context!(error_fn1(true)).unwrap_err();
    assert_eq!(error0.kind(), RUNTIME_ERROR);
    assert_eq!(error0.message(), "test error true");
    assert!(error0.backtrace().contains("test_error.rs"));
    assert!(error0.backtrace().contains("error_fn0"));
    assert!(error0.backtrace().contains("error_fn1"));
    assert!(error0.backtrace().contains("test_error_with_context"));
    let result = error_fn1(false).unwrap();
    assert_eq!(result, 1);

    let payload = object::ObjectRef::try_from(Any::from(Array::new(vec![7i64]))).unwrap();
    for shared in [false, true] {
        let error = Error::new_with_cause_and_extra_context(
            RUNTIME_ERROR,
            "outer",
            "origin",
            Some(&error0),
            Some(&payload),
        );
        let retained = shared.then(|| error.clone());
        let output = Error::with_appended_backtrace(error, " frame");
        assert_eq!(output.backtrace(), "origin frame");
        assert!(output.cause_chain().unwrap().same_as(&error0));
        assert!(output.extra_context().unwrap().same_as(&payload));
        if let Some(retained) = retained {
            assert_eq!(retained.backtrace(), "origin");
        }
    }
    assert_eq!(ObjectArc::strong_count(Error::data(&error0)), 1);
    assert_eq!(
        ObjectArc::strong_count(object::ObjectRef::data(&payload)),
        1
    );
}

#[test]
fn test_error_any_convert() {
    let error = Error::new(RUNTIME_ERROR, "test error", "test traceback");
    let any = Any::from(error.clone());
    let error2 = any.clone().try_as::<Error>().unwrap();
    assert_eq!(error2.kind(), RUNTIME_ERROR);
    assert_eq!(error2.message(), "test error");
    assert_eq!(error2.backtrace(), "test traceback");

    let res = i32::try_from(any);
    assert!(res.is_err());
    assert!(res.unwrap_err().message().contains("`ffi.Error` to `int`"));

    let anyview = AnyView::from(&error);
    let error3 = Error::try_from(anyview).unwrap();
    assert_eq!(error3.kind(), RUNTIME_ERROR);
    assert_eq!(error3.message(), "test error");
    assert_eq!(error3.backtrace(), "test traceback");
}
