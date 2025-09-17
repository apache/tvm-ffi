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
use crate::derive::{Object, ObjectRef};
use crate::object::{Object, ObjectArc};
use std::ffi::c_void;
use tvm_ffi_sys::TVMFFIBacktraceUpdateMode::kTVMFFIBacktraceUpdateModeAppend;
use tvm_ffi_sys::{
    TVMFFIByteArray, TVMFFIErrorCell, TVMFFIErrorCreate, TVMFFIErrorMoveFromRaised,
    TVMFFIErrorSetRaised, TVMFFIObjectHandle, TVMFFITypeIndex,
};

/// Error kind, wraps in a struct to be explicit
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErrorKind<'a>(&'a str);

impl<'a> ErrorKind<'a> {
    pub fn as_str(&self) -> &str {
        self.0
    }
}

impl<'a> std::fmt::Display for ErrorKind<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

pub const VALUE_ERROR: ErrorKind = ErrorKind("ValueError");
pub const TYPE_ERROR: ErrorKind = ErrorKind("TypeError");
pub const RUNTIME_ERROR: ErrorKind = ErrorKind("RuntimeError");
pub const ATTRIBUTE_ERROR: ErrorKind = ErrorKind("AttributeError");
pub const KEY_ERROR: ErrorKind = ErrorKind("KeyError");
pub const INDEX_ERROR: ErrorKind = ErrorKind("IndexError");

/// error object
#[repr(C)]
#[derive(Object)]
#[type_key = "ffi.Error"]
#[type_index(TVMFFITypeIndex::kTVMFFIError)]
pub struct ErrorObj {
    object: Object,
    cell: TVMFFIErrorCell,
}

/// Error reference class
#[derive(Clone, ObjectRef)]
pub struct Error {
    data: ObjectArc<ErrorObj>,
}

/// Default result that uses Error as the error type
pub type Result<T, E = Error> = std::result::Result<T, E>;

impl Error {
    pub fn new(kind: ErrorKind<'_>, message: &str, traceback: &str) -> Self {
        unsafe {
            let kind_data = TVMFFIByteArray::from_str(kind.as_str());
            let message_data = TVMFFIByteArray::from_str(message);
            let traceback_data = TVMFFIByteArray::from_str(traceback);
            let mut error_handle: TVMFFIObjectHandle = std::ptr::null_mut();
            let ret = TVMFFIErrorCreate(
                &kind_data,
                &message_data,
                &traceback_data,
                &mut error_handle,
            );
            assert_eq!(ret, 0, "Failed to create error object");
            let error_obj = ObjectArc::from_raw(error_handle as *const ErrorObj);
            Self { data: error_obj }
        }
    }

    /// Create a new error by moving from raised error
    ///
    /// # Returns
    /// The error from the raised error
    pub fn from_raised() -> Self {
        unsafe {
            let mut error_handle: TVMFFIObjectHandle = std::ptr::null_mut();
            TVMFFIErrorMoveFromRaised(&mut error_handle as *mut TVMFFIObjectHandle);
            assert!(
                !error_handle.is_null(),
                "Calling Error::from_raised but no error was raised"
            );
            let error_obj = ObjectArc::from_raw(error_handle as *const ErrorObj);
            Self { data: error_obj }
        }
    }

    /// Set the error as raised
    ///
    /// # Arguments
    /// * `error` - The error to set as raised
    pub fn set_raised(error: &Self) {
        unsafe {
            TVMFFIErrorSetRaised(ObjectArc::as_raw(&error.data) as TVMFFIObjectHandle);
        }
    }

    /// Get the kind of the error
    ///
    /// # Returns
    /// The kind of the error
    pub fn kind(&self) -> ErrorKind<'_> {
        ErrorKind(&self.data.cell.kind.as_str())
    }

    /// Get the message of the error
    ///
    /// # Returns
    /// The message of the error
    pub fn message(&self) -> &str {
        self.data.cell.message.as_str()
    }

    /// Get the backtrace of the error
    ///
    /// # Returns
    /// The backtrace of the error
    pub fn backtrace(&self) -> &str {
        self.data.cell.backtrace.as_str()
    }

    /// Get the traceback of the error in the order of most recent call last
    ///
    /// # Returns
    /// The traceback of the error
    pub fn traceback_most_recent_call_last(&self) -> String {
        let backtrace = self.backtrace();
        let backtrace_lines = backtrace.split('\n');
        let mut traceback = String::new();
        for line in backtrace_lines.rev() {
            traceback.push_str(line);
            traceback.push('\n');
        }
        traceback
    }

    /// Append the backtrace to the error
    ///
    /// # Arguments
    /// * `this` - The error to append the backtrace to
    /// * `backtrace` - The backtrace to append
    ///
    /// # Returns
    /// The error with the appended backtrace
    pub fn with_appended_backtrace(this: Self, backtrace: &str) -> Self {
        if ObjectArc::strong_count(&this.data) == 1 {
            // this is the only reference to the error
            // we can safely mutate the error
            unsafe {
                let backtrace_data = TVMFFIByteArray::from_str(backtrace);
                (this.data.cell.update_backtrace)(
                    ObjectArc::as_raw(&this.data) as *mut ErrorObj as *mut c_void,
                    &backtrace_data,
                    kTVMFFIBacktraceUpdateModeAppend as i32,
                );
                this
            }
        } else {
            // we need to create a new error because there is more than one unique reference
            // to the error
            let mut new_backtrace = String::new();
            new_backtrace.push_str(this.backtrace());
            new_backtrace.push_str(backtrace);
            return Error::new(this.kind(), this.message(), &new_backtrace);
        }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Traceback (most recent call last):\n{}{}: {}",
            self.traceback_most_recent_call_last(),
            self.kind().as_str(),
            self.message()
        )
    }
}

impl std::fmt::Debug for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self, f)
    }
}

impl std::error::Error for Error {}
