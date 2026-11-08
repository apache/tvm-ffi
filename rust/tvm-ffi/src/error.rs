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
//! [`Error`], the error that crosses the FFI boundary, and its [`ErrorKind`]s.

use crate::derive::{Object, ObjectRef};
use crate::object::{self, Object, ObjectArc, ObjectRefCore};
use crate::type_traits::AnyCompatible;
use std::ffi::c_void;
use tvm_ffi_sys::TVMFFIBacktraceUpdateMode::{
    kTVMFFIBacktraceUpdateModeAppend, kTVMFFIBacktraceUpdateModeReplace,
};
use tvm_ffi_sys::{
    TVMFFIAny, TVMFFIByteArray, TVMFFIErrorCell, TVMFFIErrorCreateWithCauseAndExtraContext,
    TVMFFIErrorMoveFromRaised, TVMFFIErrorSetRaised, TVMFFIObjectHandle, TVMFFITypeIndex,
};

/// The object behind an [`Error`]: its kind, message, backtrace, cause and context.
#[repr(C)]
#[derive(Object)]
#[object(type_key = "ffi.Error", unsafe type_index = TVMFFITypeIndex::kTVMFFIError)]
pub struct ErrorObj {
    object: Object,
    cell: TVMFFIErrorCell,
}

/// An error that crosses the FFI boundary, with a kind, a message and a
/// backtrace.
///
/// An `Error` raised in another language arrives in Rust as an `Err`, and one
/// returned from a Rust [`Function`](crate::Function) is raised in the caller's
/// language. Create one with [`bail!`](crate::bail) or
/// [`ensure!`](crate::ensure), which also record the source location.
///
/// # Examples
///
/// ```
/// use tvm_ffi::{bail, Result, VALUE_ERROR};
///
/// fn parse_digit(c: char) -> Result<u32> {
///     match c.to_digit(10) {
///         Some(digit) => Ok(digit),
///         None => bail!(VALUE_ERROR, "not a digit: {:?}", c),
///     }
/// }
///
/// let error = parse_digit('x').unwrap_err();
/// assert_eq!(error.kind(), VALUE_ERROR);
/// assert_eq!(error.message(), "not a digit: 'x'");
/// ```
///
/// An `Error` is `Send + Sync`, so a `Result` can cross threads.
#[repr(C)]
#[derive(Clone, ObjectRef)]
#[object_ref(skip_deref, skip_debug, skip_display)]
pub struct Error {
    data: ObjectArc<ErrorObj>,
}

// SAFETY: a shared Error is never mutated: updates copy unless this handle is unique.
unsafe impl Send for Error {}
unsafe impl Sync for Error {}

/// A `Result` whose error type defaults to [`Error`].
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// The kind of an [`Error`], such as `ValueError`, matching the exception
/// type that Python raises.
///
/// Use the constants such as [`VALUE_ERROR`] and [`TYPE_ERROR`], or
/// [`ErrorKind::new("MyError")`](ErrorKind::new) for a custom kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ErrorKind<'a>(&'a str);

impl<'a> ErrorKind<'a> {
    /// The kind named `name`, which Python raises as that exception type.
    ///
    /// ```
    /// use tvm_ffi::{Error, ErrorKind};
    /// const MY_ERROR: ErrorKind = ErrorKind::new("MyError");
    /// let error = Error::new(MY_ERROR, "bad input");
    /// assert_eq!(error.kind().as_str(), "MyError");
    /// ```
    pub const fn new(name: &'a str) -> Self {
        Self(name)
    }

    /// Get the name of the kind.
    ///
    /// # Returns
    ///
    /// The name, such as `"ValueError"`.
    pub fn as_str(&self) -> &'a str {
        self.0
    }
}

impl<'a> std::fmt::Display for ErrorKind<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// `ValueError`: an argument has the right type but a bad value.
pub const VALUE_ERROR: ErrorKind = ErrorKind::new("ValueError");
/// `TypeError`: a value has the wrong type.
pub const TYPE_ERROR: ErrorKind = ErrorKind::new("TypeError");
/// `RuntimeError`: an operation failed for another reason.
pub const RUNTIME_ERROR: ErrorKind = ErrorKind::new("RuntimeError");
/// `AttributeError`: a field, method or attribute does not exist.
pub const ATTRIBUTE_ERROR: ErrorKind = ErrorKind::new("AttributeError");
/// `KeyError`: a map has no such key.
pub const KEY_ERROR: ErrorKind = ErrorKind::new("KeyError");
/// `IndexError`: an index is out of range.
pub const INDEX_ERROR: ErrorKind = ErrorKind::new("IndexError");
/// `InternalError`: a bug, such as a broken invariant or a caught panic.
pub const INTERNAL_ERROR: ErrorKind = ErrorKind::new("InternalError");
/// `OverflowError`: a result does not fit its type.
pub const OVERFLOW_ERROR: ErrorKind = ErrorKind::new("OverflowError");
/// `ZeroDivisionError`: a division or remainder by zero.
pub const ZERO_DIVISION_ERROR: ErrorKind = ErrorKind::new("ZeroDivisionError");

impl Error {
    /// The `InternalError` for a panic caught at an FFI boundary, where unwinding
    /// into the caller would abort the process. The message is the panic's, when it
    /// is a string.
    ///
    /// ```
    /// use tvm_ffi::{Error, INTERNAL_ERROR};
    /// let payload = std::panic::catch_unwind(|| panic!("bad state")).unwrap_err();
    /// let error = Error::from_panic(payload);
    /// assert_eq!(error.kind(), INTERNAL_ERROR);
    /// assert_eq!(error.message(), "panicked: bad state");
    /// ```
    pub fn from_panic(payload: Box<dyn std::any::Any + Send>) -> Self {
        let message = payload
            .downcast_ref::<&str>()
            .copied()
            .or_else(|| {
                payload
                    .downcast_ref::<std::string::String>()
                    .map(|s| s.as_str())
            })
            .unwrap_or("unknown payload");
        Self::new(INTERNAL_ERROR, &format!("panicked: {message}"))
    }

    /// The `TypeError` for a reference that [`downcast`](crate::ObjectRefCore::downcast)
    /// handed back because it is not a `B`, naming both types.
    ///
    /// ```
    /// use tvm_ffi::{object::ObjectRef, Array, Error, ObjectRefCore, TYPE_ERROR};
    /// let object: ObjectRef = Array::from_vec(vec![1_i64]).into();
    /// let error = object.downcast::<Array<f64>>().map_err(Error::downcast_failed::<Array<f64>>);
    /// assert_eq!(error.unwrap_err().kind(), TYPE_ERROR);
    /// ```
    pub fn downcast_failed<B: AnyCompatible>(source: impl object::ObjectRefCore) -> Self {
        let mut cell = TVMFFIAny::new();
        // SAFETY: `source` stays alive while its borrowed cell is inspected.
        unsafe { AnyCompatible::copy_to_any_view(&source, &mut cell) };
        let message = format!(
            "Cannot convert from type `{}` to `{}`",
            B::get_mismatch_type_info(&cell),
            B::type_str()
        );
        Self::new(TYPE_ERROR, &message)
    }

    /// Create an error with an empty backtrace.
    ///
    /// Prefer [`format_err!`](crate::format_err) or [`bail!`](crate::bail),
    /// which also record the source location. Add the other parts with
    /// [`with_backtrace`](Self::with_backtrace),
    /// [`with_cause`](Self::with_cause) and
    /// [`with_extra_context`](Self::with_extra_context).
    ///
    /// # Arguments
    ///
    /// * `kind` - The kind of the error, such as [`VALUE_ERROR`].
    /// * `message` - The message of the error.
    ///
    /// # Returns
    ///
    /// The error.
    pub fn new(kind: ErrorKind<'_>, message: &str) -> Self {
        Self::create(kind, message, "", None, std::ptr::null_mut())
    }

    /// Set `cause` as the error that caused this one, the
    /// [`cause_chain`](Self::cause_chain). The error is updated in place when
    /// it is not shared, and copied otherwise.
    ///
    /// ```
    /// use tvm_ffi::{Error, RUNTIME_ERROR, VALUE_ERROR};
    /// let cause = Error::new(VALUE_ERROR, "bad input");
    /// let error = Error::new(RUNTIME_ERROR, "load failed").with_cause(&cause);
    /// assert_eq!(error.cause_chain().unwrap().message(), "bad input");
    /// ```
    #[inline]
    pub fn with_cause(mut self, cause: &Error) -> Self {
        *self.cause_chain_mut() = Some(cause.clone());
        self
    }

    /// Set `context` as the [`extra_context`](Self::extra_context), an object
    /// of application-specific context; `Send + Sync`, since the error is.
    /// The error is updated in place when it is not shared, and copied
    /// otherwise.
    #[inline]
    pub fn with_extra_context<T: ObjectRefCore + Send + Sync>(mut self, context: &T) -> Self {
        *self.extra_context_mut() = Some(object::into_root_object_ref(context.clone()));
        self
    }

    /// Create an error from its parts, with `extra_context` a borrowed object
    /// handle or null.
    fn create(
        kind: ErrorKind<'_>,
        message: &str,
        backtrace: &str,
        cause_chain: Option<&Error>,
        extra_context: TVMFFIObjectHandle,
    ) -> Self {
        unsafe {
            let kind_data = TVMFFIByteArray::from_str(kind.as_str());
            let message_data = TVMFFIByteArray::from_str(message);
            let backtrace_data = TVMFFIByteArray::from_str(backtrace);
            let mut error_handle: TVMFFIObjectHandle = std::ptr::null_mut();
            let ret = TVMFFIErrorCreateWithCauseAndExtraContext(
                &kind_data,
                &message_data,
                &backtrace_data,
                cause_chain.map_or(std::ptr::null_mut(), |cause| {
                    ObjectArc::as_ptr(&cause.data) as TVMFFIObjectHandle
                }),
                extra_context,
                &mut error_handle,
            );
            assert_eq!(ret, 0, "Failed to create error object");
            let error_obj = ObjectArc::from_raw(error_handle as *const ErrorObj);
            Self { data: error_obj }
        }
    }

    /// Create a new error by moving from the error that the last failed FFI call
    /// raised on this thread, as set by [`Error::set_raised`]. An associated
    /// function rather than a constructor from a value: it takes the thread's
    /// state.
    ///
    /// # Returns
    ///
    /// The raised error.
    ///
    /// # Panics
    ///
    /// Panics if no error was raised.
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

    /// Set `error` as the raised error of the calling thread, for the caller of
    /// an `extern "C"` function, which then returns `-1`, to take with
    /// [`Error::from_raised`]. An associated function rather than a method: it
    /// changes the thread's state, not `error`.
    ///
    /// # Arguments
    ///
    /// * `error` - The error to set as raised.
    pub fn set_raised(error: &Self) {
        unsafe {
            TVMFFIErrorSetRaised(ObjectArc::as_ptr(&error.data) as TVMFFIObjectHandle);
        }
    }

    /// Get the kind of the error.
    ///
    /// # Returns
    ///
    /// The kind of the error, such as [`VALUE_ERROR`].
    pub fn kind(&self) -> ErrorKind<'_> {
        // SAFETY: the error cell holds UTF-8 text that lives as long as the error.
        ErrorKind(unsafe { self.data.cell.kind.as_str() })
    }

    /// Get the message of the error.
    ///
    /// # Returns
    ///
    /// The message of the error.
    pub fn message(&self) -> &str {
        // SAFETY: the error cell holds UTF-8 text that lives as long as the error.
        unsafe { self.data.cell.message.as_str() }
    }

    /// Get the backtrace of the error.
    ///
    /// It may be rewritten when the error is re-raised through another
    /// language: copy it to keep it across FFI calls.
    ///
    /// # Returns
    ///
    /// The backtrace, most recent call first.
    pub fn backtrace(&self) -> &str {
        // SAFETY: the error cell holds UTF-8 text that lives as long as the error.
        unsafe { self.data.cell.backtrace.as_str() }
    }

    /// The error that caused this one, the next link of the cause chain; the
    /// same error that `std::error::Error::source` returns.
    pub fn cause_chain(&self) -> Option<&Error> {
        let handle = &self.data.cell.cause_chain;
        if handle.is_null() {
            return None;
        }
        // SAFETY: the cell owns a non-null handle to an error object, and an
        // `Error` is exactly that one object pointer.
        Some(unsafe { &*(handle as *const _ as *const Error) })
    }

    /// The application-specific context object, if present.
    ///
    /// The context is shared with every thread the error reaches, so it must be
    /// safe to share; a context set by another language is the setter's
    /// responsibility.
    pub fn extra_context(&self) -> Option<&object::ObjectRef> {
        let handle = &self.data.cell.extra_context;
        if handle.is_null() {
            return None;
        }
        // SAFETY: the cell owns a non-null object handle, and an `ObjectRef`
        // is exactly that one object pointer.
        Some(unsafe { &*(handle as *const _ as *const object::ObjectRef) })
    }

    /// The `cause_chain` slot, to update in place.
    fn cause_chain_mut(&mut self) -> &mut Option<Error> {
        let error = self.make_mut();
        // SAFETY: this handle is the only reference, and the slot owns a nullable
        // error handle, which is exactly an `Option<Error>`.
        unsafe { &mut *std::ptr::addr_of_mut!((*error).cell.cause_chain).cast::<Option<Error>>() }
    }

    /// The `extra_context` slot, to update in place.
    pub(crate) fn extra_context_mut(&mut self) -> &mut Option<object::ObjectRef> {
        let error = self.make_mut();
        // SAFETY: this handle is the only reference, and the slot owns a nullable
        // object handle, which is exactly an `Option<ObjectRef>`.
        unsafe {
            &mut *std::ptr::addr_of_mut!((*error).cell.extra_context)
                .cast::<Option<object::ObjectRef>>()
        }
    }

    /// Get the traceback of the error in the order of most recent call last, as
    /// Python prints it.
    ///
    /// # Returns
    ///
    /// The traceback of the error.
    pub fn traceback_most_recent_call_last(&self) -> String {
        let backtrace = self.backtrace();
        let mut traceback = String::new();
        for line in backtrace.split('\n').rev().filter(|line| !line.is_empty()) {
            traceback.push_str(line);
            traceback.push('\n');
        }
        traceback
    }

    /// Replace the backtrace, most recent call first. The error is updated in
    /// place when it is not shared, and copied otherwise.
    #[inline]
    pub fn with_backtrace(mut self, backtrace: &str) -> Self {
        let error = self.make_mut();
        // SAFETY: `make_mut` made this handle the only reference to `error`.
        unsafe {
            let backtrace_data = TVMFFIByteArray::from_str(backtrace);
            ((*error).cell.update_backtrace)(
                error.cast::<c_void>(),
                &backtrace_data,
                kTVMFFIBacktraceUpdateModeReplace as i32,
            );
        }
        self
    }

    /// Append the backtrace to the error.
    ///
    /// [`attach_context!`](crate::attach_context) appends the current source
    /// location this way. The error is updated in place when it is not shared, and
    /// copied otherwise.
    ///
    /// # Arguments
    ///
    /// * `backtrace` - The backtrace to append.
    ///
    /// # Returns
    ///
    /// The error with the appended backtrace.
    pub fn with_appended_backtrace(mut self, backtrace: &str) -> Self {
        let error = self.make_mut();
        // SAFETY: `make_mut` made this handle the only reference to `error`.
        unsafe {
            let backtrace_data = TVMFFIByteArray::from_str(backtrace);
            ((*error).cell.update_backtrace)(
                error.cast::<c_void>(),
                &backtrace_data,
                kTVMFFIBacktraceUpdateModeAppend as i32,
            );
        }
        self
    }

    /// Make this handle the only reference to its error, copying a shared one, as
    /// `Arc::make_mut`.
    fn make_mut(&mut self) -> *mut ErrorObj {
        if !ObjectArc::is_unique(&self.data) {
            *self = Self::create(
                self.kind(),
                self.message(),
                self.backtrace(),
                self.cause_chain(),
                self.data.cell.extra_context,
            );
        }
        ObjectArc::as_mut_ptr(&mut self.data)
    }
}

/// Formats the kind and the message, such as `ValueError: bad x`, as C++
/// `what()`; [`Debug`](std::fmt::Debug) adds the traceback.
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.kind().as_str(), self.message())
    }
}

/// Formats the traceback, most recent call last, then the kind and the
/// message, as Python prints an exception; `unwrap` and `expect` print it.
impl std::fmt::Debug for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.backtrace().is_empty() {
            return std::fmt::Display::fmt(self, f);
        }
        write!(
            f,
            "Traceback (most recent call last):\n{}{self}",
            self.traceback_most_recent_call_last(),
        )
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.cause_chain().map(|cause| cause as _)
    }
}
