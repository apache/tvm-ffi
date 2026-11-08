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
//! [`Function`], a callable shared with C++ and Python, and the global function
//! registry.

pub use crate::function_internal::{AsPackedCallable, PackedCallArg, PackedCallArgs};

use crate::any::{Any, AnyView};
use crate::derive::{Object, ObjectRef};
use crate::error::{Error, Result};
use crate::object::{Object, ObjectArc, ObjectCore};
use crate::type_traits::AnyCompatible;
use tvm_ffi_sys::{
    TVMFFIAny, TVMFFIByteArray, TVMFFIFunctionCell, TVMFFIFunctionCreate, TVMFFIFunctionGetGlobal,
    TVMFFIFunctionSetGlobalFromMethodInfo, TVMFFIMethodInfo, TVMFFIObjectHandle,
    TVMFFISafeCallType, TVMFFITypeIndex,
};

/// The object behind a [`Function`].
#[repr(C)]
#[derive(Object)]
#[object(type_key = "ffi.Function", unsafe type_index = TVMFFITypeIndex::kTVMFFIFunction)]
pub struct FunctionObj {
    object: Object,
    cell: TVMFFIFunctionCell,
}

/// A type-erased callable that works across languages.
///
/// A `Function` takes [`AnyView`] arguments and returns an [`Any`]. Make one
/// from a Rust closure with [`from_typed`](Self::from_typed), look one up by
/// name with [`get_global`](Self::get_global), and call it with
/// [`call_tuple`](Self::call_tuple) or [`call_packed`](Self::call_packed).
/// [`into_typed_fn!`](crate::into_typed_fn) turns it back into a typed closure.
///
/// # Examples
///
/// ```
/// use tvm_ffi::{Function, Result};
///
/// let add = Function::from_typed(|x: i64, y: i64| -> Result<i64> { Ok(x + y) });
/// let sum = i64::try_from(add.call_tuple((1_i64, 2_i64))?)?;
/// assert_eq!(sum, 3);
/// # Ok::<(), tvm_ffi::Error>(())
/// ```
///
/// A `Function` may be called from any thread, so a Rust callback must be
/// `Send + Sync`; a callback holding an `Rc` is rejected. A function written
/// in another language is as thread-safe as its implementation.
///
/// ```compile_fail,E0277
/// use std::rc::Rc;
/// let count = Rc::new(0_i64);
/// tvm_ffi::Function::from_typed(move || -> tvm_ffi::Result<i64> { Ok(*count) });
/// ```
#[derive(Clone, ObjectRef)]
#[object_ref(skip_deref)]
pub struct Function {
    data: ObjectArc<FunctionObj>,
}

// SAFETY: a `Function` is immutable, every Rust callback it wraps is `Send + Sync`,
// and `from_extern_c` requires the same of foreign callbacks.
unsafe impl Send for Function {}
unsafe impl Sync for Function {}

//------------------------------------------------------------------------
// CallbackFunctionObjImpl
//------------------------------------------------------------------------
/// Special helper class to hold a generic callback state as Object
/// Logically this Impl can be viewed as a FunctionObj
/// We can create an ObjectArc<CallbackFunctionObjImpl<F>> so the deleter
/// can correctly delete the entire object including callback part
/// then we will convert to ObjectArc<FunctionObj> to be used as function
#[repr(C)]
struct CallbackFunctionObjImpl<F: Fn(&[AnyView]) -> Result<Any> + Send + Sync + 'static> {
    function: FunctionObj,
    callback: F,
}

impl<F: Fn(&[AnyView]) -> Result<Any> + Send + Sync + 'static> CallbackFunctionObjImpl<F> {
    pub fn from_callback(callback: F) -> Self {
        Self {
            function: FunctionObj {
                object: Object::new(),
                cell: TVMFFIFunctionCell {
                    // specfic callback for F
                    safe_call: Self::invoke_callback,
                    cxx_call: std::ptr::null_mut(),
                },
            },
            callback,
        }
    }

    unsafe extern "C" fn invoke_callback(
        handle: *mut std::ffi::c_void,
        args: *const TVMFFIAny,
        num_args: i32,
        result: *mut TVMFFIAny,
    ) -> i32 {
        let this = unsafe { &*(handle as *mut Self) };
        let packed_args = unsafe { crate::function_internal::packed_args(args, num_args) };
        let ret_value = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            (this.callback)(packed_args)
        })) {
            Ok(ret_value) => ret_value,
            Err(payload) => Err(Error::from_panic(payload)),
        };
        match ret_value {
            Ok(value) => {
                unsafe { *result = Any::into_raw_ffi_any(value) };
                0
            }
            Err(error) => {
                Error::set_raised(&error);
                -1
            }
        }
    }
}

unsafe impl<F: Fn(&[AnyView]) -> Result<Any> + Send + Sync + 'static> ObjectCore
    for CallbackFunctionObjImpl<F>
{
    type Parent = FunctionObj;
    const TYPE_KEY: &'static str = FunctionObj::TYPE_KEY;
    const TYPE_DEPTH: i32 = FunctionObj::TYPE_DEPTH;
    fn type_index() -> i32 {
        FunctionObj::type_index()
    }
    unsafe fn object_header_mut(this: &mut Self) -> &mut tvm_ffi_sys::TVMFFIObject {
        unsafe { FunctionObj::object_header_mut(&mut this.function) }
    }
}

impl Function {
    /// Call the function with the arguments `packed_args`.
    ///
    /// # Arguments
    ///
    /// * `packed_args` - The arguments, one [`AnyView`] each.
    ///
    /// # Returns
    ///
    /// The function's result.
    ///
    /// # Errors
    ///
    /// Returns the error that the function raises.
    ///
    /// ```
    /// use tvm_ffi::{AnyView, Function, Result};
    /// let add = Function::from_typed(|x: i64, y: i64| -> Result<i64> { Ok(x + y) });
    /// let result = add.call_packed(&[AnyView::from(&1_i64), AnyView::from(&2_i64)])?;
    /// assert_eq!(i64::try_from(result)?, 3);
    /// # Ok::<(), tvm_ffi::Error>(())
    /// ```
    ///
    /// See also [`Function`].
    pub fn call_packed(&self, packed_args: &[AnyView]) -> Result<Any> {
        unsafe {
            let packed_args_ptr = packed_args.as_ptr() as *const TVMFFIAny;
            let mut result = Any::new();
            let ret_code = (self.data.cell.safe_call)(
                ObjectArc::as_ptr(&self.data) as *mut FunctionObj as *mut std::ffi::c_void,
                packed_args_ptr,
                packed_args.len() as i32,
                Any::as_raw_ffi_any_mut(&mut result),
            );
            if ret_code == 0 {
                Ok(result)
            } else {
                Err(Error::from_raised())
            }
        }
    }

    /// Call the function with the arguments in the tuple `tuple_args`.
    ///
    /// An element may be a value or a reference to one; a reference passes an
    /// object without retaining it, and a `&str` or `&[u8]` passes as a
    /// `String` or `Bytes`.
    ///
    /// # Arguments
    ///
    /// * `tuple_args` - The arguments, as a tuple such as `(1_i64, &array)`.
    ///
    /// # Returns
    ///
    /// The function's result.
    ///
    /// # Errors
    ///
    /// Returns the error that the function raises.
    ///
    /// See also [`Function`].
    pub fn call_tuple(&self, tuple_args: impl PackedCallArgs) -> Result<Any> {
        tuple_args.with_packed_args(|packed_args| self.call_packed(packed_args))
    }
    /// Look up the global function registered as `name`.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the function, such as `"my_ext.double"`.
    ///
    /// # Returns
    ///
    /// The global function.
    ///
    /// # Errors
    ///
    /// Returns a `ValueError` if no function is registered as `name`, as C++
    /// `Function::GetGlobalRequired`. Probe with `.ok()` where a missing
    /// function is expected.
    ///
    /// ```standalone_crate
    /// use tvm_ffi::Function;
    /// let double = Function::from_typed(|x: i64| -> tvm_ffi::Result<i64> { Ok(x * 2) });
    /// Function::register_global("my_ext.double", double)?;
    /// let double = Function::get_global("my_ext.double")?;
    /// assert_eq!(i64::try_from(double.call_tuple((21_i64,))?)?, 42);
    /// # Ok::<(), tvm_ffi::Error>(())
    /// ```
    ///
    /// See also [`register_global_typed`](Self::register_global_typed).
    pub fn get_global(name: &str) -> Result<Function> {
        unsafe {
            let name_arg = TVMFFIByteArray::from_str(name);
            let mut result: TVMFFIObjectHandle = ::std::ptr::null_mut();
            crate::check_safe_call!(TVMFFIFunctionGetGlobal(&name_arg, &mut result))?;
            if result.is_null() {
                crate::bail!(crate::error::VALUE_ERROR, "Function {} not found", name);
            }
            Ok(Self {
                data: ObjectArc::<FunctionObj>::from_raw(result as *mut FunctionObj),
            })
        }
    }

    /// Register a function as a global function, which every language can then
    /// look up by name.
    ///
    /// The function's metadata records only that it is a function, with no
    /// parameter types. Use [`Function::register_global_typed`] to record the
    /// parameter and return types of a typed function, and a doc string.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the function.
    /// * `func` - The function to register.
    ///
    /// # Returns
    ///
    /// `Ok(())` once the function is registered.
    ///
    /// # Errors
    ///
    /// Returns an error if the registration fails.
    ///
    /// See also [`get_global`](Self::get_global).
    pub fn register_global(name: &str, func: Function) -> Result<()> {
        let type_schema = crate::type_traits::type_schema(FunctionObj::TYPE_KEY, &[]);
        Self::register_global_with_info(name, &func, &type_schema, "")
    }

    /// Register a typed function as a global function, with a doc string.
    ///
    /// The function's metadata records its type schema, the parameter and
    /// return types, which `tvm_ffi.get_global_func_metadata` and the stub
    /// generator read. An empty `doc` records no doc string.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the function.
    /// * `func` - The typed function to register, as for [`Function::from_typed`].
    /// * `doc` - The doc string of the function.
    ///
    /// # Returns
    ///
    /// `Ok(())` once the function is registered.
    ///
    /// # Errors
    ///
    /// Returns an error if the registration fails.
    ///
    /// ```standalone_crate
    /// use tvm_ffi::{Function, Result};
    /// let add = |x: i64, y: i64| -> Result<i64> { Ok(x + y) };
    /// Function::register_global_typed("my_ext.add", add, "Add two integers.")?;
    /// # Ok::<(), tvm_ffi::Error>(())
    /// ```
    ///
    /// See also [`get_global`](Self::get_global).
    pub fn register_global_typed<F, I, O>(name: &str, func: F, doc: &str) -> Result<()>
    where
        F: AsPackedCallable<I, O> + Send + Sync + 'static,
    {
        let type_schema = F::type_schema();
        Self::register_global_with_info(name, &Self::from_typed(func), &type_schema, doc)
    }

    fn register_global_with_info(
        name: &str,
        func: &Function,
        type_schema: &str,
        doc: &str,
    ) -> Result<()> {
        let metadata = crate::type_traits::type_schema_metadata(type_schema);
        let mut method = TVMFFIAny::new();
        unsafe {
            <Function as AnyCompatible>::copy_to_any_view(func, &mut method);
            let info = TVMFFIMethodInfo {
                name: TVMFFIByteArray::from_str(name),
                doc: TVMFFIByteArray::from_str(doc),
                metadata: TVMFFIByteArray::from_str(&metadata),
                flags: 0,
                method,
            };
            let can_override = 0;
            crate::check_safe_call!(TVMFFIFunctionSetGlobalFromMethodInfo(&info, can_override))
        }
    }
    /// Construct a function from a packed function, which takes its arguments as a
    /// slice.
    ///
    /// Use it for a variable number of arguments; otherwise
    /// [`from_typed`](Self::from_typed) converts the arguments for you. Report
    /// errors by returning them. A panic in `func` is caught and raised to the
    /// caller as an `InternalError`, but panicking is discouraged.
    ///
    /// # Arguments
    ///
    /// * `func` - The packed function, of signature `Fn(&[AnyView]) -> Result<Any>`.
    ///
    /// # Returns
    ///
    /// The function.
    ///
    /// ```
    /// use tvm_ffi::{Any, AnyView, Function, Result};
    /// let count = Function::from_packed(|args: &[AnyView]| -> Result<Any> {
    ///     Ok(Any::from(args.len() as i64))
    /// });
    /// assert_eq!(i64::try_from(count.call_tuple((1_i64, true))?)?, 2);
    /// # Ok::<(), tvm_ffi::Error>(())
    /// ```
    ///
    /// See also [`Function`].
    pub fn from_packed<F>(func: F) -> Self
    where
        F: Fn(&[AnyView]) -> Result<Any> + Send + Sync + 'static,
    {
        unsafe {
            let callback_arc = ObjectArc::new(CallbackFunctionObjImpl::from_callback(func));
            let func_arc = ObjectArc::<FunctionObj>::from_raw(
                ObjectArc::into_raw(callback_arc) as *mut FunctionObj
            );
            Self { data: func_arc }
        }
    }

    /// Construct a function from a typed function.
    ///
    /// Each argument is converted to its parameter type, and a mismatch raises a
    /// `TypeError` to the caller. Report errors by returning them. A panic in
    /// `func` is caught and raised to the caller as an `InternalError`, but
    /// panicking is discouraged.
    ///
    /// # Arguments
    ///
    /// * `func` - The typed function, an [`AsPackedCallable`] such as
    ///   `Fn(T0, T1, ...) -> Result<O>`; each parameter is an `AnyCompatible`
    ///   type, a `TensorView` or an `RValueRef<T>`.
    ///
    /// # Returns
    ///
    /// The function.
    ///
    /// ```
    /// use tvm_ffi::{Function, Result};
    /// let scale = Function::from_typed(|x: f64, factor: f64| -> Result<f64> { Ok(x * factor) });
    /// assert_eq!(f64::try_from(scale.call_tuple((2.0_f64, 1.5_f64))?)?, 3.0);
    /// # Ok::<(), tvm_ffi::Error>(())
    /// ```
    ///
    /// See also [`Function`].
    pub fn from_typed<F, I, O>(func: F) -> Self
    where
        F: AsPackedCallable<I, O> + Send + Sync + 'static,
    {
        let closure = move |packed_args: &[AnyView]| -> Result<Any> {
            let ret_value = func.call_packed(packed_args)?;
            Ok(ret_value)
        };
        Self::from_packed(closure)
    }

    /// Wrap a C callback as a Function.
    ///
    /// # Safety
    ///
    /// `handle` (may be null) is the state `safe_call` receives; `deleter` frees it
    /// when the Function drops. All three must be safe to use from any thread.
    ///
    /// # Errors
    ///
    /// Returns an error if the runtime cannot allocate the function. The
    /// function then does not take `handle`: `deleter` is not called, and the
    /// caller still owns `handle`.
    pub unsafe fn from_extern_c(
        handle: *mut std::ffi::c_void,
        safe_call: TVMFFISafeCallType,
        deleter: Option<unsafe extern "C" fn(*mut std::ffi::c_void)>,
    ) -> Result<Self> {
        unsafe {
            let mut out_handle: TVMFFIObjectHandle = std::ptr::null_mut();
            crate::check_safe_call!(TVMFFIFunctionCreate(
                handle,
                safe_call,
                deleter,
                &mut out_handle
            ))?;
            Ok(Self {
                data: ObjectArc::<FunctionObj>::from_raw(out_handle as *mut FunctionObj),
            })
        }
    }
}
