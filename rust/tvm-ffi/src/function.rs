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

use crate::any::{Any, AnyView};
use crate::derive::{Object, ObjectRef};
use crate::error::{Error, Result};
use crate::function_internal::{AsPackedCallable, TupleAsPackedArgs};
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
#[type_key = "ffi.Function"]
#[type_index(TVMFFITypeIndex::kTVMFFIFunction)]
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
/// ```compile_fail
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
        let this = &*(handle as *mut Self);
        let packed_args = crate::function_internal::packed_args(args, num_args);
        let ret_value = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            (this.callback)(packed_args)
        })) {
            Ok(ret_value) => ret_value,
            Err(payload) => Err(crate::function_internal::panic_to_error(payload)),
        };
        match ret_value {
            Ok(value) => {
                *result = Any::into_raw_ffi_any(value);
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
    const TYPE_KEY: &'static str = FunctionObj::TYPE_KEY;
    const TYPE_DEPTH: i32 = FunctionObj::TYPE_DEPTH;
    fn type_index() -> i32 {
        FunctionObj::type_index()
    }
    unsafe fn object_header_mut(this: &mut Self) -> &mut tvm_ffi_sys::TVMFFIObject {
        FunctionObj::object_header_mut(&mut this.function)
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
    /// object without retaining it.
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
    pub fn call_tuple<TupleType>(&self, tuple_args: TupleType) -> Result<Any>
    where
        TupleType: TupleAsPackedArgs,
    {
        // This is a workaround for Rust's requirement that stack allocation size
        // must be known at compile time for generic types.
        // While we know args_len is a constant, Rust doesn't allow us to directly
        // declare [AnyView::new(); args_len] in generic contexts.
        //
        // We use a small vector optimization pattern:
        // 1. First allocate a small stack buffer (stack_args)
        // 2. If args_len exceeds STACK_LEN, allocate a heap buffer (heap_args)
        // 3. Use the appropriate buffer based on size
        //
        // Since args_len is a compile-time constant, the compiler should optimize
        // away the unused branch, making this approach efficient.
        const STACK_LEN: usize = 4;
        let mut stack_args = [AnyView::new(); STACK_LEN];
        let mut heap_args = Vec::<AnyView>::new();
        let args_len = <TupleType as TupleAsPackedArgs>::LEN;
        // get packed arguments
        let packed_args: &mut [AnyView] = if args_len <= STACK_LEN {
            &mut stack_args[..args_len]
        } else {
            heap_args.resize(args_len, AnyView::new());
            &mut heap_args[..args_len]
        };
        tuple_args.fill_any_view(packed_args);
        self.call_packed(packed_args)
    }
    /// Call the function with the `LEN` arguments in `tuple_args`.
    ///
    /// This is [`call_tuple`](Self::call_tuple) for an argument count known at
    /// compile time, without its small-vector overhead.
    ///
    /// # Arguments
    ///
    /// * `tuple_args` - The `LEN` arguments, as a tuple.
    ///
    /// # Returns
    ///
    /// The function's result.
    ///
    /// # Errors
    ///
    /// Returns the error that the function raises.
    pub fn call_tuple_with_len<const LEN: usize, TupleType>(
        &self,
        tuple_args: TupleType,
    ) -> Result<Any>
    where
        TupleType: TupleAsPackedArgs,
    {
        let mut packed_args = [AnyView::new(); LEN];
        tuple_args.fill_any_view(&mut packed_args);
        self.call_packed(&packed_args)
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
    /// Returns an error if no function is registered as `name`. Probe with
    /// `.ok()` where a missing function is expected.
    ///
    /// ```
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
                crate::bail!(crate::error::RUNTIME_ERROR, "Function {} not found", name);
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
    /// ```
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
    /// * `func` - The typed function, of signature `Fn(T0, T1, ...) -> Result<O>`.
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
