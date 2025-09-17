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
use crate::any::{Any, AnyView};
use crate::derive::{Object, ObjectRef};
use crate::error::{Error, Result};
use crate::function_internal::{AsPackedCallable, TupleAsPackedArgs};
use crate::object::{Object, ObjectArc, ObjectCore};
use tvm_ffi_sys::{
    TVMFFIAny, TVMFFIByteArray, TVMFFIFunctionCell, TVMFFIFunctionCreate, TVMFFIFunctionGetGlobal,
    TVMFFIFunctionSetGlobal, TVMFFIObjectHandle, TVMFFISafeCallType, TVMFFITypeIndex,
};

/// function object
#[repr(C)]
#[derive(Object)]
#[type_key = "ffi.Function"]
#[type_index(TVMFFITypeIndex::kTVMFFIFunction)]
pub struct FunctionObj {
    object: Object,
    cell: TVMFFIFunctionCell,
}

/// Error reference class
#[derive(Clone, ObjectRef)]
pub struct Function {
    data: ObjectArc<FunctionObj>,
}

//------------------------------------------------------------------------
// CallbackFunctionObjImpl
//------------------------------------------------------------------------
/// Special helper class to hold a generic callback state as Object
/// Logically this Impl can be viewed as a FunctionObj
/// We can create an ObjectArc<CallbackFunctionObjImpl<F>> so the deleter
/// can correctly delete the entire object including callback part
/// then we will convert to ObjectArc<FunctionObj> to be used as function
#[repr(C)]
struct CallbackFunctionObjImpl<F: Fn(&[AnyView]) -> Result<Any> + 'static> {
    function: FunctionObj,
    callback: F,
}

impl<F: Fn(&[AnyView]) -> Result<Any> + 'static> CallbackFunctionObjImpl<F> {
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
        let packed_args = std::slice::from_raw_parts(args as *const AnyView, num_args as usize);
        let ret_value = (this.callback)(packed_args);
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

unsafe impl<F: Fn(&[AnyView]) -> Result<Any> + 'static> ObjectCore for CallbackFunctionObjImpl<F> {
    const TYPE_KEY: &'static str = FunctionObj::TYPE_KEY;
    fn type_index() -> i32 {
        FunctionObj::type_index()
    }
    unsafe fn object_header_mut(this: &mut Self) -> &mut tvm_ffi_sys::TVMFFIObject {
        FunctionObj::object_header_mut(&mut this.function)
    }
}

impl Function {
    /// Call the function in packed format.
    pub fn call_packed(&self, packed_args: &[AnyView]) -> Result<Any> {
        unsafe {
            let packed_args_ptr = packed_args.as_ptr() as *const TVMFFIAny;
            let mut result = Any::new();
            let ret_code = (self.data.cell.safe_call)(
                ObjectArc::as_raw(&self.data) as *mut FunctionObj as *mut std::ffi::c_void,
                packed_args_ptr,
                packed_args.len() as i32,
                Any::as_data_ptr(&mut result),
            );
            if ret_code == 0 {
                Ok(result)
            } else {
                Err(Error::from_raised())
            }
        }
    }

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
        (&tuple_args).fill_any_view(packed_args);
        self.call_packed(packed_args)
    }
    /// Call function with compile-time known argument count
    /// This is an optimized version of call_tuple for when the argument count
    /// is known at compile time, avoiding the small vector optimization overhead.
    ///
    /// # Arguments
    /// * `tuple_args` - The tuple arguments
    ///
    /// # Returns
    /// * `Any` - The result
    pub fn call_tuple_with_len<const LEN: usize, TupleType>(
        &self,
        tuple_args: TupleType,
    ) -> Result<Any>
    where
        TupleType: TupleAsPackedArgs,
    {
        let mut packed_args = [AnyView::new(); LEN];
        (&tuple_args).fill_any_view(&mut packed_args);
        self.call_packed(&packed_args)
    }
    /// Get global function by name
    /// This function will throw an error if the function is not found.
    ///
    /// # Arguments
    /// * `name` - The name of the function
    ///
    /// # Returns
    /// * `Function` - The global function
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

    /// Register a function as a global function
    /// # Arguments
    /// * `name` - The name of the function
    /// * `func` - The function to register
    ///
    /// # Returns
    /// * `Result<()>` - The result of the registration
    pub fn register_global(name: &str, func: Function) -> Result<()> {
        unsafe {
            let name_arg = TVMFFIByteArray::from_str(name);
            let can_override = 0;
            crate::check_safe_call!(TVMFFIFunctionSetGlobal(
                &name_arg,
                ObjectArc::as_raw(&func.data) as *mut FunctionObj as TVMFFIObjectHandle,
                can_override
            ))?;
            Ok(())
        }
    }
    /// Construct a function from a packed function
    /// # Arguments
    /// * `func` - The packed function in signature of `Fn(&[AnyView]) -> Result<Any>`
    ///
    /// # Returns
    /// * `Function` - The function
    pub fn from_packed<F>(func: F) -> Self
    where
        F: Fn(&[AnyView]) -> Result<Any> + 'static,
    {
        unsafe {
            let callback_arc = ObjectArc::new(CallbackFunctionObjImpl::from_callback(func));
            let func_arc = ObjectArc::<FunctionObj>::from_raw(
                ObjectArc::into_raw(callback_arc) as *mut FunctionObj
            );
            Self { data: func_arc }
        }
    }

    /// Construct a function from a typed function
    /// # Arguments
    /// * `func` - The typed function with function signature of `F(T0, T1, ...) -> Result<O>`
    ///
    /// # Returns
    /// * `Function` - The function
    pub fn from_typed<F, I, O>(func: F) -> Self
    where
        F: AsPackedCallable<I, O> + 'static,
    {
        let closure = move |packed_args: &[AnyView]| -> Result<Any> {
            let ret_value = func.call_packed(packed_args)?;
            Ok(ret_value)
        };
        Self::from_packed(closure)
    }

    pub fn from_extern_c(
        handle: *mut std::ffi::c_void,
        safe_call: TVMFFISafeCallType,
        deleter: Option<unsafe extern "C" fn(*mut std::ffi::c_void)>,
    ) -> Self {
        unsafe {
            let mut out_handle: TVMFFIObjectHandle = std::ptr::null_mut();
            crate::check_safe_call!(TVMFFIFunctionCreate(
                handle,
                safe_call,
                deleter,
                &mut out_handle
            ))
            .unwrap();
            Self {
                data: ObjectArc::<FunctionObj>::from_raw(out_handle as *mut FunctionObj),
            }
        }
    }
}
