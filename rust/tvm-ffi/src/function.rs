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
use crate::type_traits::AnyCompatible;
use tvm_ffi_sys::{
    TVMFFIAny, TVMFFIByteArray, TVMFFIFunctionCell, TVMFFIFunctionCreate, TVMFFIFunctionGetGlobal,
    TVMFFIFunctionSetGlobalFromMethodInfo, TVMFFIGetTypeInfo, TVMFFIMethodInfo, TVMFFIObjectHandle,
    TVMFFISafeCallType, TVMFFITypeIndex, TVMFFITypeKeyToIndex,
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

unsafe impl<F: Fn(&[AnyView]) -> Result<Any> + 'static> ObjectCore for CallbackFunctionObjImpl<F> {
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
    /// ```
    /// use tvm_ffi::{AnyView, Function, Result};
    /// let add = Function::from_typed(|x: i64, y: i64| -> Result<i64> { Ok(x + y) });
    /// let result = add.call_packed(&[AnyView::from(&1_i64), AnyView::from(&2_i64)])?;
    /// assert_eq!(result.try_as::<i64>(), Some(3));
    /// # Ok::<(), tvm_ffi::Error>(())
    /// ```
    ///
    /// See also [`Function`].
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

    /// Call the function with the arguments in the tuple `tuple_args`; a reference
    /// passes an object without retaining it.
    ///
    /// ```
    /// use tvm_ffi::{Function, Result};
    /// let add = Function::from_typed(|x: i64, y: i64| -> Result<i64> { Ok(x + y) });
    /// let result = add.call_tuple((1_i64, 2_i64))?;
    /// assert_eq!(result.try_as::<i64>(), Some(3));
    /// # Ok::<(), tvm_ffi::Error>(())
    /// ```
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
        (&tuple_args).fill_any_view(packed_args);
        self.call_packed(packed_args)
    }
    /// Call the function with the `LEN` arguments in `tuple_args`, as
    /// [`call_tuple`](Self::call_tuple) does, without its small-vector
    /// overhead.
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
    /// Look up the global function registered as `name`.
    ///
    /// # Errors
    ///
    /// Returns an error if no function is registered as `name`.
    ///
    /// ```
    /// use tvm_ffi::Function;
    /// Function::register_global_typed("my_ext.double", |x: i64| -> tvm_ffi::Result<i64> { Ok(x * 2) }, "")?;
    /// let double = Function::get_global("my_ext.double")?;
    /// assert_eq!(double.call_tuple((21_i64,))?.try_as::<i64>(), Some(42));
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

    /// Look up the reflected method `method_name` of the type `type_index`.
    ///
    /// Reflected methods live in each type's method table, not the global
    /// table; an instance method takes the object as its first argument, and
    /// the constructor is named `__ffi_init__`. `type_index` must be
    /// registered: the C API treats an unregistered index as a fatal error.
    ///
    /// # Errors
    ///
    /// Returns an error if the type has no method `method_name`.
    pub fn from_type_method(type_index: i32, method_name: &str) -> Result<Function> {
        unsafe {
            let type_info = TVMFFIGetTypeInfo(type_index);
            if type_info.is_null() {
                crate::bail!(
                    crate::error::TYPE_ERROR,
                    "Cannot find type info for type_index={}",
                    type_index
                );
            }
            let type_info = &*type_info;
            for i in 0..type_info.num_methods as usize {
                let method_info = &*type_info.methods.add(i);
                if method_info.name.as_str() != method_name {
                    continue;
                }
                if !<Function as AnyCompatible>::check_any_strict(&method_info.method) {
                    crate::bail!(
                        crate::error::TYPE_ERROR,
                        "Method `{}` of type `{}` is not a Function",
                        method_name,
                        type_info.type_key.as_str()
                    );
                }
                // the table entry stores the method as a non-owning AnyView;
                // copy out a strong reference
                return Ok(<Function as AnyCompatible>::copy_from_any_view_after_check(
                    &method_info.method,
                ));
            }
            crate::bail!(
                crate::error::TYPE_ERROR,
                "Cannot find method `{}` of type `{}`",
                method_name,
                type_info.type_key.as_str()
            );
        }
    }

    /// Look up the function-valued type attribute `attr_name` of the type
    /// `type_index`.
    ///
    /// Type attributes are not inherited from base types.
    ///
    /// # Errors
    ///
    /// Returns an error if the type has no such attribute.
    pub fn from_type_attr(type_index: i32, attr_name: &str) -> Result<Function> {
        let value = crate::reflection::get_type_attr(type_index, attr_name).ok_or_else(|| {
            crate::error::Error::new(
                crate::error::TYPE_ERROR,
                &format!(
                    "Cannot find type attribute `{}` for type_index={}",
                    attr_name, type_index
                ),
                "",
            )
        })?;
        Function::try_from(value)
    }

    /// Look up the reflected method `method_name` of the type named
    /// `type_key`, as [`from_type_method`](Self::from_type_method) does.
    ///
    /// # Errors
    ///
    /// Returns an error if the type or the method does not exist.
    pub fn from_type_key_method(type_key: &str, method_name: &str) -> Result<Function> {
        unsafe {
            let type_key_arg = TVMFFIByteArray::from_str(type_key);
            let mut type_index: i32 = 0;
            crate::check_safe_call!(TVMFFITypeKeyToIndex(&type_key_arg, &mut type_index))?;
            Self::from_type_method(type_index, method_name)
        }
    }

    /// Register `func` as the global function `name`, with an untyped schema.
    ///
    /// See also [`register_global_typed`](Self::register_global_typed).
    pub fn register_global(name: &str, func: Function) -> Result<()> {
        let type_schema = crate::type_traits::type_schema(FunctionObj::TYPE_KEY, &[]);
        Self::register_global_with_info(name, &func, &type_schema, "")
    }

    /// Register the typed closure `func` as the global function `name`, which
    /// every language can then look up, with the doc string `doc`.
    ///
    /// The metadata records the parameter and return types, as C++
    /// `refl::GlobalDef().def` does.
    ///
    /// ```
    /// use tvm_ffi::{Function, Result};
    /// Function::register_global_typed("my_ext.add", |x: i64, y: i64| -> Result<i64> { Ok(x + y) }, "Add two integers.")?;
    /// # Ok::<(), tvm_ffi::Error>(())
    /// ```
    ///
    /// See also [`get_global`](Self::get_global).
    pub fn register_global_typed<F, I, O>(name: &str, func: F, doc: &str) -> Result<()>
    where
        F: AsPackedCallable<I, O> + 'static,
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
    /// Make a function from the closure `func`, which takes the arguments as a
    /// slice, for a variable number of arguments.
    ///
    /// ```
    /// use tvm_ffi::{Any, AnyView, Function, Result};
    /// let count = Function::from_packed(|args: &[AnyView]| -> Result<Any> { Ok(Any::from(args.len() as i64)) });
    /// assert_eq!(count.call_tuple((1_i64, true))?.try_as::<i64>(), Some(2));
    /// # Ok::<(), tvm_ffi::Error>(())
    /// ```
    ///
    /// See also [`Function`].
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

    /// Make a function from the typed closure `func`, of the form
    /// `Fn(T0, T1, ...) -> Result<O>`; each argument is converted to its parameter
    /// type.
    ///
    /// ```
    /// use tvm_ffi::{Function, Result};
    /// let scale = Function::from_typed(|x: f64, factor: f64| -> Result<f64> { Ok(x * factor) });
    /// assert_eq!(scale.call_tuple((2.0_f64, 1.5_f64))?.try_as::<f64>(), Some(3.0));
    /// # Ok::<(), tvm_ffi::Error>(())
    /// ```
    ///
    /// See also [`Function`].
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

    /// # Safety
    ///
    /// `handle` must be a valid pointer (or null) that is compatible with
    /// `safe_call` and `deleter`. The caller must ensure the handle outlives
    /// the returned `Function` (or that `deleter` properly frees it).
    pub unsafe fn from_extern_c(
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
