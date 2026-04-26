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
/*
 * \file tvm_ffi_python_helpers.h
 * \brief C++ based helpers for the Python FFI call to optimize performance.
 */
#ifndef TVM_FFI_PYTHON_HELPERS_H_
#define TVM_FFI_PYTHON_HELPERS_H_

#include <Python.h>
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/extra/c_env_api.h>

// Define here to avoid dependencies on non-c headers for now
#ifndef TVM_FFI_INLINE
#if defined(_MSC_VER)
#define TVM_FFI_INLINE [[msvc::forceinline]] inline
#else
#define TVM_FFI_INLINE [[gnu::always_inline]] inline
#endif
#endif

#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <vector>

///--------------------------------------------------------------------------------
/// We deliberately designed the data structure and function to be C-style
//  prefixed with TVMFFIPy so they can be easily invoked through Cython.
///--------------------------------------------------------------------------------

//------------------------------------------------------------------------------------
// Helpers for Python thread-state attachment
//------------------------------------------------------------------------------------
//
// On classic builds, PyGILState_Ensure attaches the current thread and acquires the GIL.
// On free-threaded builds, there is no process-wide GIL to acquire, but CPython still
// requires an attached thread state before manipulating Python refcounts.
class TVMFFIPyWithAttachedThreadState {
 public:
  TVMFFIPyWithAttachedThreadState() noexcept { gstate_ = PyGILState_Ensure(); }
  ~TVMFFIPyWithAttachedThreadState() { PyGILState_Release(gstate_); }

 private:
  PyGILState_STATE gstate_;
};

/*!
 * \brief Closure state carried as the resource handle for an FFI function that
 *        wraps a Python callable and optional exchange api for tensor handling.
 *
 * Created by TVMFFIPyConvertPyCallback and released by
 * TVMFFIPyCallbackClosure::Deleter when the FFI function is destroyed.
 */
struct TVMFFIPyCallbackClosure {
  /*! \brief Strong reference to the Python callable. */
  PyObject* callable;
  /*! \brief Optional DLPack exchange API used when constructing Tensor returns. */
  const DLPackExchangeAPI* dlpack_exchange_api;
  /*!
   * \brief Deleter registered with TVMFFIFunctionCreate. Runs on FFI function destroy.
   *
   * Releases the closure's strong Python reference and frees the closure.
   */
  static void Deleter(void* context) noexcept {
    TVMFFIPyWithAttachedThreadState thread_state;
    auto* closure = static_cast<TVMFFIPyCallbackClosure*>(context);
    Py_DecRef(closure->callable);
    delete closure;
  }
};

/*!
 * \brief Thread-local call stack used by TVMFFIPyCallContext.
 */
class TVMFFIPyCallStack {
 public:
  /*! \brief The stack of arguments */
  std::vector<TVMFFIAny> args_stack;
  /*! \brief The top of the argument call stack currently */
  int64_t args_stack_top = 0;
  /*!
   * \brief The stack of extra temporary Python objects that may not fit into
   * one temp per argument budget, mainly used by value protocol.
   */
  std::vector<PyObject*> extra_temp_py_objects_stack;

  /*! \brief Constructor to initialize the call stack */
  TVMFFIPyCallStack() {
    // keep it 4K as default stack size so it is page aligned
    constexpr size_t kDefaultStackSize = 4096;
    // fit everything roughly 4K stack
    args_stack.resize(kDefaultStackSize / sizeof(TVMFFIAny));
    extra_temp_py_objects_stack.reserve(16);
  }
};

//---------------------------------------------------------------------------------------------
// Support for Python -> FFI function calls.
//---------------------------------------------------------------------------------------------
/*!
 * \brief Context for each ffi call to track the stream, device and temporary arguments.
 */
class TVMFFIPyCallContext {
 public:
  /*! \brief The workspace for the packed arguments */
  TVMFFIAny* packed_args = nullptr;
  /*! \brief Detected device type, if any */
  int device_type = -1;
  /*! \brief Detected device id, if any */
  int device_id = 0;
  /*! \brief Detected stream, if any */
  void* stream = nullptr;
  /*! \brief the DLPack exchange API, if any */
  const DLPackExchangeAPI* dlpack_c_exchange_api{nullptr};
  /*! \brief pointer to the call stack space */
  TVMFFIPyCallStack* call_stack = nullptr;
  /*! \brief the temporary arguments to be recycled */
  void** temp_ffi_objects = nullptr;
  /*! \brief the temporary arguments to be recycled */
  void** temp_py_objects = nullptr;
  /*! \brief the number of temporary arguments */
  int num_temp_ffi_objects = 0;
  /*! \brief the number of temporary arguments */
  int num_temp_py_objects = 0;

  /*! \brief RAII guard constructor to create a TVMFFIPyCallContext */
  TVMFFIPyCallContext(TVMFFIPyCallStack* call_stack, int64_t num_args) : call_stack(call_stack) {
    // In most cases, it will try to allocate from temp_stack,
    // then allocate from heap if the request goes beyond the stack size.
    static_assert(sizeof(TVMFFIAny) >= (sizeof(void*) * 2));
    static_assert(alignof(TVMFFIAny) % alignof(void*) == 0);
    old_args_stack_top_ = call_stack->args_stack_top;
    int64_t requested_count = num_args * 2;
    TVMFFIAny* stack_head = call_stack->args_stack.data() + call_stack->args_stack_top;
    if (call_stack->args_stack_top + requested_count >
        static_cast<int64_t>(call_stack->args_stack.size())) {
      // allocate from heap
      heap_ptr_ = new TVMFFIAny[requested_count];
      stack_head = heap_ptr_;
    } else {
      call_stack->args_stack_top += requested_count;
    }
    this->packed_args = stack_head;
    // by default we co-locate the temporary arguments with packed arguments
    // for better cache locality with one temp per argument budget.
    this->temp_ffi_objects = reinterpret_cast<void**>(stack_head + num_args);
    this->temp_py_objects = this->temp_ffi_objects + num_args;
    this->old_extra_temp_py_objects_stack_top_ = call_stack->extra_temp_py_objects_stack.size();
  }

  ~TVMFFIPyCallContext() {
    TVMFFIPyWithAttachedThreadState thread_state;
    try {
      // recycle the temporary arguments if any
      for (int i = 0; i < this->num_temp_ffi_objects; ++i) {
        TVMFFIObjectDecRef(this->temp_ffi_objects[i]);
      }
      for (int i = 0; i < this->num_temp_py_objects; ++i) {
        Py_DecRef(static_cast<PyObject*>(this->temp_py_objects[i]));
      }
      for (size_t i = old_extra_temp_py_objects_stack_top_;
           i < call_stack->extra_temp_py_objects_stack.size(); ++i) {
        Py_DecRef(static_cast<PyObject*>(call_stack->extra_temp_py_objects_stack[i]));
      }
      call_stack->extra_temp_py_objects_stack.resize(old_extra_temp_py_objects_stack_top_);
    } catch (const std::exception& ex) {
      // very rare, catch c++ exception and set python error
      PyErr_SetString(PyExc_RuntimeError, ex.what());
    }
    // now recycle the memory of the call stack
    if (heap_ptr_ == nullptr) {
      call_stack->args_stack_top = old_args_stack_top_;
    } else {
      delete[] heap_ptr_;
    }
  }

 private:
  /*! \brief the heap pointer */
  TVMFFIAny* heap_ptr_ = nullptr;
  /*! \brief the old stack top */
  size_t old_args_stack_top_;
  /*! \brief the begin index of the temporary Python objects stack */
  size_t old_extra_temp_py_objects_stack_top_;
};

/*! \brief Argument setter for a given python argument. */
struct TVMFFIPyArgSetter {
  /*!
   * \brief Function pointer to invoke the setter.
   * \param self Pointer to this, this should be TVMFFIPyArgSetter*
   * \param call_ctx The call context.
   * \param arg The python argument to be set
   * \param out The output argument.
   * \return 0 on success, -1 on failure. PyError should be set if -1 is returned.
   */
  int (*func)(TVMFFIPyArgSetter* self, TVMFFIPyCallContext* call_ctx, PyObject* arg,
              TVMFFIAny* out);
  /*!
   * \brief Optional DLPackExchangeAPI struct pointer.
   * This is the new struct-based approach that bundles all DLPack exchange functions.
   */
  const DLPackExchangeAPI* dlpack_c_exchange_api{nullptr};
  /*!
   * \brief Invoke the setter.
   * \param call_ctx The call context.
   * \param arg The python argument to be set
   * \param out The output argument.
   * \return 0 on success, -1 on failure. PyError should be set if -1 is returned.
   */
  TVM_FFI_INLINE int operator()(TVMFFIPyCallContext* call_ctx, PyObject* arg,
                                TVMFFIAny* out) const {
    return (*func)(const_cast<TVMFFIPyArgSetter*>(this), call_ctx, arg, out);
  }
};

//---------------------------------------------------------------------------------------------
// The following section contains predefined setters for common POD types
// They ar not meant to be used directly, but instead being registered to TVMFFIPyCallManager
//---------------------------------------------------------------------------------------------
int TVMFFIPyArgSetterFloat_(TVMFFIPyArgSetter*, TVMFFIPyCallContext*, PyObject* arg,
                            TVMFFIAny* out) noexcept {
  out->type_index = kTVMFFIFloat;
  // this function getsdispatched when type is already float, so no need to worry about error
  out->v_float64 = PyFloat_AsDouble(arg);
  return 0;
}

int TVMFFIPyArgSetterInt_(TVMFFIPyArgSetter*, TVMFFIPyCallContext*, PyObject* arg,
                          TVMFFIAny* out) noexcept {
  int overflow = 0;
  out->type_index = kTVMFFIInt;
  out->v_int64 = PyLong_AsLongLongAndOverflow(arg, &overflow);

  if (overflow != 0) {
    PyErr_SetString(PyExc_OverflowError, "Python int too large to convert to int64_t");
    return -1;
  }
  return 0;
}

int TVMFFIPyArgSetterBool_(TVMFFIPyArgSetter*, TVMFFIPyCallContext*, PyObject* arg,
                           TVMFFIAny* out) noexcept {
  out->type_index = kTVMFFIBool;
  // this function getsdispatched when type is already bool, so no need to worry about error
  out->v_int64 = PyLong_AsLong(arg);
  return 0;
}

int TVMFFIPyArgSetterNone_(TVMFFIPyArgSetter*, TVMFFIPyCallContext*, PyObject* arg,
                           TVMFFIAny* out) noexcept {
  out->type_index = kTVMFFINone;
  out->v_int64 = 0;
  return 0;
}

//---------------------------------------------------------------------------------------------
// Support for PyCallback function calls.
//---------------------------------------------------------------------------------------------

/*!
 * \brief Context for a C -> Python callback call.
 *
 * Owns a temporary PyObject* array that holds arguments converted from the
 * packed FFI call. Space is first taken from the thread-local args_stack on
 * TVMFFIPyCallStack; if insufficient, we fall back to the heap.
 *
 * Unlike TVMFFIPyCallContext::~TVMFFIPyCallContext, this destructor does NOT
 * attach a thread state — callers are expected to already hold one
 * (e.g. via TVMFFIPyWithAttachedThreadState at the top of the callback).
 *
 * The destructor also decrefs every PyObject* pushed into py_args[0 ..
 * num_active_py_args-1], tracking the pushed count via `num_active_py_args`.
 */
class TVMFFIPyCallbackContext {
 public:
  /*! \brief The temporary PyObject* slots for Python call arguments. */
  PyObject** py_args = nullptr;
  /*! \brief How many slots have a live reference and need decref on cleanup. */
  int32_t num_active_py_args = 0;
  /*! \brief Number of total argument slots allocated. */
  int32_t num_args = 0;

  TVMFFIPyCallbackContext(TVMFFIPyCallStack* call_stack, int32_t num_args)
      : num_args(num_args), call_stack_(call_stack) {
    static_assert(sizeof(TVMFFIAny) % sizeof(PyObject*) == 0);
    // slots needed in the unit of TVMFFIAny
    int64_t slots_needed =
        (static_cast<int64_t>(num_args) * sizeof(PyObject*) + sizeof(TVMFFIAny) - 1) /
        sizeof(TVMFFIAny);
    old_args_stack_top_ = call_stack->args_stack_top;
    if (call_stack->args_stack_top + slots_needed <=
        static_cast<int64_t>(call_stack->args_stack.size())) {
      py_args =
          reinterpret_cast<PyObject**>(call_stack->args_stack.data() + call_stack->args_stack_top);
      call_stack->args_stack_top += slots_needed;
    } else {
      heap_ptr_ = new PyObject*[num_args];
      py_args = heap_ptr_;
    }
  }

  ~TVMFFIPyCallbackContext() {
    // caller must already hold an attached thread state; do NOT re-attach.
    // we ensure that all the pyargs are not null
    for (int32_t i = 0; i < num_active_py_args; ++i) {
      Py_DecRef(py_args[i]);
    }
    if (heap_ptr_ == nullptr) {
      call_stack_->args_stack_top = old_args_stack_top_;
    } else {
      delete[] heap_ptr_;
    }
  }

 private:
  TVMFFIPyCallStack* call_stack_ = nullptr;
  int64_t old_args_stack_top_ = 0;
  PyObject** heap_ptr_ = nullptr;
};

/*!
 * \brief A callback arg setter entry registered to handle efficient callback argument conversion.
 */
struct TVMFFIPyCallbackArgSetter {
  /*!
   * \brief Callback type that converts a borrowed TVMFFIAny (AnyView) to a new-reference PyObject*.
   * \param handle Pointer to the TVMFFIPyCallbackArgSetter (for per-type state).
   * \param dlpack_exchange_api The DLPack exchange API (may be NULL).
   * \param arg The TVMFFIAny value to convert (borrowed; setter must inc if it transfers
   * ownership).
   * \param out Output: a new-reference PyObject*.
   * \return 0 on success, -1 on failure (PyErr set).
   */
  int (*func)(TVMFFIPyCallbackArgSetter* handle, const DLPackExchangeAPI* dlpack_exchange_api,
              const TVMFFIAny* arg, PyObject** out);
};

// common callback arg setters that can be quikcly implemented in C++ and used by cython factory
// note that PyErr is propagated back so we just need to return -1 on failure.
int TVMFFIPyCallbackArgSetterNone_(TVMFFIPyCallbackArgSetter*, const DLPackExchangeAPI*,
                                   const TVMFFIAny*, PyObject** out) noexcept {
  Py_IncRef(Py_None);
  *out = Py_None;
  return 0;
}

int TVMFFIPyCallbackArgSetterBool_(TVMFFIPyCallbackArgSetter*, const DLPackExchangeAPI*,
                                   const TVMFFIAny* arg, PyObject** out) noexcept {
  *out = PyBool_FromLong(static_cast<long>(arg->v_int64));
  return (*out != nullptr) ? 0 : -1;
}

int TVMFFIPyCallbackArgSetterInt_(TVMFFIPyCallbackArgSetter*, const DLPackExchangeAPI*,
                                  const TVMFFIAny* arg, PyObject** out) noexcept {
  *out = PyLong_FromLongLong(arg->v_int64);
  return (*out != nullptr) ? 0 : -1;
}

int TVMFFIPyCallbackArgSetterFloat_(TVMFFIPyCallbackArgSetter*, const DLPackExchangeAPI*,
                                    const TVMFFIAny* arg, PyObject** out) noexcept {
  *out = PyFloat_FromDouble(arg->v_float64);
  return (*out != nullptr) ? 0 : -1;
}

int TVMFFIPyCallbackArgSetterSmallStr_(TVMFFIPyCallbackArgSetter*, const DLPackExchangeAPI*,
                                       const TVMFFIAny* arg, PyObject** out) noexcept {
  TVMFFIByteArray ba = TVMFFISmallBytesGetContentByteArray(arg);
  *out = PyUnicode_DecodeUTF8(ba.data, static_cast<Py_ssize_t>(ba.size), nullptr);
  return (*out != nullptr) ? 0 : -1;
}

int TVMFFIPyCallbackArgSetterSmallBytes_(TVMFFIPyCallbackArgSetter*, const DLPackExchangeAPI*,
                                         const TVMFFIAny* arg, PyObject** out) noexcept {
  TVMFFIByteArray ba = TVMFFISmallBytesGetContentByteArray(arg);
  *out = PyBytes_FromStringAndSize(ba.data, static_cast<Py_ssize_t>(ba.size));
  return (*out != nullptr) ? 0 : -1;
}

///--------------------------------------------------------------------------------
/// Declaring functions defined in Cython to be invoked by the C++ implementation.
/// in all cases PyErr is propagated back so we just need to return -1 on failure.
///--------------------------------------------------------------------------------
/*
 * \brief Set the error raised from Python to the FFI side.
 * \param py_err The Python error to be set.
 * \return 0 on success, -1 on failure. PyError should be set if -1 is returned.
 */
__PYX_EXTERN_C int TVMFFICyErrorSetRaisedFromPyError(PyObject* py_err);
/*
 * \brief Create an argument setter for a given Python argument type.
 * \param arg The Python argument to be set.
 * \param out The output argument setter.
 * \return 0 on success, -1 on failure. PyError should be set if -1 is returned.
 */
__PYX_EXTERN_C int TVMFFICyArgSetterFactory(PyObject* arg, TVMFFIPyArgSetter* out);
/*
 * \brief Create a callback arg setter for a given type index.
 * \param type_index The type index of the argument.
 * \param out The output callback arg setter.
 * \return 0 on success, -1 on failure. PyError should be set if -1 is returned.
 */
__PYX_EXTERN_C int TVMFFICyCallbackArgSetterFactory(int32_t type_index,
                                                    TVMFFIPyCallbackArgSetter* out);
//---------------------------------------------------------------------------------------------
// The function call manager section
//---------------------------------------------------------------------------------------------

/*!
 * \brief A manager class that handles python ffi calls.
 */
class TVMFFIPyCallManager {
 public:
  /*!
   * \brief Get the thread local call manager.
   * \return The thread local call manager.
   */
  static TVMFFIPyCallManager* ThreadLocal() {
    static thread_local TVMFFIPyCallManager inst;
    return &inst;
  }
  /*!
   * \brief Call a function with a variable number of arguments
   * \param func_handle The handle of the function to call
   * \param py_arg_tuple The arguments to the function
   * \param result The result of the function
   * \param c_api_ret_code The return code of the C-call
   * \param release_gil Whether to release the GIL
   * \param optional_out_ctx_dlpack_api The DLPack exchange API to be used for the result
   * \return 0 on when there is no python error, -1 on python error
   * \note When an error happens on FFI side, we should return 0 and set c_api_ret_code
   */
  TVM_FFI_INLINE int FuncCall(void* func_handle, PyObject* py_arg_tuple, TVMFFIAny* result,
                              int* c_api_ret_code, bool release_gil,
                              const DLPackExchangeAPI** optional_out_ctx_dlpack_api) {
    int64_t num_args = PyTuple_Size(py_arg_tuple);
    if (num_args == -1) return -1;
    try {
      // allocate a call stack
      TVMFFIPyCallContext ctx(&call_stack_, num_args);
      // Iterate over the arguments and set them
      for (int64_t i = 0; i < num_args; ++i) {
        PyObject* py_arg = PyTuple_GetItem(py_arg_tuple, i);
        TVMFFIAny* c_arg = ctx.packed_args + i;
        if (SetArgument(&ctx, py_arg, c_arg) != 0) return -1;
      }
      TVMFFIStreamHandle prev_stream = nullptr;
      DLPackManagedTensorAllocator prev_tensor_allocator = nullptr;
      // setup stream context if needed
      if (ctx.device_type != -1) {
        c_api_ret_code[0] =
            TVMFFIEnvSetStream(ctx.device_type, ctx.device_id, ctx.stream, &prev_stream);
        // setting failed, directly return
        if (c_api_ret_code[0] != 0) return 0;
      }
      if (ctx.dlpack_c_exchange_api != nullptr &&
          ctx.dlpack_c_exchange_api->managed_tensor_allocator != nullptr) {
        c_api_ret_code[0] = TVMFFIEnvSetDLPackManagedTensorAllocator(
            ctx.dlpack_c_exchange_api->managed_tensor_allocator, 0, &prev_tensor_allocator);
        if (c_api_ret_code[0] != 0) return 0;
      }
      // call the function
      if (release_gil) {
        // release the GIL
        Py_BEGIN_ALLOW_THREADS;
        c_api_ret_code[0] = TVMFFIFunctionCall(func_handle, ctx.packed_args, num_args, result);
        Py_END_ALLOW_THREADS;
      } else {
        c_api_ret_code[0] = TVMFFIFunctionCall(func_handle, ctx.packed_args, num_args, result);
      }
      // restore the original stream
      if (ctx.device_type != -1 && prev_stream != ctx.stream) {
        // always try recover first, even if error happens
        if (TVMFFIEnvSetStream(ctx.device_type, ctx.device_id, prev_stream, nullptr) != 0) {
          // recover failed, set python error
          PyErr_SetString(PyExc_RuntimeError, "Failed to recover stream");
          return -1;
        }
      }

      if (ctx.dlpack_c_exchange_api != nullptr &&
          prev_tensor_allocator != ctx.dlpack_c_exchange_api->managed_tensor_allocator) {
        // note: we cannot set the error value to c_api_ret_code[0] here because it
        // will be overwritten by the error value from the function call
        if (TVMFFIEnvSetDLPackManagedTensorAllocator(prev_tensor_allocator, 0, nullptr) != 0) {
          PyErr_SetString(PyExc_RuntimeError, "Failed to recover DLPack managed tensor allocator");
          return -1;
        }
        // return error after
        if (c_api_ret_code[0] != 0) return 0;
      }
      if (optional_out_ctx_dlpack_api != nullptr && ctx.dlpack_c_exchange_api != nullptr) {
        *optional_out_ctx_dlpack_api = ctx.dlpack_c_exchange_api;
      }
      return 0;
    } catch (const std::exception& ex) {
      // very rare, catch c++ exception and set python error
      PyErr_SetString(PyExc_RuntimeError, ex.what());
      return -1;
    }
  }

  /*
   * \brief Call a constructor with a variable number of arguments
   *
   * This function is similar to FuncCall, but it will not set the
   * stream and tensor allocator, instead, it will synchronize the TVMFFIPyCallContext
   * with the parent context. This behavior is needed for nested conversion of arguments
   * where detected argument setting needs to be synchronized with final call.
   *
   * This function will also not release  the GIL since constructor call is usually cheap.
   *
   * \param func_handle The handle of the constructor to call
   * \param py_arg_tuple The arguments to the constructor
   * \param result The result of the constructor
   * \param c_api_ret_code The return code of the constructor
   * \param parent_ctx The parent call context to
   * \return 0 on success, -1 on failure
   */
  TVM_FFI_INLINE int ConstructorCall(void* func_handle, PyObject* py_arg_tuple, TVMFFIAny* result,
                                     int* c_api_ret_code, TVMFFIPyCallContext* parent_ctx) {
    int64_t num_args = PyTuple_Size(py_arg_tuple);
    if (num_args == -1) return -1;
    try {
      // allocate a call stack
      TVMFFIPyCallContext ctx(&call_stack_, num_args);
      // Iterate over the arguments and set them
      for (int64_t i = 0; i < num_args; ++i) {
        PyObject* py_arg = PyTuple_GetItem(py_arg_tuple, i);
        TVMFFIAny* c_arg = ctx.packed_args + i;
        if (SetArgument(&ctx, py_arg, c_arg) != 0) return -1;
      }
      c_api_ret_code[0] = TVMFFIFunctionCall(func_handle, ctx.packed_args, num_args, result);
      // propagate the call context to the parent context
      if (parent_ctx != nullptr) {
        // stream and current device information
        if (parent_ctx->device_type == -1) {
          parent_ctx->device_type = ctx.device_type;
          parent_ctx->device_id = ctx.device_id;
          parent_ctx->stream = ctx.stream;
        }
        // DLPack exchange API
        if (parent_ctx->dlpack_c_exchange_api == nullptr) {
          parent_ctx->dlpack_c_exchange_api = ctx.dlpack_c_exchange_api;
        }
      }
      return 0;
    } catch (const std::exception& ex) {
      // very rare, catch c++ exception and set python error
      PyErr_SetString(PyExc_RuntimeError, ex.what());
      return -1;
    }
  }

  TVM_FFI_INLINE int SetField(void* field_setter, int64_t field_flags, void* field_ptr,
                              PyObject* py_arg, int* c_api_ret_code) {
    try {
      TVMFFIPyCallContext ctx(&call_stack_, 1);
      TVMFFIAny* c_arg = ctx.packed_args;
      if (SetArgument(&ctx, py_arg, c_arg) != 0) return -1;
      if (!(field_flags & kTVMFFIFieldFlagBitSetterIsFunctionObj)) {
        auto setter = reinterpret_cast<TVMFFIFieldSetter>(field_setter);
        c_api_ret_code[0] = (*setter)(field_ptr, c_arg);
      } else {
        TVMFFIAny args[2]{};
        args[0].type_index = kTVMFFIOpaquePtr;
        args[0].v_ptr = field_ptr;
        args[1] = *c_arg;
        TVMFFIAny result{};
        result.type_index = kTVMFFINone;
        c_api_ret_code[0] =
            TVMFFIFunctionCall(static_cast<TVMFFIObjectHandle>(field_setter), args, 2, &result);
      }
      return 0;
    } catch (const std::exception& ex) {
      // very rare, catch c++ exception and set python error
      PyErr_SetString(PyExc_RuntimeError, ex.what());
      return -1;
    }
  }

  TVM_FFI_INLINE int PyObjectToFFIAny(PyObject* py_arg, TVMFFIAny* out, int* c_api_ret_code) {
    try {
      TVMFFIPyCallContext ctx(&call_stack_, 1);
      TVMFFIAny* c_arg = ctx.packed_args;
      if (SetArgument(&ctx, py_arg, c_arg) != 0) return -1;
      c_api_ret_code[0] = TVMFFIAnyViewToOwnedAny(c_arg, out);
      return 0;
    } catch (const std::exception& ex) {
      // very rare, catch c++ exception and set python error
      PyErr_SetString(PyExc_RuntimeError, ex.what());
      return -1;
    }
  }

  /*!
   * \brief Set an py_arg to out.
   * \param ctx The call context
   * \param py_arg The python argument to be set
   * \param out The output argument
   * \return 0 on success, -1 on failure
   */
  TVM_FFI_INLINE int SetArgument(TVMFFIPyCallContext* ctx, PyObject* py_arg, TVMFFIAny* out) {
    PyTypeObject* py_type = Py_TYPE(py_arg);
    // pre-zero the output argument, modulo the type index
    out->type_index = kTVMFFINone;
    out->zero_padding = 0;
    out->v_int64 = 0;
    // find the pre-cached setter
    // This class is thread-local, so we don't need to worry about race condition
    auto it = arg_dispatch_map_.find(py_type);
    if (it != arg_dispatch_map_.end()) {
      TVMFFIPyArgSetter setter = it->second;
      // if error happens, propagate it back
      if (setter(ctx, py_arg, out) != 0) return -1;
    } else {
      // no dispatch found, query and create a new one.
      TVMFFIPyArgSetter setter;
      // propagate python error back
      if (TVMFFICyArgSetterFactory(py_arg, &setter) != 0) {
        return -1;
      }
      // update dispatch table
      arg_dispatch_map_.emplace(py_type, setter);
      if (setter(ctx, py_arg, out) != 0) return -1;
    }
    return 0;
  }

  /*!
   * \brief Get the size of the arg dispatch map
   * \return The size of the arg dispatch map
   */
  size_t GetArgDispatchMapSize() const { return arg_dispatch_map_.size(); }

  /*!
   * \brief Convert a borrowed TVMFFIAny (AnyView) into a new-reference PyObject*.
   * \param dlpack_exchange_api The DLPack exchange API (may be NULL).
   * \param arg The borrowed TVMFFIAny to convert.
   * \param py_arg The output PyObject*.
   * \return 0 on success, -1 on failure. PyError should be set if -1 is returned.
   */
  TVM_FFI_INLINE int SetPyCallbackArg(const DLPackExchangeAPI* dlpack_exchange_api,
                                      const TVMFFIAny* arg, PyObject** out) {
    size_t type_index = static_cast<size_t>(arg->type_index);
    // Mirror of SetArgument for the C++ -> Python callback path: each per-type
    // callback arg setter is responsible for its own refcount.
    // hot path: cached hit
    if (type_index < callback_arg_dispatch_table_.size() &&
        callback_arg_dispatch_table_[type_index].func != nullptr) {
      TVMFFIPyCallbackArgSetter* setter = &callback_arg_dispatch_table_[type_index];
      return setter->func(setter, dlpack_exchange_api, arg, out);
    }
    // cold path: grow and populate via factory
    if (type_index >= callback_arg_dispatch_table_.size()) {
      // initialize empty entries with nullptr
      callback_arg_dispatch_table_.resize(type_index + 1, TVMFFIPyCallbackArgSetter{nullptr});
    }
    TVMFFIPyCallbackArgSetter* setter = &callback_arg_dispatch_table_[type_index];
    if (TVMFFICyCallbackArgSetterFactory(static_cast<int32_t>(type_index), setter) != 0) {
      return -1;
    }
    return setter->func(setter, dlpack_exchange_api, arg, out);
  }

  /*!
   * \brief Python Callback function entry
   *
   * \param context The TVMFFIPyCallbackClosure* holding the Python callable
   *                and optional DLPack exchange API.
   * \param packed_args The packed FFI arguments.
   * \param num_args Number of arguments.
   * \param result Output FFI result.
   * \return 0 on success, -1 on error.
   */
  TVM_FFI_INLINE int PyCallback(void* context, const TVMFFIAny* packed_args, int32_t num_args,
                                TVMFFIAny* result) noexcept {
    TVMFFIPyWithAttachedThreadState thread_state;
    auto* closure = static_cast<TVMFFIPyCallbackClosure*>(context);
    // Wrap the body in try/catch so any C++ exception raised by the stack
    // allocator (TVMFFIPyCallbackContext / TVMFFIPyCallContext), dispatch
    // table resize in SetPyCallbackArg, or unordered_map::emplace in
    // SetArgument is converted into a PyErr + FFI error instead of
    // triggering std::terminate via the noexcept contract.
    try {
      TVMFFIPyCallbackContext cb_ctx(&call_stack_, num_args);
      // Step 1: Convert each packed arg (borrowed AnyView) to a PyObject*
      for (int32_t i = 0; i < num_args; ++i) {
        if (SetPyCallbackArg(closure->dlpack_exchange_api, &packed_args[i], &cb_ctx.py_args[i]) !=
            0) {
          ForwardPyErrorToFFI();
          return -1;
        }
        // must set active arguments count to ensure correct recycling
        cb_ctx.num_active_py_args = i + 1;
      }
      // Step 2: Call the Python function via vectorcall. Wrap py_result in
      // a RAII guard so its +1 is released on every exit path, including
      // the C++ exception path (e.g., bad_alloc from ret_ctx construction
      // or SetArgument's emplace).
#if PY_VERSION_HEX >= 0x03090000
      PyObject* py_result_raw = PyObject_Vectorcall(closure->callable, cb_ctx.py_args,
                                                    static_cast<size_t>(num_args), nullptr);
#else
      // backward compatibility for Python 3.8
      PyObject* py_result_raw = _PyObject_Vectorcall(closure->callable, cb_ctx.py_args,
                                                     static_cast<size_t>(num_args), nullptr);
#endif
      struct PyResultGuard {
        PyObject* p;
        ~PyResultGuard() {
          if (p != nullptr) Py_DecRef(p);
        }
      } py_result{py_result_raw};
      if (py_result.p == Py_None) {
        // fast path for Py_None
        result->type_index = kTVMFFINone;
        result->zero_padding = 0;
        result->v_int64 = 0;
        return 0;
      } else if (py_result.p != nullptr) {
        // normal return
        // Use SetArgument on a temporary view slot, then promote to owned.
        // Note: SetArgument only borrows py_result's chandle into `view`; it
        // does NOT inc. py_result must stay alive until AFTER
        // TVMFFIAnyViewToOwnedAny has promoted the view to an owned ref,
        // otherwise dec'ing py_result first could free the underlying object
        // (e.g. if py_result owns the last ref to a freshly created tensor).
        // The guard's destructor runs AFTER the return value is computed.
        TVMFFIPyCallContext ret_ctx(&call_stack_, 1);
        TVMFFIAny* view = ret_ctx.packed_args;
        if (SetArgument(&ret_ctx, py_result.p, view) != 0) {
          ForwardPyErrorToFFI();
          return -1;
        }
        // TLS FFI error set on failure.
        return TVMFFIAnyViewToOwnedAny(view, result);
      } else {
        // vectorcall failed
        ForwardPyErrorToFFI();
        return -1;
      }
    } catch (const std::exception& ex) {
      // very rare, catch c++ exception and set python error
      PyErr_SetString(PyExc_RuntimeError, ex.what());
      ForwardPyErrorToFFI();
      return -1;
    }
  }

  /*!
   * \brief Fetch the current Python exception and forward it to
   *        TVMFFICyErrorSetRaisedFromPyError, then clear the Python error indicator.
   *
   * This helper correctly extracts the exception *value* (not just the type
   * returned by PyErr_Occurred()) so that set_last_ffi_error can access the
   * message and traceback.
   */
  static void ForwardPyErrorToFFI() noexcept {
#if PY_VERSION_HEX >= 0x030C0000
    // Python 3.12+: PyErr_Fetch / PyErr_NormalizeException are deprecated.
    // PyErr_GetRaisedException returns an already-normalized exception
    // instance and clears the indicator. Traceback is attached as usual.
    PyObject* pvalue = PyErr_GetRaisedException();
    if (pvalue != nullptr) {
      TVMFFICyErrorSetRaisedFromPyError(pvalue);
      Py_DecRef(pvalue);
    }
#else
    // Python 3.9 - 3.11.
    PyObject *ptype, *pvalue, *ptraceback;
    PyErr_Fetch(&ptype, &pvalue, &ptraceback);
    PyErr_NormalizeException(&ptype, &pvalue, &ptraceback);
    if (ptraceback != nullptr) {
      PyException_SetTraceback(pvalue, ptraceback);
    }
    TVMFFICyErrorSetRaisedFromPyError(pvalue);
    Py_DecRef(ptype);
    Py_DecRef(pvalue);
    Py_DecRef(ptraceback);
#endif
  }

 private:
  TVMFFIPyCallManager() {
    static constexpr size_t kDefaultDispatchCapacity = 32;
    arg_dispatch_map_.reserve(kDefaultDispatchCapacity);
    // Pre-allocate callback arg dispatch table for static type indices
    static constexpr size_t kDefaultCallbackArgDispatchCapacity = 128;
    callback_arg_dispatch_table_.resize(kDefaultCallbackArgDispatchCapacity);
  }

  // internal arg dispatch map: type -> argument setter
  std::unordered_map<PyTypeObject*, TVMFFIPyArgSetter> arg_dispatch_map_;
  // call stack
  TVMFFIPyCallStack call_stack_;
  // callback arg setter dispatch table indexed by type_index (view-based path
  // used by PyCallback; see TVMFFIPyCallbackArgSetter docs above)
  std::vector<TVMFFIPyCallbackArgSetter> callback_arg_dispatch_table_;
};

/*!
 * \brief Call a function with a variable number of arguments
 * \param func_handle The handle of the function to call
 * \param py_arg_tuple The arguments to the function
 * \param result The result of the function
 * \param c_api_ret_code The return code of the function
 * \param release_gil Whether to release the GIL
 * \param out_ctx_dlpack_api The DLPack exchange API to be used for the result
 * \return 0 on success, nonzero on failure
 */
TVM_FFI_INLINE int TVMFFIPyFuncCall(void* func_handle, PyObject* py_arg_tuple, TVMFFIAny* result,
                                    int* c_api_ret_code, bool release_gil = true,
                                    const DLPackExchangeAPI** out_ctx_dlpack_api = nullptr) {
  return TVMFFIPyCallManager::ThreadLocal()->FuncCall(
      func_handle, py_arg_tuple, result, c_api_ret_code, release_gil, out_ctx_dlpack_api);
}

/*!
 * \brief Call a constructor function with a variable number of arguments
 *
 * This function is similar to TVMFFIPyFuncCall, but it will not set the
 * stream and tensor allocator. Instead, it will synchronize the TVMFFIPyCallContext
 * with the parent context. This behavior is needed for nested conversion of arguments
 * where detected argument settings need to be synchronized with the final call.
 *
 * This function will also not release the GIL since constructor call is usually cheap.
 *
 * \param func_handle The handle of the function to call
 * \param py_arg_tuple The arguments to the constructor
 * \param result The result of the constructor
 * \param c_api_ret_code The return code of the constructor
 * \param parent_ctx The parent call context
 * \param release_gil Whether to release the GIL
 * \param out_dlpack_exporter The DLPack exporter to be used for the result
 * \return 0 on success, nonzero on failure
 */
TVM_FFI_INLINE int TVMFFIPyConstructorCall(void* func_handle, PyObject* py_arg_tuple,
                                           TVMFFIAny* result, int* c_api_ret_code,
                                           TVMFFIPyCallContext* parent_ctx) {
  return TVMFFIPyCallManager::ThreadLocal()->ConstructorCall(func_handle, py_arg_tuple, result,
                                                             c_api_ret_code, parent_ctx);
}

/*!
 * \brief Set a field of a FFI object
 * \param field_setter The field setter (function pointer or FunctionObj handle)
 * \param field_flags The field flags (to dispatch between function pointer and FunctionObj)
 * \param field_ptr The pointer to the field
 * \param py_arg The python argument to be set
 * \param c_api_ret_code The return code of the function
 * \return 0 on success, nonzero on failure
 */
TVM_FFI_INLINE int TVMFFIPyCallFieldSetter(void* field_setter, int64_t field_flags, void* field_ptr,
                                           PyObject* py_arg, int* c_api_ret_code) {
  return TVMFFIPyCallManager::ThreadLocal()->SetField(field_setter, field_flags, field_ptr, py_arg,
                                                      c_api_ret_code);
}

/*!
 * \brief Set an python argument to a FFI Any using the generic dispatcher in call manager
 * \param ctx The call context
 * \param py_arg_tvm_ffi_value The python argument to be set using the __tvm_ffi_value__ protocol
 * \param out The output argument
 * \return 0 on success, nonzero on failure
 */
TVM_FFI_INLINE int TVMFFIPySetArgumentGenericDispatcher(TVMFFIPyCallContext* ctx,
                                                        PyObject* py_arg_tvm_ffi_value,
                                                        TVMFFIAny* out) {
  return TVMFFIPyCallManager::ThreadLocal()->SetArgument(ctx, py_arg_tvm_ffi_value, out);
}

/*!
 * \brief Convert a Python object to a FFI Any
 * \param py_arg The python argument to be set
 * \param out The output argument
 * \param c_api_ret_code The return code of the function
 * \return 0 on success, nonzero on failure
 */
TVM_FFI_INLINE int TVMFFIPyPyObjectToFFIAny(PyObject* py_arg, TVMFFIAny* out, int* c_api_ret_code) {
  return TVMFFIPyCallManager::ThreadLocal()->PyObjectToFFIAny(py_arg, out, c_api_ret_code);
}

/*!
 * \brief Get the size of the arg dispatch map
 * \return The size of the arg dispatch map
 */
TVM_FFI_INLINE size_t TVMFFIPyGetArgDispatchMapSize() {
  return TVMFFIPyCallManager::ThreadLocal()->GetArgDispatchMapSize();
}

//---------------------------------------------------------------------------------------------
// Free function wrapper for the Python callback path.
// Mirrors the pattern of TVMFFIPyFuncCall / TVMFFIPyConstructorCall: a top-level
// TVM_FFI_INLINE free function that forwards to the thread-local manager.
//---------------------------------------------------------------------------------------------

/*!
 * \brief C-callable Python callback entry point (TVMFFISafeCallType shape).
 *
 * Forwards to TVMFFIPyCallManager::ThreadLocal()->PyCallback. Designed to be
 * installed as the safe_call pointer for FFI functions that wrap a Python
 * callable.
 *
 * \note The `context` argument is interpreted as a TVMFFIPyCallbackClosure*
 *       by the manager (see TVMFFIPyConvertPyCallback).
 */
TVM_FFI_INLINE int TVMFFIPyCallback(void* context, const TVMFFIAny* packed_args, int32_t num_args,
                                    TVMFFIAny* result) noexcept {
  return TVMFFIPyCallManager::ThreadLocal()->PyCallback(context, packed_args, num_args, result);
}

/*!
 * \brief Create an FFI function handle from a Python callable + optional DLPack exchange API.
 *
 * Allocates a TVMFFIPyCallbackClosure on the heap, IncRefs the callable, and
 * registers it with the FFI function-creation API using TVMFFIPyCallback as the
 * safe-call entry point and TVMFFIPyCallbackClosure::Deleter as the deleter.
 *
 * Returns the raw FFI return code (TLS FFI error set on failure). The Cython
 * caller uses CHECK_CALL to translate it into a Python exception.
 *
 * \param callable The Python callable to wrap. Must be non-NULL.
 * \param dlpack_api Optional DLPack exchange API. May be NULL.
 * \param out_handle Destination for the new FFI function handle.
 * \return The return code from TVMFFIFunctionCreate (0 on success).
 */
TVM_FFI_INLINE int TVMFFIPyConvertPyCallback(PyObject* callable,
                                             const DLPackExchangeAPI* dlpack_api,
                                             TVMFFIObjectHandle* out_handle) noexcept {
  // Use nothrow new: plain `new` can throw std::bad_alloc, which in this
  // noexcept function would trigger std::terminate. On allocation failure,
  // set PyErr and return -1 so the Cython caller's CHECK_CALL surfaces it.
  auto* raw = new (std::nothrow) TVMFFIPyCallbackClosure{callable, dlpack_api};
  if (raw == nullptr) {
    PyErr_NoMemory();
    return -1;
  }
  // The callable's +1 is owned by the closure; TVMFFIPyCallbackClosure::Deleter
  // is responsible for Py_DecRef on destruction. By wiring the same Deleter as
  // the unique_ptr deleter, the failure path below (unique_ptr unwind) runs
  // the same cleanup as the success path (invoked by the FFI runtime).
  Py_IncRef(callable);
  std::unique_ptr<TVMFFIPyCallbackClosure, void (*)(void*)> closure(
      raw, &TVMFFIPyCallbackClosure::Deleter);
  int rc = TVMFFIFunctionCreate(closure.get(), &TVMFFIPyCallback, &TVMFFIPyCallbackClosure::Deleter,
                                out_handle);
  // On success, transfer ownership to the FFI function; on failure, let
  // unique_ptr unwind via Deleter (decrefs the callable, frees the closure).
  if (rc == 0) closure.release();
  return rc;
}

/*!
 * \brief Push a temporary FFI object to the call context that will be recycled after the call
 * \param ctx The call context
 * \param arg The FFI object to push
 */
TVM_FFI_INLINE void TVMFFIPyPushTempFFIObject(TVMFFIPyCallContext* ctx,
                                              TVMFFIObjectHandle arg) noexcept {
  // invariance: each ArgSetter can have at most one temporary Python object
  // so it ensures that we won't overflow the temporary Python object stack
  ctx->temp_ffi_objects[ctx->num_temp_ffi_objects++] = arg;
}

/*!
 * \brief Push a temporary Python object to the call context that will be recycled after the call
 * \param ctx The call context
 * \param arg The Python object to push
 */
TVM_FFI_INLINE void TVMFFIPyPushTempPyObject(TVMFFIPyCallContext* ctx, PyObject* arg) noexcept {
  // invariance: each ArgSetter can have at most one temporary Python object
  // so it ensures that we won't overflow the temporary Python object stack
  Py_IncRef(arg);
  ctx->temp_py_objects[ctx->num_temp_py_objects++] = arg;
}

/*!
 * \brief Push Extra temporary Python object to the call context that may go beyond one temp per
 *        argument budget, mainly used by value protocol.
 * \param ctx The call context
 * \param arg The Python object to push
 */
TVM_FFI_INLINE void TVMFFIPyPushExtraTempPyObject(TVMFFIPyCallContext* ctx, PyObject* arg) {
  Py_IncRef(arg);
  ctx->call_stack->extra_temp_py_objects_stack.emplace_back(arg);
}

//----------------------------------------------------------
// Helpers for MLIR redirection
//----------------------------------------------------------
/*!
 * \brief Function specialization that leverages MLIR packed safe call definitions.
 *
 * The MLIR execution engine generates functions that correspond to the packed signature.
 * As of now, it is hard to access the raw extern C function pointer of SafeCall
 * directly when we declare the signature in LLVM dialect.
 *
 * Note that in theory, the MLIR execution engine should be able to support
 * some form of "extern C" feature that directly exposes the function pointers
 * of C-compatible functions with an attribute tag. So we keep this feature
 * in the Python helper layer for now in case the MLIR execution engine supports it in the future.
 *
 * This helper enables us to create ffi::Function from the MLIR packed
 * safe call function pointer instead of following the redirection pattern
 * in `TVMFFIPyMLIRPackedSafeCall::Invoke`.
 *
 * \sa TVMFFIPyMLIRPackedSafeCall::Invoke
 */
class TVMFFIPyMLIRPackedSafeCall {
 public:
  TVMFFIPyMLIRPackedSafeCall(void (*mlir_packed_safe_call)(void**), PyObject* keep_alive_object)
      : mlir_packed_safe_call_(mlir_packed_safe_call), keep_alive_object_(keep_alive_object) {
    if (keep_alive_object_) {
      Py_IncRef(keep_alive_object_);
    }
  }

  ~TVMFFIPyMLIRPackedSafeCall() {
    TVMFFIPyWithAttachedThreadState thread_state;
    if (keep_alive_object_) {
      Py_DecRef(keep_alive_object_);
    }
  }

  static int Invoke(void* func, const TVMFFIAny* args, int32_t num_args, TVMFFIAny* rv) {
    TVMFFIPyMLIRPackedSafeCall* self = reinterpret_cast<TVMFFIPyMLIRPackedSafeCall*>(func);
    int ret_code = 0;
    void* handle = nullptr;
    void* mlir_args[] = {&handle, const_cast<TVMFFIAny**>(&args), &num_args, &rv, &ret_code};
    (*self->mlir_packed_safe_call_)(mlir_args);
    return ret_code;
  }

  static void Deleter(void* self) { delete static_cast<TVMFFIPyMLIRPackedSafeCall*>(self); }

 private:
  void (*mlir_packed_safe_call_)(void**);
  PyObject* keep_alive_object_;
};

/*!
 * \brief Create a TVMFFIPyMLIRPackedSafeCall handle
 * \param mlir_packed_safe_call The MLIR packed safe call function
 * \param keep_alive_object The keep alive object
 * \return The TVMFFIPyMLIRPackedSafeCall object
 */
void* TVMFFIPyMLIRPackedSafeCallCreate(void (*mlir_packed_safe_call)(void**),
                                       PyObject* keep_alive_object) {
  return new TVMFFIPyMLIRPackedSafeCall(mlir_packed_safe_call, keep_alive_object);
}

/*!
 * \brief Call the MLIR packed safe call function
 * \param self The TVMFFIPyMLIRPackedSafeCall object
 * \param args The arguments
 * \param num_args The number of arguments
 * \param rv The result
 * \return The return code
 */
int TVMFFIPyMLIRPackedSafeCallInvoke(void* self, const TVMFFIAny* args, int32_t num_args,
                                     TVMFFIAny* rv) {
  return TVMFFIPyMLIRPackedSafeCall::Invoke(self, args, num_args, rv);
}

/*!
 * \brief Delete the TVMFFIPyMLIRPackedSafeCall object
 * \param self The TVMFFIPyMLIRPackedSafeCall object
 */
void TVMFFIPyMLIRPackedSafeCallDeleter(void* self) {
  return TVMFFIPyMLIRPackedSafeCall::Deleter(self);
}

/*!
 * \brief Deleter for Python objects
 * \param py_obj The Python object to delete
 */
extern "C" void TVMFFIPyObjectDeleter(void* py_obj) noexcept {
  TVMFFIPyWithAttachedThreadState thread_state;
  Py_DecRef(static_cast<PyObject*>(py_obj));
}

/*
 * \brief Dummy target to ensure testing is linked and we can run testcases
 */
extern "C" TVM_FFI_DLL int TVMFFITestingDummyTarget();

#endif  // TVM_FFI_PYTHON_HELPERS_H_
