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
#include <tvm/ffi/any.h>
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/object.h>

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

//------------------------------------------------------------------------------------
// Helpers for direct reflected field reads
//------------------------------------------------------------------------------------
TVM_FFI_INLINE const char* TVMFFIPyFieldAddr(const void* chandle, int64_t offset) noexcept {
  return static_cast<const char*>(chandle) + offset;
}

TVM_FFI_INLINE int64_t TVMFFIPyFieldGetInt64(void* chandle, int64_t offset) noexcept {
  return *reinterpret_cast<const int64_t*>(TVMFFIPyFieldAddr(chandle, offset));
}

TVM_FFI_INLINE int64_t TVMFFIPyFieldGetInt32(void* chandle, int64_t offset) noexcept {
  return *reinterpret_cast<const int32_t*>(TVMFFIPyFieldAddr(chandle, offset));
}

TVM_FFI_INLINE int64_t TVMFFIPyFieldGetInt16(void* chandle, int64_t offset) noexcept {
  return *reinterpret_cast<const int16_t*>(TVMFFIPyFieldAddr(chandle, offset));
}

TVM_FFI_INLINE int64_t TVMFFIPyFieldGetInt8(void* chandle, int64_t offset) noexcept {
  return *reinterpret_cast<const int8_t*>(TVMFFIPyFieldAddr(chandle, offset));
}

TVM_FFI_INLINE double TVMFFIPyFieldGetFloat64(void* chandle, int64_t offset) noexcept {
  return *reinterpret_cast<const double*>(TVMFFIPyFieldAddr(chandle, offset));
}

TVM_FFI_INLINE double TVMFFIPyFieldGetFloat32(void* chandle, int64_t offset) noexcept {
  return *reinterpret_cast<const float*>(TVMFFIPyFieldAddr(chandle, offset));
}

TVM_FFI_INLINE int64_t TVMFFIPyFieldGetBool(void* chandle, int64_t offset) noexcept {
  return *reinterpret_cast<const bool*>(TVMFFIPyFieldAddr(chandle, offset));
}

TVM_FFI_INLINE void TVMFFIPyFieldGetObjectRef(void* chandle, int64_t offset,
                                              TVMFFIAny* out) noexcept {
  const auto* field =
      reinterpret_cast<const tvm::ffi::ObjectRef*>(TVMFFIPyFieldAddr(chandle, offset));
  if (!field->defined()) {
    out->type_index = tvm::ffi::TypeIndex::kTVMFFINone;
    out->zero_padding = 0;
    out->v_int64 = 0;
    return;
  }
  out->type_index = field->type_index();
  out->zero_padding = 0;
  out->v_obj = tvm::ffi::details::ObjectUnsafe::TVMFFIObjectPtrFromObjectRef(*field);
  tvm::ffi::details::ObjectUnsafe::IncRefObjectHandle(out->v_obj);
}

TVM_FFI_INLINE void TVMFFIPyFieldGetAny(void* chandle, int64_t offset, TVMFFIAny* out) noexcept {
  std::memcpy(out, TVMFFIPyFieldAddr(chandle, offset), sizeof(TVMFFIAny));
  if (out->type_index >= tvm::ffi::TypeIndex::kTVMFFIStaticObjectBegin && out->v_obj != nullptr) {
    tvm::ffi::details::ObjectUnsafe::IncRefObjectHandle(out->v_obj);
  }
}

//------------------------------------------------------------------------------------
// Fast field lookup via custom tp_getattro
//
// Instead of going through Python's property mechanism (LOAD_ATTR_PROPERTY ->
// fget frame -> Cython getter -> C read), we override tp_getattro on classes
// with reflected fields.  The custom tp_getattro does a compact field-table
// scan with interned-string pointer comparison and reads the field inline in C.
//
// For POD types (int, float, bool) the read is fully inlined.
// For complex types (ObjectRef, Any) we call back into Cython's make_ret_object
// / make_ret via registered function pointers.
//------------------------------------------------------------------------------------

#define TVM_FFI_PY_MAX_FIELD_TABLE_ENTRIES 64

/*! \brief One entry in the per-type field table. */
struct TVMFFIPyFieldTableEntry {
  PyObject* name;     /*!< Interned attribute name (pointer comparison) */
  int64_t offset;     /*!< Byte offset from chandle */
  int32_t type_index; /*!< kTVMFFIInt, kTVMFFIFloat, kTVMFFIBool, kTVMFFIAny, or
                         >=kTVMFFIStaticObjectBegin */
  int32_t field_size; /*!< sizeof(field) in bytes */
};

/*! \brief Function pointer type for tp_getattro. */
typedef PyObject* (*TVMFFIPyGetAttroFunc)(PyObject*, PyObject*);

/*! \brief Per-type field table stored in the global map. */
struct TVMFFIPyFieldTable {
  int32_t count;               /*!< Number of entries */
  Py_ssize_t chandle_pyoffset; /*!< Byte offset of CObject.chandle in the Python struct */
  TVMFFIPyGetAttroFunc
      fallback; /*!< Original tp_getattro (e.g. slot_tp_getattr_hook for types with __getattr__) */
  TVMFFIPyFieldTableEntry entries[TVM_FFI_PY_MAX_FIELD_TABLE_ENTRIES];
};

/*!
 * \brief Callback type for converting TVMFFIAny -> Python object.
 * \param result  Pointer to a filled TVMFFIAny (caller owns the ref inside).
 * \return New reference to a Python object, or NULL on error (exception set).
 */
typedef PyObject* (*TVMFFIPyMakeRetCb)(const TVMFFIAny* result);

// Module-level state — set once from Cython at init.
static PyObject* _TVMFFIPyFieldTableKey = nullptr;
static TVMFFIPyMakeRetCb _TVMFFIPyMakeRetObjectCb = nullptr;
static TVMFFIPyMakeRetCb _TVMFFIPyMakeRetCb = nullptr;

/*!
 * \brief Initialize module-level state for the field-table fast path.
 * \param key            Interned dict key (e.g. "_ffi_field_table_ptr").
 * \param make_ret_obj   Callback: TVMFFIAny* -> Python object (for ObjectRef fields).
 * \param make_ret       Callback: TVMFFIAny* -> Python object (for Any fields).
 */
TVM_FFI_INLINE void TVMFFIPyFieldTableSetup(void* key, TVMFFIPyMakeRetCb make_ret_obj,
                                            TVMFFIPyMakeRetCb make_ret) {
  _TVMFFIPyFieldTableKey = static_cast<PyObject*>(key);
  _TVMFFIPyMakeRetObjectCb = make_ret_obj;
  _TVMFFIPyMakeRetCb = make_ret;
}

/*!
 * \brief Read a POD field (int, float, bool) directly from C++ object memory.
 * \return New Python reference, or nullptr if the type is not POD.
 */
TVM_FFI_INLINE PyObject* TVMFFIPyFieldTableReadPOD(const void* chandle,
                                                   const TVMFFIPyFieldTableEntry* entry) noexcept {
  const char* ptr = static_cast<const char*>(chandle) + entry->offset;
  if (entry->type_index == tvm::ffi::TypeIndex::kTVMFFIInt) {
    switch (entry->field_size) {
      case 8:
        return PyLong_FromLongLong(*reinterpret_cast<const int64_t*>(ptr));
      case 4:
        return PyLong_FromLong(*reinterpret_cast<const int32_t*>(ptr));
      case 2:
        return PyLong_FromLong(*reinterpret_cast<const int16_t*>(ptr));
      case 1:
        return PyLong_FromLong(*reinterpret_cast<const int8_t*>(ptr));
    }
  } else if (entry->type_index == tvm::ffi::TypeIndex::kTVMFFIFloat) {
    switch (entry->field_size) {
      case 8:
        return PyFloat_FromDouble(*reinterpret_cast<const double*>(ptr));
      case 4:
        return PyFloat_FromDouble(static_cast<double>(*reinterpret_cast<const float*>(ptr)));
    }
  } else if (entry->type_index == tvm::ffi::TypeIndex::kTVMFFIBool) {
    if (*reinterpret_cast<const uint8_t*>(ptr)) {
      Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
  }
  return nullptr;
}

// ---- Global field-table map (avoids tp_dict access) ----
// Open-addressing hash table: PyTypeObject* → TVMFFIPyFieldTable*.
// Power-of-2 size for fast modular arithmetic.  Probe limit caps
// worst-case scan.  Typical TVM programs register <200 types.
#define TVM_FFI_FT_MAP_SIZE 1024
#define TVM_FFI_FT_MAP_MASK (TVM_FFI_FT_MAP_SIZE - 1)
#define TVM_FFI_FT_PROBE_LIMIT 64

static struct {
  const void* key;
  TVMFFIPyFieldTable* table;
} _TVMFFIPyFieldTableMap[TVM_FFI_FT_MAP_SIZE] = {};

TVM_FFI_INLINE TVMFFIPyFieldTable* TVMFFIPyFieldTableLookup(const void* type_ptr) noexcept {
  size_t idx = (reinterpret_cast<uintptr_t>(type_ptr) >> 4) & TVM_FFI_FT_MAP_MASK;
  for (int p = 0; p < TVM_FFI_FT_PROBE_LIMIT; p++) {
    if (_TVMFFIPyFieldTableMap[idx].key == type_ptr) return _TVMFFIPyFieldTableMap[idx].table;
    if (_TVMFFIPyFieldTableMap[idx].key == nullptr) return nullptr;
    idx = (idx + 1) & TVM_FFI_FT_MAP_MASK;
  }
  return nullptr;
}

TVM_FFI_INLINE bool TVMFFIPyFieldTableInsert(const void* type_ptr,
                                             TVMFFIPyFieldTable* table) noexcept {
  size_t idx = (reinterpret_cast<uintptr_t>(type_ptr) >> 4) & TVM_FFI_FT_MAP_MASK;
  for (int p = 0; p < TVM_FFI_FT_PROBE_LIMIT; p++) {
    if (_TVMFFIPyFieldTableMap[idx].key == nullptr || _TVMFFIPyFieldTableMap[idx].key == type_ptr) {
      _TVMFFIPyFieldTableMap[idx].key = type_ptr;
      _TVMFFIPyFieldTableMap[idx].table = table;
      return true;
    }
    idx = (idx + 1) & TVM_FFI_FT_MAP_MASK;
  }
  // Probe limit exceeded — class will fall back to property path.
  return false;
}

/*!
 * \brief Custom tp_getattro: fast field lookup for reflected fields.
 *
 * Looks up the per-type field table from a global hash map (keyed by
 * PyTypeObject*).  Scans entries with interned-string pointer comparison.
 * POD fields are read inline; complex fields (ObjectRef, Any) call back
 * into Cython.  Non-field attributes fall through to PyObject_GenericGetAttr.
 */
/*
 * NOTE on descriptor override semantics (bugs 2-4 in the Codex review):
 *
 * The field table is a snapshot taken at class-registration time.  Mutating
 * the class dict afterwards (A.x = ..., del A.x) does NOT update the table.
 * This is by design: TVM FFI field descriptors are derived from C++ reflection
 * metadata and are not meant to be replaced or deleted at runtime.
 *
 * NOTE on __getattribute__ interaction (bug 7):
 *
 * If a class defines __getattribute__, CPython may reset tp_getattro to
 * slot_tp_getattr_hook, removing our fast path.  This is expected CPython
 * behavior; the class falls back to the property-descriptor path.
 *
 * NOTE on ABI fragility (bug 10):
 *
 * The tp_getattro slot offset is discovered at init time by scanning the
 * `object` type for PyObject_GenericGetAttr.  On alternative runtimes
 * (PyPy, GraalPython) this scan may fail, leaving the offset at 0 —
 * the fast path is simply never installed and the property path is used.
 *
 * NOTE on teardown (bug 11):
 *
 * Field tables and their interned name references are module-lifetime
 * objects, intentionally never freed.  They live until process exit.
 */
static PyObject* TVMFFIPyFieldGetAttro(PyObject* obj, PyObject* name) {
  TVMFFIPyFieldTable* table = TVMFFIPyFieldTableLookup(static_cast<const void*>(Py_TYPE(obj)));
  if (table != nullptr) {
    void* chandle =
        *reinterpret_cast<void**>(reinterpret_cast<char*>(obj) + table->chandle_pyoffset);
    if (chandle != nullptr) {
      for (int32_t i = 0; i < table->count; i++) {
        if (table->entries[i].name == name) {
          // --- POD fast path (int, float, bool) ---
          PyObject* result = TVMFFIPyFieldTableReadPOD(chandle, &table->entries[i]);
          if (result != nullptr) return result;
          // --- ObjectRef ---
          if (table->entries[i].type_index >=
                  static_cast<int32_t>(tvm::ffi::TypeIndex::kTVMFFIStaticObjectBegin) &&
              _TVMFFIPyMakeRetObjectCb != nullptr) {
            TVMFFIAny any_result;
            TVMFFIPyFieldGetObjectRef(chandle, table->entries[i].offset, &any_result);
            if (any_result.type_index == static_cast<int32_t>(tvm::ffi::TypeIndex::kTVMFFINone)) {
              Py_RETURN_NONE;
            }
            return _TVMFFIPyMakeRetObjectCb(&any_result);
          }
          // --- Any ---
          if (table->entries[i].type_index ==
                  static_cast<int32_t>(tvm::ffi::TypeIndex::kTVMFFIAny) &&
              _TVMFFIPyMakeRetCb != nullptr) {
            TVMFFIAny any_result;
            TVMFFIPyFieldGetAny(chandle, table->entries[i].offset, &any_result);
            return _TVMFFIPyMakeRetCb(&any_result);
          }
          // Unknown field type — fall through to GenericGetAttr
          break;
        }
      }
    } else {
      // chandle is NULL — object was created via __new__() without __init__().
      // Check if the requested name is a registered field and raise a clear error
      // instead of letting the fallback path dereference NULL + offset (segfault).
      for (int32_t i = 0; i < table->count; i++) {
        if (table->entries[i].name == name) {
          PyErr_Format(PyExc_AttributeError,
                       "cannot access field '%U' on an uninitialized object "
                       "(internal handle is NULL). Use the class constructor "
                       "instead of __new__()",
                       name);
          return nullptr;
        }
      }
    }
  }
  // Fallback: call the original tp_getattro (which may handle __getattr__).
  if (table != nullptr && table->fallback != nullptr) {
    return table->fallback(obj, name);
  }
  return PyObject_GenericGetAttr(obj, name);
}

/*!
 * \brief Notify CPython that a type object has been modified.
 * Wraps PyType_Modified without requiring the full PyTypeObject definition.
 */
TVM_FFI_INLINE void TVMFFIPyTypeModified(void* type_ptr) noexcept {
  PyType_Modified(reinterpret_cast<PyTypeObject*>(type_ptr));
}

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
// The following section contains the dispatcher logic for function calling
//---------------------------------------------------------------------------------------------
/*!
 * \brief Factory function that creates an argument setter for a given Python argument type.
 *
 * This factory function analyzes a Python argument and creates an appropriate setter
 * that can convert Python objects of the same type to C arguments for TVM FFI calls.
 * The setter will be cached for future use for setting argument of the same type.
 *
 * \param arg The Python argument value used as a type example.
 * \param out Output parameter that receives the created argument setter.
 * \return 0 on success, -1 on failure. PyError should be set if -1 is returned.
 *
 * \note This is a callback function supplied by the caller. The factory must satisfy
 *       the invariance that the same setter can be used for other arguments with
 *       the same type as the provided example argument.
 */
typedef int (*TVMFFIPyArgSetterFactory)(PyObject* arg, TVMFFIPyArgSetter* out);

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
   * \param setter_factory The factory function to create the setter
   * \param func_handle The handle of the function to call
   * \param py_arg_tuple The arguments to the function
   * \param result The result of the function
   * \param c_api_ret_code The return code of the C-call
   * \param release_gil Whether to release the GIL
   * \param optional_out_ctx_dlpack_api The DLPack exchange API to be used for the result
   * \return 0 on when there is no python error, -1 on python error
   * \note When an error happens on FFI side, we should return 0 and set c_api_ret_code
   */
  TVM_FFI_INLINE int FuncCall(TVMFFIPyArgSetterFactory setter_factory, void* func_handle,
                              PyObject* py_arg_tuple, TVMFFIAny* result, int* c_api_ret_code,
                              bool release_gil,
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
        if (SetArgument(setter_factory, &ctx, py_arg, c_arg) != 0) return -1;
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
   * \param setter_factory The factory function to create the setter
   * \param func_handle The handle of the constructor to call
   * \param py_arg_tuple The arguments to the constructor
   * \param result The result of the constructor
   * \param c_api_ret_code The return code of the constructor
   * \param parent_ctx The parent call context to
   * \return 0 on success, -1 on failure
   */
  TVM_FFI_INLINE int ConstructorCall(TVMFFIPyArgSetterFactory setter_factory, void* func_handle,
                                     PyObject* py_arg_tuple, TVMFFIAny* result, int* c_api_ret_code,
                                     TVMFFIPyCallContext* parent_ctx) {
    int64_t num_args = PyTuple_Size(py_arg_tuple);
    if (num_args == -1) return -1;
    try {
      // allocate a call stack
      TVMFFIPyCallContext ctx(&call_stack_, num_args);
      // Iterate over the arguments and set them
      for (int64_t i = 0; i < num_args; ++i) {
        PyObject* py_arg = PyTuple_GetItem(py_arg_tuple, i);
        TVMFFIAny* c_arg = ctx.packed_args + i;
        if (SetArgument(setter_factory, &ctx, py_arg, c_arg) != 0) return -1;
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

  TVM_FFI_INLINE int SetField(TVMFFIPyArgSetterFactory setter_factory, void* field_setter,
                              int64_t field_flags, void* field_ptr, PyObject* py_arg,
                              int* c_api_ret_code) {
    try {
      TVMFFIPyCallContext ctx(&call_stack_, 1);
      TVMFFIAny* c_arg = ctx.packed_args;
      if (SetArgument(setter_factory, &ctx, py_arg, c_arg) != 0) return -1;
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

  TVM_FFI_INLINE int PyObjectToFFIAny(TVMFFIPyArgSetterFactory setter_factory, PyObject* py_arg,
                                      TVMFFIAny* out, int* c_api_ret_code) {
    try {
      TVMFFIPyCallContext ctx(&call_stack_, 1);
      TVMFFIAny* c_arg = ctx.packed_args;
      if (SetArgument(setter_factory, &ctx, py_arg, c_arg) != 0) return -1;
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
   * \param setter_factory The factory function to create the setter
   * \param ctx The call context
   * \param py_arg The python argument to be set
   * \param out The output argument
   * \return 0 on success, -1 on failure
   */
  TVM_FFI_INLINE int SetArgument(TVMFFIPyArgSetterFactory setter_factory, TVMFFIPyCallContext* ctx,
                                 PyObject* py_arg, TVMFFIAny* out) {
    PyTypeObject* py_type = Py_TYPE(py_arg);
    // pre-zero the output argument, modulo the type index
    out->type_index = kTVMFFINone;
    out->zero_padding = 0;
    out->v_int64 = 0;
    // find the pre-cached setter
    // This class is thread-local, so we don't need to worry about race condition
    auto it = dispatch_map_.find(py_type);
    if (it != dispatch_map_.end()) {
      TVMFFIPyArgSetter setter = it->second;
      // if error happens, propagate it back
      if (setter(ctx, py_arg, out) != 0) return -1;
    } else {
      // no dispatch found, query and create a new one.
      TVMFFIPyArgSetter setter;
      // propagate python error back
      if (setter_factory(py_arg, &setter) != 0) {
        return -1;
      }
      // update dispatch table
      dispatch_map_.emplace(py_type, setter);
      if (setter(ctx, py_arg, out) != 0) return -1;
    }
    return 0;
  }

  /*!
   * \brief Get the size of the dispatch map
   * \return The size of the dispatch map
   */
  size_t GetDispatchMapSize() const { return dispatch_map_.size(); }

 private:
  TVMFFIPyCallManager() {
    static constexpr size_t kDefaultDispatchCapacity = 32;
    dispatch_map_.reserve(kDefaultDispatchCapacity);
  }

  // internal dispacher
  std::unordered_map<PyTypeObject*, TVMFFIPyArgSetter> dispatch_map_;
  // call stack
  TVMFFIPyCallStack call_stack_;
};

/*!
 * \brief Call a function with a variable number of arguments
 * \param setter_factory The factory function to create the setter
 * \param func_handle The handle of the function to call
 * \param py_arg_tuple The arguments to the function
 * \param result The result of the function
 * \param c_api_ret_code The return code of the function
 * \param release_gil Whether to release the GIL
 * \param out_ctx_dlpack_api The DLPack exchange API to be used for the result
 * \return 0 on success, nonzero on failure
 */
TVM_FFI_INLINE int TVMFFIPyFuncCall(TVMFFIPyArgSetterFactory setter_factory, void* func_handle,
                                    PyObject* py_arg_tuple, TVMFFIAny* result, int* c_api_ret_code,
                                    bool release_gil = true,
                                    const DLPackExchangeAPI** out_ctx_dlpack_api = nullptr) {
  return TVMFFIPyCallManager::ThreadLocal()->FuncCall(setter_factory, func_handle, py_arg_tuple,
                                                      result, c_api_ret_code, release_gil,
                                                      out_ctx_dlpack_api);
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
 * \param setter_factory The factory function to create the setter
 * \param func_handle The handle of the function to call
 * \param py_arg_tuple The arguments to the constructor
 * \param result The result of the constructor
 * \param c_api_ret_code The return code of the constructor
 * \param parent_ctx The parent call context
 * \param release_gil Whether to release the GIL
 * \param out_dlpack_exporter The DLPack exporter to be used for the result
 * \return 0 on success, nonzero on failure
 */
TVM_FFI_INLINE int TVMFFIPyConstructorCall(TVMFFIPyArgSetterFactory setter_factory,
                                           void* func_handle, PyObject* py_arg_tuple,
                                           TVMFFIAny* result, int* c_api_ret_code,
                                           TVMFFIPyCallContext* parent_ctx) {
  return TVMFFIPyCallManager::ThreadLocal()->ConstructorCall(
      setter_factory, func_handle, py_arg_tuple, result, c_api_ret_code, parent_ctx);
}

/*!
 * \brief Set a field of a FFI object
 * \param setter_factory The factory function to create the setter
 * \param field_setter The field setter (function pointer or FunctionObj handle)
 * \param field_flags The field flags (to dispatch between function pointer and FunctionObj)
 * \param field_ptr The pointer to the field
 * \param py_arg The python argument to be set
 * \param c_api_ret_code The return code of the function
 * \return 0 on success, nonzero on failure
 */
TVM_FFI_INLINE int TVMFFIPyCallFieldSetter(TVMFFIPyArgSetterFactory setter_factory,
                                           void* field_setter, int64_t field_flags, void* field_ptr,
                                           PyObject* py_arg, int* c_api_ret_code) {
  return TVMFFIPyCallManager::ThreadLocal()->SetField(setter_factory, field_setter, field_flags,
                                                      field_ptr, py_arg, c_api_ret_code);
}

/*!
 * \brief Set an python argument to a FFI Any using the generic dispatcher in call manager
 * \param setter_factory The factory function to create the setter
 * \param ctx The call context
 * \param py_arg_tvm_ffi_value The python argument to be set using the __tvm_ffi_value__ protocol
 * \param out The output argument
 * \return 0 on success, nonzero on failure
 */
TVM_FFI_INLINE int TVMFFIPySetArgumentGenericDispatcher(TVMFFIPyArgSetterFactory setter_factory,
                                                        TVMFFIPyCallContext* ctx,
                                                        PyObject* py_arg_tvm_ffi_value,
                                                        TVMFFIAny* out) {
  return TVMFFIPyCallManager::ThreadLocal()->SetArgument(setter_factory, ctx, py_arg_tvm_ffi_value,
                                                         out);
}

/*!
 * \brief Convert a Python object to a FFI Any
 * \param setter_factory The factory function to create the setter
 * \param py_arg The python argument to be set
 * \param out The output argument
 * \param c_api_ret_code The return code of the function
 * \return 0 on success, nonzero on failure
 */
TVM_FFI_INLINE int TVMFFIPyPyObjectToFFIAny(TVMFFIPyArgSetterFactory setter_factory,
                                            PyObject* py_arg, TVMFFIAny* out, int* c_api_ret_code) {
  return TVMFFIPyCallManager::ThreadLocal()->PyObjectToFFIAny(setter_factory, py_arg, out,
                                                              c_api_ret_code);
}

/*!
 * \brief Get the size of the dispatch map
 * \return The size of the dispatch map
 */
TVM_FFI_INLINE size_t TVMFFIPyGetDispatchMapSize() {
  return TVMFFIPyCallManager::ThreadLocal()->GetDispatchMapSize();
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
