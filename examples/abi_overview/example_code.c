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
 * Example code for TVM-FFI ABI overview.
 *
 * Compilation command:
 *
 * ```bash
 * gcc $(tvm-ffi-config --cflags)             \
 *     $(tvm-ffi-config --ldflags)            \
 *     $(tvm-ffi-config --libs)               \
 *     -Wl,-rpath,$(tvm-ffi-config --libdir)  \
 *     -o ./example_code
 * ```
 */
// NOLINTBEGIN(modernize-deprecated-headers,modernize-use-nullptr)
#include <assert.h>
#include <dlpack/dlpack.h>
#include <stdio.h>
#include <string.h>
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/extra/c_env_api.h>

int IS_OWNING_ANY = 1;

// [Any_AnyView.FromInt_Float.begin]
TVMFFIAny Any_AnyView_FromInt(int64_t value) {
  TVMFFIAny any;
  any.type_index = kTVMFFIInt;
  any.zero_padding = 0;
  any.v_int64 = value;
  return any;
}

TVMFFIAny Any_AnyView_FromFloat(double value) {
  TVMFFIAny any;
  any.type_index = kTVMFFIFloat;
  any.zero_padding = 0;
  any.v_float64 = value;
  return any;
}
// [Any_AnyView.FromInt_Float.end]

// [Any_AnyView.FromObjectPtr.begin]
TVMFFIAny Any_AnyView_FromObjectPtr(TVMFFIObject* obj) {
  TVMFFIAny any;
  assert(obj != NULL);
  any.type_index = kTVMFFIObject;
  any.zero_padding = 0;
  any.v_obj = obj;
  // Increment refcount if it's Any (owning) instead of AnyView (borrowing)
  if (IS_OWNING_ANY) {
    TVMFFIObjectIncRef(obj);
  }
  return any;
}
// [Any_AnyView.FromObjectPtr.end]

// [Any_AnyView.Destroy.begin]
void Any_AnyView_Destroy(TVMFFIAny* any) {
  if (IS_OWNING_ANY) {
    // Checks if `any` holds a heap-allocated object,
    // and if so, decrements the reference count
    if (any->type_index >= kTVMFFIStaticObjectBegin) {
      TVMFFIObjectDecRef(any->v_obj);
    }
  }
  *any = (TVMFFIAny){0};  // Clears the `any` struct
}
// [Any_AnyView.Destroy.end]

// [Any_AnyView.GetInt_Float.begin]
int64_t Any_AnyView_GetInt(const TVMFFIAny* any) {
  if (any->type_index == kTVMFFIInt || any->type_index == kTVMFFIBool) {
    return any->v_int64;
  } else if (any->type_index == kTVMFFIFloat) {
    return (int64_t)(any->v_float64);
  }
  assert(0);  // FAILED to read int
  return 0;
}

double Any_AnyView_GetFloat(const TVMFFIAny* any) {
  if (any->type_index == kTVMFFIInt || any->type_index == kTVMFFIBool) {
    return (double)(any->v_int64);
  } else if (any->type_index == kTVMFFIFloat) {
    return any->v_float64;
  }
  assert(0);  // FAILED to read float
  return 0.0;
}
// [Any_AnyView.GetInt_Float.end]

// [Any_AnyView.GetDLTensor.begin]
DLTensor* Any_AnyView_GetDLTensor(const TVMFFIAny* value) {
  if (value->type_index == kTVMFFIDLTensorPtr) {
    return (DLTensor*)(value->v_ptr);
  } else if (value->type_index == kTVMFFITensor) {
    return (DLTensor*)((char*)(value->v_obj) + sizeof(TVMFFIObject));
  }
  assert(0);  // FAILED to read DLTensor
  return NULL;
}
// [Any_AnyView.GetDLTensor.end]

// [Any_AnyView.GetObject.begin]
TVMFFIObject* Any_AnyView_GetObject(const TVMFFIAny* value) {
  if (value->type_index == kTVMFFINone) {
    return NULL;  // Handling nullptr if needed
  } else if (value->type_index >= kTVMFFIStaticObjectBegin) {
    return value->v_obj;
  }
  assert(0);  // FAILED: not a TVM-FFI object
  return NULL;
}
// [Any_AnyView.GetObject.end]

// [Object.IsInstance.begin]
int Object_IsInstance(int32_t sub_type_index, int32_t super_type_index, int32_t super_type_depth) {
  const TVMFFITypeInfo* sub_type_info = NULL;
  // Everything is a subclass of object.
  if (sub_type_index == super_type_index) {
    return 1;
  }
  // Invariance: parent index is always smaller than the child.
  if (sub_type_index < super_type_index) {
    return 0;
  }
  sub_type_info = TVMFFIGetTypeInfo(sub_type_index);
  return sub_type_info->type_depth > super_type_depth &&
         sub_type_info->type_ancestors[super_type_depth]->type_index == super_type_index;
}
// [Object.IsInstance.end]

// [Object.MoveFromAny.begin]
void Object_MoveFromAny(TVMFFIAny* any, TVMFFIObject** obj) {
  assert(any->type_index >= kTVMFFIStaticObjectBegin);
  *obj = any->v_obj;
  (*any) = (TVMFFIAny){0};
  if (!IS_OWNING_ANY) {
    TVMFFIObjectIncRef(*obj);
  }
}
// [Object.MoveFromAny.end]

// [Object.Destroy.begin]
void Object_Destroy(TVMFFIObject* obj) {
  assert(obj != NULL);
  TVMFFIObjectDecRef(obj);
}
// [Object.Destroy.end]

// [Tensor.AccessDLTensor.begin]
DLTensor* Tensor_AccessDLTensor(TVMFFIObject* tensor) {
  assert(tensor != NULL);
  return (DLTensor*)((char*)tensor + sizeof(TVMFFIObject));
}
// [Tensor.AccessDLTensor.end]

// [Tensor_FromDLPack.begin]
TVMFFIObject* Tensor_FromDLPack(DLManagedTensorVersioned* from) {
  int err_code = 0;
  TVMFFIObject* out = NULL;
  err_code = TVMFFITensorFromDLPackVersioned(  //
      from,                                    // input DLPack tensor
      /*require_alignment=*/0,                 // no alignment requirement
      /*require_contiguous=*/1,                // require contiguous tensor
      (void**)(&out));
  assert(err_code == 0);
  return out;
}
// [Tensor_FromDLPack.end]

// [Tensor_Alloc.begin]
TVMFFIObject* Tensor_Alloc(DLTensor* prototype) {
  int err_code = 0;
  TVMFFIObject* out = NULL;
  assert(prototype->data == NULL);
  err_code = TVMFFIEnvTensorAlloc(prototype, (void**)(&out));
  assert(err_code == 0);
  return out;
}
// [Tensor_Alloc.end]

// [Tensor_ToDLPackVersioned.begin]
DLManagedTensorVersioned* Tensor_ToDLPackVersioned(TVMFFIObject* tensor) {
  int err_code = 0;
  DLManagedTensorVersioned* out = NULL;
  err_code = TVMFFITensorToDLPackVersioned(tensor, &out);
  assert(err_code == 0);
  return out;
}
// [Tensor_ToDLPackVersioned.end]

// [Function.Construct.begin]
TVMFFIObject* Function_Construct(void* self, TVMFFISafeCallType safe_call,
                                 void (*deleter)(void* self)) {
  int err_code;
  TVMFFIObject* out = NULL;
  err_code = TVMFFIFunctionCreate(self, safe_call, deleter, (void**)(&out));
  assert(err_code == 0);
  return out;
}
// [Function.Construct.end]

// [Function.Call.begin]
int64_t CallFunction(TVMFFIObject* func, int64_t x, int64_t y) {
  int err_code;
  TVMFFIAny args[2];
  TVMFFIAny result = (TVMFFIAny){0};
  args[0] = Any_AnyView_FromInt(x);
  args[1] = Any_AnyView_FromInt(y);
  err_code = TVMFFIFunctionCall(func, args, 2, &result);
  assert(err_code == 0);
  return Any_AnyView_GetInt(&result);
}
// [Function.Call.end]

// [Function.GetGlobal.begin]
TVMFFIObject* Function_RetrieveGlobal(const char* name) {
  TVMFFIObject* out = NULL;
  TVMFFIByteArray name_byte_array = {name, strlen(name)};
  int err_code = TVMFFIFunctionGetGlobal(&name_byte_array, (void**)(&out));
  assert(err_code == 0);
  return out;
}
// [Function.GetGlobal.end]

// [Function.SetGlobal.begin]
void Function_SetGlobal(const char* name, TVMFFIObject* func) {
  TVMFFIByteArray name_byte_array = {name, strlen(name)};
  int err_code = TVMFFIFunctionSetGlobal(&name_byte_array, func, 0);
  assert(err_code == 0);
}
// [Function.SetGlobal.end]

// [Error.Print.begin]
void PrintError(TVMFFIObject* err) {
  TVMFFIErrorCell* cell = (TVMFFIErrorCell*)((char*)err + sizeof(TVMFFIObject));
  fprintf(stderr, "%.*s: %.*s\n",                        //
          (int)cell->kind.size, cell->kind.data,         // e.g. "ValueError"
          (int)cell->message.size, cell->message.data);  // e.g. "Expected at least 2 arguments"
  if (cell->backtrace.size) {
    fprintf(stderr, "Backtrace:\n%.*s\n", (int)cell->backtrace.size, cell->backtrace.data);
  }
}
// [Error.Print.end]

// [Error.HandleReturnCode.begin]
void Error_HandleReturnCode(int rc) {
  TVMFFIObject* err = NULL;
  if (rc == -1) {
    // Move the raised error from TLS (clears TLS slot)
    TVMFFIErrorMoveFromRaised((void**)(&err));  // now `err` owns the error object
    if (err != NULL) {
      PrintError(err);  // print the error
      // IMPORTANT: Release the error object, or gets memory leaks
      TVMFFIObjectDecRef(err);
    }
  } else if (rc == -2) {
    // Frontend (e.g., Python) already has an exception set.
    // Do not fetch from TLS; consult the frontend's error mechanism.
    return;
  }
}
// [Error.HandleReturnCode.end]

// [Error.RaiseException.begin]
int Error_RaiseException(void* handle, const TVMFFIAny* args, int32_t num_args, TVMFFIAny* result) {
  TVMFFIErrorSetRaisedFromCStr("ValueError", "Expected at least 2 arguments");
  return -1;
}
// [Error.RaiseException.end]

// NOLINTEND(modernize-deprecated-headers,modernize-use-nullptr)
int main() { return 0; }
