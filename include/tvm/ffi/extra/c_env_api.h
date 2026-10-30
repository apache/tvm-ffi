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
// NOLINTBEGIN(modernize-use-using)
/*!
 * \file tvm/ffi/extra/c_env_api.h
 * \brief Extra environment API.
 */
#ifndef TVM_FFI_EXTRA_C_ENV_API_H_
#define TVM_FFI_EXTRA_C_ENV_API_H_

#include <tvm/ffi/c_api.h>

#ifdef __cplusplus
extern "C" {
#endif

// ----------------------------------------------------------------------------
// Stream context
// Focusing on minimalistic thread-local context recording stream being used.
// We explicitly not handle allocation/de-allocation of stream here.
// ----------------------------------------------------------------------------
/*!
 * \brief The type of the stream handle.
 *
 * The meaning of the handle depends on the device type:
 * - kDLCUDA: cudaStream_t
 * - kDLROCM: hipStream_t
 * - kDLOneAPI: non-owning sycl::queue*, valid for the duration of the call
 *
 * \note For kDLOneAPI, device_id is the index of the device in sycl::device::get_devices(),
 *       following the DLPack convention. It may differ from the framework device index.
 */
typedef void* TVMFFIStreamHandle;

/*!
 * \brief FFI function to set the current stream for a device
 *
 * \param device_type The type of the device.
 * \param device_id The id of the device.
 * \param stream The stream to set.
 * \param opt_out_original_stream Output original stream if the address is not nullptr.
 * \return 0 when success, nonzero when failure happens
 */
TVM_FFI_DLL int TVMFFIEnvSetStream(int32_t device_type, int32_t device_id,
                                   TVMFFIStreamHandle stream,
                                   TVMFFIStreamHandle* opt_out_original_stream);

/*!
 * \brief FFI function to get the current stream for a device
 *
 * \param device_type The type of the device.
 * \param device_id The id of the device.
 * \return The current stream of the device.
 */
TVM_FFI_DLL TVMFFIStreamHandle TVMFFIEnvGetStream(int32_t device_type, int32_t device_id);

/*!
 * \brief Set the current DLPackManagedTensorAllocator in thread-local(TLS) context
 *
 * \param allocator The allocator to set.
 * \param write_to_global_context Whether to also set the allocator to the global context.
 * \param opt_out_original_allocator Output original TLS allocator if the address is not nullptr.
 * \return 0 when success, nonzero when failure happens
 */
TVM_FFI_DLL int TVMFFIEnvSetDLPackManagedTensorAllocator(
    DLPackManagedTensorAllocator allocator, int write_to_global_context,
    DLPackManagedTensorAllocator* opt_out_original_allocator);

/*!
 * \brief FFI function get the current DLPackManagedTensorAllocator stored in context.
 *
 * This function first queries the global context, and if not found,
 * queries the thread-local context.
 *
 * \return The current setted DLPackManagedTensorAllocator
 */
TVM_FFI_DLL DLPackManagedTensorAllocator TVMFFIEnvGetDLPackManagedTensorAllocator();

/*!
 * \brief Allocate a tensor from the allocator set in thread-local(TLS) context.
 *
 * This function redirects to one of environment allocator. As of now, we only
 * support the DLPackManagedTensorAllocator set in thread-local(TLS) context.
 *
 * \param prototype The prototype DLTensor, only the dtype, ndim, shape,
 *                  and device fields are used, other fields are ignored.
 * \param out The output tensor in kTVMFFITensor type.
 * \return 0 when success, nonzero when failure happens
 * \sa TVMFFIEnvSetDLPackManagedTensorAllocator
 */
TVM_FFI_DLL int TVMFFIEnvTensorAlloc(DLTensor* prototype, TVMFFIObjectHandle* out);

/*!
 * \brief Check if there are any signals raised in the surrounding env.
 * \return 0 when success, nonzero when failure happens
 * \note Under python this function redirects to PyErr_CheckSignals
 */
TVM_FFI_DLL int TVMFFIEnvCheckSignals();

/*!
 * \brief Register a symbol into the from the surrounding env such as python
 * \param name The name of the symbol.
 * \param symbol The symbol to register.
 * \return 0 when success, nonzero when failure happens
 */
TVM_FFI_DLL int TVMFFIEnvRegisterCAPI(const char* name, void* symbol);

// ----------------------------------------------------------------------------
// Module symbol management in callee side
// ----------------------------------------------------------------------------
/*!
 * \brief FFI function to lookup a function from a module's imports.
 *
 * This is a helper function that is used by generated code.
 *
 * \param library_ctx The library context module handle.
 * \param func_name The name of the function.
 * \param out The result function.
 * \note The returned function is a weak reference that is cached/owned by the module.
 * \return 0 when no error is thrown, -1 when failure happens
 */
TVM_FFI_DLL int TVMFFIEnvModLookupFromImports(TVMFFIObjectHandle library_ctx, const char* func_name,
                                              TVMFFIObjectHandle* out);

/*!
 * \brief Register a symbol value that will be initialized when a library with the symbol is loaded.
 *
 * This function can be used to make context functions to be available in the library
 * module that wants to avoid an explicit link dependency
 *
 * \param name The name of the symbol.
 * \param symbol The symbol to register.
 * \return 0 when success, nonzero when failure happens
 */
TVM_FFI_DLL int TVMFFIEnvModRegisterContextSymbol(const char* name, void* symbol);

/*!
 * \brief Register a symbol that will be initialized when a system library is loaded.
 *
 * \param name The name of the symbol.
 * \param symbol The symbol to register.
 * \return 0 when success, nonzero when failure happens
 */
TVM_FFI_DLL int TVMFFIEnvModRegisterSystemLibSymbol(const char* name, void* symbol);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // TVM_FFI_EXTRA_C_ENV_API_H_
// NOLINTEND(modernize-use-using)
