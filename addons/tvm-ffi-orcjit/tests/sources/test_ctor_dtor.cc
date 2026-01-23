// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include <tvm/ffi/function.h>

using namespace tvm::ffi;

#define PUTS_LOG(msg)                                               \
  auto append_log_func = Function::GetGlobalRequired("append_log"); \
  append_log_func(msg);

using ctor_t = void (*)();
using dtor_t = void (*)();

__attribute__((constructor)) void init_array() { PUTS_LOG("<init_array>"); }

__attribute__((constructor(101))) void init_array_101() { PUTS_LOG("<init_array.101>"); }

__attribute__((constructor(102))) void init_array_102() { PUTS_LOG("<init_array.102>"); }
__attribute__((constructor(103))) void init_array_103() { PUTS_LOG("<init_array.103>"); }

static void ctors() { PUTS_LOG("<ctors>"); }
__attribute__((section(".ctors"), used)) static ctor_t ctors_ptr = ctors;

static void ctors_101() { PUTS_LOG("<ctors.101>"); }
__attribute__((section(".ctors.101"), used)) static ctor_t ctors_1_ptr = ctors_101;

static void ctors_102() { PUTS_LOG("<ctors.102>"); }
__attribute__((section(".ctors.102"), used)) static ctor_t ctors_2_ptr = ctors_102;

static void ctors_103() { PUTS_LOG("<ctors.103>"); }
__attribute__((section(".ctors.103"), used)) static ctor_t ctors_3_ptr = ctors_103;

__attribute__((destructor)) void fini_array() { PUTS_LOG("<fini_array>"); }

__attribute__((destructor(101))) void fini_array_101() { PUTS_LOG("<fini_array.101>"); }

__attribute__((destructor(102))) void fini_array_102() { PUTS_LOG("<fini_array.102>"); }

__attribute__((destructor(103))) void fini_array_103() { PUTS_LOG("<fini_array.103>"); }
static void dtors() { PUTS_LOG("<dtors>"); }
__attribute__((section(".dtors"), used)) static dtor_t dtors_ptr = dtors;

static void dtors_101() { PUTS_LOG("<dtors.101>"); }
__attribute__((section(".dtors.101"), used)) static dtor_t dtors_1_ptr = dtors_101;

static void dtors_102() { PUTS_LOG("<dtors.102>"); }
__attribute__((section(".dtors.102"), used)) static dtor_t dtors_2_ptr = dtors_102;

static void dtors_103() { PUTS_LOG("<dtors.103>"); }
__attribute__((section(".dtors.103"), used)) static dtor_t dtors_3_ptr = dtors_103;

void main_impl() { PUTS_LOG("<main>"); }
TVM_FFI_DLL_EXPORT_TYPED_FUNC(main, main_impl);
