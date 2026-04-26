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

// Returns the code address of this function — for arena co-location tests.
// Load into multiple libraries to verify they land in the same arena region.

#include <tvm/ffi/function.h>

#include <cstdint>

int64_t code_address_impl() { return reinterpret_cast<int64_t>(&code_address_impl); }
TVM_FFI_DLL_EXPORT_TYPED_FUNC(code_address, code_address_impl);
