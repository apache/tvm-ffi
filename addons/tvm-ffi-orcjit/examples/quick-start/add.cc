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
 * Quick Start Example - Simple Math Functions
 *
 * This file demonstrates how to export C++ functions using TVM-FFI
 * so they can be loaded dynamically at runtime with tvm-ffi-orcjit.
 */

#include <tvm/ffi/function.h>

// Simple addition function
int add_impl(int a, int b) { return a + b; }
TVM_FFI_DLL_EXPORT_TYPED_FUNC(add, add_impl);

// Multiplication function
int multiply_impl(int a, int b) { return a * b; }
TVM_FFI_DLL_EXPORT_TYPED_FUNC(multiply, multiply_impl);

// Fibonacci function (recursive)
int fib_impl(int n) {
  if (n <= 1) return n;
  return fib_impl(n - 1) + fib_impl(n - 2);
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fibonacci, fib_impl);

// String concatenation example
std::string concat_impl(std::string a, std::string b) { return a + b; }
TVM_FFI_DLL_EXPORT_TYPED_FUNC(concat, concat_impl);
