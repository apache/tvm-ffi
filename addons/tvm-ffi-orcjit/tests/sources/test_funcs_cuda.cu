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

#include <cassert>
#include <cstdio>

void checkPtr(void* ptr) {
  cudaPointerAttributes attr;
  cudaError_t err = cudaPointerGetAttributes(&attr, ptr);

  if (err != cudaSuccess) {
    printf("Pointer check failed: %s\n", cudaGetErrorString(err));
    return;
  }

  printf("Pointer is valid:\n");
  printf("  type       : %d\n", attr.type);
  printf("  device     : %d\n", attr.device);
  printf("  devicePointer: %p\n", attr.devicePointer);
  printf("  hostPointer  : %p\n", attr.hostPointer);
}

// Simple addition function
__global__ void test_add_kernel(int* a, int* b, int* c) { *c = *a + *b; }
int test_add_impl(int a, int b) {
  int c;
  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, sizeof(int));
  cudaMalloc(&d_b, sizeof(int));
  cudaMalloc(&d_c, sizeof(int));
  cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);
  printf("ttt %p %p %p\n", d_a, d_b, d_c);
  checkPtr(d_a);
  checkPtr(d_b);
  checkPtr(d_c);
  test_add_kernel<<<1, 1>>>(d_a, d_b, d_c);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) printf("Kernel launch error: %s\n", cudaGetErrorString(err));
  cudaMemcpy(&c, d_c, sizeof(int), cudaMemcpyDeviceToHost);
  printf("ggg %d %d %d", a, b, c);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  return c;
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(test_add, test_add_impl);

// Multiplication function

__global__ void test_multiply_kernel(int* a, int* b, int* c) { *c = *a * *b; }
int test_multiply_impl(int a, int b) {
  int c;
  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, sizeof(int));
  cudaMalloc(&d_b, sizeof(int));
  cudaMalloc(&d_c, sizeof(int));
  cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);
  test_multiply_kernel<<<1, 1>>>(d_a, d_b, d_c);
  cudaMemcpy(&c, d_c, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  return c;
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(test_multiply, test_multiply_impl);
