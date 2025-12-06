
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
#include <gtest/gtest.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

namespace {

using namespace tvm::ffi;

struct CPUNDAlloc {
  void AllocData(DLTensor* tensor) { tensor->data = malloc(GetDataSize(*tensor)); }
  void FreeData(DLTensor* tensor) { free(tensor->data); }
};

inline Tensor Empty(const Shape& shape, DLDataType dtype, DLDevice device) {
  return Tensor::FromNDAlloc(CPUNDAlloc(), shape, dtype, device);
}

int TestDLPackManagedTensorAllocator(DLTensor* prototype, DLManagedTensorVersioned** out,
                                     void* error_ctx,
                                     void (*SetError)(void* error_ctx, const char* kind,
                                                      const char* message)) {
  Shape shape(prototype->shape, prototype->shape + prototype->ndim);
  Tensor nd = Empty(shape, prototype->dtype, prototype->device);
  *out = nd.ToDLPackVersioned();
  return 0;
}

int TestDLPackManagedTensorAllocatorError(DLTensor* prototype, DLManagedTensorVersioned** out,
                                          void* error_ctx,
                                          void (*SetError)(void* error_ctx, const char* kind,
                                                           const char* message)) {
  SetError(error_ctx, "MemoryError", "TestDLPackManagedTensorAllocatorError");
  return -1;
}

TEST(CEnvAPI, TVMFFIEnvTensorAlloc) {
  auto old_allocator = TVMFFIEnvGetDLPackManagedTensorAllocator();
  TVMFFIEnvSetDLPackManagedTensorAllocator(TestDLPackManagedTensorAllocator, 0, nullptr);
  Tensor tensor = Tensor::FromEnvAlloc(TVMFFIEnvTensorAlloc, {1, 2, 3},
                                       DLDataType({kDLFloat, 32, 1}), DLDevice({kDLCPU, 0}));
  EXPECT_EQ(tensor.use_count(), 1);
  EXPECT_EQ(tensor.shape().size(), 3);
  EXPECT_EQ(tensor.size(0), 1);
  EXPECT_EQ(tensor.size(1), 2);
  EXPECT_EQ(tensor.size(2), 3);
  EXPECT_EQ(tensor.dtype().code, kDLFloat);
  EXPECT_EQ(tensor.dtype().bits, 32);
  EXPECT_EQ(tensor.dtype().lanes, 1);
  EXPECT_EQ(tensor.device().device_type, kDLCPU);
  EXPECT_EQ(tensor.device().device_id, 0);
  EXPECT_NE(tensor.data_ptr(), nullptr);
  TVMFFIEnvSetDLPackManagedTensorAllocator(old_allocator, 0, nullptr);
}

TEST(CEnvAPI, TVMFFIEnvTensorAllocError) {
  auto old_allocator = TVMFFIEnvGetDLPackManagedTensorAllocator();
  TVMFFIEnvSetDLPackManagedTensorAllocator(TestDLPackManagedTensorAllocatorError, 0, nullptr);

  EXPECT_THROW(
      {
        try {
          Tensor::FromEnvAlloc(TVMFFIEnvTensorAlloc, {1, 2, 3}, DLDataType({kDLFloat, 32, 1}),
                               DLDevice({kDLCPU, 0}));
        } catch (const tvm::ffi::Error& e) {
          EXPECT_EQ(e.kind(), "MemoryError");
          EXPECT_EQ(e.message(), "TestDLPackManagedTensorAllocatorError");
          throw;
        }
      },
      tvm::ffi::Error);
  TVMFFIEnvSetDLPackManagedTensorAllocator(old_allocator, 0, nullptr);
}

// Helper functions for TVMFFIHandleInitDeinitOnce test
static int InitSuccess(void** handle_addr) {
  *handle_addr = new int(42);
  return 0;
}

static int InitShouldNotBeCalled(void** handle_addr) {
  *handle_addr = new int(999);
  return 0;
}

static int DeinitSuccess(void* h) {
  delete static_cast<int*>(h);
  return 0;
}

static int DeinitShouldNotBeCalled(void* h) {
  // Should not be called when handle is already null
  return -1;
}

static int InitWithError(void** handle_addr) {
  TVMFFIErrorSetRaisedFromCStr("RuntimeError", "Initialization failed");
  return -1;
}

static int InitReturnsNull(void** handle_addr) {
  *handle_addr = nullptr;  // Invalid: must return non-null handle
  return 0;
}

static int InitForDeinitError(void** handle_addr) {
  *handle_addr = new int(100);
  return 0;
}

static int DeinitWithError(void* h) {
  delete static_cast<int*>(h);
  TVMFFIErrorSetRaisedFromCStr("RuntimeError", "Deinitialization failed");
  return -1;
}

static int InitValue123(void** handle_addr) {
  *handle_addr = new int(123);
  return 0;
}

static int InitValue456(void** handle_addr) {
  *handle_addr = new int(456);
  return 0;
}

TEST(CEnvAPI, TVMFFIHandleInitDeinitOnce) {
  // Test 1: Successful initialization
  void* handle = nullptr;
  int ret = TVMFFIHandleInitOnce(&handle, InitSuccess);
  EXPECT_EQ(ret, 0);
  EXPECT_NE(handle, nullptr);
  EXPECT_EQ(*static_cast<int*>(handle), 42);

  // Test 2: Multiple init calls should not re-initialize (idempotent)
  void* original_handle = handle;
  ret = TVMFFIHandleInitOnce(&handle, InitShouldNotBeCalled);
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(handle, original_handle);         // Handle should remain unchanged
  EXPECT_EQ(*static_cast<int*>(handle), 42);  // Value should still be 42

  // Test 3: Successful deinitialization
  ret = TVMFFIHandleDeinitOnce(&handle, DeinitSuccess);
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(handle, nullptr);

  // Test 4: Multiple deinit calls should be safe (idempotent)
  ret = TVMFFIHandleDeinitOnce(&handle, DeinitShouldNotBeCalled);
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(handle, nullptr);

  // Test 5: Init error - init_func returns error code
  void* handle2 = nullptr;
  ret = TVMFFIHandleInitOnce(&handle2, InitWithError);
  EXPECT_NE(ret, 0);
  EXPECT_EQ(handle2, nullptr);

  // Test 6: Init error - init_func returns nullptr (invalid)
  void* handle3 = nullptr;
  ret = TVMFFIHandleInitOnce(&handle3, InitReturnsNull);
  EXPECT_NE(ret, 0);
  EXPECT_EQ(handle3, nullptr);

  // Test 7: Deinit error - deinit_func returns error
  void* handle4 = nullptr;
  ret = TVMFFIHandleInitOnce(&handle4, InitForDeinitError);
  EXPECT_EQ(ret, 0);
  EXPECT_NE(handle4, nullptr);

  ret = TVMFFIHandleDeinitOnce(&handle4, DeinitWithError);
  EXPECT_NE(ret, 0);
  EXPECT_EQ(handle4, nullptr);  // Handle should still be set to nullptr

  // Test 8: Init-deinit lifecycle
  void* handle5 = nullptr;
  ret = TVMFFIHandleInitOnce(&handle5, InitValue123);
  EXPECT_EQ(ret, 0);
  EXPECT_NE(handle5, nullptr);
  EXPECT_EQ(*static_cast<int*>(handle5), 123);

  ret = TVMFFIHandleDeinitOnce(&handle5, DeinitSuccess);
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(handle5, nullptr);

  // Test 9: Ensure subsequent init after deinit works
  ret = TVMFFIHandleInitOnce(&handle5, InitValue456);
  EXPECT_EQ(ret, 0);
  EXPECT_NE(handle5, nullptr);
  EXPECT_EQ(*static_cast<int*>(handle5), 456);

  // Clean up
  ret = TVMFFIHandleDeinitOnce(&handle5, DeinitSuccess);
  EXPECT_EQ(ret, 0);
}

// Helper functions and data for multithreaded test
struct ThreadSafeCounter {
  int value;
  std::atomic<int>* init_count_ptr;
  std::atomic<int>* deinit_count_ptr;

  ThreadSafeCounter(int v, std::atomic<int>* init_ptr, std::atomic<int>* deinit_ptr)
      : value(v), init_count_ptr(init_ptr), deinit_count_ptr(deinit_ptr) {}
};

// Global pointers for the current test counters
static std::atomic<int>* g_init_count = nullptr;
static std::atomic<int>* g_deinit_count = nullptr;

static int InitWithCounter(void** handle_addr) {
  auto* counter = new ThreadSafeCounter(42, g_init_count, g_deinit_count);
  if (counter->init_count_ptr) {
    counter->init_count_ptr->fetch_add(1, std::memory_order_relaxed);
  }
  // Small delay to increase the race window
  std::this_thread::sleep_for(std::chrono::microseconds(100));
  *handle_addr = counter;
  return 0;
}

static int DeinitWithCounter(void* h) {
  auto* counter = static_cast<ThreadSafeCounter*>(h);
  if (counter->deinit_count_ptr) {
    counter->deinit_count_ptr->fetch_add(1, std::memory_order_relaxed);
  }
  // Small delay to increase the race window
  std::this_thread::sleep_for(std::chrono::microseconds(100));
  delete counter;
  return 0;
}

TEST(CEnvAPI, TVMFFIHandleInitDeinitOnceMultithreaded) {
  // Test 1: Multiple threads calling InitOnce - should initialize only once
  {
    void* handle = nullptr;
    const int num_threads = 4;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    std::vector<int> results(num_threads);
    std::mutex mtx;
    std::condition_variable cv;
    bool ready = false;
    std::atomic<int> init_count{0};

    // Set global counter pointers
    g_init_count = &init_count;
    g_deinit_count = nullptr;

    // Create threads that all try to initialize simultaneously
    for (int i = 0; i < num_threads; ++i) {
      threads.emplace_back([&handle, &results, &mtx, &cv, &ready, i]() {
        // Wait for all threads to be ready
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [&ready] { return ready; });
        lock.unlock();

        results[i] = TVMFFIHandleInitOnce(&handle, InitWithCounter);
      });
    }

    // Signal all threads to start
    {
      std::scoped_lock<std::mutex> lock(mtx);
      ready = true;
    }
    cv.notify_all();

    // Wait for all threads to complete
    for (auto& t : threads) {
      t.join();
    }

    // All threads should succeed
    for (int i = 0; i < num_threads; ++i) {
      EXPECT_EQ(results[i], 0);
    }

    // Handle should be initialized
    EXPECT_NE(handle, nullptr);
    auto* counter = static_cast<ThreadSafeCounter*>(handle);
    EXPECT_EQ(counter->value, 42);

    // Init should have been called exactly once
    EXPECT_EQ(init_count.load(), 1);

    // Clean up
    int ret = TVMFFIHandleDeinitOnce(&handle, DeinitWithCounter);
    EXPECT_EQ(ret, 0);

    // Reset global pointers
    g_init_count = nullptr;
  }

  // Test 2: Multiple threads calling DeinitOnce - should deinitialize only once
  {
    void* handle = nullptr;
    std::atomic<int> init_count{0};
    std::atomic<int> deinit_count{0};

    // Set global counter pointers
    g_init_count = &init_count;
    g_deinit_count = &deinit_count;

    // Initialize first
    int ret = TVMFFIHandleInitOnce(&handle, InitWithCounter);
    EXPECT_EQ(ret, 0);
    EXPECT_NE(handle, nullptr);

    const int num_threads = 4;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    std::vector<int> results(num_threads);
    std::mutex mtx;
    std::condition_variable cv;
    bool ready = false;

    // Create threads that all try to deinitialize simultaneously
    for (int i = 0; i < num_threads; ++i) {
      threads.emplace_back([&handle, &results, &mtx, &cv, &ready, i]() {
        // Wait for all threads to be ready
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [&ready] { return ready; });
        lock.unlock();

        results[i] = TVMFFIHandleDeinitOnce(&handle, DeinitWithCounter);
      });
    }

    // Signal all threads to start
    {
      std::scoped_lock<std::mutex> lock(mtx);
      ready = true;
    }
    cv.notify_all();

    // Wait for all threads to complete
    for (auto& t : threads) {
      t.join();
    }

    // All threads should succeed
    for (int i = 0; i < num_threads; ++i) {
      EXPECT_EQ(results[i], 0);
    }

    // Handle should be null
    EXPECT_EQ(handle, nullptr);

    // Deinit should have been called exactly once
    EXPECT_EQ(deinit_count.load(), 1);

    // Reset global pointers
    g_init_count = nullptr;
    g_deinit_count = nullptr;
  }
}
}  // namespace
