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
/*!
 * \file tvm/ffi/extra/deep_copy.h
 * \brief Reflection-based object copy utilities
 */
#ifndef TVM_FFI_EXTRA_DEEP_COPY_H_
#define TVM_FFI_EXTRA_DEEP_COPY_H_

#include <tvm/ffi/any.h>
#include <tvm/ffi/extra/base.h>

namespace tvm {
namespace ffi {

/**
 * \brief Deep copy an ffi::Any value.
 *
 * Recursively copies the value and all reachable objects in its object graph.
 * Copy-constructible types with `ObjectDef` registration automatically support deep copy.
 * Primitive types, strings, and bytes are returned as-is (they are immutable).
 * Arrays, Lists, and Maps are recursively deep copied.
 * Objects without copy support cause a runtime error.
 *
 * \param value The value to deep copy.
 * \return The deep copied value.
 */
TVM_FFI_EXTRA_CXX_API Any DeepCopy(const Any& value);

}  // namespace ffi
}  // namespace tvm
#endif  // TVM_FFI_EXTRA_DEEP_COPY_H_
