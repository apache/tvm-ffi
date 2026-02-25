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
 * \file tvm/ffi/extra/dataclass.h
 * \brief Reflection-based dataclass operations: deep copy, repr, hash, compare.
 */
#ifndef TVM_FFI_EXTRA_DATACLASS_H_
#define TVM_FFI_EXTRA_DATACLASS_H_

#include <tvm/ffi/any.h>
#include <tvm/ffi/extra/base.h>
#include <tvm/ffi/string.h>

namespace tvm {
namespace ffi {

/**
 * \brief Deep copy an ffi::Any value.
 *
 * Recursively copies the value and all reachable objects in its object graph.
 * Copy-constructible types with `ObjectDef` registration automatically support deep copy.
 * Primitive types, strings, bytes, and Shape are returned as-is (they are immutable).
 * Arrays, Lists, Maps, and Dicts are recursively deep copied.
 * Objects without copy support cause a runtime error.
 *
 * \param value The value to deep copy.
 * \return The deep copied value.
 */
TVM_FFI_EXTRA_CXX_API Any DeepCopy(const Any& value);

/**
 * \brief Produce a human-readable repr string for an ffi::Any value.
 *
 * Recursively formats the value using reflection metadata.
 * Handles cycles (prints "...") and DAGs (caches repr strings).
 * Custom __ffi_repr__ hooks are supported via the reflection type attribute system.
 *
 * \param value The value to repr-print.
 * \return The repr string.
 */
TVM_FFI_EXTRA_CXX_API String ReprPrint(const Any& value);

/**
 * \brief Compute a deterministic recursive hash of an ffi::Any value.
 *
 * Recursively hashes the value and all reachable objects using reflection.
 * Consistent with RecursiveEq: RecursiveEq(a, b) => RecursiveHash(a) == RecursiveHash(b).
 * Custom __ffi_hash__ hooks are supported.
 *
 * \param value The value to hash.
 * \return The hash as int64_t (for FFI compatibility).
 */
TVM_FFI_EXTRA_CXX_API int64_t RecursiveHash(const Any& value);

/**
 * \brief Recursive structural equality comparison.
 * \param lhs Left-hand side value.
 * \param rhs Right-hand side value.
 * \return true if the two values are structurally equal.
 */
TVM_FFI_EXTRA_CXX_API bool RecursiveEq(const Any& lhs, const Any& rhs);

/**
 * \brief Recursive structural less-than comparison.
 * \param lhs Left-hand side value.
 * \param rhs Right-hand side value.
 * \return true if lhs is structurally less than rhs.
 */
TVM_FFI_EXTRA_CXX_API bool RecursiveLt(const Any& lhs, const Any& rhs);

/**
 * \brief Recursive structural less-than-or-equal comparison.
 * \param lhs Left-hand side value.
 * \param rhs Right-hand side value.
 * \return true if lhs is structurally less than or equal to rhs.
 */
TVM_FFI_EXTRA_CXX_API bool RecursiveLe(const Any& lhs, const Any& rhs);

/**
 * \brief Recursive structural greater-than comparison.
 * \param lhs Left-hand side value.
 * \param rhs Right-hand side value.
 * \return true if lhs is structurally greater than rhs.
 */
TVM_FFI_EXTRA_CXX_API bool RecursiveGt(const Any& lhs, const Any& rhs);

/**
 * \brief Recursive structural greater-than-or-equal comparison.
 * \param lhs Left-hand side value.
 * \param rhs Right-hand side value.
 * \return true if lhs is structurally greater than or equal to rhs.
 */
TVM_FFI_EXTRA_CXX_API bool RecursiveGe(const Any& lhs, const Any& rhs);

}  // namespace ffi
}  // namespace tvm
#endif  // TVM_FFI_EXTRA_DATACLASS_H_
