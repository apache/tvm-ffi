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

#include <tvm/ffi/tvm_ffi.h>

#include <type_traits>

namespace abi_object_test {

struct BaseObj;
struct DerivedObj;
struct GrandchildObj;
struct RecursiveObj;
struct UnrelatedObj;

}  // namespace abi_object_test

// RecursiveObj is incomplete here. Specializing the trait enables its Arc storage traits before
// List<Arc<RecursiveObj>> is instantiated inside the class definition below.
template <>
inline constexpr bool tvm::ffi::is_object_subclass_v<::abi_object_test::RecursiveObj> = true;

static_assert(::tvm::ffi::is_object_subclass_v<::abi_object_test::RecursiveObj>);
static_assert(
    ::tvm::ffi::details::storage_enabled_v<::tvm::ffi::Arc<::abi_object_test::RecursiveObj>>);

namespace abi_object_test {

struct BaseObj : public ::tvm::ffi::Object {
  TVM_FFI_DECLARE_OBJECT_INFO_LOOKUP("testing.abi_object.Base", 1);
};

struct DerivedObj : public BaseObj {
  TVM_FFI_DECLARE_OBJECT_INFO_LOOKUP("testing.abi_object.Derived", 2);
};

struct GrandchildObj : public DerivedObj {
  TVM_FFI_DECLARE_OBJECT_INFO_LOOKUP("testing.abi_object.Grandchild", 3);
};

struct RecursiveObj : public ::tvm::ffi::Object {
  ::tvm::ffi::List<::tvm::ffi::Arc<RecursiveObj>> children;

  TVM_FFI_DECLARE_OBJECT_INFO_LOOKUP("testing.abi_object.Recursive", 1);
};

struct UnrelatedObj : public ::tvm::ffi::Object {
  TVM_FFI_DECLARE_OBJECT_INFO_LOOKUP("testing.abi_object.Unrelated", 1);
};

template <typename T, typename = void>
struct HasRuntimeTypeInfo : std::false_type {};

template <typename T>
struct HasRuntimeTypeInfo<T, std::void_t<decltype(T::RuntimeTypeInfo())>> : std::true_type {};

static_assert(!HasRuntimeTypeInfo<BaseObj>::value);
static_assert(std::is_base_of_v<::tvm::ffi::Object, BaseObj> &&
              std::is_base_of_v<BaseObj, DerivedObj> &&
              std::is_base_of_v<DerivedObj, GrandchildObj>);
static_assert(std::is_convertible_v<GrandchildObj*, BaseObj*>);
static_assert(
    std::is_constructible_v<::tvm::ffi::ObjectPtr<BaseObj>, ::tvm::ffi::ObjectPtr<GrandchildObj>>);
static_assert(
    std::is_assignable_v<::tvm::ffi::ObjectPtr<BaseObj>&, ::tvm::ffi::ObjectPtr<DerivedObj>>);
static_assert(std::is_constructible_v<::tvm::ffi::Arc<BaseObj>, ::tvm::ffi::Arc<GrandchildObj>>);
static_assert(std::is_assignable_v<::tvm::ffi::Arc<BaseObj>&, ::tvm::ffi::Arc<DerivedObj>>);
static_assert(::tvm::ffi::type_subsumes_v<::tvm::ffi::ObjectPtr<BaseObj>,
                                          ::tvm::ffi::ObjectPtr<GrandchildObj>>);
static_assert(
    ::tvm::ffi::type_subsumes_v<::tvm::ffi::Arc<BaseObj>, ::tvm::ffi::Arc<GrandchildObj>>);
static_assert(
    ::tvm::ffi::type_subsumes_v<::tvm::ffi::ObjectPtr<BaseObj>, ::tvm::ffi::Arc<GrandchildObj>>);
static_assert(
    !::tvm::ffi::type_subsumes_v<::tvm::ffi::Arc<BaseObj>, ::tvm::ffi::ObjectPtr<GrandchildObj>>);
static_assert(!::tvm::ffi::type_subsumes_v<::tvm::ffi::ObjectPtr<UnrelatedObj>,
                                           ::tvm::ffi::ObjectPtr<DerivedObj>>);

}  // namespace abi_object_test
