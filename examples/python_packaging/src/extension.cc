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
 * \file extension.cc
 * \brief Example of a tvm-ffi based library that registers various functions.
 */
#include <tvm/ffi/function.h>
#include <tvm/ffi/object.h>
#include <tvm/ffi/reflection/registry.h>

#include <cstdint>

namespace my_ffi_extension {

namespace ffi = tvm::ffi;

// [tvm_ffi_abi.begin]
static int AddTwo(int x) { return x + 2; }

TVM_FFI_DLL_EXPORT_TYPED_FUNC(add_two, AddTwo);
// [tvm_ffi_abi.end]

// [global_function.begin]
static int AddOne(int x) { return x + 1; }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()  //
      .def("my_ffi_extension.add_one", AddOne);
}
// [global_function.end]

/*!
 * \brief Raise a runtime error to demonstrate error propagation.
 *
 * \param msg The error message to raise.
 *
 * \code{.py}
 * import my_ffi_extension
 * try:
 *   my_ffi_extension.raise_error("boom")
 * except RuntimeError:
 *   pass
 * \endcode
 */
static void RaiseError(const ffi::String& msg) { TVM_FFI_THROW(RuntimeError) << msg; }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()  //
      .def("my_ffi_extension.raise_error", RaiseError);
}

// [object.begin]
class IntPairObj : public ffi::Object {
 public:
  int64_t a;
  int64_t b;

  IntPairObj(int64_t a, int64_t b) : a(a), b(b) {}

  int64_t Sum() const { return a + b; }

  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL(
      /*type_key=*/"my_ffi_extension.IntPair",
      /*class=*/IntPairObj,
      /*parent_class=*/ffi::Object);
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<IntPairObj>()
      .def(refl::init<int64_t, int64_t>())
      .def_rw("a", &IntPairObj::a, "the first field")
      .def_rw("b", &IntPairObj::b, "the second field")
      .def("sum", &IntPairObj::Sum, "IntPairObj::Sum() method");
}
// [object.end]
}  // namespace my_ffi_extension
