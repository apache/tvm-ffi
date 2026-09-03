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
 * \file int_pair.cc
 * \brief A tvm-ffi library that registers one object for the Rust stub generator.
 */
#include <tvm/ffi/tvm_ffi.h>

#include <cstdint>

namespace rust_stubgen {

namespace ffi = tvm::ffi;

// [object.begin]
// A polymorphic object: the vtable in front of the object header means Rust
// cannot mirror its bytes, so the generated binding reads every field through
// the reflection getters and construction stays on the C++ side.
class IntPairObj : public ffi::Object {
 public:
  int64_t a;
  int64_t b;
  int32_t kind;

  IntPairObj(int64_t a, int64_t b, int32_t kind) : a(a), b(b), kind(kind) {}
  virtual ~IntPairObj() = default;
  virtual int64_t Sum() const { return a + b; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("rust_stubgen.IntPair", IntPairObj, ffi::Object);
};

class IntPair : public ffi::ObjectRef {
 public:
  IntPair(int64_t a, int64_t b, int32_t kind) { data_ = ffi::make_object<IntPairObj>(a, b, kind); }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(IntPair, ffi::ObjectRef, IntPairObj);
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<IntPairObj>(refl::init(false))
      .def_ro("a", &IntPairObj::a, "the first operand")
      .def_ro("b", &IntPairObj::b, "the second operand")
      .def_ro("kind", &IntPairObj::kind, "0 = unordered, 1 = ordered");
  refl::GlobalDef()
      .def("rust_stubgen.IntPair",
           [](int64_t a, int64_t b, int32_t kind) { return IntPair(a, b, kind); })
      .def("rust_stubgen.IntPairSum", [](const IntPair& pair) { return pair->Sum(); });
}
// [object.end]

}  // namespace rust_stubgen
