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
 * \brief Example of a tvm-ffi based library that registers an object for Rust stubgen.
 */
#include <tvm/ffi/tvm_ffi.h>

#include <cstdint>

namespace rust_stubgen {

namespace ffi = tvm::ffi;

// [object.begin]
class IntPairObj : public ffi::Object {
 public:
  int64_t a;
  int64_t b;
  // `scale` carries a reflected default: the generated Rust builder prefills
  // it and exposes a `.scale(..)` setter instead of a required parameter.
  int64_t scale = 1;

  IntPairObj(int64_t a, int64_t b) : a(a), b(b) {}

  int64_t Sum() const { return (a + b) * scale; }

  // All fields are writable, so the generated Rust wrapper gets `DerefMut`.
  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL(
      /*type_key=*/"rust_stubgen.IntPair",
      /*class=*/IntPairObj,
      /*parent_class=*/ffi::Object);
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<IntPairObj>()
      .def(refl::init<int64_t, int64_t>())
      .def_rw("a", &IntPairObj::a, "the first field")
      .def_rw("b", &IntPairObj::b, "the second field")
      .def_rw("scale", &IntPairObj::scale, refl::init(false), refl::default_value(int64_t{1}),
              "sum multiplier (defaulted -> builder setter in Rust)")
      .def("sum", &IntPairObj::Sum, "(a + b) * scale");
}
// [object.end]
}  // namespace rust_stubgen
