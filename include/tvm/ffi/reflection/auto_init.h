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
 * \file tvm/ffi/reflection/auto_init.h
 * \brief Auto-generated ``__ffi_init__`` constructor based on reflection metadata.
 *
 * When a C++ type registered with ``ObjectDef`` does not provide an explicit
 * ``refl::init<Args...>()``, the system auto-generates a packed ``__ffi_init__``
 * that creates an object via the default creator and sets each ``init=True``
 * field from positional arguments (in field-declaration order, parent first).
 *
 * Two calling conventions are supported:
 *
 * 1. **Positional-only** ``(val0, val1, val2, ...)``: each value maps to the
 *    corresponding ``init=True`` field in declaration order.
 *
 * 2. **KWARGS** ``(pos0, ..., KWARGS, "key0", val0, "key1", val1, ...)``:
 *    positional values fill non-kw-only ``init=True`` fields; remaining
 *    key-value pairs are matched by field name.
 *
 * Fields marked ``Init(false)`` are excluded from the parameter list.  If such
 * a field has a default value, it is filled automatically; otherwise the
 * creator-default is used.
 */
#ifndef TVM_FFI_REFLECTION_AUTO_INIT_H_
#define TVM_FFI_REFLECTION_AUTO_INIT_H_

#include <tvm/ffi/c_api.h>
#include <tvm/ffi/function.h>

namespace tvm {
namespace ffi {
namespace reflection {

/*!
 * \brief Build a packed ``__ffi_init__`` function for the given type.
 *
 * The returned Function creates an instance of the type, binds positional
 * (and optionally keyword) arguments to ``init=True`` fields, fills defaults
 * for unbound fields, and returns the new ObjectRef.
 *
 * \param type_index The runtime type index of the target type.
 * \return A packed Function suitable for registration as ``__ffi_init__``.
 */
TVM_FFI_DLL Function MakeAutoInit(int32_t type_index);

}  // namespace reflection
}  // namespace ffi
}  // namespace tvm
#endif  // TVM_FFI_REFLECTION_AUTO_INIT_H_
