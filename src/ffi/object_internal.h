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
 * \file object_internal.h
 * \brief Internal declarations for object sentinel helpers.
 */
#ifndef TVM_FFI_OBJECT_INTERNAL_H_
#define TVM_FFI_OBJECT_INTERNAL_H_

#include <tvm/ffi/object.h>

namespace tvm {
namespace ffi {

/*! \brief Return the singleton "missing" sentinel object. */
ObjectRef GetMissingObject();
/*! \brief Return the singleton KWARGS sentinel object. */
ObjectRef GetKwargsObject();

}  // namespace ffi
}  // namespace tvm

#endif  // TVM_FFI_OBJECT_INTERNAL_H_
