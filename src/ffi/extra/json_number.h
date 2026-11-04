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
#ifndef TVM_FFI_SRC_EXTRA_JSON_NUMBER_H_
#define TVM_FFI_SRC_EXTRA_JSON_NUMBER_H_

#include <locale.h>
#include <tvm/ffi/error.h>

#ifdef __APPLE__
#include <xlocale.h>
#endif

namespace tvm {
namespace ffi {
namespace json {

// Keep an explicit C locale for libraries without floating-point charconv and
// for strtod's subnormal range checks. Never change the process-global locale.
inline auto GetJSONNumberLocale() {
  static const struct NumberLocale {
#ifdef _WIN32
    _locale_t value = _create_locale(LC_NUMERIC, "C");
    ~NumberLocale() { _free_locale(value); }
#else
    locale_t value = newlocale(LC_NUMERIC_MASK, "C", nullptr);
    ~NumberLocale() { freelocale(value); }
#endif
    NumberLocale() { TVM_FFI_ICHECK(value != nullptr) << "Cannot create JSON numeric locale"; }
  } locale;
  return locale.value;
}

}  // namespace json
}  // namespace ffi
}  // namespace tvm
#endif  // TVM_FFI_SRC_EXTRA_JSON_NUMBER_H_
