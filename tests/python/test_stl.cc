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
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/stl.h>

#include <array>
#include <numeric>
#include <optional>
#include <tuple>
#include <variant>
#include <vector>

namespace {

auto test_tuple(std::tuple<int, float> arg) -> std::tuple<float, int> {
  return std::make_tuple(std::get<1>(arg), std::get<0>(arg));
}

auto test_vector(std::optional<std::vector<std::array<int, 2>>> arg)
    -> std::optional<std::vector<int>> {
  if (arg) {
    auto result = std::vector<int>{};
    result.reserve(arg->size());
    for (const auto& row : *arg) {
      result.push_back(std::accumulate(row.begin(), row.end(), 0));
    }
    return result;
  } else {
    return std::nullopt;
  }
}

auto test_variant(std::variant<int, float, std::vector<std::variant<int, float>>> arg)
    -> std::variant<std::string, std::vector<std::string>> {
  if (std::holds_alternative<int>(arg)) {
    return "int";
  } else if (std::holds_alternative<float>(arg)) {
    return "float";
  } else {
    auto result = std::vector<std::string>{};
    for (const auto& item : std::get<std::vector<std::variant<int, float>>>(arg)) {
      if (std::holds_alternative<int>(item)) {
        result.emplace_back("int");
      } else {
        result.emplace_back("float");
      }
    }
    return result;
  }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(test_tuple, test_tuple);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(test_vector, test_vector);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(test_variant, test_variant);

}  // namespace
