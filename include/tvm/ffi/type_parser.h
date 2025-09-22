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

/*! \file tvm/ffi/type_parser.h
 * \brief Utilities to serialize C++ TVM FFI types into JSON strings.
 */
#ifndef TVM_FFI_TYPE_PARSER_H_
#define TVM_FFI_TYPE_PARSER_H_

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/container/variant.h>
#include <tvm/ffi/optional.h>
#include <tvm/ffi/type_traits.h>

#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace tvm {
namespace ffi {

namespace type_parser {
namespace detail {

/*! \brief Remove const and reference qualifiers from a type. */
template <typename T>
struct RemoveCVRef {
  using type = std::remove_cv_t<std::remove_reference_t<T>>;
};

template <typename T>
using RemoveCVRef_t = typename RemoveCVRef<T>::type;

/*! \brief Escape characters in a string to produce a JSON encoded value. */
TVM_FFI_DLL std::string EscapeJSONString(std::string_view input);

/*! \brief Wrap a string as a JSON string literal. */
TVM_FFI_DLL std::string QuoteJSONString(std::string_view input);

/*! \brief Assemble a composite JSON representation. */
TVM_FFI_DLL std::string ComposeComposite(std::string_view kind, std::vector<std::string> args);

/*! \brief Generic implementation that treats a type as a primitive. */
template <typename T>
struct TypeParserImpl {
  static std::string Parse() { return QuoteJSONString(TypeTraitsNoCR<T>::TypeStr()); }
};

/*! \brief Specialization for void type. */
template <>
struct TypeParserImpl<void> {
  static std::string Parse() { return QuoteJSONString("void"); }
};

template <>
struct TypeParserImpl<Any> {
  static std::string Parse() { return QuoteJSONString("Any"); }
};

template <>
struct TypeParserImpl<AnyView> {
  static std::string Parse() { return QuoteJSONString("AnyView"); }
};

/*! \brief Specialization for Optional container types. */
template <typename T>
struct TypeParserImpl<Optional<T>> {
  static std::string Parse() {
    using Element = RemoveCVRef_t<T>;
    std::vector<std::string> args;
    args.reserve(1);
    args.emplace_back(TypeParserImpl<Element>::Parse());
    return ComposeComposite("Optional", std::move(args));
  }
};

/*! \brief Specialization for Array container types. */
template <typename T>
struct TypeParserImpl<Array<T>> {
  static std::string Parse() {
    using Element = RemoveCVRef_t<T>;
    std::vector<std::string> args;
    args.reserve(1);
    args.emplace_back(TypeParserImpl<Element>::Parse());
    return ComposeComposite("Array", std::move(args));
  }
};

/*! \brief Specialization for Map container types. */
template <typename K, typename V>
struct TypeParserImpl<Map<K, V>> {
  static std::string Parse() {
    using Key = RemoveCVRef_t<K>;
    using Value = RemoveCVRef_t<V>;
    std::vector<std::string> args;
    args.reserve(2);
    args.emplace_back(TypeParserImpl<Key>::Parse());
    args.emplace_back(TypeParserImpl<Value>::Parse());
    return ComposeComposite("Map", std::move(args));
  }
};

/*! \brief Specialization for Variant container types. */
template <typename... V>
struct TypeParserImpl<Variant<V...>> {
  static std::string Parse() {
    std::vector<std::string> args;
    args.reserve(sizeof...(V));
    (args.emplace_back(TypeParserImpl<RemoveCVRef_t<V>>::Parse()), ...);
    return ComposeComposite("Variant", std::move(args));
  }
};

/*! \brief Dispatcher that normalizes the type before parsing. */
template <typename T>
std::string ParseTypeNormalized() {
  return TypeParserImpl<RemoveCVRef_t<T>>::Parse();
}

}  // namespace detail

/*! \brief Convert a C++ TVM FFI type to a JSON string description.
 *
 * The resulting string matches the grammar:
 *
 * \code
 * type = composite | primitive
 * primitive = string  // type key registered in the FFI
 * composite = { "kind": string, "args": [ type, ... ] }
 * \endcode
 */
template <typename T>
std::string ParseType() {
  return detail::ParseTypeNormalized<T>();
}

/*! \brief Get a reference to a cached JSON type schema string for the given type. */
template <typename T>
const std::string& GetTypeSchemaString() {
  using Normalized = detail::RemoveCVRef_t<T>;
  static const std::string schema = detail::ParseTypeNormalized<Normalized>();
  return schema;
}

namespace detail {

template <typename... Args>
std::vector<std::string> ParseTypeSequence() {
  std::vector<std::string> result;
  result.reserve(sizeof...(Args));
  (result.emplace_back(ParseTypeNormalized<Args>()), ...);
  return result;
}

template <typename R, typename... Args>
std::string ParseFunctionTypeNormalized() {
  auto args = ParseTypeSequence<Args...>();
  args.emplace_back(ParseTypeNormalized<R>());
  return ComposeComposite("Func", std::move(args));
}

template <typename R, typename... Args>
const std::string& GetFunctionTypeSchemaStorage() {
  static const std::string schema =
      ParseFunctionTypeNormalized<RemoveCVRef_t<R>, RemoveCVRef_t<Args>...>();
  return schema;
}

}  // namespace detail

/*! \brief Get a reference to a cached JSON function type schema string. */
template <typename R, typename... Args>
const std::string& GetFunctionTypeSchemaString() {
  return detail::GetFunctionTypeSchemaStorage<R, Args...>();
}

}  // namespace type_parser

// Expose ParseType at namespace tvm::ffi for convenience.
using type_parser::ParseType;

using type_parser::GetFunctionTypeSchemaString;
using type_parser::GetTypeSchemaString;

}  // namespace ffi
}  // namespace tvm

#endif  // TVM_FFI_TYPE_PARSER_H_
