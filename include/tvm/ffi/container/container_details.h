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
 * \file tvm/ffi/container/container_details.h
 * \brief Common utilities for typed container types.
 */
#ifndef TVM_FFI_CONTAINER_CONTAINER_DETAILS_H_
#define TVM_FFI_CONTAINER_CONTAINER_DETAILS_H_

#include <tvm/ffi/any.h>
#include <tvm/ffi/memory.h>
#include <tvm/ffi/object.h>

#include <sstream>
#include <string>
#include <type_traits>
#include <utility>

namespace tvm {
namespace ffi {
namespace details {

/*!
 * \brief iterator adapter that adapts TIter to return another type.
 * \tparam Converter a struct that contains converting function
 * \tparam TIter the content iterator type.
 */
template <typename Converter, typename TIter>
class IterAdapter {
 public:
  using difference_type = typename std::iterator_traits<TIter>::difference_type;
  using value_type = typename Converter::ResultType;
  using pointer = const typename Converter::ResultType*;
  using reference = const typename Converter::ResultType;
  using iterator_category = typename std::iterator_traits<TIter>::iterator_category;

  explicit IterAdapter(TIter iter) : iter_(iter) {}
  IterAdapter& operator++() {
    ++iter_;
    return *this;
  }
  IterAdapter& operator--() {
    --iter_;
    return *this;
  }
  IterAdapter operator++(int) {
    IterAdapter copy = *this;
    ++iter_;
    return copy;
  }
  IterAdapter operator--(int) {
    IterAdapter copy = *this;
    --iter_;
    return copy;
  }

  IterAdapter operator+(difference_type offset) const { return IterAdapter(iter_ + offset); }

  IterAdapter operator-(difference_type offset) const { return IterAdapter(iter_ - offset); }

  IterAdapter& operator+=(difference_type offset) {
    iter_ += offset;
    return *this;
  }

  IterAdapter& operator-=(difference_type offset) {
    iter_ -= offset;
    return *this;
  }

  template <typename T = IterAdapter>
  inline std::enable_if_t<std::is_same_v<iterator_category, std::random_access_iterator_tag>,
                          typename T::difference_type>
  operator-(const IterAdapter& rhs) const {
    return iter_ - rhs.iter_;
  }

  bool operator==(IterAdapter other) const { return iter_ == other.iter_; }
  bool operator!=(IterAdapter other) const { return !(*this == other); }
  reference operator*() const { return Converter::convert(*iter_); }

 private:
  TIter iter_;
};

/*!
 * \brief iterator adapter that adapts TIter to return another type.
 * \tparam Converter a struct that contains converting function
 * \tparam TIter the content iterator type.
 */
template <typename Converter, typename TIter>
class ReverseIterAdapter {
 public:
  using difference_type = typename std::iterator_traits<TIter>::difference_type;
  using value_type = typename Converter::ResultType;
  using pointer = const typename Converter::ResultType*;
  using reference = const typename Converter::ResultType;
  using iterator_category = typename std::iterator_traits<TIter>::iterator_category;

  explicit ReverseIterAdapter(TIter iter) : iter_(iter) {}
  ReverseIterAdapter& operator++() {
    --iter_;
    return *this;
  }
  ReverseIterAdapter& operator--() {
    ++iter_;
    return *this;
  }
  ReverseIterAdapter operator++(int) {
    ReverseIterAdapter copy = *this;
    --iter_;
    return copy;
  }
  ReverseIterAdapter operator--(int) {
    ReverseIterAdapter copy = *this;
    ++iter_;
    return copy;
  }
  ReverseIterAdapter operator+(difference_type offset) const {
    return ReverseIterAdapter(iter_ - offset);
  }

  template <typename T = ReverseIterAdapter>
  inline std::enable_if_t<std::is_same_v<iterator_category, std::random_access_iterator_tag>,
                          typename T::difference_type>
  operator-(const ReverseIterAdapter& rhs) const {
    return rhs.iter_ - iter_;
  }

  bool operator==(ReverseIterAdapter other) const { return iter_ == other.iter_; }
  bool operator!=(ReverseIterAdapter other) const { return !(*this == other); }
  reference operator*() const { return Converter::convert(*iter_); }

 private:
  TIter iter_;
};

/*!
 * \brief Check if T is compatible with Any.
 *
 * \tparam T The type to check.
 * \return True if T is compatible with Any, false otherwise.
 */
template <typename T>
inline constexpr bool storage_enabled_v = std::is_same_v<T, Any> || TypeTraits<T>::storage_enabled;

/*!
 * \brief Check if all T are compatible with Any.
 *
 * \tparam T The type to check.
 * \return True if T is compatible with Any, false otherwise.
 */
template <typename... T>
inline constexpr bool all_storage_enabled_v = (storage_enabled_v<T> && ...);

/*!
 * \brief Check if all T are compatible with Any.
 *
 * \tparam T The type to check.
 * \return True if T is compatible with Any, false otherwise.
 */
template <typename... T>
inline constexpr bool all_object_ref_v = (std::is_base_of_v<ObjectRef, T> && ...);
/**
 * \brief Check if Any storage of Derived can always be directly used as Base.
 *
 * \tparam Base The base type.
 * \tparam Derived The derived type.
 * \return True if Derived's storage can be used as Base's storage, false otherwise.
 */
template <typename Base, typename Derived>
inline constexpr bool type_contains_v =
    std::is_base_of_v<Base, Derived> || std::is_same_v<Base, Derived>;
// special case for Any
template <typename Derived>
inline constexpr bool type_contains_v<Any, Derived> = true;

/*!
 * \brief Create a string of the container type.
 * \tparam V The types of the elements in the container.
 * \param name The name of the container type.
 * \return A string of the container type.
 */
template <typename... V>
std::string ContainerTypeStr(const char* name) {
  std::stringstream ss;
  // helper to construct concated string of TypeStr
  class TypeStrHelper {
   public:
    TypeStrHelper(std::stringstream& stream) : stream_(stream) {}  // NOLINT(*)

    TypeStrHelper& operator<<(const std::string& str) {
      if (counter_ > 0) {
        stream_ << ", ";
      }
      stream_ << str;
      counter_++;
      return *this;
    }

   private:
    std::stringstream& stream_;  // NOLINT(*)
    int counter_ = 0;
  };
  TypeStrHelper helper(ss);
  ss << name << '<';
  (helper << ... << Type2Str<V>::v());
  ss << '>';
  return ss.str();
}

}  // namespace details
}  // namespace ffi
}  // namespace tvm
#endif  // TVM_FFI_CONTAINER_CONTAINER_DETAILS_H_
