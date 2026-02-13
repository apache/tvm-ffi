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
 * \file tvm/ffi/container/seq_base.h
 * \brief Base class for sequence containers (Array, List).
 */
#ifndef TVM_FFI_CONTAINER_SEQ_BASE_H_
#define TVM_FFI_CONTAINER_SEQ_BASE_H_

#include <tvm/ffi/any.h>
#include <tvm/ffi/object.h>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <utility>

namespace tvm {
namespace ffi {

/*!
 * \brief Base class for sequence containers (ArrayObj, ListObj).
 *
 * SeqBaseObj is transparent to the FFI type system (no type index),
 * following the same pattern as BytesObjBase.
 */
class SeqBaseObj : public Object, protected TVMFFISeqCell {
 public:
  SeqBaseObj() {
    data = nullptr;
    TVMFFISeqCell::size = 0;
    TVMFFISeqCell::capacity = 0;
    data_deleter = nullptr;
  }

  ~SeqBaseObj() {
    Any* begin = MutableBegin();
    for (int64_t i = 0; i < TVMFFISeqCell::size; ++i) {
      (begin + i)->Any::~Any();
    }
    if (data_deleter != nullptr) {
      data_deleter(data);
    }
  }

  /*! \return The size of the sequence */
  size_t size() const { return static_cast<size_t>(TVMFFISeqCell::size); }

  /*! \return The capacity of the sequence */
  size_t capacity() const { return static_cast<size_t>(TVMFFISeqCell::capacity); }

  /*! \return Whether the sequence is empty */
  bool empty() const { return TVMFFISeqCell::size == 0; }

  /*!
   * \brief Read i-th element from the sequence.
   * \param i The index
   * \return the i-th element.
   */
  const Any& at(int64_t i) const { return this->operator[](i); }

  /*!
   * \brief Read i-th element from the sequence.
   * \param i The index
   * \return the i-th element.
   */
  const Any& operator[](int64_t i) const {
    if (i < 0 || i >= TVMFFISeqCell::size) {
      TVM_FFI_THROW(IndexError) << "Index " << i << " out of bounds " << TVMFFISeqCell::size;
    }
    return static_cast<Any*>(data)[i];
  }

  /*! \return The first element */
  const Any& front() const {
    if (TVMFFISeqCell::size == 0) {
      TVM_FFI_THROW(IndexError) << "front() on empty sequence";
    }
    return static_cast<Any*>(data)[0];
  }

  /*! \return The last element */
  const Any& back() const {
    if (TVMFFISeqCell::size == 0) {
      TVM_FFI_THROW(IndexError) << "back() on empty sequence";
    }
    return static_cast<Any*>(data)[TVMFFISeqCell::size - 1];
  }

  /*! \return begin constant iterator */
  const Any* begin() const { return static_cast<Any*>(data); }

  /*! \return end constant iterator */
  const Any* end() const { return begin() + TVMFFISeqCell::size; }

  /*! \brief Release reference to all the elements */
  void clear() {
    Any* itr = MutableEnd();
    while (TVMFFISeqCell::size > 0) {
      (--itr)->Any::~Any();
      --TVMFFISeqCell::size;
    }
  }

  /*!
   * \brief Set i-th element of the sequence in-place
   * \param i The index
   * \param item The value to be set
   */
  void SetItem(int64_t i, Any item) {
    if (i < 0 || i >= TVMFFISeqCell::size) {
      TVM_FFI_THROW(IndexError) << "Index " << i << " out of bounds " << TVMFFISeqCell::size;
    }
    static_cast<Any*>(data)[i] = std::move(item);
  }

  /*! \brief Remove the last element */
  void pop_back() {
    if (TVMFFISeqCell::size == 0) {
      TVM_FFI_THROW(IndexError) << "pop_back on empty sequence";
    }
    ShrinkBy(1);
  }

  /*!
   * \brief Erase element at position idx
   * \param idx The index to erase
   */
  void erase(int64_t idx) {
    if (idx < 0 || idx >= TVMFFISeqCell::size) {
      TVM_FFI_THROW(IndexError) << "Index " << idx << " out of bounds " << TVMFFISeqCell::size;
    }
    MoveElementsLeft(idx, idx + 1, TVMFFISeqCell::size);
    ShrinkBy(1);
  }

  /*!
   * \brief Erase elements in half-open range [first, last)
   * \param first Start index (inclusive)
   * \param last End index (exclusive)
   */
  void erase(int64_t first, int64_t last) {
    if (first == last) return;
    if (first < 0 || last > TVMFFISeqCell::size || first >= last) {
      TVM_FFI_THROW(IndexError) << "Erase range [" << first << ", " << last << ") out of bounds "
                                << TVMFFISeqCell::size;
    }
    MoveElementsLeft(first, last, TVMFFISeqCell::size);
    ShrinkBy(last - first);
  }

  /*!
   * \brief Insert element at position idx
   * \param idx The index to insert at
   * \param item The value to insert
   * \note Caller must ensure capacity >= size + 1
   */
  void insert(int64_t idx, Any item) {
    int64_t sz = TVMFFISeqCell::size;
    if (idx < 0 || idx > sz) {
      TVM_FFI_THROW(IndexError) << "Index " << idx << " out of bounds [0, " << sz << "]";
    }
    EnlargeBy(1);
    MoveElementsRight(idx + 1, idx, sz);
    MutableBegin()[idx] = std::move(item);
  }

  /*!
   * \brief Insert elements from iterator range at position idx
   * \param idx The index to insert at
   * \param first Begin of iterator
   * \param last End of iterator
   * \tparam IterType The type of iterator
   * \note Caller must ensure capacity >= size + distance(first, last)
   */
  template <typename IterType>
  void insert(int64_t idx, IterType first, IterType last) {
    int64_t count = std::distance(first, last);
    if (count == 0) return;
    int64_t sz = TVMFFISeqCell::size;
    if (idx < 0 || idx > sz) {
      TVM_FFI_THROW(IndexError) << "Index " << idx << " out of bounds [0, " << sz << "]";
    }
    EnlargeBy(count);
    MoveElementsRight(idx + count, idx, sz);
    Any* dst = MutableBegin() + idx;
    for (; first != last; ++first, ++dst) {
      *dst = Any(*first);
    }
  }

  /*! \brief Reverse the elements in-place */
  void Reverse() { std::reverse(MutableBegin(), MutableBegin() + TVMFFISeqCell::size); }

  /*!
   * \brief Resize the sequence
   * \param n The new size
   * \note Caller must ensure capacity >= n when growing
   */
  void resize(int64_t n) {
    if (n < 0) {
      TVM_FFI_THROW(ValueError) << "Cannot resize to negative size";
    }
    int64_t old_size = TVMFFISeqCell::size;
    if (old_size < n) {
      EnlargeBy(n - old_size);
    } else if (old_size > n) {
      ShrinkBy(old_size - n);
    }
  }

 protected:
  /// \cond Doxygen_Suppress
  Any* MutableBegin() const { return static_cast<Any*>(this->data); }

  Any* MutableEnd() const { return MutableBegin() + TVMFFISeqCell::size; }

  template <typename... Args>
  void EmplaceInit(size_t idx, Args&&... args) {
    Any* itr = MutableBegin() + idx;
    new (itr) Any(std::forward<Args>(args)...);
  }

  void EnlargeBy(int64_t delta, const Any& val = Any()) {
    Any* itr = MutableEnd();
    while (delta-- > 0) {
      new (itr++) Any(val);
      ++TVMFFISeqCell::size;
    }
  }

  void ShrinkBy(int64_t delta) {
    Any* itr = MutableEnd();
    while (delta-- > 0) {
      (--itr)->Any::~Any();
      --TVMFFISeqCell::size;
    }
  }

  void MoveElementsLeft(int64_t dst, int64_t src_begin, int64_t src_end) {
    Any* begin = MutableBegin();
    std::move(begin + src_begin, begin + src_end, begin + dst);
  }

  void MoveElementsRight(int64_t dst, int64_t src_begin, int64_t src_end) {
    Any* begin = MutableBegin();
    std::move_backward(begin + src_begin, begin + src_end, begin + dst + (src_end - src_begin));
  }
  /// \endcond
};

/*!
 * \brief CRTP base for sequence type-traits (Array, List).
 *
 * \tparam Derived Must expose:
 *   - `static constexpr int32_t kPrimaryTypeIndex` — the canonical FFI type index
 *   - `static constexpr int32_t kOtherTypeIndex`   — an alternative accepted type index
 *   - `static constexpr const char* kTypeName`      — human-readable name for diagnostics
 */
template <typename Derived, typename SeqRef, typename T>
struct SeqTypeTraitsBase : public ObjectRefTypeTraitsBase<SeqRef> {
  using Base = ObjectRefTypeTraitsBase<SeqRef>;
  using Base::CopyFromAnyViewAfterCheck;

  TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
    if (src->type_index != Derived::kPrimaryTypeIndex) return false;
    if constexpr (std::is_same_v<T, Any>) {
      return true;
    } else {
      const SeqBaseObj* n = reinterpret_cast<const SeqBaseObj*>(src->v_obj);
      for (const Any& any_v : *n) {
        if (!details::AnyUnsafe::CheckAnyStrict<T>(any_v)) return false;
      }
      return true;
    }
  }

  TVM_FFI_INLINE static std::string GetMismatchTypeInfo(const TVMFFIAny* src) {
    if (src->type_index != Derived::kPrimaryTypeIndex &&
        src->type_index != Derived::kOtherTypeIndex) {
      return TypeTraitsBase::GetMismatchTypeInfo(src);
    }
    if constexpr (!std::is_same_v<T, Any>) {
      const SeqBaseObj* n = reinterpret_cast<const SeqBaseObj*>(src->v_obj);
      for (size_t i = 0; i < n->size(); i++) {
        const Any& any_v = n->at(static_cast<int64_t>(i));
        if (details::AnyUnsafe::CheckAnyStrict<T>(any_v)) continue;
        if (any_v.try_cast<T>()) continue;
        return std::string(Derived::kTypeName) + "[index " + std::to_string(i) + ": " +
               details::AnyUnsafe::GetMismatchTypeInfo<T>(any_v) + "]";
      }
    }
    TVM_FFI_THROW(InternalError) << "Cannot reach here";
    TVM_FFI_UNREACHABLE();
  }

  TVM_FFI_INLINE static std::optional<SeqRef> TryCastFromAnyView(const TVMFFIAny* src) {
    if (src->type_index != Derived::kPrimaryTypeIndex &&
        src->type_index != Derived::kOtherTypeIndex) {
      return std::nullopt;
    }
    const SeqBaseObj* n = reinterpret_cast<const SeqBaseObj*>(src->v_obj);
    if constexpr (!std::is_same_v<T, Any>) {
      bool storage_check = [&]() {
        for (const Any& any_v : *n) {
          if (!details::AnyUnsafe::CheckAnyStrict<T>(any_v)) return false;
        }
        return true;
      }();
      if (storage_check && src->type_index == Derived::kPrimaryTypeIndex) {
        return CopyFromAnyViewAfterCheck(src);
      }
      SeqRef result;
      result.reserve(static_cast<int64_t>(n->size()));
      for (const Any& any_v : *n) {
        if (auto opt_v = any_v.try_cast<T>()) {
          result.push_back(*std::move(opt_v));
        } else {
          return std::nullopt;
        }
      }
      return result;
    } else {
      if (src->type_index == Derived::kPrimaryTypeIndex) {
        return CopyFromAnyViewAfterCheck(src);
      }
      SeqRef result;
      result.reserve(static_cast<int64_t>(n->size()));
      for (const Any& any_v : *n) {
        result.push_back(any_v);
      }
      return result;
    }
  }

  TVM_FFI_INLINE static std::string TypeStr() {
    return std::string(Derived::kTypeName) + "<" + details::Type2Str<T>::v() + ">";
  }

 private:
  SeqTypeTraitsBase() = default;
  friend Derived;
};

}  // namespace ffi
}  // namespace tvm

#endif  // TVM_FFI_CONTAINER_SEQ_BASE_H_
