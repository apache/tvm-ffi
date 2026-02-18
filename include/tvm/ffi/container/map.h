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
 * \file tvm/ffi/container/map.h
 * \brief Runtime Map container types.
 */
#ifndef TVM_FFI_CONTAINER_MAP_H_
#define TVM_FFI_CONTAINER_MAP_H_

#include <tvm/ffi/any.h>
#include <tvm/ffi/container/container_details.h>
#include <tvm/ffi/container/map_base.h>
#include <tvm/ffi/memory.h>
#include <tvm/ffi/object.h>
#include <tvm/ffi/optional.h>

#include <unordered_map>

namespace tvm {
namespace ffi {

/*! \brief Map object */
class MapObj : public MapBaseObj {
 public:
  /// \cond Doxygen_Suppress
  static constexpr const int32_t _type_index = TypeIndex::kTVMFFIMap;
  static const constexpr bool _type_final = true;
  TVM_FFI_DECLARE_OBJECT_INFO_STATIC(StaticTypeKey::kTVMFFIMap, MapObj, Object);
  /// \endcond

 protected:
  template <typename, typename, typename>
  friend class Map;
};

/*!
 * \brief Map container of NodeRef->NodeRef in DSL graph.
 *  Map implements copy on write semantics, which means map is mutable
 *  but copy will happen when array is referenced in more than two places.
 *
 * operator[] only provide const acces, use Set to mutate the content.
 * \tparam K The key NodeRef type.
 * \tparam V The value NodeRef type.
 */
template <typename K, typename V,
          typename = typename std::enable_if_t<details::storage_enabled_v<K> &&
                                               details::storage_enabled_v<V>>>
class Map : public ObjectRef {
 public:
  /*! \brief The key type of the map */
  using key_type = K;
  /*! \brief The mapped type of the map */
  using mapped_type = V;
  /*! \brief The iterator type of the map */
  class iterator;
  /*!
   * \brief Construct an Map with UnsafeInit
   */
  explicit Map(UnsafeInit tag) : ObjectRef(tag) {}
  /*!
   * \brief default constructor
   */
  Map() { data_ = MapObj::Empty<MapObj>(); }
  /*!
   * \brief move constructor
   * \param other source
   */
  Map(Map<K, V>&& other)  // NOLINT(google-explicit-constructor)
      : ObjectRef(std::move(other.data_)) {}
  /*!
   * \brief copy constructor
   * \param other source
   */
  Map(const Map<K, V>& other)  // NOLINT(google-explicit-constructor)
      : ObjectRef(other.data_) {}

  /*!
   * \brief Move constructor
   * \param other The other map
   * \tparam KU The key type of the other map
   * \tparam VU The mapped type of the other map
   */
  template <typename KU, typename VU,
            typename = std::enable_if_t<details::type_contains_v<K, KU> &&
                                        details::type_contains_v<V, VU>>>
  Map(Map<KU, VU>&& other)  // NOLINT(google-explicit-constructor)
      : ObjectRef(std::move(other.data_)) {}

  /*!
   * \brief Copy constructor
   * \param other The other map
   * \tparam KU The key type of the other map
   * \tparam VU The mapped type of the other map
   */
  template <typename KU, typename VU,
            typename = std::enable_if_t<details::type_contains_v<K, KU> &&
                                        details::type_contains_v<V, VU>>>
  Map(const Map<KU, VU>& other) : ObjectRef(other.data_) {}  // NOLINT(google-explicit-constructor)

  /*!
   * \brief Move assignment
   * \param other The other map
   */
  Map<K, V>& operator=(Map<K, V>&& other) {
    data_ = std::move(other.data_);
    return *this;
  }

  /*!
   * \brief Copy assignment
   * \param other The other map
   */
  Map<K, V>& operator=(const Map<K, V>& other) {
    data_ = other.data_;
    return *this;
  }

  /*!
   * \brief Move assignment
   * \param other The other map
   * \tparam KU The key type of the other map
   * \tparam VU The mapped type of the other map
   */
  template <typename KU, typename VU,
            typename = std::enable_if_t<details::type_contains_v<K, KU> &&
                                        details::type_contains_v<V, VU>>>
  Map<K, V>& operator=(Map<KU, VU>&& other) {
    data_ = std::move(other.data_);
    return *this;
  }

  /*!
   * \brief Copy assignment
   * \param other The other map
   * \tparam KU The key type of the other map
   * \tparam VU The mapped type of the other map
   */
  template <typename KU, typename VU,
            typename = std::enable_if_t<details::type_contains_v<K, KU> &&
                                        details::type_contains_v<V, VU>>>
  Map<K, V>& operator=(const Map<KU, VU>& other) {
    data_ = other.data_;
    return *this;
  }
  /*!
   * \brief constructor from pointer
   * \param n the container pointer
   */
  explicit Map(ObjectPtr<Object> n) : ObjectRef(n) {}
  /*!
   * \brief constructor from iterator
   * \param begin begin of iterator
   * \param end end of iterator
   * \tparam IterType The type of iterator
   */
  template <typename IterType>
  Map(IterType begin, IterType end) {
    data_ = MapObj::CreateFromRange<MapObj>(begin, end);
  }
  /*!
   * \brief constructor from initializer list
   * \param init The initalizer list
   */
  Map(std::initializer_list<std::pair<K, V>> init) {
    data_ = MapObj::CreateFromRange<MapObj>(init.begin(), init.end());
  }
  /*!
   * \brief constructor from unordered_map
   * \param init The unordered_map
   */
  template <typename Hash, typename Equal>
  Map(const std::unordered_map<K, V, Hash, Equal>& init) {  // NOLINT(*)
    data_ = MapObj::CreateFromRange<MapObj>(init.begin(), init.end());
  }
  /*!
   * \brief Read element from map.
   * \param key The key
   * \return the corresonding element.
   */
  const V at(const K& key) const {
    return details::AnyUnsafe::CopyFromAnyViewAfterCheck<V>(GetMapObj()->at(key));
  }
  /*!
   * \brief Read element from map.
   * \param key The key
   * \return the corresonding element.
   */
  const V operator[](const K& key) const { return this->at(key); }
  /*! \return The size of the array */
  size_t size() const {
    MapObj* n = GetMapObj();
    return n == nullptr ? 0 : n->size();
  }
  /*! \return The number of elements of the key */
  size_t count(const K& key) const {
    MapObj* n = GetMapObj();
    return n == nullptr ? 0 : GetMapObj()->count(key);
  }
  /*! \return whether array is empty */
  bool empty() const { return size() == 0; }
  /*! \brief Release reference to all the elements */
  void clear() {
    MapObj* n = GetMapObj();
    if (n != nullptr) {
      data_ = MapObj::Empty<MapObj>();
    }
  }
  /*!
   * \brief set the Map.
   * \param key The index key.
   * \param value The value to be setted.
   */
  void Set(const K& key, const V& value) {
    CopyOnWrite();
    MapObj::InsertMaybeReHash<MapObj>(MapObj::KVType(key, value), &data_);
  }
  /*! \return begin iterator */
  iterator begin() const { return iterator(GetMapObj()->begin()); }
  /*! \return end iterator */
  iterator end() const { return iterator(GetMapObj()->end()); }
  /*! \return find the key and returns the associated iterator */
  iterator find(const K& key) const { return iterator(GetMapObj()->find(key)); }
  /*! \return The value associated with the key, std::nullopt if not found */
  std::optional<V> Get(const K& key) const {
    MapObj::iterator iter = GetMapObj()->find(key);
    if (iter == GetMapObj()->end()) {
      return std::nullopt;
    }
    return details::AnyUnsafe::CopyFromAnyViewAfterCheck<V>(iter->second);
  }

  /*!
   * \brief Erase the entry associated with the key
   * \param key The key
   */
  void erase(const K& key) { CopyOnWrite()->erase(key); }

  /*!
   * \brief copy on write semantics
   *  Do nothing if current handle is the unique copy of the array.
   *  Otherwise make a new copy of the array to ensure the current handle
   *  hold a unique copy.
   *
   * \return Handle to the internal node container(which guarantees to be unique)
   */
  MapObj* CopyOnWrite() {
    if (data_.get() == nullptr) {
      data_ = MapObj::Empty<MapObj>();
    } else if (!data_.unique()) {
      data_ = MapObj::CopyFrom<MapObj>(GetMapObj());
    }
    return GetMapObj();
  }
  /*! \brief specify container node */
  using ContainerType = MapObj;

  /// \cond Doxygen_Suppress
  /*! \brief Iterator of the hash map */
  class iterator {
   public:
    using iterator_category = std::bidirectional_iterator_tag;
    using difference_type = int64_t;
    using value_type = const std::pair<K, V>;
    using pointer = value_type*;
    using reference = value_type;

    iterator() : itr() {}

    /*! \brief Compare iterators */
    bool operator==(const iterator& other) const { return itr == other.itr; }
    /*! \brief Compare iterators */
    bool operator!=(const iterator& other) const { return itr != other.itr; }
    /*! \brief De-reference iterators is not allowed */
    pointer operator->() const = delete;
    /*! \brief De-reference iterators */
    reference operator*() const {
      auto& kv = *itr;
      return std::make_pair(details::AnyUnsafe::CopyFromAnyViewAfterCheck<K>(kv.first),
                            details::AnyUnsafe::CopyFromAnyViewAfterCheck<V>(kv.second));
    }
    /*! \brief Prefix self increment, e.g. ++iter */
    iterator& operator++() {
      ++itr;
      return *this;
    }
    /*! \brief Suffix self increment */
    iterator operator++(int) {
      iterator copy = *this;
      ++(*this);
      return copy;
    }

    /*! \brief Prefix self decrement, e.g. --iter */
    iterator& operator--() {
      --itr;
      return *this;
    }
    /*! \brief Suffix self decrement */
    iterator operator--(int) {
      iterator copy = *this;
      --(*this);
      return copy;
    }

   private:
    iterator(const MapObj::iterator& itr)  // NOLINT(*)
        : itr(itr) {}

    template <typename, typename, typename>
    friend class Map;

    MapObj::iterator itr;
  };
  /// \endcond

 private:
  /*! \brief Return data_ as type of pointer of MapObj */
  MapObj* GetMapObj() const { return static_cast<MapObj*>(data_.get()); }

  template <typename, typename, typename>
  friend class Map;
};

/*!
 * \brief Merge two Maps.
 * \param lhs the first Map to merge.
 * \param rhs the second Map to merge.
 * @return The merged Array. Original Maps are kept unchanged.
 */
template <typename K, typename V,
          typename = typename std::enable_if_t<details::storage_enabled_v<K> &&
                                               details::storage_enabled_v<V>>>
inline Map<K, V> Merge(Map<K, V> lhs, const Map<K, V>& rhs) {
  for (const auto& p : rhs) {
    lhs.Set(p.first, p.second);
  }
  return std::move(lhs);
}

// Traits for Map
template <typename K, typename V>
inline constexpr bool use_default_type_traits_v<Map<K, V>> = false;

template <typename K, typename V>
struct TypeTraits<Map<K, V>> : public ObjectRefTypeTraitsBase<Map<K, V>> {
  static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIMap;
  using ObjectRefTypeTraitsBase<Map<K, V>>::CopyFromAnyViewAfterCheck;

  TVM_FFI_INLINE static std::string GetMismatchTypeInfo(const TVMFFIAny* src) {
    if (src->type_index != TypeIndex::kTVMFFIMap) {
      return TypeTraitsBase::GetMismatchTypeInfo(src);
    }
    if constexpr (!std::is_same_v<K, Any> || !std::is_same_v<V, Any>) {
      const MapObj* n = reinterpret_cast<const MapObj*>(src->v_obj);
      for (const auto& kv : *n) {
        if constexpr (!std::is_same_v<K, Any>) {
          if (!details::AnyUnsafe::CheckAnyStrict<K>(kv.first) &&
              !kv.first.try_cast<K>().has_value()) {
            return "Map[some key is " + details::AnyUnsafe::GetMismatchTypeInfo<K>(kv.first) +
                   ", V]";
          }
        }
        if constexpr (!std::is_same_v<V, Any>) {
          if (!details::AnyUnsafe::CheckAnyStrict<V>(kv.second) &&
              !kv.second.try_cast<V>().has_value()) {
            return "Map[K, some value is " + details::AnyUnsafe::GetMismatchTypeInfo<V>(kv.second) +
                   "]";
          }
        }
      }
    }
    TVM_FFI_THROW(InternalError) << "Cannot reach here";
    TVM_FFI_UNREACHABLE();
  }

  TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
    if (src->type_index != TypeIndex::kTVMFFIMap) return false;
    if constexpr (std::is_same_v<K, Any> && std::is_same_v<V, Any>) {
      return true;
    } else {
      const MapObj* n = reinterpret_cast<const MapObj*>(src->v_obj);
      for (const auto& kv : *n) {
        if constexpr (!std::is_same_v<K, Any>) {
          if (!details::AnyUnsafe::CheckAnyStrict<K>(kv.first)) return false;
        }
        if constexpr (!std::is_same_v<V, Any>) {
          if (!details::AnyUnsafe::CheckAnyStrict<V>(kv.second)) return false;
        }
      }
      return true;
    }
  }

  TVM_FFI_INLINE static std::optional<Map<K, V>> TryCastFromAnyView(const TVMFFIAny* src) {
    if (src->type_index != TypeIndex::kTVMFFIMap) return std::nullopt;
    if constexpr (!std::is_same_v<K, Any> || !std::is_same_v<V, Any>) {
      const MapObj* n = reinterpret_cast<const MapObj*>(src->v_obj);
      bool storage_check = [&]() {
        for (const auto& kv : *n) {
          if constexpr (!std::is_same_v<K, Any>) {
            if (!details::AnyUnsafe::CheckAnyStrict<K>(kv.first)) return false;
          }
          if constexpr (!std::is_same_v<V, Any>) {
            if (!details::AnyUnsafe::CheckAnyStrict<V>(kv.second)) return false;
          }
        }
        return true;
      }();
      // fast path, if storage check passes, we can return the array directly.
      if (storage_check) return CopyFromAnyViewAfterCheck(src);
      // slow path, we need to create a new map and convert to the target type.
      Map<K, V> ret;
      for (const auto& kv : *n) {
        auto k = kv.first.try_cast<K>();
        auto v = kv.second.try_cast<V>();
        if (!k.has_value() || !v.has_value()) return std::nullopt;
        ret.Set(*std::move(k), *std::move(v));
      }
      return ret;
    } else {
      return CopyFromAnyViewAfterCheck(src);
    }
  }

  TVM_FFI_INLINE static std::string TypeStr() {
    return "Map<" + details::Type2Str<K>::v() + ", " + details::Type2Str<V>::v() + ">";
  }
  TVM_FFI_INLINE static std::string TypeSchema() {
    std::ostringstream oss;
    oss << R"({"type":")" << StaticTypeKey::kTVMFFIMap << R"(","args":[)";
    oss << details::TypeSchema<K>::v() << ",";
    oss << details::TypeSchema<V>::v();
    oss << "]}";
    return oss.str();
  }
};

namespace details {
template <typename K, typename V, typename KU, typename VU>
inline constexpr bool type_contains_v<Map<K, V>, Map<KU, VU>> =
    type_contains_v<K, KU> && type_contains_v<V, VU>;
}  // namespace details

}  // namespace ffi
}  // namespace tvm
#endif  // TVM_FFI_CONTAINER_MAP_H_
