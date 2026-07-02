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
 * \file tvm/ffi/object.h
 * \brief A managed object in the TVM FFI.
 */
#ifndef TVM_FFI_OBJECT_H_
#define TVM_FFI_OBJECT_H_

#include <tvm/ffi/base_details.h>
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/type_traits.h>

#include <optional>
#include <string>
#include <type_traits>
#include <utility>

namespace tvm {
namespace ffi {

/*!
 * \brief Helper tag to explicitly request unsafe initialization.
 *
 * Constructing an ObjectRefType with UnsafeInit{} will set the data_ member to nullptr.
 *
 * When initializing Object fields, ObjectRef fields can be set to UnsafeInit.
 * This enables the "construct with UnsafeInit then set all fields" pattern
 * when the object does not have a default constructor.
 *
 * Used for initialization in controlled scenarios where such unsafe
 * initialization is known to be safe.
 *
 * Each ObjectRefType should have a constructor that takes an UnsafeInit tag.
 *
 * \note As the name suggests, do not use it in normal code paths.
 */
struct UnsafeInit {};

namespace details {
// Helper to perform
// unsafe operations related to object
struct ObjectUnsafe;

/*! \brief One counter for weak reference. */
constexpr uint64_t kCombinedRefCountWeakOne = static_cast<uint64_t>(1) << 32;
/*! \brief One counter for strong reference. */
constexpr uint64_t kCombinedRefCountStrongOne = 1;
/*! \brief Both reference counts. */
constexpr uint64_t kCombinedRefCountBothOne = kCombinedRefCountWeakOne | kCombinedRefCountStrongOne;
/*! \brief Mask to get the lower 32 bits of the combined reference count. */
constexpr uint64_t kCombinedRefCountMaskUInt32 = (static_cast<uint64_t>(1) << 32) - 1;

/*!
 * Check if the type_index is an instance of TargetObjectType.
 *
 * \tparam TargetType The target object type to be checked.
 *
 * \param object_type_index The type index to be checked, caller
 *        ensures that the index is already within the object index range.
 *
 * \return Whether the target type is true.
 */
template <typename TargetType>
TVM_FFI_INLINE bool IsObjectInstance(int32_t object_type_index);

}  // namespace details

/*!
 * \brief Base class of all object containers.
 *
 * Sub-class of objects should declare the following static constexpr fields:
 *
 * - _type_index:
 *      Static type index of the object, if assigned to TypeIndex::kTVMFFIDynObject
 *      the type index will be assigned during runtime.
 *      Runtime type index can be accessed by ObjectType::TypeIndex();
 * - _type_key:
 *       The unique string identifier of the type.
 * - _type_final:
 *       Whether the type is terminal type(there is no subclass of the type in the object system).
 *       This field is automatically set by macro TVM_FFI_DECLARE_OBJECT_INFO_FINAL
 *       It is still OK to sub-class a terminal object type T and construct it using make_object.
 *       But IsInstance check will only show that the object type is T(instead of the sub-class).
 * - _type_mutable:
 *      Whether we would like to expose cast to non-constant pointer
 *      ObjectType* from Any/AnyView. By default, we set to false so it is not exposed.
 *
 * The following two fields are necessary for base classes that can be sub-classed.
 *
 * - _type_child_slots:
 *       Number of reserved type index slots for child classes.
 *       Used for runtime optimization for type checking in IsInstance.
 *       If an object's type_index is within range of [type_index, type_index + _type_child_slots]
 *       Then the object can be quickly decided as sub-class of the current object class.
 *       If not, a fallback mechanism is used to check the global type table.
 *       Recommendation: set to estimate number of children needed.
 *
 * - _type_child_slots_can_overflow:
 *       Whether we can add additional child classes even if the number of child classes
 *       exceeds the _type_child_slots. A fallback mechanism to check type table will be used.
 *       Recommendation: set to false for optimal runtime speed if we know exact number of children.
 *
 * Two macros are used to declare helper functions in the object:
 * - Use TVM_FFI_DECLARE_OBJECT_INFO for object classes that can be sub-classed.
 * - Use TVM_FFI_DECLARE_OBJECT_INFO_FINAL for object classes that cannot be sub-classed.
 *
 * New objects can be created using make_object function.
 * Which will automatically populate the type_index and deleter of the object.
 */
class Object {
 protected:
  /*! \brief header field that is the common prefix of all objects */
  TVMFFIObject header_;

 public:
  Object() {
    header_.combined_ref_count = 0;
    header_.type_index = 0;
    header_.__padding = 0;
    header_.__ensure_align = 0;
  }
  /*!
   * Check if the object is an instance of TargetType.
   * \tparam TargetType The target type to be checked.
   * \return Whether the target type is true.
   */
  template <typename TargetType>
  bool IsInstance() const {
    return details::IsObjectInstance<TargetType>(header_.type_index);
  }

  /*! \return The internal runtime type index of the object. */
  int32_t type_index() const { return header_.type_index; }

  /*!
   * \return the type key of the object.
   * \note this operation is expensive, can be used for error reporting.
   */
  std::string GetTypeKey() const {
    // the function checks that the info exists
    const TypeInfo* type_info = TVMFFIGetTypeInfo(header_.type_index);
    return std::string(type_info->type_key.data, type_info->type_key.size);
  }

  /*!
   * \return A hash value of the return of GetTypeKey.
   */
  uint64_t GetTypeKeyHash() const {
    // the function checks that the info exists
    const TypeInfo* type_info = TVMFFIGetTypeInfo(header_.type_index);
    return type_info->type_key_hash;
  }

  /*!
   * \brief Get the type key of the corresponding index from runtime.
   * \param tindex The type index.
   * \return the result.
   */
  static std::string TypeIndex2Key(int32_t tindex) {
    const TypeInfo* type_info = TVMFFIGetTypeInfo(tindex);
    return std::string(type_info->type_key.data, type_info->type_key.size);
  }

  /*!
   * \return Whether the object.use_count() == 1.
   */
  bool unique() const { return use_count() == 1; }

  /*!
   * \return The usage count of the cell.
   * \note We use STL style naming to be consistent with known API in shared_ptr.
   */
  uint64_t use_count() const {
    // only need relaxed load of counters
#ifdef _MSC_VER
    return ((reinterpret_cast<const volatile uint64_t*>(
               &header_.combined_ref_count))[0]  // NOLINT(*)
            ) &
           kCombinedRefCountMaskUInt32;
#else
    return __atomic_load_n(&(header_.combined_ref_count), __ATOMIC_RELAXED) &
           kCombinedRefCountMaskUInt32;
#endif
  }

  //----------------------------------------------------------------------------
  //  The following fields are configuration flags for subclasses of object
  //----------------------------------------------------------------------------
  /*! \brief The type key of the class */
  static constexpr const char* _type_key = StaticTypeKey::kTVMFFIObject;
  /*! \brief Whether the class is final */
  static constexpr bool _type_final = false;
  /*! \brief Whether allow mutable access to fields */
  static constexpr bool _type_mutable = false;
  /*! \brief The number of child slots of the class to pre-allocate to this type */
  static constexpr uint32_t _type_child_slots = 0;
  /*!
   * \brief Whether allow additional children beyond pre-specified by _type_child_slots
   */
  static constexpr bool _type_child_slots_can_overflow = true;
  /*! \brief The static type index of the class */
  static constexpr int32_t _type_index = TypeIndex::kTVMFFIObject;
  /*! \brief The static depth of the class in the object hierarchy */
  static constexpr int32_t _type_depth = 0;
  /*! \brief The structural equality and hash kind of the type */
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindUnsupported;
  // The following functions are provided by macro
  // TVM_FFI_DECLARE_OBJECT_INFO and TVM_FFI_DECLARE_OBJECT_INFO_FINAL
  /*!
   * \brief Get the runtime allocated type index of the type
   * \note Getting this information may need dynamic calls into a global table.
   */
  TVM_FFI_INLINE static int32_t RuntimeTypeIndex() noexcept { return TypeIndex::kTVMFFIObject; }
  /*!
   * \brief Internal function to get or allocate a runtime index.
   */
  TVM_FFI_COLD_CODE static int32_t _GetOrAllocRuntimeTypeIndex() {  // NOLINT(*)
    return TypeIndex::kTVMFFIObject;
  }

 private:
  // exposing detailed constants to here
  static constexpr uint64_t kCombinedRefCountMaskUInt32 = details::kCombinedRefCountMaskUInt32;
  static constexpr uint64_t kCombinedRefCountStrongOne = details::kCombinedRefCountStrongOne;
  static constexpr uint64_t kCombinedRefCountWeakOne = details::kCombinedRefCountWeakOne;
  static constexpr uint64_t kCombinedRefCountBothOne = details::kCombinedRefCountBothOne;
  /*! \brief increase strong reference count, the caller must already hold a strong reference */
  void IncRef() {
#ifdef _MSC_VER
    _InterlockedIncrement64(
        reinterpret_cast<volatile __int64*>(&header_.combined_ref_count));  // NOLINT(*)
#else
    __atomic_fetch_add(&(header_.combined_ref_count), 1, __ATOMIC_RELAXED);
#endif
  }
  /*!
   * \brief Try to lock the object to increase the strong reference count,
   *        the caller must already hold a strong reference.
   * \return whether the lock call is successful and object is still alive.
   */
  bool TryPromoteWeakPtr() {
#ifdef _MSC_VER
    uint64_t old_count =
        (reinterpret_cast<const volatile __int64*>(&header_.combined_ref_count))[0];  // NOLINT(*)
    while ((old_count & kCombinedRefCountMaskUInt32) != 0) {
      uint64_t new_count = old_count + kCombinedRefCountStrongOne;
      uint64_t old_count_loaded = _InterlockedCompareExchange64(
          reinterpret_cast<volatile __int64*>(&header_.combined_ref_count), new_count, old_count);
      if (old_count == old_count_loaded) {
        return true;
      }
      old_count = old_count_loaded;
    }
    return false;
#else
    uint64_t old_count = __atomic_load_n(&(header_.combined_ref_count), __ATOMIC_RELAXED);
    while ((old_count & kCombinedRefCountMaskUInt32) != 0) {
      // must do CAS to ensure that we are the only one that increases the reference count
      // avoid condition when two threads tries to promote weak to strong at same time
      // or when strong deletion happens between the load and the CAS
      uint64_t new_count = old_count + kCombinedRefCountStrongOne;
      if (__atomic_compare_exchange_n(&(header_.combined_ref_count), &old_count, new_count, true,
                                      __ATOMIC_ACQ_REL, __ATOMIC_RELAXED)) {
        return true;
      }
    }
    return false;
#endif
  }

  /*! \brief increase weak reference count */
  void IncWeakRef() {
#ifdef _MSC_VER
    _InlineInterlockedAdd64(
        reinterpret_cast<volatile __int64*>(&header_.combined_ref_count),  // NOLINT(*)
        kCombinedRefCountWeakOne);
#else
    __atomic_fetch_add(&(header_.combined_ref_count), kCombinedRefCountWeakOne, __ATOMIC_RELAXED);
#endif
  }

  /*! \brief decrease strong reference count and delete the object */
  void DecRef() {
#ifdef _MSC_VER
    // use simpler impl in windows to ensure correctness
    uint64_t count_before_sub =
        _InterlockedDecrement64(                                              //
            reinterpret_cast<volatile __int64*>(&header_.combined_ref_count)  // NOLINT(*)
            ) +
        1;
    if (count_before_sub == kCombinedRefCountBothOne) {  // NOLINT(*)
      // fast path: both reference counts will go to zero
      if (header_.deleter != nullptr) {
        // full barrrier is implicit in InterlockedDecrement
        header_.deleter(&(this->header_), kTVMFFIObjectDeleterFlagBitMaskBoth);
      }
    } else if ((count_before_sub & kCombinedRefCountMaskUInt32) == kCombinedRefCountStrongOne) {
      // strong reference count becomes zero, we need to first do strong deletion
      // then decrease weak reference count
      // full barrrier is implicit in InterlockedAdd
      if (header_.deleter != nullptr) {
        header_.deleter(&(this->header_), kTVMFFIObjectDeleterFlagBitMaskStrong);
      }
      // decrease weak reference count
      if (_InlineInterlockedAdd64(  //
              reinterpret_cast<volatile __int64*>(&header_.combined_ref_count),
              -kCombinedRefCountWeakOne) == 0) {  // NOLINT(*)
        if (header_.deleter != nullptr) {
          // full barrrier is implicit in InterlockedAdd
          header_.deleter(&(this->header_), kTVMFFIObjectDeleterFlagBitMaskWeak);
        }
      }
    }
#else
    // first do a release, note we only need to acquire for deleter
    // optimization: we only need one atomic to tell the common case
    // where both reference counts are zero
    uint64_t count_before_sub = __atomic_fetch_sub(&(header_.combined_ref_count),
                                                   kCombinedRefCountStrongOne, __ATOMIC_RELEASE);
    if (count_before_sub == kCombinedRefCountBothOne) {
      // common case, we need to delete both the object and the memory block
      // only acquire when we need to call deleter
      __atomic_thread_fence(__ATOMIC_ACQUIRE);
      if (header_.deleter != nullptr) {
        // call deleter once
        header_.deleter(&(this->header_), kTVMFFIObjectDeleterFlagBitMaskBoth);
      }
    } else if ((count_before_sub & kCombinedRefCountMaskUInt32) == kCombinedRefCountStrongOne) {
      // strong count is already zero
      // Slower path: there is still a weak reference left
      __atomic_thread_fence(__ATOMIC_ACQUIRE);
      // call destructor first, then decrease weak reference count
      if (header_.deleter != nullptr) {
        header_.deleter(&(this->header_), kTVMFFIObjectDeleterFlagBitMaskStrong);
      }
      // now decrease weak reference count
      if (__atomic_fetch_sub(&(header_.combined_ref_count), kCombinedRefCountWeakOne,
                             __ATOMIC_RELEASE) == kCombinedRefCountWeakOne) {
        __atomic_thread_fence(__ATOMIC_ACQUIRE);
        if (header_.deleter != nullptr) {
          header_.deleter(&(this->header_), kTVMFFIObjectDeleterFlagBitMaskWeak);
        }
      }
    }
#endif
  }

  /*! \brief decrease weak reference count */
  void DecWeakRef() {
#ifdef _MSC_VER
    if (_InlineInterlockedAdd64(                                               //
            reinterpret_cast<volatile __int64*>(&header_.combined_ref_count),  // NOLINT(*)
            -kCombinedRefCountWeakOne) == 0) {
      if (header_.deleter != nullptr) {
        header_.deleter(&(this->header_), kTVMFFIObjectDeleterFlagBitMaskWeak);
      }
    }
#else
    // now decrease weak reference count
    if (__atomic_fetch_sub(&(header_.combined_ref_count), kCombinedRefCountWeakOne,
                           __ATOMIC_RELEASE) == kCombinedRefCountWeakOne) {
      __atomic_thread_fence(__ATOMIC_ACQUIRE);
      if (header_.deleter != nullptr) {
        header_.deleter(&(this->header_), kTVMFFIObjectDeleterFlagBitMaskWeak);
      }
    }
#endif
  }

  // friend classes
  template <typename>
  friend class ObjectPtr;
  template <typename>
  friend class WeakObjectPtr;
  friend struct tvm::ffi::details::ObjectUnsafe;
};

/*!
 * \brief A custom smart pointer for Object.
 * \tparam T the content data type.
 * \sa make_object
 */
template <typename T>
class ObjectPtr {
 public:
  /*! \brief default constructor */
  ObjectPtr() = default;
  /*! \brief default constructor */
  ObjectPtr(std::nullptr_t) {}  // NOLINT(*)
  /*!
   * \brief copy constructor
   * \param other The value to be moved
   */
  ObjectPtr(const ObjectPtr<T>& other)  // NOLINT(*)
      : ObjectPtr(other.data_) {}
  /*!
   * \brief copy constructor
   * \param other The value to be moved
   */
  template <typename U>
  ObjectPtr(const ObjectPtr<U>& other)  // NOLINT(*)
      : ObjectPtr(other.data_) {
    static_assert(std::is_base_of_v<T, U>, "can only assign of child class ObjectPtr to parent");
  }
  /*!
   * \brief move constructor
   * \param other The value to be moved
   */
  ObjectPtr(ObjectPtr<T>&& other)  // NOLINT(*)
      : data_(other.data_) {
    other.data_ = nullptr;
  }
  /*!
   * \brief move constructor
   * \param other The value to be moved
   */
  template <typename Y>
  ObjectPtr(ObjectPtr<Y>&& other)  // NOLINT(*)
      : data_(other.data_) {
    static_assert(std::is_base_of_v<T, Y>, "can only assign of child class ObjectPtr to parent");
    other.data_ = nullptr;
  }
  /*! \brief destructor */
  ~ObjectPtr() { this->reset(); }
  /*!
   * \brief Swap this array with another Object
   * \param other The other Object
   */
  void swap(ObjectPtr<T>& other) {  // NOLINT(*)
    std::swap(data_, other.data_);
  }
  /*!
   * \return Get the content of the pointer
   */
  T* get() const { return static_cast<T*>(data_); }
  /*!
   * \return The pointer
   */
  T* operator->() const { return get(); }
  /*!
   * \return The reference
   */
  T& operator*() const {  // NOLINT(*)
    return *get();
  }
  /*!
   * \brief copy assignment
   * \param other The value to be assigned.
   * \return reference to self.
   */
  ObjectPtr<T>& operator=(const ObjectPtr<T>& other) {  // NOLINT(*)
    // takes in plane operator to enable copy elison.
    // copy-and-swap idiom
    ObjectPtr(other).swap(*this);  // NOLINT(*)
    return *this;
  }
  /*!
   * \brief move assignment
   * \param other The value to be assigned.
   * \return reference to self.
   */
  ObjectPtr<T>& operator=(ObjectPtr<T>&& other) {  // NOLINT(*)
    // copy-and-swap idiom
    ObjectPtr(std::move(other)).swap(*this);  // NOLINT(*)
    return *this;
  }
  /*!
   * \brief nullptr check
   * \return result of comparison of internal pointer with nullptr.
   */
  explicit operator bool() const { return get() != nullptr; }
  /*! \brief reset the content of ptr to be nullptr */
  void reset() {
    if (data_ != nullptr) {
      data_->DecRef();
      data_ = nullptr;
    }
  }
  /*! \return The use count of the ptr, for debug purposes */
  int use_count() const { return data_ != nullptr ? data_->use_count() : 0; }
  /*! \return whether the reference is unique */
  bool unique() const { return data_ != nullptr && data_->use_count() == 1; }
  /*! \return Whether two ObjectPtr do not equal each other */
  bool operator==(const ObjectPtr<T>& other) const { return data_ == other.data_; }
  /*! \return Whether two ObjectPtr equals each other */
  bool operator!=(const ObjectPtr<T>& other) const { return data_ != other.data_; }
  /*! \return Whether the pointer is nullptr */
  bool operator==(std::nullptr_t) const { return data_ == nullptr; }
  /*! \return Whether the pointer is not nullptr */
  bool operator!=(std::nullptr_t) const { return data_ != nullptr; }

 private:
  /*! \brief internal pointer field */
  Object* data_{nullptr};
  /*!
   * \brief constructor from Object
   * \param data The data pointer
   */
  explicit ObjectPtr(Object* data) : data_(data) {
    if (data_ != nullptr) {
      data_->IncRef();
    }
  }
  // friend classes
  friend class Object;
  friend class ObjectRef;
  friend struct ObjectPtrHash;
  template <typename>
  friend class ObjectPtr;
  template <typename>
  friend class WeakObjectPtr;
  friend struct tvm::ffi::details::ObjectUnsafe;
};

/*!
 * \brief A custom smart pointer for Object.
 * \tparam T the content data type.
 * \sa make_object
 */
template <typename T>
class WeakObjectPtr {
 public:
  /*! \brief default constructor */
  WeakObjectPtr() = default;
  /*! \brief default constructor */
  WeakObjectPtr(std::nullptr_t) {}  // NOLINT(*)
  /*!
   * \brief copy constructor
   * \param other The value to be moved
   */
  WeakObjectPtr(const WeakObjectPtr<T>& other)  // NOLINT(*)
      : WeakObjectPtr(other.data_) {}

  /*!
   * \brief copy constructor
   * \param other The value to be moved
   */
  WeakObjectPtr(const ObjectPtr<T>& other)  // NOLINT(*)
      : WeakObjectPtr(other.get()) {}
  /*!
   * \brief copy constructor
   * \param other The value to be moved
   */
  template <typename U>
  WeakObjectPtr(const WeakObjectPtr<U>& other)  // NOLINT(*)
      : WeakObjectPtr(other.data_) {
    static_assert(std::is_base_of_v<T, U>, "can only assign of child class ObjectPtr to parent");
  }
  /*!
   * \brief copy constructor
   * \param other The value to be moved
   */
  template <typename U>
  WeakObjectPtr(const ObjectPtr<U>& other)  // NOLINT(*)
      : WeakObjectPtr(other.data_) {
    static_assert(std::is_base_of_v<T, U>, "can only assign of child class ObjectPtr to parent");
  }
  /*!
   * \brief move constructor
   * \param other The value to be moved
   */
  WeakObjectPtr(WeakObjectPtr<T>&& other)  // NOLINT(*)
      : data_(other.data_) {
    other.data_ = nullptr;
  }
  /*!
   * \brief move constructor
   * \param other The value to be moved
   */
  template <typename Y>
  WeakObjectPtr(WeakObjectPtr<Y>&& other)  // NOLINT(*)
      : data_(other.data_) {
    static_assert(std::is_base_of_v<T, Y>, "can only assign of child class ObjectPtr to parent");
    other.data_ = nullptr;
  }
  /*! \brief destructor */
  ~WeakObjectPtr() { this->reset(); }
  /*!
   * \brief Swap this array with another Object
   * \param other The other Object
   */
  void swap(WeakObjectPtr<T>& other) {  // NOLINT(*)
    std::swap(data_, other.data_);
  }

  /*!
   * \brief copy assignment
   * \param other The value to be assigned.
   * \return reference to self.
   */
  WeakObjectPtr<T>& operator=(const WeakObjectPtr<T>& other) {  // NOLINT(*)
    // takes in plane operator to enable copy elison.
    // copy-and-swap idiom
    WeakObjectPtr(other).swap(*this);  // NOLINT(*)
    return *this;
  }
  /*!
   * \brief move assignment
   * \param other The value to be assigned.
   * \return reference to self.
   */
  WeakObjectPtr<T>& operator=(WeakObjectPtr<T>&& other) {  // NOLINT(*)
    // copy-and-swap idiom
    WeakObjectPtr(std::move(other)).swap(*this);  // NOLINT(*)
    return *this;
  }

  /*! \return The internal object pointer if the object is still alive, otherwise nullptr */
  ObjectPtr<T> lock() const {
    if (data_ != nullptr && data_->TryPromoteWeakPtr()) {
      ObjectPtr<T> ret;
      // we already increase the reference count, so we don't need to do it again
      ret.data_ = data_;
      return ret;
    }
    return nullptr;
  }

  /*! \brief reset the content of ptr to be nullptr */
  void reset() {
    if (data_ != nullptr) {
      data_->DecWeakRef();
      data_ = nullptr;
    }
  }

  /*! \return The use count of the ptr, for debug purposes */
  int use_count() const { return data_ != nullptr ? data_->use_count() : 0; }

  /*! \return whether the pointer is nullptr */
  bool expired() const { return data_ == nullptr || data_->use_count() == 0; }

 private:
  /*! \brief internal pointer field */
  Object* data_{nullptr};

  /*!
   * \brief constructor from Object
   * \param data The data pointer
   */
  explicit WeakObjectPtr(Object* data) : data_(data) {
    if (data_ != nullptr) {
      data_->IncWeakRef();
    }
  }

  template <typename>
  friend class WeakObjectPtr;
  friend struct tvm::ffi::details::ObjectUnsafe;
};

/*!
 * \brief Optional data type in FFI.
 * \tparam T The underlying type of the optional.
 *
 * \note Compared to std::optional, Optional<ObjectRef>
 *   akes less storage as it used nullptr to represent nullopt.
 */
template <typename T, typename = void>
class Optional;

/*! \brief Base class of all object reference */
class ObjectRef {
 public:
  /*! \brief default constructor */
  ObjectRef() = default;
  /*! \brief copy constructor */
  ObjectRef(const ObjectRef& other) = default;
  /*! \brief move constructor */
  ObjectRef(ObjectRef&& other) noexcept : data_(std::move(other.data_)) { other.data_ = nullptr; }
  /*! \brief copy assignment */
  ObjectRef& operator=(const ObjectRef& other) = default;
  /*! \brief move assignment */
  ObjectRef& operator=(ObjectRef&& other) noexcept {
    data_ = std::move(other.data_);
    other.data_ = nullptr;
    return *this;
  }
  /*! \brief Constructor from existing object ptr */
  explicit ObjectRef(ObjectPtr<Object> data) : data_(std::move(data)) {}
  /*! \brief Constructor from UnsafeInit */
  explicit ObjectRef(UnsafeInit) : data_(nullptr) {}
  /*!
   * \brief Comparator
   * \param other Another object ref.
   * \return the compare result.
   */
  bool same_as(const ObjectRef& other) const { return data_ == other.data_; }
  /*!
   * \brief Comparator
   * \param other Another object ref.
   * \return the compare result.
   */
  bool operator==(const ObjectRef& other) const { return data_ == other.data_; }
  /*!
   * \brief Comparator
   * \param other Another object ref.
   * \return the compare result.
   */
  bool operator!=(const ObjectRef& other) const { return data_ != other.data_; }
  /*!
   * \brief Comparator
   * \param other Another object ref by address.
   * \return the compare result.
   */
  bool operator<(const ObjectRef& other) const { return data_.get() < other.data_.get(); }
  /*!
   * \return whether the object is defined.
   */
  bool defined() const { return data_ != nullptr; }
  /*! \return the internal object pointer */
  const Object* get() const { return data_.get(); }
  /*! \return the internal object pointer */
  const Object* operator->() const { return get(); }
  /*! \return whether the reference is unique */
  bool unique() const { return data_.unique(); }
  /*! \return The use count of the ptr, for debug purposes */
  int use_count() const { return data_.use_count(); }

  /*!
   * \brief Try to downcast the internal Object to a
   *  raw pointer of a corresponding type.
   *
   *  The function will return a nullptr if the cast failed.
   *
   *      if (const AddNode *ptr = node_ref.as<AddNode>()) {
   *        // This is an add node
   *      }
   *
   * \tparam ObjectType the target type, must be a subtype of Object
   * \return The pointer to the requested type.
   */
  template <typename ObjectType, typename = std::enable_if_t<std::is_base_of_v<Object, ObjectType>>>
  const ObjectType* as() const& {
    if (data_ != nullptr && data_->IsInstance<ObjectType>()) {
      return static_cast<ObjectType*>(data_.get());
    } else {
      return nullptr;
    }
  }

  /*!
   * \brief Try to downcast the ObjectRef to Optional<T> of the requested type.
   *
   * If the cast fails, returns std::nullopt. If this ObjectRef is null, returns
   * a null ref for nullable target refs and std::nullopt for non-nullable targets.
   *
   * \tparam ObjectRefType the target type, must be a subtype of ObjectRef.
   * \return The optional value of the requested type.
   */
  template <typename ObjectRefType,
            typename = std::enable_if_t<std::is_base_of_v<ObjectRef, ObjectRefType>>>
  TVM_FFI_INLINE std::optional<ObjectRefType> as() const& {
    if (data_ != nullptr) {
      // Piggy back to Any TypeTraits for rich ObjectRef check, temp any_data will optimize away.
      TVMFFIAny any_data;
      any_data.type_index = data_->type_index();
      TVM_FFI_UNSAFE_ASSUME(any_data.type_index >= TypeIndex::kTVMFFIStaticObjectBegin);
      any_data.zero_padding = 0;
      TVM_FFI_CLEAR_PTR_PADDING_IN_FFI_ANY(&any_data);
      any_data.v_obj = reinterpret_cast<TVMFFIObject*>(const_cast<Object*>(data_.get()));
      if (TypeTraits<ObjectRefType>::CheckAnyStrict(&any_data)) {
        ObjectRefType result(UnsafeInit{});
        result.data_ = data_;
        return result;
      }
      return std::nullopt;
    }
    if constexpr (ObjectRefType::_type_is_nullable) {
      return ObjectRefType(UnsafeInit{});
    }
    return std::nullopt;
  }

  /*!
   * \brief Try to move-downcast the ObjectRef to Optional<T> of the requested type.
   *
   * If the cast succeeds, moves the internal object pointer to the returned
   * ObjectRefType. If the cast fails, returns std::nullopt. If this ObjectRef
   * is null, returns a null ref for nullable target refs and std::nullopt for
   * non-nullable targets.
   *
   * \tparam ObjectRefType the target type, must be a subtype of ObjectRef.
   * \return The optional value of the requested type.
   */
  template <typename ObjectRefType,
            typename = std::enable_if_t<std::is_base_of_v<ObjectRef, ObjectRefType>>>
  TVM_FFI_INLINE std::optional<ObjectRefType> as() && {
    if (data_ != nullptr) {
      // Piggy back to Any TypeTraits for rich ObjectRef check, temp any_data will optimize away.
      TVMFFIAny any_data;
      any_data.type_index = data_->type_index();
      TVM_FFI_UNSAFE_ASSUME(any_data.type_index >= TypeIndex::kTVMFFIStaticObjectBegin);
      any_data.zero_padding = 0;
      TVM_FFI_CLEAR_PTR_PADDING_IN_FFI_ANY(&any_data);
      any_data.v_obj = reinterpret_cast<TVMFFIObject*>(const_cast<Object*>(data_.get()));
      if (TypeTraits<ObjectRefType>::CheckAnyStrict(&any_data)) {
        ObjectRefType result(UnsafeInit{});
        result.data_ = std::move(data_);
        data_ = nullptr;
        return result;
      }
      return std::nullopt;
    }
    if constexpr (ObjectRefType::_type_is_nullable) {
      return ObjectRefType(UnsafeInit{});
    }
    return std::nullopt;
  }

  /*!
   * \brief Downcast the ObjectRef to the requested type or throw.
   *
   * \tparam ObjectRefType the target type, must be a subtype of ObjectRef
   * \return The requested value.
   */
  template <typename ObjectRefType,
            typename = std::enable_if_t<std::is_base_of_v<ObjectRef, ObjectRefType>>>
  TVM_FFI_INLINE ObjectRefType as_or_throw() const&;

  /*!
   * \brief Move-downcast the ObjectRef to the requested type or throw.
   *
   * If the cast succeeds, moves the internal object pointer to the returned
   * ObjectRefType.
   *
   * \tparam ObjectRefType the target type, must be a subtype of ObjectRef
   * \return The requested value.
   */
  template <typename ObjectRefType,
            typename = std::enable_if_t<std::is_base_of_v<ObjectRef, ObjectRefType>>>
  TVM_FFI_INLINE ObjectRefType as_or_throw() &&;

  /*!
   * \brief Get the type index of the ObjectRef
   * \return The type index of the ObjectRef
   */
  int32_t type_index() const {
    return data_ != nullptr ? data_->type_index() : TypeIndex::kTVMFFINone;
  }

  /*!
   * \brief Get the type key of the ObjectRef
   * \return The type key of the ObjectRef
   */
  std::string GetTypeKey() const {
    return data_ != nullptr ? data_->GetTypeKey() : StaticTypeKey::kTVMFFINone;
  }

  /*! \brief type indicate the container type. */
  using ContainerType = Object;
  /*! \brief Whether ContainerType exactly describes the objects this ref can hold. */
  static constexpr bool _type_container_is_exact = true;
  /*! \brief Whether the reference can point to nullptr */
  static constexpr bool _type_is_nullable = true;

 protected:
  /*! \brief Internal pointer that backs the reference. */
  ObjectPtr<Object> data_;
  /*! \return return a mutable internal ptr, can be used by sub-classes. */
  Object* get_mutable() const { return data_.get(); }
  // friend classes.
  friend struct ObjectPtrHash;
  friend struct tvm::ffi::details::ObjectUnsafe;
};

/*!
 * \brief Whether RefType contains every ObjectType instance.
 *
 * The containing reference type is first and the contained object type is
 * second. The template is enabled only when RefType is an object reference and
 * ObjectType is an object node. The default is true exactly when RefType has an
 * exact container type and ObjectType derives from it. Direct specializations
 * can provide the proof for non-exact reference types.
 *
 * \tparam RefType The object reference type.
 * \tparam ObjectType The object node type.
 */
template <typename RefType, typename ObjectType,
          typename = std::enable_if_t<std::is_base_of_v<ObjectRef, RefType> &&
                                      std::is_base_of_v<Object, ObjectType>>>
inline constexpr bool object_ref_contains_v =
    RefType::_type_container_is_exact &&
    std::is_base_of_v<typename RefType::ContainerType, ObjectType>;

// forward delcare variant
template <typename... V>
class Variant;

/*! \brief ObjectRef hash functor */
struct ObjectPtrHash {
  size_t operator()(const ObjectRef& a) const { return operator()(a.data_); }

  template <typename T>
  size_t operator()(const ObjectPtr<T>& a) const {
    return std::hash<Object*>()(a.get());
  }

  template <typename... V>
  TVM_FFI_INLINE size_t operator()(const Variant<V...>& a) const;
};

/*! \brief ObjectRef equal functor */
struct ObjectPtrEqual {
  bool operator()(const ObjectRef& a, const ObjectRef& b) const { return a.same_as(b); }

  template <typename T>
  bool operator()(const ObjectPtr<T>& a, const ObjectPtr<T>& b) const {
    return a == b;
  }

  template <typename... V>
  TVM_FFI_INLINE bool operator()(const Variant<V...>& a, const Variant<V...>& b) const;
};

/*!
 * \brief Helper macro to declare object information with static type index.
 *
 * For each custom object, you need to call tvm::ffi::reflection::ObjectDef<TypeName>()
 * once in your cc file to register the type index with the runtime.
 * Alternatively, you can call TypeName::_GetOrAllocRuntimeTypeIndex() once.
 *
 * \param TypeKey The type key of the current type.
 * \param TypeName The name of the current type.
 * \param ParentType The name of the ParentType
 *
 * \see tvm::ffi::reflection::ObjectDef
 */
#define TVM_FFI_DECLARE_OBJECT_INFO_STATIC(TypeKey, TypeName, ParentType)                     \
  static constexpr int32_t _type_depth = ParentType::_type_depth + 1;                         \
  TVM_FFI_COLD_CODE static int32_t _GetOrAllocRuntimeTypeIndex() {                            \
    static_assert(!ParentType::_type_final, "ParentType marked as final");                    \
    static_assert(TypeName::_type_child_slots == 0 || ParentType::_type_child_slots == 0 ||   \
                      TypeName::_type_child_slots < ParentType::_type_child_slots,            \
                  "Need to set _type_child_slots when parent specifies it.");                 \
    TVMFFIByteArray type_key{TypeName::_type_key,                                             \
                             std::char_traits<char>::length(TypeName::_type_key)};            \
    static int32_t tindex [[maybe_unused]] = TVMFFITypeGetOrAllocIndex(                       \
        &type_key, TypeName::_type_index, TypeName::_type_depth, TypeName::_type_child_slots, \
        TypeName::_type_child_slots_can_overflow, ParentType::_GetOrAllocRuntimeTypeIndex()); \
    return TypeName::_type_index;                                                             \
  }                                                                                           \
  TVM_FFI_INLINE static int32_t RuntimeTypeIndex() noexcept { return TypeName::_type_index; } \
  static constexpr const char* _type_key = TypeKey

/*!
 * \brief Helper macro to declare object information with type key already defined in class.
 *
 * \param TypeName The name of the current type.
 * \param ParentType The name of the ParentType
 */
#define TVM_FFI_DECLARE_OBJECT_INFO_PREDEFINED_TYPE_KEY(TypeName, ParentType)                 \
  static constexpr int32_t _type_depth = ParentType::_type_depth + 1;                         \
  TVM_FFI_COLD_CODE static int32_t _GetOrAllocRuntimeTypeIndex() {                            \
    static_assert(!ParentType::_type_final, "ParentType marked as final");                    \
    static_assert(TypeName::_type_child_slots == 0 || ParentType::_type_child_slots == 0 ||   \
                      TypeName::_type_child_slots < ParentType::_type_child_slots,            \
                  "Need to set _type_child_slots when parent specifies it.");                 \
    TVMFFIByteArray type_key{TypeName::_type_key,                                             \
                             std::char_traits<char>::length(TypeName::_type_key)};            \
    static int32_t tindex = TVMFFITypeGetOrAllocIndex(                                        \
        &type_key, -1, TypeName::_type_depth, TypeName::_type_child_slots,                    \
        TypeName::_type_child_slots_can_overflow, ParentType::_GetOrAllocRuntimeTypeIndex()); \
    return tindex;                                                                            \
  }                                                                                           \
  static inline const int32_t _type_index = _GetOrAllocRuntimeTypeIndex();                    \
  TVM_FFI_INLINE static int32_t RuntimeTypeIndex() noexcept { return TypeName::_type_index; }

/*!
 * \brief Helper macro to declare object information with dynamic type index.
 *
 * For each custom object, you need to call tvm::ffi::reflection::ObjectDef<TypeName>()
 * once in your cc file to register the type index with the runtime.
 * Alternatively, you can call TypeName::_GetOrAllocRuntimeTypeIndex() once.
 *
 * \param TypeKey The type key of the current type.
 * \param TypeName The name of the current type.
 * \param ParentType The name of the ParentType
 * \sa tvm::ffi::reflection::ObjectDef
 */
#define TVM_FFI_DECLARE_OBJECT_INFO(TypeKey, TypeName, ParentType) \
  static constexpr const char* _type_key = TypeKey;                \
  TVM_FFI_DECLARE_OBJECT_INFO_PREDEFINED_TYPE_KEY(TypeName, ParentType)

/*!
 * \brief Helper macro to declare object information with dynamic type index and is final.
 *
 * For each custom object, you need to call tvm::ffi::reflection::ObjectDef<TypeName>()
 * once in your cc file to register the type index with the runtime.
 * Alternatively, you can call TypeName::_GetOrAllocRuntimeTypeIndex() once.
 *
 * \param TypeKey The type key of the current type.
 * \param TypeName The name of the current type.
 * \param ParentType The name of the ParentType
 * \sa tvm::ffi::reflection::ObjectDef
 */
#define TVM_FFI_DECLARE_OBJECT_INFO_FINAL(TypeKey, TypeName, ParentType) \
  static const constexpr int _type_child_slots [[maybe_unused]] = 0;     \
  static const constexpr bool _type_final [[maybe_unused]] = true;       \
  TVM_FFI_DECLARE_OBJECT_INFO(TypeKey, TypeName, ParentType)

/*!
 * \brief Define object reference methods.
 *
 * \param TypeName The object type name
 * \param ParentType The parent type of the objectref
 * \param ObjectName The type name of the object.
 *
 * \note This macro also defines the default constructor that puts the ObjectRef
 *       in undefined state initially.
 */
#define TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TypeName, ParentType, ObjectName)         \
  TypeName() = default;                                                                      \
  explicit TypeName(::tvm::ffi::ObjectPtr<ObjectName> n) : ParentType(std::move(n)) {}       \
  explicit TypeName(::tvm::ffi::UnsafeInit tag) : ParentType(tag) {}                         \
  TVM_FFI_DEFINE_DEFAULT_COPY_MOVE_AND_ASSIGN(TypeName)                                      \
  using __PtrType = std::conditional_t<(ObjectName::_type_mutable),                          \
                                       ObjectName*, /* NOLINT(bugprone-macro-parentheses) */ \
                                       const ObjectName*>;                                   \
  __PtrType operator->() const { return static_cast<__PtrType>(data_.get()); }               \
  __PtrType get() const { return static_cast<__PtrType>(data_.get()); }                      \
  [[maybe_unused]] static constexpr bool _type_is_nullable = true;                           \
  using ContainerType = ObjectName

/*!
 * \brief Define object reference methods do not have undefined state.
 *
 * \param TypeName The object type name
 * \param ParentType The parent type of the objectref
 * \param ObjectName The type name of the object.
 */
#define TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(TypeName, ParentType, ObjectName)      \
  explicit TypeName(::tvm::ffi::UnsafeInit tag) : ParentType(tag) {}                         \
  TVM_FFI_DEFINE_DEFAULT_COPY_MOVE_AND_ASSIGN(TypeName)                                      \
  using __PtrType = std::conditional_t<(ObjectName::_type_mutable),                          \
                                       ObjectName*, /* NOLINT(bugprone-macro-parentheses) */ \
                                       const ObjectName*>;                                   \
  __PtrType operator->() const { return static_cast<__PtrType>(data_.get()); }               \
  __PtrType get() const { return static_cast<__PtrType>(data_.get()); }                      \
  [[maybe_unused]] static constexpr bool _type_is_nullable = false;                          \
  using ContainerType = ObjectName

namespace details {

template <typename TargetType>
TVM_FFI_INLINE bool IsObjectInstance(int32_t object_type_index) {
  static_assert(std::is_base_of_v<Object, TargetType>);
  // Everything is a subclass of object.
  if constexpr (std::is_same_v<TargetType, Object>) {
    return true;
  } else if constexpr (TargetType::_type_final) {
    // if the target type is a final type
    // then we only need to check the equivalence.
    return object_type_index == TargetType::RuntimeTypeIndex();
  } else {
    // Explicitly enclose in else to eliminate this branch early in compilation.
    // if target type is a non-leaf type
    // Check if type index falls into the range of reserved slots.
    int32_t target_type_index = TargetType::RuntimeTypeIndex();
    int32_t begin = target_type_index;
    // The condition will be optimized by constant-folding.
    if constexpr (TargetType::_type_child_slots != 0) {
      // total_slots = child_slots + 1 (including self)
      int32_t end = begin + TargetType::_type_child_slots + 1;
      if (object_type_index >= begin && object_type_index < end) return true;
    } else {
      if (object_type_index == begin) return true;
    }
    if constexpr (TargetType::_type_child_slots_can_overflow) {
      // Invariance: parent index is always smaller than the child.
      if (object_type_index < target_type_index) return false;
      // Do a runtime lookup of type information
      // the function checks that the info exists
      const TypeInfo* type_info = TVMFFIGetTypeInfo(object_type_index);
      return (type_info->type_depth > TargetType::_type_depth &&
              type_info->type_ancestors[TargetType::_type_depth]->type_index == target_type_index);
    } else {
      return false;
    }
  }
}

/*!
 * \brief Return whether a runtime type index matches a target type index.
 *
 * \param actual_type_index Runtime type index of the value being checked.
 * \param target_type_index Target type index to match against.
 * \return Whether \p actual_type_index is compatible with \p target_type_index.
 */
TVM_FFI_INLINE bool RuntimeTypeIndexMatch(int32_t actual_type_index, int32_t target_type_index) {
  if (actual_type_index == target_type_index) {
    return true;
  }
  // Any target matches all runtime values.
  if (target_type_index == TypeIndex::kTVMFFIAny) {
    return true;
  }
  // str/bytes targets also match their small inline variants.
  if (target_type_index == TypeIndex::kTVMFFIStr) {
    return actual_type_index == TypeIndex::kTVMFFISmallStr;
  }
  if (target_type_index == TypeIndex::kTVMFFIBytes) {
    return actual_type_index == TypeIndex::kTVMFFISmallBytes;
  }
  // Everything is a subclass of object.
  if (target_type_index == TypeIndex::kTVMFFIObject) {
    return actual_type_index >= TypeIndex::kTVMFFIStaticObjectBegin;
  }
  // Non-object type indices can only match through exact equality handled above.
  if (actual_type_index < TypeIndex::kTVMFFIStaticObjectBegin ||
      target_type_index < TypeIndex::kTVMFFIStaticObjectBegin) {
    return false;
  }
  // Invariance: parent index is always smaller than the child.
  if (actual_type_index < target_type_index) {
    return false;
  }
  // Fall back to runtime ancestry metadata.
  const TypeInfo* actual_type_info = TVMFFIGetTypeInfo(actual_type_index);
  const TypeInfo* target_type_info = TVMFFIGetTypeInfo(target_type_index);
  return actual_type_info->type_depth > target_type_info->type_depth &&
         actual_type_info->type_ancestors[target_type_info->type_depth]->type_index ==
             target_type_index;
}

/*!
 * \brief Namespace to internally manipulate object class.
 * \note These functions are only supposed to be used by internal
 * implementations and not external users of the tvm::ffi
 */
struct ObjectUnsafe {
  // NOTE: get ffi header from an object
  TVM_FFI_INLINE static TVMFFIObject* GetHeader(const Object* src) {
    return const_cast<TVMFFIObject*>(&(src->header_));
  }

  /*!
   * \brief Recover the TVMFFIObjectAllocHeader for a TVMFFIObject pointer.
   * \param ptr The pointer to the space of the object.
   * \return The header set by the allocator that produced ``ptr``.
   */
  TVM_FFI_INLINE static TVMFFIObjectAllocHeader* GetObjectAllocHeaderFromPtr(void* ptr) {
    return reinterpret_cast<TVMFFIObjectAllocHeader*>(static_cast<char*>(ptr) -
                                                      sizeof(TVMFFIObjectAllocHeader));
  }

// Suppress -Winvalid-offsetof: we intentionally use offsetof on non-standard-layout types
// to avoid undefined behavior from null pointer arithmetic that sanitizers flag.
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winvalid-offsetof"
#endif
  template <typename Class>
  TVM_FFI_INLINE static int64_t GetObjectOffsetToSubclass() {
    return static_cast<int64_t>(__builtin_offsetof(Class, header_)) -
           static_cast<int64_t>(__builtin_offsetof(Object, header_));
  }
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

  template <typename T>
  TVM_FFI_INLINE static T ObjectRefFromObjectPtr(const ObjectPtr<Object>& ptr) {
    T ref(UnsafeInit{});
    ref.data_ = ptr;
    return ref;
  }

  template <typename T>
  TVM_FFI_INLINE static T ObjectRefFromObjectPtr(ObjectPtr<Object>&& ptr) {
    T ref(UnsafeInit{});
    ref.data_ = std::move(ptr);
    return ref;
  }

  template <typename T>
  TVM_FFI_INLINE static ObjectPtr<T> ObjectPtrFromObjectRef(const ObjectRef& ref) {
    if constexpr (std::is_same_v<T, Object>) {
      return ref.data_;
    } else {
      return tvm::ffi::ObjectPtr<T>(ref.data_.data_);
    }
  }

  template <typename T>
  TVM_FFI_INLINE static ObjectPtr<T> ObjectPtrFromObjectRef(ObjectRef&& ref) {
    if constexpr (std::is_same_v<T, Object>) {
      return std::move(ref.data_);
    } else {
      ObjectPtr<T> result;
      result.data_ = std::move(ref.data_.data_);
      ref.data_.data_ = nullptr;
      return result;
    }
  }

  template <typename T>
  TVM_FFI_INLINE static ObjectPtr<T> ObjectPtrFromOwned(Object* raw_ptr) {
    tvm::ffi::ObjectPtr<T> ptr;
    ptr.data_ = raw_ptr;
    return ptr;
  }

  template <typename T>
  TVM_FFI_INLINE static ObjectPtr<T> ObjectPtrFromOwned(TVMFFIObject* obj_ptr) {
    return ObjectPtrFromOwned<T>(reinterpret_cast<Object*>(obj_ptr));
  }

  template <typename T>
  TVM_FFI_INLINE static T* RawObjectPtrFromUnowned(TVMFFIObject* obj_ptr) {
    // NOTE: this is important to first cast to Object*
    // then cast back to T* because objptr and tptr may not be the same
    // depending on how sub-class allocates the space.
    return static_cast<T*>(reinterpret_cast<Object*>(obj_ptr));
  }

  // Create ObjectPtr from unowned ptr
  template <typename T>
  TVM_FFI_INLINE static ObjectPtr<T> ObjectPtrFromUnowned(Object* raw_ptr) {
    return tvm::ffi::ObjectPtr<T>(raw_ptr);
  }

  template <typename T>
  TVM_FFI_INLINE static ObjectPtr<T> ObjectPtrFromUnowned(TVMFFIObject* obj_ptr) {
    return tvm::ffi::ObjectPtr<T>(reinterpret_cast<Object*>(obj_ptr));
  }

  TVM_FFI_INLINE static void DecRefObjectHandle(TVMFFIObjectHandle handle) {
    if (handle) reinterpret_cast<Object*>(handle)->DecRef();
  }

  TVM_FFI_INLINE static void IncRefObjectHandle(TVMFFIObjectHandle handle) {
    reinterpret_cast<Object*>(handle)->IncRef();
  }

  TVM_FFI_INLINE static Object* RawObjectPtrFromObjectRef(const ObjectRef& src) {
    return src.data_.data_;
  }

  TVM_FFI_INLINE static TVMFFIObject* TVMFFIObjectPtrFromObjectRef(const ObjectRef& src) {
    return GetHeader(src.data_.data_);
  }

  template <typename T>
  TVM_FFI_INLINE static TVMFFIObject* TVMFFIObjectPtrFromObjectPtr(const ObjectPtr<T>& src) {
    return GetHeader(src.data_);
  }

  template <typename T>
  TVM_FFI_INLINE static TVMFFIObject* MoveObjectPtrToTVMFFIObjectPtr(ObjectPtr<T>&& src) {
    Object* obj_ptr = src.data_;
    src.data_ = nullptr;
    return GetHeader(obj_ptr);
  }

  TVM_FFI_INLINE static TVMFFIObject* MoveObjectRefToTVMFFIObjectPtr(ObjectRef&& src) {
    Object* obj_ptr = src.data_.data_;
    src.data_.data_ = nullptr;
    return GetHeader(obj_ptr);
  }
};

}  // namespace details

template <typename T>
struct TypeToRuntimeTypeIndex<T, std::enable_if_t<std::is_base_of_v<ObjectRef, T>>> {
  static int32_t v() { return T::ContainerType::RuntimeTypeIndex(); }
};

template <typename TObjRef>
struct ObjectRefTypeTraitsBase : public TypeTraitsBase {
  static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIObject;
  using ContainerType = typename TObjRef::ContainerType;

  TVM_FFI_INLINE static void CopyToAnyView(const TObjRef& src, TVMFFIAny* result) {
    if constexpr (TObjRef::_type_is_nullable) {
      if (!src.defined()) {
        TypeTraits<std::nullptr_t>::CopyToAnyView(nullptr, result);
        return;
      }
    }
    TVMFFIObject* obj_ptr = details::ObjectUnsafe::TVMFFIObjectPtrFromObjectRef(src);
    result->type_index = obj_ptr->type_index;
    result->zero_padding = 0;
    TVM_FFI_CLEAR_PTR_PADDING_IN_FFI_ANY(result);
    result->v_obj = obj_ptr;
  }

  TVM_FFI_INLINE static void MoveToAny(TObjRef src, TVMFFIAny* result) {
    if constexpr (TObjRef::_type_is_nullable) {
      if (!src.defined()) {
        TypeTraits<std::nullptr_t>::CopyToAnyView(nullptr, result);
        return;
      }
    }
    TVMFFIObject* obj_ptr = details::ObjectUnsafe::MoveObjectRefToTVMFFIObjectPtr(std::move(src));
    result->type_index = obj_ptr->type_index;
    result->zero_padding = 0;
    TVM_FFI_CLEAR_PTR_PADDING_IN_FFI_ANY(result);
    result->v_obj = obj_ptr;
  }

  TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
    if constexpr (TObjRef::_type_is_nullable) {
      if (src->type_index == TypeIndex::kTVMFFINone) return true;
    }
    return src->type_index >= TypeIndex::kTVMFFIStaticObjectBegin &&
           details::IsObjectInstance<ContainerType>(src->type_index);
  }

  TVM_FFI_INLINE static TObjRef CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
    if constexpr (TObjRef::_type_is_nullable) {
      if (src->type_index == TypeIndex::kTVMFFINone) {
        return details::ObjectUnsafe::ObjectRefFromObjectPtr<TObjRef>(nullptr);
      }
    }
    return details::ObjectUnsafe::ObjectRefFromObjectPtr<TObjRef>(
        details::ObjectUnsafe::ObjectPtrFromUnowned<Object>(src->v_obj));
  }

  TVM_FFI_INLINE static TObjRef MoveFromAnyAfterCheck(TVMFFIAny* src) {
    if constexpr (TObjRef::_type_is_nullable) {
      if (src->type_index == TypeIndex::kTVMFFINone) {
        return details::ObjectUnsafe::ObjectRefFromObjectPtr<TObjRef>(nullptr);
      }
    }
    ObjectPtr<ContainerType> obj_ptr =
        details::ObjectUnsafe::ObjectPtrFromOwned<ContainerType>(src->v_obj);
    TypeTraits<std::nullptr_t>::MoveToAny(nullptr, src);
    return details::ObjectUnsafe::ObjectRefFromObjectPtr<TObjRef>(std::move(obj_ptr));
  }

  TVM_FFI_INLINE static std::optional<TObjRef> TryCastFromAnyView(const TVMFFIAny* src) {
    if constexpr (TObjRef::_type_is_nullable) {
      if (src->type_index == TypeIndex::kTVMFFINone) {
        return details::ObjectUnsafe::ObjectRefFromObjectPtr<TObjRef>(nullptr);
      }
    }
    if (src->type_index >= TypeIndex::kTVMFFIStaticObjectBegin &&
        details::IsObjectInstance<ContainerType>(src->type_index)) {
      return details::ObjectUnsafe::ObjectRefFromObjectPtr<TObjRef>(
          details::ObjectUnsafe::ObjectPtrFromUnowned<ContainerType>(src->v_obj));
    }
    return std::nullopt;
  }

  TVM_FFI_INLINE static std::string TypeStr() { return ContainerType::_type_key; }
  TVM_FFI_INLINE static std::string TypeSchema() {
    return R"({"type":")" + std::string(ContainerType::_type_key) + R"("})";
  }
};

template <typename TObjRef>
struct TypeTraits<TObjRef, std::enable_if_t<std::is_base_of_v<ObjectRef, TObjRef> &&
                                            use_default_type_traits_v<TObjRef>>>
    : public ObjectRefTypeTraitsBase<TObjRef> {};

/*!
 * \brief Helper class to define ObjectRef that can be auto-converted from a
 *        fallback type, the Traits<ObjectRefType> must be derived from it
 *        and define a static methods named ConvertFallbackValue for each
 *        FallbackType
 *
 *        The conversion will go through the FallbackTypes in the order
 *        specified in the template parameter.
 * \tparam ObjectRefType The type of the ObjectRef.
 * \tparam FallbackTypes The type of the fallback value.
 */
template <typename ObjectRefType, typename... FallbackTypes>
struct ObjectRefWithFallbackTraitsBase : public ObjectRefTypeTraitsBase<ObjectRefType> {
  /// \cond Doxygen_Suppress
  TVM_FFI_INLINE static std::optional<ObjectRefType> TryCastFromAnyView(const TVMFFIAny* src) {
    if (auto opt_obj = ObjectRefTypeTraitsBase<ObjectRefType>::TryCastFromAnyView(src)) {
      return opt_obj;
    }
    return TryFallbackTypes<FallbackTypes...>(src);
  }

  template <typename FallbackType, typename... Rest>
  TVM_FFI_INLINE static std::optional<ObjectRefType> TryFallbackTypes(const TVMFFIAny* src) {
    static_assert(!std::is_same_v<bool, FallbackType>,
                  "Using bool as FallbackType can cause bug because int will be detected as bool, "
                  "use tvm::ffi::StrictBool instead");
    if (auto opt_fallback = TypeTraits<FallbackType>::TryCastFromAnyView(src)) {
      return TypeTraits<ObjectRefType>::ConvertFallbackValue(*std::move(opt_fallback));
    }
    if constexpr (sizeof...(Rest) > 0) {
      return TryFallbackTypes<Rest...>(src);
    }
    return std::nullopt;
  }
  /// \endcond
};

template <typename TObject>
struct TypeTraits<TObject*, std::enable_if_t<std::is_base_of_v<Object, TObject>>>
    : public TypeTraitsBase {
  TVM_FFI_INLINE static void CopyToAnyView(TObject* src, TVMFFIAny* result) {
    TVMFFIObject* obj_ptr = details::ObjectUnsafe::GetHeader(src);
    result->type_index = obj_ptr->type_index;
    result->zero_padding = 0;
    TVM_FFI_CLEAR_PTR_PADDING_IN_FFI_ANY(result);
    result->v_obj = obj_ptr;
  }

  TVM_FFI_INLINE static void MoveToAny(TObject* src, TVMFFIAny* result) {
    TVMFFIObject* obj_ptr = details::ObjectUnsafe::GetHeader(src);
    result->type_index = obj_ptr->type_index;
    result->zero_padding = 0;
    TVM_FFI_CLEAR_PTR_PADDING_IN_FFI_ANY(result);
    result->v_obj = obj_ptr;
    details::ObjectUnsafe::IncRefObjectHandle(result->v_obj);
  }

  TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
    return src->type_index >= TypeIndex::kTVMFFIStaticObjectBegin &&
           details::IsObjectInstance<TObject>(src->type_index);
  }

  TVM_FFI_INLINE static TObject* CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
    if constexpr (!std::is_const_v<TObject>) {
      static_assert(TObject::_type_mutable, "TObject must be mutable to enable cast from Any");
    }
    return details::ObjectUnsafe::RawObjectPtrFromUnowned<TObject>(src->v_obj);
  }

  TVM_FFI_INLINE static std::optional<TObject*> TryCastFromAnyView(const TVMFFIAny* src) {
    if constexpr (!std::is_const_v<TObject>) {
      static_assert(TObject::_type_mutable, "TObject must be mutable to enable cast from Any");
    }
    if (CheckAnyStrict(src)) return CopyFromAnyViewAfterCheck(src);
    return std::nullopt;
  }

  TVM_FFI_INLINE static std::string TypeStr() { return TObject::_type_key; }
  TVM_FFI_INLINE static std::string TypeSchema() {
    return R"({"type":")" + std::string(TObject::_type_key) + R"("})";
  }
};

}  // namespace ffi
}  // namespace tvm

#endif  // TVM_FFI_OBJECT_H_
