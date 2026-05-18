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
 * \file tvm/ffi/extra/structural_visit.h
 * \brief Structural visit implementation
 */
#ifndef TVM_FFI_EXTRA_STRUCTURAL_VISIT_H_
#define TVM_FFI_EXTRA_STRUCTURAL_VISIT_H_

#include <tvm/ffi/any.h>
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/container/variant.h>
#include <tvm/ffi/extra/base.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/optional.h>
#include <tvm/ffi/reflection/access_path.h>

#include <type_traits>
#include <utility>

namespace tvm {
namespace ffi {

/*!
 * \brief Object node carrying the optional payload for an interrupted structural visit.
 */
class VisitInterruptObj : public Object {
 public:
  /*! \brief Payload returned with the interrupt, or FFI None for no payload. */
  Any value;

  VisitInterruptObj() = default;
  /*!
   * \brief Construct a VisitInterruptObj with a payload.
   * \param value The payload carried by the interrupt.
   */
  explicit VisitInterruptObj(Any value) : value(std::move(value)) {}

  /// \cond Doxygen_Suppress
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ffi.VisitInterrupt", VisitInterruptObj, Object);
  /// \endcond
};

/*!
 * \brief ObjectRef wrapper for VisitInterruptObj.
 */
class VisitInterrupt : public ObjectRef {
 public:
  /*! \brief Construct an interrupt with no payload. */
  VisitInterrupt() : VisitInterrupt(Any(nullptr)) {}
  /*!
   * \brief Construct an interrupt with a user-defined payload.
   * \param value The payload carried by the interrupt.
   */
  explicit VisitInterrupt(Any value)
      : ObjectRef(make_object<VisitInterruptObj>(std::move(value))) {}

  /// \cond Doxygen_Suppress
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(VisitInterrupt, ObjectRef, VisitInterruptObj);
  /// \endcond
};

/*!
 * \brief C-ABI safe-call style visit function pointer.
 *
 * Used as the primary entry point of \ref StructuralVisitor so that non-C++
 * bindings (e.g. Rust) can implement and invoke visitors without crossing a
 * C++ exception boundary.
 *
 * \param self  Opaque visitor self pointer (the value stored in
 *              \ref StructuralVisitor::self).
 * \param value The object being visited.
 * \param out   Out parameter: set to ``std::nullopt`` on no-interrupt, or to a
 *              \ref VisitInterrupt to halt traversal. Must be pointer to a
 *              default-initialized \c Optional<VisitInterrupt>.
 * \return 0 on success, non-zero on error. On error, the error is set via
 *         \c TVMFFIErrorSetRaised and may be retrieved with
 *         \c TVMFFIErrorMoveFromRaised.
 *
 * \sa TVMFFISafeCallType
 */
using FStructuralVisitSafe = int (*)(void* self, const ObjectRef& value,
                                     Optional<VisitInterrupt>* out);

/*!
 * \brief C++ fast-path visit function pointer (throws on error).
 *
 * Optional companion to \ref FStructuralVisitSafe. When non-null, callers in
 * the same C++ ABI may invoke it directly and let exceptions propagate, saving
 * the catch/rethrow round-trip of the safe-call path.
 *
 * Always set to \c nullptr for visitors authored outside C++.
 */
using FStructuralVisitCpp = Optional<VisitInterrupt> (*)(void* self, const ObjectRef& value);

/*!
 * \brief Structural visitor driving recursive traversal of an Object tree.
 *
 * The visitor is a layout-stable POD-shaped struct exposing a small
 * function-pointer table (``safe_visit`` / ``cpp_visit``) and an opaque
 * ``self`` pointer. This mirrors the design of \ref TVMFFIFunctionCell and
 * makes the visitor authorable / invokable from non-C++ bindings.
 *
 * Construction modes:
 * - Default-constructed visitors dispatch through the per-type structural
 *   visit attribute registry (\c reflection::type_attr::kStructuralVisit) and
 *   fall back to a reflection-driven field walk when no override is registered.
 * - Derived visitors should fill in ``self`` and the function-pointer table in
 *   their constructor. \ref StructuralVisitorImpl is a convenience template
 *   that wires this up automatically for a C++ callable.
 *
 * The class deliberately avoids virtual functions so the layout is stable
 * across the FFI boundary. Custom dispatch is expressed through the
 * function-pointer table rather than virtual overrides.
 */
class StructuralVisitor {
 public:
  // -------- C-ABI layout: keep these fields first and in this order ---------
  /*! \brief Required C-ABI safe-call entry. Never null on a constructed visitor. */
  FStructuralVisitSafe safe_visit = nullptr;
  /*!
   * \brief Optional C++ fast-path entry. ``nullptr`` for non-C++ visitors.
   *
   * Stored as ``void*`` (rather than \ref FStructuralVisitCpp) to keep this
   * struct free of C++-specific signatures so that language bindings can
   * mirror its layout with a plain pointer field.
   */
  void* cpp_visit = nullptr;
  /*! \brief Opaque self pointer forwarded to ``safe_visit`` / ``cpp_visit``. */
  void* self = nullptr;
  /*! \brief Current def-region context for structural eq/hash semantics. */
  TVMFFIDefRegionKind def_region_mode = kTVMFFIDefRegionKindNone;

  // --------------------------- C++-only API ---------------------------------

  /*!
   * \brief Construct the default structural visitor.
   *
   * Wires up ``safe_visit`` / ``cpp_visit`` to the default dispatcher
   * implemented in \c structural_visit.cc, which consults the structural-visit
   * type attribute registry and falls back to a reflection-driven field walk.
   */
  StructuralVisitor();

  ~StructuralVisitor() = default;
  StructuralVisitor(const StructuralVisitor&) = default;
  StructuralVisitor(StructuralVisitor&&) = default;
  StructuralVisitor& operator=(const StructuralVisitor&) = default;
  StructuralVisitor& operator=(StructuralVisitor&&) = default;

  /*!
   * \brief Visit a value, dispatching through this visitor's function table.
   *
   * Prefers ``cpp_visit`` when available (no catch/rethrow), otherwise routes
   * through ``safe_visit`` and rethrows any raised error as a C++ exception.
   *
   * \param value The object to visit.
   * \return ``std::nullopt`` to continue traversal, or a \ref VisitInterrupt
   *         to halt the entire visit.
   */
  TVM_FFI_INLINE Optional<VisitInterrupt> Visit(const ObjectRef& value) {
    // Use cpp_visit fast path when present, mirroring FunctionObj::CallPacked.
    if (cpp_visit != nullptr) {
      return reinterpret_cast<FStructuralVisitCpp>(cpp_visit)(self, value);
    }
    Optional<VisitInterrupt> out;
    TVM_FFI_CHECK_SAFE_CALL(safe_visit(self, value, &out));
    return out;
  }

  /*!
   * \brief Default visit behavior: type-attr lookup, then reflection fallback.
   *
   * Custom visitors typically invoke this after performing their own
   * type-specific handling to obtain standard recursion semantics. Any
   * exceptions raised by registered visit hooks propagate as C++ exceptions.
   */
  Optional<VisitInterrupt> DefaultVisit(const ObjectRef& value);

  /*! \brief Get the current def-region context. */
  TVM_FFI_INLINE TVMFFIDefRegionKind def_region_kind() const { return def_region_mode; }

  /*!
   * \brief Temporarily switch the def-region context while invoking \p callback.
   * \param kind The def-region kind to set during the callback.
   * \param callback A nullary callable whose return value is forwarded.
   * \return The return value of \p callback.
   */
  template <typename Callback>
  TVM_FFI_INLINE auto WithDefRegionKind(TVMFFIDefRegionKind kind, Callback&& callback)
      -> decltype(std::forward<Callback>(callback)()) {
    class Scope {
     public:
      Scope(StructuralVisitor* visitor, TVMFFIDefRegionKind kind)
          : visitor_(visitor), old_kind_(visitor->def_region_mode) {
        visitor_->def_region_mode = kind;
      }
      ~Scope() { visitor_->def_region_mode = old_kind_; }
      Scope(const Scope&) = delete;
      Scope& operator=(const Scope&) = delete;

     private:
      StructuralVisitor* visitor_;
      TVMFFIDefRegionKind old_kind_;
    };
    Scope scope(this, kind);
    return std::forward<Callback>(callback)();
  }
};

/*!
 * \brief Convenience template that adapts a C++ callable into a StructuralVisitor.
 *
 * Mirrors \c details::FunctionObjImpl: holds a callable and wires up the
 * ``safe_visit`` / ``cpp_visit`` thunks so that the inline \ref
 * StructuralVisitor::Visit dispatch reaches the callable.
 *
 * The callable must be invocable as
 * ``Optional<VisitInterrupt>(StructuralVisitor* visitor, const ObjectRef& value)``.
 * It receives the active visitor so it can recurse via ``visitor->Visit(child)``
 * or fall back to default behavior via ``visitor->DefaultVisit(child)``.
 *
 * \tparam TCallable A non-cv-qualified, non-reference callable type.
 */
template <typename TCallable>
class StructuralVisitorImpl : public StructuralVisitor {
 public:
  static_assert(std::is_same_v<TCallable, std::remove_cv_t<std::remove_reference_t<TCallable>>>,
                "TCallable of StructuralVisitorImpl cannot be const or reference type");

  template <typename... Args>
  explicit StructuralVisitorImpl(Args&&... args) : callable_(std::forward<Args>(args)...) {
    this->self = static_cast<StructuralVisitor*>(this);
    this->cpp_visit = reinterpret_cast<void*>(&CppVisit);
    this->safe_visit = &SafeVisit;
  }

  StructuralVisitorImpl(const StructuralVisitorImpl&) = delete;
  StructuralVisitorImpl& operator=(const StructuralVisitorImpl&) = delete;

 private:
  static Optional<VisitInterrupt> CppVisit(void* self, const ObjectRef& value) {
    auto* p = static_cast<StructuralVisitorImpl*>(static_cast<StructuralVisitor*>(self));
    return p->callable_(static_cast<StructuralVisitor*>(p), value);
  }

  static int SafeVisit(void* self, const ObjectRef& value, Optional<VisitInterrupt>* out) {
    TVM_FFI_SAFE_CALL_BEGIN();
    *out = CppVisit(self, value);
    TVM_FFI_SAFE_CALL_END();
  }

  TCallable callable_;
};

// ---------------------------------------------------------------------------
// Walk helpers — kept compiling against the new StructuralVisitor shape.
// (Walk semantics themselves are not the focus of this redesign; see
//  the inline Walker below for the minimal adaptation.)
// ---------------------------------------------------------------------------

enum class WalkOrder : int32_t {
  kPreOrder = 0,
  kPostOrder = 1,
};

enum class WalkResult : int32_t {
  kAdvance = 0,
  kSkip = 1,
};

namespace details {
template <typename... T>
struct FirstTypeImpl;

template <typename T, typename... Rest>
struct FirstTypeImpl<T, Rest...> {
  using type = T;
};

template <typename... T>
using FirstType = typename FirstTypeImpl<T...>::type;
}  // namespace details

template <typename... T>
using StructuralWalkCallbackArg =
    std::conditional_t<sizeof...(T) == 1, details::FirstType<T...>, Variant<T...>>;

template <typename U, typename... T>
void TryMatchCallbackArg(AnyView value, Optional<StructuralWalkCallbackArg<T...>>* result) {
  if (result->has_value()) return;
  if (std::optional<U> opt = value.as<U>()) {
    if constexpr (sizeof...(T) == 1) {
      *result = *opt;
    } else {
      *result = Variant<T...>(*opt);
    }
  }
}

template <typename... T>
Optional<StructuralWalkCallbackArg<T...>> MatchCallbackArg(AnyView value) {
  Optional<StructuralWalkCallbackArg<T...>> result = std::nullopt;
  (TryMatchCallbackArg<T, T...>(value, &result), ...);
  return result;
}

inline Optional<VisitInterrupt> ToInterrupt(Variant<WalkResult, VisitInterrupt> result) {
  if (std::optional<VisitInterrupt> interrupt = result.as<VisitInterrupt>()) {
    return *interrupt;
  }
  return std::nullopt;
}

template <typename... T, typename F>
Optional<VisitInterrupt> structuralWalk(AnyView root, F&& callback,
                                        WalkOrder order = WalkOrder::kPreOrder) {
  using Callback = std::decay_t<F>;

  // The Walker holds the user callback and delegates default recursion to
  // StructuralVisitor::DefaultVisit. Because StructuralVisitor::Visit is no
  // longer virtual, the override is expressed through the function-pointer
  // table (cpp_visit / safe_visit) wired up in the constructor.
  class Walker : public StructuralVisitor {
   public:
    Walker(Callback callback, WalkOrder order)
        : callback_(std::move(callback)), order_(order) {
      this->self = static_cast<StructuralVisitor*>(this);
      this->cpp_visit = reinterpret_cast<void*>(&Walker::CppVisit);
      this->safe_visit = &Walker::SafeVisit;
    }

   private:
    Optional<VisitInterrupt> VisitImpl(const ObjectRef& value) {
      if (order_ == WalkOrder::kPreOrder) {
        if (auto matched = MatchCallbackArg<T...>(value)) {
          Variant<WalkResult, VisitInterrupt> result = callback_(*matched, this->def_region_kind());
          if (auto interrupt = ToInterrupt(result)) {
            return interrupt;
          }
          if (result.template get<WalkResult>() == WalkResult::kSkip) {
            return std::nullopt;
          }
        }
      }
      if (auto interrupt = this->DefaultVisit(value)) {
        return interrupt;
      }
      if (order_ == WalkOrder::kPostOrder) {
        if (auto matched = MatchCallbackArg<T...>(value)) {
          Variant<WalkResult, VisitInterrupt> result = callback_(*matched, this->def_region_kind());
          if (auto interrupt = ToInterrupt(result)) {
            return interrupt;
          }
        }
      }
      return std::nullopt;
    }

    static Optional<VisitInterrupt> CppVisit(void* self, const ObjectRef& value) {
      return static_cast<Walker*>(static_cast<StructuralVisitor*>(self))->VisitImpl(value);
    }

    static int SafeVisit(void* self, const ObjectRef& value, Optional<VisitInterrupt>* out) {
      TVM_FFI_SAFE_CALL_BEGIN();
      *out = CppVisit(self, value);
      TVM_FFI_SAFE_CALL_END();
    }

    Callback callback_;
    WalkOrder order_;
  };

  Walker walker(std::forward<F>(callback), order);
  return walker.Visit(root.cast<ObjectRef>());
}

}  // namespace ffi
}  // namespace tvm
#endif  // TVM_FFI_EXTRA_STRUCTURAL_VISIT_H_
