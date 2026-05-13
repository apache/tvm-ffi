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
 * \file tvm/ffi/extra/structural_error_context.h
 * \brief StructuralErrorContext: typed payload for Error::extra_context that records the
 *        chain of ObjectRefs visited during a recursive structural visit when an error is thrown.
 */
#ifndef TVM_FFI_EXTRA_STRUCTURAL_ERROR_CONTEXT_H_
#define TVM_FFI_EXTRA_STRUCTURAL_ERROR_CONTEXT_H_

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/base.h>
#include <tvm/ffi/memory.h>
#include <tvm/ffi/object.h>
#include <tvm/ffi/optional.h>
#include <tvm/ffi/reflection/access_path.h>

namespace tvm {
namespace ffi {

/*!
 * \brief Object class for StructuralErrorContext.
 *
 * \sa StructuralErrorContext
 */
class StructuralErrorContextObj : public Object {
 public:
  /*!
   * \brief Visit records that get populated, which include the object visit
   *        path pattern in innermost-first order. Best-effort — not exhaustive.
   */
  Array<ObjectRef> reverse_visit_pattern;

  /*!
   * \brief Pre-existing Error::extra_context payload before we placed the
   *        StructuralErrorContext.
   */
  Optional<ObjectRef> previous_error_context;

  /// \cond Doxygen_Suppress
  static constexpr bool _type_mutable = true;
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindUnsupported;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ffi.StructuralErrorContext", StructuralErrorContextObj,
                                    Object);
  /// \endcond
};

/*!
 * \brief Typed payload attached to Error::extra_context to support
 *        structural-context-aware error reporting.
 *
 * The StructuralErrorContext captures the reverse_visit_pattern —
 * the chain of nodes visited before an error was thrown — so callers
 * can translate it via FindAccessPaths into a structured access path
 * for richer error messages.
 *
 * Typical usage:
 *
 *   1. A recursive structural visit is instrumented with
 *      TVM_FFI_STRUCTURAL_VISIT_BEGIN / _END(node). On throw, the
 *      macros record the visited nodes into a StructuralErrorContext
 *      and attach it to the Error's extra_context.
 *
 *   2. The root catch handler retrieves the context via
 *      TryGetFromError(err), then resolves the chain into one or more
 *      reflection::AccessPath instances via FindAccessPaths(root, ctx).
 *
 *   3. The caller uses the AccessPath to enrich the error message
 *      with structured position info (e.g., ".body[2].cond.lhs").
 */
class StructuralErrorContext : public ObjectRef {
 public:
  /*! \brief Get the StructuralErrorContext attached to err's extra_context.
   *  \param err The error to inspect.
   *  \return The attached StructuralErrorContext, or NullOpt if absent.
   */
  static Optional<StructuralErrorContext> TryGetFromError(const Error& err) {
    std::optional<ObjectRef> ec = err.extra_context();
    if (ec) {
      return ec->as<StructuralErrorContext>();
    }
    return std::nullopt;
  }

  /*! \brief Find all access paths that match the pattern specified in the
   *         StructuralErrorContext.
   *  \param root The root ObjectRef to search from.
   *  \param structural_context The StructuralErrorContext to match against.
   *  \param allow_prefix_match If true, also report paths where only a
   *                            prefix of the pattern was matched (i.e.,
   *                            the algorithm descended through some
   *                            matched records but could not find further
   *                            matches before reaching a leaf). Default
   *                            false — only full pattern matches are
   *                            reported.
   *  \return Array of matched access paths.
   */
  TVM_FFI_EXTRA_CXX_API static Array<reflection::AccessPath> FindAccessPaths(
      const ObjectRef& root, const StructuralErrorContext& structural_context,
      bool allow_prefix_match = false);

  /// \cond Doxygen_Suppress
  explicit StructuralErrorContext(ObjectPtr<StructuralErrorContextObj> n)
      : ObjectRef(std::move(n)) {}
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(StructuralErrorContext, ObjectRef,
                                                StructuralErrorContextObj);
  /// \endcond
};

/*!
 * \brief Begin a structural-visit try block.
 *
 * Must be paired with TVM_FFI_STRUCTURAL_VISIT_END(node) at the end of the
 * visit body. Expands to an open `try {` — a mismatched _END macro is a
 * compile error (unclosed try block).
 *
 * \code{.cpp}
 * void MyVisitor::VisitNode(const ObjectRef& node) {
 *   TVM_FFI_STRUCTURAL_VISIT_BEGIN();
 *   DispatchVisit(node);
 *   TVM_FFI_STRUCTURAL_VISIT_END(node);
 * }
 * \endcode
 */
#define TVM_FFI_STRUCTURAL_VISIT_BEGIN() try {
/*!
 * \brief End a structural-visit try block and catch+re-throw any Error,
 *        appending node to the StructuralErrorContext on the way up.
 *
 * Must be paired with TVM_FFI_STRUCTURAL_VISIT_BEGIN() above the visit body.
 *
 * \param node The ObjectRef at the current visit level (appended to the
 *             error context's reverse_visit_pattern on exception).
 */
#define TVM_FFI_STRUCTURAL_VISIT_END(node)                                          \
  }                                                                                 \
  catch (::tvm::ffi::Error & _tvm_ffi_visit_err_) {                                 \
    ::tvm::ffi::details::UpdateStructuralErrorContext(_tvm_ffi_visit_err_, (node)); \
    throw;                                                                          \
  }

namespace details {
/*!
 * \brief Implementation helper for TVM_FFI_STRUCTURAL_VISIT_END(node).
 *        Calling convention may change; do not call directly from user code.
 *
 * \note Safe to call only while the Error is owned by the current thread during
 *       exception unwind, before any external observer (Python wrapper, logging
 *       hook) takes a reference. After the catch handler returns to wider scope,
 *       treat the error as immutable. Writes directly to ErrorObj::extra_context
 *       using ObjectUnsafe refcount-aware patterns (no separate fn-ptr needed).
 */
inline void UpdateStructuralErrorContext(Error err, ObjectRef node) {
  std::optional<ObjectRef> ec = err.extra_context();
  if (ec) {
    Optional<StructuralErrorContext> sec = ec->as<StructuralErrorContext>();
    if (sec) {
      sec.value()->reverse_visit_pattern.push_back(node);
      return;
    }
  }
  // Build a fresh StructuralErrorContext, preserving any pre-existing payload.
  ObjectPtr<StructuralErrorContextObj> obj = make_object<StructuralErrorContextObj>();
  obj->reverse_visit_pattern = Array<ObjectRef>{node};
  if (ec) obj->previous_error_context = *ec;

  // Write directly to the ErrorObj cell using ObjectUnsafe patterns.
  // This is safe here because we are the sole owner during exception unwind.
  ErrorObj* error_obj =
      static_cast<ErrorObj*>(details::ObjectUnsafe::RawObjectPtrFromObjectRef(err));
  if (error_obj->extra_context != nullptr) {
    details::ObjectUnsafe::DecRefObjectHandle(error_obj->extra_context);
  }
  error_obj->extra_context = details::ObjectUnsafe::MoveObjectPtrToTVMFFIObjectPtr(std::move(obj));
}
}  // namespace details

}  // namespace ffi
}  // namespace tvm
#endif  // TVM_FFI_EXTRA_STRUCTURAL_ERROR_CONTEXT_H_
