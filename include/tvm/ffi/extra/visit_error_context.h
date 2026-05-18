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
 * \file tvm/ffi/extra/visit_error_context.h
 * \brief VisitErrorContext: typed payload for Error::extra_context that records the
 *        chain of ObjectRefs visited during a recursive visit when an error is thrown.
 */
#ifndef TVM_FFI_EXTRA_VISIT_ERROR_CONTEXT_H_
#define TVM_FFI_EXTRA_VISIT_ERROR_CONTEXT_H_

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/list.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/base.h>
#include <tvm/ffi/memory.h>
#include <tvm/ffi/object.h>
#include <tvm/ffi/optional.h>
#include <tvm/ffi/reflection/access_path.h>

namespace tvm {
namespace ffi {

/*!
 * \brief Object class for VisitErrorContext.
 *
 * \sa VisitErrorContext
 */
class VisitErrorContextObj : public Object {
 public:
  /*!
   * \brief Visit records that get populated, which include the object visit
   *        path pattern in innermost-first order. Best-effort — not exhaustive.
   */
  List<ObjectRef> reverse_visit_pattern;

  /*!
   * \brief Pre-existing Error::extra_context payload before we placed the
   *        VisitErrorContext.
   */
  Optional<ObjectRef> prev_error_context;

  /// \cond Doxygen_Suppress
  static constexpr bool _type_mutable = true;
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindUnsupported;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ffi.VisitErrorContext", VisitErrorContextObj, Object);
  /// \endcond
};

/*!
 * \brief Typed payload attached to Error::extra_context to support
 *        visit-context-aware error reporting.
 *
 * The VisitErrorContext captures the reverse_visit_pattern —
 * the chain of nodes visited before an error was thrown — so callers
 * can translate it via FindAccessPaths into a structured access path
 * for richer error messages.
 *
 * Typical usage:
 *
 *   1. A recursive visit is instrumented with
 *      TVM_FFI_VISIT_BEGIN / _END(node). The deepest detection
 *      point throws via TVM_FFI_VISIT_THROW(kind, node), which
 *      seeds the context with the throw site; enclosing BEGIN/END pairs
 *      append parent nodes on rethrow.
 *
 *   2. The root catch handler retrieves the context via
 *      TryGetFromError(err), then resolves the chain into one or more
 *      reflection::AccessPath instances via FindAccessPaths(root, ctx).
 *
 *   3. The caller uses the AccessPath to enrich the error message
 *      with structured position info (e.g., ".body[2].cond.lhs").
 */
class VisitErrorContext : public ObjectRef {
 public:
  /*! \brief Get the VisitErrorContext attached to err's extra_context.
   *  \param err The error to inspect.
   *  \return The attached VisitErrorContext, or NullOpt if absent.
   */
  TVM_FFI_COLD_CODE
  static Optional<VisitErrorContext> TryGetFromError(const Error& err) {
    std::optional<ObjectRef> extra_context = err.extra_context();
    if (extra_context) {
      return extra_context->as<VisitErrorContext>();
    }
    return std::nullopt;
  }

  /*! \brief Find all access paths that match the pattern specified in the
   *         VisitErrorContext.
   *  \param root The root ObjectRef to search from.
   *  \param visit_context The VisitErrorContext to match against.
   *  \param allow_prefix_match If true, also report paths where only a
   *                            prefix of the pattern was matched (i.e.,
   *                            the algorithm descended through some
   *                            matched records but could not find further
   *                            matches before reaching a leaf). Default
   *                            false — only full pattern matches are
   *                            reported.
   *  \return Array of matched access paths.
   */
  TVM_FFI_COLD_CODE
  TVM_FFI_EXTRA_CXX_API static Array<reflection::AccessPath> FindAccessPaths(
      const ObjectRef& root, const VisitErrorContext& visit_context,
      bool allow_prefix_match = false);

  /// \cond Doxygen_Suppress
  explicit VisitErrorContext(ObjectPtr<VisitErrorContextObj> n) : ObjectRef(std::move(n)) {}
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(VisitErrorContext, ObjectRef, VisitErrorContextObj);
  /// \endcond
};

/*!
 * \brief Begin a visit try block.
 *
 * Must be paired with TVM_FFI_VISIT_END(node) at the end of the
 * visit body. Expands to an open `try {` — a mismatched _END macro is a
 * compile error (unclosed try block).
 *
 * \code{.cpp}
 * void MyVisitor::VisitNode(const ObjectRef& node) {
 *   TVM_FFI_VISIT_BEGIN();
 *   DispatchVisit(node);
 *   TVM_FFI_VISIT_END(node);
 * }
 * \endcode
 */
#define TVM_FFI_VISIT_BEGIN() try {
/*!
 * \brief End a visit try block and catch+re-throw any Error,
 *        appending node to the VisitErrorContext on the way up.
 *
 * Must be paired with TVM_FFI_VISIT_BEGIN() above the visit body.
 *
 * \param node The ObjectRef at the current visit level (appended to the
 *             error context's reverse_visit_pattern on exception).
 */
#define TVM_FFI_VISIT_END(node)                                                \
  }                                                                            \
  catch (::tvm::ffi::Error & _tvm_ffi_visit_err_) {                            \
    ::tvm::ffi::details::UpdateVisitErrorContext(_tvm_ffi_visit_err_, (node)); \
    throw;                                                                     \
  }

/*!
 * \brief End a visit try block and catch any Error as an Expected error,
 *        appending node to the VisitErrorContext on the way up.
 *
 * Must be paired with TVM_FFI_VISIT_BEGIN() above the visit body.
 *
 * \param node The current visit value. Object-backed values are appended to the
 *             error context's reverse_visit_pattern on exception.
 */
#define TVM_FFI_VISIT_END_RETURN_EXPECTED(node)                                \
  }                                                                            \
  catch (::tvm::ffi::Error & _tvm_ffi_visit_err_) {                            \
    ::tvm::ffi::details::UpdateVisitErrorContext(_tvm_ffi_visit_err_, (node)); \
    return ::tvm::ffi::Unexpected(_tvm_ffi_visit_err_);                        \
  }

/*!
 * \brief Throw an error from inside a visit, with `node` recorded
 *        as the innermost frame of the resulting VisitErrorContext.
 *
 * Use this when the bad spot is somewhere the BEGIN/END pair does not
 * already record — typically a child field of the currently-visited node,
 * or a helper called from a visit that has no BEGIN/END of its own. The
 * throw site is seeded as the innermost frame; enclosing
 * TVM_FFI_VISIT_BEGIN/END pairs continue to append their nodes
 * on rethrow as the stack unwinds. The macro mirrors TVM_FFI_THROW —
 * it returns an ostream you stream a message into.
 *
 * If `node` here is the same as the enclosing END's node (a redundant
 * throw at the same level), FindAccessPaths normalizes the consecutive
 * duplicate during matching, so user code does not need to guard against
 * it — but in that case the throw would have been recorded by END
 * anyway and a plain TVM_FFI_THROW would suffice.
 *
 * \code{.cpp}
 * // Visiting a TPair node; pin the bad subfield (.lhs) as the throw
 * // site so the resulting AccessPath ends at .lhs, not at the TPair
 * // node itself. The surrounding END appends `node` as the next frame.
 * void Visitor::Visit(const ObjectRef& node) {
 *   TVM_FFI_VISIT_BEGIN();
 *   if (auto pair = node.as<TPair>()) {
 *     if (!IsValid(pair.value()->lhs)) {
 *       TVM_FFI_VISIT_THROW(ValueError, pair.value()->lhs)
 *           << "invalid lhs";
 *     }
 *   }
 *   TVM_FFI_VISIT_END(node);
 * }
 * \endcode
 *
 * \param ErrorKind The kind of error to throw (e.g. TypeError, ValueError).
 * \param node The ObjectRef at the throw site (innermost frame).
 */
#define TVM_FFI_VISIT_THROW(ErrorKind, node)                                                    \
  ::tvm::ffi::details::ErrorBuilder(                                                            \
      #ErrorKind, TVMFFIBacktrace(__FILE__, __LINE__, TVM_FFI_FUNC_SIG, 0),                     \
      TVM_FFI_ALWAYS_LOG_BEFORE_THROW, ::std::nullopt,                                          \
      ::std::optional<::tvm::ffi::ObjectRef>(::tvm::ffi::details::MakeVisitErrorContext(node))) \
      .stream()

namespace details {
/*!
 * \brief Build a fresh VisitErrorContext seeded with `node` as the
 *        innermost (and currently only) frame of reverse_visit_pattern.
 *
 *        Used by TVM_FFI_VISIT_THROW to attach the throw-site
 *        node to the Error's extra_context at construction time.
 */
TVM_FFI_COLD_CODE
inline VisitErrorContext MakeVisitErrorContext(const ObjectRef& node) {
  ObjectPtr<VisitErrorContextObj> obj = make_object<VisitErrorContextObj>();
  obj->reverse_visit_pattern = List<ObjectRef>{node};
  return VisitErrorContext(std::move(obj));
}

/*!
 * \brief Implementation helper for TVM_FFI_VISIT_END(node).
 *        Calling convention may change; do not call directly from user code.
 */
TVM_FFI_COLD_CODE
inline void UpdateVisitErrorContext(Error& err, const ObjectRef& node) {  // NOLINT(*)
  // NOTE: This function mutates the ErrorObj in place via ObjectUnsafe.
  // Expected to run only inside the exception throw chain, where the Error
  // is single-owned by this thread. The tradeoff avoids reallocating a
  // fresh Error per catch frame; the immutability invariant returns once
  // the unwind window closes.
  std::optional<ObjectRef> extra_context = err.extra_context();
  if (extra_context) {
    Optional<VisitErrorContext> visit_context = extra_context->as<VisitErrorContext>();
    if (visit_context) {
      visit_context.value()->reverse_visit_pattern.push_back(node);
      return;
    }
  }
  // Build a fresh VisitErrorContext, preserving any pre-existing payload.
  ObjectPtr<VisitErrorContextObj> new_context = make_object<VisitErrorContextObj>();
  new_context->reverse_visit_pattern = List<ObjectRef>{node};
  if (extra_context) new_context->prev_error_context = *extra_context;

  ErrorObj* error_obj =
      static_cast<ErrorObj*>(details::ObjectUnsafe::RawObjectPtrFromObjectRef(err));
  if (error_obj->extra_context != nullptr) {
    details::ObjectUnsafe::DecRefObjectHandle(error_obj->extra_context);
  }
  error_obj->extra_context =
      details::ObjectUnsafe::MoveObjectPtrToTVMFFIObjectPtr(std::move(new_context));
}
}  // namespace details

}  // namespace ffi
}  // namespace tvm
#endif  // TVM_FFI_EXTRA_VISIT_ERROR_CONTEXT_H_
