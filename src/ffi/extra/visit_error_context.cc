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
/*
 * \file src/ffi/extra/visit_error_context.cc
 * \brief VisitErrorContext implementation — breadcrumb-trail collection and access-path
 *        extraction for recursive Object visits.
 */
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/dict.h>
#include <tvm/ffi/container/list.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/extra/visit_error_context.h>
#include <tvm/ffi/memory.h>
#include <tvm/ffi/reflection/accessor.h>
#include <tvm/ffi/string.h>

#include <utility>
#include <vector>

namespace tvm {
namespace ffi {

/**
 * \brief Internal handler for finding all access paths in a root ObjectRef
 *        that match the breadcrumb pattern stored in a VisitErrorContext.
 */
class VisitErrorAccessPathFinder {
 public:
  TVM_FFI_COLD_CODE
  explicit VisitErrorAccessPathFinder(VisitErrorContext context, bool allow_prefix_match)
      : context_(std::move(context)), allow_prefix_match_(allow_prefix_match) {
    // Normalize the breadcrumb pattern before matching. Two kinds of noise can
    // accumulate in reverse_visit_pattern at recording time:
    //
    //   1. Consecutive duplicates of the same ObjectRef. This is expected when
    //      TVM_FFI_VISIT_THROW(kind, node) and an enclosing
    //      TVM_FFI_VISIT_END(node) at the same level both record
    //      the same `node` — the throw site seeds the context with `node`, and
    //      the catch handler immediately above appends `node` again. Treat any
    //      run of identical adjacent entries as a single frame.
    //
    //   2. Null / undefined ObjectRefs. These can leak in when a visited node
    //      was mutated or torn down between recording and matching, leaving a
    //      stale slot. We can never match a null pointer against a live tree,
    //      so drop them silently rather than letting them break the chain.
    //
    // The matcher operates on `records_` from here on; the original
    // `context_->reverse_visit_pattern` is left untouched so callers that
    // inspect the context for debugging still see the raw history.
    const List<ObjectRef>& raw = context_->reverse_visit_pattern;
    records_.reserve(raw.size());
    for (const ObjectRef& entry : raw) {
      if (!entry.defined()) continue;
      if (!records_.empty() && entry.same_as(records_.back())) continue;
      records_.push_back(entry);
    }
  }

  TVM_FFI_COLD_CODE
  Array<reflection::AccessPath> Find(const ObjectRef& root) {
    this->VisitObject(root);
    return Array<reflection::AccessPath>(results_.begin(), results_.end());
  }

 private:
  /**
   * \brief Stack-allocated mirror of AccessStepObj used during the descent hot path.
   *        For kAttr: stores const TVMFFIFieldInfo* encoded as void* in key (via
   *        kTVMFFIOpaquePtr). String allocation is deferred to ToAccessStep().
   *        Not an Object — pure struct, no Object header / refcount.
   */
  struct TempAccessStep {
    reflection::AccessKind kind;
    Any key{};  // For kAttr: FieldInfo pointer encoded as void* (kTVMFFIOpaquePtr).
                // For kArrayItem: int64 index.
                // For kMapItem: the user's key.

    static TempAccessStep Attr(const TVMFFIFieldInfo* fi) {
      TempAccessStep s;
      s.kind = reflection::AccessKind::kAttr;
      // We store `fi` as an OPAQUE POINTER inside `Any` (kTVMFFIOpaquePtr).
      // `Any` is used here purely as a value-carrying slot for the bits of
      // the pointer — it does NOT take ownership, does NOT dereference
      // through it, and does NOT mutate the pointee. The FieldInfo struct
      // is read-only metadata owned by the type-info registry and lives
      // for the lifetime of the type info, so the underlying object is
      // unaffected by the const_cast below.
      //
      // The const_cast is needed only because `Any` provides TypeTraits for
      // `void*` but not for `const void*`. On retrieval (ToAccessStep()
      // below) we cast back to `const TVMFFIFieldInfo*` before any use.
      s.key = Any(const_cast<void*>(static_cast<const void*>(fi)));
      return s;
    }

    static TempAccessStep ArrayItem(int64_t index) {
      TempAccessStep s;
      s.kind = reflection::AccessKind::kArrayItem;
      s.key = Any(index);
      return s;
    }

    static TempAccessStep MapItem(Any k) {
      TempAccessStep s;
      s.kind = reflection::AccessKind::kMapItem;
      s.key = std::move(k);
      return s;
    }

    /*! \brief Materialize the heap-allocated AccessStep. Called only at match time.
     *         The String allocation for fi->name happens HERE, once. */
    reflection::AccessStep ToAccessStep() const {
      if (kind == reflection::AccessKind::kAttr) {
        // Recover the original `const TVMFFIFieldInfo*` from the opaque
        // pointer stored in `key`. The bits round-trip unchanged; the
        // const-qualifier is restored here, immediately before any use.
        const TVMFFIFieldInfo* fi = static_cast<const TVMFFIFieldInfo*>(key.cast<void*>());
        return reflection::AccessStep::Attr(String(fi->name.data, fi->name.size));
      } else if (kind == reflection::AccessKind::kArrayItem) {
        return reflection::AccessStep::ArrayItem(key.cast<int64_t>());
      } else {
        // kMapItem
        return reflection::AccessStep::MapItem(key);
      }
    }
  };

  void VisitAny(Any value) {
    // Skip null Any silently — error-path defensive.
    if (value == nullptr) return;
    const int32_t type_index = value.type_index();
    if (type_index < TypeIndex::kTVMFFIStaticObjectBegin) {
      // Primitive — cannot hold an ObjectRef chain entry.
      return;
    }
    switch (type_index) {
      case TypeIndex::kTVMFFIArray:
        this->VisitSequence(
            details::AnyUnsafe::MoveFromAnyAfterCheck<Array<Any>>(std::move(value)));
        break;
      case TypeIndex::kTVMFFIList:
        this->VisitSequence(details::AnyUnsafe::MoveFromAnyAfterCheck<List<Any>>(std::move(value)));
        break;
      case TypeIndex::kTVMFFIMap:
        this->VisitMap(details::AnyUnsafe::MoveFromAnyAfterCheck<Map<Any, Any>>(std::move(value)));
        break;
      case TypeIndex::kTVMFFIDict:
        this->VisitMap(details::AnyUnsafe::MoveFromAnyAfterCheck<Dict<Any, Any>>(std::move(value)));
        break;
      default:
        if (type_index >= TypeIndex::kTVMFFIStaticObjectBegin) {
          ObjectRef obj = details::AnyUnsafe::MoveFromAnyAfterCheck<ObjectRef>(std::move(value));
          this->VisitObject(obj);
        }
        break;
    }
  }

  void VisitObject(const ObjectRef& node) {
    // Defensive: error path; never throw.
    if (!node.defined()) return;
    const TVMFFITypeInfo* type_info = TVMFFIGetTypeInfo(node->type_index());
    if (type_info == nullptr || type_info->metadata == nullptr) return;

    bool matched_step = num_pattern_step_matched_ < records_.size() &&
                        node.same_as(records_[records_.size() - 1 - num_pattern_step_matched_]);
    if (matched_step) {
      ++num_pattern_step_matched_;
      if (num_pattern_step_matched_ == records_.size()) {
        // Full match — materialize the AccessPath and record.
        results_.push_back(this->MaterializeAccessPath());
        --num_pattern_step_matched_;
        return;
      }
    }

    this->VisitChildrenFields(node, type_info);

    if (matched_step) --num_pattern_step_matched_;
  }

  void VisitChildrenFields(ObjectRef node, const TVMFFITypeInfo* type_info) {
    // Snapshot results_.size() before descent to detect any inner full or
    // prefix match recorded by a deeper call. If results_ grew, some inner
    // node was recorded — we do not record again here.
    size_t results_before = results_.size();
    reflection::ForEachFieldInfo(type_info, [&](const TVMFFIFieldInfo* field_info) {
      reflection::FieldGetter getter(field_info);
      Any child_val = getter(node);
      this->PushStep(TempAccessStep::Attr(field_info));
      this->VisitAny(std::move(child_val));
      this->PopStep();
    });

    // Leaf-with-prefix-match: this node contributed a match step, but no
    // inner result was recorded from its subtree. Record the current node's
    // path as the best-effort prefix. path_steps_ reflects the path TO this
    // node (the step leading here was pushed by our caller), so
    // MaterializeAccessPath() yields the correct prefix path.
    // Skip when path_steps_ is empty — AccessPath::Root() gives no useful info.
    if (allow_prefix_match_ && num_pattern_step_matched_ > 0 &&
        num_pattern_step_matched_ < records_.size() && !path_steps_.empty() &&
        results_.size() == results_before) {
      results_.push_back(this->MaterializeAccessPath());
    }
  }

  template <typename SeqType>
  void VisitSequence(const SeqType& seq) {
    for (size_t i = 0; i < seq.size(); ++i) {
      Any item = seq[static_cast<int64_t>(i)];
      if (item == nullptr) continue;
      this->PushStep(TempAccessStep::ArrayItem(static_cast<int64_t>(i)));
      this->VisitAny(std::move(item));
      this->PopStep();
    }
  }

  template <typename MapType>
  void VisitMap(const MapType& m) {
    for (const std::pair<Any, Any>& kv : m) {
      if (kv.first == nullptr || kv.second == nullptr) continue;
      this->PushStep(TempAccessStep::MapItem(kv.first));
      this->VisitAny(kv.second);
      this->PopStep();
    }
  }

  /*! \brief Append a TempAccessStep to the descent stack; cache unchanged. */
  void PushStep(TempAccessStep step) { path_steps_.push_back(std::move(step)); }

  /*! \brief Pop the top TempAccessStep; truncate materialized_paths_ to match. */
  void PopStep() {
    path_steps_.pop_back();
    if (materialized_paths_.size() > path_steps_.size()) {
      materialized_paths_.erase(
          materialized_paths_.begin() + static_cast<std::ptrdiff_t>(path_steps_.size()),
          materialized_paths_.end());
    }
  }

  // Cache invariant maintained jointly with PushStep / PopStep:
  //   materialized_paths_[k] (when present) is the AccessPath built
  //   from path_steps_[0..k+1] as they currently exist, and
  //   materialized_paths_.size() <= path_steps_.size().
  //
  // - PushStep:              appends to path_steps_; cache unchanged.
  // - PopStep:               pops path_steps_; truncates cache to match.
  // - MaterializeAccessPath: extends cache up to path_steps_.size(),
  //                          chaining new AccessPath nodes via Extend.
  //
  // Lazy materialization avoids per-descent AccessPath allocation;
  // cache amortizes across consecutive matches with shared prefix
  // (LCA sharing).
  /*! \brief Materialize the AccessPath at the current descent depth. */
  reflection::AccessPath MaterializeAccessPath() {
    // Root-itself match: matching fired before any descent step was pushed,
    // so the access path is just Root(). Returning early also avoids reading
    // back() from an empty materialized_paths_.
    if (path_steps_.empty()) return reflection::AccessPath::Root();
    for (size_t idx = materialized_paths_.size(); idx < path_steps_.size(); ++idx) {
      reflection::AccessPath parent =
          (idx == 0) ? reflection::AccessPath::Root() : materialized_paths_[idx - 1];
      materialized_paths_.push_back(parent->Extend(path_steps_[idx].ToAccessStep()));
    }
    return materialized_paths_.back();
  }

  // The visit error context whose pattern we're matching against.
  VisitErrorContext context_;
  // When true, record prefix matches at leaves (partial pattern match).
  bool allow_prefix_match_;
  // Normalized breadcrumb pattern derived from context_->reverse_visit_pattern
  // by dropping null/undefined entries and collapsing consecutive duplicates.
  // See the constructor for rationale.
  std::vector<ObjectRef> records_;
  // Count of pattern entries matched so far (root-closest first). Incremented on
  // match, decremented on unwind; full match when equal to pattern size.
  size_t num_pattern_step_matched_{0};
  // Current descent path — entries pushed on field/index/key descent, popped on unwind.
  std::vector<TempAccessStep> path_steps_;
  // Lazy cache of materialized AccessPath nodes, parallel to path_steps_.
  // materialized_paths_[k] corresponds to path_steps_[0..k+1] as they currently
  // stand. Extended on match; truncated by PopStep. size <= path_steps_.size().
  std::vector<reflection::AccessPath> materialized_paths_;
  // Recorded full-match paths.
  std::vector<reflection::AccessPath> results_;
};

// ---------------------------------------------------------------------------
// VisitErrorContext::FindAccessPaths
// ---------------------------------------------------------------------------

TVM_FFI_COLD_CODE
Array<reflection::AccessPath> VisitErrorContext::FindAccessPaths(
    const ObjectRef& root, const VisitErrorContext& visit_context, bool allow_prefix_match) {
  VisitErrorAccessPathFinder finder(visit_context, allow_prefix_match);
  return finder.Find(root);
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<VisitErrorContextObj>()
      .def_ro("reverse_visit_pattern", &VisitErrorContextObj::reverse_visit_pattern)
      .def_ro("prev_error_context", &VisitErrorContextObj::prev_error_context);
  refl::GlobalDef().def("ffi.VisitErrorContext.FindAccessPaths",
                        &VisitErrorContext::FindAccessPaths);
}

}  // namespace ffi
}  // namespace tvm
