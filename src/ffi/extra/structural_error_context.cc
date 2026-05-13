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
 * \file src/ffi/extra/structural_error_context.cc
 * \brief StructuralErrorContext implementation — breadcrumb-trail collection and access-path
 *        extraction for recursive Object visits.
 */
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/dict.h>
#include <tvm/ffi/container/list.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/extra/structural_error_context.h>
#include <tvm/ffi/memory.h>
#include <tvm/ffi/reflection/accessor.h>
#include <tvm/ffi/string.h>

#include <utility>
#include <vector>

namespace tvm {
namespace ffi {

/**
 * \brief Internal handler for finding all access paths in a root ObjectRef
 *        that match the breadcrumb pattern stored in a StructuralErrorContext.
 */
class StructuralErrorAccessPathFinder {
 public:
  explicit StructuralErrorAccessPathFinder(StructuralErrorContext context, bool allow_prefix_match)
      : context_(std::move(context)),
        allow_prefix_match_(allow_prefix_match),
        num_pattern_step_matched_(0) {}

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
      // Any has TypeTraits<void*> (kTVMFFIOpaquePtr) but not const void*.
      // Cast away const for storage; restore on retrieval.
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
          this->VisitObject(std::move(obj));
        }
        break;
    }
  }

  void VisitObject(ObjectRef node) {
    // Defensive: error path; never throw.
    if (!node.defined()) return;
    const TVMFFITypeInfo* type_info = TVMFFIGetTypeInfo(node->type_index());
    if (type_info == nullptr || type_info->metadata == nullptr) return;

    const Array<ObjectRef>& records = context_->reverse_visit_pattern;
    bool matched_step = num_pattern_step_matched_ < records.size() &&
                        node.same_as(records[records.size() - 1 - num_pattern_step_matched_]);
    if (matched_step) {
      ++num_pattern_step_matched_;
      if (num_pattern_step_matched_ == records.size()) {
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
        num_pattern_step_matched_ < context_->reverse_visit_pattern.size() &&
        !path_steps_.empty() && results_.size() == results_before) {
      results_.push_back(this->MaterializeAccessPath());
    }
  }

  template <typename SeqType>
  void VisitSequence(SeqType seq) {
    for (size_t i = 0; i < seq.size(); ++i) {
      Any item = seq[i];
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
      materialized_paths_.erase(materialized_paths_.begin() + path_steps_.size(),
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
    for (size_t idx = materialized_paths_.size(); idx < path_steps_.size(); ++idx) {
      reflection::AccessPath parent =
          (idx == 0) ? reflection::AccessPath::Root() : materialized_paths_[idx - 1];
      materialized_paths_.push_back(parent->Extend(path_steps_[idx].ToAccessStep()));
    }
    return materialized_paths_.back();
  }

  // The structural error context whose pattern we're matching against.
  StructuralErrorContext context_;
  // When true, record prefix matches at leaves (partial pattern match).
  bool allow_prefix_match_;
  // Count of pattern entries matched so far (root-closest first). Incremented on
  // match, decremented on unwind; full match when equal to pattern size.
  size_t num_pattern_step_matched_;
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
// StructuralErrorContext::FindAccessPaths
// ---------------------------------------------------------------------------

Array<reflection::AccessPath> StructuralErrorContext::FindAccessPaths(
    const ObjectRef& root, const StructuralErrorContext& structural_context,
    bool allow_prefix_match) {
  StructuralErrorAccessPathFinder finder(structural_context, allow_prefix_match);
  return finder.Find(root);
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<StructuralErrorContextObj>()
      .def_ro("reverse_visit_pattern", &StructuralErrorContextObj::reverse_visit_pattern)
      .def_ro("previous_error_context", &StructuralErrorContextObj::previous_error_context);
  // Register FindAccessPaths with the 3-arg signature (UpdateError takes Error& which is
  // not supported by the packed function protocol).
  refl::GlobalDef().def("ffi.StructuralErrorContext.FindAccessPaths",
                        static_cast<Array<reflection::AccessPath> (*)(
                            const ObjectRef&, const StructuralErrorContext&, bool)>(
                            &StructuralErrorContext::FindAccessPaths));
}

}  // namespace ffi
}  // namespace tvm
