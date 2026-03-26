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
 * \file src/ffi/extra/pyast_trait_print.cc
 * \brief Trait-driven printer and default (Level 0) printer.
 *
 * TraitPrint dispatches on the runtime type of the trait object to produce
 * the appropriate AST nodes. DefaultPrint renders any reflected object as
 * TypeKey(field1=val1, field2=val2, ...).
 */
#include <tvm/ffi/container/map_base.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/extra/ir_traits.h>
#include <tvm/ffi/extra/pyast.h>
#include <tvm/ffi/reflection/accessor.h>

#include <cstring>
#include <string>
#include <string_view>
#include <unordered_map>

namespace tvm {
namespace ffi {
namespace pyast {

// Forward declarations
NodeAST DefaultPrint(ObjectRef obj, IRPrinter printer, AccessPath path);

// ============================================================================
// Helpers
// ============================================================================

namespace {

// ---- Helper: parse a callee string like "T.evaluate" or "I.GlobalVar" into ExprAST ----
ExprAST ParseCalleeString(const std::string& callee_str) {
  // Check for "prefix.name" pattern (e.g. "T.prim_func", "I.GlobalVar")
  auto dot = callee_str.find('.');
  if (dot != std::string::npos) {
    std::string prefix = callee_str.substr(0, dot);
    std::string name = callee_str.substr(dot + 1);
    return ExprAttr(IdAST(prefix), name);
  }
  return IdAST(callee_str);
}

/*! \brief Check if an Any value holds a string type (String, SmallStr, or RawStr). */
inline bool IsString(AnyView v) {
  int32_t type_index = v.type_index();
  return type_index == TypeIndex::kTVMFFIStr || type_index == TypeIndex::kTVMFFISmallStr ||
         type_index == TypeIndex::kTVMFFIRawStr;
}

/*! \brief Check if an object is a list/array container (ffi.List or ffi.Array). */
inline bool IsListLike(AnyView obj) {
  int32_t type_index = obj.type_index();
  return type_index == TypeIndex::kTVMFFIList || type_index == TypeIndex::kTVMFFIArray;
}

/*! \brief Get the current frame from the printer, or null if no frames. */
AnyView GetCurrentFrame(const IRPrinter& printer) {
  if (printer->frames.empty()) return AnyView();
  return printer->frames.back();
}

/************** Field-Reference Resolver **************/

const TVMFFIFieldInfo* FindField(int32_t type_index, std::string_view name) {
  const TVMFFITypeInfo* info = TVMFFIGetTypeInfo(type_index);
  for (int32_t d = 1; d < info->type_depth; ++d) {
    const TVMFFITypeInfo* ancestor = info->type_ancestors[d];
    for (int32_t i = 0; i < ancestor->num_fields; ++i) {
      if (name.size() == ancestor->fields[i].name.size &&
          std::memcmp(name.data(), ancestor->fields[i].name.data, name.size()) == 0) {
        return &ancestor->fields[i];
      }
    }
  }
  for (int32_t i = 0; i < info->num_fields; ++i) {
    if (name.size() == info->fields[i].name.size &&
        std::memcmp(name.data(), info->fields[i].name.data, name.size()) == 0) {
      return &info->fields[i];
    }
  }
  return nullptr;
}

Optional<Function> FindMethod(int32_t type_index, std::string_view name) {
  const TVMFFITypeInfo* info = TVMFFIGetTypeInfo(type_index);
  for (int32_t i = 0; i < info->num_methods; ++i) {
    if (name.size() == info->methods[i].name.size &&
        std::memcmp(name.data(), info->methods[i].name.data, name.size()) == 0) {
      return AnyView::CopyFromTVMFFIAny(info->methods[i].method).cast<Function>();
    }
  }
  for (int32_t d = info->type_depth - 1; d >= 1; --d) {
    const TVMFFITypeInfo* ancestor = info->type_ancestors[d];
    for (int32_t i = 0; i < ancestor->num_methods; ++i) {
      if (name.size() == ancestor->methods[i].name.size &&
          std::memcmp(name.data(), ancestor->methods[i].name.data, name.size()) == 0) {
        return AnyView::CopyFromTVMFFIAny(ancestor->methods[i].method).cast<Function>();
      }
    }
  }
  return {};
}

// ---- Helper: get a named field from an object via reflection ----
Any FindFieldValue(AnyView obj, std::string_view name) {
  const TVMFFIFieldInfo* fi = FindField(obj.type_index(), name);
  if (fi == nullptr) {
    TVM_FFI_THROW(ValueError) << "Object of type " << obj.GetTypeKey() << " has no field named '"
                              << name << "'";
  }
  Any result;
  const void* addr = reinterpret_cast<const char*>(obj.cast<const Object*>()) + fi->offset;
  TVM_FFI_CHECK_SAFE_CALL(
      fi->getter(const_cast<void*>(addr), reinterpret_cast<TVMFFIAny*>(&result)));
  return result;
}

Any ResolveWithPrinter(const String& ref, AnyView obj, AnyView printer, AnyView frame) {
  std::string_view sv(ref.data(), ref.size());
  if (sv.size() > 8 && sv.substr(0, 8) == "$global:") {
    std::string_view name = sv.substr(8);
    Function fn = Function::GetGlobalRequired(name);
    Any result;
    AnyView args[2] = {printer, obj};
    fn.CallPacked(args, 2, &result);
    return result;
  }
  if (sv.size() > 8 && sv.substr(0, 8) == "$method:") {
    std::string_view name = sv.substr(8);
    Optional<Function> method = FindMethod(obj.type_index(), name);
    if (!method.has_value()) {
      TVM_FFI_THROW(ValueError) << "Cannot resolve method reference '" << ref << "' on type "
                                << obj.GetTypeKey();
    }
    Any result;
    AnyView args[2] = {printer, obj};
    method.value().CallPacked(args, 2, &result);
    return result;
  }
  if (sv.size() > 7 && sv.substr(0, 7) == "$field:") {
    std::string_view name = sv.substr(7);
    return FindFieldValue(obj, name);
  }
  return ref;
}

/*! \brief Resolve a field reference with printer context and print it through the printer. */
ExprAST ResolveAndPrint(const String& ref, AnyView obj, const IRPrinter& printer,
                        const AccessPath& path) {
  Any value = ResolveWithPrinter(ref, obj, printer, GetCurrentFrame(printer));
  return printer->operator()(std::move(value), path).cast<ExprAST>();
}

Optional<ir_traits::IRTraits> GetTrait(AnyView obj) {
  static reflection::TypeAttrColumn col("__ffi_ir_traits__");
  AnyView v = col[obj.type_index()];
  if (v.type_index() == TypeIndex::kTVMFFINone) return {};
  return v.cast<ir_traits::IRTraits>();
}

/*! \brief Resolve an optional ref with printer context; return null ExprAST if ref is absent. */
Optional<ExprAST> ResolveAndPrintOpt(const Optional<String>& ref, AnyView obj,
                                     const IRPrinter& printer, const AccessPath& path) {
  if (!ref.has_value()) return {};
  Any value = ResolveWithPrinter(ref.value(), obj, printer, GetCurrentFrame(printer));
  return printer->operator()(std::move(value), path).cast<ExprAST>();
}

/*! \brief Resolve a body reference and print each element as statements. */
List<StmtAST> PrintBody(const String& body_ref, AnyView obj, const IRPrinter& printer,
                        const AccessPath& path) {
  Any body_val = ResolveWithPrinter(body_ref, obj, printer, GetCurrentFrame(printer));
  DefaultFrame frame;
  printer->FramePush(frame);

  // If the body value is a list, iterate and print each element individually
  if (body_val.type_index() >= TypeIndex::kTVMFFIStaticObjectBegin) {
    ObjectRef body_obj = body_val.cast<ObjectRef>();
    if (IsListLike(body_obj)) {
      List<Any> items = body_val.cast<List<Any>>();
      for (int64_t i = 0; i < static_cast<int64_t>(items.size()); ++i) {
        Any printed = printer->operator()(items[i], path->Attr("body")->ArrayItem(i));
        if (printed.type_index() >= TypeIndex::kTVMFFIStaticObjectBegin) {
          NodeAST node = printed.cast<NodeAST>();
          if (const auto* block = node.as<StmtBlockASTObj>()) {
            frame->stmts.insert(frame->stmts.end(), block->stmts.begin(), block->stmts.end());
          } else if (node->IsInstance<StmtASTObj>()) {
            frame->stmts.push_back(GetRef<StmtAST>(node.as<StmtASTObj>()));
          } else if (node->IsInstance<ExprASTObj>()) {
            frame->stmts.push_back(ExprStmtAST(GetRef<ExprAST>(node.as<ExprASTObj>())));
          }
        }
      }
      printer->FramePop();
      return frame->stmts;
    }
  }

  Any printed = printer->operator()(std::move(body_val), path->Attr("body"));
  printer->FramePop();
  List<StmtAST> stmts = frame->stmts;
  if (printed.type_index() >= TypeIndex::kTVMFFIStaticObjectBegin) {
    NodeAST node = printed.cast<NodeAST>();
    if (const auto* block = node.as<StmtBlockASTObj>()) {
      stmts.insert(stmts.end(), block->stmts.begin(), block->stmts.end());
    } else if (node->IsInstance<StmtASTObj>()) {
      stmts.push_back(GetRef<StmtAST>(node.as<StmtASTObj>()));
    } else if (node->IsInstance<ExprASTObj>()) {
      stmts.push_back(ExprStmtAST(GetRef<ExprAST>(node.as<ExprASTObj>())));
    }
  }
  return stmts;
}

/*! \brief Print the def-site of a Value-traited variable (name + type annotation). */
AssignAST PrintValueDef(const ObjectRef& var_obj, const IRPrinter& printer, const AccessPath& path,
                        const Optional<ObjectRef>& frame) {
  // Try to get the Value trait from the variable
  auto var_trait = GetTrait(var_obj);
  String name_hint("v");
  Optional<ExprAST> ty_annotation;
  if (var_trait.has_value()) {
    if (const auto* vt = var_trait.value().as<ir_traits::ValueTraitsObj>()) {
      Any name_val = ResolveWithPrinter(vt->name, var_obj, printer, GetCurrentFrame(printer));
      if (IsString(name_val)) {
        name_hint = name_val.cast<String>();
      }
      // Type annotation from text_printer_type or ty
      if (vt->text_printer_type.has_value()) {
        Any ty_val = ResolveWithPrinter(vt->text_printer_type.value(), var_obj, printer,
                                        GetCurrentFrame(printer));
        if (ty_val.type_index() != TypeIndex::kTVMFFINone) {
          ty_annotation =
              printer->operator()(std::move(ty_val), path->Attr("type")).cast<ExprAST>();
        }
      } else if (vt->ty.has_value()) {
        ty_annotation = ResolveAndPrintOpt(vt->ty, var_obj, printer, path->Attr("type"));
      }
    } else {
      // Fallback: try to extract a name from the object's reflected "name" field
      const TVMFFITypeInfo* info = TVMFFIGetTypeInfo(var_obj->type_index());
      reflection::ForEachFieldInfo(info, [&](const TVMFFIFieldInfo* fi) {
        std::string_view fname(fi->name.data, fi->name.size);
        if (fname == "name") {
          Any field_val;
          const void* addr = reinterpret_cast<const char*>(var_obj.get()) + fi->offset;
          TVM_FFI_CHECK_SAFE_CALL(
              fi->getter(const_cast<void*>(addr), reinterpret_cast<TVMFFIAny*>(&field_val)));
          if (IsString(field_val)) {
            name_hint = field_val.cast<String>();
          }
        }
      });
    }
  }
  IdAST id = printer->VarDef(name_hint, var_obj, frame);
  return AssignAST(std::move(id), Optional<ExprAST>{}, ty_annotation);
}

/*! \brief Define variables from a region's def_values and return param AssignASTs. */
List<AssignAST> DefineRegionVars(const ir_traits::RegionTraits& region, AnyView obj,
                                 const IRPrinter& printer, const AccessPath& path,
                                 const Optional<ObjectRef>& frame) {
  List<AssignAST> params;
  if (!region->def_values.has_value()) return params;
  Any dv = ResolveWithPrinter(region->def_values.value(), obj, printer, GetCurrentFrame(printer));
  // If def_values resolves to a list/array, define each element
  if (dv.type_index() >= TypeIndex::kTVMFFIStaticObjectBegin) {
    ObjectRef dv_obj = dv.cast<ObjectRef>();
    if (IsListLike(dv_obj)) {
      List<Any> items = dv.cast<List<Any>>();
      for (int64_t i = 0; i < static_cast<int64_t>(items.size()); ++i) {
        ObjectRef item = items[i].cast<ObjectRef>();
        params.push_back(PrintValueDef(item, printer, path->Attr("params")->ArrayItem(i), frame));
      }
    } else {
      // Single variable
      params.push_back(PrintValueDef(dv_obj, printer, path->Attr("param"), frame));
    }
  }
  return params;
}

/*! \brief Map unary operator string to OperationAST::Kind. */
OperationASTObj::Kind UnaryOpStringToKind(std::string_view op) {
  static const std::unordered_map<std::string_view, OperationASTObj::Kind> kMap = {
      {"-", OperationASTObj::kUSub},
      {"~", OperationASTObj::kInvert},
      {"not", OperationASTObj::kNot},
      {"+", OperationASTObj::kUAdd},
  };
  auto it = kMap.find(op);
  if (it != kMap.end()) return it->second;
  TVM_FFI_THROW(ValueError) << "Unknown unary operator: " << op;
  TVM_FFI_UNREACHABLE();
}

/*! \brief Call text_printer_pre/post method if present. */
void CallPrinterHook(const Optional<String>& hook_ref, AnyView obj, const IRPrinter& printer,
                     const AccessPath& path, const DefaultFrame& frame) {
  if (!hook_ref.has_value()) return;
  Any hook_fn_val = ResolveWithPrinter(hook_ref.value(), obj, printer, GetCurrentFrame(printer));
  // The hook is expected to be a Function(obj, printer, frame)
  if (hook_fn_val.type_index() >= TypeIndex::kTVMFFIStaticObjectBegin) {
    Function hook_fn = hook_fn_val.cast<Function>();
    Any result;
    AnyView args[3] = {AnyView(obj), AnyView(printer), AnyView(frame)};
    hook_fn.CallPacked(args, 3, &result);
  }
}

/*! \brief Build a range() call AST from For trait fields. */
ExprAST BuildRangeCall(AnyView obj, const ir_traits::ForTraitsObj& trait, const IRPrinter& printer,
                       const AccessPath& path) {
  if (trait.text_printer_kind.has_value()) {
    // Resolve the kind string (e.g. "$method:kind_prefix" → "T.serial" or "")
    Any kind_val =
        ResolveWithPrinter(trait.text_printer_kind.value(), obj, printer, GetCurrentFrame(printer));
    // If it's an empty string, fall through to plain range()
    if (IsString(kind_val)) {
      String kind_str = kind_val.cast<String>();
      if (kind_str.empty()) goto use_range;
    }
    // Custom kind: kind(start, end, step=..., ...)
    ExprAST callee(ffi::UnsafeInit{});
    if (IsString(kind_val)) {
      callee = ParseCalleeString(std::string(kind_val.cast<String>()));
    } else {
      callee = printer->operator()(std::move(kind_val), path->Attr("kind")).cast<ExprAST>();
    }
    List<ExprAST> args;
    // Helper: check if a value is "zero" (raw int 0 or IntImm with value 0)
    auto is_zero_val = [](AnyView v) -> bool {
      if (v.type_index() == TypeIndex::kTVMFFIInt) return v.cast<int64_t>() == 0;
      if (v.type_index() >= TypeIndex::kTVMFFIStaticObjectBegin) {
        Any field = FindFieldValue(v, "value");
        if (field.type_index() == TypeIndex::kTVMFFIInt) return field.cast<int64_t>() == 0;
      }
      return false;
    };
    auto is_one_val = [](AnyView v) -> bool {
      if (v.type_index() == TypeIndex::kTVMFFIInt) return v.cast<int64_t>() == 1;
      if (v.type_index() >= TypeIndex::kTVMFFIStaticObjectBegin) {
        Any field = FindFieldValue(v, "value");
        if (field.type_index() == TypeIndex::kTVMFFIInt) return field.cast<int64_t>() == 1;
      }
      return false;
    };
    auto is_empty_map = [](AnyView v) -> bool {
      if (v.type_index() == TypeIndex::kTVMFFINone) return true;
      if (const auto* m = v.as<MapObj>()) {
        return m->size() == 0;
      }
      return false;
    };

    if (trait.start.has_value()) {
      Any start_val =
          ResolveWithPrinter(trait.start.value(), obj, printer, GetCurrentFrame(printer));
      if (!is_zero_val(start_val)) {
        args.push_back(
            printer->operator()(std::move(start_val), path->Attr("start")).cast<ExprAST>());
      }
    }
    if (trait.end.has_value()) {
      args.push_back(ResolveAndPrint(trait.end.value(), obj, printer, path->Attr("end")));
    }
    List<String> kwarg_keys;
    List<ExprAST> kwarg_values;
    if (trait.step.has_value()) {
      Any step_val = ResolveWithPrinter(trait.step.value(), obj, printer, GetCurrentFrame(printer));
      if (!is_one_val(step_val) && step_val.type_index() != TypeIndex::kTVMFFINone) {
        kwarg_keys.push_back("step");
        kwarg_values.push_back(
            printer->operator()(std::move(step_val), path->Attr("step")).cast<ExprAST>());
      }
    }
    if (trait.attrs.has_value()) {
      Any attrs_val =
          ResolveWithPrinter(trait.attrs.value(), obj, printer, GetCurrentFrame(printer));
      if (!is_empty_map(attrs_val)) {
        kwarg_keys.push_back("annotations");
        kwarg_values.push_back(
            printer->operator()(std::move(attrs_val), path->Attr("attrs")).cast<ExprAST>());
      }
    }
    return CallAST(callee, args, kwarg_keys, kwarg_values);
  }
use_range:

  // Default: range(start?, end, step?)
  List<ExprAST> args;
  bool has_start = false;
  if (trait.start.has_value()) {
    Any start_val = ResolveWithPrinter(trait.start.value(), obj, printer, GetCurrentFrame(printer));
    bool is_zero =
        (start_val.type_index() == TypeIndex::kTVMFFIInt && start_val.cast<int64_t>() == 0);
    if (!is_zero || trait.step.has_value()) {
      args.push_back(
          printer->operator()(std::move(start_val), path->Attr("start")).cast<ExprAST>());
      has_start = true;
    }
  }
  if (trait.end.has_value()) {
    args.push_back(ResolveAndPrint(trait.end.value(), obj, printer, path->Attr("end")));
  }
  if (trait.step.has_value()) {
    Any step_val = ResolveWithPrinter(trait.step.value(), obj, printer, GetCurrentFrame(printer));
    bool is_one = (step_val.type_index() == TypeIndex::kTVMFFIInt && step_val.cast<int64_t>() == 1);
    if (!is_one) {
      if (!has_start && !args.empty()) {
        // Need to insert start=0 before end
        args.insert(args.begin(), LiteralAST::Int(0));
      }
      args.push_back(printer->operator()(std::move(step_val), path->Attr("step")).cast<ExprAST>());
    }
  }
  return CallAST(IdAST("range"), args, {}, {});
}

// ============================================================================
// Expression-level trait printers
// ============================================================================

NodeAST PrintBinOp(AnyView obj, const ir_traits::BinOpTraitsObj& t, const IRPrinter& printer,
                   const AccessPath& path) {
  ExprAST lhs = ResolveAndPrint(t.lhs, obj, printer, path->Attr("lhs"));
  ExprAST rhs = ResolveAndPrint(t.rhs, obj, printer, path->Attr("rhs"));
  std::string_view op_sv(t.op.data(), t.op.size());
  // Check dynamic sugar guard: if text_printer_sugar_check is set and returns false,
  // fall back to function-call form T.OpName(lhs, rhs).
  bool use_sugar = true;
  if (t.text_printer_sugar_check.has_value()) {
    Any check_val = ResolveWithPrinter(t.text_printer_sugar_check.value(), obj, printer,
                                       GetCurrentFrame(printer));
    if (check_val.type_index() == TypeIndex::kTVMFFIBool) {
      use_sugar = check_val.cast<bool>();
    } else if (check_val.type_index() == TypeIndex::kTVMFFIInt) {
      use_sugar = check_val.cast<int64_t>() != 0;
    }
  }
  if (use_sugar) {
    // Try standard Python operator first
    static const std::unordered_map<std::string_view, OperationASTObj::Kind> kMap = {
        {"+", OperationASTObj::kAdd},       {"-", OperationASTObj::kSub},
        {"*", OperationASTObj::kMult},      {"/", OperationASTObj::kDiv},
        {"//", OperationASTObj::kFloorDiv}, {"%", OperationASTObj::kMod},
        {"**", OperationASTObj::kPow},      {"<<", OperationASTObj::kLShift},
        {">>", OperationASTObj::kRShift},   {"&", OperationASTObj::kBitAnd},
        {"|", OperationASTObj::kBitOr},     {"^", OperationASTObj::kBitXor},
        {"<", OperationASTObj::kLt},        {"<=", OperationASTObj::kLtE},
        {"==", OperationASTObj::kEq},       {"!=", OperationASTObj::kNotEq},
        {">", OperationASTObj::kGt},        {">=", OperationASTObj::kGtE},
        {"and", OperationASTObj::kAnd},     {"or", OperationASTObj::kOr},
    };
    auto it = kMap.find(op_sv);
    if (it != kMap.end()) {
      return OperationAST(static_cast<int64_t>(it->second), {lhs, rhs});
    }
  }
  // Non-standard operator or sugar check failed: render as T.FuncName(lhs, rhs)
  // Use text_printer_func_name if available (e.g. "Add", "FloorDiv"), else fall back to op.
  std::string func_name;
  if (t.text_printer_func_name.has_value()) {
    func_name = std::string(t.text_printer_func_name.value());
  } else {
    func_name = std::string(op_sv);
  }
  ExprAST callee = ExprAttr(IdAST("T"), func_name);
  return CallAST(callee, {lhs, rhs}, {}, {});
}

NodeAST PrintUnaryOp(AnyView obj, const ir_traits::UnaryOpTraitsObj& t, const IRPrinter& printer,
                     const AccessPath& path) {
  ExprAST operand = ResolveAndPrint(t.operand, obj, printer, path->Attr("operand"));
  std::string_view op_sv(t.op.data(), t.op.size());
  return OperationAST(static_cast<int64_t>(UnaryOpStringToKind(op_sv)), {operand});
}

NodeAST PrintValueUse(AnyView obj, const ir_traits::ValueTraitsObj& t, const IRPrinter& printer,
                      const AccessPath& path) {
  ObjectRef obj_ref = obj.cast<ObjectRef>();
  // Use-site: just return the variable reference
  Optional<ExprAST> existing = printer->VarGet(obj_ref);
  if (existing.has_value()) return existing.value();
  // Auto-define as free variable if allowed
  if (printer->cfg->def_free_var) {
    Any name_val = ResolveWithPrinter(t.name, obj, printer, GetCurrentFrame(printer));
    String name_hint("v");
    if (IsString(name_val)) {
      name_hint = name_val.cast<String>();
    }
    return printer->VarDef(name_hint, obj_ref, Optional<ObjectRef>{});
  }
  TVM_FFI_THROW(ValueError) << "Undefined variable: " << obj.GetTypeKey();
  TVM_FFI_UNREACHABLE();
}

/*! \brief Resolve an indices reference and return the list of index/slice ExprASTs. */
List<ExprAST> ResolveIndices(const String& indices_ref, AnyView obj, const IRPrinter& printer,
                             const AccessPath& path) {
  Any indices_val = ResolveWithPrinter(indices_ref, obj, printer, GetCurrentFrame(printer));
  List<ExprAST> idx_list;
  if (indices_val.type_index() >= TypeIndex::kTVMFFIStaticObjectBegin) {
    ObjectRef idx_obj = indices_val.cast<ObjectRef>();
    if (IsListLike(idx_obj)) {
      List<Any> items = indices_val.cast<List<Any>>();
      for (int64_t i = 0; i < static_cast<int64_t>(items.size()); ++i) {
        Any item = items[i];
        if (item.type_index() >= TypeIndex::kTVMFFIStaticObjectBegin) {
          ObjectRef item_obj = item.cast<ObjectRef>();
          if (IsListLike(item_obj)) {
            List<Any> pair = item.cast<List<Any>>();
            if (pair.size() == 2 || pair.size() == 3) {
              AccessPath ip = path->Attr("indices")->ArrayItem(i);
              ExprAST start = printer->operator()(pair[0], ip->ArrayItem(0)).cast<ExprAST>();
              ExprAST stop = printer->operator()(pair[1], ip->ArrayItem(1)).cast<ExprAST>();
              Optional<ExprAST> step;
              if (pair.size() == 3) {
                step = printer->operator()(pair[2], ip->ArrayItem(2)).cast<ExprAST>();
              }
              idx_list.push_back(SliceAST(start, stop, step));
              continue;
            }
          }
        }
        idx_list.push_back(
            printer->operator()(items[i], path->Attr("indices")->ArrayItem(i)).cast<ExprAST>());
      }
    } else {
      idx_list.push_back(
          printer->operator()(std::move(indices_val), path->Attr("indices")).cast<ExprAST>());
    }
  } else {
    idx_list.push_back(
        printer->operator()(std::move(indices_val), path->Attr("indices")).cast<ExprAST>());
  }
  return idx_list;
}

NodeAST PrintLoad(AnyView obj, const ir_traits::LoadTraitsObj& t, const IRPrinter& printer,
                  const AccessPath& path) {
  ExprAST source = ResolveAndPrint(t.source, obj, printer, path->Attr("source"));
  if (!t.indices.has_value()) return source;
  List<ExprAST> idx_list = ResolveIndices(t.indices.value(), obj, printer, path);
  if (t.predicate.has_value()) {
    Any pred_val = ResolveWithPrinter(t.predicate.value(), obj, printer, GetCurrentFrame(printer));
    if (pred_val.type_index() != TypeIndex::kTVMFFINone) {
      ExprAST pred_expr =
          printer->operator()(std::move(pred_val), path->Attr("predicate")).cast<ExprAST>();
      List<ExprAST> args;
      args.push_back(TupleAST({}, std::move(idx_list)));
      List<String> kw_keys;
      List<ExprAST> kw_vals;
      kw_keys.push_back(String("predicate"));
      kw_vals.push_back(pred_expr);
      return CallAST(ExprAttr(source, "vload"), std::move(args), kw_keys, kw_vals);
    }
  }
  return IndexAST(source, idx_list);
}

}  // namespace

// ============================================================================
// Statement-level trait printers
// ============================================================================

namespace {

NodeAST PrintStore(AnyView obj, const ir_traits::StoreTraitsObj& t, const IRPrinter& printer,
                   const AccessPath& path) {
  ExprAST target = ResolveAndPrint(t.target, obj, printer, path->Attr("target"));
  ExprAST value = ResolveAndPrint(t.value, obj, printer, path->Attr("value"));
  if (t.indices.has_value()) {
    List<ExprAST> idx_list = ResolveIndices(t.indices.value(), obj, printer, path);
    if (t.predicate.has_value()) {
      Any pred_val =
          ResolveWithPrinter(t.predicate.value(), obj, printer, GetCurrentFrame(printer));
      if (pred_val.type_index() != TypeIndex::kTVMFFINone) {
        ExprAST pred_expr =
            printer->operator()(std::move(pred_val), path->Attr("predicate")).cast<ExprAST>();
        List<ExprAST> args;
        args.push_back(TupleAST({}, std::move(idx_list)));
        args.push_back(value);
        List<String> kw_keys;
        List<ExprAST> kw_vals;
        kw_keys.push_back(String("predicate"));
        kw_vals.push_back(pred_expr);
        return ExprStmtAST(CallAST(ExprAttr(target, "vstore"), std::move(args), kw_keys, kw_vals));
      }
    }
    return AssignAST(IndexAST(target, idx_list), value);
  }
  // target = value
  return AssignAST(target, value);
}

/*! \brief Collect hook-emitted stmts and prepend them before result. */
NodeAST WrapWithHookStmts(NodeAST result, DefaultFrame& frame, int64_t pre_count) {
  int64_t post_count = static_cast<int64_t>(frame->stmts.size());
  if (post_count <= pre_count) return result;
  List<StmtAST> all;
  for (int64_t i = pre_count; i < post_count; ++i) all.push_back(frame->stmts[i]);
  frame->stmts.erase(frame->stmts.begin() + pre_count, frame->stmts.end());
  if (result->IsInstance<StmtBlockASTObj>()) {
    auto blk = GetRef<StmtBlockAST>(result.as<StmtBlockASTObj>());
    all.insert(all.end(), blk->stmts.begin(), blk->stmts.end());
  } else if (result->IsInstance<StmtASTObj>()) {
    all.push_back(GetRef<StmtAST>(result.as<StmtASTObj>()));
  }
  return StmtBlockAST(std::move(all));
}

NodeAST PrintAssign(AnyView obj, const ir_traits::AssignTraitsObj& t, const IRPrinter& printer,
                    const AccessPath& path) {
  // Get current frame for pre/post hooks
  DefaultFrame frame = printer->frames.back().cast<DefaultFrame>();
  int64_t pre_count = static_cast<int64_t>(frame->stmts.size());
  CallPrinterHook(t.text_printer_pre, obj, printer, path, frame);

  // ---- Expression-statement mode (no LHS) ----
  if (!t.def_values.has_value()) {
    ExprAST expr = ResolveAndPrint(t.rhs, obj, printer, path->Attr("rhs"));
    // Check if this should be rendered as a return statement
    if (t.text_printer_return_check.has_value()) {
      Any check_val = ResolveWithPrinter(t.text_printer_return_check.value(), obj, printer,
                                         GetCurrentFrame(printer));
      bool is_return = false;
      if (check_val.type_index() == TypeIndex::kTVMFFIBool) {
        is_return = check_val.cast<bool>();
      } else if (check_val.type_index() == TypeIndex::kTVMFFIInt) {
        is_return = check_val.cast<int64_t>() != 0;
      }
      if (is_return) return ReturnAST(expr);
    }
    // Optional wrapper callee (e.g. "T.evaluate")
    if (t.text_printer_kind.has_value()) {
      std::string_view kind_sv(t.text_printer_kind.value().data(),
                               t.text_printer_kind.value().size());
      if (kind_sv.substr(0, 1) == "$") {
        Any kind_val =
            ResolveWithPrinter(t.text_printer_kind.value(), obj, printer, GetCurrentFrame(printer));
        if (kind_val.type_index() == TypeIndex::kTVMFFINone) return ExprStmtAST(expr);
        if (IsString(kind_val)) {
          String kind_str = kind_val.cast<String>();
          if (kind_str.size() == 0) return ExprStmtAST(expr);
          return ExprStmtAST(CallAST(ParseCalleeString(std::string(kind_str)), {expr}, {}, {}));
        }
      }
      return ExprStmtAST(
          CallAST(ParseCalleeString(std::string(t.text_printer_kind.value())), {expr}, {}, {}));
    }
    return ExprStmtAST(expr);
  }

  // ---- Assignment mode ----
  Any def_vals = ResolveWithPrinter(t.def_values.value(), obj, printer, GetCurrentFrame(printer));
  Any rhs_val = ResolveWithPrinter(t.rhs, obj, printer, GetCurrentFrame(printer));

  // RHS is void/None -> expression statement of def_values
  if (rhs_val.type_index() == TypeIndex::kTVMFFINone) {
    ExprAST printed =
        printer->operator()(std::move(def_vals), path->Attr("target")).cast<ExprAST>();
    CallPrinterHook(t.text_printer_post, obj, printer, path, frame);
    return WrapWithHookStmts(ExprStmtAST(printed), frame, pre_count);
  }

  // Define LHS variable (before printing RHS, so it's in scope)
  ExprAST lhs_expr(ffi::UnsafeInit{});
  Optional<ExprAST> ty_annotation;
  if (def_vals.type_index() >= TypeIndex::kTVMFFIStaticObjectBegin) {
    ObjectRef var_obj = def_vals.cast<ObjectRef>();
    AssignAST def = PrintValueDef(var_obj, printer, path->Attr("target"), Optional<ObjectRef>{});
    lhs_expr = def->lhs;
    ty_annotation = def->annotation;
  } else {
    lhs_expr = printer->operator()(std::move(def_vals), path->Attr("target")).cast<ExprAST>();
  }

  // Print RHS -- may produce ExprAST or StmtAST (e.g. Function, If, StmtBlock)
  Any rhs_result = printer->operator()(std::move(rhs_val), path->Attr("rhs"));

  if (rhs_result.type_index() >= TypeIndex::kTVMFFIStaticObjectBegin) {
    NodeAST rhs_node = rhs_result.cast<NodeAST>();
    // Function RHS: replace function name with the LHS variable name
    if (auto* func = rhs_node.as<FunctionASTObj>()) {
      if (lhs_expr->IsInstance<IdASTObj>()) {
        IdAST lhs_id = GetRef<IdAST>(lhs_expr.as<IdASTObj>());
        NodeAST result =
            FunctionAST(lhs_id, func->args, func->decorators, func->return_type, func->body);
        CallPrinterHook(t.text_printer_post, obj, printer, path, frame);
        return WrapWithHookStmts(std::move(result), frame, pre_count);
      }
    }
    // Statement-level RHS (If, StmtBlock): return directly
    if (rhs_node->IsInstance<IfASTObj>() || rhs_node->IsInstance<StmtBlockASTObj>()) {
      CallPrinterHook(t.text_printer_post, obj, printer, path, frame);
      return WrapWithHookStmts(std::move(rhs_node), frame, pre_count);
    }
  }

  // Normal case: RHS is an expression
  StmtAST result_stmt = AssignAST(lhs_expr, rhs_result.cast<ExprAST>(), ty_annotation);
  CallPrinterHook(t.text_printer_post, obj, printer, path, frame);
  return WrapWithHookStmts(std::move(result_stmt), frame, pre_count);
}

NodeAST PrintAssert(AnyView obj, const ir_traits::AssertTraitsObj& t, const IRPrinter& printer,
                    const AccessPath& path) {
  ExprAST cond = ResolveAndPrint(t.cond, obj, printer, path->Attr("cond"));
  Optional<ExprAST> msg;
  if (t.message.has_value()) {
    Any msg_val = ResolveWithPrinter(t.message.value(), obj, printer, GetCurrentFrame(printer));
    if (msg_val.type_index() >= TypeIndex::kTVMFFIStaticObjectBegin) {
      ObjectRef msg_obj = msg_val.cast<ObjectRef>();
      if (IsListLike(msg_obj)) {
        // $method:structured_msg returns Array — render as tuple where inner Arrays
        // become lists: (error_kind, [part0, part1, ...])
        List<Any> items = msg_val.cast<List<Any>>();
        List<ExprAST> elts;
        for (int64_t i = 0; i < static_cast<int64_t>(items.size()); ++i) {
          Any item = items[i];
          if (item.type_index() >= TypeIndex::kTVMFFIStaticObjectBegin) {
            ObjectRef item_obj = item.cast<ObjectRef>();
            if (IsListLike(item_obj)) {
              // Inner Array → render as ListAST [elem0, elem1, ...]
              List<Any> inner = item.cast<List<Any>>();
              List<ExprAST> inner_elts;
              for (int64_t j = 0; j < static_cast<int64_t>(inner.size()); ++j) {
                inner_elts.push_back(
                    printer->operator()(inner[j], path->Attr("message")->ArrayItem(i)->ArrayItem(j))
                        .cast<ExprAST>());
              }
              elts.push_back(ListAST(std::move(inner_elts)));
              continue;
            }
          }
          elts.push_back(printer->operator()(std::move(item), path->Attr("message")->ArrayItem(i))
                             .cast<ExprAST>());
        }
        msg = TupleAST(std::move(elts));
      } else {
        msg = printer->operator()(std::move(msg_val), path->Attr("message")).cast<ExprAST>();
      }
    } else {
      msg = printer->operator()(std::move(msg_val), path->Attr("message")).cast<ExprAST>();
    }
  }
  return AssertAST(cond, msg);
}

NodeAST PrintReturn(AnyView obj, const ir_traits::ReturnTraitsObj& t, const IRPrinter& printer,
                    const AccessPath& path) {
  ExprAST value = ResolveAndPrint(t.value, obj, printer, path->Attr("value"));
  return ReturnAST(value);
}

}  // namespace

// ============================================================================
// Scope-level trait printers
// ============================================================================

namespace {

NodeAST PrintFunc(AnyView obj, const ir_traits::FuncTraitsObj& t, const IRPrinter& printer,
                  const AccessPath& path) {
  // Resolve symbol name
  Any symbol_val = ResolveWithPrinter(t.symbol, obj, printer, GetCurrentFrame(printer));
  String symbol_str = symbol_val.cast<String>();
  IdAST name = IdAST(symbol_str);

  // Decorators from text_printer_kind
  List<ExprAST> decorators;
  if (t.text_printer_kind.has_value()) {
    Any kind_val =
        ResolveWithPrinter(t.text_printer_kind.value(), obj, printer, GetCurrentFrame(printer));
    if (IsString(kind_val)) {
      decorators.push_back(IdAST(kind_val.cast<String>()));
    } else {
      decorators.push_back(
          printer->operator()(std::move(kind_val), path->Attr("decorator")).cast<ExprAST>());
    }
  }

  // Push frame for function body
  DefaultFrame frame;
  printer->FramePush(frame);

  // Define parameters
  List<AssignAST> params = DefineRegionVars(t.region, obj, printer, path, frame);

  // Call text_printer_pre hook
  CallPrinterHook(t.text_printer_pre, obj, printer, path, frame);

  // Print body
  List<StmtAST> body = PrintBody(t.region->body, obj, printer, path);

  // Handle region.ret
  if (t.region->ret.has_value()) {
    ExprAST ret_val = ResolveAndPrint(t.region->ret.value(), obj, printer, path->Attr("ret"));
    body.push_back(ReturnAST(ret_val));
  }

  // Merge frame stmts into body
  List<StmtAST> all_body;
  all_body.insert(all_body.end(), frame->stmts.begin(), frame->stmts.end());
  all_body.insert(all_body.end(), body.begin(), body.end());

  printer->FramePop();

  // Class-style rendering: no params, no return type → ClassAST
  if (params.empty() && !t.region->ret.has_value()) {
    // Resolve bases from attrs field
    List<ExprAST> bases;
    if (t.attrs.has_value()) {
      Any attrs_val = ResolveWithPrinter(t.attrs.value(), obj, printer, path->Attr("bases"));
      if (attrs_val.type_index() != TypeIndex::kTVMFFINone) {
        bases.push_back(
            printer->operator()(std::move(attrs_val), path->Attr("bases")).cast<ExprAST>());
      }
    }
    return ClassAST(name, bases, decorators, all_body, {}, {});
  }
  return FunctionAST(name, params, decorators, Optional<ExprAST>{}, all_body);
}

NodeAST PrintFor(AnyView obj, const ir_traits::ForTraitsObj& t, const IRPrinter& printer,
                 const AccessPath& path) {
  // Push frame for loop body
  DefaultFrame frame;
  printer->FramePush(frame);

  // Define loop variable
  List<AssignAST> loop_var_defs = DefineRegionVars(t.region, obj, printer, path, frame);

  // Build the LHS (loop variable as expression)
  ExprAST lhs;
  if (loop_var_defs.size() == 1) {
    lhs = loop_var_defs[0]->lhs;
  } else if (loop_var_defs.size() > 1) {
    List<ExprAST> elts;
    for (const auto& d : loop_var_defs) {
      elts.push_back(d->lhs);
    }
    lhs = TupleAST(elts);
  } else {
    lhs = IdAST("_");
  }

  // Build the iterator expression (RHS)
  ExprAST rhs;
  if (t.end.has_value()) {
    rhs = BuildRangeCall(obj, t, printer, path);
  } else if (t.region->def_expr.has_value()) {
    rhs = ResolveAndPrint(t.region->def_expr.value(), obj, printer, path->Attr("iter"));
  } else {
    rhs = CallAST(IdAST("range"), {LiteralAST::Int(0)}, {}, {});
  }

  // Call text_printer_pre hook (no hook for For in the design, but kept for extensibility)

  // Print body
  List<StmtAST> body = PrintBody(t.region->body, obj, printer, path);

  // Carry pattern (def_carry/carry_init) is reserved for future use.

  // Handle region.ret (yield for carry)
  if (t.region->ret.has_value()) {
    ExprAST ret_val = ResolveAndPrint(t.region->ret.value(), obj, printer, path->Attr("ret"));
    body.push_back(ExprStmtAST(YieldAST(ret_val)));
  }

  // Merge frame stmts into body
  List<StmtAST> all_body;
  all_body.insert(all_body.end(), frame->stmts.begin(), frame->stmts.end());
  all_body.insert(all_body.end(), body.begin(), body.end());

  printer->FramePop();

  return ForAST(lhs, rhs, all_body);
}

NodeAST PrintWith(AnyView obj, const ir_traits::WithTraitsObj& t, const IRPrinter& printer,
                  const AccessPath& path) {
  bool no_frame = t.text_printer_no_frame.has_value() && t.text_printer_no_frame.value();
  bool is_inline = !t.text_printer_kind.has_value() && !t.region->def_values.has_value() &&
                   !t.region->def_expr.has_value() && !t.text_printer_pre.has_value() &&
                   !t.text_printer_post.has_value();

  // Inline sequence mode: no with-header, just flatten body elements
  if (is_inline) {
    List<StmtAST> stmts;
    if (no_frame) {
      Any elems_val = ResolveWithPrinter(t.region->body, obj, printer, GetCurrentFrame(printer));
      if (elems_val.type_index() >= TypeIndex::kTVMFFIStaticObjectBegin) {
        ObjectRef elems_obj = elems_val.cast<ObjectRef>();
        if (IsListLike(elems_obj)) {
          List<Any> items = elems_val.cast<List<Any>>();
          for (int64_t i = 0; i < static_cast<int64_t>(items.size()); ++i) {
            Any printed = printer->operator()(items[i], path->Attr("body")->ArrayItem(i));
            if (printed.type_index() >= TypeIndex::kTVMFFIStaticObjectBegin) {
              NodeAST node = printed.cast<NodeAST>();
              if (const auto* block = node.as<StmtBlockASTObj>()) {
                stmts.insert(stmts.end(), block->stmts.begin(), block->stmts.end());
              } else if (node->IsInstance<StmtASTObj>()) {
                stmts.push_back(GetRef<StmtAST>(node.as<StmtASTObj>()));
              } else if (node->IsInstance<ExprASTObj>()) {
                stmts.push_back(ExprStmtAST(GetRef<ExprAST>(node.as<ExprASTObj>())));
              }
            }
          }
        }
      }
    } else {
      stmts = PrintBody(t.region->body, obj, printer, path);
    }
    if (t.region->ret.has_value()) {
      ExprAST ret_val = ResolveAndPrint(t.region->ret.value(), obj, printer, path->Attr("ret"));
      stmts.push_back(ReturnAST(ret_val));
    }
    return StmtBlockAST(std::move(stmts));
  }

  // Build context expression
  ExprAST ctx_expr;
  if (t.text_printer_kind.has_value()) {
    Any kind_val =
        ResolveWithPrinter(t.text_printer_kind.value(), obj, printer, GetCurrentFrame(printer));
    if (IsString(kind_val)) {
      // kind(def_expr_args...)
      ExprAST callee = IdAST(kind_val.cast<String>());
      List<ExprAST> args;
      if (t.region->def_expr.has_value()) {
        args.push_back(
            ResolveAndPrint(t.region->def_expr.value(), obj, printer, path->Attr("def_expr")));
      }
      ctx_expr = CallAST(callee, args, {}, {});
    } else {
      ctx_expr = printer->operator()(std::move(kind_val), path->Attr("kind")).cast<ExprAST>();
    }
  } else if (t.region->def_expr.has_value()) {
    ctx_expr = ResolveAndPrint(t.region->def_expr.value(), obj, printer, path->Attr("def_expr"));
  } else {
    ctx_expr = IdAST("_context");
  }

  // Push frame
  DefaultFrame frame;
  printer->FramePush(frame);

  // Define as-variables
  Optional<ExprAST> as_var;
  List<AssignAST> var_defs = DefineRegionVars(t.region, obj, printer, path, frame);
  if (var_defs.size() == 1) {
    as_var = var_defs[0]->lhs;
  } else if (var_defs.size() > 1) {
    List<ExprAST> elts;
    for (const auto& d : var_defs) {
      elts.push_back(d->lhs);
    }
    as_var = TupleAST(elts);
  }

  // Call text_printer_pre hook
  CallPrinterHook(t.text_printer_pre, obj, printer, path, frame);

  // Print body — when a post hook is present, print body items inline
  // (without a sub-frame) so that variables defined during body printing
  // remain accessible to the post hook via printer->VarGet.
  List<StmtAST> body;
  if (t.text_printer_post.has_value()) {
    Any body_val = ResolveWithPrinter(t.region->body, obj, printer, GetCurrentFrame(printer));
    if (body_val.type_index() >= TypeIndex::kTVMFFIStaticObjectBegin) {
      ObjectRef body_obj = body_val.cast<ObjectRef>();
      if (IsListLike(body_obj)) {
        List<Any> items = body_val.cast<List<Any>>();
        for (int64_t i = 0; i < static_cast<int64_t>(items.size()); ++i) {
          Any printed = printer->operator()(items[i], path->Attr("body")->ArrayItem(i));
          if (printed.type_index() >= TypeIndex::kTVMFFIStaticObjectBegin) {
            NodeAST node = printed.cast<NodeAST>();
            if (const auto* blk = node.as<StmtBlockASTObj>()) {
              body.insert(body.end(), blk->stmts.begin(), blk->stmts.end());
            } else if (node->IsInstance<StmtASTObj>()) {
              body.push_back(GetRef<StmtAST>(node.as<StmtASTObj>()));
            } else if (node->IsInstance<ExprASTObj>()) {
              body.push_back(ExprStmtAST(GetRef<ExprAST>(node.as<ExprASTObj>())));
            }
          }
        }
      } else {
        Any printed = printer->operator()(std::move(body_val), path->Attr("body"));
        if (printed.type_index() >= TypeIndex::kTVMFFIStaticObjectBegin) {
          NodeAST node = printed.cast<NodeAST>();
          if (const auto* blk = node.as<StmtBlockASTObj>()) {
            body.insert(body.end(), blk->stmts.begin(), blk->stmts.end());
          } else if (node->IsInstance<StmtASTObj>()) {
            body.push_back(GetRef<StmtAST>(node.as<StmtASTObj>()));
          } else if (node->IsInstance<ExprASTObj>()) {
            body.push_back(ExprStmtAST(GetRef<ExprAST>(node.as<ExprASTObj>())));
          }
        }
      }
    }
  } else {
    body = PrintBody(t.region->body, obj, printer, path);
  }

  // Snapshot frame stmts before post hook (these are pre-body stmts)
  int64_t pre_stmts_end = static_cast<int64_t>(frame->stmts.size());

  // Call text_printer_post hook (new stmts go after body)
  CallPrinterHook(t.text_printer_post, obj, printer, path, frame);

  // Handle region.ret
  if (t.region->ret.has_value()) {
    ExprAST ret_val = ResolveAndPrint(t.region->ret.value(), obj, printer, path->Attr("ret"));
    body.push_back(ExprStmtAST(YieldAST(ret_val)));
  }

  // Merge: pre-stmts, then body, then post-stmts
  List<StmtAST> all_body;
  all_body.insert(all_body.end(), frame->stmts.begin(), frame->stmts.begin() + pre_stmts_end);
  all_body.insert(all_body.end(), body.begin(), body.end());
  all_body.insert(all_body.end(), frame->stmts.begin() + pre_stmts_end, frame->stmts.end());

  printer->FramePop();

  return WithAST(as_var, ctx_expr, all_body);
}

NodeAST PrintWhile(AnyView obj, const ir_traits::WhileTraitsObj& t, const IRPrinter& printer,
                   const AccessPath& path) {
  ExprAST cond = ResolveAndPrint(t.cond, obj, printer, path->Attr("cond"));

  // Print body
  List<StmtAST> body = PrintBody(t.region->body, obj, printer, path);
  return WhileAST(cond, body);
}

NodeAST PrintIf(AnyView obj, const ir_traits::IfTraitsObj& t, const IRPrinter& printer,
                const AccessPath& path) {
  ExprAST cond = ResolveAndPrint(t.cond, obj, printer, path->Attr("cond"));
  List<StmtAST> then_branch = PrintBody(t.then_region->body, obj, printer, path->Attr("then"));
  List<StmtAST> else_branch;
  if (t.else_region.has_value()) {
    Any else_val =
        ResolveWithPrinter(t.else_region.value()->body, obj, printer, GetCurrentFrame(printer));
    if (else_val.type_index() != TypeIndex::kTVMFFINone) {
      else_branch = PrintBody(t.else_region.value()->body, obj, printer, path->Attr("else"));
    }
  }
  return IfAST(cond, then_branch, else_branch);
}

// ---- Helper: resolve an Any value and flatten it as args list ----
List<ExprAST> ResolveAsArgList(const Any& args_val, const IRPrinter& printer,
                               const AccessPath& path) {
  List<ExprAST> arg_docs;
  if (args_val.type_index() == TypeIndex::kTVMFFINone) return arg_docs;
  if (args_val.type_index() >= TypeIndex::kTVMFFIStaticObjectBegin) {
    ObjectRef args_obj = args_val.cast<ObjectRef>();
    if (IsListLike(args_obj)) {
      auto items = args_val.cast<List<Any>>();
      for (int i = 0; i < static_cast<int>(items.size()); ++i) {
        arg_docs.push_back(printer->operator()(items[i], path->ArrayItem(i)).cast<ExprAST>());
      }
      return arg_docs;
    }
  }
  // Single value: print as one arg
  arg_docs.push_back(printer->operator()(Any(args_val), path).cast<ExprAST>());
  return arg_docs;
}

// ---- Literal: dtype-aware printing ----
// When format="int": reads the object's dtype field. int32→bare number, bool→T.bool(...), else
// T.<dtype>(value). When format="float": reads dtype. void→bare float, else T.<dtype>(value).
// Otherwise: just print the resolved value.
NodeAST PrintLiteral(AnyView obj, const ir_traits::LiteralTraitsObj& t, const IRPrinter& printer,
                     const AccessPath& path) {
  Any value = ResolveWithPrinter(t.value, obj, printer, GetCurrentFrame(printer));
  if (t.format.has_value()) {
    std::string fmt(t.format.value());
    // Try to read dtype from the object itself (IntImm/FloatImm have a dtype field in the
    // PrimExprNode base via the `dtype` DataType field which is part of the object layout)
    Any dtype_any = FindFieldValue(obj, "dtype");
    if (dtype_any.type_index() == TypeIndex::kTVMFFIDataType) {
      DLDataType dtype = dtype_any.cast<DLDataType>();

      if (fmt == "int") {
        int64_t int_val = (value.type_index() == TypeIndex::kTVMFFIInt) ? value.cast<int64_t>() : 0;
        // int32 → bare number
        if (dtype.code == kDLInt && dtype.bits == 32 && dtype.lanes == 1) {
          return LiteralAST::Int(int_val, {path->Attr("value")});
        }
        // bool → T.bool(True/False) (kDLBool=6, bits=8 per DLPack convention)
        if (dtype.code == kDLBool) {
          return CallAST(ExprAttr(IdAST("T"), "bool"),
                         {LiteralAST::Bool(static_cast<bool>(int_val), {path->Attr("value")})}, {},
                         {});
        }
        // other → T.<dtype>(value)
        std::string ds = DLDataTypeToString(dtype);
        return CallAST(ExprAttr(IdAST("T"), ds), {LiteralAST::Int(int_val, {path->Attr("value")})},
                       {}, {});
      }
      if (fmt == "float") {
        double float_val =
            (value.type_index() == TypeIndex::kTVMFFIFloat) ? value.cast<double>() : 0.0;
        // void → bare float
        if (dtype.bits == 0 && dtype.lanes == 0) {
          return LiteralAST::Float(float_val, {path->Attr("value")});
        }
        std::string ds = DLDataTypeToString(dtype);
        return CallAST(ExprAttr(IdAST("T"), ds),
                       {LiteralAST::Float(float_val, {path->Attr("value")})}, {}, {});
      }
    }
  }
  // Default: just print the resolved value
  return printer->operator()(std::move(value), path->Attr("value")).cast<ExprAST>();
}

// ---- Seq: flatten elements as a StmtBlock ----
// ---- Call: print op(args...) ----
// The `op` field can be a $field/$method reference OR a literal callee string
// like "I.GlobalVar" or "T.Ramp".
NodeAST PrintCall(AnyView obj, const ir_traits::CallTraitsObj& t, const IRPrinter& printer,
                  const AccessPath& path) {
  // Call text_printer_pre hook if set
  if (t.text_printer_pre.has_value()) {
    DefaultFrame frame = printer->frames.back().cast<DefaultFrame>();
    CallPrinterHook(t.text_printer_pre, obj, printer, path, frame);
  }
  // Resolve callee -- dynamic override takes priority, but falls through to t.op if None
  ExprAST callee(ffi::UnsafeInit{});
  bool callee_resolved = false;
  if (t.text_printer_callee.has_value()) {
    Any callee_val =
        ResolveWithPrinter(t.text_printer_callee.value(), obj, printer, GetCurrentFrame(printer));
    if (IsString(callee_val)) {
      callee = ParseCalleeString(std::string(callee_val.cast<String>()));
      callee_resolved = true;
    } else if (callee_val.type_index() != TypeIndex::kTVMFFINone) {
      // Check if the value is already an ExprAST (e.g. from $printer: method returning IdAST)
      if (callee_val.type_index() >= TypeIndex::kTVMFFIStaticObjectBegin) {
        ObjectRef callee_obj = callee_val.cast<ObjectRef>();
        if (callee_obj->IsInstance<ExprASTObj>()) {
          callee = GetRef<ExprAST>(callee_obj.as<ExprASTObj>());
          callee_resolved = true;
        }
      }
      if (!callee_resolved) {
        callee = printer->operator()(std::move(callee_val), path->Attr("op")).cast<ExprAST>();
        callee_resolved = true;
      }
    }
    // If None, fall through to t.op below
  }
  if (!callee_resolved) {
    std::string_view op_sv(t.op.data(), t.op.size());
    if (op_sv.substr(0, 1) == "$") {
      // $field: or $method: reference
      Any op_val = ResolveWithPrinter(t.op, obj, printer, GetCurrentFrame(printer));
      if (IsString(op_val)) {
        callee = ParseCalleeString(std::string(op_val.cast<String>()));
      } else {
        callee = printer->operator()(std::move(op_val), path->Attr("op")).cast<ExprAST>();
      }
    } else {
      // Literal callee string like "I.GlobalVar" or "T.Ramp"
      callee = ParseCalleeString(std::string(op_sv));
    }
  }
  // Resolve args
  Any args_val = ResolveWithPrinter(t.args, obj, printer, GetCurrentFrame(printer));
  List<ExprAST> arg_docs = ResolveAsArgList(args_val, printer, path->Attr("args"));
  // Handle attrs (single named attribute)
  List<String> kw_keys;
  List<ExprAST> kw_vals;
  if (t.attrs.has_value()) {
    Any attrs_val = ResolveWithPrinter(t.attrs.value(), obj, printer, GetCurrentFrame(printer));
    if (attrs_val.type_index() != TypeIndex::kTVMFFINone) {
      kw_keys.push_back(String("attrs"));
      kw_vals.push_back(
          printer->operator()(std::move(attrs_val), path->Attr("attrs")).cast<ExprAST>());
    }
  }
  // Handle kwargs ($method: returning Map/Dict with String keys)
  if (t.kwargs.has_value()) {
    Any kwargs_val = ResolveWithPrinter(t.kwargs.value(), obj, printer, GetCurrentFrame(printer));
    if (kwargs_val.type_index() >= TypeIndex::kTVMFFIStaticObjectBegin) {
      ObjectRef kwargs_obj = kwargs_val.cast<ObjectRef>();
      // Check if it's any map-like container (ffi.Map or ffi.Dict)
      if (const auto* map_base = kwargs_obj.as<MapBaseObj>()) {
        for (const auto& kv : *map_base) {
          kw_keys.push_back(kv.first.cast<String>());
          kw_vals.push_back(
              printer->operator()(Any(kv.second), path->Attr("kwargs")).cast<ExprAST>());
        }
      }
    }
  }
  if (!kw_keys.empty()) {
    return CallAST(callee, std::move(arg_docs), std::move(kw_keys), std::move(kw_vals));
  }
  return CallAST(callee, std::move(arg_docs), {}, {});
}

// ---- Type trait printers ----

NodeAST PrintPrimTy(AnyView obj, const ir_traits::PrimTyTraitsObj& t, const IRPrinter& printer,
                    const AccessPath& path) {
  Any dtype_val = ResolveWithPrinter(t.dtype, obj, printer, GetCurrentFrame(printer));
  // If it's a DataType, render as T.<dtype>
  if (dtype_val.type_index() == TypeIndex::kTVMFFIDataType) {
    DLDataType dtype = dtype_val.cast<DLDataType>();
    std::string s = (dtype.bits == 0 && dtype.lanes == 0) ? "void" : DLDataTypeToString(dtype);
    return ExprAttr(IdAST("T"), s);
  }
  // Fallback: resolve and print
  return ResolveAndPrint(t.dtype, obj, printer, path->Attr("dtype"));
}

NodeAST PrintTupleTy(AnyView obj, const ir_traits::TupleTyTraitsObj& t, const IRPrinter& printer,
                     const AccessPath& path) {
  Any fields_val = ResolveWithPrinter(t.fields, obj, printer, GetCurrentFrame(printer));
  List<ExprAST> field_docs = ResolveAsArgList(fields_val, printer, path->Attr("fields"));
  if (field_docs.empty()) {
    return LiteralAST::Null({path});
  }
  return CallAST(ExprAttr(IdAST("T"), "Tuple"), std::move(field_docs), {}, {});
}

NodeAST PrintFuncTy(AnyView obj, const ir_traits::FuncTyTraitsObj& t, const IRPrinter& printer,
                    const AccessPath& path) {
  List<ExprAST> args;
  if (t.params.has_value()) {
    args.push_back(ResolveAndPrint(t.params.value(), obj, printer, path->Attr("params")));
  }
  if (t.ret.has_value()) {
    args.push_back(ResolveAndPrint(t.ret.value(), obj, printer, path->Attr("ret")));
  }
  return CallAST(ExprAttr(IdAST("I"), "FuncType"), std::move(args), {}, {});
}

// ---- BufferTy: T.Buffer(shape, dtype, ...) ----
NodeAST PrintBufferTy(AnyView obj, const ir_traits::BufferTyTraitsObj& t, const IRPrinter& printer,
                      const AccessPath& path) {
  // If buffer is already defined as a variable, return the var ref
  Optional<ExprAST> existing = printer->VarGet(obj.cast<ObjectRef>());
  if (existing.has_value()) return existing.value();

  // Resolve shape
  ExprAST shape_expr = ResolveAndPrint(t.shape, obj, printer, path->Attr("shape"));

  // Resolve dtype -- render as a string
  Any dtype_val = ResolveWithPrinter(t.dtype, obj, printer, GetCurrentFrame(printer));
  ExprAST dtype_expr(ffi::UnsafeInit{});
  if (dtype_val.type_index() == TypeIndex::kTVMFFIDataType) {
    DLDataType dtype = dtype_val.cast<DLDataType>();
    std::string ds = DLDataTypeToString(dtype);
    dtype_expr = LiteralAST::Str(ds);
  } else {
    dtype_expr = printer->operator()(std::move(dtype_val), path->Attr("dtype")).cast<ExprAST>();
  }

  ExprAST callee = ExprAttr(IdAST("T"), "Buffer");
  List<ExprAST> args;
  args.push_back(shape_expr);
  args.push_back(dtype_expr);

  // Optional kwargs: strides, offset, scope -- elide defaults
  List<String> kw_keys;
  List<ExprAST> kw_vals;

  if (t.strides.has_value()) {
    Any strides_val = ResolveWithPrinter(t.strides.value(), obj, printer, GetCurrentFrame(printer));
    // Skip empty strides (None, or empty array/list)
    bool is_default = (strides_val.type_index() == TypeIndex::kTVMFFINone);
    if (!is_default && strides_val.type_index() >= TypeIndex::kTVMFFIStaticObjectBegin) {
      ObjectRef strides_obj = strides_val.cast<ObjectRef>();
      if (IsListLike(strides_obj)) {
        List<Any> items = strides_val.cast<List<Any>>();
        is_default = items.empty();
      }
    }
    if (!is_default) {
      kw_keys.push_back(String("strides"));
      kw_vals.push_back(
          printer->operator()(std::move(strides_val), path->Attr("strides")).cast<ExprAST>());
    }
  }

  if (t.offset.has_value()) {
    Any offset_val = ResolveWithPrinter(t.offset.value(), obj, printer, GetCurrentFrame(printer));
    // Skip zero offset
    bool is_default = (offset_val.type_index() == TypeIndex::kTVMFFINone);
    if (!is_default && offset_val.type_index() == TypeIndex::kTVMFFIInt) {
      is_default = (offset_val.cast<int64_t>() == 0);
    }
    if (!is_default) {
      kw_keys.push_back(String("elem_offset"));
      kw_vals.push_back(
          printer->operator()(std::move(offset_val), path->Attr("offset")).cast<ExprAST>());
    }
  }

  if (t.scope.has_value()) {
    Any scope_val = ResolveWithPrinter(t.scope.value(), obj, printer, GetCurrentFrame(printer));
    // Skip default scope "global" or empty
    bool is_default = (scope_val.type_index() == TypeIndex::kTVMFFINone);
    if (!is_default) {
      if (IsString(scope_val)) {
        String scope_str = scope_val.cast<String>();
        is_default = (std::string_view(scope_str.data(), scope_str.size()) == "global" ||
                      scope_str.size() == 0);
      }
    }
    if (!is_default) {
      kw_keys.push_back(String("scope"));
      kw_vals.push_back(
          printer->operator()(std::move(scope_val), path->Attr("scope")).cast<ExprAST>());
    }
  }

  return CallAST(callee, args, kw_keys, kw_vals);
}

}  // namespace

// ============================================================================
// TraitPrint — dispatch on trait runtime type
// ============================================================================

NodeAST TraitPrint(AnyView obj_view, const ObjectRef& trait_ref, IRPrinter printer,
                   AccessPath path) {
  ObjectRef obj = obj_view.cast<ObjectRef>();
  ir_traits::IRTraits trait = AnyView(trait_ref).cast<ir_traits::IRTraits>();
  // Type traits: if obj is already defined in the printer, return its name.
  if (trait->IsInstance<ir_traits::TyTraitsObj>()) {
    if (auto doc = printer->VarGet(obj)) {
      return doc.value();
    }
    if (auto* t = trait.as<ir_traits::PrimTyTraitsObj>()) {
      return PrintPrimTy(obj, *t, printer, path);
    }
    if (auto* t = trait.as<ir_traits::TupleTyTraitsObj>()) {
      return PrintTupleTy(obj, *t, printer, path);
    }
    if (auto* t = trait.as<ir_traits::FuncTyTraitsObj>()) {
      return PrintFuncTy(obj, *t, printer, path);
    }
    if (auto* t = trait.as<ir_traits::BufferTyTraitsObj>()) {
      return PrintBufferTy(obj, *t, printer, path);
    }
    return DefaultPrint(std::move(obj), std::move(printer), std::move(path));
  }
  // Expression traits
  if (auto* t = trait.as<ir_traits::BinOpTraitsObj>()) return PrintBinOp(obj, *t, printer, path);
  if (auto* t = trait.as<ir_traits::UnaryOpTraitsObj>()) {
    return PrintUnaryOp(obj, *t, printer, path);
  }
  if (auto* t = trait.as<ir_traits::ValueTraitsObj>()) return PrintValueUse(obj, *t, printer, path);
  if (auto* t = trait.as<ir_traits::LiteralTraitsObj>()) {
    return PrintLiteral(obj, *t, printer, path);
  }
  if (auto* t = trait.as<ir_traits::CallTraitsObj>()) return PrintCall(obj, *t, printer, path);
  if (auto* t = trait.as<ir_traits::LoadTraitsObj>()) return PrintLoad(obj, *t, printer, path);
  // Statement traits
  if (auto* t = trait.as<ir_traits::StoreTraitsObj>()) return PrintStore(obj, *t, printer, path);
  if (auto* t = trait.as<ir_traits::AssignTraitsObj>()) return PrintAssign(obj, *t, printer, path);
  if (auto* t = trait.as<ir_traits::AssertTraitsObj>()) return PrintAssert(obj, *t, printer, path);
  if (auto* t = trait.as<ir_traits::ReturnTraitsObj>()) return PrintReturn(obj, *t, printer, path);
  if (auto* t = trait.as<ir_traits::FuncTraitsObj>()) return PrintFunc(obj, *t, printer, path);
  if (auto* t = trait.as<ir_traits::ForTraitsObj>()) return PrintFor(obj, *t, printer, path);
  if (auto* t = trait.as<ir_traits::WithTraitsObj>()) return PrintWith(obj, *t, printer, path);
  if (auto* t = trait.as<ir_traits::WhileTraitsObj>()) return PrintWhile(obj, *t, printer, path);
  if (auto* t = trait.as<ir_traits::IfTraitsObj>()) return PrintIf(obj, *t, printer, path);
  return DefaultPrint(std::move(obj), std::move(printer), std::move(path));
}

// ============================================================================
// DefaultPrint — Level 0 printer: TypeKey(field1=val1, field2=val2, ...)
// ============================================================================

NodeAST DefaultPrint(ObjectRef obj, IRPrinter printer, AccessPath path) {
  int32_t type_index = obj->type_index();
  const TVMFFITypeInfo* info = TVMFFIGetTypeInfo(type_index);
  String type_key(info->type_key.data, info->type_key.size);

  ExprAST callee = IdAST(type_key);
  List<String> kwarg_keys;
  List<ExprAST> kwarg_values;

  // Collect all fields (including inherited) via ForEachFieldInfo
  reflection::ForEachFieldInfo(info, [&](const TVMFFIFieldInfo* fi) {
    String name(fi->name.data, fi->name.size);
    Any field_val;
    const void* addr = reinterpret_cast<const char*>(obj.get()) + fi->offset;
    TVM_FFI_CHECK_SAFE_CALL(
        fi->getter(const_cast<void*>(addr), reinterpret_cast<TVMFFIAny*>(&field_val)));
    ExprAST printed = printer->operator()(std::move(field_val), path->Attr(name)).cast<ExprAST>();
    kwarg_keys.push_back(name);
    kwarg_values.push_back(printed);
  });

  return CallAST(callee, {}, kwarg_keys, kwarg_values);
}

}  // namespace pyast
}  // namespace ffi
}  // namespace tvm

namespace tvm {
namespace ffi {
namespace ir_traits {

/************** ObjectDef Registrations **************/

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = ::tvm::ffi::reflection;
  namespace tr = ::tvm::ffi::ir_traits;

  // Ensure the __ffi_ir_traits__ column exists
  refl::EnsureTypeAttrColumn("__ffi_ir_traits__");

  // Region
  refl::ObjectDef<tr::RegionTraitsObj>()
      .def_ro("body", &tr::RegionTraitsObj::body)
      .def_ro("def_values", &tr::RegionTraitsObj::def_values)
      .def_ro("def_expr", &tr::RegionTraitsObj::def_expr)
      .def_ro("ret", &tr::RegionTraitsObj::ret)
      .def(refl::init<String, Optional<String>, Optional<String>, Optional<String>>());

  // Trait (base, no init)
  {
    refl::ObjectDef<tr::IRTraitsObj> def(refl::init(false));
  }
  {
    refl::ObjectDef<tr::ExprTraitsObj> def(refl::init(false));
  }
  {
    refl::ObjectDef<tr::StmtTraitsObj> def(refl::init(false));
  }

  // BinOp
  refl::ObjectDef<tr::BinOpTraitsObj>()
      .def_ro("lhs", &tr::BinOpTraitsObj::lhs)
      .def_ro("rhs", &tr::BinOpTraitsObj::rhs)
      .def_ro("op", &tr::BinOpTraitsObj::op)
      .def_ro("text_printer_sugar_check", &tr::BinOpTraitsObj::text_printer_sugar_check)
      .def_ro("text_printer_func_name", &tr::BinOpTraitsObj::text_printer_func_name)
      .def(refl::init<String, String, String, Optional<String>, Optional<String>>());

  // UnaryOp
  refl::ObjectDef<tr::UnaryOpTraitsObj>()
      .def_ro("operand", &tr::UnaryOpTraitsObj::operand)
      .def_ro("op", &tr::UnaryOpTraitsObj::op)
      .def(refl::init<String, String>());

  // Value
  refl::ObjectDef<tr::ValueTraitsObj>()
      .def_ro("name", &tr::ValueTraitsObj::name)
      .def_ro("ty", &tr::ValueTraitsObj::ty)
      .def_ro("text_printer_type", &tr::ValueTraitsObj::text_printer_type)
      .def(refl::init<String, Optional<String>, Optional<String>>());

  // Assign
  refl::ObjectDef<tr::AssignTraitsObj>()
      .def_ro("def_values", &tr::AssignTraitsObj::def_values)
      .def_ro("rhs", &tr::AssignTraitsObj::rhs)
      .def_ro("text_printer_pre", &tr::AssignTraitsObj::text_printer_pre)
      .def_ro("text_printer_post", &tr::AssignTraitsObj::text_printer_post)
      .def_ro("text_printer_kind", &tr::AssignTraitsObj::text_printer_kind)
      .def_ro("text_printer_return_check", &tr::AssignTraitsObj::text_printer_return_check)
      .def(refl::init<Optional<String>, String, Optional<String>, Optional<String>,
                      Optional<String>, Optional<String>>());

  // Load
  refl::ObjectDef<tr::LoadTraitsObj>()
      .def_ro("source", &tr::LoadTraitsObj::source)
      .def_ro("indices", &tr::LoadTraitsObj::indices)
      .def_ro("predicate", &tr::LoadTraitsObj::predicate)
      .def(refl::init<String, Optional<String>, Optional<String>>());

  // Store
  refl::ObjectDef<tr::StoreTraitsObj>()
      .def_ro("target", &tr::StoreTraitsObj::target)
      .def_ro("value", &tr::StoreTraitsObj::value)
      .def_ro("indices", &tr::StoreTraitsObj::indices)
      .def_ro("predicate", &tr::StoreTraitsObj::predicate)
      .def(refl::init<String, String, Optional<String>, Optional<String>>());

  // Assert
  refl::ObjectDef<tr::AssertTraitsObj>()
      .def_ro("cond", &tr::AssertTraitsObj::cond)
      .def_ro("message", &tr::AssertTraitsObj::message)
      .def(refl::init<String, Optional<String>>());

  // Return
  refl::ObjectDef<tr::ReturnTraitsObj>()
      .def_ro("value", &tr::ReturnTraitsObj::value)
      .def(refl::init<String>());

  // Func
  refl::ObjectDef<tr::FuncTraitsObj>()
      .def_ro("symbol", &tr::FuncTraitsObj::symbol)
      .def_ro("region", &tr::FuncTraitsObj::region)
      .def_ro("attrs", &tr::FuncTraitsObj::attrs)
      .def_ro("text_printer_kind", &tr::FuncTraitsObj::text_printer_kind)
      .def_ro("text_printer_pre", &tr::FuncTraitsObj::text_printer_pre)
      .def(
          refl::init<String, RegionTraits, Optional<String>, Optional<String>, Optional<String>>());

  // For
  refl::ObjectDef<tr::ForTraitsObj>()
      .def_ro("region", &tr::ForTraitsObj::region)
      .def_ro("start", &tr::ForTraitsObj::start)
      .def_ro("end", &tr::ForTraitsObj::end)
      .def_ro("step", &tr::ForTraitsObj::step)
      .def_ro("def_carry", &tr::ForTraitsObj::def_carry)
      .def_ro("carry_init", &tr::ForTraitsObj::carry_init)
      .def_ro("attrs", &tr::ForTraitsObj::attrs)
      .def_ro("text_printer_kind", &tr::ForTraitsObj::text_printer_kind)
      .def(refl::init<RegionTraits, Optional<String>, Optional<String>, Optional<String>,
                      Optional<String>, Optional<String>, Optional<String>, Optional<String>>());

  // With
  refl::ObjectDef<tr::WithTraitsObj>()
      .def_ro("region", &tr::WithTraitsObj::region)
      .def_ro("def_carry", &tr::WithTraitsObj::def_carry)
      .def_ro("carry_init", &tr::WithTraitsObj::carry_init)
      .def_ro("text_printer_kind", &tr::WithTraitsObj::text_printer_kind)
      .def_ro("text_printer_pre", &tr::WithTraitsObj::text_printer_pre)
      .def_ro("text_printer_post", &tr::WithTraitsObj::text_printer_post)
      .def_ro("text_printer_no_frame", &tr::WithTraitsObj::text_printer_no_frame)
      .def(refl::init<RegionTraits, Optional<String>, Optional<String>, Optional<String>,
                      Optional<String>, Optional<String>, Optional<bool>>());

  // While
  refl::ObjectDef<tr::WhileTraitsObj>()
      .def_ro("cond", &tr::WhileTraitsObj::cond)
      .def_ro("region", &tr::WhileTraitsObj::region)
      .def(refl::init<String, RegionTraits>());

  // If
  refl::ObjectDef<tr::IfTraitsObj>()
      .def_ro("cond", &tr::IfTraitsObj::cond)
      .def_ro("then_region", &tr::IfTraitsObj::then_region)
      .def_ro("else_region", &tr::IfTraitsObj::else_region)
      .def(refl::init<String, RegionTraits, Optional<RegionTraits>>());

  // Literal
  refl::ObjectDef<tr::LiteralTraitsObj>()
      .def_ro("value", &tr::LiteralTraitsObj::value)
      .def_ro("format", &tr::LiteralTraitsObj::format)
      .def(refl::init<String, Optional<String>>());

  // Call
  refl::ObjectDef<tr::CallTraitsObj>()
      .def_ro("op", &tr::CallTraitsObj::op)
      .def_ro("args", &tr::CallTraitsObj::args)
      .def_ro("attrs", &tr::CallTraitsObj::attrs)
      .def_ro("kwargs", &tr::CallTraitsObj::kwargs)
      .def_ro("text_printer_callee", &tr::CallTraitsObj::text_printer_callee)
      .def_ro("text_printer_pre", &tr::CallTraitsObj::text_printer_pre)
      .def(refl::init<String, String, Optional<String>, Optional<String>, Optional<String>>());

  // Ty (base, no init)
  {
    refl::ObjectDef<tr::TyTraitsObj> def(refl::init(false));
  }

  // TensorTy
  refl::ObjectDef<tr::TensorTyTraitsObj>()
      .def_ro("shape", &tr::TensorTyTraitsObj::shape)
      .def_ro("dtype", &tr::TensorTyTraitsObj::dtype)
      .def_ro("device", &tr::TensorTyTraitsObj::device)
      .def(refl::init<Optional<String>, Optional<String>, Optional<String>>());

  // BufferTy
  refl::ObjectDef<tr::BufferTyTraitsObj>()
      .def_ro("shape", &tr::BufferTyTraitsObj::shape)
      .def_ro("dtype", &tr::BufferTyTraitsObj::dtype)
      .def_ro("strides", &tr::BufferTyTraitsObj::strides)
      .def_ro("offset", &tr::BufferTyTraitsObj::offset)
      .def_ro("scope", &tr::BufferTyTraitsObj::scope)
      .def(refl::init<String, String, Optional<String>, Optional<String>, Optional<String>>());

  // PrimTy
  refl::ObjectDef<tr::PrimTyTraitsObj>()
      .def_ro("dtype", &tr::PrimTyTraitsObj::dtype)
      .def(refl::init<String>());

  // FuncTy
  refl::ObjectDef<tr::FuncTyTraitsObj>()
      .def_ro("params", &tr::FuncTyTraitsObj::params)
      .def_ro("ret", &tr::FuncTyTraitsObj::ret)
      .def(refl::init<Optional<String>, Optional<String>>());

  // TupleTy
  refl::ObjectDef<tr::TupleTyTraitsObj>()
      .def_ro("fields", &tr::TupleTyTraitsObj::fields)
      .def(refl::init<String>());

  // ShapeTy
  refl::ObjectDef<tr::ShapeTyTraitsObj>()
      .def_ro("dims", &tr::ShapeTyTraitsObj::dims)
      .def_ro("ndim", &tr::ShapeTyTraitsObj::ndim)
      .def(refl::init<Optional<String>, Optional<String>>());
}

}  // namespace ir_traits
}  // namespace ffi
}  // namespace tvm
