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
 * \file tvm/ffi/extra/ir_traits.h
 * \brief Merged header: semantic IR trait types + trait-driven printer declarations.
 *
 * Traits describe WHAT each IR node IS (a loop, a value, a function)
 * and are consumed by tools such as the text printer, optimizers,
 * and visualizers.  Trait objects are immutable FFI objects stored
 * in TypeAttrColumn("__ffi_ir_traits__").
 *
 * Each concrete trait type registers a "__ffi_traits_print__" TypeAttrColumn
 * handler used by the text printer for dispatch.
 */
#ifndef TVM_FFI_EXTRA_IR_TRAITS_H_
#define TVM_FFI_EXTRA_IR_TRAITS_H_

#include <tvm/ffi/any.h>
#include <tvm/ffi/cast.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/object.h>
#include <tvm/ffi/optional.h>
#include <tvm/ffi/reflection/accessor.h>
#include <tvm/ffi/string.h>

namespace tvm {
namespace ffi {
namespace ir_traits {

/************** Helpers **************/

/*! \brief Convert nullable const char* to Optional<String>. */
inline Optional<String> ToOptString(const char* s) {
  return s ? Optional<String>(String(s)) : Optional<String>();
}

/************** IRTraits (base) **************/

/*! \brief Base class for all semantic traits. */
struct IRTraitsObj : public Object {
  /// \cond Doxygen_Suppress
  TVM_FFI_DECLARE_OBJECT_INFO("ffi.ir_traits.IRTraits", IRTraitsObj, Object);
  /// \endcond
};

/// \cond Doxygen_Suppress
/*! \brief Reference wrapper for IRTraitsObj. */
struct IRTraits : public ObjectRef {
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(IRTraits, ObjectRef, IRTraitsObj);
};

/************** ExprTraits / StmtTraits (intermediate) **************/

/*! \brief Base class for expression-level traits (produce a value). */
struct ExprTraitsObj : public IRTraitsObj {
  /// \cond Doxygen_Suppress
  TVM_FFI_DECLARE_OBJECT_INFO("ffi.ir_traits.ExprTraits", ExprTraitsObj, IRTraitsObj);
  /// \endcond
};

/// \cond Doxygen_Suppress
struct ExprTraits : public IRTraits {
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(ExprTraits, IRTraits, ExprTraitsObj);
};
/// \endcond

/*! \brief Base class for statement-level traits (side effects, control flow, scope). */
struct StmtTraitsObj : public IRTraitsObj {
  /// \cond Doxygen_Suppress
  TVM_FFI_DECLARE_OBJECT_INFO("ffi.ir_traits.StmtTraits", StmtTraitsObj, IRTraitsObj);
  /// \endcond
};

/// \cond Doxygen_Suppress
struct StmtTraits : public IRTraits {
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(StmtTraits, IRTraits, StmtTraitsObj);
};
/// \endcond
/// \endcond

/************** Region **************/

/*! \brief Embedded scope descriptor for body regions. */
struct RegionTraitsObj : public IRTraitsObj {
  /*! \brief Body reference ("$field:..." or "$method:..."). */
  String body;
  /*! \brief Variables defined before body. */
  Optional<String> def_values;
  /*! \brief Expression that binds def_values. */
  Optional<String> def_expr;
  /*! \brief Return / yield value. */
  Optional<String> ret;
  /// \cond Doxygen_Suppress
  explicit RegionTraitsObj(String body, Optional<String> def_values, Optional<String> def_expr,
                           Optional<String> ret)
      : body(std::move(body)),
        def_values(std::move(def_values)),
        def_expr(std::move(def_expr)),
        ret(std::move(ret)) {}
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ffi.ir_traits.RegionTraits", RegionTraitsObj, IRTraitsObj);
  /// \endcond
};

/// \cond Doxygen_Suppress
/*! \brief Reference wrapper for RegionTraitsObj. */
struct RegionTraits : public IRTraits {
  explicit RegionTraits(String body, Optional<String> def_values = {},
                        Optional<String> def_expr = {}, Optional<String> ret = {})
      : RegionTraits(make_object<RegionTraitsObj>(std::move(body), std::move(def_values),
                                                  std::move(def_expr), std::move(ret))) {}
  explicit RegionTraits(const char* body, const char* def_values = nullptr,
                        const char* def_expr = nullptr, const char* ret = nullptr)
      : RegionTraits(String(body), ToOptString(def_values), ToOptString(def_expr),
                     ToOptString(ret)) {}
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(RegionTraits, IRTraits, RegionTraitsObj);
};
/// \endcond

/************** Expression traits **************/

/*! \brief Binary operation trait. */
struct BinOpTraitsObj : public ExprTraitsObj {
  /*! \brief Left-hand side reference. */
  String lhs;
  /*! \brief Right-hand side reference. */
  String rhs;
  /*! \brief Operator string ("+", "-", etc.). */
  String op;
  /*! \brief When set, $method: returning bool. If false, fall back to T.OpName(a, b) form. */
  Optional<String> text_printer_sugar_check;
  /*! \brief Fallback function name (e.g. "Add", "FloorDiv") when sugar check fails. */
  Optional<String> text_printer_func_name;
  /// \cond Doxygen_Suppress
  explicit BinOpTraitsObj(String lhs, String rhs, String op,
                          Optional<String> text_printer_sugar_check = {},
                          Optional<String> text_printer_func_name = {})
      : lhs(std::move(lhs)),
        rhs(std::move(rhs)),
        op(std::move(op)),
        text_printer_sugar_check(std::move(text_printer_sugar_check)),
        text_printer_func_name(std::move(text_printer_func_name)) {}
  explicit BinOpTraitsObj(const char* lhs, const char* rhs, const char* op,
                          const char* sugar_check = nullptr, const char* func_name = nullptr)
      : lhs(lhs),
        rhs(rhs),
        op(op),
        text_printer_sugar_check(ToOptString(sugar_check)),
        text_printer_func_name(ToOptString(func_name)) {}
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ffi.ir_traits.BinOpTraits", BinOpTraitsObj, ExprTraitsObj);
  /// \endcond
};
/// \cond Doxygen_Suppress
struct BinOpTraits : public ExprTraits {
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(BinOpTraits, ExprTraits, BinOpTraitsObj);
};
/// \endcond

/*! \brief Unary operation trait. */
struct UnaryOpTraitsObj : public ExprTraitsObj {
  /*! \brief Operand reference. */
  String operand;
  /*! \brief Operator string ("-", "~", "not"). */
  String op;
  /// \cond Doxygen_Suppress
  explicit UnaryOpTraitsObj(String operand, String op)
      : operand(std::move(operand)), op(std::move(op)) {}
  explicit UnaryOpTraitsObj(const char* operand, const char* op) : operand(operand), op(op) {}
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ffi.ir_traits.UnaryOpTraits", UnaryOpTraitsObj, ExprTraitsObj);
  /// \endcond
};
/// \cond Doxygen_Suppress
struct UnaryOpTraits : public ExprTraits {
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(UnaryOpTraits, ExprTraits, UnaryOpTraitsObj);
};
/// \endcond

/*! \brief SSA variable trait. */
struct ValueTraitsObj : public ExprTraitsObj {
  /*! \brief Name reference. */
  String name;
  /*! \brief Type annotation reference. */
  Optional<String> ty;
  /*! \brief Custom type rendering at def site. */
  Optional<String> text_printer_type;
  /// \cond Doxygen_Suppress
  explicit ValueTraitsObj(String name, Optional<String> ty, Optional<String> text_printer_type)
      : name(std::move(name)), ty(std::move(ty)), text_printer_type(std::move(text_printer_type)) {}
  explicit ValueTraitsObj(const char* name, const char* ty = nullptr,
                          const char* text_printer_type = nullptr)
      : name(name), ty(ToOptString(ty)), text_printer_type(ToOptString(text_printer_type)) {}
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ffi.ir_traits.ValueTraits", ValueTraitsObj, ExprTraitsObj);
  /// \endcond
};
/// \cond Doxygen_Suppress
struct ValueTraits : public ExprTraits {
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(ValueTraits, ExprTraits, ValueTraitsObj);
};
/// \endcond

/************** Statement traits **************/

/*!
 * \brief Assignment / binding / expression-statement trait.
 *
 * When def_values is set: renders as `def_values = rhs` (assignment).
 * When def_values is absent: renders `rhs` as a statement (expression-statement).
 * In expr-statement mode, text_printer_kind wraps the expression (e.g. `T.evaluate(expr)`)
 * and text_printer_return_check can render it as `return expr`.
 */
struct AssignTraitsObj : public StmtTraitsObj {
  /*! \brief Target variable(s) reference. Absent for expression-statements. */
  Optional<String> def_values;
  /*! \brief Right-hand side / expression reference. */
  String rhs;
  /*! \brief Method called before printing. */
  Optional<String> text_printer_pre;
  /*! \brief Method called after printing. */
  Optional<String> text_printer_post;
  /*! \brief Wrapper callee for expr-stmt mode (e.g. "T.evaluate"). */
  Optional<String> text_printer_kind;
  /*! \brief $method: returning bool — if true, render as `return expr` instead of stmt. */
  Optional<String> text_printer_return_check;
  /// \cond Doxygen_Suppress
  explicit AssignTraitsObj(Optional<String> def_values, String rhs,
                           Optional<String> text_printer_pre = {},
                           Optional<String> text_printer_post = {},
                           Optional<String> text_printer_kind = {},
                           Optional<String> text_printer_return_check = {})
      : def_values(std::move(def_values)),
        rhs(std::move(rhs)),
        text_printer_pre(std::move(text_printer_pre)),
        text_printer_post(std::move(text_printer_post)),
        text_printer_kind(std::move(text_printer_kind)),
        text_printer_return_check(std::move(text_printer_return_check)) {}
  explicit AssignTraitsObj(const char* def_values, const char* rhs, const char* pre = nullptr,
                           const char* post = nullptr, const char* kind = nullptr,
                           const char* return_check = nullptr)
      : def_values(def_values ? Optional<String>(String(def_values)) : Optional<String>()),
        rhs(rhs),
        text_printer_pre(ToOptString(pre)),
        text_printer_post(ToOptString(post)),
        text_printer_kind(ToOptString(kind)),
        text_printer_return_check(ToOptString(return_check)) {}
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ffi.ir_traits.AssignTraits", AssignTraitsObj, StmtTraitsObj);
  /// \endcond
};
/// \cond Doxygen_Suppress
struct AssignTraits : public StmtTraits {
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(AssignTraits, StmtTraits, AssignTraitsObj);
};
/// \endcond

/*! \brief Memory load trait. */
struct LoadTraitsObj : public ExprTraitsObj {
  /*! \brief Source buffer/tensor reference. */
  String source;
  /*! \brief Index expressions reference. */
  Optional<String> indices;
  /*! \brief Predicate reference (for conditional loads). */
  Optional<String> predicate;
  /// \cond Doxygen_Suppress
  explicit LoadTraitsObj(String source, Optional<String> indices, Optional<String> predicate = {})
      : source(std::move(source)), indices(std::move(indices)), predicate(std::move(predicate)) {}
  explicit LoadTraitsObj(const char* source, const char* indices = nullptr,
                         const char* predicate = nullptr)
      : source(source), indices(ToOptString(indices)), predicate(ToOptString(predicate)) {}
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ffi.ir_traits.LoadTraits", LoadTraitsObj, ExprTraitsObj);
  /// \endcond
};
/// \cond Doxygen_Suppress
struct LoadTraits : public ExprTraits {
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(LoadTraits, ExprTraits, LoadTraitsObj);
};
/// \endcond

/*! \brief Memory store trait. */
struct StoreTraitsObj : public StmtTraitsObj {
  /*! \brief Target buffer/tensor reference. */
  String target;
  /*! \brief Value to store reference. */
  String value;
  /*! \brief Index expressions reference. */
  Optional<String> indices;
  /*! \brief Predicate reference (for conditional stores). */
  Optional<String> predicate;
  /// \cond Doxygen_Suppress
  explicit StoreTraitsObj(String target, String value, Optional<String> indices,
                          Optional<String> predicate = {})
      : target(std::move(target)),
        value(std::move(value)),
        indices(std::move(indices)),
        predicate(std::move(predicate)) {}
  explicit StoreTraitsObj(const char* target, const char* value, const char* indices = nullptr,
                          const char* predicate = nullptr)
      : target(target),
        value(value),
        indices(ToOptString(indices)),
        predicate(ToOptString(predicate)) {}
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ffi.ir_traits.StoreTraits", StoreTraitsObj, StmtTraitsObj);
  /// \endcond
};
/// \cond Doxygen_Suppress
struct StoreTraits : public StmtTraits {
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(StoreTraits, StmtTraits, StoreTraitsObj);
};
/// \endcond

/*! \brief Assertion trait. */
struct AssertTraitsObj : public StmtTraitsObj {
  /*! \brief Condition reference. */
  String cond;
  /*! \brief Message reference. */
  Optional<String> message;
  /// \cond Doxygen_Suppress
  explicit AssertTraitsObj(String cond, Optional<String> message)
      : cond(std::move(cond)), message(std::move(message)) {}
  explicit AssertTraitsObj(const char* cond, const char* message = nullptr)
      : cond(cond), message(ToOptString(message)) {}
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ffi.ir_traits.AssertTraits", AssertTraitsObj, StmtTraitsObj);
  /// \endcond
};
/// \cond Doxygen_Suppress
struct AssertTraits : public StmtTraits {
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(AssertTraits, StmtTraits, AssertTraitsObj);
};
/// \endcond

/*! \brief Return trait. */
struct ReturnTraitsObj : public StmtTraitsObj {
  /*! \brief Return value reference. */
  String value;
  /// \cond Doxygen_Suppress
  explicit ReturnTraitsObj(String value) : value(std::move(value)) {}
  explicit ReturnTraitsObj(const char* value) : value(value) {}
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ffi.ir_traits.ReturnTraits", ReturnTraitsObj, StmtTraitsObj);
  /// \endcond
};
/// \cond Doxygen_Suppress
struct ReturnTraits : public StmtTraits {
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(ReturnTraits, StmtTraits, ReturnTraitsObj);
};
/// \endcond

/*! \brief Literal / leaf value trait. */
struct LiteralTraitsObj : public ExprTraitsObj {
  /*! \brief Value reference. */
  String value;
  /*! \brief Format hint (e.g. "int", "float", "string"). */
  Optional<String> format;
  /// \cond Doxygen_Suppress
  explicit LiteralTraitsObj(String value, Optional<String> format = {})
      : value(std::move(value)), format(std::move(format)) {}
  explicit LiteralTraitsObj(const char* value, const char* format = nullptr)
      : value(value), format(ToOptString(format)) {}
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ffi.ir_traits.LiteralTraits", LiteralTraitsObj, ExprTraitsObj);
  /// \endcond
};
/// \cond Doxygen_Suppress
struct LiteralTraits : public ExprTraits {
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(LiteralTraits, ExprTraits, LiteralTraitsObj);
};
/// \endcond

/*! \brief Call expression trait (function call with op and args). */
struct CallTraitsObj : public ExprTraitsObj {
  /*! \brief Callee / operator reference (literal string or $field:/$method:). */
  String op;
  /*! \brief Arguments reference. */
  String args;
  /*! \brief Attributes reference. */
  Optional<String> attrs;
  /*! \brief Keyword args reference ($method: returning Map<String, Any>). */
  Optional<String> kwargs;
  /*! \brief Dynamic callee override ($method: returning String callee name). */
  Optional<String> text_printer_callee;
  /*! \brief Hook called before printing ($global:). */
  Optional<String> text_printer_pre;
  /// \cond Doxygen_Suppress
  explicit CallTraitsObj(String op, String args, Optional<String> attrs = {},
                         Optional<String> kwargs = {}, Optional<String> text_printer_callee = {},
                         Optional<String> text_printer_pre = {})
      : op(std::move(op)),
        args(std::move(args)),
        attrs(std::move(attrs)),
        kwargs(std::move(kwargs)),
        text_printer_callee(std::move(text_printer_callee)),
        text_printer_pre(std::move(text_printer_pre)) {}
  explicit CallTraitsObj(const char* op, const char* args, const char* attrs = nullptr,
                         const char* kwargs = nullptr, const char* callee = nullptr,
                         const char* pre = nullptr)
      : op(op),
        args(args),
        attrs(ToOptString(attrs)),
        kwargs(ToOptString(kwargs)),
        text_printer_callee(ToOptString(callee)),
        text_printer_pre(ToOptString(pre)) {}
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ffi.ir_traits.CallTraits", CallTraitsObj, ExprTraitsObj);
  /// \endcond
};
/// \cond Doxygen_Suppress
struct CallTraits : public ExprTraits {
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(CallTraits, ExprTraits, CallTraitsObj);
};
/// \endcond

/*! \brief Function trait. */
struct FuncTraitsObj : public StmtTraitsObj {
  /*! \brief Symbol name reference. */
  String symbol;
  /*! \brief Function body region. */
  RegionTraits region;
  /*! \brief Attributes reference. */
  Optional<String> attrs;
  /*! \brief Decorator/callee text. */
  Optional<String> text_printer_kind;
  /*! \brief Method called before body. */
  Optional<String> text_printer_pre;
  /// \cond Doxygen_Suppress
  explicit FuncTraitsObj(String symbol, RegionTraits region, Optional<String> attrs,
                         Optional<String> text_printer_kind, Optional<String> text_printer_pre)
      : symbol(std::move(symbol)),
        region(std::move(region)),
        attrs(std::move(attrs)),
        text_printer_kind(std::move(text_printer_kind)),
        text_printer_pre(std::move(text_printer_pre)) {}
  explicit FuncTraitsObj(const char* symbol, RegionTraits region, const char* attrs = nullptr,
                         const char* kind = nullptr, const char* pre = nullptr)
      : symbol(symbol),
        region(std::move(region)),
        attrs(ToOptString(attrs)),
        text_printer_kind(ToOptString(kind)),
        text_printer_pre(ToOptString(pre)) {}
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ffi.ir_traits.FuncTraits", FuncTraitsObj, StmtTraitsObj);
  /// \endcond
};
/// \cond Doxygen_Suppress
struct FuncTraits : public StmtTraits {
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(FuncTraits, StmtTraits, FuncTraitsObj);
};
/// \endcond

/*! \brief For-loop trait. */
struct ForTraitsObj : public StmtTraitsObj {
  /*! \brief Loop body region. */
  RegionTraits region;
  /*! \brief Loop start reference. */
  Optional<String> start;
  /*! \brief Loop end reference. */
  Optional<String> end;
  /*! \brief Loop step reference. */
  Optional<String> step;
  /*! \brief Carried variables reference. */
  Optional<String> def_carry;
  /*! \brief Initial carry values reference. */
  Optional<String> carry_init;
  /*! \brief Attributes reference. */
  Optional<String> attrs;
  /*! \brief Custom iterator kind text. */
  Optional<String> text_printer_kind;
  /// \cond Doxygen_Suppress
  explicit ForTraitsObj(RegionTraits region, Optional<String> start, Optional<String> end,
                        Optional<String> step, Optional<String> def_carry,
                        Optional<String> carry_init, Optional<String> attrs,
                        Optional<String> text_printer_kind)
      : region(std::move(region)),
        start(std::move(start)),
        end(std::move(end)),
        step(std::move(step)),
        def_carry(std::move(def_carry)),
        carry_init(std::move(carry_init)),
        attrs(std::move(attrs)),
        text_printer_kind(std::move(text_printer_kind)) {}
  explicit ForTraitsObj(RegionTraits region, const char* start = nullptr, const char* end = nullptr,
                        const char* step = nullptr, const char* def_carry = nullptr,
                        const char* carry_init = nullptr, const char* attrs = nullptr,
                        const char* kind = nullptr)
      : region(std::move(region)),
        start(ToOptString(start)),
        end(ToOptString(end)),
        step(ToOptString(step)),
        def_carry(ToOptString(def_carry)),
        carry_init(ToOptString(carry_init)),
        attrs(ToOptString(attrs)),
        text_printer_kind(ToOptString(kind)) {}
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ffi.ir_traits.ForTraits", ForTraitsObj, StmtTraitsObj);
  /// \endcond
};
/// \cond Doxygen_Suppress
struct ForTraits : public StmtTraits {
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(ForTraits, StmtTraits, ForTraitsObj);
};
/// \endcond

/*!
 * \brief Context manager / sequence trait.
 *
 * When text_printer_kind is set: renders as `with kind(...) as var:` block.
 * When text_printer_kind is absent and region has no def_values: renders as
 * an inline sequence of the region body (replaces SeqTraits).
 */
struct WithTraitsObj : public StmtTraitsObj {
  /*! \brief Scoped body region. */
  RegionTraits region;
  /*! \brief Carried variables reference. */
  Optional<String> def_carry;
  /*! \brief Initial carry values reference. */
  Optional<String> carry_init;
  /*! \brief Context expression kind text. */
  Optional<String> text_printer_kind;
  /*! \brief Method called before body. */
  Optional<String> text_printer_pre;
  /*! \brief Method called after body. */
  Optional<String> text_printer_post;
  /*! \brief When true, skip frame push/pop (elements stay in parent scope). */
  Optional<bool> text_printer_no_frame;
  /// \cond Doxygen_Suppress
  explicit WithTraitsObj(RegionTraits region, Optional<String> def_carry,
                         Optional<String> carry_init, Optional<String> text_printer_kind,
                         Optional<String> text_printer_pre, Optional<String> text_printer_post,
                         Optional<bool> text_printer_no_frame = {})
      : region(std::move(region)),
        def_carry(std::move(def_carry)),
        carry_init(std::move(carry_init)),
        text_printer_kind(std::move(text_printer_kind)),
        text_printer_pre(std::move(text_printer_pre)),
        text_printer_post(std::move(text_printer_post)),
        text_printer_no_frame(std::move(text_printer_no_frame)) {}
  explicit WithTraitsObj(RegionTraits region, const char* def_carry = nullptr,
                         const char* carry_init = nullptr, const char* kind = nullptr,
                         const char* pre = nullptr, const char* post = nullptr,
                         Optional<bool> no_frame = {})
      : region(std::move(region)),
        def_carry(ToOptString(def_carry)),
        carry_init(ToOptString(carry_init)),
        text_printer_kind(ToOptString(kind)),
        text_printer_pre(ToOptString(pre)),
        text_printer_post(ToOptString(post)),
        text_printer_no_frame(std::move(no_frame)) {}
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ffi.ir_traits.WithTraits", WithTraitsObj, StmtTraitsObj);
  /// \endcond
};
/// \cond Doxygen_Suppress
struct WithTraits : public StmtTraits {
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(WithTraits, StmtTraits, WithTraitsObj);
};
/// \endcond

/*! \brief While-loop trait. */
struct WhileTraitsObj : public StmtTraitsObj {
  /*! \brief Condition reference. */
  String cond;
  /*! \brief Loop body region. */
  RegionTraits region;
  /// \cond Doxygen_Suppress
  explicit WhileTraitsObj(String cond, RegionTraits region)
      : cond(std::move(cond)), region(std::move(region)) {}
  explicit WhileTraitsObj(const char* cond, RegionTraits region)
      : cond(cond), region(std::move(region)) {}
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ffi.ir_traits.WhileTraits", WhileTraitsObj, StmtTraitsObj);
  /// \endcond
};
/// \cond Doxygen_Suppress
struct WhileTraits : public StmtTraits {
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(WhileTraits, StmtTraits, WhileTraitsObj);
};
/// \endcond

/*! \brief Conditional trait. */
struct IfTraitsObj : public StmtTraitsObj {
  /*! \brief Condition reference. */
  String cond;
  /*! \brief Then-branch region. */
  RegionTraits then_region;
  /*! \brief Else-branch region (optional). */
  Optional<RegionTraits> else_region;
  /// \cond Doxygen_Suppress
  explicit IfTraitsObj(String cond, RegionTraits then_region, Optional<RegionTraits> else_region)
      : cond(std::move(cond)),
        then_region(std::move(then_region)),
        else_region(std::move(else_region)) {}
  explicit IfTraitsObj(const char* cond, RegionTraits then_region,
                       Optional<RegionTraits> else_region = {})
      : cond(cond), then_region(std::move(then_region)), else_region(std::move(else_region)) {}
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ffi.ir_traits.IfTraits", IfTraitsObj, StmtTraitsObj);
  /// \endcond
};
/// \cond Doxygen_Suppress
struct IfTraits : public StmtTraits {
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(IfTraits, StmtTraits, IfTraitsObj);
};
/// \endcond

/************** Type traits **************/

/*! \brief Base type marker trait. */
struct TyTraitsObj : public IRTraitsObj {
  /// \cond Doxygen_Suppress
  TVM_FFI_DECLARE_OBJECT_INFO("ffi.ir_traits.TyTraits", TyTraitsObj, IRTraitsObj);
  /// \endcond
};
/// \cond Doxygen_Suppress
struct TyTraits : public IRTraits {
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TyTraits, IRTraits, TyTraitsObj);
};
/// \endcond

/*! \brief Tensor type trait. */
struct TensorTyTraitsObj : public TyTraitsObj {
  /*! \brief Shape reference. */
  Optional<String> shape;
  /*! \brief Data type reference. */
  Optional<String> dtype;
  /*! \brief Device reference. */
  Optional<String> device;
  /// \cond Doxygen_Suppress
  explicit TensorTyTraitsObj(Optional<String> shape, Optional<String> dtype,
                             Optional<String> device)
      : shape(std::move(shape)), dtype(std::move(dtype)), device(std::move(device)) {}
  explicit TensorTyTraitsObj(const char* shape = nullptr, const char* dtype = nullptr,
                             const char* device = nullptr)
      : shape(ToOptString(shape)), dtype(ToOptString(dtype)), device(ToOptString(device)) {}
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ffi.ir_traits.TensorTyTraits", TensorTyTraitsObj, TyTraitsObj);
  /// \endcond
};
/// \cond Doxygen_Suppress
struct TensorTyTraits : public TyTraits {
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TensorTyTraits, TyTraits, TensorTyTraitsObj);
};
/// \endcond

/*! \brief Buffer / MemRef type trait. */
struct BufferTyTraitsObj : public TyTraitsObj {
  /*! \brief Shape reference. */
  String shape;
  /*! \brief Data type reference. */
  String dtype;
  /*! \brief Strides reference. */
  Optional<String> strides;
  /*! \brief Element offset reference. */
  Optional<String> offset;
  /*! \brief Memory scope reference. */
  Optional<String> scope;
  /// \cond Doxygen_Suppress
  explicit BufferTyTraitsObj(String shape, String dtype, Optional<String> strides,
                             Optional<String> offset, Optional<String> scope)
      : shape(std::move(shape)),
        dtype(std::move(dtype)),
        strides(std::move(strides)),
        offset(std::move(offset)),
        scope(std::move(scope)) {}
  explicit BufferTyTraitsObj(const char* shape, const char* dtype, const char* strides = nullptr,
                             const char* offset = nullptr, const char* scope = nullptr)
      : shape(shape),
        dtype(dtype),
        strides(ToOptString(strides)),
        offset(ToOptString(offset)),
        scope(ToOptString(scope)) {}
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ffi.ir_traits.BufferTyTraits", BufferTyTraitsObj, TyTraitsObj);
  /// \endcond
};
/// \cond Doxygen_Suppress
struct BufferTyTraits : public TyTraits {
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(BufferTyTraits, TyTraits, BufferTyTraitsObj);
};
/// \endcond

/*! \brief Scalar primitive type trait. */
struct PrimTyTraitsObj : public TyTraitsObj {
  /*! \brief Data type reference. */
  String dtype;
  /// \cond Doxygen_Suppress
  explicit PrimTyTraitsObj(String dtype) : dtype(std::move(dtype)) {}
  explicit PrimTyTraitsObj(const char* dtype) : dtype(dtype) {}
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ffi.ir_traits.PrimTyTraits", PrimTyTraitsObj, TyTraitsObj);
  /// \endcond
};
/// \cond Doxygen_Suppress
struct PrimTyTraits : public TyTraits {
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(PrimTyTraits, TyTraits, PrimTyTraitsObj);
};
/// \endcond

/*! \brief Function signature type trait. */
struct FuncTyTraitsObj : public TyTraitsObj {
  /*! \brief Parameter types reference. */
  Optional<String> params;
  /*! \brief Return type reference. */
  Optional<String> ret;
  /// \cond Doxygen_Suppress
  explicit FuncTyTraitsObj(Optional<String> params, Optional<String> ret)
      : params(std::move(params)), ret(std::move(ret)) {}
  explicit FuncTyTraitsObj(const char* params = nullptr, const char* ret = nullptr)
      : params(ToOptString(params)), ret(ToOptString(ret)) {}
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ffi.ir_traits.FuncTyTraits", FuncTyTraitsObj, TyTraitsObj);
  /// \endcond
};
/// \cond Doxygen_Suppress
struct FuncTyTraits : public TyTraits {
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(FuncTyTraits, TyTraits, FuncTyTraitsObj);
};
/// \endcond

/*! \brief Tuple / product type trait. */
struct TupleTyTraitsObj : public TyTraitsObj {
  /*! \brief Element types reference. */
  String fields;
  /// \cond Doxygen_Suppress
  explicit TupleTyTraitsObj(String fields) : fields(std::move(fields)) {}
  explicit TupleTyTraitsObj(const char* fields) : fields(fields) {}
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ffi.ir_traits.TupleTyTraits", TupleTyTraitsObj, TyTraitsObj);
  /// \endcond
};
/// \cond Doxygen_Suppress
struct TupleTyTraits : public TyTraits {
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TupleTyTraits, TyTraits, TupleTyTraitsObj);
};
/// \endcond

/*! \brief Shape descriptor type trait. */
struct ShapeTyTraitsObj : public TyTraitsObj {
  /*! \brief Dimension expressions reference. */
  Optional<String> dims;
  /*! \brief Rank reference. */
  Optional<String> ndim;
  /// \cond Doxygen_Suppress
  explicit ShapeTyTraitsObj(Optional<String> dims, Optional<String> ndim)
      : dims(std::move(dims)), ndim(std::move(ndim)) {}
  explicit ShapeTyTraitsObj(const char* dims = nullptr, const char* ndim = nullptr)
      : dims(ToOptString(dims)), ndim(ToOptString(ndim)) {}
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ffi.ir_traits.ShapeTyTraits", ShapeTyTraitsObj, TyTraitsObj);
  /// \endcond
};
/// \cond Doxygen_Suppress
struct ShapeTyTraits : public TyTraits {
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(ShapeTyTraits, TyTraits, ShapeTyTraitsObj);
};
/// \endcond

}  // namespace ir_traits
}  // namespace ffi
}  // namespace tvm

#endif  // TVM_FFI_EXTRA_IR_TRAITS_H_
