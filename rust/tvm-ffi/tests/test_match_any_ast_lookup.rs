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

use tvm_ffi::derive::{Object, ObjectRef};
use tvm_ffi::object::{Object as ObjectBase, ObjectArc, ObjectCore};
use tvm_ffi::{match_any, Any, Array, Shape, TypeIndex};

#[repr(C)]
#[derive(Object)]
#[type_key = "testing.match_any.Expr"]
#[type_register]
#[type_child_slots = 20]
#[type_child_slots_can_overflow = false]
struct ExprObj {
    base: ObjectBase,
}

fn register_type<T: ObjectCore>() -> i32 {
    let type_index = T::type_index();
    assert!(type_index >= TypeIndex::kTVMFFIStaticObjectBegin as i32);
    type_index
}

macro_rules! define_expr_leaves {
    ($(($object:ident, $handle:ident, $type_key:literal)),+ $(,)?) => {
        $(
            #[repr(C)]
            #[derive(Object)]
            #[type_key = $type_key]
            #[type_register]
            #[type_final]
            struct $object {
                base: ExprObj,
            }

            #[repr(C)]
            #[derive(ObjectRef, Clone)]
            struct $handle {
                data: ObjectArc<$object>,
            }

            impl Default for $handle {
                fn default() -> Self {
                    Self {
                        data: ObjectArc::new($object {
                            base: ExprObj {
                                base: ObjectBase::new(),
                            },
                        }),
                    }
                }
            }
        )+

        fn register_expr_types() {
            register_type::<ExprObj>();
            $(
                register_type::<$object>();
            )+
        }
    };
}

// These final nodes model the kind of downstream AST hierarchy that motivates
// exact-leaf dispatch. This integration-test binary intentionally has one test,
// so its process-wide type registrations happen before any concurrent lookup.
define_expr_leaves!(
    (AddExprObj, AddExpr, "testing.match_any.AddExpr"),
    (SubExprObj, SubExpr, "testing.match_any.SubExpr"),
    (MulExprObj, MulExpr, "testing.match_any.MulExpr"),
    (DivExprObj, DivExpr, "testing.match_any.DivExpr"),
    (ModExprObj, ModExpr, "testing.match_any.ModExpr"),
    (NegExprObj, NegExpr, "testing.match_any.NegExpr"),
    (CallExprObj, CallExpr, "testing.match_any.CallExpr"),
    (LetExprObj, LetExpr, "testing.match_any.LetExpr"),
    (IfExprObj, IfExpr, "testing.match_any.IfExpr"),
    (TupleExprObj, TupleExpr, "testing.match_any.TupleExpr"),
    (
        TupleGetItemExprObj,
        TupleGetItemExpr,
        "testing.match_any.TupleGetItemExpr"
    ),
    (CastExprObj, CastExpr, "testing.match_any.CastExpr"),
    (LoadExprObj, LoadExpr, "testing.match_any.LoadExpr"),
    (StoreExprObj, StoreExpr, "testing.match_any.StoreExpr"),
    (ForExprObj, ForExpr, "testing.match_any.ForExpr"),
    (WhileExprObj, WhileExpr, "testing.match_any.WhileExpr"),
    (SeqExprObj, SeqExpr, "testing.match_any.SeqExpr"),
    (ReturnExprObj, ReturnExpr, "testing.match_any.ReturnExpr"),
    (
        ConstantExprObj,
        ConstantExpr,
        "testing.match_any.ConstantExpr"
    ),
    (VarExprObj, VarExpr, "testing.match_any.VarExpr"),
);

#[test]
fn dispatches_representative_ast_leaf_nodes() {
    register_expr_types();

    fn classify(value: Any) -> &'static str {
        match_any! {
            value {
                AddExpr(_) => "add",
                SubExpr(_) => "sub",
                MulExpr(_) => "mul",
                DivExpr(_) => "div",
                ModExpr(_) => "mod",
                NegExpr(_) => "neg",
                CallExpr(_) => "call",
                LetExpr(_) => "let",
                IfExpr(_) => "if",
                TupleExpr(_) => "tuple",
                TupleGetItemExpr(_) => "tuple_get_item",
                CastExpr(_) => "cast",
                LoadExpr(_) => "load",
                StoreExpr(_) => "store",
                ForExpr(_) => "for",
                WhileExpr(_) => "while",
                SeqExpr(_) => "seq",
                ReturnExpr(_) => "return",
                ConstantExpr(_) => "constant",
                VarExpr(_) => "var",
                _ => "unsupported",
            }
        }
    }

    let cases = [
        (Any::from(AddExpr::default()), "add"),
        (Any::from(SubExpr::default()), "sub"),
        (Any::from(MulExpr::default()), "mul"),
        (Any::from(DivExpr::default()), "div"),
        (Any::from(ModExpr::default()), "mod"),
        (Any::from(NegExpr::default()), "neg"),
        (Any::from(CallExpr::default()), "call"),
        (Any::from(LetExpr::default()), "let"),
        (Any::from(IfExpr::default()), "if"),
        (Any::from(TupleExpr::default()), "tuple"),
        (Any::from(TupleGetItemExpr::default()), "tuple_get_item"),
        (Any::from(CastExpr::default()), "cast"),
        (Any::from(LoadExpr::default()), "load"),
        (Any::from(StoreExpr::default()), "store"),
        (Any::from(ForExpr::default()), "for"),
        (Any::from(WhileExpr::default()), "while"),
        (Any::from(SeqExpr::default()), "seq"),
        (Any::from(ReturnExpr::default()), "return"),
        (Any::from(ConstantExpr::default()), "constant"),
        (Any::from(VarExpr::default()), "var"),
    ];
    for (value, expected) in cases {
        assert_eq!(classify(value), expected);
    }

    assert_eq!(classify(Any::from(Shape::from([1_i64, 2]))), "unsupported");
    assert_eq!(classify(Any::from(1_i64)), "unsupported");

    // This call site has enough syntactically eligible arms to consider leaf
    // lookup, but Array<T> requires complete type conversion. The entire match
    // must therefore retain source-ordered dispatch.
    fn classify_mixed_patterns(value: Any) -> &'static str {
        match_any! {
            value {
                AddExpr(_) => "add",
                SubExpr(_) => "sub",
                MulExpr(_) => "mul",
                DivExpr(_) => "div",
                ModExpr(_) => "mod",
                NegExpr(_) => "neg",
                CallExpr(_) => "call",
                LetExpr(_) => "let",
                IfExpr(_) => "if",
                TupleExpr(_) => "tuple",
                TupleGetItemExpr(_) => "tuple_get_item",
                CastExpr(_) => "cast",
                LoadExpr(_) => "load",
                StoreExpr(_) => "store",
                ForExpr(_) => "for",
                WhileExpr(_) => "while",
                SeqExpr(_) => "seq",
                ReturnExpr(_) => "return",
                Array::<i64>(_) => "integer_array",
                Array::<f64>(_) => "float_array",
                _ => "unsupported",
            }
        }
    }

    assert_eq!(
        classify_mixed_patterns(Any::from(AddExpr::default())),
        "add"
    );
    let array = [1.5_f64, 2.5].into_iter().collect::<Array<f64>>();
    assert_eq!(classify_mixed_patterns(Any::from(array)), "float_array");
}
