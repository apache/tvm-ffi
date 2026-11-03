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
use tvm_ffi::{
    dispatch, match_any, Any, AnyView, Array, Dispatcher, MutationView, Shape, TypeIndex,
};

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
    ($(($object:ident, $handle:ident, $method:ident, $label:literal, $type_key:literal)),+ $(,)?) => {
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

        struct OwnedClassifier;
        #[dispatch(classify)]
        impl OwnedClassifier {
            $(fn $method(&self, _value: $handle) -> &'static str { $label })+
        }

        struct BorrowedClassifier;
        #[dispatch(classify)]
        impl BorrowedClassifier {
            $(fn $method(&self, _value: &$object) -> &'static str { $label })+
        }

        struct CapabilityClassifier;
        #[dispatch(classify, MutationView)]
        impl CapabilityClassifier {
            $(fn $method(&self, _value: MutationView<'_, $handle>) -> &'static str { $label })+
        }

        struct MixedClassifier;
        #[dispatch(classify)]
        impl MixedClassifier {
            $(fn $method(&self, _value: $handle) -> &'static str { $label })+
            fn classify_integer_array(&self, _value: Array<i64>) -> &'static str {
                "integer_array"
            }
            fn classify_float_array(&self, _value: Array<f64>) -> &'static str {
                "float_array"
            }
        }

        fn classify_borrowed_pattern(value: AnyView<'_>) -> &'static str {
            match_any! { value {
                $($object(ref node) => { let _: &$object = node; $label },)+
                _ => "unsupported",
            }}
        }

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
    (
        AddExprObj,
        AddExpr,
        classify_add,
        "add",
        "testing.match_any.AddExpr"
    ),
    (
        SubExprObj,
        SubExpr,
        classify_sub,
        "sub",
        "testing.match_any.SubExpr"
    ),
    (
        MulExprObj,
        MulExpr,
        classify_mul,
        "mul",
        "testing.match_any.MulExpr"
    ),
    (
        DivExprObj,
        DivExpr,
        classify_div,
        "div",
        "testing.match_any.DivExpr"
    ),
    (
        ModExprObj,
        ModExpr,
        classify_mod,
        "mod",
        "testing.match_any.ModExpr"
    ),
    (
        NegExprObj,
        NegExpr,
        classify_neg,
        "neg",
        "testing.match_any.NegExpr"
    ),
    (
        CallExprObj,
        CallExpr,
        classify_call,
        "call",
        "testing.match_any.CallExpr"
    ),
    (
        LetExprObj,
        LetExpr,
        classify_let,
        "let",
        "testing.match_any.LetExpr"
    ),
    (
        IfExprObj,
        IfExpr,
        classify_if,
        "if",
        "testing.match_any.IfExpr"
    ),
    (
        TupleExprObj,
        TupleExpr,
        classify_tuple,
        "tuple",
        "testing.match_any.TupleExpr"
    ),
    (
        TupleGetItemExprObj,
        TupleGetItemExpr,
        classify_tuple_get_item,
        "tuple_get_item",
        "testing.match_any.TupleGetItemExpr"
    ),
    (
        CastExprObj,
        CastExpr,
        classify_cast,
        "cast",
        "testing.match_any.CastExpr"
    ),
    (
        LoadExprObj,
        LoadExpr,
        classify_load,
        "load",
        "testing.match_any.LoadExpr"
    ),
    (
        StoreExprObj,
        StoreExpr,
        classify_store,
        "store",
        "testing.match_any.StoreExpr"
    ),
    (
        ForExprObj,
        ForExpr,
        classify_for,
        "for",
        "testing.match_any.ForExpr"
    ),
    (
        WhileExprObj,
        WhileExpr,
        classify_while,
        "while",
        "testing.match_any.WhileExpr"
    ),
    (
        SeqExprObj,
        SeqExpr,
        classify_seq,
        "seq",
        "testing.match_any.SeqExpr"
    ),
    (
        ReturnExprObj,
        ReturnExpr,
        classify_return,
        "return",
        "testing.match_any.ReturnExpr"
    ),
    (
        ConstantExprObj,
        ConstantExpr,
        classify_constant,
        "constant",
        "testing.match_any.ConstantExpr"
    ),
    (
        VarExprObj,
        VarExpr,
        classify_var,
        "var",
        "testing.match_any.VarExpr"
    ),
);

#[test]
fn dispatches_representative_ast_leaf_nodes() {
    register_expr_types();

    // The ref adapter must expose final-node metadata, including in mixed lists.
    use tvm_ffi::match_any_internal::{
        LeafPatternMetadata, LeafPatternProbe, PatternConversionProbe, RefPatternMetadata,
    };
    let node_probe = PatternConversionProbe::<AddExprObj>::new();
    let mixed = LeafPatternProbe::<(SubExpr, ())>::new().prepend((&&&node_probe).leaf_pattern());
    assert!((&mixed).leaf_pattern_list_id().is_some());
    let mut indices = [0; 2];
    (&mixed).fill_leaf_type_indices(&mut indices);
    assert_eq!(
        indices,
        [AddExprObj::type_index(), SubExprObj::type_index()]
    );

    type Owned<T> = T;
    use tvm_ffi::MutationView;
    macro_rules! classify_all {
        ($value:expr; $wrapper:ident) => {
            match_any! {
                $value {
                    $wrapper<AddExpr>(_) => "add",
                    $wrapper<SubExpr>(_) => "sub",
                    $wrapper<MulExpr>(_) => "mul",
                    $wrapper<DivExpr>(_) => "div",
                    $wrapper<ModExpr>(_) => "mod",
                    $wrapper<NegExpr>(_) => "neg",
                    $wrapper<CallExpr>(_) => "call",
                    $wrapper<LetExpr>(_) => "let",
                    $wrapper<IfExpr>(_) => "if",
                    $wrapper<TupleExpr>(_) => "tuple",
                    $wrapper<TupleGetItemExpr>(_) => "tuple_get_item",
                    $wrapper<CastExpr>(_) => "cast",
                    $wrapper<LoadExpr>(_) => "load",
                    $wrapper<StoreExpr>(_) => "store",
                    $wrapper<ForExpr>(_) => "for",
                    $wrapper<WhileExpr>(_) => "while",
                    $wrapper<SeqExpr>(_) => "seq",
                    $wrapper<ReturnExpr>(_) => "return",
                    $wrapper<ConstantExpr>(_) => "constant",
                    $wrapper<VarExpr>(_) => "var",
                    _ => "unsupported",
                }
            }
        };
    }
    fn classify(value: Any) -> &'static str {
        classify_all!(value; Owned)
    }

    fn checked<T>(_: &T, value: &Any, count: usize, name: &'static str) -> &'static str {
        assert_eq!(value.debug_strong_count(), Some(count));
        name
    }

    fn classify_refs<T: ObjectCore>(value: &Any) -> &'static str {
        match_any! {
            value {
                T(ref node) => checked(node, value, 1, "add"),
                SubExpr(ref handle) => checked(handle, value, 2, "sub"),
                MulExprObj(ref node) => checked(node, value, 1, "mul"),
                DivExpr(ref handle) => checked(handle, value, 2, "div"),
                ModExprObj(ref node) => checked(node, value, 1, "mod"),
                NegExpr(ref handle) => checked(handle, value, 2, "neg"),
                CallExprObj(ref node) => checked(node, value, 1, "call"),
                LetExpr(ref handle) => checked(handle, value, 2, "let"),
                IfExprObj(ref node) => checked(node, value, 1, "if"),
                TupleExpr(ref handle) => checked(handle, value, 2, "tuple"),
                TupleGetItemExprObj(ref node) => checked(node, value, 1, "tuple_get_item"),
                CastExpr(ref handle) => checked(handle, value, 2, "cast"),
                LoadExprObj(ref node) => checked(node, value, 1, "load"),
                StoreExpr(ref handle) => checked(handle, value, 2, "store"),
                ForExprObj(ref node) => checked(node, value, 1, "for"),
                WhileExpr(ref handle) => checked(handle, value, 2, "while"),
                SeqExprObj(ref node) => checked(node, value, 1, "seq"),
                ReturnExpr(ref handle) => checked(handle, value, 2, "return"),
                ConstantExprObj(ref node) => checked(node, value, 1, "constant"),
                VarExpr(ref handle) => checked(handle, value, 2, "var"),
                AddExpr(_) => "duplicate_must_not_win",
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
        for _ in 0..2 {
            assert_eq!(classify_refs::<AddExprObj>(&value), expected);
            assert_eq!(value.debug_strong_count(), Some(1));
        }
        let view = AnyView::from(&value);
        assert_eq!(OwnedClassifier.dispatch_tuple((view,)), Some(expected));
        assert_eq!(BorrowedClassifier.dispatch_tuple((view,)), Some(expected));
        assert_eq!(
            CapabilityClassifier.dispatch_tuple((MutationView::borrowed(view),)),
            Some(expected)
        );
        assert_eq!(MixedClassifier.dispatch_tuple((view,)), Some(expected));
        assert_eq!(classify_borrowed_pattern(view), expected);
        assert_eq!(classify_all!(&value; Owned), expected);
        assert_eq!(
            classify_all!(MutationView::borrowed(tvm_ffi::AnyView::from(&value)); MutationView),
            expected
        );
        assert_eq!(value.debug_strong_count(), Some(1));
        assert_eq!(classify(value), expected);
    }

    let scalar = 1_i64;
    let view = AnyView::from(&scalar);
    assert_eq!(OwnedClassifier.dispatch_tuple((view,)), None);
    assert_eq!(BorrowedClassifier.dispatch_tuple((view,)), None);
    assert_eq!(
        CapabilityClassifier.dispatch_tuple((MutationView::borrowed(view),)),
        None
    );

    assert_eq!(classify(Any::from(Shape::from([1_i64, 2]))), "unsupported");
    assert_eq!(classify(Any::from(1_i64)), "unsupported");

    // The call-site cache was initialized for AddExprObj; another generic
    // instantiation must take the ordered path and preserve its first arm.
    assert_eq!(
        classify_refs::<SubExprObj>(&Any::from(SubExpr::default())),
        "add"
    );

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
                Array::<f64>(ref array) => {
                    assert_eq!(array.len(), 2);
                    "float_array"
                },
                _ => "unsupported",
            }
        }
    }

    assert_eq!(
        classify_mixed_patterns(Any::from(AddExpr::default())),
        "add"
    );
    let array = [1.5_f64, 2.5].into_iter().collect::<Array<f64>>();
    assert_eq!(
        MixedClassifier.dispatch_tuple((AnyView::from(&array),)),
        Some("float_array")
    );
    assert_eq!(classify_mixed_patterns(Any::from(array)), "float_array");
}
