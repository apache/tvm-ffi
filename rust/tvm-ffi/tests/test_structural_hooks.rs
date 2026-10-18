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

//! Rust structural hooks: per-type descent registered from Rust and consulted
//! by the visit, walk, map and mutate engines ahead of C++ hooks and
//! reflected fields.

use std::sync::Once;

use tvm_ffi::derive::{Object, ObjectRef};
use tvm_ffi::{
    register_structural_hooks, register_structural_hooks_for_type_index, structural_map,
    structural_mutate, structural_walk, Any, DefRegionKind, Error, Function, InplaceValue,
    MapDispatch, MutateChildren, Object, ObjectArc, ObjectRefCore, Result, StructuralHooks,
    StructuralMutator, StructuralVarRemap, StructuralView, TypeIndex, UnchangedOr, VisitChildren,
    VisitHalt, WalkOrder, WalkResult, RUNTIME_ERROR,
};
use tvm_ffi_sys::{TVMFFIByteArray, TVMFFITypeKeyToIndex};

// A pair node whose descent is defined only by its Rust hooks: it has no
// reflected fields, so without hooks the engines see a leaf.
#[repr(C)]
#[derive(Object)]
#[type_key = "testing.rust.HookPair"]
#[type_register]
#[type_final]
struct HookPairObj {
    base: Object,
    left: Any,
    right: Any,
    // Selects the region of `right`: 0 inherits, 1 Simple, 2 Pattern, 3 None.
    // Mutation adds 1 per copy and 100 per in-place update, to tell them apart.
    tag: i64,
}

#[repr(C)]
#[derive(ObjectRef, Clone)]
struct HookPair {
    data: ObjectArc<HookPairObj>,
}

impl HookPair {
    fn new(left: impl Into<Any>, right: impl Into<Any>, tag: i64) -> Self {
        Self {
            data: ObjectArc::new(HookPairObj {
                base: Object::new(),
                left: left.into(),
                right: right.into(),
                tag,
            }),
        }
    }

    fn node(&self) -> &HookPairObj {
        &self.data
    }
}

// The same shape without hooks, to show what the engines do by default.
#[repr(C)]
#[derive(Object)]
#[type_key = "testing.rust.PlainPair"]
#[type_register]
#[type_final]
struct PlainPairObj {
    base: Object,
    left: Any,
    right: Any,
}

#[repr(C)]
#[derive(ObjectRef, Clone)]
struct PlainPair {
    data: ObjectArc<PlainPairObj>,
}

// A variable and its binding, with identity remapping owned by the hooks.
#[repr(C)]
#[derive(Object)]
#[type_key = "testing.rust.HookVar"]
#[type_register]
#[type_final]
struct HookVarObj {
    base: Object,
    name: i64,
}

#[repr(C)]
#[derive(ObjectRef, Clone)]
struct HookVar {
    data: ObjectArc<HookVarObj>,
}

impl HookVar {
    fn new(name: i64) -> Self {
        Self {
            data: ObjectArc::new(HookVarObj {
                base: Object::new(),
                name,
            }),
        }
    }
}

#[repr(C)]
#[derive(Object)]
#[type_key = "testing.rust.HookBind"]
#[type_register]
#[type_final]
struct HookBindObj {
    base: Object,
    var: HookVar,
    value: Any,
}

#[repr(C)]
#[derive(ObjectRef, Clone)]
struct HookBind {
    data: ObjectArc<HookBindObj>,
}

impl HookBind {
    fn new(var: HookVar, value: impl Into<Any>) -> Self {
        Self {
            data: ObjectArc::new(HookBindObj {
                base: Object::new(),
                var,
                value: value.into(),
            }),
        }
    }
}

fn right_region(tag: i64, inherited: DefRegionKind) -> DefRegionKind {
    match tag {
        1 => DefRegionKind::Simple,
        2 => DefRegionKind::Pattern,
        3 => DefRegionKind::None,
        _ => inherited,
    }
}

fn pair_visit(value: &StructuralView, children: &mut VisitChildren<'_>) -> Result<(), VisitHalt> {
    let node = value
        .as_node::<HookPairObj>()
        .expect("registered for HookPair");
    children.visit(&node.left)?;
    let region = right_region(node.tag, children.def_region_kind());
    children.visit_with(&node.right, region)
}

fn pair_mutate(
    value: &StructuralView,
    children: &mut MutateChildren<'_>,
) -> Result<UnchangedOr<Any>> {
    let node = value
        .as_node::<HookPairObj>()
        .expect("registered for HookPair");
    let left = children.mutate(&node.left)?;
    let region = right_region(node.tag, children.def_region_kind());
    let right = children.mutate_with(&node.right, region)?;
    if left.unchanged_or_same_as(&node.left) && right.unchanged_or_same_as(&node.right) {
        return Ok(UnchangedOr::unchanged());
    }
    let left = left.value_or_else(|| node.left.clone());
    let right = right.value_or_else(|| node.right.clone());
    let tag = node.tag;
    // SAFETY: `node` is not used after this point.
    if let Some(node) = unsafe { children.inplace_node::<HookPairObj>() } {
        node.left = left;
        node.right = right;
        node.tag = tag + 100;
        return Ok(UnchangedOr::unchanged());
    }
    Ok(UnchangedOr::changed(
        HookPair::new(left, right, tag + 1).into(),
    ))
}

fn var_visit(_value: &StructuralView, _children: &mut VisitChildren<'_>) -> Result<(), VisitHalt> {
    Ok(())
}

// C++ `VarMutate`: a recorded replacement wins; a use outside any definition
// keeps its identity; a definition records itself under a pattern region.
fn var_mutate(
    value: &StructuralView,
    children: &mut MutateChildren<'_>,
) -> Result<UnchangedOr<Any>> {
    if let Some(mapped) = children.var_remap_get(value)? {
        return Ok(mapped);
    }
    if children.def_region_kind() == DefRegionKind::None {
        return Ok(UnchangedOr::unchanged());
    }
    if children.def_region_kind() == DefRegionKind::Pattern {
        children.var_remap_set(value, &UnchangedOr::unchanged())?;
    }
    Ok(UnchangedOr::unchanged())
}

fn bind_visit(value: &StructuralView, children: &mut VisitChildren<'_>) -> Result<(), VisitHalt> {
    let node = value
        .as_node::<HookBindObj>()
        .expect("registered for HookBind");
    children.visit_with(&node.var, DefRegionKind::Simple)?;
    children.visit(&node.value)
}

// C++ `BindMutate`: the value, then the variable as a simple definition whose
// replacement later uses resolve through.
fn bind_mutate(
    value: &StructuralView,
    children: &mut MutateChildren<'_>,
) -> Result<UnchangedOr<Any>> {
    let node = value
        .as_node::<HookBindObj>()
        .expect("registered for HookBind");
    let mapped_value = children.mutate(&node.value)?;
    let mapped_var = children.mutate_with(&node.var, DefRegionKind::Simple)?;
    let var_changed = !mapped_var.unchanged_or_same_as(&Any::from(node.var.clone()));
    if var_changed {
        children.var_remap_set(&node.var, &mapped_var)?;
    }
    if !var_changed && mapped_value.unchanged_or_same_as(&node.value) {
        return Ok(UnchangedOr::unchanged());
    }
    let var = if var_changed {
        HookVar::try_from(mapped_var.into_option().unwrap())?
    } else {
        node.var.clone()
    };
    let value = mapped_value.value_or_else(|| node.value.clone());
    Ok(UnchangedOr::changed(HookBind::new(var, value).into()))
}

fn register_test_hooks() {
    static REGISTERED: Once = Once::new();
    REGISTERED.call_once(|| {
        register_structural_hooks::<HookPairObj>(StructuralHooks {
            visit: Some(pair_visit),
            mutate: Some(pair_mutate),
        })
        .unwrap();
        register_structural_hooks::<HookVarObj>(StructuralHooks {
            visit: Some(var_visit),
            mutate: Some(var_mutate),
        })
        .unwrap();
        register_structural_hooks::<HookBindObj>(StructuralHooks {
            visit: Some(bind_visit),
            mutate: Some(bind_mutate),
        })
        .unwrap();
    });
}

/// Every integer reached by a walk, with the region it was reached under.
fn walk_integers<R>(root: &R, order: WalkOrder) -> Vec<(i64, DefRegionKind)>
where
    for<'x> tvm_ffi::AnyView<'x>: From<&'x R>,
{
    let mut seen = Vec::new();
    let result = structural_walk(
        root,
        |value: &StructuralView, kind: DefRegionKind| {
            if let Some(integer) = value.cast::<i64>() {
                seen.push((integer, kind));
            }
            WalkResult::Advance
        },
        order,
    );
    assert!(result.unwrap().is_none());
    seen
}

struct IncrementIntegers;

impl MapDispatch for IncrementIntegers {
    fn dispatch_map(
        &mut self,
        value: &StructuralView,
        _def_region_kind: DefRegionKind,
    ) -> Option<Result<Any>> {
        value
            .cast::<i64>()
            .map(|integer| Ok(Any::from(integer + 1)))
    }
}

fn pair_pointer(value: &Any) -> *const HookPairObj {
    let pair = HookPair::try_from(value.clone()).unwrap();
    unsafe { ObjectArc::as_raw(HookPair::data(&pair)) }
}

#[test]
fn visit_hook_defines_descent_and_regions() {
    register_test_hooks();
    // Without hooks a Rust object with no reflected fields is a leaf.
    let plain = PlainPair {
        data: ObjectArc::new(PlainPairObj {
            base: Object::new(),
            left: Any::from(1i64),
            right: Any::from(2i64),
        }),
    };
    assert!(walk_integers(&plain, WalkOrder::PreOrder).is_empty());

    // Outer: right under Simple. Inner (under Simple): left inherits Simple,
    // right asks for None and gets it. Innermost under Pattern: a request for
    // None inside a pattern region has no effect.
    let root = HookPair::new(
        1i64,
        HookPair::new(
            2i64,
            HookPair::new(HookPair::new(4i64, 5i64, 3), 6i64, 0),
            3,
        ),
        1,
    );
    let inner = HookPair::new(9i64, HookPair::new(7i64, 8i64, 3), 2);
    let root = HookPair::new(root, inner, 0);
    for order in [WalkOrder::PreOrder, WalkOrder::PostOrder] {
        let seen = walk_integers(&root, order);
        assert_eq!(
            seen,
            vec![
                (1, DefRegionKind::None),
                (2, DefRegionKind::Simple),
                (4, DefRegionKind::None),
                (5, DefRegionKind::None),
                (6, DefRegionKind::None),
                (9, DefRegionKind::None),
                (7, DefRegionKind::Pattern),
                (8, DefRegionKind::Pattern),
            ]
        );
    }
}

#[test]
fn visit_hook_propagates_interrupts_and_errors() {
    register_test_hooks();
    let root = HookPair::new(1i64, HookPair::new(2i64, 3i64, 0), 0);
    let mut seen = Vec::new();
    let result = structural_walk(
        &root,
        |value: &StructuralView| {
            let Some(integer) = value.cast::<i64>() else {
                return Ok(WalkResult::Advance);
            };
            seen.push(integer);
            match integer {
                2 => Ok(WalkResult::InterruptWith(Any::from(20i64))),
                _ => Ok(WalkResult::Advance),
            }
        },
        WalkOrder::PreOrder,
    );
    let interrupt = result.unwrap().expect("interrupted");
    assert_eq!(i64::try_from(interrupt.value).unwrap(), 20);
    assert_eq!(seen, vec![1, 2]);

    let result = structural_walk(
        &root,
        |value: &StructuralView| -> Result<WalkResult> {
            match value.cast::<i64>() {
                Some(3) => Err(Error::new(RUNTIME_ERROR, "stop at three", "")),
                _ => Ok(WalkResult::Advance),
            }
        },
        WalkOrder::PostOrder,
    );
    let error = result.err().expect("error");
    assert!(error.message().contains("stop at three"));
}

#[test]
fn mutate_hook_copies_shared_nodes_and_updates_unique_nodes_in_place() {
    register_test_hooks();
    let root = HookPair::new(1i64, HookPair::new(2i64, 3i64, 0), 0);
    let root_pointer = unsafe { ObjectArc::as_raw(HookPair::data(&root)) };

    // Shared root: both levels are copied, the original is untouched.
    let mapped =
        structural_map(root.clone(), &mut IncrementIntegers, WalkOrder::PostOrder).unwrap();
    assert_ne!(pair_pointer(&mapped), root_pointer);
    let outer = HookPair::try_from(mapped).unwrap();
    assert_eq!(outer.node().tag, 1);
    assert_eq!(i64::try_from(outer.node().left.clone()).unwrap(), 2);
    let inner = HookPair::try_from(outer.node().right.clone()).unwrap();
    assert_eq!(inner.node().tag, 1);
    assert_eq!(i64::try_from(inner.node().left.clone()).unwrap(), 3);
    assert_eq!(i64::try_from(inner.node().right.clone()).unwrap(), 4);
    assert_eq!(root.node().tag, 0);
    assert_eq!(i64::try_from(root.node().left.clone()).unwrap(), 1);

    // Unique root: both levels are updated in place.
    let mapped = structural_map(root, &mut IncrementIntegers, WalkOrder::PostOrder).unwrap();
    assert_eq!(pair_pointer(&mapped), root_pointer);
    let outer = HookPair::try_from(mapped).unwrap();
    assert_eq!(outer.node().tag, 100);
    assert_eq!(i64::try_from(outer.node().left.clone()).unwrap(), 2);
    let inner = HookPair::try_from(outer.node().right.clone()).unwrap();
    assert_eq!(inner.node().tag, 100);
    assert_eq!(i64::try_from(inner.node().right.clone()).unwrap(), 4);

    // A unique root over a shared child: the root in place, the child copied.
    let shared_child = HookPair::new(5i64, 6i64, 0);
    let root = HookPair::new(shared_child.clone(), 7i64, 0);
    let root_pointer = unsafe { ObjectArc::as_raw(HookPair::data(&root)) };
    let mapped = structural_map(root, &mut IncrementIntegers, WalkOrder::PreOrder).unwrap();
    assert_eq!(pair_pointer(&mapped), root_pointer);
    let outer = HookPair::try_from(mapped).unwrap();
    assert_eq!(outer.node().tag, 100);
    let child = HookPair::try_from(outer.node().left.clone()).unwrap();
    assert_eq!(child.node().tag, 1);
    assert_eq!(i64::try_from(child.node().left.clone()).unwrap(), 6);
    assert_eq!(i64::try_from(shared_child.node().left.clone()).unwrap(), 5);

    // Nothing to change: unchanged, and the same object comes back.
    let root = HookPair::new(Any::new(), Any::new(), 0);
    let root_pointer = unsafe { ObjectArc::as_raw(HookPair::data(&root)) };
    let mapped =
        structural_map(root.clone(), &mut IncrementIntegers, WalkOrder::PostOrder).unwrap();
    assert_eq!(pair_pointer(&mapped), root_pointer);
}

#[test]
fn mutate_hook_regions_select_callback_behavior() {
    register_test_hooks();
    struct IncrementUnderSimple;
    impl MapDispatch for IncrementUnderSimple {
        fn dispatch_map(
            &mut self,
            value: &StructuralView,
            def_region_kind: DefRegionKind,
        ) -> Option<Result<Any>> {
            let integer = value.cast::<i64>()?;
            (def_region_kind == DefRegionKind::Simple).then(|| Ok(Any::from(integer + 10)))
        }
    }
    // Only `right` of a tag-1 pair is a simple definition.
    let root = HookPair::new(1i64, HookPair::new(2i64, 3i64, 0), 1);
    let mapped = structural_map(root, &mut IncrementUnderSimple, WalkOrder::PostOrder).unwrap();
    let outer = HookPair::try_from(mapped).unwrap();
    assert_eq!(i64::try_from(outer.node().left.clone()).unwrap(), 1);
    let inner = HookPair::try_from(outer.node().right.clone()).unwrap();
    assert_eq!(i64::try_from(inner.node().left.clone()).unwrap(), 12);
    assert_eq!(i64::try_from(inner.node().right.clone()).unwrap(), 13);
}

#[test]
fn mutate_hooks_own_variable_remapping() {
    register_test_hooks();
    // Rename every variable at its definition; uses follow through the remap.
    struct RenameDefinitions;
    impl MapDispatch for RenameDefinitions {
        fn dispatch_map(
            &mut self,
            value: &StructuralView,
            def_region_kind: DefRegionKind,
        ) -> Option<Result<Any>> {
            let var = value.as_node::<HookVarObj>()?;
            (def_region_kind == DefRegionKind::Simple)
                .then(|| Ok(Any::from(HookVar::new(var.name + 1))))
        }
    }
    let var = HookVar::new(1);
    let root = HookPair::new(HookBind::new(var.clone(), 5i64), var.clone(), 0);
    let mapped =
        structural_map(root.clone(), &mut RenameDefinitions, WalkOrder::PostOrder).unwrap();
    let outer = HookPair::try_from(mapped).unwrap();
    let bind = HookBind::try_from(outer.node().left.clone()).unwrap();
    assert_eq!(bind.data.var.data.name, 2);
    let use_site = HookVar::try_from(outer.node().right.clone()).unwrap();
    assert!(use_site.same_as(&bind.data.var));
    assert!(!use_site.same_as(&var));

    // A use with no enclosing definition keeps its identity.
    let root = HookPair::new(var.clone(), 5i64, 0);
    let mapped =
        structural_map(root.clone(), &mut RenameDefinitions, WalkOrder::PostOrder).unwrap();
    let outer = HookPair::try_from(mapped).unwrap();
    assert!(HookVar::try_from(outer.node().left.clone())
        .unwrap()
        .same_as(&var));

    // Pre-order: the definition is renamed before the hook descends, so the
    // bind hook sees the already-mapped node and remaps the original variable.
    let var = HookVar::new(7);
    let root = HookPair::new(HookBind::new(var.clone(), 5i64), var.clone(), 0);
    let mapped = structural_map(root, &mut RenameDefinitions, WalkOrder::PreOrder).unwrap();
    let outer = HookPair::try_from(mapped).unwrap();
    let bind = HookBind::try_from(outer.node().left.clone()).unwrap();
    assert_eq!(bind.data.var.data.name, 8);
    let use_site = HookVar::try_from(outer.node().right.clone()).unwrap();
    assert!(use_site.same_as(&bind.data.var));
}

#[test]
fn user_mutators_reenter_through_hooks() {
    register_test_hooks();
    #[derive(Default)]
    struct DoubleIntegers {
        remap: StructuralVarRemap,
        entries: usize,
    }
    impl StructuralMutator for DoubleIntegers {
        fn dispatch_mutate(
            &mut self,
            value: &StructuralView,
            def_region_kind: DefRegionKind,
        ) -> Result<Any> {
            self.entries += 1;
            if let Some(integer) = value.cast::<i64>() {
                Ok(Any::from(integer * 2))
            } else {
                self.default_mutate(value, def_region_kind)
            }
        }
        fn dispatch_maybe_inplace_mutate(
            &mut self,
            value: InplaceValue<'_>,
            def_region_kind: DefRegionKind,
        ) -> Result<Any> {
            self.entries += 1;
            if let Some(integer) = value.as_value().cast::<i64>() {
                Ok(Any::from(integer * 2))
            } else {
                self.default_maybe_inplace_mutate(value, def_region_kind)
            }
        }
        fn var_remap_get(&mut self, var: &StructuralView) -> Result<Option<Any>> {
            self.remap.get(var)
        }
        fn var_remap_set(&mut self, var: &StructuralView, mutated_value: &Any) -> Result<()> {
            self.remap.set(var, mutated_value)
        }
    }
    let root = HookPair::new(1i64, HookPair::new(2i64, 3i64, 0), 0);
    let mut mutator = DoubleIntegers::default();
    let mapped = structural_mutate(root.clone(), &mut mutator).unwrap();
    // Root, 1, inner pair, 2, 3.
    assert_eq!(mutator.entries, 5);
    let outer = HookPair::try_from(mapped).unwrap();
    assert_eq!(i64::try_from(outer.node().left.clone()).unwrap(), 2);
    let inner = HookPair::try_from(outer.node().right.clone()).unwrap();
    assert_eq!(i64::try_from(inner.node().right.clone()).unwrap(), 6);
}

#[test]
fn rust_hooks_take_precedence_over_cpp_hooks() {
    // Reference the test library so its C++ registrations are linked.
    assert_eq!(unsafe { tvm_ffi_sys::TVMFFITestingDummyTarget() }, 0);
    let graph = Function::get_global("testing.make_visit_region_graph")
        .unwrap()
        .call_tuple((true,))
        .unwrap();
    // The C++ hook visits the child under Simple, then the integer 4.
    assert_eq!(
        walk_integers(&graph, WalkOrder::PreOrder),
        vec![
            (1, DefRegionKind::Simple),
            (2, DefRegionKind::Pattern),
            (3, DefRegionKind::Simple),
            (4, DefRegionKind::None),
        ]
    );

    fn rust_hook(
        value: &StructuralView,
        children: &mut VisitChildren<'_>,
    ) -> Result<(), VisitHalt> {
        let object = tvm_ffi::object::ObjectRef::try_from(value.to_owned()).unwrap();
        let child = tvm_ffi::FieldGetter::new(value.type_index(), "child")
            .unwrap()
            .get::<_, tvm_ffi::object::ObjectRef>(&**tvm_ffi::object::ObjectRef::data(&object))
            .unwrap();
        children.visit_with(&child, DefRegionKind::Pattern)?;
        children.visit(&7i64)
    }
    let key = unsafe { TVMFFIByteArray::from_str("testing.TestVisitHook") };
    let mut type_index = -1;
    assert_eq!(unsafe { TVMFFITypeKeyToIndex(&key, &mut type_index) }, 0);
    register_structural_hooks_for_type_index(
        type_index,
        StructuralHooks {
            visit: Some(rust_hook),
            mutate: None,
        },
    )
    .unwrap();
    assert_eq!(
        walk_integers(&graph, WalkOrder::PreOrder),
        vec![
            (1, DefRegionKind::Pattern),
            (2, DefRegionKind::Pattern),
            (3, DefRegionKind::Pattern),
            (7, DefRegionKind::None),
        ]
    );
    // Clearing the hooks restores the C++ hook.
    register_structural_hooks_for_type_index(type_index, StructuralHooks::default()).unwrap();
    assert_eq!(walk_integers(&graph, WalkOrder::PreOrder).len(), 4);
    assert_eq!(walk_integers(&graph, WalkOrder::PreOrder)[3].0, 4);
}

#[test]
fn registration_rejects_non_object_and_container_types() {
    let hooks = StructuralHooks {
        visit: Some(var_visit),
        mutate: None,
    };
    let error = register_structural_hooks_for_type_index(TypeIndex::kTVMFFIInt as i32, hooks)
        .err()
        .unwrap();
    assert!(error.message().contains("object types"));
    for index in [
        TypeIndex::kTVMFFIArray,
        TypeIndex::kTVMFFIList,
        TypeIndex::kTVMFFIMap,
        TypeIndex::kTVMFFIDict,
        TypeIndex::kTVMFFIStr,
        TypeIndex::kTVMFFIBytes,
    ] {
        let error = register_structural_hooks_for_type_index(index as i32, hooks)
            .err()
            .unwrap();
        assert!(error.message().contains("descend natively"));
    }
    let error = register_structural_hooks_for_type_index(1 << 20, hooks)
        .err()
        .unwrap();
    assert!(error.message().contains("beyond"));
}
