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

use std::cell::{Cell, RefCell};
use std::sync::LazyLock;
use tvm_ffi::reflection::TypeAttrColumn;
use tvm_ffi::tvm_ffi_sys::{
    TVMFFIAny, TVMFFIByteArray, TVMFFIGetTypeInfo, TVMFFIObject, TVMFFITypeGetOrAllocIndex,
    TVMFFITypeMetadata, TVMFFITypeRegisterMetadata,
};
use tvm_ffi::{
    dispatch, structural_visit, Any, AnyView, Array, DefRegionKind, Dispatcher, Error, Function,
    IntoDispatcher, Result, StructuralVisitEngine, StructuralVisitorBase, StructuralVisitorObj,
    StructuralVisitorView, VisitInterrupt, RUNTIME_ERROR,
};

thread_local! {
    static DROPPED: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
    static FAIL_INIT: Cell<bool> = const { Cell::new(false) };
    static BASE_RESULT: RefCell<Option<Result<Option<VisitInterrupt>>>> = const { RefCell::new(None) };
}

// This base deliberately moves the engine dispatcher well beyond the ordinary
// prefix. Its private fields have a registered native size but no owning wrapper.
#[repr(C)]
struct CountingBase {
    prefix: StructuralVisitorObj,
    cookie: [u64; 8],
    defaults: Cell<usize>,
}

impl Drop for CountingBase {
    fn drop(&mut self) {
        assert_eq!(self.cookie, [0xfeed_dead_beef; 8]);
        DROPPED.with_borrow_mut(|drops| drops.push(self.defaults.get()));
    }
}

// SAFETY: the C prefix is at offset zero, the complete size is registered, and
// the only mutable field uses Cell. No safe API exposes an owning runtime. The
// reused ordinary hooks obey the synchronous visitor contract and use its vtable.
unsafe impl StructuralVisitorBase for CountingBase {
    fn new() -> Result<Self> {
        if FAIL_INIT.get() {
            return Err(Error::new(RUNTIME_ERROR, "base initialization failed", ""));
        }
        Ok(Self {
            prefix: StructuralVisitorObj::default(),
            cookie: [0xfeed_dead_beef; 8],
            defaults: Cell::new(0),
        })
    }

    fn type_index() -> i32 {
        static INDEX: LazyLock<i32> = LazyLock::new(|| unsafe {
            let parent = StructuralVisitorObj::type_index();
            let depth = (*TVMFFIGetTypeInfo(parent)).type_depth + 1;
            let index = TVMFFITypeGetOrAllocIndex(
                &TVMFFIByteArray::from_str("test.CountingVisitor"),
                -1,
                depth,
                0,
                0,
                parent,
            );
            assert!(index >= 0);
            assert_eq!(
                TVMFFITypeRegisterMetadata(
                    index,
                    &TVMFFITypeMetadata {
                        doc: TVMFFIByteArray::from_str(""),
                        creator: None,
                        total_size: std::mem::size_of::<CountingBase>() as i32,
                        structural_eq_hash_kind: 0,
                    }
                ),
                0
            );
            index
        });
        *INDEX
    }

    unsafe fn default_visit_raw(runtime: *mut Self, value: AnyView<'_>) -> TVMFFIAny {
        assert_eq!(
            (*runtime.cast::<TVMFFIObject>()).type_index,
            Self::type_index()
        );
        (*runtime).defaults.set((*runtime).defaults.get() + 1);
        // A default with observable behavior: redirect 10 to children 11 and 12.
        if value.downcast_ref::<i64>().copied() == Some(10) {
            let children = Array::from_vec(vec![11_i64, 12]);
            return StructuralVisitorObj::default_visit_raw(runtime.cast(), (&children).into());
        }
        if value.downcast_ref::<i64>().copied() == Some(-1) {
            let result = BASE_RESULT.with_borrow_mut(Option::take).unwrap();
            return Any::into_raw_ffi_any(match result {
                Ok(None) => Any::new(),
                Ok(Some(interrupt)) => interrupt.into(),
                Err(error) => error.into(),
            });
        }
        // Exercise the shared resolved-hook entry as well as ordinary fallback.
        if value.downcast_ref::<Array<Any>>().cloned().is_some() {
            let hook = TypeAttrColumn::lookup("__s_visit__")
                .and_then(|column| column.get(value.type_index()))
                .unwrap();
            return StructuralVisitorObj::call_structural_visit_hook_raw(
                runtime.cast(),
                *value.as_raw_ffi_any(),
                *AnyView::from(&hook).as_raw_ffi_any(),
            );
        }
        StructuralVisitorObj::default_visit_raw(runtime.cast(), value)
    }
}

type View<'a, D> = StructuralVisitorView<'a, D, CountingBase>;

fn visit<'a, H, M, D, O>(root: impl Into<AnyView<'a>>, visitor: H) -> Result<Option<VisitInterrupt>>
where
    O: std::borrow::BorrowMut<D>,
    for<'v, 'c, 'r> H: IntoDispatcher<
        (
            AnyView<'v>,
            &'c mut StructuralVisitorView<'r, D, CountingBase>,
        ),
        Option<Result<Option<VisitInterrupt>>>,
        M,
        Target = D,
        Output = O,
    >,
    for<'s, 'v, 'c, 'r> &'s mut D: Dispatcher<
        (
            AnyView<'v>,
            &'c mut StructuralVisitorView<'r, D, CountingBase>,
        ),
        Option<Result<Option<VisitInterrupt>>>,
    >,
{
    StructuralVisitEngine::<CountingBase>::visit_root(root, visitor)
}

#[derive(Default)]
struct Trace {
    leaves: Vec<(i64, DefRegionKind)>,
    defaults: Vec<usize>,
}

#[dispatch(visit, (AnyView<'_>, &mut View<'_, Self>), Result<Option<VisitInterrupt>>)]
impl Trace {
    fn visit_integer(
        &mut self,
        value: i64,
        visitor: &mut View<'_, Self>,
    ) -> Result<Option<VisitInterrupt>> {
        self.leaves.push((value, visitor.def_region_kind()));
        self.defaults.push(visitor.base().defaults.get());
        match value {
            0 => visitor.with_def_region_kind(DefRegionKind::Pattern, |view| {
                let base = view.base();
                let before = base.defaults.get();
                let result = view.default_visit(self, &10_i64);
                assert_eq!(base.defaults.get(), before + 1);
                assert_eq!(base.cookie, [0xfeed_dead_beef; 8]);
                assert_eq!(view.def_region_kind(), DefRegionKind::Pattern);
                result
            }),
            1 => {
                visitor.visit(self, &Array::from_vec(vec![2_i64, 3]))?;
                // Resume the parent dispatcher after native recursion, then recurse again.
                self.leaves.push((4, visitor.def_region_kind()));
                visitor.visit(self, &5_i64)
            }
            _ => Ok(None),
        }
    }
}

#[test]
fn larger_base_defaults_recurse_and_restore_dispatcher_and_region() {
    DROPPED.with_borrow_mut(Vec::clear);
    assert!(
        std::mem::size_of::<StructuralVisitEngine<CountingBase>>()
            > std::mem::size_of::<StructuralVisitEngine>()
    );
    assert_eq!(
        std::mem::size_of::<View<'_, Trace>>(),
        std::mem::size_of::<usize>()
    );
    let root = Array::from_vec(vec![0_i64, 1, 6]);
    let mut trace = Trace::default();
    assert!(visit(&root, &mut trace).unwrap().is_none());
    use DefRegionKind::{None as Use, Pattern};
    assert_eq!(
        trace.leaves,
        [
            (0, Use),
            (11, Pattern),
            (12, Pattern),
            (1, Use),
            (2, Use),
            (3, Use),
            (4, Use),
            (5, Use),
            (6, Use)
        ]
    );
    assert_eq!(trace.defaults, [1, 2, 2, 2, 3, 3, 3, 3]);
    DROPPED.with_borrow(|drops| assert_eq!(*drops, [3]));
}

#[test]
fn unmatched_scalar_uses_selected_default_and_owned_dispatcher() {
    struct Empty;
    impl<Dummy> Dispatcher<Dummy, Option<Result<Option<VisitInterrupt>>>> for &mut Empty {
        fn dispatch_tuple(self, _: Dummy) -> Option<Result<Option<VisitInterrupt>>> {
            None
        }
    }
    DROPPED.with_borrow_mut(Vec::clear);
    assert!(visit(&10_i64, Empty).unwrap().is_none());
    DROPPED.with_borrow(|drops| assert_eq!(*drops, [3]));
    assert!(structural_visit(&Array::from_vec(vec![1_i64]), Empty)
        .unwrap()
        .is_none());
    assert!(structural_visit(&Array::from_vec(vec![1_i64]), &mut Empty)
        .unwrap()
        .is_none());
}

#[test]
fn ordinary_reflected_fallback_uses_specialized_vtable() {
    assert_eq!(
        unsafe { tvm_ffi::tvm_ffi_sys::TVMFFITestingDummyTarget() },
        0
    );
    let root = Function::get_global("testing.make_visit_region_graph")
        .unwrap()
        .call_tuple((false,))
        .unwrap();
    let mut trace = Trace::default();
    assert!(visit(&root, &mut trace).unwrap().is_none());
    assert!(trace.leaves.iter().any(|&(value, _)| value == 2));
    assert!(trace.leaves.iter().any(|&(value, _)| value == 3));
}

#[test]
fn actual_dispatcher_identity_is_checked_before_recursive_publication() {
    struct Nonzero(usize);
    impl<'a, 'b, 'c>
        Dispatcher<(AnyView<'a>, &'b mut View<'c, Nonzero>), Option<Result<Option<VisitInterrupt>>>>
        for &mut Nonzero
    {
        fn dispatch_tuple(
            self,
            (_, view): (AnyView<'a>, &'b mut View<'c, Nonzero>),
        ) -> Option<Result<Option<VisitInterrupt>>> {
            let mut other = Nonzero(1);
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                view.default_visit(&mut other, &1_i64)
            }));
            assert!(result.is_err());
            self.0 += 1;
            Some(Ok(None))
        }
    }
    let mut visitor = Nonzero(0);
    visit(&1_i64, &mut visitor).unwrap();
    assert_eq!(visitor.0, 1);
}

#[test]
fn base_failure_and_owned_results_release_exactly_once() {
    struct Unmatched;
    impl<Args> Dispatcher<Args, Option<Result<Option<VisitInterrupt>>>> for &mut Unmatched {
        fn dispatch_tuple(self, _: Args) -> Option<Result<Option<VisitInterrupt>>> {
            None
        }
    }
    DROPPED.with_borrow_mut(Vec::clear);
    FAIL_INIT.set(true);
    assert!(visit(&1_i64, Unmatched)
        .err()
        .unwrap()
        .to_string()
        .contains("initialization failed"));
    FAIL_INIT.set(false);
    DROPPED.with_borrow(|drops| assert!(drops.is_empty()));

    let payload = Any::from(Array::from_vec(vec![42_i64]));
    BASE_RESULT
        .with_borrow_mut(|slot| *slot = Some(Ok(Some(VisitInterrupt::new(payload.clone())))));
    let result = visit(&Array::from_vec(vec![-1_i64, 99]), Unmatched)
        .unwrap()
        .unwrap();
    assert_eq!(payload.debug_strong_count(), Some(2));
    DROPPED.with_borrow(|drops| assert_eq!(*drops, [2]));
    drop(result);
    assert_eq!(payload.debug_strong_count(), Some(1));

    let error = Error::new(RUNTIME_ERROR, "base descent failed", "");
    let owner = Any::from(error.clone());
    BASE_RESULT.with_borrow_mut(|slot| *slot = Some(Err(error)));
    let result = visit(&Array::from_vec(vec![-1_i64, 99]), Unmatched)
        .err()
        .unwrap();
    assert!(result.to_string().contains("base descent failed"));
    drop(result);
    assert_eq!(owner.debug_strong_count(), Some(1));
    DROPPED.with_borrow(|drops| assert_eq!(*drops, [2, 2]));
}

#[test]
fn callback_outcomes_preserve_reborrows_and_owners_through_specialized_defaults() {
    struct Halt {
        result: Option<Result<Option<VisitInterrupt>>>,
        seen: Vec<i64>,
    }
    #[dispatch(visit, (AnyView<'_>, &mut View<'_, Self>), Result<Option<VisitInterrupt>>)]
    impl Halt {
        fn visit_integer(
            &mut self,
            value: i64,
            view: &mut View<'_, Self>,
        ) -> Result<Option<VisitInterrupt>> {
            self.seen.push(value);
            match value {
                10 => {
                    let result = view.with_def_region_kind(DefRegionKind::Pattern, |view| {
                        view.default_visit(self, &10_i64)
                    });
                    assert_eq!(view.def_region_kind(), DefRegionKind::None);
                    // The parent borrow must be usable after either halted descent.
                    view.visit(self, &99_i64)?;
                    result
                }
                11 => {
                    assert_eq!(view.def_region_kind(), DefRegionKind::Pattern);
                    self.result.take().unwrap()
                }
                99 => Ok(None),
                _ => panic!("descent continued after the callback halted it"),
            }
        }
    }
    for fail in [false, true] {
        DROPPED.with_borrow_mut(Vec::clear);
        let payload = Any::from(Array::from_vec(vec![42_i64]));
        let error = Error::new(RUNTIME_ERROR, "callback failed", "");
        let error_owner = Any::from(error.clone());
        let mut visitor = Halt {
            result: Some(if fail {
                Err(error)
            } else {
                drop(error);
                Ok(Some(VisitInterrupt::new(payload.clone())))
            }),
            seen: vec![],
        };
        let result = visit(&Array::from_vec(vec![10_i64, 100]), &mut visitor);
        assert_eq!(visitor.seen, [10, 11, 99]);
        DROPPED.with_borrow(|drops| assert_eq!(*drops, [2]));
        if fail {
            assert!(result
                .as_ref()
                .err()
                .unwrap()
                .to_string()
                .contains("callback failed"));
        } else {
            assert_eq!(payload.debug_strong_count(), Some(2));
        }
        drop(result);
        assert_eq!(payload.debug_strong_count(), Some(1));
        assert_eq!(error_owner.debug_strong_count(), Some(1));
    }
}
