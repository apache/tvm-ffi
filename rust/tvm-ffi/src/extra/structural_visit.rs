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

//! Native Rust structural visiting.
//!
//! This module separates the two jobs involved in a visit:
//!
//! * [`VisitValue`] provides borrowed matching for generated Rust dispatch.
//! * `NativeWalker` owns recursion through containers and reflected fields.
//!
//! The runtime object registry is open, so the walker uses the stable tvm-ffi
//! reflection ABI for arbitrary registered object types. That ABI is only the
//! object-description boundary: traversal, control flow, typed dispatch,
//! visitor state, and definition-region propagation remain in Rust.
//!
//! A Rust handler may override a type's children by visiting them through
//! [`VisitCtx`] and returning [`WalkResult::Skip`]. No C++
//! `ffi.StructuralVisitor` is constructed and no C++ default-visit function is
//! called. A non-container type with a foreign `__s_visit__` hook must be
//! handled this way; advancing into its default children is rejected instead
//! of silently substituting reflection with potentially different semantics.

use std::ops::ControlFlow;
use std::os::raw::c_void;
use std::ptr::NonNull;

use crate::any::{Any, AnyView};
use crate::error::{Error, Result, RUNTIME_ERROR, TYPE_ERROR};
use crate::function::Function;
use crate::object::ObjectCore;
use crate::tvm_ffi_sys::TVMFFIFieldFlagBitMask::{
    kTVMFFIFieldFlagBitMaskSEqHashDefNonRecursive, kTVMFFIFieldFlagBitMaskSEqHashDefRecursive,
    kTVMFFIFieldFlagBitMaskSEqHashIgnore,
};
use crate::tvm_ffi_sys::{
    TVMFFIAny, TVMFFIByteArray, TVMFFIDefRegionKind, TVMFFIFieldInfo, TVMFFIGetTypeAttrColumn,
    TVMFFIGetTypeInfo, TVMFFIObject, TVMFFITypeAttrColumn, TVMFFITypeIndex,
};

const STRUCTURAL_VISIT_ATTR: &str = "__s_visit__";
const FLAG_SEQ_HASH_IGNORE: i64 = kTVMFFIFieldFlagBitMaskSEqHashIgnore as i64;
const FLAG_SEQ_HASH_DEF_RECURSIVE: i64 = kTVMFFIFieldFlagBitMaskSEqHashDefRecursive as i64;
const FLAG_SEQ_HASH_DEF_NON_RECURSIVE: i64 = kTVMFFIFieldFlagBitMaskSEqHashDefNonRecursive as i64;

/// What a callback asks the Rust walker to do with the current value.
pub enum WalkResult {
    /// Continue and visit this value's children.
    Advance,
    /// Continue without visiting this value's children or firing its exit hook.
    Skip,
    /// Halt the entire traversal.
    Interrupt,
    /// Halt the entire traversal and return a payload to the caller.
    InterruptWith(Any),
}

impl WalkResult {
    /// Halt traversal with an FFI-compatible payload.
    pub fn interrupt_with<T: Into<Any>>(payload: T) -> Self {
        Self::InterruptWith(payload.into())
    }
}

/// Convert either an infallible or fallible typed handler result.
///
/// This keeps simple handlers terse while allowing a handler to return
/// `tvm_ffi::Result<WalkResult>` and use `?`.
pub trait IntoVisitResult {
    fn into_visit_result(self) -> Result<WalkResult>;
}

impl IntoVisitResult for WalkResult {
    fn into_visit_result(self) -> Result<WalkResult> {
        Ok(self)
    }
}

impl IntoVisitResult for Result<WalkResult> {
    fn into_visit_result(self) -> Result<WalkResult> {
        self
    }
}

/// Whether a callback runs before or after a value's children.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// Before the value's children.
    Enter,
    /// After the value's children.
    Exit,
}

/// Callback order for [`structural_walk`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WalkOrder {
    /// Run the typed handler before the current value's children.
    #[default]
    PreOrder,
    /// Run the typed handler after the current value's children.
    PostOrder,
}

/// Definition-region state active at the current value.
///
/// Reflected fields marked `SEqHashDefRecursive` or
/// `SEqHashDefNonRecursive` override the inherited state for that field's
/// complete recursive visit.
#[repr(i32)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DefRegionKind {
    /// The value is outside a definition region.
    #[default]
    None = 0,
    /// Definitions apply recursively through the visited value.
    Recursive = 1,
    /// Definitions apply to the visited value using non-recursive semantics.
    NonRecursive = 2,
}

const _: () = {
    assert!(DefRegionKind::None as i32 == TVMFFIDefRegionKind::kTVMFFIDefRegionKindNone as i32);
    assert!(
        DefRegionKind::Recursive as i32
            == TVMFFIDefRegionKind::kTVMFFIDefRegionKindRecursive as i32
    );
    assert!(
        DefRegionKind::NonRecursive as i32
            == TVMFFIDefRegionKind::kTVMFFIDefRegionKindNonRecursive as i32
    );
};

/// Result of a completed Rust walk.
///
/// `Continue(())` means the whole graph was visited. `Break(payload)` means a
/// handler interrupted it; a payload-less interrupt carries `ffi::None`.
pub type VisitOutcome = ControlFlow<Any>;

/// Fallible result returned by generated typed dispatch.
#[doc(hidden)]
pub type VisitResult = Result<WalkResult>;

/// A borrowed view of a raw tvm-ffi value.
///
/// Generated visitors match this value without taking ownership: borrowed
/// object-node handlers use [`VisitValue::as_node`], while POD or object-ref
/// value handlers use [`VisitValue::cast`].
#[repr(transparent)]
pub struct VisitValue(TVMFFIAny);

impl VisitValue {
    #[inline]
    fn from_raw(raw: TVMFFIAny) -> Self {
        VisitValue(raw)
    }

    /// Convert the value into an owned typed handle.
    #[inline]
    pub fn cast<R: crate::type_traits::AnyCompatible>(&self) -> Option<R> {
        unsafe {
            if R::check_any_strict(&self.0) {
                Some(R::copy_from_any_view_after_check(&self.0))
            } else {
                None
            }
        }
    }

    /// Runtime type index stored in this value.
    #[inline]
    pub fn type_index(&self) -> i32 {
        self.0.type_index
    }

    /// Borrow the value as node type `N` if it is an instance of that type.
    #[inline]
    pub fn as_node<N: ObjectCore>(&self) -> Option<&N> {
        if self.0.type_index < TVMFFITypeIndex::kTVMFFIStaticObjectBegin as i32 {
            return None;
        }
        if !is_instance(self.0.type_index, N::type_index()) {
            return None;
        }
        Some(unsafe { &*(self.0.data_union.v_obj as *const N) })
    }
}

enum NativeHalt {
    Interrupt(Any),
    Error(Error),
}

impl From<Error> for NativeHalt {
    fn from(error: Error) -> Self {
        NativeHalt::Error(error)
    }
}

type NativeResult = std::result::Result<(), NativeHalt>;

/// Typed dispatch implemented by the visitor object itself.
///
/// [`crate::dispatch`] tests the implementation's `visit_*` methods in source
/// order. Borrowed node arguments use refcount-free subtype checks, owned
/// FFI-compatible arguments use exact value casts, and `&VisitValue` is a
/// catch-all. `None` asks the Rust walker to continue normally.
pub trait VisitDispatch: Sized {
    fn dispatch_visit(&mut self, value: &VisitValue, ctx: &mut VisitCtx<'_>)
        -> Option<VisitResult>;
}

/// Recursive traversal access passed to a typed handler.
///
/// The context contains the walker, not the visitor. A handler lends its
/// current `&mut self` back to [`VisitCtx::visit`], so nested traversal is an
/// ordinary checked Rust reborrow and needs no raw visitor pointer.
pub struct VisitCtx<'a> {
    walker: &'a NativeWalker,
    order: WalkOrder,
    def_region_kind: DefRegionKind,
    halted: Option<NativeHalt>,
}

impl VisitCtx<'_> {
    /// Return the definition-region state active at the current node.
    pub fn def_region_kind(&self) -> DefRegionKind {
        self.def_region_kind
    }

    /// Visit `child` immediately with the same typed dispatcher.
    pub fn visit<V, T>(&mut self, visitor: &mut V, child: &T) -> bool
    where
        V: VisitDispatch,
        for<'x> AnyView<'x>: From<&'x T>,
    {
        self.visit_with_def_region(visitor, child, self.def_region_kind)
    }

    /// Visit `child` under an explicitly selected definition-region state.
    ///
    /// The override is scoped to this recursive call. The current context is
    /// unchanged after success, error, or interruption.
    pub fn visit_with_def_region<V, T>(
        &mut self,
        visitor: &mut V,
        child: &T,
        def_region_kind: DefRegionKind,
    ) -> bool
    where
        V: VisitDispatch,
        for<'x> AnyView<'x>: From<&'x T>,
    {
        if self.halted.is_some() {
            return false;
        }
        let mut dispatch = DispatchVisitor {
            visitor,
            order: self.order,
        };
        let result =
            self.walker
                .visit_raw(raw_of(AnyView::from(child)), &mut dispatch, def_region_kind);
        self.absorb(result)
    }

    fn absorb(&mut self, result: NativeResult) -> bool {
        match result {
            Ok(()) => true,
            Err(halt) => {
                self.halted = Some(halt);
                false
            }
        }
    }
}

trait NativeVisit {
    fn order(&self) -> WalkOrder {
        WalkOrder::PreOrder
    }

    fn enter(&mut self, value: &VisitValue, ctx: &mut VisitCtx<'_>) -> Result<WalkResult>;

    fn exit(&mut self, _value: &VisitValue, _ctx: &mut VisitCtx<'_>) -> Result<WalkResult> {
        Ok(WalkResult::Advance)
    }
}

struct DispatchVisitor<'a, V> {
    visitor: &'a mut V,
    order: WalkOrder,
}

impl<V: VisitDispatch> NativeVisit for DispatchVisitor<'_, V> {
    fn order(&self) -> WalkOrder {
        self.order
    }

    fn enter(&mut self, value: &VisitValue, ctx: &mut VisitCtx<'_>) -> Result<WalkResult> {
        match self.order {
            WalkOrder::PreOrder => self
                .visitor
                .dispatch_visit(value, ctx)
                .unwrap_or(Ok(WalkResult::Advance)),
            WalkOrder::PostOrder => Ok(WalkResult::Advance),
        }
    }

    fn exit(&mut self, value: &VisitValue, ctx: &mut VisitCtx<'_>) -> Result<WalkResult> {
        match self.order {
            WalkOrder::PreOrder => Ok(WalkResult::Advance),
            WalkOrder::PostOrder => self
                .visitor
                .dispatch_visit(value, ctx)
                .unwrap_or(Ok(WalkResult::Advance)),
        }
    }
}

struct CallbackVisitor<F>(F);

impl<F, O> NativeVisit for CallbackVisitor<F>
where
    F: FnMut(&VisitValue, Phase, DefRegionKind) -> O,
    O: IntoVisitResult,
{
    fn enter(&mut self, value: &VisitValue, ctx: &mut VisitCtx<'_>) -> Result<WalkResult> {
        (self.0)(value, Phase::Enter, ctx.def_region_kind()).into_visit_result()
    }

    fn exit(&mut self, value: &VisitValue, ctx: &mut VisitCtx<'_>) -> Result<WalkResult> {
        (self.0)(value, Phase::Exit, ctx.def_region_kind()).into_visit_result()
    }
}

/// Stateless Rust recursion engine.
struct NativeWalker {
    structural_visit: Option<TypeAttrColumn>,
}

impl NativeWalker {
    fn new() -> Self {
        Self {
            structural_visit: type_attr_column(STRUCTURAL_VISIT_ATTR),
        }
    }

    fn visit_raw<V: NativeVisit>(
        &self,
        value: TVMFFIAny,
        visitor: &mut V,
        def_region_kind: DefRegionKind,
    ) -> NativeResult {
        if value.type_index == TVMFFITypeIndex::kTVMFFINone as i32 {
            return Ok(());
        }

        let visit_value = VisitValue::from_raw(value);
        let mut ctx = VisitCtx {
            walker: self,
            order: visitor.order(),
            def_region_kind,
            halted: None,
        };
        let enter = match visitor.enter(&visit_value, &mut ctx) {
            Ok(flow) => flow,
            Err(error) => return Err(Self::with_value_context(error.into(), value)),
        };
        if let Some(halt) = ctx.halted.take() {
            return Err(Self::with_value_context(halt, value));
        }
        match enter {
            WalkResult::Advance => {}
            WalkResult::Skip => return Ok(()),
            WalkResult::Interrupt => return Err(NativeHalt::Interrupt(Any::new())),
            WalkResult::InterruptWith(payload) => return Err(NativeHalt::Interrupt(payload)),
        }

        if let Err(halt) = self.visit_children_raw(value, visitor, def_region_kind) {
            return Err(Self::with_value_context(halt, value));
        }

        let exit = match visitor.exit(&visit_value, &mut ctx) {
            Ok(flow) => flow,
            Err(error) => return Err(Self::with_value_context(error.into(), value)),
        };
        if let Some(halt) = ctx.halted.take() {
            return Err(Self::with_value_context(halt, value));
        }
        match exit {
            WalkResult::Interrupt => Err(NativeHalt::Interrupt(Any::new())),
            WalkResult::InterruptWith(payload) => Err(NativeHalt::Interrupt(payload)),
            WalkResult::Advance | WalkResult::Skip => Ok(()),
        }
    }

    fn visit_children_raw<V: NativeVisit>(
        &self,
        value: TVMFFIAny,
        visitor: &mut V,
        def_region_kind: DefRegionKind,
    ) -> NativeResult {
        match value.type_index {
            x if x == TVMFFITypeIndex::kTVMFFIArray as i32
                || x == TVMFFITypeIndex::kTVMFFIList as i32 =>
            {
                return self.visit_sequence(value, visitor, def_region_kind);
            }
            x if x == TVMFFITypeIndex::kTVMFFIMap as i32
                || x == TVMFFITypeIndex::kTVMFFIDict as i32 =>
            {
                return self.visit_map(value, visitor, def_region_kind);
            }
            _ => {}
        }

        self.reject_foreign_structural_visit(value.type_index)?;
        if value.type_index < TVMFFITypeIndex::kTVMFFIStaticObjectBegin as i32 {
            Ok(())
        } else {
            self.visit_reflected_fields(value, visitor, def_region_kind)
        }
    }

    fn visit_sequence<V: NativeVisit>(
        &self,
        value: TVMFFIAny,
        visitor: &mut V,
        def_region_kind: DefRegionKind,
    ) -> NativeResult {
        let seq = unsafe { &*(value.data_union.v_obj as *const SeqPrefix) };
        if seq.size < 0 {
            return Err(runtime_error("native visitor: sequence reports a negative size").into());
        }
        if seq.data.is_null() && seq.size != 0 {
            return Err(runtime_error(
                "native visitor: non-empty sequence has a null data pointer",
            )
            .into());
        }
        let size = usize::try_from(seq.size)
            .map_err(|_| runtime_error("native visitor: sequence size does not fit usize"))?;
        if size == 0 {
            return Ok(());
        }

        if value.type_index == TVMFFITypeIndex::kTVMFFIList as i32 {
            // List storage may be invalidated by a re-entrant callback. Own a
            // snapshot before running the first callback.
            let children: Vec<Any> = {
                let cells = unsafe { std::slice::from_raw_parts(seq.data, size) };
                cells
                    .iter()
                    .map(|cell| Any::from(unsafe { view_of(cell) }))
                    .collect()
            };
            for (index, mut child) in children.into_iter().enumerate() {
                let raw = raw_of_owned(&mut child);
                self.visit_raw(raw, visitor, def_region_kind)
                    .map_err(|halt| {
                        with_error_context(halt, &format!("sequence item [{index}]"))
                    })?;
            }
            return Ok(());
        }

        // Array is immutable, so its element cells remain stable throughout
        // recursive callbacks and need no refcounted snapshot.
        let cells = unsafe { std::slice::from_raw_parts(seq.data, size) };
        for (index, child) in cells.iter().enumerate() {
            self.visit_raw(*child, visitor, def_region_kind)
                .map_err(|halt| with_error_context(halt, &format!("sequence item [{index}]")))?;
        }
        Ok(())
    }

    fn visit_map<V: NativeVisit>(
        &self,
        value: TVMFFIAny,
        visitor: &mut V,
        def_region_kind: DefRegionKind,
    ) -> NativeResult {
        // Map storage is private C++. The Rust binding itself uses these public
        // iterator functors; using them here does not invoke structural
        // visiting or transfer traversal control out of Rust.
        let is_dict = value.type_index == TVMFFITypeIndex::kTVMFFIDict as i32;
        let (size_name, iter_name) = if is_dict {
            ("ffi.DictSize", "ffi.DictForwardIterFunctor")
        } else {
            ("ffi.MapSize", "ffi.MapForwardIterFunctor")
        };
        let size = Function::get_global(size_name)?
            .call_packed(&[unsafe { view_of(&value) }])
            .and_then(i64::try_from)?;
        if size < 0 {
            return Err(runtime_error("native visitor: map reports a negative size").into());
        }
        let size = usize::try_from(size)
            .map_err(|_| runtime_error("native visitor: map size does not fit usize"))?;
        if size == 0 {
            return Ok(());
        }

        let iter_any =
            Function::get_global(iter_name)?.call_packed(&[unsafe { view_of(&value) }])?;
        let iter = Function::try_from(iter_any)?;

        if is_dict {
            // Dict mutation invalidates its iterator, so snapshot all entries
            // before dispatching to user code.
            let mut entries = Vec::with_capacity(size);
            for index in 0..size {
                let key = iter.call_packed(&[AnyView::from(&0i64)])?;
                let map_value = iter.call_packed(&[AnyView::from(&1i64)])?;
                entries.push((key, map_value));
                if index + 1 != size {
                    iter.call_packed(&[AnyView::from(&2i64)])?;
                }
            }

            for (index, (mut key, mut map_value)) in entries.into_iter().enumerate() {
                let key_raw = raw_of_owned(&mut key);
                self.visit_raw(key_raw, visitor, def_region_kind)
                    .map_err(|halt| with_error_context(halt, &format!("dict key [{index}]")))?;
                let value_raw = raw_of_owned(&mut map_value);
                self.visit_raw(value_raw, visitor, def_region_kind)
                    .map_err(|halt| with_error_context(halt, &format!("dict value [{index}]")))?;
            }
            return Ok(());
        }

        // Map is immutable. Retain only the current owned key/value pair.
        for index in 0..size {
            let mut key = iter.call_packed(&[AnyView::from(&0i64)])?;
            let mut map_value = iter.call_packed(&[AnyView::from(&1i64)])?;
            let key_raw = raw_of_owned(&mut key);
            self.visit_raw(key_raw, visitor, def_region_kind)
                .map_err(|halt| with_error_context(halt, &format!("map key [{index}]")))?;
            let value_raw = raw_of_owned(&mut map_value);
            self.visit_raw(value_raw, visitor, def_region_kind)
                .map_err(|halt| with_error_context(halt, &format!("map value [{index}]")))?;
            if index + 1 != size {
                iter.call_packed(&[AnyView::from(&2i64)])?;
            }
        }
        Ok(())
    }

    fn visit_reflected_fields<V: NativeVisit>(
        &self,
        value: TVMFFIAny,
        visitor: &mut V,
        def_region_kind: DefRegionKind,
    ) -> NativeResult {
        if unsafe { TVMFFIGetTypeInfo(value.type_index) }.is_null() {
            return Err(runtime_error(&format!(
                "native visitor: unregistered type index {}",
                value.type_index
            ))
            .into());
        }
        let object = unsafe { value.data_union.v_obj } as *mut u8;
        let halted = unsafe {
            for_each_field(value.type_index, |field| {
                match self.visit_reflected_field(object, field, visitor, def_region_kind) {
                    Ok(()) => ControlFlow::Continue(()),
                    Err(halt) => ControlFlow::Break(halt),
                }
            })
        };
        halted.map_or(Ok(()), Err)
    }

    unsafe fn visit_reflected_field<V: NativeVisit>(
        &self,
        object: *mut u8,
        field: &TVMFFIFieldInfo,
        visitor: &mut V,
        inherited_region: DefRegionKind,
    ) -> NativeResult {
        if field.flags & FLAG_SEQ_HASH_IGNORE != 0 {
            return Ok(());
        }

        let Some(getter) = field.getter else {
            return Err(NativeHalt::Error(runtime_error(&format!(
                "native visitor: reflected field `{}` has no getter",
                field.name.as_str()
            ))));
        };
        let address = object.offset(field.offset as isize) as *mut c_void;
        let mut child_raw = TVMFFIAny::new();
        if getter(address, &mut child_raw) != 0 {
            return Err(with_error_context(
                NativeHalt::Error(Error::from_raised()),
                &format!("field `{}`", field.name.as_str()),
            ));
        }

        // A reflection getter returns an owned Any. Keep it alive while the
        // recursive walk borrows its raw cell.
        let mut child = Any::from_raw_ffi_any(child_raw);
        let borrowed = raw_of_owned(&mut child);
        let child_region = field_def_region(field, inherited_region);
        self.visit_raw(borrowed, visitor, child_region)
            .map_err(|halt| with_error_context(halt, &format!("field `{}`", field.name.as_str())))
    }

    fn reject_foreign_structural_visit(&self, type_index: i32) -> Result<()> {
        let Some(attr) = self
            .structural_visit
            .and_then(|column| column.get(type_index))
        else {
            return Ok(());
        };
        match attr.type_index {
            x if x == TVMFFITypeIndex::kTVMFFINone as i32 => Ok(()),
            x if x == TVMFFITypeIndex::kTVMFFIOpaquePtr as i32
                || x == TVMFFITypeIndex::kTVMFFIFunction as i32 =>
            {
                let value_type = if type_index < TVMFFITypeIndex::kTVMFFIStaticObjectBegin as i32 {
                    format!("type index {type_index}")
                } else {
                    format!("type `{}`", type_key_of(type_index))
                };
                Err(runtime_error(&format!(
                    "native visitor: {value_type} registers foreign `{STRUCTURAL_VISIT_ATTR}`; \
                     use a matching pre-order Rust handler, visit its children through \
                     `VisitCtx`, and return `WalkResult::Skip`"
                )))
            }
            _ => Err(Error::new(
                TYPE_ERROR,
                &format!(
                    "{STRUCTURAL_VISIT_ATTR} must be an opaque function pointer or ffi.Function"
                ),
                "",
            )),
        }
    }

    fn with_value_context(halt: NativeHalt, value: TVMFFIAny) -> NativeHalt {
        if value.type_index < TVMFFITypeIndex::kTVMFFIStaticObjectBegin as i32 {
            halt
        } else {
            with_error_context(halt, &format!("object `{}`", type_key_of(value.type_index)))
        }
    }
}

/// Visit `root` in pre-order with typed handlers stored in `visitor`.
pub fn structural_visit<R, V>(root: &R, visitor: &mut V) -> Result<VisitOutcome>
where
    V: VisitDispatch,
    for<'x> AnyView<'x>: From<&'x R>,
{
    structural_walk(root, visitor, WalkOrder::PreOrder)
}

/// Walk `root` with typed handlers and state stored in `walker`.
///
/// `walker` may use [`crate::dispatch`] exactly like a visitor. Each matching
/// handler runs once, before or after the value's children according to
/// `order`.
pub fn structural_walk<R, W>(root: &R, walker: &mut W, order: WalkOrder) -> Result<VisitOutcome>
where
    W: VisitDispatch,
    for<'x> AnyView<'x>: From<&'x R>,
{
    let native_walker = NativeWalker::new();
    let mut dispatch = DispatchVisitor {
        visitor: walker,
        order,
    };
    finish(native_walker.visit_raw(
        raw_of(AnyView::from(root)),
        &mut dispatch,
        DefRegionKind::None,
    ))
}

/// Native pre/post walk used by analyses that need to observe every raw value.
pub fn walk<R, F, O>(root: &R, mut callback: F) -> Result<VisitOutcome>
where
    for<'x> AnyView<'x>: From<&'x R>,
    F: FnMut(&VisitValue, Phase) -> O,
    O: IntoVisitResult,
{
    walk_with_context(root, move |value, phase, _def_region_kind| {
        callback(value, phase)
    })
}

/// Native pre/post walk whose callback also receives definition-region state.
pub fn walk_with_context<R, F, O>(root: &R, callback: F) -> Result<VisitOutcome>
where
    for<'x> AnyView<'x>: From<&'x R>,
    F: FnMut(&VisitValue, Phase, DefRegionKind) -> O,
    O: IntoVisitResult,
{
    let walker = NativeWalker::new();
    let mut callback = CallbackVisitor(callback);
    finish(walker.visit_raw(
        raw_of(AnyView::from(root)),
        &mut callback,
        DefRegionKind::None,
    ))
}

fn finish(result: NativeResult) -> Result<VisitOutcome> {
    match result {
        Ok(()) => Ok(ControlFlow::Continue(())),
        Err(NativeHalt::Error(error)) => Err(error),
        Err(NativeHalt::Interrupt(payload)) => Ok(ControlFlow::Break(payload)),
    }
}

fn field_def_region(field: &TVMFFIFieldInfo, inherited: DefRegionKind) -> DefRegionKind {
    if field.flags & FLAG_SEQ_HASH_DEF_NON_RECURSIVE != 0 {
        DefRegionKind::NonRecursive
    } else if field.flags & FLAG_SEQ_HASH_DEF_RECURSIVE != 0 {
        DefRegionKind::Recursive
    } else {
        inherited
    }
}

fn with_error_context(halt: NativeHalt, frame: &str) -> NativeHalt {
    match halt {
        NativeHalt::Error(error) => NativeHalt::Error(Error::with_appended_backtrace(
            error,
            &format!("[native structural visit] {frame}\n"),
        )),
        interrupt => interrupt,
    }
}

fn runtime_error(message: &str) -> Error {
    Error::new(RUNTIME_ERROR, message, "")
}

/// Layout prefix shared by the C++ `ArrayObj` and `ListObj`.
#[repr(C)]
struct SeqPrefix {
    _header: TVMFFIObject,
    data: *const TVMFFIAny,
    size: i64,
}

const _: () = {
    assert!(std::mem::offset_of!(SeqPrefix, data) == 24);
    assert!(std::mem::offset_of!(SeqPrefix, size) == 32);
};

#[derive(Clone, Copy)]
struct TypeAttrColumn(NonNull<TVMFFITypeAttrColumn>);

impl TypeAttrColumn {
    /// Copy one borrowed cell; ownership remains with the registry.
    fn get(self, type_index: i32) -> Option<TVMFFIAny> {
        unsafe {
            let column = self.0.as_ref();
            let index = type_index - column.begin_index;
            if index < 0 || index >= column.size || column.data.is_null() {
                None
            } else {
                Some(*column.data.offset(index as isize))
            }
        }
    }
}

fn type_attr_column(attr_name: &str) -> Option<TypeAttrColumn> {
    unsafe {
        let attr_name = TVMFFIByteArray::from_str(attr_name);
        NonNull::new(TVMFFIGetTypeAttrColumn(&attr_name).cast_mut()).map(TypeAttrColumn)
    }
}

fn type_key_of(type_index: i32) -> String {
    unsafe {
        let info = TVMFFIGetTypeInfo(type_index);
        if info.is_null() {
            format!("<type_index {type_index}>")
        } else {
            (*info).type_key.as_str().to_string()
        }
    }
}

fn is_instance(object_type_index: i32, base_type_index: i32) -> bool {
    if object_type_index == base_type_index {
        return true;
    }
    unsafe {
        let info = TVMFFIGetTypeInfo(object_type_index);
        let base_info = TVMFFIGetTypeInfo(base_type_index);
        if info.is_null() || base_info.is_null() {
            return false;
        }
        let base_depth = (*base_info).type_depth;
        if (*info).type_depth <= base_depth {
            return false;
        }
        let ancestors = (*info).type_acenstors;
        if ancestors.is_null() {
            return false;
        }
        let ancestor = *ancestors.offset(base_depth as isize);
        !ancestor.is_null() && (*ancestor).type_index == base_type_index
    }
}

/// Visit every reflected field of `type_index` and its ancestors in the same
/// parent-to-child order as C++ `ForEachFieldInfoWithEarlyStop`.
///
/// # Safety
///
/// `type_index` must be a registered type index.
unsafe fn for_each_field<B>(
    type_index: i32,
    mut callback: impl FnMut(&'static TVMFFIFieldInfo) -> ControlFlow<B>,
) -> Option<B> {
    let info = TVMFFIGetTypeInfo(type_index);
    if info.is_null() {
        return None;
    }

    // Ancestor slot 0 is the root Object. C++ starts at slot 1, walks toward
    // the immediate parent, then visits the concrete type's own fields.
    for depth in 1..(*info).type_depth {
        let ancestor = *(*info).type_acenstors.offset(depth as isize);
        if let Some(value) = visit_field_level(ancestor, &mut callback) {
            return Some(value);
        }
    }
    visit_field_level(info, &mut callback)
}

unsafe fn visit_field_level<B>(
    info: *const crate::tvm_ffi_sys::TVMFFITypeInfo,
    callback: &mut impl FnMut(&'static TVMFFIFieldInfo) -> ControlFlow<B>,
) -> Option<B> {
    if info.is_null() || (*info).fields.is_null() {
        return None;
    }
    let fields = std::slice::from_raw_parts((*info).fields, (*info).num_fields as usize);
    for field in fields {
        // C reflection tables are immortal once registered.
        let field: &'static TVMFFIFieldInfo = &*(field as *const TVMFFIFieldInfo);
        if let ControlFlow::Break(value) = callback(field) {
            return Some(value);
        }
    }
    None
}

fn raw_of(view: AnyView<'_>) -> TVMFFIAny {
    *view.as_raw_ffi_any()
}

fn raw_of_owned(any: &mut Any) -> TVMFFIAny {
    *any.as_raw_ffi_any()
}

unsafe fn view_of(raw: &TVMFFIAny) -> AnyView<'_> {
    unsafe { AnyView::from_raw_ffi_any(*raw) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Array;

    struct RegionProbe(Vec<DefRegionKind>);

    impl NativeVisit for RegionProbe {
        fn enter(&mut self, _value: &VisitValue, ctx: &mut VisitCtx<'_>) -> Result<WalkResult> {
            self.0.push(ctx.def_region_kind());
            Ok(WalkResult::Advance)
        }
    }

    #[derive(Default)]
    struct TypedRegionProbe(Vec<DefRegionKind>);

    #[crate::dispatch(visit)]
    impl TypedRegionProbe {
        fn visit_integer(&mut self, _value: i64, ctx: &mut VisitCtx<'_>) -> WalkResult {
            self.0.push(ctx.def_region_kind());
            WalkResult::Advance
        }
    }

    unsafe extern "C" fn clone_any_field(field: *mut c_void, result: *mut TVMFFIAny) -> i32 {
        let value = &*(field as *const Any);
        *result = Any::into_raw_ffi_any(value.clone());
        0
    }

    #[test]
    fn def_region_is_inherited_through_containers() {
        let root = Array::new(vec![1i64, 2]);
        let walker = NativeWalker::new();
        let mut probe = RegionProbe(Vec::new());
        assert!(walker
            .visit_raw(
                raw_of(AnyView::from(&root)),
                &mut probe,
                DefRegionKind::Recursive,
            )
            .is_ok());
        assert_eq!(probe.0, vec![DefRegionKind::Recursive; 3]);
    }

    #[test]
    fn reflected_field_def_region_reaches_typed_handler_and_restores() {
        let walker = NativeWalker::new();
        let mut probe = TypedRegionProbe::default();
        let mut dispatch = DispatchVisitor {
            visitor: &mut probe,
            order: WalkOrder::PreOrder,
        };
        let mut value = Any::from(7i64);
        let mut field: TVMFFIFieldInfo = unsafe { std::mem::zeroed() };
        field.name = unsafe { TVMFFIByteArray::from_str("value") };
        field.getter = Some(clone_any_field);
        let object = (&mut value as *mut Any).cast::<u8>();

        for flags in [
            FLAG_SEQ_HASH_DEF_RECURSIVE,
            0,
            FLAG_SEQ_HASH_DEF_NON_RECURSIVE,
            FLAG_SEQ_HASH_DEF_NON_RECURSIVE | FLAG_SEQ_HASH_DEF_RECURSIVE,
            FLAG_SEQ_HASH_IGNORE,
        ] {
            field.flags = flags;
            assert!(unsafe {
                walker.visit_reflected_field(object, &field, &mut dispatch, DefRegionKind::None)
            }
            .is_ok());
        }
        assert_eq!(
            probe.0,
            vec![
                DefRegionKind::Recursive,
                DefRegionKind::None,
                DefRegionKind::NonRecursive,
                DefRegionKind::NonRecursive,
            ]
        );
    }
}
