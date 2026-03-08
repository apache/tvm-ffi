# Rust Stubgen Guide

`tvm-ffi-stubgen` generates Rust stubs from TVM-FFI reflection metadata.
This document is design-oriented and focuses on generated interface forms and implementation choices.

## Table of Contents

- [Document Scope](#document-scope)
- [Generated Interface Forms](#generated-interface-forms)
- [Object Model and Inheritance](#object-model-and-inheritance)
- [Field Accessor Style](#field-accessor-style)
- [Subtyping and Cast Rules](#subtyping-and-cast-rules)
- [repr(C) Decision Rules](#reprc-decision-rules)
- [Safety and Fallback Strategy](#safety-and-fallback-strategy)
- [TODO](#todo)
- [Related User Guide](#related-user-guide)

## Document Scope

This README intentionally does not duplicate full command-line tutorial content.
For command usage and end-to-end calling examples, see:

- `docs/packaging/rust_stubgen.md`

## Generated Interface Forms

Stubgen emits a public facade (`src/lib.rs`) plus detail modules:

- `src/_tvm_ffi_stubgen_detail/functions.rs`
- `src/_tvm_ffi_stubgen_detail/types.rs`

By default the generator runs `cargo fmt` on the emitted crate after writing these files.
Pass `--no-format` to keep the raw generated text when debugging formatting-sensitive output.

### Function Wrappers

#### Typed wrapper path

When type schema is fully known, function wrappers are generated as typed Rust APIs:

```rust
pub fn add_one(_0: i64) -> Result<i64> { ... }
```

#### Packed fallback path

When schema is not fully resolved, wrappers use packed calling style:

```rust
pub fn echo(args: &[Any]) -> Result<Any> { ... }
```

### Type Wrappers

#### repr(C) path (preferred)

For types with known `total_size`:

- `#[repr(C)] <Type>Obj` with typed fields and `[u8; N]` gaps
- `#[derive(ObjectRef, Clone)] <Type>`
- `impl_object_hierarchy!(...)`
- direct-field `get_` accessors

Example shape:

```rust
#[repr(C)]
pub struct PrimExprObj {
    __tvm_ffi_object_parent: BaseExprObj,
    dtype: tvm_ffi::DLDataType,
    _gap0: [u8; 4],  // C++ tail padding
}
```

Gaps cover C++ tail padding, vtable pointers, and fields whose type schema
is not mappable to Rust.  This allows the vast majority of types to use
repr(C) layout even when metadata is incomplete.

#### fallback wrapper path

For types without `total_size` metadata (no `ObjectDef` registered):

- `define_object_wrapper!(Type, "type.key")`
- field access via `FieldGetter<T>`

### Object Method Lookup Path

Object methods (including `__ffi_init__`) are generated from type reflection metadata,
not from global function registry names:

- generated code calls `tvm_ffi::object_wrapper::resolve_type_method(type_key, method_name)`
- runtime lookup path is `TVMFFITypeKeyToIndex -> TVMFFIGetTypeInfo -> methods[]`
- the `method` entry is converted from `AnyView` to owned `Any`, then to `ffi.Function`

Global wrappers under `functions.rs` still use `Function::get_global`, but type methods in
`types.rs` no longer assume `<type>.<method>` is globally registered.

For constructor-like methods (`__ffi_init__`), stubgen emits `new(...)` directly as the public
Rust API (or `ffi_init` only when a user-defined `new` method already exists).

## Object Model and Inheritance

repr(C) object inheritance is modeled by composition and deref chain:

### Obj-level layout

Derived object stores parent object as first field:

```rust
#[repr(C)]
pub struct DerivedObj {
    __tvm_ffi_object_parent: BaseObj,
    extra: i64,
}
```

### Ref-level inheritance

Ref wrappers use `impl_object_hierarchy!` to establish:

- `Deref<Derived -> Base>`
- `From<Derived> for Base/ObjectRef` (upcast)
- `TryFrom<Base/ObjectRef> for Derived` (downcast)

## Field Accessor Style

Getter generation follows a single style:

- name prefix is always `get_`
- only direct fields of current type generate getters
- inherited getters are available via deref auto-coercion

### Return type rules

- POD field -> return by value
- object/container field -> clone and return user-facing type

Example:

```rust
impl TestObjectDerived {
    pub fn get_v_map(&self) -> tvm_ffi::Map<tvm_ffi::AnyValue, tvm_ffi::AnyValue> {
        self.data.v_map.clone()
    }
}
```

## Subtyping and Cast Rules

Stubgen-generated repr(C) refs use standard Rust traits as the only user-facing cast API:

- borrow upcast: `Deref`
- consuming upcast: `From` / `.into()`
- consuming downcast: `TryFrom` / `.try_into()`

This avoids custom cast traits and keeps compile-time type constraints explicit.

## repr(C) Decision Rules

`check_repr_c` gates repr(C) generation using a **gap-filling** strategy.

### Hard requirements (cause fallback to `define_object_wrapper!`)

- Type must have `total_size > 0` (i.e. `ObjectDef` was called for it)
- No overlapping fields

### Soft handling (does NOT cause fallback)

- **Tail padding / vtable / unregistered fields**: byte ranges between registered
  fields (or between the last field and `total_size`) are emitted as `[u8; N]` gap
  members in the `#[repr(C)]` struct.
- **Parent type not in type_map or not repr(C)-compatible**: the parent region is
  treated as a gap after the `Object` header. The struct uses `tvm_ffi::object::Object`
  as the parent field and gap-fills the bytes between Object and the first known field.
- **Field type schema not mappable to Rust**: the field is skipped in the struct layout
  (covered by a gap) but still accessible via runtime `FieldGetter` if needed.

### Schema mapping rules

Representative mappings include:

- `Any` / `ffi.Any` -> `tvm_ffi::AnyValue`
- `ffi.Array<T>` -> `tvm_ffi::Array<T>`
- `ffi.Array` (no args) -> `tvm_ffi::Array<tvm_ffi::object::ObjectRef>`
- `ffi.Map<K,V>` -> `tvm_ffi::Map<K, V>`
- `Optional<T>` -> `Option<T>`
- `Optional` (no args) -> `Option<tvm_ffi::object::ObjectRef>`

## Multi-Prefix Generation

`--init-prefix` accepts multiple values. Behavior depends on the count:

- **Single prefix** (e.g. `--init-prefix testing`): the prefix is stripped and items land
  at the crate root. This is the default backward-compatible mode.
- **Multiple prefixes** (e.g. `--init-prefix tl --init-prefix ir --init-prefix tir`):
  no prefix is stripped; each prefix naturally becomes a top-level module.

Example with multiple prefixes:

```
tl.KernelLaunch     → crate::tl::KernelLaunch
ir.Span             → crate::ir::Span
tir.BufferLoad      → crate::tir::BufferLoad
script.ir_builder.* → crate::script::ir_builder::*
```

This allows a single generated crate to cover multiple namespaces. Cross-namespace
`repr(C)` inheritance (e.g. `tir.PrimFunc` extending `ir.BaseFunc`) resolves within
the crate without workarounds.

## Safety and Fallback Strategy

Generated user-facing code is intended to remain safe Rust.

### Safety boundary

- unsafe operations are encapsulated in `tvm-ffi` internals and macros
- generated wrappers and getters are safe APIs

### Built-in filtering and fallback

- built-in `ffi.*` primitives are not re-generated as wrapper types
- only types without `total_size` metadata fall back to `define_object_wrapper!`
- types with incomplete field schemas or unmappable parents still get repr(C) layout
  via gap-filling

### Logging

Stubgen uses the `log` crate. Set `RUST_LOG` to control verbosity:

- `RUST_LOG=debug` — shows repr(C) pass/fail decisions and field mapping failures
- `RUST_LOG=trace` — additionally shows per-field offset/size/schema details

## TODO

Known gaps and design issues that remain open.

### Ancestor chain is truncated when direct parent is not repr(C)-mappable

When `check_repr_c` cannot map the direct parent type, `repr_c.rs` falls back to
`tvm_ffi::object::Object` as the layout parent and fills the missing bytes with a gap.
However, the second pass in `generate.rs` that builds `ancestor_chain` only propagates
through types whose `parent_type_key` is set in `ReprCInfo`; when it is `None` the chain
collapses to `[tvm_ffi::object::ObjectRef]`.

Consequence: if the C++ hierarchy is `Object → A (mappable) → B (not mappable) → C`,
the generated code for `C` emits

```rust
tvm_ffi::impl_object_hierarchy!(C: tvm_ffi::object::ObjectRef);
```

instead of

```rust
tvm_ffi::impl_object_hierarchy!(C: A, tvm_ffi::object::ObjectRef);
```

This means `From<C> for A` and `TryFrom<A> for C` are not generated, and getters
inherited from `A` are inaccessible via deref on `C` even though the layout is correct.

The ancestor chain logic should be derived from the runtime type ancestry table
(`TVMFFIGetTypeInfo → type_acenstors`) independently of layout mappability, so that
upcast/downcast correctness is preserved regardless of whether every intermediate type
has a usable repr(C) layout.

### Common interface between fallback and repr(C) paths

`define_object_wrapper!` types and repr(C) types currently expose different API surfaces:

- repr(C) types: `Deref` chain, `From`/`TryFrom`, direct `get_*` accessors
- fallback types: `from_object` / `as_object_ref` / `into_object_ref`, runtime `FieldGetter`

Code that depends on a given type must know which generation path was used, and that
path can change between stubgen versions as reflection metadata improves.  A type that
was a thin wrapper in version N may become a repr(C) type in version N+1, silently
breaking downstream call sites that relied on `from_object` or `as_object_ref`.

A stable, version-independent interface layer is needed so that user code does not need
to distinguish between the two paths, and so that crates built against one stubgen
version remain source-compatible with crates built against a later one.

## Related User Guide

For generation command-line usage and step-by-step invocation examples, see:

- `docs/packaging/rust_stubgen.md`
