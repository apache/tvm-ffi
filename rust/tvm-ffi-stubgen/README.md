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
- [Related User Guide](#related-user-guide)

## Document Scope

This README intentionally does not duplicate full command-line tutorial content.
For command usage and end-to-end calling examples, see:

- `docs/packaging/rust_stubgen.md`

## Generated Interface Forms

Stubgen emits a public facade (`src/lib.rs`) plus detail modules:

- `src/_tvm_ffi_stubgen_detail/functions.rs`
- `src/_tvm_ffi_stubgen_detail/types.rs`

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

For layout-compatible object types:

- `#[repr(C)] <Type>Obj`
- `#[derive(ObjectRef, Clone)] <Type>`
- `impl_object_hierarchy!(...)`
- direct-field `get_` accessors

Example shape:

```rust
#[repr(C)]
pub struct TestObjectDerivedObj {
    parent: TestObjectBaseObj,
    v_map: tvm_ffi::Map<tvm_ffi::AnyValue, tvm_ffi::AnyValue>,
    v_array: tvm_ffi::Array<tvm_ffi::AnyValue>,
}
```

#### fallback wrapper path

For non-repr(C)-compatible types:

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
    parent: BaseObj,
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

`check_repr_c` gates repr(C) generation.

### Required metadata checks

- `total_size > 0`
- valid field `offset/size/alignment`
- field order and no overlap
- aligned placement (`field.offset == align_up(pos, alignment)`)
- parent boundary matches first direct field offset
- parent type is also repr(C)-compatible

### Schema mapping rules

Representative mappings include:

- `Any` / `ffi.Any` -> `tvm_ffi::AnyValue`
- `ffi.Array<T>` -> `tvm_ffi::Array<T>`
- `ffi.Map<K,V>` -> `tvm_ffi::Map<K, V>`

## Safety and Fallback Strategy

Generated user-facing code is intended to remain safe Rust.

### Safety boundary

- unsafe operations are encapsulated in `tvm-ffi` internals and macros
- generated wrappers and getters are safe APIs

### Built-in filtering and fallback

- built-in `ffi.*` primitives are not re-generated as wrapper types
- unsupported/non-layout-compatible object types fall back to `define_object_wrapper!`

## Related User Guide

For generation command-line usage and step-by-step invocation examples, see:

- `docs/packaging/rust_stubgen.md`
