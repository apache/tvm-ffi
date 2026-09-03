<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# Rust Stub Generation

`tvm-ffi-stubgen --target rust` turns the reflection metadata of a C++ library
into Rust bindings. This example registers two objects in `src/int_pair.cc`
and lets CMake regenerate `rust/src/generated/` after every build.

Every object gets a `#[repr(C)]` wrapper, a reference type, `Deref`, and the
upcasts along its ancestor chain. What the wrapper holds depends on whether the
reflected fields account for every byte of the object:

- `rust_stubgen.IntRange` does, so its binding is *complete*: the struct mirrors
  the fields at their real offsets and widths, and Rust reads `range.begin`
  directly. A `const` assertion pins the struct's size and alignment to the
  reflected facts. The object is allocated in Rust, by a generated function
  that takes every field by value; a C++ function then reads it back.
- `rust_stubgen.IntPair` has a vtable in front of the object header, so its
  binding is *opaque*: the struct embeds only the parent and every field is read
  through an accessor that calls the C ABI getter. It is constructed by a
  registered global function.

A builtin parent such as `ffi.IntEnum` has no `<Leaf>Obj` in the crate; the
import section defines a header-only stand-in per builtin ancestor so the
derived type depth matches the registry.

## Build and run

```bash
# 1. Build the C++ library; the post-build step runs the stub generator.
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# 2. Build and run the Rust program against it.
cd rust && cargo run
```

The Rust crate depends on the `tvm-ffi` crate of this repository and needs
`tvm-ffi-config` on `PATH` (activate the virtual environment where the
`apache-tvm-ffi` package is installed).

## Directives

The generated file keeps one-line directives the generator reads on every run.
`rust/src/generated/rust_stubgen/mod.rs` declares the integer field `kind` as an
open newtype:

```rust
// tvm-ffi-stubgen(enum): rust_stubgen.IntPair.kind -> PairKind(i32) { Unordered=0, Ordered=1 }
```

It also marks `IntRange` as having a hand-written `new` (in `main.rs`, where it
validates the extent), so the generator emits none of its own:

```rust
// tvm-ffi-stubgen(custom-new): rust_stubgen.IntRange
```

Without the marker the generator emits `IntRange::new` itself, and a
hand-written one is a duplicate definition. Three more directives are
available: `field` names the Rust type of a field
(`// tvm-ffi-stubgen(field): rust_stubgen.IntPair.a -> MyInt`), `nullable`
wraps it in `Option` (`// tvm-ffi-stubgen(nullable): rust_stubgen.IntPair.a`),
and `upcast` adds a conversion to a hand-written typed view
(`// tvm-ffi-stubgen(upcast): rust_stubgen.IntRange -> MyView`).
