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
into Rust bindings. This example registers one object, `rust_stubgen.IntPair`
(`src/int_pair.cc`), and lets CMake regenerate `rust/src/generated/` after
every build.

Every object gets a `#[repr(C)]` wrapper, a reference type, `Deref`, and the
upcasts along its ancestor chain. `IntPair` is plain data, so its reflected
fields account for every byte and the binding is *complete*: the struct mirrors
the fields at their real offsets and widths, a `const` assertion pins its size
and alignment to the reflected facts, and a generated `new` allocates the object
in Rust. `main.rs` builds one that way, reads `pair.a` directly, and hands it to
a C++ function that reads it back.

An object whose layout cannot be reproduced (a polymorphic one, say, with a
vtable in front of the object header) is bound *opaquely* instead: the struct
embeds only the parent, every field is read through an accessor that calls the
C ABI getter, and construction stays on the C++ side.

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

Four more directives are available: `field` names the Rust type of a field
(`// tvm-ffi-stubgen(field): rust_stubgen.IntPair.a -> MyInt`), `nullable`
wraps it in `Option` (`// tvm-ffi-stubgen(nullable): rust_stubgen.IntPair.a`),
`upcast` adds a conversion to a hand-written typed view
(`// tvm-ffi-stubgen(upcast): rust_stubgen.IntPair -> MyView`), and
`custom-new` names the generated allocator `from_complete_fields` when `new`
is hand-written (`// tvm-ffi-stubgen(custom-new): rust_stubgen.IntPair`).

## Partial generation

Every type an object refers to (its parent, its ancestors, the types of its
fields) has to be provided in the same `tvm-ffi-stubgen` run: `ffi.*` types
come from the `tvm-ffi` crate, an `object/<key>` block in any processed file
generates it, and a `ty-map` points at a hand-written binding whose object
struct is named `<Name>Obj`
(`// tvm-ffi-stubgen(ty-map): rust_stubgen.IntPair -> crate::hand::IntPair`).
Anything else is an error listing the missing keys, so a partial binding never
references a module that does not exist.
