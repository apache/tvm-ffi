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

Every object is bound *opaquely*: Rust gets a `#[repr(C)]` wrapper that embeds
only the parent, a reference type, `Deref`, the upcasts along the ancestor
chain, and one accessor per reflected field that reads through the C ABI
getter. The object's bytes are never reproduced, so the binding is correct for
any registered type; construction goes through the registered global functions.

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

Two more are available: `field` names the Rust type of a field's accessor
(`// tvm-ffi-stubgen(field): rust_stubgen.IntPair.a -> MyInt`) and `nullable`
wraps it in `Option` (`// tvm-ffi-stubgen(nullable): rust_stubgen.IntPair.a`).
