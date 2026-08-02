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

# Rust Packages

(Experimental) Rust support for the `tvm-ffi` ABI.
Currently, the rust support is in an experimental stage.
This workspace contains three crates:

- `tvm-ffi`: Safe, ergonomic Rust bindings over the ABI.
- `tvm-ffi-sys`: Low-level exposure of raw C ABIs.
- `tvm-ffi-macros`: Procedural macros used by `tvm-ffi` (derive/object helpers and exported function helpers).

The overall project focuses on low-level, direct access to the ABI when possible for maximum
efficiency while maintaining interoperability.

## Structural Visitors and Walkers

The `tvm-ffi` crate provides native Rust structural traversal over FFI values,
built-in containers, and reflected object fields, mirroring the two layers of
the C++ API:

- **Walk layer (observer)** — `structural_walk`, the analog of C++
  `StructuralWalk`: the walker owns recursion; handlers observe each value in
  pre- or post-order (`WalkOrder`) and steer traversal through the returned
  `WalkResult` (`Advance`, `Skip`, `Interrupt`/`InterruptWith`).
- **Visitor layer (user-driven)** — `structural_visit` with a
  `StructuralVisitor`, the analog of a C++ `StructuralVisitorObj`: your
  `visit` method runs for the root and controls all recursion itself.

Every traversal returns `Result<Option<VisitInterrupt>>`: `Ok(None)` means
the whole graph was visited, `Ok(Some(interrupt))` carries an interrupting
handler's payload. Handlers may also return `Result<WalkResult>` to propagate
errors with `?`.

### Observer walks

`#[dispatch(visit)]` turns the `visit_*` methods of an inherent impl into a
typed, stateful observer. Each value dispatches to the first handler matching
its runtime type (a `&VisitValue` handler acts as the catch-all); a handler
that needs the definition-region state declares a trailing `DefRegionKind`
argument:

```rust
use tvm_ffi::{dispatch, structural_walk, Array, WalkOrder, WalkResult};

#[derive(Default)]
struct Calculator {
    value: f64,
}

#[dispatch(visit)]
impl Calculator {
    fn visit_integer(&mut self, value: i64) -> WalkResult {
        self.value += value as f64;
        WalkResult::Advance
    }

    fn visit_float(&mut self, value: f64) -> WalkResult {
        self.value -= value;
        WalkResult::Advance
    }
}

let values = Array::new(vec![10_i64, 2]);
let mut calculator = Calculator::default();
assert!(structural_walk(&values, &mut calculator, WalkOrder::PreOrder)
    .unwrap()
    .is_none());
assert_eq!(calculator.value, 12.0);
```

`structural_walk` also accepts a bare closure taking `&VisitValue` (with an
optional trailing `DefRegionKind`) as a catch-all observer.

### User-driven visitors

When traversal itself is part of the analysis — visiting selected children,
custom orders, definition-region overrides — implement `StructuralVisitor`.
`visit` receives each value with its definition-region state and descends
only where it chooses: `default_visit_children` delegates the default child
recursion, and `visit_child` visits one child under an explicit state:

```rust,ignore
use tvm_ffi::{DefRegionKind, Result, StructuralVisitor, VisitInterrupt, VisitValue};

struct FuncVisitor;

impl StructuralVisitor for FuncVisitor {
    fn visit(
        &mut self,
        value: &VisitValue,
        def_region_kind: DefRegionKind,
    ) -> Result<Option<VisitInterrupt>> {
        if let Some(func) = value.as_node::<FuncObj>() {
            // Parameters bind recursively; the body inherits the state.
            if let Some(interrupt) = self.visit_child(&func.params, DefRegionKind::Recursive)? {
                return Ok(Some(interrupt));
            }
            return self.visit_child(&func.body, def_region_kind);
        }
        self.default_visit_children(value, def_region_kind)
    }
}
```

Recursion runs natively in Rust; no C++ visitor is constructed. Mutable
`List`/`Dict` contents are snapshotted before callbacks run, and a
non-container type with a foreign `__s_visit__` hook is rejected rather than
silently walked through reflection.

## Installation

The Rust support depends on `libtvm_ffi`.
Please install the `tvm-ffi` pip package by running:

```bash
pip install -v ..
```

Confirm that `tvm-ffi-config` is available with:

```bash
tvm-ffi-config --libdir
```

Then build the workspace with:

```bash
cargo build
```

The build will:

- Query `tvm-ffi-config --libdir` to add the appropriate link search path.
- Link against `tvm_ffi`.
- Update the appropriate dynamic loader path environment variable for `cargo run` and `cargo test`.

For running downstream applications, you need to set the `LD_LIBRARY_PATH` so `libtvm_ffi` is available in the path.

```bash
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:`tvm-ffi-config --libdir`
```

## Running Examples

You can run an optional library-loading example similar to the quick_start examples in [examples/quick_start](../examples/quick_start/).

```bash
cargo run --example load_library --features example
```

Check out the [load_library.rs](tvm-ffi/examples/load_library.rs) for details.
