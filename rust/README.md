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

The `tvm-ffi` crate provides a native Rust structural walker over FFI values,
built-in containers, and reflected object fields. `#[dispatch(visit)]` turns
the `visit_*` methods in an inherent implementation into a typed, stateful
visitor. Each value is automatically dispatched to the handler matching its
runtime type:

```rust
use tvm_ffi::{dispatch, structural_visit, Function, WalkResult};

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

let values = Function::get_global("ffi.Array")
    .unwrap()
    .call_tuple((10_i64, 2.5_f64))
    .unwrap();
let mut calculator = Calculator::default();
assert!(structural_visit(&values, &mut calculator)
    .unwrap()
    .is_continue());
assert_eq!(calculator.value, 7.5);
```

Typed handlers are tested in source order: borrowed `ObjectCore` node types use
runtime subtype checks, owned arguments use `AnyCompatible` casts, and a final
`&VisitValue` handler acts as a catch-all. Use `structural_walk` to select
pre-order or post-order dispatch, or `walk`/`walk_with_context` for raw
callbacks.

`WalkResult::Advance` visits reflected or container children, `Skip` suppresses
the current value's default recursion, and `Interrupt` stops the complete
traversal.

A handler may declare an optional third `DefRegionKind` parameter to observe
the definition-region state at the current value (maintained automatically
from reflected field flags):

```rust,ignore
fn visit_var(&mut self, var: &VarObj, kind: DefRegionKind) -> WalkResult {
    // kind tells a definition position apart from a use position.
    WalkResult::Advance
}
```

To take over a value's children, a pre-order handler visits them through
`VisitDispatch::subvisit` and returns `Skip`. `subvisit` reports a nested
interrupt as `ControlFlow::Break`; propagate it (and errors, via `?`) instead
of dropping the result:

```rust,ignore
fn visit_func(&mut self, func: &FuncObj, kind: DefRegionKind) -> Result<WalkResult> {
    if let ControlFlow::Break(payload) = self.subvisit(&func.params, DefRegionKind::Recursive)? {
        return Ok(WalkResult::InterruptWith(payload));
    }
    if let ControlFlow::Break(payload) = self.subvisit(&func.body, kind)? {
        return Ok(WalkResult::InterruptWith(payload));
    }
    Ok(WalkResult::Skip)
}
```

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
