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

# Rust Guide

```{note}
The Rust support is currently in an experimental stage.
```

This guide demonstrates how to use TVM FFI from Rust applications.

## Installation

### Prerequisites

The Rust support depends on `libtvm_ffi`. First, install the `tvm-ffi` Python package:

```bash
pip install -v -e .
```

Confirm that `tvm-ffi-config` is available:

```bash
tvm-ffi-config --libdir
```

### Adding to Your Project

Add to your `Cargo.toml`:

```toml
[dependencies]
tvm-ffi = { path = "path/to/tvm-ffi/rust/tvm-ffi" }
```

For published versions (when available):

```toml
[dependencies]
tvm-ffi = "0.1.0-alpha.0"
```

### Environment Setup

Set the library path so `libtvm_ffi` can be found at runtime:

```bash
export LD_LIBRARY_PATH=$(tvm-ffi-config --libdir):$LD_LIBRARY_PATH
```

## Basic Usage

### Loading a Module

Load a compiled TVM FFI module and call its functions:

```rust
use tvm_ffi::{Module, Result};

fn main() -> Result<()> {
    // Load compiled module
    let module = Module::load_from_file("build/add_one_cpu.so")?;

    // Get function by name
    let add_fn = module.get_function("add_one_cpu")?;

    Ok(())
}
```

### Working with Tensors

Create and manipulate tensors:

```rust
use tvm_ffi::Tensor;

// Create a tensor from a slice
let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
let tensor = Tensor::from_slice(&data, &[2, 3])?;
```

### Calling Functions

Call functions with tensors:

```rust
use tvm_ffi::{Module, Tensor, Result};

fn run_example() -> Result<()> {
    let module = Module::load_from_file("build/add_one_cpu.so")?;
    let func = module.get_function("add_one_cpu")?;

    // Create input and output tensors
    let input = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4])?;
    let output = Tensor::from_slice(&[0.0f32; 4], &[4])?;

    // Call function
    func.call_tuple((&input, &output))?;

    Ok(())
}
```

## Advanced Topics

### Global Functions

Register and access global functions:

```rust
use tvm_ffi::Function;

// Get global function
let func = Function::get_global("my_function")?;

// Register a new global function
let my_func = Function::from_packed(|args: &[AnyView]| -> Result<Any> {
    // Function implementation
    Ok(Any::default())
});
Function::register_global("my_custom_func", my_func)?;
```

### Type-Erased Functions

Create functions from Rust closures:

```rust
use tvm_ffi::{Function, Any, AnyView, Result};

// From packed closure
let func = Function::from_packed(|args: &[AnyView]| -> Result<Any> {
    // Process args and return result
    Ok(Any::default())
});

// From typed closure
let typed_func = Function::from_typed(|x: i64, y: i64| -> Result<i64> {
    Ok(x + y)
});
```

### Error Handling

TVM FFI uses standard Rust `Result` types:

```rust
use tvm_ffi::{Error, Module, Result, VALUE_ERROR};

fn may_fail(value: i32) -> Result<()> {
    // Operations that may fail
    let module = Module::load_from_file("path.so")?;

    // Custom errors
    if value < 0 {
        return Err(Error::new(
            VALUE_ERROR,
            "Value must be non-negative",
            ""
        ));
    }

    Ok(())
}
```

### Structural Walk and Visit

Rust provides equivalents of the C++ `StructuralWalk`/`StructuralVisitor`
APIs. Put `#[dispatch(visit)]` on an impl to turn its `visit_*` methods into
typed handlers, then pass it to `structural_walk`; each handler returns a
`WalkResult` (`Advance`, `Skip`, or `Interrupt`) to steer the traversal.
Handlers dispatch on their argument type and may take an optional trailing
`DefRegionKind` argument:

```rust
use tvm_ffi::{dispatch, structural_walk, Array, DefRegionKind, WalkOrder, WalkResult};

#[derive(Default)]
struct Probe {
    total: i64,
    floats: usize,
}

#[dispatch(visit)]
impl Probe {
    fn visit_integer(&mut self, value: i64) -> WalkResult {
        self.total += value;
        WalkResult::Advance
    }

    fn visit_float(&mut self, _value: f64, _kind: DefRegionKind) -> WalkResult {
        self.floats += 1;
        WalkResult::Advance
    }
}

let values = Array::new(vec![1_i64, 2, 3]);
let mut probe = Probe::default();
structural_walk(&values, &mut probe, WalkOrder::PreOrder)?;
assert_eq!(probe.total, 6);
```

A closure works as a catch-all walker, over `&VisitValue` alone or with the
definition-region state as a second argument:

```rust
use tvm_ffi::{structural_walk, Array, DefRegionKind, VisitValue, WalkOrder, WalkResult};

let values = Array::new(vec![1_i64, 2, 3]);
let mut integers = 0;
structural_walk(
    &values,
    |value: &VisitValue| {
        if value.cast::<i64>().is_some() {
            integers += 1;
        }
        WalkResult::Advance
    },
    WalkOrder::PreOrder,
)?;
assert_eq!(integers, 3);

let mut uses = 0;
structural_walk(
    &values,
    |value: &VisitValue, kind: DefRegionKind| {
        if value.cast::<i64>().is_some() && kind == DefRegionKind::None {
            uses += 1;
        }
        WalkResult::Advance
    },
    WalkOrder::PreOrder,
)?;
assert_eq!(uses, 3);
```

To drive recursion yourself, implement `StructuralVisitor` and call
`structural_visit`; `visit` runs for each value and descends through
`default_visit_children` (or `visit_child` for selected children):

```rust
use tvm_ffi::{
    structural_visit, Array, DefRegionKind, Result, StructuralVisitor, VisitInterrupt, VisitValue,
};

#[derive(Default)]
struct Depth {
    max: usize,
    current: usize,
}

impl StructuralVisitor for Depth {
    fn visit(
        &mut self,
        value: &VisitValue,
        def_region_kind: DefRegionKind,
    ) -> Result<Option<VisitInterrupt>> {
        self.current += 1;
        self.max = self.max.max(self.current);
        let interrupt = self.default_visit_children(value, def_region_kind)?;
        self.current -= 1;
        Ok(interrupt)
    }
}

let values = Array::new(vec![1_i64, 2]);
let mut depth = Depth::default();
structural_visit(&values, &mut depth)?;
assert_eq!(depth.max, 2);
```

## Examples

The repository includes a complete example in `rust/tvm-ffi/examples/load_library.rs`.

Run it with:

```bash
cd rust
cargo run --example load_library --features example
```

## Building the Workspace

Build the entire Rust workspace:

```bash
cd rust
cargo build
```

Run tests:

```bash
cargo test
```

## API Reference

For detailed API documentation, see the [Rust API Reference](../reference/rust/index.rst).

## Related Resources

- [Quick Start Guide](../get_started/quickstart.rst) - General TVM FFI introduction
- [C++ Guide](./cpp_lang_guide.md) - C++ API usage
- [Python Guide](./python_lang_guide.md) - Python API usage
