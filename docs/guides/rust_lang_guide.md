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

### Reflected Type Methods

Libraries that register their API through the C++ reflection registry
(`refl::ObjectDef<T>().def(...)`) store methods in a per-type method table
rather than the global function table. Resolve them by type key (or type
index) and method name; constructors registered via `refl::init` are
reachable under the reserved name `__ffi_init__`:

```rust
use tvm_ffi::{AnyView, Function};

// Resolve the reflected constructor and construct an instance
let ctor = Function::from_type_key_method("testing.TestIntPair", "__ffi_init__")?;
let pair = ctor.call_tuple((1i64, 2i64))?;

// Resolve an instance method; the first packed argument is the object itself
let sum = Function::from_type_key_method("testing.TestIntPair", "sum")?;
let result = sum.call_packed(&[AnyView::from(&pair)])?;
assert_eq!(i64::try_from(result)?, 3);
```

`Function::from_type_method(type_index, name)` performs the same lookup when
the type index is already known (e.g. from `Any::type_index`).

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

Lambdas also work — pass a single typed lambda, or a tuple of them (up to 8)
tried in order with the first matching argument type winning, like the
variadic C++ `StructuralWalk(root, callbacks...)` chain. Unmatched values
simply advance; a `&VisitValue` lambda acts as a catch-all and must come
last, since links after an always-matching one never run. Each lambda may
take a trailing `DefRegionKind` argument:

```rust
use tvm_ffi::{structural_walk, Array, DefRegionKind, Object, WalkOrder, WalkResult};

let values = Array::new(vec![1_i64, 2, 3]);

let mut total = 0;
structural_walk(
    &values,
    |value: i64| {
        total += value;
        WalkResult::Advance
    },
    WalkOrder::PreOrder,
)?;
assert_eq!(total, 6);

let mut evens = 0;
let mut objects = 0;
structural_walk(
    &values,
    (
        |value: i64| {
            if value % 2 == 0 {
                evens += 1;
            }
            WalkResult::Advance
        },
        |_object: &Object, _kind: DefRegionKind| {
            objects += 1;
            WalkResult::Advance
        },
    ),
    WalkOrder::PreOrder,
)?;
assert_eq!((evens, objects), (1, 1));
```

Both entry points return `Result<Option<VisitInterrupt>>`: `Ok(None)` means
the whole graph was visited, and a handler stops the walk early by returning
`WalkResult::interrupt_with(payload)`, which comes back to the caller as
`Ok(Some(interrupt))`. Handlers may also return `Result<WalkResult>` and
propagate errors with `?`:

```rust
use tvm_ffi::{structural_walk, Array, WalkOrder, WalkResult};

let values = Array::new(vec![1_i64, 2, 3]);
let found = structural_walk(
    &values,
    |value: i64| {
        if value == 2 {
            return WalkResult::interrupt_with(value);
        }
        WalkResult::Advance
    },
    WalkOrder::PreOrder,
)?;
assert_eq!(found.map(|i| i64::try_from(i.value).unwrap()), Some(2));
```

To drive recursion yourself, implement `StructuralVisitor` and call
`structural_visit`; `visit` runs for each value and descends through
`default_visit_children`, or through `visit_child`, which visits one
selected child and can override the def-region state for it (e.g.
`DefRegionKind::Recursive` when descending into a binder's parameters):

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

Two safety notes: mutable `List`/`Dict` contents are snapshotted before
callbacks run, so mutation during traversal cannot invalidate the walk; and
a non-container type with a foreign `__s_visit__` hook is rejected rather
than silently walked through reflection — visit such a type's children
explicitly from a `StructuralVisitor`, or skip it with a pre-order
`WalkResult::Skip`.

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
