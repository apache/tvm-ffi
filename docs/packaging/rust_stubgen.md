<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->
<!--- -->
<!---   http://www.apache.org/licenses/LICENSE-2.0 -->
<!--- -->
<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# Rust Stubgen Guide

```{note}
The Rust stub generation flow is currently experimental and may evolve.
```

This guide covers practical usage of `tvm-ffi-stubgen`: generation command, output crate, and how to call generated APIs.

## Generate a Stub Crate

Run from `3rdparty/tvm/3rdparty/tvm-ffi/rust`:

```bash
cargo run -p tvm-ffi-stubgen -- <OUT_DIR> \
  --init-prefix testing \
  --init-crate tvm-ffi-testing \
  --dlls /abs/path/to/libtvm_ffi_testing.so \
  --overwrite
```

### Arguments

- `OUT_DIR`: positional output directory
- `--dlls`: one or more dynamic libraries for reflection metadata
- `--init-prefix`: registry prefix filter (functions/types to include)
- `--init-crate`: generated crate name
- `--tvm-ffi-path`: optional local path override for `tvm-ffi`
- `--overwrite`: overwrite non-empty output directory

## Generated Output Layout

The output is a standalone Rust crate:

- `Cargo.toml`
- `src/lib.rs`
- `src/_tvm_ffi_stubgen_detail/functions.rs`
- `src/_tvm_ffi_stubgen_detail/types.rs`

`src/lib.rs` re-exports generated wrappers and provides:

```rust
pub fn load_library(path: &str) -> tvm_ffi::Result<tvm_ffi::Module>
```

## Using Generated Crate

Using the generated stubs is straightforward—simply load the runtime library, call exported functions, and work with generated object wrappers and subtyping as needed. The full process is shown in the following example, covering typical usage:

```rust
use tvm_ffi_testing as stub;

fn main() -> tvm_ffi::Result<()> {
    // Load FFI library (required before any calls)
    stub::load_library("/abs/path/to/libtvm_ffi_testing.so")?;

    // Call a generated function with typed arguments
    let y = stub::add_one(1)?;
    assert_eq!(y, 2);

    // Call a function via packed interface for dynamic signature
    let _out = stub::echo(&[tvm_ffi::Any::from(1_i64)])?;

    // Use object-returning wrappers and ObjectRef-based APIs
    let obj = stub::make_unregistered_object()?;
    let count = stub::object_use_count(obj.clone())?;
    assert!(count >= 1);

    // Fallback wrapper can be built from ObjectRef directly
    let _wrapped: stub::TestUnregisteredObject = obj.into();

    Ok(())
}
```

- Load the library once before using the APIs.
- Generated functions support typed signatures when possible and fall back to `Any` for dynamic calling.
- Generated object-returning wrappers integrate with `ObjectRef` APIs and wrapper conversions.


## Related Docs

- Rust language guide: `guides/rust_lang_guide.md`
- Rust stubgen design details (implementation-oriented): `rust/tvm-ffi-stubgen/README.md`
