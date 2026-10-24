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

## Installation

Cargo builds `libtvm_ffi` from source and dynamically links it. No Python package
is required. Install CMake 3.26+ and a C/C++17 toolchain; Unix builds also need
`sh` and `make` for libbacktrace. Windows builds use MSVC.

In a source checkout, initialize the submodules first:

```bash
git submodule update --init --recursive
cd rust
cargo build
cargo test --workspace
```

Keep the complete repository layout when using these crates as path dependencies.
Cargo makes the shared runtime available to `cargo run` and `cargo test`.

When running a binary outside Cargo, deploy the shared runtime and configure
the platform's loader path or an application rpath.

## Running Examples

The optional library-loading example is similar to [examples/quickstart](../examples/quickstart/)
and uses the Python `tvm_ffi` package to build its module. Install the Python
package and activate its environment first
(see [the source build guide](../docs/dev/source_build.rst)).

```bash
cargo run --example load_library --features example
```

Check out the [load_library.rs](tvm-ffi/examples/load_library.rs) for details.
