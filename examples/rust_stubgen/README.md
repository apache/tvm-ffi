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

# TVM FFI Rust Stubgen Example

This is an example project that registers a C++ object (`IntPair`) and
generates typed Rust bindings for it with `tvm-ffi-stubgen --target rust`,
as documented in [docs/packaging/stubgen.rst](../../docs/packaging/stubgen.rst).

## Build the library and generate the bindings

Install tvm-ffi and activate the virtualenv first (from the repo root):

```bash
uv pip install -e .
source .venv/bin/activate
```

Then build the C++ shared library:

```bash
cd examples/rust_stubgen
cmake -B build
cmake --build build
```

Stub generation runs as a post-build step
(`tvm_ffi_configure_target(... STUB_TARGET rust STUB_INIT ON)` in
`CMakeLists.txt`) and refreshes `rust/src/generated/`. The equivalent CLI
invocation is:

```bash
tvm-ffi-stubgen rust/src/generated --target rust --dlls build/librust_stubgen.so \
  --init-lib rust_stubgen --init-pypkg rust_stubgen --init-prefix "rust_stubgen."
```

## Run the example

After building the C++ library, run the Rust demo:

```bash
cd rust
cargo run
```

This runs four flows: constructing an `IntPair` via the generated builder
(`ffi_new().a(1).b(2).build()`; the defaulted `scale` field may be omitted),
calling the `sum` method, overriding the default through the `.scale(..)`
setter, and writing a field through `DerefMut`.
