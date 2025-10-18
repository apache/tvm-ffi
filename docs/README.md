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
# TVM FFI Documentation

This guide walks through building and maintaining the TVM FFI documentation set.

## Prerequisites

- [`uv`](https://docs.astral.sh/uv/) manages the Python environment for all docs commands.
- Ensure you are in the repository root before running the commands below.
- Optional: Install `Doxygen` if you plan to generate the C++ API reference (see "Build with C++ Docs").

## Build Workflows

### Interactive build (auto-reload)

Rebuilds and serves the documentation locally with live reload:

```bash
uv run --group docs sphinx-autobuild docs docs/_build/html --ignore docs/reference/cpp/generated
```

By default, open `http://127.0.0.1:8000` in your browser after the initial build completes.

### One-off build

Generates the HTML documentation once, without running a server:

```bash
uv run --group docs sphinx-build -M html docs docs/_build
```

### Build with C++ Docs

Generating the C++ reference takes longer and requires Doxygen:

```bash
brew install doxygen        # macOS
sudo apt install doxygen    # Linux
```

Set `BUILD_CPP_DOCS=1` on the desired build command to enable the extra step:

```bash
# Interactive build with auto-rebuild on C++ header changes
BUILD_CPP_DOCS=1 uv run --group docs sphinx-autobuild docs docs/_build/html --ignore docs/reference/cpp/generated --watch include
# One-off build
BUILD_CPP_DOCS=1 uv run --group docs sphinx-build -M html docs docs/_build
```

The C++ documentation will automatically rebuild when header files change during `sphinx-autobuild` sessions.

### Build with Rust Docs

Generating the Rust reference requires `cargo` to be installed:

```bash
# Install Rust toolchain if not already installed
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Set `BUILD_RUST_DOCS=1` on the desired build command to enable Rust documentation:

```bash
# Interactive build with auto-rebuild on Rust source changes
BUILD_RUST_DOCS=1 uv run --group docs sphinx-autobuild docs docs/_build/html --ignore docs/reference/rust/generated --watch rust
# One-off build (clean rebuild)
BUILD_RUST_DOCS=1 uv run --group docs sphinx-build -M html docs docs/_build
```

The Rust documentation will automatically rebuild when source files change during `sphinx-autobuild` sessions.

### Build All Documentation

To build documentation with all language references enabled:

```bash
# Interactive build
BUILD_CPP_DOCS=1 BUILD_RUST_DOCS=1 uv run --group docs sphinx-autobuild docs docs/_build/html --ignore docs/reference/cpp/generated --ignore docs/reference/rust/generated --watch include --watch rust
# One-off build
BUILD_CPP_DOCS=1 BUILD_RUST_DOCS=1 uv run --group docs sphinx-build -M html docs docs/_build
```

## Cleanup

Remove generated artifacts when they are no longer needed:

```bash
rm -rf docs/_build/
rm -rf docs/reference/python/generated
rm -rf docs/reference/cpp/generated
rm -rf docs/reference/rust/generated
```
