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

# TVM FFI: Open ABI and FFI for Machine Learning Systems

📚 [Documentation](https://tvm.apache.org/ffi/) | 🚀 [Quickstart](https://tvm.apache.org/ffi/get_started/quickstart.html)

Apache TVM FFI is an open ABI and FFI for machine learning systems. It is a minimal, framework-agnostic,
yet flexible open convention with the following systems in mind:

- **Kernel libraries** - ship one wheel to support multiple frameworks, Python versions, and different languages.
- **Kernel DSLs** - reusable open ABI for JIT and AOT kernel exposure frameworks and runtimes.
- **Frameworks and runtimes** - a uniform extension point for ABI-compliant libraries and DSLs.
- **ML infrastructure** - out-of-the-box bindings and interop for Python, C++, and Rust.
- **Coding agents** - a unified mechanism for shipping generated code in production.

## Features

- **Stable, minimal C ABI** designed for kernels, DSLs, and runtime extensibility.
- **Zero-copy interop** across PyTorch, JAX, and CuPy using [DLPack protocol](https://data-apis.org/array-api/2024.12/design_topics/data_interchange.html).
- **Compact value and call convention** covering common data types for ultra low-overhead ML applications.
- **Multi-language support** out of the box: Python, C++, and Rust (with a path towards more languages).

These enable broad **interoperability** across frameworks, libraries, DSLs, and agents; the ability to **ship one wheel** for multiple frameworks and Python versions (including free-threaded Python); and consistent infrastructure across environments.

## Status and Release Versioning

**C ABI stability** is our top priority.

**Status: RFC** Main features are complete and ABI stable. We recognize potential needs for evolution to ensure
it works best for the machine learning systems community, and would like to work together with the
community for such evolution. We plan to stay in the RFC stage for three months from the v0.1.0 release.

Releases during the RFC stage will be `0.X.Y`, where bumps in `X` indicate C ABI-breaking changes
and `Y` indicates other changes. We anticipate the RFC stage will last for three months, then we will start following
[Semantic Versioning](https://packaging.python.org/en/latest/discussions/versioning/)
(`major.minor.patch`) going forward.
