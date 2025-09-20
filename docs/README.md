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

## Build Locally with uv

All documentation dependencies are managed via [`uv`](https://docs.astral.sh/uv/).
Run the following commands from the repository root:


```bash
uv run --extra docs make -C docs livehtml
```

For a one-off build you can run `uv run --extra docs make -C docs html` instead.

`uv run` executes the `make` targets inside the synced environment, so manual
virtualenv activation is unnecessary.

## Build with C++ Docs

To include the generated C++ API documentation, install Doxygen and set the
`BUILD_CPP_DOCS=1` environment variable when invoking the build target:

```bash
BUILD_CPP_DOCS=1 uv run --extra docs make -C docs livehtml
```

Generating the C++ docs can take noticeably longer, so it remains opt-in.
