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
# Torch C DLPack Extension

This folder contains the source for the `torch-c-dlpack-ext` package, which provides an
Ahead-Of-Time (AOT) compiled module to support faster DLPack conversion in DLPack v1.2.
By default, `tvm-ffi` will JIT-compile a version of this functionality during loading,
and use a safe-fallback if JIT-compilation fails.
Installing this wheel allows users to avoid this JIT compilation overhead and
also avoid the cases where the user environment does not necessarily have a compiler toolchain
to run JIT-compilation.

```bash
pip install torch-c-dlpack-ext
```
