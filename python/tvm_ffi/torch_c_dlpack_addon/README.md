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

# The torch_c_dlpack addon

This is a standalone python module that is used to build a shared library as a patch to make old version of pytorch to support the dlpack protocol.

Two kinds of usage:

1. used as a standalone python module to build the shared library for each combination of `<python-version>` and `<pytorch-version>` and build a package to contain the built libraries.
2. used as a jit module in tvm-ffi when there is not a prebuilt patch found.

The module is designed to be only dependent on pytorch.

This module should be included in the tvm-ffi distribution to support JIT use-case.

## AOT Packaging

```bash
python python/tvm_ffi/torch_c_dlpack_addon/build.py --build_dir <build-dir>
```

Then there will be a shared library under the given `<build-dir>` (either `libtorch_c_dlpack_addon.so` or `libtorch_c_dlpack_addon.dll`, depends on the platform). The built shared library is specific to the current
python version and pytorch version.

## JIT Usage

We launched a python interpreter to run the `build.py` script. It will store the built shared library in tvm-ffi cache. See `python/tvm-ffi/_optional_torch_c_dlpack.py`.
