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

This package preserves the `torch_c_dlpack_ext` import for consumers that
already depend on it. PyTorch 2.10 and newer provide the native DLPack exchange
API, so the package uses that API without loading or compiling a separate
library. The pure Python wheel also works on Python 3.14t and 3.15t when a
compatible PyTorch wheel is installed.

```bash
uv pip install torch-c-dlpack-ext
```

AOT libraries for older PyTorch ABIs are no longer bundled. TVM FFI's own
optional JIT/fallback path remains available for older PyTorch users.
