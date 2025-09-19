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

# tvm ffi

[![CI](https://github.com/apache/tvm-ffi/actions/workflows/ci_test.yml/badge.svg)](https://github.com/apache/tvm-ffi/actions/workflows/ci_test.yml)

## Development with Docker

The repository ships a development container that contains the full toolchain for
building the core library, and running examples.

```bash
# Build the image (from the repository root)
docker build -t tvm-ffi-dev .

# Start an interactive shell
docker run --rm -it \
    -v "$(pwd)":/workspace/tvm-ffi \
    -w /workspace/tvm-ffi \
    tvm-ffi-dev bash

# Start an interactive shell with GPU access
docker run --rm -it --gpus all \
    -v "$(pwd)":/workspace/tvm-ffi \
    -w /workspace/tvm-ffi \
    tvm-ffi-dev bash

> **Note** Ensure the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
> is installed on the host to make GPUs available inside the container.
```

Inside the container you can install the project in editable mode and run the quick
start example exactly as described in `examples/quick_start/README.md`:

```bash
pip install -ve .

cd examples/quick_start
bash run_example.sh
```

All build artifacts are written to the mounted workspace on the host machine, so you
can continue editing files with your local tooling.
