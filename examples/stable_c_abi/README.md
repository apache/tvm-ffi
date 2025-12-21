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

# Stable C ABI Code Example

This directory contains all the source code for [tutorial](https://tvm.apache.org/ffi/get_started/stable_c_abi.html).

## Run everything

On Linux/macOS:

```bash
bash run_all.sh
```

On Windows:

```batch
run_all.bat
```

## Compile and Distribute `add_one_cpu` manually

To compile the C Example:

```bash
cmake . -B build -DEXAMPLE_NAME="kernel" -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --config RelWithDebInfo
```

This produces `build/add_one_cpu.so`.

## Load the Distributed `add_one_cpu` manually

To run library loading example in C:

```bash
cmake . -B build -DEXAMPLE_NAME="load" -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --config RelWithDebInfo
build/load
```

The executable is emitted as `build/load` (`build/load.exe` on Windows).
