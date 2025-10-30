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

# Contributing to TVM FFI

We welcome contributions of all kinds, including bug fixes, documentation improvements, enhancements,
and more. To ensure a smooth process, Here is a general guide to contributing.

## Installation

For development, you can install through editable installation:

```bash
git clone https://github.com/apache/tvm-ffi --recursive
cd tvm-ffi
pip install --no-build-isolation -e . -v
```

We recommend using the `--no-build-isolation` flag to ensure compatibility with your existing environment.

## Setting Up Pre-commit Hooks

This project uses [pre-commit](https://pre-commit.com/) to maintain code quality and consistency.
Pre-commit hooks automatically check your code for common issues before you commit changes.

### Installing Pre-commit

First, install pre-commit (requires version 2.18.0 or later):

```bash
pip install pre-commit
```

### Installing the Git Hooks

After cloning the repository, install the pre-commit hooks:

```bash
cd tvm-ffi
pre-commit install
```

This configures git to automatically run the hooks before each commit.

### Running Hooks Manually

You can run the hooks manually on all files:

```bash
pre-commit run --all-files
```

Or run them only on staged files:

```bash
pre-commit run
```

### What the Hooks Check

The pre-commit configuration includes checks for:

- **License headers**: Ensures all files have proper Apache Software Foundation headers
- **Code formatting**: Runs clang-format (C++), ruff (Python), shfmt (Shell scripts)
- **Linting**: Runs clang-tidy, ruff, shellcheck, markdownlint, yamllint, and more
- **Type checking**: Runs mypy for Python type annotations
- **File quality**: Checks for trailing whitespace, file sizes, merge conflicts, etc.

### Troubleshooting

If you encounter errors:

1. **Version issues**: Ensure you have pre-commit 2.18.0 or later:

   ```bash
   pre-commit --version
   pip install --upgrade pre-commit
   ```

2. **Cache issues**: Clean the pre-commit cache:

   ```bash
   pre-commit clean
   ```

3. **Hook failures**: Most formatting hooks will automatically fix issues. Review the changes and stage them:

   ```bash
   git add -u
   git commit
   ```

## Contributing Workflow

You can contribute to the repo through the following steps.

- Fork the repository and create a new branch for your work.
- Push your changes to your fork and open a pull request to the main repository.
  - Please provide a clear description of your changes and link to the relevant issue if one exists.
  - Create necessary test cases and documentation.
- Work with the community by incorporating feedback from reviewers until the change is ready to be merged.

For significant changes, it's often a good idea to open a GitHub issue first (with `[RFC] title`) to discuss your proposal.
It is optional, but can be very helpful as it allows the maintainers and the community to provide feedback and helps ensure your
work aligns with the project's goals.

## Development with Docker

The repository ships a development container that contains the full toolchain for
building the core library, and running examples.

```bash
# Build the image (from the repository root)
docker build -t tvm-ffi-dev -f tests/docker/Dockerfile tests/docker

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
# In /workspace/tvm-ffi/ see https://tvm.apache.org/ffi/guides/build_from_source.html for reference
pip install --force-reinstall --verbose -e . \
  --config-settings cmake.define.TVM_FFI_ATTACH_DEBUG_SYMBOLS=ON

# Change working directory to sample
cd examples/quickstart

# Install dependency, Build and run all examples
bash raw_compile.sh
```

All build artifacts are written to the mounted workspace on the host machine, so you
can continue editing files with your local tooling.

## Stability and Minimalism

C ABI stability is the top priority of this project. We also prioritize minimalism and efficiency
in the core so it is portable and can be used broadly.
We also recognize potential needs for evolution to ensure it works best for the machine
learning systems community, and would like to work together collectively with the community for such evolution.

- When proposing a change, consider first whether things can be built on top of the current mechanism.
- We welcome RFCs and DISCUSS discussions for ABI discussions and rationales.
- We will communicate very mindfully to the community about major changes.

Feature improvements in the Python layer and C++ headers (that do not impact the ABI) have
comparatively less significant impacts on downstream users, so they can follow normal contribution
process in general open source projects.

## Status and Release Versioning

The project is in the RFC stage, which means the main features are complete.
We anticipate the API will be reasonably stable but will continue to work with the community to evolve it.
More importantly, the RFC stage is a period where we are working with the open source communities
to ensure we evolve the ABI to meet the needs of frameworks.

Releases during the RFC stage will be `0.X.Y`, where bumps in `X` indicate C ABI-breaking changes
and `Y` indicates other changes. We may also produce pre-releases (beta releases) during this period.

We anticipate the RFC stage will last for a few months, then we will start to follow
[Semantic Versioning](https://packaging.python.org/en/latest/discussions/versioning/)
(`major.minor.patch`) going forward.
