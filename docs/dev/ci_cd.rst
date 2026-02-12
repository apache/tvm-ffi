..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

Reproduce CI/CD
===============

This guide explains how to reproduce CI checks and tests locally,
and how wheel builds and releases work.
All CI/CD workflows are defined under `.github/workflows/ <https://github.com/apache/tvm-ffi/tree/main/.github/workflows>`__.
For building the project from source, see :doc:`source_build`.

Linters
-------

Pre-commit
~~~~~~~~~~

The project uses `pre-commit <https://pre-commit.com/>`__ to run linters and
formatters. All hooks are defined in `.pre-commit-config.yaml <https://github.com/apache/tvm-ffi/blob/main/.pre-commit-config.yaml>`__. Install and
register the git hooks so they run automatically before each commit:

.. code-block:: bash

   uv tool install pre-commit
   pre-commit install

You can also run hooks manually:

.. code-block:: bash

   # Run all hooks on every file
   pre-commit run --all-files

   # Run only on staged files
   pre-commit run

   # Run a single hook in isolation
   pre-commit run ruff-check --all-files
   pre-commit run clang-format --all-files

The main linters per language are:

- **Python** -- ``ruff`` (lint + format), ``ty`` (type checking)
- **C/C++** -- ``clang-format`` (format), ``clang-tidy`` (lint, see below)
- **Cython** -- ``cython-lint``
- **CMake** -- ``cmake-format``, ``cmake-lint``
- **Shell** -- ``shfmt``, ``shellcheck``

If you run into issues with pre-commit:

- **Version problems** -- ensure you have pre-commit 2.18.0 or later
  (``pre-commit --version``).
- **Stale cache** -- run ``pre-commit clean`` to clear the hook cache.
- **Auto-fixed files** -- most formatting hooks fix issues in place. Review the
  changes, stage them with ``git add -u``, and commit again.

clang-tidy
~~~~~~~~~~

``clang-tidy`` is run as a separate CI job (not as a pre-commit hook) and only
checks C++ files that have changed. To reproduce it locally:

.. code-block:: bash

   # Run clang-tidy on specific files
   uv run --no-project --with "clang-tidy==21.1.1" \
     python tests/lint/clang_tidy_precommit.py \
       --build-dir=build-pre-commit \
       --jobs=$(nproc) \
       include/tvm/ffi/c_api.h src/some_file.cc

   # Or run on all C++ sources
   uv run --no-project --with "clang-tidy==21.1.1" \
     python tests/lint/clang_tidy_precommit.py \
       --build-dir=build-pre-commit \
       --jobs=$(nproc) \
       ./src/ ./include ./tests

.. note::

   On macOS, ``clang-tidy`` is resolved through ``xcrun``. The wrapper
   ``tests/lint/clang_tidy_precommit.py`` handles this automatically.


C++ Tests
---------

Build and run locally. First, set ``CMAKE_BUILD_PARALLEL_LEVEL`` to speed up
the build:

.. code-block:: bash

   export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)   # Linux
   export CMAKE_BUILD_PARALLEL_LEVEL=$(sysctl -n hw.ncpu)  # macOS

Then configure, build, and run:

.. code-block:: bash

   # Configure with tests enabled
   cmake . -B build_test -DTVM_FFI_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Debug

   # Build the test target
   cmake --build build_test --clean-first --config Debug --target tvm_ffi_tests

   # Run tests
   ctest -V -C Debug --test-dir build_test --output-on-failure

.. note::

   On Windows, make sure you run the build from a
   **Developer Command Prompt for VS** or have the MSVC toolchain on your
   ``PATH``.


Python Tests
------------

Reproduce locally with:

.. code-block:: bash

   # Install the project in editable mode with test dependencies
   uv pip install --reinstall --verbose --group test -e .

   # Run the full test suite
   uv run pytest -vvs tests/python


Rust Tests
----------

Rust tests live in the ``rust/`` workspace. Run them with:

.. code-block:: bash

   cd rust && cargo test

This tests all workspace members (``tvm-ffi``, ``tvm-ffi-sys``,
``tvm-ffi-macros``).

.. note::

   CI runs Rust tests only after the Python package is installed (
   ``uv pip install --group test -e .``), because the Rust FFI bindings link
   against the built shared library. Make sure the Python package is
   installed before running ``cargo test``.


Build Python Wheels
-------------------

CI builds wheels using `cibuildwheel <https://cibuildwheel.pypa.io/>`__
on Linux (x86_64, aarch64), Windows (AMD64), and macOS (arm64).
The wheel configuration lives in the ``[tool.cibuildwheel]`` section of
``pyproject.toml``.

To build a wheel locally:

.. code-block:: bash

   uv tool install cibuildwheel
   cibuildwheel --output-dir dist

You can restrict the build to a single platform:

.. code-block:: bash

   # Build only for the current platform
   cibuildwheel --only cp312-macosx_arm64

Use environment variables to control the target platform:

.. code-block:: bash

   # Choose manylinux image (e.g. manylinux2014, manylinux_2_28)
   CIBW_MANYLINUX_X86_64_IMAGE=manylinux_2_28 cibuildwheel --output-dir dist

   # Set macOS deployment target
   CIBW_ENVIRONMENT_MACOS="MACOSX_DEPLOYMENT_TARGET=10.14" cibuildwheel --output-dir dist

   # Build only specific Python versions
   CIBW_BUILD="cp312-*" cibuildwheel --output-dir dist

.. seealso::

   - :doc:`../packaging/python_packaging`: Packaging shared libraries as
     Python wheels with scikit-build-core.
   - :doc:`release_process`: Publishing wheels and creating release artifacts.
