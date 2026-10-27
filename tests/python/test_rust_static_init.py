# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""The static init blocks of a Rust library run when Python loads it.

``rust/tvm-ffi/examples/static_init.rs`` is built by cargo into a library of its own, which a fresh
interpreter loads: its block registers a global function, as a C++ library does when it loads.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "rust" / "Cargo.toml"


def find_cargo() -> str | None:
    """Find a cargo to build the crate with, rustup's first."""
    candidates = [str(Path.home() / ".cargo" / "bin" / "cargo"), shutil.which("cargo")]
    for cargo in candidates:
        if cargo and os.access(cargo, os.X_OK):
            return cargo
    return None


@pytest.fixture(scope="module")
def static_init_library() -> Path:
    """Build the example library with a static init block with cargo."""
    cargo = find_cargo()
    if cargo is None:
        raise pytest.skip.Exception("building rust/ needs cargo")
    if not shutil.which("tvm-ffi-config") or not MANIFEST.exists():
        raise pytest.skip.Exception(
            "building rust/ needs tvm-ffi-config on PATH and the source tree"
        )
    subprocess.run(
        [
            cargo,
            "build",
            "--manifest-path",
            str(MANIFEST),
            "-p",
            "tvm-ffi",
            "--example",
            "static_init",
        ],
        check=True,
    )
    metadata = subprocess.run(
        [cargo, "metadata", "--manifest-path", str(MANIFEST), "--no-deps", "--format-version", "1"],
        capture_output=True,
        text=True,
        check=True,
    )
    target = Path(json.loads(metadata.stdout)["target_directory"])
    if sys.platform == "win32":
        prefix, suffix = "", ".dll"
    elif sys.platform == "darwin":
        prefix, suffix = "lib", ".dylib"
    else:
        prefix, suffix = "lib", ".so"
    library = target / "debug" / "examples" / f"{prefix}static_init{suffix}"
    assert library.exists(), library
    return library


def run_python(code: str, *libraries: Path) -> str:
    """Run `code` in a fresh interpreter, with the paths of `libraries` as its arguments, and
    return its standard error.
    """
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code), *map(str, libraries)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stderr


def test_a_library_runs_its_blocks_when_python_loads_it(
    static_init_library: Path, tmp_path: Path
) -> None:
    copy = tmp_path / static_init_library.name
    shutil.copy(static_init_library, copy)
    stderr = run_python(
        """
        import ctypes
        import gc
        import sys

        import tvm_ffi

        library, copy = sys.argv[1:]

        # Loading the library registers its function, as loading a C++ library does. The library
        # stays loaded when Python drops its module, as the registration keeps its code.
        module = tvm_ffi.load_module(library)
        del module
        gc.collect()
        add_one = tvm_ffi.get_global_func("static_init_example.add_one")
        assert add_one(1) == 2

        # The copy has the block of the library, as two libraries that link the same crate do:
        # loading it registers nothing again, where registering the name again would fail.
        ctypes.CDLL(copy)
        assert add_one(2) == 3
        """,
        static_init_library,
        copy,
    )
    assert "static init block" not in stderr, stderr


def test_a_block_that_fails_when_its_library_loads_is_reported(
    static_init_library: Path,
) -> None:
    stderr = run_python(
        """
        import sys

        import tvm_ffi

        # Another library registered the name of the example function: the block fails when the
        # library loads, and the error is printed, as it cannot reach a caller then.
        tvm_ffi.register_global_func("static_init_example.add_one", lambda x: x)
        module = tvm_ffi.load_module(sys.argv[1])

        # `run_static_init_blocks` raises the error again.
        try:
            module.run_static_init_blocks()
        except Exception as error:
            assert "`static_init_example.add_one` is already registered" in str(error), error
        else:
            raise AssertionError("the block that failed was not reported")
        """,
        static_init_library,
    )
    assert "the static init block at static_init:" in stderr, stderr
    assert "is already registered" in stderr, stderr
