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
"""Version consistency linter across Python, C++, and Rust.

This script checks that:
  1) C++ version macros in headers are internally consistent and match the
     canonical project version (major/minor/micro) derived from setuptools_scm.
  2) Rust crate versions are internally consistent and compatible with
     the canonical project version.

Usage: python tests/lint/check_version.py [--cpp] [--rust]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import setuptools_scm
import tomli
from packaging.version import Version as packaging_version

RE_MAJOR = re.compile(r"^\s*#\s*define\s+TVM_FFI_VERSION_MAJOR\s+(\d+)\b")
RE_MINOR = re.compile(r"^\s*#\s*define\s+TVM_FFI_VERSION_MINOR\s+(\d+)\b")
RE_PATCH = re.compile(r"^\s*#\s*define\s+TVM_FFI_VERSION_PATCH\s+(\d+)\b")


def _version_info() -> dict[str, Any]:
    """Return project version information using setuptools_scm."""
    version = setuptools_scm.get_version()
    v = packaging_version(version)

    return {
        "full": version,
        "major": v.major,
        "minor": v.minor,
        "micro": v.micro,
        "pre": v.pre,
        "dev": v.dev,
        "local": v.local,
        "post": v.post,
        "release": v.release,
        "is_prerelease": v.is_prerelease,
        "is_postrelease": v.is_postrelease,
        "is_devrelease": v.is_devrelease,
        "base_version": v.base_version,
        "public": v.public,
    }


def _map_pep440_pre_to_semver(pre: tuple[str, int] | None) -> str | None:
    if pre is None:
        return None
    tag, num = pre
    tag = tag.lower()
    if tag in {"a", "alpha"}:
        tag = "alpha"
    elif tag in {"b", "beta"}:
        tag = "beta"
    elif tag in {"rc", "c"}:
        tag = "rc"
    else:
        return None
    return f"{tag}.{num}"


def _check_cpp(version_info: dict) -> list[str]:
    errors: list[str] = []
    c_api_path = Path("include") / "tvm" / "ffi" / "c_api.h"
    if not c_api_path.exists():
        errors.append(f"[C++] Missing expected header file: {c_api_path}")
        return errors

    def _scan_cpp_macros() -> tuple[int, int, int]:
        file = c_api_path
        major = minor = patch = None
        for line in file.read_text(encoding="utf-8").splitlines():
            if m := RE_MAJOR.match(line):
                major = int(m.group(1))
            if m := RE_MINOR.match(line):
                minor = int(m.group(1))
            if m := RE_PATCH.match(line):
                patch = int(m.group(1))
        return major, minor, patch

    (major, minor, patch) = _scan_cpp_macros()

    if major is None or minor is None or patch is None:
        errors.append(f"[C++] {c_api_path}: No version macros found: {major=}, {minor=}, {patch=}.")
        return errors

    exp_major, exp_minor, exp_patch = (
        version_info["major"],
        version_info["minor"],
        version_info["micro"],
    )
    if (major, minor, patch) != (exp_major, exp_minor, exp_patch):
        errors.append(
            f"[C++] {c_api_path}: Macro version mismatch: found {major}.{minor}.{patch}, "
            f"expected {exp_major}.{exp_minor}.{exp_patch}."
        )
    return errors


def _check_rust(version_info: dict) -> list[str]:
    errors: list[str] = []
    rust_dir = Path("rust")
    found_versions: dict[Path, str] = {}
    for path in [
        rust_dir / "tvm-ffi" / "Cargo.toml",
        rust_dir / "tvm-ffi-macros" / "Cargo.toml",
        rust_dir / "tvm-ffi-sys" / "Cargo.toml",
    ]:
        found_versions[path] = tomli.loads(path.read_text(encoding="utf-8"))["package"]["version"]

    if not found_versions:
        # No crates found, skip silently
        return errors

    # 1) All crates must agree on a single version
    unique_versions = set(found_versions.values())
    if len(unique_versions) > 1:
        errors.append(
            "[Rust] Crates have inconsistent versions: "
            + ", ".join(f"{p} -> {v}" for p, v in sorted(found_versions.items()))
        )

    # 2) Optionally enforce compatibility with Python version
    base = version_info["base_version"]
    allowed: set[str] = {base}
    pre = _map_pep440_pre_to_semver(version_info.get("pre"))
    if pre:
        allowed.add(f"{base}-{pre}")
    allowed = sorted(allowed)
    for path, v in found_versions.items():
        if v not in allowed:
            errors.append(
                f"[Rust] {path}: version not compatible with project version. Allowed: {allowed}; got: {v}."
            )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Check version consistency across languages")
    parser.add_argument("--cpp", action="store_true", help="Check C++ version macros")
    parser.add_argument("--rust", action="store_true", help="Check Rust crate versions")
    args = parser.parse_args()
    info = _version_info()
    print(
        f"Project version: {info['full']}\n"
        f"  major: {info['major']}\n"
        f"  minor: {info['minor']}\n"
        f"  micro: {info['micro']}\n"
        f"  pre: {info['pre']}\n"
        f"  dev: {info['dev']}\n"
        f"  local: {info['local']}\n"
        f"  post: {info['post']}\n"
        f"  base_version: {info['base_version']}\n"
        f"  release: {info['release']}\n"
        f"  public: {info['public']}\n"
        f"  is_prerelease: {info['is_prerelease']}\n"
        f"  is_postrelease: {info['is_postrelease']}\n"
        f"  is_devrelease: {info['is_devrelease']}"
    )
    errors: list[str] = []
    if args.cpp:
        errors += _check_cpp(info)
    if args.rust:
        errors += _check_rust(info)
    if errors:
        print("\n".join(errors))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
