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
"""Helper tool to check version consistency between pyproject.toml and __init__.py."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import tomli


def read_pyproject_version(pyproject_path: Path) -> str | None:
    """Read version from pyproject.toml."""
    with pyproject_path.open("rb") as f:
        data = tomli.load(f)

    return data.get("project", {}).get("version")


def read_init_version(init_path: Path) -> str | None:
    """Read __version__ from __init__.py."""
    with init_path.open(encoding="utf-8") as f:
        content = f.read()

    # Look for __version__ = "..." pattern
    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
    if match:
        return match.group(1)
    return None


def update_init_version(init_path: Path, new_version: str) -> bool:
    """Update __version__ in __init__.py."""
    with init_path.open(encoding="utf-8") as f:
        content = f.read()

    # Replace the version line
    new_content = re.sub(
        r'__version__\s*=\s*["\'][^"\']+["\']', f'__version__ = "{new_version}"', content
    )

    with init_path.open("w", encoding="utf-8") as f:
        f.write(new_content)

    return True


def main() -> int:
    """Execute the main function."""
    # Hardcoded paths
    pyproject_path = Path("pyproject.toml")
    init_path = Path("python/tvm_ffi/__init__.py")

    # Read versions
    pyproject_version = read_pyproject_version(pyproject_path)
    init_version = read_init_version(init_path)

    if pyproject_version is None or init_version is None:
        return 1

    if pyproject_version == init_version:
        print("Version check passed!")
        return 0
    else:
        print("Version check failed!")
        print(f"pyproject.toml version: {pyproject_version}")
        print(f"__init__.py version: {init_version}")
        print("Run precommit locally to fix the version.")
        update_init_version(init_path, pyproject_version)
        return 1


if __name__ == "__main__":
    sys.exit(main())
