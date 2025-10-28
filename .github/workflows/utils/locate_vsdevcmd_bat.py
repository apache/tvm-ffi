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

"""Locate the VsDevCmd.bat file for the current Visual Studio installation."""

import os
import subprocess
from pathlib import Path


def main() -> None:
    """Locate the VsDevCmd.bat file for the current Visual Studio installation.

    Raise exception if not found. If found, print the path to stdout.
    """
    # Path to vswhere.exe
    vswhere_path = str(
        Path(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)"))
        / "Microsoft Visual Studio"
        / "Installer"
        / "vswhere.exe"
    )

    if not Path(vswhere_path).exists():
        raise FileNotFoundError("vswhere.exe not found.")

    # Find the Visual Studio installation path
    vs_install_path = subprocess.run(
        [
            vswhere_path,
            "-latest",
            "-prerelease",
            "-products",
            "*",
            "-property",
            "installationPath",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    if not vs_install_path:
        raise FileNotFoundError("No Visual Studio installation found.")

    # Construct the path to the VsDevCmd.bat file
    vsdevcmd_path = str(Path(vs_install_path) / "Common7" / "Tools" / "VsDevCmd.bat")
    print(vsdevcmd_path)


if __name__ == "__main__":
    main()
