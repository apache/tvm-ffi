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
"""Build Torch C DLPack Addon."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import sysconfig
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.utils.cpp_extension
from lockfile import FileLock  # type: ignore[import-not-found]

IS_WINDOWS = sys.platform == "win32"


def find_dlpack_include_path() -> str:
    """Find dlpack header files for C compilation."""
    install_include_path = Path(__file__).resolve().parent.parent / "include"
    if (install_include_path / "dlpack").is_dir():
        return str(install_include_path)

    source_include_path = (
        Path(__file__).resolve().parent.parent / ".." / ".." / "3rdparty" / "dlpack" / "include"
    )
    if source_include_path.is_dir():
        return str(source_include_path)

    raise RuntimeError("Cannot find include path.")


def _run_command_in_dev_prompt(
    args: list[str],
    cwd: str | os.PathLike[str],
    capture_output: bool,
) -> subprocess.CompletedProcess:
    """Locates the Developer Command Prompt and runs a command within its environment."""
    try:
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

        if not Path(vsdevcmd_path).exists():
            raise FileNotFoundError(f"VsDevCmd.bat not found at: {vsdevcmd_path}")

        # Use cmd.exe to run the batch file and then your command.
        # The /k flag keeps the command prompt open after the batch file runs.
        # The "&" symbol chains the commands.
        cmd_command = '"{vsdevcmd_path}" -arch=x64 & {command}'.format(
            vsdevcmd_path=vsdevcmd_path, command=" ".join(args)
        )

        # Execute the command in a new shell
        return subprocess.run(
            cmd_command, check=False, cwd=cwd, capture_output=capture_output, shell=True
        )

    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        raise RuntimeError(
            "Failed to run the following command in MSVC developer environment: {}".format(
                " ".join(args)
            )
        ) from e


def _generate_ninja_build(
    build_dir: Path,
    libname: str,
    source_path: str,
    extra_cflags: Sequence[str],
    extra_ldflags: Sequence[str],
    extra_include_paths: Sequence[str],
) -> None:
    """Generate the content of build.ninja for building the module."""
    if IS_WINDOWS:
        default_cflags = [
            "/std:c++17",
            "/MD",
            "/wd4819",
            "/wd4251",
            "/wd4244",
            "/wd4267",
            "/wd4275",
            "/wd4018",
            "/wd4190",
            "/wd4624",
            "/wd4067",
            "/wd4068",
            "/EHsc",
        ]
        default_ldflags = ["/DLL"]
    else:
        default_cflags = ["-std=c++17", "-fPIC", "-O2"]
        default_ldflags = ["-shared"]

    cflags = default_cflags + [flag.strip() for flag in extra_cflags]
    ldflags = default_ldflags + [flag.strip() for flag in extra_ldflags]
    include_paths = [find_dlpack_include_path()] + [str(Path(path).resolve()) for path in extra_include_paths]

    # append include paths
    for path in include_paths:
        cflags.append("-I{}".format(path.replace(":", "$:")))

    # flags
    ninja = []
    ninja.append("ninja_required_version = 1.3")
    ninja.append("cxx = {}".format(os.environ.get("CXX", "cl" if IS_WINDOWS else "c++")))
    ninja.append("cflags = {}".format(" ".join(cflags)))
    ninja.append("ldflags = {}".format(" ".join(ldflags)))

    # rules
    ninja.append("")
    ninja.append("rule compile")
    if IS_WINDOWS:
        ninja.append("  command = $cxx /showIncludes $cflags -c $in /Fo$out")
        ninja.append("  deps = msvc")
    else:
        ninja.append("  depfile = $out.d")
        ninja.append("  deps = gcc")
        ninja.append("  command = $cxx -MMD -MF $out.d $cflags -c $in -o $out")
    ninja.append("")

    ninja.append("rule link")
    if IS_WINDOWS:
        ninja.append("  command = $cxx $in /link $ldflags /out:$out")
    else:
        ninja.append("  command = $cxx $in $ldflags -o $out")
    ninja.append("")

    # build targets
    ninja.append(
        "build main.o: compile {}".format(str(Path(source_path).resolve()).replace(":", "$:"))
    )

    # Use appropriate extension based on platform
    ninja.append(f"build {libname}: link main.o")
    ninja.append("")

    # default target
    ninja.append(f"default {libname}")
    ninja.append("")

    with open(build_dir / "build.ninja", "w") as f:  # noqa: PTH123
        f.write("\n".join(ninja))


def _build(build_dir: Path) -> None:
    # Run ninja build
    if IS_WINDOWS:
        result = _run_command_in_dev_prompt(
            args=["ninja", "-f", "build.ninja"],
            cwd=str(build_dir),
            capture_output=False,
        )
        if result.returncode != 0:
            raise RuntimeError("Ninja build failed on Windows.")
    else:
        result = subprocess.run(
            ["ninja", "-f", "build.ninja"],
            check=False,
            cwd=str(build_dir),
            capture_output=False,
        )
        if result.returncode != 0:
            raise RuntimeError("Ninja build failed on Unix-like system.")
    pass


parser = argparse.ArgumentParser()
parser.add_argument(
    "--build_dir",
    type=str,
    default=str(Path("~/.tvm_ffi/torch_c_dlpack_addon").expanduser()),
    help="Directory to store the built extension library.",
)
parser.add_argument(
    '--build_with_cuda',
    action='store_true',
    help="Build with CUDA support."
)


def main() -> None:
    """Build the torch c dlpack extension."""
    args = parser.parse_args()
    build_dir = Path(args.build_dir)

    if not build_dir.exists():
        build_dir.mkdir(parents=True, exist_ok=True)

    name = "libtorch_c_dlpack_addon"
    libname = name + (".dll" if IS_WINDOWS else ".so")
    tmp_libname = libname + ".tmp"

    with FileLock(str(build_dir / "build.lock")):
        if (build_dir / libname).exists():
            # already built
            return

        # resolve configs
        include_paths = []
        ldflags = []
        cflags = []
        include_paths.append(sysconfig.get_paths()["include"])

        if args.build_with_cuda:
            cflags.append('-DBUILD_WITH_CUDA')
            include_paths.extend(torch.utils.cpp_extension.include_paths('cuda'))
        else:
            include_paths.extend(torch.utils.cpp_extension.include_paths('cpu'))

        for lib_dir in torch.utils.cpp_extension.library_paths():
            ldflags.append(f"-L{lib_dir}")
        ldflags.append("-ltorch_python")

        # generate ninja build file
        _generate_ninja_build(
            build_dir=build_dir,
            libname=tmp_libname,
            source_path=str(Path(__file__).parent / "addon.cc"),
            extra_cflags=cflags,
            extra_ldflags=ldflags,
            extra_include_paths=include_paths,
        )

        # build the shared library
        _build(build_dir=build_dir)

        # rename the tmp file to final libname
        shutil.move(str(build_dir / tmp_libname), str(build_dir / libname))


if __name__ == "__main__":
    main()
