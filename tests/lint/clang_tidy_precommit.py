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
from __future__ import annotations

import os
import subprocess
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DEFAULT_SOURCE_EXTS = (".c", ".cc", ".cpp", ".cxx", ".m", ".mm")


def parse_args(argv: Sequence[str]) -> Namespace:
    p = ArgumentParser(
        prog="clang-tidy-precommit",
        description=(
            "Run clang-tidy on changed files using a compile_commands.json "
            "generated in a dedicated build directory."
        ),
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--build-dir",
        "-B",
        default="build-pre-commit",
        help="CMake build directory containing compile_commands.json.",
    )
    p.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="Maximum parallel clang-tidy processes.",
    )
    p.add_argument(
        "--fix",
        action="store_true",
        help="Enable clang-tidy in-place fixes (-fix).",
    )
    p.add_argument(
        "files",
        nargs="*",
        help="Files passed by pre-commit (will be filtered by extension).",
    )
    return p.parse_args(list(argv))


def filter_files(files: Sequence[str]) -> list[str]:
    kept: list[str] = []
    for f in files:
        if not f:
            continue
        p = Path(f)
        if p.is_dir():
            for child in p.rglob("*"):
                if child.is_file() and child.suffix.lower() in DEFAULT_SOURCE_EXTS:
                    kept.append(str(child))
            continue
        if p.suffix.lower() in DEFAULT_SOURCE_EXTS:
            kept.append(str(p))
    return kept


def ensure_compile_commands(build_dir: Path) -> Path:
    cc_path = build_dir / "compile_commands.json"
    if cc_path.exists():
        return cc_path
    build_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "cmake",
        "-S",
        ".",
        "-B",
        str(build_dir),
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DTVM_FFI_BUILD_TESTS=ON",
    ]
    ret = subprocess.run(cmd, check=False)
    if ret.returncode != 0:
        print(
            "[clang-tidy-precommit] Failed to generate compile_commands.json via CMake.",
            file=sys.stderr,
        )
        sys.exit(ret.returncode)
    if not cc_path.exists():
        print(
            f"[clang-tidy-precommit] Missing {cc_path}. Ensure CMake config succeeded.",
            file=sys.stderr,
        )
        sys.exit(2)
    return cc_path


def build_base_cmd(p_arg: Path, fix: bool) -> list[str]:
    cmd: list[str] = ["clang-tidy", f"-p={p_arg!s}", "-quiet"]
    if fix:
        cmd.append("-fix")
    return cmd


def run_parallel(cmd: list[str], files: list[str], jobs: int) -> int:
    def one(f: str) -> tuple[int, str, list[str]]:
        full_cmd = [*cmd, f]
        print(f"[RUNNING] {' '.join(full_cmd)}", file=sys.stderr)
        proc = subprocess.run(full_cmd, check=False, capture_output=True, text=True)
        out = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode, out, " ".join(full_cmd)

    jobs = max(1, int(jobs or 1))
    rc = 0
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = [ex.submit(one, f) for f in files]
        for fut in as_completed(futs):
            code, output, full_cmd = fut.result()
            output = output.strip()
            if code != 0:
                print(f"[FAILED] {full_cmd}\n{output}")
                rc = 1
    return rc


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    files = filter_files(args.files)
    if not files:
        print("[clang-tidy-precommit] No relevant files to lint.")
        return 0

    build_dir = Path(args.build_dir).resolve()
    cc_path = ensure_compile_commands(build_dir)
    base_cmd = build_base_cmd(p_arg=cc_path.parent, fix=args.fix)
    if sys.platform == "darwin":
        base_cmd = ["xcrun", *base_cmd]
    rc = run_parallel(base_cmd, files, args.jobs)
    if rc != 0 and args.fix:
        print(
            "[clang-tidy-precommit] clang-tidy reported issues and applied fixes. "
            "Re-stage your changes if files were modified.",
            file=sys.stderr,
        )
    return rc


if __name__ == "__main__":
    sys.exit(main())
