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
"""Build test object files for all available compiler variants.

Uses ``tvm_ffi.cpp.build`` to compile C/C++ test sources to relocatable
object files.  Detects platform and available compilers:

- Linux:   LLVM Clang (default) + GCC
- macOS:   LLVM Clang (default) + Apple Clang
- Windows: LLVM Clang (default) + MSVC + clang-cl

Objects are placed under a temporary directory and skipped if they already
exist.  LLVM clang is found via ``$LLVM_PREFIX/bin`` or ``$PATH``; no
hardcoded install path is assumed.
"""

from __future__ import annotations

import os
import platform
import shutil
import tempfile
from pathlib import Path

import tvm_ffi.cpp

SOURCES_DIR = Path(__file__).resolve().parent / "sources"
SOURCES_C = SOURCES_DIR / "c"
SOURCES_CC = SOURCES_DIR / "cc"
SOURCES_CUDA = SOURCES_DIR / "cuda"

_DEFAULT_OUT_DIR = Path(tempfile.gettempdir()) / "tvm_ffi_orcjit_tests"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extra_cflags() -> list[str]:
    machine = platform.machine()
    if machine in ("aarch64", "arm64"):
        return ["-mno-outline-atomics"]
    return []


def _extra_cuda_cflags() -> list[str]:
    machine = platform.machine()
    if machine in ("aarch64", "arm64"):
        return ["-Xcompiler", "-mno-outline-atomics"]
    return []


def _build_objects(
    src_dir: Path,
    out_dir: Path,
    *,
    ext_glob: str,
    extra_cflags: list[str],
    extra_cuda_cflags: list[str] | None = None,
) -> None:
    """Compile all sources in *src_dir* to object files in *out_dir*."""
    out_dir.mkdir(parents=True, exist_ok=True)

    for src in sorted(src_dir.glob(ext_glob)):
        dest = out_dir / f"{src.stem}.o"
        if dest.exists():
            continue
        build_dir = out_dir / f".build_{src.stem}"
        obj_path = tvm_ffi.cpp.build(
            name=src.stem,
            sources=[str(src)],
            output=f"{src.stem}.o",
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags or [],
            build_directory=str(build_dir),
        )
        shutil.copy2(obj_path, dest)


def _build_variant(
    name: str,
    *,
    cc: str | None,
    cxx: str | None,
    extra_cflags: list[str],
    c_outdir: Path,
    cc_outdir: Path | None,
) -> None:
    """Build all test objects for one compiler variant."""
    saved_cc = os.environ.get("CC")
    saved_cxx = os.environ.get("CXX")
    try:
        if cc:
            os.environ["CC"] = cc
        if cxx:
            os.environ["CXX"] = cxx

        if cc:
            _build_objects(SOURCES_C, c_outdir, ext_glob="*.c", extra_cflags=extra_cflags)
        if cxx and cc_outdir:
            _build_objects(SOURCES_CC, cc_outdir, ext_glob="*.cc", extra_cflags=extra_cflags)
    finally:
        if saved_cc is None:
            os.environ.pop("CC", None)
        else:
            os.environ["CC"] = saved_cc
        if saved_cxx is None:
            os.environ.pop("CXX", None)
        else:
            os.environ["CXX"] = saved_cxx


# ---------------------------------------------------------------------------
# Compiler detection
# ---------------------------------------------------------------------------


def _find_llvm_clang() -> tuple[str, str] | None:
    """Find LLVM clang/clang++ via LLVM_PREFIX or PATH.

    Returns (clang, clang++) paths, or None if not found.
    """
    llvm_prefix = os.environ.get("LLVM_PREFIX")
    if llvm_prefix:
        p = Path(llvm_prefix)
        bin_dir = p / "Library" / "bin" if (p / "Library" / "bin").exists() else p / "bin"
        cc = bin_dir / ("clang.exe" if platform.system() == "Windows" else "clang")
        if cc.exists():
            cxx = bin_dir / ("clang++.exe" if platform.system() == "Windows" else "clang++")
            return str(cc), str(cxx)

    cc = shutil.which("clang")
    if cc:
        cxx = shutil.which("clang++")
        return cc, cxx or cc

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_test_objects(out_dir: Path | None = None) -> Path:
    """Build test objects for all available compiler variants.

    Objects are placed under *out_dir* (default:
    ``{tempdir}/tvm_ffi_orcjit_tests``).  Already-built objects are
    skipped.  Returns the output directory.
    """
    if out_dir is None:
        out_dir = _DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    system = platform.system()
    extra = _extra_cflags()

    if system in ("Linux", "Darwin"):
        llvm = _find_llvm_clang()
        if llvm:
            clang, clangxx = llvm
            _build_variant(
                "LLVM Clang",
                cc=clang,
                cxx=clangxx,
                extra_cflags=extra,
                c_outdir=out_dir / "c",
                cc_outdir=out_dir / "cc",
            )
        if system == "Linux" and shutil.which("gcc"):
            _build_variant(
                "GCC",
                cc="gcc",
                cxx="g++",
                extra_cflags=extra,
                c_outdir=out_dir / "c-gcc",
                cc_outdir=out_dir / "cc-gcc",
            )
            # PIE variant: -fpie forces R_X86_64_PC32 for hidden-visibility
            # externals like __dso_handle (instead of GOTPCRELX with -fPIC).
            # Used to reproduce __dso_handle Delta32 overflow on x86_64.
            _build_variant(
                "GCC (PIE)",
                cc=None,
                cxx="g++",
                extra_cflags=[*extra, "-fpie"],
                c_outdir=out_dir / "c-gcc-pie",
                cc_outdir=out_dir / "cc-gcc-pie",
            )
        if system == "Darwin" and Path("/usr/bin/clang").exists():
            _build_variant(
                "Apple Clang",
                cc="/usr/bin/clang",
                cxx="/usr/bin/clang++",
                extra_cflags=extra,
                c_outdir=out_dir / "c-appleclang",
                cc_outdir=out_dir / "cc-appleclang",
            )

    elif system == "Windows":
        llvm = _find_llvm_clang()
        if llvm:
            clang_cl = Path(llvm[0]).parent / "clang-cl.exe"
            if clang_cl.exists():
                _build_variant(
                    "LLVM clang-cl",
                    cc=str(clang_cl),
                    cxx=None,
                    extra_cflags=["/GS-"],
                    c_outdir=out_dir / "c",
                    cc_outdir=None,
                )
        if shutil.which("cl"):
            _build_variant(
                "MSVC",
                cc="cl",
                cxx=None,
                extra_cflags=["/GS-"],
                c_outdir=out_dir / "c-msvc",
                cc_outdir=None,
            )
        if shutil.which("clang-cl"):
            _build_variant(
                "clang-cl",
                cc="clang-cl",
                cxx=None,
                extra_cflags=["/GS-"],
                c_outdir=out_dir / "c-clang-cl",
                cc_outdir=None,
            )

    # CUDA (platform-independent, uses nvcc)
    if shutil.which("nvcc"):
        _build_objects(
            SOURCES_CUDA,
            out_dir / "cuda",
            ext_glob="*.cu",
            extra_cflags=[],
            extra_cuda_cflags=_extra_cuda_cflags(),
        )

    return out_dir
