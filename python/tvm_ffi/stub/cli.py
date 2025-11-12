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
# tvm-ffi-stubgen(skip-file)
"""TVM-FFI Stub Generator (``tvm-ffi-stubgen``)."""

from __future__ import annotations

import argparse
import ctypes
import sys
import traceback
from pathlib import Path
from typing import Callable

from . import codegen as G
from . import consts as C
from .analysis import collect_global_funcs, collect_ty_maps
from .file_utils import FileInfo, collect_files
from .utils import Options


def _fn_ty_map(ty_map: dict[str, str], ty_used: set[str]) -> Callable[[str], str]:
    def _run(name: str) -> str:
        nonlocal ty_map, ty_used
        if (ret := ty_map.get(name)) is not None:
            name = ret
        if (ret := C.TY_TO_IMPORT.get(name)) is not None:
            name = ret
        if "." in name:
            ty_used.add(name)
        return name.rsplit(".", 1)[-1]

    return _run


def __main__() -> int:
    """Command line entry point for ``tvm-ffi-stubgen``.

    This generates in-place type stubs inside special ``tvm-ffi-stubgen`` blocks
    in the given files or directories. See the module docstring for an
    overview and examples of the block syntax.
    """
    opt = _parse_args()
    dlls = [ctypes.CDLL(lib) for lib in opt.dlls]
    global_funcs = collect_global_funcs()
    files: list[FileInfo] = collect_files([Path(f) for f in opt.files])

    # Stage 1: Process `tvm-ffi-stubgen(ty-map)`
    ty_map: dict[str, str] = collect_ty_maps(files, opt)

    # Stage 2: Process
    # - `tvm-ffi-stubgen(begin): global/...`
    # - `tvm-ffi-stubgen(begin): object/...`

    def _stage_2(file: FileInfo) -> None:
        if opt.verbose:
            print(f"{C.TERM_CYAN}[File] {file.path}{C.TERM_RESET}")
        ty_used: set[str] = set()
        ty_on_file: set[str] = set()
        fn_ty_map_fn = _fn_ty_map(ty_map, ty_used)
        # Stage 2.1. Process `tvm-ffi-stubgen(begin): global/...`
        for code in file.code_blocks:
            if code.kind == "global":
                G.generate_global_funcs(code, global_funcs, fn_ty_map_fn, opt)
        # Stage 2.2. Process `tvm-ffi-stubgen(begin): object/...`
        for code in file.code_blocks:
            if code.kind == "object":
                G.generate_object(code, fn_ty_map_fn, opt)
                ty_on_file.add(ty_map.get(code.param, code.param))
        # Stage 2.3. Add imports for used types.
        for code in file.code_blocks:
            if code.kind == "import":
                G.generate_imports(code, ty_used - ty_on_file, opt)
                break  # Only one import block per file is supported for now.
        file.update(show_diff=opt.verbose, dry_run=opt.dry_run)

    for file in files:
        try:
            _stage_2(file)
        except:
            print(
                f'{C.TERM_RED}[Failed] File "{file.path}": {traceback.format_exc()}{C.TERM_RESET}'
            )
    del dlls
    return 0


def _parse_args() -> Options:
    class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
        pass

    parser = argparse.ArgumentParser(
        prog="tvm-ffi-stubgen",
        description=(
            "Generate in-place type stubs for TVM FFI.\n\n"
            "It scans .py/.pyi files for tvm-ffi-stubgen blocks and fills them with\n"
            "TYPE_CHECKING-only annotations derived from TVM runtime metadata."
        ),
        formatter_class=HelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Single file\n"
            "  tvm-ffi-stubgen python/tvm_ffi/_ffi_api.py\n\n"
            "  # Recursively scan directories\n"
            "  tvm-ffi-stubgen python/tvm_ffi examples/packaging/python/my_ffi_extension\n\n"
            "  # Preload TVM runtime / extension libraries\n"
            "  tvm-ffi-stubgen --dlls build/libtvm_runtime.so build/libmy_ext.so my_pkg/_ffi_api.py\n\n"
            "Stub block syntax (placed in your source):\n"
            "  # tvm-ffi-stubgen(begin): global/<registry-prefix>\n"
            "  ... generated function stubs ...\n"
            "  # tvm-ffi-stubgen(end)\n\n"
            "  # tvm-ffi-stubgen(begin): object/<type_key>\n"
            "  # tvm-ffi-stubgen(ty_map): list -> Sequence\n"
            "  # tvm-ffi-stubgen(ty_map): dict -> Mapping\n"
            "  ... generated fields and methods ...\n"
            "  # tvm-ffi-stubgen(end)\n\n"
            "  # Skip a file entirely\n"
            "  # tvm-ffi-stubgen(skip-file)\n\n"
            "Tips:\n"
            "  - Only .py/.pyi files are updated; directories are scanned recursively.\n"
            "  - Import any aliases you use in ty_map under TYPE_CHECKING, e.g.\n"
            "      from collections.abc import Mapping, Sequence\n"
            "  - Use --dlls to preload shared libraries when function/type metadata\n"
            "    is provided by native extensions.\n"
        ),
    )
    parser.add_argument(
        "--dlls",
        nargs="*",
        metavar="LIB",
        help=(
            "Shared libraries to preload before generation (e.g. TVM runtime or "
            "your extension). This ensures global function and object metadata "
            "is available. Accepts multiple paths; platform-specific suffixes "
            "like .so/.dylib/.dll are supported."
        ),
        default=[],
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=4,
        help=(
            "Extra spaces added inside each generated block, relative to the "
            "indentation of the corresponding '# tvm-ffi-stubgen(begin):' line."
        ),
    )
    parser.add_argument(
        "files",
        nargs="*",
        metavar="PATH",
        help=(
            "Files or directories to process. Directories are scanned recursively; "
            "only .py and .pyi files are modified. Use tvm-ffi-stubgen markers to "
            "select where stubs are generated."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "Print a unified diff of changes to each file. This is useful for "
            "debugging or previewing changes before applying them."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Don't write changes to files. This is useful for previewing changes "
            "without modifying any files."
        ),
    )
    opt = Options(**vars(parser.parse_args()))
    if not opt.files:
        parser.print_help()
        sys.exit(1)
    return opt


if __name__ == "__main__":
    sys.exit(__main__())
