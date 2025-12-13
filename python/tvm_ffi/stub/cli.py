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
"""TVM-FFI Stub Generator (``tvm-ffi-stubgen``)."""

from __future__ import annotations

import argparse
import ctypes
import importlib
import sys
import traceback
from pathlib import Path
from typing import Callable

from . import codegen as G
from . import consts as C
from .analysis import collect_global_funcs, collect_type_keys
from .file_utils import FileInfo, collect_files
from .utils import FuncInfo, Options


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
    for imp in opt.imports or []:
        importlib.import_module(imp)
    if opt.init_path:
        opt.files.append(opt.init_path)
    dlls = [ctypes.CDLL(lib) for lib in opt.dlls]
    files: list[FileInfo] = collect_files([Path(f) for f in opt.files])
    global_funcs: dict[str, list[FuncInfo]] = collect_global_funcs()

    # Stage 1: Collect information
    # - type maps: `tvm-ffi-stubgen(ty-map)`
    # - defined global functions: `tvm-ffi-stubgen(begin): global/...`
    # - defined object types: `tvm-ffi-stubgen(begin): object/...`
    ty_map: dict[str, str] = C.TY_MAP_DEFAULTS.copy()
    for file in files:
        try:
            _stage_1(file, ty_map)
        except Exception:
            print(
                f'{C.TERM_RED}[Failed] File "{file.path}": {traceback.format_exc()}{C.TERM_RESET}'
            )

    # Stage 2. Generate stubs if they are not defined on the file.
    if opt.init_path:
        _stage_2(
            files,
            init_path=Path(opt.init_path).resolve(),
            global_funcs=global_funcs,
        )

    # Stage 3: Process
    # - `tvm-ffi-stubgen(begin): global/...`
    # - `tvm-ffi-stubgen(begin): object/...`
    for file in files:
        if opt.verbose:
            print(f"{C.TERM_CYAN}[File] {file.path}{C.TERM_RESET}")
        try:
            _stage_3(file, opt, ty_map, global_funcs)
        except Exception:
            print(
                f'{C.TERM_RED}[Failed] File "{file.path}": {traceback.format_exc()}{C.TERM_RESET}'
            )
    del dlls
    return 0


def _stage_1(
    file: FileInfo,
    ty_map: dict[str, str],
) -> None:
    for code in file.code_blocks:
        if code.kind == "ty-map":
            try:
                assert isinstance(code.param, str)
                lhs, rhs = code.param.split("->")
            except ValueError as e:
                raise ValueError(
                    f"Invalid ty_map format at line {code.lineno_start}. Example: `A.B -> C.D`"
                ) from e
            ty_map[lhs.strip()] = rhs.strip()


def _stage_2(
    files: list[FileInfo],
    init_path: Path,
    global_funcs: dict[str, list[FuncInfo]],
) -> None:
    def _find_or_insert_file(path: Path) -> FileInfo:
        ret: FileInfo | None
        if not path.exists():
            ret = FileInfo(path=path, lines=(), code_blocks=[])
        else:
            for file in files:
                if path.samefile(file.path):
                    return file
            ret = FileInfo.from_file(file=path)
            assert ret is not None, f"Failed to read file: {path}"
        files.append(ret)
        return ret

    # Step 0. Find out functions and classes already defined on files.
    defined_func_prefixes: set[str] = {  # type: ignore[union-attr]
        code.param[0] for file in files for code in file.code_blocks if code.kind == "global"
    }
    defined_objs: set[str] = {  # type: ignore[assignment]
        code.param for file in files for code in file.code_blocks if code.kind == "object"
    } | C.BUILTIN_TYPE_KEYS

    # Step 1. Generate missing `_ffi_api.py` and `__init__.py` under each prefix.
    prefixes: dict[str, list[str]] = collect_type_keys()
    for prefix in global_funcs:
        prefixes.setdefault(prefix, [])

    for prefix, obj_names in prefixes.items():
        if prefix.startswith("testing") or prefix.startswith("ffi"):
            continue
        funcs = sorted(
            [] if prefix in defined_func_prefixes else global_funcs.get(prefix, []),
            key=lambda f: f.schema.name,
        )
        objs = sorted(set(obj_names) - defined_objs)
        if not funcs and not objs:
            continue
        # Step 1.1. Create target directory if not exists
        directory = init_path / prefix.replace(".", "/")
        directory.mkdir(parents=True, exist_ok=True)
        # Step 1.2. Generate `_ffi_api.py`
        target_path = directory / "_ffi_api.py"
        target_file = _find_or_insert_file(target_path)
        with target_path.open("a", encoding="utf-8") as f:
            f.write(G.generate_ffi_api(target_file.code_blocks, prefix, objs))
        target_file.reload()
        # Step 1.3. Generate `__init__.py`
        target_path = directory / "__init__.py"
        target_file = _find_or_insert_file(target_path)
        with target_path.open("a", encoding="utf-8") as f:
            f.write(G.generate_init(target_file.code_blocks, prefix, submodule="_ffi_api"))
        target_file.reload()


def _stage_3(
    file: FileInfo,
    opt: Options,
    ty_map: dict[str, str],
    global_funcs: dict[str, list[FuncInfo]],
) -> None:
    all_defined = set()
    ty_used: set[str] = set()
    ty_on_file: set[str] = set()
    fn_ty_map_fn = _fn_ty_map(ty_map, ty_used)
    # Stage 2.1. Process `tvm-ffi-stubgen(begin): global/...`
    for code in file.code_blocks:
        if code.kind == "global":
            funcs = global_funcs.get(code.param[0], [])
            for func in funcs:
                all_defined.add(func.schema.name)
            G.generate_global_funcs(code, funcs, fn_ty_map_fn, opt)
    # Stage 2.2. Process `tvm-ffi-stubgen(begin): object/...`
    for code in file.code_blocks:
        if code.kind == "object":
            type_key = code.param
            assert isinstance(type_key, str)
            ty_on_file.add(ty_map.get(type_key, type_key))
            G.generate_object(code, fn_ty_map_fn, opt)
    # Stage 2.3. Add imports for used types.
    for code in file.code_blocks:
        if code.kind == "import":
            G.generate_imports(code, ty_used - ty_on_file, opt)
            break  # Only one import block per file is supported for now.
    # Stage 2.4. Add `__all__` for defined classes and functions.
    for code in file.code_blocks:
        if code.kind == "__all__":
            G.generate_all(code, all_defined | ty_on_file, opt)
            break  # Only one __all__ block per file is supported for now.
    # Stage 2.5. Process `tvm-ffi-stubgen(begin): export/...`
    for code in file.code_blocks:
        if code.kind == "export":
            G.generate_export(code)
    # Finalize: write back to file
    file.update(verbose=opt.verbose, dry_run=opt.dry_run)


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
            f"  {C.STUB_BEGIN} global/<registry-prefix>\n"
            "  ... generated function stubs ...\n"
            f"  {C.STUB_END}\n\n"
            f"  {C.STUB_BEGIN} object/<type_key>\n"
            f"  {C.STUB_TY_MAP}: list -> Sequence\n"
            f"  {C.STUB_TY_MAP}: dict -> Mapping\n"
            "  ... generated fields and methods ...\n"
            f"  {C.STUB_END}\n\n"
            "  # Skip a file entirely\n"
            f"  {C.STUB_SKIP_FILE}\n\n"
            "Tips:\n"
            "  - Only .py/.pyi files are updated; directories are scanned recursively.\n"
            "  - Import any aliases you use in ty_map under TYPE_CHECKING, e.g.\n"
            "      from collections.abc import Mapping, Sequence\n"
            "  - Use --dlls to preload shared libraries when function/type metadata\n"
            "    is provided by native extensions.\n"
        ),
    )
    parser.add_argument(
        "--imports",
        nargs="*",
        metavar="IMPORTS",
        help=("Additional imports to load before generation."),
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
        "--init-path",
        type=str,
        default="",
        help="If specified, generate stubs under the given package prefix.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=4,
        help=(
            "Extra spaces added inside each generated block, relative to the "
            f"indentation of the corresponding '{C.STUB_BEGIN}' line."
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
