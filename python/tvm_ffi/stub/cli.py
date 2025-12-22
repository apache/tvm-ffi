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

from . import codegen as G
from . import consts as C
from .file_utils import FileInfo, collect_files
from .lib_state import (
    collect_global_funcs,
    collect_type_keys,
    object_info_from_type_key,
    toposort_objects,
)
from .utils import FuncInfo, ImportItem, InitConfig, Options


def __main__() -> int:
    """Command line entry point for ``tvm-ffi-stubgen``.

    This generates in-place type stubs inside special ``tvm-ffi-stubgen`` blocks
    in the given files or directories. See the module docstring for an
    overview and examples of the block syntax.
    """
    opt = _parse_args()
    for imp in opt.imports or []:
        importlib.import_module(imp)
    dlls = [ctypes.CDLL(lib) for lib in opt.dlls]
    files: list[FileInfo] = collect_files([Path(f) for f in opt.files])
    global_funcs: dict[str, list[FuncInfo]] = collect_global_funcs()
    init_path: Path | None = None
    if opt.files:
        init_path = Path(opt.files[0]).resolve()
        if init_path.is_file():
            init_path = init_path.parent

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
    if opt.init:
        assert init_path is not None, "init-path could not be determined"
        _stage_2(
            files,
            ty_map,
            init_cfg=opt.init,
            init_path=init_path,
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
    ty_map: dict[str, str],
    init_cfg: InitConfig,
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
            ret = FileInfo.from_file(file=path, include_empty=True)
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

    # Step 0. Generate missing `_ffi_api.py` and `__init__.py` under each prefix.
    prefix_filter = init_cfg.prefix.strip()
    if prefix_filter and not prefix_filter.endswith("."):
        prefix_filter += "."
    root_prefix = prefix_filter.rstrip(".")
    prefixes: dict[str, list[str]] = collect_type_keys()
    for prefix in global_funcs:
        prefixes.setdefault(prefix, [])
    for prefix, obj_names in prefixes.items():
        if not (prefix == root_prefix or prefix.startswith(prefix_filter)):
            continue
        funcs = sorted(
            [] if prefix in defined_func_prefixes else global_funcs.get(prefix, []),
            key=lambda f: f.schema.name,
        )
        objs = sorted(set(obj_names) - defined_objs)
        object_infos = toposort_objects(objs)
        if not funcs and not object_infos:
            continue
        # Step 1. Create target directory if not exists
        directory = init_path / prefix.replace(".", "/")
        directory.mkdir(parents=True, exist_ok=True)
        # Step 2. Generate `_ffi_api.py`
        target_path = directory / "_ffi_api.py"
        target_file = _find_or_insert_file(target_path)
        with target_path.open("a", encoding="utf-8") as f:
            f.write(
                G.generate_ffi_api(
                    target_file.code_blocks,
                    ty_map,
                    prefix,
                    object_infos,
                    init_cfg,
                    is_root=prefix == root_prefix,
                )
            )
        target_file.reload()
        # Step 3. Generate `__init__.py`
        target_path = directory / "__init__.py"
        target_file = _find_or_insert_file(target_path)
        with target_path.open("a", encoding="utf-8") as f:
            f.write(G.generate_init(target_file.code_blocks, prefix, submodule="_ffi_api"))
        target_file.reload()


def _stage_3(  # noqa: PLR0912
    file: FileInfo,
    opt: Options,
    ty_map: dict[str, str],
    global_funcs: dict[str, list[FuncInfo]],
) -> None:
    defined_funcs: set[str] = set()
    defined_types: set[str] = set()
    imports: list[ImportItem] = []
    ffi_load_lib_imported = False
    # Stage 1. Collect `tvm-ffi-stubgen(import-object): ...`
    for code in file.code_blocks:
        if code.kind == "import-object":
            name, type_checking_only, alias = code.param  # type: ignore[misc]
            imports.append(
                ImportItem(
                    name,
                    type_checking_only=(
                        bool(type_checking_only)
                        and isinstance(type_checking_only, str)
                        and type_checking_only.lower() == "true"
                    ),
                    alias=alias if alias else None,
                )
            )
            if (alias and alias == "_FFI_LOAD_LIB") or name.endswith("libinfo.load_lib_module"):
                ffi_load_lib_imported = True
    # Stage 2. Process `tvm-ffi-stubgen(begin): global/...`
    for code in file.code_blocks:
        if code.kind == "global":
            funcs = global_funcs.get(code.param[0], [])
            for func in funcs:
                defined_funcs.add(func.schema.name)
            G.generate_global_funcs(code, funcs, ty_map, imports, opt)
    # Stage 3. Process `tvm-ffi-stubgen(begin): object/...`
    for code in file.code_blocks:
        if code.kind == "object":
            type_key = code.param
            assert isinstance(type_key, str)
            obj_info = object_info_from_type_key(type_key)
            type_key = ty_map.get(type_key, type_key)
            full_name = ImportItem(type_key).full_name
            defined_types.add(full_name)
            G.generate_object(code, ty_map, imports, opt, obj_info)
    # Stage 4. Add imports for used types.
    imports = [i for i in imports if i.full_name not in defined_types]
    for code in file.code_blocks:
        if code.kind == "import-section":
            G.generate_import_section(code, imports, opt)
            break  # Only one import block per file is supported for now.
    # Stage 5. Add `__all__` for defined classes and functions.
    for code in file.code_blocks:
        if code.kind == "__all__":
            export_names = defined_funcs | defined_types
            if ffi_load_lib_imported:
                export_names = export_names | {"LIB"}
            G.generate_all(code, export_names, opt)
            break  # Only one __all__ block per file is supported for now.
    # Stage 6. Process `tvm-ffi-stubgen(begin): export/...`
    for code in file.code_blocks:
        if code.kind == "export":
            G.generate_export(code)
    # Finalize: write back to file
    file.update(verbose=opt.verbose, dry_run=opt.dry_run)


def _parse_args() -> Options:
    class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
        pass

    def _split_list_arg(arg: str | None) -> list[str]:
        if not arg:
            return []
        return [item.strip() for item in arg.split(";") if item.strip()]

    parser = argparse.ArgumentParser(
        prog="tvm-ffi-stubgen",
        description=(
            "Generate type stubs for TVM FFI extensions.\n\n"
            "In `--init-*` mode, it generates missing `_ffi_api.py` and `__init__.py` files, "
            "based on the registered global functions and object types in the loaded libraries.\n\n"
            "In normal mode, it processes the given files/directories in-place, generating "
            "type stubs inside special `tvm-ffi-stubgen` blocks. Scroo down for more details."
        ),
        formatter_class=HelpFormatter,
        epilog=(
            "========\n"
            "Examples\n"
            "========\n\n"
            "  # Single file\n"
            "  tvm-ffi-stubgen python/tvm_ffi/_ffi_api.py\n\n"
            "  # Recursively scan directories\n"
            "  tvm-ffi-stubgen python/tvm_ffi examples/packaging/python/my_ffi_extension\n\n"
            "  # Preload extension libraries\n"
            "  tvm-ffi-stubgen --dlls build/libmy_ext.so;build/libmy_2nd_ext.so my_pkg/_ffi_api.py\n\n"
            "  # Package-level init (my-ffi-extension)\n"
            "  tvm-ffi-stubgen examples/packaging/python \\\n"
            "    --dlls examples/packaging/build/libmy_ffi_extension.dylib \\\n"
            "    --init-pypkg my-ffi-extension \\\n"
            "    --init-lib my_ffi_extension \\\n"
            '    --init-prefix "my_ffi_extension."\n\n'
            "=====================\n"
            "Syntax of stub blocks\n"
            "=====================\n\n"
            "Global functions\n"
            "~~~~~~~~~~~~~~~~\n\n"
            "    ```\n"
            f"    {C.STUB_BEGIN} global/<registry-prefix>@<import-from (default: tvm_ffi)>\n"
            f"    {C.STUB_END}\n"
            "    ```\n\n"
            "Generates TYPE_CHECKING-only stubs for functions in the global registry under the prefix.\n\n"
            "Example:\n\n"
            "    ```\n"
            f"    {C.STUB_BEGIN} global/ffi@.registry\n"
            "    # fmt: off\n"
            '    _FFI_INIT_FUNC("ffi", __name__)\n'
            "    if TYPE_CHECKING:\n"
            "        def Array(*args: Any) -> Any: ...\n"
            "        def ArrayGetItem(_0: Sequence[Any], _1: int, /) -> Any: ...\n"
            "        def ArraySize(_0: Sequence[Any], /) -> int: ...\n"
            "        def Bytes(_0: bytes, /) -> bytes: ...\n"
            "        ...\n"
            "        def StructuralHash(_0: Any, _1: bool, _2: bool, /) -> int: ...\n"
            "        def SystemLib(*args: Any) -> Any: ...\n"
            "        def ToJSONGraph(_0: Any, _1: Any, /) -> Any: ...\n"
            "        def ToJSONGraphString(_0: Any, _1: Any, /) -> str: ...\n"
            "    # fmt: on\n"
            f"    {C.STUB_END}\n"
            "    ```\n\n"
            "Objects\n"
            "~~~~~~~\n\n"
            "    ```\n"
            f"    {C.STUB_BEGIN} object/<type_key>\n"
            f"    {C.STUB_END}\n"
            "    ```\n\n"
            "Generates fields/methods for a class defined using TVM-FFI Object APIs.\n\n"
            "Example:\n\n"
            "    ```\n"
            '    @register_object("ffi.reflection.AccessPath")\n'
            "    class AccessPath(tvm_ffi.Object):\n"
            f"        {C.STUB_BEGIN} object/ffi.reflection.AccessPath\n"
            "        # fmt: off\n"
            "        parent: Object | None\n"
            "        step: AccessStep | None\n"
            "        depth: int\n"
            "        if TYPE_CHECKING:\n"
            "            @staticmethod\n"
            "            def _root() -> AccessPath: ...\n"
            "            def _extend(self, _1: AccessStep, /) -> AccessPath: ...\n"
            "            def _attr(self, _1: str, /) -> AccessPath: ...\n"
            "            def _array_item(self, _1: int, /) -> AccessPath: ...\n"
            "            def _map_item(self, _1: Any, /) -> AccessPath: ...\n"
            "            def _attr_missing(self, _1: str, /) -> AccessPath: ...\n"
            "            def _array_item_missing(self, _1: int, /) -> AccessPath: ...\n"
            "            def _map_item_missing(self, _1: Any, /) -> AccessPath: ...\n"
            "            def _is_prefix_of(self, _1: AccessPath, /) -> bool: ...\n"
            "            def _to_steps(self, /) -> Sequence[AccessStep]: ...\n"
            "            def _path_equal(self, _1: AccessPath, /) -> bool: ...\n"
            "        # fmt: on\n"
            f"        {C.STUB_END}\n"
            "    ```\n\n"
            "Import section\n"
            "~~~~~~~~~~~~~~\n\n"
            "    ```\n"
            f"    {C.STUB_BEGIN} import-section\n"
            "    # fmt: off\n"
            "    # isort: off\n"
            "    from __future__ import annotations\n"
            "    from ..registry import init_ffi_api as _FFI_INIT_FUNC\n"
            "    from typing import TYPE_CHECKING\n"
            "    if TYPE_CHECKING:\n"
            "        from collections.abc import Mapping, Sequence\n"
            "        from tvm_ffi import Device, Object, Tensor, dtype\n"
            "        from tvm_ffi.testing import TestIntPair\n"
            "        from typing import Any, Callable\n"
            "    # isort: on\n"
            "    # fmt: on\n"
            f"    {C.STUB_END}\n"
            "    ```\n\n"
            "Auto-populates imports used by generated stubs.\n\n"
            "Export\n"
            "~~~~~~\n\n"
            "    ```\n"
            f"    {C.STUB_BEGIN} export/_ffi_api\n"
            "    # fmt: off\n"
            "    # isort: off\n"
            "    from ._ffi_api import *  # noqa: F403\n"
            "    from ._ffi_api import __all__ as _ffi_api__all__\n"
            '    if "__all__" not in globals():\n'
            "        __all__ = []\n"
            "    __all__.extend(_ffi_api__all__)\n"
            "    # isort: on\n"
            "    # fmt: on\n"
            f"    {C.STUB_END}\n"
            "    ```\n\n"
            "Re-exports a generated submodule's __all__ into the parent.\n\n"
            "__all__\n"
            "~~~~~~~\n\n"
            "    ```\n"
            "    __all__ = [\n"
            f"        {C.STUB_BEGIN} __all__\n"
            '        "LIB",\n'
            '        "IntPair",\n'
            '        "raise_error",\n'
            f"        {C.STUB_END}\n"
            "    ]\n"
            "    ```\n\n"
            "Populates __all__ with generated classes/functions and LIB (if present).\n\n"
            "Type map\n"
            "~~~~~~~~\n\n"
            "    ```\n"
            f"    {C.STUB_TY_MAP} <type_key> -> <python_type>\n"
            "    ```\n\n"
            "Maps runtime type keys to Python types used in generation.\n\n"
            "Example:\n\n"
            "    ```\n"
            f"    {C.STUB_TY_MAP} ffi.reflection.AccessStep -> ffi.access_path.AccessStep\n"
            "    ```\n\n"
            "Import object\n"
            "~~~~~~~~~~~~~\n\n"
            "    ```\n"
            f"    {C.STUB_IMPORT_OBJECT} <from>; <type_checking_only>; <alias>\n"
            "    ```\n\n"
            "Injects a custom import into generated code, optionally TYPE_CHECKING-only.\n\n"
            "Example:\n\n"
            "    ```\n"
            f"    {C.STUB_IMPORT_OBJECT} ffi.Object;False;_ffi_Object\n"
            "    ```\n\n"
            "Skip file\n"
            "~~~~~~~~~\n\n"
            "    ```\n"
            f"    {C.STUB_SKIP_FILE}\n"
            "    ```\n\n"
            "Prevents stubgen from modifying the file."
        ),
    )
    parser.add_argument(
        "--imports",
        type=str,
        default="",
        metavar="IMPORTS",
        help=(
            "Additional imports to load before generation, separated by ';' "
            "(e.g. 'pkgA;pkgB.submodule')."
        ),
    )
    parser.add_argument(
        "--dlls",
        type=str,
        default="",
        metavar="LIBS",
        help=(
            "Shared libraries to preload before generation (e.g. TVM runtime or "
            "your extension), separated by ';'. This ensures global function and "
            "object metadata is available. Platform-specific suffixes like "
            ".so/.dylib/.dll are supported."
        ),
    )
    parser.add_argument(
        "--init-pypkg",
        type=str,
        default="",
        help=(
            "Python package name to generate stubs for (e.g. apache-tvm-ffi). "
            "Required together with --init-lib and --init-prefix."
        ),
    )
    parser.add_argument(
        "--init-lib",
        type=str,
        default="",
        help=(
            "CMake target that produces the shared library to load for stub generation "
            "(e.g. tvm_ffi_shared). Required together with --init-pypkg and "
            "--init-prefix."
        ),
    )
    parser.add_argument(
        "--init-prefix",
        type=str,
        default="",
        help=(
            "Global function/object prefix to include when generating stubs "
            "(e.g. tvm_ffi.). Required together with --init-pypkg and --init-lib."
        ),
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
    args = parser.parse_args()

    init_flags = [args.init_pypkg, args.init_lib, args.init_prefix]
    init_cfg: InitConfig | None = None
    if any(init_flags):
        if not all(init_flags):
            parser.error("--init-pypkg, --init-lib, and --init-prefix must be provided together")
        init_cfg = InitConfig(
            pkg=args.init_pypkg,
            shared_target=args.init_lib,
            prefix=args.init_prefix,
        )

    if not args.files:
        parser.print_help()
        sys.exit(1)

    return Options(
        imports=_split_list_arg(args.imports),
        dlls=_split_list_arg(args.dlls),
        init=init_cfg,
        indent=args.indent,
        files=args.files,
        verbose=args.verbose,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    sys.exit(__main__())
