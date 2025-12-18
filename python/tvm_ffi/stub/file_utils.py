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
"""Utilities for parsing and generating stub files for TVM FFI."""

from __future__ import annotations

import dataclasses
import difflib
import os
import traceback
from pathlib import Path
from typing import Callable, Generator, Iterable

from . import consts as C


@dataclasses.dataclass
class CodeBlock:
    """A block of code to be generated in a stub file."""

    kind: C.STUB_BLOCK_KINDS
    param: str | tuple[str, ...]
    lineno_start: int
    lineno_end: int | None
    lines: list[str]

    def __post_init__(self) -> None:
        """Validate the code block after initialization."""
        assert self.kind in {
            "global",
            "object",
            "ty-map",
            "import-section",
            "import-object",
            "export",
            "__all__",
            None,
        }

    @property
    def indent(self) -> int:
        """Calculate the indentation level of the block based on the first line."""
        if not self.lines:
            return 0
        first_line = self.lines[0]
        return len(first_line) - len(first_line.lstrip(" "))

    @staticmethod
    def from_begin_line(lineo: int, line: str) -> CodeBlock:
        """Parse a line to create a CodeBlock if it contains a stub begin marker."""
        if line.startswith(C.STUB_TY_MAP):
            line = line[len(C.STUB_TY_MAP) :].strip()
            return CodeBlock(
                kind="ty-map",
                param=line,
                lineno_start=lineo,
                lineno_end=lineo,
                lines=[],
            )
        elif line.startswith(C.STUB_IMPORT_OBJECT):
            line = line[len(C.STUB_IMPORT_OBJECT) :].strip()
            splits = [p.strip() for p in line.split(";")]
            if len(splits) < 3:
                splits += [""] * (3 - len(splits))
            return CodeBlock(
                kind="import-object",
                param=tuple(splits),
                lineno_start=lineo,
                lineno_end=lineo,
                lines=[],
            )
        assert line.startswith(C.STUB_BEGIN)
        param: str | tuple[str, ...]
        stub = line[len(C.STUB_BEGIN) :].strip()
        if stub.startswith("global/"):
            kind = "global"
            param = stub[len("global/") :].strip()
            if "@" in param:
                param = tuple(param.split("@"))
            else:
                param = (param, "")
        elif stub.startswith("object/"):
            kind = "object"
            param = stub[len("object/") :].strip()
        elif stub.startswith("ty-map/"):
            kind = "ty-map"
            param = stub[len("ty-map/") :].strip()
        elif stub == "import-section":
            kind = "import-section"
            param = ""
        elif stub.startswith("export/"):
            kind = "export"
            param = stub[len("export/") :].strip()
        elif stub == "__all__":
            kind = "__all__"
            param = ""
        else:
            raise ValueError(f"Unknown stub type `{stub}` at line {lineo}")
        return CodeBlock(
            kind=kind,  # type: ignore[arg-type]
            param=param,
            lineno_start=lineo,
            lineno_end=None,
            lines=[],
        )


@dataclasses.dataclass
class FileInfo:
    """Information about a file being processed."""

    path: Path
    lines: tuple[str, ...]
    code_blocks: list[CodeBlock]

    def update(self, verbose: bool, dry_run: bool) -> bool:
        """Update the file's lines based on the current code blocks and optionally show a diff."""
        new_lines = tuple(line for block in self.code_blocks for line in block.lines)
        if self.lines == new_lines:
            if verbose:
                print(f"{C.TERM_CYAN}-----> Unchanged{C.TERM_RESET}")
            return False
        if verbose:
            for line in difflib.unified_diff(self.lines, new_lines, lineterm=""):
                # Skip placeholder headers when fromfile/tofile are unspecified
                if line.startswith("---") or line.startswith("+++"):
                    continue
                if line.startswith("-") and not line.startswith("---"):
                    print(f"{C.TERM_RED}{line}{C.TERM_RESET}")  # Red for removals
                elif line.startswith("+") and not line.startswith("+++"):
                    print(f"{C.TERM_GREEN}{line}{C.TERM_RESET}")  # Green for additions
                elif line.startswith("?"):
                    print(f"{C.TERM_YELLOW}{line}{C.TERM_RESET}")  # Yellow for hints
                else:
                    print(line)
        self.lines = new_lines
        if not dry_run:
            self.path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        return True

    @staticmethod
    def from_file(file: Path, include_empty: bool = False) -> FileInfo | None:  # noqa: PLR0912
        """Parse a file to extract code blocks based on stub markers."""
        assert file.is_file(), f"Expected a file, but got: {file}"
        file = file.resolve()
        has_marker = False
        lines: list[str] = file.read_text(encoding="utf-8").splitlines()
        for _, line in enumerate(lines, start=1):
            if line.strip().startswith(C.STUB_SKIP_FILE):
                return None
            if line.strip().startswith(C.STUB_PREFIX):
                has_marker = True
        if not has_marker and not include_empty:
            return None
        del has_marker

        codes: list[CodeBlock] = []
        code: CodeBlock | None = None
        for lineno, line in enumerate(lines, 1):
            clean_line = line.strip()
            if clean_line.startswith(C.STUB_BEGIN):
                # Process "# tvm-ffi-stubgen(begin)"
                if code is not None:
                    raise ValueError(f"Nested stub not permitted at line {lineno}")
                code = CodeBlock.from_begin_line(lineno, clean_line)
                code.lineno_start = lineno
                code.lines.append(line)
            elif clean_line.startswith(C.STUB_END):
                # Process "# tvm-ffi-stubgen(end)"
                if code is None:
                    raise ValueError(f"Unmatched `{C.STUB_END}` found at line {lineno}")
                code.lineno_end = lineno
                code.lines.append(line)
                codes.append(code)
                code = None
            elif clean_line.startswith(C.STUB_TY_MAP):
                # Process "# tvm-ffi-stubgen(ty_map)"
                ty_code = CodeBlock.from_begin_line(lineno, clean_line)
                ty_code.lineno_end = lineno
                ty_code.lines.append(line)
                codes.append(ty_code)
                del ty_code
            elif clean_line.startswith(C.STUB_IMPORT_OBJECT):
                # Process "# tvm-ffi-stubgen(import-object)"
                imp_code = CodeBlock.from_begin_line(lineno, clean_line)
                imp_code.lineno_end = lineno
                imp_code.lines.append(line)
                codes.append(imp_code)
                del imp_code
            elif clean_line.startswith(C.STUB_PREFIX):
                raise ValueError(f"Unknown stub type at line {lineno}: {clean_line}")
            elif code is None:
                # Process a plain line outside of any stub block
                codes.append(
                    CodeBlock(
                        kind=None,
                        param="",
                        lineno_start=lineno,
                        lineno_end=lineno,
                        lines=[line],
                    )
                )
            else:  # Process a line inside a stub block
                code.lines.append(line)
        if code is not None:
            raise ValueError("Unclosed stub block at end of file")
        return FileInfo(path=file, lines=tuple(lines), code_blocks=codes)

    def reload(self) -> None:
        """Reload the code blocks from disk while preserving original `lines`."""
        source = FileInfo.from_file(self.path)
        assert source is not None, f"File no longer exists or valid: {self.path}"
        self.code_blocks = source.code_blocks


def collect_files(paths: list[Path]) -> list[FileInfo]:
    """Collect all files from the given paths and parse them into FileInfo objects."""

    def _on_error(e: Exception) -> None:
        print(
            f"{C.TERM_RED}[Error]\n{traceback.format_exc()}{C.TERM_RESET}",
            end="",
            flush=True,
        )

    def _walk_recursive() -> Generator[Path, None, None]:
        for p in paths:
            if p.is_file():
                yield p
                continue
            for root, _dirs, files in path_walk(p, follow_symlinks=False, on_error=_on_error):
                for file in files:
                    f = Path(root) / file
                    if f.suffix.lower() not in C.DEFAULT_SOURCE_EXTS:
                        continue
                    yield f

    filenames = list(_walk_recursive())
    filenames = sorted(filenames, key=lambda f: str(f))
    files = []
    for file in filenames:
        try:
            content = FileInfo.from_file(file)
        except Exception as e:
            _on_error(e)
        else:
            if content is not None:
                files.append(content)
    return files


def path_walk(
    p: Path,
    *,
    top_down: bool = True,
    on_error: Callable[[Exception], None] | None = None,
    follow_symlinks: bool = False,
) -> Iterable[tuple[Path, list[str], list[str]]]:
    """Compat wrapper for Path.walk (3.12+) with a fallback for < 3.12."""
    if not p.exists():
        return
    # Python 3.12+ - just delegate to `Path.walk`
    if hasattr(p, "walk"):
        yield from p.walk(  # type: ignore[attr-defined]
            top_down=top_down,
            on_error=on_error,
            follow_symlinks=follow_symlinks,
        )
        return
    # Python < 3.12 - use `os.walk``
    for root_str, dirnames, filenames in os.walk(
        p,
        topdown=top_down,
        onerror=on_error,
        followlinks=follow_symlinks,
    ):
        root = Path(root_str)
        # dirnames and filenames are lists of *names*, not full paths,
        # just like Path.walk()'s documented behavior.
        yield root, dirnames, filenames
