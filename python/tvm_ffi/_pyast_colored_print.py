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
"""Syntax-highlighted printing for Python-style AST output.

Uses Pygments when available. Falls back to plain ``print()`` with a one-time
warning suggesting ``pip install pygments``.
"""

from __future__ import annotations

import sys
import typing
import warnings

if typing.TYPE_CHECKING:
    from pygments.style import Style

_PYGMENTS_WARNED = False


def cprint(text: str, style: str | None = None) -> None:
    """Print Python source with Pygments syntax highlighting.

    Parameters
    ----------
    text
        The source code string to print.
    style
        Pygments style name or one of the built-in aliases ``"light"``,
        ``"dark"``, ``"ansi"``.  When ``None``, ``"light"`` is used inside
        Jupyter notebooks and ``"ansi"`` in terminals.

    Notes
    -----
    When the optional ``pygments`` package is not installed, plain text is
    printed with a one-time warning suggesting installation.  Additional
    Pygments styles are listed at https://pygments.org/styles/.

    """
    is_notebook = "ipykernel" in sys.modules
    pygment_style = _resolve_style(style, is_notebook)
    if pygment_style is None:
        print(text)
        return
    from pygments import highlight  # noqa: PLC0415
    from pygments.formatters import (  # noqa: PLC0415
        HtmlFormatter,  # ty: ignore[unresolved-import]
        Terminal256Formatter,  # ty: ignore[unresolved-import]
    )
    from pygments.lexers.python import Python3Lexer  # noqa: PLC0415

    if is_notebook:
        from IPython import display  # noqa: PLC0415

        formatter = HtmlFormatter(style=pygment_style, noclasses=True)
        display.display(display.HTML(highlight(text, Python3Lexer(), formatter)))
    else:
        print(highlight(text, Python3Lexer(), Terminal256Formatter(style=pygment_style)))


def _resolve_style(
    style: str | None,
    is_notebook: bool,
) -> type[Style] | str | None:
    """Return a Pygments style class/name, or ``None`` if Pygments is missing."""
    global _PYGMENTS_WARNED  # noqa: PLW0603
    try:
        from pygments.style import Style  # noqa: PLC0415
        from pygments.token import (  # noqa: PLC0415
            Comment,
            Keyword,
            Name,
            Number,
            Operator,
            String,
        )
    except ModuleNotFoundError:
        if not _PYGMENTS_WARNED:
            _PYGMENTS_WARNED = True
            warnings.warn(
                "pygments is not installed; falling back to plain text. "
                "Install it for syntax-highlighted output: pip install pygments",
                stacklevel=3,
            )
        return None

    class _JupyterLight(Style):
        """Jupyter-Notebook-like Pygments style (alias: ``"light"``)."""

        background_color = ""
        styles: typing.ClassVar = {
            Keyword: "bold #008000",
            Keyword.Type: "nobold #008000",
            Name.Function: "#0000FF",
            Name.Class: "bold #0000FF",
            Name.Decorator: "#AA22FF",
            String: "#BA2121",
            Number: "#008000",
            Operator: "bold #AA22FF",
            Operator.Word: "bold #008000",
            Comment: "italic #007979",
        }

    class _VSCDark(Style):
        """VSCode-Dark-like Pygments style (alias: ``"dark"``)."""

        background_color = ""
        styles: typing.ClassVar = {
            Keyword: "bold #c586c0",
            Keyword.Type: "#82aaff",
            Keyword.Namespace: "#4ec9b0",
            Name.Class: "bold #569cd6",
            Name.Function: "bold #dcdcaa",
            Name.Decorator: "italic #fe4ef3",
            String: "#ce9178",
            Number: "#b5cea8",
            Operator: "#bbbbbb",
            Operator.Word: "#569cd6",
            Comment: "italic #6a9956",
        }

    class _AnsiTerminal(Style):
        """ANSI terminal Pygments style (alias: ``"ansi"``)."""

        background_color = ""
        styles: typing.ClassVar = {
            Keyword: "bold ansigreen",
            Keyword.Type: "nobold ansigreen",
            Name.Class: "bold ansiblue",
            Name.Function: "bold ansiblue",
            Name.Decorator: "italic ansibrightmagenta",
            String: "ansiyellow",
            Number: "ansibrightgreen",
            Operator: "bold ansimagenta",
            Operator.Word: "bold ansigreen",
            Comment: "italic ansibrightblack",
        }

    _BUILTIN_STYLES: dict[str, type[Style]] = {
        "light": _JupyterLight,
        "dark": _VSCDark,
        "ansi": _AnsiTerminal,
    }
    if style is not None:
        return _BUILTIN_STYLES.get(style, style)
    return _JupyterLight if is_notebook else _AnsiTerminal
