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
# pylint: disable=invalid-name
"""Error handling."""

from __future__ import annotations

import ast
import re
import sys
import types
from typing import Any

from . import core


def _parse_backtrace(backtrace: str) -> list[tuple[str, int, str]]:
    """Parse the backtrace string into a list of (filename, lineno, func).

    Parameters
    ----------
    backtrace
        The backtrace string.

    Returns
    -------
    result
        The list of (filename, lineno, func)

    """
    pattern = r'File "(.+?)", line (\d+), in (.+)'
    result = []
    for line in backtrace.split("\n"):
        match = re.match(pattern, line.strip())
        if match:
            try:
                filename = match.group(1)
                lineno = int(match.group(2))
                func = match.group(3)
                result.append((filename, lineno, func))
            except ValueError:
                pass
    return result


class TracebackManager:
    """Helper to manage traceback generation."""

    def __init__(self) -> None:
        """Initialize the traceback manager and its cache."""
        self._code_cache: dict[tuple[str, int, str], types.CodeType] = {}

    def _get_cached_code_object(self, filename: str, lineno: int, func: str) -> types.CodeType:
        # Hack to create a code object that points to the correct
        # line number and function name
        key = (filename, lineno, func)
        # cache the code object to avoid re-creating it
        if key in self._code_cache:
            return self._code_cache[key]
        # Parse to AST and zero out column info
        # since column info are not accurate in original trace
        tree = ast.parse("_getframe()", filename=filename, mode="eval")
        for node in ast.walk(tree):
            if hasattr(node, "col_offset"):
                node.col_offset = 0
            if hasattr(node, "end_col_offset"):
                node.end_col_offset = 0
        # call into get frame, bt changes the context
        code_object = compile(tree, filename, "eval")
        # replace the function name and line number
        code_object = code_object.replace(co_name=func, co_firstlineno=lineno)
        self._code_cache[key] = code_object
        return code_object

    def _create_frame(self, filename: str, lineno: int, func: str) -> types.FrameType:
        """Create a frame object from the filename, lineno, and func."""
        code_object = self._get_cached_code_object(filename, lineno, func)
        # call into get frame, but changes the context so the code
        # points to the correct frame
        context = {"_getframe": sys._getframe}
        # pylint: disable=eval-used
        return eval(code_object, context, context)

    def append_traceback(
        self,
        tb: types.TracebackType | None,
        filename: str,
        lineno: int,
        func: str,
    ) -> types.TracebackType:
        """Append a traceback to the given traceback.

        Parameters
        ----------
        tb
            The traceback to append to.
        filename
            The filename of the traceback
        lineno
            The line number of the traceback
        func
            The function name of the traceback

        Returns
        -------
        new_tb
            The new traceback with the appended frame.

        """

        # This approach avoids binding the created frame object to a local variable
        # in `append_traceback`, which would create a reference cycle. By using a
        # nested function, the frame object is a temporary that is not held by
        # the locals of `append_traceback`. See the diagram in `_with_append_backtrace`
        # and PR #327 for more details.
        def create(
            tb: types.TracebackType | None, frame: types.FrameType, lineno: int
        ) -> types.TracebackType:
            return types.TracebackType(tb, frame, frame.f_lasti, lineno)

        return create(tb, self._create_frame(filename, lineno, func), lineno)


_TRACEBACK_MANAGER = TracebackManager()


def _with_append_backtrace(py_error: BaseException, backtrace: str) -> BaseException:
    """Append the backtrace to the py_error and return it."""
    # We manually delete py_error and tb to avoid reference cycle, making it faster to gc the locals inside the frame
    # please see pull request #327 for more details
    #
    # Memory Cycle Diagram:
    #
    #         [Stack Frames]                            [Heap Objects]
    #     +-------------------+
    #     | outside functions | -----------------------> [ Tensor ]
    #     +-------------------+                   (Held by cycle, slow to free)
    #             ^
    #             | f_back
    #     +-------------------+  locals      py_error
    #     | py_error (this)   | -----+--------------> [ BaseException ]
    #     +-------------------+      |                       |
    #             ^                  |                       | (with_traceback)
    #             | f_back           |                       v
    #     +-------------------+      +--------------> [ Traceback Obj ]
    #     | append_traceback  |                   tb         |
    #     +-------------------+                              |
    #             ^                                          |
    #             | f_back                                   |
    #     +-------------------+                              |
    #     | _create_frame     |                              |
    #     +-------------------+                              |
    #             ^                                          |
    #             | f_back                                   |
    #     +-------------------+                              |
    #     | _get_frame        | <----------------------------+
    #     +-------------------+      (Cycle closes here)
    tb = py_error.__traceback__
    try:
        for filename, lineno, func in _parse_backtrace(backtrace):
            tb = _TRACEBACK_MANAGER.append_traceback(tb, filename, lineno, func)
        return py_error.with_traceback(tb)
    finally:
        # We explicitly break the reference cycle here. The `finally` block is
        # executed just before the function returns, after the `return` expression
        # in the `try` block has been evaluated. Deleting `py_error` and `tb`
        # here ensures they are not held by this function's frame's locals,
        # which resolves the cycle.
        del py_error, tb


def _traceback_to_backtrace_str(tb: types.TracebackType | None) -> str:
    """Convert the traceback to a string."""
    lines = []
    while tb is not None:
        frame = tb.tb_frame
        lineno = tb.tb_lineno
        filename = frame.f_code.co_filename
        funcname = frame.f_code.co_name
        lines.append(f'  File "{filename}", line {lineno}, in {funcname}\n')
        tb = tb.tb_next
    # needs to reverse the order of the lines so backtrace stores in
    # the reverse order of python traceback
    return "".join(reversed(lines))


core._WITH_APPEND_BACKTRACE = _with_append_backtrace
core._TRACEBACK_TO_BACKTRACE_STR = _traceback_to_backtrace_str


def register_error(
    name_or_cls: str | type | None = None,
    cls: type | None = None,
) -> Any:
    """Register an error class so it can be recognized by the ffi error handler.

    Parameters
    ----------
    name_or_cls
        The name of the error class.

    cls
        The class to register.

    Returns
    -------
    fregister
        Register function if f is not specified.

    Examples
    --------
    .. code-block:: python

        import tvm_ffi


        # Register a custom Python exception so tvm_ffi.Error maps to it
        @tvm_ffi.error.register_error
        class MyError(RuntimeError):
            pass


        # Convert a Python exception to an FFI Error and back
        ffi_err = tvm_ffi.convert(MyError("boom"))
        py_err = ffi_err.py_error()
        assert isinstance(py_err, MyError)

    """
    if callable(name_or_cls):
        cls = name_or_cls
        name_or_cls = cls.__name__

    def register(mycls: type) -> type:
        """Register the error class name with the FFI core."""
        err_name = name_or_cls if isinstance(name_or_cls, str) else mycls.__name__
        core.ERROR_NAME_TO_TYPE[err_name] = mycls
        core.ERROR_TYPE_TO_NAME[mycls] = err_name
        return mycls

    if cls is None:
        return register
    return register(cls)


register_error("RuntimeError", RuntimeError)
register_error("ValueError", ValueError)
register_error("TypeError", TypeError)
register_error("AttributeError", AttributeError)
register_error("KeyError", KeyError)
register_error("IndexError", IndexError)
register_error("AssertionError", AssertionError)
register_error("MemoryError", MemoryError)
