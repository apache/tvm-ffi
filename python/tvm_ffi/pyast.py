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
"""Python-style AST node definitions, printer classes, and rendering utilities.

This module defines the abstract syntax tree (AST) used by the text printer to
represent Python-style source code. The hierarchy is:

* ``Node`` -- base class for all AST nodes, providing ``to_python()`` and
  ``print_python()`` methods as well as source-path tracking.
* ``Expr(Node)`` -- base class for expression nodes (literals, identifiers,
  attribute access, indexing, calls, operations, etc.). Supports Python
  operator overloading so that AST fragments can be composed with ``+``,
  ``-``, ``*``, comparison operators, and more.
* ``Stmt(Node)`` -- base class for statement nodes (assignments, control flow,
  function/class definitions, comments, etc.).

Concrete node types correspond closely to the Python AST: ``Literal``, ``Id``,
``Attr``, ``Index``, ``Call``, ``Operation``, ``Lambda``, ``Tuple``, ``List``,
``Dict``, ``Slice`` (expressions) and ``StmtBlock``, ``Assign``, ``If``,
``While``, ``For``, ``With``, ``ExprStmt``, ``Assert``, ``Return``,
``Function``, ``Class``, ``Comment``, ``DocString`` (statements).
"""

# ruff: noqa: D102
# tvm-ffi-stubgen(begin): import-section
# fmt: off
# isort: off
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from collections.abc import MutableMapping, MutableSequence
    from tvm_ffi import Object
    from tvm_ffi.access_path import AccessPath
    from typing import Any, Callable
# isort: on
# fmt: on
# tvm-ffi-stubgen(end)

import contextlib
from collections.abc import Generator, Sequence
from typing import Any, TypeVar

from tvm_ffi import Object
from tvm_ffi.access_path import AccessPath
from tvm_ffi.dataclasses import c_class


@c_class("ffi.pyast.PrinterConfig", init=False)
class PrinterConfig(Object):
    """Configuration for the Python-style text printer.

    Controls formatting behavior such as indentation, line numbering, and
    how free variables and duplicate variable names are handled.

    Attributes
    ----------
    def_free_var
        Whether to automatically define free variables that
        appear in the output. Default ``True``.
    indent_spaces
        Number of spaces per indentation level. Default ``2``.
    print_line_numbers
        If greater than zero, prefix each output line with
        its line number. ``0`` disables line numbers (default).
    num_context_lines
        Number of context lines to show around underlined
        regions when ``path_to_underline`` is set. ``-1`` means show all
        lines (default).
    print_addr_on_dup_var
        When ``True``, append an object address suffix
        to disambiguate variables that share the same name. Default
        ``False``.
    path_to_underline
        A list of ``AccessPath`` instances identifying
        sub-expressions to underline in the printed output. Default
        is an empty list.

    Examples
    --------
    .. code-block:: python

        from tvm_ffi import pyast

        cfg = ast.PrinterConfig(indent_spaces=4, print_line_numbers=1)
        node = ast.Id(name="x")
        print(node.to_python(cfg))

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.PrinterConfig
    # fmt: off
    def_free_var: bool
    indent_spaces: int
    print_line_numbers: int
    num_context_lines: int
    print_addr_on_dup_var: bool
    path_to_underline: MutableSequence[AccessPath]
    if TYPE_CHECKING:
        def __init__(self, _0: bool, _1: int, _2: int, _3: int, _4: bool, _5: MutableSequence[AccessPath], /) -> None: ...
        def __ffi_shallow_copy__(self, /) -> Object: ...
        @staticmethod
        def __c_ffi_init__(_0: bool, _1: int, _2: int, _3: int, _4: bool, _5: MutableSequence[AccessPath], /) -> Object: ...
    # fmt: on
    # tvm-ffi-stubgen(end)

    def __init__(
        self,
        def_free_var: bool = True,
        indent_spaces: int = 2,
        print_line_numbers: int = 0,
        num_context_lines: int = -1,
        print_addr_on_dup_var: bool = False,
        path_to_underline: list[AccessPath] | None = None,
    ) -> None:
        """Initialize a PrinterConfig.

        Parameters
        ----------
        def_free_var
            Whether to automatically define free variables. Default ``True``.
        indent_spaces
            Number of spaces per indentation level. Default ``2``.
        print_line_numbers
            If greater than zero, prefix each output line with its line
            number. Default ``0``.
        num_context_lines
            Number of context lines to show around underlined regions.
            ``-1`` means show all lines. Default ``-1``.
        print_addr_on_dup_var
            When ``True``, append an object address suffix to disambiguate
            variables that share the same name. Default ``False``.
        path_to_underline
            A list of ``AccessPath`` instances identifying sub-expressions
            to underline. Default ``None`` (empty list).

        """
        if path_to_underline is None:
            path_to_underline = []
        self.__ffi_init__(
            def_free_var,
            indent_spaces,
            print_line_numbers,
            num_context_lines,
            print_addr_on_dup_var,
            path_to_underline,
        )


@c_class("ffi.pyast.Node", init=False)
class Node(Object):
    """Base class for all text-printer AST nodes.

    Every AST node carries an optional list of ``source_paths`` that trace
    the node back to the original IR object it was derived from. The two
    main entry points for rendering are ``to_python()`` (returns a string)
    and ``print_python()`` (prints to stdout).

    Attributes
    ----------
    source_paths
        Access paths linking this node to the original IR
        objects it represents. Default is an empty list.

    Examples
    --------
    .. code-block:: python

        from tvm_ffi import pyast

        node = ast.Id(name="x")
        source = node.to_python()  # "x"
        node.print_python()  # prints "x" to stdout

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.Node
    # fmt: off
    source_paths: MutableSequence[AccessPath]
    lineno: int
    col_offset: int
    end_lineno: int
    end_col_offset: int
    if TYPE_CHECKING:
        def __ffi_shallow_copy__(self, /) -> Object: ...
        def _to_python(self, _1: PrinterConfig, /) -> str: ...
    # fmt: on
    # tvm-ffi-stubgen(end)

    def to_python(self, config: PrinterConfig | None = None) -> str:
        """Render this AST node as Python-style source code.

        Parameters
        ----------
        config
            Printer configuration. Uses default settings when ``None``.

        Returns
        -------
        source
            The rendered source code as a string.

        Examples
        --------
        .. code-block:: python

            from tvm_ffi import pyast

            node = ast.Id(name="my_var")
            node.to_python()  # "my_var"

        """
        if config is None:
            config = PrinterConfig()
        return self._to_python(config)

    def print_python(
        self,
        config: PrinterConfig | None = None,
        style: str | None = None,
    ) -> None:
        """Print this AST node as Python-style source code to stdout.

        Uses Pygments syntax highlighting when available.

        Parameters
        ----------
        config
            Printer configuration. Uses default settings when ``None``.
        style
            Pygments style name or one of ``"light"``, ``"dark"``,
            ``"ansi"``. Defaults to ``"light"`` in notebooks, ``"ansi"``
            in terminals.

        """
        from ._pyast_colored_print import cprint  # noqa: PLC0415

        cprint(self.to_python(config), style=style)

    def add_path(self, path: Any) -> Node:
        """Append a source path to this node and return the node itself.

        This allows chaining, e.g. ``node.add_path(p1).add_path(p2)``.

        Parameters
        ----------
        path
            The access path to append.

        Returns
        -------
        self
            ``self``, for fluent chaining.

        """
        self.source_paths.append(path)
        return self


@c_class("ffi.pyast.Expr", init=False)
class Expr(Node):
    """Base class for expression AST nodes.

    ``Expr`` extends ``Node`` with Python operator overloading and builder
    methods so that AST fragments can be composed using natural syntax:

    * **Arithmetic**: ``+``, ``-``, ``*``, ``/``, ``//``, ``%``, ``**``
    * **Bitwise**: ``&``, ``|``, ``^``, ``~``, ``<<``, ``>>``
    * **Comparison**: ``<``, ``<=``, ``>``, ``>=``, ``.eq()``, ``.ne()``
    * **Logical**: ``.logical_and()``, ``.logical_or()``
    * **Ternary**: ``.if_then_else(then, else_)``
    * **Access**: ``.attr("name")``, ``.index([...])``, ``[...]``
    * **Calls**: ``.call(*args)``, ``.call_kw(args, keys, values)``

    Examples
    --------
    .. code-block:: python

        from tvm_ffi import pyast

        x = ast.Id(name="x")
        y = ast.Id(name="y")
        expr = (x + y).attr("shape").call(ast.Literal(0))
        expr.print_python()  # (x + y).shape(0)

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.Expr
    # fmt: off
    source_paths: MutableSequence[AccessPath]
    if TYPE_CHECKING:
        def __ffi_shallow_copy__(self, /) -> Object: ...
    # fmt: on
    # tvm-ffi-stubgen(end)

    def __neg__(self) -> Expr:
        return Operation(OperationKind.USub, [self])

    def __invert__(self) -> Expr:
        return Operation(OperationKind.Invert, [self])

    def __add__(self, other: Expr) -> Expr:
        return Operation(OperationKind.Add, [self, other])

    def __sub__(self, other: Expr) -> Expr:
        return Operation(OperationKind.Sub, [self, other])

    def __mul__(self, other: Expr) -> Expr:
        return Operation(OperationKind.Mult, [self, other])

    def __truediv__(self, other: Expr) -> Expr:
        return Operation(OperationKind.Div, [self, other])

    def __floordiv__(self, other: Expr) -> Expr:
        return Operation(OperationKind.FloorDiv, [self, other])

    def __mod__(self, other: Expr) -> Expr:
        return Operation(OperationKind.Mod, [self, other])

    def __pow__(self, other: Expr) -> Expr:
        return Operation(OperationKind.Pow, [self, other])

    def __lshift__(self, other: Expr) -> Expr:
        return Operation(OperationKind.LShift, [self, other])

    def __rshift__(self, other: Expr) -> Expr:
        return Operation(OperationKind.RShift, [self, other])

    def __and__(self, other: Expr) -> Expr:
        return Operation(OperationKind.BitAnd, [self, other])

    def __or__(self, other: Expr) -> Expr:
        return Operation(OperationKind.BitOr, [self, other])

    def __xor__(self, other: Expr) -> Expr:
        return Operation(OperationKind.BitXor, [self, other])

    def __lt__(self, other: Expr) -> Expr:
        return Operation(OperationKind.Lt, [self, other])

    def __le__(self, other: Expr) -> Expr:
        return Operation(OperationKind.LtE, [self, other])

    def __gt__(self, other: Expr) -> Expr:
        return Operation(OperationKind.Gt, [self, other])

    def __ge__(self, other: Expr) -> Expr:
        return Operation(OperationKind.GtE, [self, other])

    def logical_and(self, other: Expr) -> Expr:
        """Build a logical ``and`` operation (``self and other``).

        Python's ``and`` operator cannot be overloaded, so this explicit
        method is provided instead.

        Parameters
        ----------
        other
            The right-hand operand.

        Returns
        -------
        result
            An ``Operation`` node representing ``self and other``.

        """
        return Operation(OperationKind.And, [self, other])

    def logical_or(self, other: Expr) -> Expr:
        """Build a logical ``or`` operation (``self or other``).

        Python's ``or`` operator cannot be overloaded, so this explicit
        method is provided instead.

        Parameters
        ----------
        other
            The right-hand operand.

        Returns
        -------
        result
            An ``Operation`` node representing ``self or other``.

        """
        return Operation(OperationKind.Or, [self, other])

    def if_then_else(self, then: Expr, else_: Expr) -> Expr:
        """Build a ternary conditional expression (``then if self else else_``).

        Parameters
        ----------
        then
            The value when the condition is true.
        else_
            The value when the condition is false.

        Returns
        -------
        result
            An ``Operation`` node representing the ternary expression.

        Examples
        --------
        .. code-block:: python

            from tvm_ffi import pyast

            cond = ast.Id(name="flag")
            result = cond.if_then_else(ast.Literal(1), ast.Literal(0))
            result.print_python()  # 1 if flag else 0

        """
        return Operation(OperationKind.IfThenElse, [self, then, else_])

    def eq(self, other: Expr) -> Expr:
        """Build an equality comparison (``self == other``).

        Python's ``__eq__`` is not overloaded to preserve standard object
        identity semantics, so this explicit method is provided instead.

        Parameters
        ----------
        other
            The right-hand operand.

        Returns
        -------
        result
            An ``Operation`` node representing ``self == other``.

        """
        return Operation(OperationKind.Eq, [self, other])

    def ne(self, other: Expr) -> Expr:
        """Build an inequality comparison (``self != other``).

        Python's ``__ne__`` is not overloaded to preserve standard object
        identity semantics, so this explicit method is provided instead.

        Parameters
        ----------
        other
            The right-hand operand.

        Returns
        -------
        result
            An ``Operation`` node representing ``self != other``.

        """
        return Operation(OperationKind.NotEq, [self, other])

    def attr(self, name: str) -> Expr:
        """Build an attribute access expression (``self.name``).

        Parameters
        ----------
        name
            The attribute name.

        Returns
        -------
        result
            An ``Attr`` node representing ``self.name``.

        Examples
        --------
        .. code-block:: python

            from tvm_ffi import pyast

            obj = ast.Id(name="module")
            obj.attr("forward").print_python()  # module.forward

        """
        return Attr(self, name)

    def index(self, indices: MutableSequence[Expr]) -> Expr:
        """Build a subscript/index expression (``self[indices]``).

        Parameters
        ----------
        indices
            A sequence of index expressions.

        Returns
        -------
        result
            An ``Index`` node representing ``self[indices]``.

        Examples
        --------
        .. code-block:: python

            from tvm_ffi import pyast

            arr = ast.Id(name="arr")
            arr.index([ast.Literal(0)]).print_python()  # arr[0]

        """
        return Index(self, indices)

    def call(self, *args: Expr) -> Expr:
        """Build a positional-only call expression (``self(args...)``).

        Parameters
        ----------
        *args
            Positional argument expressions.

        Returns
        -------
        result
            A ``Call`` node representing ``self(*args)``.

        Examples
        --------
        .. code-block:: python

            from tvm_ffi import pyast

            fn = ast.Id(name="relu")
            fn.call(ast.Id(name="x")).print_python()  # relu(x)

        """
        return Call(
            self,
            args,  # ty: ignore[invalid-argument-type]
            [],
            [],
        )

    def call_kw(
        self,
        args: Sequence[Expr],
        kwargs_keys: Sequence[str],
        kwargs_values: Sequence[Expr],
    ) -> Expr:
        """Build a call expression with keyword arguments.

        Renders as ``self(*args, key0=val0, key1=val1, ...)``.

        Parameters
        ----------
        args
            Positional argument expressions.
        kwargs_keys
            Keyword argument names.
        kwargs_values
            Keyword argument value expressions, in the same
            order as *kwargs_keys*.

        Returns
        -------
        result
            A ``Call`` node representing the keyword call.

        Examples
        --------
        .. code-block:: python

            from tvm_ffi import pyast

            fn = ast.Id(name="conv2d")
            fn.call_kw(
                args=[ast.Id(name="x")],
                kwargs_keys=["stride"],
                kwargs_values=[ast.Literal(2)],
            ).print_python()  # conv2d(x, stride=2)

        """
        if not isinstance(args, Sequence):
            args = (args,)
        if not isinstance(kwargs_keys, Sequence):
            kwargs_keys = (kwargs_keys,)
        if not isinstance(kwargs_values, Sequence):
            kwargs_values = (kwargs_values,)
        return Call(
            self,
            args,  # ty: ignore[invalid-argument-type]
            kwargs_keys,  # ty: ignore[invalid-argument-type]
            kwargs_values,  # ty: ignore[invalid-argument-type]
        )

    def __getitem__(self, indices: Expr | Sequence[Expr]) -> Expr:
        """Build a subscript expression via Python's ``[]`` syntax.

        Delegates to ``self.index()``, wrapping a single index in a tuple
        if necessary.

        Parameters
        ----------
        indices
            One or more index expressions.

        Returns
        -------
        result
            An ``Index`` node representing ``self[indices]``.

        """
        if isinstance(indices, Sequence):
            return self.index(indices)  # ty: ignore[invalid-argument-type]
        return self.index([indices])


@c_class("ffi.pyast.Stmt", init=False)
class Stmt(Node):
    """Base class for statement AST nodes.

    Statements represent executable constructs (assignments, loops,
    conditionals, etc.). Every statement may carry an optional trailing
    ``comment`` that is rendered as ``# comment`` after the statement.

    Attributes
    ----------
    comment
        An optional inline comment string. Default ``None``.

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.Stmt
    # fmt: off
    source_paths: MutableSequence[AccessPath]
    comment: str | None
    if TYPE_CHECKING:
        def __ffi_shallow_copy__(self, /) -> Object: ...
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("ffi.pyast.StmtBlock")
class StmtBlock(Stmt):
    """A sequence of statements rendered as a block.

    Represents a group of statements that are printed together, typically
    as the body of a function, class, loop, or conditional.

    Attributes
    ----------
    stmts
        The list of statements in this block.

    Examples
    --------
    .. code-block:: python

        from tvm_ffi import pyast

        block = ast.StmtBlock(stmts=[ast.ExprStmt(expr=ast.Id(name="x"))])
        block.print_python()  # x

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.StmtBlock
    # fmt: off
    stmts: MutableSequence[Stmt]
    if TYPE_CHECKING:
        def __init__(self, _0: MutableSequence[Stmt], /) -> None: ...
        def __ffi_shallow_copy__(self, /) -> Object: ...
        @staticmethod
        def __c_ffi_init__(_0: MutableSequence[Stmt], /) -> Object: ...
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("ffi.pyast.Literal")
class Literal(Expr):
    """A literal value expression (``42``, ``3.14``, ``"hello"``, ``True``, ``None``).

    Wraps an arbitrary Python value and renders it using its ``repr()``.

    Attributes
    ----------
    value
        The literal Python value.

    Examples
    --------
    .. code-block:: python

        from tvm_ffi import pyast

        ast.Literal(42).print_python()  # 42
        ast.Literal("hello").print_python()  # "hello"

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.Literal
    # fmt: off
    value: Any
    if TYPE_CHECKING:
        def __init__(self, _0: Any, /) -> None: ...
        def __ffi_shallow_copy__(self, /) -> Object: ...
        @staticmethod
        def __c_ffi_init__(_0: Any, /) -> Object: ...
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("ffi.pyast.Id")
class Id(Expr):
    """An identifier / variable name expression.

    Renders as the bare name string.

    Attributes
    ----------
    name
        The identifier string.

    Examples
    --------
    .. code-block:: python

        from tvm_ffi import pyast

        ast.Id(name="x").print_python()  # x

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.Id
    # fmt: off
    name: str
    if TYPE_CHECKING:
        def __init__(self, _0: str, /) -> None: ...
        def __ffi_shallow_copy__(self, /) -> Object: ...
        @staticmethod
        def __c_ffi_init__(_0: str, /) -> Object: ...
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("ffi.pyast.Attr")
class Attr(Expr):
    """An attribute access expression (``obj.name``).

    Attributes
    ----------
    obj
        The object expression being accessed.
    name
        The attribute name.

    Examples
    --------
    .. code-block:: python

        from tvm_ffi import pyast

        ast.Attr(obj=ast.Id(name="self"), name="weight").print_python()  # self.weight

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.Attr
    # fmt: off
    obj: Expr
    name: str
    if TYPE_CHECKING:
        def __init__(self, _0: Expr, _1: str, /) -> None: ...
        def __ffi_shallow_copy__(self, /) -> Object: ...
        @staticmethod
        def __c_ffi_init__(_0: Expr, _1: str, /) -> Object: ...
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("ffi.pyast.Index")
class Index(Expr):
    """A subscript / index expression (``obj[idx0, idx1, ...]``).

    Attributes
    ----------
    obj
        The object expression being indexed.
    idx
        A list of index expressions.

    Examples
    --------
    .. code-block:: python

        from tvm_ffi import pyast

        ast.Index(obj=ast.Id(name="x"), idx=[ast.Literal(0)]).print_python()  # x[0]

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.Index
    # fmt: off
    obj: Expr
    idx: MutableSequence[Expr]
    if TYPE_CHECKING:
        def __init__(self, _0: Expr, _1: MutableSequence[Expr], /) -> None: ...
        def __ffi_shallow_copy__(self, /) -> Object: ...
        @staticmethod
        def __c_ffi_init__(_0: Expr, _1: MutableSequence[Expr], /) -> Object: ...
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("ffi.pyast.Call")
class Call(Expr):
    """A function call expression (``callee(args..., key=val, ...)``).

    Attributes
    ----------
    callee
        The callable expression.
    args
        Positional argument expressions.
    kwargs_keys
        Keyword argument names.
    kwargs_values
        Keyword argument value expressions, aligned with
        *kwargs_keys*.

    Examples
    --------
    .. code-block:: python

        from tvm_ffi import pyast

        ast.Call(
            callee=ast.Id(name="f"),
            args=[ast.Literal(1)],
            kwargs_keys=["dim"],
            kwargs_values=[ast.Literal(0)],
        ).print_python()  # f(1, dim=0)

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.Call
    # fmt: off
    callee: Expr
    args: MutableSequence[Expr]
    kwargs_keys: MutableSequence[str]
    kwargs_values: MutableSequence[Expr]
    if TYPE_CHECKING:
        def __init__(self, _0: Expr, _1: MutableSequence[Expr], _2: MutableSequence[str], _3: MutableSequence[Expr], /) -> None: ...
        def __ffi_shallow_copy__(self, /) -> Object: ...
        @staticmethod
        def __c_ffi_init__(_0: Expr, _1: MutableSequence[Expr], _2: MutableSequence[str], _3: MutableSequence[Expr], /) -> Object: ...
    # fmt: on
    # tvm-ffi-stubgen(end)


class OperationKind:
    """Enum-like class defining operation kinds for ``Operation`` nodes.

    Integer constants are grouped into three ranges:

    * **Unary** (``_UnaryStart`` .. ``_UnaryEnd``): ``USub`` (``-x``),
      ``Invert`` (``~x``), ``Not`` (``not x``).
    * **Binary** (``_BinaryStart`` .. ``_BinaryEnd``): arithmetic
      (``Add``, ``Sub``, ``Mult``, ``Div``, ``FloorDiv``, ``Mod``,
      ``Pow``), bitwise (``LShift``, ``RShift``, ``BitAnd``, ``BitOr``,
      ``BitXor``), comparison (``Lt``, ``LtE``, ``Eq``, ``NotEq``,
      ``Gt``, ``GtE``), and logical (``And``, ``Or``).
    * **Special** (``_SpecialStart`` .. ``SpecialEnd``): ``IfThenElse``
      (ternary conditional ``a if cond else b``).

    These constants are used as the ``op`` field of ``Operation`` nodes.

    Examples
    --------
    .. code-block:: python

        from tvm_ffi import pyast

        x = ast.Id(name="x")
        y = ast.Id(name="y")
        add_op = ast.Operation(ast.OperationKind.Add, [x, y])
        add_op.print_python()  # x + y

    """

    Undefined = -1
    _UnaryStart = 0
    USub = 1
    Invert = 2
    Not = 3
    UAdd = 4
    _UnaryEnd = 5
    _BinaryStart = 5
    Add = 6
    Sub = 7
    Mult = 8
    Div = 9
    FloorDiv = 10
    Mod = 11
    Pow = 12
    LShift = 13
    RShift = 14
    BitAnd = 15
    BitOr = 16
    BitXor = 17
    Lt = 18
    LtE = 19
    Eq = 20
    NotEq = 21
    Gt = 22
    GtE = 23
    And = 24
    Or = 25
    MatMult = 26
    Is = 27
    IsNot = 28
    In = 29
    NotIn = 30
    _BinaryEnd = 31
    _SpecialStart = 32
    IfThenElse = 33
    ChainedCompare = 34
    Parens = 35
    SpecialEnd = 36


@c_class("ffi.pyast.Operation")
class Operation(Expr):
    """A unary, binary, or special operation expression.

    The ``op`` field is one of the integer constants defined in
    ``OperationKind``. The ``operands`` list contains one element for
    unary ops, two for binary ops, or three for ``IfThenElse``.

    Attributes
    ----------
    op
        The operation kind (an ``OperationKind`` constant).
    operands
        The operand expressions.

    Examples
    --------
    .. code-block:: python

        from tvm_ffi import pyast

        x = ast.Id(name="x")
        y = ast.Id(name="y")
        expr = ast.Operation(ast.OperationKind.Add, [x, y])
        expr.print_python()  # x + y

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.Operation
    # fmt: off
    op: int
    operands: MutableSequence[Expr]
    if TYPE_CHECKING:
        def __init__(self, _0: int, _1: MutableSequence[Expr], /) -> None: ...
        def __ffi_shallow_copy__(self, /) -> Object: ...
        @staticmethod
        def __c_ffi_init__(_0: int, _1: MutableSequence[Expr], /) -> Object: ...
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("ffi.pyast.Lambda")
class Lambda(Expr):
    """A lambda expression (``lambda args: body``).

    Attributes
    ----------
    args
        The parameter identifiers.
    body
        The body expression.

    Examples
    --------
    .. code-block:: python

        from tvm_ffi import pyast

        ast.Lambda(args=[ast.Id(name="x")], body=ast.Id(name="x")).print_python()
        # lambda x: x

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.Lambda
    # fmt: off
    args: MutableSequence[Expr]
    body: Expr
    if TYPE_CHECKING:
        def __init__(self, _0: MutableSequence[Expr], _1: Expr, /) -> None: ...
        def __ffi_shallow_copy__(self, /) -> Object: ...
        @staticmethod
        def __c_ffi_init__(_0: MutableSequence[Expr], _1: Expr, /) -> Object: ...
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("ffi.pyast.Tuple")
class Tuple(Expr):
    """A tuple expression (``(a, b, c)``).

    Attributes
    ----------
    values
        The element expressions.

    Examples
    --------
    .. code-block:: python

        from tvm_ffi import pyast

        ast.Tuple(values=[ast.Literal(1), ast.Literal(2)]).print_python()  # (1, 2)

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.Tuple
    # fmt: off
    values: MutableSequence[Expr]
    if TYPE_CHECKING:
        def __init__(self, _0: MutableSequence[Expr], /) -> None: ...
        def __ffi_shallow_copy__(self, /) -> Object: ...
        @staticmethod
        def __c_ffi_init__(_0: MutableSequence[Expr], /) -> Object: ...
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("ffi.pyast.List")
class List(Expr):
    """A list expression (``[a, b, c]``).

    Attributes
    ----------
    values
        The element expressions.

    Examples
    --------
    .. code-block:: python

        from tvm_ffi import pyast

        ast.List(values=[ast.Literal(1), ast.Literal(2)]).print_python()  # [1, 2]

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.List
    # fmt: off
    values: MutableSequence[Expr]
    if TYPE_CHECKING:
        def __init__(self, _0: MutableSequence[Expr], /) -> None: ...
        def __ffi_shallow_copy__(self, /) -> Object: ...
        @staticmethod
        def __c_ffi_init__(_0: MutableSequence[Expr], /) -> Object: ...
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("ffi.pyast.Dict")
class Dict(Expr):
    """A dictionary expression (``{k0: v0, k1: v1, ...}``).

    Attributes
    ----------
    keys
        The key expressions.
    values
        The value expressions, aligned with *keys*.

    Examples
    --------
    .. code-block:: python

        from tvm_ffi import pyast

        ast.Dict(
            keys=[ast.Literal("a")],
            values=[ast.Literal(1)],
        ).print_python()  # {"a": 1}

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.Dict
    # fmt: off
    keys: MutableSequence[Expr]
    values: MutableSequence[Expr]
    if TYPE_CHECKING:
        def __init__(self, _0: MutableSequence[Expr], _1: MutableSequence[Expr], /) -> None: ...
        def __ffi_shallow_copy__(self, /) -> Object: ...
        @staticmethod
        def __c_ffi_init__(_0: MutableSequence[Expr], _1: MutableSequence[Expr], /) -> Object: ...
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("ffi.pyast.Slice", init=False)
class Slice(Expr):
    """A slice expression (``start:stop:step``).

    All three components are optional. A ``None`` component is omitted
    from the rendered output.

    Attributes
    ----------
    start
        The start expression, or ``None``.
    stop
        The stop expression, or ``None``.
    step
        The step expression, or ``None``.

    Examples
    --------
    .. code-block:: python

        from tvm_ffi import pyast

        ast.Slice(start=ast.Literal(0), stop=ast.Literal(10)).print_python()  # 0:10

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.Slice
    # fmt: off
    start: Expr | None
    stop: Expr | None
    step: Expr | None
    if TYPE_CHECKING:
        def __init__(self, _0: Expr | None, _1: Expr | None, _2: Expr | None, /) -> None: ...
        def __ffi_shallow_copy__(self, /) -> Object: ...
        @staticmethod
        def __c_ffi_init__(_0: Expr | None, _1: Expr | None, _2: Expr | None, /) -> Object: ...
    # fmt: on
    # tvm-ffi-stubgen(end)

    def __init__(
        self,
        start: Expr | None = None,
        stop: Expr | None = None,
        step: Expr | None = None,
    ) -> None:
        """Initialize a Slice expression.

        Parameters
        ----------
        start
            The start expression, or ``None``. Default ``None``.
        stop
            The stop expression, or ``None``. Default ``None``.
        step
            The step expression, or ``None``. Default ``None``.

        """
        self.__ffi_init__(start, stop, step)


@c_class("ffi.pyast.Assign", init=False)
class Assign(Stmt):
    """An assignment statement (``lhs = rhs`` or ``lhs: annotation = rhs``).

    When ``rhs`` is ``None`` the statement renders as a bare declaration
    (``lhs: annotation``). When ``annotation`` is ``None`` the type
    annotation is omitted.

    Attributes
    ----------
    lhs
        The left-hand-side target expression.
    rhs
        The right-hand-side value expression, or ``None``.
    annotation
        An optional type annotation expression.

    Examples
    --------
    .. code-block:: python

        from tvm_ffi import pyast

        ast.Assign(lhs=ast.Id(name="x"), rhs=ast.Literal(42)).print_python()  # x = 42

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.Assign
    # fmt: off
    lhs: Expr
    rhs: Expr | None
    annotation: Expr | None
    aug_op: int
    if TYPE_CHECKING:
        def __init__(self, _0: Expr, _1: Expr | None, _2: Expr | None, _3: int, /) -> None: ...
        def __ffi_shallow_copy__(self, /) -> Object: ...
        @staticmethod
        def __c_ffi_init__(_0: Expr, _1: Expr | None, _2: Expr | None, _3: int, /) -> Object: ...
    # fmt: on
    # tvm-ffi-stubgen(end)

    def __init__(
        self,
        lhs: Expr,
        rhs: Expr | None = None,
        annotation: Expr | None = None,
        aug_op: int = -1,
    ) -> None:
        """Initialize an Assign statement.

        Parameters
        ----------
        lhs
            The left-hand-side target expression.
        rhs
            The right-hand-side value expression, or ``None``.
            Default ``None``.
        annotation
            An optional type annotation expression. Default ``None``.
        aug_op
            Augmented-assignment operator kind (``OperationKind`` value),
            or -1 for plain assignment. Default -1.

        """
        self.__ffi_init__(lhs, rhs, annotation, aug_op)


@c_class("ffi.pyast.If")
class If(Stmt):
    """An ``if / elif / else`` conditional statement.

    Attributes
    ----------
    cond
        The condition expression.
    then_branch
        Statements executed when the condition is true.
    else_branch
        Statements executed when the condition is false
        (may be empty).

    Examples
    --------
    .. code-block:: python

        from tvm_ffi import pyast

        ast.If(
            cond=ast.Id(name="flag"),
            then_branch=[ast.ExprStmt(expr=ast.Id(name="a"))],
            else_branch=[ast.ExprStmt(expr=ast.Id(name="b"))],
        ).print_python()
        # if flag:
        #   a
        # else:
        #   b

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.If
    # fmt: off
    cond: Expr
    then_branch: MutableSequence[Stmt]
    else_branch: MutableSequence[Stmt]
    if TYPE_CHECKING:
        def __init__(self, _0: Expr, _1: MutableSequence[Stmt], _2: MutableSequence[Stmt], /) -> None: ...
        def __ffi_shallow_copy__(self, /) -> Object: ...
        @staticmethod
        def __c_ffi_init__(_0: Expr, _1: MutableSequence[Stmt], _2: MutableSequence[Stmt], /) -> Object: ...
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("ffi.pyast.While", init=False)
class While(Stmt):
    """A ``while`` loop statement.

    Attributes
    ----------
    cond
        The loop condition expression.
    body
        The loop body statements.

    Examples
    --------
    .. code-block:: python

        from tvm_ffi import pyast

        ast.While(
            cond=ast.Id(name="running"),
            body=[ast.ExprStmt(expr=ast.Id(name="step"))],
        ).print_python()
        # while running:
        #   step

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.While
    # fmt: off
    cond: Expr
    body: MutableSequence[Stmt]
    orelse: MutableSequence[Stmt]
    if TYPE_CHECKING:
        def __init__(self, _0: Expr, _1: MutableSequence[Stmt], _2: MutableSequence[Stmt], /) -> None: ...
        def __ffi_shallow_copy__(self, /) -> Object: ...
        @staticmethod
        def __c_ffi_init__(_0: Expr, _1: MutableSequence[Stmt], _2: MutableSequence[Stmt], /) -> Object: ...
    # fmt: on
    # tvm-ffi-stubgen(end)

    def __init__(
        self,
        cond: Expr,
        body: MutableSequence[Stmt],
        orelse: MutableSequence[Stmt] | None = None,
    ) -> None:
        if orelse is None:
            orelse = []
        self.__ffi_init__(cond, body, orelse)


@c_class("ffi.pyast.For", init=False)
class For(Stmt):
    """A ``for`` loop statement (``for lhs in rhs: body``).

    Attributes
    ----------
    lhs
        The loop variable expression.
    rhs
        The iterable expression.
    body
        The loop body statements.

    Examples
    --------
    .. code-block:: python

        from tvm_ffi import pyast

        ast.For(
            lhs=ast.Id(name="i"),
            rhs=ast.Id(name="items"),
            body=[ast.ExprStmt(expr=ast.Id(name="process"))],
        ).print_python()
        # for i in items:
        #   process

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.For
    # fmt: off
    lhs: Expr
    rhs: Expr
    body: MutableSequence[Stmt]
    is_async: bool
    orelse: MutableSequence[Stmt]
    if TYPE_CHECKING:
        def __init__(self, _0: Expr, _1: Expr, _2: MutableSequence[Stmt], _3: bool, _4: MutableSequence[Stmt], /) -> None: ...
        def __ffi_shallow_copy__(self, /) -> Object: ...
        @staticmethod
        def __c_ffi_init__(_0: Expr, _1: Expr, _2: MutableSequence[Stmt], _3: bool, _4: MutableSequence[Stmt], /) -> Object: ...
    # fmt: on
    # tvm-ffi-stubgen(end)

    def __init__(
        self,
        lhs: Expr,
        rhs: Expr,
        body: MutableSequence[Stmt],
        is_async: bool = False,
        orelse: MutableSequence[Stmt] | None = None,
    ) -> None:
        if orelse is None:
            orelse = []
        self.__ffi_init__(lhs, rhs, body, is_async, orelse)


@c_class("ffi.pyast.With", init=False)
class With(Stmt):
    """A ``with`` context-manager statement (``with rhs as lhs: body``).

    When ``lhs`` is ``None``, the ``as lhs`` clause is omitted.

    Attributes
    ----------
    lhs
        The optional target expression (``as`` variable), or ``None``.
    rhs
        The context-manager expression.
    body
        The body statements.

    Examples
    --------
    .. code-block:: python

        from tvm_ffi import pyast

        ast.With(
            lhs=ast.Id(name="f"),
            rhs=ast.Id(name="open_file"),
            body=[ast.ExprStmt(expr=ast.Id(name="read"))],
        ).print_python()
        # with open_file as f:
        #   read

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.With
    # fmt: off
    lhs: Expr | None
    rhs: Expr
    body: MutableSequence[Stmt]
    is_async: bool
    if TYPE_CHECKING:
        def __init__(self, _0: Expr | None, _1: Expr, _2: MutableSequence[Stmt], _3: bool, /) -> None: ...
        def __ffi_shallow_copy__(self, /) -> Object: ...
        @staticmethod
        def __c_ffi_init__(_0: Expr | None, _1: Expr, _2: MutableSequence[Stmt], _3: bool, /) -> Object: ...
    # fmt: on
    # tvm-ffi-stubgen(end)

    def __init__(
        self,
        lhs: Expr | None,
        rhs: Expr,
        body: MutableSequence[Stmt],
        is_async: bool = False,
    ) -> None:
        self.__ffi_init__(lhs, rhs, body, is_async)


@c_class("ffi.pyast.ExprStmt")
class ExprStmt(Stmt):
    """An expression used as a statement (e.g. a bare function call).

    Attributes
    ----------
    expr
        The expression to evaluate as a statement.

    Examples
    --------
    .. code-block:: python

        from tvm_ffi import pyast

        ast.ExprStmt(expr=ast.Id(name="do_something")).print_python()  # do_something

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.ExprStmt
    # fmt: off
    expr: Expr
    if TYPE_CHECKING:
        def __init__(self, _0: Expr, /) -> None: ...
        def __ffi_shallow_copy__(self, /) -> Object: ...
        @staticmethod
        def __c_ffi_init__(_0: Expr, /) -> Object: ...
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("ffi.pyast.Assert", init=False)
class Assert(Stmt):
    """An ``assert`` statement (``assert cond, msg``).

    When ``msg`` is ``None``, only the condition is rendered.

    Attributes
    ----------
    cond
        The condition expression.
    msg
        An optional message expression, or ``None``.

    Examples
    --------
    .. code-block:: python

        from tvm_ffi import pyast

        ast.Assert(
            cond=ast.Id(name="x"), msg=ast.Literal("x must be set")
        ).print_python()
        # assert x, "x must be set"

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.Assert
    # fmt: off
    cond: Expr
    msg: Expr | None
    if TYPE_CHECKING:
        def __init__(self, _0: Expr, _1: Expr | None, /) -> None: ...
        def __ffi_shallow_copy__(self, /) -> Object: ...
        @staticmethod
        def __c_ffi_init__(_0: Expr, _1: Expr | None, /) -> Object: ...
    # fmt: on
    # tvm-ffi-stubgen(end)

    def __init__(
        self,
        cond: Expr,
        msg: Expr | None = None,
    ) -> None:
        """Initialize an Assert statement.

        Parameters
        ----------
        cond
            The condition expression.
        msg
            An optional message expression, or ``None``. Default ``None``.

        """
        self.__ffi_init__(cond, msg)


@c_class("ffi.pyast.Return")
class Return(Stmt):
    """A ``return`` statement.

    When ``value`` is ``None``, renders as a bare ``return``.

    Attributes
    ----------
    value
        The return value expression, or ``None``.

    Examples
    --------
    .. code-block:: python

        from tvm_ffi import pyast

        ast.Return(value=ast.Literal(42)).print_python()  # return 42

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.Return
    # fmt: off
    value: Expr | None
    if TYPE_CHECKING:
        def __init__(self, _0: Expr | None, /) -> None: ...
        def __ffi_shallow_copy__(self, /) -> Object: ...
        @staticmethod
        def __c_ffi_init__(_0: Expr | None, /) -> Object: ...
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("ffi.pyast.Function", init=False)
class Function(Stmt):
    """A ``def`` function definition statement.

    Attributes
    ----------
    name
        The function name identifier.
    args
        The parameter list, each represented as an ``Assign`` node
        (the ``lhs`` is the parameter name; ``annotation`` and ``rhs``
        provide type hints and default values).
    decorators
        Decorator expressions applied above the function.
    return_type
        An optional return-type annotation expression.
    body
        The function body statements.

    Examples
    --------
    .. code-block:: python

        from tvm_ffi import pyast

        ast.Function(
            name=ast.Id(name="add"),
            args=[
                ast.Assign(lhs=ast.Id(name="a")),
                ast.Assign(lhs=ast.Id(name="b")),
            ],
            decorators=[],
            return_type=None,
            body=[ast.Return(value=ast.Id(name="a") + ast.Id(name="b"))],
        ).print_python()
        # def add(a, b):
        #   return a + b

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.Function
    # fmt: off
    name: Id
    args: MutableSequence[Assign]
    decorators: MutableSequence[Expr]
    return_type: Expr | None
    body: MutableSequence[Stmt]
    is_async: bool
    if TYPE_CHECKING:
        def __init__(self, _0: Id, _1: MutableSequence[Assign], _2: MutableSequence[Expr], _3: Expr | None, _4: MutableSequence[Stmt], _5: bool, /) -> None: ...
        def __ffi_shallow_copy__(self, /) -> Object: ...
        @staticmethod
        def __c_ffi_init__(_0: Id, _1: MutableSequence[Assign], _2: MutableSequence[Expr], _3: Expr | None, _4: MutableSequence[Stmt], _5: bool, /) -> Object: ...
    # fmt: on
    # tvm-ffi-stubgen(end)

    def __init__(
        self,
        name: Id,
        args: MutableSequence[Assign],
        decorators: MutableSequence[Expr],
        return_type: Expr | None,
        body: MutableSequence[Stmt],
        is_async: bool = False,
    ) -> None:
        self.__ffi_init__(name, args, decorators, return_type, body, is_async)


@c_class("ffi.pyast.Class", init=False)
class Class(Stmt):
    """A ``class`` definition statement.

    Attributes
    ----------
    name
        The class name identifier.
    decorators
        Decorator expressions applied above the class.
    body
        The class body statements.

    Examples
    --------
    .. code-block:: python

        from tvm_ffi import pyast

        ast.Class(
            name=ast.Id(name="MyClass"),
            decorators=[],
            body=[ast.ExprStmt(expr=ast.Id(name="pass"))],
        ).print_python()
        # class MyClass:
        #   pass

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.Class
    # fmt: off
    name: Id
    bases: MutableSequence[Expr]
    decorators: MutableSequence[Expr]
    body: MutableSequence[Stmt]
    kwargs_keys: MutableSequence[str]
    kwargs_values: MutableSequence[Expr]
    if TYPE_CHECKING:
        def __init__(self, _0: Id, _1: MutableSequence[Expr], _2: MutableSequence[Expr], _3: MutableSequence[Stmt], _4: MutableSequence[str], _5: MutableSequence[Expr], /) -> None: ...
        def __ffi_shallow_copy__(self, /) -> Object: ...
        @staticmethod
        def __c_ffi_init__(_0: Id, _1: MutableSequence[Expr], _2: MutableSequence[Expr], _3: MutableSequence[Stmt], _4: MutableSequence[str], _5: MutableSequence[Expr], /) -> Object: ...
    # fmt: on
    # tvm-ffi-stubgen(end)

    def __init__(
        self,
        name: Id,
        bases: MutableSequence[Expr] | None = None,
        decorators: MutableSequence[Expr] | None = None,
        body: MutableSequence[Stmt] | None = None,
        kwargs_keys: MutableSequence[str] | None = None,
        kwargs_values: MutableSequence[Expr] | None = None,
    ) -> None:
        if bases is None:
            bases = []
        if decorators is None:
            decorators = []
        if body is None:
            body = []
        if kwargs_keys is None:
            kwargs_keys = []
        if kwargs_values is None:
            kwargs_values = []
        self.__ffi_init__(name, bases, decorators, body, kwargs_keys, kwargs_values)


@c_class("ffi.pyast.Comment", init=False)
class Comment(Stmt):
    """A standalone ``# comment`` line.

    The ``comment`` field (inherited from ``Stmt``) holds the comment text.
    It is rendered as a full-line comment rather than an inline comment.

    Examples
    --------
    .. code-block:: python

        from tvm_ffi import pyast

        ast.Comment("TODO: refactor this").print_python()  # # TODO: refactor this

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.Comment
    # fmt: off
    if TYPE_CHECKING:
        def __init__(self, _0: str | None, /) -> None: ...
        def __ffi_shallow_copy__(self, /) -> Object: ...
        @staticmethod
        def __c_ffi_init__(_0: str | None, /) -> Object: ...
    # fmt: on
    # tvm-ffi-stubgen(end)

    def __init__(self, comment: str | None) -> None:
        """Initialize a Comment statement.

        Parameters
        ----------
        comment
            The comment text, or ``None``.

        """
        self.__ffi_init__(comment)


@c_class("ffi.pyast.DocString", init=False)
class DocString(Stmt):
    r"""A triple-quoted docstring statement.

    Renders as a ``\"\"\"...\"\"\"``. The ``comment`` field (inherited from
    ``Stmt``) holds the docstring text.

    Examples
    --------
    .. code-block:: python

        from tvm_ffi import pyast

        ast.DocString("This is a docstring.").print_python()

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.DocString
    # fmt: off
    if TYPE_CHECKING:
        def __init__(self, _0: str | None, /) -> None: ...
        def __ffi_shallow_copy__(self, /) -> Object: ...
        @staticmethod
        def __c_ffi_init__(_0: str | None, /) -> Object: ...
    # fmt: on
    # tvm-ffi-stubgen(end)

    def __init__(self, comment: str | None) -> None:
        """Initialize a DocString statement.

        Parameters
        ----------
        comment
            The docstring text, or ``None``.

        """
        self.__ffi_init__(comment)


@c_class("ffi.pyast.Set")
class Set(Expr):
    """A set expression (``{a, b, c}``).

    Attributes
    ----------
    values
        The element expressions.

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.Set
    # fmt: off
    values: MutableSequence[Expr]
    if TYPE_CHECKING:
        def __init__(self, _0: MutableSequence[Expr], /) -> None: ...
        def __ffi_shallow_copy__(self, /) -> Object: ...
        @staticmethod
        def __c_ffi_init__(_0: MutableSequence[Expr], /) -> Object: ...
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("ffi.pyast.ComprehensionIter")
class ComprehensionIter(Node):
    """One ``for target in iter [if cond]...`` clause in a comprehension.

    Attributes
    ----------
    target
        The loop variable expression.
    iter
        The iterable expression.
    ifs
        Zero or more filter-condition expressions.

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.ComprehensionIter
    # fmt: off
    target: Expr
    iter: Expr
    ifs: MutableSequence[Expr]
    if TYPE_CHECKING:
        def __init__(self, _0: Expr, _1: Expr, _2: MutableSequence[Expr], /) -> None: ...
        def __ffi_shallow_copy__(self, /) -> Object: ...
        @staticmethod
        def __c_ffi_init__(_0: Expr, _1: Expr, _2: MutableSequence[Expr], /) -> Object: ...
    # fmt: on
    # tvm-ffi-stubgen(end)


class ComprehensionKind:
    """Enum-like class for comprehension kinds."""

    List = 0
    Set = 1
    Dict = 2
    Generator = 3


@c_class("ffi.pyast.Comprehension")
class Comprehension(Expr):
    """A comprehension expression.

    Covers list comprehensions (``[elt for ...]``), set comprehensions
    (``{elt for ...}``), dict comprehensions (``{key: value for ...}``),
    and generator expressions (``(elt for ...)``).

    Attributes
    ----------
    kind
        The comprehension kind (a ``ComprehensionKind`` constant).
    elt
        The element expression (or key for dict comprehensions).
    value
        The value expression (only for dict comprehensions; ``None`` otherwise).
    iters
        The list of ``ComprehensionIter`` clauses.

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.Comprehension
    # fmt: off
    kind: int
    elt: Expr
    value: Expr | None
    iters: MutableSequence[ComprehensionIter]
    if TYPE_CHECKING:
        def __init__(self, _0: int, _1: Expr, _2: Expr | None, _3: MutableSequence[ComprehensionIter], /) -> None: ...
        def __ffi_shallow_copy__(self, /) -> Object: ...
        @staticmethod
        def __c_ffi_init__(_0: int, _1: Expr, _2: Expr | None, _3: MutableSequence[ComprehensionIter], /) -> Object: ...
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("ffi.pyast.Yield", init=False)
class Yield(Expr):
    """A yield expression (``yield value``).

    Attributes
    ----------
    value
        The yielded value, or ``None`` for bare ``yield``.

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.Yield
    # fmt: off
    value: Expr | None
    if TYPE_CHECKING:
        def __init__(self, _0: Expr | None, /) -> None: ...
        def __ffi_shallow_copy__(self, /) -> Object: ...
        @staticmethod
        def __c_ffi_init__(_0: Expr | None, /) -> Object: ...
    # fmt: on
    # tvm-ffi-stubgen(end)

    def __init__(self, value: Expr | None = None) -> None:
        """Initialize a Yield expression."""
        self.__ffi_init__(value)


@c_class("ffi.pyast.YieldFrom")
class YieldFrom(Expr):
    """A yield-from expression (``yield from iterable``).

    Attributes
    ----------
    value
        The iterable to yield from.

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.YieldFrom
    # fmt: off
    value: Expr
    if TYPE_CHECKING:
        def __init__(self, _0: Expr, /) -> None: ...
        def __ffi_shallow_copy__(self, /) -> Object: ...
        @staticmethod
        def __c_ffi_init__(_0: Expr, /) -> Object: ...
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("ffi.pyast.StarredExpr")
class StarredExpr(Expr):
    """A starred expression (``*value``)."""

    # tvm-ffi-stubgen(begin): object/ffi.pyast.StarredExpr
    # fmt: off
    value: Expr
    if TYPE_CHECKING:
        def __init__(self, _0: Expr, /) -> None: ...
        def __ffi_shallow_copy__(self, /) -> Object: ...
        @staticmethod
        def __c_ffi_init__(_0: Expr, /) -> Object: ...
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("ffi.pyast.Await")
class AwaitExpr(Expr):
    """An await expression (``await value``)."""

    # tvm-ffi-stubgen(begin): object/ffi.pyast.Await
    # fmt: off
    value: Expr
    if TYPE_CHECKING:
        def __init__(self, _0: Expr, /) -> None: ...
        def __ffi_shallow_copy__(self, /) -> Object: ...
        @staticmethod
        def __c_ffi_init__(_0: Expr, /) -> Object: ...
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("ffi.pyast.WalrusExpr")
class WalrusExpr(Expr):
    """A walrus / named expression (``target := value``)."""

    # tvm-ffi-stubgen(begin): object/ffi.pyast.WalrusExpr
    # fmt: off
    target: Expr
    value: Expr
    if TYPE_CHECKING:
        def __init__(self, _0: Expr, _1: Expr, /) -> None: ...
        def __ffi_shallow_copy__(self, /) -> Object: ...
        @staticmethod
        def __c_ffi_init__(_0: Expr, _1: Expr, /) -> Object: ...
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("ffi.pyast.FStr")
class FStr(Expr):
    """An f-string expression (``f"...{x}..."``).

    ``values`` is a list of ``Literal(str)`` for text parts and
    ``FStrValue`` for interpolated expressions.
    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.FStr
    # fmt: off
    values: MutableSequence[Expr]
    if TYPE_CHECKING:
        def __init__(self, _0: MutableSequence[Expr], /) -> None: ...
        def __ffi_shallow_copy__(self, /) -> Object: ...
        @staticmethod
        def __c_ffi_init__(_0: MutableSequence[Expr], /) -> Object: ...
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("ffi.pyast.FStrValue", init=False)
class FStrValue(Expr):
    """A formatted value inside an f-string (``{value!r:.2f}``)."""

    # tvm-ffi-stubgen(begin): object/ffi.pyast.FStrValue
    # fmt: off
    value: Expr
    conversion: int
    format_spec: Expr | None
    if TYPE_CHECKING:
        def __init__(self, _0: Expr, _1: int, _2: Expr | None, /) -> None: ...
        def __ffi_shallow_copy__(self, /) -> Object: ...
        @staticmethod
        def __c_ffi_init__(_0: Expr, _1: int, _2: Expr | None, /) -> Object: ...
    # fmt: on
    # tvm-ffi-stubgen(end)

    def __init__(
        self,
        value: Expr,
        conversion: int = -1,
        format_spec: Expr | None = None,
    ) -> None:
        self.__ffi_init__(value, conversion, format_spec)


@c_class("ffi.pyast.ExceptHandler")
class ExceptHandler(Node):
    """One ``except [Type [as name]]:`` clause in a try statement."""

    # tvm-ffi-stubgen(begin): object/ffi.pyast.ExceptHandler
    # fmt: off
    type: Expr | None
    name: str | None
    body: MutableSequence[Stmt]
    if TYPE_CHECKING:
        def __init__(self, _0: Expr | None, _1: str | None, _2: MutableSequence[Stmt], /) -> None: ...
        def __ffi_shallow_copy__(self, /) -> Object: ...
        @staticmethod
        def __c_ffi_init__(_0: Expr | None, _1: str | None, _2: MutableSequence[Stmt], /) -> Object: ...
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("ffi.pyast.Try", init=False)
class Try(Stmt):
    """A ``try / except / else / finally`` statement."""

    body: MutableSequence[Stmt]
    handlers: MutableSequence[ExceptHandler]
    orelse: MutableSequence[Stmt]
    finalbody: MutableSequence[Stmt]

    def __init__(
        self,
        body: MutableSequence[Stmt],
        handlers: MutableSequence[ExceptHandler],
        orelse: MutableSequence[Stmt] | None = None,
        finalbody: MutableSequence[Stmt] | None = None,
    ) -> None:
        if orelse is None:
            orelse = []
        if finalbody is None:
            finalbody = []
        self.__ffi_init__(body, handlers, orelse, finalbody)


@c_class("ffi.pyast.MatchCase")
class MatchCase(Node):
    """One ``case pattern [if guard]:`` clause in a match statement."""

    # tvm-ffi-stubgen(begin): object/ffi.pyast.MatchCase
    # fmt: off
    pattern: Expr
    guard: Expr | None
    body: MutableSequence[Stmt]
    if TYPE_CHECKING:
        def __init__(self, _0: Expr, _1: Expr | None, _2: MutableSequence[Stmt], /) -> None: ...
        def __ffi_shallow_copy__(self, /) -> Object: ...
        @staticmethod
        def __c_ffi_init__(_0: Expr, _1: Expr | None, _2: MutableSequence[Stmt], /) -> Object: ...
    # fmt: on
    # tvm-ffi-stubgen(end)


@c_class("ffi.pyast.Match")
class Match(Stmt):
    """A ``match / case`` statement."""

    # tvm-ffi-stubgen(begin): object/ffi.pyast.Match
    # fmt: off
    subject: Expr
    cases: MutableSequence[MatchCase]
    if TYPE_CHECKING:
        def __init__(self, _0: Expr, _1: MutableSequence[MatchCase], /) -> None: ...
        def __ffi_shallow_copy__(self, /) -> Object: ...
        @staticmethod
        def __c_ffi_init__(_0: Expr, _1: MutableSequence[MatchCase], /) -> Object: ...
    # fmt: on
    # tvm-ffi-stubgen(end)


def from_py(source: Any) -> Node:
    """Convert a Python source string or ``ast.AST`` node to a TVM-FFI text AST node.

    Parameters
    ----------
    source : str | ast.AST
        Either a Python source string or a Python standard-library ``ast``
        node to convert.

    Returns
    -------
    Node
        The corresponding TVM-FFI text AST node.

    Examples
    --------
    .. code-block:: python

        from tvm_ffi import pyast

        node = ast.from_py("x + 1")
        node.print_python()  # x + 1

    """
    from ._pyast_translator import ast_translate  # noqa: PLC0415

    return ast_translate(source)


@c_class("ffi.pyast.VarInfo")
class VarInfo(Object):
    """Metadata for a variable tracked by ``IRPrinter``.

    Attributes
    ----------
    name
        The display name assigned to the variable, or ``None`` if
        a name has not yet been chosen (see ``var_def_no_name``).
    creator
        A ``Function`` callable that, when invoked by the printer,
        produces the definition site AST for this variable.

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.VarInfo
    # fmt: off
    name: str | None
    creator: Callable[..., Any]
    if TYPE_CHECKING:
        def __init__(self, _0: str | None, _1: Callable[..., Any], /) -> None: ...
        def __ffi_shallow_copy__(self, /) -> Object: ...
        @staticmethod
        def __c_ffi_init__(_0: str | None, _1: Callable[..., Any], /) -> Object: ...
    # fmt: on
    # tvm-ffi-stubgen(end)


FrameType = TypeVar("FrameType", bound=Object)


@c_class("ffi.pyast.DefaultFrame", init=False)
class DefaultFrame(Object):
    """The default scoping frame used by ``IRPrinter``.

    A frame collects statements emitted while it is active on the printer's
    frame stack. ``DefaultFrame`` is the simplest frame type and simply
    holds a mutable list of ``Stmt`` nodes.

    Attributes
    ----------
    stmts
        The list of statements accumulated in this frame.

    Examples
    --------
    .. code-block:: python

        printer = IRPrinter()
        with printer.with_frame(DefaultFrame()) as frame:
            # ... emit statements ...
            pass
        print(frame.stmts)

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.DefaultFrame
    # fmt: off
    stmts: MutableSequence[Stmt]
    if TYPE_CHECKING:
        def __init__(self, _0: MutableSequence[Stmt], /) -> None: ...
        def __ffi_shallow_copy__(self, /) -> Object: ...
        @staticmethod
        def __c_ffi_init__(_0: MutableSequence[Stmt], /) -> Object: ...
    # fmt: on
    # tvm-ffi-stubgen(end)

    def __init__(self, stmts: list[Stmt] | None = None) -> None:
        if stmts is None:
            stmts = []
        self.__ffi_init__(stmts)


@c_class("ffi.pyast.IRPrinter", init=False)
class IRPrinter(Object):
    """Stateful printer that converts TVM FFI objects into text-printer AST nodes.

    ``IRPrinter`` manages variable bindings and a stack of scoping frames.
    When called on an object, it dispatches to the object's registered
    printer handler to produce AST nodes, automatically defining and
    referencing variables as needed.

    Attributes
    ----------
    cfg
        The ``PrinterConfig`` controlling output formatting.
    obj2info
        Mapping from IR objects to their ``VarInfo`` metadata.
    defined_names
        Mapping from variable name strings to usage counts
        (used for de-duplication).
    frames
        The current stack of scoping frames.
    frame_vars
        Mapping from frame objects to the set of variables
        defined within that frame.

    Examples
    --------
    .. code-block:: python

        from tvm_ffi.pyast import IRPrinter
        from tvm_ffi.pyast import PrinterConfig
        from tvm_ffi.access_path import AccessPath

        printer = IRPrinter(PrinterConfig(indent_spaces=4))
        node = printer(my_obj, AccessPath.root())
        print(node.to_python())

    """

    # tvm-ffi-stubgen(begin): object/ffi.pyast.IRPrinter
    # fmt: off
    cfg: PrinterConfig
    obj2info: MutableMapping[Any, VarInfo]
    defined_names: MutableMapping[str, int]
    frames: MutableSequence[Any]
    frame_vars: MutableMapping[Any, Any]
    if TYPE_CHECKING:
        def __init__(self, _0: PrinterConfig, _1: MutableMapping[Any, VarInfo], _2: MutableMapping[str, int], _3: MutableSequence[Any], _4: MutableMapping[Any, Any], /) -> None: ...
        def __ffi_shallow_copy__(self, /) -> Object: ...
        @staticmethod
        def __c_ffi_init__(_0: PrinterConfig, _1: MutableMapping[Any, VarInfo], _2: MutableMapping[str, int], _3: MutableSequence[Any], _4: MutableMapping[Any, Any], /) -> Object: ...
        def var_is_defined(self, _1: Object, /) -> bool: ...
        def var_def(self, _1: str, _2: Object, _3: Object | None, /) -> Id: ...
        def var_def_no_name(self, _1: Callable[..., Any], _2: Object, _3: Object | None, /) -> None: ...
        def var_remove(self, _1: Object, /) -> None: ...
        def var_get(self, _1: Object, /) -> Expr | None: ...
        def frame_push(self, _1: Object, /) -> None: ...
        def frame_pop(self, /) -> None: ...
        def __call__(self, _1: Any, _2: AccessPath, /) -> Any: ...
    # fmt: on
    # tvm-ffi-stubgen(end)

    def __init__(self, cfg: PrinterConfig | None = None) -> None:
        if cfg is None:
            cfg = PrinterConfig()
        self.__ffi_init__(cfg, {}, {}, [], {})

    def __call__(self, obj: Any, path: AccessPath) -> Any:
        """Convert *obj* to a text format AST node using this printer's state.

        Parameters
        ----------
        obj
            The TVM FFI object to convert.
        path
            The access path describing how *obj* was reached.

        Returns
        -------
        Any
            The resulting AST node.

        Examples
        --------
        .. code-block:: python

            from tvm_ffi.access_path import AccessPath

            printer = IRPrinter()
            node = printer(my_obj, AccessPath.root())

        """
        info = type(self).__tvm_ffi_type_info__  # type: ignore[attr-defined]
        call_fn = next(m.func for m in info.methods if m.name == "__call__")
        return call_fn(self, obj, path)

    @contextlib.contextmanager
    def with_frame(self, frame: FrameType) -> Generator[FrameType, None, None]:
        """Context manager that pushes *frame* and pops it on exit.

        Any variables defined while the frame is active are associated with
        it and cleaned up when the frame is popped.

        Parameters
        ----------
        frame
            The frame object to activate.

        Yields
        ------
        FrameType
            The same *frame* object, for convenience.

        Examples
        --------
        .. code-block:: python

            printer = IRPrinter()
            with printer.with_frame(DefaultFrame()) as f:
                # statements emitted here go into f.stmts
                pass

        """
        self.frame_push(frame)
        try:
            yield frame
        finally:
            self.frame_pop()


def to_python(obj: Any, cfg: PrinterConfig | None = None) -> str:
    """Convert any TVM FFI object to Python-style source code."""
    if cfg is None:
        cfg = PrinterConfig()
    printer = IRPrinter(cfg)
    with printer.with_frame(DefaultFrame()) as frame:
        ret = printer(obj, AccessPath.root())
    if not frame.stmts:
        return ret.to_python(cfg)
    if isinstance(ret, StmtBlock):
        frame.stmts.extend(ret.stmts)
    elif isinstance(ret, Expr):
        frame.stmts.append(ExprStmt(ret))
    elif isinstance(ret, Stmt):
        frame.stmts.append(ret)
    return StmtBlock(frame.stmts).to_python(cfg)
