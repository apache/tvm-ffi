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
"""Tests for tvm_ffi.pyast printer, ported from mlc-python's test_printer_ir_printer.py."""

from __future__ import annotations

import re

import pytest
from tvm_ffi import pyast
from tvm_ffi.access_path import AccessPath
from tvm_ffi.testing.testing import ToyAdd as Add
from tvm_ffi.testing.testing import ToyAssign as Assign
from tvm_ffi.testing.testing import ToyFunc as Func
from tvm_ffi.testing.testing import ToyStmt as Stmt
from tvm_ffi.testing.testing import ToyVar as Var


def test_var_print() -> None:
    a = Var(name="a")
    assert pyast.to_python(a) == "a"


def test_var_print_name_normalize() -> None:
    a = Var(name="a/0/b")
    assert pyast.to_python(a) == "a_0_b"
    assert pyast.to_python(a) == "a_0_b"


def test_add_print() -> None:
    a = Var(name="a")
    b = Var(name="b")
    c = Add(lhs=a, rhs=b)
    assert pyast.to_python(c) == "a + b"


def test_assign_print() -> None:
    a = Var(name="a")
    b = Var(name="b")
    c = Assign(lhs=a, rhs=b)
    assert pyast.to_python(c) == "a = b"


def test_func_print() -> None:
    a = Var(name="a")
    b = Var(name="b")
    c = Var(name="c")
    d = Var(name="d")
    e = Var(name="e")
    stmts: list[Stmt] = [
        Assign(lhs=d, rhs=Add(a, b)),
        Assign(lhs=e, rhs=Add(d, c)),
    ]
    f = Func(name="f", args=[a, b, c], stmts=stmts, ret=e)
    assert (
        pyast.to_python(f)
        == """
def f(a, b, c):
  d = a + b
  e = d + c
  return e
""".strip()
    )


def test_print_none() -> None:
    printer = pyast.IRPrinter()
    path = AccessPath.root()
    node = printer(None, path)
    assert node.to_python() == "None"


def test_print_int() -> None:
    printer = pyast.IRPrinter()
    path = AccessPath.root()
    node = printer(42, path)
    assert node.to_python() == "42"


def test_print_str() -> None:
    printer = pyast.IRPrinter()
    path = AccessPath.root()
    node = printer("hey", path)
    assert node.to_python() == '"hey"'


def test_print_bool() -> None:
    printer = pyast.IRPrinter()
    path = AccessPath.root()
    node = printer(True, path)
    assert node.to_python() == "True"


def test_duplicated_vars() -> None:
    a = Var(name="a")
    b = Var(name="a")
    f = Func(
        name="f",
        args=[a],
        stmts=[Assign(lhs=b, rhs=Add(a, a))],
        ret=b,
    )
    assert (
        pyast.to_python(f)
        == """
def f(a):
  a_1 = a + a
  return a_1
""".strip()
    )
    assert re.fullmatch(
        r"^def f\(a\):\n"
        r"  a_0x[0-9A-Fa-f]+ = a \+ a\n"
        r"  return a_0x[0-9A-Fa-f]+$",
        pyast.to_python(f, pyast.PrinterConfig(print_addr_on_dup_var=True)),
    )


@pytest.mark.parametrize(
    "path, expected",
    [
        (
            AccessPath.root().attr("args").array_item(0),
            """
def f(a, b):
      ^
  c = a + b
  return c
""",
        ),
        (
            AccessPath.root().attr("args").array_item(1),
            """
def f(a, b):
         ^
  c = a + b
  return c
""",
        ),
        (
            AccessPath.root().attr("stmts").array_item(0),
            """
def f(a, b):
  c = a + b
  ^^^^^^^^^
  return c
""",
        ),
        (
            AccessPath.root().attr("stmts").array_item(0),
            """
def f(a, b):
  c = a + b
  ^^^^^^^^^
  return c
""",
        ),
        (
            AccessPath.root().attr("stmts").array_item(0).attr("lhs"),
            """
def f(a, b):
  c = a + b
  ^
  return c
""",
        ),
        (
            AccessPath.root().attr("stmts").array_item(0).attr("rhs"),
            """
def f(a, b):
  c = a + b
      ^^^^^
  return c
""",
        ),
        (
            AccessPath.root().attr("stmts").array_item(0).attr("rhs").attr("lhs"),
            """
def f(a, b):
  c = a + b
      ^
  return c
""",
        ),
        (
            AccessPath.root().attr("stmts").array_item(0).attr("rhs").attr("rhs"),
            """
def f(a, b):
  c = a + b
          ^
  return c
""",
        ),
        (
            AccessPath.root().attr("ret"),
            """
def f(a, b):
  c = a + b
  return c
         ^
""",
        ),
    ],
)
def test_print_underscore(path: AccessPath, expected: str) -> None:
    a = Var(name="a")
    b = Var(name="b")
    c = Var(name="c")
    f = Func(
        name="f",
        args=[a, b],
        stmts=[
            Assign(lhs=c, rhs=Add(a, b)),
        ],
        ret=c,
    )
    actual = pyast.to_python(
        f,
        pyast.PrinterConfig(path_to_underline=[path]),
    )
    assert actual.strip() == expected.strip()
