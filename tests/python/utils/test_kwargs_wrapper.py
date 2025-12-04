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

import inspect
from typing import Any

import pytest
from tvm_ffi.utils.kwargs_wrapper import make_kwargs_wrapper, make_kwargs_wrapper_from_signature


def test_basic_wrapper() -> None:
    """Test basic wrapper functionality with various argument combinations."""

    def target(*args: Any) -> int:
        return sum(args)

    # No defaults - all required
    wrapper = make_kwargs_wrapper(target, ["a", "b", "c"])
    assert wrapper(1, 2, 3) == 6
    assert wrapper(a=1, b=2, c=3) == 6
    assert wrapper(1, b=2, c=3) == 6

    # Single default argument
    wrapper = make_kwargs_wrapper(target, ["a", "b", "c"], arg_defaults=(10,))
    assert wrapper(1, 2) == 13  # c=10
    assert wrapper(1, 2, 3) == 6  # c=3 explicit
    assert wrapper(1, 2, c=5) == 8  # c=5 via keyword

    # Multiple defaults (right-aligned)
    wrapper = make_kwargs_wrapper(target, ["a", "b", "c"], arg_defaults=(20, 30))
    assert wrapper(1) == 51  # b=20, c=30
    assert wrapper(1, 2) == 33  # b=2, c=30
    assert wrapper(1, 2, 3) == 6  # all explicit

    # All defaults
    wrapper = make_kwargs_wrapper(target, ["a", "b", "c"], arg_defaults=(1, 2, 3))
    assert wrapper() == 6
    assert wrapper(10) == 15
    assert wrapper(10, 20, 30) == 60

    # Bound methods
    class Calculator:
        def __init__(self, base: int) -> None:
            self.base = base

        def add(self, a: int, b: int) -> int:
            return self.base + a + b

    calc = Calculator(100)
    wrapper = make_kwargs_wrapper(calc.add, ["a", "b"], arg_defaults=(5,))
    assert wrapper(1) == 106


def test_keyword_only_arguments() -> None:
    """Test wrapper with keyword-only arguments."""

    def target(*args: Any) -> int:
        return sum(args)

    # Optional keyword-only arguments (with defaults)
    wrapper = make_kwargs_wrapper(
        target,
        ["a", "b"],
        arg_defaults=(),
        kwonly_names=["c", "d"],
        kwonly_defaults={"c": 100, "d": 200},
    )
    assert wrapper(1, 2) == 303  # c=100, d=200
    assert wrapper(1, 2, c=10) == 213  # d=200
    assert wrapper(1, 2, c=10, d=20) == 33

    wrapper = make_kwargs_wrapper(
        target, ["a", "b"], arg_defaults=(), kwonly_names=["c", "d"], kwonly_defaults={}
    )
    assert wrapper(1, 2, c=10, d=20) == 33  # c and d are required

    wrapper = make_kwargs_wrapper(
        target,
        ["a", "b"],
        arg_defaults=(),
        kwonly_names=["c", "d"],
        kwonly_defaults={"d": 100},
    )
    assert wrapper(1, 2, c=10) == 113  # c required, d=100
    assert wrapper(1, 2, c=10, d=20) == 33  # both explicit

    wrapper = make_kwargs_wrapper(
        target,
        ["a", "b", "c"],
        arg_defaults=(10,),
        kwonly_names=["d", "e"],
        kwonly_defaults={"d": 20, "e": 30},
    )
    assert wrapper(1, 2) == 63  # c=10, d=20, e=30
    assert wrapper(1, 2, 5, d=15) == 53  # c=5 explicit, e=30


def test_validation_errors() -> None:
    """Test input validation and error handling."""
    target = lambda *args: sum(args)

    # Duplicate positional argument names
    with pytest.raises(ValueError, match="Duplicate argument names found"):
        make_kwargs_wrapper(target, ["a", "b", "a"])

    # Duplicate keyword-only argument names
    with pytest.raises(ValueError, match="Duplicate keyword-only argument names found"):
        make_kwargs_wrapper(target, ["a"], kwonly_names=["b", "c", "b"])

    # Invalid argument name types
    with pytest.raises(TypeError, match="Argument name must be a string"):
        make_kwargs_wrapper(target, ["a", 123])  # type: ignore[list-item]

    # Invalid Python identifiers
    with pytest.raises(ValueError, match="not a valid Python identifier"):
        make_kwargs_wrapper(target, ["a", "b-c"])

    # Python keywords cannot be used as parameter names
    with pytest.raises(
        ValueError, match="is a Python keyword and cannot be used as a parameter name"
    ):
        make_kwargs_wrapper(target, ["a", "if"])

    # arg_defaults not a tuple
    with pytest.raises(TypeError, match="arg_defaults must be a tuple"):
        make_kwargs_wrapper(target, ["a", "b"], arg_defaults=[10])  # type: ignore[arg-type]

    # arg_defaults too long
    with pytest.raises(ValueError, match=r"arg_defaults has .* values but only"):
        make_kwargs_wrapper(target, ["a"], arg_defaults=(1, 2, 3))

    # Overlap between positional and keyword-only
    with pytest.raises(ValueError, match="cannot be both positional and keyword-only"):
        make_kwargs_wrapper(target, ["a", "b"], kwonly_names=["b"])

    # kwonly_defaults key not in kwonly_names
    with pytest.raises(ValueError, match="not in kwonly_names"):
        make_kwargs_wrapper(target, ["a", "b"], kwonly_names=["c"], kwonly_defaults={"d": 10})

    # Internal name conflict
    with pytest.raises(ValueError, match="conflict with internal names"):
        make_kwargs_wrapper(target, ["__i_target_func", "b"])


def test_special_default_values() -> None:
    """Test wrapper with special default values like None and objects."""

    def target(a: Any, b: Any, c: Any) -> tuple[Any, Any, Any]:
        return (a, b, c)

    # None as default
    wrapper = make_kwargs_wrapper(target, ["a", "b", "c"], arg_defaults=(None, None))
    assert wrapper(1) == (1, None, None)

    # Complex objects as defaults (verify object reference is preserved)
    default_list = [1, 2, 3]
    wrapper = make_kwargs_wrapper(target, ["a", "b", "c"], arg_defaults=(default_list, None))
    result = wrapper(1)
    assert result[1] is default_list


def test_wrapper_with_signature() -> None:
    """Test make_kwargs_wrapper_from_signature."""
    target = lambda *args: sum(args)

    def source_func(a: Any, b: Any, c: int = 10, d: int = 20) -> None:
        """Source function documentation."""
        pass

    sig = inspect.signature(source_func)
    wrapper = make_kwargs_wrapper_from_signature(target, sig)
    assert wrapper(1, 2) == 33  # 1 + 2 + 10 + 20
    assert wrapper(1, 2, 3) == 26  # 1 + 2 + 3 + 20
    assert wrapper(1, 2, 3, 4) == 10  # 1 + 2 + 3 + 4

    # Test metadata preservation when prototype is provided
    wrapper_with_metadata = make_kwargs_wrapper_from_signature(target, sig, source_func)
    assert wrapper_with_metadata.__name__ == "source_func"
    assert wrapper_with_metadata.__doc__ == "Source function documentation."

    # With keyword-only arguments
    def source_kwonly(a: Any, b: Any, *, c: int = 10, d: int = 20) -> None:
        pass

    wrapper = make_kwargs_wrapper_from_signature(target, inspect.signature(source_kwonly))
    assert wrapper(1, 2) == 33
    assert wrapper(1, 2, c=5, d=6) == 14

    # With required keyword-only arguments
    def source_required_kwonly(a: Any, b: Any, *, c: Any, d: int = 20) -> None:
        pass

    wrapper = make_kwargs_wrapper_from_signature(target, inspect.signature(source_required_kwonly))
    assert wrapper(1, 2, c=10) == 33  # c required, d=20
    assert wrapper(1, 2, c=10, d=5) == 18  # both explicit

    # Reject *args and **kwargs
    def with_varargs(a: Any, *args: Any) -> None:
        pass

    with pytest.raises(ValueError, match=r"\*args not supported"):
        make_kwargs_wrapper_from_signature(target, inspect.signature(with_varargs))

    def with_kwargs(a: Any, **kwargs: Any) -> None:
        pass

    with pytest.raises(ValueError, match=r"\*\*kwargs not supported"):
        make_kwargs_wrapper_from_signature(target, inspect.signature(with_kwargs))

    # Test exclude_arg_names - ignore certain arguments from the signature
    def source_with_skip(a: Any, b: Any, c: int = 10, d: int = 20) -> None:
        pass

    wrapper = make_kwargs_wrapper_from_signature(
        target, inspect.signature(source_with_skip), exclude_arg_names=["c"]
    )
    # c is ignored, so wrapper should only have a, b, d
    assert wrapper(1, 2) == 23  # 1 + 2 + 20 (d=20)
    assert wrapper(1, 2, d=5) == 8  # 1 + 2 + 5

    # Test ignoring multiple arguments
    wrapper = make_kwargs_wrapper_from_signature(
        target, inspect.signature(source_with_skip), exclude_arg_names=["b", "d"]
    )
    # b and d are ignored, so wrapper should only have a, c
    assert wrapper(1) == 11  # 1 + 10 (c=10)
    assert wrapper(1, c=5) == 6  # 1 + 5

    # Test ignoring keyword-only arguments
    def source_kwonly_skip(a: Any, b: Any, *, c: int = 10, d: int = 20) -> None:
        pass

    wrapper = make_kwargs_wrapper_from_signature(
        target, inspect.signature(source_kwonly_skip), exclude_arg_names=["c"]
    )
    # c is skipped, so wrapper should only have a, b, d
    assert wrapper(1, 2) == 23  # 1 + 2 + 20 (d=20)
    assert wrapper(1, 2, d=5) == 8  # 1 + 2 + 5

    # Test excluding a non-existent argument (should be silently ignored)
    wrapper = make_kwargs_wrapper_from_signature(
        target, inspect.signature(source_with_skip), exclude_arg_names=["non_existent"]
    )
    # Should be the same as no exclusion
    assert wrapper(1, 2) == 33  # 1 + 2 + 10 + 20
    assert wrapper(1, 2, 3, 4) == 10  # 1 + 2 + 3 + 4

    # Test excluding both existing and non-existent arguments
    wrapper = make_kwargs_wrapper_from_signature(
        target,
        inspect.signature(source_with_skip),
        exclude_arg_names=["c", "non_existent", "also_missing"],
    )
    # Only c should be excluded, non-existent names are ignored
    assert wrapper(1, 2) == 23  # 1 + 2 + 20 (d=20, c excluded)
    assert wrapper(1, 2, d=5) == 8  # 1 + 2 + 5


def test_exception_propagation() -> None:
    """Test that exceptions from the target function are properly propagated."""

    def raising_func(a: int, b: int, c: str) -> int:
        if a == 0:
            raise ValueError("a cannot be zero")
        if b < 0:
            raise RuntimeError(f"b must be non-negative, got {b}")
        if c != "valid":
            raise TypeError(f"c must be 'valid', got {c!r}")
        return a + b

    # Test with positional defaults
    wrapper = make_kwargs_wrapper(raising_func, ["a", "b", "c"], arg_defaults=(10, "valid"))
    assert wrapper(5) == 15

    with pytest.raises(ValueError, match="a cannot be zero"):
        wrapper(0)

    with pytest.raises(RuntimeError, match="b must be non-negative"):
        wrapper(1, -5)

    # Test with keyword-only arguments
    wrapper_kwonly = make_kwargs_wrapper(
        raising_func,
        ["a"],
        kwonly_names=["b", "c"],
        kwonly_defaults={"b": 10, "c": "valid"},
    )
    assert wrapper_kwonly(5) == 15

    with pytest.raises(ValueError, match="a cannot be zero"):
        wrapper_kwonly(0)

    with pytest.raises(RuntimeError, match="b must be non-negative"):
        wrapper_kwonly(5, b=-5)

    with pytest.raises(TypeError, match="c must be 'valid'"):
        wrapper_kwonly(5, c="invalid")


def test_metadata_preservation() -> None:
    """Test that function metadata is preserved when prototype is provided."""

    def my_function(x: int, y: int = 10) -> int:
        """Document the function."""
        return x + y

    target = lambda *args: sum(args)

    wrapper = make_kwargs_wrapper(target, ["x", "y"], arg_defaults=(10,), prototype=my_function)
    assert wrapper.__name__ == "my_function"
    assert wrapper.__doc__ == "Document the function."
    assert wrapper.__annotations__ == my_function.__annotations__
    assert wrapper(5) == 15


def test_optimized_default_types() -> None:
    """Test that None, bool, and str defaults work correctly.

    This test verifies the optimization where None and bool defaults are
    directly embedded in the generated signature, while str defaults use
    the MISSING sentinel for safety.
    """

    def target(*args: Any) -> tuple[Any, ...]:
        return args

    # Test None default (should be optimized - directly embedded)
    wrapper = make_kwargs_wrapper(target, ["a", "b", "c"], arg_defaults=(None,))
    assert wrapper(1, 2) == (1, 2, None)
    assert wrapper(1, 2, 3) == (1, 2, 3)
    assert wrapper(1, 2, c=None) == (1, 2, None)

    # Test bool defaults (should be optimized - directly embedded)
    wrapper = make_kwargs_wrapper(target, ["a", "flag", "debug"], arg_defaults=(True, False))
    assert wrapper(1) == (1, True, False)
    assert wrapper(1, False) == (1, False, False)
    assert wrapper(1, flag=False, debug=True) == (1, False, True)

    # Test str default (should use MISSING sentinel for safety)
    wrapper = make_kwargs_wrapper(target, ["a", "b", "name"], arg_defaults=("default",))
    assert wrapper(1, 2) == (1, 2, "default")
    assert wrapper(1, 2, "custom") == (1, 2, "custom")
    assert wrapper(1, 2, name="custom") == (1, 2, "custom")

    # Test keyword-only with None, bool, and str
    wrapper = make_kwargs_wrapper(
        target,
        ["a"],
        kwonly_names=["b", "flag", "name"],
        kwonly_defaults={"b": None, "flag": True, "name": "default"},
    )
    assert wrapper(1) == (1, None, True, "default")
    assert wrapper(1, b=2) == (1, 2, True, "default")
    assert wrapper(1, flag=False) == (1, None, False, "default")
    assert wrapper(1, name="custom") == (1, None, True, "custom")
    assert wrapper(1, b=2, flag=False, name="test") == (1, 2, False, "test")
