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
"""Utilities for creating high-performance keyword argument wrapper functions.

This module provides tools for wrapping positional-only callables with
keyword argument support using code generation techniques.
"""

from __future__ import annotations

import functools
import inspect
from typing import Any, Callable

# Sentinel object for missing arguments
MISSING = object()


def _validate_argument_names(names: list[str], arg_type: str) -> None:
    """Validate that argument names are valid Python identifiers and unique.

    Parameters
    ----------
    names
        List of argument names to validate.
    arg_type
        Description of the argument type (e.g., "Argument", "Keyword-only argument").

    """
    # Check for duplicate names
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate {arg_type.lower()} names found in: {names}")

    # Validate each name is a valid identifier
    for name in names:
        if not isinstance(name, str):
            raise TypeError(
                f"{arg_type} name must be a string, got {type(name).__name__}: {name!r}"
            )
        if not name.isidentifier():
            raise ValueError(
                f"Invalid {arg_type.lower()} name: {name!r} is not a valid Python identifier"
            )


def _validate_wrapper_args(
    args_names: list[str],
    args_defaults: tuple,
    kwargsonly_names: list[str],
    kwargsonly_defaults: dict[str, Any],
    reserved_names: set[str],
) -> None:
    """Validate all input arguments for make_kwargs_wrapper.

    Parameters
    ----------
    args_names
        List of positional argument names.
    args_defaults
        Tuple of default values for positional arguments.
    kwargsonly_names
        List of keyword-only argument names.
    kwargsonly_defaults
        Dictionary of default values for keyword-only arguments.
    reserved_names
        Set of reserved internal names that cannot be used as argument names.

    """
    # Validate args_names are valid Python identifiers and unique
    _validate_argument_names(args_names, "Argument")

    # Validate args_defaults is a tuple
    if not isinstance(args_defaults, tuple):
        raise TypeError(f"args_defaults must be a tuple, got {type(args_defaults).__name__}")

    # Validate args_defaults length doesn't exceed args_names length
    if len(args_defaults) > len(args_names):
        raise ValueError(
            f"args_defaults has {len(args_defaults)} values but only "
            f"{len(args_names)} positional arguments"
        )

    # Validate kwargsonly_names are valid identifiers and unique
    _validate_argument_names(kwargsonly_names, "Keyword-only argument")

    # Validate kwargsonly_defaults keys are in kwargsonly_names
    kwargsonly_names_set = set(kwargsonly_names)
    for key in kwargsonly_defaults:
        if key not in kwargsonly_names_set:
            raise ValueError(
                f"Default provided for '{key}' which is not in kwargsonly_names: {kwargsonly_names}"
            )

    # Validate no overlap between positional and keyword-only arguments
    args_names_set = set(args_names)
    overlap = args_names_set & kwargsonly_names_set
    if overlap:
        raise ValueError(f"Arguments cannot be both positional and keyword-only: {overlap}")

    # Validate no conflict between user argument names and internal names
    all_user_names = args_names_set | kwargsonly_names_set
    conflicts = all_user_names & reserved_names
    if conflicts:
        raise ValueError(
            f"Argument names conflict with internal names: {conflicts}. "
            f'Please avoid using names starting with "__i_"'
        )


def make_kwargs_wrapper(
    target_func: Callable,
    args_names: list[str],
    args_defaults: tuple = (),
    kwargsonly_names: list[str] | None = None,
    kwargsonly_defaults: dict[str, Any] | None = None,
    prototype_func: Callable | None = None,
) -> Callable:
    """Create a wrapper with kwargs support for a function that only accepts positional arguments.

    This function dynamically generates a wrapper using code generation to minimize overhead.

    Parameters
    ----------
    target_func
        The underlying function to be called by the wrapper. This function must only
        accept positional arguments.
    args_names
        A list of ALL positional argument names in order. These define the positional
        parameters that the wrapper will accept. Must not overlap with kwargsonly_names.
    args_defaults
        A tuple of default values for positional arguments, right-aligned to args_names
        (matching Python's __defaults__ behavior). The length of this tuple determines
        how many trailing arguments have defaults.
        Example: (10, 20) with args_names=['a', 'b', 'c', 'd'] means c=10, d=20.
        Empty tuple () means no defaults.
    kwargsonly_names
        A list of keyword-only argument names. These arguments can only be passed by name,
        not positionally, and appear after a '*' separator in the signature. Can include both
        required and optional keyword-only arguments. Must not overlap with args_names.
        Example: ['debug', 'timeout'] creates wrapper(..., *, debug, timeout).
    kwargsonly_defaults
        Optional dictionary of default values for keyword-only arguments (matching Python's
        __kwdefaults__ behavior). Keys must be a subset of kwargsonly_names. Keyword-only
        arguments not in this dict are required.
        Example: {'timeout': 30} with kwargsonly_names=['debug', 'timeout'] means 'debug'
        is required and 'timeout' is optional.
    prototype_func
        Optional prototype function to copy metadata (__name__, __doc__, __module__,
        __qualname__, __annotations__) from. If None, no metadata is copied.

    Returns
    -------
        A dynamically generated wrapper function with the specified signature

    Notes
    -----
    The generated wrapper will directly embed default values for None and bool types
    and use a MISSING sentinel object to distinguish between explicitly
    passed arguments and those that should use default values for other types to ensure
    the generated code does not contain unexpected str repr.

    """
    # Normalize inputs
    if kwargsonly_names is None:
        kwargsonly_names = []
    if kwargsonly_defaults is None:
        kwargsonly_defaults = {}

    # Internal variable names used in generated code to avoid user argument conflicts
    _INTERNAL_TARGET_FUNC = "__i_target_func"
    _INTERNAL_MISSING = "__i_MISSING"
    _INTERNAL_DEFAULTS_DICT = "__i_args_defaults"
    _INTERNAL_NAMES = {_INTERNAL_TARGET_FUNC, _INTERNAL_MISSING, _INTERNAL_DEFAULTS_DICT}

    # Validate all input arguments
    _validate_wrapper_args(
        args_names, args_defaults, kwargsonly_names, kwargsonly_defaults, _INTERNAL_NAMES
    )

    # Build positional defaults dictionary (right-aligned)
    # Example: args_names=["a","b","c","d"], args_defaults=(10,20) -> {"c":10, "d":20}
    args_defaults_dict = (
        dict(zip(args_names[-len(args_defaults) :], args_defaults)) if args_defaults else {}
    )

    # Build wrapper signature and call arguments
    # Note: this code must be in this function so all code generation and exec are self-contained
    # We construct runtime_defaults dict for only non-safe defaults that need MISSING sentinel
    arg_parts = []
    call_parts = []
    runtime_defaults = {}

    def _add_param_with_default(name: str, default_value: Any) -> None:
        """Add a parameter with a default value to arg_parts and call_parts."""
        # Rationale: we directly embed default values for None and bool
        # since they are common case and safe to be directly included in generated code.
        #
        # For other cases (including int/str), we use the MISSING sentinel to ensure
        # generated code do not contain unexpected str repr and instead they are passed
        # through runtime_defaults[name].
        #
        # we deliberately skip int/str since bring their string representation
        # may involve __str__ / __repr__ that could be updated by subclasses.
        # The missing check is generally fast enough and more controllable.
        if default_value is None:
            # Safe to use the default value None directly in the signature
            arg_parts.append(f"{name}=None")
            call_parts.append(name)
        elif type(default_value) is bool:
            # we deliberately not use isinstance to avoid subclasses of bool
            # we also explicitly avoid repr for safety
            default_value_str = "True" if default_value else "False"
            arg_parts.append(f"{name}={default_value_str}")
            call_parts.append(name)
        else:
            # For all other cases, we use the MISSING sentinel
            arg_parts.append(f"{name}={_INTERNAL_MISSING}")
            runtime_defaults[name] = default_value
            # The conditional check runs
            call_parts.append(
                f'{_INTERNAL_DEFAULTS_DICT}["{name}"] if {name} is {_INTERNAL_MISSING} else {name}'
            )

    # Handle positional arguments
    for name in args_names:
        if name in args_defaults_dict:
            _add_param_with_default(name, args_defaults_dict[name])
        else:
            arg_parts.append(name)
            call_parts.append(name)

    # Handle keyword-only arguments
    if kwargsonly_names:
        arg_parts.append("*")  # Separator for keyword-only args
        for name in kwargsonly_names:
            if name in kwargsonly_defaults:
                _add_param_with_default(name, kwargsonly_defaults[name])
            else:
                # Required keyword-only arg (no default)
                arg_parts.append(name)
                call_parts.append(name)

    arg_list = ", ".join(arg_parts)
    call_list = ", ".join(call_parts)

    code_str = f"""
def wrapper({arg_list}):
    return {_INTERNAL_TARGET_FUNC}({call_list})
"""
    # Execute the generated code
    exec_globals = {
        _INTERNAL_TARGET_FUNC: target_func,
        _INTERNAL_MISSING: MISSING,
        _INTERNAL_DEFAULTS_DICT: runtime_defaults,
    }
    namespace: dict[str, Any] = {}
    # Note: this is a limited use of exec that is safe.
    # We ensure generated code does not contain any untrusted input.
    # The argument names are validated and the default values are not part of generated code.
    # Instead default values are set to MISSING sentinel object and explicitly passed as exec_globals.
    # This is a practice adopted by `dataclasses` and `pydantic`
    exec(code_str, exec_globals, namespace)
    new_func = namespace["wrapper"]

    # Copy metadata from prototype_func if provided
    if prototype_func is not None:
        functools.update_wrapper(new_func, prototype_func, updated=())

    return new_func


def make_kwargs_wrapper_from_signature(
    target_func: Callable,
    signature: inspect.Signature,
    prototype_func: Callable | None = None,
) -> Callable:
    """Create a wrapper with kwargs support for a function that only accepts positional arguments.

    This is a convenience function that extracts parameter information from a signature
    object and calls make_kwargs_wrapper with the appropriate arguments. Supports both
    required and optional keyword-only arguments.

    Parameters
    ----------
    target_func
        The underlying function to be called by the wrapper.
    signature
        An inspect.Signature object describing the desired wrapper signature.
    prototype_func
        Optional prototype function to copy metadata (__name__, __doc__, __module__,
        __qualname__, __annotations__) from. If None, no metadata is copied.

    Returns
    -------
        A dynamically generated wrapper function with the specified signature.

    Raises
    ------
    ValueError
        If the signature contains *args or **kwargs.

    """
    # Extract positional and keyword-only parameters
    args_names = []
    args_defaults_list = []
    kwargsonly_names = []
    kwargsonly_defaults = {}

    # Track when we start seeing defaults for positional args
    has_seen_positional_default = False

    for param_name, param in signature.parameters.items():
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            raise ValueError("*args not supported in wrapper generation")
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            raise ValueError("**kwargs not supported in wrapper generation")
        elif param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            args_names.append(param_name)
            if param.default is not inspect.Parameter.empty:
                has_seen_positional_default = True
                args_defaults_list.append(param.default)
            elif has_seen_positional_default:
                # Required arg after optional arg (invalid in Python)
                raise ValueError(
                    f"Required positional parameter '{param_name}' cannot follow "
                    f"parameters with defaults"
                )
        elif param.kind == inspect.Parameter.KEYWORD_ONLY:
            kwargsonly_names.append(param_name)
            if param.default is not inspect.Parameter.empty:
                kwargsonly_defaults[param_name] = param.default

    # Convert defaults list to tuple (right-aligned to args_names)
    args_defaults = tuple(args_defaults_list)

    return make_kwargs_wrapper(
        target_func,
        args_names,
        args_defaults,
        kwargsonly_names,
        kwargsonly_defaults,
        prototype_func,
    )
