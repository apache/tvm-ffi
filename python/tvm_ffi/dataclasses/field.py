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
"""Field descriptor and ``field()`` helper for Python-defined TVM-FFI types."""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any, ClassVar

from ..core import MISSING, TypeSchema

# Re-export the stdlib KW_ONLY sentinel so type checkers recognise
# ``_: KW_ONLY`` as a keyword-only boundary rather than a real field.
# dataclasses.KW_ONLY was added in Python 3.10; on older runtimes we
# define a class sentinel (a class, not an instance, so that ``_: KW_ONLY``
# is a valid type annotation for static analysers targeting 3.9).
if sys.version_info >= (3, 10):
    from dataclasses import KW_ONLY
else:

    class KW_ONLY:
        """Sentinel type: annotations after ``_: KW_ONLY`` are keyword-only."""


class Field:
    """Descriptor for a single field in a Python-defined TVM-FFI type.

    When constructed directly (low-level API), *name* and *_ty_schema*
    should be provided.  When returned by :func:`field` (``@py_class``
    workflow), both are ``None`` and filled in by the decorator.

    Parameters
    ----------
    name : str | None
        The field name.  ``None`` when created via :func:`field`; filled
        in by the ``@py_class`` decorator.
    _ty_schema : TypeSchema | None
        Private: the internal :class:`TypeSchema` used by the reflection
        layer.  ``None`` when created via :func:`field`; filled in by
        the ``@py_class`` decorator.  Consumers should use :attr:`type`
        instead.
    type : Any
        The resolved Python annotation (e.g. ``int``, ``list[str]``,
        ``Optional[X]``).  Filled in by the ``@py_class`` / ``@c_class``
        decorator via :func:`typing.get_type_hints`; ``None`` until then
        or when the annotation cannot be resolved.
    default : object
        Default value for the field. Mutually exclusive with *default_factory*.
        ``MISSING`` when not set.
    default_factory : Callable[[], object] | None
        A zero-argument callable that produces the default value.
        Mutually exclusive with *default*.  ``None`` when not set.
    frozen : bool
        Whether this field is read-only after ``__init__``.
    init : bool
        Whether this field appears in the auto-generated ``__init__``.
    repr : bool
        Whether this field appears in ``__repr__`` output.
    hash : bool | None
        Whether this field participates in recursive hashing.
        ``None`` means "follow *compare*" (the native dataclass default).
    compare : bool
        Whether this field participates in recursive comparison.
    kw_only : bool | None
        Whether this field is keyword-only in ``__init__``.
        ``None`` means "inherit from the decorator-level *kw_only* flag".
    structural_eq : str | None
        Structural equality/hashing annotation for this field.  Valid
        values are:

        - ``None`` (default): the field participates normally in
          structural comparison and hashing.
        - ``"ignore"``: the field is excluded from structural equality
          and hashing entirely (e.g. source spans, caches).
        - ``"def"``: the field is a **definition region** that introduces
          new variable bindings.  Free variables encountered inside this
          field are mapped by position, enabling alpha-equivalence.
    doc : str | None
        Optional docstring for the field.

    """

    __slots__ = (
        "_ty_schema",
        "compare",
        "default",
        "default_factory",
        "doc",
        "frozen",
        "hash",
        "init",
        "kw_only",
        "name",
        "repr",
        "structural_eq",
        "type",
    )
    name: str | None
    _ty_schema: TypeSchema | None
    type: Any
    default: object
    default_factory: Callable[[], object] | None
    frozen: bool
    init: bool
    repr: bool
    hash: bool | None
    compare: bool
    kw_only: bool | None
    structural_eq: str | None
    doc: str | None

    #: Valid values for the *structural_eq* parameter.
    _VALID_STRUCTURAL_EQ_VALUES: ClassVar[frozenset[str | None]] = frozenset(
        {None, "ignore", "def"}
    )

    def __init__(  # noqa: PLR0913
        self,
        name: str | None = None,
        _ty_schema: TypeSchema | None = None,
        *,
        default: object = MISSING,
        default_factory: Callable[[], object] | None = MISSING,  # type: ignore[assignment]
        frozen: bool = False,
        init: bool = True,
        repr: bool = True,
        hash: bool | None = True,
        compare: bool = False,
        kw_only: bool | None = False,
        structural_eq: str | None = None,
        doc: str | None = None,
    ) -> None:
        # MISSING means "parameter not provided".
        # An explicit None from the user fails the callable() check,
        # matching stdlib dataclasses semantics.
        if default_factory is not MISSING:
            if default is not MISSING:
                raise ValueError("cannot specify both default and default_factory")
            if not callable(default_factory):
                raise TypeError(
                    f"default_factory must be a callable, got {type(default_factory).__name__}"
                )
        if structural_eq not in Field._VALID_STRUCTURAL_EQ_VALUES:
            raise ValueError(
                f"structural_eq must be one of "
                f"{sorted(Field._VALID_STRUCTURAL_EQ_VALUES, key=str)}, "
                f"got {structural_eq!r}"
            )
        self.name = name
        self._ty_schema = _ty_schema
        self.type = None
        self.default = default
        self.default_factory = default_factory
        self.frozen = frozen
        self.init = init
        self.repr = repr
        self.hash = hash
        self.compare = compare
        self.kw_only = kw_only
        self.structural_eq = structural_eq
        self.doc = doc


def field(
    *,
    default: object = MISSING,
    default_factory: Callable[[], object] | None = MISSING,  # type: ignore[assignment]
    frozen: bool = False,
    init: bool = True,
    repr: bool = True,
    hash: bool | None = None,
    compare: bool = True,
    kw_only: bool | None = None,
    structural_eq: str | None = None,
    doc: str | None = None,
) -> Any:
    """Customize a field in a ``@py_class``-decorated class.

    Returns a :class:`Field` sentinel whose *name* and *_ty_schema*
    are ``None``.  The ``@py_class`` decorator fills them in later
    from the class annotations.

    The return type is ``Any`` because ``dataclass_transform`` field
    specifiers must be assignable to any annotated type (e.g.
    ``x: int = field(default=0)``).

    Parameters
    ----------
    default
        Default value for the field.  Mutually exclusive with *default_factory*.
    default_factory
        A zero-argument callable that produces the default value.
        Mutually exclusive with *default*.
    frozen
        Whether this field is read-only after ``__init__``.  When True,
        normal attribute assignment is blocked; use
        ``type(obj)._force_set_field(obj, "field_name", value)``
        when mutation is necessary.
    init
        Whether this field appears in the auto-generated ``__init__``.
    repr
        Whether this field appears in ``__repr__`` output.
    hash
        Whether this field participates in recursive hashing.
        ``None`` (default) means "follow *compare*".
    compare
        Whether this field participates in recursive comparison.
    kw_only
        Whether this field is keyword-only in ``__init__``.
        ``None`` means "inherit from the decorator-level ``kw_only`` flag".
    structural_eq
        Structural equality/hashing annotation. ``None`` (default) means
        the field participates normally. ``"ignore"`` excludes the field
        from structural comparison and hashing. ``"def"`` marks the field
        as a definition region for variable binding.
    doc
        Optional docstring for the field.

    Returns
    -------
    Any
        A :class:`Field` sentinel recognised by ``@py_class``.

    Examples
    --------
    .. code-block:: python

        @py_class
        class Point(Object):
            x: float
            y: float = field(default=0.0, repr=False)


        @py_class(structural_eq="tree")
        class MyFunc(Object):
            params: Array = field(structural_eq="def")
            body: Expr
            span: Object = field(structural_eq="ignore")

    """
    return Field(
        default=default,
        default_factory=default_factory,
        frozen=frozen,
        init=init,
        repr=repr,
        hash=hash,
        compare=compare,
        kw_only=kw_only,
        structural_eq=structural_eq,
        doc=doc,
    )
