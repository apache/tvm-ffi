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
"""AST visitor and transformer for :mod:`tvm_ffi.pyast` nodes.

Modeled after :class:`ast.NodeVisitor` and :class:`ast.NodeTransformer` from
the Python standard library, adapted for the TVM FFI pyast node hierarchy.
"""

from __future__ import annotations

import json
from collections.abc import Generator, MutableSequence
from functools import lru_cache
from typing import Any


class _FieldInfo:
    """Pre-computed metadata for a single field descriptor."""

    __slots__ = ("field", "optional")

    def __init__(self, field: Any) -> None:
        self.field = field
        schema_json = field.metadata.get("type_schema", "")
        if schema_json:
            self.optional = json.loads(schema_json).get("type") == "Optional"
        else:
            self.optional = False


@lru_cache(maxsize=None)
def _collect_fields(cls: Any) -> list[_FieldInfo]:
    """Return deduplicated field info for *cls* (parent-first order).

    Walks the ``__tvm_ffi_type_info__`` parent chain and collects fields,
    skipping any field whose name was already seen so that shadowed
    re-declarations (e.g. ``source_paths`` on both ``Node`` and ``Expr``)
    are yielded only once.

    The result is cached per class, so repeated calls are free.
    """
    ti = cls.__tvm_ffi_type_info__
    chain: list[Any] = []
    while ti is not None:
        chain.append(ti)
        ti = ti.parent_type_info
    seen: set[str] = set()
    result: list[_FieldInfo] = []
    for ancestor_info in reversed(chain):
        if ancestor_info.fields is not None:
            for field in ancestor_info.fields:
                if field.name not in seen:
                    seen.add(field.name)
                    result.append(_FieldInfo(field))
    return result


def iter_fields(node: Any) -> Generator[tuple[str, Any], None, None]:
    """Yield ``(field_name, field_value)`` pairs for each field of *node*.

    Fields are discovered via the TVM FFI reflection system
    (``__tvm_ffi_type_info__``), walking the parent chain so that inherited
    fields are included.  Shadowed re-declarations are yielded only once.

    Parameters
    ----------
    node
        A :class:`~tvm_ffi.pyast.Node` instance.

    Yields
    ------
    tuple[str, Any]
        ``(name, value)`` for every registered field on *node*.

    """
    for fi in _collect_fields(type(node)):
        yield fi.field.name, getattr(node, fi.field.name)


def iter_child_nodes(node: Any) -> Generator[Any, None, None]:
    """Yield all direct child nodes of *node*.

    Child nodes are field values (or items inside list-valued fields) that
    are themselves :class:`~tvm_ffi.pyast.Node` instances.

    Parameters
    ----------
    node
        A :class:`~tvm_ffi.pyast.Node` instance.

    Yields
    ------
    Node
        Each direct child node.

    """
    from .pyast import Node  # noqa: PLC0415

    for _name, value in iter_fields(node):
        if isinstance(value, Node):
            yield value
        elif isinstance(value, (list, MutableSequence)):
            for item in value:
                if isinstance(item, Node):
                    yield item


class NodeVisitor:
    """Walk a pyast tree and call a visitor function for every node found.

    This class is meant to be subclassed, with the subclass adding visitor
    methods.

    Per default the visitor functions for the nodes are ``'visit_'`` +
    class name of the node.  So an ``If`` node visit function would be
    ``visit_If``.  This behavior can be changed by overriding the
    :meth:`visit` method.  If no visitor function exists for a node
    (return value ``None``) the :meth:`generic_visit` visitor is used instead.

    Don't use the :class:`NodeVisitor` if you want to apply changes to nodes
    during traversing.  For this a special visitor exists
    (:class:`NodeTransformer`) that allows modifications.
    """

    def visit(self, node: Any) -> Any:
        """Visit a node."""
        method = "visit_" + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node: Any) -> None:
        """Traverse all child nodes when no explicit visitor exists."""
        from .pyast import Node  # noqa: PLC0415

        for _name, value in iter_fields(node):
            if isinstance(value, Node):
                self.visit(value)
            elif isinstance(value, (list, MutableSequence)):
                for item in value:
                    if isinstance(item, Node):
                        self.visit(item)


class NodeTransformer(NodeVisitor):
    """Walk the pyast tree and allow modification of nodes.

    The :class:`NodeTransformer` will walk the AST and use the return value
    of the visitor methods to replace or remove the old node.  If the return
    value of the visitor method is ``None``, the node will be removed from
    its location, otherwise it is replaced with the return value.  The return
    value may be the original node in which case no replacement takes place.

    Here is an example transformer that rewrites all ``Id`` nodes named
    ``"x"`` to ``"y"``::

        class RenameX(NodeTransformer):
            def visit_Id(self, node):
                if node.name == "x":
                    return pyast.Id("y")
                return node

    Keep in mind that if the node you're operating on has child nodes you
    must either transform the child nodes yourself or call the
    :meth:`generic_visit` method for the node first.

    For nodes that were part of a collection of statements (that applies to
    all statement nodes), the visitor may also return a list of nodes rather
    than just a single node.

    Usually you use the transformer like this::

        node = YourTransformer().visit(node)
    """

    def generic_visit(self, node: Any) -> Any:
        """Transform all child nodes when no explicit visitor exists."""
        from .pyast import Node  # noqa: PLC0415

        for fi in _collect_fields(type(node)):
            old_value = getattr(node, fi.field.name)
            if isinstance(old_value, (list, MutableSequence)):
                new_values = []
                for item in old_value:
                    if isinstance(item, Node):
                        new_item = self.visit(item)
                        if new_item is None:
                            continue
                        elif not isinstance(new_item, Node):
                            new_values.extend(new_item)
                            continue
                        else:
                            new_values.append(new_item)
                    else:
                        new_values.append(item)
                old_value[:] = new_values
            elif isinstance(old_value, Node):
                new_node = self.visit(old_value)
                if new_node is not None:
                    fi.field.setter(node, new_node)
                elif fi.optional:
                    fi.field.setter(node, None)
                # else: non-optional field — keep the original node
        return node
