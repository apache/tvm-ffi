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
"""The one-line directive grammar shared by every generator.

A directive names a subject, a target, and optional keys::

    <comment> tvm-ffi-stubgen(<subject>): <target>
    <comment> tvm-ffi-stubgen(<subject>): <target>(<key>, <key> = <value>, ...)

A key is a flag or takes a value; commas and ``=`` inside ``()``, ``[]``, ``{}`` or
``<>`` belong to the value, so ``pub(crate)`` and ``Option<A, B>`` are one value.
Lines for the same subject and target merge: a repeatable key collects every value,
any other key may repeat only with the same value. The pipeline reads the
``object`` / ``file`` keys of :data:`consts.PIPELINE_DIRECTIVE_KEYS`; the active
generator declares the rest (``Generator.directive_keys``).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Union

#: A parsed value: ``True`` for a flag, the text, or every text of a repeatable key.
Value = Union[bool, str, list[str]]


@dataclasses.dataclass(frozen=True)
class Key:
    """How a key is written: a flag, or ``key = value`` (``values`` lists the legal ones)."""

    takes_value: bool = False
    repeatable: bool = False
    values: tuple[str, ...] = ()


#: Key tables: subject -> key -> spelling. ``"*"`` accepts any other key with a value.
KeyTable = Mapping[str, Mapping[str, Key]]


class Directives:
    """The directives of one file: ``settings[subject][target][key]``."""

    def __init__(self, keys: KeyTable) -> None:
        self.keys = keys
        self.settings: dict[str, dict[str, dict[str, Value]]] = {s: {} for s in keys}
        self._lines: dict[tuple[str, str, str], int] = {}

    def add(self, subject: str, payload: str, lineno: int) -> None:
        """Parse and merge one directive; raise ``ValueError`` on a malformed one."""
        if subject not in self.keys:
            raise ValueError(f"Unknown directive `{subject}` at line {lineno}")

        def invalid(reason: str) -> ValueError:
            return ValueError(f"Invalid `{subject}` directive at line {lineno}: {reason}")

        target, paren, rest = payload.strip().partition("(")
        target = target.strip()
        if paren and not rest.endswith(")"):
            raise invalid("expected `<target>(<key>, <key> = <value>, ...)`")
        if not target or any(c.isspace() for c in target):
            raise invalid(f"bad target `{target}`")
        if (subject == "field" and "." not in target.strip(".")) or (
            subject == "file" and target != "mod"
        ):
            raise invalid("the target is `<type_key>.<field>`" if subject == "field" else "`mod`")
        entry = self.settings[subject].setdefault(target, {})
        for arg in filter(str.strip, _split(rest[:-1], ",")):
            key, *value_parts = _split(arg, "=")
            key, eq, value = key.strip(), bool(value_parts), "=".join(value_parts).strip()
            spec = self.keys[subject].get(key) or self.keys[subject].get("*")
            if not key.isidentifier() or spec is None:
                raise invalid(f"unknown key `{key}`")
            if eq != spec.takes_value or (eq and not value):
                raise invalid(
                    f"`{key}` takes a value" if spec.takes_value else f"`{key}` is a flag"
                )
            if spec.values and value not in spec.values:
                raise invalid(f"`{key}` is one of {', '.join(spec.values)}")
            if spec.repeatable:
                collected = entry.setdefault(key, [])
                assert isinstance(collected, list)
                collected.append(value)
                continue
            new: Value = value if eq else True
            first = self._lines.setdefault((subject, target, key), lineno)
            if entry.setdefault(key, new) != new:
                raise invalid(f"`{key}` conflicts with line {first}")


def _split(text: str, sep: str) -> list[str]:
    """Split ``text`` on each ``sep`` outside brackets."""
    parts, depth, start = [], 0, 0
    for i, char in enumerate(text):
        depth += (char in "([{<") - (char in ")]}>")
        if char == sep and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    return [*parts, text[start:]]
