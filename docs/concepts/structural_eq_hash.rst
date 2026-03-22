..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

Structural Equality and Hashing
===============================

TVM FFI provides ``structural_equal`` and ``structural_hash`` for the
object graph. These compare objects by **content** — recursively walking
fields — rather than by pointer identity.

The behavior is controlled by two layers of annotation on
:func:`~tvm_ffi.dataclasses.py_class`:

1. **Type-level** ``structure=`` — what *role* does this type play in the
   IR graph?
2. **Field-level** ``structure=`` on :func:`~tvm_ffi.dataclasses.field` —
   should this field be skipped, or does it introduce new variable bindings?

This document explains what each annotation means, when to use it, and how
they compose.


Type-Level Annotation
---------------------

The ``structure`` parameter on ``@py_class`` declares how instances of the
type participate in structural equality and hashing:

.. code-block:: python

   @py_class(structure="tree")
   class Expr(Object):
       ...

Quick reference
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 37 45

   * - ``structure=``
     - Meaning
     - Use when...
   * - ``"tree"``
     - A regular IR node
     - Default for most IR nodes
   * - ``"const-tree"``
     - An immutable value node (with pointer shortcut)
     - The type has no transitive ``"var"`` children
   * - ``"dag"``
     - A node in a dataflow graph
     - Pointer sharing is semantically meaningful
   * - ``"var"``
     - A bound variable
     - The type represents a variable binding
   * - ``"singleton"``
     - A singleton
     - Exactly one instance per logical identity (e.g. registry entries)
   * - ``None``
     - Not comparable
     - The type should never be compared structurally


``"tree"`` — The Default
-------------------------

.. code-block:: python

   @py_class(structure="tree")
   class Add(Object):
       lhs: Expr
       rhs: Expr

**Meaning**: "This node is defined by its fields. Two nodes are equal if and
only if all their fields are recursively equal."

This is the right choice for the vast majority of IR nodes: expressions,
statements, types, attributes, buffers, etc.

**Example.**

.. code-block:: text

   1 + 2  vs  1 + 2   →  Equal
   1 + 2  vs  1 + 3   →  Not equal (rhs differs)

Sharing is invisible
~~~~~~~~~~~~~~~~~~~~

``"tree"`` treats every reference independently. If the same object is
referenced multiple times, each reference is compared by content separately.
Sharing is **not** part of the structural identity:

.. code-block:: text

   let s = x + 1

   (s, s)                   ← same object referenced twice
   (x + 1, x + 1)          ← two independent copies with same content

   These are EQUAL under "tree" — sharing is not detected.

The following diagram illustrates this. Under ``"tree"``, the **DAG** on the
left and the **tree** on the right are considered structurally equal because
every node has the same content:

.. mermaid::

   graph TD
       subgraph "DAG — shared node"
           T1["(_, _)"]
           S1["s = x + 1"]
           T1 -->|".0"| S1
           T1 -->|".1"| S1
       end

       subgraph "Tree — independent copies"
           T2["(_, _)"]
           A1["x + 1"]
           A2["x + 1"]
           T2 -->|".0"| A1
           T2 -->|".1"| A2
       end

       style S1 fill:#d4edda
       style A1 fill:#d4edda
       style A2 fill:#d4edda

If sharing needs to matter, use ``"dag"`` instead.


``"const-tree"`` — Tree with a Fast Path
-----------------------------------------

.. code-block:: python

   @py_class(structure="const-tree")
   class DeviceMesh(Object):
       shape: list[int]
       device_ids: list[int]

**Meaning**: "Same as ``"tree"``, but if two references point to the same
object, they are guaranteed equal — skip the field comparison."

This is purely a **performance optimization**. The only behavioral difference
from ``"tree"`` is that pointer identity short-circuits to ``True``.

When is this safe?
~~~~~~~~~~~~~~~~~~

When the type satisfies two conditions:

1. **Immutable** — content doesn't change after construction, so same-pointer
   always implies same-content.
2. **No transitive** ``"var"`` **children** — skipping field traversal won't
   cause variable mappings to be missed (see :ref:`var-kind` for why this
   matters).

Why not use it everywhere?
~~~~~~~~~~~~~~~~~~~~~~~~~~

Most IR nodes are immutable, but many transitively contain variables
(e.g., ``x + 1`` contains the ``"var"`` node ``x``). If the pointer
shortcut fires, the traversal skips ``x``, and a variable mapping that should
have been established is silently missed.

The following diagram shows the danger. Suppose the ``+`` node were
incorrectly annotated as ``"const-tree"``. When comparing two trees that
share a sub-expression, the pointer shortcut fires on the shared node, and
the ``"var"`` ``x`` inside it is never visited — so no ``x ↔ y`` mapping
is recorded:

.. mermaid::

   graph TD
       subgraph "lhs"
           LT["(_, _)"]
           LE["x + 1"]
           LX["x : var"]
           LT -->|".0"| LE
           LT -->|".1"| LX
           LE -->|".lhs"| LX2["x"]
       end

       subgraph "rhs"
           RT["(_, _)"]
           RE["y + 1"]
           RY["y : var"]
           RT -->|".0"| RE
           RT -->|".1"| RY
           RE -->|".lhs"| RY2["y"]
       end

       LE -. "const-tree would skip here<br/>(misses x ↔ y mapping)" .-> RE
       LX -. "Later comparison fails:<br/>x has no recorded mapping" .-> RY

       style LE fill:#fff3cd
       style RE fill:#fff3cd
       style LX fill:#f8d7da
       style RY fill:#f8d7da
       style LX2 fill:#f8d7da
       style RY2 fill:#f8d7da


``"dag"`` — Sharing-Aware Comparison
-------------------------------------

.. code-block:: python

   @py_class(structure="dag")
   class Binding(Object):
       var: Var
       value: Expr

**Meaning**: "This node lives in a graph where pointer sharing is
semantically meaningful. Two graphs are equal only if they have the same
content **and** the same sharing structure."

Why it exists
~~~~~~~~~~~~~

In dataflow IR, sharing matters. Consider:

.. code-block:: text

   # Program A: shared — compute once, use twice
   let s = x + 1 in (s, s)

   # Program B: independent — compute twice
   (x + 1, x + 1)

Program A computes ``x + 1`` once and references it twice; Program B
computes it independently twice. Under ``"tree"`` these are equal;
under ``"dag"`` they are **not**:

.. mermaid::

   graph TD
       subgraph "Program A — DAG"
           TA["(_, _)"]
           SA["s = x + 1"]
           TA -->|".0"| SA
           TA -->|".1"| SA
       end

       subgraph "Program B — Tree"
           TB["(_, _)"]
           A1["x + 1"]
           A2["x + 1"]
           TB -->|".0"| A1
           TB -->|".1"| A2
       end

       SA -. "NOT EQUAL under dag<br/>(sharing structure differs)" .-> A1

       style SA fill:#d4edda
       style A1 fill:#d4edda
       style A2 fill:#f8d7da

How ``"dag"`` detects sharing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``"dag"`` maintains a bijective (one-to-one) mapping between objects that
have been successfully compared. When the same object appears again, it
checks whether the *pairing* is consistent:

.. code-block:: text

   Comparing Program A vs Program B:

   .0:  s ↔ (x+1)₁  →  content equal, record pairing: s ↔ (x+1)₁
   .1:  s ↔ (x+1)₂  →  s already paired with (x+1)₁, not (x+1)₂
                      →  NOT EQUAL

The mapping is **bijective**: if ``a`` is paired with ``b``, no other object
can pair with either ``a`` or ``b``. This prevents false positives in both
directions.

**Example of the reverse direction.**

.. code-block:: text

   lhs: (a, b)     rhs: (a, a)     where a ≅ b (same content)

   .0: a₁ ↔ a₂  →  equal, record a₁ ↔ a₂
   .1: b₁ ↔ a₂  →  b₁ is new, but a₂ already paired with a₁
                  →  NOT EQUAL

Without the reverse check, the second comparison would proceed to content
comparison, find ``b₁ ≅ a₂``, and incorrectly succeed.

Full comparison: ``"tree"`` vs ``"dag"``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 48 13 13

   * - Scenario
     - ``"tree"``
     - ``"dag"``
   * - both trees with same content
     - Equal
     - Equal
   * - both DAGs, same sharing shape
     - Equal
     - Equal
   * - ``let s = e in (s, s)`` vs ``(e, e')`` where ``e ≅ e'``
     - Equal
     - **Not equal**
   * - ``(a, b)`` vs ``(a, a)`` where ``a ≅ b``
     - Equal
     - **Not equal**


.. _var-kind:

``"var"`` — Bound Variables
----------------------------

.. code-block:: python

   @py_class(structure="var")
   class Var(Object):
       name: str

**Meaning**: "This is a variable. Two variables are equal if they are
**bound in corresponding positions**, not if they have the same name or
content."

The problem
~~~~~~~~~~~

.. code-block:: text

   fun x → x + 1       should equal       fun y → y + 1

Variables are not defined by their content (name, type annotation). They
are defined by **where they are introduced** and **how they are used**.
``x`` and ``y`` above are interchangeable because they occupy the same
binding position and are used in the same way.

How it works: definition regions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``"var"`` works together with ``field(structure="def")`` (see
:ref:`field-annotations`). A field marked ``structure="def"`` is a
**definition region** — it's where new variable bindings are introduced.

- **Inside a definition region**: encountering two different variables
  establishes a correspondence ("treat ``x`` as equivalent to ``y``").
- **Outside a definition region**: variables are only equal if a prior
  correspondence already exists, or they are the same pointer.

The following diagram traces the comparison of two alpha-equivalent functions:

.. mermaid::

   sequenceDiagram
       participant C as Comparator
       participant L as lhs: fun x → x + 1
       participant R as rhs: fun y → y + 1

       Note over C: Field "params" has structure="def"
       C->>L: get params → [x]
       C->>R: get params → [y]
       Note over C: Enter definition region
       C->>C: Compare x ↔ y: both are Vars
       Note over C: Record mapping: x ↔ y
       Note over C: Exit definition region

       Note over C: Field "body" — normal region
       C->>L: get body → x + 1
       C->>R: get body → y + 1
       C->>C: Compare + fields...
       C->>C: x ↔ y: lookup finds x→y ✓
       C->>C: 1 ↔ 1: equal ✓
       Note over C: Result: EQUAL ✓

**Without** a definition region, the same variables would **not** be equal:

.. code-block:: text

   # Bare expressions, no enclosing function:
   x + 1  vs  y + 1   →  NOT EQUAL (no definition region, different pointers)

Full comparison: with and without definition regions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 55 22 22

   * - Scenario
     - With ``"def"``
     - Without
   * - ``fun x → x + 1`` vs ``fun y → y + 1``
     - Equal
     - n/a
   * - ``fun x → x + 1`` vs ``fun y → x + 1``
     - **Not equal** (body uses ``x`` but mapping says ``y``)
     - n/a
   * - ``fun (x, y) → x + y`` vs ``fun (a, b) → a + b``
     - Equal (x↔a, y↔b)
     - n/a
   * - ``fun (x, y) → x + y`` vs ``fun (a, b) → b + a``
     - **Not equal** (x↔a but body uses ``x`` where ``b`` appears)
     - n/a
   * - ``x + 1`` vs ``y + 1`` (bare)
     - n/a
     - **Not equal**
   * - ``x + 1`` vs ``x + 1`` (same pointer)
     - n/a
     - Equal

Inconsistent variable usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The bijective mapping catches inconsistencies. Consider:

.. code-block:: text

   fun (x, y) → x + x    vs    fun (a, b) → a + b

.. mermaid::

   sequenceDiagram
       participant C as Comparator
       participant L as lhs: fun (x, y) → x + x
       participant R as rhs: fun (a, b) → a + b

       Note over C: Definition region (params)
       C->>C: x ↔ a → record x↔a ✓
       C->>C: y ↔ b → record y↔b ✓

       Note over C: Body: x + x vs a + b
       C->>C: x ↔ a → lookup x→a, matches ✓
       C->>C: x ↔ b → lookup x→a, but rhs is b ≠ a → FAIL ✗
       Note over C: Result: NOT EQUAL ✓

The ``map_free_vars`` flag
~~~~~~~~~~~~~~~~~~~~~~~~~~

``structural_equal(lhs, rhs, map_free_vars=True)`` starts the comparison
in "definition region" mode. This is useful for comparing standalone
expressions where you want alpha-equivalence at the top level without an
enclosing function:

.. code-block:: python

   # With map_free_vars=True:
   structural_equal(x + 1, y + 1, map_free_vars=True)   # → True

   # With map_free_vars=False (default):
   structural_equal(x + 1, y + 1)                        # → False


``"singleton"`` — Singletons
------------------------------

.. code-block:: python

   @py_class(structure="singleton")
   class Op(Object):
       name: str

**Meaning**: "There is exactly one instance of this object per logical
identity. Pointer equality is the only valid comparison."

No content comparison is ever performed. Different pointers are always
unequal; same pointer is always equal.

.. code-block:: python

   op_conv = Op.get("nn.conv2d")
   op_relu = Op.get("nn.relu")

   structural_equal(op_conv, op_conv)   # → True  (same pointer)
   structural_equal(op_conv, op_relu)   # → False (different pointers)


.. _field-annotations:

Field-Level Annotations
-----------------------

The ``structure`` parameter on :func:`~tvm_ffi.dataclasses.field` controls
how structural equality/hashing treats that specific field.

``structure="ignore"`` — Exclude a field
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   @py_class(structure="tree")
   class MyNode(Object):
       value: int
       span: str = field(structure="ignore")

**Meaning**: "This field is not part of the node's structural identity.
Skip it during comparison and hashing."

Use for:

- **Source locations** (``span``) — where the node came from in source code
  doesn't affect what it means.
- **Cached/derived values** — computed from other fields, would be
  redundant to compare.
- **Debug annotations** — names, comments, metadata for human consumption.

``structure="def"`` — Definition region
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   @py_class(structure="tree")
   class Lambda(Object):
       params: list[Var] = field(structure="def")
       body: Expr

**Meaning**: "This field introduces new variable bindings. When comparing
or hashing this field, allow new variable correspondences to be
established."

This is the counterpart to ``"var"``. A ``"var"`` type says "I am a
variable"; ``structure="def"`` says "this field is where variables are
defined." Together they enable alpha-equivalence: comparing functions up
to consistent variable renaming.

Use for:

- **Function parameter lists**
- **Let-binding left-hand sides**
- **Any field that introduces names into scope**


Custom Equality and Hashing
----------------------------

For types where the default field-by-field traversal is insufficient, you
can register custom callbacks as type attributes:

- **``__s_equal__``** — custom equality logic
- **``__s_hash__``** — custom hashing logic

These are registered per type via ``EnsureTypeAttrColumn``. When present,
they replace the default field iteration. The system still manages all
kind-specific logic (``"dag"`` memoization, ``"var"`` mapping, etc.) — the
custom callback only controls which sub-values to compare/hash and in what
order.


All Kinds at a Glance
---------------------

The following diagram visualizes the five comparable kinds, arranged by how
much structural information they track:

.. mermaid::

   graph LR
       UI["singleton<br/><i>pointer only</i>"]
       TN["tree<br/><i>content only</i>"]
       CTN["const-tree<br/><i>content + pointer shortcut</i>"]
       DN["dag<br/><i>content + sharing</i>"]
       FV["var<br/><i>content + binding position</i>"]

       UI --- TN
       TN --- CTN
       TN --- DN
       TN --- FV

       style UI fill:#e2e3e5
       style TN fill:#d4edda
       style CTN fill:#d4edda
       style DN fill:#cce5ff
       style FV fill:#fff3cd

.. list-table::
   :header-rows: 1
   :widths: 18 18 18 18 18

   * -
     - Content comparison
     - Pointer shortcut
     - Tracks sharing
     - Tracks binding position
   * - ``"singleton"``
     - No
     - Yes (only)
     - No
     - No
   * - ``"tree"``
     - Yes
     - No
     - No
     - No
   * - ``"const-tree"``
     - Yes
     - Yes (fast path)
     - No
     - No
   * - ``"dag"``
     - Yes
     - No
     - Yes
     - No
   * - ``"var"``
     - Yes
     - No
     - No
     - Yes


Decision Guide
--------------

When defining a new type:

.. mermaid::

   graph TD
       Start["New @py_class type"] --> Q1{"Singleton?<br/>(one instance per<br/>logical identity)"}
       Q1 -->|Yes| UI["structure=&quot;singleton&quot;"]
       Q1 -->|No| Q2{"Represents a<br/>variable binding?"}
       Q2 -->|Yes| FV["structure=&quot;var&quot;"]
       Q2 -->|No| Q3{"Pointer sharing<br/>semantically<br/>meaningful?"}
       Q3 -->|Yes| DN["structure=&quot;dag&quot;"]
       Q3 -->|No| Q4{"Immutable AND<br/>no transitive<br/>var children?"}
       Q4 -->|Yes| CTN["structure=&quot;const-tree&quot;"]
       Q4 -->|No| TN["structure=&quot;tree&quot;"]

       style UI fill:#e2e3e5
       style FV fill:#fff3cd
       style DN fill:#cce5ff
       style CTN fill:#d4edda
       style TN fill:#d4edda

For fields:

.. mermaid::

   graph TD
       Start["field() parameter"] --> Q1{"Irrelevant to<br/>structural identity?<br/>(span, cache, debug)"}
       Q1 -->|Yes| IGN["structure=&quot;ignore&quot;"]
       Q1 -->|No| Q2{"Introduces new<br/>variable bindings?"}
       Q2 -->|Yes| DEF["structure=&quot;def&quot;"]
       Q2 -->|No| NONE["No flag needed"]

       style IGN fill:#f8d7da
       style DEF fill:#fff3cd
       style NONE fill:#d4edda


Worked Example
--------------

Putting it all together for a function node with parameters, body, and
source location:

.. code-block:: python

   @py_class(structure="tree")
   class Lambda(Object):
       params: list[Var] = field(structure="def")
       body: Expr
       span: str = field(structure="ignore", default="")

   @py_class(structure="var")
   class Var(Object):
       name: str

   @py_class(structure="singleton")
   class Op(Object):
       name: str

With these annotations, alpha-equivalent functions are structurally equal:

.. code-block:: text

   # These two are structurally equal:
   fun [x] → x + 1       (span="a.py:1")
   fun [y] → y + 1       (span="b.py:5")

   #  - params has structure="def" → x maps to y
   #  - body uses that mapping → (x + 1) ≅ (y + 1)
   #  - span has structure="ignore" → locations don't matter

And in Python:

.. code-block:: python

   from tvm_ffi import structural_equal, structural_hash

   x, y = Var("x"), Var("y")
   f1 = Lambda([x], x + 1, span="a.py:1")
   f2 = Lambda([y], y + 1, span="b.py:5")

   assert structural_equal(f1, f2)                   # alpha-equivalent
   assert structural_hash(f1) == structural_hash(f2)  # same hash
