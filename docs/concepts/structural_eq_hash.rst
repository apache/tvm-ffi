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

Structural Equality, Hashing, Walking, and Mapping
==================================================

TVM FFI provides ``structural_equal`` and ``structural_hash`` for the
object graph. These compare objects by **content** — recursively walking
fields — rather than by pointer identity.

The same reflection metadata also drives ``structural_walk`` for analyses and
``structural_map`` for transformations.  Their low-level engines,
``StructuralVisitor`` and ``StructuralMutator``, let custom object types
participate in the same traversal protocol.

The behavior is controlled by two layers of annotation on
:func:`~tvm_ffi.dataclasses.py_class`:

1. **Type-level** ``structural_eq=`` — what *role* does this type play in the
   IR graph?
2. **Field-level** ``structural_eq=`` on :func:`~tvm_ffi.dataclasses.field` —
   should this field be skipped, or does it introduce new variable bindings?

This document explains what each annotation means, when to use it, and how
they compose.

.. note::

   Structural equality and hashing **never call Python-level** ``__eq__``
   or ``__hash__``.  ``structural_equal`` / ``structural_hash`` dispatch
   entirely through a C++ walker driven by the kind metadata registered
   via ``structural_eq=``; the Python ``a == b`` / ``hash(a)`` dunders
   are independent (they default to pointer identity and handle address,
   inherited from ``Object``).  To customize how a specific type
   participates in *structural* comparison, register the
   :ref:`sequal-shash` hooks described below — do **not** override
   ``__eq__`` or ``__hash__``.


Type-Level Annotation
---------------------

The ``structural_eq`` parameter on ``@py_class`` declares how instances of the
type participate in structural equality and hashing. It defaults to ``"tree"``,
so the following class is compared recursively by its fields:

.. code-block:: python

   @py_class
   class Expr(Object):
       ...

Specify ``structural_eq`` only when the type needs another role, such as
``"var"``, ``"dag"``, or ``"singleton"``, or pass ``None`` explicitly to opt
out of structural equality and hashing.

Quick reference
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 37 45

   * - ``structural_eq=``
     - Meaning
     - Use when...
   * - ``"tree"``
     - A regular IR node
     - Default for ``@py_class`` and most IR nodes
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
     - Explicitly opt out of structural comparison


``"tree"`` — The Default
-------------------------

.. code-block:: python

   @py_class
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

   @py_class(structural_eq="const-tree")
   class DeviceMesh(Object):
       shape: list[int]
       device_ids: list[int]

**Meaning**: "Same as ``"tree"``, but if two references point to the same
object, they are guaranteed equal — skip the field comparison."

This is purely a **performance optimization**. The only behavioral difference
from ``"tree"`` is that pointer identity short-circuits to ``True``.

When is this safe (and worth it)?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Three conditions decide whether ``"const-tree"`` is the right choice:

1. **Immutable** — content doesn't change after construction, so same-pointer
   always implies same-content.
2. **No transitive** ``"var"`` **children** — skipping field traversal won't
   cause variable mappings to be missed (see :ref:`var-kind` for why this
   matters).
3. **Sharing is common** — instances are interned or canonicalized, so the
   same pointer actually appears on both sides of real comparisons. Without
   interning, the shortcut never fires and ``"const-tree"`` behaves like
   ``"tree"`` with a dead branch.

Conditions 1 and 2 are correctness requirements: violating them is a bug,
not a performance regression. Condition 3 is the payoff — ``"const-tree"``
is worth reaching for only when it will actually save work.

A useful rule of thumb: *does the system go out of its way to make two
equal instances of this type share a pointer?* Canonical types, interned
constants, cached shapes, and op metadata usually do. General expression
and statement nodes usually don't — and also fail condition 2. Prefer
``"const-tree"`` for the type / attribute / metadata layer of the IR, not
the expression / statement layer.

Note also that condition 2 is a *whole-subgraph* property: once a field
holds an ``Expr`` (which may one day contain a ``Var``), the annotation
silently commits the type to that invariant — a later refactor embedding
a ``Var`` becomes a correctness break rather than a local change.

Why not use it everywhere?
~~~~~~~~~~~~~~~~~~~~~~~~~~

Most IR nodes are immutable, but many transitively contain variables
(e.g., ``x + 1`` contains the ``"var"`` node ``x``). The pointer shortcut
fires only when both sides of a comparison reference the **same object** —
but when that sharing exists, skipping traversal also skips the variable
occurrences inside, and mappings that should have been recorded are
silently missed.

Suppose the ``+`` node were incorrectly annotated as ``"const-tree"``, and
consider comparing two tuples that share the ``+`` subtree via pointer
identity:

.. code-block:: text

   shared = x + 1                     # pointer P, contains var x

   lhs = (shared, x)                  # .0 = P, .1 = var x
   rhs = (shared, y)                  # .0 = P, .1 = var y  (different Var)

   structural_equal(lhs, rhs, map_free_vars=True)

With the ``+`` annotated as plain ``"tree"`` (correct):

- ``.0``: traverse into ``shared`` on both sides, visit ``x`` at ``.lhs``,
  record the mapping ``x ↔ x``.
- ``.1``: look up ``x`` → maps to ``x``, but rhs is ``y``. **NOT EQUAL** ✓

With the ``+`` annotated as ``"const-tree"`` (the bug):

- ``.0``: pointer shortcut fires on ``shared`` (both sides reference P).
  Fields are skipped, ``x`` inside is never visited, no mapping is recorded.
- ``.1``: compare ``x`` vs ``y``. No existing mapping, and
  ``map_free_vars=True`` lets a new one be recorded as ``x ↔ y``.
  **EQUAL** ✗ (wrong)

The following diagram illustrates the shared structure. The ``+`` node
(``shared``) has two incoming ``.0`` edges — one from each side — which
is exactly the situation in which the pointer shortcut fires:

.. mermaid::

   graph TD
       LT["lhs: (_, _)"]
       RT["rhs: (_, _)"]
       ADD["shared = x + 1<br/>const-tree<br/><i>same pointer on both sides</i>"]
       X["x : var"]
       ONE["1"]
       Y["y : var"]

       LT -->|".0"| ADD
       RT -->|".0"| ADD
       LT -->|".1"| X
       RT -->|".1"| Y
       ADD -->|".lhs"| X
       ADD -->|".rhs"| ONE

       style ADD fill:#fff3cd
       style X fill:#f8d7da
       style Y fill:#f8d7da

The same failure mode arises whenever a shared subtree containing a
``"var"`` is compared inside any definition region (e.g., the body of a
``Lambda`` whose params field is ``structural_eq="def"``), not only under
``map_free_vars=True``.


``"dag"`` — Sharing-Aware Comparison
-------------------------------------

.. code-block:: python

   @py_class(structural_eq="dag")
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

   @py_class(structural_eq="var")
   class Var(Object):
       name: str = field(structural_eq="ignore")   # alpha-equivalent vars differ in name
       type: Type                                  # participates in equality

**Meaning**: "This is a variable. Two variables are equal if they are
**bound in corresponding positions**, not if they have the same name."
The ``name`` field is almost always marked ``structural_eq="ignore"``
because alpha-equivalent variables have different names. Other fields
such as ``type`` *are* compared — but only at the binding site (see
:ref:`var-fields`).

The problem
~~~~~~~~~~~

.. code-block:: text

   fun x → x + 1       should equal       fun y → y + 1

Variables are not defined by their content, such as their name. They
are defined by **where they are introduced** and **how they are used**.
``x`` and ``y`` above are interchangeable because they occupy the same
binding position and are used in the same way.

How it works: definition regions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``"var"`` works together with ``field(structural_eq="def")`` (see
:ref:`field-annotations`). A field marked ``structural_eq="def"`` is a
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

       Note over C: Field "params" has structural_eq="def"
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

.. _var-fields:

Fields and the sticky mapping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A ``"var"`` type still has fields, and non-ignored fields *are* compared —
but only on the **first** encounter of a var pair. Once a mapping is
recorded, subsequent occurrences look up the mapping and skip field
comparison entirely.

Take the ``Var`` declaration from the top of this section: ``name`` is
ignored, but ``type`` is not. The first time a pair of vars is seen in
a definition region, their ``type`` fields are compared and the mapping
is only established if they match. After that, the mapping is **sticky**
— later occurrences trust the correspondence regardless of those fields:

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Scenario
     - Result
   * - ``Var("x", int)`` vs ``Var("y", int)`` on first encounter
     - Fields match → mapping ``x ↔ y`` recorded → **Equal**
   * - ``Var("x", int)`` vs ``Var("y", float)`` on first encounter
     - Fields differ → **Not equal**
   * - ``Var("x", int)`` vs ``Var("y", float)`` when ``x ↔ y`` already mapped
     - Lookup succeeds → **Equal** (types are *not* rechecked)

For IRs where type consistency is part of well-formedness, this is
usually sufficient: a well-formed program uses each var with a consistent
type at every occurrence, so the first-encounter check at the binding
site covers the rest. If you truly want types re-verified at every use,
they don't belong on the ``"var"`` node — lift them into the surrounding
expression/statement node where they participate in normal ``"tree"``
comparison.

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

   @py_class(structural_eq="singleton")
   class Op(Object):
       name: str

**Meaning**: "There is exactly one instance of this object per logical
identity. Pointer equality is the only valid comparison."

No content comparison is ever performed. Different pointers are always
unequal; same pointer is always equal.

The :class:`~tvm_ffi.dataclasses.Enum` hierarchy always uses this kind by
default. This includes :class:`~tvm_ffi.dataclasses.IntEnum`,
:class:`~tvm_ffi.dataclasses.StrEnum`, and every subclass of these enum bases.
Each registered enum variant is therefore structurally equal only to that same
singleton variant.

.. code-block:: python

   op_conv = Op.get("nn.conv2d")
   op_relu = Op.get("nn.relu")

   structural_equal(op_conv, op_conv)   # → True  (same pointer)
   structural_equal(op_conv, op_relu)   # → False (different pointers)


.. _field-annotations:

Field-Level Annotations
-----------------------

The ``structural_eq`` parameter on :func:`~tvm_ffi.dataclasses.field` controls
how structural equality/hashing treats that specific field.

``structural_eq="ignore"`` — Exclude a field
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   @py_class(structural_eq="tree")
   class MyNode(Object):
       value: int
       span: str = field(structural_eq="ignore")

**Meaning**: "This field is not part of the node's structural identity.
Skip it during comparison and hashing."

Use for:

- **Source locations** (``span``) — where the node came from in source code
  doesn't affect what it means.
- **Cached/derived values** — computed from other fields, would be
  redundant to compare.
- **Debug annotations** — names, comments, metadata for human consumption.

``structural_eq="def-recursive"`` / ``"def-non-recursive"`` — Definition region
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   @py_class(structural_eq="tree")
   class Lambda(Object):
       params: list[Var] = field(structural_eq="def-recursive")
       body: Expr

**Meaning**: "This field introduces new variable bindings. When comparing
or hashing this field, allow new variable correspondences to be
established."

This is the counterpart to ``"var"``. A ``"var"`` type says "I am a
variable"; the ``"def-*"`` flags on a field say "this field is where
variables are defined." Together they enable alpha-equivalence:
comparing functions up to consistent variable renaming.

There are two flavors of definition region, distinguished by what
happens when a ``"var"`` reached through the field carries its own
sub-fields (for example, a shape annotation in the var's type):

- ``"def-recursive"`` (alias: ``"def"``) — the variable's sub-fields
  stay inside the definition region. Any free variables encountered
  in those sub-fields are themselves treated as fresh definitions at
  the same site. One example is **function parameter lists**, where
  the value var and any shape parameters in its type are co-introduced
  together at the function boundary.

- ``"def-non-recursive"`` — only the immediate variable(s) reached
  through the field bind. The variable's sub-fields are walked
  outside the definition region, so any free variables there are
  *use* references that must resolve against an outer-scope binding.
  One example is a **normal binding** whose value type references
  outer-scope shape parameters (a ``let v = expr`` where ``v``'s
  type refers to vars defined earlier).

When the distinction does not matter (no nested free vars under the
bound variable), either flavor works and ``"def-recursive"`` is the
conventional default — that's why the bare ``"def"`` alias resolves
to it.

Use for:

- **Function parameter lists** — ``"def-recursive"`` so shape
  parameters in each param's type co-introduce at the same site.
- **Normal binding left-hand sides** (let bindings, for-loop
  iterators) whose value type references outer-scope vars —
  ``"def-non-recursive"`` so those references don't rebind.
- **Any field that introduces names into scope** — pick the flavor
  that matches the binding form's contract; default to
  ``"def-recursive"`` when in doubt.


.. _sequal-shash:

Custom Equality and Hashing: ``__s_equal__`` / ``__s_hash__``
--------------------------------------------------------------

For types where the default field-by-field traversal is insufficient (for
example, fields that need to be visited in a specific order, cross-field
invariants, or sub-values that need a different ``def_region`` setting
than the declarative field flags allow), you can register custom
callbacks as **type attributes**:

- ``__s_equal__`` — custom structural equality logic.
- ``__s_hash__`` — custom structural hashing logic.

These are the *only* supported way to override structural comparison.
``structural_equal`` / ``structural_hash`` never consult Python
``__eq__`` / ``__hash__`` — those dunders serve a separate purpose
(``==`` and ``hash()``, which default to pointer identity).

When either hook is registered, it replaces the default field iteration
for that type.  All kind-specific machinery (``"dag"`` memoization,
``"var"`` mapping, the pointer shortcut of ``"const-tree"``, etc.) is
still managed by the framework — the custom callback only controls
*which* sub-values are compared or hashed, *in what order*, and *with
what* ``def_region`` flag.

Signatures
~~~~~~~~~~

``__s_equal__``:

.. code-block:: text

   (self, other, eq_cb) -> bool

   eq_cb(lhs, rhs, def_region_kind: int, field_name: str) -> bool

``__s_hash__``:

.. code-block:: text

   (self, init_hash: int, hash_cb) -> int

   hash_cb(value, init_hash: int, def_region_kind: int) -> int

The ``def_region_kind`` argument on each recursive call mirrors the
field-level ``"def-*"`` flags and controls whether the sub-value is
compared/hashed inside a definition region:

- ``0`` — not in a def region (matches ``None`` on a field).
- ``1`` — recursive def region (matches ``"def-recursive"``, alias
  ``"def"``).
- ``2`` — non-recursive def region (matches ``"def-non-recursive"``).

For back-compat with the original single-flag API, the callback also
accepts a plain ``bool``: ``True`` is treated as ``1`` (recursive) and
``False`` as ``0`` (not in a def region). The Python examples below
use ``True`` / ``False`` for that reason; pass an explicit ``2`` (or
the ``kTVMFFIDefRegionKindNonRecursive`` enum value from C++) when the
non-recursive kind is needed.

The ``field_name`` argument on ``eq_cb`` is used only for mismatch path
reporting from :py:func:`~tvm_ffi.get_first_structural_mismatch`.

Example (Python)
~~~~~~~~~~~~~~~~

.. code-block:: python

   @py_class(structural_eq="tree")
   class Lambda(Object):
       params: list
       body: Any
       comment: str  # not part of identity, but also not iterated below

       def __s_equal__(self, other, eq_cb):
           # params is a definition region; body is not.
           if not eq_cb(self.params, other.params, True, "params"):
               return False
           if not eq_cb(self.body, other.body, False, "body"):
               return False
           return True

       def __s_hash__(self, init_hash, hash_cb):
           h = hash_cb(self.params, init_hash, True)
           h = hash_cb(self.body, h, False)
           return h

The two methods must agree: if ``__s_equal__`` considers two instances
equal, ``__s_hash__`` must produce the same hash for them.

Example (C++)
~~~~~~~~~~~~~

.. code-block:: c++

   class MyNodeObj : public Object {
    public:
     Array<Var> params;
     Array<ObjectRef> body;

     bool SEqual(const MyNodeObj* other,
                 ffi::TypedFunction<bool(AnyView, AnyView, bool, AnyView)> cmp) const {
       if (!cmp(params, other->params, /*def_region=*/true, "params")) return false;
       if (!cmp(body, other->body, /*def_region=*/false, "body")) return false;
       return true;
     }

     int64_t SHash(int64_t init_hash,
                   ffi::TypedFunction<int64_t(AnyView, int64_t, bool)> hash) const {
       int64_t h = hash(params, init_hash, /*def_region=*/true);
       h = hash(body, h, /*def_region=*/false);
       return h;
     }

     static void RegisterReflection() {
       namespace refl = tvm::ffi::reflection;
       refl::ObjectDef<MyNodeObj>()
           .def_ro("params", &MyNodeObj::params)
           .def_ro("body", &MyNodeObj::body);
       refl::TypeAttrDef<MyNodeObj>()
           .def(refl::type_attr::kSEqual, &MyNodeObj::SEqual)
           .def(refl::type_attr::kSHash, &MyNodeObj::SHash);
     }

     static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
     TVM_FFI_DECLARE_OBJECT_INFO_FINAL("my.Node", MyNodeObj, Object);
   };

See :cpp:var:`tvm::ffi::reflection::type_attr::kSEqual` and
:cpp:var:`tvm::ffi::reflection::type_attr::kSHash` in
``include/tvm/ffi/reflection/accessor.h`` for the full reference.


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
       Start["New non-enum @py_class type"] --> Q1{"Singleton?<br/>(one instance per<br/>logical identity)"}
       Q1 -->|Yes| UI["structural_eq=&quot;singleton&quot;"]
       Q1 -->|No| Q2{"Represents a<br/>variable binding?"}
       Q2 -->|Yes| FV["structural_eq=&quot;var&quot;"]
       Q2 -->|No| Q3{"Pointer sharing<br/>semantically<br/>meaningful?"}
       Q3 -->|Yes| DN["structural_eq=&quot;dag&quot;"]
       Q3 -->|No| Q4{"Immutable AND<br/>no transitive<br/>var children?"}
       Q4 -->|Yes| CTN["structural_eq=&quot;const-tree&quot;"]
       Q4 -->|No| TN["structural_eq=&quot;tree&quot;"]

       style UI fill:#e2e3e5
       style FV fill:#fff3cd
       style DN fill:#cce5ff
       style CTN fill:#d4edda
       style TN fill:#d4edda

Enum types do not need this decision process: :class:`~tvm_ffi.dataclasses.Enum`
and all of its subclasses default to ``"singleton"``.

For fields:

.. mermaid::

   graph TD
       Start["field() parameter"] --> Q1{"Irrelevant to<br/>structural identity?<br/>(span, cache, debug)"}
       Q1 -->|Yes| IGN["structural_eq=&quot;ignore&quot;"]
       Q1 -->|No| Q2{"Introduces new<br/>variable bindings?"}
       Q2 -->|Yes| DEF["structural_eq=&quot;def&quot;"]
       Q2 -->|No| NONE["No flag needed"]

       style IGN fill:#f8d7da
       style DEF fill:#fff3cd
       style NONE fill:#d4edda


Worked Example
--------------

Putting it all together for a function node with parameters, body, and
source location:

.. code-block:: python

   @py_class(structural_eq="tree")
   class Lambda(Object):
       params: list[Var] = field(structural_eq="def")
       body: Expr
       span: str = field(structural_eq="ignore", default="")

   @py_class(structural_eq="var")
   class Var(Object):
       name: str = field(structural_eq="ignore")

   @py_class(structural_eq="singleton")
   class Op(Object):
       name: str

With these annotations, alpha-equivalent functions are structurally equal:

.. code-block:: text

   # These two are structurally equal:
   fun [x] → x + 1       (span="a.py:1")
   fun [y] → y + 1       (span="b.py:5")

   #  - params has structural_eq="def" → x maps to y
   #  - body uses that mapping → (x + 1) ≅ (y + 1)
   #  - span has structural_eq="ignore" → locations don't matter

And in Python:

.. code-block:: python

   from tvm_ffi import structural_equal, structural_hash

   x, y = Var("x"), Var("y")
   f1 = Lambda([x], x + 1, span="a.py:1")
   f2 = Lambda([y], y + 1, span="b.py:5")

   assert structural_equal(f1, f2)                   # alpha-equivalent
   assert structural_hash(f1) == structural_hash(f2)  # same hash


Structural Walk and Map
------------------------------

Structural walk and map use the same type metadata, field flags, and
container registrations as structural equality and hashing.  The default
reflected traversal visits only structural fields, skips fields marked
``structural_eq="ignore"``, and preserves definition-region information from
fields marked as definitions.

There are two layers of API:

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - API
     - Purpose
     - Typical use
   * - :func:`~tvm_ffi.structural_walk`
     - Inspect a value graph without replacing values
     - Collect information, validate IR, or stop at a match
   * - :func:`~tvm_ffi.structural_map`
     - Recursively replace values and rebuild changed paths
     - Rewriting and compiler transformation passes
   * - :class:`~tvm_ffi.StructuralVisitor`
     - Low-level recursive visit engine
     - Implementing ``__s_visit__`` or a language binding
   * - :class:`~tvm_ffi.StructuralMutator`
     - Low-level recursive mutation engine
     - Implementing custom mutation hooks and identity substitution

``structural_walk`` and ``structural_map`` construct the corresponding low-level
object, install callback-aware dispatch, run it on the root, and return the final
result.  Applications normally use these two functions directly.  Custom object
hooks receive the low-level visitor or mutator so that recursive calls remain in
the same traversal.

StructuralVisitor and StructuralMutator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A :class:`~tvm_ffi.StructuralVisitor` carries recursive dispatch, the current
definition-region kind, and any early-interruption state.  Its main operations
are:

- ``visitor.visit(value)`` visits a child with the same visitor.
- ``visitor.def_region_kind()`` reports the active definition-region kind.
- ``visitor.with_def_region_kind(kind, callback)`` temporarily changes that kind
  while ``callback`` performs recursive visits.

The default visitor dispatches to a type's ``__s_visit__`` hook when present.
Otherwise POD values are leaves and object-backed values are visited through
their reflected structural fields.  Array, List, Map, and Dict have built-in
hooks that visit their elements or key-value pairs.

A :class:`~tvm_ffi.StructuralMutator` adds ownership and replacement semantics.
Its main operations are:

- ``mutator.mutate(value)`` maps without intentionally modifying ``value``.
- ``mutator.maybe_inplace_mutate(value)`` permits a type-specific implementation
  to reuse a safely mutable value and otherwise falls back to ``mutate``.
- ``mutator.var_remap_get(var)`` and ``mutator.var_remap_set(var, mapped)``
  access the current identity-substitution environment.
- ``def_region_kind`` and ``with_def_region_kind`` have the same role as on the
  visitor.

For reflected objects, ``mutate`` starts from a shallow copy, recursively maps
each structural field, and installs mapped fields in that copy.  If no field
changes, it returns the original object instead.  A nested change therefore
copies only the objects along the changed path; unchanged children remain shared.
``maybe_inplace_mutate`` is an explicit optimization path.  A type-specific
``__s_maybe_inplace_mutate__`` hook owns the safety policy and may reuse its
input.  Without that hook, the default implementation calls ``mutate``.

.. note::

   Visitor and mutator instances are supplied by an active traversal.  Python
   code normally receives them as arguments to ``__s_visit__``, ``__s_mutate__``,
   or ``__s_maybe_inplace_mutate__`` rather than constructing them directly.

Structural Walk
~~~~~~~~~~~~~~~

:func:`~tvm_ffi.structural_walk` invokes an analysis callback at each matching
value.  Callback entries are ordered, and only the first matching entry runs.
A walk is post-order by default.
A callback may return:

- :attr:`~tvm_ffi.WalkResult.ADVANCE` or ``None`` to continue.
- :attr:`~tvm_ffi.WalkResult.SKIP` to skip the current value's children.  This is
  primarily useful with pre-order traversal.
- :class:`~tvm_ffi.VisitInterrupt` to stop the entire traversal and return a
  payload.

For example, the following analysis records integer leaves and stops at the
first negative value:

.. code-block:: python

   import tvm_ffi

   integers = []

   def visit_int(value):
       integers.append(value)
       if value < 0:
           return tvm_ffi.VisitInterrupt(value)
       return tvm_ffi.WalkResult.ADVANCE

   interrupted = tvm_ffi.structural_walk(
       function,
       (int, visit_int),
   )

   if interrupted is not None:
       print("first negative value:", interrupted.value)

Walking never replaces a value.  Side effects should be limited to the
analysis state owned by the callback.

Structural Map
~~~~~~~~~~~~~~

:func:`~tvm_ffi.structural_map` uses the same typed callback selection but each
callback returns the mapped value: either its input unchanged or a replacement.
Mapping always visits all structural children; there is no ``SKIP`` or
``VisitInterrupt`` result.  A callback exception aborts the transformation and
is propagated with structural visit context.  Mapping is post-order by default,
so a callback receives a value whose children have already been mapped.

Post-order is natural for bottom-up compiler rewrites because children have
already been mapped when the callback runs:

.. code-block:: python

   def fold_add(add):
       if isinstance(add.lhs, IntImm) and isinstance(add.rhs, IntImm):
           return IntImm(add.lhs.value + add.rhs.value)
       return add

   optimized = tvm_ffi.structural_map(
       function,
       (Add, fold_add),
   )

Map callbacks must follow map semantics: they must not mutate their input in
place.  A node with a matching callback is always processed by ``mutate`` so
that the callback observes or returns a mapped value.

Callback Selection and Order
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both Python functions accept the same callback forms:

- ``(Type, callback)`` for one type.
- ``((TypeA, TypeB), callback)`` to share one callback across types.
- A sequence of callback entries.
- A bare callable as a ``typing.Any`` catch-all.

``typing.Any`` and ``object`` match both POD and object-backed values.
``tvm_ffi.Object`` matches only object-backed FFI values.  Entries are tested in
the order supplied, so place specific types before broad catch-all callbacks.

Both APIs default to post-order.  The ``order`` argument controls the
relationship between callbacks and children:

- In pre-order, a walk callback runs before the children.  For mapping, the
  callback result becomes the value whose children are subsequently mapped.
- In post-order, children are processed first.  A map callback therefore receives
  the value with its mapped children already installed.

Definition Regions
~~~~~~~~~~~~~~~~~~

Callbacks passed through ``with_def_region_kind`` receive
``(value, def_region_kind)``.  The kind is one of:

- ``DefRegionKind.NONE`` for an ordinary use.
- ``DefRegionKind.DEF_RECURSIVE`` for a recursive definition region.
- ``DefRegionKind.DEF_NON_RECURSIVE`` for a non-recursive definition.

The field annotations described earlier in this document establish these
regions.  Recursive definitions propagate the definition mode into the defined
value's children.  A non-recursive definition applies to the FreeVar identity
itself, while its unannotated children are treated as ordinary uses.  Explicit
nested definition annotations establish their own region.

.. code-block:: python

   uses = []

   tvm_ffi.structural_walk(
       function,
       with_def_region_kind=(
           Var,
           lambda var, kind: (
               uses.append(var)
               if kind == tvm_ffi.DefRegionKind.NONE
               else None
           ),
       ),
   )

``structural_map`` accepts the same def-region-aware callback form, but the
callback must return the mapped value.

Custom Visit and Mutation Hooks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A type with non-standard child storage can define ``__s_visit__``.  The hook
receives the active visitor and the current value, recursively visits every
structural child, and returns an interrupt if one occurs:

.. code-block:: python

   @staticmethod
   def __s_visit__(visitor, value):
       return visitor.visit(value.children)

A custom ``__s_mutate__`` hook similarly receives the active mutator.  It should
recursively call ``mutator.mutate`` and return a new value only when needed.
An optional ``__s_maybe_inplace_mutate__`` hook may implement an in-place
optimization, but it owns the complete safety policy and receives no automatic
uniqueness guarantee.  A type defining it must also define ``__s_mutate__``.  If
the optional hook is absent, ``maybe_inplace_mutate`` delegates to ``mutate``;
generic reflected fields are never mutated in place automatically.

When an object marked ``structural_eq="var"`` that represents a FreeVar identity
registers either ``__s_mutate__`` or ``__s_maybe_inplace_mutate__`` hooks, it should:

1. call ``var_remap_get`` before recursively mutating the value;
2. immediately return the mapped value on a hit;
3. compute the final result on a miss; and
4. call ``var_remap_set`` with that final result before returning it.

C++ APIs
~~~~~~~~

C++ provides typed counterparts.  Callback dispatch uses the first argument
type and accepts an optional second ``TVMFFIDefRegionKind`` argument.  The
``Expected`` forms report failures without throwing:

.. code-block:: cpp

   Expected<Optional<VisitInterrupt>> walked =
       StructuralWalkExpected<WalkOrder::kPreOrder>(
           root,
           [&](const Add& add) -> Expected<WalkResult> {
             ++num_adds;
             return WalkResult::Advance();
           });

   Expected<Any> mapped = StructuralMapExpected<WalkOrder::kPostOrder>(
       root,
       [&](const Add& add) -> Expected<Any> {
         return FoldAdd(add);
       });

Walk callbacks return ``Expected<WalkResult>``.  Map callbacks return
``Expected<Any>`` and must obey the same non-in-place callback contract as the
Python API.  ``StructuralWalk`` and ``StructuralMap`` are the corresponding
throwing convenience forms.
