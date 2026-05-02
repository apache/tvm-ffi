<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

---
name: object-registration
description: Reference for registering TVM-FFI objects (C++ ObjectDef, Python @register_object/@c_class, global functions, field traits, auto-init).
argument-hint: "[cpp | python | global-func | field-traits | init | example]"
---

# TVM-FFI Object Registration Reference

How to define, register, and expose TVM-FFI objects across C++ and Python.

## C++ Object Declaration

### Step 1: Obj class (data)

```cpp
class FooObj : public Object {
 public:
  int64_t x;
  String name;
  TVM_FFI_DECLARE_OBJECT_INFO("my.Foo", FooObj, Object);
};
```

Macro variants:

| Macro | Use |
|-------|-----|
| `TVM_FFI_DECLARE_OBJECT_INFO(key, Type, Parent)` | Standard — allows subclassing |
| `TVM_FFI_DECLARE_OBJECT_INFO_FINAL(key, Type, Parent)` | Sealed — no subclasses allowed |

### Step 2: Ref wrapper (handle)

```cpp
class Foo : public ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Foo, ObjectRef, FooObj);
};
```

See the **type-system** skill for nullable vs non-nullable details.

### Step 3: Register via ObjectDef

```cpp
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<FooObj>()
      .def(refl::init<int64_t, String>())            // constructor
      .def_ro("x", &FooObj::x, "The x field")        // read-only field
      .def_rw("name", &FooObj::name, "The name")      // read-write field
      .def("method", &Foo::SomeMethod, "A method")    // instance method
      .def_static("make", &Foo::Make, "Factory");      // static method
}
```

## ObjectDef API (include/tvm/ffi/reflection/registry.h)

### Field registration

| Method | Mutability | Python access |
|--------|-----------|---------------|
| `.def_ro(name, &Obj::field, ...)` | Read-only | `obj.field` (property, no setter) |
| `.def_rw(name, &Obj::field, ...)` | Read-write | `obj.field` (property with setter) |

Extra arguments after the field pointer can be: a docstring, and any number of **field traits** (see below).

### Method registration

| Method | Binding |
|--------|---------|
| `.def(name, func, ...)` | Instance method — first arg is `self` |
| `.def_static(name, func, ...)` | Static method |
| `.def(refl::init<Args...>())` | Explicit constructor |

### Constructor via ObjectDef destructor (auto-init)

If no explicit `refl::init<>()` or manual `__ffi_init__` is registered, `~ObjectDef()` auto-generates one from field metadata. This is the common case for data classes.

## Field Traits (include/tvm/ffi/reflection/registry.h)

Pass these as extra args to `def_ro`/`def_rw`:

| Trait | Effect |
|-------|--------|
| `refl::default_value(val)` | Sets default; field becomes optional in init |
| `refl::default_factory(fn)` | Default from callable (e.g., empty list) |
| `refl::init(false)` | Exclude field from auto-generated `__init__` |
| `refl::kw_only(true)` | Field is keyword-only in `__init__` |
| `refl::repr(false)` | Exclude from `__repr__` |
| `refl::compare(false)` | Exclude from structural equality/ordering |
| `refl::hash(false)` | Exclude from structural hashing |
| `refl::Metadata{{"key", value}}` | Attach arbitrary metadata |

Class-level init suppression (no `__ffi_init__` at all):

```cpp
refl::ObjectDef<FooObj>(refl::init(false))
    .def_rw("x", &FooObj::x);
// No auto-init generated; provide factory methods instead
```

### Init signature generation rules

1. Required positional fields come first (no default, not kw_only)
2. Optional positional fields next (has default, not kw_only)
3. Keyword-only fields last
4. Parent fields appear before child fields
5. Fields with `init(false)` are excluded entirely

## Type Attribute Registration

```cpp
refl::TypeAttrDef<FooObj>().def(
    refl::type_attr::kConvert,
    &refl::details::FFIConvertFromAnyViewToObjectRef<Foo>);
```

Standard type attributes:

| Constant | Name | Purpose |
|----------|------|---------|
| `kInit` | `__ffi_init__` | Constructor |
| `kShallowCopy` | `__ffi_shallow_copy__` | Copy support |
| `kRepr` | `__ffi_repr__` | String repr |
| `kHash` | `__ffi_hash__` | Recursive hash |
| `kEq` | `__ffi_eq__` | Recursive equality |
| `kCompare` | `__ffi_compare__` | Ordering |
| `kConvert` | `__ffi_convert__` | AnyView -> typed ref conversion |

## Global Function Registration

### C++ side

```cpp
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("my.add_one", [](int x) { return x + 1; })
      .def_packed("my.nop", [](PackedArgs args, Any* ret) {});
}
```

| Method | Signature style |
|--------|----------------|
| `.def(name, func)` | Typed — auto-converts args |
| `.def_packed(name, func)` | Raw packed — `(PackedArgs, Any*)` |

### Python side

```python
# As decorator
@tvm_ffi.register_global_func("my.echo")
def echo(x):
    return x

# Direct registration
tvm_ffi.register_global_func("my.add", lambda x, y: x + y)

# Retrieval
f = tvm_ffi.get_global_func("my.echo")
f = tvm_ffi.get_global_func("my.echo", allow_missing=True)  # returns None if missing
```

## Python: @register_object (python/tvm_ffi/registry.py)

Connects a Python class to an existing C++ type key.

```python
@tvm_ffi.register_object("my.Foo")
class Foo(tvm_ffi.Object):
    pass  # fields/methods auto-installed from C++ reflection
```

What it does:
1. Looks up the C++ type index from the type key
2. Registers the Python class as the wrapper for that type index
3. Installs field properties (from `def_ro`/`def_rw`)
4. Installs methods (from `.def()`/`.def_static()`)
5. Generates `__init__` from C++ `__ffi_init__` metadata

### __post_init__ support

```python
@tvm_ffi.register_object("my.Foo")
class Foo(tvm_ffi.Object):
    def __post_init__(self):
        # Called after C++ __ffi_init__ completes
        self._cache = compute(self.x)
```

## Python: @c_class (python/tvm_ffi/dataclasses/c_class.py)

Superset of `@register_object` — also installs structural dunders.

```python
@tvm_ffi.dataclasses.c_class("my.Foo", eq=True, unsafe_hash=True)
class Foo(tvm_ffi.Object):
    x: int
    name: str
```

Parameters:

| Param | Default | Effect |
|-------|---------|--------|
| `type_key` | required | C++ type key |
| `init` | `True` | Install `__init__` from C++ metadata |
| `repr` | `True` | Install `__repr__` |
| `eq` | `False` | Install `__eq__`/`__ne__` via RecursiveEq |
| `order` | `False` | Install `__lt__`/`__le__`/`__gt__`/`__ge__` |
| `unsafe_hash` | `False` | Install `__hash__` via RecursiveHash |

User-defined dunders in the class body are never overwritten.

### When to use which

| Decorator | Use for |
|-----------|---------|
| `@register_object` | Most types — just need field access and init |
| `@c_class` | Data classes needing `==`, hashing, ordering |

## Complete End-to-End Example

### C++ (src/ffi/testing/testing.cc)

```cpp
class TestIntPairObj : public tvm::ffi::Object {
 public:
  int64_t a;
  int64_t b;
  TestIntPairObj() = default;
  TestIntPairObj(int64_t a, int64_t b) : a(a), b(b) {}
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("testing.TestIntPair", TestIntPairObj, Object);
};

class TestIntPair : public tvm::ffi::ObjectRef {
 public:
  explicit TestIntPair(int64_t a, int64_t b) {
    data_ = tvm::ffi::make_object<TestIntPairObj>(a, b);
  }
  int64_t Sum() const { return get()->a + get()->b; }
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TestIntPair, ObjectRef, TestIntPairObj);
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TestIntPairObj>()
      .def(refl::init<int64_t, int64_t>())
      .def_ro("a", &TestIntPairObj::a, "Field `a`")
      .def_ro("b", &TestIntPairObj::b, "Field `b`")
      .def("sum", &TestIntPair::Sum, "Method to compute sum of a and b");
  refl::TypeAttrDef<TestIntPairObj>().def(
      refl::type_attr::kConvert,
      &refl::details::FFIConvertFromAnyViewToObjectRef<TestIntPair>);
}
```

### Python

```python
@tvm_ffi.register_object("testing.TestIntPair")
class TestIntPair(tvm_ffi.Object):
    pass

pair = TestIntPair(3, 4)
assert pair.a == 3
assert pair.b == 4
assert pair.sum() == 7
```

## Python Introspection

Access type metadata at runtime:

```python
info = obj.__tvm_ffi_type_info__
info.type_key        # "my.Foo"
info.type_index      # runtime int
info.fields          # list[TypeField]
info.methods         # list[TypeMethod]

# Per-field metadata
for f in info.fields:
    f.name           # field name
    f.frozen         # True if def_ro
    f.c_init         # participates in __init__
    f.c_kw_only      # keyword-only
    f.c_has_default  # has default value
    f.metadata       # dict from refl::Metadata
```
