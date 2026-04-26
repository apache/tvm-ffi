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
name: type-system
description: Reference for TVM-FFI's type system (Any, AnyView, Object, ObjectRef, casting, reflection).
argument-hint: "[any | object | casting | reflection | string | containers]"
---

# TVM-FFI Type System Reference

Quick reference for working with TVM-FFI's core type abstractions.

## Any / AnyView (include/tvm/ffi/any.h)

Type-erased value containers.

| Type | Owning | Use |
|------|--------|-----|
| `AnyView` | No (borrowed) | Function parameters, temporary reads |
| `Any` | Yes (ref-counted) | Return values, stored fields |

**Key methods** (both types):
- `.type_index()` -> `int32_t` runtime type tag
- `.cast<T>()` -> T, throws `TypeError` on mismatch
- `.as<T>()` -> `std::optional<T>`, no conversion, returns nullopt on mismatch
- `.as<ObjectType*>()` -> `const ObjectType*` or nullptr
- `.GetTypeKey()` -> std::string

## Object System (include/tvm/ffi/object.h)

### Hierarchy

```
Object          base class, holds TVMFFIObject header (ref count + type_index)
  ObjectPtr<T>  smart pointer, manages strong reference count
  ObjectRef     base ref wrapper, nullable, wraps ObjectPtr<Object>
```

### The Obj+Ref Pattern

Every type is a pair: `FooObj` (data) + `Foo` (ref handle).

```cpp
// FooObj: the data class
struct FooObj : public Object {
  int32_t value;
  TVM_FFI_DECLARE_OBJECT_INFO("my.Foo", FooObj, Object);
};

// Foo: the ref wrapper
struct Foo : public ObjectRef {
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Foo, ObjectRef, FooObj);
};
```

Access data through `->`: `Foo foo = ...; foo->value;`

### Nullable vs Non-Nullable

| Macro | Default ctor | _type_is_nullable | Use for |
|-------|-------------|-------------------|---------|
| `TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE` | Yes (nullptr) | true | Base/abstract refs (NodeAST, ExprAST, StmtAST) |
| `TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE` | Deleted | false | Concrete types (IdAST, CallAST, etc.) |

### UnsafeInit

`ExprAST x(ffi::UnsafeInit{})` bypasses null-check on construction. **Unnecessary for nullable types** - just default-construct: `ExprAST x;`

Required only for non-nullable types that must be assigned later (deferred init).

### Creating Objects

```cpp
ObjectPtr<FooObj> ptr = make_object<FooObj>(args...);
Foo ref(std::move(ptr));
```

## Casting (include/tvm/ffi/cast.h)

### GetRef: raw Obj* -> Ref wrapper

```cpp
// Given a const FooObj* from .as<FooObj>():
const FooObj* raw = some_ref.as<FooObj>();
Foo ref = GetRef<Foo>(raw);  // handles const_cast internally
```

### .as<>() two overloads on ObjectRef

```cpp
// Overload 1: ObjectType* (raw pointer) - for field access
if (const auto* foo = ref.as<FooObj>()) {
    use(foo->field);
}

// Overload 2: RefType (optional ref) - for passing around
if (auto foo = ref.as<Foo>()) {
    accept_foo(*foo);  // dereference std::optional
}
```

Both return nullptr/nullopt on type mismatch. The ref overload is preferred when you need a typed ref, not just field access.

### Idiomatic downcast patterns

```cpp
// GOOD: get ref directly via optional overload
if (auto stmt = node.as<StmtAST>()) {
    stmts.push_back(*stmt);
}

// GOOD: raw pointer for field access
if (const auto* func = node.as<FunctionASTObj>()) {
    use(func->name, func->args);
}

// OK but verbose: GetRef from raw pointer
if (node->IsInstance<StmtASTObj>()) {
    stmts.push_back(GetRef<StmtAST>(node.as<StmtASTObj>()));
}

// BAD: redundant IsInstance - .as<>() already checks
if (node->IsInstance<FooObj>()) {  // unnecessary
    const auto* p = node.as<FooObj>();  // checks again
}
```

### AnyView/Any casting

```cpp
AnyView v = ...;
v.cast<int64_t>();       // throws on mismatch
v.cast<ObjectRef>();     // throws on mismatch
v.cast<String>();        // throws on mismatch
v.as<Foo>();             // returns std::optional<Foo>
```

## Reflection (include/tvm/ffi/reflection/accessor.h)

### FieldGetter / FieldSetter

Higher-level field access via runtime reflection. Encapsulates the raw offset+getter pointer mechanics.

```cpp
// By field info pointer (when you already have it):
reflection::FieldGetter getter(field_info);
Any value = getter(obj.get());

// By type key + name (does lookup):
reflection::FieldGetter getter("my.Foo", "value");
Any value = getter(some_obj);

// Setter:
reflection::FieldSetter setter(field_info);
setter(obj.get(), new_value);
```

Prefer `FieldGetter`/`FieldSetter` over manual `reinterpret_cast<char*>(ptr) + offset` arithmetic.

### TypeAttrColumn

Per-type dispatch via attribute columns:

```cpp
static reflection::TypeAttrColumn col("__ffi_ir_traits__");
AnyView trait = col[obj->type_index()];
```

### ForEachFieldInfo

Iterate all fields (including inherited):

```cpp
reflection::ForEachFieldInfo(type_info, [&](const TVMFFIFieldInfo* fi) {
    Any val = reflection::FieldGetter(fi)(obj.get());
    // ...
});
```

## String (include/tvm/ffi/string.h)

**No implicit conversion to `std::string_view`**. Must extract explicitly:

```cpp
String s = ...;
std::string_view sv(s.data(), s.size());  // explicit
```

Key constructors:
```cpp
String()                          // empty
String("hello")                   // from const char*
String(ptr, size)                 // from char* + size
String(std::string("hello"))      // from std::string (copy)
String(std::move(std_str))        // from std::string (move)
```

Has `operator std::string()` for converting TO std::string.

## Containers (include/tvm/ffi/container/)

| Type | Mutability | Backing | Notes |
|------|-----------|---------|-------|
| `Array<T>` | Immutable | - | Frozen after creation |
| `List<T>` | Mutable | - | push_back, insert, erase |
| `Map<K,V>` | Immutable | hash map | Frozen key-value |
| `Dict<K,V>` | Mutable | hash map | set, erase |
| `String` | Immutable | SSO + heap | See above |
| `Tuple` | Immutable | - | Heterogeneous fixed-size |
