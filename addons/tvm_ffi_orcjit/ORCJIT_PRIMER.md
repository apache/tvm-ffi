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

# ORC JIT v2 Primer — Background for the `tvm_ffi_orcjit` Addon

This document explains the background knowledge needed to understand the
`tvm_ffi_orcjit` addon: object file formats, the classical linker model, and LLVM's
ORC JIT v2 architecture. It then maps those concepts onto the addon's implementation.

---

## 1. Object File Formats

A **compiled object file** (`.o` / `.obj`) is not an executable. It is an intermediate
container that holds machine code, data, and metadata about unresolved references. The
three dominant formats are:

| Format | Platform | File extensions |
| --- | --- | --- |
| **ELF** (Executable and Linkable Format) | Linux, most Unix | `.o`, `.so`, `.elf` |
| **Mach-O** (Mach Object) | macOS, iOS | `.o`, `.dylib`, `.macho` |
| **COFF/PE** (Common Object File Format / Portable Executable) | Windows | `.obj`, `.dll`, `.exe` |

Despite surface differences, all three share the same conceptual structure:

```text
┌─────────────────────────────────┐
│ File Header                     │  magic, target arch, section count
├─────────────────────────────────┤
│ Section Headers                 │  name, type, file offset, size, flags
├─────────────────────────────────┤
│ .text        (code)             │  machine instructions
│ .rodata      (read-only data)   │  string literals, constants
│ .data        (writable data)    │  initialized globals
│ .bss         (zero-init data)   │  uninitialized globals (no file bytes)
│ .init_array  (constructors)     │  array of function pointers — C++ ctors
│ .fini_array  (destructors)      │  array of function pointers — C++ dtors
│ .eh_frame    (unwind info)      │  exception/stack-unwind tables
│ … other sections …              │
├─────────────────────────────────┤
│ Symbol Table                    │  name → (section, offset, binding, type)
├─────────────────────────────────┤
│ Relocation Tables               │  (section, offset, symbol, type, addend)
└─────────────────────────────────┘
```

### Symbols

A **symbol** is a named location in the file — a function entry point, a global variable,
or a section boundary. Symbols have:

- **binding**: `LOCAL` (file-private), `GLOBAL` (externally visible), `WEAK` (override-able default)
- **definition status**: *defined* (has an address in this file) vs. *undefined* (imported — must be
  resolved at link time)

### Relocations

A **relocation** is a "fixup recipe" stored alongside the machine code. It says:

> "At byte offset X in section S, patch in the address of symbol FOO, using formula R."

The formula R is a **relocation type** that encodes what arithmetic to apply:

| Type | Meaning |
| --- | --- |
| `R_X86_64_64` | Absolute 64-bit address of the symbol |
| `R_X86_64_PC32` | Symbol address minus the patch location (PC-relative 32-bit) |
| `R_X86_64_PLT32` | PC-relative call through the PLT (procedure linkage table) |
| `IMAGE_REL_AMD64_ADDR32NB` (COFF) | Address relative to image base (Pointer32NB) |

The linker (or JIT) processes these tables to produce the final binary.

### Platform-Specific Initialization Sections

C++ global constructors and destructors must run before/after `main`. Compilers encode
them as arrays of function pointers in special sections:

| Platform | Constructors | Destructors | Notes |
| --- | --- | --- | --- |
| ELF (Linux) | `.init_array` | `.fini_array` | Priority suffix `.init_array.NNN` (lower = earlier) |
| ELF (legacy) | `.ctors` | `.dtors` | Older GCC style |
| Mach-O | `__DATA,__mod_init_func` | `__DATA,__mod_term_func` | Processed by dyld |
| COFF (Windows) | `.CRT$XCU` (default) | `.CRT$XTZ` | Suffix encodes priority via ASCII ordering |

The linker or OS loader is responsible for iterating these arrays and calling each pointer
in priority order before handing control to user code. A JIT that loads `.o` files must
replicate this behavior itself — this is a key responsibility of `tvm_ffi_orcjit`.

---

## 2. The Classical Linker Model

A **static linker** (`ld`, `link.exe`) takes multiple object files and libraries and
produces a single loadable image (executable or shared library). The process is:

```text
Object files + Archives
        │
        ▼
┌───────────────────┐
│ Symbol Resolution │  Match every undefined symbol to a definition.
│                   │  Archive (.a) members pulled in on demand.
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ Section Merging   │  All .text sections → one .text; same for .data, etc.
│                   │  Assign virtual addresses (VMA) to each section.
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ Relocation        │  Apply each relocation fixup now that all VMAs are known.
│                   │  Patch raw bytes in the output image.
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ Output Emission   │  Write ELF/PE/Mach-O header + sections + dynamic table.
└───────────────────┘
```

### Dynamic Linking

At **load time** (when `dlopen` / `LoadLibrary` runs), the OS dynamic linker
(`ld.so`, `dyld`, `ntdll`) performs a *reduced* version of the above:

1. Map shared library sections into the process address space.
2. Resolve cross-library symbol references (PLT/GOT stubs).
3. Apply load-time relocations for position-dependent code.
4. Run `.init_array` / `__mod_init_func` constructors.

### Why JIT Is Different

A JIT allocates memory at arbitrary addresses *at runtime*. It cannot use precomputed
link-time addresses. Every relocation must be re-evaluated against the JIT-allocated
addresses. Furthermore, if the JIT loads object files incrementally (one at a time),
symbol resolution must be deferred until all relevant objects are present.

---

## 3. LLVM ORC JIT v2

**ORC** stands for *On Request Compilation*. LLVM ORC JIT v2 (introduced in LLVM 9,
stabilized in LLVM 13+) is a complete redesign of LLVM's JIT infrastructure. It is
designed to be composable, asynchronous, and correct for production use (unlike the
older `MCJIT` which had several fundamental limitations around multi-module linking).

### 3.1 Core Concepts

#### ExecutionSession

`llvm::orc::ExecutionSession` is the root object of any ORC JIT instance. It owns:

- A set of `JITDylib`s (the symbol namespaces).
- The dispatch mechanism for asynchronous compilation tasks.
- The global symbol interning table (maps string → `SymbolStringPtr`).

Think of it as the "JIT process" — one per logical JIT environment.

#### JITDylib (JIT Dynamic Library)

A `JITDylib` is a **symbol namespace** that loosely mirrors a shared library. It:

- Holds a **symbol table**: `name → {flags, address}`.
- Has a **link order** (a list of other `JITDylib`s to search for unresolved symbols).
- Can be populated via *materialization units* (object files, LLVM IR, inline asm, etc.).

Multiple `JITDylib`s can coexist in one `ExecutionSession`, enabling isolation: e.g.,
one library per compiled kernel, sharing a common runtime library.

```text
ExecutionSession
  ├── JITDylib "main"           ← default, process symbols
  ├── JITDylib "libA"           ← user kernel A
  │     link_order: [libA, libB, main]
  └── JITDylib "libB"           ← user kernel B (shared by A)
```

#### LLJIT

`llvm::orc::LLJIT` is the high-level, batteries-included wrapper around
`ExecutionSession`. It:

- Sets up a target machine and data layout.
- Creates a default `JITDylib` ("main") with process symbol support.
- Configures the linking pipeline (see below).
- Exposes `addObjectFile()`, `addIRModule()`, and `lookup()`.

`LLJIT` is what `tvm_ffi_orcjit` wraps in `ORCJITExecutionSessionObj`.

#### MaterializationUnit and MaterializationResponsibility

A **MaterializationUnit** is a lazy producer of symbols. When a symbol is first looked
up and not yet defined, ORC triggers its materialization unit to produce the definition
asynchronously. This is the "on request" in ORC.

An object file becomes a `StaticLibraryDefinitionGenerator` or `ObjectLayer`-level unit:
the object is parsed, linked, and its symbols resolved only when someone asks for them.

#### Layers

ORC processes objects through a pipeline of **layers**, each transforming the input:

```text
addObjectFile(buffer)
       │
       ▼
┌─────────────────────┐
│ ObjectTransformLayer│  (optional) transform raw object bytes before linking
│  e.g. strip .pdata  │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ ObjectLinkingLayer  │  runs JITLink to:
│  (uses JITLink)     │    • parse object format
│                     │    • resolve symbols across JITDylibs
│                     │    • apply relocations
│                     │    • allocate + write JIT memory
└────────┬────────────┘
         │
         ▼
   JITDylib symbol table updated; code is live
```

#### JITLink

`JITLink` is LLVM's low-level, graph-based linker used inside `ObjectLinkingLayer`. It:

1. **Parses** the object file into an in-memory `LinkGraph` (nodes = sections/atoms,
   edges = relocations).
2. **Runs pass pipelines** (pre-prune, post-allocation, post-fixup) — plugins can
   inspect and modify the graph at each stage.
3. **Allocates** JIT memory (code + data) via a `JITLinkMemoryManager`.
4. **Resolves** relocations using the current `ExecutionSession` symbol lookup.
5. **Finalizes** by writing machine code into the allocated pages and marking them
   executable.

The pass pipeline is where `tvm_ffi_orcjit`'s `InitFiniPlugin` does its work.

### 3.2 Symbol Lookup and Resolution

When `session.lookup(search_order, symbol_name)` is called:

```text
1. Check each JITDylib in search_order, in order.
2. If symbol is defined → return its ExecutorAddr.
3. If symbol is in a MaterializationUnit not yet realized → trigger materialization.
4. Materialization runs JITLink for the relevant object → symbol becomes defined.
5. Return the address.
```

Symbol names are **mangled** (C++ name mangling) or explicitly prefixed. `tvm_ffi_orcjit`
uses the `__tvm_ffi_` prefix to namespace exported functions.

### 3.3 DefinitionGenerators

A `DefinitionGenerator` is a fallback attached to a `JITDylib`. When a symbol is not
found in the dylib's own table, the generator is invoked to dynamically create a
definition — typically by looking up a symbol in the host process or another library.

`LLJIT` attaches a `DynamicLibrarySearchGenerator` ("ProcessSymbols") to the main
dylib, which resolves symbols like `malloc`, `printf`, or any C runtime function by
looking them up in the running process.

On Windows, `tvm_ffi_orcjit` adds a custom `DLLImportDefinitionGenerator` that handles
`__imp_XXX` import stubs that MSVC-compiled objects expect.

### 3.4 Platform Support

#### What an ORC Platform Is

When the OS dynamic linker loads a shared library, it does more than just map bytes
into memory — it runs constructors, registers `atexit` handlers, and sets up
thread-local storage. An **ORC platform** is the JIT-side object that replicates this
OS loader behavior for JIT-linked code.

Concretely, an ORC platform:

1. **Intercepts `__cxa_atexit`**: C++ objects with destructors register their cleanup
   via `__cxa_atexit`. The platform installs an interposer so these registrations are
   scoped to a `JITDylib` and can be drained when that dylib is torn down — instead of
   running at process exit like normal.
2. **Drives initialization**: After an object file is linked into a `JITDylib`, calling
   `jit_->initialize(dylib)` asks the platform to run constructors (e.g., iterate
   `__mod_init_func` on macOS, `.init_array` on ELF).
3. **Drives deinitialization**: `jit_->deinitialize(dylib)` drains the `atexit`
   handlers registered during step 2 and runs destructors.

This requires a small **ORC runtime library** (part of LLVM's `compiler-rt`) to be
compiled for the target and loaded into the JIT — it provides the actual
`__cxa_atexit` interposer and initialization trampolines that the platform object
coordinates with.

The three platform objects in LLVM are:

| Platform | OS | Init section driven |
| --- | --- | --- |
| `MachOPlatform` | macOS / iOS | `__DATA,__mod_init_func` |
| `ELFNativePlatform` | Linux / ELF | `.init_array`, TLS |
| `COFFPlatform` | Windows | `.CRT$XC*` init, `__cxa_atexit` interop |

`ExecutorNativePlatform` is a convenience builder that auto-selects the right platform
for the host OS and loads the ORC runtime from a given path.

#### How `tvm_ffi_orcjit` Uses (or Avoids) ORC Platforms

The addon takes a different approach on each platform:

- **macOS**: ORC platform support is *optional*. When the caller passes an ORC runtime
  path to `ExecutionSession`, `ExecutorNativePlatform` activates `MachOPlatform`.
  `jit_->initialize(dylib)` and `jit_->deinitialize(dylib)` then drive `__mod_init_func`
  and `__cxa_atexit` teardown natively. Without the path, the addon falls back to its
  own `InitFiniPlugin`.
- **Windows**: `COFFPlatform` is skipped entirely because it requires MSVC CRT symbols
  (`_CxxThrowException`, RTTI vtables, iostream objects) that are not resolvable in
  the JIT context. Instead, `InitFiniPlugin` manually handles `.CRT$XC*` / `.CRT$XT*`
  init/fini sections.
- **Linux**: `ELFNativePlatform` is not used. `InitFiniPlugin` handles `.init_array` /
  `.fini_array` / `.ctors` / `.dtors` directly, without the ORC runtime.

---

## 4. How `tvm_ffi_orcjit` Uses ORC JIT v2

With the background above, here is how the addon maps onto the concepts.

### 4.1 Object Model

```text
Python / C++ API                   LLVM ORC v2 concept
─────────────────────────────────────────────────────────────────
default_session()                  shared (leaked) LLJIT + ExecutionSession
ExecutionSession                   LLJIT + ExecutionSession
session.load_module(objs)          createJITDylib + addObjectFile(s) + wire imports
(returned) Module                  JITDylib, wrapped as a tvm_ffi.Module
module.get_function("add")         ExecutionSession.lookup("__tvm_ffi_add")
```

`session.load_module(objects)` is the sole public loader: it creates one
`JITDylib`, adds every object (path or in-memory bytes), injects context symbols
eagerly, expands any embedded library binary into an import tree, and returns a
plain `tvm_ffi.Module`. There is no incremental library API — a module is a
unit: load all its objects at once, then look up functions on the result.

### 4.2 Loading and lookup

`load_module` adds each object to the JITDylib, and JITLink parses it into a
`LinkGraph`, resolves relocations, and allocates executable JIT memory.
`get_function` looks the symbol up (materializing lazily), then wraps the raw
pointer as a `tvm_ffi::Function`. Symbols resolve against the dylib's own
default link order (this dylib → Platform → process/runtime symbols); objects
that reference each other must be loaded together, since there is no linking
between separate `load_module` results.

Two addon-specific pieces sit in this pipeline:

- **`InitFiniPlugin`** — a JITLink pass plugin that keeps init/fini sections
  (`.init_array`/`.ctors`/`.fini_array`/`.dtors`, `__mod_init_func`, `.CRT$XC*`)
  live, then collects their function pointers after fixup. The addon runs them
  in priority order at first lookup and at teardown, replacing the ORC
  platform's initializer machinery. See `llvm_patches/init_fini_plugin.h`.
- **Windows DLL import stubs** — `DLLImportDefinitionGenerator` resolves
  `__imp_XXX` references to host-DLL functions by emitting JIT-memory pointer +
  trampoline stubs, keeping `PCRel32` fixups within ±2 GB of the JIT code. See
  `llvm_patches/win_dll_import_generator.h`.

---

## 5. End-to-End Example

```python
import tvm_ffi_orcjit as oj

# 1. Get the shared ExecutionSession (wraps LLJIT; created once per process)
sess = oj.default_session()

# 2. Load a compiled object file into a fresh JITDylib
#    → object parsed, JITLink links it, InitFiniPlugin collects ctors,
#      context symbols injected eagerly, embedded binary (if any) expanded
mod = sess.load_module("add.o")   # returns a tvm_ffi.Module

# 3. Look up and call a function
#    → LLVM resolves "__tvm_ffi_add"; pending constructors fire on first lookup
result = mod.add(3, 4)   # → 7
```

The corresponding C++ side of `add.o`:

```cpp
// add.cc  — compiled to add.o with clang++ -c -O2 add.cc
#include <tvm/ffi/function.h>

static tvm::ffi::Function add_impl = [](int a, int b) { return a + b; };

// Exports symbol "__tvm_ffi_add" using TVMFFISafeCallType ABI
TVM_FFI_DLL_EXPORT_TYPED_FUNC(add, add_impl);
```

---

## 6. Key Concepts Summary

| Concept | What it is | Where in the addon |
| --- | --- | --- |
| Object file | Container of machine code, data, symbols, relocations | Input to `session.load_module()` |
| Relocation | Recipe to patch a code address at link/JIT time | Applied by JITLink |
| `.init_array` / `.ctors` | Array of C++ constructor pointers in ELF objects | Collected by `InitFiniPlugin` |
| `ExecutionSession` | Root of the ORC JIT environment | `ORCJITExecutionSessionObj` |
| `LLJIT` | High-level ORC JIT wrapper | Stored in `ORCJITExecutionSessionObj::jit_` |
| `JITDylib` | Symbol namespace / virtual shared library | `ORCJITDynamicLibraryObj::dylib_` |
| `JITLink` | LLVM's JIT-aware linker | Used inside `ObjectLinkingLayer` |
| JITLink pass pipeline | Pre-prune → post-alloc → post-fixup hooks | Where `InitFiniPlugin` runs |
| `DefinitionGenerator` | Fallback symbol provider | `DLLImportDefinitionGenerator` (Win) |
| Link order | Search path across JITDylibs for symbol resolution | LLJIT default (Main → Platform → ProcessSymbols) |
| `__tvm_ffi_` prefix | Namespace for TVM-FFI exported functions | Used in `GetFunction()` |

---

## 7. Further Reading

- LLVM ORC JIT documentation: <https://llvm.org/docs/ORCv2.html>
- JITLink design: <https://llvm.org/docs/JITLink.html>
- ELF specification: <https://refspecs.linuxfoundation.org/elf/elf.pdf>
- PE/COFF specification: <https://learn.microsoft.com/en-us/windows/win32/debug/pe-format>
- Mach-O reference: <https://developer.apple.com/documentation/kernel/mach-o_file_format_reference>
- Ian Lance Taylor's linker series (20-part blog): foundational reading on linkers
