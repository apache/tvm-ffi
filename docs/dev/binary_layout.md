<!--
  Licensed to the Apache Software Foundation (ASF) under one
  or more contributor license agreements.  See the NOTICE file
  distributed with this work for additional information
  regarding copyright ownership.  The ASF licenses this file
  to you under the Apache License, Version 2.0 (the
  "License"); you may not use this file except in compliance
  with the License.  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing,
  software distributed under the License is distributed on an
  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  KIND, either express or implied.  See the License for the
  specific language governing permissions and limitations
  under the License.
-->

# Binary Layout: Hot/Cold Code Separation

## Goal

Group functions that only run on error / setup / teardown paths into a
separate cold region of `.text`, away from the hot ABI dispatch and
container code that takes the bulk of execution time. The cold region
sits at one end of `.text`; the hot region fills the remaining bulk and
benefits from better instruction-cache locality.

Two ingredients enable this:

1. A portable `TVM_FFI_COLD_CODE` attribute (`[[gnu::cold]]` on
   GCC/Clang, no-op on MSVC) marking individual functions.
2. The `-ffunction-sections` compile flag, which emits each function
   into its own ELF section so the linker can place cold-marked
   functions independently. The default GNU linker script gathers
   `.text.unlikely.*` into a contiguous slot inside `.text`. No
   linker-side `--gc-sections` is needed (and is deliberately
   not enabled — see *Why no `--gc-sections`* below).

`TVM_FFI_UNLIKELY` (a thin wrapper around `__builtin_expect`) is
applied to a few representative error-check branches to give the
optimizer a hint to keep the hot fall-through contiguous and let the
out-of-line error block be split into the cold section.

## Measurements

All three builds: Release, GCC 11.4, ld.bfd 2.38, x86_64 Linux.
Sizes are for the stripped shared library.

| Build    | Stripped size | Delta vs baseline | `.text` size |
| -------- | ------------: | ----------------: | -----------: |
| baseline |     1,887,800 |                 - |    1,499,450 |
| seed     |     1,830,472 |   -57,328 (-3.0%) |    1,452,947 |
| final    |     1,826,368 |   -61,432 (-3.3%) |    1,452,691 |

* baseline — `apache/tvm-ffi` `main` at `d4cfd86`, no annotations and
  no `-ffunction-sections`.
* seed — `TVM_FFI_COLD_CODE` defined, `ErrorBuilder` ctors and dtor
  marked cold, `-ffunction-sections` enabled on `tvm_ffi_objs` /
  `tvm_ffi_extra_objs`.
* final — also marks the `TVMFFIError*` C ABI helpers and the
  `SafeCallContext::Set*` /`MoveFromRaised` methods cold, and adds
  `TVM_FFI_UNLIKELY` to (a) the `TVM_FFI_CHECK_SAFE_CALL` macro,
  (b) the `TVM_FFI_CHECK` macro, and (c) the
  `GlobalFunctionTable::Update` already-registered branch.

The size drop comes almost entirely from `-ffunction-sections`. With
per-function sections the linker can pack functions tightly without
the inter-function padding that an in-TU `.text` requires for
alignment between adjacent functions. There is no source-level dead
code being removed (and `--gc-sections` is not in use).

### Hot vs cold symbol layout (baseline)

```text
0x000084f1 t TVMFFIBacktrace.cold              <- only the .cold split-off
0x0000adfa t TVMFFIErrorSetRaisedFromCStr.cold <- only the .cold split-off
...
0x00032e00 T TVMFFIBacktrace                   <- hot body in middle of .text
0x00039f90 t ErrorBuilder ctor                 <- in middle of .text
0x0003c260 t ErrorBuilder dtor                 <- in middle of .text
0x00057cd0 T TVMFFIErrorSetRaisedFromCStr      <- interleaved with C ABI
0x00059020 T TVMFFIErrorSetRaised
```

In baseline GCC's automatic cold-splitting pass already produces
`.cold` partial-function sections (the early-region cluster at
`0x84f1..0xaebe`). What it does NOT do is pull the full bodies of
`ErrorBuilder` ctor/dtor or the `TVMFFIError*` C ABI exports into the
cold region: those bodies sit at `0x39f90`, `0x3c260`, `0x57cd0`,
`0x59020`, interleaved with hot C-ABI exports.

### Hot vs cold symbol layout (final)

```text
0x00008450 t BacktraceFullCallback.cold
0x00008488 t BacktraceSyminfoCallback.cold
0x00008511 t TVMFFIBacktrace.cold
0x000086f2 t ErrorBuilder ctor                 <- moved into cold region
0x00008e54 t ErrorBuilder dtor                 <- moved into cold region
0x0000b98a T TVMFFIErrorSetRaised              <- moved into cold region
0x0000c4cd T TVMFFIErrorSetRaisedFromCStr      <- moved into cold region
0x0000c7fd T TVMFFIErrorSetRaisedFromCStrParts <- moved into cold region
...
0x00047b80 T TVMFFIBacktrace                   <- hot body starts well after
                                                cold cluster
```

`ErrorBuilder` ctor/dtor and the cold-marked C ABI exports now live
inside the cold cluster at the head of `.text` (offsets `0x86f2`
through `0xc7fd`). The hot `TVMFFIBacktrace` body has moved from
offset `0x32e00` (baseline) to `0x47b80` (final), reflecting that the
cold cluster grew. Every `TVM_FFI_THROW` site keeps its `ErrorBuilder`
out-of-line in this cluster instead of side-by-side with C ABI
exports.

### `.text.unlikely.*` in the object files

`-ffunction-sections` is doing its job at the object-file level. For
`build_final/CMakeFiles/tvm_ffi_objs.dir/src/ffi/error.cc.o`:

```console
$ readelf -SW build_final/CMakeFiles/tvm_ffi_objs.dir/src/ffi/error.cc.o \
    | grep -c '\.text\.'
37
```

vs a single `.text` section in baseline. Section names such as
`.text.unlikely.TVMFFIErrorSetRaisedFromCStr` are emitted for
`[[gnu::cold]]`-marked functions, and the default linker script
gathers all of them together in `.text` at link time.

### Linker placement detail (ld.bfd / lld)

The relevant fragment of the default GNU linker script is:

```text
.text :
{
  *(.text.unlikely .text.*_unlikely .text.unlikely.*)
  *(.text.exit .text.exit.*)
  *(.text.startup .text.startup.*)
  *(.text.hot .text.hot.*)
  *(SORT(.text.sorted.*))
  *(.text .stub .text.* .gnu.linkonce.t.*)
}
```

`.text.unlikely.*` is matched first, so cold-marked functions cluster
at the START of `.text` (the lowest offsets), and the bulk of the
ordinary `.text.*` follows. The separation goal — keeping cold code
out of the hot instruction stream — is achieved regardless of whether
cold sits at the head or the tail.

## Why no `--gc-sections`

`-ffunction-sections` is sufficient for cold-region separation. The
related optimization `--gc-sections` (dead-section elimination) is
intentionally not enabled here. tvm-ffi uses static-initializer
registration patterns (`TVM_FFI_STATIC_INIT_BLOCK`, global function
registries) where symbols are reachable only via runtime tables that
the linker cannot trace statically. Enabling `--gc-sections` without
auditing every such registration for `__attribute__((used))`
(or equivalent) risks silently stripping registered globals at link
time. Left for a follow-up.

## How to apply `TVM_FFI_COLD_CODE`

```cpp
// header
TVM_FFI_COLD_CODE
[[noreturn]] void ReportFatalError(const std::string& message);

// source
TVM_FFI_COLD_CODE
int TVMFFIErrorCreate(const TVMFFIByteArray* kind, ...) {
  // ... only runs at error-construction time
}
```

Apply it conservatively, only to functions that run exclusively on
error / fatal / setup / teardown paths. Constructors of objects that
also flow through non-error paths (e.g. `Error` itself, used for
`EnvErrorAlreadySet()` and Python error propagation) must NOT be
marked cold.

## How to apply `TVM_FFI_UNLIKELY`

```cpp
int ret_code = TVMFFIThing(&arg);
if (TVM_FFI_UNLIKELY(ret_code != 0)) {
  throw ::tvm::ffi::details::MoveFromSafeCallRaised();
}
```

Limit it to error-check branches in performance-sensitive call sites
(C ABI dispatch macros, frequently-invoked check helpers). It is not a
substitute for `TVM_FFI_COLD_CODE` on whole functions, and over-use
makes ordinary branches harder to read.
