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

# LLVM patches

This directory holds workarounds for upstream LLVM bugs and missing
features.  **A file belongs here if and only if its entire reason to
exist is an LLVM defect, and we would delete the whole file once
upstream catches up.**

Features that happen to coexist with a bug (e.g. the arena memory
manager, which we keep for THP and contiguous layout regardless of
LLVM state) live at the top level of `src/ffi/`, not here.

Each patch file opens with a fixed-shape header describing:

- which LLVM issue it addresses (link or "not yet filed"),
- affected version range,
- exact trigger conditions,
- symptom without the patch, and
- a `## Removal` section listing the `#include` and plugin-registration
  line(s) to delete when the upstream fix lands and the project's
  minimum LLVM version bumps past it.

## Index

- **COFF ctor/dtor** (Windows branch of `init_fini_plugin.{h,cc}`)
  LLVM issue: COFFPlatform stalled.
  Upstream status: stalled 2+ years.
  Remove when: COFFPlatform becomes usable end-to-end with clang-cl /
  MSVC objects.

- **Mach-O `__cxa_atexit` scoping** (`macho_cxa_atexit_shim.{h,cc}`)
  Upstream dependency: re-enabling `MachOPlatform` after the compact-unwind
  per-graph DSO-base fix is available.
  Remove when: the addon can use `MachOPlatform` end-to-end.

- **COFF unwind-data stripping** (`win_coff_pdata_strip.{h,cc}`)
  Upstream dependency: usable `COFFPlatform` support with SEH registration.
  Remove when: `.pdata` / `.xdata` can be registered and relocated normally.

- **Windows DLL import stubs** (`win_dll_import_generator.{h,cc}`)
  Upstream dependency: usable `COFFPlatform` and in-range DLL call stubs.
  Remove when: host DLL imports work end-to-end without the custom generator.

## Removal checklist

When deleting a patch file:

1. Delete the `.h` and `.cc` pair.
2. Remove the matching `#include "llvm_patches/<file>.h"` in
   `orcjit_session.cc`.
3. Remove the plugin-registration line(s) in `orcjit_session.cc`
   identified in the file's `## Removal` header block.
4. Remove the corresponding sources from `addons/tvm_ffi_orcjit/CMakeLists.txt`.
5. Update the index above.
