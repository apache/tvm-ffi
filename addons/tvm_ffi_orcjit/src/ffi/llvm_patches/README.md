<!---
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
--->

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

| Patch                 | File                                    | LLVM issue                                                 | Upstream status       | Remove when                     |
|-----------------------|-----------------------------------------|------------------------------------------------------------|-----------------------|----------------------------------|
| GOTPCRELX relaxation  | `gotpcrelx_fix.{h,cc}`                  | TBD — issue not yet filed                                  | open                  | LLVM floor bumps past the fix    |
| ELF init/fini         | `init_fini_plugin.{h,cc}` (Linux)       | [llvm/llvm-project#175981](https://github.com/llvm/llvm-project/issues/175981) | open, patch submitted | LLVM floor bumps past the fix    |
| COFF ctor/dtor        | `init_fini_plugin.{h,cc}` (Windows)     | COFFPlatform stalled                                       | stalled 2+ years      | COFFPlatform becomes usable      |

macOS already has working `MachOPlatform`, so no patch file is needed
for that platform.

## Removal checklist

When deleting a patch file:

1. Delete the `.h` and `.cc` pair.
2. Remove the matching `#include "llvm_patches/<file>.h"` in
   `orcjit_session.cc`.
3. Remove the plugin-registration line(s) in `orcjit_session.cc`
   identified in the file's `## Removal` header block.
4. Remove the corresponding sources from `addons/tvm_ffi_orcjit/CMakeLists.txt`.
5. Update the table above.
