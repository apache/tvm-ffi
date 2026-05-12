/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file win_coff_pdata_strip.h
 * \brief COFF `.pdata` / `.xdata` relocation stripper for Windows JIT.
 *
 * Installed as an `ObjTransformLayer` transform on Windows.  clang-cl
 * places static functions in COMDAT sections, and the `.pdata` SEH
 * unwind data has relocations targeting COMDAT leader symbols.
 * JITLink's `COFFLinkGraphBuilder` doesn't register COMDAT leaders in
 * its symbol table when the second COMDAT symbol is `CLASS_STATIC`
 * (not `CLASS_EXTERNAL`), causing "Could not find symbol" errors at
 * graph-build time.
 *
 * `init_fini_plugin.cc` also strips `.pdata` / `.xdata` edges in a
 * `PostAllocationPass`; this transform moves the stripping earlier
 * (pre-graph-build) to prevent the graph builder error.  Both pieces
 * disappear once `COFFPlatform` becomes usable.
 *
 * Trigger: Windows x86_64 JIT of clang-cl objects that contain
 *          static-linkage functions (i.e. almost any C++ input).
 * Symptom without the transform: JITLink graph-build failure with
 *          "Could not find symbol" referencing a COMDAT leader.
 *
 * We avoid `llvm/Object/COFF.h` because `windows.h` (included
 * transitively by `LLJIT.h`) defines `IMAGE_*` macros that conflict
 * with LLVM's COFF enums; we parse the COFF header with raw
 * `memcpy`s instead.
 *
 * ## Removal
 *
 * Tied to enabling `COFFPlatform`.  See the Windows removal notes in
 * init_fini_plugin.h.
 */
#ifndef TVM_FFI_ORCJIT_LLVM_PATCHES_WIN_COFF_PDATA_STRIP_H_
#define TVM_FFI_ORCJIT_LLVM_PATCHES_WIN_COFF_PDATA_STRIP_H_

#ifdef _WIN32

#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>

#include <memory>

namespace tvm {
namespace ffi {
namespace orcjit {

/*! \brief Zero out `PointerToRelocations` / `NumberOfRelocations` on
 *         `.pdata` and `.xdata` section headers before the COFF object
 *         reaches JITLink's graph builder.
 *
 *  Returns the (possibly copy-modified) buffer.  Safe to install as the
 *  `ObjTransformLayer` transform — invokes no LLVM object parsers and
 *  short-circuits for any non-x86_64 or too-small input.
 */
llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>> StripCoffPdataXdata(
    std::unique_ptr<llvm::MemoryBuffer> Buf);

}  // namespace orcjit
}  // namespace ffi
}  // namespace tvm

#endif  // _WIN32

#endif  // TVM_FFI_ORCJIT_LLVM_PATCHES_WIN_COFF_PDATA_STRIP_H_
