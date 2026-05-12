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
 * \file macho_cxa_atexit_shim.h
 * \brief Per-JITDylib `__cxa_atexit` interposer for macOS JIT.
 *
 * We skip `MachOPlatform` entirely on macOS to sidestep the
 * compact-unwind 32-bit-delta bug in JITLink's `CompactUnwindSupport`
 * (see the analysis in orcjit_session.cc and
 * fix-machoplatform-libunwind-dso-base.patch at the repo root).  With
 * no Platform in the picture, clang-lowered
 * `__attribute__((destructor))` and C++ global dtors — which register
 * through `__cxa_atexit(fn, arg, &__dso_handle)` during init — would
 * fall through to libSystem's `___cxa_atexit`, orphaning those
 * callbacks from our drop-time drain.
 *
 * This shim:
 *   1. Installs an absolute symbol for `___cxa_atexit` on each user
 *      JITDylib that points at our own capture function
 *      (`InstallCxaAtexitShim`).
 *   2. Publishes the owning dylib's `CxaAtexitRecords` vector in TLS
 *      (`CxaAtexitRecordsScope`) so the capture function knows where
 *      to push `(fn, arg)` pairs.
 *   3. Drains the captured records LIFO at dylib destruction time
 *      (`DrainCxaAtexit`).
 *
 * Trigger: any macOS JIT object containing static destructors,
 *          `__attribute__((destructor))` functions, or C++ global
 *          objects with non-trivial dtors.
 * Symptom without the shim: destructors never run; any resource
 *          held by a JIT global leaks for the lifetime of the host
 *          process.
 *
 * ## Removal
 *
 * Tied to re-enabling `MachOPlatform`.  See the macOS removal notes
 * in init_fini_plugin.h.  When `MachOPlatform` is restored, delete
 * this file, drop the `#include` in orcjit_session.cc / orcjit_dylib.h,
 * and remove the `cxa_atexit_records_` field from `ORCJITDynamicLibraryObj`.
 */
#ifndef TVM_FFI_ORCJIT_LLVM_PATCHES_MACHO_CXA_ATEXIT_SHIM_H_
#define TVM_FFI_ORCJIT_LLVM_PATCHES_MACHO_CXA_ATEXIT_SHIM_H_

#ifdef __APPLE__

#include <llvm/ExecutionEngine/Orc/Core.h>

#include <utility>
#include <vector>

namespace tvm {
namespace ffi {
namespace orcjit {

/*! \brief `(dtor, arg)` pairs captured from per-dylib `__cxa_atexit` calls. */
using CxaAtexitRecords = std::vector<std::pair<void (*)(void*), void*>>;

/*! \brief RAII scope that publishes a per-dylib `CxaAtexitRecords` vector
 *         to the TLS slot consulted by the `___cxa_atexit` shim.
 *
 *  The shim reads the TLS pointer and pushes `(fn, arg)` into the pointed-to
 *  vector; outside any active scope the shim silently drops registrations.
 *  Wrap any JIT entry point that can run ctors / dtors (e.g. `GetSymbol`'s
 *  initialize call, the dtor's drain loop) with one of these, constructed
 *  with `&cxa_atexit_records_` from the owning `ORCJITDynamicLibraryObj`.
 */
class CxaAtexitRecordsScope {
 public:
  explicit CxaAtexitRecordsScope(CxaAtexitRecords* records);
  ~CxaAtexitRecordsScope();
  CxaAtexitRecordsScope(const CxaAtexitRecordsScope&) = delete;
  CxaAtexitRecordsScope& operator=(const CxaAtexitRecordsScope&) = delete;

 private:
  CxaAtexitRecords* prev_;
};

/*! \brief Install the `___cxa_atexit` absolute-symbol shim on \p jd.
 *
 *  Must be called once per user JITDylib, before any JIT code on that dylib
 *  materializes.  Placing the definition on the dylib itself (rather than
 *  injecting into the link order) ensures it wins over `<Platform>`'s
 *  libSystem fallback — JITDylib::define-time symbols are searched before
 *  the link order.
 */
void InstallCxaAtexitShim(llvm::orc::ExecutionSession& ES, llvm::orc::JITDylib& jd);

/*! \brief Drain captured `(fn, arg)` records LIFO, running each dtor.
 *
 *  Pop-and-call order handles re-entrant registrations from within a dtor —
 *  the internal `CxaAtexitRecordsScope` keeps the TLS pointer live so any
 *  `___cxa_atexit` call from inside a dtor also lands in \p records.
 */
void DrainCxaAtexit(CxaAtexitRecords& records);

}  // namespace orcjit
}  // namespace ffi
}  // namespace tvm

#endif  // __APPLE__

#endif  // TVM_FFI_ORCJIT_LLVM_PATCHES_MACHO_CXA_ATEXIT_SHIM_H_
