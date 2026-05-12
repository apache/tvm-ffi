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
 * \file macho_cxa_atexit_shim.cc
 * \brief Per-JITDylib `__cxa_atexit` interposer implementation.
 *
 * See macho_cxa_atexit_shim.h for the trigger, symptom, and removal
 * procedure.
 */

#include "macho_cxa_atexit_shim.h"

#ifdef __APPLE__

#include <llvm/ExecutionEngine/Orc/AbsoluteSymbols.h>
#include <llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h>
#include <llvm/ExecutionEngine/Orc/Shared/ExecutorSymbolDef.h>
#include <llvm/Support/Error.h>

namespace tvm {
namespace ffi {
namespace orcjit {

namespace {
// TLS pointer to the currently-active dylib's records vector.
// Each CxaAtexitRecordsScope saves the previous pointer and restores it
// on exit, so nested scopes compose correctly across re-entrant init/fini.
//
// We cannot override ___dso_handle (LLJIT's Platform has already defined
// it in every user JITDylib), so we don't rely on the `dso_handle` arg
// passed to the shim.  Instead, each ORCJITDynamicLibraryObj publishes its
// own records vector via this TLS slot, scoped around any JIT entry point
// that may run ctors / dtors.  The shim pushes (fn, arg) into the
// TLS-pointed vector, or silently drops if no scope is active (which
// would be a stray call from outside any JIT execution — acceptable
// degradation).
thread_local CxaAtexitRecords* g_active_cxa_records = nullptr;

extern "C" int tvm_ffi_cxa_atexit_shim(void (*fn)(void*), void* arg,
                                       void* /*dso_handle*/) noexcept {
  if (fn == nullptr || g_active_cxa_records == nullptr) return 0;
  g_active_cxa_records->emplace_back(fn, arg);
  return 0;
}
}  // namespace

CxaAtexitRecordsScope::CxaAtexitRecordsScope(CxaAtexitRecords* records)
    : prev_(g_active_cxa_records) {
  g_active_cxa_records = records;
}
CxaAtexitRecordsScope::~CxaAtexitRecordsScope() { g_active_cxa_records = prev_; }

void InstallCxaAtexitShim(llvm::orc::ExecutionSession& ES, llvm::orc::JITDylib& jd) {
  llvm::orc::SymbolMap shim_syms;
  shim_syms[ES.intern("___cxa_atexit")] = {
      llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(&tvm_ffi_cxa_atexit_shim)),
      llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable};
  llvm::cantFail(jd.define(llvm::orc::absoluteSymbols(std::move(shim_syms))));
}

void DrainCxaAtexit(CxaAtexitRecords& records) {
  CxaAtexitRecordsScope scope(&records);
  while (!records.empty()) {
    auto [fn, arg] = records.back();
    records.pop_back();
    fn(arg);
  }
}

}  // namespace orcjit
}  // namespace ffi
}  // namespace tvm

#endif  // __APPLE__
