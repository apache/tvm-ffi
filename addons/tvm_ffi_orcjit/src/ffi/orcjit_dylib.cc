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
 * \file orcjit_dylib.cc
 * \brief LLVM ORC JIT DynamicLibrary implementation
 */

#include "orcjit_dylib.h"

#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/Object/ObjectFile.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/cast.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/variant.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "orcjit_session.h"
#include "orcjit_utils.h"

namespace tvm {
namespace ffi {
namespace orcjit {

namespace {
// When JIT thunks start reading their ctx/handle arg, grow this wrapper with
// a DylibFnContext prefix whose first field is the thunk pointer, and flip
// `safe_call` at a slab-emitted redirect that dispatches via ctx[0]. Example:
//
//     struct DylibFnContext { TVMFFISafeCallType fn; /* + closure fields */ };
//     struct DylibFnContextWithModule {
//       DylibFnContext ctx;  // first — pointer-interconvertible with wrapper
//       Module module_ref;
//     };
struct DylibFnContextWithModule {
  Module module_ref;  // keeps the owning dylib (and its slab) alive
};

void DeleteDylibFnContextWithModule(void* p) { delete static_cast<DylibFnContextWithModule*>(p); }

// Minimal little-endian reader for the embedded library-binary blob. The addon
// links only against public tvm-ffi headers, so it re-implements the reduced
// subset of the core parser it needs (see ProcessEmbeddedLibraryBin). All
// supported targets are little-endian (x86_64, aarch64, arm64).
class BlobReader {
 public:
  BlobReader(const char* data, size_t size) : data_(data), size_(size) {}

  // All bounds checks use subtraction (size_ - cursor_ never underflows because
  // cursor_ <= size_ is an invariant) to avoid cursor_ + n overflowing.
  uint64_t ReadU64() {
    TVM_FFI_CHECK(size_ - cursor_ >= sizeof(uint64_t), RuntimeError)
        << "Corrupt library binary: unexpected end of blob";
    uint64_t value = 0;
    for (size_t i = 0; i < sizeof(uint64_t); ++i) {
      value |= static_cast<uint64_t>(static_cast<unsigned char>(data_[cursor_ + i])) << (i * 8);
    }
    cursor_ += sizeof(uint64_t);
    return value;
  }

  std::string ReadString() {
    uint64_t nbytes = ReadU64();
    TVM_FFI_CHECK(nbytes <= size_ - cursor_, RuntimeError)
        << "Corrupt library binary: string length exceeds blob";
    std::string out(data_ + cursor_, static_cast<size_t>(nbytes));
    cursor_ += static_cast<size_t>(nbytes);
    return out;
  }

  std::vector<uint64_t> ReadU64Vector() {
    uint64_t count = ReadU64();
    // Guard reserve() against a corrupt count: each element needs 8 more bytes,
    // so count can't exceed the remaining byte budget. Prevents an OOM abort.
    TVM_FFI_CHECK(count <= (size_ - cursor_) / sizeof(uint64_t), RuntimeError)
        << "Corrupt library binary: vector size exceeds remaining blob";
    std::vector<uint64_t> out;
    out.reserve(static_cast<size_t>(count));
    for (uint64_t i = 0; i < count; ++i) {
      out.push_back(ReadU64());
    }
    return out;
  }

 private:
  const char* data_;
  size_t size_;
  size_t cursor_{0};
};

// Reduced re-implementation of the core ProcessLibraryBin parser.
// Deserializes the modules embedded in a __tvm_ffi__library_bin blob, wires up
// the import tree, and returns the root module. The "_lib" placeholder is
// filled by `lib_module` (the JIT dylib module itself). Custom modules are
// deserialized through the public ffi.Module.load_from_bytes.<kind> registry.
//
// Blob layout (little-endian):
//   <nbytes: u64> <indptr: vec<u64>> <child_indices: vec<u64>>
//   <kind0: str> [<bytes0: str>] <kind1: str> [<bytes1: str>] ...
// where vec<u64> = <count: u64> <u64 * count>, str = <len: u64> <bytes>,
// and the import tree is a CSR: module i imports child_indices[indptr[i]..].
Module ProcessEmbeddedLibraryBin(const char* library_bin, const Module& lib_module) {
  uint64_t nbytes = 0;
  for (size_t i = 0; i < sizeof(nbytes); ++i) {
    nbytes |= static_cast<uint64_t>(static_cast<unsigned char>(library_bin[i])) << (i * 8);
  }
  BlobReader reader(library_bin + sizeof(nbytes), static_cast<size_t>(nbytes));

  std::vector<uint64_t> indptr = reader.ReadU64Vector();
  std::vector<uint64_t> child_indices = reader.ReadU64Vector();
  // Validate the CSR shape up front so the wiring loop below cannot read out of
  // bounds: indptr has n+1 entries for n>=1 modules, starts at 0, is
  // non-decreasing, and its tail stays within child_indices.
  TVM_FFI_CHECK(indptr.size() >= 2, RuntimeError)
      << "Corrupt library binary: import tree indptr must have at least 2 entries";
  TVM_FFI_CHECK(indptr.front() == 0, RuntimeError)
      << "Corrupt library binary: import tree indptr must start at 0";
  for (size_t i = 0; i + 1 < indptr.size(); ++i) {
    TVM_FFI_CHECK(indptr[i] <= indptr[i + 1], RuntimeError)
        << "Corrupt library binary: import tree indptr must be non-decreasing";
  }
  TVM_FFI_CHECK(indptr.back() <= child_indices.size(), RuntimeError)
      << "Corrupt library binary: import tree indptr exceeds child index count";
  size_t num_modules = indptr.size() - 1;

  std::vector<Module> modules;
  modules.reserve(num_modules);
  for (size_t i = 0; i < num_modules; ++i) {
    std::string kind = reader.ReadString();
    if (kind == "_lib") {
      // Placeholder for the symbol source — the JIT dylib module itself.
      modules.push_back(lib_module);
    } else {
      std::string module_bytes = reader.ReadString();
      auto floader = Function::GetGlobal("ffi.Module.load_from_bytes." + kind);
      TVM_FFI_CHECK(floader.has_value(), RuntimeError)
          << "Library binary embeds a {" << kind << "} module but loader "
          << "ffi.Module.load_from_bytes." << kind << " is not registered";
      // cast<Module> throws if the loader returned a non-Module (e.g. None), and
      // Module is non-nullable, so the result is always a valid module.
      modules.push_back((*floader)(Bytes(module_bytes)).cast<Module>());
    }
  }

  // Wire the import tree (CSR) using the public ModuleObj::ImportModule.
  for (size_t i = 0; i < modules.size(); ++i) {
    for (uint64_t j = indptr[i]; j < indptr[i + 1]; ++j) {
      uint64_t child = child_indices[j];
      TVM_FFI_CHECK(child < modules.size(), RuntimeError)
          << "Corrupt library binary: child index out of range";
      modules[i]->ImportModule(modules[static_cast<size_t>(child)]);
    }
  }
  return modules[0];
}
}  // namespace

ORCJITDynamicLibraryObj::ORCJITDynamicLibraryObj(ORCJITExecutionSession session,
                                                 llvm::orc::JITDylib* dylib, llvm::orc::LLJIT* jit,
                                                 String name)
    : session_(std::move(session)), dylib_(dylib), jit_(jit), name_(std::move(name)) {
  TVM_FFI_CHECK(dylib_ != nullptr, ValueError) << "JITDylib cannot be null";
  TVM_FFI_CHECK(jit_ != nullptr, ValueError) << "LLJIT cannot be null";
}

ORCJITDynamicLibraryObj::~ORCJITDynamicLibraryObj() {
  // Step 1: run this dylib's static destructors. Drain the entries under the
  // lock but run them released — a JIT'd dtor may re-enter the session on this
  // thread, which the plain mutex could not survive while held.
  std::vector<ORCJITExecutionSessionObj::InitFiniEntry> deinit;
  {
    std::lock_guard<std::mutex> lock(session_->mutex_);
    deinit = session_->DrainPendingDeinitializers(GetJITDylib());
  }
  ORCJITExecutionSessionObj::RunInitFiniEntries(deinit);
#ifdef __APPLE__
  // Drain per-dylib __cxa_atexit registrations (LIFO) captured during init; see
  // llvm_patches/macho_cxa_atexit_shim.h.
  DrainCxaAtexit(cxa_atexit_records_);
#endif
  // Step 2: remove the JITDylib, releasing its JIT memory via the memory
  // manager's deallocate() (see orcjit_session.cc — no Platform teardown here).
  {
    std::lock_guard<std::mutex> lock(session_->mutex_);
    session_->RemoveDylib(dylib_);
  }
  dylib_ = nullptr;
}

void ORCJITDynamicLibraryObj::AddObjectFile(const String& path) {
  // Read object file from disk into an owned MemoryBuffer.
  auto buffer_or_err = llvm::MemoryBuffer::getFile(path.c_str());
  if (!buffer_or_err) {
    TVM_FFI_THROW(IOError) << "Failed to read object file: " << path;
  }
  AddObjectBuffer(std::move(*buffer_or_err));
}

void ORCJITDynamicLibraryObj::AddObjectBytes(const Bytes& bytes) {
  // Copy the bytes into an owned MemoryBuffer: LLVM takes ownership and the
  // source bytes may not outlive linking.
  auto buffer = llvm::MemoryBuffer::getMemBufferCopy(llvm::StringRef(bytes.data(), bytes.size()),
                                                     name_.operator std::string());
  AddObjectBuffer(std::move(buffer));
}

void ORCJITDynamicLibraryObj::AddObjectBuffer(std::unique_ptr<llvm::MemoryBuffer> buffer) {
  // Leaf op (holds mutex_): addObjectFile only defines the object's
  // materialization unit — no JIT code runs here.
  std::lock_guard<std::mutex> lock(session_->mutex_);
  TVM_FFI_ORCJIT_LLVM_CALL(jit_->addObjectFile(*dylib_, std::move(buffer)));
}

Module ORCJITExecutionSessionObj::LoadModule(const Array<Variant<String, Bytes>>& objects,
                                             const String& name) {
  // Hold no lock here: the callees each lock mutex_ at the leaf, and the fresh
  // dylib is unpublished until this returns, so no other thread can race it.
  ORCJITDynamicLibrary dylib = CreateDynamicLibrary(name);
  ORCJITDynamicLibraryObj* self = dylib.get();

  for (const Variant<String, Bytes>& object : objects) {
    if (auto opt_path = object.as<String>()) {
      self->AddObjectFile(*opt_path);
    } else {
      self->AddObjectBytes(object.get<Bytes>());
    }
  }

  return self->Finalize();
}

Module ORCJITDynamicLibraryObj::Finalize() {
  // Called once, from LoadModule, holding no session lock (its callees lock at
  // the leaf). The guard prevents a second pass from appending duplicate imports
  // (ImportModule does not dedup).
  TVM_FFI_CHECK(!finalized_, InternalError) << "ORCJIT dynamic library already finalized";
  finalized_ = true;

  // Inject context symbols eagerly rather than deferring to first lookup.
  InitContextSymbols();

  // If the objects embed a library binary, reconstruct the import tree so the
  // result behaves like a normally-loaded tvm-ffi module. Otherwise this is a
  // plain JIT dylib module.
  if (const char* library_bin =
          reinterpret_cast<const char*>(GetSymbol(symbol::tvm_ffi_library_bin))) {
    return ProcessEmbeddedLibraryBin(library_bin, GetRef<Module>(this));
  }
  return GetRef<Module>(this);
}

void* ORCJITDynamicLibraryObj::GetSymbol(const String& name) {
  // Search this dylib only. Its JITDylib link order (set at creation) already
  // chains to Main → Platform → ProcessSymbols for host/runtime symbols, so a
  // single-entry search order resolves everything a self-contained module needs.
  llvm::orc::JITDylibSearchOrder search_order;
  search_order.emplace_back(dylib_, llvm::orc::JITDylibLookupFlags::MatchAllSymbols);

  // Hold the lock across the lookup: materialization runs on this thread and,
  // via InitFiniPlugin, populates the session's pending maps (lock-guarded).
  // Drain under the lock; run constructors after release (see below).
  llvm::Expected<llvm::orc::ExecutorSymbolDef> symbol_or_err = llvm::orc::ExecutorSymbolDef();
  std::vector<ORCJITExecutionSessionObj::InitFiniEntry> init;
  {
    std::lock_guard<std::mutex> lock(session_->mutex_);
    symbol_or_err =
        jit_->getExecutionSession().lookup(search_order, jit_->mangleAndIntern(name.c_str()));
    init = session_->DrainPendingInitializers(GetJITDylib());
  }

  // Run this dylib's constructors (drained above) with the lock released.
#ifdef __APPLE__
  // Route any __cxa_atexit registrations made during init to this dylib's
  // records; see llvm_patches/macho_cxa_atexit_shim.h.
  CxaAtexitRecordsScope scope(&cxa_atexit_records_);
#endif
  ORCJITExecutionSessionObj::RunInitFiniEntries(init);

  if (!symbol_or_err) {
    llvm::Error remaining =
        llvm::handleErrors(symbol_or_err.takeError(), [](const llvm::orc::SymbolsNotFound&) {});
    if (remaining) TVM_FFI_ORCJIT_LLVM_CALL(std::move(remaining));
    return nullptr;
  }
  return symbol_or_err->getAddress().toPtr<void*>();
}

void ORCJITDynamicLibraryObj::InitContextSymbols() {
  // Called once from Finalize before the dylib is published, so no guard is
  // needed. Point the library-context slot at this module and inject any
  // registered context symbols.
  if (void** ctx_addr = reinterpret_cast<void**>(GetSymbol(symbol::tvm_ffi_library_ctx))) {
    *ctx_addr = this;
  }
  Module::VisitContextSymbols([this](const String& name, void* symbol) {
    if (void** ctx_addr = reinterpret_cast<void**>(GetSymbol(name))) {
      *ctx_addr = symbol;
    }
  });
}

llvm::orc::JITDylib& ORCJITDynamicLibraryObj::GetJITDylib() {
  TVM_FFI_CHECK(dylib_ != nullptr, InternalError) << "JITDylib is null";
  return *dylib_;
}

Optional<Function> ORCJITDynamicLibraryObj::GetFunction(const String& name) {
  // Pure symbol lookup. Context symbols were injected once at load time (see
  // Finalize), so this holds no lock and does no refresh — the returned
  // Function, once resolved, is invoked lock-free on the hot path.
  //
  // TVM-FFI exports have the __tvm_ffi_ prefix.
  std::string symbol_name = symbol::tvm_ffi_symbol_prefix + std::string(name);
  if (void* symbol = GetSymbol(symbol_name)) {
    TVMFFISafeCallType c_func = reinterpret_cast<TVMFFISafeCallType>(symbol);
    auto* wrapper = new DylibFnContextWithModule{GetRef<Module>(this)};
    return Function::FromExternC(wrapper, c_func, DeleteDylibFnContextWithModule);
  }
  return std::nullopt;
}

//-------------------------------------
// Registration
//-------------------------------------

static void RegisterOrcJITFunctions() {
  static bool registered = false;
  if (registered) return;
  registered = true;

  namespace refl = tvm::ffi::reflection;

  refl::ObjectDef<ORCJITExecutionSessionObj>();

  refl::GlobalDef()
      .def("tvm_ffi_orcjit.ExecutionSession",
           [](const Optional<Variant<String, Bytes>>& orc_rt, int64_t slab_size_bytes) {
             return ORCJITExecutionSession(orc_rt, slab_size_bytes);
           })
      .def("tvm_ffi_orcjit.GlobalDefaultSession",
           []() { return ORCJITExecutionSessionObj::GlobalDefault(); })
      .def("tvm_ffi_orcjit.SessionLoadModule",
           [](const ORCJITExecutionSession& session, const Array<Variant<String, Bytes>>& objects,
              const String& name) -> Module { return session->LoadModule(objects, name); })
      .def("tvm_ffi_orcjit.SessionClearFreeSlabs",
           [](const ORCJITExecutionSession& session) -> int64_t {
             return session->ClearFreeSlabs();
           });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  // This block may not execute when loaded via dlopen on some platforms.
  // Call TVMFFIOrcJITInitialize() explicitly if functions are not registered.
  RegisterOrcJITFunctions();
}

}  // namespace orcjit
}  // namespace ffi
}  // namespace tvm

// C API for explicit initialization
extern "C" {

TVM_FFI_DLL_EXPORT void TVMFFIOrcJITInitialize() { tvm::ffi::orcjit::RegisterOrcJITFunctions(); }
}
