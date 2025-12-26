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
 * \file orcjit_session.cc
 * \brief LLVM ORC JIT ExecutionSession implementation
 */

#include "orcjit_session.h"

#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/TargetSelect.h>
#include <tvm/ffi/cast.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/reflection/registry.h>

#include <sstream>

#include "orcjit_dylib.h"
#include "tvm/ffi/object.h"

namespace tvm {
namespace ffi {
namespace orcjit {

// Initialize LLVM native target (only once)
struct LLVMInitializer {
  LLVMInitializer() {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();
  }
};

static LLVMInitializer llvm_initializer;

ORCJITExecutionSessionObj::ORCJITExecutionSessionObj() : jit_(nullptr), dylib_counter_(0) {}

void ORCJITExecutionSessionObj::Initialize() {
  // Create LLJIT instance
  auto jit_or_err = llvm::orc::LLJITBuilder()
                        .setPlatformSetUp(llvm::orc::ExecutorNativePlatform(
                            "/usr/lib/llvm-20/lib/clang/20/lib/linux/liborc_rt-x86_64.a"))
                        .create();
  if (!jit_or_err) {
    auto err = jit_or_err.takeError();
    std::string err_msg;
    llvm::handleAllErrors(std::move(err),
                          [&](const llvm::ErrorInfoBase& eib) { err_msg = eib.message(); });
    TVM_FFI_THROW(InternalError) << "Failed to create LLJIT: " << err_msg;
  }
  jit_ = std::move(*jit_or_err);
}

ORCJITExecutionSession::ORCJITExecutionSession() {
  ObjectPtr<ORCJITExecutionSessionObj> obj = make_object<ORCJITExecutionSessionObj>();
  obj->Initialize();
  data_ = std::move(obj);
}

ORCJITDynamicLibrary ORCJITExecutionSessionObj::CreateDynamicLibrary(const String& name) {
  TVM_FFI_CHECK(jit_ != nullptr, InternalError) << "ExecutionSession not initialized";

  // Generate name if not provided
  String lib_name = name;
  if (lib_name.empty()) {
    std::ostringstream oss;
    oss << "dylib_" << dylib_counter_++;
    lib_name = oss.str();
  }

  // Check if library with this name already exists
  TVM_FFI_CHECK(dylibs_.find(lib_name) == dylibs_.end(), ValueError)
      << "DynamicLibrary with name '" << lib_name << "' already exists";

  // Create a new JITDylib
  // auto& jd = jit_->getExecutionSession().createBareJITDylib(lib_name.c_str());
  auto& jd = jit_->getMainJITDylib();

  // Add process symbol resolver to make C/C++ stdlib available
  auto dlsg = llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
      jit_->getDataLayout().getGlobalPrefix());
  if (!dlsg) {
    TVM_FFI_THROW(InternalError) << "Failed to create process symbol resolver";
  }
  jd.addGenerator(std::move(*dlsg));

  // Add __dso_handle by compiling a minimal LLVM IR module containing it.
  // This ensures __dso_handle is allocated in JIT memory (within 2GB of code),
  // avoiding "relocation out of range" errors with optimized code.
  //
  // We create an IR module with a global variable for __dso_handle, then compile
  // it through the normal IR compilation path. JITLink will allocate it properly.
  auto Ctx = std::make_unique<llvm::LLVMContext>();
  auto M = std::make_unique<llvm::Module>("__dso_handle_module", *Ctx);
  M->setDataLayout(jit_->getDataLayout());
  M->setTargetTriple(jit_->getTargetTriple().str());

  // Create a global variable: i8 __dso_handle = 0
  auto* Int8Ty = llvm::Type::getInt8Ty(*Ctx);
  auto* DsoHandle = new llvm::GlobalVariable(
      *M, Int8Ty,
      false,                              // not constant
      llvm::GlobalValue::WeakAnyLinkage,  // Use weak linkage so multiple dylibs can define it
      llvm::ConstantInt::get(Int8Ty, 0), "__dso_handle");
  DsoHandle->setVisibility(llvm::GlobalValue::DefaultVisibility);

  // Add the module to THIS specific JITDylib using the IR layer
  auto& CompileLayer = jit_->getIRCompileLayer();
  if (auto Err = CompileLayer.add(jd, llvm::orc::ThreadSafeModule(std::move(M), std::move(Ctx)))) {
    std::string err_msg;
    llvm::handleAllErrors(std::move(Err),
                          [&](const llvm::ErrorInfoBase& eib) { err_msg = eib.message(); });
    TVM_FFI_THROW(InternalError) << "Failed to add __dso_handle module: " << err_msg;
  }

  // Create the wrapper object
  auto dylib = ORCJITDynamicLibrary(make_object<ORCJITDynamicLibraryObj>(
      GetObjectPtr<ORCJITExecutionSessionObj>(this), &jd, jit_.get(), lib_name));

  // Store for lifetime management
  dylibs_.insert({lib_name, dylib});

  return dylib;
}

llvm::orc::ExecutionSession& ORCJITExecutionSessionObj::GetLLVMExecutionSession() {
  TVM_FFI_CHECK(jit_ != nullptr, InternalError) << "ExecutionSession not initialized";
  return jit_->getExecutionSession();
}

llvm::orc::LLJIT& ORCJITExecutionSessionObj::GetLLJIT() {
  TVM_FFI_CHECK(jit_ != nullptr, InternalError) << "ExecutionSession not initialized";
  return *jit_;
}

}  // namespace orcjit
}  // namespace ffi
}  // namespace tvm
