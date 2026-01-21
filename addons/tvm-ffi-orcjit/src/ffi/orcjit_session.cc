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
#include <llvm/Support/Error.h>
#include <llvm/Support/TargetSelect.h>
#include <tvm/ffi/cast.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/reflection/registry.h>

#include "orcjit_dylib.h"
#include "orcjit_utils.h"
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

ORCJITExecutionSessionObj::ORCJITExecutionSessionObj(const std::string& orc_rt_path)
    : jit_(nullptr) {
  if (!orc_rt_path.empty()) {
    jit_ = std::move(call_llvm(llvm::orc::LLJITBuilder()
                                   .setPlatformSetUp(llvm::orc::ExecutorNativePlatform(orc_rt_path))
                                   .create(),
                               "Failed to create LLJIT with ORC runtime"));
  } else {
    jit_ = std::move(call_llvm(llvm::orc::LLJITBuilder().create(), "Failed to create LLJIT"));
    auto jit_or_err = llvm::orc::LLJITBuilder().create();
  }
}

ORCJITExecutionSession::ORCJITExecutionSession(const std::string& orc_rt_path) {
  ObjectPtr<ORCJITExecutionSessionObj> obj = make_object<ORCJITExecutionSessionObj>(orc_rt_path);
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

  llvm::orc::JITDylib& jit_dylib =
      call_llvm(jit_->getExecutionSession().createJITDylib(lib_name.c_str()));
  jit_dylib.addToLinkOrder(jit_->getMainJITDylib());
  if (auto* PlatformJD = jit_->getPlatformJITDylib().get()) {
    jit_dylib.addToLinkOrder(*PlatformJD);
  }
  std::unique_ptr<llvm::orc::DynamicLibrarySearchGenerator> generator =
      call_llvm(llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
                    jit_->getDataLayout().getGlobalPrefix()),
                "Failed to create process symbol resolver");
  jit_dylib.addGenerator(std::move(generator));

  auto dylib = ORCJITDynamicLibrary(make_object<ORCJITDynamicLibraryObj>(
      GetObjectPtr<ORCJITExecutionSessionObj>(this), &jit_dylib, jit_.get(), lib_name));

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
