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
 * \file orcjit_session.h
 * \brief LLVM ORC JIT ExecutionSession wrapper
 */
#ifndef TVM_FFI_ORCJIT_ORCJIT_SESSION_H_
#define TVM_FFI_ORCJIT_ORCJIT_SESSION_H_

#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <tvm/ffi/object.h>
#include <tvm/ffi/string.h>

#include <string>

namespace tvm {
namespace ffi {
namespace orcjit {

// Forward declaration
class ORCJITDynamicLibrary;

/*!
 * \brief ExecutionSession object for LLVM ORC JIT v2
 *
 * This class manages the lifetime of an LLVM ExecutionSession and provides
 * functionality to create and manage multiple JITDylibs (DynamicLibraries).
 */
class ORCJITExecutionSessionObj : public Object {
 public:
  /*!
   * \brief Default constructor (for make_object)
   */
  explicit ORCJITExecutionSessionObj(const std::string& orc_rt_path = "");

  ~ORCJITExecutionSessionObj() { puts("ORCJITExecutionSessionObj destructed"); };

  /*!
   * \brief Create a new DynamicLibrary (JITDylib) in this session
   * \param name Optional name for the library (for debugging)
   * \return The created dynamic library instance
   */
  ORCJITDynamicLibrary CreateDynamicLibrary(const String& name);

  /*!
   * \brief Get the underlying LLVM ExecutionSession
   * \return Reference to the LLVM ExecutionSession
   */
  llvm::orc::ExecutionSession& GetLLVMExecutionSession();

  /*!
   * \brief Get the underlying LLJIT instance
   * \return Reference to the LLJIT instance
   */
  llvm::orc::LLJIT& GetLLJIT();

  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("orcjit.ExecutionSession", ORCJITExecutionSessionObj, Object);

 private:
  /*! \brief The LLVM ORC JIT instance */
  std::unique_ptr<llvm::orc::LLJIT> jit_;

  /*! \brief Counter for auto-generating library names */
  int dylib_counter_ = 0;
};

/*!
 * \brief Reference wrapper for ORCJITExecutionSessionObj
 *
 * A reference wrapper serves as a reference-counted pointer to the session object.
 */
class ORCJITExecutionSession : public ObjectRef {
 public:
  /*!
   * \brief Create a new ExecutionSession
   * \return The created execution session instance
   */
  explicit ORCJITExecutionSession(const std::string& orc_rt_path = "");
  // Required: define object reference methods
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ORCJITExecutionSession, ObjectRef,
                                                ORCJITExecutionSessionObj);
};

}  // namespace orcjit
}  // namespace ffi
}  // namespace tvm

#endif  // TVM_FFI_ORCJIT_ORCJIT_SESSION_H_
