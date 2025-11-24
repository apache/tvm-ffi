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
 * \file orcjit_dylib.h
 * \brief LLVM ORC JIT DynamicLibrary (JITDylib) wrapper
 */
#ifndef TVM_FFI_ORCJIT_ORCJIT_DYLIB_H_
#define TVM_FFI_ORCJIT_ORCJIT_DYLIB_H_

#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/object.h>
#include <tvm/ffi/string.h>

namespace tvm {
namespace ffi {
namespace orcjit {

class ORCJITExecutionSessionObj;

class ORCJITDynamicLibraryObj : public ModuleObj {
 public:
  /*!
   * \brief Constructor
   * \param session The parent execution session
   * \param dylib The LLVM JITDylib
   * \param jit The LLJIT instance
   * \param name The library name
   */
  ORCJITDynamicLibraryObj(ObjectPtr<ORCJITExecutionSessionObj> session, llvm::orc::JITDylib* dylib,
                          llvm::orc::LLJIT* jit, String name);

  const char* kind() const final { return "orcjit"; }

  Optional<Function> GetFunction(const String& name) override;

 private:
  /*!
   * \brief Add an object file to this library
   * \param path Path to the object file to load
   */
  void AddObjectFile(const String& path);

  /*!
   * \brief Set the link order for symbol resolution
   * \param dylibs Vector of libraries to search for symbols (in order)
   *
   * When resolving symbols, this library will search in the specified libraries
   * in the order provided. This replaces any previous link order.
   */
  void SetLinkOrder(const std::vector<llvm::orc::JITDylib*>& dylibs);

  /*!
   * \brief Look up a symbol in this library
   * \param name The symbol name to look up
   * \return Pointer to the symbol, or nullptr if not found
   */
  void* GetSymbol(const String& name);

  /*!
   * \brief Get the underlying LLVM JITDylib
   * \return Reference to the LLVM JITDylib
   */
  llvm::orc::JITDylib& GetJITDylib();

  /*!
   * \brief Get the name of this library
   * \return The library name
   */
  String GetName() const { return name_; }

  /*! \brief Parent execution session (for lifetime management) */
  ObjectPtr<ORCJITExecutionSessionObj> session_;

  /*! \brief The LLVM JITDylib */
  llvm::orc::JITDylib* dylib_;

  /*! \brief The LLJIT instance (for addObjectFile API) */
  llvm::orc::LLJIT* jit_;

  /*! \brief Library name */
  String name_;

  /*! \brief Link order tracking (to support incremental linking) */
  llvm::orc::JITDylibSearchOrder link_order_;
};

/*!
 * \brief DynamicLibrary wrapper for LLVM ORC JIT v2 JITDylib
 *
 * This class wraps an LLVM JITDylib and provides functionality to:
 * - Load object files
 * - Link against other dynamic libraries
 * - Look up symbols
 */
class ORCJITDynamicLibrary : public Module {
 public:
  explicit ORCJITDynamicLibrary(const ObjectPtr<ORCJITDynamicLibraryObj>& ptr) : Module(ptr) {};
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ORCJITDynamicLibrary, Module,
                                                ORCJITDynamicLibraryObj);
};

}  // namespace orcjit
}  // namespace ffi
}  // namespace tvm

#endif  // TVM_FFI_ORCJIT_ORCJIT_DYLIB_H_
