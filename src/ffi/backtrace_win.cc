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
 * \file backtrace_win.cc
 * \brief Backtrace implementation on windows platform
 * \note We use the term "backtrace" to be consistent with python naming convention.
 */
#ifdef _MSC_VER

// clang-format off
#include <windows.h>
#include <dbghelp.h>  // NOLINT(*)
// clang-format on

#include <tvm/ffi/c_api.h>
#include <tvm/ffi/error.h>

#include <iostream>
#include <mutex>
#include <vector>

#include "./backtrace_utils.h"

namespace {

/*! \brief Global singleton holding tvm_ffi's DbgHelp symbol session. */
struct DbgHelpSession {
  /*! \brief Serializes every DbgHelp call made by tvm_ffi. */
  std::mutex mutex;
  /*! \brief Session handle, or nullptr when no session could be established. */
  HANDLE handle = nullptr;

  static DbgHelpSession* Global() {
    static DbgHelpSession inst;
    return &inst;
  }

  DbgHelpSession() {
    HANDLE current_process_handle = GetCurrentProcess();
    // Duplicate rather than key the session on GetCurrentProcess() directly: that
    // value is the same in every component, so another DbgHelp user in this process
    // could close our symbols. A duplicated handle is a key only we hold.
    if (!DuplicateHandle(current_process_handle, current_process_handle, current_process_handle,
                         &handle, 0, FALSE, DUPLICATE_SAME_ACCESS)) {
      handle = nullptr;
      return;
    }
    // LOAD_LINES keeps file and line information; UNDNAME demangles C++ names.
    SymSetOptions(SYMOPT_LOAD_LINES | SYMOPT_UNDNAME);
    // TRUE enumerates loaded modules now, so no module is discovered later while
    // the mutex is held. Unlike the previous code the result is checked: a session
    // that failed to initialize must not be cached as usable.
    if (!SymInitialize(handle, NULL, TRUE)) {
      // Nothing was filed under this key, so there is no session to SymCleanup;
      // release the handle and leave it null to report the failure.
      CloseHandle(handle);
      handle = nullptr;
    }
  }
};

}  // namespace

const TVMFFIByteArray* TVMFFIBacktrace(const char* filename, int lineno, const char* func,
                                       int cross_ffi_boundary) {
  static thread_local std::string backtrace_str;
  static thread_local TVMFFIByteArray backtrace_array;

  // pass in current line as here so last line of backtrace is always accurate
  tvm::ffi::BacktraceStorage backtrace;
  backtrace.stop_at_boundary = cross_ffi_boundary == 0;
  if (filename != nullptr && func != nullptr) {
    // need to skip TVMFFIBacktrace and the caller function
    // which is already included in filename and func
    backtrace.skip_frame_count = 2;
    if (!tvm::ffi::ShouldExcludeFrame(filename, func)) {
      backtrace.Append(filename, func, lineno);
    }
  }

  HANDLE thread = GetCurrentThread();

  DbgHelpSession* session = DbgHelpSession::Global();
  std::lock_guard<std::mutex> lock(session->mutex);
  HANDLE process = session->handle;
  if (process == nullptr) {
    // StackWalk64 resolves through the session, so without one there is nothing
    // to walk. Report the caller's own frame instead of using a dead session.
    backtrace_str = backtrace.GetBacktrace();
    backtrace_array.data = backtrace_str.data();
    backtrace_array.size = backtrace_str.size();
    return &backtrace_array;
  }
  // Register modules loaded since the session was created. StackWalk64 needs a
  // module's unwind tables to step through it, so a DLL DbgHelp does not know
  // truncates the walk at its first frame rather than just losing the name.
  SymRefreshModuleList(process);

  CONTEXT context = {};
  RtlCaptureContext(&context);

  STACKFRAME64 stack = {};
  DWORD machine_type;

#ifdef _M_IX86
  machine_type = IMAGE_FILE_MACHINE_I386;
  stack.AddrPC.Offset = context.Eip;
  stack.AddrStack.Offset = context.Esp;
  stack.AddrFrame.Offset = context.Ebp;
#elif _M_X64
  machine_type = IMAGE_FILE_MACHINE_AMD64;
  stack.AddrPC.Offset = context.Rip;
  stack.AddrStack.Offset = context.Rsp;
  stack.AddrFrame.Offset = context.Rbp;
#elif _M_ARM64
  machine_type = IMAGE_FILE_MACHINE_ARM64;
  stack.AddrPC.Offset = context.Pc;
  stack.AddrStack.Offset = context.Sp;
  stack.AddrFrame.Offset = context.Fp;
#else
#error "Unsupported architecture"
#endif

  stack.AddrPC.Mode = AddrModeFlat;
  stack.AddrFrame.Mode = AddrModeFlat;
  stack.AddrStack.Mode = AddrModeFlat;

  while (!backtrace.ExceedBacktraceLimit()) {
    if (!StackWalk64(machine_type, process, thread, &stack, &context, nullptr,
                     SymFunctionTableAccess64, SymGetModuleBase64, nullptr)) {
      break;
    }

    if (stack.AddrPC.Offset == 0) {
      break;
    }
    const char* filename = nullptr;
    const char* symbol = "<unknown>";
    int lineno = 0;
    // Get file and line number
    IMAGEHLP_LINE64 line_info;
    ZeroMemory(&line_info, sizeof(IMAGEHLP_LINE64));
    line_info.SizeOfStruct = sizeof(IMAGEHLP_LINE64);
    DWORD displacement32 = 0;

    if (SymGetLineFromAddr64(process, stack.AddrPC.Offset, &displacement32, &line_info)) {
      filename = line_info.FileName;
      lineno = line_info.LineNumber;
    }
    // allocate symbol info that aligns to the SYMBOL_INFO
    // we use u64 here to be safe
    size_t total_symbol_bytes = sizeof(SYMBOL_INFO) + MAX_SYM_NAME * sizeof(TCHAR);
    size_t total_u64_words = (total_symbol_bytes + 7) / 8;
    static_assert(8 % alignof(SYMBOL_INFO) == 0);
    std::vector<uint64_t> symbol_buffer(total_u64_words, 0);
    if (filename != nullptr) {
      // only run symbol translation if we have the file name
      // this is because SymFromAddr can return wrong symbol which becomes even more
      // confusing when pdb file do not exist
      PSYMBOL_INFO symbol_info = reinterpret_cast<PSYMBOL_INFO>(symbol_buffer.data());
      symbol_info->SizeOfStruct = sizeof(SYMBOL_INFO);
      symbol_info->MaxNameLen = MAX_SYM_NAME;
      DWORD64 displacement = 0;
      if (SymFromAddr(process, stack.AddrPC.Offset, &displacement, symbol_info)) {
        symbol = symbol_info->Name;
      }
    }
    if (backtrace.stop_at_boundary && tvm::ffi::DetectFFIBoundary(filename, symbol)) {
      break;
    }
    // skip extra frames
    if (backtrace.skip_frame_count > 0) {
      backtrace.skip_frame_count--;
      continue;
    }
    if (tvm::ffi::ShouldExcludeFrame(filename, symbol)) {
      continue;
    }
    backtrace.Append(filename, symbol, lineno);
  }
  backtrace_str = backtrace.GetBacktrace();
  backtrace_array.data = backtrace_str.data();
  backtrace_array.size = backtrace_str.size();
  return &backtrace_array;
}
#endif  // _MSC_VER
