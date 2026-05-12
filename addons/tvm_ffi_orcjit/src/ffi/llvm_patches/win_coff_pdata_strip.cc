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
 * \file win_coff_pdata_strip.cc
 * \brief COFF `.pdata` / `.xdata` relocation stripper implementation.
 *
 * See win_coff_pdata_strip.h for the trigger, symptom, and removal
 * procedure.
 */

#include "win_coff_pdata_strip.h"

#ifdef _WIN32

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>

namespace tvm {
namespace ffi {
namespace orcjit {

llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>> StripCoffPdataXdata(
    std::unique_ptr<llvm::MemoryBuffer> Buf) {
  const char* Data = Buf->getBufferStart();
  size_t Size = Buf->getBufferSize();
  if (Size < 20) return std::move(Buf);

  // Parse COFF header (regular or bigobj format)
  uint16_t w0, w1;
  std::memcpy(&w0, Data, 2);
  std::memcpy(&w1, Data + 2, 2);
  bool bigobj = (w0 == 0 && w1 == 0xFFFF);

  uint16_t machine;
  uint32_t num_sections, ptr_to_symtab, num_symbols;
  size_t sec_hdr_start, sym_entry_size;
  if (bigobj) {
    if (Size < 56) return std::move(Buf);
    std::memcpy(&machine, Data + 6, 2);
    std::memcpy(&num_sections, Data + 44, 4);
    std::memcpy(&ptr_to_symtab, Data + 48, 4);
    std::memcpy(&num_symbols, Data + 52, 4);
    sec_hdr_start = 56;
    sym_entry_size = 20;
  } else {
    machine = w0;
    uint16_t ns, opt_hdr_size;
    std::memcpy(&ns, Data + 2, 2);
    std::memcpy(&opt_hdr_size, Data + 16, 2);
    std::memcpy(&ptr_to_symtab, Data + 8, 4);
    std::memcpy(&num_symbols, Data + 12, 4);
    num_sections = ns;
    sec_hdr_start = 20 + opt_hdr_size;
    sym_entry_size = 18;
  }
  if (machine != 0x8664) return std::move(Buf);

  // String table follows the symbol table
  size_t strtab_start = ptr_to_symtab + static_cast<size_t>(num_symbols) * sym_entry_size;

  // Resolve a section name (inline 8-byte or "/offset" string table ref)
  constexpr size_t kSecHdrSize = 40;
  auto resolve_name = [&](size_t hdr_off) -> llvm::StringRef {
    const char* raw = Data + hdr_off;
    if (raw[0] == '/' && raw[1] >= '0' && raw[1] <= '9') {
      uint32_t offset = 0;
      for (int j = 1; j < 8 && raw[j] >= '0' && raw[j] <= '9'; ++j)
        offset = offset * 10 + (raw[j] - '0');
      size_t pos = strtab_start + offset;
      if (pos < Size) {
        size_t len = 0;
        while (pos + len < Size && Data[pos + len]) ++len;
        return {Data + pos, len};
      }
    }
    size_t len = 0;
    while (len < 8 && raw[len]) ++len;
    return {raw, len};
  };

  // Collect section header offsets needing relocation stripping
  llvm::SmallVector<size_t, 8> strip_offsets;
  for (uint32_t i = 0; i < num_sections; ++i) {
    size_t off = sec_hdr_start + i * kSecHdrSize;
    if (off + kSecHdrSize > Size) break;
    auto name = resolve_name(off);
    if (name.starts_with(".pdata") || name.starts_with(".xdata")) {
      uint16_t num_relocs;
      std::memcpy(&num_relocs, Data + off + 32, 2);
      if (num_relocs > 0) strip_offsets.push_back(off);
    }
  }
  if (strip_offsets.empty()) return std::move(Buf);

  // Create mutable copy, zero out PointerToRelocations and NumberOfRelocations
  llvm::SmallVector<char> MutableBuf(Data, Data + Size);
  for (auto off : strip_offsets) {
    std::memset(&MutableBuf[off + 24], 0, 4);  // PointerToRelocations
    std::memset(&MutableBuf[off + 32], 0, 2);  // NumberOfRelocations
  }
  return llvm::MemoryBuffer::getMemBufferCopy(llvm::StringRef(MutableBuf.data(), MutableBuf.size()),
                                              Buf->getBufferIdentifier());
}

}  // namespace orcjit
}  // namespace ffi
}  // namespace tvm

#endif  // _WIN32
