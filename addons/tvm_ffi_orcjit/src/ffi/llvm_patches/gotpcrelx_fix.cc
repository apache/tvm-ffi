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
 * \file gotpcrelx_fix.cc
 * \brief LLVM JITLink GOTPCRELX relaxation bug workaround (x86_64).
 *
 * See gotpcrelx_fix.h for the trigger, symptom, and removal procedure.
 */
#include "gotpcrelx_fix.h"

#if defined(__linux__) && (defined(__x86_64__) || defined(_M_X64))

#include <llvm/ADT/DenseMap.h>
#include <llvm/ExecutionEngine/JITLink/JITLink.h>
#include <llvm/ExecutionEngine/JITLink/x86_64.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MathExtras.h>

#include <cstdint>
#include <string>

namespace tvm {
namespace ffi {
namespace orcjit {

namespace {

/*! \brief Correct broken GOTPCRELX relaxations produced by
 *         optimizeGOTAndStubAccesses().
 *
 * Strategy:
 *  1. Build target-symbol → GOT-entry-symbol map (O(B+S) up front).
 *  2. For every Pointer32 edge whose preceding bytes are 67 e8
 *     (relaxed call) or e9 (relaxed jmp):
 *     - If the target is reachable via a signed 32-bit PC-relative
 *       displacement, change the edge to BranchPCRel32.
 *     - Otherwise revert the relaxation: restore the original
 *       indirect-call/jmp opcode bytes (ff 15 / ff 25), retarget
 *       the edge to the GOT entry, and use PCRel32 with addend 0
 *       (JITLink normalises GOTPCRELX addends to 0).
 */
llvm::Error fixBrokenGOTPCRELXRelaxation(llvm::jitlink::LinkGraph& G) {
  using namespace llvm::jitlink;
  // Build block → first symbol at offset 0 (for GOT entry symbol lookup).
  llvm::DenseMap<Block*, Symbol*> BlockToSym;
  for (auto* Sym : G.defined_symbols()) {
    if (Sym->getOffset() == 0 && !BlockToSym.count(&Sym->getBlock())) {
      BlockToSym[&Sym->getBlock()] = Sym;
    }
  }

  // Build target symbol → GOT entry symbol map.
  // GOT entries are pointer-sized blocks with exactly one Pointer64 edge.
  llvm::DenseMap<Symbol*, Symbol*> SymToGOTSym;
  for (auto* B : G.blocks()) {
    if (B->getSize() != G.getPointerSize()) continue;
    if (B->edges_size() != 1) continue;
    auto& E = *B->edges().begin();
    if (E.getKind() == x86_64::Pointer64) {
      auto It = BlockToSym.find(B);
      if (It != BlockToSym.end()) {
        SymToGOTSym[&E.getTarget()] = It->second;
      }
    }
  }

  for (auto* B : G.blocks()) {
    for (auto& E : B->edges()) {
      if (E.getKind() != x86_64::Pointer32) continue;
      if (E.getOffset() < 2) continue;

      auto MutableContent = B->getMutableContent(G);
      auto* FixupData = reinterpret_cast<uint8_t*>(MutableContent.data()) + E.getOffset();
      uint8_t Prev2 = FixupData[-2];
      uint8_t Prev1 = FixupData[-1];

      bool isRelaxedCall = (Prev2 == 0x67 && Prev1 == 0xe8);
      bool isRelaxedJmp = (Prev1 == 0xe9);
      if (!isRelaxedCall && !isRelaxedJmp) continue;

      // Check if PC-relative displacement would fit.
      auto TargetAddr = E.getTarget().getAddress();
      auto FixupAddr = B->getFixupAddress(E);
      int64_t Displacement = TargetAddr.getValue() - (FixupAddr.getValue() + 4) + E.getAddend();
      if (llvm::isInt<32>(Displacement)) {
        E.setKind(x86_64::BranchPCRel32);
        continue;
      }

      // Distance doesn't fit — revert to indirect call/jmp through GOT.
      auto It = SymToGOTSym.find(&E.getTarget());
      if (It == SymToGOTSym.end()) {
        return llvm::make_error<llvm::StringError>(
            "Cannot revert GOTPCRELX relaxation: no GOT entry for " +
                (E.getTarget().hasName() ? std::string(*E.getTarget().getName())
                                         : std::string("<anon>")),
            llvm::inconvertibleErrorCode());
      }

      Symbol* GOTSym = It->second;
      if (isRelaxedCall) {
        // Restore: 67 e8 → ff 15 (call *[rip+disp32])
        FixupData[-2] = 0xff;
        FixupData[-1] = 0x15;
      } else {
        // Restore: e9 XX XX XX XX 90 → ff 25 XX XX XX XX
        FixupData[-1] = 0xff;
        FixupData[0] = 0x25;
        // For jmp, the optimization shifted offset by -1; shift back.
        E.setOffset(E.getOffset() + 1);
      }
      E.setKind(x86_64::PCRel32);
      E.setTarget(*GOTSym);
      E.setAddend(0);
    }
  }
  return llvm::Error::success();
}

}  // namespace

void GOTPCRELXFixPlugin::modifyPassConfig(llvm::orc::MaterializationResponsibility& MR,
                                          llvm::jitlink::LinkGraph& G,
                                          llvm::jitlink::PassConfiguration& Config) {
  Config.PreFixupPasses.emplace_back(fixBrokenGOTPCRELXRelaxation);
}

llvm::Error GOTPCRELXFixPlugin::notifyFailed(llvm::orc::MaterializationResponsibility& MR) {
  return llvm::Error::success();
}

llvm::Error GOTPCRELXFixPlugin::notifyRemovingResources(llvm::orc::JITDylib& JD,
                                                        llvm::orc::ResourceKey K) {
  return llvm::Error::success();
}

void GOTPCRELXFixPlugin::notifyTransferringResources(llvm::orc::JITDylib& JD,
                                                     llvm::orc::ResourceKey DstKey,
                                                     llvm::orc::ResourceKey SrcKey) {}

}  // namespace orcjit
}  // namespace ffi
}  // namespace tvm

#endif  // __linux__ && __x86_64__
