//===- EmitHGLDD.cpp - HGLDD debug info emission --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Debug/DebugDialect.h"
#include "circt/Dialect/Debug/DebugLocations.h"
#include "circt/Target/DebugInfo.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "di"

using namespace circt;

struct DIPort {
  std::string name;
  // TODO: type
  // TODO: source_loc
  // TODO: output_loc
  // TODO: value expr
};

struct DIHierarchy {
  SmallVector<DIPort, 0> ports;
};

struct DebugInfo {
  DebugInfo(Operation *op);

  llvm::BumpPtrAllocator allocator;
};

DebugInfo::DebugInfo(Operation *op) {
  // TODO: Traverse the op, collect all the debug info attributes, and combine
  // them into a different debug info data structure.
  LLVM_DEBUG(llvm::dbgs() << "Should collect DI now\n");

  op->walk([&](Operation *op) {
    LLVM_DEBUG(llvm::dbgs() << "- Visiting " << op->getName() << " at "
                            << op->getLoc() << "\n");
  });
}

LogicalResult debuginfo::emitHGLDD(ModuleOp module, llvm::raw_ostream &os) {
  module.getContext()->loadDialect<debug::DebugDialect>();

  auto loc = debug::MagicLoc::get(module.getContext(),
                                  StringAttr::get(module.getContext(), "hello"),
                                  module.getLoc());
  llvm::errs() << "Attribute: " << loc << "\n";

  DebugInfo di(module);
  return success();
  // module.emitOpError("should emit HGLDD; not implemented yet");
  // return failure();
}
