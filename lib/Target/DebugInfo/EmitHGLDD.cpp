//===- EmitHGLDD.cpp - HGLDD debug info emission --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Target/DebugInfo.h"
#include "mlir/IR/BuiltinOps.h"

using namespace circt;

LogicalResult debuginfo::emitHGLDD(ModuleOp module, llvm::raw_ostream &os) {
  module.emitOpError("should emit HGLDD; not implemented yet");
  return failure();
}
