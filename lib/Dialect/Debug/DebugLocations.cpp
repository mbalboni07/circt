//===- DebugLocations.cpp - Debug location attrs --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Debug/DebugLocations.h"
#include "circt/Dialect/Debug/DebugDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Location.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace debug;

// Dialect implementation generated from `DebugLocations.td`
#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/Debug/DebugLocations.cpp.inc"

void DebugDialect::registerLocations() {
  llvm::errs() << "Adding attributes\n";
  addAttributes<
#define GET_ATTRDEF_LIST
#include "circt/Dialect/Debug/DebugLocations.cpp.inc"
      >();
}
