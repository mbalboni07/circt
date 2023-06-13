//===- DebugInfo.h - Debug info emission ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares entry points to emit debug information.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_TARGET_DEBUGINFO_H
#define CIRCT_TARGET_DEBUGINFO_H

#include "circt/Support/LLVM.h"
#include "llvm/Support/raw_ostream.h"

namespace circt {
namespace debuginfo {

/// Register all debug information emission flavors as from-MLIR translations.
void registerTranslations();

/// Serializes the debug information in the given `module` into the HGLDD format
/// and writes it to `output`.
LogicalResult emitHGLDD(ModuleOp module, llvm::raw_ostream &os);

} // namespace debuginfo
} // namespace circt

#endif // CIRCT_TARGET_DEBUGINFO_H
